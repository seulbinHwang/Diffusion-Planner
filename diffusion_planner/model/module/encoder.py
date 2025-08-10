import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

import math

from diffusion_planner.model.module.mixer import MixerBlock


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.future_len + 1 + config.agent_num + config.static_objects_num + config.lane_num
        self.ego_future_encoder = EgoFutureEncoder(
            config.future_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth)
        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth)
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim)
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth)

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device)

        # position embedding encode x, y, cos, sin, type (ego, neighbor, static, lane)
        self.pos_emb = nn.Linear(8, config.hidden_dim)

    def forward(self, inputs):
        encoder_outputs = {}
        # ego
        ego_past = inputs["ego_agent_past"]  # (B, V=21, D=11) -> (B, 1, V, D)
        ego_past = ego_past.unsqueeze(1)  # Add a dimension for P
        # agents
        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # ego future plan
        ego_future = inputs['ego_future']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]
        encoding_future, future_mask, future_pos, future_global = self.ego_future_encoder(ego_future)
        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(ego_past,
            neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit)

        encoding_input = torch.cat(
            [encoding_future, encoding_neighbors, encoding_static, encoding_lanes],
            dim=1)
        encoding_pos = torch.cat([future_pos, neighbor_pos, static_pos, lane_pos],
                                 dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat(
            [future_mask, neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim),
                                          device=encoding_pos.device)
        encoding_pos_result[
            ~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(
            B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num))
        encoder_outputs['ego_future_global'] = future_global

        return encoder_outputs


class SelfAttentionBlock(nn.Module):

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU,
                       drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class AgentFusionEncoder(nn.Module):

    def __init__(self,
                 time_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        # x,y,cos,sin + time(sin,cos,rel) + valid_mask
        self.channel_pre_project = Mlp(in_features=8 + 3 + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=time_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, x_ego, x):
        '''
        x_ego: B, P=1, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        '''
        x = torch.cat([x_ego, x], dim=1)  # (B, M=P+1, V, D)
        neighbor_type = x[:, :, -1, 8:]  # (B, M, 3)
        x = x[..., :8] # (B, M, V, 8)

        pos = x[:, :, -1, :8].clone()  # x, y, cos, sin # (B, M, 8)
        # neighbor: [0,1,0,0]
        pos[..., -4:] = 0.0
        pos[..., -3] = 1.0
        # ego: [1, 0, 0, 0]
        pos[:, 0, -4:] = 0.0
        pos[:, 0, -4] = 1.0

        B, M, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        base_time = torch.linspace(-0.1 * (V - 1), 0.0, V, device=x.device)
        time_norm = (base_time - base_time.min()) / (base_time.max() - base_time.min())
        time_rel = base_time / (0.1 * (V - 1))
        time_feat = torch.cat(
            [torch.sin(2 * math.pi * time_norm),
             torch.cos(2 * math.pi * time_norm),
             time_rel], dim=-1)
        time_feat = time_feat.view(1, 1, V, 3).expand(B, M, -1, -1)
        x = torch.cat([x, time_feat, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * M, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

            # pooling
        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * M, -1) # (B * M, 3)
        neighbor_type = neighbor_type[valid_indices] # (valid_indices.sum(), 3)
        type_embedding = self.type_emb(
            neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * M, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, M, -1), mask_p.reshape(B,
                                                       -1), pos.view(B, M, -1)


class EgoFutureEncoder(nn.Module):

    def __init__(self,
                 future_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=None,
                 channels_mlp_dim=128):
        super().__init__()

        if tokens_mlp_dim is None:
            tokens_mlp_dim = future_len

        self._future_len = future_len
        # x,y,cos,sin + time(sin,cos,rel) + valid_mask
        self.channel_pre_project = Mlp(in_features=4 + 3 + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=future_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)
        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, ego_future):
        """
        ego_future: (B, F, 4)
        Returns:
            tokens: (B, F, hidden_dim)
            mask: (B, F) bool
            pos: (B, F, 8)
            global_vec: (B, hidden_dim)
        """
        B, F, _ = ego_future.shape

        mask = torch.sum(torch.ne(ego_future, 0), dim=-1) == 0

        time_idx = torch.arange(F, device=ego_future.device, dtype=ego_future.dtype)
        time_norm = (time_idx / (F - 1)).view(1, F, 1)
        time_sec = (time_idx + 1) * 0.1
        time_rel = (time_sec / (0.1 * F)).view(1, F, 1)
        time_feat = torch.cat([
            torch.sin(2 * math.pi * time_norm),
            torch.cos(2 * math.pi * time_norm),
            time_rel
        ], dim=-1)
        time_feat = time_feat.expand(B, -1, -1)

        x = torch.cat([ego_future, time_feat, (~mask).float().unsqueeze(-1)], dim=-1)
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        tokens = self.emb_project(x.reshape(B * F, -1)).view(B, F, -1)
        global_vec = self.emb_project(x.mean(dim=1))

        pos = torch.zeros((B, F, 8), device=ego_future.device)
        pos[..., :4] = ego_future
        # ego future tokens: [1,0,0,0]
        pos[..., -4:] = 0.0
        pos[..., -4] = 1.0
        pos[mask] = 0.0

        return tokens, mask, pos, global_vec


class StaticFusionEncoder(nn.Module):

    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim,
                              hidden_features=hidden_dim,
                              out_features=hidden_dim,
                              act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        '''
        B, P, _ = x.shape

        pos = x[:, :, :8].clone()  # x, y, cos, sin
        # static: [0, 0,1,0]
        pos[..., -4:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):

    def __init__(self,
                 lane_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :8].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0, 0,0,1]
        pos[..., -4:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros(
            (speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1)
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(
            traffic)  # Traffic light embedding for valid data

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B,
                                                       -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=192,
                 num_heads=6,
                 drop_path_rate=0.3,
                 depth=3,
                 device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)
