import torch.nn as nn
from timm.models.layers import Mlp

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()
        # tokens_mlp_dim: 64
        # channels_mlp_dim: 128
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1) # (N, tokens, C) -> (N, C, tokens)
        y = self.tokens_mlp(y) # (N, C, tokens) -> (N, C, tokens)
        y = y.permute(0, 2, 1) # (N, C, tokens) -> (N, tokens, C)
        x = x + y # skip connection
        y = self.norm2(x)
        return x + self.channels_mlp(y) # (N, tokens, C) -> (N, tokens, C)