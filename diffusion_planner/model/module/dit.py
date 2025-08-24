import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp


def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat(
            [x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest],
            dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32) /
                          half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=approx_gelu,
                        drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim,
                                                heads,
                                                dropout,
                                                batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=approx_gelu,
                        drop=0)
        self.gate_cross = nn.Parameter(torch.tensor(0.0))
        self.gate_mlp2 = nn.Parameter(torch.tensor(0.0))

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cross_c, y, attn_mask, cross_mask):
        """
        Input shapes:
            x: (B, Pnn, D=192)
            cross_c: (B, N=token_num, D=192)
            y: (B, D=192)
            attn_mask: near_current_mask: (B, Pnn)
            cross_mask: (B, token_num)

        Note:
            softmax 연산은 입력이 모두 마스킹된 경우 NaN을 발생시킬 수 있다.
            이를 방지하기 위해 완전히 마스킹된 배치는 어텐션을 건너뛰고
            출력 텐서를 0으로 초기화한다.
        """
        # y: (B, D=192)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
         gate_mlp) = self.adaLN_modulation(y).chunk(6, dim=1)


        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # 모든 토큰이 마스킹된 배치는 NaN을 방지하기 위해 어텐션을 생략한다
        msa_out = torch.zeros_like(x)
        valid_mask = ~attn_mask.all(dim=1)
        if valid_mask.any():
            msa_out[valid_mask] = self.attn(
                modulated_x[valid_mask],
                modulated_x[valid_mask],
                modulated_x[valid_mask],
                key_padding_mask=attn_mask[valid_mask],
                need_weights=False,
            )[0]
        msa_out = torch.nan_to_num(msa_out, nan=0.0, posinf=0.0,
                                   neginf=0.0)  # (B, P, D)
        x = x + gate_msa.unsqueeze(1) * msa_out  # (B, P, D)

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        q = self.norm3(x)
        cross_out = torch.zeros_like(x)
        valid_cross = ~cross_mask.all(dim=1)
        if valid_cross.any():
            cross_out[valid_cross] = self.cross_attn(
                q[valid_cross],
                cross_c[valid_cross],
                cross_c[valid_cross],
                key_padding_mask=cross_mask[valid_cross],
                need_weights=False,
            )[0]
        # 교차 어텐션에서도 완전히 마스크된 배치는 0으로 채워 NaN을 방지한다
        cross_out = torch.nan_to_num(cross_out, nan=0.0, posinf=0.0,
                                     neginf=0.0)
        x = x + self.gate_cross * cross_out
        x = x + self.gate_mlp2 * self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            # nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj[-1].weight)  # proj의 마지막 Linear
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, y):
        B, P, _ = x.shape
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x
