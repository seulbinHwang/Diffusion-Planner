import torch.nn as nn
from timm.models.layers import Mlp


class MixerBlock(nn.Module):
    """(N, T, C) 입력을 '길이 축(T) 섞기' + '특징 길이(C) 섞기'로 처리하는 블록.

    입력/출력 모양(Shape)
        - 입력:  (N, T, C)
        - 출력:  (N, T, C)

        N: 묶음 개수(예: 유효 lane 개수)
        T: 길이 축(예: 요약된 점 개수)
        C: 특징 길이(예: 192)

    channels_mlp_ratio
        - 1.0: 기존처럼 특징 길이(C) 쪽 변환을 크게 사용
        - 1.0보다 작게: 특징 길이 쪽 변환의 '중간 길이'를 줄여서,
          (토큰마다 반복되는 큰 계산을) 가볍게 만듭니다.
    """

    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        drop_path_rate: float,
        *,
        channels_mlp_ratio: float = 1.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)

        hidden_c: int = max(16, int(float(channels_mlp_dim) * float(channels_mlp_ratio)))
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_c,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=float(drop_path_rate),
        )

        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=float(drop_path_rate),
        )

    def forward(self, x):
        if x.size(0) == 0:  # N==0
            return x

        y = self.norm1(x)
        y = y.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        y = self.tokens_mlp(y)  # (N, C, T) -> (N, C, T)
        y = y.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = x + y

        y = self.norm2(x)
        return x + self.channels_mlp(y)
