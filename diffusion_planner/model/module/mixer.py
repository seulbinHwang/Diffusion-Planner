import torch
import torch.nn as nn
from timm.models.layers import Mlp
from typing import Any, List, Optional


class Conv1dMlp(nn.Module):
    """(N, Cin, L)에서 Cin 축을 섞는 MLP(Conv1d kernel=1로 구현).

    입력/출력 모양(Shape)
        - 입력:  (N, Cin, L)
        - 출력:  (N, Cout, L)

    핵심(쉽게 설명)
        - kernel_size=1인 Conv1d는, 각 위치(L의 각 칸)마다
          "Cin 길이 벡터 -> Cout 길이 벡터" 선형 변환을 합니다.
        - 즉, 마지막 축(L)을 그대로 둔 채 Cin 축에 Linear를 적용한 것과 완전히 같습니다.
        - 그래서 토큰 축(T)이나 lane_len 축 같은 "길이 축 변환"을
          transpose/permute 없이 처리할 수 있습니다.

    체크포인트 호환
        - 기존에 timm Mlp(Linear)로 저장된 fc1.weight/fc2.weight가 (out, in) 2D여도
          여기서는 (out, in, 1) 3D로 자동 변환해서 로드할 수 있게 처리합니다.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        drop_p: float = 0.0,
        act_layer: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(hidden_channels),
            kernel_size=1,
            bias=True,
        )
        self.act = act_layer
        self.drop1 = nn.Dropout(float(drop_p)) if float(drop_p) > 0.0 else nn.Identity()

        self.fc2 = nn.Conv1d(
            in_channels=int(hidden_channels),
            out_channels=int(out_channels),
            kernel_size=1,
            bias=True,
        )
        self.drop2 = nn.Dropout(float(drop_p)) if float(drop_p) > 0.0 else nn.Identity()

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Linear(2D) weight -> Conv1d(3D) weight 자동 변환 로더."""
        for name in ("fc1.weight", "fc2.weight"):
            key = prefix + name
            if key in state_dict:
                w = state_dict[key]
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    # (out, in) -> (out, in, 1)
                    state_dict[key] = w.unsqueeze(-1)

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (N, Cin, L)

        Returns:
            (N, Cout, L)
        """
        if x.numel() == 0:
            return x
        x = self.fc1(x)          # (N, hidden, L)
        x = self.act(x)          # (N, hidden, L)
        x = self.drop1(x)        # (N, hidden, L)
        x = self.fc2(x)          # (N, out, L)
        x = self.drop2(x)        # (N, out, L)
        return x


class MixerBlock(nn.Module):
    """(N, T, C) 입력을 '길이 축(T) 섞기' + '특징 길이(C) 섞기'로 처리하는 블록.

    입력/출력 모양(Shape)
        - 입력:  (N, T, C)
        - 출력:  (N, T, C)

    변경점(핵심)
        - 기존: (N,T,C) -> permute해서 (N,C,T)로 만든 뒤, Linear(=timm Mlp)를 T축에 적용
        - 변경: (N,T,C) 모양을 유지한 채, Conv1d(kernel=1)로 T축을 섞어서 permute 제거

    장점
        - 차원 순서 변경(permute/transpose) 때문에 생기는 "숨은 복사" 가능성을 줄이고,
          LayerNorm/Linear이 마지막 축(C)을 연속된 형태로 보기 쉬워집니다.
    """

    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        drop_path_rate: float,
        *,
        channels_mlp_ratio: float = 1.0,
    ) -> None:
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

        # ✅ 토큰 축(T) 섞기: (N, T, C)에서 T 축을 Conv1d(kernel=1)로 섞는다.
        # 입력을 (N, T, C) 그대로 Conv1d에 넣으면, Conv1d는 (N, in_channels=T, length=C)로 해석한다.
        self.tokens_mlp = Conv1dMlp(
            in_channels=int(tokens_mlp_dim),
            hidden_channels=int(tokens_mlp_dim),
            out_channels=int(tokens_mlp_dim),
            drop_p=float(drop_path_rate),
            act_layer=nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:  # N==0
            return x

        # (1) 토큰 축 섞기 (permute 없이)
        # y: (N, T, C)
        y = self.norm1(x)
        y = self.tokens_mlp(y)  # (N, T, C)
        x = x + y

        # (2) 채널 축(C) 섞기 (기존 그대로)
        y = self.norm2(x)
        return x + self.channels_mlp(y)
