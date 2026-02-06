import torch
import torch.nn as nn
from timm.models.layers import Mlp
from typing import List, Any


class Conv1dMlp(nn.Module):
    """(N, Cin, L) -> (N, Cout, L) 형태로 동작하는 작은 변환 블록입니다.

    목적
    - 입력의 "길이 축(L)"은 그대로 두고,
      각 위치마다 "채널(Cin)"을 섞어서 "채널(Cout)"로 바꿉니다.
    - Conv1d(kernel_size=1)을 써서, 불필요한 permute/transpose 없이
      채널 섞기를 수행하려고 사용합니다.

    입력/출력 모양(Shape)
    - 입력:  (N, Cin, L)
    - 출력:  (N, Cout, L)
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
            int(in_channels),
            int(hidden_channels),
            kernel_size=1,
            bias=True,
        )
        self.act = act_layer
        self.drop1 = nn.Dropout(float(drop_p)) if float(drop_p) > 0.0 else nn.Identity()
        self.fc2 = nn.Conv1d(
            int(hidden_channels),
            int(out_channels),
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
        """예전 체크포인트(Linear, 2D weight)를 Conv1d weight(3D)로 안전하게 받기 위한 처리입니다.

        설명
        - Conv1d의 weight는 보통 (Cout, Cin, 1) 모양입니다.
        - 과거에 같은 이름(fc1.weight/fc2.weight)을 Linear로 저장한 경우 (Cout, Cin) 모양일 수 있습니다.
        - 이때 마지막에 길이 축(1)을 한 칸 추가해서 (Cout, Cin, 1)로 맞춥니다.
        """
        for name in ("fc1.weight", "fc2.weight"):
            key = prefix + name
            if key in state_dict:
                w = state_dict[key]
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    state_dict[key] = w.unsqueeze(-1)  # (Cout, Cin) -> (Cout, Cin, 1)

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
        """앞에서 설명한 (N, Cin, L) -> (N, Cout, L) 변환을 수행합니다."""
        if x.numel() == 0:
            return x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """(N, T, C) 입력을 '길이 축(T) 섞기' + '특징 길이(C) 섞기'로 처리하는 블록.

    입력/출력 모양(Shape)
        - 입력:  (N, T, C)
        - 출력:  (N, T, C)

    변경점(after_weak)
        - tokens_mlp에서 permute(0,2,1)를 없애고,
          (N, T, C)를 (N, Cin=T, L=C)로 보고 Conv1d(kernel=1)로 처리합니다.
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

        # (N, T, C)를 (N, Cin=T, L=C)로 보고 채널(T)만 섞음
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

        # (1) 토큰 축(T) 섞기: permute 없이 Conv1d로 처리
        y = self.norm1(x)          # (N, T, C)
        y = self.tokens_mlp(y)     # (N, T, C)
        x = x + y

        # (2) 채널 축(C) 섞기: 기존과 동일
        y = self.norm2(x)
        return x + self.channels_mlp(y)
