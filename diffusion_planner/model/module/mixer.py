import inspect
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from typing import Any, List, Optional, Tuple, Union


# ------------------------------------------------------------
# FlashAttention의 fused LayerNorm / fused MLP (가능하면 사용)
# ------------------------------------------------------------
try:
    from flash_attn.ops.layer_norm import LayerNorm as _FlashAttnLayerNorm
    from flash_attn.ops.fused_dense import FusedMLP as _FlashAttnFusedMLP

    _FLASH_FUSED_AVAILABLE: bool = True
    _FLASH_FUSED_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    _FlashAttnLayerNorm = None
    _FlashAttnFusedMLP = None
    _FLASH_FUSED_AVAILABLE = False
    _FLASH_FUSED_IMPORT_ERR = _e


NormalizedShape = Union[int, Tuple[int, ...]]


def _try_build_flash_fused_mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    drop_p: float,
) -> Optional[nn.Module]:
    """flash-attn fused MLP를 만들 수 있으면 만들고, 아니면 None을 반환합니다.

    Args:
        in_features (int): 입력 마지막 축 길이.
        hidden_features (int): 중간 길이.
        out_features (int): 출력 마지막 축 길이.
        drop_p (float): dropout 확률.

    Returns:
        Optional[nn.Module]:
            - 만들 수 있으면 fused MLP 모듈
            - 실패하면 None
    """
    if (not _FLASH_FUSED_AVAILABLE) or (_FlashAttnFusedMLP is None):
        return None

    # flash-attn 쪽 API가 버전별로 조금 달라질 수 있어서,
    # 실패 가능성이 낮은 호출 패턴을 몇 개 정해 순서대로 시도합니다.
    candidates = [
        ((), {
            "in_features": int(in_features),
            "hidden_features": int(hidden_features),
            "out_features": int(out_features),
            "dropout": float(drop_p),
            "return_residual": False,
        }),
        ((), {
            "in_features": int(in_features),
            "hidden_features": int(hidden_features),
            "out_features": int(out_features),
            "dropout_p": float(drop_p),
            "return_residual": False,
        }),
        ((int(in_features), int(hidden_features), int(out_features)), {
            "dropout": float(drop_p),
            "return_residual": False,
        }),
        ((int(in_features), int(hidden_features), int(out_features)), {
            "dropout_p": float(drop_p),
            "return_residual": False,
        }),
        ((int(in_features), int(hidden_features)), {
            "out_features": int(out_features),
            "dropout": float(drop_p),
            "return_residual": False,
        }),
        ((int(in_features), int(hidden_features)), {
            "out_features": int(out_features),
            "dropout_p": float(drop_p),
            "return_residual": False,
        }),
    ]

    init_sig = inspect.signature(_FlashAttnFusedMLP.__init__)
    valid_keys = set(init_sig.parameters.keys())

    for args, kwargs in candidates:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        try:
            return _FlashAttnFusedMLP(*args, **filtered_kwargs)
        except Exception:
            continue

    return None


class FastLayerNorm(nn.Module):
    """가능하면 flash-attn LayerNorm을 쓰고, 아니면 torch LayerNorm을 쓰는 모듈입니다.

    입력/출력 모양(Shape)
        - 입력:  (..., D)
        - 출력:  (..., D)

    Note:
        - 수식은 같지만 내부 구현이 달라 아주 미세한 수치 차이는 생길 수 있습니다.
    """

    def __init__(
        self,
        normalized_shape: NormalizedShape,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps: float = float(eps)

        if _FLASH_FUSED_AVAILABLE and (_FlashAttnLayerNorm is not None):
            self._impl = _FlashAttnLayerNorm(normalized_shape, eps=self.eps)
        else:
            self._impl = nn.LayerNorm(normalized_shape, eps=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


class FastMlp(nn.Module):
    """가능하면 flash-attn fused MLP를 쓰고, 아니면 timm Mlp를 쓰는 모듈입니다.

    목적(쉽게 설명)
        - Linear -> GELU -> Linear(및 dropout)을 더 적은 GPU 호출로 수행하는 구현을 우선 사용합니다.
        - 입력이 2D든 3D든, 마지막 축을 in_features로 보고 처리합니다.

    입력/출력 모양(Shape)
        - 입력:  (..., in_features)
        - 출력:  (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features)
        self.out_features: int = int(out_features)
        self.drop: float = float(drop)

        fused = _try_build_flash_fused_mlp(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            drop_p=self.drop,
        )
        if fused is not None:
            self._impl: nn.Module = fused
            self._is_fused: bool = True
        else:
            self._impl = Mlp(
                in_features=self.in_features,
                hidden_features=self.hidden_features,
                out_features=self.out_features,
                act_layer=nn.GELU,
                drop=self.drop,
            )
            self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if int(x.shape[-1]) != int(self.in_features):
            raise ValueError(
                f"FastMlp input last dim mismatch. got {int(x.shape[-1])}, expected {int(self.in_features)}"
            )

        # fused MLP가 2D 입력을 가정하는 경우가 있어, 항상 2D로 펴서 처리합니다.
        orig_shape = x.shape                      # (..., in_features)
        x2 = x.reshape(-1, orig_shape[-1])        # (N, in_features)

        y2 = self._impl(x2)
        if isinstance(y2, tuple):
            y2 = y2[0]

        return y2.reshape(*orig_shape[:-1], -1)   # (..., out_features)


class FastLayerNormMlp(nn.Module):
    """LayerNorm -> MLP를 한 덩어리로 묶은 모듈입니다.

    목적
        - 코드에서 'norm + mlp' 패턴을 단순화합니다.
        - norm과 mlp 각각도 가능하면 fused 구현을 씁니다.

    입력/출력 모양(Shape)
        - 입력:  (..., in_features)
        - 출력:  (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        norm_eps: float = 1e-5,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = FastLayerNorm(int(in_features), eps=float(norm_eps))
        self.mlp = FastMlp(
            in_features=int(in_features),
            hidden_features=int(hidden_features),
            out_features=int(out_features),
            drop=float(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class Conv1dMlp(nn.Module):
    """(N, Cin, L)에서 Cin 축을 섞는 MLP(Conv1d kernel=1로 구현).

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

    변경점(핵심)
        - norm2 + channels_mlp를 FastLayerNormMlp로 교체해
          (가능하면) fused LayerNorm / fused MLP를 사용합니다.
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

        # (1) 토큰 축(T) 섞기 쪽: LayerNorm + Conv1dMlp
        self.norm1 = FastLayerNorm(channels_mlp_dim)
        self.tokens_mlp = Conv1dMlp(
            in_channels=int(tokens_mlp_dim),
            hidden_channels=int(tokens_mlp_dim),
            out_channels=int(tokens_mlp_dim),
            drop_p=float(drop_path_rate),
            act_layer=nn.GELU(),
        )

        # (2) 채널 축(C) 섞기 쪽: (LayerNorm -> MLP) 한 덩어리로
        hidden_c: int = max(16, int(float(channels_mlp_dim) * float(channels_mlp_ratio)))
        self.channels_norm_mlp = FastLayerNormMlp(
            in_features=int(channels_mlp_dim),
            hidden_features=int(hidden_c),
            out_features=int(channels_mlp_dim),
            drop=float(drop_path_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:  # N==0
            return x

        # (1) 토큰 축 섞기
        y = self.norm1(x)       # (N,T,C)
        y = self.tokens_mlp(y)  # (N,T,C)
        x = x + y

        # (2) 채널 축 섞기 (norm + mlp)
        return x + self.channels_norm_mlp(x)
