import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from typing import Any, List, Optional, Tuple, Union


# ------------------------------------------------------------
# FlashAttention fused MLP (가능하면 사용)
#   - LayerNorm은 "CUDA 확장"이 5090(sm_120)에서 터질 수 있으므로,
#     여기서는 Triton LN을 사용하도록 별도 처리합니다.
# ------------------------------------------------------------
try:
    from flash_attn.ops.fused_dense import FusedMLP as _FlashAttnFusedMLP
    _FLASH_FUSED_MLP_AVAILABLE: bool = True
    _FLASH_FUSED_MLP_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    _FlashAttnFusedMLP = None
    _FLASH_FUSED_MLP_AVAILABLE = False
    _FLASH_FUSED_MLP_IMPORT_ERR = _e


# ------------------------------------------------------------
# FlashAttention Triton LayerNorm (권장)
#   - 5090(sm_120)에서 CUDA 확장 LN(dropout_layer_norm)이 "no kernel image"로 터질 수 있어
#     Triton layer_norm_fn으로 강제합니다.
# ------------------------------------------------------------
try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn as _triton_layer_norm_fn
    _TRITON_LN_AVAILABLE: bool = True
    _TRITON_LN_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    _triton_layer_norm_fn = None
    _TRITON_LN_AVAILABLE = False
    _TRITON_LN_IMPORT_ERR = _e


NormalizedShape = Union[int, Tuple[int, ...]]


def _try_build_flash_fused_mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    drop_p: float,
) -> Optional[nn.Module]:
    """flash-attn fused MLP를 만들 수 있으면 만들고, 아니면 None을 반환합니다."""
    if (not _FLASH_FUSED_MLP_AVAILABLE) or (_FlashAttnFusedMLP is None):
        raise RuntimeError("flash-attn fused MLP is not available") from _FLASH_FUSED_MLP_IMPORT_ERR
        return None

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
    ]

    try:
        init_sig = inspect.signature(_FlashAttnFusedMLP.__init__)
        valid_keys = set(init_sig.parameters.keys())
    except Exception:
        valid_keys = set()

    for args, kwargs in candidates:
        filtered_kwargs = {k: v for k, v in kwargs.items() if (not valid_keys) or (k in valid_keys)}
        try:
            return _FlashAttnFusedMLP(*args, **filtered_kwargs)
        except Exception:
            continue

    return None


class FastLayerNorm(nn.Module):
    """Triton LayerNorm을 우선 사용하고, 안 되면 torch LayerNorm으로 처리합니다.

    입력/출력 모양
        - 입력: (..., D)
        - 출력: (..., D)

    주의
        - Triton LN은 내부 구현 차이로 아주 미세한 수치 차이가 날 수 있습니다.
        - CUDA 확장 LN(dropout_layer_norm)은 5090(sm_120)에서 런타임 에러가 날 수 있어
          여기서는 사용하지 않습니다.
    """

    def __init__(
        self,
        normalized_shape: NormalizedShape,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._ln = nn.LayerNorm(normalized_shape, eps=float(eps))

    @staticmethod
    def _use_triton_ln(x: torch.Tensor) -> bool:
        return bool(
            _TRITON_LN_AVAILABLE
            and (x.is_cuda)
            and (x.dtype in (torch.float16, torch.bfloat16))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            # 빈 텐서도 안전하게 처리
            return F.layer_norm(x, self._ln.normalized_shape, self._ln.weight, self._ln.bias, self._ln.eps)

        if self._use_triton_ln(x):
            w = self._ln.weight
            b = self._ln.bias

            # Triton LN은 weight/bias dtype/device가 입력과 맞을 때가 안전합니다(작은 텐서라 비용 작음)
            if w is not None and (w.dtype != x.dtype or w.device != x.device):
                w = w.to(device=x.device, dtype=x.dtype)
            if b is not None and (b.dtype != x.dtype or b.device != x.device):
                b = b.to(device=x.device, dtype=x.dtype)

            try:
                y = _triton_layer_norm_fn(
                    x,
                    w,
                    b,
                    eps=float(self._ln.eps),
                    dropout_p=0.0,
                    is_rms_norm=False,
                )
                # 버전에 따라 (y, ...) 튜플로 올 수도 있어 안전 처리
                if isinstance(y, tuple):
                    y = y[0]
                return y
            except Exception:
                raise RuntimeError("Triton LN failed") from None
                # Triton 경로가 어떤 이유로든 실패하면 안전 경로로 복귀
                return F.layer_norm(x, self._ln.normalized_shape, self._ln.weight, self._ln.bias, self._ln.eps)
        raise RuntimeError("Triton LN is not available")
        # 안전 경로
        return self._ln(x)


class FastMlp(nn.Module):
    """가능하면 flash-attn fused MLP를 쓰고, 아니면 timm Mlp를 씁니다."""

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
        else:
            self._impl = Mlp(
                in_features=self.in_features,
                hidden_features=self.hidden_features,
                out_features=self.out_features,
                act_layer=nn.GELU,
                drop=self.drop,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if int(x.shape[-1]) != int(self.in_features):
            raise ValueError(
                f"FastMlp input last dim mismatch. got {int(x.shape[-1])}, expected {int(self.in_features)}"
            )

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])  # (N, in_features)
        y2 = self._impl(x2)
        if isinstance(y2, tuple):
            y2 = y2[0]
        return y2.reshape(*orig_shape[:-1], -1)


class FastLayerNormMlp(nn.Module):
    """LayerNorm -> MLP를 묶은 모듈."""

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
    """(N, Cin, L)에서 Cin 축을 섞는 MLP(Conv1d kernel=1)."""

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
        self.fc1 = nn.Conv1d(int(in_channels), int(hidden_channels), kernel_size=1, bias=True)
        self.act = act_layer
        self.drop1 = nn.Dropout(float(drop_p)) if float(drop_p) > 0.0 else nn.Identity()
        self.fc2 = nn.Conv1d(int(hidden_channels), int(out_channels), kernel_size=1, bias=True)
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
        if x.numel() == 0:
            return x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """(N, T, C) -> (N, T, C)"""

    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        drop_path_rate: float,
        *,
        channels_mlp_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        # (1) 토큰 축 섞기: LN + Conv1dMlp
        self.norm1 = FastLayerNorm(channels_mlp_dim)
        self.tokens_mlp = Conv1dMlp(
            in_channels=int(tokens_mlp_dim),
            hidden_channels=int(tokens_mlp_dim),
            out_channels=int(tokens_mlp_dim),
            drop_p=float(drop_path_rate),
            act_layer=nn.GELU(),
        )

        # (2) 채널 축 섞기: (LN -> MLP)
        hidden_c: int = max(16, int(float(channels_mlp_dim) * float(channels_mlp_ratio)))
        self.channels_norm_mlp = FastLayerNormMlp(
            in_features=int(channels_mlp_dim),
            hidden_features=int(hidden_c),
            out_features=int(channels_mlp_dim),
            drop=float(drop_path_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x

        y = self.norm1(x)       # (N,T,C)
        y = self.tokens_mlp(y)  # (N,T,C)
        x = x + y

        return x + self.channels_norm_mlp(x)
