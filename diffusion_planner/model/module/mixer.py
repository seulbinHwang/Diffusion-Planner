# diffusion_planner/model/module/mixer.py (REPLACE THE WHOLE FILE)

import torch
import torch.nn as nn
from typing import Optional


# ============================================================
# Force flash-attn fused LayerNorm / fused MLP
# - If unavailable, raise immediately (NO fallback)
# ============================================================

# (1) FusedMLP
try:
    from flash_attn.ops.fused_dense import FusedMLP as _FlashFusedMLP
except Exception as e:
    raise ImportError(
        "Required: cannot import flash_attn.ops.fused_dense.FusedMLP. "
        "This project must use FastMlp (no fallback). "
        f"(cause: {repr(e)})"
    ) from e


# (2) LayerNorm (flash-attn implementation)
try:
    from flash_attn.ops.layer_norm import layer_norm as _flash_layer_norm_fn
except Exception as e1:
    try:
        from flash_attn.ops.triton.layer_norm import layer_norm as _flash_layer_norm_fn
    except Exception as e2:
        raise ImportError(
            "Required: cannot import flash-attn layer_norm function. "
            "This project must use FastLayerNorm (no fallback). "
            f"(cause1: {repr(e1)} / cause2: {repr(e2)})"
        ) from e2

class _FlashLayerNorm(nn.Module):
    """flash-attn의 layer_norm(함수)을 nn.Module처럼 쓰기 위한 래퍼.

    이 모듈은 torch.nn.LayerNorm과 같은 의미를 가지도록
    weight/bias 파라미터(길이 D)를 내부에 들고,
    forward에서 flash-attn layer_norm 함수를 호출합니다.

    입력/출력 모양:
        - 입력:  (..., D)
        - 출력: (..., D)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """
        Args:
            normalized_shape (int): 마지막 차원 D
            eps (float): 0으로 나누기 방지용 작은 값
        """
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)

        # (D,)
        self.weight: nn.Parameter = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., D)"""
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"_FlashLayerNorm: last dim must be {self.normalized_shape}. got {int(x.shape[-1])}"
            )

        if x.numel() == 0:
            return x

        orig_shape = tuple(x.shape)
        D = self.normalized_shape

        # layer_norm 함수가 2D 입력을 가정하는 경우도 있어서 안전하게 펼쳤다가 복원
        x2d = x.reshape(-1, D).contiguous()  # (M, D)

        # dtype/device는 입력 x에 맞춰서 호출(학습/AMP에서 꼬임 방지)
        w = self.weight.to(device=x2d.device, dtype=x2d.dtype).contiguous()  # (D,)
        b = self.bias.to(device=x2d.device, dtype=x2d.dtype).contiguous()    # (D,)

        # flash-attn layer_norm은 보통 (x, weight, bias, epsilon) 형태
        y2d = _flash_layer_norm_fn(x2d, w, b, self.eps)  # (M, D)
        return y2d.reshape(orig_shape)


def _create_flash_layernorm(hidden_size: int, eps: float) -> nn.Module:
    """flash-attn layer_norm 기반 LayerNorm 모듈 생성."""
    return _FlashLayerNorm(int(hidden_size), eps=float(eps))



class FastLayerNorm(nn.Module):
    """A LayerNorm wrapper that *forces* flash-attn LayerNorm.

    Shape:
        - Input:  (..., D)
        - Output: (..., D)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)
        self._ln: nn.Module = _create_flash_layernorm(
            hidden_size=self.normalized_shape,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ln(x)


class FastMlp(nn.Module):
    """An MLP wrapper that *forces* flash-attn FusedMLP.

    Key behavior:
        - Always flatten to 2D (M, D) before calling fused MLP
        - Restore original shape afterward

    Shape:
        - Input:  (..., in_features)
        - Output: (..., out_features)

    Note:
        - If drop > 0, this applies output dropout.
          (Some fused MLP variants may not implement internal dropout in the same way.)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        *,
        activation: str = "gelu_approx",
        checkpoint_lvl: int = 0,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features) if hidden_features is not None else int(in_features)
        self.out_features: int = int(out_features) if out_features is not None else int(in_features)
        self.drop_p: float = float(drop)

        self._mlp: nn.Module = _FlashFusedMLP(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            activation=str(activation),
            return_residual=False,
            checkpoint_lvl=int(checkpoint_lvl),
        )

        self._drop: nn.Module = nn.Dropout(self.drop_p) if self.drop_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"FastMlp: last dim must be in_features={self.in_features}. got {int(x.shape[-1])}"
            )

        orig_shape = tuple(x.shape)
        if x.numel() == 0:
            out_shape = (*orig_shape[:-1], self.out_features)
            return x.new_zeros(out_shape)

        x2d: torch.Tensor = x.reshape(-1, self.in_features)  # (M, D)
        y2d: torch.Tensor = self._mlp(x2d)                   # (M, out_features)
        y2d = self._drop(y2d)

        y: torch.Tensor = y2d.reshape(*orig_shape[:-1], self.out_features)
        return y


class FastLayerNormMlp(nn.Module):
    """A single-call module that chains FastLayerNorm -> FastMlp.

    Shape:
        - Input:  (..., in_features)
        - Output: (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
        *,
        eps: float = 1e-5,
        activation: str = "gelu_approx",
        checkpoint_lvl: int = 0,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features)
        self.out_features: int = int(out_features)

        self.norm: FastLayerNorm = FastLayerNorm(self.in_features, eps=float(eps))
        self.mlp: FastMlp = FastMlp(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            drop=float(drop),
            activation=str(activation),
            checkpoint_lvl=int(checkpoint_lvl),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class MixerBlock(nn.Module):
    """A block that mixes along token length (T) and feature/channel (C).

    Shape:
        - Input:  (N, T, C)
        - Output: (N, T, C)
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

        # (A) Token-axis mixing: LN (over C) + MLP over T (applied on (N, C, T))
        self.norm1: FastLayerNorm = FastLayerNorm(int(channels_mlp_dim))
        self.tokens_mlp: FastMlp = FastMlp(
            in_features=int(tokens_mlp_dim),
            hidden_features=int(tokens_mlp_dim),
            out_features=int(tokens_mlp_dim),
            drop=float(drop_path_rate),
        )

        # (B) Channel-axis mixing: fuse LN + MLP into a single call
        hidden_c: int = max(16, int(float(channels_mlp_dim) * float(channels_mlp_ratio)))
        self.channels_norm_mlp: FastLayerNormMlp = FastLayerNormMlp(
            in_features=int(channels_mlp_dim),
            hidden_features=int(hidden_c),
            out_features=int(channels_mlp_dim),
            drop=float(drop_path_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x

        # (1) Token-axis mixing
        y: torch.Tensor = self.norm1(x)       # (N, T, C)
        y = y.permute(0, 2, 1)                # (N, C, T)
        y = self.tokens_mlp(y)                # (N, C, T)
        y = y.permute(0, 2, 1)                # (N, T, C)
        x = x + y

        # (2) Channel-axis mixing (single-call LN+MLP)
        return x + self.channels_norm_mlp(x)
