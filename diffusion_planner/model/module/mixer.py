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
    from flash_attn.ops.layer_norm import LayerNorm as _FlashLayerNorm
except Exception as e1:
    try:
        from flash_attn.ops.triton.layer_norm import LayerNorm as _FlashLayerNorm
    except Exception as e2:
        raise ImportError(
            "Required: cannot import flash-attn LayerNorm implementation. "
            "This project must use FastLayerNorm (no fallback). "
            f"(cause1: {repr(e1)} / cause2: {repr(e2)})"
        ) from e2


def _create_flash_layernorm(hidden_size: int, eps: float) -> nn.Module:
    """Create a flash-attn LayerNorm module with minimal version-coupling.

    Args:
        hidden_size: Feature size (last dim).
        eps: Small value to avoid division by zero.

    Returns:
        A flash-attn LayerNorm module.
    """
    # Different flash-attn builds may use eps / epsilon; try both, then positional.
    try:
        return _FlashLayerNorm(int(hidden_size), eps=float(eps))
    except TypeError:
        try:
            return _FlashLayerNorm(int(hidden_size), epsilon=float(eps))
        except TypeError:
            return _FlashLayerNorm(int(hidden_size), float(eps))


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
