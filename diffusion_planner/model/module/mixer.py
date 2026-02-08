# diffusion_planner/model/module/mixer.py (REPLACE THE WHOLE FILE)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ============================================================
# Prefer flash-attn fused LayerNorm / fused MLP
# - If unavailable:
#   - use_fallback=True  -> fall back to plain PyTorch
#   - use_fallback=False -> raise an error (but NOT at import-time)
# ============================================================

_FLASH_IMPORT_ERRORS: Dict[str, str] = {}
_FLASH_FUSED_MLP_AVAILABLE: bool = False
_FLASH_LAYER_NORM_AVAILABLE: bool = False

# (1) FusedMLP (flash-attn)
try:
    from flash_attn.ops.fused_dense import FusedMLP as _FlashFusedMLP  # type: ignore
    _FLASH_FUSED_MLP_AVAILABLE = True
except Exception as e:
    _FlashFusedMLP = None  # type: ignore
    _FLASH_FUSED_MLP_AVAILABLE = False
    _FLASH_IMPORT_ERRORS["fused_mlp"] = repr(e)

# (2) LayerNorm function (flash-attn)
_flash_layer_norm_fn = None
try:
    from flash_attn.ops.layer_norm import layer_norm as _flash_layer_norm_fn  # type: ignore
    _FLASH_LAYER_NORM_AVAILABLE = True
except Exception as e1:
    try:
        from flash_attn.ops.triton.layer_norm import layer_norm as _flash_layer_norm_fn  # type: ignore
        _FLASH_LAYER_NORM_AVAILABLE = True
    except Exception as e2:
        _flash_layer_norm_fn = None
        _FLASH_LAYER_NORM_AVAILABLE = False
        _FLASH_IMPORT_ERRORS["layer_norm"] = f"cause1: {repr(e1)} / cause2: {repr(e2)}"


def _require_flash_component_or_raise(
    component_key: str,
    *,
    available: bool,
    use_fallback: bool,
    what: str,
) -> None:
    """flash-attn 구성요소가 없을 때, 폴백 허용 여부에 따라 에러를 낼지 결정합니다.

    Args:
        component_key: _FLASH_IMPORT_ERRORS에 저장된 키 ("fused_mlp" 또는 "layer_norm").
        available: 해당 구성요소 import 성공 여부.
        use_fallback: True면 폴백 허용(에러 안 냄), False면 폴백 금지(에러 냄).
        what: 에러 메시지에 넣을 설명 문자열.

    Raises:
        RuntimeError: use_fallback=False인데 import가 실패한 경우.
    """
    if use_fallback:
        return
    if available:
        return
    cause = _FLASH_IMPORT_ERRORS.get(component_key, "unknown")
    raise RuntimeError(
        f"{what} 를 반드시 써야 하는데(use_fallback=False), flash-attn import에 실패했습니다.\n"
        f"- 실패 항목: {component_key}\n"
        f"- 원인: {cause}\n"
        "해결: flash-attn을 fused_dense / layer_norm 지원까지 포함되도록 설치/빌드한 뒤 다시 실행하세요."
    )


def _create_activation(activation: str) -> nn.Module:
    """문자열로 활성 함수를 선택해 nn.Module로 만듭니다.

    Args:
        activation (str): 활성 함수 이름.
            - "gelu_approx": GELU(근사)
            - "gelu": GELU
            - "relu": ReLU
            - "silu": SiLU

    Returns:
        nn.Module: 활성 함수 모듈.
    """
    name = str(activation).lower().strip()

    if name in ("gelu_approx", "gelu_fast", "gelu_tanh"):
        try:
            return nn.GELU(approximate="tanh")
        except TypeError:
            return nn.GELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unsupported activation: {activation}")


class _FlashLayerNorm(nn.Module):
    """flash-attn의 layer_norm(함수)을 nn.Module처럼 쓰기 위한 래퍼.

    동작:
    - flash-attn layer_norm 함수가 있고,
      입력이 CUDA + (fp16/bf16)일 때: flash-attn 함수 사용
    - 그 외(패키지 없음/CPU/fp32 등): PyTorch layer_norm 사용

    입력/출력 모양:
    - 입력:  (..., D)
    - 출력: (..., D)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)

        # (D,)
        self.weight: nn.Parameter = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.normalized_shape))

    @staticmethod
    def _can_use_flash(x: torch.Tensor) -> bool:
        if not _FLASH_LAYER_NORM_AVAILABLE:
            return False
        if _flash_layer_norm_fn is None:
            return False
        if not x.is_cuda:
            return False
        return x.dtype in (torch.float16, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"_FlashLayerNorm: last dim must be {self.normalized_shape}. got {int(x.shape[-1])}"
            )

        if x.numel() == 0:
            return x

        orig_shape = tuple(x.shape)
        D = self.normalized_shape

        # 일부 구현은 2D 입력을 기대할 수 있어 안전하게 펼쳤다가 복원
        x2d = x.reshape(-1, D).contiguous()  # (M, D)

        # weight/bias는 입력 dtype/device에 맞춰 사용
        w = self.weight.to(device=x2d.device, dtype=x2d.dtype).contiguous()  # (D,)
        b = self.bias.to(device=x2d.device, dtype=x2d.dtype).contiguous()    # (D,)

        if self._can_use_flash(x2d):
            y2d = _flash_layer_norm_fn(x2d, w, b, self.eps)  # type: ignore[misc]
        else:
            y2d = F.layer_norm(x2d, (D,), weight=w, bias=b, eps=self.eps)  # (M, D)

        return y2d.reshape(orig_shape)


def _create_layernorm(hidden_size: int, eps: float, *, use_fallback: bool) -> nn.Module:
    """LayerNorm 모듈을 만듭니다.

    정책:
    - use_fallback=True:
        - flash-attn layer_norm import 실패해도 PyTorch로 동작
    - use_fallback=False:
        - import 실패 상태면 여기서 즉시 에러

    Args:
        hidden_size: 마지막 차원 D
        eps: 작은 값
        use_fallback: 폴백 허용 여부

    Returns:
        nn.Module: LayerNorm 동작 모듈
    """
    _require_flash_component_or_raise(
        "layer_norm",
        available=_FLASH_LAYER_NORM_AVAILABLE and (_flash_layer_norm_fn is not None),
        use_fallback=bool(use_fallback),
        what="flash-attn layer_norm",
    )
    return _FlashLayerNorm(int(hidden_size), eps=float(eps))


class FastLayerNorm(nn.Module):
    """LayerNorm 래퍼.

    - use_fallback=True: flash-attn이 없으면 PyTorch로 동작
    - use_fallback=False: flash-attn import가 안 된 상태면 생성 시점에 에러
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, *, use_fallback: bool = True) -> None:
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)
        self.use_fallback: bool = bool(use_fallback)

        self._ln: nn.Module = _create_layernorm(
            hidden_size=self.normalized_shape,
            eps=self.eps,
            use_fallback=self.use_fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ln(x)


class _TorchMlp(nn.Module):
    """PyTorch 기본 연산으로 만든 2단 MLP (폴백용).

    구성:
    - 선형 변환 1번
    - 활성 함수 1번
    - 선형 변환 1번

    입력/출력 모양:
    - 입력:  (M, in_features)
    - 출력: (M, out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        activation: str = "gelu_approx",
        checkpoint_lvl: int = 0,
        return_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features)
        self.out_features: int = int(out_features)
        self.checkpoint_lvl: int = int(checkpoint_lvl)
        self.return_residual: bool = bool(return_residual)

        self.fc1: nn.Linear = nn.Linear(self.in_features, self.hidden_features, bias=True)
        self.act: nn.Module = _create_activation(str(activation))
        self.fc2: nn.Linear = nn.Linear(self.hidden_features, self.out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"_TorchMlp expects 2D (M,D). got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_features:
            raise ValueError(f"_TorchMlp: last dim must be {self.in_features}. got {int(x.shape[1])}")

        y = self.fc2(self.act(self.fc1(x)))  # (M, out_features)

        if self.return_residual:
            raise RuntimeError("_TorchMlp fallback does not support return_residual=True in this project.")

        return y


def _create_mlp_backend(
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    activation: str,
    checkpoint_lvl: int,
    use_fallback: bool,
) -> nn.Module:
    """MLP 백엔드를 만듭니다.

    정책:
    - flash-attn FusedMLP import 성공:
        - 항상 flash-attn 모듈 사용
    - import 실패:
        - use_fallback=True -> PyTorch MLP로 진행
        - use_fallback=False -> 생성 시점에 에러

    Args:
        in_features: 입력 차원 D
        hidden_features: 중간 차원 H
        out_features: 출력 차원 O
        activation: 활성 함수 이름
        checkpoint_lvl: flash-attn 쪽 옵션
        use_fallback: 폴백 허용 여부

    Returns:
        nn.Module: (M, D) -> (M, O)
    """
    if _FLASH_FUSED_MLP_AVAILABLE and _FlashFusedMLP is not None:
        return _FlashFusedMLP(  # type: ignore[call-arg]
            in_features=int(in_features),
            hidden_features=int(hidden_features),
            out_features=int(out_features),
            activation=str(activation),
            return_residual=False,
            checkpoint_lvl=int(checkpoint_lvl),
        )

    _require_flash_component_or_raise(
        "fused_mlp",
        available=False,
        use_fallback=bool(use_fallback),
        what="flash-attn FusedMLP",
    )

    return _TorchMlp(
        in_features=int(in_features),
        hidden_features=int(hidden_features),
        out_features=int(out_features),
        activation=str(activation),
        checkpoint_lvl=int(checkpoint_lvl),
        return_residual=False,
    )


class FastMlp(nn.Module):
    """MLP 래퍼.

    - use_fallback=True: flash-attn이 없으면 PyTorch로 동작
    - use_fallback=False: flash-attn import가 안 된 상태면 생성 시점에 에러

    Shape:
    - Input:  (..., in_features)
    - Output: (..., out_features)
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
        use_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features) if hidden_features is not None else int(in_features)
        self.out_features: int = int(out_features) if out_features is not None else int(in_features)
        self.drop_p: float = float(drop)
        self.use_fallback: bool = bool(use_fallback)

        self._mlp: nn.Module = _create_mlp_backend(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            activation=str(activation),
            checkpoint_lvl=int(checkpoint_lvl),
            use_fallback=self.use_fallback,
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
    """LayerNorm -> MLP를 이어서 수행하는 모듈."""

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
        use_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: int = int(hidden_features)
        self.out_features: int = int(out_features)
        self.use_fallback: bool = bool(use_fallback)

        self.norm: FastLayerNorm = FastLayerNorm(self.in_features, eps=float(eps), use_fallback=self.use_fallback)
        self.mlp: FastMlp = FastMlp(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            drop=float(drop),
            activation=str(activation),
            checkpoint_lvl=int(checkpoint_lvl),
            use_fallback=self.use_fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class MixerBlock(nn.Module):
    """토큰 길이(T) 방향과 채널(C) 방향을 번갈아 섞는 블록.

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
        use_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.use_fallback: bool = bool(use_fallback)

        # (A) Token-axis mixing: LN (over C) + MLP over T (applied on (N, C, T))
        self.norm1: FastLayerNorm = FastLayerNorm(int(channels_mlp_dim), use_fallback=self.use_fallback)
        self.tokens_mlp: FastMlp = FastMlp(
            in_features=int(tokens_mlp_dim),
            hidden_features=int(tokens_mlp_dim),
            out_features=int(tokens_mlp_dim),
            drop=float(drop_path_rate),
            use_fallback=self.use_fallback,
        )

        # (B) Channel-axis mixing: LN + MLP
        hidden_c: int = max(16, int(float(channels_mlp_dim) * float(channels_mlp_ratio)))
        self.channels_norm_mlp: FastLayerNormMlp = FastLayerNormMlp(
            in_features=int(channels_mlp_dim),
            hidden_features=int(hidden_c),
            out_features=int(channels_mlp_dim),
            drop=float(drop_path_rate),
            use_fallback=self.use_fallback,
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

        # (2) Channel-axis mixing
        return x + self.channels_norm_mlp(x)
