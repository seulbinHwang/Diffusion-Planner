# diffusion_planner/model/module/mixer.py (REPLACE THE WHOLE FILE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ============================================================
# Prefer flash-attn fused LayerNorm / fused MLP
# - If unavailable, fall back to plain PyTorch (no import-time error)
# ============================================================

_FLASH_IMPORT_ERRORS: Dict[str, str] = {}
_FLASH_FUSED_MLP_AVAILABLE: bool = False
_FLASH_LAYER_NORM_AVAILABLE: bool = False

# (1) FusedMLP (flash-attn)
try:
    from flash_attn.ops.fused_dense import FusedMLP as _FlashFusedMLP  # type: ignore
    _FLASH_FUSED_MLP_AVAILABLE = True
except Exception as e:
    raise ImportError(
        "Required: cannot import flash_attn.ops.fused_dense.FusedMLP. "
        "This project must use FastMlp (no fallback). "
        f"(cause: {repr(e)})"
    ) from e
    _FlashFusedMLP = None  # type: ignore
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
        raise ImportError(
            "Required: cannot import flash-attn layer_norm function. "
            "This project must use FastLayerNorm (no fallback). "
            f"(cause1: {repr(e1)} / cause2: {repr(e2)})"
        ) from e2
        _flash_layer_norm_fn = None
        _FLASH_LAYER_NORM_AVAILABLE = False
        _FLASH_IMPORT_ERRORS["layer_norm"] = f"cause1: {repr(e1)} / cause2: {repr(e2)}"


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
        # PyTorch 버전에 따라 approximate 옵션이 없을 수도 있어 안전하게 처리
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
    """flash-attn의 layer_norm(함수)을 nn.Module처럼 쓰기 위한 래퍼(폴백 포함).

    동작:
        - flash-attn layer_norm 함수가 있고,
          입력이 CUDA + (fp16/bf16)일 때: flash-attn 함수 사용
        - 그 외(패키지 없음/CPU/fp32 등): PyTorch layer_norm으로 폴백

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

    @staticmethod
    def _can_use_flash(x: torch.Tensor) -> bool:
        """현재 입력에서 flash-attn layer_norm을 안전하게 쓸 수 있는지 확인합니다."""
        if not _FLASH_LAYER_NORM_AVAILABLE:
            return False
        if _flash_layer_norm_fn is None:
            return False
        if not x.is_cuda:
            return False
        return x.dtype in (torch.float16, torch.bfloat16)

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

        # 일부 구현은 2D 입력을 기대할 수 있어 안전하게 펼쳤다가 복원
        x2d = x.reshape(-1, D).contiguous()  # (M, D)

        # weight/bias는 입력 dtype/device에 맞춰 사용
        w = self.weight.to(device=x2d.device, dtype=x2d.dtype).contiguous()  # (D,)
        b = self.bias.to(device=x2d.device, dtype=x2d.dtype).contiguous()    # (D,)

        if self._can_use_flash(x2d):
            # flash-attn layer_norm: 보통 (x, weight, bias, epsilon) 형태
            y2d = _flash_layer_norm_fn(x2d, w, b, self.eps)  # type: ignore[misc]  # (M, D)
        else:
            # PyTorch 폴백
            y2d = F.layer_norm(x2d, (D,), weight=w, bias=b, eps=self.eps)  # (M, D)

        return y2d.reshape(orig_shape)


def _create_layernorm(hidden_size: int, eps: float) -> nn.Module:
    """LayerNorm 모듈을 만듭니다.

    - 가능하면 flash-attn 기반을 쓰고,
    - 불가능하면 PyTorch 구현으로 폴백합니다.

    Args:
        hidden_size (int): 마지막 차원 크기 D
        eps (float): 작은 값

    Returns:
        nn.Module: LayerNorm 동작을 하는 모듈
    """
    # flash-attn이 없더라도 _FlashLayerNorm 내부에서 자동 폴백되지만,
    # 패키지가 아예 없을 때도 안정적으로 동작하도록 여기서도 분기 가능.
    return _FlashLayerNorm(int(hidden_size), eps=float(eps))


class FastLayerNorm(nn.Module):
    """LayerNorm 래퍼(폴백 포함).

    - flash-attn layer_norm이 준비되어 있으면 자동으로 사용
    - 없으면 PyTorch layer_norm으로 자동 폴백

    Shape:
        - Input:  (..., D)
        - Output: (..., D)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)
        self._ln: nn.Module = _create_layernorm(
            hidden_size=self.normalized_shape,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ln(x)


class _TorchMlp(nn.Module):
    """PyTorch 기본 연산으로 만든 2단 MLP (flash-attn 미사용 폴백).

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
        """
        Args:
            in_features (int): 입력 마지막 차원 크기
            hidden_features (int): 중간 차원 크기
            out_features (int): 출력 마지막 차원 크기
            activation (str): 활성 함수 이름
            checkpoint_lvl (int): 호환용 인자(여기서는 0만 권장). 0이 아니어도 동작은 합니다.
            return_residual (bool): 호환용 인자(이 파일에서는 False만 사용).
        """
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
        """(M, in_features) -> (M, out_features)"""
        if x.dim() != 2:
            raise ValueError(f"_TorchMlp expects 2D (M,D). got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_features:
            raise ValueError(f"_TorchMlp: last dim must be {self.in_features}. got {int(x.shape[1])}")

        y = self.fc2(self.act(self.fc1(x)))  # (M, out_features)

        # 이 파일의 사용 방식에서는 return_residual=False만 사용합니다.
        # (True일 때의 반환 형태를 강제로 맞추면, 호출부가 바뀌어야 해서 여기서는 지원하지 않습니다.)
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
) -> nn.Module:
    """MLP 백엔드를 만듭니다.

    - flash-attn FusedMLP가 있으면: 그걸 사용
    - 없으면: PyTorch MLP로 폴백

    Args:
        in_features (int): 입력 차원 D
        hidden_features (int): 중간 차원 H
        out_features (int): 출력 차원 O
        activation (str): 활성 함수 이름
        checkpoint_lvl (int): flash-attn 쪽 옵션(폴백에서는 유지용 인자)

    Returns:
        nn.Module: (M, D) -> (M, O) 를 수행하는 모듈
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

    return _TorchMlp(
        in_features=int(in_features),
        hidden_features=int(hidden_features),
        out_features=int(out_features),
        activation=str(activation),
        checkpoint_lvl=int(checkpoint_lvl),
        return_residual=False,
    )


class FastMlp(nn.Module):
    """MLP 래퍼(폴백 포함).

    Key behavior:
        - 입력을 2D로 펼쳐서 (M, D) 형태로 만든 뒤 MLP 호출
        - 다시 원래 모양으로 복원

    Backend:
        - flash-attn FusedMLP가 있으면 자동 사용
        - 없으면 PyTorch 연산으로 폴백

    Shape:
        - Input:  (..., in_features)
        - Output: (..., out_features)

    Note:
        - drop > 0 이면 출력에 dropout을 적용합니다.
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

        self._mlp: nn.Module = _create_mlp_backend(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            activation=str(activation),
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
    """한 번에 LayerNorm -> MLP를 이어서 수행하는 모듈(폴백 포함).

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

        # (B) Channel-axis mixing: LN + MLP
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

        # (2) Channel-axis mixing
        return x + self.channels_norm_mlp(x)
