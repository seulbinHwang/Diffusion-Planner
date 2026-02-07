import math
import torch
import torch.nn as nn
from typing import Dict
from timm.models.layers import Mlp
from typing import Tuple, Optional
from flash_attn.bert_padding import unpad_input, pad_input
from diffusion_planner.model.module.pram_v2 import ModulationTriplet
from diffusion_planner.model.module.pram_v2 import (
    apply_pram_v2_path_modulation,
    style_queries_for_cross_attention,
)
from typing import NamedTuple

import torch.nn.functional as F
# ===== FlashAttention-2 varlen import (2.x 표준 경로 + 백업 경로) =====
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    try:
        # 일부 버전(옛 코드) 표기
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_q_kvpacked_func as flash_attn_varlen_cross_func)
    except ImportError:
        # flash-attn 2.8.x의 정식 이름
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_kvpacked_func as flash_attn_varlen_cross_func)
    _FA2_AVAILABLE = True
    _FA2_IMPORT_ERR = None
except Exception as _e1:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        try:
            from flash_attn import (flash_attn_varlen_q_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        except ImportError:
            from flash_attn import (flash_attn_varlen_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        _FA2_AVAILABLE = True
        _FA2_IMPORT_ERR = None
    except Exception as _e2:
        _FA2_AVAILABLE = False
        _FA2_IMPORT_ERR = Exception(
            f"interface import err: {_e1}; top-level err: {_e2}")
        flash_attn_varlen_cross_func = None

# ===== fused MLP (Linear -> GELU -> Linear) hard-require =====
try:
    # flash-attn 2.8.x 환경에서 흔히 존재
    from flash_attn.ops.fused_dense import fused_mlp_func, FusedMLP
except Exception as _e_fused_mlp:
    raise RuntimeError(
        "This code path requires fused MLP only.\n"
        "Failed to import flash_attn.ops.fused_dense.fused_mlp_func / FusedMLP.\n"
        "Please install/build flash-attn with fused_dense support and retry.\n"
        f"(cause: {_e_fused_mlp})"
    ) from _e_fused_mlp

def _fused_linear_gelu_linear(
    x2d: torch.Tensor,  # (N, Din)
    w1: torch.Tensor,  # (H, Din)
    b1: Optional[torch.Tensor],  # (H,) or None
    w2: torch.Tensor,  # (Dout, H)
    b2: Optional[torch.Tensor],  # (Dout,) or None
    is_training: bool,
) -> torch.Tensor:
    """fused_mlp_func로 Linear -> GELU(근사) -> Linear을 수행합니다."""
    act = "gelu_approx"

    try:
        return fused_mlp_func(
            x2d,
            w1,
            w2,
            b1,
            b2,
            activation=act,
            save_pre_act=is_training,
            return_residual=False,
        )
    except TypeError:
        # 환경에 따라 일부 키워드가 없을 수 있어 단계적으로 줄여서 재시도
        try:
            return fused_mlp_func(
                x2d,
                w1,
                w2,
                b1,
                b2,
                activation=act,
                save_pre_act=is_training,
            )
        except TypeError:
            try:
                return fused_mlp_func(
                    x2d,
                    w1,
                    w2,
                    b1,
                    b2,
                    activation=act,
                )
            except Exception as e:
                raise RuntimeError(
                    "fused_mlp_func call failed. "
                    "Your flash-attn build may have an incompatible fused_dense interface."
                    f" (cause: {e})"
                ) from e


def _require_fused_mlp_ready(
    x2d: torch.Tensor,  # (N, Din)
    fc1: nn.Linear,
    fc2: nn.Linear,
    *,
    strict_param_dtype: bool = False,
) -> None:
    if x2d.dim() != 2:
        raise RuntimeError(f"Fused MLP input must be 2D (N, Din). Got {tuple(x2d.shape)}")

    if not x2d.is_cuda:
        raise RuntimeError("Fused MLP is hard-required to run on CUDA (GPU) only.")

    if x2d.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Fused MLP is hard-required to use fp16/bf16 only. "
            f"Current dtype={x2d.dtype}. Use autocast or cast inputs."
        )

    params = [("fc1.weight", fc1.weight), ("fc2.weight", fc2.weight)]
    if fc1.bias is not None:
        params.append(("fc1.bias", fc1.bias))
    if fc2.bias is not None:
        params.append(("fc2.bias", fc2.bias))

    for name, p in params:
        if not p.is_cuda:
            raise RuntimeError(f"Fused MLP parameter {name} is not on CUDA.")

        # ✅ 여기만 조건부로: strict_param_dtype=False면 dtype 불일치 체크를 안 함
        if strict_param_dtype and (p.dtype != x2d.dtype):
            raise RuntimeError(
                f"Fused MLP parameter {name} dtype({p.dtype}) does not match input dtype({x2d.dtype})."
            )



def _cast_linear_params_like_ref(
    fc: nn.Linear,
    ref: torch.Tensor,
) -> None:
    """Linear 파라미터(weight/bias)를 ref 텐서와 같은 dtype/device로 맞춥니다.

    주의:
        - 이 함수는 파라미터의 "저장 dtype" 자체를 바꿉니다.
        - 보통은 optimizer 생성 전에 1회 맞추는 게 안전합니다.

    Args:
        fc (nn.Linear): 대상 Linear 레이어.
        ref (torch.Tensor): 기준 텐서.
            - shape: (N, Din) 또는 (B, L, Din) 등 (모양은 상관 없음)
            - dtype/device: 이 값으로 파라미터를 맞춥니다.
    """
    target_device: torch.device = ref.device
    target_dtype: torch.dtype = ref.dtype

    # weight
    if (fc.weight.device != target_device) or (fc.weight.dtype != target_dtype):
        fc.weight.data = fc.weight.data.to(device=target_device, dtype=target_dtype)
        if fc.weight.grad is not None:
            fc.weight.grad = None

    # bias
    if fc.bias is not None:
        if (fc.bias.device != target_device) or (fc.bias.dtype != target_dtype):
            fc.bias.data = fc.bias.data.to(device=target_device, dtype=target_dtype)
            if fc.bias.grad is not None:
                fc.bias.grad = None


class FusedMlpGelu(nn.Module):
    """flash-attn fused_mlp_func만 사용하는 Linear -> GELU -> Linear 모듈."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        auto_align_param_dtype: bool = False,
    ) -> None:
        """초기화.

        Args:
            in_features: 입력 차원(Din).
            hidden_features: 중간 차원(H).
            out_features: 출력 차원(Dout). None이면 Din.
            bias: bias 사용 여부.
            auto_align_param_dtype:
                True면 forward에서 입력 x의 dtype/device에 맞춰
                fc1/fc2 파라미터 저장 dtype/device를 자동으로 맞춥니다.
        """
        super().__init__()
        out_features = int(in_features) if out_features is None else int(out_features)

        self.fc1 = nn.Linear(int(in_features), int(hidden_features), bias=bias)
        self.fc2 = nn.Linear(int(hidden_features), int(out_features), bias=bias)

        self._auto_align_param_dtype = bool(auto_align_param_dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fused Linear->GELU->Linear 실행.

        Args:
            x (torch.Tensor):
                - shape: (N, Din) 또는 (B, L, Din)
                - device: CUDA
                - dtype: fp16 또는 bf16

        Returns:
            torch.Tensor:
                - shape: (N, Dout) 또는 (B, L, Dout)
        """
        if x.dim() == 2:
            x2d = x  # (N, Din)
            restore_shape = None
        elif x.dim() == 3:
            B, L, Din = x.shape
            x2d = x.reshape(B * L, Din)  # (B*L, Din)
            restore_shape = (B, L)
        else:
            raise RuntimeError(f"Fused MLP supports only 2D/3D inputs. Got {tuple(x.shape)}")

        # (선택) 파라미터 저장 dtype/device를 입력과 동일하게 맞춤
        if self._auto_align_param_dtype:
            _cast_linear_params_like_ref(self.fc1, x2d)
            _cast_linear_params_like_ref(self.fc2, x2d)

        # 여기서는 dtype 불일치를 허용하지 않도록 강제하는 걸 추천
        _require_fused_mlp_ready(x2d, self.fc1, self.fc2, strict_param_dtype=True)

        # Empty input can fail in fused kernel; handle safely while keeping graph "touched".
        N = int(x2d.shape[0])
        Dout = int(self.fc2.out_features)
        if N == 0:
            out2d = x2d.new_zeros((0, Dout))  # (0, Dout)
            touch = (
                self.fc1.weight.view(-1)[:1].sum()
                + (self.fc1.bias.view(-1)[:1].sum()
                   if self.fc1.bias is not None else 0.0)
                + self.fc2.weight.view(-1)[:1].sum()
                + (self.fc2.bias.view(-1)[:1].sum()
                   if self.fc2.bias is not None else 0.0)
            ) * 0.0
            out2d = out2d + touch
        else:
            # Fused kernels often prefer contiguous inputs/weights.
            x2d = x2d.contiguous()  # (N, Din)
            w1 = self.fc1.weight.contiguous()  # (H, Din)
            b1 = self.fc1.bias.contiguous(
            ) if self.fc1.bias is not None else None  # (H,) or None
            w2 = self.fc2.weight.contiguous()  # (Dout, H)
            b2 = self.fc2.bias.contiguous(
            ) if self.fc2.bias is not None else None  # (Dout,) or None

            # Hard-require fused path only (via fused_mlp_func).
            out2d = _fused_linear_gelu_linear(
                x2d=x2d,
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                is_training=self.training,
            )  # (N, Dout)

        if restore_shape is None:
            return out2d  # (N, Dout)

        B, L = restore_shape
        return out2d.reshape(B, L, Dout)  # (B, L, Dout)


# ===========================================================
class FlashAttnKVCache(NamedTuple):
    """Cross-Attention에서 K/V 쪽(scene 토큰)을 한 번만 펼쳐서 재사용하기 위한 캐시.

    Attributes:
        kv_unpad (torch.Tensor): (Tk, D) 유효한 scene 토큰만 모은 텐서
        cu_seqlens_k (torch.Tensor): (B+1,) 각 배치의 누적 길이(int32)
        max_seqlen_k (int): 배치 내 최대 유효 길이
    """
    kv_unpad: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_k: int


def modulate(
    x: torch.Tensor,  # (B, L, D)
    shift: torch.Tensor,  # (B, D) 또는 (B, L, D)
    scale: torch.Tensor,  # (B, D) 또는 (B, L, D)
    only_first: bool = False,
) -> torch.Tensor:
    """브로드캐스트 친화적 adaLN 모듈레이션.

    전역/에이전트별 모듈레이션 벡터를 x에 적용합니다.
    shift/scale는 (B, D) 또는 (B, L, D) 형태를 모두 허용합니다.

    Args:
        x: (B, L, D) 입력 시퀀스 임베딩.
        shift: (B, D) 또는 (B, L, D). 이동(shift) 항.
        scale: (B, D) 또는 (B, L, D). 스케일(scale) 항.
        only_first: True면 첫 토큰에만 모듈레이션을 적용합니다.

    Returns:
        torch.Tensor: (B, L, D) 모듈레이션 적용 결과.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (B, L, D), got {tuple(x.shape)}")

    # (B, D) → (B, 1, D)로 승격하여 L축으로 브로드캐스트
    if shift.dim() == 2 and scale.dim() == 2:
        shift = shift.unsqueeze(1)  # (B, 1, D)
        scale = scale.unsqueeze(1)  # (B, 1, D)
    elif shift.dim() == 3 and scale.dim() == 3:
        # 이미 (B, L, D)
        pass
    else:
        raise ValueError(
            f"shift/scale must be (B,D) or (B,L,D); "
            f"got shift={tuple(shift.shape)}, scale={tuple(scale.shape)}")
    # ★ 추가: dtype/device 정렬
    shift = shift.to(dtype=x.dtype, device=x.device)
    scale = scale.to(dtype=x.dtype, device=x.device)

    if only_first:
        x_first = x[:, :1] * (1 + scale[:, :1]) + shift[:, :1]
        x_rest = x[:, 1:]
        return torch.cat([x_first, x_rest], dim=1)
    else:
        return x * (1 + scale) + shift


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

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = float(max_period)

        # -----------------------------
        # ✅ 방법 1) freqs를 1번만 만들고 재사용 (register_buffer)
        # -----------------------------
        half: int = int(self.frequency_embedding_size // 2)
        if half > 0:
            exponent = (-math.log(self.max_period)) * (
                torch.arange(start=0, end=half, dtype=torch.float32) /
                float(half))  # (half,)
            freqs = torch.exp(exponent)  # (half,) float32
        else:
            freqs = torch.empty((0,), dtype=torch.float32)

        # persistent=False: 체크포인트 호환(새 buffer로 인해 strict 로딩 실패) 위험 줄이기
        self.register_buffer("_freqs", freqs, persistent=False)  # (half,)

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
        # (원본 호환 유지용: 외부에서 static 호출할 가능성 대비)
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

    def _timestep_embedding_cached(self, t: torch.Tensor) -> torch.Tensor:
        """시간값 t를 sin/cos 임베딩으로 바꿉니다(캐시된 freqs 사용).

        Args:
            t (torch.Tensor):
                시간값 텐서.
                - shape: (N,)
                - dtype: 무관 (내부에서 float32로 계산)

        Returns:
            torch.Tensor:
                sin/cos 임베딩.
                - shape: (N, frequency_embedding_size)
                - dtype: float32
                - device: t.device
        """
        if t.dim() != 1:
            raise ValueError(f"t must be 1D (N,), got {tuple(t.shape)}")

        N: int = int(t.shape[0])
        half: int = int(self.frequency_embedding_size // 2)

        if half == 0:
            # 거의 안 쓰는 케이스지만 안전하게 처리
            return torch.zeros((N, self.frequency_embedding_size),
                               device=t.device,
                               dtype=torch.float32)

        freqs: torch.Tensor = self._freqs
        if freqs.device != t.device:
            freqs = freqs.to(device=t.device)

        args = t[:, None].float() * freqs[None, :]  # (N, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)],
                        dim=-1)  # (N, 2*half)

        if (self.frequency_embedding_size % 2) == 1:
            pad = torch.zeros((N, 1), device=t.device, dtype=emb.dtype)  # (N,1)
            emb = torch.cat([emb, pad], dim=-1)  # (N, 2*half+1)

        return emb  # float32

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        if t.dim() != 1:
            raise ValueError(f"t must be 1D (B,), got {tuple(t.shape)}")

        B: int = int(t.shape[0])
        if B == 0:
            # (0, H)
            out_dim: int = int(self.mlp[-1].out_features)
            return torch.zeros((0, out_dim),
                               device=t.device,
                               dtype=torch.float32)

        # -----------------------------
        # ✅ 방법 2) 배치의 t가 전부 같은 경우: 1번만 계산 + expand
        #   - GPU 동기화 없이 처리(마스크로 나머지만 계산)
        # -----------------------------
        t0 = t[:1]  # (1,)
        same_mask = (t == t0)  # (B,) bool

        # (1) 첫 값 임베딩 1회
        t0_freq = self._timestep_embedding_cached(t0)  # (1, F)
        t0_emb = self.mlp(t0_freq)  # (1, H)

        # (2) 다른 값이 있는 샘플만 추가 계산
        t_other = t[~same_mask]  # (N_other,)
        if t_other.numel() == 0:
            return t0_emb.expand(B, -1)  # (B, H)

        t_other_freq = self._timestep_embedding_cached(t_other)  # (N_other, F)
        t_other_emb = self.mlp(t_other_freq)  # (N_other, H)

        out = t0_emb.expand(B, -1).clone()  # (B, H)
        out[~same_mask] = t_other_emb
        return out


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    def __init__(self, dim=192, heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, f"dim({dim}) must be divisible by heads({heads})"
        if (self.head_dim % 8) != 0:
            # FlashAttention-2는 FP16/BF16에서 head_dim이 8의 배수일 때 최적화가 가장 좋습니다.
            print(f"[Warning] head_dim={self.head_dim} is not a multiple of 8; "
                  f"FlashAttention-2 성능이 저하될 수 있습니다.")
            raise RuntimeError("head_dim이 8의 배수가 되도록 dim 또는 heads를 조정하세요.")

        # === 원본 LayerNorm/MLP/게이팅은 그대로 유지 ===
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # ✅ fused MLP로 강제 교체 (Linear -> GELU -> Linear)
        self.mlp1 = FusedMlpGelu(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            bias=True,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        # ✅ fused MLP로 강제 교체 (Linear -> GELU -> Linear)
        self.mlp2 = FusedMlpGelu(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            bias=True,
        )
        # === FlashAttention-2용 프로젝션 (Self-Attn) ===
        # QKV/Out은 패딩 제거된 유효 토큰에만 적용됩니다.
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # === FlashAttention-2용 프로젝션 (Cross-Attn) ===
        self.q_proj_cross = nn.Linear(dim, dim, bias=True)
        self.kv_proj_cross = nn.Linear(dim, 2 * dim, bias=True)
        self.out_proj_cross = nn.Linear(dim, dim, bias=True)

        # === 게이트(원본 유지) ===
        self.gate_mlp2 = nn.Parameter(torch.tensor(0.0))

        # Dropout 확률(Train일 때만 FA2에 전달)
        self._attn_dropout_p = dropout

    # ====================== 유틸/헬퍼 함수들 ======================

    @staticmethod
    def _check_flash_available() -> None:
        """FlashAttention‑2 설치 여부 확인 및 친절한 에러 메시지.

        Raises:
            RuntimeError: FlashAttention‑2를 불러오지 못한 경우.
        """
        if not _FA2_AVAILABLE:
            raise RuntimeError("FlashAttention‑2(varlen) 모듈을 불러오지 못했습니다. "
                               "pip install flash-attn>=2.3 등으로 설치 후 다시 시도하세요. "
                               f"(원인: {_FA2_IMPORT_ERR})")

    @staticmethod
    def _get_compute_dtype(x: torch.Tensor) -> torch.dtype:
        """연산 dtype을 선택합니다(BF16/FP16 우선).

        1) autocast가 켜져 있으면 해당 dtype을 우선 사용합니다.
        2) 입력 텐서 `x`가 FP16/BF16이면 그대로 사용합니다.
        3) 그 외에는 GPU 아키텍처에 따라 BF16(>=SM80) 또는 FP16을 반환합니다.

        Args:
            x (torch.Tensor): 임의 텐서. (shape 무관)

        Returns:
            torch.dtype: 연산에 사용할 dtype (torch.float16 또는 torch.bfloat16).
        """
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                pass

        if x.dtype in (torch.float16, torch.bfloat16):
            return x.dtype

        if x.is_cuda and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(x.device)
            return torch.bfloat16 if major >= 8 else torch.float16

        return torch.float16

    def _self_attn_flash_varlen(
            self,
            x: torch.Tensor,  # (B, L, D)
            attn_mask: torch.Tensor,  # (B, L)  True=pad
    ) -> torch.Tensor:
        """FlashAttention‑2(varlen) 기반 Self‑Attention.

        패딩 토큰을 제거(unpad)한 뒤, **varlen self‑attention** 커널로 계산하고,
        최종 결과를 원래 배치 모양으로 복원합니다. Q/K/V 및 출력 프로젝션은
        **유효 토큰에만** 적용되어 연산/메모리를 함께 절약합니다.

        Args:
            x (torch.Tensor): 쿼리=키=값 입력. 모양 (B, L, D).
            attn_mask (torch.Tensor): 키 패딩 마스크(True=pad). 모양 (B, L).

        Returns:
            torch.Tensor: Self‑Attention 출력. 모양 (B, L, D).

        Raises:
            RuntimeError: FlashAttention‑2 모듈이 없는 경우.
        """
        self._check_flash_available()
        B, L, D = x.shape

        # (1) 언패드
        # x_unpad, idx, cu, max_len, seqlens = self._unpad_from_mask(
        #     x, attn_mask)  # x_unpad: (T, D)
        attention_mask = (~attn_mask).to(torch.bool)  # True=valid

        res = unpad_input(x, attention_mask)

        # v2.7 이하: 4개 / v2.8.x: 5개
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        T = x_unpad.shape[0]
        if T == 0 or max_seqlen == 0:
            touch = (self.qkv_proj.weight.view(-1)[:1].sum() +
                     (self.qkv_proj.bias.view(-1)[:1].sum()
                      if self.qkv_proj.bias is not None else 0) +
                     self.out_proj.weight.view(-1)[:1].sum() +
                     (self.out_proj.bias.view(-1)[:1].sum()
                      if self.out_proj.bias is not None else 0)) * 0.0
            return torch.zeros_like(x) + touch

        # (2) QKV 프로젝션 (유효 토큰만)
        qkv = self.qkv_proj(x_unpad)  # (T, 3*D)
        qkv = qkv.reshape(T, 3, self.num_heads, self.head_dim)  # (T, 3, H, Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        # (3) FlashAttention‑2 varlen (Self-Attn)
        # out: (T, H, Hd)
        out = flash_attn_varlen_qkvpacked_func(
            qkv,  # (T, 3, H, Hd)
            cu_seqlens=cu_seqlens.to(torch.int32),
            max_seqlen=max_seqlen,  # int  배치 내 최대 유효 길이.
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (T, H, Hd)

        # (4) 출력 프로젝션 + pad back
        out = out.reshape(T, self.num_heads * self.head_dim)  # (T, D)
        out = self.out_proj(out.to(x.dtype))  # (T, D) -> 원 dtype
        out = pad_input(out, indices, B, L)  # (B, L, D)
        return out

    def _cross_attn_flash_varlen(
            self,
            q_in: torch.Tensor,  # (B, Lq, D)
            kv_in: torch.Tensor,  # (B, Lk, D)
            q_mask: torch.Tensor,  # (B, Lq) True=pad
            kv_mask: torch.Tensor,  # (B, Lk) True=pad
    ) -> torch.Tensor:
        """FlashAttention‑2(varlen) 기반 Cross‑Attention.

        Q(이웃 시퀀스)와 K/V(장면 컨텍스트)를 각각 언패드한 뒤,
        **varlen Q vs varlen KV** 커널로 계산합니다. Q/K/V 및 출력 프로젝션은
        **유효 토큰에만** 적용되어 연산과 메모리를 효율화합니다.

        Args:
            q_in (torch.Tensor): 쿼리 입력(이웃). 모양 (B, Lq, D).
            kv_in (torch.Tensor): 키/값 입력(컨텍스트). 모양 (B, Lk, D).
            q_mask (torch.Tensor): Q 패딩 마스크(True=pad). 모양 (B, Lq).
            kv_mask (torch.Tensor): K/V 패딩 마스크(True=pad). 모양 (B, Lk).

        Returns:
            torch.Tensor: Cross‑Attention 출력(Q 자리수). 모양 (B, Lq, D).

        Raises:
            RuntimeError: FlashAttention‑2 모듈이 없는 경우.

        Note:
            - 모든 토큰이 마스크된 배치(전부 pad)인 경우 (B, Lq, D) 영 텐서를 반환합니다.
            - 드롭아웃은 학습 시에만 활성화됩니다.
        """
        # TODO: cpu 만 사용가능할 때, PyTorch SDPA(패딩 포함) 사용하는 옵션 추가
        self._check_flash_available()
        B, Lq, D = q_in.shape
        _, Lk, _ = kv_in.shape

        # (1) 언패드(Q, KV 각각)
        q_mask_valid = (~q_mask).to(torch.bool)
        kv_mask_valid = (~kv_mask).to(torch.bool)
        res = unpad_input(q_in, q_mask_valid)  # (Tq, D)
        # v2.7 이하: 4개 / v2.8.x: 5개
        if len(res) == 4:
            q_unpad, q_idx, cu_q, max_q = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_q[1:] - cu_q[:-1]).to(torch.int32)
        elif len(res) == 5:
            q_unpad, q_idx, cu_q, max_q, seqlens = res
        res = unpad_input(kv_in, kv_mask_valid)  # (Tk, D)
        # v2.7 이하: 4개 / v2.8.x: 5개
        if len(res) == 4:
            kv_unpad, indices, cu_k, max_k = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_k[1:] - cu_k[:-1]).to(torch.int32)
        elif len(res) == 5:
            kv_unpad, indices, cu_k, max_k, seqlens = res

        if q_unpad.numel() == 0 or kv_unpad.numel(
        ) == 0 or max_q == 0 or max_k == 0:
            out_zeros = torch.zeros(B,
                                    Lq,
                                    D,
                                    device=q_in.device,
                                    dtype=q_in.dtype)
            touch = (self.q_proj_cross.weight.view(-1)[:1].sum() +
                     (self.q_proj_cross.bias.view(-1)[:1].sum()
                      if self.q_proj_cross.bias is not None else 0) +
                     self.kv_proj_cross.weight.view(-1)[:1].sum() +
                     (self.kv_proj_cross.bias.view(-1)[:1].sum()
                      if self.kv_proj_cross.bias is not None else 0) +
                     self.out_proj_cross.weight.view(-1)[:1].sum() +
                     (self.out_proj_cross.bias.view(-1)[:1].sum()
                      if self.out_proj_cross.bias is not None else 0)) * 0.0
            return out_zeros + touch

        Tq = q_unpad.shape[0]
        Tk = kv_unpad.shape[0]

        # (2) Q / KV 프로젝션 (유효 토큰만)
        q = self.q_proj_cross(q_unpad).reshape(Tq, self.num_heads,
                                               self.head_dim)  # (Tq, H, Hd)
        kv = self.kv_proj_cross(kv_unpad).reshape(
            Tk, 2, self.num_heads, self.head_dim)  # (Tk, 2, H, Hd)
        comp_dtype = self._get_compute_dtype(q)
        q = q.to(comp_dtype)
        kv = kv.to(comp_dtype)

        # (3) FlashAttention‑2 varlen (Cross-Attn)
        # out: (Tq, H, Hd)
        out = flash_attn_varlen_cross_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (Tq, H, Hd)

        # (4) 출력 프로젝션 + pad back
        out = out.reshape(Tq, self.num_heads * self.head_dim)  # (Tq, D)
        out = self.out_proj_cross(out.to(q_in.dtype))  # (Tq, D)
        # (5) 패드 복원(★ Q 인덱스 사용)
        out = pad_input(out, q_idx, B, Lq)  # (B, Lq, D)
        return out

    # ====================== (추가) 함수화된 per‑agent adaLN 로직 ======================

    def _self_attn_flash_varlen_packed(
        self,
        x_unpad: torch.Tensor,  # (Tq, D)
        cu_seqlens_q: torch.Tensor,  # (B+1,) int32
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """패딩이 제거된(unpad) 토큰(Tq, D)에 대해서만 Self-Attention을 계산합니다.

        Args:
            x_unpad (torch.Tensor): (Tq, D) 유효 에이전트 토큰
            cu_seqlens_q (torch.Tensor): (B+1,) 누적 길이(int32)
            max_seqlen_q (int): 배치 내 최대 길이

        Returns:
            torch.Tensor: (Tq, D) Self-Attention 출력
        """
        self._check_flash_available()

        if x_unpad.dim() != 2:
            raise ValueError(
                f"x_unpad must be 2D (Tq,D). got {tuple(x_unpad.shape)}")

        Tq, D = x_unpad.shape
        if Tq == 0 or int(max_seqlen_q) == 0:
            # 파라미터가 “사용된 것처럼” 그래프에 등장시키기 위한 터치
            touch = (self.qkv_proj.weight.view(-1)[:1].sum() +
                     (self.qkv_proj.bias.view(-1)[:1].sum()
                      if self.qkv_proj.bias is not None else 0) +
                     self.out_proj.weight.view(-1)[:1].sum() +
                     (self.out_proj.bias.view(-1)[:1].sum()
                      if self.out_proj.bias is not None else 0)) * 0.0
            return x_unpad.new_zeros((Tq, D)) + touch

        qkv = self.qkv_proj(x_unpad)  # (Tq, 3*D)
        qkv = qkv.reshape(Tq, 3, self.num_heads,
                          self.head_dim)  # (Tq, 3, H, Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens_q.to(torch.int32),
            max_seqlen=int(max_seqlen_q),
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (Tq, H, Hd)

        out = out.reshape(Tq, self.num_heads * self.head_dim)  # (Tq, D)
        out = self.out_proj(out.to(dtype=x_unpad.dtype))  # (Tq, D)
        return out

    def _cross_attn_flash_varlen_packed(
        self,
        q_unpad: torch.Tensor,  # (Tq, D)
        cu_seqlens_q: torch.Tensor,  # (B+1,) int32
        max_seqlen_q: int,
        kv_cache: FlashAttnKVCache,
    ) -> torch.Tensor:
        """패딩이 제거된 Q(Tq,D)와, 캐시된 KV(Tk,D)로 Cross-Attention을 계산합니다.

        Args:
            q_unpad (torch.Tensor): (Tq, D) 유효 에이전트 토큰
            cu_seqlens_q (torch.Tensor): (B+1,) 누적 길이(int32)
            max_seqlen_q (int): 배치 내 최대 길이
            kv_cache (FlashAttnKVCache): scene 토큰 KV 캐시

        Returns:
            torch.Tensor: (Tq, D) Cross-Attention 출력
        """
        self._check_flash_available()

        if q_unpad.dim() != 2:
            raise ValueError(
                f"q_unpad must be 2D (Tq,D). got {tuple(q_unpad.shape)}")

        kv_unpad = kv_cache.kv_unpad
        cu_k = kv_cache.cu_seqlens_k
        max_k = int(kv_cache.max_seqlen_k)

        Tq, D = q_unpad.shape
        Tk = int(kv_unpad.shape[0])

        if Tq == 0 or Tk == 0 or int(max_seqlen_q) == 0 or max_k == 0:
            touch = (self.q_proj_cross.weight.view(-1)[:1].sum() +
                     (self.q_proj_cross.bias.view(-1)[:1].sum()
                      if self.q_proj_cross.bias is not None else 0) +
                     self.kv_proj_cross.weight.view(-1)[:1].sum() +
                     (self.kv_proj_cross.bias.view(-1)[:1].sum()
                      if self.kv_proj_cross.bias is not None else 0) +
                     self.out_proj_cross.weight.view(-1)[:1].sum() +
                     (self.out_proj_cross.bias.view(-1)[:1].sum()
                      if self.out_proj_cross.bias is not None else 0)) * 0.0
            return q_unpad.new_zeros((Tq, D)) + touch

        q = self.q_proj_cross(q_unpad).reshape(Tq, self.num_heads,
                                               self.head_dim)  # (Tq, H, Hd)
        kv = self.kv_proj_cross(kv_unpad).reshape(
            Tk, 2, self.num_heads, self.head_dim)  # (Tk, 2, H, Hd)

        comp_dtype = self._get_compute_dtype(q)
        q = q.to(comp_dtype)
        kv = kv.to(comp_dtype)

        out = flash_attn_varlen_cross_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_k),
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (Tq, H, Hd)

        out = out.reshape(Tq, self.num_heads * self.head_dim)  # (Tq, D)
        out = self.out_proj_cross(out.to(dtype=q_unpad.dtype))  # (Tq, D)
        return out

    @staticmethod
    def _cast_to_block_compute_dtype(
        x: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """AMP(autocast) 환경에서 다음 무거운 연산에 맞게 dtype을 정리합니다.

        목적
        - LayerNorm 같은 연산은 AMP에서 float32로 나오는 경우가 있습니다.
        - 하지만 이후 MLP/FlashAttention은 bf16으로 돌고 싶은 경우가 많습니다.
        - 그래서 "LayerNorm 결과"를 autocast가 선택한 dtype(bfloat16/float16)로 다시 맞춥니다.

        Args:
            x (torch.Tensor): dtype을 맞출 대상 텐서. shape: 임의
            ref (torch.Tensor): device/dtype 판단 기준 텐서. shape: 임의

        Returns:
            torch.Tensor:
                - shape: x와 동일
                - dtype:
                    - autocast가 켜져 있고 CUDA면: torch.get_autocast_gpu_dtype()
                    - 그 외: ref.dtype
        """
        target_dtype: torch.dtype = DiTBlock._get_compute_dtype(ref)
        if x.device != ref.device:
            x = x.to(device=ref.device)
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        return x

    def forward_packed(
        self,
        x_unpad: torch.Tensor,  # (Tq, D)
        cu_seqlens_q: torch.Tensor,  # (B+1,) int32
        max_seqlen_q: int,
        cross_kv_cache: FlashAttnKVCache,
        pram_v2_modulations: Dict[str, ModulationTriplet],  # 각 값 텐서 shape: (Tq, D)
    ) -> torch.Tensor:
        """블록 전체를 (Tq, D) packed 토큰에서만 수행합니다.

        - LayerNorm은 AMP 정책대로 fp32가 될 수 있습니다.
        - 하지만 그 다음 MLP/Attention 경로는 bf16이 되도록, LN 출력만 다시 캐스팅합니다.

        Args:
            x_unpad: (Tq, D)
            cu_seqlens_q: (B+1,)
            max_seqlen_q: int
            cross_kv_cache: KV 캐시
            pram_v2_modulations: {"SA","FFN","CA"} 각 ModulationTriplet
                - delta_scale/shift/gate: (Tq, D)

        Returns:
            torch.Tensor: (Tq, D)
        """
        if x_unpad.dim() != 2:
            raise ValueError(f"x_unpad must be 2D (Tq,D). got {tuple(x_unpad.shape)}")

        # (A) 블록 내부 기준 dtype을 "autocast가 선택한 dtype"으로 맞춤
        # - autocast 꺼져 있으면 사실상 no-op
        x_unpad = self._cast_to_block_compute_dtype(x_unpad, ref=x_unpad)

        # ----- SA -----
        sa_mod: ModulationTriplet = pram_v2_modulations["SA"]

        y1 = self.norm1(x_unpad)  # (Tq, D)  (AMP에서 float32일 수 있음)
        y1 = self._cast_to_block_compute_dtype(y1, ref=x_unpad)  # (Tq, D) bf16로 복귀

        ds = sa_mod.delta_scale.to(dtype=y1.dtype, device=y1.device)  # (Tq, D)
        sh = sa_mod.shift.to(dtype=y1.dtype, device=y1.device)        # (Tq, D)
        y_tilde = y1 * (1.0 + ds) + sh                                 # (Tq, D)

        f_sa = self._self_attn_flash_varlen_packed(
            x_unpad=y_tilde,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )  # (Tq, D)

        g = sa_mod.gate.to(dtype=x_unpad.dtype, device=x_unpad.device)  # (Tq, D)
        x_unpad = x_unpad + g * f_sa                                     # (Tq, D)

        # ----- FFN(MLP1) -----
        ffn_mod: ModulationTriplet = pram_v2_modulations["FFN"]

        y2 = self.norm2(x_unpad)  # (Tq, D) (AMP에서 float32일 수 있음)
        y2 = self._cast_to_block_compute_dtype(y2, ref=x_unpad)  # (Tq, D) bf16로 복귀

        ds = ffn_mod.delta_scale.to(dtype=y2.dtype, device=y2.device)  # (Tq, D)
        sh = ffn_mod.shift.to(dtype=y2.dtype, device=y2.device)        # (Tq, D)
        y_tilde = y2 * (1.0 + ds) + sh                                   # (Tq, D)

        f_ffn = self.mlp1(y_tilde)                                       # (Tq, D)

        g = ffn_mod.gate.to(dtype=x_unpad.dtype, device=x_unpad.device)  # (Tq, D)
        x_unpad = x_unpad + g * f_ffn                                     # (Tq, D)

        # ----- CA -----
        ca_mod: ModulationTriplet = pram_v2_modulations["CA"]

        y3 = self.norm3(x_unpad)  # (Tq, D)
        y3 = self._cast_to_block_compute_dtype(y3, ref=x_unpad)  # (Tq, D)

        ds = ca_mod.delta_scale.to(dtype=y3.dtype, device=y3.device)  # (Tq, D)
        sh = ca_mod.shift.to(dtype=y3.dtype, device=y3.device)        # (Tq, D)
        q_styled = y3 * (1.0 + ds) + sh                                 # (Tq, D)

        f_ca = self._cross_attn_flash_varlen_packed(
            q_unpad=q_styled,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            kv_cache=cross_kv_cache,
        )  # (Tq, D)

        g = ca_mod.gate.to(dtype=x_unpad.dtype, device=x_unpad.device)  # (Tq, D)
        x_unpad = x_unpad + g * f_ca                                     # (Tq, D)

        # ----- MLP2 -----
        y4 = self.norm4(x_unpad)  # (Tq, D)
        y4 = self._cast_to_block_compute_dtype(y4, ref=x_unpad)  # (Tq, D)

        mlp2_out = self.mlp2(y4)  # (Tq, D)
        gate2 = self.gate_mlp2.to(dtype=x_unpad.dtype, device=x_unpad.device)  # ()
        x_unpad = x_unpad + gate2 * mlp2_out  # (Tq, D)

        return x_unpad

    def _apply_modulated_mlp1(
            self,
            x: torch.Tensor,  # (B, P, D)
            shift_mlp_pa: torch.Tensor,  # (B, P, D)
            scale_mlp_pa: torch.Tensor,  # (B, P, D)
            gate_mlp_pa: torch.Tensor,  # (B, P, D)
    ) -> torch.Tensor:
        """per‑agent 모듈레이션을 적용한 MLP1 경로."""
        modulated_x = modulate(self.norm2(x), shift_mlp_pa,
                               scale_mlp_pa)  # (B, P, D)
        x = x + gate_mlp_pa * self.mlp1(modulated_x)  # (B, P, D)
        return x

    def forward(
        self,
        x: torch.Tensor,  # [B, (1+)Pnn, D]
        cross_c: torch.Tensor,  # [B, N_c, D]
        pram_v2_modulations: Dict[
            str,
            ModulationTriplet],  # {"SA": ModulationTriplet, "FFN":ModulationTriplet, "CA":ModulationTriplet}
        target_current_mask: torch.Tensor,  # [B, (1+)Pnn] True=pad
        cross_mask: torch.Tensor  # [B, N_c]  True=pad
    ) -> torch.Tensor:
        """PRAM‑v2 주입 경로 포워드.

        경로별 동작:
          - SA : Y=LN1(X) → Y~=(1+Δs)⊙Y + b → Self‑Attn → X + g⊙F_SA
          - FFN: Y=LN2(X) → Y~=(1+Δs)⊙Y + b →   MLP1   → X + g⊙F_FFN
          - CA : Y=LN3(X) → Q~=(1+Δs)⊙Y + b (Q‑only) → Cross‑Attn → X + g⊙F_CA
                 (K/V는 cross_c로부터 그대로 생성; 안정/저비용)
          - 마지막: 원본 MLP2 경로는 그대로 유지 (게이트 self.gate_mlp2)

        Args:
            x:                [B,(1+)Pnn,D]     입력 토큰
            cross_c:          [B,N_c,D]     컨텍스트 토큰
            pram_v2_modulations:
                               dict{"SA"|"FFN"|"CA" → ModulationTriplet(Δs,b,g)}
            target_current_mask:[B,(1+)Pnn]       True=pad (무효 에이전트)
            cross_mask:       [B,N_c]       True=pad (컨텍스트 패딩)

        Returns:
            torch.Tensor: [B,(1+)Pnn,D]
        """
        # ------ SA ------
        sa_mod: ModulationTriplet = pram_v2_modulations["SA"]
        y = self.norm1(x)  # [B,(1+)Pnn,D]
        y_tilde = apply_pram_v2_path_modulation(y, sa_mod)  # [B,(1+)Pnn,D]
        f_sa = self._self_attn_flash_varlen(
            y_tilde, target_current_mask)  # [B,(1+)Pnn,D]
        x = x + sa_mod.gate.to(dtype=x.dtype,
                               device=x.device) * f_sa  # [B,(1+)Pnn,D]

        # ------ FFN(MLP1) ------
        ffn_mod: ModulationTriplet = pram_v2_modulations["FFN"]
        y = self.norm2(x)  # [B,(1+)Pnn,D]
        y_tilde = apply_pram_v2_path_modulation(y, ffn_mod)  # [B,(1+)Pnn,D]
        y_tilde = self._cast_to_block_compute_dtype(y_tilde, ref=x)  # ✅ 추가

        f_ffn = self.mlp1(y_tilde)  # [B,(1+)Pnn,D]
        x = x + ffn_mod.gate.to(dtype=x.dtype,
                                device=x.device) * f_ffn  # [B,(1+)Pnn,D]

        # ------ CA (Q만 스타일링) ------
        ca_mod: ModulationTriplet = pram_v2_modulations["CA"]
        y = self.norm3(x)  # [B,(1+)Pnn,D]
        q_styled = apply_pram_v2_path_modulation(y, ca_mod)  # [B,(1+)Pnn,D]
        f_ca = self._cross_attn_flash_varlen(q_styled, cross_c,
                                             target_current_mask,
                                             cross_mask)  # [B,(1+)Pnn,D]
        # ★ 권장: self.gate_cross 제거, SA/FFN과 동일한 형태
        x = x + ca_mod.gate.to(dtype=x.dtype,
                               device=x.device) * f_ca  # (B,(1+)Pnn,D)

        # ------ 원본 MLP2 경로 유지 ------
        y4 = self.norm4(x)
        y4 = self._cast_to_block_compute_dtype(y4, ref=x)  # ✅ 추가
        x = x + self.gate_mlp2.to(dtype=x.dtype, device=x.device) * self.mlp2(
            y4)

        # 무효 에이전트 0‑클램프 (안전)
        x = x.masked_fill(target_current_mask.unsqueeze(-1),
                          0.0)  # [B,(1+)Pnn,D]
        return x
