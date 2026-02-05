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
import inspect
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# (추가) 빠른 LayerNorm / 빠른 MLP를 "있으면 쓰고, 없으면 안전하게 복귀"하기 위한 유틸
# ============================================================

# ---- flash-attn fused LayerNorm (선택) ----
try:
    from flash_attn.ops.layer_norm import layer_norm as _flash_layer_norm  # type: ignore
except Exception:
    try:
        # 일부 빌드/버전에서 이름이 다를 수 있어 백업
        from flash_attn.ops.layer_norm import layer_norm_fn as _flash_layer_norm  # type: ignore
    except Exception:
        _flash_layer_norm = None

# ---- flash-attn fused Dense+GELU+Dense (선택) ----
try:
    from flash_attn.ops.fused_dense import fused_dense_gelu_dense as _flash_fused_dense_gelu_dense  # type: ignore
except Exception:
    _flash_fused_dense_gelu_dense = None

# fused_dense_gelu_dense가 받는 추가 인자(버전별 차이)를 미리 파악해 둠
_FUSED_DGDD_KWARGS: Dict[str, Any] = {}
if _flash_fused_dense_gelu_dense is not None:
    try:
        sig = inspect.signature(_flash_fused_dense_gelu_dense)
        params = sig.parameters
        if "checkpoint_lvl" in params:
            _FUSED_DGDD_KWARGS["checkpoint_lvl"] = 0
        if "heuristic" in params:
            _FUSED_DGDD_KWARGS["heuristic"] = 0
        if "save_pre_act" in params:
            _FUSED_DGDD_KWARGS["save_pre_act"] = False
    except Exception:
        _FUSED_DGDD_KWARGS = {}


def _is_cuda_fp16_or_bf16(x: torch.Tensor) -> bool:
    """GPU에서 작은 dtype(fp16/bf16)로 계산 중인지 확인합니다.

    Args:
        x (torch.Tensor): 임의 텐서. shape: 임의

    Returns:
        bool: CUDA + (float16 또는 bfloat16) 이면 True
    """
    return bool(x.is_cuda and x.dtype in (torch.float16, torch.bfloat16))


def _fast_layer_norm(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """LayerNorm을 가능한 빠른 경로로 실행합니다.

    동작:
        1) flash-attn의 LayerNorm 함수가 있고, GPU + fp16/bf16이면 그걸 먼저 시도합니다.
        2) 실패하거나 조건이 안 맞으면, PyTorch의 layer_norm으로 안전하게 실행합니다.

    Args:
        x (torch.Tensor): 입력 텐서. shape: (..., D)
        ln (nn.LayerNorm): LayerNorm 모듈. (weight/bias/eps 사용)

    Returns:
        torch.Tensor: 정규화 결과. shape: (..., D)
    """
    if x.numel() == 0:
        # 빈 텐서는 그대로 처리(안전)
        return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)

    if _flash_layer_norm is not None and _is_cuda_fp16_or_bf16(x):
        # flash-attn layer_norm은 버전/빌드에 따라 인자명이 다를 수 있어 2가지 형태로 시도
        try:
            return _flash_layer_norm(x, ln.weight, ln.bias, ln.eps)  # type: ignore[misc]
        except TypeError:
            try:
                return _flash_layer_norm(x, ln.weight, ln.bias, eps=ln.eps)  # type: ignore[misc]
            except Exception:
                pass
        except Exception:
            pass

    # 기본 경로(항상 동작)
    return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)


def _fast_modulated_layer_norm(
    x: torch.Tensor,               # shape: (..., D)
    ln: nn.LayerNorm,
    delta_scale: torch.Tensor,     # shape: (..., D)
    shift: torch.Tensor,           # shape: (..., D)
) -> torch.Tensor:
    """LayerNorm 이후에 (1+delta_scale)과 shift를 적용합니다.

    계산식(그대로 유지):
        y = LayerNorm(x)
        out = y * (1 + delta_scale) + shift

    구현(속도 목적):
        - (1 + delta_scale) 를 따로 만들지 않고,
          y*(1+ds) = y + ds*y 로 바꿔서 addcmul로 처리합니다.
        - 이렇게 하면 작은 텐서에서 GPU 호출 횟수를 줄일 수 있습니다.

    Args:
        x (torch.Tensor): 입력. shape: (..., D)
        ln (nn.LayerNorm): 정규화 모듈
        delta_scale (torch.Tensor): 스케일 보정. shape: (..., D)
        shift (torch.Tensor): 이동 보정. shape: (..., D)

    Returns:
        torch.Tensor: 보정된 출력. shape: (..., D)
    """
    # y: (..., D)
    y = _fast_layer_norm(x, ln)

    # dtype/device 맞추기(이미 같으면 복사 없음)
    if delta_scale.device != y.device or delta_scale.dtype != y.dtype:
        delta_scale = delta_scale.to(device=y.device, dtype=y.dtype)
    if shift.device != y.device or shift.dtype != y.dtype:
        shift = shift.to(device=y.device, dtype=y.dtype)

    # y*(1+ds) = y + ds*y  (addcmul 1회)
    y_scaled = torch.addcmul(y, delta_scale, y)  # (..., D)

    # + shift (1회)
    out = y_scaled + shift  # (..., D)
    return out


def _fast_mlp_gelu(
    x: torch.Tensor,      # shape: (..., D)
    mlp: nn.Module,
) -> torch.Tensor:
    """Linear -> GELU -> Linear 형태 MLP를 가능한 빠른 경로로 실행합니다.

    동작:
        1) flash-attn fused 함수가 있고, 입력/가중치 dtype이 fp16/bf16이면 fused를 시도합니다.
        2) 조건이 안 맞거나 실패하면, 기존 mlp(x)로 실행합니다.

    Args:
        x (torch.Tensor): 입력. shape: (..., D)
        mlp (nn.Module): timm Mlp 같은 형태를 기대합니다.
            - mlp.fc1, mlp.fc2 (nn.Linear)가 있어야 합니다.

    Returns:
        torch.Tensor: 출력. shape: (..., D_out)
    """
    if _flash_fused_dense_gelu_dense is None:
        return mlp(x)

    if x.numel() == 0:
        return mlp(x)

    if not _is_cuda_fp16_or_bf16(x):
        return mlp(x)

    # timm.Mlp 호환: fc1/fc2가 있어야 함
    if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
        return mlp(x)

    fc1 = getattr(mlp, "fc1")
    fc2 = getattr(mlp, "fc2")
    if not (isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)):
        return mlp(x)

    # fused_dense는 보통 가중치 dtype이 입력과 같아야 안정적입니다.
    if fc1.weight.dtype != x.dtype or fc2.weight.dtype != x.dtype:
        return mlp(x)
    if (fc1.weight.device != x.device) or (fc2.weight.device != x.device):
        return mlp(x)

    # x_2d: (N, D)
    orig_shape = tuple(int(s) for s in x.shape)
    x_2d = x.reshape(-1, orig_shape[-1])

    # fused 실행
    try:
        y_2d = _flash_fused_dense_gelu_dense(  # type: ignore[misc]
            x_2d,
            fc1.weight, fc1.bias,
            fc2.weight, fc2.bias,
            **_FUSED_DGDD_KWARGS,
        )  # (N, D_out)
    except Exception:
        return mlp(x)

    # 원래 shape로 복원: (..., D_out)
    out_shape = orig_shape[:-1] + (int(y_2d.shape[-1]),)
    return y_2d.reshape(out_shape)


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
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=approx_gelu,
                        drop=0)

        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=approx_gelu,
                        drop=0)

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

    def forward_packed(
            self,
            x_unpad: torch.Tensor,  # (Tq, D)
            cu_seqlens_q: torch.Tensor,  # (B+1,) int32
            max_seqlen_q: int,
            cross_kv_cache: FlashAttnKVCache,
            pram_v2_modulations: Dict[str, ModulationTriplet],
            # 값 텐서 shape: (Tq, D)
    ) -> torch.Tensor:
        """블록 전체를 (Tq, D) packed 토큰에서만 수행합니다.

        변경점(속도 목적):
            - LayerNorm은 가능하면 flash-attn 경로로 실행합니다(없으면 기존 경로).
            - y*(1+ds)+sh 는 addcmul을 이용해 적은 호출로 계산합니다.
            - x + g*f 도 addcmul로 계산해 호출 수를 줄입니다.
            - MLP는 가능하면 flash-attn fused 경로를 사용합니다(없으면 기존 timm Mlp).

        Args:
            x_unpad: (Tq, D) 유효 에이전트 토큰
            cu_seqlens_q: (B+1,) 누적 길이(int32)
            max_seqlen_q: 배치 내 최대 길이
            cross_kv_cache: scene KV 캐시
            pram_v2_modulations: {"SA","FFN","CA"} 각 ModulationTriplet
                - delta_scale/shift/gate: (Tq, D)

        Returns:
            torch.Tensor: (Tq, D)
        """
        if x_unpad.dim() != 2:
            raise ValueError(
                f"x_unpad must be 2D (Tq,D). got {tuple(x_unpad.shape)}")

        # ----- SA -----
        sa_mod: ModulationTriplet = pram_v2_modulations["SA"]

        # y_tilde: (Tq, D)
        y_tilde = _fast_modulated_layer_norm(
            x=x_unpad,  # (Tq, D)
            ln=self.norm1,  # LayerNorm 파라미터 그대로 사용
            delta_scale=sa_mod.delta_scale,  # (Tq, D)
            shift=sa_mod.shift,  # (Tq, D)
        )

        # f_sa: (Tq, D)
        f_sa = self._self_attn_flash_varlen_packed(
            x_unpad=y_tilde,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )

        # x = x + g * f_sa  (addcmul 1회)
        g_sa = sa_mod.gate
        if g_sa.device != x_unpad.device or g_sa.dtype != x_unpad.dtype:
            g_sa = g_sa.to(device=x_unpad.device, dtype=x_unpad.dtype)
        x_unpad = torch.addcmul(x_unpad, g_sa, f_sa)  # (Tq, D)

        # ----- FFN(MLP1) -----
        ffn_mod: ModulationTriplet = pram_v2_modulations["FFN"]

        y_tilde = _fast_modulated_layer_norm(
            x=x_unpad,  # (Tq, D)
            ln=self.norm2,
            delta_scale=ffn_mod.delta_scale,  # (Tq, D)
            shift=ffn_mod.shift,  # (Tq, D)
        )

        # f_ffn: (Tq, D)
        f_ffn = _fast_mlp_gelu(y_tilde, self.mlp1)

        # x = x + g * f_ffn  (addcmul 1회)
        g_ffn = ffn_mod.gate
        if g_ffn.device != x_unpad.device or g_ffn.dtype != x_unpad.dtype:
            g_ffn = g_ffn.to(device=x_unpad.device, dtype=x_unpad.dtype)
        x_unpad = torch.addcmul(x_unpad, g_ffn, f_ffn)  # (Tq, D)

        # ----- CA -----
        ca_mod: ModulationTriplet = pram_v2_modulations["CA"]

        q_styled = _fast_modulated_layer_norm(
            x=x_unpad,  # (Tq, D)
            ln=self.norm3,
            delta_scale=ca_mod.delta_scale,  # (Tq, D)
            shift=ca_mod.shift,  # (Tq, D)
        )

        # f_ca: (Tq, D)
        f_ca = self._cross_attn_flash_varlen_packed(
            q_unpad=q_styled,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            kv_cache=cross_kv_cache,
        )

        # x = x + g * f_ca  (addcmul 1회)
        g_ca = ca_mod.gate
        if g_ca.device != x_unpad.device or g_ca.dtype != x_unpad.dtype:
            g_ca = g_ca.to(device=x_unpad.device, dtype=x_unpad.dtype)
        x_unpad = torch.addcmul(x_unpad, g_ca, f_ca)  # (Tq, D)

        # ----- MLP2 -----
        # norm4도 가능한 빠른 경로로
        y4 = _fast_layer_norm(x_unpad, self.norm4)  # (Tq, D)
        mlp2_out = _fast_mlp_gelu(y4, self.mlp2)  # (Tq, D)

        # 게이트는 학습되는 파라미터라 item()로 빼면 안 됩니다.
        gate2 = self.gate_mlp2.to(dtype=x_unpad.dtype,
                                  device=x_unpad.device)  # shape: ()
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
        x = x + self.gate_mlp2.to(dtype=x.dtype, device=x.device) * self.mlp2(
            self.norm4(x))

        # 무효 에이전트 0‑클램프 (안전)
        x = x.masked_fill(target_current_mask.unsqueeze(-1),
                          0.0)  # [B,(1+)Pnn,D]
        return x
