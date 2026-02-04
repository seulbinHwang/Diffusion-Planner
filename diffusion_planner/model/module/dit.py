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
from typing import Dict, Tuple
import torch
import torch.nn as nn
from diffusion_planner.model.module.pram_v2 import ModulationTriplet

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
                torch.arange(start=0, end=half, dtype=torch.float32) / float(half)
            )  # (half,)
            freqs = torch.exp(exponent)  # (half,) float32
        else:
            freqs = torch.empty((0,), dtype=torch.float32)

        # persistent=False: 체크포인트 호환(새 buffer로 인해 strict 로딩 실패) 위험 줄이기
        self.register_buffer("_freqs", freqs, persistent=False)  # (half,)


    @staticmethod
    def _to_token_view_from_indices_or_cu(
        t: torch.Tensor,
        *,
        indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """(B,L,...) 또는 (B,...) 형태 텐서를 (T,...) 유효 토큰 순서로 맞춥니다.

        목적:
            - pram 모듈레이션(scale/shift/gate)이 (B,L,...) 이면 indices로 unpad
            - (B,...) 이면 배치별 유효 토큰 수만큼 repeat해서 (T,...)로 변환

        Args:
            t (torch.Tensor):
                - (B, L, ...) 또는 (B, ...) 또는 스칼라 형태
            indices (torch.Tensor):
                unpad_input이 준 indices. shape: (T,)
            cu_seqlens (torch.Tensor):
                unpad_input이 준 cu_seqlens. shape: (B+1,)

        Returns:
            torch.Tensor:
                (T, ...) 모양의 텐서
        """
        if t.dim() >= 3:
            # (B,L,...) -> (B*L, ...) -> indices로 선택
            flat = t.reshape(-1, *t.shape[2:])
            return flat.index_select(0, indices)

        if t.dim() == 2:
            # (B,D) -> 배치별 유효 토큰 수만큼 repeat -> (T,D)
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # (B,)
            return torch.repeat_interleave(t, seqlens, dim=0)

        if t.dim() == 1:
            # (B,) -> (T,1)
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # (B,)
            out = torch.repeat_interleave(t, seqlens, dim=0)  # (T,)
            return out.unsqueeze(-1)

        # 스칼라() -> (T,1)
        return t.view(1, 1).expand(int(indices.numel()), 1)

    def _self_attn_flash_varlen_packed(
        self,
        x_unpad: torch.Tensor,      # (T, D)
        cu_seqlens: torch.Tensor,   # (B+1,)
        max_seqlen: int,
    ) -> torch.Tensor:
        """Self-Attn을 packed(T,D)에서 바로 계산합니다.

        Args:
            x_unpad: 유효 토큰만 모은 텐서. shape: (T, D)
            cu_seqlens: 배치별 누적 길이. shape: (B+1,)
            max_seqlen: 배치 내 최대 유효 길이. int

        Returns:
            out_unpad: shape (T, D)
        """
        self._check_flash_available()

        T, D = x_unpad.shape
        if T == 0 or int(max_seqlen) == 0:
            touch = (self.qkv_proj.weight.view(-1)[:1].sum() +
                     (self.qkv_proj.bias.view(-1)[:1].sum()
                      if self.qkv_proj.bias is not None else 0) +
                     self.out_proj.weight.view(-1)[:1].sum() +
                     (self.out_proj.bias.view(-1)[:1].sum()
                      if self.out_proj.bias is not None else 0)) * 0.0
            return x_unpad + touch  # (0,D) 유지

        qkv = self.qkv_proj(x_unpad)  # (T, 3*D)
        qkv = qkv.reshape(T, 3, self.num_heads, self.head_dim)  # (T,3,H,Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens.to(torch.int32),
            max_seqlen=int(max_seqlen),
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (T,H,Hd)

        out = out.reshape(T, self.num_heads * self.head_dim)  # (T,D)
        out = self.out_proj(out.to(x_unpad.dtype))  # (T,D)
        return out

    def _cross_attn_flash_varlen_packed(
        self,
        q_unpad: torch.Tensor,          # (Tq, D)
        cu_seqlens_q: torch.Tensor,     # (B+1,)
        max_seqlen_q: int,
        kv_unpad: torch.Tensor,         # (Tk, D)
        cu_seqlens_k: torch.Tensor,     # (B+1,)
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Cross-Attn을 packed(T,D)에서 바로 계산합니다. (KV는 바깥에서 캐시된 것을 사용)

        Args:
            q_unpad: 유효 쿼리 토큰. shape: (Tq, D)
            cu_seqlens_q: q의 누적 길이. shape: (B+1,)
            max_seqlen_q: q의 배치 내 최대 길이. int
            kv_unpad: 유효 KV 토큰(캐시). shape: (Tk, D)
            cu_seqlens_k: k의 누적 길이. shape: (B+1,)
            max_seqlen_k: k의 배치 내 최대 길이. int

        Returns:
            out_unpad: q 자리 출력. shape: (Tq, D)
        """
        self._check_flash_available()

        Tq, D = q_unpad.shape
        Tk = int(kv_unpad.shape[0])

        if Tq == 0 or Tk == 0 or int(max_seqlen_q) == 0 or int(max_seqlen_k) == 0:
            touch = (self.q_proj_cross.weight.view(-1)[:1].sum() +
                     (self.q_proj_cross.bias.view(-1)[:1].sum()
                      if self.q_proj_cross.bias is not None else 0) +
                     self.kv_proj_cross.weight.view(-1)[:1].sum() +
                     (self.kv_proj_cross.bias.view(-1)[:1].sum()
                      if self.kv_proj_cross.bias is not None else 0) +
                     self.out_proj_cross.weight.view(-1)[:1].sum() +
                     (self.out_proj_cross.bias.view(-1)[:1].sum()
                      if self.out_proj_cross.bias is not None else 0)) * 0.0
            return q_unpad + touch  # (Tq,D) 유지(빈 경우도 OK)

        q = self.q_proj_cross(q_unpad).reshape(Tq, self.num_heads, self.head_dim)  # (Tq,H,Hd)
        kv = self.kv_proj_cross(kv_unpad).reshape(Tk, 2, self.num_heads, self.head_dim)  # (Tk,2,H,Hd)

        comp_dtype = self._get_compute_dtype(q)
        q = q.to(comp_dtype)
        kv = kv.to(comp_dtype)

        out = flash_attn_varlen_cross_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (Tq,H,Hd)

        out = out.reshape(Tq, self.num_heads * self.head_dim)  # (Tq,D)
        out = self.out_proj_cross(out.to(q_unpad.dtype))       # (Tq,D)
        return out

    def forward_packed(
        self,
        x_unpad: torch.Tensor,          # (T, D)
        *,
        indices: torch.Tensor,          # (T,)
        cu_seqlens_q: torch.Tensor,     # (B+1,)
        max_seqlen_q: int,
        kv_unpad: torch.Tensor,         # (Tk, D)
        cu_seqlens_k: torch.Tensor,     # (B+1,)
        max_seqlen_k: int,
        pram_v2_modulations: Dict[str, ModulationTriplet],
    ) -> torch.Tensor:
        """블록 전체를 packed(T,D)에서 수행합니다.

        핵심:
            - LayerNorm/MLP까지 유효 토큰만 계산 (패딩 에이전트 계산 제거)
            - Cross-Attn의 KV는 바깥에서 캐시된 kv_unpad/cu_k/max_k를 사용

        Args:
            x_unpad:
                유효 에이전트 토큰. shape: (T, D)
            indices:
                unpad_input이 준 indices. shape: (T,)
            cu_seqlens_q:
                Q(에이전트)의 누적 길이. shape: (B+1,)
            max_seqlen_q:
                Q의 배치 내 최대 길이. int
            kv_unpad:
                scene KV 유효 토큰(캐시). shape: (Tk, D)
            cu_seqlens_k:
                KV의 누적 길이. shape: (B+1,)
            max_seqlen_k:
                KV의 배치 내 최대 길이. int
            pram_v2_modulations:
                {"SA","FFN","CA"} 각각의 모듈레이션(shift/scale/gate)

        Returns:
            torch.Tensor:
                블록 출력(유효 토큰만). shape: (T, D)
        """
        # --- SA ---
        sa_mod: ModulationTriplet = pram_v2_modulations["SA"]
        sa_scale = self._to_token_view_from_indices_or_cu(sa_mod.scale, indices=indices, cu_seqlens=cu_seqlens_q)
        sa_shift = self._to_token_view_from_indices_or_cu(sa_mod.shift, indices=indices, cu_seqlens=cu_seqlens_q)
        sa_gate  = self._to_token_view_from_indices_or_cu(sa_mod.gate,  indices=indices, cu_seqlens=cu_seqlens_q)

        sa_scale = sa_scale.to(dtype=x_unpad.dtype, device=x_unpad.device)
        sa_shift = sa_shift.to(dtype=x_unpad.dtype, device=x_unpad.device)
        sa_gate  = sa_gate.to(dtype=x_unpad.dtype, device=x_unpad.device)

        y = self.norm1(x_unpad)  # (T,D)
        y = y * (1.0 + sa_scale) + sa_shift
        f_sa = self._self_attn_flash_varlen_packed(y, cu_seqlens_q, max_seqlen_q)  # (T,D)
        x_unpad = x_unpad + sa_gate * f_sa  # (T,D)

        # --- FFN(MLP1) ---
        ffn_mod: ModulationTriplet = pram_v2_modulations["FFN"]
        ffn_scale = self._to_token_view_from_indices_or_cu(ffn_mod.scale, indices=indices, cu_seqlens=cu_seqlens_q)
        ffn_shift = self._to_token_view_from_indices_or_cu(ffn_mod.shift, indices=indices, cu_seqlens=cu_seqlens_q)
        ffn_gate  = self._to_token_view_from_indices_or_cu(ffn_mod.gate,  indices=indices, cu_seqlens=cu_seqlens_q)

        ffn_scale = ffn_scale.to(dtype=x_unpad.dtype, device=x_unpad.device)
        ffn_shift = ffn_shift.to(dtype=x_unpad.dtype, device=x_unpad.device)
        ffn_gate  = ffn_gate.to(dtype=x_unpad.dtype, device=x_unpad.device)

        y = self.norm2(x_unpad)  # (T,D)
        y = y * (1.0 + ffn_scale) + ffn_shift
        f_ffn = self.mlp1(y)  # (T,D)
        x_unpad = x_unpad + ffn_gate * f_ffn  # (T,D)

        # --- CA (KV 캐시 사용) ---
        ca_mod: ModulationTriplet = pram_v2_modulations["CA"]
        ca_scale = self._to_token_view_from_indices_or_cu(ca_mod.scale, indices=indices, cu_seqlens=cu_seqlens_q)
        ca_shift = self._to_token_view_from_indices_or_cu(ca_mod.shift, indices=indices, cu_seqlens=cu_seqlens_q)
        ca_gate  = self._to_token_view_from_indices_or_cu(ca_mod.gate,  indices=indices, cu_seqlens=cu_seqlens_q)

        ca_scale = ca_scale.to(dtype=x_unpad.dtype, device=x_unpad.device)
        ca_shift = ca_shift.to(dtype=x_unpad.dtype, device=x_unpad.device)
        ca_gate  = ca_gate.to(dtype=x_unpad.dtype, device=x_unpad.device)

        y = self.norm3(x_unpad)  # (T,D)
        q = y * (1.0 + ca_scale) + ca_shift
        f_ca = self._cross_attn_flash_varlen_packed(
            q_unpad=q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            kv_unpad=kv_unpad,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )  # (T,D)
        x_unpad = x_unpad + ca_gate * f_ca  # (T,D)

        # --- 원본 MLP2 ---
        y = self.norm4(x_unpad)  # (T,D)
        x_unpad = x_unpad + self.gate_mlp2.to(dtype=x_unpad.dtype, device=x_unpad.device) * self.mlp2(y)  # (T,D)

        return x_unpad

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
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
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
                               device=t.device, dtype=torch.float32)

        freqs: torch.Tensor = self._freqs
        if freqs.device != t.device:
            freqs = freqs.to(device=t.device)

        args = t[:, None].float() * freqs[None, :]  # (N, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, 2*half)

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
            return torch.zeros((0, out_dim), device=t.device, dtype=torch.float32)

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

    def _combine_global_and_route_modulations(
        self,
        global_modulations: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor,
                                  torch.Tensor],  # (B,D)×6
        route_residuals_mods: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor,
                                    torch.Tensor],  # (B,P,D)×6
        ego_fut_residuals: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor,
                                 torch.Tensor],  # (B,P,D)×6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """전역 모듈레이션(B,D)과 per‑agent 잔차(B,P,D)를 결합해 (B,P,D) 6개를 반환합니다.

        결합식:
            shift_msa^p = shift_msa + route_msa_alpha * Δshift_msa^p
            scale_msa^p = scale_msa + route_msa_alpha * Δscale_msa^p
            gate_msa^p  = gate_msa  + route_msa_alpha * Δgate_msa^p
            (MLP 경로도 동일; α는 학습 가능한 스칼라)
        """
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
         gate_mlp) = global_modulations
        (d_shift_msa, d_scale_msa, d_gate_msa, d_shift_mlp, d_scale_mlp,
         d_gate_mlp) = route_residuals_mods

        # ★ FIX: 스칼라 파라미터 α를 연산 dtype에 맞춤 (수치/성능 안정화)
        alpha_msa = self.route_msa_alpha.to(dtype=shift_msa.dtype)
        alpha_mlp = self.route_mlp_alpha.to(dtype=shift_mlp.dtype)

        # (B, D) → (B, 1, D) 승격 후 (B, P, D) 잔차와 합
        shift_msa_pa = shift_msa.unsqueeze(1) + alpha_msa * d_shift_msa
        scale_msa_pa = scale_msa.unsqueeze(1) + alpha_msa * d_scale_msa
        gate_msa_pa = gate_msa.unsqueeze(1) + alpha_msa * d_gate_msa

        shift_mlp_pa = shift_mlp.unsqueeze(1) + alpha_mlp * d_shift_mlp
        scale_mlp_pa = scale_mlp.unsqueeze(1) + alpha_mlp * d_scale_mlp
        gate_mlp_pa = gate_mlp.unsqueeze(1) + alpha_mlp * d_gate_mlp

        (dsh_msa_e, dsc_msa_e, dgt_msa_e, dsh_mlp_e, dsc_mlp_e,
         dgt_mlp_e) = ego_fut_residuals

        a_msa = self.ego_msa_alpha.to(shift_msa.dtype)
        a_mlp = self.ego_mlp_alpha.to(shift_mlp.dtype)
        shift_msa_pa = shift_msa_pa + a_msa * dsh_msa_e
        scale_msa_pa = scale_msa_pa + a_msa * dsc_msa_e
        gate_msa_pa = gate_msa_pa + a_msa * dgt_msa_e
        shift_mlp_pa = shift_mlp_pa + a_mlp * dsh_mlp_e
        scale_mlp_pa = scale_mlp_pa + a_mlp * dsc_mlp_e
        gate_mlp_pa = gate_mlp_pa + a_mlp * dgt_mlp_e

        return shift_msa_pa, scale_msa_pa, gate_msa_pa, shift_mlp_pa, scale_mlp_pa, gate_mlp_pa

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
