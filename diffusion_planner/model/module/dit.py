import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from typing import Tuple, Optional
from flash_attn.bert_padding import unpad_input, pad_input

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
# ===========================================================


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

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


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

        # 전역(ego_fut_global + t)에서 (B, D) 모듈레이션 6개 생성
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

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
        self.gate_cross = nn.Parameter(torch.tensor(0.0))
        self.gate_mlp2 = nn.Parameter(torch.tensor(0.0))

        # Dropout 확률(Train일 때만 FA2에 전달)
        self._attn_dropout_p = dropout

        # ==== (추가) per-agent route 기반 잔차 모듈레이션 ====
        # 입력: near_agents_route_lane_emb (B, Pnn, D) → 출력: (B, Pnn, 6D)
        self.route_adaLN_modulation = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # 잔차 0 초기화(adaLN‑Zero 정신 유지): 초기엔 전역만 작동
        nn.init.zeros_(self.route_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.route_adaLN_modulation[-1].bias)

        # 경로 잔차의 전체 스케일(학습 가능한 스칼라, 0에서 시작)
        self.route_msa_alpha = nn.Parameter(torch.tensor(0.0))  # Self-Attn 경로용
        self.route_mlp_alpha = nn.Parameter(torch.tensor(0.0))  # MLP1 경로용

        # ==== (신규) per-agent ego 계획 기반 잔차 모듈레이션 ====
        # 입력: concat([near_agents_route_lane_emb, ego_fut_global⊕]) → (B,P,2D)
        # 출력: (B,P,6D) → Δshift/scale/gate (MSA/MLP1 경로)
        self.ego_adaLN_mod = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.ego_adaLN_mod[-1].weight)
        nn.init.zeros_(self.ego_adaLN_mod[-1].bias)

        # ego 잔차 스케일 (0에서 시작 → 학습되며 서서히 켜짐)
        self.ego_msa_alpha = nn.Parameter(torch.tensor(0.0))
        self.ego_mlp_alpha = nn.Parameter(torch.tensor(0.0))

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
        x_unpad, idx, cu, max_len = unpad_input(x, attention_mask)
        T = x_unpad.shape[0]
        if T == 0 or max_len == 0:
            return torch.zeros_like(x)

        # (2) QKV 프로젝션 (유효 토큰만)
        qkv = self.qkv_proj(x_unpad)  # (T, 3*D)
        qkv = qkv.reshape(T, 3, self.num_heads, self.head_dim)  # (T, 3, H, Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        # (3) FlashAttention‑2 varlen (Self-Attn)
        # out: (T, H, Hd)
        out = flash_attn_varlen_qkvpacked_func(
            qkv,  # (T, 3, H, Hd)
            cu_seqlens=cu.to(torch.int32),
            max_seqlen=max_len,  # int  배치 내 최대 유효 길이.
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (T, H, Hd)

        # (4) 출력 프로젝션 + pad back
        out = out.reshape(T, self.num_heads * self.head_dim)  # (T, D)
        out = self.out_proj(out.to(x.dtype))  # (T, D) -> 원 dtype
        out = pad_input(out, idx, B, L)  # (B, L, D)
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
        q_unpad, q_idx, cu_q, max_q = unpad_input(q_in, q_mask_valid)  # (Tq, D)
        kv_unpad, _, cu_k, max_k = unpad_input(kv_in, kv_mask_valid)  # (Tk, D)
        if q_unpad.numel() == 0 or kv_unpad.numel(
        ) == 0 or max_q == 0 or max_k == 0:
            return torch.zeros(B, Lq, D, device=q_in.device, dtype=q_in.dtype)

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

    def _compute_global_adaln(
        self,
        global_condition: torch.
        Tensor,  # (B, D)  = ego_fut_global + t_embedding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """전역 조건으로부터 (B, D) 모듈레이션 6개를 계산합니다."""
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
         gate_mlp) = self.adaLN_modulation(global_condition).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _compute_route_residual_adaln(
        self,
        per_agent_route_lane_emb: torch.Tensor,  # (B, P, D)
        route_known_mask: torch.Tensor
        # (B, P)  True=route known
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """에이전트별 route 임베딩(B, P, D)으로부터 (B, P, D) 잔차 모듈레이션 6개를 계산합니다.

        Args:
            per_agent_route_lane_emb (torch.Tensor): (B, P, D) route 기반 임베딩.
            route_known_mask (Optional[torch.Tensor]): (B, P) bool, True=route 정보 있음.
                - 제공되면, 알려지지 않은 에이전트(=False)의 잔차 Δ를 0으로 강제합니다.

        Returns:
            Tuple[torch.Tensor, ...]: (delta_shift_msa, delta_scale_msa, delta_gate_msa,
                                       delta_shift_mlp, delta_scale_mlp, delta_gate_mlp),
                                       각각 (B, P, D).
        """
        # route_modulation: (B, P, 6D)
        route_modulation: torch.Tensor = self.route_adaLN_modulation(
            per_agent_route_lane_emb)
        (delta_shift_msa, delta_scale_msa, delta_gate_msa, delta_shift_mlp,
         delta_scale_mlp, delta_gate_mlp) = route_modulation.chunk(6, dim=-1)
        if route_known_mask.dtype != torch.bool:
            raise ValueError(
                f"route_known_mask must be bool, got {route_known_mask.dtype}")
        if route_known_mask.shape != per_agent_route_lane_emb.shape[:2]:
            raise ValueError(
                f"route_known_mask shape must be (B,P)={per_agent_route_lane_emb.shape[:2]}, "
                f"got {tuple(route_known_mask.shape)}")
        # (B,P) → (B,P,1)로 승격 후 Δ항을 0으로 강제
        keep = route_known_mask.unsqueeze(-1)  # True=keep Δ, False=zero Δ
        delta_shift_msa = delta_shift_msa.masked_fill(~keep, 0)
        delta_scale_msa = delta_scale_msa.masked_fill(~keep, 0)
        delta_gate_msa = delta_gate_msa.masked_fill(~keep, 0)
        delta_shift_mlp = delta_shift_mlp.masked_fill(~keep, 0)
        delta_scale_mlp = delta_scale_mlp.masked_fill(~keep, 0)
        delta_gate_mlp = delta_gate_mlp.masked_fill(~keep, 0)

        return (delta_shift_msa, delta_scale_msa, delta_gate_msa,
                delta_shift_mlp, delta_scale_mlp, delta_gate_mlp)

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

    def _apply_modulated_self_attention(
            self,
            x: torch.Tensor,  # (B, P, D)
            attn_mask: torch.Tensor,  # (B, P) True=pad
            shift_msa_pa: torch.Tensor,  # (B, P, D)
            scale_msa_pa: torch.Tensor,  # (B, P, D)
            gate_msa_pa: torch.Tensor,  # (B, P, D)
    ) -> torch.Tensor:
        """per‑agent 모듈레이션을 적용한 Self‑Attention 경로."""
        modulated_x = modulate(self.norm1(x), shift_msa_pa,
                               scale_msa_pa)  # (B, P, D)
        msa_out = self._self_attn_flash_varlen(modulated_x,
                                               attn_mask)  # (B, P, D)
        x = x + gate_msa_pa * msa_out  # (B, P, D)
        return x

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

    def _compute_ego_future_adaln(
        self,
        ego_fut_global_expand: torch.Tensor,  # (B, Pnn, D)
        near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
        route_known_mask: torch.Tensor  # (B, Pnn) True=known
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        ego_fut_global_expand 에서, 마지막 차원의 값이 전부 0. 이면 -> 무효한 에이전트이므로, 잔차 Δ를 0으로 강제합니다.

        """
        # (B,Pnn) True=valid ego_fut_global_expand
        valid_ego_mask = (ego_fut_global_expand.abs().sum(dim=-1) != 0)

        pair = torch.cat([near_agents_route_lane_emb, ego_fut_global_expand],
                         dim=-1)  # (B,Pnn,2D)
        d_ego = self.ego_adaLN_mod(pair)  # (B,Pnn,6D)
        (dsh_msa_e, dsc_msa_e, dgt_msa_e, dsh_mlp_e, dsc_mlp_e,
         dgt_mlp_e) = d_ego.chunk(6, dim=-1)  # 각각 (B,Pnn,D)
        keep = (valid_ego_mask & route_known_mask).unsqueeze(-1)  # (B,P,1)
        dsh_msa_e = dsh_msa_e.masked_fill(~keep, 0)
        dsc_msa_e = dsc_msa_e.masked_fill(~keep, 0)
        dgt_msa_e = dgt_msa_e.masked_fill(~keep, 0)
        dsh_mlp_e = dsh_mlp_e.masked_fill(~keep, 0)
        dsc_mlp_e = dsc_mlp_e.masked_fill(~keep, 0)
        dgt_mlp_e = dgt_mlp_e.masked_fill(~keep, 0)
        return dsh_msa_e, dsc_msa_e, dgt_msa_e, dsh_mlp_e, dsc_mlp_e, dgt_mlp_e

    def forward(
        self,
        x: torch.Tensor,  # (B, Pnn, D)
        cross_c: torch.Tensor,  # (B, token_num, D)
        t_embedding: torch.Tensor,  # (B, D)  t_embedding
        ego_fut_global: torch.Tensor,  # (B, D)
        near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
        attn_mask: torch.Tensor,  # (B, Pnn) True=pad # near_current_mask
        cross_mask: torch.Tensor,  # (B, token_num)   True=pad
        route_known_mask: torch.Tensor  # (B, Pnn) True=known
    ) -> torch.Tensor:
        """
        순서:
            1) 전역 모듈레이션(B,D) 6개 계산
            2) per‑agent route 잔차(B,Pnn,D) 6개 계산
            3) 결합하여 (B,Pnn,D) 6개 모듈레이션 생성
            4) Self‑Attention → MLP1 (per‑agent 모듈레이션 적용)
            5) Cross‑Attention → MLP2 (원본 게이트 유지)

        Note:
            - FlashAttention‑2 varlen 경로로 무효 토큰(패딩)을 완전히 건너뜁니다.
            - pad back 시 마스크된 위치는 자연스럽게 0이 되며, gradient도 올바르게 흘러갑니다.
        """
        B, Pnn, D = x.shape
        # t_embedding: (B, D=192)
        # 1) 전역(B,D)
        # global_mods: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        global_mods = self._compute_global_adaln(t_embedding)  # (B,D)×6
        # 2) per-agent 잔차(B,Pnn,D) 6개 (+선택적 마스킹 적용)
        route_residuals_mods = self._compute_route_residual_adaln(
            near_agents_route_lane_emb.to(x.dtype),
            route_known_mask=route_known_mask)  # (B,Pnn,D)×6
        #################
        ego_fut_global_expand = ego_fut_global.to(x.dtype).unsqueeze(1).expand(
            B, Pnn, D)  # (B,Pnn,D)
        near_agents_route_lane_emb = near_agents_route_lane_emb.to(x.dtype)
        # dsh_msa_e, dsc_msa_e, dgt_msa_e, dsh_mlp_e, dsc_mlp_e, dgt_mlp_e
        ego_fut_residuals = self._compute_ego_future_adaln(
            ego_fut_global_expand, near_agents_route_lane_emb, route_known_mask)

        #################
        # 3) 결합(B,Pnn,D)
        (shift_msa_pa, scale_msa_pa, gate_msa_pa, shift_mlp_pa, scale_mlp_pa,
         gate_mlp_pa) = self._combine_global_and_route_modulations(
             global_mods, route_residuals_mods, ego_fut_residuals)

        # 4) Self‑Attention + MLP1 (per‑agent 모듈레이션)
        x = self._apply_modulated_self_attention(x, attn_mask, shift_msa_pa,
                                                 scale_msa_pa, gate_msa_pa)
        x = self._apply_modulated_mlp1(x, shift_mlp_pa, scale_mlp_pa,
                                       gate_mlp_pa)
        # 5) Cross‑Attention (원본 유지) + MLP2(게이트)
        q = self.norm3(x)  # (B, Pnn, D)
        cross_out = self._cross_attn_flash_varlen(q, cross_c, attn_mask,
                                                  cross_mask)  # (B, Pnn, D)
        gate_cross = self.gate_cross.to(x.dtype)
        gate_mlp2 = self.gate_mlp2.to(x.dtype)
        x = x + gate_cross * cross_out
        x = x + gate_mlp2 * self.mlp2(self.norm4(x))
        x = x.masked_fill(attn_mask.unsqueeze(-1), 0.0)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            # nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj[-1].weight)  # proj의 마지막 Linear
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, t_embedding):
        B, P, _ = x.shape
        shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x
