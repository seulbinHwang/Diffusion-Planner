# -*- coding: utf-8 -*-
# file: diffusion_planner/model/module/pram_v2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
from timm.models.layers import Mlp

PathName = Literal["SA", "FFN", "CA"]
# -*- coding: utf-8 -*-


class PRAMV2StateTokenEncoder(nn.Module):
    """현재 프레임 (x,y,cos,sin) → [B,Pnn,D] 토큰으로 얕게 인코딩.

    Args:
        hidden_dim (int): 출력 토큰 차원 D(= DiT hidden_dim).
        mid_dim (Optional[int]): 중간 폭. 기본은 hidden_dim//2.
        use_rmsnorm (bool): 입력 4채널 RMSNorm 적용 여부.
        zero_init_last (bool): 마지막 선형(mlp.fc2) 0초기화 여부.

    Inputs:
        near_cur_norm: [B, Pnn, 4]  정규화 공간의 현재 프레임 (x,y,cos,sin)
        near_current_mask: [B, Pnn]  True=무효 에이전트(패딩)

    Returns:
        state_token_in: [B, Pnn, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        mid_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        zero_init_last: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mid_dim = hidden_dim // 2 if mid_dim is None else mid_dim
        self.input_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.token_mlp = Mlp(
            in_features=9,
            hidden_features=self.mid_dim,
            out_features=self.hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        if zero_init_last:
            nn.init.zeros_(self.token_mlp.fc2.weight)
            nn.init.zeros_(self.token_mlp.fc2.bias)

    @staticmethod
    def _project_unit_circle(
            x: torch.Tensor,  # (..., 9)
            eps: float = 1e-6) -> torch.Tensor:
        """(cos,sin) → 단위원 정규화. x: [...,4] -> [...,4]"""
        cs = x[..., 2:4]
        norm = torch.linalg.norm(cs, dim=-1, keepdim=True).clamp_min(eps)
        cs = cs / norm
        return torch.cat([x[..., :2], cs, x[..., 4:]], dim=-1)  # (...,9)

    @staticmethod
    def _mask_zero(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """마스크(True=무효) 위치 0."""
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        target_cur_norm: torch.
        Tensor,  # [B, (1+)Pnn, 11] # [x,y,cos,sin,vx,vy,w,l,one_hot(3)]
        target_current_mask: torch.Tensor  # [B, (1+)Pnn]
    ) -> torch.Tensor:
        """현재 프레임을 얕게 투영해 state_token_in을 생성."""
        xy_cos_sin = target_cur_norm[..., 0:4]  # (B, (1+)Pnn, 4)
        width_and_length = target_cur_norm[..., 6:8]  # (B, (1+)Pnn, 2)
        one_hot_type = target_cur_norm[..., 8:11]  # (B, (1+)Pnn, 3)
        target_cur_norm = torch.cat(
            [xy_cos_sin, width_and_length, one_hot_type],
            dim=-1)  # (B, (1+)Pnn, 9)
        x = self._project_unit_circle(target_cur_norm)  # [...,9]
        x = self.input_norm(x)
        state_token_in = self.token_mlp(x)  # [B,(1+)Pnn,D]
        state_token_in = self._mask_zero(state_token_in, target_current_mask)
        return state_token_in  # [B,(1+)Pnn,D]


# -----------------------------
# 데이터 컨테이너(타입·shape 명시)
# -----------------------------


@dataclass
class ModulationTriplet:
    """경로별 모듈레이션 3종(Δscale, shift, gate).

    Attributes:
        delta_scale: [B, Pnn, H]
        shift:       [B, Pnn, H]
        gate:        [B, Pnn, H]  (sigmoid 직후 [0,1] 범위 권장)
    """
    delta_scale: torch.Tensor
    shift: torch.Tensor
    gate: torch.Tensor


@dataclass
class ComposerOutputs:
    """Composer(한 번 계산) 출력.

    Attributes:
        delta_scale_base: [B, Pnn, H]
        shift_base:       [B, Pnn, H]
        logit_gate_base:  [B, Pnn, H]  (σ 전 단계)
    """
    delta_scale_base: torch.Tensor
    shift_base: torch.Tensor
    logit_gate_base: torch.Tensor


@dataclass
class TimeModulationOutputs:
    """확산 시간(t) 기반 전역 모듈레이션.

    Attributes:
        delta_scale_time: [B, 1, H]  (에이전트 축으로 방송 예정)
        shift_time:       [B, 1, H]
        logit_gate_time:  [B, 1, H]
    """
    delta_scale_time: torch.Tensor
    shift_time: torch.Tensor
    logit_gate_time: torch.Tensor


# -----------------------------
# 유틸 함수 / 모듈 (함수 내 함수 사용 금지)
# -----------------------------


def build_mlp(in_features: int,
              hidden_features: int,
              out_features: int,
              activation: str = "gelu") -> nn.Sequential:
    """두 층 MLP 생성 유틸.

    Args:
        in_features: 입력 차원
        hidden_features: 중간 폭
        out_features: 출력 차원
        activation: "gelu" 또는 "silu"

    Returns:
        nn.Sequential: Linear(in→hidden) → Act → Linear(hidden→out)
    """
    act = nn.GELU() if activation.lower() == "gelu" else nn.SiLU()
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        act,
        nn.Linear(hidden_features, out_features),
    )


class RMSNormNoParam(nn.Module):
    """학습 파라미터가 없는 RMSNorm (스케일만 정규화).

    y = x / sqrt(mean(x^2) + eps)

    Args:
        eps: 수치 안정성용 epsilon
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., H]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


def zero_init_linear(linear: nn.Linear) -> None:
    """Linear 레이어의 weight/bias를 0으로 초기화."""
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


# -----------------------------
# Composer / Time / Scalars
# -----------------------------


class PRAMV2Composer(nn.Module):
    """PRAM‑v2의 Composer.

    S/E/R를 저차원으로 정리(RMSNorm 포함) → 쌍곱(SE/ER/RS) → 혼합 MLP → base 모듈레이션(Δs,b,logit g)을 산출.

    Args:
        hidden_dim: 최종 H(=DiT hidden_dim)
        adapter_hidden_dim: 어댑터 내부 중간 폭 m (~2h)
        composed_hidden_dim: 쌍곱 포함 후 혼합용 폭 c (h 또는 2h 권장)
        activation: 'gelu' 또는 'silu'
        gate_init_bias: 게이트 바이어스 초기값(충분히 음수 권장: -2 ~ -4)

    Shapes:
        state_token_in:              [B, Pnn, D]
        ego_future_global:           [B, D]
        near_agents_route_lane_emb:  [B, Pnn, D]
        route_known_mask:            [B, Pnn]  (True=route 제공)
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_hidden_dim: int = 128,
        composed_hidden_dim: int = 128,
        activation: str = "gelu",
        gate_init_bias: float = -3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.composed_hidden_dim = composed_hidden_dim
        self.activation = activation
        self.gate_init_bias = gate_init_bias

        # 어댑터: D -> m -> h  (세 경로 동일 차원 D 입력 가정)
        # h는 composed_hidden_dim과 동일/유사 규모를 권장하나, 분리해도 무방
        # 여기서는 h := composed_hidden_dim 로 둡니다.
        h = composed_hidden_dim
        m = adapter_hidden_dim

        self.adapt_S = build_mlp(
            in_features=hidden_dim,  # D (=192)
            hidden_features=m,  # 128
            out_features=h,  # 128
            activation=activation)

        # RMSNorm(무파라미터): 곱(원소곱) 전·후 안정화
        self.rms_pre = RMSNormNoParam()
        self.rms_post = RMSNormNoParam()


        # 출력 헤드(Δs_base, b_base, logit_g_base) : [B, Pnn, H]
        self.head_delta_scale = nn.Linear(self.composed_hidden_dim,
                                          self.hidden_dim)
        self.head_shift = nn.Linear(self.composed_hidden_dim, self.hidden_dim)
        self.head_logit_gate = nn.Linear(self.composed_hidden_dim,
                                         self.hidden_dim)

        # 중요 초기화:
        # Δs, b는 0에서 시작(원 DiT 그대로), gate 바이어스는 충분히 음수에서 시작
        zero_init_linear(self.head_delta_scale)
        zero_init_linear(self.head_shift)
        nn.init.zeros_(self.head_logit_gate.weight)
        nn.init.constant_(self.head_logit_gate.bias, gate_init_bias)

        # 입력 정규화(채널 기준)
        self.in_norm_S = nn.LayerNorm(hidden_dim)
        self.in_norm_E = nn.LayerNorm(hidden_dim)
        self.in_norm_R = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _is_ego_provided(ego_fut_global: torch.Tensor,
                         eps: float = 0.0) -> torch.Tensor:
        """[B,D]에서 전부 0(또는 비유효)이면 미제공으로 보는 배치 마스크.

        Returns:
            ego_known_mask_b: [B]  (True=제공됨, False=미제공)
        """
        is_finite = torch.isfinite(ego_fut_global).all(dim=-1)  # [B]
        nonzero = (ego_fut_global.abs().sum(dim=-1) > eps)  # [B]
        return is_finite & nonzero

    @staticmethod
    def _broadcast_ego(ego_fut_global: torch.Tensor, pnn: int) -> torch.Tensor:
        """[B,D] → [B,Pnn,D] 브로드캐스트."""
        return ego_fut_global[:, None, :].expand(-1, pnn, -1)

    @staticmethod
    def _mask_ego_by_batch(e_b: torch.Tensor,
                           ego_known_mask_b: torch.Tensor) -> torch.Tensor:
        """배치 단위(전부 0) 미제공은 전체 Pnn에서 0으로."""
        return torch.where(
            ego_known_mask_b[:, None, None],  # [B,1,1]
            e_b,
            torch.zeros_like(e_b))

    def forward(
            self,
            state_token_in: torch.Tensor,  # [B, (1+)Pnn, D]
    ) -> ComposerOutputs:
        """(요약) state/ego/route → Adapt → (SE/ER/RS) → 혼합 → Δs_base/b_base/logit_g_base.

        1) “입력 요약” 만들기 (한 번만 계산 → 모든 블록 재사용)
        2) 쌍곱(상호작용) 특징 만들기 — (SE, ER, RS)
        3) 쌍곱 포함해 한 덩어리로 묶고 u 만들기 — LN → 작은 MLP
        5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
        """
        """
        1) “입력 요약” 만들기 (한 번만 계산 → 모든 블록 재사용)
        """

        # ★ 무효 agent 마스크: state_token_in의 D차원이 전부 0이면 무효(True)
        #    state_known_mask: [B,(1+)Pnn] (True=유효), invalid_mask: [B,(1+)Pnn,1] (True=무효)
        state_known_mask = state_token_in.ne(0).any(
            dim=-1)  # [B,(1+)Pnn] True=유효
        invalid_mask = (~state_known_mask).unsqueeze(
            -1)  # [B,(1+)Pnn,1] True=무효

        # --- S 경로 ---
        S_in = self.in_norm_S(state_token_in)  # [B,(1+)Pnn,D]
        s = self.adapt_S(S_in)  # [B,(1+)Pnn,h]
        s = self.rms_pre(s)
        s = s.masked_fill(invalid_mask, 0.0)  # ★ 무효 agent는 S 경로 0

        """
        5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
        """
        # --- 헤드 ---
        delta_scale_base = self.head_delta_scale(s)  # [B,(1+)Pnn,H]
        shift_base = self.head_shift(s)  # [B,(1+)Pnn,H]
        logit_gate_base = self.head_logit_gate(s)  # [B,(1+)Pnn,H]

        # ★ 최종 출력도 무효 agent에서는 모두 0 보장
        delta_scale_base = delta_scale_base.masked_fill(invalid_mask,
                                                        0.0)  # [B,(1+)Pnn,H]
        shift_base = shift_base.masked_fill(invalid_mask, 0.0)  # [B,(1+)Pnn,H]
        logit_gate_base = logit_gate_base.masked_fill(invalid_mask,
                                                      0.0)  # [B,(1+)Pnn,H]

        return ComposerOutputs(
            delta_scale_base=delta_scale_base,  # [B, (1+)Pnn, H]
            shift_base=shift_base,  # [B, (1+)Pnn, H]
            logit_gate_base=logit_gate_base,  # [B, (1+)Pnn, H]
        )


class PRAMV2TimeModulator(nn.Module):
    """확산 시간(t) 임베딩으로부터 전역(time) 모듈레이션을 생성.

    Args:
        hidden_dim: H(=DiT hidden_dim)

    Inputs:
        t_embedding: [B, H]

    Outputs (broadcast-ready):
        delta_scale_time: [B, 1, H]
        shift_time:       [B, 1, H]
        logit_gate_time:  [B, 1, H]
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 간단히 Linear 3개. (필요 시 더 깊게 확장 가능)
        self.lin_delta_scale = nn.Linear(hidden_dim, hidden_dim)
        self.lin_shift = nn.Linear(hidden_dim, hidden_dim)
        self.lin_logit_gate = nn.Linear(hidden_dim, hidden_dim)

        # ★ 0-initialization (권장 설정)
        nn.init.zeros_(self.lin_delta_scale.weight)
        nn.init.zeros_(self.lin_delta_scale.bias)

        nn.init.zeros_(self.lin_shift.weight)
        nn.init.zeros_(self.lin_shift.bias)

        nn.init.zeros_(self.lin_logit_gate.weight)
        nn.init.constant_(self.lin_logit_gate.bias,
                          -3.0)  # 게이트 초기 닫힘(≈σ(-3)≈0.05)

        # 초기화는 기본값 유지(시간 스타일은 학습 통해 조정되도록)

    def forward(self, t_embedding: torch.Tensor) -> TimeModulationOutputs:
        """시간 기반 모듈레이션 계산.

        Args:
            t_embedding: [B, H]

        Returns:
            TimeModulationOutputs: 각 텐서는 [B, 1, H]
        """
        assert t_embedding.dim(
        ) == 2 and t_embedding.shape[-1] == self.hidden_dim

        dlt = self.lin_delta_scale(t_embedding).unsqueeze(1)  # [B, 1, H]
        shf = self.lin_shift(t_embedding).unsqueeze(1)  # [B, 1, H]
        lgt = self.lin_logit_gate(t_embedding).unsqueeze(1)  # [B, 1, H]

        return TimeModulationOutputs(
            delta_scale_time=dlt,
            shift_time=shf,
            logit_gate_time=lgt,
        )


class PRAMV2BlockPathScalars(nn.Module):
    """블록×경로 토글 스칼라 및 게이트 편향.

    내부에 학습 가능한 스칼라 4종을 둔다:
      - k_s     : Δscale 강도 스케일
      - k_sh    : shift 강도 스케일
      - k_g     : gate 강도 스케일
      - beta_g  : gate 전용 편향

    Args:
        depth: DiT 블록 개수
        num_paths: 3 (SA/FFN/CA)
    """

    PATH_INDEX = {"SA": 0, "FFN": 1, "CA": 2}

    def __init__(self, depth: int, num_paths: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.num_paths = num_paths

        # [depth, num_paths] 모양의 파라미터로 구현
        # 초기화: k_*는 1.0 부근, beta_g는 0.0 부근
        self.k_s = nn.Parameter(torch.ones(depth, num_paths))
        self.k_sh = nn.Parameter(torch.ones(depth, num_paths))
        self.k_g = nn.Parameter(torch.ones(depth, num_paths))
        self.beta_g = nn.Parameter(torch.zeros(depth, num_paths))

    def _path_to_index(self, path: PathName) -> int:
        """경로명 → 인덱스 변환.

        # path: PathName = Literal["SA", "FFN", "CA"]
        # PATH_INDEX = {"SA": 0, "FFN": 1, "CA": 2}
        """
        return self.PATH_INDEX[path]

    def get_scalars(
        self,
        block_index: int,
        path: PathName  # Literal["SA", "FFN", "CA"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """블록/경로별 스칼라 4종(k_s, k_sh, k_g, beta_g)을 반환.

        Args:
            block_index: 0..depth-1
            path: "SA" | "FFN" | "CA"

        Returns:
            (k_s, k_sh, k_g, beta_g): 각 0-D 또는 [1] 텐서(브로드캐스트 용이)
        """
        i = block_index
        j = self._path_to_index(path)
        # 0-D 텐서 반환 (필요 시 .view(1,1,1)로 브로드캐스트)
        return (
            self.k_s[i, j],
            self.k_sh[i, j],
            self.k_g[i, j],
            self.beta_g[i, j],
        )


def compute_pram_v2_modulations_for_block(
    composer_out: ComposerOutputs,  # 3개 [B, Pnn, H]
    time_out: TimeModulationOutputs,  # 3개 [B, 1, H]
    path_scalars: PRAMV2BlockPathScalars,
    block_index: int,
    batch_size: int,
    one_or_Pnn: int,
    hidden_dim: int,
) -> Dict[PathName, ModulationTriplet]:
    """블록 b에서 SA/FFN/CA 경로별 최종 모듈레이션(Δs, b, gate)을 합성합니다.

    공식(경로별 동일):
        Δs^(b,p) = Δs_time + k_s^(b,p)  * Δs_base
        b^(b,p)  = b_time  + k_sh^(b,p) * b_base
        g^(b,p)  = σ( logit_time + k_g^(b,p) * logit_base + β_g^(b,p) )

    Args:
        composer_out: Δs_base / b_base / logit_g_base  (각 [B,Pnn,H])
        time_out:     Δs_time / b_time / logit_g_time  (각 [B,1,H])
        path_scalars: 블록×경로 토글 스칼라 관리자
        block_index:  현재 블록 인덱스(0‑based)
        batch_size:   B (shape 검증용)
        one_or_Pnn: Pnn (shape 검증용)
        hidden_dim:   H (shape 검증용)

    Returns:
        Dict[PathName, ModulationTriplet]:
        {"SA": ModulationTriplet, "FFN":ModulationTriplet, "CA":ModulationTriplet}
    """
    # ----- shape 정리 -----
    B, Pnn, H = composer_out.delta_scale_base.shape
    assert B == batch_size and Pnn == one_or_Pnn and H == hidden_dim, \
        f"[compute_pram_v2_modulations_for_block] shape mismatch: " \
        f"got composer_out {composer_out.delta_scale_base.shape}, expected {(batch_size, one_or_Pnn, hidden_dim)}"

    device = composer_out.delta_scale_base.device
    dtype = composer_out.delta_scale_base.dtype

    # 시간 모듈레이션 [B,1,H] → [B,Pnn,H] (브로드캐스트)
    delta_scale_time = time_out.delta_scale_time.to(dtype=dtype,
                                                    device=device).expand(
                                                        B, Pnn, H)
    shift_time = time_out.shift_time.to(dtype=dtype,
                                        device=device).expand(B, Pnn, H)
    logit_gate_time = time_out.logit_gate_time.to(dtype=dtype,
                                                  device=device).expand(
                                                      B, Pnn, H)

    # Base (이미 [B,Pnn,H])
    delta_scale_base = composer_out.delta_scale_base.to(dtype=dtype,
                                                        device=device)
    shift_base = composer_out.shift_base.to(dtype=dtype, device=device)
    logit_gate_base = composer_out.logit_gate_base.to(dtype=dtype,
                                                      device=device)

    out: Dict[PathName, ModulationTriplet] = {}

    for path in ("SA", "FFN", "CA"):
        # k_s, k_sh, k_g, beta_g: 0‑D 텐서(파라미터) → dtype/device 정렬
        k_s, k_sh, k_g, beta_g = path_scalars.get_scalars(block_index,
                                                          path)  # 0-D Tensors
        k_s = k_s.to(dtype=dtype, device=device)
        k_sh = k_sh.to(dtype=dtype, device=device)
        k_g = k_g.to(dtype=dtype, device=device)
        beta_g = beta_g.to(dtype=dtype, device=device)

        # 최종 합성
        delta_scale = delta_scale_time + k_s * delta_scale_base  # [B,Pnn,H]
        shift = shift_time + k_sh * shift_base  # [B,Pnn,H]
        gate = torch.sigmoid(logit_gate_time + k_g * logit_gate_base +
                             beta_g)  # [B,Pnn,H]

        out[path] = ModulationTriplet(delta_scale=delta_scale,
                                      shift=shift,
                                      gate=gate)

    return out


def apply_pram_v2_path_modulation(
        normalized_token: torch.Tensor,  # [B, Pnn, H]
        modulation: ModulationTriplet,  # Δs / b / g  (각 [B,Pnn,H])
) -> torch.Tensor:
    """(경로 공통) LN 출력에 모듈레이션을 적용합니다.

    Y~ = Y ⊙ (1 + Δs) + b

    Args:
        normalized_token: [B, Pnn, H]  # LN 결과
        modulation:       ModulationTriplet (Δs/shift/gate)

    Returns:
        torch.Tensor: [B, Pnn, H]  # 모듈레이션 적용 결과
    """
    y = normalized_token
    ds = modulation.delta_scale.to(dtype=y.dtype, device=y.device)  # [B,Pnn,H]
    sh = modulation.shift.to(dtype=y.dtype, device=y.device)  # [B,Pnn,H]
    return y * (1.0 + ds) + sh


def style_queries_for_cross_attention(
        normalized_token: torch.Tensor,  # [B, Pnn, H]
        modulation: ModulationTriplet,  # Δs / b / g  (각 [B,Pnn,H])
) -> torch.Tensor:
    """Cross‑Attention에서 **Q에만** 적용할 스타일링.

    구현은 path‑modulation과 동일: Q~ = Q ⊙ (1 + Δs) + b

    Args:
        normalized_token: [B, Pnn, H]  # LN 결과(=Q의 원천)
        modulation:       ModulationTriplet (Δs/shift/gate)

    Returns:
        torch.Tensor: [B, Pnn, H]  # 스타일링된 Q 원천
    """
    return apply_pram_v2_path_modulation(normalized_token, modulation)


def apply_pram_v2_final_layer(
    x: torch.Tensor,  # [B, (1+)Pnn, H]
    composer_out: ComposerOutputs,
    time_out: TimeModulationOutputs,
    final_norm: nn.LayerNorm,  # LN(H) 모듈
    out_proj: nn.Sequential,  # Linear(H -> (T)*4)
    final_scalars: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """PRAM‑v2 9단계: 최종 모듈레이션 + 최종 투영까지 수행.

    공식(게이트는 보통 생략):
        Δs_final = Δs_time + k_s^final  * Δs_base
        b_final  = b_time  + k_sh^final * b_base
        Y        = LN(X)
        Y~       = (1 + Δs_final) ⊙ Y + b_final
        out      = Linear(Y~)

    Args:
        x:             [B, (1+)Pnn, H]          마지막 블록 출력(hidden)
        composer_out:  Δs_base / b_base / logit_g_base  (각 [B,(1+)Pnn,H])
        time_out:      Δs_time / b_time / logit_g_time  (각 [B,1,H])
        final_norm:    최종 LayerNorm(H)
        out_proj:      최종 선형( H → T*4 )
        final_scalars: (k_final_s, k_final_sh)  # 각 0‑D 텐서 또는 None(미제공 시 1.0)

    Returns:
        x_out: [B, (1+)Pnn, T*4]
    """
    B, one_or_Pnn, H = x.shape
    device, dtype = x.device, x.dtype

    # 시간 모듈레이션 [B,1,H] → [B,(1+)Pnn,H]
    delta_scale_time = time_out.delta_scale_time.to(dtype=dtype,
                                                    device=device).expand(
                                                        B, one_or_Pnn, H)
    shift_time = time_out.shift_time.to(dtype=dtype,
                                        device=device).expand(B, one_or_Pnn, H)

    # Base
    delta_scale_base = composer_out.delta_scale_base.to(
        dtype=dtype, device=device)  # [B,(1+)Pnn,H]
    shift_base = composer_out.shift_base.to(dtype=dtype,
                                            device=device)  # [B,(1+)Pnn,H]

    # 최종 스칼라 (미제공 시 1.0)
    if final_scalars is None:
        k_final_s = torch.tensor(1.0, dtype=dtype, device=device)
        k_final_sh = torch.tensor(1.0, dtype=dtype, device=device)
    else:
        k_final_s, k_final_sh = final_scalars
        k_final_s = k_final_s.to(dtype=dtype, device=device)
        k_final_sh = k_final_sh.to(dtype=dtype, device=device)

    # 최종 합성
    delta_scale_final = delta_scale_time + k_final_s * delta_scale_base  # [B,(1+)Pnn,H]
    shift_final = shift_time + k_final_sh * shift_base  # [B,(1+)Pnn,H]

    # LN → (1+Δs) ⊙ · + b → Linear
    y = final_norm(x)  # [B,(1+)Pnn,H]
    y_tilde = y * (1.0 + delta_scale_final) + shift_final  # [B,(1+)Pnn,H]
    x_out = out_proj(y_tilde)  # [B,(1+)Pnn,T*4]
    return x_out
