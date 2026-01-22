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
    """현재 프레임 상태를 더 풍부한 토큰으로 바꿉니다(그룹별 처리 후 합치기).

    이 버전은 입력을 성격별로 나눠서 각각 따로 처리한 뒤 합칩니다.

    - 위치 그룹: (x, y)
    - 방향 그룹: (cos, sin)  (길이를 1로 정리한 뒤 사용)
    - 크기 그룹: (w, l)
    - 종류 그룹: type_one_hot[3]

    Args:
        hidden_dim (int):
            최종 출력 토큰의 마지막 차원 크기 D.
        branch_hidden_dim (Optional[int]):
            (x,y)와 (cos,sin) 그룹 전용 작은 네트워크의 중간 크기.
            None이면 hidden_dim//2 를 사용합니다.
        branch_out_dim (Optional[int]):
            각 그룹 전용 작은 네트워크의 출력 크기.
            None이면 max(16, hidden_dim//3) 을 사용합니다.
            (그룹 출력 크기는 동일해야 원소별 곱을 만들 수 있습니다.)
        size_type_hidden_dim (Optional[int]):
            (w,l) / type 그룹 전용 작은 네트워크의 중간 크기.
            None이면 branch_hidden_dim보다 작게 자동 설정합니다(파라미터 2~3배 목표 유지).
        fuse_hidden_dim (Optional[int]):
            합친 뒤 최종으로 섞는 작은 네트워크의 중간 크기.
            None이면 branch_out_dim 을 사용합니다.
        use_rmsnorm (bool):
            입력값을 “크기 기준”으로 정리할지 여부입니다.
            True면 각 그룹 입력을 간단히 정리한 뒤 네트워크에 넣습니다.
        zero_init_last (bool):
            마지막 합치기 네트워크의 마지막 선형을 0으로 초기화할지 여부입니다.
            주의: 이 프로젝트에서는 state_token_in이 전부 0이면 무효로 판정될 수 있어,
            기본 False가 안전합니다.

    Inputs:
        target_cur_norm (torch.Tensor):
            현재 프레임 상태.
            shape: (B, (1+)Pnn, 11)
            채널 의미(현재 코드 기준):
              0: x, 1: y, 2: cos, 3: sin, 4: vx, 5: vy, 6: w, 7: l, 8~10: type one-hot
            여기서는 속도(vx,vy)는 사용하지 않습니다.
        target_current_mask (torch.Tensor):
            True면 무효(패딩) 에이전트.
            shape: (B, (1+)Pnn)

    Returns:
        torch.Tensor:
            state_token_in
            shape: (B, (1+)Pnn, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        branch_hidden_dim: Optional[int] = None,
        branch_out_dim: Optional[int] = None,
        size_type_hidden_dim: Optional[int] = None,
        fuse_hidden_dim: Optional[int] = None,
        use_rmsnorm: bool = True,
        zero_init_last: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        b_hidden: int = self._default_branch_hidden_dim(
            hidden_dim=self.hidden_dim, branch_hidden_dim=branch_hidden_dim)
        b_out: int = self._default_branch_out_dim(hidden_dim=self.hidden_dim,
                                                  branch_out_dim=branch_out_dim)
        st_hidden: int = self._default_size_type_hidden_dim(
            branch_hidden_dim=b_hidden,
            size_type_hidden_dim=size_type_hidden_dim)
        f_hidden: int = self._default_fuse_hidden_dim(
            branch_out_dim=b_out, fuse_hidden_dim=fuse_hidden_dim)

        self.branch_hidden_dim = b_hidden
        self.branch_out_dim = b_out
        self.size_type_hidden_dim = st_hidden
        self.fuse_hidden_dim = f_hidden

        # 그룹별 입력 정리(학습 파라미터 없음)
        self.xy_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.heading_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.size_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.type_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()

        # 그룹별 전용 작은 네트워크
        self.xy_mlp = Mlp(
            in_features=2,
            hidden_features=self.branch_hidden_dim,
            out_features=self.branch_out_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.heading_mlp = Mlp(
            in_features=2,
            hidden_features=self.branch_hidden_dim,
            out_features=self.branch_out_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # ✅ (요청 반영) 크기(w,l)와 type(3)을 분리
        self.size_mlp = Mlp(
            in_features=2,  # (w, l)
            hidden_features=self.size_type_hidden_dim,
            out_features=self.branch_out_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.type_mlp = Mlp(
            in_features=3,  # type one-hot(3)
            hidden_features=self.size_type_hidden_dim,
            out_features=self.branch_out_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # 합치기 네트워크
        # 입력은 6개 묶음:
        #   [xy, heading, shape, xy*heading, heading*shape, shape*xy]
        fuse_in_dim: int = int(self.branch_out_dim * 6)
        self.fuse_mlp = Mlp(
            in_features=fuse_in_dim,
            hidden_features=self.fuse_hidden_dim,
            out_features=self.hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        if zero_init_last:
            nn.init.zeros_(self.fuse_mlp.fc2.weight)
            nn.init.zeros_(self.fuse_mlp.fc2.bias)

    @staticmethod
    def _default_branch_hidden_dim(
        hidden_dim: int,
        branch_hidden_dim: Optional[int],
    ) -> int:
        """(x,y)/(cos,sin) 그룹 전용 네트워크의 중간 크기를 정합니다.

        Args:
            hidden_dim (int): 최종 출력 크기 D.
            branch_hidden_dim (Optional[int]): 사용자가 지정한 값 또는 None.

        Returns:
            int: 중간 크기.
        """
        if branch_hidden_dim is not None:
            return int(branch_hidden_dim)
        return max(32, int(hidden_dim // 2))

    @staticmethod
    def _default_branch_out_dim(
        hidden_dim: int,
        branch_out_dim: Optional[int],
    ) -> int:
        """그룹 전용 네트워크의 출력 크기를 정합니다.

        Args:
            hidden_dim (int): 최종 출력 크기 D.
            branch_out_dim (Optional[int]): 사용자가 지정한 값 또는 None.

        Returns:
            int: 출력 크기.
        """
        if branch_out_dim is not None:
            return int(branch_out_dim)
        return max(16, int(hidden_dim // 3))

    @staticmethod
    def _default_size_type_hidden_dim(
        branch_hidden_dim: int,
        size_type_hidden_dim: Optional[int],
    ) -> int:
        """(w,l) / type 그룹 전용 네트워크의 중간 크기를 정합니다.

        기본 목표:
            - 분리로 인해 파라미터가 과하게 늘지 않도록,
              branch_hidden_dim보다 작게 자동 설정합니다.

        Args:
            branch_hidden_dim (int): (x,y)/(cos,sin) 그룹의 중간 크기.
            size_type_hidden_dim (Optional[int]): 사용자가 지정한 값 또는 None.

        Returns:
            int: (w,l) / type 그룹 전용 중간 크기.
        """
        if size_type_hidden_dim is not None:
            return int(size_type_hidden_dim)
        # 기본: branch_hidden_dim * 5/8 (예: 96 -> 60), 최소 32
        return max(32, int(branch_hidden_dim * 5 // 8))

    @staticmethod
    def _default_fuse_hidden_dim(
        branch_out_dim: int,
        fuse_hidden_dim: Optional[int],
    ) -> int:
        """합치기 네트워크의 중간 크기를 정합니다.

        Args:
            branch_out_dim (int): 그룹 전용 네트워크 출력 크기.
            fuse_hidden_dim (Optional[int]): 사용자가 지정한 값 또는 None.

        Returns:
            int: 합치기 네트워크 중간 크기.
        """
        if fuse_hidden_dim is not None:
            return int(fuse_hidden_dim)
        return int(branch_out_dim)

    @staticmethod
    def _normalize_cos_sin(
        cos_sin: torch.Tensor,  # (B, P, 2)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """(cos, sin)의 길이를 1로 정리합니다.

        Args:
            cos_sin (torch.Tensor):
                방향 정보.
                shape: (B, P, 2)
            eps (float):
                0에 가까운 경우 나눗셈이 불안정해지는 것을 막기 위한 값.

        Returns:
            torch.Tensor:
                길이가 1로 정리된 (cos, sin).
                shape: (B, P, 2)
        """
        raw_norm: torch.Tensor = torch.linalg.norm(cos_sin,
                                                   dim=-1,
                                                   keepdim=True)  # (B, P, 1)
        safe_norm: torch.Tensor = raw_norm.clamp_min(float(eps))  # (B, P, 1)
        normalized: torch.Tensor = cos_sin / safe_norm  # (B, P, 2)

        ones: torch.Tensor = torch.ones_like(normalized[..., :1])
        zeros: torch.Tensor = torch.zeros_like(normalized[..., :1])
        fallback: torch.Tensor = torch.cat([ones, zeros], dim=-1)  # (B, P, 2)

        too_small: torch.Tensor = (raw_norm < float(eps))  # (B, P, 1)
        normalized = torch.where(too_small, fallback, normalized)
        return normalized

    @staticmethod
    def _mask_zero(
            x: torch.Tensor,  # (B, P, D)
            mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """무효(패딩) 위치를 0으로 만듭니다.

        Args:
            x (torch.Tensor):
                입력 텐서.
                shape: (B, P, D)
            mask (torch.Tensor):
                True면 무효.
                shape: (B, P)

        Returns:
            torch.Tensor:
                무효 위치가 0으로 정리된 텐서.
                shape: (B, P, D)
        """
        mask_bool: torch.Tensor = mask.to(dtype=torch.bool)  # (B, P)
        return x.masked_fill(mask_bool.unsqueeze(-1), 0.0)

    def _split_groups(
        self,
        target_cur_norm: torch.Tensor,  # (B, P, 11)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """입력을 (x,y)/(cos,sin)/(w,l)/type으로 나눕니다.

        Args:
            target_cur_norm (torch.Tensor):
                현재 프레임 상태.
                shape: (B, P, 11)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - xy: (B, P, 2)
                - cos_sin: (B, P, 2)
                - size_wl: (B, P, 2) = (w, l)
                - type_one_hot: (B, P, 3)
        """
        xy: torch.Tensor = target_cur_norm[..., 0:2]  # (B, P, 2)
        cos_sin: torch.Tensor = target_cur_norm[..., 2:4]  # (B, P, 2)
        size_wl: torch.Tensor = target_cur_norm[..., 6:8]  # (B, P, 2)
        type_one_hot: torch.Tensor = target_cur_norm[..., 8:11]  # (B, P, 3)
        return xy, cos_sin, size_wl, type_one_hot

    @staticmethod
    def _combine_size_and_type_features(
            size_feat: torch.Tensor,  # (B, P, O)
            type_feat: torch.Tensor,  # (B, P, O)
    ) -> torch.Tensor:
        """크기 특징과 종류 특징을 하나의 'shape' 특징으로 합칩니다.

        합치는 규칙(파라미터 없음):
            shape_feat = size_feat + type_feat + (size_feat * type_feat)

        Args:
            size_feat (torch.Tensor):
                크기(w,l)에서 뽑은 특징.
                shape: (B, P, O)
            type_feat (torch.Tensor):
                type(one-hot)에서 뽑은 특징.
                shape: (B, P, O)

        Returns:
            torch.Tensor:
                결합된 shape 특징.
                shape: (B, P, O)
        """
        return size_feat + type_feat + (size_feat * type_feat)

    def forward(
            self,
            target_cur_norm: torch.Tensor,  # (B, (1+)Pnn, 11)
            target_current_mask: torch.Tensor,  # (B, (1+)Pnn)
    ) -> torch.Tensor:
        """그룹별 처리 후 합쳐서 state_token_in을 만듭니다."""
        # xy: (B, P, 2), cos_sin: (B, P, 2), size_wl: (B, P, 2), type_one_hot: (B, P, 3)
        xy, cos_sin, size_wl, type_one_hot = self._split_groups(target_cur_norm)

        cos_sin_unit: torch.Tensor = self._normalize_cos_sin(
            cos_sin)  # (B, P, 2)

        # 그룹별 입력 정리
        xy_in: torch.Tensor = self.xy_norm(xy)  # (B, P, 2)
        heading_in: torch.Tensor = self.heading_norm(cos_sin_unit)  # (B, P, 2)
        size_in: torch.Tensor = self.size_norm(size_wl)  # (B, P, 2)
        type_in: torch.Tensor = self.type_norm(type_one_hot)  # (B, P, 3)

        # 그룹별 특징 추출
        xy_feat: torch.Tensor = self.xy_mlp(xy_in)  # (B, P, O)
        heading_feat: torch.Tensor = self.heading_mlp(heading_in)  # (B, P, O)
        size_feat: torch.Tensor = self.size_mlp(size_in)  # (B, P, O)
        type_feat: torch.Tensor = self.type_mlp(type_in)  # (B, P, O)

        # ✅ 크기+종류 결합(파라미터 없음)
        shape_feat: torch.Tensor = self._combine_size_and_type_features(
            size_feat=size_feat, type_feat=type_feat)  # (B, P, O)

        # 그룹 사이 “동시 패턴”(파라미터 없이)
        xy_heading: torch.Tensor = xy_feat * heading_feat  # (B, P, O)
        heading_shape: torch.Tensor = heading_feat * shape_feat  # (B, P, O)
        shape_xy: torch.Tensor = shape_feat * xy_feat  # (B, P, O)

        # fuse_in: (B, P, 6*O)
        fuse_in: torch.Tensor = torch.cat(
            [
                xy_feat, heading_feat, shape_feat, xy_heading, heading_shape,
                shape_xy
            ],
            dim=-1,
        )

        # state_token_in: (B, P, D)
        state_token_in: torch.Tensor = self.fuse_mlp(fuse_in)

        # 무효 에이전트는 0으로 정리
        state_token_in = self._mask_zero(state_token_in, target_current_mask)
        return state_token_in


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
    """학습 파라미터가 없는 RMSNorm (스케일만 정리).

    - 입력의 마지막 축(H)을 기준으로, 값의 "크기"를 이용해 나눠서 정리합니다.
    - 입력이 전부 0인 경우에도 0으로 나누는 문제가 생기지 않도록 안전장치를 둡니다.
    - 연산 속도를 위해 작은 숫자 형식(float16/bfloat16)을 쓸 때도 안전하도록,
      분모 계산은 float32로 수행한 뒤 원래 형식으로 되돌립니다.

    Args:
        eps (float):
            분모가 너무 작아지는 것을 막기 위한 아주 작은 양수입니다.
            기본값은 float16 환경에서도 안전한 1e-6 입니다.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """정규화(정리)를 적용합니다.

        Args:
            x (torch.Tensor):
                입력 텐서
                shape: (..., H)

        Returns:
            torch.Tensor:
                정리된 텐서 (입력과 같은 dtype)
                shape: (..., H)
        """
        # x: [..., H]
        # float16/bfloat16에서는 eps가 0으로 취급되거나(또는 분모가 0이 되는) 문제가 날 수 있어
        # 분모 계산만 float32로 올려서 안전하게 처리합니다.
        if x.dtype in (torch.float16, torch.bfloat16):
            x_compute: torch.Tensor = x.to(dtype=torch.float32)  # [..., H]
            compute_dtype = torch.float32
        else:
            x_compute = x
            compute_dtype = x.dtype

        # mean_sq: [..., 1]
        mean_sq: torch.Tensor = x_compute.pow(2).mean(dim=-1, keepdim=True)

        # eps를 더하고, 최소값을 보장해 분모가 0으로 가지 않게 합니다.
        eps_t: torch.Tensor = torch.tensor(self.eps, dtype=compute_dtype, device=x_compute.device)
        mean_sq = (mean_sq + eps_t).clamp_min(eps_t)  # [..., 1]

        # rms: [..., 1]
        rms: torch.Tensor = mean_sq.sqrt()

        # y: [..., H]
        y: torch.Tensor = x_compute / rms

        # 원래 dtype으로 복원
        return y.to(dtype=x.dtype)


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
            target_current_mask: torch.Tensor,  # [B, (1+)Pnn]  (True=무효)
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

        # --- S 경로 ---
        S_in = self.in_norm_S(state_token_in)  # [B,(1+)Pnn,D]
        s = self.adapt_S(S_in)  # [B,(1+)Pnn,h]
        s = self.rms_pre(s)
        s = s.masked_fill(target_current_mask, 0.0)  # ★ 무효 agent는 S 경로 0
        """
        5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
        """
        # --- 헤드 ---
        delta_scale_base = self.head_delta_scale(s)  # [B,(1+)Pnn,H]
        shift_base = self.head_shift(s)  # [B,(1+)Pnn,H]
        logit_gate_base = self.head_logit_gate(s)  # [B,(1+)Pnn,H]

        # ★ 최종 출력도 무효 agent에서는 모두 0 보장
        delta_scale_base = delta_scale_base.masked_fill(target_current_mask,
                                                        0.0)  # [B,(1+)Pnn,H]
        shift_base = shift_base.masked_fill(target_current_mask, 0.0)  # [B,(1+)Pnn,H]
        logit_gate_base = logit_gate_base.masked_fill(target_current_mask,
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

    def forward(self, t_embedding: torch.Tensor,
                ) -> TimeModulationOutputs:
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
    target_current_mask: torch.Tensor, # [B, (1+)Pnn]
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

        # 무효 에이전트는 모두 0으로 정리
        delta_scale = delta_scale.masked_fill(target_current_mask, 0.0)
        shift = shift.masked_fill(target_current_mask, 0.0)
        gate = gate.masked_fill(target_current_mask, 0.0)

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
target_current_mask: torch.Tensor,  # [B, (1+)Pnn]  (True=무효)
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

    # 무효 에이전트는 모두 0으로 정리
    delta_scale_final = delta_scale_final.masked_fill(target_current_mask, 0.0)
    shift_final = shift_final.masked_fill(target_current_mask, 0.0)

    # LN → (1+Δs) ⊙ · + b → Linear
    y = final_norm(x)  # [B,(1+)Pnn,H]
    y_tilde = y * (1.0 + delta_scale_final) + shift_final  # [B,(1+)Pnn,H]
    x_out = out_proj(y_tilde)  # [B,(1+)Pnn,T*4]
    return x_out
