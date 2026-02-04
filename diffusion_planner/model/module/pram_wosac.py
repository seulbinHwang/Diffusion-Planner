# -*- coding: utf-8 -*-
# file: diffusion_planner/model/module/pram_v2.py
from __future__ import annotations
from flash_attn.bert_padding import unpad_input, pad_input
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
from timm.models.layers import Mlp

PathName = Literal["SA", "FFN", "CA"]
# -*- coding: utf-8 -*-


class PRAMV2StateTokenEncoder(nn.Module):
    """현재 프레임 상태를 더 풍부한 토큰으로 바꿉니다(그룹별 처리 후 합치기).

    변경점(성능)
    ----------
    1) target_current_mask=True(무효/패딩) 에이전트는 아예 계산에서 제외합니다.
       - 유효 에이전트만 unpad_input으로 뽑아서 계산
       - 결과를 pad_input으로 원래 자리로 되돌림
       - 무효 에이전트 결과는 자동으로 0이 됩니다.

    2) xy/heading/size/type 4개 작은 MLP를 "블록 대각선"으로 묶어
       fc1 1번 + fc2 1번 호출로 수학적으로 동일하게 계산합니다.
       - 훈련: torch.block_diag로 매번 구성(그래디언트 정상 전파)
       - 추론(eval): 한 번 만든 가중치를 캐시해서 재사용
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

        self.branch_hidden_dim = int(b_hidden)
        self.branch_out_dim = int(b_out)
        self.size_type_hidden_dim = int(st_hidden)
        self.fuse_hidden_dim = int(f_hidden)

        # 옵션: 필요하면 외부에서 False로 꺼도 됩니다.
        self.enable_varlen_unpad: bool = True
        self.enable_fused_group_mlp: bool = True  # 4개 MLP -> 블록대각선 1회 호출

        # 그룹별 입력 정리(학습 파라미터 없음)
        self.xy_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.heading_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.size_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()
        self.type_norm = RMSNormNoParam() if use_rmsnorm else nn.Identity()

        # 그룹별 전용 작은 네트워크(파라미터는 그대로 유지: 기존 체크포인트 호환에 유리)
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

        # --- eval용 캐시(블록대각선 가중치) ---
        # persistent=False: state_dict에 저장하지 않음(로드/세이브 영향 최소화)
        self.register_buffer("_fused_fc1_w", torch.empty(0), persistent=False)
        self.register_buffer("_fused_fc1_b", torch.empty(0), persistent=False)
        self.register_buffer("_fused_fc2_w", torch.empty(0), persistent=False)
        self.register_buffer("_fused_fc2_b", torch.empty(0), persistent=False)

    @staticmethod
    def _default_branch_hidden_dim(hidden_dim: int,
                                   branch_hidden_dim: Optional[int]) -> int:
        if branch_hidden_dim is not None:
            return int(branch_hidden_dim)
        return max(32, int(hidden_dim // 2))

    @staticmethod
    def _default_branch_out_dim(hidden_dim: int,
                                branch_out_dim: Optional[int]) -> int:
        if branch_out_dim is not None:
            return int(branch_out_dim)
        return max(16, int(hidden_dim // 3))

    @staticmethod
    def _default_size_type_hidden_dim(
            branch_hidden_dim: int, size_type_hidden_dim: Optional[int]) -> int:
        if size_type_hidden_dim is not None:
            return int(size_type_hidden_dim)
        return max(32, int(branch_hidden_dim * 5 // 8))

    @staticmethod
    def _default_fuse_hidden_dim(branch_out_dim: int,
                                 fuse_hidden_dim: Optional[int]) -> int:
        if fuse_hidden_dim is not None:
            return int(fuse_hidden_dim)
        return int(branch_out_dim)

    @staticmethod
    def _normalize_cos_sin(cos_sin: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
        """(cos, sin)의 길이를 1로 정리합니다.

        Args:
            cos_sin (torch.Tensor): 방향 정보
                - shape: (..., 2)
            eps (float): 0 나눗셈 방지

        Returns:
            torch.Tensor:
                - shape: (..., 2)
        """
        raw_norm: torch.Tensor = torch.linalg.norm(cos_sin,
                                                   dim=-1,
                                                   keepdim=True)  # (..., 1)
        safe_norm: torch.Tensor = raw_norm.clamp_min(float(eps))
        normalized: torch.Tensor = cos_sin / safe_norm

        ones: torch.Tensor = torch.ones_like(normalized[..., :1])
        zeros: torch.Tensor = torch.zeros_like(normalized[..., :1])
        fallback: torch.Tensor = torch.cat([ones, zeros], dim=-1)  # (..., 2)

        too_small: torch.Tensor = (raw_norm < float(eps))
        normalized = torch.where(too_small, fallback, normalized)
        return normalized

    def _split_groups(
        self,
        target_cur_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """입력을 (x,y)/(cos,sin)/(w,l)/type으로 나눕니다.

        Args:
            target_cur_norm (torch.Tensor):
                - shape: (..., 11)

        Returns:
            xy:          (..., 2)
            cos_sin:      (..., 2)
            size_wl:      (..., 2)
            type_one_hot: (..., 3)
        """
        xy: torch.Tensor = target_cur_norm[..., 0:2]
        cos_sin: torch.Tensor = target_cur_norm[..., 2:4]
        size_wl: torch.Tensor = target_cur_norm[..., 6:8]
        type_one_hot: torch.Tensor = target_cur_norm[..., 8:11]
        return xy, cos_sin, size_wl, type_one_hot

    @staticmethod
    def _combine_size_and_type_features(
        size_feat: torch.Tensor,
        type_feat: torch.Tensor,
    ) -> torch.Tensor:
        """크기 특징과 종류 특징을 하나의 특징으로 합칩니다.

        규칙(파라미터 없음):
            shape_feat = size_feat + type_feat + (size_feat * type_feat)

        Args:
            size_feat (torch.Tensor): (..., O)
            type_feat (torch.Tensor): (..., O)

        Returns:
            torch.Tensor: (..., O)
        """
        return size_feat + type_feat + (size_feat * type_feat)

    def _ddp_touch_scalar(self) -> torch.Tensor:
        """(예외 케이스) 유효 토큰이 0개일 때도 파라미터가 '사용된 것'으로 보이게 합니다.

        Returns:
            torch.Tensor:
                - shape: ()  (스칼라)
        """
        # 아주 작은 일부만 더해도 목적(파라미터 사용 흔적)에는 충분합니다.
        touch = (self.xy_mlp.fc1.weight.view(-1)[:1].sum() +
                 self.xy_mlp.fc2.weight.view(-1)[:1].sum() +
                 self.fuse_mlp.fc1.weight.view(-1)[:1].sum() +
                 self.fuse_mlp.fc2.weight.view(-1)[:1].sum()) * 0.0
        return touch

    def _unpad_valid_agents(
            self,
            target_cur_norm: torch.Tensor,  # (B, P, 11)
            target_current_mask: torch.Tensor,  # (B, P)  True=무효
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """유효 에이전트만 뽑아서 2D로 펴 줍니다.

        Args:
            target_cur_norm (torch.Tensor):
                - shape: (B, P, 11)
            target_current_mask (torch.Tensor):
                - shape: (B, P)
                - True=무효(패딩), False=유효

        Returns:
            x_unpad (torch.Tensor):
                - shape: (T_total, 11)  (T_total=유효 토큰 총합)
            indices (torch.Tensor):
                - shape: (T_total,)
            batch_size (int): B
            agent_slots (int): P
        """
        B: int = int(target_cur_norm.shape[0])
        P: int = int(target_cur_norm.shape[1])

        mask_bool = target_current_mask.to(dtype=torch.bool)  # (B, P)
        valid_mask = (~mask_bool)  # (B, P) True=유효

        res = unpad_input(target_cur_norm, valid_mask)
        if len(res) == 4:
            x_unpad, indices, _, _ = res
        else:
            x_unpad, indices, _, _, _ = res
        return x_unpad, indices, B, P

    def _pad_back_agents(
            self,
            token_unpad: torch.Tensor,  # (T_total, D)
            indices: torch.Tensor,  # (T_total,)
            batch_size: int,  # B
            agent_slots: int,  # P
    ) -> torch.Tensor:
        """unpad 상태의 결과를 원래 (B,P,*)로 되돌립니다.

        Args:
            token_unpad (torch.Tensor):
                - shape: (T_total, D)
            indices (torch.Tensor):
                - shape: (T_total,)
            batch_size (int): B
            agent_slots (int): P

        Returns:
            torch.Tensor:
                - shape: (B, P, D)
                - pad 위치는 0
        """
        return pad_input(token_unpad, indices, int(batch_size),
                         int(agent_slots))

    def _build_fused_group_weights_train(
        self,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """훈련 모드에서 4개 그룹 MLP 가중치를 블록 대각선으로 묶어 반환합니다.

        반환 텐서는 원래 파라미터로부터 만들어지므로, 그래디언트가 정상적으로 전파됩니다.

        Returns:
            fc1_w: (H_sum, In_sum)
            fc1_b: (H_sum,)
            fc2_w: (Out_sum, H_sum)
            fc2_b: (Out_sum,)
        """
        # fc1 blocks
        w1_xy = self.xy_mlp.fc1.weight
        w1_hd = self.heading_mlp.fc1.weight
        w1_sz = self.size_mlp.fc1.weight
        w1_tp = self.type_mlp.fc1.weight
        fc1_w = torch.block_diag(w1_xy, w1_hd, w1_sz, w1_tp)

        def _bias_or_zeros(bias: Optional[torch.Tensor],
                           like: torch.Tensor) -> torch.Tensor:
            if bias is None:
                return torch.zeros((like.shape[0],),
                                   device=like.device,
                                   dtype=like.dtype)
            return bias

        b1_xy = _bias_or_zeros(self.xy_mlp.fc1.bias, w1_xy)
        b1_hd = _bias_or_zeros(self.heading_mlp.fc1.bias, w1_hd)
        b1_sz = _bias_or_zeros(self.size_mlp.fc1.bias, w1_sz)
        b1_tp = _bias_or_zeros(self.type_mlp.fc1.bias, w1_tp)
        fc1_b = torch.cat([b1_xy, b1_hd, b1_sz, b1_tp], dim=0)

        # fc2 blocks
        w2_xy = self.xy_mlp.fc2.weight
        w2_hd = self.heading_mlp.fc2.weight
        w2_sz = self.size_mlp.fc2.weight
        w2_tp = self.type_mlp.fc2.weight
        fc2_w = torch.block_diag(w2_xy, w2_hd, w2_sz, w2_tp)

        b2_xy = _bias_or_zeros(self.xy_mlp.fc2.bias, w2_xy)
        b2_hd = _bias_or_zeros(self.heading_mlp.fc2.bias, w2_hd)
        b2_sz = _bias_or_zeros(self.size_mlp.fc2.bias, w2_sz)
        b2_tp = _bias_or_zeros(self.type_mlp.fc2.bias, w2_tp)
        fc2_b = torch.cat([b2_xy, b2_hd, b2_sz, b2_tp], dim=0)

        return fc1_w, fc1_b, fc2_w, fc2_b

    def _maybe_rebuild_fused_group_cache_eval(self) -> None:
        """추론(eval) 모드에서 블록 대각선 가중치를 1회 캐시로 만들어 둡니다."""
        w_ref = self.xy_mlp.fc1.weight
        need = (self._fused_fc1_w.numel() == 0 or
                self._fused_fc1_w.device != w_ref.device or
                self._fused_fc1_w.dtype != w_ref.dtype)
        if not need:
            return

        with torch.no_grad():
            fc1_w, fc1_b, fc2_w, fc2_b = self._build_fused_group_weights_train()
            self._fused_fc1_w = fc1_w.detach()
            self._fused_fc1_b = fc1_b.detach()
            self._fused_fc2_w = fc2_w.detach()
            self._fused_fc2_b = fc2_b.detach()

    def _group_mlps_forward(
        self,
        xy_in: torch.Tensor,  # (N, 2)
        heading_in: torch.Tensor,  # (N, 2)
        size_in: torch.Tensor,  # (N, 2)
        type_in: torch.Tensor,  # (N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """4개 그룹 MLP를 계산합니다(합쳐서 1~2번 호출로 처리).

        Args:
            xy_in: (N, 2)
            heading_in: (N, 2)
            size_in: (N, 2)
            type_in: (N, 3)

        Returns:
            xy_feat: (N, O)
            heading_feat: (N, O)
            size_feat: (N, O)
            type_feat: (N, O)
        """
        if not self.enable_fused_group_mlp:
            xy_feat = self.xy_mlp(xy_in)
            heading_feat = self.heading_mlp(heading_in)
            size_feat = self.size_mlp(size_in)
            type_feat = self.type_mlp(type_in)
            return xy_feat, heading_feat, size_feat, type_feat

        x_cat = torch.cat([xy_in, heading_in, size_in, type_in],
                          dim=-1)  # (N, 9)

        if self.training:
            fc1_w, fc1_b, fc2_w, fc2_b = self._build_fused_group_weights_train()
        else:
            self._maybe_rebuild_fused_group_cache_eval()
            fc1_w, fc1_b = self._fused_fc1_w, self._fused_fc1_b
            fc2_w, fc2_b = self._fused_fc2_w, self._fused_fc2_b

        h = F.linear(x_cat, fc1_w, fc1_b)  # (N, H_sum)
        h = F.gelu(h)  # (N, H_sum)
        y = F.linear(h, fc2_w, fc2_b)  # (N, 4*O)

        O = int(self.branch_out_dim)
        xy_feat, heading_feat, size_feat, type_feat = y.split(O, dim=-1)
        return xy_feat, heading_feat, size_feat, type_feat

    def _forward_unpadded_tokens(
            self,
            target_cur_norm_unpad: torch.Tensor,  # (T_total, 11)
    ) -> torch.Tensor:
        """유효 토큰(2D)만으로 state_token_in을 계산합니다.

        Args:
            target_cur_norm_unpad:
                - shape: (T_total, 11)

        Returns:
            state_token_unpad:
                - shape: (T_total, D)
        """
        xy, cos_sin, size_wl, type_one_hot = self._split_groups(
            target_cur_norm_unpad)  # (T,2),(T,2),(T,2),(T,3)
        cos_sin_unit = self._normalize_cos_sin(cos_sin)  # (T_total, 2)

        xy_in = self.xy_norm(xy)  # (T_total, 2)
        heading_in = self.heading_norm(cos_sin_unit)  # (T_total, 2)
        size_in = self.size_norm(size_wl)  # (T_total, 2)
        type_in = self.type_norm(type_one_hot)  # (T_total, 3)

        xy_feat, heading_feat, size_feat, type_feat = self._group_mlps_forward(
            xy_in=xy_in,
            heading_in=heading_in,
            size_in=size_in,
            type_in=type_in,
        )  # 각 (T_total, O)

        shape_feat = self._combine_size_and_type_features(
            size_feat=size_feat, type_feat=type_feat)  # (T_total, O)

        xy_heading = xy_feat * heading_feat  # (T_total, O)
        heading_shape = heading_feat * shape_feat  # (T_total, O)
        shape_xy = shape_feat * xy_feat  # (T_total, O)

        fuse_in = torch.cat(
            [
                xy_feat, heading_feat, shape_feat, xy_heading, heading_shape,
                shape_xy
            ],
            dim=-1,
        )  # (T_total, 6*O)

        state_token_unpad = self.fuse_mlp(fuse_in)  # (T_total, D)
        return state_token_unpad

    def forward(
            self,
            target_cur_norm: torch.Tensor,  # (B, (1+)Pnn, 11)
            target_current_mask: torch.Tensor,  # (B, (1+)Pnn) True=무효
    ) -> torch.Tensor:
        """그룹별 처리 후 합쳐서 state_token_in을 만듭니다.

        Returns:
            torch.Tensor:
                - shape: (B, (1+)Pnn, D)
                - 무효(패딩) 에이전트 위치는 0
        """
        if not self.enable_varlen_unpad:
            # (기존 방식) 전체 계산 후 마스크로 0 처리
            xy, cos_sin, size_wl, type_one_hot = self._split_groups(
                target_cur_norm)
            cos_sin_unit = self._normalize_cos_sin(cos_sin)

            xy_in = self.xy_norm(xy)
            heading_in = self.heading_norm(cos_sin_unit)
            size_in = self.size_norm(size_wl)
            type_in = self.type_norm(type_one_hot)

            xy_feat, heading_feat, size_feat, type_feat = self._group_mlps_forward(
                xy_in=xy_in.reshape(-1, 2),
                heading_in=heading_in.reshape(-1, 2),
                size_in=size_in.reshape(-1, 2),
                type_in=type_in.reshape(-1, 3),
            )
            B, P = target_cur_norm.shape[:2]
            O = self.branch_out_dim
            xy_feat = xy_feat.reshape(B, P, O)
            heading_feat = heading_feat.reshape(B, P, O)
            size_feat = size_feat.reshape(B, P, O)
            type_feat = type_feat.reshape(B, P, O)

            shape_feat = self._combine_size_and_type_features(
                size_feat=size_feat, type_feat=type_feat)
            xy_heading = xy_feat * heading_feat
            heading_shape = heading_feat * shape_feat
            shape_xy = shape_feat * xy_feat

            fuse_in = torch.cat([
                xy_feat, heading_feat, shape_feat, xy_heading, heading_shape,
                shape_xy
            ],
                                dim=-1)
            state_token_in = self.fuse_mlp(fuse_in)

            mask_bool = target_current_mask.to(torch.bool)
            return state_token_in.masked_fill(mask_bool.unsqueeze(-1), 0.0)

        # ✅ (조언 1) 유효 에이전트만 뽑아서 계산
        x_unpad, indices, B, P = self._unpad_valid_agents(
            target_cur_norm=target_cur_norm,
            target_current_mask=target_current_mask,
        )  # x_unpad: (T_total, 11)

        if x_unpad.numel() == 0:
            zeros = target_cur_norm.new_zeros((B, P, self.hidden_dim))
            return zeros + self._ddp_touch_scalar()

        token_unpad = self._forward_unpadded_tokens(x_unpad)  # (T_total, D)

        # ✅ 결과를 원래 자리로 복원(pad 위치는 0)
        token = self._pad_back_agents(
            token_unpad=token_unpad,
            indices=indices,
            batch_size=B,
            agent_slots=P,
        )  # (B, P, D)
        return token


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
        eps_t: torch.Tensor = torch.tensor(self.eps,
                                           dtype=compute_dtype,
                                           device=x_compute.device)
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
        s = self.rms_pre(s)  # s: [B,(1+)Pnn,h]
        mask = target_current_mask.to(torch.bool)  # [B, P]
        s = s.masked_fill(mask.unsqueeze(-1),
                          0.0)  # [B, P, 1] -> [B, P, H]로 방송됨
        """
        5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
        """
        # --- 헤드 ---
        delta_scale_base = self.head_delta_scale(s)  # [B,(1+)Pnn,H]
        shift_base = self.head_shift(s)  # [B,(1+)Pnn,H]
        logit_gate_base = self.head_logit_gate(s)  # [B,(1+)Pnn,H]

        # ★ 최종 출력도 무효 agent에서는 모두 0 보장
        delta_scale_base = delta_scale_base.masked_fill(mask.unsqueeze(-1), 0.0)
        shift_base = shift_base.masked_fill(mask.unsqueeze(-1), 0.0)
        logit_gate_base = logit_gate_base.masked_fill(mask.unsqueeze(-1), 0.0)

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

    def forward(
        self,
        t_embedding: torch.Tensor,
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
        target_current_mask: torch.Tensor,  # [B, (1+)Pnn]
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
        target_current_mask = target_current_mask.to(torch.bool)
        # 무효 에이전트는 모두 0으로 정리
        delta_scale = delta_scale.masked_fill(target_current_mask.unsqueeze(-1),
                                              0.0)
        shift = shift.masked_fill(target_current_mask.unsqueeze(-1), 0.0)
        gate = gate.masked_fill(target_current_mask.unsqueeze(-1), 0.0)

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
    target_current_mask = target_current_mask.to(torch.bool)
    delta_scale_final = delta_scale_final.masked_fill(
        target_current_mask.unsqueeze(-1), 0.0)
    shift_final = shift_final.masked_fill(target_current_mask.unsqueeze(-1),
                                          0.0)

    # LN → (1+Δs) ⊙ · + b → Linear
    y = final_norm(x)  # [B,(1+)Pnn,H]
    y_tilde = y * (1.0 + delta_scale_final) + shift_final  # [B,(1+)Pnn,H]
    x_out = out_proj(y_tilde)  # [B,(1+)Pnn,T*4]
    return x_out
