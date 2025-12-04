import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from typing import Optional, Dict, Tuple
from flash_attn.bert_padding import unpad_input, pad_input
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, \
    StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock
from diffusion_planner.loss import _require_finite
# decoder.py 상단 import 섹션에 추가
from diffusion_planner.model.module.pram_v2 import (
    PRAMV2Composer,
    PRAMV2TimeModulator,
    PRAMV2BlockPathScalars,
    PRAMV2StateTokenEncoder,
    compute_pram_v2_modulations_for_block,
    apply_pram_v2_final_layer,  # ← 9단계 마무리 보정 호출용(스켈레톤이어도 OK)
)
from typing import Tuple, Optional
from diffusion_planner.model.module.feasible import FeasibleProjector
from diffusion_planner.loss import AMP_DTYPE

from typing import NamedTuple

# decoder.py 또는 feasible.py 상단 import 근처에 추가
import time
from contextlib import contextmanager
from typing import Iterator
import torch


def compute_cond_last_mask(cond_last_pos: torch.Tensor) -> torch.Tensor:
    """마지막 위치 조건(cond_last_pos)이 '실제로 존재하는지'를 판단하는 마스크를 만든다.

    Args:
        cond_last_pos: 마지막 위치 조건 텐서.
            shape: (batch_size, num_agents, D)
            - 유효한 goal이면 finite(real 숫자) + 0-벡터가 아닌 값
            - goal이 없으면:
                - NaN / inf 로 채우거나, 또는
                - (0, 0, 0, 0) 같은 0-벡터 sentinel 로 표현된다고 가정

    Returns:
        cond_last_mask: shape (batch_size, num_agents) 의 bool 텐서.
            - True  : 유효한 goal이 존재함
            - False : goal이 없거나, 값이 비정상(NaN/inf) 이거나, 0-벡터 sentinel 인 경우
    """
    # 1) 값이 유한한지 (NaN/inf 아닌지)
    is_finite: torch.Tensor = torch.isfinite(cond_last_pos).all(
        dim=-1)  # (B, Pnn)

    # 2) 완전한 0-벡터가 아닌지
    #    (필요하면 여기서 eps 기준으로 바꿀 수도 있음: (cond_last_pos.abs() > eps).any(dim=-1))
    is_non_zero: torch.Tensor = cond_last_pos.ne(0).any(dim=-1)  # (B, Pnn)

    cond_last_mask: torch.Tensor = is_finite & is_non_zero
    return cond_last_mask


@contextmanager
def profile_block(name: str,
                  enabled: bool = True,
                  device_type: str = "cuda") -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력하는 간단한 프로파일러.

    Args:
        name: 출력에 사용할 블록 이름.
        enabled: False 이면 아무 것도 하지 않고 그냥 통과.
        device_type: "cuda" 인 경우, GPU 연산 정합을 위해 앞/뒤에 synchronize 호출.
    """
    if not enabled:
        # 아무 것도 하지 않고 블록만 실행
        yield
        return

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time: float = time.perf_counter()

    yield  # 실제 코드 실행

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms: float = (time.perf_counter() - start_time) * 1000.0
    print(f"[PROFILE] {name}: {elapsed_ms:.3f} ms")


def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """ref 텐서의 dtype/device로 x를 캐스팅합니다."""
    return x.to(dtype=ref.dtype, device=ref.device)


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len: int = config.future_len
        self._sde = VPSDE_linear()
        self._cond_last_prob: float = getattr(config, "cond_last_prob",
                                              0.0)  # 20%

        if self.config.use_current_input:
            output_dim = (config.future_len + 1) * 4  # x, y, cos, sin
        else:
            output_dim = (config.future_len) * 4  # x, y, cos, sin

        self.dit = DiT(
            config=config,
            sde=self._sde,
            depth=config.decoder_depth,
            output_dim=output_dim,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type)

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = getattr(config, 'guidance_fn', None)

    @property
    def sde(self):
        return self._sde

    # TODO: 점검하기
    def _maybe_apply_last_pos_condition_training(
        self,
        near_cur_future_norm_xT: torch.Tensor,
        # (B, Pnn, (T) * 4)  # 정규화 상태
        near_current_mask: torch.Tensor,  # (B, Pnn)  True=무효 에이전트
        cond_last_pos_norm: torch.Tensor,  # (B, Pnn, 4)  # 정규화 목표점
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        훈련 시, **배치 단위**로 cond_last_prob 확률(기본 20%)에 따라
        마지막 스텝(목표점)을 관측(노이즈 0)으로 **전원 적용/전원 미적용**한다.

        배치 토글:
            - 토글=True (약 20%): 이번 배치의 모든 유효 Pnn에 목표를 주입
            - 토글=False (약 80%): 이번 배치에는 전혀 목표를 주지 않음

        Args:
            near_cur_future_norm_xT: (B, Pnn, T * 4), 정규화된 입력 시퀀스.
            near_current_mask: (B, Pnn), True=무효 이웃 에이전트.
            cond_last_pos_norm: (B, Pnn, 4), 정규화된 목표점(마지막 프레임 값).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x_t_out: (B, Pnn, T * 4)
                    * 배치 토글이 True일 때, 모든 유효 에이전트의 마지막 프레임을
                      cond_last_pos_norm으로 치환(=노이즈 0).
                    * 배치 토글이 False면 입력을 그대로 반환.
                - apply_mask: (B, Pnn) 또는 None
                    * 이번 배치에서 실제로 목표를 적용한 에이전트 마스크(로깅/평가용).
                    * 미적용 시 None.
        """
        B, Pnn, flat = near_cur_future_norm_xT.shape
        assert flat % 4 == 0, "near_cur_future_norm_xT의 마지막 차원은 4의 배수여야 합니다."
        T_plus1: int = flat // 4
        assert T_plus1 >= 2, "미래 길이 T는 최소 1 이상이어야 합니다."
        assert cond_last_pos_norm.shape == (B, Pnn,
                                            4), "cond_last_pos_norm shape 불일치"
        cond_last_pos_norm = _cast_like(cond_last_pos_norm,
                                        near_cur_future_norm_xT)
        # cond_last_prob==0 이면 빠르게 종료
        if self._cond_last_prob <= 0.0:
            return near_cur_future_norm_xT, None

        # (1) 적용 가능 위치: 유효 에이전트 & has_goal(목표값이 전부 0 이 아닌)
        has_goal: torch.Tensor = torch.isfinite(cond_last_pos_norm).all(
            dim=-1) & (cond_last_pos_norm.ne(0).sum(dim=-1) > 0)  # (B, Pnn)
        can_apply: torch.Tensor = (~near_current_mask) & has_goal  # (B, Pnn)

        # (2) 배치 단일 베르누이 샘플: 약 20% 확률로 전체 적용
        batch_toggle: bool = (torch.rand(
            (), device=near_cur_future_norm_xT.device).item()
                              < float(self._cond_last_prob))

        if not batch_toggle:
            # 이번 배치는 전원 미적용
            return near_cur_future_norm_xT, None

        # (3) 전체 적용: 유효한 위치(can_apply)에만 주입
        if not can_apply.any().item():
            # 유효한 에이전트가 하나도 없으면 변경 없음
            return near_cur_future_norm_xT, None

        x_seq = near_cur_future_norm_xT.view(B, Pnn, T_plus1,
                                             4).clone()  # (B, Pnn, 1 + T, 4)

        # 마지막 프레임만 뽑아서 (B, Pnn, 4) 모양으로 맞춘 뒤 where로 바꿔끼우기
        last = x_seq[:, :, -1, :]  # (B, Pnn, 4)
        apply_cond = can_apply.unsqueeze(-1)  # (B, Pnn, 1) → 4에 브로드캐스트
        # cond_last_pos_norm은 이미 _cast_like로 dtype/device 맞춤
        last = torch.where(apply_cond, cond_last_pos_norm, last)  # (B, Pnn, 4)
        x_seq[:, :, -1, :] = last

        # x_out: (B, Pnn, T * 4)
        # can_apply: (B, Pnn)
        x_out = x_seq.view(B, Pnn, flat)
        return x_out, can_apply

    def _get_near_past_current_infos(
        self,
        target_agents_mask: Optional[torch.Tensor],  # [B, max_agent_num] bool
        neighbor_agents_past: torch.Tensor,  # [B, max_agent_num, time_len, 11]
            predicted_agents_num: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            near_past_current: [B, pnn, time_len, 11]
            near_past_current_mask: [B, pnn, time_len]  True=빈 슬롯(무효 에이전트)
            near_class_one_hot: [B, pnn, 3]  one-hot class vector
        """
        # neighbor_agents_past_mask : [B, max_agent_num, time_len] bool # True=무효 점
        neighbor_agents_past_mask = torch.sum(
            torch.ne(neighbor_agents_past,
                     0), dim=-1) == 0  # (B, max_agent_num, time_len)

        neighbor_agents_class_one_hot = neighbor_agents_past[
            ..., 0, 8:11]  # (B, max_agent_num, 3)
        # 마지막 타임스텝만 추출: [B, max_agent_num, 4]

        if target_agents_mask is None:
            assert predicted_agents_num is not None, "predicted_agents_num must be provided if target_agents_mask is None"
            # near_past_current, near_past_current_mask,
            near_past_current = neighbor_agents_past[:, :predicted_agents_num]  # [B, pnn, time_len, 11]
            near_past_current_mask = neighbor_agents_past_mask[:, :predicted_agents_num]  # [B, pnn, time_len]

            near_class_one_hot = neighbor_agents_class_one_hot[:, :predicted_agents_num, :]  # [B, pnn, 3]

        else:
            B, max_agent_num, time_len, D11 = neighbor_agents_past.shape
            # near_past_current:
            near_past_current = torch.zeros(
                (B, max_agent_num, time_len, D11),
                dtype=neighbor_agents_past.dtype,
                device=neighbor_agents_past.device,
            )
            near_past_current_mask = torch.ones(
                (B, max_agent_num, time_len),
                dtype=neighbor_agents_past_mask.dtype,
                device=neighbor_agents_past_mask.device,
            )  # True=빈 슬롯(무효 에이전트)
            near_class_one_hot = torch.zeros(
                (B, max_agent_num, 3),
                dtype=neighbor_agents_class_one_hot.dtype,
                device=neighbor_agents_class_one_hot.device,
            )

            # 마스크가 True인 “그 자리”에 값 대입 (슬롯 유지)
            # near_past_current: [B, max_agent_num, time_len, 11]
            near_past_current[target_agents_mask] = neighbor_agents_past[
                target_agents_mask]
            # near_past_current_mask : [B, max_agent_num, time_len]  True=빈 슬롯(무효 점)
            near_past_current_mask[
                target_agents_mask] = neighbor_agents_past_mask[
                    target_agents_mask]

            near_class_one_hot[
                target_agents_mask] = neighbor_agents_class_one_hot[
                    target_agents_mask]  #
            # TODO: 임시 -> _predicted_neighbor_num 에 대한 의존성 타파?
            near_past_current = near_past_current[:, :predicted_agents_num]  # [B, pnn, time_len, 11]
            near_class_one_hot = near_class_one_hot[:, :predicted_agents_num, :]  # [B, pnn, 3]
            # near_past_current_mask : [B, pnn, time_len]  True=빈 슬롯(무효 에이전트)
            near_past_current_mask = near_past_current_mask[:, :predicted_agents_num]

        return near_past_current, near_past_current_mask, near_class_one_hot

    def _get_near_past_cur_future_valid(
        self,
        near_past_current_mask: torch.
        Tensor,  # [B, pnn, time_len]  True=빈 슬롯(무효 에이전트)
        near_future_valid: Optional[
            torch.Tensor] = None,  # [B, pnn, future_len] bool
    ) -> torch.Tensor:  # [B, pnn, 1 + future_len] bool
        near_past_current_valid = ~near_past_current_mask  # [B, pnn, time_len]  True=유효 에이전트
        near_current_valid = near_past_current_valid[:, :, -1]  # [B, pnn]

        if near_future_valid is None:
            near_future_valid = near_current_valid.unsqueeze(-1).expand(
                -1, -1, self._future_len)  # [B, pnn, future_len] bool
        near_past_cur_future_valid = torch.cat(
            [near_past_current_valid, near_future_valid],
            dim=-1)  # [B, pnn, time_len + future_len] bool
        return near_past_cur_future_valid

    def _reshape_xt_with_current_state(
            self,
            xt: torch.Tensor,  # shape: (B, Pnn, flattened_dim)
            batch_size: int,
            predicted_neighbor_num: int,
            near_current_xyyaw: torch.Tensor  # shape: (B, Pnn, 4)
    ) -> torch.Tensor:
        """샘플 xt 를 (B, Pnn, 1+T, 4) 모양으로 펼치고
        '현재 상태 프레임'을 맨 앞에 올바르게 넣어주는 메서드.

        - use_current_input=True  인 경우: xt 안에 이미 현재+미래가 다 들어 있어서
          모양만 (B, Pnn, 1+T, 4) 로 바꿔주고 0번째 프레임을 현재 상태로 덮어쓴다.
        - use_current_input=False 인 경우: xt 안에는 미래 T프레임만 있고,
          0번째 프레임은 나중 단계에서 따로 채운다고 생각하고 모양만 (B, Pnn, T, 4) 로 만든다.
        """
        if self.config.use_current_input:
            # xt_reshaped: (B, Pnn, 1+T, 4)
            xt_reshaped = xt.reshape(
                batch_size,
                predicted_neighbor_num,
                1 + self._future_len,
                4,
            )
            xt_reshaped[:, :, 0, :] = near_current_xyyaw  # 현재 상태 주입 (B,Pnn,4)
        else:
            # xt_reshaped: (B, Pnn, T, 4)
            xt_reshaped = xt.reshape(
                batch_size,
                predicted_neighbor_num,
                self._future_len,
                4,
            )
        return xt_reshaped

    @staticmethod
    def _inject_last_position_condition_if_available(
        xt: torch.Tensor,  # (B, Pnn, T, C) 정도라고 가정
        cond_last_pos: Optional[torch.Tensor]  # (B, Pnn, C_pos)
    ) -> torch.Tensor:
        """cond_last_pos가 유효한 샘플에만 마지막 프레임 위치를 덮어쓴다."""
        if cond_last_pos is None:
            return xt

        cond_last_mask: torch.Tensor = compute_cond_last_mask(
            cond_last_pos)  # (B, Pnn)

        # cond_last_mask == True 인 애들만 마지막 프레임에 cond_last_pos를 주입
        # 예시: 마지막 프레임의 (x, y)만 덮어쓴다면
        xt_last = xt[..., -1, :]  # (B, Pnn, C)
        new_last = xt_last.clone()

        # cond_last_pos의 (x, y)만 사용한다고 가정
        new_last_xy = cond_last_pos[..., :2]  # (B, Pnn, 2)
        new_last[..., :2] = torch.where(
            cond_last_mask[..., None],  # (B, Pnn, 1)
            new_last_xy,
            xt_last[..., :2],
        )

        xt[..., -1, :] = new_last
        return xt

    def _project_future_yaw_to_unit_circle(
            self,
            xt_sequence: torch.Tensor,  # shape: (B, Pnn, time_len, 4)
    ) -> torch.Tensor:
        """각 프레임의 (cos, sin) 부분이 길이 1 이 되도록
        단위원 위로 다시 정리해 주는 메서드.

        - yaw 벡터가 (0,0)에 가까운 경우를 대비해서 작은 epsilon 으로 나눗셈을 안정화한다.
        """
        # yaw_direction: (B, Pnn, time_len, 2)
        yaw_direction = xt_sequence[:, :, :, 2:4]

        # yaw_norm: (B, Pnn, time_len, 1)
        yaw_norm = torch.linalg.norm(
            yaw_direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        # 정규화된 yaw 를 다시 써 넣기
        xt_sequence[:, :, :,
                    2:4] = yaw_direction / yaw_norm  # (B,Pnn,time_len,2)
        return xt_sequence

    def _compute_feasible_blend_beta(
            self,
            diffusion_time_step: torch.Tensor,  # shape: () or (1,)  
            reference_tensor_for_device: torch.
        Tensor,  # shape: (B, Pnn, time_len, 4)  
    ) -> torch.Tensor:  # shape: ()
        """DPM-Solver가 넘겨주는 현재 시간 t 로 prox-snap 비율 β(t)를 계산하는 함수.  

        - t 값이 클 때(초반, 노이즈 많을 때)는 β=0 으로 두고 아무 것도 하지 않는다.  
        - t 값이 threshold 이하(후반, 노이즈가 거의 빠진 구간)에서만 β를 0→β_max 로 서서히 키운다.  
        - β_max, 지수 p 는 config 에서 읽어오되, 없으면 기본값(0.1, 1.0)을 쓴다.  
        """

        # 학습 때 쓰던 threshold 그대로 재사용 (예: 0.30)
        t_threshold: float = float(
            getattr(self.config, "feasible_learn_noise_thresh", 0.3))
        if t_threshold <= 0.0:
            return reference_tensor_for_device.new_tensor(0.0)

        beta_max: float = float(getattr(self.config, "prox_snap_beta_max", 0.1))
        beta_power: float = float(
            getattr(self.config, "prox_snap_beta_power", 1.0))
        if beta_max <= 0.0:
            return reference_tensor_for_device.new_tensor(0.0)

        # diffusion_time_step: t (shape: () or (1,))
        t_value: torch.Tensor = diffusion_time_step.detach().float()
        if t_value.dim() > 0:
            # 여러 값이 들어와도 첫 번째 값만 사용 (배치 전체가 같은 t 를 쓰기 때문)
            t_value = t_value.view(-1)[0]

        # s(t) = clip( (t_th - t) / t_th, 0, 1 )
        safe_t_threshold: float = max(t_threshold, 1e-6)
        s_raw: torch.Tensor = (t_threshold - t_value) / safe_t_threshold
        s_clamped: torch.Tensor = torch.clamp(s_raw, 0.0, 1.0)

        beta_value: torch.Tensor = beta_max * (s_clamped**beta_power)
        beta_value = beta_value.to(
            dtype=reference_tensor_for_device.dtype,
            device=reference_tensor_for_device.device,
        )
        return beta_value

    def _get_latest_feasible_integrated_trajectory_future(
        self,
        batch_size: int,
        predicted_neighbor_num: int,
        future_len: int,
        reference_tensor_for_device: torch.
        Tensor,  # shape: (B, Pnn, time_len, 4)  
    ) -> Optional[torch.Tensor]:  # returns: (B, Pnn, future_len, 4) or None
        """DiT가 방금 저장해 둔 FeasibleProjector 적분 결과를 꺼내오는 메서드.  

        - 학습/추론 공통으로 self.dit.dit_returns.integrated_trajectory 에 저장된 값을 사용한다.  
        - shape 이나 배치 크기가 안 맞으면 None 을 돌려서 prox-snap 을 건너뛴다.  
        """
        dit_returns = getattr(self.dit, "dit_returns", None)
        if dit_returns is None:
            return None

        integrated_future = getattr(dit_returns, "integrated_trajectory",
                                    None)  # (B, Pnn, T, 4)
        if integrated_future is None:
            return None
        if integrated_future.dim() != 4 or integrated_future.shape[-1] != 4:
            return None

        B_f, Pnn_f, T_f, _ = integrated_future.shape
        if B_f != batch_size or Pnn_f != predicted_neighbor_num:
            return None
        if T_f != future_len:
            # future_len 이 다르면 섞을 수 없으니 안전하게 스킵
            return None

        return integrated_future.to(  # shape: (B, Pnn, future_len, 4)  
            dtype=reference_tensor_for_device.dtype,
            device=reference_tensor_for_device.device,
        )

    def _apply_feasible_blend_with_feasible_projection(
            self,
            xt_sequence: torch.Tensor,  # shape: (B, Pnn, time_len, 4)  
            near_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, time_len_total) bool  
            cond_last_mask: torch.Tensor,  # shape: (B, Pnn) bool  
            feasible_blend_beta_scalar: torch.Tensor,  # shape: ()  
    ) -> torch.Tensor:  # shape: (B, Pnn, time_len, 4)
        """현재 샘플 궤적과 FeasibleProjector 적분 궤적을 β 비율로 섞어 주는 메서드.  

        동작 요약:  
            1) DiT 가 저장해 둔 integrated_trajectory (미래 T 프레임)을 가져온다.  
            2) 현재 xt_sequence 에서 '미래 부분'만 잘라서 (B, Pnn, T, 4) 로 맞춘다.  
            3) near_past_cur_future_valid 로 유효한 미래 타임스텝만 골라서,  
               x_new = (1-β) * x_t + β * Π(x_t) 형태로 섞는다.  
            4) cond-last 가 있는 에이전트는 마지막 프레임(T-1)은 보호한다(β=0).  
        """
        # β==0 이면 아무 것도 하지 않고 바로 반환
        if feasible_blend_beta_scalar.detach().item() <= 0.0:
            return xt_sequence

        if not getattr(self.config, "use_feasible", False):
            return xt_sequence

        batch_size, predicted_neighbor_num, time_len, _ = xt_sequence.shape
        future_len: int = int(self._future_len)

        # FeasibleProjector 적분 결과 (미래 T 프레임) 가져오기
        # integrated_future: (B, Pnn, future_len, 4) 또는 None
        integrated_future = self._get_latest_feasible_integrated_trajectory_future(
            batch_size=batch_size,
            predicted_neighbor_num=predicted_neighbor_num,
            future_len=future_len,
            reference_tensor_for_device=
            xt_sequence,  # shape: (B, Pnn, 80 or 81, 4)
        )
        if integrated_future is None:
            return xt_sequence

        # xt_sequence 에서 "미래" 부분만 (B, Pnn, future_len, 4) 형태로 뽑기
        if self.config.use_current_input:
            # xt_sequence: (B, Pnn, 1+future_len, 4)  -> 미래만 사용
            if time_len != future_len + 1:
                # time_len 이 다르면 안전하게 스킵
                return xt_sequence
            xt_future = xt_sequence[:, :, 1:, :]  # (B, Pnn, future_len, 4)
        else:
            # xt_sequence: (B, Pnn, future_len, 4) 가 바로 미래 궤적
            if time_len != future_len:
                return xt_sequence
            xt_future = xt_sequence  # (B, Pnn, future_len, 4)

        # near_past_cur_future_valid 에서 "현재+미래" 마스크만 추출
        if near_past_cur_future_valid.shape[-1] < (future_len + 1):
            return xt_sequence
        near_cur_future_valid = near_past_cur_future_valid[:, :, -(
            future_len + 1):]  # (B, Pnn, 1+T)
        future_valid = near_cur_future_valid[:, :, 1:]  # (B, Pnn, T)

        # 유효한 미래 타임스텝만 섞기 위해 마스크 생성
        mix_valid_mask = future_valid.unsqueeze(-1).to(  # (B, Pnn, T, 1)  
            dtype=xt_future.dtype)

        # cond-last 를 쓴 에이전트는 마지막 프레임(T-1)을 섞지 않도록 보호
        if cond_last_mask is not None and cond_last_mask.any():
            cond_last_mask_expanded = cond_last_mask.unsqueeze(-1).unsqueeze(
                -1)  # (B, Pnn, 1, 1)
            # 마지막 타임스텝에 대해서만 cond-last=True 인 곳은 0 으로 만든다.
            mix_valid_mask[:, :, -1:, :] = mix_valid_mask[:, :, -1:, :] * (
                (~cond_last_mask_expanded).to(mix_valid_mask.dtype))

        # β 를 (B, Pnn, T, 1) 로 브로드캐스트
        beta_broadcast = feasible_blend_beta_scalar.view(1, 1, 1,
                                                         1)  # (1,1,1,1)
        beta_mask = beta_broadcast * mix_valid_mask  # (B, Pnn, T, 1)

        # 실제로 섞기: x_new = (1-β) * x_t + β * Π(x_t)
        integrated_future = integrated_future.to(dtype=xt_future.dtype,
                                                 device=xt_future.device)
        new_future = (
            1.0 - beta_mask
        ) * xt_future + beta_mask * integrated_future  # (B, Pnn, T, 4)

        # xt_sequence 에 다시 써 넣기
        if self.config.use_current_input:
            xt_sequence[:, :, 1:, :] = new_future
        else:
            xt_sequence[:, :, :, :] = new_future

        return xt_sequence

    def _prepare_near_trajectories_and_masks(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        int, int,
    ]:
        """배치 dict에서 이웃 에이전트 관련 공통 텐서를 만들어준다.

        - 이웃 과거/현재 상태
        - 유효/무효 에이전트 마스크
        - 과거+현재+미래 유효 프레임 마스크
        - 에이전트 종류 one-hot
        - 마지막 목표 위치(cond_last_pos_norm)

        Args:
            inputs: DataLoader 에서 넘어온 배치 dict.

        Returns:
            near_past: (B, Pnn, past_len, 11) 과거 이웃 상태.
            near_current_xyyaw: (B, Pnn, 4) 현재 이웃 상태 (x, y, cos, sin).
            near_current_mask: (B, Pnn) True면 해당 위치에 이웃이 없음.
            near_past_cur_future_valid:
                (B, Pnn, time_len + future_len) 과거+현재+미래 유효 프레임 마스크.
            near_class_one_hot: (B, Pnn, 3) 에이전트 종류 one-hot.
            cond_last_pos_norm:
                (B, Pnn, 4) 정규화된 마지막 목표 위치(없으면 NaN/0 로 채움).
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.
        """
        neighbor_agents_past: torch.Tensor = inputs[
            "neighbor_agents_past"
        ]  # (B, max_agent_num, time_len, 11)

        near_future_gt_3_dim: Optional[torch.Tensor] = inputs.get(
            "near_future_gt_3_dim", None
        )  # (B, Pnn, future_len, 3) 또는 None

        predicted_agents_num: Optional[int] = (
            int(near_future_gt_3_dim.shape[1])
            if near_future_gt_3_dim is not None
            else None
        )

        target_agents_mask: Optional[torch.Tensor] = inputs.get(
            "target_agents_mask", None
        )  # (B, max_agent_num) bool 또는 None

        # near_past_current: (B, Pnn, time_len, 11)
        # near_past_current_mask: (B, Pnn, time_len)  True=빈 슬롯
        # near_class_one_hot: (B, Pnn, 3)
        (
            near_past_current,
            near_past_current_mask,
            near_class_one_hot,
        ) = self._get_near_past_current_infos(
            target_agents_mask=target_agents_mask,
            neighbor_agents_past=neighbor_agents_past,
            predicted_agents_num=predicted_agents_num,
        )

        # 과거 부분만 분리 (마지막 한 프레임은 현재)
        near_past: torch.Tensor = near_past_current[:, :, :-1, :].detach()
        # near_past: (B, Pnn, past_len, 11)

        # 현재 포함 xy-yaw 시퀀스
        near_past_current_xyyaw: torch.Tensor = near_past_current[:, :, :, :4]
        # near_past_current_xyyaw: (B, Pnn, time_len, 4)

        near_current_xyyaw: torch.Tensor = near_past_current_xyyaw[:, :, -1, :]
        # near_current_xyyaw: (B, Pnn, 4)

        near_current_mask: torch.Tensor = near_past_current_mask[:, :, -1]
        # near_current_mask: (B, Pnn)  True=빈 슬롯

        # 나중 모듈에서 쓰기 위해 inputs 에도 넣어둠(기존 로직 유지)
        inputs["near_current_mask"] = near_current_mask

        # 과거+현재+미래 유효 프레임 마스크
        near_future_valid: Optional[torch.Tensor] = inputs.get(
            "near_future_valid", None
        )  # (B, Pnn, future_len) 또는 None
        near_past_cur_future_valid: torch.Tensor = (
            self._get_near_past_cur_future_valid(
                near_past_current_mask=near_past_current_mask,
                near_future_valid=near_future_valid,
            )
        )  # (B, Pnn, time_len + future_len)

        batch_size, predicted_neighbor_num, _ = near_current_xyyaw.shape

        # cond_last_pos_norm 준비
        if "cond_last_pos_norm" in inputs:
            cond_last_pos_norm: torch.Tensor = inputs[
                "cond_last_pos_norm"
            ]  # (B, Pnn, 4)
        else:
            cond_last_pos_norm = near_current_xyyaw.new_full(
                (batch_size, predicted_neighbor_num, 4),
                float("nan"),
            )  # (B, Pnn, 4)

        # dtype/device 를 near_current_xyyaw 와 맞춤
        cond_last_pos_norm = _cast_like(
            cond_last_pos_norm, near_current_xyyaw
        )  # (B, Pnn, 4)

        return (
            near_past, # (B, Pnn, past_len, 11)
            near_current_xyyaw, # (B, Pnn, 4)
            near_current_mask, # (B, Pnn)
            near_past_cur_future_valid, # (B, Pnn, time_len + future_len)
            near_class_one_hot, # (B, Pnn, 3)
            cond_last_pos_norm, # (B, Pnn, 4)
            batch_size, # int
            predicted_neighbor_num, # int
        )

    def _unpack_encoder_outputs(
        self,
        encoder_outputs: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """ encoder_outputs dict 에서 디코더가 쓰는 텐서들을 꺼낸다.

        Args:
            encoder_outputs: Encoder 단계에서 만들어진 출력 dict.
            batch_size: 배치 크기 B (shape 확인용).

        Returns:
            scene_encoding_token: (B, token_num, D) 장면 토큰.
            scene_encoding_token_mask: (B, token_num) bool 마스크.
            ego_fut_global: (B, D) 이고 에이전트 미래 요약 벡터.
            near_agents_route_lane_emb: (B, Pnn, D) 이웃+경로 요약 벡터.
            route_known_mask: (B, Pnn) bool, 경로 유효 여부 마스크.
        """
        scene_encoding_token: torch.Tensor = encoder_outputs[
            "encoding"
        ]  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor = encoder_outputs[
            "encoding_mask"
        ]  # (B, token_num) bool
        ego_fut_global: torch.Tensor = encoder_outputs[
            "ego_fut_global"
        ]  # (B, D)
        near_agents_route_lane_emb: torch.Tensor = encoder_outputs[
            "near_agents_route_lane_emb"
        ]  # (B, Pnn, D)
        route_known_mask: torch.Tensor = encoder_outputs[
            "route_known_mask"
        ]  # (B, Pnn) bool

        # 원래 코드에 있던 shape 체크는 그대로 유지
        assert ego_fut_global.shape == (
            batch_size,
            scene_encoding_token.shape[-1],
        )

        return (
            scene_encoding_token,
            scene_encoding_token_mask,
            ego_fut_global,
            near_agents_route_lane_emb,
            route_known_mask,
        )

    def _forward_training_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        ego_fut_global: torch.Tensor,
        near_agents_route_lane_emb: torch.Tensor,
        route_known_mask: torch.Tensor,
        near_past: torch.Tensor,
        near_current_xyyaw: torch.Tensor,
        near_current_mask: torch.Tensor,
        near_past_cur_future_valid: torch.Tensor,
        near_class_one_hot: torch.Tensor,
        cond_last_pos_norm: torch.Tensor,
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> Dict[str, torch.Tensor]:
        """훈련 모드에서 한 배치에 대해 decoder 를 한 번 돌린다.

        Args:
            inputs: DataLoader 에서 온 배치 dict.
            scene_encoding_token: (B, token_num, D) 장면 토큰.
            scene_encoding_token_mask: (B, token_num) bool 마스크.
            ego_fut_global: (B, D) 이고 미래 요약 벡터.
            near_agents_route_lane_emb: (B, Pnn, D) 이웃+경로 임베딩.
            route_known_mask: (B, Pnn) bool, 경로 유효 여부.
            near_past: (B, Pnn, past_len, 11) 이웃 과거 상태.
            near_current_xyyaw: (B, Pnn, 4) 이웃 현재 상태.
            near_current_mask: (B, Pnn) True면 이웃이 없는 자리.
            near_past_cur_future_valid:
                (B, Pnn, time_len + future_len) 과거+현재+미래 유효 마스크.
            near_class_one_hot: (B, Pnn, 3) 에이전트 종류 one-hot.
            cond_last_pos_norm: (B, Pnn, 4) 정규화된 목표 위치.
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.

        Returns:
            Dict[str, torch.Tensor]: "score" (필수)와
                "integrated_trajectory", "control_constraint_diff"(옵션)를 포함한 dict.
        """
        return_: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # near_cur_future_norm_xT: (B, Pnn, 1+T, 4)
        near_cur_future_norm_xT: torch.Tensor = inputs[
            "near_cur_future_norm_xT"
        ]
        # near_cur_norm_xT: (B, Pnn, 4) 현재 프레임
        near_cur_norm_xT: torch.Tensor = near_cur_future_norm_xT[:, :, 0, :]
        # near_future_norm_xT: (B, Pnn, T, 4)
        near_future_norm_xT: torch.Tensor = near_cur_future_norm_xT[:, :, 1:, :]

        if self.config.use_current_input:
            # xT_input: (B, Pnn, 1+T, 4)
            xT_input: torch.Tensor = near_cur_future_norm_xT
        else:
            # xT_input: (B, Pnn, T, 4)
            xT_input = near_future_norm_xT

        # flatten: (B, Pnn, F)  F = (1+T)*4 또는 T*4
        xT_input = xT_input.reshape(B, Pnn, -1)

        # cond_last_pos_norm 을 사용해 마지막 프레임을 학습용으로 일부 교체
        xT_input, _ = self._maybe_apply_last_pos_condition_training(
            near_cur_future_norm_xT=xT_input,  # (B, Pnn, F)
            near_current_mask=near_current_mask,  # (B, Pnn)
            cond_last_pos_norm=cond_last_pos_norm,  # (B, Pnn, 4)
        )

        diffusion_time: torch.Tensor = inputs["diffusion_time"]  # (B,)

        # DiT 호출
        score: torch.Tensor = self.dit(
            xT_input,  # (B, Pnn, F)
            diffusion_time,  # (B,)
            scene_encoding_token,  # (B, token_num, D)
            ego_fut_global,  # (B, D)
            near_agents_route_lane_emb,  # (B, Pnn, D)
            near_past_cur_future_valid,  # (B, Pnn, time_len+future_len)
            scene_encoding_token_mask,  # (B, token_num) bool
            route_known_mask,  # (B, Pnn) bool
            near_class_one_hot,  # (B, Pnn, 3)
            near_current_xyyaw=near_cur_norm_xT,  # (B, Pnn, 4)  정규화된 현재
            near_past=near_past,  # (B, Pnn, past_len, 11)
        )
        _require_finite("decoder_dit_output", score)

        if self.config.use_current_input:
            # score: (B, Pnn, 1+T, 4)
            score = score.reshape(B, Pnn, 1 + self._future_len, 4)
        else:
            # score: (B, Pnn, T, 4)
            score = score.reshape(B, Pnn, self._future_len, 4)
            # 앞에 현재 프레임 붙이기: (B, Pnn, 1+T, 4)
            score = torch.cat(
                [near_current_xyyaw.unsqueeze(2), score],
                dim=2,
            )

        return_["score"] = score  # (B, Pnn, 1+T, 4)

        if self.config.use_feasible:
            # integrated_trajectory: (B, Pnn, T, 4)
            integrated_trajectory: torch.Tensor = (
                self.dit.dit_returns.integrated_trajectory
            )
            integrated_trajectory = torch.cat(
                [near_current_xyyaw.unsqueeze(2), integrated_trajectory],
                dim=2,
            )  # (B, Pnn, 1+T, 4)
            return_["integrated_trajectory"] = integrated_trajectory
            # control_constraint_diff: (B, Pnn, T, 3)
            return_["control_constraint_diff"] = (
                self.dit.dit_returns.control_constraint_diff
            )

        return return_

    def _forward_inference_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        ego_fut_global: torch.Tensor,
        near_agents_route_lane_emb: torch.Tensor,
        route_known_mask: torch.Tensor,
        near_past: torch.Tensor,
        near_current_xyyaw: torch.Tensor,
        near_current_mask: torch.Tensor,
        near_past_cur_future_valid: torch.Tensor,
        near_class_one_hot: torch.Tensor,
        cond_last_pos_norm: torch.Tensor,
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> Dict[str, torch.Tensor]:
        """추론/평가 모드에서 한 배치에 대해 decoder 를 한 번 돌린다.

        - 랜덤 노이즈에서 시작해 DPM-Solver로 샘플링.
        - 중간 단계마다 현재 상태/목표/feasible 보정을 넣는다.
        - 마지막에 상태 역정규화까지 해서 score 에 넣어준다.

        Args:
            inputs: DataLoader 에서 온 배치 dict.
            scene_encoding_token: (B, token_num, D) 장면 토큰.
            scene_encoding_token_mask: (B, token_num) bool 마스크.
            ego_fut_global: (B, D) 이고 미래 요약 벡터.
            near_agents_route_lane_emb: (B, Pnn, D) 이웃+경로 임베딩.
            route_known_mask: (B, Pnn) bool, 경로 유효 여부.
            near_past: (B, Pnn, past_len, 11) 이웃 과거 상태.
            near_current_xyyaw: (B, Pnn, 4) 이웃 현재 상태.
            near_current_mask: (B, Pnn) True면 이웃이 없는 자리.
            near_past_cur_future_valid:
                (B, Pnn, time_len + future_len) 과거+현재+미래 유효 마스크.
            near_class_one_hot: (B, Pnn, 3) 에이전트 종류 one-hot.
            cond_last_pos_norm: (B, Pnn, 4) 정규화된 목표 위치.
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.

        Returns:
            Dict[str, torch.Tensor]: "score"(필수)와
                "integrated_trajectory"(옵션)를 포함한 dict.
        """
        return_: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # 초기 노이즈 궤적 샘플링
        noise: torch.Tensor = near_current_xyyaw.new_empty(
            (B, Pnn, self._future_len, 4),
        ).normal_(0.0, 0.5)  # (B, Pnn, T, 4)

        if self.config.use_current_input:
            # xT: (B, Pnn, (1+T)*4)
            xT: torch.Tensor = torch.cat(
                [
                    near_current_xyyaw[:, :, None, :],  # (B, Pnn, 1, 4)
                    noise,  # (B, Pnn, T, 4)
                ],
                dim=2,
            ).reshape(B, Pnn, -1)
        else:
            # xT: (B, Pnn, T*4)
            xT = noise.reshape(B, Pnn, -1)

        cond_last_pos: Optional[torch.Tensor] = None
        # cond_last_pos_norm 안에 유한한 값이 하나라도 있으면 cond-last를 사용
        if torch.isfinite(cond_last_pos_norm).any():
            cond_last_pos = cond_last_pos_norm  # (B, Pnn, 4)

        if cond_last_pos is not None:
            cond_last_mask: torch.Tensor = compute_cond_last_mask(
                cond_last_pos
            )  # (B, Pnn) True=goal 있음
        else:
            cond_last_mask = torch.zeros(
                B,
                Pnn,
                dtype=torch.bool,
                device=xT.device,
            )  # (B, Pnn)

        def initial_state_constraint(
            xt: torch.Tensor,  # (B, Pnn, F)
            t: torch.Tensor,  # () 또는 (1,)
            step: int,
        ) -> torch.Tensor:
            """DPM-Solver 중간 단계에서 한 번 호출되는 보정 함수.

            - 0번째 프레임을 항상 현재 상태로 맞춰주고
            - cond-last 정보가 있다면 마지막 프레임에 목표를 덮어쓰고
            - 저노이즈 구간에서는 feasible 궤적과 조금 섞은 뒤
            - yaw(cos, sin)을 단위원 위로 다시 정리한다.

            Args:
                xt: (B, Pnn, F) 현재 단계의 샘플.
                t: 현재 시간 값.
                step: 현재 스텝 번호(사용하지 않지만 시그니처 맞춤용).

            Returns:
                torch.Tensor: (B, Pnn, F) 보정된 샘플.
            """
            # xt_sequence: (B, Pnn, time_len, 4)
            xt_sequence: torch.Tensor = self._reshape_xt_with_current_state(
                xt=xt,
                batch_size=B,
                predicted_neighbor_num=Pnn,
                near_current_xyyaw=near_current_xyyaw,  # (B, Pnn, 4)
            )

            # cond-last 정보가 있으면 마지막 프레임에 목표 위치 주입
            # xt_sequence: (B, Pnn, time_len, 4)
            xt_sequence = self._inject_last_position_condition_if_available(
                xt=xt_sequence,
                cond_last_pos=cond_last_pos_norm,  # (B, Pnn, 4)
            )

            # 저노이즈 구간에서만 feasible 궤적과 섞기
            if self.config.use_feasible_blend:
                assert self.config.use_feasible, (
                    "use_feasible_blend 옵션은 "
                    "use_feasible 이 켜져 있을 때만 동작합니다."
                )
                feasible_blend_beta_scalar: torch.Tensor = (
                    self._compute_feasible_blend_beta(
                        diffusion_time_step=t,  # () 또는 (1,)
                        reference_tensor_for_device=xt_sequence,  # (B,Pnn,time_len,4)
                    )
                )  # ()

                if feasible_blend_beta_scalar.detach().item() > 0.0:
                    xt_sequence = (
                        self._apply_feasible_blend_with_feasible_projection(
                            xt_sequence=xt_sequence,  # (B, Pnn, time_len, 4)
                            near_past_cur_future_valid=(
                                near_past_cur_future_valid
                            ),  # (B, Pnn, time_len_total) bool
                            cond_last_mask=cond_last_mask,  # (B, Pnn)
                            feasible_blend_beta_scalar=(
                                feasible_blend_beta_scalar
                            ),  # ()
                        )
                    )

            # yaw (cos, sin)을 단위원 위로 다시 정규화
            xt_sequence = self._project_future_yaw_to_unit_circle(
                xt_sequence=xt_sequence,
            )  # (B, Pnn, time_len, 4)

            # 다시 평탄화해서 반환: (B, Pnn, F)
            return xt_sequence.reshape(B, Pnn, -1)

        # DPM-Solver 샘플링
        x0: torch.Tensor = dpm_sampler(
            self.dit,
            xT.float(),  # (B, Pnn, F)
            other_model_params={
                "cross_c": scene_encoding_token,  # (B, token_num, D)
                "ego_fut_global": ego_fut_global,  # (B, D)
                "near_agents_route_lane_emb":
                    near_agents_route_lane_emb,  # (B, Pnn, D)
                "near_past_cur_future_valid":
                    near_past_cur_future_valid,  # (B, Pnn, time_len+future_len)
                "cross_mask": scene_encoding_token_mask,  # (B, token_num)
                "route_known_mask": route_known_mask,  # (B, Pnn)
                "near_class_one_hot": near_class_one_hot,  # (B, Pnn, 3)
                "near_current_xyyaw": near_current_xyyaw,  # (B, Pnn, 4)
                "near_past": near_past,  # (B, Pnn, past_len, 11)
            },
            dpm_solver_params={
                "correcting_xt_fn": initial_state_constraint,
            },
            model_wrapper_params={
                "classifier_fn": self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit,
                    "model_condition": {
                        "cross_c": scene_encoding_token,
                        "ego_fut_global": ego_fut_global,
                        "near_agents_route_lane_emb":
                            near_agents_route_lane_emb,
                        "near_past_cur_future_valid":
                            near_past_cur_future_valid,
                        "cross_mask": scene_encoding_token_mask,
                        "route_known_mask": route_known_mask,
                        "near_class_one_hot": near_class_one_hot,
                        "near_current_xyyaw": near_current_xyyaw,
                        "near_past": near_past,
                    },
                    "inputs": inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                    "config": self.config,
                },
                "guidance_scale": 1.0,
                "guidance_type": (
                    "classifier"
                    if self._guidance_fn is not None
                    else "uncond"
                ),
            },
        )

        x0 = x0.to(xT.dtype)

        if self.config.use_current_input:
            # x0: (B, Pnn, (1+T)*4)
            assert x0.shape == (
                B,
                Pnn,
                (1 + self._future_len) * 4,
            )
            x0 = x0.reshape(B, Pnn, -1, 4)  # (B, Pnn, 1+T, 4)
        else:
            # x0: (B, Pnn, T*4)
            assert x0.shape == (B, Pnn, self._future_len * 4)
            x0 = torch.cat(
                [
                    near_current_xyyaw.unsqueeze(2),  # (B, Pnn, 1, 4)
                    x0.reshape(B, Pnn, -1, 4),  # (B, Pnn, T, 4)
                ],
                dim=2,
            )  # (B, Pnn, 1+T, 4)

        # 비정규화 궤적
        unnorm_x0: torch.Tensor = self._state_normalizer.inverse(
            x0
        )  # (B, Pnn, 1+T, 4)
        unnorm_x0[near_current_mask] = 0.0  # 패딩 슬롯은 0으로 정리

        if self.config.use_feasible:
            integrated_trajectory: torch.Tensor = (
                self.dit.dit_returns.integrated_trajectory
            )  # (B, Pnn, T, 4)
            integrated_trajectory = torch.cat(
                [
                    near_current_xyyaw.unsqueeze(2),  # (B, Pnn, 1, 4)
                    integrated_trajectory,  # (B, Pnn, T, 4)
                ],
                dim=2,
            )  # (B, Pnn, 1+T, 4)

            unnorm_integrated_trajectory: torch.Tensor = (
                self._state_normalizer.inverse(integrated_trajectory)
            )  # (B, Pnn, 1+T, 4)
            unnorm_integrated_trajectory[near_current_mask] = 0.0

            return_["integrated_trajectory"] = unnorm_integrated_trajectory

        return_["score"] = unnorm_x0  # (B, Pnn, 1+T, 4)
        return return_

    def forward(
        self,
        encoder_outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Decoder 전체 한 스텝을 수행하는 진입점.

        - 공통으로 이웃 에이전트 관련 텐서와 장면 인코딩을 준비하고
        - self.training 여부에 따라 훈련/추론 경로로 나눠 처리한다.

        Args:
            encoder_outputs: Encoder 에서 나온 출력 dict.
                - "encoding":                  (B, token_num, hidden_dim)
                - "encoding_mask":             (B, token_num)
                - "ego_fut_global":            (B, hidden_dim)
                - "near_agents_route_lane_emb":(B, Pnn, hidden_dim)
                - "route_known_mask":          (B, Pnn)
            inputs: DataLoader 에서 온 배치 dict.
                ego_agent_past : (B, time_len, 11) #
                ego_future_gt_3_dim : (B, future_len, 3)
                neighbor_agents_past : (B, agent_num, time_len, 11) #
                lanes : (B, lane_num, lane_len, 12) #
                lanes_speed_limit : (B, lane_num, 1) #
                lanes_has_speed_limit : (B, lane_num, 1) #
                route_lanes : (B, route_num, route_len, 12)
                route_lanes_speed_limit : (B, route_num, 1)
                route_lanes_has_speed_limit : (B, route_num, 1)
                static_objects : (B, static_num, 10) #
                near_future_gt_3_dim: (B, Pnn, future_len, 3)
                planner_future_11_dim: (B, future_len, 11) #
                agent_route_lane_order: (B, agent_num, lane_num)

                near_future_valid: (B, Pnn, future_len) 미래 유효 마스크.
                near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4) 현재+미래 x_T.
                batch_diffusion_time: (B,) diffusion 시간.
                cond_last_pos_norm: (B, Pnn, 4) cond 용 마지막 위치.

        Returns:
            Dict[str, torch.Tensor]: 최소한 "score" 키를 가지며,
            설정에 따라 "integrated_trajectory",
            "control_constraint_diff" 도 포함될 수 있다.
        """
        (
            near_past,  # (B, Pnn, past_len, 11)
            near_current_xyyaw,  # (B, Pnn, 4)
            near_current_mask,  # (B, Pnn)
            near_past_cur_future_valid,  # (B, Pnn, time_len+future_len)
            near_class_one_hot,  # (B, Pnn, 3)
            cond_last_pos_norm,  # (B, Pnn, 4)
            batch_size,
            predicted_neighbor_num,
        ) = self._prepare_near_trajectories_and_masks(inputs)

        (
            scene_encoding_token,  # (B, token_num, D)
            scene_encoding_token_mask,  # (B, token_num)
            ego_fut_global,  # (B, D)
            near_agents_route_lane_emb,  # (B, Pnn, D)
            route_known_mask,  # (B, Pnn)
        ) = self._unpack_encoder_outputs(
            encoder_outputs=encoder_outputs,
            batch_size=batch_size,
        )

        if self.training:
            return self._forward_training_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                ego_fut_global=ego_fut_global,
                near_agents_route_lane_emb=near_agents_route_lane_emb,
                route_known_mask=route_known_mask,
                near_past=near_past,
                near_current_xyyaw=near_current_xyyaw,
                near_current_mask=near_current_mask,
                near_past_cur_future_valid=near_past_cur_future_valid,
                near_class_one_hot=near_class_one_hot,
                cond_last_pos_norm=cond_last_pos_norm,
                batch_size=batch_size,
                predicted_neighbor_num=predicted_neighbor_num,
            )
        else:
            return self._forward_inference_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                ego_fut_global=ego_fut_global,
                near_agents_route_lane_emb=near_agents_route_lane_emb,
                route_known_mask=route_known_mask,
                near_past=near_past,
                near_current_xyyaw=near_current_xyyaw,
                near_current_mask=near_current_mask,
                near_past_cur_future_valid=near_past_cur_future_valid,
                near_class_one_hot=near_class_one_hot,
                cond_last_pos_norm=cond_last_pos_norm,
                batch_size=batch_size,
                predicted_neighbor_num=predicted_neighbor_num,
            )

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DiTReturns:
    integrated_trajectory: torch.Tensor  # (B, Pnn, T, 4)
    control_constraint_diff: torch.Tensor  # (B, Pnn, T, 3)


class DiT(nn.Module):

    def __init__(self,
                 config,
                 sde: SDE,
                 depth,
                 output_dim,
                 hidden_dim=192,
                 heads=6,
                 dropout=0.1,
                 mlp_ratio=4.0,
                 model_type="x_start"):
        super().__init__()
        self.dit_returns = None
        self.config = config
        self._future_len: int = config.future_len  # <추가하자>

        assert model_type in ["score",
                              "x_start"], f"Unknown model type: {model_type}"
        self.final_hidden_tokens = None
        if self.config.use_feasible:
            self.feasible_projector = FeasibleProjector(
                self.config, hidden_dim, self.config.use_feasible_dl,
                self.config.use_feasible_filter)
            self.feasible_projector.enable_profile = self.config.profile_feasible

        self._model_type = model_type
        self.preproj = Mlp(in_features=output_dim,
                           hidden_features=512,
                           out_features=hidden_dim,
                           act_layer=nn.GELU,
                           drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout, mlp_ratio)
            for i in range(depth)
        ])
        ##################
        # ----- PRAM‑v2 구성요소 추가 -----
        # S/E/R를 저차원으로 정리(RMSNorm 포함) → 쌍곱(SE/ER/RS) → 혼합 MLP → base 모듈레이션(Δs,b,logit g)을 산출.
        self.pram_v2_composer = PRAMV2Composer(
            hidden_dim=hidden_dim,
            adapter_hidden_dim=128,
            composed_hidden_dim=128,
            activation="gelu",
            gate_init_bias=-3.0,
        )
        # 확산 시간(t) 임베딩으로부터 전역(time) 모듈레이션을 생성.
        self.pram_v2_time_mod = PRAMV2TimeModulator(hidden_dim=hidden_dim)

        self.pram_v2_state_token_encoder = PRAMV2StateTokenEncoder(
            hidden_dim=hidden_dim)
        # 블록×경로 토글 스칼라 및 게이트 편향.
        self.pram_v2_block_path_scalars = PRAMV2BlockPathScalars(depth=depth)

        # 9단계: v2 전용 최종 LN/Linear(출력 투영)
        self.pram_v2_final_norm = nn.LayerNorm(hidden_dim)
        self.pram_v2_out_proj = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_dim, hidden_dim * 4, bias=True),
            nn.GELU(approximate="tanh"),
            # nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_dim * 4, output_dim, bias=True))
        nn.init.zeros_(
            self.pram_v2_out_proj[-1].weight)  # pram_v2_out_proj의 마지막 Linear
        nn.init.zeros_(self.pram_v2_out_proj[-1].bias)

        # [추가] TimestepEmbedder MLP 초기화 (여기서 1회만)
        #  - 구조: Sequential[ Linear(256→H), SiLU, Linear(H→H) ]
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None:
            nn.init.constant_(self.t_embedder.mlp[0].bias, 0.0)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.t_embedder.mlp[2].bias is not None:
            nn.init.constant_(self.t_embedder.mlp[2].bias, 0.0)

        # [선택] 가독성 차원에서 명시(기본값과 동일)
        nn.init.constant_(self.pram_v2_final_norm.weight, 1.0)
        nn.init.constant_(self.pram_v2_final_norm.bias, 0.0)

        # 9단계 마무리 보정용 스칼라 (k^{final}_s, k^{final}_{sh})
        # 실제 보정 함수는 스켈레톤 상태여도 호출부만 준비해 둡니다.
        self.pram_v2_final_scale_scalar = nn.Parameter(torch.tensor(1.0))
        self.pram_v2_final_shift_scalar = nn.Parameter(torch.tensor(1.0))
        #################
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    def preproj_varlen(
            self,
            near_cur_future_norm_xT: torch.Tensor,  # (B, Pnn, F=(1+T)*4)
            near_current_mask: torch.Tensor,  # (B, Pnn) True=pad(무효 에이전트)
    ) -> torch.Tensor:
        """pre‑proj(Mlp)를 **유효 에이전트 토큰**에만 적용하는 varlen 전처리.

        기존 pre‑proj는 (B,Pnn,F) 전체에 적용되어 K≪Pnn일 때 불필요한 연산이 발생합니다.
        이 함수는 마스크 기반 **unpad→pre‑proj→pad‑back**으로 pre‑proj도 K에 비례로 줄입니다.

        Args:
            near_cur_future_norm_xT: 정규화된 (현재+미래) 입력.
                **shape:** (B, Pnn, F)  (F=(1+T)*4, 예: 81*4=324)
            near_current_mask: 키 패딩 마스크(True=pad=무효 에이전트).
                **shape:** (B, Pnn)

        Returns:
            x: pre‑proj 결과.
                **shape:** (B, Pnn, D)  (D=hidden_dim, 예: 192)
                무효 위치는 0으로 채워져 있습니다.
        """
        B, Pnn, F = near_cur_future_norm_xT.shape
        # 유효 토큰만 추출
        # unpad_input은 'True=유효' 마스크를 기대 → 반전 필요
        attention_mask = (~near_current_mask).to(
            torch.bool)  # (B, Pnn), True=유효

        res = unpad_input(near_cur_future_norm_xT, attention_mask)
        # x_unpad: (T, F), indices: (T,), cu: (B+1,), max_seqlen: int
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        # 모든 토큰이 pad인 극단 케이스 방어
        if x_unpad.numel() == 0:
            D_out = self.preproj.fc2.out_features  # timm Mlp의 최종 out_features
            zeros = near_cur_future_norm_xT.new_zeros((B, Pnn, D_out))
            # timm Mlp(fc1/fc2)의 파라미터를 0-스케일로 터치(그래프 포함)
            touch = (self.preproj.fc1.weight.view(-1)[:1].sum() +
                     (self.preproj.fc1.bias.view(-1)[:1].sum()
                      if self.preproj.fc1.bias is not None else 0) +
                     self.preproj.fc2.weight.view(-1)[:1].sum() +
                     (self.preproj.fc2.bias.view(-1)[:1].sum()
                      if self.preproj.fc2.bias is not None else 0)) * 0.0
            return zeros + touch

        # 유효 토큰만 pre‑proj 수행  (T, F) -> (T, D)
        x_unpad = self.preproj(x_unpad)  # (T, D)

        # 배치 모양으로 복원 (pad 위치는 0)
        x = pad_input(x_unpad, indices, B, Pnn)  # (B, Pnn, D)
        return x

    @property
    def model_type(self):
        return self._model_type

    def forward(
            self,
            near_future_norm_xT: torch.Tensor,  # (B, Pnn, T*4)
            diffusion_time: torch.Tensor,  # (B,)
            cross_c: torch.Tensor,  # (B, token_num, D)
            ego_fut_global: torch.Tensor,  # (B, D)
            near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
            near_past_cur_future_valid: torch.Tensor,
            # [B, pnn, time_len(=1+past_len) + future_len] bool # 과거-현재-미래
            cross_mask: torch.Tensor,  # (B, token_num) True=pad
            route_known_mask: torch.Tensor,  # (B, Pnn) True=known
            near_class_one_hot: torch.Tensor,
            # (B, Pnn, 3) # 0: 차량, 1: 보행자, 2: 자전거
            near_current_xyyaw: torch.Tensor,  # ★ 추가: (B, Pnn, 4)
            near_past: torch.Tensor,  # (B, Pnn, past_len, 11),
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        near_future_norm_xT:  [B, Pnn, T * 4] # (80*4 = 324)
        diffusion_time:  [B,]                 -> Diffusion time uniformly sampled in [eps, 1]
        cross_c: [B, N = token_num, D = 192]
        ego_fut_global: [B, D]   -> Global encoding of the future trajectory of the ego agent.
        near_past_cur_future_valid: torch.Tensor, # [B, pnn, time_len(=1+past_len) + future_len] bool # 과거-현재-미래
        near_agents_route_lane_emb, # (B, Pnn, D)
        cross_mask: (B, token_num)
        """
        near_cur_future_valid = near_past_cur_future_valid[:, :, -(
            self._future_len + 1):]  # [B, Pnn, 1 + future_len] bool
        near_current_valid = near_cur_future_valid[:, :,
                                                   0]  # [B, Pnn] True=유효 에이전트
        near_current_mask = ~near_current_valid  # [B, Pnn] True=무효 에이전트
        B, Pnn, _ = near_future_norm_xT.shape
        # (B, Pnn, 324) -> (B, Pnn, D=192)
        # x = self.preproj(near_future_norm_xT)
        device_type = near_future_norm_xT.device.type
        with profile_block(
                "DiT.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            x = self.preproj_varlen(near_future_norm_xT, near_current_mask)

            x = x.masked_fill(near_current_mask.unsqueeze(-1),
                              0.0)  # ← 무효 토큰 0 클램프

            # diffusion_time: [B,]
            # t_embedding: (B, D=192)
            t_embedding = self.t_embedder(diffusion_time)
            t_embedding = t_embedding.to(x.dtype)
            ego_fut_global = ego_fut_global.to(x.dtype)  # 방어적 정렬
            ############################
            # ★ 추가: state_token_in 생성 (현재 프레임만 사용)
            B, Pnn, _ = near_future_norm_xT.shape
            # state_token_in: [B,Pnn,D]
            state_token_in = self.pram_v2_state_token_encoder(
                # PRAMV2StateTokenEncoder
                near_cur_norm=near_current_xyyaw.to(x.dtype),  # [B,Pnn,4]
                near_current_mask=near_current_mask  # [B,Pnn]
            )
            # [V2 - START]  ✅ Composer/Time 1회 계산 → 블록별 합성 → v2 전용 포워드
            """
            1) “입력 요약” 만들기 (한 번만 계산 → 모든 블록 재사용)
            2) 쌍곱(상호작용) 특징 만들기 — (SE, ER, RS)
            3) 쌍곱 포함해 한 덩어리로 묶고 z 만들기 — LN → 작은 MLP
            5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
            """
            # composer_out: ComposerOutputs
            #   Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
            composer_out = self.pram_v2_composer(
                state_token_in=state_token_in,  # [B, Pnn, D]
                ego_fut_global=ego_fut_global,  # [B, D] # (배치 단위 0벡터 처리)
                near_agents_route_lane_emb=near_agents_route_lane_emb,
                # [B, Pnn, D]
                route_known_mask=route_known_mask  # [B, Pnn] True=known
            )
            """
            7) 확산 시간 t의 글로벌 모듈레이션
    
            time_out : TimeModulationOutputs 
                delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
            """
            time_out = self.pram_v2_time_mod(
                t_embedding)  # [B,1,H] 3종 # PRAMV2TimeModulator

            for block_index, block in enumerate(self.blocks):
                """ pram_mods
                Dict[PathName, ModulationTriplet]: 
                    {"SA": ModulationTriplet, "FFN":ModulationTriplet, "CA":ModulationTriplet}
                """
                pram_mods = compute_pram_v2_modulations_for_block(
                    # Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
                    composer_out=composer_out,
                    # delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
                    time_out=time_out,
                    path_scalars=self.
                    pram_v2_block_path_scalars,  # PRAMV2BlockPathScalars
                    block_index=block_index,
                    batch_size=B,
                    predicted_neighbor_num=Pnn,
                    hidden_dim=x.shape[-1],
                )  # -> {"SA": ModulationTriplet, "FFN": ..., "CA": ...}

                # ★ DiTBlock에 추가한 v2 전용 진입점(스켈레톤; 구현은 이후 단계)
                x = block(
                    x=x,  # [B, Pnn, H]
                    cross_c=cross_c,  # [B, N_c, H]
                    pram_v2_modulations=
                    pram_mods,  # Dict[PathName, ModulationTriplet]
                    near_current_mask=near_current_mask,  # [B, Pnn]
                    cross_mask=cross_mask  # [B, N_c]
                )
                x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)
            if self.config.feasible_grad_to_dit:
                self.final_hidden_tokens = x.float()
            else:
                self.final_hidden_tokens = x.detach().clone().float(
                )  # (B, Pnn, H)
            # [V2 - END]
            # --- ✅ PRAM‑v2: 9단계 최종 보정 + 최종 투영(= FinalLayer 완전 대체) ---
            x = apply_pram_v2_final_layer(
                x=x,  # [B, Pnn, H]
                # Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
                composer_out=composer_out,
                # delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
                time_out=time_out,
                final_norm=self.pram_v2_final_norm,  # LN(H)
                out_proj=self.pram_v2_out_proj,  # Linear(H -> (T)*4)
                final_scalars=(
                    self.pram_v2_final_scale_scalar,  # k_final_s
                    self.pram_v2_final_shift_scalar,  # k_final_sh
                ),
            )  # -> [B, Pnn, (T)*4]

            # 마스크(무효 토큰) 0 클램프 유지
            x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)
        if self._model_type == "score":
            std = self.marginal_prob_std(diffusion_time).float()[:, None,
                                                                 None]  # FP32
            out = (x.float() / (std + 1e-6)).to(
                x.dtype)  # 계산은 FP32, 최종만 원래 dtype
            return out
        elif self._model_type == "x_start":
            # CURRENT DEFAULT OPTION: "x_start"
            # x: (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
            if self.config.use_feasible:
                if self.config.feasible_grad_to_dit:
                    x_for_feasible = x.float()  # (B,Pnn, T*4 or (1+T)*4)
                    near_current_xyyaw_for_feasible = near_current_xyyaw.float(
                    )  # (B,Pnn,4)
                    near_past_cur_future_valid_for_feasible = near_past_cur_future_valid
                else:
                    x_for_feasible = x.detach().float(
                    )  # (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
                    near_current_xyyaw_for_feasible = near_current_xyyaw.detach(
                    ).float()  # (B, Pnn, 4)
                    near_past_cur_future_valid_for_feasible = near_past_cur_future_valid.detach(
                    )
                # DiT.forward (model_type == "x_start" 분기 내부)
                if getattr(self.config, "use_current_input", True):
                    # x: (B, Pnn, (1+T)*4) → 이미 현재 프레임 포함
                    diffusion_trajectory = x_for_feasible.reshape(
                        B, Pnn, -1, 4).contiguous()  # (B,Pnn,1+T,4)
                    diffusion_trajectory[:, :,
                                         0, :] = near_current_xyyaw_for_feasible
                else:
                    # x: (B, Pnn, T*4) → 현재 프레임을 앞에 붙여서 1+T로 맞춤
                    diffusion_trajectory = torch.cat(
                        [
                            near_current_xyyaw_for_feasible.unsqueeze(2),
                            x_for_feasible.reshape(B, Pnn, -1, 4)
                        ],
                        dim=2).contiguous()  # (B,Pnn,1+T,4)
                t_threshold = self.config.feasible_learn_noise_thresh  # 0.30
                if not self.config.use_direct_loss:
                    t_threshold = 1.
                low_t_mask = (diffusion_time <= t_threshold)  # [B]  True=저노이즈

                # (B, Pnn, T, 4) -> (B, Pnn, T*4)

                self._feasible_projection(
                    diffusion_trajectory,
                    near_class_one_hot,
                    near_past_cur_future_valid_for_feasible,
                    near_past,  # [B, pnn, past_len=(time_len - 1), 11]
                    low_t_mask,  # [B]  True=저노이즈
                )
                if not self.training and not self.config.use_direct_loss:
                    integrated_trajectory = self.dit_returns.integrated_trajectory  # (B, Pnn, T, 4)
                    if self.config.use_current_input:
                        x = torch.cat(
                            [
                                near_current_xyyaw_for_feasible.unsqueeze(
                                    2),  # (
                                integrated_trajectory
                            ],
                            dim=2)  # (B, Pnn, 1+T, 4)
                    else:
                        x = integrated_trajectory  # (B,Pnn,T,4)
                    # x: (B, Pnn, T, 4) or (B, Pnn, (1+T), 4) -> (B, Pnn, T*4) or (B, Pnn, (1+T)*4)
                    x = x.reshape(B, Pnn, -1)
                    return x
            return x  # (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    # : stride 기반 down/up 샘플링을 통합한 새 파이프라인
    def _feasible_projection_core(
            self,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
            near_past_cur_future_valid: torch.Tensor,
            # (B, Pnn, time_len=1+past_len+future_len) bool
            near_past: Optional[torch.Tensor],  # (B, Pnn, past_len, 11) 또는 None
            final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
    ) -> None:
        """FeasibleProjector 전체 파이프라인을 한 번에 실행합니다.

        한 에이전트 기준 단계 요약:
            1) (과거 + 현재 + 미래) 궤적을 stride_step 간격으로 단순 stride 샘플링.
               - 현재 시점, 마지막 미래 시점은 항상 포함.
            2) 다운샘플 궤적에 Savitzky–Golay 필터를 적용해
               세계 기준 속도/요각속도 시퀀스를 얻는다.
            3) 점 제어 → 구간(중점) 제어로 바꾸고,
               FeasibleProjector 네트워크(TCN)로 구간 제어를 보정.
            4) 보정된 “서브샘플 구간 제어”를 선형 보간으로
               원래 future_len 개수의 구간으로 업샘플링.
            5) 업샘플된 현재~미래 구간 제어를 가지고
               filter_and_integrate 로 최종 적분 궤적과 제약 위반량을 계산.
        """
        device_type: str = diffusion_trajectory.device.type
        use_profile: bool = bool(getattr(self.config, "profile_feasible",
                                         False))

        # Feasible 파트는 모두 FP32로 고정
        with torch.autocast(device_type=device_type, enabled=False):
            diffusion_trajectory = diffusion_trajectory.float(
            )  # (B, Pnn, 1+T, 4)
            near_class_one_hot = near_class_one_hot.float()  # (B, Pnn, 3)
            near_past_cur_future_valid = near_past_cur_future_valid.to(
                torch.bool)  # (B, Pnn, time_len)
            if near_past is not None:
                near_past = near_past.float()  # (B, Pnn, past_len, 11)

            B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
            future_len: int = int(one_plus_T - 1)  # T

            # stride 및 SG 윈도 관련 하이퍼 계산
            (
                stride_step,  # int
                dt_for_savgol,  # float
                max_window_len_xy,  # int
                max_window_len_yaw,  # int
            ) = self.feasible_projector.get_feasible_stride_params(future_len)
            # 역정규화 현재+미래 궤적 및 현재 상태
            unnorm_diffusion_trajectory = self.config.state_normalizer.inverse(
                diffusion_trajectory)  # (B, Pnn, 1+T, 4)
            unnorm_near_current_state = unnorm_diffusion_trajectory[:, :,
                                                                    0, :]  # (B, Pnn, 4)

            # 과거 xy-yaw (정규화/역정규화) 준비
            if near_past is not None and near_past.numel() > 0:
                near_past_xyyaw = near_past[..., :4]  # (B, Pnn, past_len, 4)
                unnorm_near_past_xyyaw = self.config.state_normalizer.inverse(
                    near_past_xyyaw)  # (B, Pnn, past_len, 4)
            else:
                near_past_xyyaw = None
                unnorm_near_past_xyyaw = None

            # --- (1) stride 기준 다운샘플 궤적/마스크 생성 ---
            (
                unnorm_diffusion_trajectory_stride,  # (B, Pnn, 1+T_ds, 4)
                unnorm_near_past_xyyaw_stride,  # (B, Pnn, past_len_ds, 4) or None
                near_past_cur_future_valid_stride,  # (B, Pnn, past_len_ds+1+T_ds) bool
                diffusion_trajectory_stride_norm,  # (B, Pnn, 1+T_ds, 4)
                near_past_xyyaw_stride_norm,  # (B, Pnn, past_len_ds, 4) or None
                past_len_ds,  # int
                future_len_ds,  # int (T_ds)
            ) = self.feasible_projector.build_downsampled_feasible_inputs(
                diffusion_trajectory=diffusion_trajectory,
                near_past=near_past,
                near_past_cur_future_valid=near_past_cur_future_valid,
                unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
                unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
                stride_step=stride_step,
            )

            # --- (2) Savitzky–Golay + 점 제어 계산 ---
            with profile_block(
                    "feasible.savgol_filter_for_control",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                # (B, Pnn, point_len_ds, 3)  [v_x^w, v_y^w, ω]
                unnorm_points_world_control_stride = \
                    self.feasible_projector.savgol_filter_for_control(
                        unnorm_diffusion_trajectory_stride,    # (B, Pnn, 1+T_ds, 4)
                        unnorm_near_past_xyyaw_stride,         # (B, Pnn, past_len_ds, 4) or None
                        near_past_cur_future_valid_stride,     # (B, Pnn, past_len_ds+1+T_ds) bool
                        dt=dt_for_savgol,
                        polyorder=2,
                        max_window_len_xy=max_window_len_xy,
                        max_window_len_yaw=max_window_len_yaw,
                    )

            with profile_block(
                    "feasible.compute_midpoint_controls",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                # (B, Pnn, segment_len_ds, 3)  [v_x^b, v_y^b, ω]_mid (stride 타임라인 기준)
                unnorm_seg_body_control_stride = \
                    self.feasible_projector.compute_midpoint_controls(
                        unnorm_diffusion_trajectory_stride,    # (B, Pnn, 1+T_ds, 4)
                        unnorm_near_past_xyyaw_stride,         # (B, Pnn, past_len_ds, 4) or None
                        unnorm_points_world_control_stride,    # (B, Pnn, point_len_ds, 3)
                        near_past_cur_future_valid_stride,     # (B, Pnn, past_len_ds+1+T_ds) bool
                    )

            # --- (3) 관측 정규화 → FeasibleProjector 네트워크(TCN) 보정 ---
            temp_dict = {"seg_body_control": unnorm_seg_body_control_stride}
            temp_dict = self.config.observation_normalizer(temp_dict)
            seg_body_control_stride = temp_dict[
                "seg_body_control"]  # (B, Pnn, segment_len_ds, 3)

            with torch.autocast(
                    device_type=device_type,
                    dtype=AMP_DTYPE,
                    enabled=(device_type == "cuda"),
            ):
                with profile_block(
                        "feasible.network_forward",
                        enabled=use_profile,
                        device_type=device_type,
                ):
                    # (B, Pnn, segment_len_ds, 3)  정규화 공간 제어
                    seg_body_control_stride_ref = self.feasible_projector(
                        near_past_cur_future_valid_stride,  # (B, Pnn, past_len_ds+1+T_ds)
                        diffusion_trajectory_stride_norm,  # (B, Pnn, 1+T_ds, 4)
                        near_past_xyyaw_stride_norm,  # (B, Pnn, past_len_ds, 4) or None
                        seg_body_control_stride,  # (B, Pnn, segment_len_ds, 3)
                        final_hidden_tokens,  # (B, Pnn, H)
                    )

            seg_body_control_stride_ref = seg_body_control_stride_ref.float()
            temp_dict = {"seg_body_control": seg_body_control_stride_ref}
            temp_dict = self.config.observation_normalizer.inverse(temp_dict)
            unnorm_seg_body_control_stride_ref = temp_dict[
                "seg_body_control"]  # (B, Pnn, segment_len_ds, 3)

            # --- (4) 미래 구간 제어만 원래 future_len 개수로 업샘플링 ---
            # (B, Pnn, future_len, 3)
            unnorm_fut_seg_body_control = self.feasible_projector.upsample_future_controls_from_stride(
                unnorm_seg_body_control_stride=
                unnorm_seg_body_control_stride_ref,  # (B, Pnn, segment_len_ds, 3)
                past_len_ds=past_len_ds,  # int
                future_len_ds=future_len_ds,  # int
                future_len_full=future_len,  # int
                stride_step=stride_step,  # int
            )
            # --- (5) 제약 기반 필터 + 적분 ---
            near_cur_future_valid = near_past_cur_future_valid[:, :, -(
                1 + future_len):]  # (B, Pnn, 1+future_len) bool

            with profile_block(
                    "feasible.filter_and_integrate",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                (
                    unnorm_integrated_trajectory,  # (B, Pnn, future_len, 4)
                    unnorm_control_constraint_diff,  # (B, Pnn, future_len, 3)
                ) = self.feasible_projector.filter_and_integrate(
                    unnorm_near_current_state,  # (B, Pnn, 4)
                    near_cur_future_valid,  # (B, Pnn, 1+future_len) bool
                    unnorm_fut_seg_body_control,  # (B, Pnn, future_len, 3)
                    near_class_one_hot,  # (B, Pnn, 3)
                )

            # 정규화해서 DiTReturns 로 저장
            integrated_trajectory = self.config.state_normalizer(
                unnorm_integrated_trajectory)  # (B, Pnn, future_len, 4)
            near_future_mask = near_cur_future_valid[:, :,
                                                     1:]  # (B, Pnn, future_len) bool
            integrated_trajectory = integrated_trajectory.masked_fill(
                ~near_future_mask.unsqueeze(-1), 0.0)

            temp_dict = {"seg_body_control": unnorm_control_constraint_diff}
            temp_dict = self.config.observation_normalizer(temp_dict)
            control_constraint_diff = temp_dict[
                "seg_body_control"]  # (B, Pnn, future_len, 3)
            control_constraint_diff = control_constraint_diff.masked_fill(
                ~near_future_mask.unsqueeze(-1), 0.0)

            self.dit_returns = DiTReturns(
                integrated_trajectory=integrated_trajectory,
                control_constraint_diff=control_constraint_diff,
            )

    # [추가!!!] 학습 시 low‑t 샘플에 대해서만 FeasibleProjector를 돌리는 래퍼
    def _feasible_projection(
        self,
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
        near_past_cur_future_valid: torch.Tensor,
        # (B, Pnn, time_len(=1+past_len) + future_len) bool
        near_past: torch.Tensor,  # (B, Pnn, past_len, 11) 또는 None
        feasible_low_t_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """FeasibleProjector 호출을 t‑기반으로 부분적으로만 수행하는 래퍼.

        - 학습이 아닐 때(`self.training == False`)나
          `feasible_low_t_mask` 가 없을 때는
          기존 전체 파이프라인(_feasible_projection_full)을 그대로 사용.
        - 학습 + `feasible_low_t_mask` 가 주어졌을 때는
          diffusion_time <= threshold 인 샘플(batch index) 에 대해서만
          FeasibleProjector 전체 파이프라인을 실행하고,
          나머지 샘플은
            * integrated_trajectory  : 원 궤적(diffusion_trajectory) 그대로
            * control_constraint_diff: 0
          로 채운 뒤 `self.dit_returns` 에 통합해서 저장합니다.

        이렇게 하면
        - high‑t 샘플에 대해 불필요한 Savitzky–Golay + TCN forward 를 줄이면서,
        - 적어도 한 샘플에 대해서는 항상 FeasibleProjector가 그래프에 등장하므로
          DDP(find_unused_parameters=False) 환경에서도
          "unused parameter" 에러 가능성을 줄일 수 있습니다.
        """
        # 학습이 아니거나, low‑t 마스크가 없으면 기존 전체 경로 유지
        if feasible_low_t_mask is None:
            self._feasible_projection_core(
                diffusion_trajectory,
                near_class_one_hot,
                near_past_cur_future_valid,
                near_past,
                self.final_hidden_tokens,
            )
            return

        B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
        T = one_plus_T - 1
        device = diffusion_trajectory.device
        dtype = diffusion_trajectory.dtype

        # [B] -> bool 로 정리
        low_t_mask = feasible_low_t_mask.to(device=device)
        if low_t_mask.dtype != torch.bool:
            # 0/1 float 같은 케이스 방어용
            low_t_mask = low_t_mask > 0.5

        # 기본값:
        #   - integrated_trajectory: 원 궤적(x_start) 그대로 (t=1..T)
        #   - control_constraint_diff: 전부 0
        # diffusion_trajectory 는 정규화 상태라고 가정
        base_integrated = diffusion_trajectory[:, :,
                                               1:, :].detach()  # (B,Pnn,T,4)
        base_constraint = torch.zeros((B, Pnn, T, 3),
                                      device=device,
                                      dtype=dtype)  # (B,Pnn,T,3)

        integrated_all = base_integrated.clone()  # (B,Pnn,T,4)
        constraint_all = base_constraint.clone()  # (B,Pnn,T,3)

        # 실제로 FeasibleProjector를 돌릴 배치 인덱스 선택
        active_idx = torch.nonzero(low_t_mask,
                                   as_tuple=False).squeeze(-1)  # (N_active,)

        # 만약 이번 배치가 전부 high‑t 라면, 그래도 최소 1개 샘플(0번)은
        # FeasibleProjector를 한 번 태워서 그래프에는 항상 등장하도록 한다.
        if active_idx.numel() == 0:
            active_idx = torch.tensor([0], device=device,
                                      dtype=torch.long)  # (1,)

        # 서브 배치만 골라서 전체 파이프라인 실행
        self._feasible_projection_core(
            diffusion_trajectory[active_idx],
            near_class_one_hot[active_idx],
            near_past_cur_future_valid[active_idx],
            near_past[active_idx] if near_past is not None else None,
            self.final_hidden_tokens[active_idx],
        )

        # `_feasible_projection_core` 은 서브 배치 기준으로 self.dit_returns 를 채운다.
        integ_active = self.dit_returns.integrated_trajectory  # (N_active, Pnn, T, 4)
        const_active = self.dit_returns.control_constraint_diff  # (N_active, Pnn, T, 3)

        # 선택된 active 샘플에 대해서만 결과를 덮어쓰기
        integrated_all[active_idx] = integ_active
        constraint_all[active_idx] = const_active

        # 전체 배치 기준으로 다시 dit_returns 업데이트
        self.dit_returns = DiTReturns(
            integrated_trajectory=integrated_all,  # (B,Pnn,T,4)
            control_constraint_diff=constraint_all,  # (B,Pnn,T,3)
        )
