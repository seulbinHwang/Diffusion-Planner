from functools import partial
from typing import Callable
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
        self._future_len: int = config.future_len
        self._sde = VPSDE_linear()
        self._cond_last_prob: float = getattr(config, "cond_last_prob",
                                              0.0)  # 20%
        if self.config.use_past_dit_input:
            output_dim = (config.time_len +
                          config.future_len) * 4  # x, y, cos, sin
        else:
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

    def _build_training_dit_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        near_current_mask: torch.Tensor,  # (B, Pnn)
        cond_last_pos_norm: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """훈련 모드에서 DiT에 넣을 입력(xT_input)과 현재 프레임(정규화)을 만든다.

        이 함수는 훈련에서 매번 반복되는 전처리를 한 곳에 모아,
        `_forward_training_mode`를 짧고 읽기 쉽게 만드는 목적입니다.

        처리 흐름:
            1) inputs["near_cur_future_norm_xT"]에서 현재/미래를 분리합니다.
            2) config.use_current_input 설정에 따라
               - 현재+미래를 쓰거나
               - 미래만 쓰도록 선택합니다.
            3) (B, Pnn, time, 4) -> (B, Pnn, F) 로 펼칩니다.
            4) (옵션) 목표점(cond_last_pos_norm)을 마지막 프레임에 주입합니다.
               - 배치 단위 확률 토글 방식(기존 로직 유지)

        Args:
            inputs: DataLoader에서 넘어온 배치 dict.
                - "near_cur_future_norm_xT": (B, Pnn, 1+T, 4)
            near_current_mask: 현재 프레임 기준 무효 에이전트 마스크.
                shape: (B, Pnn)  True=무효(패딩)
            cond_last_pos_norm: 목표점(마지막 위치) 조건(정규화).
                shape: (B, Pnn, 4)
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.

        Returns:
            xT_input_flat:
                DiT 입력(flatten).
                - use_past_dit_input=True  -> shape (B, Pnn, (time_len+T)*4)
                - use_current_input=True  -> shape (B, Pnn, (1+T)*4)
                - use_current_input=False -> shape (B, Pnn, T*4)
            near_cur_norm_xT:
                현재 프레임(정규화).
                shape: (B, Pnn, 4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # near_cur_future_norm_xT: (B, Pnn, 1+T, 4)
        near_cur_future_norm_xT: torch.Tensor = inputs[
            "near_cur_future_norm_xT"]

        # near_agents_past : (B, Pnn, time_len, 11)
        near_agents_past = inputs["near_agents_past"]

        # near_cur_norm_xT: (B, Pnn, 4)
        near_cur_norm_xT: torch.Tensor = near_cur_future_norm_xT[:, :, 0, :]

        # near_future_norm_xT: (B, Pnn, T, 4)
        near_future_norm_xT: torch.Tensor = near_cur_future_norm_xT[:, :, 1:, :]

        # xT_input_seq: (B, Pnn, time, 4)
        if self.config.use_past_dit_input:
            # xT_input_seq : ( time_len + future_len )  전체 사용
            xT_input_seq: torch.Tensor = torch.cat(
                [near_agents_past, near_future_norm_xT],
                dim=2,
            )  # (B, Pnn, time_len + T, 4)
        else:
            if self.config.use_current_input:
                xT_input_seq: torch.Tensor = near_cur_future_norm_xT  # (B, Pnn, 1+T, 4)
            else:
                xT_input_seq = near_future_norm_xT  # (B, Pnn, T, 4)

        # xT_input_flat: (B, Pnn, F)
        xT_input_flat: torch.Tensor = xT_input_seq.reshape(B, Pnn, -1)

        # 목표점(cond_last_pos_norm)을 학습용으로 일부 배치에서만 마지막 프레임에 주입
        xT_input_flat, _ = self._maybe_apply_last_pos_condition_training(
            near_cur_future_norm_xT=xT_input_flat,  # (B, Pnn, F)
            near_current_mask=near_current_mask,  # (B, Pnn)
            cond_last_pos_norm=cond_last_pos_norm,  # (B, Pnn, 4)
        )
        return xT_input_flat, near_cur_norm_xT

    def _run_dit_training_forward(
            self,
            near_agents_past: Optional[
                torch.Tensor],  # (B, Pnn, time_len(=past_len+1), 11) 또는 None
            xT_input_flat: torch.Tensor,  # (B, Pnn, F)
            diffusion_time: torch.Tensor,  # (B,)
            scene_encoding_token: torch.Tensor,  # (B, token_num, D)
            scene_encoding_token_mask: torch.Tensor,  # (B, token_num)
            ego_fut_global: torch.Tensor,  # (B, D)
            near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
            near_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len+future_len)
            route_known_mask: torch.Tensor,  # (B, Pnn)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
            near_cur_norm_xT: torch.Tensor,  # (B, Pnn, 4)
            near_past: torch.Tensor,  # (B, Pnn, past_len, 11)
    ) -> torch.Tensor:
        """훈련 모드에서 DiT를 1회 호출하고 raw 출력(flat)을 얻는다.

        Args:
            near_agents_past: (B, Pnn, time_len(=past_len+1), 11) 또는 None
            xT_input_flat: (B, Pnn, F)
            diffusion_time: (B,)
            scene_encoding_token: (B, token_num, D)
            scene_encoding_token_mask: (B, token_num)
            ego_fut_global: (B, D)
            near_agents_route_lane_emb: (B, Pnn, D)
            near_past_cur_future_valid: (B, Pnn, time_len+future_len)
            route_known_mask: (B, Pnn)
            near_class_one_hot: (B, Pnn, 3)
            near_cur_norm_xT: (B, Pnn, 4)
            near_past: (B, Pnn, past_len, 11)

        Returns:
            score_flat:
                DiT 출력(flatten).
                shape: (B, Pnn, F_out)
        """
        score_flat: torch.Tensor = self.dit(
            xT_input_flat,  # (B, Pnn, F)
            near_agents_past,
            diffusion_time,  # (B,)
            scene_encoding_token,  # (B, token_num, D)
            ego_fut_global,  # (B, D)
            near_agents_route_lane_emb,  # (B, Pnn, D)
            near_past_cur_future_valid,  # (B, Pnn, time_len+future_len)
            scene_encoding_token_mask,  # (B, token_num)
            route_known_mask,  # (B, Pnn)
            near_class_one_hot,  # (B, Pnn, 3)
            near_current_xyyaw=near_cur_norm_xT,  # (B, Pnn, 4)
            near_past=near_past,  # (B, Pnn, past_len, 11)
        )
        _require_finite("decoder_dit_output", score_flat)
        return score_flat

    def _reshape_training_score_to_sequence(
        self,
        score_flat: torch.Tensor,  # (B, Pnn, F_out)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """훈련 모드에서 DiT 출력(flat)을 (B, Pnn, 1+T, 4) 형태로 바꾼다.

        - use_past_dit_input / use_current_input 설정 조합을 그대로 반영합니다.
        - 최종적으로 Decoder가 리턴하는 score는 항상 (B, Pnn, 1+T, 4) 형태가 되도록 맞춥니다.

        Args:
            score_flat: DiT 출력(flat).
                shape: (B, Pnn, F_out)
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.

        Returns:
            score_seq:
                shape: (B, Pnn, 1+T, 4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        if self.config.use_past_dit_input:
            # score_flat: (B, Pnn, (time_len + future_len) * 4)
            score_seq = score_flat.reshape(
                B,
                Pnn,
                self.config.time_len + self._future_len,
                4,
            )
            # 마지막 (1+T)만 사용: (B, Pnn, 1+T, 4)
            score_seq = score_seq[:, :, -(1 + self._future_len):, :]
            return score_seq

        if self.config.use_current_input:
            # score_seq: (B, Pnn, 1+T, 4)
            score_seq = score_flat.reshape(B, Pnn, 1 + self._future_len, 4)
            return score_seq

        # use_current_input=False 인 경우: DiT 출력은 미래만 (B,Pnn,T,4)
        score_future = score_flat.reshape(B, Pnn, self._future_len,
                                          4)  # (B,Pnn,T,4)
        score_seq = torch.cat(
            [near_current_xyyaw.unsqueeze(2), score_future],
            dim=2,
        )  # (B,Pnn,1+T,4)
        return score_seq

    def _append_training_feasible_outputs(
            self,
            outputs: Dict[str, torch.Tensor],
            near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
    ) -> None:
        """훈련 모드에서 feasible 출력(integrated_trajectory 등)을 outputs dict에 추가한다.

        Args:
            outputs: `_forward_training_mode`가 반환할 dict (in-place 업데이트).
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)

        Returns:
            None
        """
        if not getattr(self.config, "use_feasible", False):
            return

        # integrated_trajectory: (B, Pnn, T, 4)
        integrated_trajectory: torch.Tensor = self.dit.dit_returns.integrated_trajectory
        integrated_trajectory = torch.cat(
            [near_current_xyyaw.unsqueeze(2), integrated_trajectory],
            dim=2,
        )  # (B, Pnn, 1+T, 4)

        outputs["integrated_trajectory"] = integrated_trajectory
        outputs[
            "control_constraint_diff"] = self.dit.dit_returns.control_constraint_diff

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
        has_goal = compute_cond_last_mask(cond_last_pos_norm)  # (B, Pnn)
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
        near_agents_past: torch.Tensor,  # shape: (B, Pnn, time_len, 11)
        near_current_xyyaw: torch.Tensor  # shape: (B, Pnn, 4)
    ) -> torch.Tensor:
        """샘플 xt 를 (B, Pnn, 1+T, 4) 모양으로 펼치고
        '현재 상태 프레임'을 맨 앞에 올바르게 넣어주는 메서드.

        - use_current_input=True  인 경우: xt 안에 이미 현재+미래가 다 들어 있어서
          모양만 (B, Pnn, 1+T, 4) 로 바꿔주고 0번째 프레임을 현재 상태로 덮어쓴다.
        - use_current_input=False 인 경우: xt 안에는 미래 T프레임만 있고,
          0번째 프레임은 나중 단계에서 따로 채운다고 생각하고 모양만 (B, Pnn, T, 4) 로 만든다.
        """
        if self.config.use_past_dit_input:
            # xt_reshaped: (B, Pnn, time_len+T, 4)
            xt_reshaped = xt.reshape(
                batch_size,
                predicted_neighbor_num,
                self.config.time_len + self._future_len,
                4,
            )
            # 과거~현재 상태 주입
            xt_reshaped[:, :, :self.config.
                        time_len, :] = near_agents_past[:, :, :, 0:4]
        else:
            if self.config.use_current_input:
                # xt_reshaped: (B, Pnn, 1+T, 4)
                xt_reshaped = xt.reshape(
                    batch_size,
                    predicted_neighbor_num,
                    1 + self._future_len,
                    4,
                )
                xt_reshaped[:, :,
                            0, :] = near_current_xyyaw  # 현재 상태 주입 (B,Pnn,4)
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
            xt: torch.Tensor,  # (B, Pnn, _, 4) 정도라고 가정
            cond_last_pos: Optional[torch.Tensor],  # (B, Pnn, 4) 또는 None
    ) -> torch.Tensor:
        """목표점(cond-last)이 있는 에이전트만 마지막 프레임 위치를 덮어쓴다.

        처리 흐름:
            1) 목표점 텐서가 없으면(=None) 그대로 반환합니다.
            2) 목표점 텐서가 있으면 xt와 dtype/device를 맞춥니다.
               - 샘플링 중 xt dtype이 바뀌는 경우가 있어서, 여기서 한 번 더 안전하게 맞춥니다.
            3) 목표가 있는 에이전트만 골라 마지막 프레임의 (x, y) 값을 목표로 바꿉니다.

        Args:
            xt (torch.Tensor): 현재 샘플 시퀀스.
                shape: (B, Pnn, _, 4)
            cond_last_pos (Optional[torch.Tensor]): 목표점(정규화) 또는 None.
                - None 또는 shape: (B, Pnn, 4)

        Returns:
            torch.Tensor:
                목표점이 있는 에이전트만 마지막 프레임이 보정된 xt.
                shape: (B, Pnn, _, 4)
        """
        if cond_last_pos is None:
            return xt

        # dtype/device 정합 (중요)
        cond_last_pos = _cast_like(cond_last_pos, xt)  # (B, Pnn, 4)

        # cond_last_mask: (B, Pnn)
        cond_last_mask: torch.Tensor = compute_cond_last_mask(cond_last_pos)

        # xt_last: (B, Pnn, 4)
        xt_last = xt[..., -1, :]
        new_last = xt_last.clone()

        # 목표점의 (x, y): (B, Pnn, 2)
        new_last_xy = cond_last_pos[..., :2]

        # 목표가 있는 에이전트만 마지막 프레임 (x,y) 교체
        new_last[..., :2] = torch.where(
            cond_last_mask[..., None],  # (B, Pnn, 1)
            new_last_xy,
            xt_last[..., :2],
        )

        # in-place 갱신: xt[..., -1, :] shape (B, Pnn, 4)
        xt[..., -1, :] = new_last
        return xt

    def _project_future_yaw_to_unit_circle(
            self,
            xt_sequence: torch.Tensor,  # shape: (B, Pnn, _, 4)
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
        xt_sequence[:, :, :, 2:4] = yaw_direction / yaw_norm  # (B,Pnn,_,2)
        return xt_sequence

    def _compute_feasible_blend_beta(
            self,
            diffusion_time_step: torch.Tensor,  # shape: () or (1,)  
            reference_tensor_for_device: torch.Tensor,  # shape: (B, Pnn, _, 4)
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
        reference_tensor_for_device: torch.Tensor,  # shape: (B, Pnn, _, 4)
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
            xt_sequence: torch.Tensor,  # shape: (B, Pnn, _, 4)
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
            reference_tensor_for_device=xt_sequence,  # shape: (B, Pnn, _, 4)
        )
        if integrated_future is None:
            return xt_sequence

        # xt_sequence 에서 "미래" 부분만 (B, Pnn, future_len, 4) 형태로 뽑기
        if self.config.use_past_dit_input:
            if time_len != self.config.time_len + future_len:
                # time_len 이 다르면 안전하게 스킵
                raise ValueError(
                    "시간 길이가 맞지 않습니다. time_len={}, future_len={}".format(
                        time_len, future_len))
            xt_future = xt_sequence[:, :, self.config.
                                    time_len:, :]  # (B, Pnn, future_len, 4)
        else:
            if self.config.use_current_input:
                # xt_sequence: (B, Pnn, 1+future_len, 4)  -> 미래만 사용
                if time_len != future_len + 1:
                    # time_len 이 다르면 안전하게 스킵
                    raise ValueError(
                        "시간 길이가 맞지 않습니다. time_len={}, future_len={}".format(
                            time_len, future_len))
                xt_future = xt_sequence[:, :, 1:, :]  # (B, Pnn, future_len, 4)
            else:
                # xt_sequence: (B, Pnn, future_len, 4) 가 바로 미래 궤적
                if time_len != future_len:
                    raise ValueError(
                        "시간 길이가 맞지 않습니다. time_len={}, future_len={}".format(
                            time_len, future_len))
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
        if self.config.use_past_dit_input:
            xt_sequence[:, :, self.config.time_len:, :] = new_future
        else:
            if self.config.use_current_input:
                xt_sequence[:, :, 1:, :] = new_future
            else:
                xt_sequence[:, :, :, :] = new_future

        return xt_sequence

    def _prepare_near_trajectories_and_masks(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
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
        near_agents_past = inputs["near_agents_past"]  # (B, Pnn, time_len, 11)
        near_past = near_agents_past[:, :, :-1, :].detach(
        )  # (B, Pnn, past_len, 11)
        near_current_xyyaw = near_agents_past[:, :, -1, :4]  # (B, Pnn, 4)
        # 과거+현재+미래 유효 프레임 마스크
        near_future_valid: Optional[torch.Tensor] = inputs.get(
            "near_future_valid", None)  # (B, Pnn, future_len) 또는 None
        near_past_current_mask = torch.sum(torch.ne(
            near_agents_past[:, :, :, :8], 0),
                                           dim=-1) == 0  # (B, pnn, time_len)
        near_current_mask = near_past_current_mask[:, :, -1]  # (B, Pnn)
        near_past_cur_future_valid: torch.Tensor = (
            self._get_near_past_cur_future_valid(
                near_past_current_mask=
                near_past_current_mask,  # [B, pnn, time_len]  True=빈 슬롯(무효 에이전트)
                near_future_valid=near_future_valid,
            ))  # (B, Pnn, time_len + future_len)
        near_class_one_hot = near_agents_past[:, :, -1, 8:11]  # (B, Pnn, 3)
        # cond_last_pos_norm 준비
        batch_size, predicted_neighbor_num, _ = near_current_xyyaw.shape

        if "cond_last_pos_norm" in inputs:
            cond_last_pos_norm: torch.Tensor = inputs[
                "cond_last_pos_norm"]  # (B, Pnn, 4)
        else:
            cond_last_pos_norm = near_current_xyyaw.new_full(
                (batch_size, predicted_neighbor_num, 4),
                float("nan"),
            )  # (B, Pnn, 4)

        # dtype/device 를 near_current_xyyaw 와 맞춤
        cond_last_pos_norm = _cast_like(cond_last_pos_norm,
                                        near_current_xyyaw)  # (B, Pnn, 4)

        return (
            near_past,  # (B, Pnn, past_len, 11)
            near_current_xyyaw,  # (B, Pnn, 4)
            near_current_mask,  # (B, Pnn)
            near_past_cur_future_valid,  # (B, Pnn, time_len + future_len)
            near_class_one_hot,  # (B, Pnn, 3)
            cond_last_pos_norm,  # (B, Pnn, 4)
            batch_size,  # int
            predicted_neighbor_num,  # int
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
            "encoding"]  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor = encoder_outputs[
            "encoding_mask"]  # (B, token_num) bool
        ego_fut_global: torch.Tensor = encoder_outputs[
            "ego_fut_global"]  # (B, D)
        near_agents_route_lane_emb: torch.Tensor = encoder_outputs[
            "near_agents_route_lane_emb"]  # (B, Pnn, D)
        route_known_mask: torch.Tensor = encoder_outputs[
            "route_known_mask"]  # (B, Pnn) bool

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

    def _sample_inference_noise(
        self,
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """추론 모드에서 시작점으로 쓸 미래 노이즈 궤적을 만든다.

        Args:
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.

        Returns:
            noise:
                미래 프레임 노이즈.
                shape: (B, Pnn, T, 4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num
        noise: torch.Tensor = near_current_xyyaw.new_empty(
            (B, Pnn, self._future_len, 4),).normal_(0.0, 0.5)  # (B, Pnn, T, 4)
        return noise

    def _build_inference_xT_from_noise(
        self,
        noise: torch.Tensor,  # (B, Pnn, T, 4)
        near_agents_past: torch.Tensor,  # (B, Pnn, time_len, 11)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """샘플링 시작점 xT(flat)를 만든다.

        - use_current_input=True: (현재 + 미래노이즈) 를 이어 붙여 flatten.
        - use_current_input=False: 미래노이즈만 flatten.

        Args:
            noise: (B, Pnn, T, 4)
            near_current_xyyaw: (B, Pnn, 4)
            batch_size: B
            predicted_neighbor_num: Pnn

        Returns:
            xT:
                shape:
                  - use_past_dit_input=True  -> (B, Pnn, (time_len+T)*4)
                  - use_current_input=True  -> (B, Pnn, (1+T)*4)
                  - use_current_input=False -> (B, Pnn, T*4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num
        if self.config.use_past_dit_input:
            # XT: (B, Pnn, time_len+T, 4)
            xT: torch.Tensor = torch.cat(
                [
                    near_agents_past[:, :, :, :4],  # (B, Pnn, time_len, 4)
                    noise,  # (B, Pnn, T, 4)
                ],
                dim=2,
            ).reshape(B, Pnn, -1)  # (B, Pnn, (time_len+T)*4)
        else:
            if self.config.use_current_input:
                xT: torch.Tensor = torch.cat(
                    [
                        near_current_xyyaw[:, :, None, :],  # (B, Pnn, 1, 4)
                        noise,  # (B, Pnn, T, 4)
                    ],
                    dim=2,
                ).reshape(B, Pnn, -1)  # (B, Pnn, (1+T)*4)
                return xT

            xT = noise.reshape(B, Pnn, -1)  # (B, Pnn, T*4)
        return xT

    def _build_inference_cond_last_info(
        self,
        cond_last_pos_norm: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
        reference_tensor_for_device: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """목표점(cond-last) 텐서와 '목표가 있는 에이전트' 마스크를 만든다.

        이 함수는 추론(샘플링)에서 목표점 정보를 안전하게 정리하는 역할을 합니다.

        처리 흐름:
            1) cond_last_pos_norm으로부터 '목표가 실제로 있는지' 마스크를 먼저 만듭니다.
               - cond_last_mask: (B, Pnn)
            2) 마스크가 전부 False면(목표가 아무도 없으면),
               - cond_last_pos는 None으로 내려서 불필요한 텐서 전달을 피합니다.
            3) 목표가 하나라도 있으면,
               - cond_last_pos_norm을 샘플 텐서(reference_tensor_for_device)와 같은 dtype/device로 맞춰
                 cond_last_pos로 반환합니다.

        Args:
            cond_last_pos_norm (torch.Tensor): 목표점(정규화).
                shape: (B, Pnn, 4)
            batch_size (int): 배치 크기 B.
            predicted_neighbor_num (int): 이웃 수 Pnn.
            reference_tensor_for_device (torch.Tensor): 결과 텐서의 dtype/device 기준 텐서.
                shape: 보통 (B, Pnn, F)

        Returns:
            Tuple[Optional[torch.Tensor], torch.Tensor]:
                cond_last_pos:
                    - 목표가 없으면 None
                    - 목표가 있으면 shape: (B, Pnn, 4)
                cond_last_mask:
                    - 목표가 있는 에이전트만 True
                    - shape: (B, Pnn)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num
        assert cond_last_pos_norm.shape[:2] == (B, Pnn), (
            f"cond_last_pos_norm shape mismatch: got {tuple(cond_last_pos_norm.shape)}, "
            f"expected (B,Pnn,4)=({B},{Pnn},4)")

        # cond_last_mask: (B, Pnn)
        cond_last_mask: torch.Tensor = compute_cond_last_mask(
            cond_last_pos_norm)
        cond_last_mask = cond_last_mask.to(
            device=reference_tensor_for_device.device)

        # 목표가 아무도 없으면, 텐서를 들고 다니지 않도록 None 처리
        if not cond_last_mask.any().item():
            return None, cond_last_mask

        # 목표가 있으면 dtype/device 정렬해서 전달
        cond_last_pos: torch.Tensor = _cast_like(
            cond_last_pos_norm, reference_tensor_for_device)  # (B, Pnn, 4)
        return cond_last_pos, cond_last_mask

    def _initial_state_constraint(
            self,
            xt: torch.Tensor,  # (B, Pnn, F)
            t: torch.Tensor,  # () 또는 (1,)
            step: int,
            *,
            batch_size: int,  # DONE
            predicted_neighbor_num: int,  # DONE
            near_agents_past: torch.Tensor,  # (B, Pnn, time_len, 11)
            near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
            near_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len_total)
            cond_last_pos_norm: Optional[torch.Tensor],  # (B, Pnn, 4) 또는 None
            cond_last_mask: torch.Tensor,  # (B, Pnn)
    ) -> torch.Tensor:
        """샘플링 중간 단계에서 샘플을 '현재/목표/기본 규칙'에 맞게 정리한다.

        이 함수는 샘플링이 진행되는 동안 여러 번 호출되며,
        매번 다음을 수행합니다.

        1) (옵션) 현재 상태 프레임을 강제로 맞춥니다.
           - use_current_input=True일 때만 현재 프레임이 xt 안에 존재하므로,
             그 경우 0번째 프레임을 near_current_xyyaw로 덮어씁니다.
        2) (옵션) 목표점(cond-last)이 있는 에이전트에 대해
           마지막 프레임의 (x,y)를 목표로 덮어씁니다.
        3) (옵션) feasible 보정 결과와 일부 섞습니다.
           - 노이즈가 거의 빠진 구간에서만 섞도록 beta(t)로 조절합니다.
           - 목표점이 있는 에이전트는 마지막 프레임을 보호합니다.
        4) (cos, sin)이 단위원(길이 1) 위에 있도록 다시 정리합니다.

        Args:
            xt: 현재 단계 샘플(flat).
                shape: (B, Pnn, F)
            t: 현재 시간 값.
                shape: () 또는 (1,)
            step: 단계 번호(시그니처 맞춤용). 내부 로직에서는 사용하지 않습니다.
            batch_size: 배치 크기 B.
            predicted_neighbor_num: 이웃 수 Pnn.
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            near_past_cur_future_valid: 과거~미래 유효 마스크.
                shape: (B, Pnn, time_len_total)
            cond_last_pos_norm: 목표점(정규화) 또는 None.
                - None 또는 shape: (B, Pnn, 4)
            cond_last_mask: 목표점이 실제로 있는 에이전트 마스크.
                shape: (B, Pnn)

        Returns:
            xt_out: 보정 후 flat 샘플.
                shape: (B, Pnn, F)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # xt_sequence: (B, Pnn, _, 4)
        xt_sequence: torch.Tensor = self._reshape_xt_with_current_state(
            xt=xt,
            batch_size=B,
            predicted_neighbor_num=Pnn,
            near_agents_past=near_agents_past,  # (B, Pnn, time_len, 11)
            near_current_xyyaw=near_current_xyyaw,  # (B, Pnn, 4)
        )

        # (옵션) 목표점 주입
        xt_sequence = self._inject_last_position_condition_if_available(
            xt=xt_sequence,  # (B, Pnn, _, 4)
            cond_last_pos=cond_last_pos_norm,  # (B, Pnn, 4) or None
        )

        # (옵션) feasible 결과와 섞기
        if getattr(self.config, "use_feasible_blend", False):
            assert getattr(self.config, "use_feasible", False), (
                "use_feasible_blend 옵션은 use_feasible 이 켜져 있을 때만 동작합니다.")

            beta_scalar: torch.Tensor = self._compute_feasible_blend_beta(
                diffusion_time_step=t,  # () 또는 (1,)
                reference_tensor_for_device=xt_sequence,  # (B, Pnn, _, 4)
            )  # shape: ()

            if beta_scalar.detach().item() > 0.0:
                xt_sequence = self._apply_feasible_blend_with_feasible_projection(
                    xt_sequence=xt_sequence,  # (B, Pnn, _, 4)
                    near_past_cur_future_valid=
                    near_past_cur_future_valid,  # (B,Pnn,time_len_total)
                    cond_last_mask=cond_last_mask,  # (B, Pnn)
                    feasible_blend_beta_scalar=beta_scalar,  # ()
                )

        # yaw (cos, sin) 정리
        xt_sequence = self._project_future_yaw_to_unit_circle(
            xt_sequence=xt_sequence,  # (B, Pnn, _, 4)
        )

        # 다시 flatten: (B, Pnn, F)
        return xt_sequence.reshape(B, Pnn, -1)

    def _build_inference_correcting_xt_fn(
        self,
        batch_size: int,
        predicted_neighbor_num: int,
        near_agents_past: torch.Tensor,  # (B, Pnn, time_len, 11)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        near_past_cur_future_valid: torch.Tensor,  # (B, Pnn, time_len_total)
        cond_last_pos_norm: Optional[torch.Tensor],  # (B, Pnn, 4) 또는 None
        cond_last_mask: torch.Tensor,  # (B, Pnn)
    ) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        """샘플링 루프가 요구하는 시그니처(xt,t,step) 형태의 보정 함수를 만든다.

        내부적으로는 self._initial_state_constraint 를 사용하고,
        batch/마스크/현재상태 같은 '고정 정보'는 partial로 미리 묶어서 전달합니다.

        Args:
            batch_size: B
            predicted_neighbor_num: Pnn
            near_current_xyyaw: (B, Pnn, 4)
            near_past_cur_future_valid: (B, Pnn, time_len_total)
            cond_last_pos_norm: (B, Pnn, 4) 또는 None
            cond_last_mask: (B, Pnn)

        Returns:
            correcting_xt_fn:
                callable(xt, t, step) -> xt_out
                xt: (B, Pnn, F), xt_out: (B, Pnn, F)
        """
        return partial(
            self._initial_state_constraint,
            batch_size=batch_size,
            predicted_neighbor_num=predicted_neighbor_num,
            near_agents_past=near_agents_past,
            near_current_xyyaw=near_current_xyyaw,
            near_past_cur_future_valid=near_past_cur_future_valid,
            cond_last_pos_norm=cond_last_pos_norm,
            cond_last_mask=cond_last_mask,
        )

    def _run_dpm_sampler_for_inference(
        self,
        xT: torch.Tensor,  # (B, Pnn, F)
        near_agents_past: Optional[
            torch.Tensor],  # (B, Pnn, time_len(=past_len+1), 11) 또는 None
        scene_encoding_token: torch.Tensor,  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor,  # (B, token_num)
        ego_fut_global: torch.Tensor,  # (B, D)
        near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
        near_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len+future_len)
        route_known_mask: torch.Tensor,  # (B, Pnn)
        near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        near_past: torch.Tensor,  # (B, Pnn, past_len, 11)
        inputs: Dict[str, torch.Tensor],
        correcting_xt_fn: Callable[[torch.Tensor, torch.Tensor, int],
                                   torch.Tensor],
    ) -> torch.Tensor:
        """dpm_sampler를 호출해 최종 샘플 x0(flat)을 얻는다.

        Args:
            xT: 시작 샘플(flat).
                shape: (B, Pnn, F)
            near_agents_past: (B, Pnn, time_len(=past_len+1), 11) 또는 None
            scene_encoding_token: (B, token_num, D)
            scene_encoding_token_mask: (B, token_num)
            ego_fut_global: (B, D)
            near_agents_route_lane_emb: (B, Pnn, D)
            near_past_cur_future_valid: (B, Pnn, time_len+future_len)
            route_known_mask: (B, Pnn)
            near_class_one_hot: (B, Pnn, 3)
            near_current_xyyaw: (B, Pnn, 4)
            near_past: (B, Pnn, past_len, 11)
            inputs: 배치 dict (가이던스 함수에 전달).
            correcting_xt_fn: callable(xt,t,step)->xt_out

        Returns:
            x0:
                샘플링 결과(flat).
                shape: (B, Pnn, F)
        """

        x0: torch.Tensor = dpm_sampler(
            self.dit,
            xT.float(),  # (B, Pnn, F)
            other_model_params={
                "near_agents_past": near_agents_past,
                "cross_c": scene_encoding_token,
                "ego_fut_global": ego_fut_global,
                "near_agents_route_lane_emb": near_agents_route_lane_emb,
                "near_past_cur_future_valid": near_past_cur_future_valid,
                "cross_mask": scene_encoding_token_mask,
                "route_known_mask": route_known_mask,
                "near_class_one_hot": near_class_one_hot,
                "near_current_xyyaw": near_current_xyyaw,
                "near_past": near_past,
            },
            dpm_solver_params={
                "correcting_xt_fn": correcting_xt_fn,
            },
            model_wrapper_params={
                "classifier_fn":
                    self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit,
                    "model_condition": {
                        "near_agents_past":
                            near_agents_past,
                        "cross_c":
                            scene_encoding_token,
                        "ego_fut_global":
                            ego_fut_global,
                        "near_agents_route_lane_emb":
                            near_agents_route_lane_emb,
                        "near_past_cur_future_valid":
                            near_past_cur_future_valid,
                        "cross_mask":
                            scene_encoding_token_mask,
                        "route_known_mask":
                            route_known_mask,
                        "near_class_one_hot":
                            near_class_one_hot,
                        "near_current_xyyaw":
                            near_current_xyyaw,
                        "near_past":
                            near_past,
                    },
                    "inputs": inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                    "config": self.config,
                },
                "guidance_scale":
                    1.0,
                "guidance_type":
                    ("classifier" if self._guidance_fn is not None else "uncond"
                    ),
            },
        )
        return x0

    def _reshape_inference_x0_to_sequence(
        self,
        x0: torch.Tensor,  # (B, Pnn, F)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """샘플링 결과 x0(flat)를 (B, Pnn, 1+T, 4) 형태로 복원한다.

        Args:
            x0: 샘플링 결과(flat).
                shape: (B, Pnn, F)
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            batch_size: B
            predicted_neighbor_num: Pnn

        Returns:
            x0_seq:
                shape: (B, Pnn, 1+T, 4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        if self.config.use_past_dit_input:
            x0_seq = x0.reshape(
                B,
                Pnn,
                self.config.time_len + self._future_len,
                4,
            )
            x0_seq = x0_seq[:, :, -(1 + self._future_len):, :]  # (B,Pnn,1+T,4)
            return x0_seq

        if self.config.use_current_input:
            assert x0.shape == (B, Pnn, (1 + self._future_len) * 4)
            x0_seq = x0.reshape(B, Pnn, -1, 4)  # (B,Pnn,1+T,4)
            return x0_seq

        assert x0.shape == (B, Pnn, self._future_len * 4)
        x0_seq = torch.cat(
            [
                near_current_xyyaw.unsqueeze(2),  # (B,Pnn,1,4)
                x0.reshape(B, Pnn, -1, 4),  # (B,Pnn,T,4)
            ],
            dim=2,
        )  # (B,Pnn,1+T,4)
        return x0_seq

    def _inverse_normalize_and_cleanup_padding(
            self,
            x_norm: torch.Tensor,  # (B, Pnn, 1+T, 4)
            near_current_mask: torch.Tensor,  # (B, Pnn)
    ) -> torch.Tensor:
        """정규화된 궤적을 역정규화하고, 패딩 슬롯은 0으로 정리한다.

        Args:
            x_norm: 정규화된 궤적.
                shape: (B, Pnn, 1+T, 4)
            near_current_mask: True=무효 에이전트.
                shape: (B, Pnn)

        Returns:
            x_denorm:
                역정규화된 궤적.
                shape: (B, Pnn, 1+T, 4)
        """
        x_denorm: torch.Tensor = self._state_normalizer.inverse(
            x_norm)  # (B,Pnn,1+T,4)
        x_denorm[near_current_mask] = 0.0
        return x_denorm

    def _append_inference_feasible_outputs(
            self,
            outputs: Dict[str, torch.Tensor],
            near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
            near_current_mask: torch.Tensor,  # (B, Pnn)
    ) -> None:
        """추론 모드에서 feasible 출력(integrated_trajectory)을 outputs에 추가한다.

        Args:
            outputs: 반환 dict (in-place 업데이트).
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            near_current_mask: True=무효 에이전트.
                shape: (B, Pnn)

        Returns:
            None
        """
        if not getattr(self.config, "use_feasible", False):
            return

        # integrated_trajectory: (B, Pnn, T, 4)  (정규화 상태)
        integrated_trajectory: torch.Tensor = self.dit.dit_returns.integrated_trajectory
        integrated_trajectory = torch.cat(
            [near_current_xyyaw.unsqueeze(2), integrated_trajectory],
            dim=2,
        )  # (B, Pnn, 1+T, 4)

        unnorm_integrated: torch.Tensor = self._state_normalizer.inverse(
            integrated_trajectory)
        unnorm_integrated[near_current_mask] = 0.0

        outputs["integrated_trajectory"] = unnorm_integrated

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
        """훈련 모드에서 한 배치에 대해 decoder 를 한 번 돌린다."""
        return_: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # 1) DiT 입력 준비 (flatten + 목표점 주입 옵션)
        xT_input_flat, near_cur_norm_xT = self._build_training_dit_inputs(
            inputs=inputs,
            near_current_mask=near_current_mask,  # (B, Pnn)
            cond_last_pos_norm=cond_last_pos_norm,  # (B, Pnn, 4)
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )

        # 2) DiT 1회 호출
        diffusion_time: torch.Tensor = inputs["diffusion_time"]  # (B,)
        near_agents_past: Optional[torch.Tensor] = inputs[
            "near_agents_past"]  # (B,Pnn,time_len(=past_len+1),11) 또는 None

        score_flat: torch.Tensor = self._run_dit_training_forward(
            near_agents_past=near_agents_past,
            xT_input_flat=xT_input_flat,  # (B, Pnn, F)
            diffusion_time=diffusion_time,  # (B,)
            scene_encoding_token=scene_encoding_token,  # (B, token_num, D)
            scene_encoding_token_mask=scene_encoding_token_mask,  # (B, token_num)
            ego_fut_global=ego_fut_global,  # (B, D)
            near_agents_route_lane_emb=near_agents_route_lane_emb,  # (B, Pnn, D)
            near_past_cur_future_valid=
            near_past_cur_future_valid,  # (B, Pnn, time_len+future_len)
            route_known_mask=route_known_mask,  # (B, Pnn)
            near_class_one_hot=near_class_one_hot,  # (B, Pnn, 3)
            near_cur_norm_xT=near_cur_norm_xT,  # (B, Pnn, 4)
            near_past=near_past,  # (B, Pnn, past_len, 11)
        )

        # 3) (B,Pnn,F_out) -> (B,Pnn,1+T,4)
        score_seq: torch.Tensor = self._reshape_training_score_to_sequence(
            score_flat=score_flat,
            near_current_xyyaw=near_current_xyyaw,  # (B, Pnn, 4)
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )
        return_["score"] = score_seq

        # 4) feasible 출력 추가(옵션)
        self._append_training_feasible_outputs(
            outputs=return_,
            near_current_xyyaw=near_current_xyyaw,  # (B, Pnn, 4)
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
        """추론/평가 모드에서 한 배치에 대해 decoder 를 한 번 돌린다."""
        return_: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # 1) 시작 노이즈 및 xT(flat) 생성
        noise: torch.Tensor = self._sample_inference_noise(
            near_current_xyyaw=near_current_xyyaw,  # (B,Pnn,4)
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )  # (B,Pnn,T,4)
        near_agents_past = input[
            "near_agents_past"]  # (B,Pnn,time_len(=past_len+1),11)
        xT: torch.Tensor = self._build_inference_xT_from_noise(
            noise=noise,  # (B,Pnn,T,4)
            near_agents_past=near_agents_past,
            near_current_xyyaw=near_current_xyyaw,  # (B,Pnn,4)
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )  # (B,Pnn,F)

        # 2) 목표점(cond-last) 정보 준비
        cond_last_pos, cond_last_mask = self._build_inference_cond_last_info(
            cond_last_pos_norm=cond_last_pos_norm,  # (B,Pnn,4)
            batch_size=B,
            predicted_neighbor_num=Pnn,
            reference_tensor_for_device=xT,
        )

        # 3) 샘플링 단계에서 쓸 보정 함수 만들기(메서드 + partial)

        # 4) dpm_sampler 실행
        near_agents_past: Optional[torch.Tensor] = inputs[
            "near_agents_past"]  # (B,Pnn,time_len(=past_len+1),11) 또는 None

        correcting_xt_fn = self._build_inference_correcting_xt_fn(
            batch_size=B,
            predicted_neighbor_num=Pnn,
            near_agents_past=near_agents_past,  # (B,Pnn,time_len, 11)
            near_current_xyyaw=near_current_xyyaw,  # (B,Pnn,4)
            near_past_cur_future_valid=
            near_past_cur_future_valid,  # (B,Pnn,time_total)
            cond_last_pos_norm=cond_last_pos,  # (B,Pnn,4) or None
            cond_last_mask=cond_last_mask,  # (B,Pnn)
        )

        x0: torch.Tensor = self._run_dpm_sampler_for_inference(
            xT=xT,  # (B,Pnn,F)
            near_agents_past=
            near_agents_past,  # (B,Pnn,time_len(=past_len+1),11) or None
            scene_encoding_token=scene_encoding_token,
            scene_encoding_token_mask=scene_encoding_token_mask,
            ego_fut_global=ego_fut_global,
            near_agents_route_lane_emb=near_agents_route_lane_emb,
            near_past_cur_future_valid=near_past_cur_future_valid,
            route_known_mask=route_known_mask,
            near_class_one_hot=near_class_one_hot,
            near_current_xyyaw=near_current_xyyaw,
            near_past=near_past,
            inputs=inputs,
            correcting_xt_fn=correcting_xt_fn,
        )

        # 원래 코드와 동일하게 dtype 맞춤
        x0 = x0.to(xT.dtype)

        # 5) (B,Pnn,F) -> (B,Pnn,1+T,4) 복원
        x0_seq_norm: torch.Tensor = self._reshape_inference_x0_to_sequence(
            x0=x0,
            near_current_xyyaw=near_current_xyyaw,
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )  # (B,Pnn,1+T,4)

        # 6) 역정규화 및 패딩 정리
        unnorm_x0: torch.Tensor = self._inverse_normalize_and_cleanup_padding(
            x_norm=x0_seq_norm,
            near_current_mask=near_current_mask,
        )  # (B,Pnn,1+T,4)

        # 7) feasible 출력(옵션)
        self._append_inference_feasible_outputs(
            outputs=return_,
            near_current_xyyaw=near_current_xyyaw,
            near_current_mask=near_current_mask,
        )

        return_["score"] = unnorm_x0
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


from dataclasses import dataclass


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
            near_input_norm_xT: torch.Tensor,  # (B, Pnn, F=(1+T)*4)
            near_current_mask: torch.Tensor,  # (B, Pnn) True=pad(무효 에이전트)
    ) -> torch.Tensor:
        """pre-proj MLP 를 유효 에이전트 토큰에만 적용하는 전처리 함수.

        마스크를 이용해 유효 토큰만 펼친 뒤 MLP 를 통과시키고,
        다시 배치 모양으로 되돌립니다.

        Args:
            near_input_norm_xT (torch.Tensor):
                정규화된 현재+미래 입력.
                shape: (B, Pnn, F)  (F=(1+T)*4)
            near_current_mask (torch.Tensor):
                현재 프레임 기준 무효 에이전트 마스크(True=패딩).
                shape: (B, Pnn)

        Returns:
            torch.Tensor:
                pre-proj 후 토큰.
                shape: (B, Pnn, D)
        """
        B, Pnn, F = near_input_norm_xT.shape  # (B, Pnn, F)
        # unpad_input 은 True=유효 이므로 반전 필요
        attention_mask = (~near_current_mask).to(torch.bool)  # (B, Pnn)

        res = unpad_input(near_input_norm_xT, attention_mask)
        # x_unpad: (T_total, F), indices: (T_total,)
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        else:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        if x_unpad.numel() == 0:
            D_out = self.preproj.fc2.out_features
            zeros = near_input_norm_xT.new_zeros((B, Pnn, D_out))
            touch = (self.preproj.fc1.weight.view(-1)[:1].sum() +
                     (self.preproj.fc1.bias.view(-1)[:1].sum()
                      if self.preproj.fc1.bias is not None else 0) +
                     self.preproj.fc2.weight.view(-1)[:1].sum() +
                     (self.preproj.fc2.bias.view(-1)[:1].sum()
                      if self.preproj.fc2.bias is not None else 0)) * 0.0
            return zeros + touch

        # 유효 토큰만 pre-proj 수행
        x_unpad = self.preproj(x_unpad)  # (T_total, D)

        # 다시 배치 모양으로 복원 (pad 위치는 0)
        x = pad_input(x_unpad, indices, B, Pnn)  # (B, Pnn, D)
        return x

    def _select_tensors_for_feasible_projection(
        self,
        x: torch.Tensor,  # (B, Pnn, F_out)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        near_past_cur_future_valid: torch.Tensor,  # (B, Pnn, time_len_total)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FeasibleProjector에 넣기 전, 텐서의 '그래디언트 유지/차단' 방식을 정한다.

        - feasible_grad_to_dit=True:
            학습이 가능한 경로를 유지하기 위해 x/near_current를 그대로 사용하고 float32로만 바꿉니다.
        - feasible_grad_to_dit=False:
            FeasibleProjector 쪽으로는 학습 신호가 흐르지 않게 detach한 뒤 float32로 바꿉니다.

        Args:
            x: DiT 본체 출력(flat).
                shape: (B, Pnn, F_out)
            near_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            near_past_cur_future_valid: 과거~미래 유효 마스크.
                shape: (B, Pnn, time_len_total)

        Returns:
            x_for_feasible:
                shape: (B, Pnn, F_out), dtype=float32
            near_current_xyyaw_for_feasible:
                shape: (B, Pnn, 4), dtype=float32
            near_past_cur_future_valid_for_feasible:
                shape: (B, Pnn, time_len_total)
        """
        if getattr(self.config, "feasible_grad_to_dit", False):
            x_for_feasible = x.float()
            near_current_xyyaw_for_feasible = near_current_xyyaw.float()
            near_past_cur_future_valid_for_feasible = near_past_cur_future_valid
        else:
            x_for_feasible = x.detach().float()
            near_current_xyyaw_for_feasible = near_current_xyyaw.detach().float(
            )
            near_past_cur_future_valid_for_feasible = near_past_cur_future_valid.detach(
            )
        return (
            x_for_feasible,
            near_current_xyyaw_for_feasible,
            near_past_cur_future_valid_for_feasible,
        )

    def _build_diffusion_trajectory_for_feasible_projection(
        self,
        x_for_feasible: torch.Tensor,  # (B, Pnn, F_out)
        near_current_xyyaw_for_feasible: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """FeasibleProjector 입력으로 쓸 (현재+미래) 궤적 텐서를 만든다.

        이 함수는 x(flat)를 (B, Pnn, 1+T, 4)로 바꾸고,
        첫 프레임(현재)을 near_current_xyyaw로 맞춰줍니다.

        Args:
            x_for_feasible: DiT 출력(flat).
                shape: (B, Pnn, F_out)
            near_current_xyyaw_for_feasible: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            batch_size: B
            predicted_neighbor_num: Pnn

        Returns:
            diffusion_trajectory:
                shape: (B, Pnn, 1+T, 4)
        """
        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        # diffusion_trajectory: (B, Pnn, 1+T, 4)
        if self.config.use_past_dit_input:
            # x_for_feasible: (B, Pnn, (past_len + 1+T)*4) 같은 형태
            diffusion_trajectory = (x_for_feasible.reshape(
                B, Pnn, -1, 4).contiguous()[:, :, -(self._future_len + 1):, :]
                                   )  # (B, Pnn, 1+T, 4)
            diffusion_trajectory[:, :, 0, :] = near_current_xyyaw_for_feasible
            return diffusion_trajectory.contiguous()

        if self.config.use_current_input:
            diffusion_trajectory = x_for_feasible.reshape(
                B, Pnn, -1, 4).contiguous()  # (B,Pnn,1+T,4)
            diffusion_trajectory[:, :, 0, :] = near_current_xyyaw_for_feasible
            return diffusion_trajectory.contiguous()

        # use_current_input=False: x_for_feasible는 미래만 (B,Pnn,T*4)
        diffusion_trajectory = torch.cat(
            [
                near_current_xyyaw_for_feasible.unsqueeze(2),  # (B,Pnn,1,4)
                x_for_feasible.reshape(B, Pnn, -1, 4),  # (B,Pnn,T,4)
            ],
            dim=2,
        ).contiguous()  # (B,Pnn,1+T,4)
        return diffusion_trajectory

    def _compute_feasible_low_t_mask(
            self,
            diffusion_time: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """현재 시간 값으로 'feasible을 돌릴 배치' 마스크를 만든다.

        - diffusion_time이 작을수록(노이즈가 적다고 보는 구간) feasible을 적극 적용합니다.
        - direct loss를 쓰지 않는 설정에서는 모든 배치에 대해 feasible을 돌리기 위해
          threshold를 1.0으로 둡니다(기존 로직 유지).

        Args:
            diffusion_time: 확산 시간.
                shape: (B,)

        Returns:
            low_t_mask:
                feasible을 실제로 실행할 배치만 True.
                shape: (B,)
        """
        t_threshold: float = float(self.config.feasible_learn_noise_thresh)
        if not getattr(self.config, "use_direct_loss", False):
            t_threshold = 1.0
        low_t_mask: torch.Tensor = (diffusion_time <= t_threshold)  # (B,)
        return low_t_mask

    def _maybe_replace_x_with_integrated_trajectory_for_eval(
        self,
        x: torch.Tensor,  # (B, Pnn, F_out)
        integrated_trajectory: torch.Tensor,  # (B, Pnn, T, 4)
        near_current_xyyaw_for_feasible: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        predicted_neighbor_num: int,
    ) -> torch.Tensor:
        """평가 모드에서 integrated_trajectory를 최종 출력으로 쓸지 결정한다.

        기존 규칙(그대로 유지):
            - training 모드이거나 direct loss 사용 시: x 그대로 반환
            - 평가 모드 + direct loss 미사용 시:
                * past_dit_input: x의 과거+현재 부분 + integrated_trajectory를 이어붙여 반환
                * current_input: 현재 + integrated_trajectory
                * current_input=False: integrated_trajectory만 반환

        Args:
            x: DiT 출력(flat).
                shape: (B, Pnn, F_out)
            integrated_trajectory: feasible 적분 결과(정규화 상태).
                shape: (B, Pnn, T, 4)
            near_current_xyyaw_for_feasible: 현재 상태(정규화).
                shape: (B, Pnn, 4)
            batch_size: B
            predicted_neighbor_num: Pnn

        Returns:
            x_out:
                최종 출력(flat).
                shape: (B, Pnn, F_out)
        """
        if self.training or getattr(self.config, "use_direct_loss", False):
            return x

        B: int = batch_size
        Pnn: int = predicted_neighbor_num

        if self.config.use_past_dit_input:
            # x의 과거+현재 부분만 유지한 뒤, 미래를 integrated_trajectory로 교체
            x_past_cur = (
                x.reshape(B, Pnn, -1,
                          4).contiguous()[:, :, :(self.config.past_len + 1), :]
            )  # (B, Pnn, past_len+1, 4)

            x_new = torch.cat(
                [x_past_cur, integrated_trajectory],
                dim=2,
            )  # (B, Pnn, past_len+1+T, 4)

            return x_new.reshape(B, Pnn, -1)

        if self.config.use_current_input:
            x_new = torch.cat(
                [
                    near_current_xyyaw_for_feasible.unsqueeze(2),  # (B,Pnn,1,4)
                    integrated_trajectory,  # (B,Pnn,T,4)
                ],
                dim=2,
            )  # (B,Pnn,1+T,4)
            return x_new.reshape(B, Pnn, -1)

        # use_current_input=False: 미래만 반환
        return integrated_trajectory.reshape(B, Pnn, -1)

    @property
    def model_type(self):
        return self._model_type

    def _run_dit_core_with_pram_v2(
            self,
            near_input_norm_xT: torch.Tensor,  # (B, Pnn, _*4)
            diffusion_time: torch.Tensor,  # (B,)
            cross_c: torch.Tensor,  # (B, token_num, D)
            ego_fut_global: torch.Tensor,  # (B, D)
            near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
            cross_mask: torch.Tensor,  # (B, token_num)
            route_known_mask: torch.Tensor,  # (B, Pnn)
            near_current: torch.Tensor,  # (B, Pnn, 11)
            near_current_mask: torch.Tensor,  # (B, Pnn)
    ) -> torch.Tensor:
        """DiT 본체(프리프로젝션 + PRAM-v2 블록 + 최종 투영)를 한 번 수행한다.

        - 현재 유효 에이전트만 pre-proj MLP 를 통과시킨 뒤
        - PRAM-v2 composer/time 모듈레이션으로 각 블록에 스케일/시프트를 주고
        - 모든 블록을 통과한 후 최종 LayerNorm+Linear 로 (T*4) 차원 출력을 만든다.

        Args:
            near_input_norm_xT (torch.Tensor):
                정규화된  궤적(flatten 형태).
                shape: (B, Pnn, F)
            diffusion_time (torch.Tensor):
                확산 시간 t.
                shape: (B,)
            cross_c (torch.Tensor):
                장면 토큰.
                shape: (B, token_num, D)
            ego_fut_global (torch.Tensor):
                ego 미래 요약 벡터.
                shape: (B, D)
            near_agents_route_lane_emb (torch.Tensor):
                이웃+경로 임베딩.
                shape: (B, Pnn, D)
            cross_mask (torch.Tensor):
                cross 토큰 마스크(True=패딩).
                shape: (B, token_num)
            route_known_mask (torch.Tensor):
                경로 유효 여부(True=known).
                shape: (B, Pnn)
            near_current (torch.Tensor):
                현재 프레임 (x, y, cos, sin).
                shape: (B, Pnn, 11)
            near_current_mask (torch.Tensor):
                현재 프레임 기준 무효 에이전트 마스크(True=패딩).
                shape: (B, Pnn)

        Returns:
            torch.Tensor:
                DiT 본체 출력 (아직 model_type 보정 전).
                shape: (B, Pnn, F_out)  (F_out=T*4 또는 (1+T)*4)
        """
        B, Pnn, _ = near_input_norm_xT.shape  # (B, Pnn, F)

        # 1) pre-proj (varlen)
        # x: (B, Pnn, H)
        x: torch.Tensor = self.preproj_varlen(
            near_input_norm_xT=near_input_norm_xT,
            near_current_mask=near_current_mask,
        )
        # 무효 토큰은 0으로 고정
        x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)  # (B, Pnn, H)

        # 2) timestep embedding
        # t_embedding: (B, H)
        t_embedding: torch.Tensor = self.t_embedder(diffusion_time)
        t_embedding = t_embedding.to(x.dtype)
        ego_fut_global = ego_fut_global.to(x.dtype)

        # 3) state_token_in 준비 (현재 프레임 기반)
        # state_token_in: (B, Pnn, H)
        state_token_in: torch.Tensor = self.pram_v2_state_token_encoder(
            near_cur_norm=near_current.to(x.dtype),  # (B, Pnn, 11)
            near_current_mask=near_current_mask,  # (B, Pnn)
        )

        # 4) PRAM-v2 composer + time modulation
        composer_out = self.pram_v2_composer(
            state_token_in=state_token_in,  # (B, Pnn, H)
            ego_fut_global=ego_fut_global,  # (B, H)
            near_agents_route_lane_emb=near_agents_route_lane_emb,  # (B,Pnn,H)
            route_known_mask=route_known_mask,  # (B, Pnn)
        )
        time_out = self.pram_v2_time_mod(t_embedding)  # (B, 1, H) 3종

        # 5) DiT 블록 반복
        for block_index, block in enumerate(self.blocks):
            pram_mods = compute_pram_v2_modulations_for_block(
                composer_out=composer_out,  #
                time_out=time_out,  # (B, 1, H)
                path_scalars=self.pram_v2_block_path_scalars,
                block_index=block_index,
                batch_size=B,
                predicted_neighbor_num=Pnn,
                hidden_dim=x.shape[-1],
            )
            # x: (B, Pnn, H)
            x = block(
                x=x,
                cross_c=cross_c,  # (B, token_num, H)
                pram_v2_modulations=pram_mods,
                near_current_mask=near_current_mask,  # (B, Pnn)
                cross_mask=cross_mask,  # (B, token_num)
            )
            x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)

        # 6) feasible 에서 사용할 최종 hidden 저장
        if self.config.feasible_grad_to_dit:
            self.final_hidden_tokens = x.float()  # (B, Pnn, H)
        else:
            self.final_hidden_tokens = x.detach().clone().float()  # (B, Pnn, H)

        # 7) PRAM-v2 최종 레이어 (LN + MLP + 보정)
        # x: (B, Pnn, F_out)
        x = apply_pram_v2_final_layer(
            x=x,
            composer_out=composer_out,
            time_out=time_out,
            final_norm=self.pram_v2_final_norm,
            out_proj=self.pram_v2_out_proj,
            final_scalars=(
                self.pram_v2_final_scale_scalar,
                self.pram_v2_final_shift_scalar,
            ),
        )

        # 마스크 위치는 다시 0으로 정리
        x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)
        return x

    def _forward_score_branch(
            self,
            x: torch.Tensor,  # (B, Pnn, F_out)
            diffusion_time: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """model_type 이 'score' 인 경우 출력 텐서를 만드는 함수.

        Args:
            x (torch.Tensor):
                DiT 본체 출력.
                shape: (B, Pnn, F_out)
            diffusion_time (torch.Tensor):
                확산 시간 t.
                shape: (B,)

        Returns:
            torch.Tensor:
                score(x_t) 추정값.
                shape: (B, Pnn, F_out)
        """
        # std: (B, 1, 1)  FP32
        std: torch.Tensor = self.marginal_prob_std(diffusion_time).float()[:,
                                                                           None,
                                                                           None]
        out: torch.Tensor = (x.float() / (std + 1e-6)).to(x.dtype)
        return out

    def _forward_x_start_branch(
        self,
        x: torch.Tensor,  # (B, Pnn, F_out)
        diffusion_time: torch.Tensor,  # (B,)
        near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        near_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, 1+past_len+future_len)
        near_past: torch.Tensor,  # (B, Pnn, past_len, 11)
        near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
        B: int,
        Pnn: int,
    ) -> torch.Tensor:
        """model_type 이 'x_start' 인 경우 출력 텐서를 만드는 함수."""
        # 기본 경로: feasible 을 쓰지 않을 때
        if not getattr(self.config, "use_feasible", False):
            return x

        # 1) feasible에 넣을 텐서 준비(그래디언트 유지/차단 포함)
        (
            x_for_feasible,  # (B,Pnn,F_out) float32
            near_current_xyyaw_for_feasible,  # (B,Pnn,4) float32
            near_past_cur_future_valid_for_feasible,  # (B,Pnn,time_len_total)
        ) = self._select_tensors_for_feasible_projection(
            x=x,
            near_current_xyyaw=near_current_xyyaw,
            near_past_cur_future_valid=near_past_cur_future_valid,
        )

        # 2) (현재+미래) 궤적 텐서 만들기: (B,Pnn,1+T,4)
        diffusion_trajectory: torch.Tensor = self._build_diffusion_trajectory_for_feasible_projection(
            x_for_feasible=x_for_feasible,
            near_current_xyyaw_for_feasible=near_current_xyyaw_for_feasible,
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )

        # 3) low-t 마스크 계산
        low_t_mask: torch.Tensor = self._compute_feasible_low_t_mask(
            diffusion_time=diffusion_time,  # (B,)
        )  # (B,)

        # 4) FeasibleProjector 실행(dit_returns 갱신)
        self._feasible_projection(
            diffusion_trajectory=diffusion_trajectory,  # (B,Pnn,1+T,4)
            near_class_one_hot=near_class_one_hot,  # (B,Pnn,3)
            near_past_cur_future_valid=near_past_cur_future_valid_for_feasible,
            near_past=near_past,  # (B,Pnn,past_len,11)
            feasible_low_t_mask=low_t_mask,  # (B,)
        )

        # 5) 평가 모드에서는 integrated_trajectory로 출력 교체(옵션, 기존 로직 유지)
        integrated_trajectory: torch.Tensor = self.dit_returns.integrated_trajectory  # (B,Pnn,T,4)
        x_out: torch.Tensor = self._maybe_replace_x_with_integrated_trajectory_for_eval(
            x=x,
            integrated_trajectory=integrated_trajectory,
            near_current_xyyaw_for_feasible=near_current_xyyaw_for_feasible,
            batch_size=B,
            predicted_neighbor_num=Pnn,
        )
        return x_out

    def _compute_near_current_valid_and_mask(
        self,
        near_past_cur_future_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거+현재+미래 유효 마스크에서 현재 프레임 기준 마스크를 만든다.

        Args:
            near_past_cur_future_valid (torch.Tensor):
                과거~미래 전체 유효 마스크.
                shape: (B, Pnn, 1+past_len+future_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - near_cur_future_valid:
                    현재+미래 부분 유효 마스크.
                    shape: (B, Pnn, 1+future_len)
                - near_current_mask:
                    현재 프레임 기준 무효 에이전트 마스크(True=패딩).
                    shape: (B, Pnn)
        """
        # near_cur_future_valid: (B, Pnn, 1+T)
        near_cur_future_valid: torch.Tensor = near_past_cur_future_valid[:, :, -(
            self._future_len + 1):]
        # near_current_valid: (B, Pnn)  True=유효
        near_current_valid: torch.Tensor = near_cur_future_valid[:, :, 0]
        # near_current_mask: (B, Pnn)  True=무효(패딩)
        near_current_mask: torch.Tensor = ~near_current_valid
        return near_cur_future_valid, near_current_mask

    def forward(
            self,
            near_input_norm_xT: torch.
        Tensor,  # (B, Pnn, (time_len+ T) *4) or (B, Pnn, T*4) or (B, Pnn, (1+T)*4)
            near_agents_past: torch.
        Tensor,  # (B, Pnn, time_len(=past_len+1), 11)
            diffusion_time: torch.Tensor,  # (B,)
            cross_c: torch.Tensor,  # (B, token_num, D)
            ego_fut_global: torch.Tensor,  # (B, D)
            near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
            near_past_cur_future_valid: torch.Tensor,
            # (B, Pnn, 1+past_len+future_len) bool
            cross_mask: torch.Tensor,  # (B, token_num)
            route_known_mask: torch.Tensor,  # (B, Pnn)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
            near_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
            near_past: torch.Tensor,  # (B, Pnn, past_len, 11)
    ) -> torch.Tensor:
        """DiT 전체 forward 를 수행하는 진입점.

        순서 개요:
            1) 과거+현재+미래 유효 마스크에서 현재 에이전트 마스크를 만든다.
            2) pre-proj + PRAM-v2 블록을 통과해 기본 출력 x 를 만든다.
            3) model_type 에 따라 "score" 또는 "x_start" 분기 처리한다.
               - "x_start" 인 경우, use_feasible 설정에 따라
                 FeasibleProjector 기반 보정을 추가로 수행한다.

        Args:
            near_input_norm_xT (torch.Tensor):
                정규화된 현재+미래 입력(flatten).
                shape: (B, Pnn, F)
            diffusion_time (torch.Tensor):
                확산 시간 t.
                shape: (B,)
            cross_c (torch.Tensor):
                장면 토큰.
                shape: (B, token_num, D)
            ego_fut_global (torch.Tensor):
                ego 미래 요약 벡터.
                shape: (B, D)
            near_agents_route_lane_emb (torch.Tensor):
                이웃+경로 임베딩.
                shape: (B, Pnn, D)
            near_past_cur_future_valid (torch.Tensor):
                과거~현재~미래 유효 마스크.
                shape: (B, Pnn, 1+past_len+future_len)
            cross_mask (torch.Tensor):
                cross 토큰 마스크(True=패딩).
                shape: (B, token_num)
            route_known_mask (torch.Tensor):
                경로 유효 여부(True=known).
                shape: (B, Pnn)
            near_class_one_hot (torch.Tensor):
                에이전트 종류 one-hot.
                shape: (B, Pnn, 3)
            near_current_xyyaw (torch.Tensor):
                현재 프레임 상태(x, y, cos, sin).
                shape: (B, Pnn, 4)
            near_past (torch.Tensor):
                과거 궤적.
                shape: (B, Pnn, past_len, 11)

        Returns:
            torch.Tensor:
                model_type 에 따른 최종 출력.
                - "score": score(x_t) (B, Pnn, F)
                - "x_start": x_start 또는 feasible 보정 후 궤적 (B, Pnn, F)
        """
        # 현재+미래 유효 마스크 및 현재 무효 에이전트 마스크
        """
        near_cur_future_valid :  (B, Pnn, 1+future_len)
        near_current_mask :  (B, Pnn)
        """
        near_cur_future_valid, near_current_mask = \
            self._compute_near_current_valid_and_mask(
                near_past_cur_future_valid=near_past_cur_future_valid
            )
        B, Pnn, _ = near_input_norm_xT.shape  # (B, Pnn, F)
        # time_len(=past_len+1)
        near_agents_past_xyyaw = near_agents_past[:, :, :, :4]
        near_current = near_agents_past[:, :, -1, :]  # (B, Pnn, 11)
        device_type: str = near_input_norm_xT.device.type

        # DiT 본체(프리프로젝션+블록+최종 투영)를 프로파일링 블록 안에서 수행
        with profile_block(
                "DiT.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            # x: (B, Pnn, F_out) # (F_out=T*4 또는 (1+T)*4 또는 (past_len+1+T)*4)
            x: torch.Tensor = self._run_dit_core_with_pram_v2(
                near_input_norm_xT=near_input_norm_xT,
                diffusion_time=diffusion_time,
                cross_c=cross_c,
                ego_fut_global=ego_fut_global,
                near_agents_route_lane_emb=near_agents_route_lane_emb,
                cross_mask=cross_mask,
                route_known_mask=route_known_mask,
                near_current=near_current,
                near_current_mask=near_current_mask,
            )

        # model_type 분기
        if self._model_type == "score":
            return self._forward_score_branch(
                x=x,
                diffusion_time=diffusion_time,
            )
        elif self._model_type == "x_start":
            return self._forward_x_start_branch(
                x=x,
                diffusion_time=diffusion_time,
                near_current_xyyaw=near_current_xyyaw,
                near_past_cur_future_valid=near_past_cur_future_valid,
                near_past=near_past,
                near_class_one_hot=near_class_one_hot,
                B=B,
                Pnn=Pnn,
            )
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
