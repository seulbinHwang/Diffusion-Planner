from functools import partial
from typing import Callable
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from typing import Optional, Dict, Tuple, Any
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, \
    StateNormalizer
from flash_attn.bert_padding import unpad_input, pad_input
# FlashAttention-2 varlen qkvpacked (fallback 로직 없음: 없으면 즉시 에러)
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except Exception as e:
    raise ImportError(
        "flash_attn_varlen_qkvpacked_func import 실패. "
        "flash-attn(FlashAttention-2)이 설치/빌드되어 있어야 합니다."
    ) from e
# decoder.py 상단 import 섹션에 추가
from diffusion_planner.model.module.pram_wosac import (
    PRAMV2Composer,
    PRAMV2TimeModulator,
    PRAMV2BlockPathScalars,
    PRAMV2StateTokenEncoder,
    ModulationTriplet,
    apply_pram_v2_final_layer,  # ← 9단계 마무리 보정 호출용(스켈레톤이어도 OK)
)
from typing import Tuple, Optional
from diffusion_planner.model.module.feasible import FeasibleProjector
from diffusion_planner.loss import AMP_DTYPE
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


# decoder.py 또는 feasible.py 상단 import 근처에 추가
import time
from contextlib import contextmanager
from typing import Iterator
import torch
from typing import Tuple
import math
import torch.nn.functional as F
from diffusion_planner.model.module.dit import TimestepEmbedder

# ---------------------------
# (DEBUG) 같은 경고를 너무 많이 찍지 않기 위한 "한 번만 출력" 헬퍼
# ---------------------------
from typing import Dict


def _infer_fast_compute_dtype(reference_tensor: torch.Tensor) -> torch.dtype:
    """모델이 실제로 계산할 때 쓸 가능성이 큰 dtype을 고릅니다.

    목적:
        - cross_c 같은 큰 텐서를 매번 fp32 -> bf16/fp16 로 복사하는 일을 줄입니다.
        - 이미 같은 dtype이면 복사가 일어나지 않도록 합니다.

    Args:
        reference_tensor (torch.Tensor): device 확인용 기준 텐서. shape: 임의

    Returns:
        torch.dtype: 계산에 쓸 dtype
    """
    # GPU에서 "작은 dtype으로 계산하는 모드"가 켜져 있으면 그 dtype을 따릅니다.
    if reference_tensor.is_cuda and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_gpu_dtype()
        except Exception:
            return reference_tensor.dtype
    return reference_tensor.dtype


def _prepare_bool_mask_on_device(
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """마스크를 bool + 지정 device로 정리합니다.

    Args:
        mask (torch.Tensor): (....) shape, bool 또는 0/1 float/int 가능
        device (torch.device): 올릴 device

    Returns:
        torch.Tensor: bool 마스크, device는 지정값
    """
    m = _to_bool_mask(mask)
    if m.device != device:
        m = m.to(device=device)
    return m

def _prepare_cross_pos_2d_for_dit(
    cross_pos_2d: torch.Tensor,  # (B, token_num, 9)
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    """scene token의 (x,y,...) 정보를 DiT에서 쓰기 좋게 정리합니다.

    목표
    ----
    - device를 reference_tensor와 동일하게 맞춥니다.
    - dtype은 "모델이 실제로 계산할 dtype"(대개 bf16/fp16)을 따르도록 하여,
      큰 텐서(cross_pos_2d)가 불필요하게 float32로 커지는 것을 피합니다.

    Args:
        cross_pos_2d:
            shape: (B, token_num, 9)
        reference_tensor:
            dtype/device 기준 텐서. shape: 임의

    Returns:
        torch.Tensor:
            shape: (B, token_num, 9)
            device: reference_tensor.device
            dtype: (가능하면) bf16/fp16, 아니면 reference_tensor.dtype
    """
    if cross_pos_2d.device != reference_tensor.device:
        cross_pos_2d = cross_pos_2d.to(device=reference_tensor.device)

    dtype = _infer_fast_compute_dtype(reference_tensor)
    if cross_pos_2d.dtype != dtype:
        cross_pos_2d = cross_pos_2d.to(dtype=dtype)

    return cross_pos_2d

def _prepare_cross_inputs_for_dit(
    cross_c: torch.Tensor,  # (B, token_num, D)
    cross_mask: torch.Tensor,  # (B, token_num) bool/0-1
    reference_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DiT에 넣기 전에 cross 입력(인코더 토큰/마스크)을 미리 정리합니다.

    목표:
        - DiT 내부에서 cross_c를 다시 dtype/device 변환하지 않게(복사 없게) 만들기
        - cross_mask를 bool + 같은 device로 맞추기

    Args:
        cross_c (torch.Tensor): (B, token_num, D)
        cross_mask (torch.Tensor): (B, token_num) bool 또는 0/1
        reference_tensor (torch.Tensor): device/dtype 기준 텐서. shape: 임의

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            cross_c_out: (B, token_num, D)  reference_tensor와 같은 device + (가능하면) 계산 dtype
            cross_mask_out: (B, token_num) bool + 같은 device
    """
    device = reference_tensor.device
    dtype = _infer_fast_compute_dtype(reference_tensor)

    # 큰 텐서는 "필요할 때만" 변환(이미 동일하면 복사 없음)
    if cross_c.device != device or cross_c.dtype != dtype:
        cross_c = cross_c.to(device=device, dtype=dtype)

    cross_mask = _prepare_bool_mask_on_device(cross_mask, device=device)
    return cross_c, cross_mask


def _prepare_diffusion_time_for_dit(
        diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        reference_tensor: torch.Tensor,  # device 기준
) -> torch.Tensor:
    """diffusion_time을 DiT가 바로 쓰기 좋은 형태(device + float32)로 맞춥니다.

    Args:
        diffusion_time (torch.Tensor): (B,) 또는 (B, future_len)
        reference_tensor (torch.Tensor): device 기준 텐서. shape: 임의

    Returns:
        torch.Tensor: diffusion_time_f32 (원 shape 유지), device는 reference_tensor.device, dtype=float32
    """
    if diffusion_time.device != reference_tensor.device or diffusion_time.dtype != torch.float32:
        diffusion_time = diffusion_time.to(device=reference_tensor.device,
                                           dtype=torch.float32)
    return diffusion_time


def _ensure_tensor_on_ref(noise: torch.Tensor,
                          ref: torch.Tensor) -> torch.Tensor:
    """noise를 ref와 같은 device 맞춥니다."""
    return noise.to(device=ref.device)


def _to_bool_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """입력 마스크를 bool 텐서로 바꿉니다.

    이 함수는 "마스크 의미(예: True=유효, True=패딩)"를 바꾸지 않습니다.
    단지 dtype만 bool로 통일합니다.

    Args:
        mask (torch.Tensor):
            입력 마스크 텐서.
            - shape: 임의
            - dtype: bool 또는 0/1 형태의 int/float 텐서일 수 있습니다.
        threshold (float):
            float 마스크일 때 True로 볼 기준값.
            기본은 0.5 입니다.

    Returns:
        torch.Tensor:
            bool 마스크 텐서.
            shape: 입력과 동일
    """
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return mask > float(threshold)
    return mask != 0


import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch

# name -> 호출 횟수 / 평균 계산에 포함된 횟수 / 평균 누적(ms)
_PROFILE_TOTAL_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_TOTAL_MS: Dict[str, float] = {}

# 각 name별로 처음 N번 호출은 avg 계산에서 제외
_PROFILE_WARMUP_CALLS: int = 30


@contextmanager
def profile_block(
    name: str,
    enabled: bool = True,
    device_type: str = "cuda",
) -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력하고, 평균(avg)은 워밍업 이후만 누적합니다.

    동작:
        - name(문자열)별로 호출 횟수를 셉니다.
        - 각 name에서 처음 _PROFILE_WARMUP_CALLS번 호출은 avg 누적에서 제외합니다.
        - 그 다음 호출부터 avg_ms = (워밍업 제외 누적 시간) / (워밍업 제외 호출 횟수) 로 출력합니다.

    Args:
        name (str): 출력/누적의 키로 쓸 블록 이름.
        enabled (bool): False이면 계측 없이 그대로 실행합니다.
        device_type (str): "cuda"면 GPU 작업까지 포함해 재기 위해 앞/뒤로 synchronize 합니다.

    Yields:
        None
    """
    if not enabled:
        yield
        return

    is_cuda: bool = isinstance(device_type,
                               str) and device_type.startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0: float = time.perf_counter()
    yield

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms: float = (time.perf_counter() - t0) * 1000.0

    # (1) 전체 호출 카운트(워밍업 포함)
    total_prev: int = _PROFILE_TOTAL_CALL_COUNT.get(name, 0)
    total_now: int = total_prev + 1
    _PROFILE_TOTAL_CALL_COUNT[name] = total_now

    # (2) avg 누적(워밍업 제외)
    avg_cnt_prev: int = _PROFILE_AVG_CALL_COUNT.get(name, 0)
    avg_sum_prev: float = _PROFILE_AVG_TOTAL_MS.get(name, 0.0)

    if total_now > int(_PROFILE_WARMUP_CALLS):
        avg_cnt_now: int = avg_cnt_prev + 1
        avg_sum_now: float = avg_sum_prev + float(elapsed_ms)
        _PROFILE_AVG_CALL_COUNT[name] = avg_cnt_now
        _PROFILE_AVG_TOTAL_MS[name] = avg_sum_now
    else:
        avg_cnt_now = avg_cnt_prev
        avg_sum_now = avg_sum_prev

    avg_ms: float = (avg_sum_now /
                     float(avg_cnt_now)) if avg_cnt_now > 0 else 0.0
    warmup_left: int = max(0, int(_PROFILE_WARMUP_CALLS) - total_now)

    print(
        f"[PROFILE] {name}: {elapsed_ms:.3f} ms | avg {avg_ms:.3f} ms | n={avg_cnt_now} | warmup_left={warmup_left}"
    )


def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """ref 텐서의 dtype/device로 x를 맞춥니다(이미 같으면 그대로 반환).

    Args:
        x (torch.Tensor): 입력 텐서. shape: 임의
        ref (torch.Tensor): 기준 텐서. shape: 임의

    Returns:
        torch.Tensor: x를 ref와 같은 dtype/device로 맞춘 텐서.
    """
    if x.device == ref.device and x.dtype == ref.dtype:
        return x
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
            if self.config.pose_based:
                output_dim = (config.time_len +
                              config.future_len) * 4  # x, y, cos, sin
            else:
                output_dim = (config.time_len - 1 + config.future_len) * 3
        else:
            if self.config.pose_based:
                if self.config.use_current_input:
                    output_dim = (config.future_len + 1) * 4  # x, y, cos, sin
                else:
                    output_dim = (config.future_len) * 4  # x, y, cos, sin
            else:
                output_dim = (config.future_len) * 3
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
        # = <diffusion_planner/model/guidance/guidance_wrapper.py> 의 GuidanceWrapper 인스턴스가 들어옴
        self._guidance_fn = getattr(config, "guidance_fn", None)
        self._amortized_buffer = None  # (B, 1+Pnn, T, 4 or 3)
        # ✅ (추가) z를 만들 때 사용한 표준정규 노이즈(eps) 버퍼
        # shape: (B, 1+Pnn, T, 4 or 3)
        self._amortized_buffer_eps = None
        future_len = self.config.future_len  # e.g., 80
        # Future-only amortized schedule: [1/T, 2/T, ..., 1]
        # ✅ [수정] amortized schedule에서 t=1을 정확히 쓰지 않도록 마지막을 1-eps로 제한
        # - config에 값이 있으면 그걸 쓰고, 없으면 1e-3 사용
        t_eps: float = float(getattr(config, "diffusion_time_eps", 1e-3))
        t_max: float = max(1.0 - t_eps, 0.0)
        self.t_tau = torch.arange(
            1, future_len + 1, dtype=torch.float32) / float(future_len)  # (T,)
        self.t_tau = torch.clamp(self.t_tau, max=t_max)  # (T,)

    def _get_amortized_random_noise_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        batch_size: int,
        one_or_Pnn: int,
        target_current_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        """amortized + step_idx>0에서 사용할 랜덤 텐서를 inputs에서 꺼냅니다.

        이 랜덤은 Decoder가 기존에 torch.randn_like(...)로 만들던 값(표준정규)을
        바깥(eval 코드)에서 cand_idx 기반 seed로 만든 뒤 전달받는 용도입니다.

        Args:
            inputs (Dict[str, torch.Tensor]):
                모델 입력 dict.
                - amortized_random_noise: (B, (1+)Pnn, future_len, 4 or 3)
            batch_size (int):
                B
            one_or_Pnn (int):
                (1+Pnn)
            target_current_xyyaw (torch.Tensor):
                dtype/device 기준 텐서.
                shape: (B, (1+)Pnn, 4)

        Returns:
            torch.Tensor:
                amortized random noise 텐서.
                shape: (B, (1+)Pnn, future_len, 4 or 3)
        """
        noise = inputs.get("amortized_random_noise", None)
        if noise is None:
            raise KeyError(
                "amortized(step_idx>0) 추론에는 inputs['amortized_random_noise']가 필요합니다. "
                "eval 코드에서 cand_idx 기반 seed로 만든 텐서를 넣어 주세요.")
        if not isinstance(noise, torch.Tensor):
            raise TypeError(
                "inputs['amortized_random_noise']는 torch.Tensor여야 합니다.")
        if self.config.pose_based:
            expected_dim = 4
        else:
            expected_dim = 3
        expected = (int(batch_size), int(one_or_Pnn), int(self._future_len),
                    expected_dim)
        if tuple(noise.shape) != expected:
            raise ValueError(
                "inputs['amortized_random_noise'] shape가 예상과 다릅니다. "
                f"expected={expected}, got={tuple(noise.shape)}")

        return _ensure_tensor_on_ref(noise, target_current_xyyaw)

    def _get_inference_noise_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        batch_size: int,
        one_or_Pnn: int,
        target_current_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        """추론 시작에 사용할 noise를 inputs에서 꺼냅니다.

        규칙
        ----
        - Decoder는 더 이상 내부에서 noise를 만들지 않습니다.
        - 반드시 inputs["inference_noise"]가 있어야 합니다.

        Args:
            inputs (Dict[str, torch.Tensor]):
                모델 입력 dict.
                - inference_noise: shape (B, (1+)Pnn, future_len, 4 or 3)
            batch_size (int):
                B. shape: ()
            one_or_Pnn (int):
                (1+Pnn). shape: ()
            target_current_xyyaw (torch.Tensor):
                dtype/device 기준 텐서.
                shape: (B, (1+)Pnn, 4)

        Returns:
            torch.Tensor:
                inference noise 텐서.
                shape: (B, (1+)Pnn, future_len, 4)
        """
        noise = inputs.get("inference_noise", None)
        if noise is None:
            raise KeyError("Decoder 추론에는 inputs['inference_noise']가 필요합니다. "
                           "eval 코드에서 (rollout_idx 기준으로 만든 noise)를 넣어 주세요.")
        if not isinstance(noise, torch.Tensor):
            raise TypeError("inputs['inference_noise']는 torch.Tensor여야 합니다.")
        if self.config.pose_based:
            expected_dim = 4
        else:
            expected_dim = 3
        expected = (int(batch_size), int(one_or_Pnn), int(self._future_len),
                    expected_dim)
        if tuple(noise.shape) != expected:
            raise ValueError("inputs['inference_noise'] shape가 예상과 다릅니다. "
                             f"expected={expected}, got={tuple(noise.shape)}")

        return _ensure_tensor_on_ref(noise, target_current_xyyaw)

    @property
    def sde(self):
        return self._sde

    def _build_training_dit_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        target_seq_past: torch.
        Tensor,  # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
    ) -> torch.Tensor:
        """훈련 모드에서 DiT에 넣을 입력(xT_input_flat)과 현재 프레임(정규화)을 만든다.


        Args:
            inputs:
                - "target_seq_norm_xT":
                    - (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
                - target_seq_past: (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
        Returns:
            xT_input_flat:
                - use_past_dit_input=True  -> (B, (1+)Pnn, (time_len+future_len)*4) or (B, (1+)Pnn, (past_len + future_len)*3)
                - use_current_input=True   -> (B, (1+)Pnn, (1+future_len)*4) or (B, (1+)Pnn, (future_len)*3)
                - use_current_input=False  -> (B, (1+)Pnn, (future_len)*4) or (B, (1+)Pnn, (future_len)*3)
        """
        B, one_or_Pnn = target_seq_past.shape[:2]
        # target_seq_norm_xT: (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
        target_seq_norm_xT: torch.Tensor = inputs["target_seq_norm_xT"]
        # (B, (1+)Pnn, time_len, 4)

        #  (B, (1+)Pnn, future_len, 4 or 3)
        target_future_norm_xT: torch.Tensor = target_seq_norm_xT[:, :,
                                                                 -self.config.
                                                                 future_len:, :]
        # dtype/device 정합: concat 전에 맞춰두는 게 안전
        target_seq_past = _cast_like(target_seq_past, target_future_norm_xT)
        # 현재 프레임(정규화): 과거~현재 시퀀스의 마지막 프레임을 기준으로 잡아 정합성 강화
        # xT_input_seq 구성
        if self.config.use_past_dit_input:

            # xT_input_seq: (B, (1+)Pnn, (time_len + future_len, 4) or (past_len + future_len)*3)
            # target_seq_past: (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
            # target_future_norm_xT : (B, (1+)Pnn, future_len, 4 or 3)
            xT_input_seq: torch.Tensor = torch.cat(
                [target_seq_past, target_future_norm_xT],
                dim=2,
            )
        else:
            if self.config.use_current_input:
                # (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
                xT_input_seq = target_seq_norm_xT
            else:
                # (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
                xT_input_seq = target_future_norm_xT

        # flatten: (B, (1+)Pnn, F)
        xT_input_flat: torch.Tensor = xT_input_seq.reshape(B, one_or_Pnn, -1)

        return xT_input_flat

    def _extract_future_from_diffusion_output(
        self,
        diffusion_output: torch.Tensor,
        # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        batch_size: int,
        one_or_Pnn: int,
    ) -> torch.Tensor:  #(B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
        """훈련 모드에서 DiT 출력(flat)을 (B, Pnn, T, 4 or 3) 형태로 바꾼다.

        Args:
            diffusion_output / xT_input_flat
            if pose_based
                (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            else
                (B, (1+)Pnn, (past_len + T) *4) or (B, (1+)Pnn, T*4)

        Returns:
            diffusion_future_output:
                shape: (B,(1+)Pnn, T, 4) or (B, (1+)Pnn, T, 3)
        """
        B: int = batch_size
        if self.config.pose_based:
            last_dim = 4
        else:
            last_dim = 3
        diffusion_output = diffusion_output.reshape(B, one_or_Pnn, -1, last_dim)
        # 마지막 (1+T)만 사용: (B, Pnn, T, 4 or 3)
        diffusion_future_output = diffusion_output[:, :,
                                                   -(self._future_len):, :]
        return diffusion_future_output

    def _append_training_feasible_outputs(
            self,
            decoder_output_dict: Dict[str, torch.Tensor],
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
    ) -> None:
        """훈련 모드에서 feasible 출력(integrated_trajectory 등)을 outputs dict에 추가한다.

        Returns:
            None
        """
        if not self.config.use_feasible:
            return

        # integrated_trajectory: (B, (1+)Pnn, T, 4)
        integrated_trajectory: torch.Tensor = self.dit.norm_dit_returns.integrated_trajectory
        integrated_trajectory = torch.cat(
            [target_current_xyyaw.unsqueeze(2), integrated_trajectory],
            dim=2,
        )  # (B, (1+)Pnn, 1+T, 4)

        decoder_output_dict["integrated_trajectory"] = integrated_trajectory
        decoder_output_dict[
            "control_constraint_diff"] = self.dit.norm_dit_returns.control_constraint_diff  # (B, (1+)Pnn, T, 3)
        if not bool(self.config.pose_based):
            control_sequence = getattr(self.dit.norm_dit_returns,
                                       "control_sequence", None)
            if isinstance(control_sequence, torch.Tensor):
                # dtype/device는 integrated_trajectory와 맞춰 둠(불필요한 변환 방지)
                decoder_output_dict["control_sequence"] = _cast_like(
                    control_sequence, integrated_trajectory)

    def _inpainting_past_cur_and_reshape(
        self,
        xt: torch.Tensor,  # shape: (B, Pnn, flattened_dim)
        batch_size: int,
        one_or_Pnn: int,
        target_seq_past: torch.Tensor,
        # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
    ) -> torch.Tensor:
        """샘플 xt 를 (B, Pnn, _, 4) 모양으로 펼치고
        '과거~현재 상태 프레임'을 맨 앞에 올바르게 넣어주는 메서드.
        """
        if self.config.pose_based:
            last_dim = 4
        else:
            last_dim = 3
        past_seq_len = target_seq_past.shape[
            -2]  # time_len or past_len(=time_len-1)
        if self.config.use_past_dit_input:
            # xt_reshaped: (B, Pnn, time_len+T, 4) or (B, Pnn, past_len+T, 3)
            xt_reshaped = xt.reshape(batch_size, one_or_Pnn, -1, last_dim)
            # 과거~현재 상태 주입
            xt_reshaped[:, :, :past_seq_len, :] = target_seq_past
        else:
            if self.config.use_current_input:
                if self.config.pose_based:
                    # xt_reshaped: (B, Pnn, 1+T, 4)
                    xt_reshaped = xt.reshape(
                        batch_size,
                        one_or_Pnn,
                        1 + self._future_len,
                        -1,
                    )
                    target_current_xyyaw = target_seq_past[:, :,
                                                           -1, :]  # (B,Pnn,4)
                    xt_reshaped[:, :,
                                0, :] = target_current_xyyaw  # 현재 상태 주입 (B,Pnn,4)
                else:
                    # xt_reshaped: (B, Pnn, T, 3)
                    xt_reshaped = xt.reshape(
                        batch_size,
                        one_or_Pnn,
                        self._future_len,
                        -1,
                    )
            else:
                # xt_reshaped: (B, Pnn, T, 4 or 3)
                xt_reshaped = xt.reshape(
                    batch_size,
                    one_or_Pnn,
                    self._future_len,
                    -1,
                )
        return xt_reshaped

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
            diffusion_time_step: torch.Tensor,  # shape: (B) or (B, future_len)
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

        # diffusion_time_step: t (shape: (B)
        t_value: torch.Tensor = diffusion_time_step.detach().float()
        if t_value.dim() == 1:
            # 여러 값이 들어와도 첫 번째 값만 사용 (배치 전체가 같은 t 를 쓰기 때문)
            t_value = t_value.view(-1)[0]  # shape: ()
            # s(t) = clip( (t_th - t) / t_th, 0, 1 )
            safe_t_threshold: float = max(t_threshold, 1e-6)
            s_raw: torch.Tensor = (t_threshold - t_value) / safe_t_threshold
            s_clamped: torch.Tensor = torch.clamp(s_raw, 0.0, 1.0)

            beta_value: torch.Tensor = beta_max * (s_clamped**beta_power
                                                  )  # shape: ()
            beta_value = beta_value.to(
                dtype=reference_tensor_for_device.dtype,
                device=reference_tensor_for_device.device,
            )
        elif t_value.dim() == 2:
            # beta_value : torch Tensor filled with zero. shape = ()
            beta_value = reference_tensor_for_device.new_tensor(
                0.0)  # shape: ()
        else:
            raise ValueError("diffusion_time_step의 차원 수가 예상과 다릅니다.")

        return beta_value

    def _get_latest_feasible_integrated_trajectory_future(
        self,
        batch_size: int,
        one_or_Pnn: int,
        future_len: int,
        reference_tensor_for_device: torch.Tensor,  # shape: (B, Pnn, _, 4)
    ) -> Optional[torch.Tensor]:  # returns: (B, Pnn, future_len, 4) or None
        """DiT가 방금 저장해 둔 FeasibleProjector 적분 결과를 꺼내오는 메서드.

        - 학습/추론 공통으로 self.dit.norm_dit_returns.integrated_trajectory 에 저장된 값을 사용한다.
        - shape 이나 배치 크기가 안 맞으면 None 을 돌려서 prox-snap 을 건너뛴다.
        """
        norm_dit_returns = getattr(self.dit, "norm_dit_returns", None)
        if norm_dit_returns is None:
            return None
        """
        # integrated_trajectory : (B, (1+)Pnn, future_len, 4)
        # control_constraint_diff : (B, (1+)Pnn, future_len, 3)
        """
        integrated_future = getattr(norm_dit_returns, "integrated_trajectory",
                                    None)  # (B, (1+)Pnn, T, 4)
        if integrated_future is None:
            return None
        if integrated_future.dim() != 4 or integrated_future.shape[-1] != 4:
            return None

        B_f, Pnn_f, T_f, _ = integrated_future.shape
        if B_f != batch_size or Pnn_f != one_or_Pnn:
            return None
        if T_f != future_len:
            # future_len 이 다르면 섞을 수 없으니 안전하게 스킵
            return None

        return integrated_future.to(  # shape: (B, (1+)Pnn, future_len, 4)
            dtype=reference_tensor_for_device.dtype,
            device=reference_tensor_for_device.device,
        )

    def _apply_feasible_blend_with_feasible_projection(
            self,
            xt_sequence: torch.Tensor,  # shape: (B, (1+)Pnn, _, 4)
            target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, (1+)Pnn, time_len_total) bool
            feasible_blend_beta_scalar: torch.Tensor,  # shape: ()
    ) -> torch.Tensor:  # shape: (B, Pnn, time_len, 4)
        """현재 샘플 궤적과 FeasibleProjector 적분 궤적을 β 비율로 섞어 주는 메서드.

        동작 요약:
            1) DiT 가 저장해 둔 integrated_trajectory (미래 T 프레임)을 가져온다.
            2) 현재 xt_sequence 에서 '미래 부분'만 잘라서 (B, Pnn, T, 4) 로 맞춘다.
            3) target_past_cur_future_valid 로 유효한 미래 타임스텝만 골라서,
               x_new = (1-β) * x_t + β * Π(x_t) 형태로 섞는다.
        """
        # β==0 이면 아무 것도 하지 않고 바로 반환
        if feasible_blend_beta_scalar.detach().item() <= 0.0:
            return xt_sequence

        if not getattr(self.config, "use_feasible", False):
            return xt_sequence

        batch_size, one_or_Pnn, T_len, _ = xt_sequence.shape
        future_len: int = int(self._future_len)

        # FeasibleProjector 적분 결과 (미래 T 프레임) 가져오기
        # integrated_future: (B, (1+)Pnn, future_len, 4) 또는 None
        integrated_future = self._get_latest_feasible_integrated_trajectory_future(
            batch_size=batch_size,
            one_or_Pnn=one_or_Pnn,
            future_len=future_len,
            reference_tensor_for_device=xt_sequence,  # shape: (B, Pnn, _, 4)
        )
        if integrated_future is None:
            return xt_sequence

        # xt_sequence 에서 "미래" 부분만 (B, (1+)Pnn, future_len, 4) 형태로 뽑기
        if self.config.use_past_dit_input:
            if T_len != self.config.time_len + future_len:
                # time_len 이 다르면 안전하게 스킵
                raise ValueError(
                    "시간 길이가 맞지 않습니다. T_len={}, future_len={}".format(
                        T_len, future_len))
            xt_future = xt_sequence[:, :, self.config.
                                    time_len:, :]  # (B, (1+)Pnn, future_len, 4)
        else:
            if self.config.use_current_input:
                # xt_sequence: (B, (1+)Pnn, 1+future_len, 4)  -> 미래만 사용
                if T_len != future_len + 1:
                    # time_len 이 다르면 안전하게 스킵
                    raise ValueError(
                        "시간 길이가 맞지 않습니다. T_len={}, future_len={}".format(
                            T_len, future_len))
                xt_future = xt_sequence[:, :,
                                        1:, :]  # (B, (1+)Pnn, future_len, 4)
            else:
                # xt_sequence: (B, (1+)Pnn, future_len, 4) 가 바로 미래 궤적
                if T_len != future_len:
                    raise ValueError(
                        "시간 길이가 맞지 않습니다. T_len={}, future_len={}".format(
                            T_len, future_len))
                xt_future = xt_sequence  # (B, (1+)Pnn, future_len, 4)

        # target_past_cur_future_valid 에서 "현재+미래" 마스크만 추출
        if target_past_cur_future_valid.shape[-1] < (future_len + 1):
            return xt_sequence
        target_cur_future_valid = target_past_cur_future_valid[:, :, -(
            future_len + 1):]  # (B, (1+)Pnn, 1+T)
        future_valid = target_cur_future_valid[:, :, 1:]  # (B, (1+)Pnn, T)

        # 유효한 미래 타임스텝만 섞기 위해 마스크 생성
        mix_valid_mask = future_valid.unsqueeze(-1).to(  # (B, (1+)Pnn, T, 1)
            dtype=xt_future.dtype)

        # β 를 (B, Pnn, T, 1) 로 브로드캐스트
        beta_broadcast = feasible_blend_beta_scalar.view(1, 1, 1,
                                                         1)  # (1,1,1,1)
        beta_mask = beta_broadcast * mix_valid_mask  # (B, Pnn, T, 1)

        # 실제로 섞기: x_new = (1-β) * x_t + β * Π(x_t)
        integrated_future = integrated_future.to(dtype=xt_future.dtype,
                                                 device=xt_future.device)
        new_future = (
            1.0 - beta_mask
        ) * xt_future + beta_mask * integrated_future  # (B, (1+)Pnn, T, 4)

        # xt_sequence 에 다시 써 넣기
        if self.config.use_past_dit_input:
            xt_sequence[:, :, self.config.time_len:, :] = new_future
        else:
            if self.config.use_current_input:
                xt_sequence[:, :, 1:, :] = new_future
            else:
                xt_sequence[:, :, :, :] = new_future

        return xt_sequence

    def _prepare_target_trajectories_and_masks(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
    ]:
        """배치 dict에서 이웃 에이전트 관련 공통 텐서를 만든다.
        Args:
            inputs (Dict[str, torch.Tensor]):
                DataLoader에서 넘어온 배치 dict.

        Returns:
            Tuple:
                target_agents_past:
                    (B, (1+)Pnn, time_len, 11)
                target_seq_past:
                    (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
                target_past_cur_future_valid:
                    (B, (1+)Pnn, time_len + future_len) bool (True=유효)
                batch_size (int):
                    B
                predicted_neighbor_num (int):
                    Pnn
        """
        near_agents_past: torch.Tensor = inputs[
            "near_agents_past"]  # (B,Pnn,time_len,11)
        near_agents_past_is_valid = inputs[
            "near_agents_past_is_valid"]  # (B, Pnn, time_len) bool

        # target_agents_past: (B, (1+)Pnn, time_len, 11)
        if self.config.do_ego_predict:
            ego_agent_past: torch.Tensor = inputs[
                "ego_agent_past"]  # (B,time_len,11)
            target_agents_past = torch.cat(
                [
                    ego_agent_past.unsqueeze(1),  # (B,1,time_len,11)
                    near_agents_past
                ],
                dim=1,
            )  # (B, 1+Pnn, time_len, 11)
            ego_agent_past_is_valid = inputs[
                "ego_agent_past_is_valid"]  # (B, time_len) bool

            target_past_current_is_valid = torch.cat(
                [
                    ego_agent_past_is_valid.unsqueeze(1),  # (B,1,time_len)
                    near_agents_past_is_valid
                ],
                dim=1,
            )  # (B, (1+)Pnn, time_len) bool
        else:
            target_agents_past = near_agents_past  # (B, Pnn, time_len, 11)
            target_past_current_is_valid = near_agents_past_is_valid  # (B, Pnn, time_len) bool

        # target_future_valid: (B, (1 +) Pnn, future_len) 또는 None
        target_future_valid: torch.Tensor = inputs["target_future_valid"]
        """ target_past_cur_future_valid : (B, (1+)Pnn, time_len + future_len) bool

        target_past_current_is_valid : (B, (1+)Pnn, time_len) bool
        target_future_valid: (B, (1 +) Pnn, future_len) bool
        """
        target_past_cur_future_valid = torch.cat(
            [target_past_current_is_valid, target_future_valid], dim=-1)
        target_past_cur_future_valid = _to_bool_mask(
            target_past_cur_future_valid).to(device=target_agents_past.device)

        batch_size, one_or_Pnn, _, _ = target_agents_past.shape

        if self.config.pose_based:
            # target_agents_past: (B, (1+)Pnn, time_len, 11)
            #target_seq_past: (B, (1+)Pnn, time_len, 4)
            target_seq_past = target_agents_past[:, :, :, :4]
        else:
            # past_seg_control_gt_3_dim:  (B, (1+)Pnn, past_len, 3) or None
            past_seg_control_gt_3_dim = inputs.get("past_seg_control_gt_3_dim",
                                                   None)
            assert past_seg_control_gt_3_dim is not None, \
                f"past_seg_control_gt_3_dim is None, but config.pose_based={self.config.pose_based}"
            target_seq_past = past_seg_control_gt_3_dim  # (B, (1+)Pnn, past_len, 3)
        return (
            target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            target_seq_past,  # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
            target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
            batch_size,
            one_or_Pnn,
        )

    def _unpack_encoder_outputs(
            self,
            encoder_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """encoder_outputs dict 에서 디코더가 쓰는 텐서들을 꺼낸다.

        Returns:
            scene_encoding_token:
                (B, token_num, D)
            scene_encoding_token_mask:
                (B, token_num)  True=pad
            scene_encoding_pos_2d:
                (B, token_num, 9)
        """
        scene_encoding_token: torch.Tensor = encoder_outputs[
            "encoding"]  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor = encoder_outputs[
            "encoding_mask"]  # (B, token_num) True=pad
        scene_encoding_pos_2d: torch.Tensor = encoder_outputs[
            "encoding_pos_2d"]  # (B, token_num, 9)
        return (
            scene_encoding_token,
            scene_encoding_token_mask,
            scene_encoding_pos_2d,
        )
    def _build_inference_xT_from_noise(
        self,
        noise: torch.Tensor,  # (B,(1+)Pnn,T,4 or 3)
        target_seq_past: torch.Tensor,
        # (B, (1+)Pnn, (time_len, 4) or (past_len, 3))
    ) -> torch.Tensor:
        """샘플링 시작점 xT(flat)를 만든다.

        Returns:
            pose_based=True:
                (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, (1+)Pnn, T*4)
            pose_based=False:
                (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, T*3)
        """
        B, one_or_Pnn = noise.shape[:2]

        if self.config.use_past_dit_input:
            target_seq_past = _cast_like(target_seq_past, noise)

            xT: torch.Tensor = torch.cat(
                [
                    target_seq_past,  # (B, (1+)Pnn, past_len, 4 or 3)
                    noise,  # (B, (1+)Pnn, future_len, 4 or 3)
                ],
                dim=2,
            ).reshape(B, one_or_Pnn, -1)
            return xT

        if self.config.use_current_input:
            if self.config.pose_based:
                # target_current_xyyaw: (B, (1+)Pnn, 4)
                target_current_xyyaw = target_seq_past[:, :, -1, :]
                target_current_xyyaw = _cast_like(target_current_xyyaw, noise)

                xT = torch.cat(
                    [
                        target_current_xyyaw[:, :, None, :],
                        # (B, (1+)Pnn, 1, 4)
                        noise,  # (B, (1+)Pnn, T, 4)
                    ],
                    dim=2,
                ).reshape(B, one_or_Pnn, -1)
                return xT

            # pose_based=False + use_current_input=True: current 프레임이 입력에 따로 없으므로 noise만 사용
            return noise.reshape(B, one_or_Pnn, -1)

        # use_current_input=False: 미래만
        return noise.reshape(B, one_or_Pnn, -1)

    def _mask_invalid_timesteps_to_zero(
            self,
            x_seq: torch.Tensor,  # (B, P, S, 4)
            target_past_cur_future_valid: torch.Tensor,  # (B, P, T_all)
    ) -> torch.Tensor:
        """시간축에서 무효(빈칸)로 표시된 칸을 0으로 정리합니다.

        이 함수가 하는 일
        --------------
        - x_seq는 (B, P, S, 4) 형태의 "현재/미래(또는 과거 포함)" 값입니다.
        - target_past_cur_future_valid는 (B, P, T_all) 형태로, True=유효 / False=무효 입니다.
        - x_seq의 시간 길이 S에 맞춰, valid 마스크의 "뒤쪽 S칸"을 잘라서 적용합니다.
          (x_seq가 '미래만'이거나 '현재+미래'인 경우에도 뒤쪽 구간이 정확히 대응합니다.)

        Args:
            x_seq (torch.Tensor):
                값 텐서.
                shape: (B, P, S, 4)
            target_past_cur_future_valid (torch.Tensor):
                유효 여부 텐서(True=유효, False=무효).
                shape: (B, P, T_all)

        Returns:
            torch.Tensor:
                무효 칸이 0으로 정리된 값 텐서.
                shape: (B, P, S, 4)

        Raises:
            ValueError: shape이 기대와 다를 때
        """
        B, P, S, _ = x_seq.shape
        # 뒤쪽 S칸이 x_seq의 시간축과 대응(현재+미래 / 미래만 / 과거+미래 모두 대응)
        valid_sliced: torch.Tensor = _to_bool_mask(
            target_past_cur_future_valid[..., -int(S):]).to(
                device=x_seq.device, dtype=torch.bool)  # (B,P,S)
        # (B,P,S,4)에서 무효 칸을 0으로
        x_seq_out: torch.Tensor = x_seq.masked_fill(~valid_sliced.unsqueeze(-1),
                                                    0.0)
        return x_seq_out

    def _mask_zero_at_invalid_timestep_in_seq_flat(
            self,
            x_flat: torch.Tensor,
            # (B, (1+)Pnn, (time_len+future_len)*4) or (B, (1+)Pnn, (past_len + future_len)*3)
            target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, time_len + future_len)
    ) -> torch.Tensor:
        """
        # 2) xT(flat) 생성
        pose_based = True
            (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
        pose_based = False
            (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, (T)*3)
        """
        """flat 형태(F=시간*4) 텐서에서 무효 시간 칸을 0으로 정리합니다.

        Args:
            target_past_cur_future_valid (torch.Tensor):
                shape:  # (B, (1+)Pnn, time_len + future_len), True=유효 / False=무효

        Returns:
            torch.Tensor:
                shape: (B, P, F), 무효 칸은 0
        """

        B, P, F = x_flat.shape
        if self.config.pose_based:
            S: int = int(
                F //
                4)  # S: (time_len + future_len) or 1+future_len or future_len
            x_seq = x_flat.reshape(B, P, S, 4)  # (B,P,S,4)
            x_seq = self._mask_invalid_timesteps_to_zero(
                x_seq=x_seq,
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, P, T_all)
            )
        else:
            S: int = int(F // 3)  # S: (past_len + future_len) or future_len
            x_seq = x_flat.reshape(B, P, S, 3)
            # (B, P, T_all)
            # # (B, (1+)Pnn, time_len + future_len), True=유효 / False=무효
            target_past_future_seg_valid = (
                target_past_cur_future_valid[..., :-1] &
                target_past_cur_future_valid[..., 1:]
            )  # (B, P, past_len + future_len)
            x_seq = self._mask_invalid_timesteps_to_zero(
                x_seq=x_seq,
                target_past_cur_future_valid=target_past_future_seg_valid,
            )
        return x_seq.reshape(B, P, F)

    def _initial_state_constraint(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,  # (B) or (B,future_len)
            step: int,
            *,
            batch_size: int,  # DONE
            one_or_Pnn: int,  # DONE
            target_seq_past: torch.Tensor,
            # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
            target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, time_len_total)
    ) -> torch.Tensor:
        """샘플링 중간 단계에서 샘플을 '현재/목표/기본 규칙'에 맞게 정리한다.

        이 함수는 샘플링이 진행되는 동안 여러 번 호출되며,
        매번 다음을 수행합니다.

        1) (옵션) 현재 상태 프레임을 강제로 맞춥니다.
           - use_current_input=True일 때만 현재 프레임이 xt 안에 존재하므로,
             그 경우 0번째 프레임을 near_current_xyyaw로 덮어씁니다.
        2) (옵션) feasible 보정 결과와 일부 섞습니다.
           - 노이즈가 거의 빠진 구간에서만 섞도록 beta(t)로 조절합니다.
        3) (cos, sin)이 단위원(길이 1) 위에 있도록 다시 정리합니다.

        Args:

        Returns:
            xt_out: 보정 후 flat 샘플.
                shape: (B, Pnn, F)
        """
        B: int = batch_size

        # xt_sequence: (B, Pnn, _, 4)
        """
        # 2) xT and x0
        pose_based = True
            (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
        pose_based = False
            (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, (T)*3)
        """
        xt_sequence: torch.Tensor = self._inpainting_past_cur_and_reshape(
            xt=xt,
            batch_size=B,
            one_or_Pnn=one_or_Pnn,
            target_seq_past=target_seq_past,
        )

        # (옵션) feasible 결과와 섞기
        if self.config.use_feasible_blend:
            raise NotImplementedError("현재 use_feasible_blend 옵션은 구현 중입니다.")
            assert self.config.use_feasible, (
                "use_feasible_blend 옵션은 use_feasible 이 켜져 있을 때만 동작합니다.")

            beta_scalar: torch.Tensor = self._compute_feasible_blend_beta(
                diffusion_time_step=t,  # (B,) or (B,future_len)
                reference_tensor_for_device=xt_sequence,  # (B, Pnn, _, 4)
            )  # shape: ()

            if beta_scalar.detach().item() > 0.0:
                xt_sequence = self._apply_feasible_blend_with_feasible_projection(
                    xt_sequence=xt_sequence,  # (B, (1+)Pnn, _, 4)
                    target_past_cur_future_valid=
                    target_past_cur_future_valid,  # (B,(1+)Pnn,time_len_total)
                    feasible_blend_beta_scalar=beta_scalar,  # ()
                )

        # yaw (cos, sin) 정리
        if self.config.pose_based:
            xt_sequence = self._project_future_yaw_to_unit_circle(
                xt_sequence=xt_sequence,  # (B, (1+)Pnn, _, 4)
            )
        xt_sequence_flattened = xt_sequence.reshape(B, one_or_Pnn, -1)
        # ✅ 추가: 무효(패딩) 타임스텝을 항상 0으로 강제
        xt_sequence_flattened = self._mask_zero_at_invalid_timestep_in_seq_flat(
            x_flat=xt_sequence_flattened,  # (B, (1+)Pnn, F)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
        )
        # 다시 flatten: (B, Pnn, F)
        return xt_sequence_flattened

    def _build_inference_correcting_xt_fn(
        self,
        batch_size: int,
        one_or_Pnn: int,
        target_seq_past: torch.Tensor,
        target_past_cur_future_valid: torch.Tensor,
        # (B, Pnn, time_len_total)
    ) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        """샘플링 루프가 요구하는 시그니처(xt,t,step) 형태의 보정 함수를 만든다.

        내부적으로는 self._initial_state_constraint 를 사용하고,
        batch/마스크/현재상태 같은 '고정 정보'는 partial로 미리 묶어서 전달합니다.

        Args:
            batch_size: B
            one_or_Pnn: Pnn
            target_past_cur_future_valid: (B, Pnn, time_len_total)

        Returns:
            correcting_xt_fn:
                callable(xt, t, step) -> xt_out
                xt: (B, Pnn, F), xt_out: (B, Pnn, F)
        """
        return partial(
            self._initial_state_constraint,
            batch_size=batch_size,
            one_or_Pnn=one_or_Pnn,
            target_seq_past=target_seq_past,
            target_past_cur_future_valid=target_past_cur_future_valid,
        )

    def _build_amortized_t_tau(
        self,
        batch_size: int,
        reference_tensor_for_device: torch.Tensor,
    ) -> torch.Tensor:
        """amortized 1-step에서 사용할 프레임별 시간값(t_tau)을 배치 크기에 맞게 만든다.

        Args:
            batch_size (int): 배치 크기 B.
            reference_tensor_for_device (torch.Tensor): device 기준 텐서.
                - shape: 임의

        Returns:
            torch.Tensor: 프레임별 시간값.
                - shape: (B, future_len)
                - dtype: float32
                - device: reference_tensor_for_device.device
        """
        # self.t_tau: (future_len,)
        t_base: torch.Tensor = self.t_tau.to(
            device=reference_tensor_for_device.device,
            dtype=torch.float32,
        )  # (future_len,)

        # t_tau: (B, future_len)
        t_tau: torch.Tensor = t_base.unsqueeze(0).expand(int(batch_size),
                                                         -1).contiguous()
        return t_tau

    def _compute_amortized_t_eff(self, t_tau: torch.Tensor) -> torch.Tensor:
        """(B, future_len) 형태의 시간값을 (B,) 대표 시간값으로 요약한다.

        이 대표 시간값은 guidance_fn / correcting_xt_fn 같이
        '시간을 스칼라로 받는 함수'에 넣기 위한 용도입니다.

        현재 DiT 구현은 (B, future_len) 시간 입력을 내부에서 평균으로 요약해 씁니다.
        (DiT._get_time_embedding에서 t.mean(dim=1)을 사용)

        Args:
            t_tau (torch.Tensor): 프레임별 시간값.
                - shape: (B, future_len)

        Returns:
            torch.Tensor: 대표 시간값.
                - shape: (B,)
                - dtype: float32
        """
        if t_tau.dim() != 2:
            raise ValueError(
                f"t_tau must be 2D (B,future_len). got {tuple(t_tau.shape)}")

        # t_eff: (B,)
        t_eff: torch.Tensor = t_tau.mean(dim=1).to(torch.float32)
        return t_eff

    def _build_classifier_kwargs_for_guidance(
            self,
            *,
            target_agents_past: Optional[torch.Tensor],
            scene_encoding_token: torch.Tensor,  # (B, token_num, D)
            scene_encoding_token_mask: torch.Tensor,  # (B, token_num) True=pad
            scene_encoding_pos_2d: torch.Tensor,  # (B, token_num, 9)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len_total)
            inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """guidance 함수가 필요로 하는 부가 정보를 한 번에 구성합니다.

        Returns:
            Dict[str, object]:
                GuidanceWrapper가 받는 classifier_kwargs dict.
        """
        model_condition: Dict[str, Any] = {
            "target_agents_past": target_agents_past,
            "target_past_cur_future_valid": target_past_cur_future_valid,
            "cross_c": scene_encoding_token,
            "cross_mask": scene_encoding_token_mask,
            "cross_pos_2d": scene_encoding_pos_2d,  # ✅ (B, token_num, 9)
        }

        classifier_kwargs: Dict[str, object] = {
            "model": self.dit,
            "model_condition": model_condition,
            "inputs": inputs,
            "observation_normalizer": self._observation_normalizer,
            "state_normalizer": self._state_normalizer,
            "config": self.config,
        }
        return classifier_kwargs

    def _compute_sigma2_over_alpha_for_guidance(
        self,
        t_eff: torch.Tensor,
        reference_tensor_for_device: torch.Tensor,
    ) -> torch.Tensor:
        """대표 시간값 t_eff에서 guidance 보정 스케일(sigma^2/alpha)을 계산합니다.

        이 구현은 DPM-Solver가 쓰는 NoiseScheduleVP(schedule="linear")의
        marginal_alpha / marginal_std를 그대로 재사용합니다.

        Args:
            t_eff (torch.Tensor):
                대표 시간값.
                shape: (B,)
                dtype: float32 권장
            reference_tensor_for_device (torch.Tensor):
                device/dtype 기준 텐서.
                shape: 임의

        Returns:
            torch.Tensor:
                sigma^2/alpha 값.
                shape: (B, 1, 1)
                dtype: float32
        """
        if t_eff.dim() != 1:
            raise ValueError(f"t_eff must be 1D (B,). got {tuple(t_eff.shape)}")

        B: int = int(t_eff.shape[0])
        device = reference_tensor_for_device.device

        t: torch.Tensor = t_eff.detach().to(device=device,
                                            dtype=torch.float32).view(B)  # (B,)

        # VPSDE_linear와 동일한 beta 범위를 NoiseScheduleVP에 그대로 전달
        beta_0: float = float(getattr(self._sde, "_beta_min", 0.1))
        beta_1: float = float(getattr(self._sde, "_beta_max", 20.0))

        noise_schedule = dpm.NoiseScheduleVP(
            schedule="linear",
            continuous_beta_0=beta_0,
            continuous_beta_1=beta_1,
            dtype=torch.float32,
        )

        alpha: torch.Tensor = noise_schedule.marginal_alpha(t)  # (B,)
        sigma: torch.Tensor = noise_schedule.marginal_std(t)  # (B,)

        alpha_safe: torch.Tensor = torch.clamp(alpha, min=1e-6)
        sigma2_over_alpha: torch.Tensor = (sigma * sigma) / alpha_safe  # (B,)

        return sigma2_over_alpha.view(B, 1, 1)  # (B,1,1)

    def _compute_guidance_grad_wrt_x(
            self,
            x_t_flat: torch.Tensor,
            t_eff: torch.Tensor,
            classifier_kwargs: Dict[str, object],
    ) -> torch.Tensor:
        """guidance_fn 출력(점수)을 x에 대해 미분한 값을 구합니다.

        핵심 안전장치
        - 바깥이 autocast(bf16) 컨텍스트여도, guidance_fn 계산 구간만 FP32로 고정합니다.
        - 바깥이 torch.no_grad()/inference_mode 여도, 여기서는 미분이 가능하도록 다시 켭니다.

        Args:
            x_t_flat (torch.Tensor):
                점수를 평가할 입력 x.
                shape: (B, P, F)
            t_eff (torch.Tensor):
                대표 시간값.
                shape: (B,)
            classifier_kwargs (Dict[str, object]):
                guidance_fn에 전달할 추가 정보.

        Returns:
            torch.Tensor:
                guidance_fn 점수의 x에 대한 미분값.
                shape: (B, P, F)
                dtype: float32
        """
        if self._guidance_fn is None:
            raise RuntimeError(
                "self._guidance_fn is None, but guidance grad was requested."
            )
        if x_t_flat.dim() != 3:
            raise ValueError(
                f"x_t_flat must be 3D (B,P,F). got {tuple(x_t_flat.shape)}"
            )
        if t_eff.dim() != 1:
            raise ValueError(f"t_eff must be 1D (B,). got {tuple(t_eff.shape)}")

        device_type: str = "cuda" if x_t_flat.is_cuda else "cpu"

        # 바깥이 torch.no_grad / torch.inference_mode 여도 여기서는 미분이 가능해야 합니다.
        with torch.inference_mode(False), torch.enable_grad():
            x_in: torch.Tensor = (
                x_t_flat.detach()
                .to(dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )  # (B,P,F) float32, requires_grad=True

            t_in: torch.Tensor = t_eff.detach().to(
                device=x_in.device,
                dtype=torch.float32,
            )  # (B,)

            # ✅ guidance_fn 계산 구간만 autocast OFF로 고정 (bf16 노출 차단)
            with torch.autocast(device_type=device_type, enabled=False):
                try:
                    score = self._guidance_fn(x_in, t_in, None,
                                              **classifier_kwargs)
                except TypeError:
                    score = self._guidance_fn(x_in, t_in, **classifier_kwargs)

                if not isinstance(score, torch.Tensor):
                    raise TypeError(
                        "guidance_fn must return a torch.Tensor. "
                        f"got {type(score)}"
                    )
                # ✅ no_grad/inference_mode/연결 끊김으로 guidance가 조용히 무력화되는 걸 방지
                if not score.requires_grad:
                    raise RuntimeError(
                        "guidance_fn output does not require grad w.r.t x. "
                        "guidance가 no_grad/inference_mode 영향으로 조용히 꺼졌거나, "
                        "guidance_fn 내부에서 x와의 연결이 끊겼을 수 있습니다."
                    )

                score_sum: torch.Tensor = score.float().sum()

            grad = torch.autograd.grad(
                outputs=score_sum,
                inputs=x_in,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        if grad is None:
            raise RuntimeError(
                "guidance gradient is None. guidance_fn 내부에서 x와의 연결이 끊겼을 수 있습니다."
            )

        return grad.detach()  # (B,P,F) float32

    def _run_dpm_sampler_for_inference(
            self,
            xT: torch.Tensor,
            target_agents_past: Optional[torch.Tensor],
            target_seq_past: Optional[torch.Tensor],
            scene_encoding_token: torch.Tensor,
            scene_encoding_token_mask: torch.Tensor,
            scene_encoding_pos_2d: torch.Tensor,  # ✅ 추가 (B, token_num, 9)
            target_past_cur_future_valid: torch.Tensor,
            inputs: Dict[str, torch.Tensor],
            correcting_xt_fn: Callable[
                [torch.Tensor, torch.Tensor, int], torch.Tensor],
            diffusion_steps: int,
    ) -> torch.Tensor:
        """dpm_sampler(또는 amortized 1-step)로 diffusion_sequence(flat)을 얻습니다."""
        classifier_kwargs: Dict[
            str, object] = self._build_classifier_kwargs_for_guidance(
            target_agents_past=target_agents_past,
            scene_encoding_token=scene_encoding_token,
            scene_encoding_token_mask=scene_encoding_token_mask,
            scene_encoding_pos_2d=scene_encoding_pos_2d,  # ✅ 추가
            target_past_cur_future_valid=target_past_cur_future_valid,
            inputs=inputs,
        )

        if diffusion_steps > 1:
            diffusion_sequence: torch.Tensor = dpm_sampler(
                self.dit,
                xT.float(),
                diffusion_steps=diffusion_steps,
                other_model_params={
                    "target_agents_past": target_agents_past,
                    "target_past_cur_future_valid": target_past_cur_future_valid,
                    "cross_c": scene_encoding_token,
                    "cross_mask": scene_encoding_token_mask,
                    "cross_pos_2d": scene_encoding_pos_2d,  # ✅ 여기 NameError 방지
                },
                dpm_solver_params={"correcting_xt_fn": correcting_xt_fn},
                model_wrapper_params={
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": classifier_kwargs,
                    "guidance_scale": self.config.guidance_scale,
                    "guidance_type": (
                        "classifier" if self._guidance_fn is not None else "uncond"),
                },
            )
            if not self.config.use_amortized_diffusion:
                return diffusion_sequence

            # amortized warm-up 로직은 네 코드 그대로 이어짐...
            # (중요) 아래 amortized 1-step에서 self.dit 호출할 때도 cross_pos_2d를 넘겨야 함
            B, one_or_Pnn, _ = diffusion_sequence.shape
            target_current_xyyaw = target_agents_past[:, :, -1, :4]
            amortized_random_noise = self._get_amortized_random_noise_from_inputs(
                inputs=inputs,
                batch_size=int(B),
                one_or_Pnn=int(one_or_Pnn),
                target_current_xyyaw=target_current_xyyaw,
            )

            diffusion_cur_future_sequence = self._reshape_inference_x0_to_sequence(
                diffusion_sequence=diffusion_sequence,
                target_current_xyyaw=target_current_xyyaw,
            )
            diffusion_future_sequence = diffusion_cur_future_sequence[
                :, :, -(self.config.future_len):, :]

            self._set_amortized_buffer_from_sequence(
                diffusion_future_sequence=diffusion_future_sequence,
                random_noise=amortized_random_noise,
                do_shift=False,
            )

            xT = self._build_inference_xT_from_noise(
                noise=self._amortized_buffer,
                target_seq_past=target_seq_past,
            )
            xT = self._mask_zero_at_invalid_timestep_in_seq_flat(
                x_flat=xT,
                target_past_cur_future_valid=target_past_cur_future_valid,
            )
            diffusion_steps = 1

        # -------- diffusion_steps == 1 (amortized 1-step) --------
        assert diffusion_steps == 1
        B = int(xT.shape[0])
        xT_f32 = xT.float()

        t_tau = self._build_amortized_t_tau(batch_size=B,
                                            reference_tensor_for_device=xT_f32)
        low_t_mask = torch.ones((B,), dtype=torch.bool, device=xT.device)

        _ = self.dit(
            target_input_norm_xT=xT_f32,
            diffusion_time=t_tau,
            target_agents_past=target_agents_past,
            target_past_cur_future_valid=target_past_cur_future_valid,
            cross_c=scene_encoding_token,
            cross_mask=scene_encoding_token_mask,
            cross_pos_2d=scene_encoding_pos_2d,  # ✅ 여기 NameError 방지
            low_t_mask=low_t_mask,
        )

        x0_base_flat = self.dit.diffusion_sequence_flat.to(device=xT_f32.device,
                                                           dtype=torch.float32)
        t_eff = self._compute_amortized_t_eff(t_tau).to(device=xT_f32.device,
                                                        dtype=torch.float32)

        x0_guided_flat = x0_base_flat
        guidance_scale: float = float(
            getattr(self.config, "guidance_scale", 0.0))
        if (self._guidance_fn is not None) and (guidance_scale != 0.0):
            grad = self._compute_guidance_grad_wrt_x(
                x_t_flat=xT_f32,
                t_eff=t_eff,
                classifier_kwargs=classifier_kwargs,
            )
            sigma2_over_alpha = self._compute_sigma2_over_alpha_for_guidance(
                t_eff=t_eff,
                reference_tensor_for_device=xT_f32,
            )
            x0_guided_flat = x0_base_flat + (
                        guidance_scale * sigma2_over_alpha) * grad

        diffusion_sequence = correcting_xt_fn(x0_guided_flat.detach(), t_eff, 0)
        return diffusion_sequence

    def _reshape_inference_x0_to_sequence(
            self,
            diffusion_sequence: torch.Tensor,
            # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
    ) -> torch.Tensor:
        """
        # 2) diffusion_sequence
        pose_based = True
            (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
        pose_based = False
            (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, (T)*3)
        """
        """
        Returns:
            diffusion_cur_future_sequence:
                shape: (B, (1+)Pnn, 1+T, 4) / (B, (1+)Pnn, T, 3)
        """
        if self.config.pose_based:
            last_dim = 4
            seq_len = 1 + self._future_len
        else:
            last_dim = 3
            seq_len = self._future_len

        batch_size, one_or_Pnn = diffusion_sequence.shape[:2]
        B: int = batch_size
        if self.config.use_past_dit_input:
            diffusion_cur_future_sequence = diffusion_sequence.reshape(
                B,
                one_or_Pnn,
                -1,
                last_dim,
            )
            diffusion_cur_future_sequence = diffusion_cur_future_sequence[:, :,
                                                                          -seq_len:, :]
            return diffusion_cur_future_sequence
        if self.config.use_current_input:
            diffusion_cur_future_sequence = diffusion_sequence.reshape(
                B, one_or_Pnn, seq_len, last_dim)
            return diffusion_cur_future_sequence
        # self.config.use_current_input = False 일 때
        if self.config.pose_based:
            diffusion_cur_future_sequence = torch.cat(
                [
                    target_current_xyyaw.unsqueeze(2),  # (B,Pnn,1,4)
                    diffusion_sequence.reshape(B, one_or_Pnn, self._future_len,
                                               last_dim),  # (B,Pnn,T,4)
                ],
                dim=2,
            )  # (B,Pnn,1+T,4)
        else:
            diffusion_cur_future_sequence = diffusion_sequence.reshape(
                B, one_or_Pnn, seq_len, last_dim)
        return diffusion_cur_future_sequence

    def _append_inference_feasible_outputs(
            self,
            decoder_output_dict: Dict[str, torch.Tensor],
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
    ) -> None:
        """추론 모드에서 feasible 결과를 outputs에 추가합니다.

        추가되는 키
        ----------
        - "integrated_trajectory":
            Feasible가 적분한 미래 궤적을 현재 프레임과 붙여 반환합니다.
            shape: (B, (1+)Pnn, 1+T, 4)

        - (pose_based=False인 경우) "control_sequence":
            Feasible가 최종으로 적분에 사용한 control 시퀀스입니다.
            (필터가 켜져 있으면 필터 적용 후 값, 꺼져 있으면 raw와 동일)
            shape: (B, (1+)Pnn, T, 3)

        목적
        ----
        pose_based=False & use_feasible=True에서
        - pose 업데이트는 integrated_trajectory(=Feasible 기반)
        - 다음 chunk 입력 past control 업데이트는 control_sequence(=Feasible 최종 control)
        로 맞춰, 입력 불일치(past pose vs past control)를 없앱니다.
        """
        if not self.config.use_feasible:
            return

        norm_returns = getattr(self.dit, "norm_dit_returns", None)
        if norm_returns is None:
            return

        integrated_future = getattr(norm_returns, "integrated_trajectory", None)
        if not isinstance(integrated_future, torch.Tensor):
            return

        # integrated_trajectory: (B, (1+)Pnn, 1+T, 4)
        integrated_trajectory: torch.Tensor = torch.cat(
            [target_current_xyyaw.unsqueeze(2), integrated_future],
            dim=2,
        )
        decoder_output_dict["integrated_trajectory"] = integrated_trajectory

        # pose_based=False인 경우: Feasible가 최종으로 사용한 control_sequence를 함께 노출
        if not bool(self.config.pose_based):
            control_sequence = getattr(norm_returns, "control_sequence", None)
            if isinstance(control_sequence, torch.Tensor):
                # dtype/device는 integrated_trajectory와 맞춰 둠(불필요한 변환 방지)
                decoder_output_dict["control_sequence"] = _cast_like(
                    control_sequence, integrated_trajectory)

    def _forward_training_mode(
            self,
            inputs: Dict[str, torch.Tensor],
            scene_encoding_token: torch.Tensor,
            scene_encoding_token_mask: torch.Tensor,
            scene_encoding_pos_2d: torch.Tensor,  # ✅ 추가
            target_agents_past: torch.Tensor,
            target_seq_past: torch.Tensor,
            target_past_cur_future_valid: torch.Tensor,
            batch_size: int,
            one_or_Pnn: int,
    ) -> Dict[str, torch.Tensor]:
        """훈련 모드에서 한 배치에 대해 decoder 를 한 번 돌린다."""
        decoder_output_dict: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        diffusion_time: torch.Tensor = inputs["diffusion_time"]

        xT_input_flat = self._build_training_dit_inputs(inputs=inputs,
                                                        target_seq_past=target_seq_past)
        xT_input_flat = self._mask_zero_at_invalid_timestep_in_seq_flat(
            x_flat=xT_input_flat,
            target_past_cur_future_valid=target_past_cur_future_valid,
        )

        diffusion_time = _prepare_diffusion_time_for_dit(
            diffusion_time=diffusion_time, reference_tensor=xT_input_flat)
        low_t_mask = _prepare_bool_mask_on_device(inputs["low_t_mask"],
                                                  device=xT_input_flat.device)

        diffusion_output: torch.Tensor = self.dit(
            target_input_norm_xT=xT_input_flat,
            diffusion_time=diffusion_time,
            target_agents_past=target_agents_past,
            target_past_cur_future_valid=target_past_cur_future_valid,
            cross_c=scene_encoding_token,
            cross_mask=scene_encoding_token_mask,
            cross_pos_2d=scene_encoding_pos_2d,  # ✅ 여기 NameError 방지
            low_t_mask=low_t_mask,
        )

        target_current_xyyaw = target_agents_past[:, :, -1, :4]
        diffusion_future_output: torch.Tensor = self._extract_future_from_diffusion_output(
            diffusion_output=diffusion_output,
            batch_size=B,
            one_or_Pnn=one_or_Pnn,
        )
        decoder_output_dict["diffusion_output"] = diffusion_future_output
        decoder_output_dict[
            "diffusion_sequence"] = self.dit.norm_dit_returns.diffusion_sequence

        self._append_training_feasible_outputs(
            decoder_output_dict=decoder_output_dict,
            target_current_xyyaw=target_current_xyyaw,
        )
        return decoder_output_dict
    # =========================================================
    # 3) Decoder._set_amortized_buffer_from_sequence 를 교체
    # =========================================================
    def _set_amortized_buffer_from_sequence(
            self,
            diffusion_future_sequence: torch.Tensor,
            # (B, (1+)Pnn, future_len, 4 or 3)  == "x0" 역할
            random_noise: torch.Tensor,
            # (B, (1+)Pnn, future_len, 4 or 3)  == "새로 받은 eps 후보(전체 T)"  (do_shift=True면 tail만 사용)
            rollout_time_chunk_size: Optional[int] = None,
            do_shift: bool = True,
    ) -> None:
        """diffusion_future_sequence(x0)와 eps로 다음 스텝의 noisy 버퍼(z)를 만듭니다.

        핵심 동작
        ----------
        - Warm-up(do_shift=False):
          - eps 전체(T칸)를 1번 저장합니다.
          - z = alpha(t_tau) * x0 + sigma(t_tau) * eps 로 만들어 self._amortized_buffer(z)에 저장합니다.
        - AR 업데이트(do_shift=True):
          - x0도 gap만큼 shift합니다.
          - eps도 "기존 eps 버퍼"를 gap만큼 shift합니다.
          - 새로 생긴 tail(gap칸)만 random_noise의 tail을 사용해 eps를 채웁니다.
          - z는 전체 T칸을 한 번에 다시 계산합니다.
          - self._amortized_buffer(z)와 self._amortized_buffer_eps(eps)를 같이 갱신합니다.

        Args:
            diffusion_future_sequence (torch.Tensor):
                x0 역할 텐서.
                shape: (B, (1+)Pnn, future_len, C)  C=4(pose) 또는 3(control)
            random_noise (torch.Tensor):
                표준정규 노이즈(eps) 텐서.
                shape: (B, (1+)Pnn, future_len, C)
                - do_shift=True일 때는 tail(gap) 구간만 사용합니다.
            rollout_time_chunk_size (Optional[int]):
                gap. do_shift=True일 때만 필요합니다.
            do_shift (bool):
                True면 AR 업데이트(shift + tail append),
                False면 warm-up(전체 eps 저장) 입니다.

        Returns:
            None
        """
        if not isinstance(random_noise, torch.Tensor):
            raise TypeError("random_noise must be torch.Tensor.")

        if tuple(random_noise.shape) != tuple(diffusion_future_sequence.shape):
            raise ValueError(
                "random_noise shape가 diffusion_future_sequence와 같아야 합니다. "
                f"random_noise={tuple(random_noise.shape)}, "
                f"diffusion_future_sequence={tuple(diffusion_future_sequence.shape)}"
            )

        B = int(diffusion_future_sequence.shape[0])
        future_len = int(diffusion_future_sequence.shape[2])
        device_type = diffusion_future_sequence.device.type

        # t_tau: (B, future_len)
        batch_diffusion_time: torch.Tensor = self.t_tau.unsqueeze(0).repeat(B,
                                                                            1).to(
            device=diffusion_future_sequence.device
        )

        # -------------------------
        # (A) x0 shift 준비
        # -------------------------
        if do_shift:
            assert rollout_time_chunk_size is not None, "rollout_time_chunk_size must be set when do_shift=True."
            gap = int(rollout_time_chunk_size)
            gap = max(0, min(gap, future_len))

            x0_shifted = torch.zeros_like(
                diffusion_future_sequence)  # (B,P,T,C)
            if gap > 0 and gap < future_len:
                x0_shifted[:, :, :future_len - gap, :] = \
                diffusion_future_sequence[:, :, gap:, :]
            elif gap == 0:
                x0_shifted = diffusion_future_sequence
            # gap == future_len 이면 전부 0 유지
        else:
            gap = 0
            x0_shifted = diffusion_future_sequence

        # -------------------------
        # (B) eps shift + tail append
        # -------------------------
        if do_shift:
            if self._amortized_buffer_eps is None:
                raise RuntimeError(
                    "do_shift=True 인데 self._amortized_buffer_eps가 없습니다. "
                    "warm-up(do_shift=False)에서 eps 버퍼를 먼저 세팅해야 합니다."
                )
            if tuple(self._amortized_buffer_eps.shape) != tuple(
                    diffusion_future_sequence.shape):
                raise RuntimeError(
                    "self._amortized_buffer_eps shape가 현재 diffusion_future_sequence와 다릅니다. "
                    f"eps={tuple(self._amortized_buffer_eps.shape)}, "
                    f"x0={tuple(diffusion_future_sequence.shape)}"
                )

            prev_eps = self._amortized_buffer_eps.to(
                device=diffusion_future_sequence.device,
                dtype=diffusion_future_sequence.dtype,
            )

            eps_shifted = torch.zeros_like(
                diffusion_future_sequence)  # (B,P,T,C)

            if gap > 0 and gap < future_len:
                # 기존 eps를 앞으로 당김
                eps_shifted[:, :, :future_len - gap, :] = prev_eps[
                    :, :, gap:, :]
                # tail(gap)만 새 eps로 채움 (random_noise의 tail만 사용)
                eps_shifted[:, :, future_len - gap:, :] = random_noise[
                    :, :, future_len - gap:, :]
            elif gap == 0:
                eps_shifted = prev_eps
            else:
                # gap == future_len: 전부 새 tail(=전체)로
                eps_shifted = random_noise.to(
                    device=diffusion_future_sequence.device,
                    dtype=diffusion_future_sequence.dtype,
                )
        else:
            # warm-up: eps 전체를 그대로 저장
            eps_shifted = random_noise.to(
                device=diffusion_future_sequence.device,
                dtype=diffusion_future_sequence.dtype,
            )

        # -------------------------
        # (C) z 재계산: z = alpha*x0 + sigma*eps
        #     (계산은 fp32로)
        # -------------------------
        with torch.autocast(device_type=device_type, enabled=False):
            x0_f32 = x0_shifted.to(dtype=torch.float32)
            t_f32 = batch_diffusion_time.to(dtype=torch.float32)

            mean_f32, std_f32 = self.sde.marginal_prob(x0_f32,
                                                       t_f32)  # mean=alpha*x0, std=sigma
            eps_f32 = eps_shifted.to(device=x0_f32.device, dtype=torch.float32)

            z_f32 = mean_f32 + std_f32 * eps_f32  # (B,P,T,C)

        # -------------------------
        # (D) 버퍼 갱신 (dtype/device는 x0 기준으로 맞춤)
        # -------------------------
        self._amortized_buffer = z_f32.to(
            device=diffusion_future_sequence.device,
            dtype=diffusion_future_sequence.dtype,
        ).detach()

        self._amortized_buffer_eps = eps_shifted.to(
            device=diffusion_future_sequence.device,
            dtype=diffusion_future_sequence.dtype,
        ).detach()

    @staticmethod
    def _normalize_cos_sin_for_rotation(
        cos_values: torch.Tensor,
        sin_values: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin)을 회전에 쓰기 좋게 길이 1로 맞춥니다.

        - 길이가 충분히 크면: (cos, sin)을 길이 1로 나눠 정리합니다.
        - 길이가 거의 0이면: 회전을 못하므로 (1, 0)으로 둡니다(회전 없음).

        Args:
            cos_values: shape = (...)
            sin_values: shape = (...)
            eps: 0 나눗셈 방지 값.

        Returns:
            cos_normalized: shape = (...)
            sin_normalized: shape = (...)
        """
        raw_norm = torch.sqrt(cos_values**2 + sin_values**2)
        safe_norm = torch.clamp(raw_norm, min=eps)

        cos_normalized = cos_values / safe_norm
        sin_normalized = sin_values / safe_norm

        too_small = raw_norm < eps
        cos_normalized = torch.where(too_small, torch.ones_like(cos_normalized),
                                     cos_normalized)
        sin_normalized = torch.where(too_small,
                                     torch.zeros_like(sin_normalized),
                                     sin_normalized)
        return cos_normalized, sin_normalized

    def _extract_delta_pose_params(
        self,
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ego 다음 포즈에서 이동/회전에 필요한 값들을 뽑습니다.

        Args:
            normed_ego_next_pose: (B, 4) = (x, y, cos(각도), sin(각도))

        Returns:
            delta_xy: (B, 2)
            cos_delta: (B,)
            sin_delta: (B,)
            yaw_delta: (B,)
        """
        if normed_ego_next_pose.dim(
        ) != 2 or normed_ego_next_pose.shape[-1] != 4:
            raise ValueError("normed_ego_next_pose는 (B, 4) 여야 합니다. "
                             f"현재 shape={tuple(normed_ego_next_pose.shape)}")

        delta_xy = normed_ego_next_pose[:, 0:2]  # (B, 2)
        cos_raw = normed_ego_next_pose[:, 2]  # (B,)
        sin_raw = normed_ego_next_pose[:, 3]  # (B,)

        cos_delta, sin_delta = self._normalize_cos_sin_for_rotation(
            cos_raw, sin_raw)
        yaw_delta = torch.atan2(sin_delta, cos_delta)
        return delta_xy, cos_delta, sin_delta, yaw_delta

    @staticmethod
    def _transform_points_to_new_origin(
            points_xy: torch.Tensor,  # (B, ..., 2)
            delta_xy: torch.Tensor,  # (B, 2)
            cos_delta: torch.Tensor,  # (B,)
            sin_delta: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """(x, y) 점들을 새 원점 기준으로 변환합니다(이동+회전)."""
        if points_xy.shape[-1] != 2:
            raise ValueError(
                f"points_xy의 마지막 차원은 2여야 합니다. shape={tuple(points_xy.shape)}")

        delta_xy_b = delta_xy
        cos_b = cos_delta
        sin_b = sin_delta
        for _ in range(points_xy.dim() - 2):
            delta_xy_b = delta_xy_b.unsqueeze(1)
            cos_b = cos_b.unsqueeze(1)
            sin_b = sin_b.unsqueeze(1)

        dx = points_xy[..., 0] - delta_xy_b[..., 0]
        dy = points_xy[..., 1] - delta_xy_b[..., 1]

        x_new = cos_b * dx + sin_b * dy
        y_new = -sin_b * dx + cos_b * dy
        return torch.stack([x_new, y_new], dim=-1)

    @staticmethod
    def _rotate_heading_cos_sin_to_new_origin(
            cos_heading: torch.Tensor,  # (B, ...)
            sin_heading: torch.Tensor,  # (B, ...)
            cos_delta: torch.Tensor,  # (B,)
            sin_delta: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) 방향을 새 기준으로 바꿉니다(각도 빼기)."""
        cos_b = cos_delta
        sin_b = sin_delta
        for _ in range(cos_heading.dim() - 1):
            cos_b = cos_b.unsqueeze(1)
            sin_b = sin_b.unsqueeze(1)

        cos_new = cos_heading * cos_b + sin_heading * sin_b
        sin_new = sin_heading * cos_b - cos_heading * sin_b
        return cos_new, sin_new

    def _transform_pose_4_dim_inplace(
        self,
        pose_4_dim: torch.Tensor,
        delta_xy: torch.Tensor,
        cos_delta: torch.Tensor,
        sin_delta: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """(x, y, cos, sin) 포즈 텐서를 "새 원점/방향" 기준으로 바꿉니다.

        이 함수가 하는 일
        --------------
        - pose_4_dim의 위치(x, y)를
          1) delta_xy 만큼 이동(빼기)
          2) cos_delta/sin_delta 만큼 회전
          해서 새 기준 좌표로 바꿉니다.
        - pose_4_dim의 방향(cos, sin)도 같은 기준으로 회전해 줍니다.
        - valid_mask가 False인 칸(무효 타임스텝)은 값을 그대로 둡니다.

        Args:
            pose_4_dim (torch.Tensor):
                포즈 텐서.
                shape: (B, T, 4) 또는 (B, P, T, 4) 등 (..., 4)
                마지막 4는 (x, y, cos, sin)
            delta_xy (torch.Tensor):
                새 원점으로 삼을 ego 위치.
                shape: (B, 2)
            cos_delta (torch.Tensor):
                새 기준 회전에 쓰는 cos 값.
                shape: (B,)
            sin_delta (torch.Tensor):
                새 기준 회전에 쓰는 sin 값.
                shape: (B,)
            valid_mask (torch.Tensor):
                변환을 적용할 칸(True) / 그대로 둘 칸(False).
                shape: pose_4_dim.shape[:-1]
                dtype: torch.bool

        Returns:
            None
        """
        if pose_4_dim.dim() < 2 or int(pose_4_dim.shape[-1]) != 4:
            raise ValueError("pose_4_dim은 (..., 4) 이어야 합니다. "
                             f"현재 shape={tuple(pose_4_dim.shape)}")

        expected_mask_shape = tuple(int(x) for x in pose_4_dim.shape[:-1])
        if tuple(int(x) for x in valid_mask.shape) != expected_mask_shape:
            raise ValueError(
                "valid_mask shape가 pose_4_dim과 맞지 않습니다. "
                f"expected={expected_mask_shape}, got={tuple(valid_mask.shape)}"
            )

        valid_mask_bool = valid_mask.to(dtype=torch.bool)

        # 1) (x, y) 변환
        pos_xy = pose_4_dim[..., 0:2]  # shape: (..., 2)
        pos_xy_new = self._transform_points_to_new_origin(
            points_xy=pos_xy,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
        )  # shape: (..., 2)
        pose_4_dim[..., 0:2] = torch.where(
            valid_mask_bool[..., None],
            pos_xy_new,
            pos_xy,
        )

        # 2) (cos, sin) 방향 변환
        cos_h = pose_4_dim[..., 2]  # shape: (...)
        sin_h = pose_4_dim[..., 3]  # shape: (...)
        cos_new, sin_new = self._rotate_heading_cos_sin_to_new_origin(
            cos_heading=cos_h,
            sin_heading=sin_h,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
        )  # shape: (...), (...)

        pose_4_dim[..., 2] = torch.where(valid_mask_bool, cos_new, cos_h)
        pose_4_dim[..., 3] = torch.where(valid_mask_bool, sin_new, sin_h)

    def _update_diffusion_future_sequence_origin(
            self,
            diffusion_future_sequence: torch.Tensor,  # (B, 1+Pnn, T, 4 or 3)
            rollout_time_chunk_size: int,
            target_future_valid: torch.Tensor,  # (B, (1+)Pnn, future_len)
    ) -> torch.Tensor:  # (B, 1+Pnn, T, 4 or 3)
        """amortized 추론에서 재사용하는 'diffusion_future_sequence'의 좌표 기준을 다음 스텝 기준으로 맞춥니다.

        이 함수가 필요한 이유
        -------------------
        평가/롤아웃 코드에서는 매 스텝(또는 chunk)마다 입력 전체를 "새 ego 기준(0,0)"으로 바꿉니다.
        그런데 Decoder 안의 미래 버퍼(self._amortized_buffer)는 다음 스텝에도 재사용되므로,
        diffusion_future_sequence도 똑같이 기준을 바꾸지 않으면,
        다음 스텝에서 "과거/현재(새 기준)"과 "미래 버퍼(옛 기준)"가 섞일 수 있습니다.

        처리 순서(중요)
        -------------
        diffusion_future_sequence 는 모델이 쓰는 스케일(정규화된 값)로 저장되어 있습니다.
        이동/회전은 실제 단위에서 하는 것이 안전하므로 아래 순서를 지킵니다.

          1) diffusion_future_sequence를 state_normalizer.inverse 로 실제 단위로 되돌립니다.
          2) rollout_time_chunk_size 시점의 ego 포즈를 기준점으로 삼습니다.
          3) diffusion_future_sequence 전체를 같은 기준으로 이동/회전합니다.
          4) 다시 state_normalizer 로 모델 입력 스케일로 맞춥니다.
          5) (cos, sin)이 길이 1이 되도록 한 번 더 정리합니다.

        """

        # buffer_norm: (B, (1+)Pnn, future_len, 4)
        buffer_norm: torch.Tensor = diffusion_future_sequence

        # dtype/device 보존용
        original_dtype = buffer_norm.dtype
        device = buffer_norm.device

        # 1) 버퍼를 실제 단위로 되돌리기
        # buffer_unnorm: (B, (1+)Pnn, future_len, 4)
        # target_future_valid: (B, (1+)Pnn, future_len)
        buffer_unnorm: torch.Tensor = self.config.state_normalizer.inverse(
            buffer_norm, target_future_valid)

        # 2) ego가 기준이 될 시점(rollout_time_chunk_size)에서의 ego 포즈 뽑기

        step_index: int = int(rollout_time_chunk_size - 1)
        # unnorm_ego_next_4_dim: (B, 4)
        unnorm_ego_next_4_dim: torch.Tensor = buffer_unnorm[:, 0,
                                                            step_index, :].detach(
                                                            )

        # 3) 이동/회전에 필요한 값 계산
        # delta_xy: (B, 2), cos_delta: (B,), sin_delta: (B,)
        delta_xy, cos_delta, sin_delta, _ = self._extract_delta_pose_params(
            unnorm_ego_next_4_dim)

        # 4) 버퍼 전체를 새 기준으로 이동/회전 (유효 프레임만)
        # valid_mask: (B, (1+)Pnn, future_len)
        valid_mask: torch.Tensor = target_future_valid.to(device=device,
                                                          dtype=torch.bool)
        self._transform_pose_4_dim_inplace(
            pose_4_dim=buffer_unnorm,  # (B, (1+)Pnn, future_len, 4)  (실제 단위)
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=valid_mask,  # (B, (1+)Pnn, future_len)
        )

        # 5) 다시 모델 스케일로 맞추기
        # buffer_norm_new: (B, (1+)Pnn, future_len, 4)
        # target_future_valid: (B, (1+)Pnn, future_len)
        buffer_norm_new: torch.Tensor = self.config.state_normalizer(
            buffer_unnorm, target_future_valid)

        # dtype/device 원복 + 안전하게 detach
        buffer_norm_new = buffer_norm_new.to(device=device,
                                             dtype=original_dtype).detach()

        # 6) (cos, sin) 길이 1 정리
        buffer_norm_new = self._project_future_yaw_to_unit_circle(
            buffer_norm_new)

        diffusion_future_sequence = buffer_norm_new
        return diffusion_future_sequence

    def _forward_inference_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        scene_encoding_pos_2d: torch.Tensor,
        # (B, (1+)Pnn, time_len, 11)
        target_agents_past: torch.Tensor,
        # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
        target_seq_past: torch.Tensor,
        target_past_cur_future_valid: torch.Tensor,
        batch_size: int,
        one_or_Pnn: int,
    ) -> Dict[str, torch.Tensor]:
        """ decoder_output_dict
        diffusion_sequence : (B, (1+)Pnn, 1+T, 4) or (B, (1+)Pnn, T, 3)
        "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
        "control_constraint_diff" : (B, (1+)Pnn, T, 3)
        "control_sequence" : (B, (1+)Pnn, T, 3)
        """
        with torch.no_grad():
            target_current_xyyaw = target_agents_past[:, :,
                                                      -1, :4]  # (B,(1+)Pnn,4)
            decoder_output_dict: Dict[str, torch.Tensor] = {}
            B: int = batch_size
            # 1) 시작 노이즈 생성 (미래만)
            # ✅ (변경) 외부에서 만든 noise를 반드시 입력으로 받는다.
            # noise: (B,(1+)Pnn,T,4)
            """
            noise_future_sequence 가 들어온 경우
                1) use_amortized_diffusion 가 False인 경우
                2) use_amortized_diffusion 가 True인 경우이면서 첫번쨰 샘플링인 경우

            noise_future_sequence 가 들어오지 않은 경우
                1) use_amortized_diffusion 가 True이면서 첫 스텝이 아닌 경우
            """
            noise_future_sequence = inputs.get("inference_noise", None)
            if self.config.use_amortized_diffusion:
                if noise_future_sequence is not None:  # 첫번쨰 샘플링
                    # ✅ (중요) 새 rollout 시작이면 z/eps 둘 다 리셋
                    self._amortized_buffer = None
                    self._amortized_buffer_eps = None
                    # noise_future_sequence: (B,(1+)Pnn,T,4 or 3)
                    noise_future_sequence = self._get_inference_noise_from_inputs(
                        inputs=inputs,
                        batch_size=int(B),
                        one_or_Pnn=int(one_or_Pnn),
                        target_current_xyyaw=target_current_xyyaw,
                    )
                    diffusion_steps = 16
                else:  # 첫 스텝이 아닌 경우
                    diffusion_steps = 1
                    assert self._amortized_buffer is not None, (
                        "When using amortized diffusion during inference, "
                        "if inference_noise is not provided, "
                        "self._amortized_buffer must be set.")
                    # noise_future_sequence ; (B,(1+)Pnn,T,4 or 3)
                    noise_future_sequence = self._amortized_buffer
            else:  # 기존 DPM-Solver 경로
                self._amortized_buffer_eps = None
                assert self._amortized_buffer is None, (
                    "self._amortized_buffer should be None when not using amortized diffusion."
                )
                diffusion_steps = 10
                if noise_future_sequence is None:
                    raise ValueError(
                        "inference_noise must be provided "
                        "in inputs when not using amortized diffusion.")
                else:
                    noise_future_sequence: torch.Tensor = self._get_inference_noise_from_inputs(
                        inputs=inputs,
                        batch_size=int(B),
                        one_or_Pnn=int(one_or_Pnn),
                        target_current_xyyaw=target_current_xyyaw,
                    )  # (B,(1+)Pnn,T,4)
            """
            # 2) xT(flat) 생성
            pose_based = True
                (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
            pose_based = False
                (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, (T)*3)
            """
            xT: torch.Tensor = self._build_inference_xT_from_noise(
                noise=noise_future_sequence,  # (B,(1+)Pnn,T,4 or 3)
                target_seq_past=target_seq_past,
                # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
            )  # (B,Pnn,F)
            xT = self._mask_zero_at_invalid_timestep_in_seq_flat(
                x_flat=xT,
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
            )
            # 4) 샘플링 중 보정 함수 구성
            correcting_xt_fn = self._build_inference_correcting_xt_fn(
                batch_size=B,
                one_or_Pnn=one_or_Pnn,
                target_seq_past=target_seq_past,
                # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B,Pnn,time_total)
            )

            # 5) dpm_sampler 실행
            """
            # 2) xT and diffusion_sequence (x,y,cos,sin) or (v_x^b, v_y^b, yaw_rate)
            pose_based = True
                (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
            pose_based = False
                (B, (1+)Pnn, (past_len+T)*3) or (B, (1+)Pnn, (T)*3)
            """
            diffusion_sequence: torch.Tensor = self._run_dpm_sampler_for_inference(
                xT=xT,
                target_agents_past=target_agents_past,  # (B,Pnn,time_len,11)
                target_seq_past=
                target_seq_past,  # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                scene_encoding_pos_2d=scene_encoding_pos_2d,
                target_past_cur_future_valid=target_past_cur_future_valid,
                inputs=inputs,
                correcting_xt_fn=correcting_xt_fn,
                diffusion_steps=diffusion_steps,
            )

            # dtype 맞춤(기존 로직 유지)
            diffusion_sequence = diffusion_sequence.to(xT.dtype)

            # 6) diffusion_cur_future_sequence: (B, (1+)Pnn, 1+T, 4) / (B, (1+)Pnn, T, 3)
            diffusion_cur_future_sequence: torch.Tensor = self._reshape_inference_x0_to_sequence(
                diffusion_sequence=diffusion_sequence,
                target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            )
            # 6) diffusion_cur_future_sequence: (B, (1+)Pnn, 1+T, 4) / (B, (1+)Pnn, T, 3)
            decoder_output_dict[
                "diffusion_sequence"] = diffusion_cur_future_sequence
            """
            4.3 # self._amortized_buffer 채우기 (noise 주고 채워야 한다!)
            """
            # self.dit.norm_dit_returns.integrated_trajectory : (B, (1+)Pnn, T, 4) 정규화 상태
            if self.config.use_amortized_diffusion:
                if self.config.use_feasible:
                    if self.config.pose_based:
                        integrated = self.dit.norm_dit_returns.integrated_trajectory.to(
                            device=target_current_xyyaw.device,
                            dtype=target_current_xyyaw.dtype,
                        ).detach()
                        diffusion_future_sequence = integrated  # (B, 1+Pnn, T, 4)
                    else:
                        control_sequence = self.dit.norm_dit_returns.control_sequence.to(
                            device=target_current_xyyaw.device,
                            dtype=target_current_xyyaw.dtype,
                        ).detach()
                        diffusion_future_sequence = control_sequence  # (B, 1+Pnn, T, 3)

                else:
                    # (B, 1+Pnn, T, 4 or 3)
                    diffusion_future_sequence = diffusion_cur_future_sequence[:, :,
                                                                              -self
                                                                              .
                                                                              config
                                                                              .
                                                                              future_len:, :].detach(
                                                                              )
                rollout_time_chunk_size = inputs["rollout_time_chunk_size"]
                rollout_time_chunk_size_int = int(
                    rollout_time_chunk_size[0].item())

                # ✅ cand_idx 기반 seed로 만든 랜덤을 inputs에서 받아 사용
                # amortized_random_noise: (B*R, (1+)Pnn, future_len, 4 or 3)
                amortized_random_noise = self._get_amortized_random_noise_from_inputs(
                    inputs=inputs,
                    batch_size=int(B),
                    one_or_Pnn=int(one_or_Pnn),
                    target_current_xyyaw=target_current_xyyaw,
                )

                # target_future_valid: (B, (1+)Pnn, T)
                target_future_valid = target_past_cur_future_valid[:, :, -self.
                                                                   _future_len:].detach(
                                                                   )
                # ✅ 미래 버퍼도 다음 스텝 기준 좌표로 맞추기
                if self.config.pose_based:
                    diffusion_future_sequence = self._update_diffusion_future_sequence_origin(
                        diffusion_future_sequence=diffusion_future_sequence,
                        rollout_time_chunk_size=rollout_time_chunk_size_int,
                        target_future_valid=target_future_valid,
                    )

                # noise_future_sequence ; (B,(1+)Pnn,T,4 or 3)
                self._set_amortized_buffer_from_sequence(
                    diffusion_future_sequence=
                    diffusion_future_sequence,  # (B, 1+Pnn, T, 4 or 3)
                    random_noise=
                    amortized_random_noise,  # (B*R, (1+)Pnn, future_len, 4or 3)
                    rollout_time_chunk_size=rollout_time_chunk_size_int,
                    do_shift=True,
                )

            # 8) Feasible 출력(옵션)
            # decoder_output_dict["integrated_trajectory"] : (B, (1+)Pnn, 1+T, 4)
            self._append_inference_feasible_outputs(
                decoder_output_dict=decoder_output_dict,
                target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            )
            """ decoder_output_dict
            "diffusion_output" : (B, (1+)Pnn, T, 4) / (B, (1+)Pnn, T, 3)
            "diffusion_sequence" : (B, (1+)Pnn, T, 4) / (B, (1+)Pnn, T, 3)
            "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
            "control_constraint_diff" : (B, (1+)Pnn, T, 3)
            "control_sequence" : (B, (1+)Pnn, T, 3)
            """
            return decoder_output_dict

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
                - "route_known_mask":          (B, Pnn)
        Returns:
            Dict[str, torch.Tensor]: 최소한 "diffusion_output" 키를 가지며,
            설정에 따라 "integrated_trajectory",
            "control_constraint_diff" 도 포함될 수 있다.
        """
        (
            target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
            target_seq_past,
            target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
            batch_size,
            one_or_Pnn,
        ) = self._prepare_target_trajectories_and_masks(inputs)
        (
            scene_encoding_token,
            scene_encoding_token_mask,
            scene_encoding_pos_2d,
        ) = self._unpack_encoder_outputs(encoder_outputs=encoder_outputs)

        scene_encoding_token, scene_encoding_token_mask = _prepare_cross_inputs_for_dit(
            cross_c=scene_encoding_token,
            cross_mask=scene_encoding_token_mask,
            reference_tensor=target_agents_past,
        )

        scene_encoding_pos_2d = _prepare_cross_pos_2d_for_dit(
            cross_pos_2d=scene_encoding_pos_2d,  # (B, token_num, 9)
            reference_tensor=target_agents_past,  # device 기준
        )

        if self.training:
            """ decoder_output_dict
            diffusion_output : (B, (1+)Pnn, 1+T, 4) or (B, (1+)Pnn, T, 3)
            diffusion_sequence : (B, (1+)Pnn, 1+T, 4) or (B, (1+)Pnn, T, 3)
            "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
            "control_constraint_diff" : (B, (1+)Pnn, T, 3)
            "control_sequence" : (B, (1+)Pnn, T, 3)
            """
            return self._forward_training_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                scene_encoding_pos_2d=scene_encoding_pos_2d,  # ✅ 추가
                target_agents_past=
                target_agents_past,  # (B, (1+)Pnn, time_len, 11)
                target_seq_past=
                target_seq_past,  # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
                batch_size=batch_size,
                one_or_Pnn=one_or_Pnn,
            )
        else:
            """ decoder_output_dict
            diffusion_sequence : (B, (1+)Pnn, 1+T, 4) or (B, (1+)Pnn, T, 3)
            "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
            "control_constraint_diff" : (B, (1+)Pnn, T, 3)
            "control_sequence" : (B, (1+)Pnn, T, 3)
            """
            return self._forward_inference_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                scene_encoding_pos_2d=scene_encoding_pos_2d,  # ✅ 추가

                target_agents_past=target_agents_past,
                # (B, (1+)Pnn, time_len, 11)
                target_seq_past=target_seq_past,
                # (B, (1+)Pnn, (time_len, 4) or (past_len(=time_len-1), 3))
                target_past_cur_future_valid=target_past_cur_future_valid,
                batch_size=batch_size,
                one_or_Pnn=one_or_Pnn,
            )


from dataclasses import dataclass


@dataclass(frozen=False)
class DiTReturns:
    """
    diffusion_sequence
        pose_based:
            (B, (1+)Pnn, (1+T),4)
        else:
            (B, (1+)Pnn, T,3)

    """
    diffusion_sequence: Optional[torch.Tensor] = None
    integrated_trajectory: Optional[torch.Tensor] = None  # (B, Pnn, T, 4)
    control_constraint_diff: Optional[torch.Tensor] = None  # (B, Pnn, T, 3)
    control_sequence: Optional[torch.Tensor] = None  # (B, Pnn, T, 3)

class SimpleSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by heads. hidden_dim={hidden_dim}, heads={heads}"
            )
        self.heads = int(heads)
        self.head_dim = int(hidden_dim // heads)

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(float(proj_drop))

    @staticmethod
    def _get_compute_dtype(x: torch.Tensor) -> torch.dtype:
        """FlashAttention 계산에 사용할 dtype을 정합니다."""
        if x.is_cuda and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        return x.dtype

    def _touch_params_zero(self, ref: torch.Tensor) -> torch.Tensor:
        """토큰이 0개일 때도 DDP에서 파라미터가 '사용됨'으로 보이게 0계수로 연결."""
        touch = ref.new_zeros(())
        for p in self.parameters():
            if p is None or p.numel() == 0:
                continue
            touch = touch + p.view(-1)[:1].sum().to(dtype=ref.dtype, device=ref.device)
        return touch * 0.0

    def forward(
        self,
        x: torch.Tensor,  # (B, P, H)
        key_padding_mask: torch.Tensor,  # (B, P) True=pad
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be (B,P,H). got {tuple(x.shape)}")
        B, P, H = x.shape

        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # 유효 토큰(True=valid)만 unpad
        valid_mask = (~key_padding_mask).to(torch.bool)  # (B,P)

        res = unpad_input(x, valid_mask)
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, _ = res
        else:
            raise RuntimeError(f"unpad_input() return format unexpected. len={len(res)}")

        T = int(x_unpad.shape[0])
        max_seqlen_int = int(max_seqlen) if not isinstance(max_seqlen, int) else int(max_seqlen)

        # 유효 토큰이 0개면: 출력은 전부 0 (하지만 파라미터는 0계수로 연결)
        if T == 0 or max_seqlen_int == 0:
            out = x.new_zeros((B, P, H))
            out = out + self._touch_params_zero(out)
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return out

        # FlashAttention은 보통 fp16/bf16만 안정적으로 지원하는 경우가 많음
        comp_dtype = self._get_compute_dtype(x_unpad)
        if comp_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"FlashAttention compute dtype must be fp16/bf16. got {comp_dtype}. "
                "autocast(bf16/fp16) 켜거나 모델/입력을 half/bfloat16으로 맞춰주세요."
            )

        # qkv: (T, 3H) -> (T, 3, heads, head_dim)
        qkv = self.qkv(x_unpad)  # (T, 3H)
        qkv = qkv.view(T, 3, self.heads, self.head_dim).to(dtype=comp_dtype)

        cu = cu_seqlens.to(device=qkv.device, dtype=torch.int32)

        drop_p = self.attn_drop if self.training else 0.0

        # out: (T, heads, head_dim)
        out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu,
            max_seqlen=max_seqlen_int,
            dropout_p=float(drop_p),
            softmax_scale=None,  # 기본 1/sqrt(head_dim)
            causal=False,
        )

        # (T, H)
        out = out.reshape(T, H)

        # out_proj + dropout (dtype는 입력과 맞춤)
        out = self.out_proj(out.to(dtype=x_unpad.dtype))
        out = self.proj_drop(out)

        # pad back: (B, P, H)
        out = pad_input(out, indices, B, P)
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class TopKScoreBiasedCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim: int,
        heads: int,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by heads. hidden_dim={hidden_dim}, heads={heads}"
            )

        self.topk_k: int = int(getattr(config, "ca_topk_k", 64))
        self.s_d: float = float(getattr(config, "ca_topk_s_d", 10.0))

        neg_cfg: float = float(getattr(config, "ca_topk_neg", -1e9))
        self.neg_value: float = min(neg_cfg, -1e9)

        self.eps: float = float(getattr(config, "ca_topk_eps", 1e-6))
        self.agent_chunk_size: int = int(getattr(config, "ca_topk_agent_chunk_size", 64))

        self.heads = int(heads)
        self.head_dim = int(hidden_dim // heads)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_drop = nn.Dropout(float(proj_drop))

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,  # (..., K)
        mask: torch.Tensor,    # (..., K) True=가릴 위치
        dim: int = -1,
    ) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        logits_masked = logits.masked_fill(mask, float("-inf"))
        attn = torch.softmax(logits_masked, dim=dim)
        attn = attn.masked_fill(mask, 0.0)
        return attn

    @staticmethod
    def _choose_distance_dtype(query: torch.Tensor) -> torch.dtype:
        """거리 계산에 사용할 dtype을 고릅니다.

        - query가 bf16/fp16이면 그 dtype을 사용해 메모리를 줄입니다.
        - 그 외에는 float32로 둡니다.

        Args:
            query (torch.Tensor): (B, P, H)

        Returns:
            torch.dtype: 거리 계산 dtype
        """
        if query.dtype in (torch.float16, torch.bfloat16):
            return query.dtype
        return torch.float32

    def forward(
        self,
        *,
        query: torch.Tensor,  # (B, P, H)
        agent_pose4: torch.Tensor,  # (B, P, 4)
        target_current_mask: torch.Tensor,  # (B, P) True=pad
        scene_encoding_token: torch.Tensor,  # (B, token_num, H)
        scene_encoding_token_mask: torch.Tensor,  # (B, token_num) True=pad
        scene_encoding_pos_2d: torch.Tensor,  # (B, token_num, 9)
        score_beta: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError(f"query must be (B,P,H). got {tuple(query.shape)}")
        if agent_pose4.dim() != 3 or int(agent_pose4.shape[-1]) != 4:
            raise ValueError(f"agent_pose4 must be (B,P,4). got {tuple(agent_pose4.shape)}")
        if scene_encoding_token.dim() != 3:
            raise ValueError(
                f"scene_encoding_token must be (B,token_num,H). got {tuple(scene_encoding_token.shape)}"
            )
        if scene_encoding_pos_2d.dim() != 3 or int(scene_encoding_pos_2d.shape[-1]) != 9:
            raise ValueError(
                f"scene_encoding_pos_2d must be (B,token_num,9). got {tuple(scene_encoding_pos_2d.shape)}"
            )

        if target_current_mask.dtype != torch.bool:
            target_current_mask = target_current_mask.to(torch.bool)
        if scene_encoding_token_mask.dtype != torch.bool:
            scene_encoding_token_mask = scene_encoding_token_mask.to(torch.bool)

        B, P, H = query.shape
        token_num: int = int(scene_encoding_token.shape[1])

        if token_num <= 0 or self.topk_k <= 0:
            return query.new_zeros((B, P, H))

        K: int = min(int(self.topk_k), token_num)
        s_d: float = max(float(self.s_d), 1e-6)
        NEG: float = float(self.neg_value)

        # (핵심) 거리 계산 dtype: 가능하면 bf16/fp16 사용
        dist_dtype: torch.dtype = self._choose_distance_dtype(query)

        # token_xy: (B, token_num, 2)
        token_xy: torch.Tensor = scene_encoding_pos_2d[:, :, 0:2].to(
            device=query.device, dtype=dist_dtype
        )
        token_x: torch.Tensor = token_xy[:, :, 0]  # (B, token_num)
        token_y: torch.Tensor = token_xy[:, :, 1]  # (B, token_num)

        out_all = query.new_zeros((B, P, H))  # (B, P, H)
        chunk: int = max(1, int(self.agent_chunk_size))
        beta_f32: torch.Tensor = score_beta.to(device=query.device, dtype=torch.float32)

        for start in range(0, P, chunk):
            end = min(P, start + chunk)
            q_chunk = query[:, start:end, :]                    # (B, Pc, H)
            pose_chunk = agent_pose4[:, start:end, :]           # (B, Pc, 4)
            agent_pad_chunk = target_current_mask[:, start:end] # (B, Pc) True=pad
            Pc = int(end - start)

            # agent_xy: (B, Pc, 2)
            agent_xy = pose_chunk[..., 0:2].to(device=query.device, dtype=dist_dtype)
            agent_x = agent_xy[..., 0]  # (B, Pc)
            agent_y = agent_xy[..., 1]  # (B, Pc)

            # (B, Pc, token_num)
            dx = token_x.unsqueeze(1) - agent_x.unsqueeze(2)
            dy = token_y.unsqueeze(1) - agent_y.unsqueeze(2)

            # dist: (B, Pc, token_num)
            dist = torch.sqrt(dx * dx + dy * dy + float(self.eps))

            # score_for_topk: (B, Pc, token_num)
            # - invalid는 -inf로 두고, topk에서 제외
            invalid_score_mask = scene_encoding_token_mask.unsqueeze(1) | agent_pad_chunk.unsqueeze(-1)
            score_for_topk = -(dist / s_d)
            score_for_topk = score_for_topk.masked_fill(invalid_score_mask, float("-inf"))

            # topk_idx: (B, Pc, K)
            topk_idx = torch.topk(score_for_topk, k=K, dim=-1, largest=True).indices

            # score_keep: (B, Pc, K)
            # - 예전 코드의 "NEG 채우기" 의미를 유지하기 위해 -inf는 NEG로 바꿉니다.
            score_keep = torch.gather(score_for_topk, dim=-1, index=topk_idx)
            score_keep = torch.clamp(score_keep, min=NEG)

            # sel_pad_mask: (B, Pc, K)
            sel_pad_mask = scene_encoding_token_mask.unsqueeze(1).expand(B, Pc, token_num).gather(dim=2, index=topk_idx)

            # kv_sel: (B, Pc, K, H)
            scene_exp = scene_encoding_token.unsqueeze(1).expand(B, Pc, token_num, H)
            idx_exp = topk_idx.unsqueeze(-1).expand(B, Pc, K, H)
            kv_sel = torch.gather(scene_exp, dim=2, index=idx_exp)
            kv_sel = kv_sel.masked_fill(sel_pad_mask.unsqueeze(-1), 0.0)

            q_lin = self.q_proj(q_chunk)  # (B, Pc, H)
            k_lin = self.k_proj(kv_sel)   # (B, Pc, K, H)
            v_lin = self.v_proj(kv_sel)   # (B, Pc, K, H)

            qh = q_lin.view(B, Pc, self.heads, self.head_dim).permute(0, 2, 1, 3)        # (B,h,Pc,hd)
            kh = k_lin.view(B, Pc, K, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,h,Pc,K,hd)
            vh = v_lin.view(B, Pc, K, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B,h,Pc,K,hd)

            # logits는 softmax 안정 위해 float32 유지
            logits = (qh.to(torch.float32).unsqueeze(3) * kh.to(torch.float32)).sum(dim=-1) * float(self.scale)  # (B,h,Pc,K)
            logits = logits + beta_f32 * score_keep.to(torch.float32).unsqueeze(1)  # (B,h,Pc,K)

            attn = self._masked_softmax(logits, mask=sel_pad_mask.unsqueeze(1), dim=-1)  # (B,h,Pc,K)
            out = (attn.unsqueeze(-1) * vh.to(torch.float32)).sum(dim=3)  # (B,h,Pc,hd)

            out = out.permute(0, 2, 1, 3).contiguous().view(B, Pc, H).to(dtype=q_chunk.dtype)  # (B,Pc,H)
            out = self.out_proj(out)
            out = self.proj_drop(out)

            out = out.masked_fill(agent_pad_chunk.unsqueeze(-1), 0.0)
            out_all[:, start:end, :] = out

        return out_all

class DiTBlock(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim: int,
        heads: int,
        dropout: float,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(hidden_dim)
        self.norm_ca = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.sa = SimpleSelfAttention(hidden_dim=hidden_dim, heads=heads)
        self.ca = TopKScoreBiasedCrossAttention(config=config, hidden_dim=hidden_dim, heads=heads)

        mlp_hidden = int(hidden_dim * float(mlp_ratio))
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        drop_path_rate = float(dropout)
        self.drop_path_sa = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path_ca = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    @staticmethod
    def _apply_modulation(
        x_norm: torch.Tensor,  # (B,P,H)
        mod: ModulationTriplet,  # delta_scale/shift/gate: (B,P,H)
    ) -> torch.Tensor:
        return x_norm * (1.0 + mod.delta_scale) + mod.shift

    def forward(
        self,
        *,
        x: torch.Tensor,  # (B, P, H)
        target_current_mask: torch.Tensor,  # (B, P) True=pad
        cross_c: torch.Tensor,  # (B, token_num, H)
        cross_mask: torch.Tensor,  # (B, token_num) True=pad
        cross_pos_2d: torch.Tensor,  # (B, token_num, 9)
        agent_pose4: torch.Tensor,  # (B, P, 4)
        ca_score_beta: torch.Tensor,  # scalar
        pram_v2_modulations: Dict[str, ModulationTriplet],
    ) -> torch.Tensor:
        """SA + CA(topK+bias) + FFN 블록.

        Returns:
            x_out:
                shape: (B, P, H)
        """
        if target_current_mask.dtype != torch.bool:
            target_current_mask = target_current_mask.to(torch.bool)

        # ---- SA ----
        sa_mod = pram_v2_modulations["SA"]
        h = self.norm_sa(x)
        h = self._apply_modulation(h, sa_mod)  # (B,P,H)
        sa_out = self.sa(h, key_padding_mask=target_current_mask)  # (B,P,H)
        sa_out = sa_out * sa_mod.gate
        x = x + self.drop_path_sa(sa_out)

        # ---- CA ----
        ca_mod = pram_v2_modulations["CA"]
        h = self.norm_ca(x)
        h = self._apply_modulation(h, ca_mod)  # (B,P,H)
        ca_out = self.ca(
            query=h,
            agent_pose4=agent_pose4,
            target_current_mask=target_current_mask,
            scene_encoding_token=cross_c,
            scene_encoding_token_mask=cross_mask,
            scene_encoding_pos_2d=cross_pos_2d,
            score_beta=ca_score_beta,
        )  # (B,P,H)
        ca_out = ca_out * ca_mod.gate
        x = x + self.drop_path_ca(ca_out)

        # ---- FFN ----
        ffn_mod = pram_v2_modulations["FFN"]
        h = self.norm_ffn(x)
        h = self._apply_modulation(h, ffn_mod)
        ffn_out = self.mlp(h)
        ffn_out = ffn_out * ffn_mod.gate
        x = x + self.drop_path_ffn(ffn_out)

        return x

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
        self.norm_dit_returns = None
        self.diffusion_sequence_flat = None
        self.config = config
        self._future_len: int = config.future_len  # <추가하자>
        self._time_len: int = config.time_len  # <추가하자>
        self._past_len = self._time_len - 1

        assert model_type in ["score", "v",
                              "x_start"], f"Unknown model type: {model_type}"
        self.final_hidden_tokens = None
        if self.config.use_feasible:
            self.feasible_projector = FeasibleProjector(
                self.config, hidden_dim, self.config.use_feasible_dl,
                self.config.use_feasible_filter)
            self.feasible_projector.enable_profile = self.config.profile_feasible

        self._model_type = model_type
        # if self.config.use_amortized_diffusion:
        if self.config.pose_based:
            output_dim_pre = int(output_dim / 4 * 6)
        else:
            output_dim_pre = int(output_dim / 3 * 5)
        # else:
        #     output_dim_pre = output_dim
        self.preproj = Mlp(
            in_features=output_dim_pre,
            # x, y, cos(yaw), sin(yaw), noising timestep, validity
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.)
        # [추가] 현재 상태 토큰(state_token_in)을 에이전트 토큰(preproj 출력)에 직접 더하기 위한 레이어
        # - 0 초기화: 처음에는 기존 출력/동작과 완전히 동일하게 시작
        self.state_token_injector = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.state_token_injector.weight)
        if self.state_token_injector.bias is not None:
            nn.init.zeros_(self.state_token_injector.bias)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(config, hidden_dim, heads, dropout, mlp_ratio)
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
        # [추가] pose_based=False일 때만:
        # 현재 motion(마지막 과거 구간 control: vxᵇ, vyᵇ, yaw_rate)을 state_token_in에 더하기 위한 프로젝션
        # - weight를 0으로 시작하면, 학습 시작 시점에는 기존과 완전히 동일(추가 토큰이 항상 0)
        # - bias를 없애면, motion 정보가 0일 때 항상 0이 더해져서 더 보수적으로 동작
        self.pram_v2_motion_proj: Optional[nn.Linear] = None
        if not self.config.pose_based:
            self.pram_v2_motion_proj = nn.Linear(3, hidden_dim, bias=False)
            nn.init.zeros_(self.pram_v2_motion_proj.weight)

        # 블록×경로 토글 스칼라 및 게이트 편향.
        # ---- CA 게이트 초기 오픈(기본 0.25) ----
        # - config로 조절 가능:
        #   - pram_ca_gate_init_prob: float (기본 0.25)
        ca_gate_init_prob: float = float(
            getattr(config, "pram_ca_gate_init_prob", 0.25))

        # 초기 base logit(대략 -6)을 "실제 모듈 바이어스"에서 읽어 계산
        # - time 모듈 바이어스(보통 -3) + composer head 바이어스(보통 -3)
        time_bias: float = float(
            self.pram_v2_time_mod.lin_logit_gate.bias.detach().view(-1)[
                0].item())
        base_bias: float = float(
            self.pram_v2_composer.head_logit_gate.bias.detach().view(-1)[
                0].item())
        base_gate_logit_bias: float = time_bias + base_bias  # 보통 -6.0

        self.pram_v2_block_path_scalars = PRAMV2BlockPathScalars(
            depth=depth,
            ca_gate_init_prob=ca_gate_init_prob,
            base_gate_logit_bias=base_gate_logit_bias,
        )

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
        self.ca_score_beta = nn.Parameter(torch.tensor(0.0))

    from typing import Optional



    def _get_ca_score_beta_for_cross_attention(self) -> torch.Tensor:
        """Cross-Attention에서 사용할 거리 가중치 beta를 안전하게 만듭니다.

        이 beta는 "거리 점수(score_keep)"에 곱해져 attention 점수(logits)에 더해집니다.
        - score_keep는 거리가 멀수록 더 작은 값(더 음수)입니다.
        - beta가 음수가 되면, 멀리 있는 토큰이 오히려 유리해질 수 있습니다.

        이 함수는 설정에 따라 beta를 항상 0 이상이 되도록 바꿔 반환합니다.

        Returns:
            torch.Tensor:
                beta 값 텐서
                - shape: ()
                - dtype: self.ca_score_beta.dtype
                - device: self.ca_score_beta.device
        """
        force_nonneg: bool = bool(
            getattr(self.config, "ca_score_beta_force_nonneg", True)
        )
        if not force_nonneg:
            return self.ca_score_beta  # shape: ()

        # softplus의 샤프니스(클수록 0 근처 출력이 더 작아져, 초기 동작이 거의 0과 비슷해집니다)
        softplus_beta: float = float(
            getattr(self.config, "ca_score_beta_softplus_beta", 1000.0)
        )
        if (not math.isfinite(softplus_beta)) or (softplus_beta <= 0.0):
            softplus_beta = 1000.0

        # 항상 0 이상
        return F.softplus(self.ca_score_beta, beta=softplus_beta)  # shape: ()

    def _build_ca_gate_type_logit_bias(
            self,
            *,
            target_current_11_dim: torch.Tensor,  # (B, P, 11)
            target_current_mask: torch.Tensor,  # (B, P) True=pad
            reference_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """에이전트 타입에 따라 CA 게이트(logit)에 더할 bias를 (B,P,1)로 만듭니다."""
        enabled: bool = bool(
            getattr(self.config, "pram_ca_type_gate_bias_enabled", True))
        if not enabled:
            return None

        if target_current_11_dim.dim() != 3 or int(
                target_current_11_dim.shape[-1]) != 11:
            raise ValueError(
                f"target_current_11_dim must be (B,P,11). got {tuple(target_current_11_dim.shape)}")

        if target_current_mask.dtype != torch.bool:
            target_current_mask = target_current_mask.to(torch.bool)

        type_one_hot: torch.Tensor = target_current_11_dim[..., 8:11].to(
            torch.float32)  # (B,P,3)

        veh_b: float = float(
            getattr(self.config, "pram_ca_vehicle_gate_logit_bias", 1.0))
        ped_b: float = float(
            getattr(self.config, "pram_ca_ped_gate_logit_bias", 0.0))
        bik_b: float = float(
            getattr(self.config, "pram_ca_bike_gate_logit_bias", 0.0))

        bias_vec = torch.tensor([veh_b, ped_b, bik_b],
                                device=type_one_hot.device,
                                dtype=torch.float32)  # (3,)
        bias_bp = (type_one_hot * bias_vec.view(1, 1, 3)).sum(dim=-1)  # (B,P)

        # pad agent는 0
        bias_bp = bias_bp.masked_fill(target_current_mask, 0.0)

        return bias_bp.unsqueeze(-1).to(device=reference_tensor.device,
                                        dtype=reference_tensor.dtype)  # (B,P,1)

    def _get_time_embedding(self, diffusion_time: torch.Tensor,
                            ref: torch.Tensor) -> torch.Tensor:
        """
        diffusion_time:
          - (B,)          : 일반 확산 시간
          - (B, future_len): amortized 스케줄 시간
        ref: dtype/device 기준 텐서 (보통 x)
        returns: (B, H)
        """
        if diffusion_time.ndim == 1:
            # (B,) -> (B,H)
            t = diffusion_time.to(device=ref.device, dtype=torch.float32)
            return self.t_embedder(t).to(dtype=ref.dtype, device=ref.device)
        if diffusion_time.ndim == 2:
            # (B,T) -> 대표 스칼라 (B,) -> (B,H)
            t = diffusion_time.to(device=ref.device, dtype=torch.float32)
            B, T = t.shape
            if T != self._future_len:
                raise ValueError(
                    f"diffusion_time shape mismatch: expected (B,{self._future_len}), got {tuple(t.shape)}"
                )
            # ✅ 평균 노이즈 수준을 대표값으로 사용
            t_global = t.mean(dim=1)  # (B,)
            return self.t_embedder(t_global).to(dtype=ref.dtype,
                                                device=ref.device)

        raise ValueError(
            f"diffusion_time must be (B,) or (B,T). got {tuple(diffusion_time.shape)}"
        )

    def _select_tensors_for_feasible_projection(
        self,
        x: torch.Tensor,  # (B, P, F)
        target_current_xyyaw: torch.Tensor,  # (B, P, 4)
        target_past_cur_future_valid: torch.Tensor,  # (B, P, T_all)  True=유효
        target_current_control: Optional[
            torch.Tensor] = None,  # (B, P, 3) vxᵇ, vyᵇ, yaw_rate
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        """FeasibleProjector에 넘길 텐서들을 '정책' 기준으로만 정리합니다.

        - detach 여부/float32 변환/valid(bool) 정리만 수행합니다.

        Args:
            x (torch.Tensor):
                DiT 출력(flat).
                shape: (B, P, F)
            target_current_xyyaw (torch.Tensor):
                현재 포즈(정규화).
                shape: (B, P, 4)
            target_past_cur_future_valid (torch.Tensor):
                노드(valid) 마스크(True=유효).
                shape: (B, P, T_all)
            target_current_control (torch.Tensor, optional):
                현재 제어(정규화). pose_based=False일 때 사용.
                shape: (B, P, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x_for_feasible:
                    Feasible에 넘길 x(flat).
                    shape: (B, P, F), dtype=float32
                target_current_xyyaw_for_feasible:
                    현재 포즈.
                    shape: (B, P, 4), dtype=float32
                target_past_cur_future_valid_for_feasible:
                    노드(valid) 마스크(True=유효).
                    shape: (B, P, T_all), dtype=bool
                target_current_control_for_feasible:
                    현재 제어. pose_based=False일 때 사용.
                    shape: (B, P, 3), dtype=float32

        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B,P,F). got {tuple(x.shape)}")

        # valid 마스크: bool + x.device로 통일
        valid_bool = _to_bool_mask(target_past_cur_future_valid)
        if valid_bool.device != x.device:
            valid_bool = valid_bool.to(device=x.device)

        if bool(getattr(self.config, "feasible_grad_to_dit", False)):
            x_for_feasible = x.to(dtype=torch.float32)
            target_current_xyyaw_for_feasible = target_current_xyyaw.to(
                device=x.device, dtype=torch.float32)
            target_past_cur_future_valid_for_feasible = valid_bool
        else:
            x_for_feasible = x.detach().to(dtype=torch.float32)
            target_current_xyyaw_for_feasible = target_current_xyyaw.detach(
            ).to(device=x.device, dtype=torch.float32)
            target_past_cur_future_valid_for_feasible = valid_bool.detach()

        # target_current_control
        target_current_control_for_feasible = None
        if target_current_control is not None:
            if not bool(getattr(self.config, "feasible_grad_to_dit", False)):
                target_current_control_for_feasible = target_current_control.detach(
                ).to(device=x.device, dtype=torch.float32)
            else:
                target_current_control_for_feasible = target_current_control.to(
                    device=x.device, dtype=torch.float32)

        return (
            x_for_feasible,  # (B,P,F) float32
            target_current_xyyaw_for_feasible,  # (B,P,4) float32
            target_past_cur_future_valid_for_feasible,  # (B,P,T_all) bool
            target_current_control_for_feasible,  # (B,P,3) float32 or None
        )

    def _build_diffusion_inputs_for_feasible_projection(
            self,
            x_for_feasible: torch.Tensor,  # (B, P, F) float32
            target_current_xyyaw_for_feasible: torch.
        Tensor,  # (B, P, 4) float32
            target_past_cur_future_valid_for_feasible: torch.
        Tensor,  # (B, P, T_all) bool
    ) -> torch.Tensor:
        """FeasibleProjector에 넣을 입력을 모드에 맞게 만듭니다(단일 함수).

        - pose_based=True:
            (현재+미래) pose 궤적을 만듭니다.
            반환 shape: (B, P, 1+T, 4)
        - pose_based=False:
            미래 control만 뽑고(길이 T),
            노드 valid로부터 구간 valid를 만들어 무효 구간을 0으로 만듭니다.
            반환 shape: (B, P, T, 3)

        Args:
            x_for_feasible (torch.Tensor):
                DiT 출력(flat).
                shape: (B, P, F), dtype=float32
            target_current_xyyaw_for_feasible (torch.Tensor):
                현재 포즈(정규화).
                shape: (B, P, 4), dtype=float32
            target_past_cur_future_valid_for_feasible (torch.Tensor):
                노드(valid) 마스크(True=유효).
                shape: (B, P, T_all), dtype=bool

        Returns:
            torch.Tensor:
                pose_based=True  -> (B, P, 1+T, 4)
                pose_based=False -> (B, P, T, 3)
        """
        if x_for_feasible.dim() != 3:
            raise ValueError(
                f"x_for_feasible must be 3D (B,P,F). got {tuple(x_for_feasible.shape)}"
            )

        B: int = int(x_for_feasible.shape[0])
        P: int = int(x_for_feasible.shape[1])
        future_len: int = int(self._future_len)

        if self.config.pose_based:
            # -------- pose_based=True: 기존 로직 그대로 --------
            if int(x_for_feasible.shape[-1]) % 4 != 0:
                raise ValueError(
                    f"pose_based=True expects last dim multiple of 4. got F={int(x_for_feasible.shape[-1])}"
                )

            if self.config.use_past_dit_input:
                # x_for_feasible: (B,P, (time_len+T)*4) -> (B,P,time_len+T,4) -> 마지막 1+T만
                diffusion_sequence = x_for_feasible.reshape(B, P, -1,
                                                            4).contiguous()
                diffusion_sequence = diffusion_sequence[:, :,
                                                        -(future_len +
                                                          1):, :]  # (B,P,1+T,4)
                diffusion_sequence[:, :,
                                   0, :] = target_current_xyyaw_for_feasible
                return diffusion_sequence.contiguous()

            if self.config.use_current_input:
                # x_for_feasible: (B,P,(1+T)*4) -> (B,P,1+T,4)
                diffusion_sequence = x_for_feasible.reshape(B, P, -1,
                                                            4).contiguous()
                diffusion_sequence[:, :,
                                   0, :] = target_current_xyyaw_for_feasible
                return diffusion_sequence.contiguous()

            # use_current_input=False: x_for_feasible는 미래만 (B,P,T*4)
            diffusion_future = x_for_feasible.reshape(B, P, -1, 4)  # (B,P,T,4)
            diffusion_sequence = torch.cat(
                [
                    target_current_xyyaw_for_feasible.unsqueeze(2),
                    diffusion_future
                ],
                dim=2,
            ).contiguous()  # (B,P,1+T,4)
            return diffusion_sequence

        # -------- pose_based=False: future control 만들기 --------
        if int(x_for_feasible.shape[-1]) % 3 != 0:
            raise ValueError(
                f"pose_based=False expects last dim multiple of 3. got F={int(x_for_feasible.shape[-1])}"
            )

        F: int = int(x_for_feasible.shape[-1])
        T_any: int = int(F // 3)
        if T_any < future_len:
            raise ValueError("pose_based=False: x 길이가 future_len보다 짧습니다. "
                             f"T_any={T_any}, future_len={future_len}")

        # (B,P,T_any,3) -> 마지막 T만
        x_seq = x_for_feasible.reshape(B, P, T_any, 3)
        diffusion_control_traj = x_seq[:, :, -future_len:, :].contiguous(
        )  # (B,P,T,3)

        # 노드 valid(현재+미래) -> 구간 valid(T)
        valid_bool = target_past_cur_future_valid_for_feasible
        if valid_bool.dtype != torch.bool:
            valid_bool = _to_bool_mask(valid_bool)
        if valid_bool.device != diffusion_control_traj.device:
            valid_bool = valid_bool.to(device=diffusion_control_traj.device)

        if int(valid_bool.shape[-1]) < (future_len + 1):
            raise ValueError(
                "target_past_cur_future_valid 길이가 future_len+1보다 짧습니다. "
                f"valid_len={int(valid_bool.shape[-1])}, future_len={future_len}"
            )

        cur_future_valid = valid_bool[:, :, -(future_len + 1):]  # (B,P,1+T)
        seg_valid = cur_future_valid[:, :, :
                                     -1] & cur_future_valid[:, :, 1:]  # (B,P,T)

        # 무효 구간은 0
        diffusion_control_traj = diffusion_control_traj.masked_fill(
            ~seg_valid.unsqueeze(-1), 0.0)
        return diffusion_control_traj

    def _compute_feasible_low_t_mask(
            self,
            diffusion_time: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """현재 시간 값으로 '저노이즈 배치' 마스크를 만든다.

        의미(중요)
        ----------
        - pose_based=True:
            low_t_mask==True 인 배치만 FeasibleProjector 경로(보정/필터/적분)를 실제로 실행합니다.
        - pose_based=False:
            filter_and_integrate(적분)는 항상 수행하고,
            low_t_mask==True 인 배치에만 "제약(필터)"를 적용합니다.

        Args:
            diffusion_time: 확산 시간. shape: (B,)

        Returns:
            low_t_mask:
                pose_based=True  -> feasible 전체 실행 배치 마스크
                pose_based=False -> 제약(필터) 적용 배치 마스크
                shape: (B,)
        """
        assert diffusion_time.ndim == 1, \
            f"diffusion_time must be (B,), got {tuple(diffusion_time.shape)}"
        t_threshold: float = float(self.config.feasible_learn_noise_thresh)
        low_t_mask: torch.Tensor = (diffusion_time <= t_threshold)  # (B,)
        return low_t_mask

    @property
    def model_type(self):
        return self._model_type

    def _extract_last_past_segment_control_for_pram(
        self,
        target_input_norm_xT: torch.
        Tensor,  # (B, P, T_any*5)  (pose_based=False 기준)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """pose_based=False에서 PRAM에 넣을 '현재 motion'을 뽑습니다.

        여기서 '현재 motion'은 "현재 시점 바로 직전 구간"의 control(vxᵇ, vyᵇ, yaw_rate) 입니다.
        즉, past_seg_control_gt_3_dim[..., -1, :]과 같은 의미입니다.

        Args:
            target_input_norm_xT (torch.Tensor):
                DiT에 들어가는 flat 입력.
                pose_based=False일 때는 한 구간이 아래 5개 값으로 구성됩니다.
                  - control 3개: (vxᵇ, vyᵇ, yaw_rate)
                  - 노이즈 시간값 1개
                  - 유효 여부 1개(0/1)
                shape: (B, P, T_any*5)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                cur_motion_norm:
                    마지막 과거 구간 control(정규화된 값).
                    shape: (B, P, 3)
                cur_motion_valid:
                    그 구간이 유효하면 True.
                    shape: (B, P) bool
        """
        if target_input_norm_xT.dim() != 3:
            raise ValueError(
                f"target_input_norm_xT must be 3D (B,P,F). got {tuple(target_input_norm_xT.shape)}"
            )

        B, P, F = target_input_norm_xT.shape
        step_dim: int = 5  # (vxᵇ, vyᵇ, yaw_rate, 노이즈시간, 유효여부)

        if F % step_dim != 0:
            raise ValueError(
                f"pose_based=False expects last dim multiple of {step_dim}. got F={F}"
            )

        T_any: int = int(F // step_dim)
        future_len: int = int(self._future_len)
        past_len: int = int(T_any - future_len)

        # 과거 구간이 없으면(=현재 motion을 정의할 수 없으면) 0으로 반환
        if past_len <= 0:
            zeros = target_input_norm_xT.new_zeros((B, P, 3))
            valid = torch.zeros(
                (B, P),
                device=target_input_norm_xT.device,
                dtype=torch.bool,
            )
            return zeros, valid

        last_past_index: int = int(past_len - 1)

        x_seq = target_input_norm_xT.reshape(B, P, T_any, step_dim)
        cur_motion_norm = x_seq[:, :, last_past_index, 0:3]  # (B,P,3)
        cur_motion_valid = x_seq[:, :, last_past_index, 4] > 0.5  # (B,P) bool

        return cur_motion_norm, cur_motion_valid

    def _run_dit_core_with_pram_v2(
            self,
            target_input_norm_xT: torch.Tensor,  # (B, (1+)Pnn, _ * 6 or 5)
            diffusion_time: torch.Tensor,
            cross_c: torch.Tensor,  # (B, token_num, H)
            cross_mask: torch.Tensor,  # (B, token_num) True=pad
            cross_pos_2d: torch.Tensor,  # (B, token_num, 9)
            target_current_11_dim: torch.Tensor,  # (B, P, 11)
            target_current_valid: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """DiT 본체(프리프로젝션 + PRAM-v2 블록 + 최종 투영)를 수행합니다.

        변경점(핵심)
        ----------
        - FlashAttention/unpad/pad/KV-cache/packed 경로 제거
        - cross-attn은 "거리 기반 top-K + score bias(beta)"만 사용

        Returns:
            x_out:
                shape: (B, P, output_dim)
        """
        target_current_mask = (
            ~target_current_valid.to(torch.bool))  # (B,P) True=pad
        if target_current_mask.dtype != torch.bool:
            target_current_mask = target_current_mask.to(torch.bool)


        B, P, _ = target_input_norm_xT.shape
        depth: int = int(len(self.blocks))

        # 1) pre-proj (B,P,H)
        x: torch.Tensor = self.preproj(target_input_norm_xT)
        x = x.masked_fill(target_current_mask.unsqueeze(-1), 0.0)
        # 2) cross 입력 정리
        cross_c = _cast_like(cross_c, x)
        cross_mask = _to_bool_mask(cross_mask).to(device=x.device)  # True=pad
        cross_pos_2d = cross_pos_2d.to(device=x.device,
                                       dtype=torch.float32)  # dist 계산용


        # 3) timestep embedding
        t_embedding: torch.Tensor = self._get_time_embedding(diffusion_time,
                                                             ref=x)

        # 4) state_token_in (B,P,H)
        state_token_in: torch.Tensor = self.pram_v2_state_token_encoder(
            target_cur_norm=target_current_11_dim.to(dtype=x.dtype,
                                                     device=x.device),
            target_current_mask=target_current_mask,
        )

        # pose_based=False일 때 motion token 추가(기존 로직 유지)
        if self.pram_v2_motion_proj is not None:
            cur_motion_norm, cur_motion_valid = self._extract_last_past_segment_control_for_pram(
                target_input_norm_xT=target_input_norm_xT,
            )
            cur_motion_norm = _cast_like(cur_motion_norm, state_token_in)
            cur_motion_valid_f = cur_motion_valid.to(
                dtype=cur_motion_norm.dtype,
                device=cur_motion_norm.device).unsqueeze(-1)  # (B,P,1)
            cur_motion_norm = cur_motion_norm * cur_motion_valid_f

            motion_token = self.pram_v2_motion_proj(cur_motion_norm)  # (B,P,H)
            motion_token = motion_token.masked_fill(
                target_current_mask.unsqueeze(-1), 0.0)
            motion_token = _cast_like(motion_token, state_token_in)
            state_token_in = state_token_in + motion_token

        # 5) state_token_in을 agent 토큰에 주입(0-init Linear라 초기에는 영향 0)
        state_add = self.state_token_injector(state_token_in)
        state_add = _cast_like(state_add, x)
        x = x + state_add
        x = x.masked_fill(target_current_mask.unsqueeze(-1), 0.0)

        # 6) composer + time modulation
        composer_out = self.pram_v2_composer(
            state_token_in=state_token_in,
            target_current_mask=target_current_mask,
        )
        time_out = self.pram_v2_time_mod(t_embedding)

        # 7) 블록별 PRAM 모듈레이션 준비(일반(B,P,H))
        ds_base = composer_out.delta_scale_base.to(dtype=x.dtype,
                                                   device=x.device)  # (B,P,H)
        sh_base = composer_out.shift_base.to(dtype=x.dtype,
                                             device=x.device)  # (B,P,H)
        lg_base = composer_out.logit_gate_base.to(dtype=x.dtype,
                                                  device=x.device)  # (B,P,H)

        ds_time = time_out.delta_scale_time.to(dtype=x.dtype,
                                               device=x.device)  # (B,1,H)
        sh_time = time_out.shift_time.to(dtype=x.dtype,
                                         device=x.device)  # (B,1,H)
        lg_time = time_out.logit_gate_time.to(dtype=x.dtype,
                                              device=x.device)  # (B,1,H)

        # (B,1,H) -> (B,P,H)
        ds_time = ds_time.expand(B, P, -1).contiguous()
        sh_time = sh_time.expand(B, P, -1).contiguous()
        lg_time = lg_time.expand(B, P, -1).contiguous()

        # 타입별 CA gate bias (B,P,1)
        ca_type_bias = self._build_ca_gate_type_logit_bias(
            target_current_11_dim=target_current_11_dim,
            target_current_mask=target_current_mask,
            reference_tensor=x,
        )

        # agent_pose4: (B,P,4)
        agent_pose4 = target_current_11_dim[..., 0:4]  # (B,P,4)

        # path scalars: (depth,3)
        k_s = self.pram_v2_block_path_scalars.k_s.to(dtype=x.dtype,
                                                     device=x.device)
        k_sh = self.pram_v2_block_path_scalars.k_sh.to(dtype=x.dtype,
                                                       device=x.device)
        k_g = self.pram_v2_block_path_scalars.k_g.to(dtype=x.dtype,
                                                     device=x.device)
        beta_g = self.pram_v2_block_path_scalars.beta_g.to(dtype=x.dtype,
                                                           device=x.device)
        ca_score_beta_for_ca: torch.Tensor = self._get_ca_score_beta_for_cross_attention()  # shape: ()

        for block_index, block in enumerate(self.blocks):
            # SA mods
            ds_sa = ds_time + ds_base * k_s[block_index, 0]
            sh_sa = sh_time + sh_base * k_sh[block_index, 0]
            gate_sa = torch.sigmoid(
                lg_time + lg_base * k_g[block_index, 0] + beta_g[
                    block_index, 0])

            # FFN mods
            ds_ffn = ds_time + ds_base * k_s[block_index, 1]
            sh_ffn = sh_time + sh_base * k_sh[block_index, 1]
            gate_ffn = torch.sigmoid(
                lg_time + lg_base * k_g[block_index, 1] + beta_g[
                    block_index, 1])

            # CA mods (+ type bias)
            ds_ca = ds_time + ds_base * k_s[block_index, 2]
            sh_ca = sh_time + sh_base * k_sh[block_index, 2]
            gate_ca_logit = lg_time + lg_base * k_g[block_index, 2] + beta_g[
                block_index, 2]
            if ca_type_bias is not None:
                gate_ca_logit = gate_ca_logit + ca_type_bias  # (B,P,H)로 broadcast
            gate_ca = torch.sigmoid(gate_ca_logit)

            pram_mods = {
                "SA": ModulationTriplet(delta_scale=ds_sa, shift=sh_sa,
                                        gate=gate_sa),
                "FFN": ModulationTriplet(delta_scale=ds_ffn, shift=sh_ffn,
                                         gate=gate_ffn),
                "CA": ModulationTriplet(delta_scale=ds_ca, shift=sh_ca,
                                        gate=gate_ca),
            }

            x = block(
                x=x,  # (B,P,H)
                target_current_mask=target_current_mask,  # (B,P)
                cross_c=cross_c,  # (B,token_num,H)
                cross_mask=cross_mask,  # (B,token_num)
                cross_pos_2d=cross_pos_2d,  # (B,token_num,9)
                agent_pose4=agent_pose4,  # (B,P,4)
                ca_score_beta=ca_score_beta_for_ca,  # ✅ (shape: ())
                pram_v2_modulations=pram_mods,
            )

        # feasible용 저장
        if getattr(self.config, "feasible_grad_to_dit", False):
            self.final_hidden_tokens = x.float()
        else:
            self.final_hidden_tokens = x.detach().clone().float()

        # 9) PRAM-v2 final layer
        x_out = apply_pram_v2_final_layer(
            x=x,
            composer_out=composer_out,
            time_out=time_out,
            final_norm=self.pram_v2_final_norm,
            out_proj=self.pram_v2_out_proj,
            target_current_mask=target_current_mask,
            final_scalars=(self.pram_v2_final_scale_scalar,
                           self.pram_v2_final_shift_scalar),
        )

        x_out = x_out.masked_fill(target_current_mask.unsqueeze(-1), 0.0)

        return x_out

    def do_feasible_projection(
            self,
            x: torch.Tensor,
            low_t_mask: Optional[torch.Tensor],  # (B,)
            diffusion_time: torch.Tensor,  # (B,)
            target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len+future_len)
            target_current_control: Optional[torch.Tensor],  # (B, (1+)Pnn, 3)
    ) -> None:
        # target_past_11_dim: (B, (1+)Pnn, past_len, 11)
        target_past_11_dim = target_agents_past[:, :, :-1, :]
        target_current_xyyaw = target_agents_past[:, :, -1, :4]  # (B,P,4)
        target_class_one_hot = target_agents_past[:, :, -1, 8:11]  # (B,P,3)

        # low_t_mask: bool + device 통일
        if low_t_mask is None:
            low_t_mask = self._compute_feasible_low_t_mask(
                diffusion_time=diffusion_time)  # (B,)
        low_t_mask = _to_bool_mask(low_t_mask).to(device=x.device)  # (B,) bool

        # (공통) 정책만 적용: x(flat) 그대로 반환
        (
            x_for_feasible,  # (B,P,F) float32
            target_current_xyyaw_for_feasible,  # (B,P,4) float32
            target_past_cur_future_valid_for_feasible,  # (B,P,T_all) bool
            target_current_control_for_feasible,  # (B,P, 3) float32 or None
        ) = self._select_tensors_for_feasible_projection(
            x=x,
            target_current_xyyaw=target_current_xyyaw,
            target_past_cur_future_valid=target_past_cur_future_valid,
            target_current_control=target_current_control,  # (B, (1+)Pnn, 3)
        )

        if self.config.pose_based:
            diffusion_sequence = self._build_diffusion_inputs_for_feasible_projection(
                x_for_feasible=x_for_feasible,
                target_current_xyyaw_for_feasible=
                target_current_xyyaw_for_feasible,
                target_past_cur_future_valid_for_feasible=
                target_past_cur_future_valid_for_feasible,
            )  # (B,P,1+T,4)

            self._feasible_projection(
                diffusion_sequence=diffusion_sequence,
                target_past_11_dim=target_past_11_dim,
                target_class_one_hot=target_class_one_hot,
                target_past_cur_future_valid=
                target_past_cur_future_valid_for_feasible,
                low_t_mask=low_t_mask,
            )
        else:
            diffusion_control_traj = self._build_diffusion_inputs_for_feasible_projection(
                x_for_feasible=x_for_feasible,
                target_current_xyyaw_for_feasible=
                target_current_xyyaw_for_feasible,
                target_past_cur_future_valid_for_feasible=
                target_past_cur_future_valid_for_feasible,
            )  # (B,P,T,3)
            self._feasible_projection_vel(
                diffusion_control_traj=diffusion_control_traj,
                target_current_xyyaw=target_current_xyyaw_for_feasible,
                target_class_one_hot=target_class_one_hot,
                target_past_cur_future_valid=
                target_past_cur_future_valid_for_feasible,
                low_t_mask=low_t_mask,
                target_current_control_for_feasible=
                target_current_control_for_feasible,
            )

    def _compute_target_current_valid(
        self,
        target_past_cur_future_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거+현재+미래 유효 마스크에서 현재 프레임 기준 마스크를 만든다.

        Args:
            target_past_cur_future_valid (torch.Tensor):
                과거~미래 전체 유효 마스크.
                shape: (B, (1+)Pnn, 1+past_len+future_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - target_cur_future_valid:
                    현재+미래 부분 유효 마스크.
                    shape: (B, (1+)Pnn, 1+future_len)
                - target_current_valid:
                    현재 프레임 기준 무효 에이전트 마스크(True=유효).
                    shape: (B, (1+)Pnn)
        """
        # target_cur_future_valid: (B, (1+)Pnn, 1+T)
        target_cur_future_valid: torch.Tensor = target_past_cur_future_valid[:, :, -(
            1 + self._future_len):]
        # target_current_valid: (B, (1+)Pnn)  True=유효
        target_current_valid: torch.Tensor = target_cur_future_valid[:, :, 0]

        return target_cur_future_valid, target_current_valid

    def _apply_diffusion_timestep_and_validity(
        self,
        target_input_norm_xT: torch.Tensor,
        diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, past_len+1+future_len)
    ) -> torch.Tensor:  # (B, (1+)Pnn, F=_*6)
        """
if pose_based
    (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
else
    (B, (1+)Pnn, (past_len + T) *3) or (B, (1+)Pnn, T*3)
        5: x, y, cos, sin 에서 diffusion noise time step  + validity 추가

        diffusion noise time step
            참고로 1+future_len 은 현재+미래. 현재에는 노이즈를 추가하지 않을 것이므로, 0으로 채움
            미래 future_len 에 대해서는, diffusion_time 에 따라 노이즈 time step을 채움

        """
        diffusion_time = diffusion_time.to(device=target_input_norm_xT.device)
        if self.config.pose_based:
            final_dim = 4
        else:
            final_dim = 3
        # target_input_norm_xT: (B, (1+)Pnn, _ * 4 or 3) -> (B, (1+)Pnn, _, 4 or 3)
        target_input_norm_xT = target_input_norm_xT.reshape(
            target_input_norm_xT.shape[0],
            target_input_norm_xT.shape[1],
            -1,
            final_dim,
        )  # (B, (1+)Pnn, T_any, 4 or 3)

        if diffusion_time.ndim == 1:
            diffusion_time_for_future = diffusion_time[:, None].repeat(
                1, self._future_len)  # (B, future_len)
        elif diffusion_time.ndim == 2:
            if diffusion_time.shape[1] != self._future_len:
                raise ValueError(
                    f"diffusion_time must have shape (B,{self._future_len}) when 2D. got {tuple(diffusion_time.shape)}"
                )
            diffusion_time_for_future = diffusion_time  # (B, future_len)
        else:
            raise ValueError(
                f"diffusion_time must be (B,) or (B, future_len). got {diffusion_time.shape}"
            )
        """
if pose_based
    (B, (1+)Pnn, (time_len+ T) ,4) or (B, (1+)Pnn, T,4) or (B, (1+)Pnn, (1+T),4)
else
    (B, (1+)Pnn, (past_len + T) ,3) or (B, (1+)Pnn, T,3)
        """
        B, P, T_any, _ = target_input_norm_xT.shape

        past_cur_time_len = T_any - self._future_len  # past_len + 1
        # diffusion_time_for_past_cur: (B, past_cur_time_len)
        diffusion_time_for_past_cur = torch.zeros(
            (B, past_cur_time_len),
            dtype=diffusion_time.dtype,
            device=diffusion_time.device,
        )
        # (B, past_cur_time_len + future_len)
        diffusion_time_full = torch.cat(
            [diffusion_time_for_past_cur, diffusion_time_for_future],
            dim=1,
        )
        # (B, past_cur_time_len + future_len) -> (B, 1, past_cur_time_len + future_len, 1)
        diffusion_time_full = diffusion_time_full[:, None, :, None]
        # (B, P, past_cur_time_len + future_len, 1)
        diffusion_time_full = diffusion_time_full.expand(B, P, T_any, 1)
        if self.config.pose_based:
            # (B,P,T_any) bool
            validity = target_past_cur_future_valid[:, :, -T_any:]
        else:
            target_past_future_seg_valid = (
                target_past_cur_future_valid[..., :-1] &
                target_past_cur_future_valid[..., 1:]
            )  # (B, P, past_len + future_len)
            validity = target_past_future_seg_valid[:, :, -T_any:]
        validity = validity.to(device=target_input_norm_xT.device)  # 안전
        validity_f = validity.to(dtype=target_input_norm_xT.dtype).unsqueeze(
            -1)  # (B,P,T_any,1) float
        diffusion_time_full = diffusion_time_full * validity_f  # (B,P,T_any,1)
        diffusion_time_full = _cast_like(diffusion_time_full,
                                         target_input_norm_xT)
        """
        target_input_norm_xT
if pose_based
    (B, (1+)Pnn, (time_len+ T) ,4) or (B, (1+)Pnn, T,4) or (B, (1+)Pnn, (1+T),4)
else
    (B, (1+)Pnn, (past_len + T) ,3) or (B, (1+)Pnn, T,3)


        diffusion_time_full : (B, (1+)Pnn, past_cur_time_len + future_len, 1)
        validity : (B, (1+)Pnn, past_cur_time_len + future_len, 1)
        """
        target_input_norm_xT = torch.cat(
            [
                target_input_norm_xT,
                # (B, (1+)Pnn, past_cur_time_len + future_len, 4)
                diffusion_time_full,
                # (B, (1+)Pnn, past_cur_time_len + future_len, 1)
                validity_f,  # (B, (1+)Pnn, past_cur_time_len + future_len, 1)
            ],
            dim=-1,
        )  # (B, (1+)Pnn, past_cur_time_len + future_len, 6 or 5)
        target_input_norm_xT = target_input_norm_xT.reshape(
            B,
            P,
            -1,
        )  # (B, (1+)Pnn, _ * 6 or 5)
        return target_input_norm_xT

    def _get_x0_from_v(
        self,
        diffusion_output: torch.Tensor,
        diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, time_len+future_len)
        x_t_flat: torch.Tensor) -> torch.Tensor:
        """ diffusion_output / x_t_flat
        pose_based:
            (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        else:
            (B, (1+)Pnn, (past_len + T) *3) or (B, (1+)Pnn, T*3)

        $\hat{x}_0=\alpha_t x_t-\sigma_t\hat{v}_t$ 수식을 이용해서, x_0 를 복원하는 코딩을 구현하고 싶다.
        """
        # v_hat -> x0_hat: x0_hat = alpha_t * x_t - sigma_t * v_hat

        per_step_dim: int = 4 if self.config.pose_based else 3  # C
        B_in: int = int(x_t_flat.shape[0])
        P_in: int = int(x_t_flat.shape[1])
        F_in: int = int(x_t_flat.shape[2])

        if F_in % per_step_dim != 0:
            raise ValueError(
                f"v-branch: x_t_flat last dim must be multiple of {per_step_dim}. "
                f"got F={F_in}, pose_based={bool(self.config.pose_based)}")
        # 전체 step 수(과거 포함일 수 있음)
        T_any: int = int(
            F_in //
            per_step_dim)  # (time_len+T) or (1+T) or (T) or (past_len+T)
        future_len: int = int(self._future_len)
        if int(diffusion_output.shape[2]) != F_in:
            raise ValueError(
                "v-branch: model output shape mismatch with input. "
                f"x.shape={tuple(diffusion_output.shape)}, x_t_flat.shape={tuple(x_t_flat.shape)}"
            )
        past_cur_len: int = int(T_any - future_len)  # 과거(+현재가 있으면 현재) 구간 길이

        # --- (1) t_full 만들기: 과거/현재는 0, 미래는 diffusion_time ---
        if diffusion_time.ndim == 1:
            # diffusion_time: (B,) -> 미래 T개로 반복
            # t_future : (B, T)
            t_future = diffusion_time.to(device=diffusion_output.device,
                                         dtype=torch.float32).view(
                                             B_in,
                                             1).repeat(1, future_len)  # (B, T)
        elif diffusion_time.ndim == 2:
            # diffusion_time: (B, T)
            if tuple(diffusion_time.shape) != (B_in, future_len):
                raise ValueError(
                    "v-branch: diffusion_time shape mismatch. "
                    f"expected={(B_in, future_len)}, got={tuple(diffusion_time.shape)}"
                )
            t_future = diffusion_time.to(device=diffusion_output.device,
                                         dtype=torch.float32)  # (B, T)
        else:
            raise ValueError(
                f"v-branch: diffusion_time must be (B,) or (B,T). got {tuple(diffusion_time.shape)}"
            )

        if past_cur_len > 0:
            t_prefix = torch.zeros((B_in, past_cur_len),
                                   device=diffusion_output.device,
                                   dtype=torch.float32)  # (B, past_cur_len)
            t_full = torch.cat([t_prefix, t_future], dim=1)  # (B, T_any)
        else:
            t_full = t_future  # (B, T_any)  (future-only)

        # --- (2) alpha, sigma 구하기 ---
        # VPSDE_linear.marginal_prob(x0, t) = (alpha(t)*x0, sigma(t))
        # x0를 1로 두면 mean이 alpha(t)가 됨.
        x_dummy = torch.ones((B_in, 1, T_any, 1),
                             device=diffusion_output.device,
                             dtype=torch.float32)  # (B,1,T_any,1)
        # alpha:(B,1,T_any,1), sigma:(B,1,T_any,1)
        alpha, sigma = self._sde.marginal_prob(x_dummy, t_full)

        # --- (3) x0 복원 ---
        x_t_seq = x_t_flat.reshape(B_in, P_in, T_any, per_step_dim).to(
            torch.float32)  # (B,P,T_any,C)
        v_hat_seq = diffusion_output.reshape(B_in, P_in,
                                             T_any, per_step_dim).to(
                                                 torch.float32)  # (B,P,T_any,C)

        x0_hat_seq = alpha * x_t_seq - sigma * v_hat_seq  # (B,P,T_any,C)
        x0_hat_seq = x0_hat_seq.to(dtype=diffusion_output.dtype)

        # --- (4) 무효(step) 안전 마스킹: invalid는 항상 0 ---
        valid_nodes = _to_bool_mask(target_past_cur_future_valid).to(
            device=x0_hat_seq.device)  # (B,P, ...)
        if self.config.pose_based:
            valid_step = valid_nodes[:, :, -T_any:]  # (B,P,T_any)
        else:
            seg_valid_all = valid_nodes[..., :-1] & valid_nodes[
                ..., 1:]  # (B,P, seg_len)
            valid_step = seg_valid_all[:, :, -T_any:]  # (B,P,T_any)

        x0_hat_seq = x0_hat_seq.masked_fill(~valid_step.unsqueeze(-1), 0.0)

        # flat로 복원
        x0_hat_flat = x0_hat_seq.reshape(B_in, P_in,
                                         T_any * per_step_dim)  # (B,P,F)
        return x0_hat_flat

    def _get_target_current_control(
        self,
        target_input_norm_xT: torch.Tensor,
    ):
        # target_current_control: (B, (1+)Pnn, 3)  마지막 과거 구간 control (vxᵇ, vyᵇ, yaw_rate)
        # target_input_norm_xT 로 부터 추출하자.
        # not self.config.pose_based and self.config.use_past_dit_input 조건에서는
        # target_input_norm_xT는 (B, (1+)Pnn, (past_len + T) *3) shape일거야.
        B_tmp, P_tmp, F_tmp = target_input_norm_xT.shape
        if F_tmp % 3 != 0:
            raise ValueError(
                "pose_based=False expects target_input_norm_xT last dim to be multiple of 3. "
                f"got shape={tuple(target_input_norm_xT.shape)}")
        total_seg_len = F_tmp // 3
        assert total_seg_len == (
            self._future_len + self._past_len
        ), f"Expected total_seg_len to be future_len + past_len = {self._future_len + self._past_len}, but got {total_seg_len} from target_input_norm_xT shape {tuple(target_input_norm_xT.shape)}"
        past_len = total_seg_len - int(self._future_len)
        # target_current_control : (B, (1+)Pnn, 3)  마지막 과거 구간 control (vxᵇ, vyᵇ, yaw_rate)
        target_current_control = target_input_norm_xT.reshape(
            B_tmp, P_tmp, total_seg_len, 3)[:, :, past_len - 1, :]
        return target_current_control

    def forward(
            self,
            target_input_norm_xT: torch.Tensor,
            diffusion_time: torch.Tensor,
            target_agents_past: torch.Tensor,
            target_past_cur_future_valid: torch.Tensor,
            cross_c: torch.Tensor,
            cross_mask: torch.Tensor,
            cross_pos_2d: torch.Tensor,  # ✅ 추가 (B, token_num, 9)
            low_t_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DiT 전체 forward 를 수행하는 진입점.
        target_input_norm_xT (input) 혹은 xT_input_flat (output)
            if pose_based
                (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            else
                (B, (1+)Pnn, (past_len + T) *3) or (B, (1+)Pnn, T*3)
        """
        if not self.config.pose_based and self.config.use_past_dit_input:
            target_current_control = self._get_target_current_control(
                target_input_norm_xT)

        else:
            target_current_control = None

        if self.config.profile_feasible:
            print("=============[DEBUG] DiT.forward 호출 =============")
        # 현재+미래 유효 마스크 및 현재 무효 에이전트 마스크
        """ target_past_cur_future_valid: (B, (1+)Pnn, 1+past_len+future_len) bool
        target_cur_future_valid :  (B, (1+)Pnn, 1+future_len)
        target_current_valid :  (B, (1+)Pnn)
        """
        target_cur_future_valid, target_current_valid = \
            self._compute_target_current_valid(
                target_past_cur_future_valid=target_past_cur_future_valid
            )
        B, one_or_Pnn, _ = target_input_norm_xT.shape  # (B, (1+)Pnn, F)
        # time_len(=past_len+1)
        target_current_11_dim = target_agents_past[:, :, -1, :]  # (B, Pnn, 11)
        device_type: str = target_input_norm_xT.device.type

        # DiT 본체(프리프로젝션+블록+최종 투영)를 프로파일링 블록 안에서 수행
        with profile_block(
                "DiT.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            device_type: str = target_input_norm_xT.device.type

            # ✅ [추가] v-예측에서 x0 복원에 사용할 x_t(flat) 보관
            x_t_flat: torch.Tensor = target_input_norm_xT  # (B, (1+)Pnn, F)
            with profile_block(
                    "DiT._apply_diffusion_timestep_and_validity",
                    enabled=self.config.profile_feasible,
                    device_type=device_type,
            ):
                # target_input_norm_xT: (B, (1+)Pnn, _ * 6 or 5)
                target_input_norm_xT = self._apply_diffusion_timestep_and_validity(
                    target_input_norm_xT, diffusion_time,
                    target_past_cur_future_valid)
            """ diffusion_output
        pose_based:
            (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        else:
            (B, (1+)Pnn, (past_len + T) *3) or (B, (1+)Pnn, T*3)
            """
            diffusion_output: torch.Tensor = self._run_dit_core_with_pram_v2(
                target_input_norm_xT=target_input_norm_xT,
                diffusion_time=diffusion_time,
                cross_c=cross_c,
                cross_mask=cross_mask,
                cross_pos_2d=cross_pos_2d,  # ✅ 추가
                target_current_11_dim=target_current_11_dim,
                target_current_valid=target_current_valid,
            )
        self.norm_dit_returns = DiTReturns()
        # model_type 분기
        if self._model_type == "score":
            raise NotImplementedError("Score model type is not implemented.")
        elif self._model_type == "v":
            x0 = self._get_x0_from_v(diffusion_output, diffusion_time,
                                     target_past_cur_future_valid, x_t_flat)
        elif self._model_type == "x_start":
            # x 를 대부분의 경우에 그대로 반환
            x0 = diffusion_output
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
        """ self.diffusion_sequence_flat = x0
        pose_based:
            (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        else:
            (B, (1+)Pnn, (past_len + T) *3) or (B, (1+)Pnn, T*3)
        """
        self.diffusion_sequence_flat = x0
        """ x0_4dim
        pose_based:
            (B, (1+)Pnn, (time_len+ T) ,4) or (B, (1+)Pnn, T,4) or (B, (1+)Pnn, (1+T),4)
        else:
            (B, (1+)Pnn, (past_len + T) ,3) or (B, (1+)Pnn, T,3)
        """
        if self.config.pose_based:
            x0_4dim = x0.reshape(B, one_or_Pnn, -1, 4)  # (B, (1+)Pnn, T_any, 4)
            target_current_xyyaw = target_agents_past[:, :,
                                                      -1:, :4]  # (B,P,1,4)
            x0_future_4dim = x0_4dim[:, :,
                                     -self._future_len:, :]  # (B,P,F_future,4)
            x0_4dim = torch.cat([target_current_xyyaw, x0_future_4dim],
                                dim=2)  # (B,P,1+T,4)
        else:
            x0_4dim = x0.reshape(B, one_or_Pnn, -1, 3)  # (B, (1+)Pnn, T_any, 3)
            x0_4dim = x0_4dim[:, :, -self._future_len:, :]  # (B,P,F_future,3)
        """
        diffusion_sequence
            pose_based:
                (B, (1+)Pnn, (1+T),4)
            else:
                (B, (1+)Pnn, T,3)
        """
        self.norm_dit_returns.diffusion_sequence = x0_4dim
        if self.config.use_feasible:
            self.do_feasible_projection(
                x=x0,
                low_t_mask=low_t_mask,  # (B,)
                diffusion_time=diffusion_time,
                target_agents_past=
                target_agents_past,  # (B, (1+)Pnn, time_len, 11)
                target_past_cur_future_valid=target_past_cur_future_valid,
                # (B, (1+)Pnn, time_len+future_len)
                target_current_control=target_current_control,
                # (B, (1+)Pnn, 3) or None
            )
        return diffusion_output

    def _prepare_prev_control_for_filter_and_integrate(
        self,
        *,
        target_past_cur_future_valid: torch.
        Tensor,  # (B,Pnn, 1+past_len+future_len) bool/0-1
        target_current_control_norm: Optional[
            torch.Tensor],  # (B,Pnn,3) or None (정규화)
        future_len: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """k=0에서 쓸 '이전 구간 control'을 안전하게 준비합니다.

        하는 일
        - target_current_control_norm(정규화)이 있으면:
          - 마지막 과거 구간이 유효한지(prev_valid)를 계산합니다.
          - control을  unnorm으로 바꿉니다.
          - 유효하지 않은 행은 0으로 눌러 둡니다.
        - past가 없거나(prev 구간이 정의되지 않으면) None을 반환합니다.

        Args:
            target_past_cur_future_valid (torch.Tensor):
                과거~현재~미래 노드 유효 마스크.
                shape: (B, Pnn, 1+past_len+future_len)
            target_current_control_norm (Optional[torch.Tensor]):
                "마지막 과거 구간" control(정규화).
                shape: (B, Pnn, 3)
            future_len (int):
                미래 구간 개수 T.
            device (torch.device):
                반환 텐서 device.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - prev_control_unnorm:
                    (B, Pnn, 3) unnorm control. (유효하지 않은 행은 0)
                    없으면 None
                - prev_control_valid:
                    (B, Pnn) bool. 마지막 과거 구간이 유효하면 True
                    없으면 None
        """
        if target_current_control_norm is None:
            return None, None

        if target_past_cur_future_valid.dim() != 3:
            raise ValueError(
                "target_past_cur_future_valid는 (B,Pnn,T_total) 3D여야 합니다. "
                f"got {tuple(target_past_cur_future_valid.shape)}")

        valid_all = target_past_cur_future_valid
        if valid_all.dtype != torch.bool:
            valid_all = valid_all > 0.5
        valid_all = valid_all.to(device=device)

        total_len = int(valid_all.shape[-1])  # = 1+past_len+future_len
        past_len = total_len - (1 + int(future_len))  # past 노드 개수
        if past_len <= 0:
            # 과거가 없으면 "마지막 과거 구간" 자체가 없음
            return None, None

        # 마지막 과거 구간(seg = past_len-1): node[past_len-1] & node[past_len(=current)]
        prev_valid = valid_all[..., past_len -
                               1] & valid_all[..., past_len]  # (B,Pnn)

        # 전부 무효면 굳이 넘기지 않고 기존 동작으로 두는 게 안전
        if not bool(prev_valid.any().item()):
            return None, None

        if target_current_control_norm.dim() != 3 or int(
                target_current_control_norm.shape[-1]) != 3:
            raise ValueError("target_current_control_norm은 (B,Pnn,3) 이어야 합니다. "
                             f"got {tuple(target_current_control_norm.shape)}")

        ctrl_norm = target_current_control_norm.to(
            device=device, dtype=torch.float32)  # (B,Pnn,3)
        ctrl_norm_t = ctrl_norm.unsqueeze(2)  # (B,Pnn,1,3)
        prev_valid_unsqueeze = prev_valid.unsqueeze(-1) # (B,Pnn,1)

        # state_normalizer.inverse는 (B,Pnn,T,3) 입력을 기대하므로 T=1로 맞춤
        ctrl_unnorm_t = self.config.state_normalizer.inverse(ctrl_norm_t,
                                                                   valid_mask=prev_valid_unsqueeze)
        ctrl_unnorm = ctrl_unnorm_t.squeeze(2)  # (B,Pnn,3)
        # 유효하지 않은 행은 0으로 (혹시라도 잘못 쓰여도 안전)
        ctrl_unnorm = ctrl_unnorm.masked_fill(~prev_valid.unsqueeze(-1), 0.0)

        return ctrl_unnorm, prev_valid

    # : stride 기반 down/up 샘플링을 통합한 새 파이프라인
    def _feasible_projection_core(
            self,
            diffusion_sequence: torch.Tensor,  # (B, (1+)Pnn, 1+future_len, 4)
            target_class_one_hot: torch.Tensor,  # (B, (1+)Pnn, 3)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len=1+past_len+future_len) bool
            target_past_11_dim: Optional[
                torch.Tensor],  # (B, (1+)Pnn, past_len, 11) 또는 None
            final_hidden_tokens: torch.Tensor,  # (B, (1+)Pnn, H)
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
        device_type: str = diffusion_sequence.device.type
        use_profile: bool = bool(getattr(self.config, "profile_feasible",
                                         False))

        # Feasible 파트는 모두 FP32로 고정
        with torch.autocast(device_type=device_type, enabled=False):
            diffusion_sequence = diffusion_sequence.float()  # (B, Pnn, 1+T, 4)
            target_class_one_hot = target_class_one_hot.float()  # (B, Pnn, 3)
            target_past_cur_future_valid = target_past_cur_future_valid.to(
                torch.bool)  # (B, Pnn, time_len+future_len)
            if target_past_11_dim is not None:
                target_past_11_dim = target_past_11_dim.float(
                )  # (B, Pnn, past_len, 11)

            B, Pnn, one_future_len, _ = diffusion_sequence.shape
            future_len: int = int(one_future_len - 1)  # T

            # stride 및 SG 윈도 관련 하이퍼 계산
            (
                stride_step,  # int
                dt_for_savgol,  # float
                max_window_len_xy,  # int
                max_window_len_yaw,  # int
            ) = self.feasible_projector.get_feasible_stride_params(future_len)
            # 역정규화 현재+미래 궤적 및 현재 상태
            target_cur_future_valid = target_past_cur_future_valid[:, :, -(
                one_future_len):]  # (B, Pnn, 1+T) bool
            unnorm_diffusion_trajectory = self.config.state_normalizer.inverse(
                data=diffusion_sequence,  # (B, Pnn, 1+T, 4)
                valid_mask=target_cur_future_valid)  # (B, Pnn, 1+T)
            unnorm_near_current_state = unnorm_diffusion_trajectory[:, :,
                                                                    0, :]  # (B, Pnn, 4)

            # 과거 xy-yaw (정규화/역정규화) 준비

            if target_past_11_dim is not None and target_past_11_dim.numel(
            ) > 0:
                # target_past_xyyaw: (B, Pnn, past_len, 4)
                target_past_xyyaw = target_past_11_dim[..., :4]
                # target_past_valid : (B, Pnn, past_len) bool
                target_past_valid = target_past_cur_future_valid[:, :, :-(
                    one_future_len)]
                # (B, Pnn, past_len, 4)
                unnorm_target_past_xyyaw = self.config.state_normalizer.inverse(
                    data=target_past_xyyaw, valid_mask=target_past_valid)
            else:
                unnorm_target_past_xyyaw = None

            # --- (1) stride 기준 다운샘플 궤적/마스크 생성 ---
            (
                unnorm_diffusion_trajectory_stride,  # (B, Pnn, 1+T_ds, 4)
                unnorm_near_past_xyyaw_stride,
                # (B, 1+Pnn, past_len_ds, 4) or None
                target_past_cur_future_valid_stride,
                # (B, Pnn, past_len_ds+1+T_ds) bool
                diffusion_trajectory_stride_norm,  # (B, Pnn, 1+T_ds, 4)
                near_past_xyyaw_stride_norm,  # (B, Pnn, past_len_ds, 4) or None
                past_len_ds,  # int
                future_len_ds,  # int (T_ds)
            ) = self.feasible_projector.build_downsampled_feasible_inputs(
                diffusion_trajectory=diffusion_sequence,  # (B, Pnn, 1+T, 4)
                target_past=target_past_11_dim,
                # (B, Pnn, past_len, 11) or None
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, Pnn, past_len+1+T) bool
                unnorm_diffusion_trajectory=
                unnorm_diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                unnorm_near_past_xyyaw=
                unnorm_target_past_xyyaw,  # (B, Pnn, past_len, 4) or None
                stride_step=stride_step,  # int
            )

            # --- (2) Savitzky–Golay + 점 제어 계산 ---
            with profile_block(
                    "feasible.savgol_filter_for_control",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                # (B, Pnn, point_len_ds, 3)  [v_x^w, v_y^w, ω]
                # + point_len_inputs_stride: (prepare_points_and_masks 결과 재사용용)
                (
                    unnorm_points_world_control_stride,
                    point_len_inputs_stride,
                ) = self.feasible_projector.savgol_filter_for_control(
                    unnorm_diffusion_trajectory_stride,  # (B, Pnn, 1+T_ds, 4)
                    unnorm_near_past_xyyaw_stride,
                    # (B, Pnn, past_len_ds, 4) or None
                    target_past_cur_future_valid_stride,
                    # (B, Pnn, past_len_ds+1+T_ds) bool
                    dt=dt_for_savgol,
                    polyorder=2,
                    max_window_len_xy=max_window_len_xy,
                    max_window_len_yaw=max_window_len_yaw,
                    return_point_len_inputs=True,
                )

            with profile_block(
                    "feasible.compute_midpoint_controls",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                # (B, Pnn, segment_len_ds, 3)  [v_x^b, v_y^b, ω]_mid (stride 타임라인 기준)
                """무효 구간(점이 무효한 구간)은 0.0으로 출력됩니다."""
                # # (B, Pnn, segment_len_ds, 3) # (B, Pnn, segment_len_ds)
                unnorm_seg_body_control_stride, seg_valid = self.feasible_projector.compute_midpoint_controls(
                    unnorm_diffusion_trajectory_stride,  # (B, Pnn, 1+T_ds, 4)
                    unnorm_near_past_xyyaw_stride,
                    # (B, Pnn, past_len_ds, 4) or None
                    unnorm_points_world_control_stride,
                    # (B, Pnn, point_len_ds, 3)
                    target_past_cur_future_valid_stride,
                    # (B, Pnn, past_len_ds+1+T_ds) bool
                    point_len_inputs=point_len_inputs_stride,
                )
                seg_valid = seg_valid > 0.5

            # --- (3) 관측 정규화 → FeasibleProjector 네트워크(TCN) 보정 ---
            # (B, Pnn, segment_len_ds, 3)
            seg_body_control_stride = self.config.state_normalizer(data=unnorm_seg_body_control_stride,
                                                                   valid_mask=seg_valid
                                                                   )

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
                        target_past_cur_future_valid_stride,
                        # (B, Pnn, past_len_ds+1+T_ds)
                        diffusion_trajectory_stride_norm,  # (B, Pnn, 1+T_ds, 4)
                        near_past_xyyaw_stride_norm,
                        # (B, Pnn, past_len_ds, 4) or None
                        seg_body_control_stride,  # (B, Pnn, segment_len_ds, 3)
                        final_hidden_tokens,  # (B, Pnn, H)
                    )
                    touch = self._ddp_touch_feasible_projector_params(
                        device=seg_body_control_stride_ref.device,
                        dtype=seg_body_control_stride_ref.dtype,
                    )
                    seg_body_control_stride_ref = seg_body_control_stride_ref + touch

            seg_body_control_stride_ref = seg_body_control_stride_ref.float()
            # (B, Pnn, segment_len_ds, 3)
            unnorm_seg_body_control_stride_ref = self.config.state_normalizer.inverse(
                data=seg_body_control_stride_ref,
                valid_mask=seg_valid,
            )


            # --- (4) 미래 구간 제어만 원래 future_len 개수로 업샘플링 --- (여기서 과거 걸 짜르는듯)
            # (B, Pnn, future_len, 3)
            unnorm_fut_seg_body_control = self.feasible_projector.upsample_future_controls_from_stride(
                unnorm_seg_body_control_stride=
                unnorm_seg_body_control_stride_ref,
                # (B, Pnn, segment_len_ds, 3)
                past_len_ds=past_len_ds,  # int
                future_len_ds=future_len_ds,  # int
                future_len_full=future_len,  # int
                stride_step=stride_step,  # int
            )
            # --- (5) 제약 기반 필터 + 적분 ---
            target_cur_future_valid = target_past_cur_future_valid[:, :, -(
                1 + future_len):]  # (B, Pnn, 1+future_len) bool

            with profile_block(
                    "feasible.filter_and_integrate",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                """무효 구간(점이 무효한 구간)은 0.0으로 출력됩니다."""
                """
                pose_based = True 에서는, noise 레벨이 일정 이하인 데이터만 
                _feasible_projection_core 에 넘어왔습니다. -> filter_and_integrate 로 전달
                """
                (
                    unnorm_integrated_trajectory,  # (B, Pnn, future_len, 4)
                    unnorm_control_constraint_diff,  # (B, Pnn, future_len, 3)
                    unnorm_control_sequence,  # (B, Pnn, future_len, 3)
                ) = self.feasible_projector.filter_and_integrate(
                    unnorm_near_current_state,  # (B, Pnn, 4)
                    target_cur_future_valid,  # (B, Pnn, 1+future_len) bool
                    unnorm_fut_seg_body_control,  # (B, Pnn, future_len, 3)
                    target_class_one_hot,  # (B, Pnn, 3)
                )
            # 정규화해서 DiTReturns 로 저장
            target_future_valid = target_cur_future_valid[:, :,
                                                          1:]  # (B, Pnn, future_len) bool
            # target_cur_future_valid: (B, Pnn, 1+future_len) # 앞뒤 점이 모두 유호이면 유효
            # target_segment_valid : (B, Pnn, future_len)
            target_segment_valid = target_cur_future_valid[:, :, 1:] & target_cur_future_valid[:, :, :-1]
            integrated_trajectory = self.config.state_normalizer(
                data=unnorm_integrated_trajectory,
                valid_mask=target_future_valid)  # (B, Pnn, future_len, 4)

            # (B, Pnn, future_len, 3)
            control_constraint_diff = self.config.state_normalizer(unnorm_control_constraint_diff,
                                                                  valid_mask=target_segment_valid)
            control_constraint_diff = control_constraint_diff.masked_fill(
                ~target_segment_valid.unsqueeze(-1), 0.0)

            self.norm_dit_returns.integrated_trajectory = integrated_trajectory  # (B,(1+)Pnn,future_len,4)
            self.norm_dit_returns.control_constraint_diff = control_constraint_diff  # (B,(1+)Pnn,future_len,3)

    def _feasible_projection_core_vel(
        self,
        diffusion_control_traj: torch.Tensor,  # (B, (1+)Pnn, future_len, 3)
        target_class_one_hot: torch.Tensor,  # (B, (1+)Pnn, 3)
        target_past_cur_future_valid: torch.Tensor,
        # (B, (1+)Pnn, 1+past_len+future_len) bool
        target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
        filter_active_idx: torch.Tensor,
        target_current_control: Optional[torch.Tensor],
        # (B, (1+)Pnn, 3)  (정규화 값)
    ) -> None:
        device_type: str = diffusion_control_traj.device.type
        use_profile: bool = bool(getattr(self.config, "profile_feasible",
                                         False))

        with torch.autocast(device_type=device_type, enabled=False):
            B, Pnn, future_len, _ = diffusion_control_traj.shape
            one_future_len = 1 + future_len

            diffusion_control_traj = diffusion_control_traj.float(
            )  # (B,Pnn,T,3)
            target_class_one_hot = target_class_one_hot.float()
            target_past_cur_future_valid = target_past_cur_future_valid.to(
                torch.bool)

            # 현재~미래 valid
            target_cur_future_valid = target_past_cur_future_valid[:, :, -(
                one_future_len):]  # (B,Pnn,1+T)
            target_cur_valid = target_cur_future_valid[:, :, 0]  # (B,Pnn)

            # 현재 포즈 unnorm
            unnorm_target_current_xyyaw = self.config.state_normalizer.inverse(
                data=target_current_xyyaw,
                valid_mask=target_cur_valid,
            )  # (B,Pnn,4)
            # target_cur_future_valid: (B,Pnn,1+T)
            # target_seg_future_valid : (B,Pnn,T)
            target_seg_future_valid = target_cur_future_valid[:, :, 1:] & target_cur_future_valid[:, :, :-1]
            # 미래 control unnorm
            # (B,Pnn,T,3)
            unnorm_diffusion_control_traj = self.config.state_normalizer.inverse(data=diffusion_control_traj,
                                                                valid_mask=target_seg_future_valid,
                                                                                 )


            # ✅ (1) prev control을 unnorm으로 변환 + ✅ (2) prev_valid 계산
            unnorm_prev_control, prev_control_valid = self._prepare_prev_control_for_filter_and_integrate(
                target_past_cur_future_valid=target_past_cur_future_valid,
                # (B,Pnn,1+past_len+T)
                target_current_control_norm=target_current_control,
                # (B,Pnn,3) norm
                future_len=int(future_len),
                device=diffusion_control_traj.device,
            )

            with profile_block(
                    "feasible.filter_and_integrate",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                use_body_vel = bool(getattr(self.config, "use_body_vel", True))
                (
                    unnorm_integrated_trajectory,  # (B,Pnn,T,4)
                    unnorm_control_constraint_diff,  # (B,Pnn,T,3)
                    unnorm_control_sequence,  # (B,Pnn,T,3)
                ) = self.feasible_projector.filter_and_integrate(
                    unnorm_target_current_xyyaw,  # (B,Pnn,4)
                    target_cur_future_valid,  # (B,Pnn,1+T)
                    unnorm_diffusion_control_traj,  # (B,Pnn,T,3)
                    target_class_one_hot,  # (B,Pnn,3)
                    filter_active_idx,
                    target_current_control=unnorm_prev_control,  # ✅ unnorm으로 넣기
                    target_current_control_valid=prev_control_valid,
                    use_body_vel=use_body_vel
                    # ✅ k=0 S2 적용 마스크
                )

            target_future_valid = target_cur_future_valid[:, :, 1:]  # (B,Pnn,T)
            integrated_trajectory = self.config.state_normalizer(
                data=unnorm_integrated_trajectory,
                valid_mask=target_future_valid,
            )

            control_constraint_diff = self.config.state_normalizer(unnorm_control_constraint_diff,
                                                                   valid_mask=target_seg_future_valid)
            control_constraint_diff = control_constraint_diff.masked_fill(
                ~target_seg_future_valid.unsqueeze(-1), 0.0)

            norm_control_sequence = self.config.state_normalizer(unnorm_control_sequence
                                                                 ,valid_mask=target_seg_future_valid) #
            norm_control_sequence = norm_control_sequence.masked_fill(
                ~target_seg_future_valid.unsqueeze(-1), 0.0)

            self.norm_dit_returns.integrated_trajectory = integrated_trajectory  # (B,(1+)Pnn,future_len,4)
            self.norm_dit_returns.control_constraint_diff = control_constraint_diff  # (B,(1+)Pnn,future_len,3)
            self.norm_dit_returns.control_sequence = norm_control_sequence  # (B,(1+)Pnn,future_len,3)

    def _ddp_touch_feasible_projector_params(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """FeasibleProjector 파라미터를 0배로 한 번 만져서(사용 흔적) 그래프에 포함시킵니다.

        목적:
            - 이번 배치에서 low_t_mask가 전부 False라 feasible을 "실행"하지 않아도,
              DDP에서 FeasibleProjector 파라미터가 unused로 판단되는 상황을 피하려고,
              모든 파라미터를 계산 그래프에 한 번씩 등장시킵니다.
            - 결과 스칼라는 항상 0이므로, 어떤 출력 텐서에 더해도 값이 바뀌지 않습니다.

        Args:
            device (torch.device): 결과 스칼라를 만들 device
            dtype (torch.dtype): 결과 스칼라 dtype

        Returns:
            torch.Tensor:
                - shape: ()
                - 값: 0.0 (출력값 변화 없음)
        """
        if not hasattr(self,
                       "feasible_projector") or self.feasible_projector is None:
            return torch.zeros((), device=device, dtype=dtype)

        touch: torch.Tensor = torch.zeros((), device=device, dtype=dtype)

        # ✅ 핵심: FeasibleProjector의 "모든" 학습 파라미터가 그래프에 등장하도록 1원소씩 터치
        for p in self.feasible_projector.parameters():
            if p is None:
                continue
            if not p.requires_grad:
                continue
            if p.numel() == 0:
                continue

            # dtype/device 정렬 후 0을 곱해 값은 0, 그래프에는 파라미터가 등장
            touch = touch + (p.view(-1)[0].to(device=device, dtype=dtype) * 0.0)

        return touch

    def _feasible_projection(
            self,
            diffusion_sequence: torch.Tensor,  # (B, (1+)Pnn, 1+future_len, 4)
            target_past_11_dim: torch.Tensor,
            # (B, (1+)Pnn, past_len, 11) 또는 None
            target_class_one_hot: torch.Tensor,  # (B, (1+)Pnn, 3)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len(=1+past_len) + future_len) bool
            low_t_mask: torch.Tensor,  # (B, )
    ) -> None:
        B, Pnn, one_future_len, _ = diffusion_sequence.shape
        future_len = one_future_len - 1
        device = diffusion_sequence.device

        # [B] -> bool 로 정리
        low_t_mask = low_t_mask.to(device=device)
        low_t_mask_bool = low_t_mask if low_t_mask.dtype == torch.bool else (
            low_t_mask > 0.5)

        # 기본값:
        #   - integrated_trajectory: 원 궤적(x_start) 그대로 (t=1..future_len)
        #   - control_constraint_diff: 전부 0
        # diffusion_sequence 는 정규화 상태라고 가정
        base_integrated = diffusion_sequence[:, :, 1:, :].detach(
        )  # (B,Pnn,future_len,4)
        base_constraint = diffusion_sequence.new_zeros(
            (B, Pnn, future_len, 3))  # (B,Pnn,future_len,3)

        # ✅ DDP 안전용: feasible_projector 파라미터를 "0배로" 그래프에 등장시키는 스칼라
        touch = self._ddp_touch_feasible_projector_params(
            device=device,
            dtype=base_integrated.dtype,
        )

        # 실제로 FeasibleProjector를 돌릴 배치 인덱스 선택
        active_idx = torch.nonzero(low_t_mask_bool,
                                   as_tuple=False).squeeze(-1)  # (N_active,)

        # low-t 샘플이 없으면:
        #  - 출력은 base로 유지
        #  - DDP unused 방지용 touch만 더해 둔다(값 변화 없음)
        if int(active_idx.numel()) == 0:
            integrated_all = base_integrated + touch
            constraint_all = base_constraint + touch
            self.norm_dit_returns.integrated_trajectory = integrated_all  # (B,Pnn,future_len,4)
            self.norm_dit_returns.control_constraint_diff = constraint_all  # (B,Pnn,future_len,3)
            return

        # ---- active subset만 projector 파이프라인 실행 ----
        self._feasible_projection_core(
            diffusion_sequence[active_idx],
            target_class_one_hot[active_idx],
            target_past_cur_future_valid[active_idx],
            target_past_11_dim[active_idx]
            if target_past_11_dim is not None else None,
            self.final_hidden_tokens[active_idx],
        )

        # `_feasible_projection_core` 은 서브 배치 기준으로 self.norm_dit_returns 를 채운다.
        integ_active = self.norm_dit_returns.integrated_trajectory  # (N_active,Pnn,future_len,4)
        const_active = self.norm_dit_returns.control_constraint_diff  # (N_active,Pnn,future_len,3)

        # dtype 맞추기
        integ_active = integ_active.to(device=device,
                                       dtype=base_integrated.dtype)
        const_active = const_active.to(device=device,
                                       dtype=base_constraint.dtype)

        # ✅ 핵심: in-place 대입이 아니라 "새 텐서"를 만들어서 연결 유지
        integrated_all = base_integrated.index_copy(0, active_idx, integ_active)
        constraint_all = base_constraint.index_copy(0, active_idx, const_active)

        # ✅ 추가 안전장치:
        #    - FeasibleProjector 내부가 어떤 이유로든 조기 return해서 "파라미터 미사용"이 발생해도,
        #      최종 출력에서 touch가 들어가므로 DDP unused 방지
        integrated_all = integrated_all + touch
        constraint_all = constraint_all + touch
        self.norm_dit_returns.integrated_trajectory = integrated_all  # (B,Pnn,future_len,4)
        self.norm_dit_returns.control_constraint_diff = constraint_all  # (B,Pnn,future_len,3)

    def _feasible_projection_vel(
            self,
            diffusion_control_traj: torch.Tensor,  # (B, (1+)Pnn, future_len, 3)
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
            target_class_one_hot: torch.Tensor,  # (B, (1+)Pnn, 3)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len_total)
            low_t_mask: torch.Tensor,  # (B,)
            target_current_control_for_feasible: Optional[
                torch.Tensor],  # (B, (1+)Pnn, 3)
    ) -> None:
        B, Pnn, future_len, _ = diffusion_control_traj.shape
        device = diffusion_control_traj.device

        low_t_mask = low_t_mask.to(device=device)
        low_t_mask_bool = low_t_mask if low_t_mask.dtype == torch.bool else (
            low_t_mask > 0.5)

        # active_idx는 "제약(필터) 적용 배치" 인덱스입니다. (pose_based=False에서는 적분은 항상 수행)
        # - active_idx가 비면: 제약(필터)은 적용하지 않고, 적분만 수행됩니다.
        active_idx = torch.nonzero(low_t_mask_bool, as_tuple=False).squeeze(-1)

        self._feasible_projection_core_vel(
            diffusion_control_traj,  # (B, (1+)Pnn, future_len, 3)
            target_class_one_hot,  # (B, (1+)Pnn, 3)
            target_past_cur_future_valid,  # (B, (1+)Pnn, time_len_total)
            target_current_xyyaw,  # (B, (1+)Pnn, 4)
            filter_active_idx=active_idx,  # (B_filter,)
            target_current_control=
            target_current_control_for_feasible,  # (B, (1+)Pnn, 3)
        )
