from functools import partial
from typing import Callable
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from typing import Optional, Dict, Tuple, Any
from flash_attn.bert_padding import unpad_input, pad_input
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, \
    StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock
from diffusion_planner.loss import _require_finite
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

from typing import NamedTuple

# decoder.py 또는 feasible.py 상단 import 근처에 추가
import time
from contextlib import contextmanager
from typing import Iterator
import torch
from typing import Tuple
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FlashAttnKVCache


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
    """noise를 ref와 같은 device/dtype으로 맞춥니다."""
    return noise.to(device=ref.device, dtype=ref.dtype)


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


def _compute_time_padding_mask_from_state(
    target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
    check_dims: int = 8,
    eps: float = 0.0,
) -> torch.Tensor:
    """에이전트 상태 시퀀스에서 '빈 슬롯(패딩)'을 time 축으로 판단하는 마스크를 만듭니다.

    판단 규칙(프레임 단위)
    ----------------------
    아래 중 하나라도 만족하면 그 프레임은 패딩(True)으로 봅니다.
      1) check_dims 범위 안에 NaN/inf가 섞여 있음
      2) check_dims 범위 값들이 전부 0(또는 eps 이하)로 비어 있음

    Args:
        target_agents_past (torch.Tensor):
            에이전트 상태 시퀀스.
            shape: (B, (1+)Pnn, time_len, 11)
        check_dims (int):
            패딩 판정에 사용할 마지막 차원 D의 앞부분 길이.
            보통 (x,y,cos,sin,vx,vy,w,l) 같은 연속값 영역만 체크하려고 8을 사용합니다.
        eps (float):
            "전부 0" 판정을 조금 느슨하게 하고 싶을 때 쓰는 값.
            기본은 0.0 입니다.

    Returns:
        target_past_current_mask: (B, Pnn, (1+)time_len)
            패딩 마스크(True=패딩, False=유효).

    Raises:
        ValueError:
            target_agents_past shape이 4D가 아니면 발생합니다.
    """
    if target_agents_past.dim() != 4:
        raise ValueError(
            f"target_agents_past must be 4D (B,Pnn,time_len,D), got {tuple(target_agents_past.shape)}"
        )

    last_dim: int = int(target_agents_past.size(-1))
    used_dims: int = int(min(max(check_dims, 1), last_dim))

    state_slice = target_agents_past[
        ..., :used_dims]  # (B,(1+)Pnn,time_len,used_dims)

    # NaN/inf 방어
    is_finite = torch.isfinite(state_slice).all(dim=-1)  # (B,(1+)Pnn,time_len)

    # 0(또는 거의 0) 방어
    is_non_zero = state_slice.abs().sum(dim=-1) > float(
        eps)  # (B,(1+)Pnn,time_len)

    is_valid = is_finite & is_non_zero  # (B,(1+)Pnn,time_len)
    target_past_current_mask = ~is_valid
    return target_past_current_mask

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

    is_cuda: bool = isinstance(device_type, str) and device_type.startswith("cuda")
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

    avg_ms: float = (avg_sum_now / float(avg_cnt_now)) if avg_cnt_now > 0 else 0.0
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
        # = <diffusion_planner/model/guidance/guidance_wrapper.py> 의 GuidanceWrapper 인스턴스가 들어옴
        self._guidance_fn = getattr(config, "guidance_fn", None)
        self._x0_for_amortized_inference = None  # # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
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
                - amortized_random_noise: (B, (1+)Pnn, future_len, 4)
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
                shape: (B, (1+)Pnn, future_len, 4)
        """
        noise = inputs.get("amortized_random_noise", None)
        if noise is None:
            raise KeyError(
                "amortized(step_idx>0) 추론에는 inputs['amortized_random_noise']가 필요합니다. "
                "eval 코드에서 cand_idx 기반 seed로 만든 텐서를 넣어 주세요.")
        if not isinstance(noise, torch.Tensor):
            raise TypeError(
                "inputs['amortized_random_noise']는 torch.Tensor여야 합니다.")

        expected = (int(batch_size), int(one_or_Pnn), int(self._future_len), 4)
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
                - inference_noise: shape (B, (1+)Pnn, future_len, 4)
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

        expected = (int(batch_size), int(one_or_Pnn), int(self._future_len), 4)
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
            target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
    ) -> torch.Tensor:
        """훈련 모드에서 DiT에 넣을 입력(xT_input_flat)과 현재 프레임(정규화)을 만든다.

        처리 흐름:
            1) near_cur_future_norm_xT 에서 "미래(노이즈가 섞인 구간)"을 꺼냅니다.
               - shape: (B, Pnn, future_len, 4)
            2) near_agents_past 에서 (x, y, cos, sin) 4개 값만 꺼내 과거~현재 시퀀스를 만듭니다.
               - shape: (B, Pnn, time_len, 4)
            3) use_past_dit_input=True 이면:
               - (과거~현재) + (미래) 를 시간축으로 이어서 (past+current+future) 시퀀스를 만들고 flatten 합니다.
            4) use_past_dit_input=False 이면:
               - 기존처럼 (현재+미래) 또는 (미래만)으로 flatten 합니다.

        Args:
            inputs:
                - "target_cur_future_norm_xT": (B, Pnn, 1+future_len, 4)
                - "near_agents_past": (B, Pnn, time_len(=past_len+1), 11)  # 가정
            cond_last_pos_norm: (B, (1+)Pnn, 4)
        Returns:
            xT_input_flat:
                - use_past_dit_input=True  -> (B, (1+)Pnn, (time_len+future_len)*4)
                - use_current_input=True   -> (B, (1+)Pnn, (1+future_len)*4)
                - use_current_input=False  -> (B, (1+)Pnn, (future_len)*4)
        """
        B, one_or_Pnn = target_agents_past.shape[:2]
        # target_cur_future_norm_xT: (B, (1+)Pnn, 1+T, 4)
        target_cur_future_norm_xT: torch.Tensor = inputs[
            "target_cur_future_norm_xT"]
        # (B, (1+)Pnn, time_len, 4)
        target_agents_past_4_dim = target_agents_past[..., :4]

        #  (B, (1+)Pnn, T, 4)
        target_future_norm_xT: torch.Tensor = target_cur_future_norm_xT[:, :,
                                                                        1:, :]
        # 현재 프레임(정규화): 과거~현재 시퀀스의 마지막 프레임을 기준으로 잡아 정합성 강화
        # xT_input_seq 구성
        if self.config.use_past_dit_input:
            # dtype/device 정합: concat 전에 맞춰두는 게 안전
            target_agents_past_4_dim = _cast_like(target_agents_past_4_dim,
                                                  target_future_norm_xT)
            # (B, (1+)Pnn, time_len + T, 4)
            # (B, (1+)Pnn, time_len, 4) + (B, (1+)Pnn, T, 4)
            xT_input_seq: torch.Tensor = torch.cat(
                [target_agents_past_4_dim, target_future_norm_xT],
                dim=2,
            )
        else:
            if self.config.use_current_input:
                # (B, (1+)Pnn, 1+T, 4)
                xT_input_seq = target_cur_future_norm_xT
            else:
                # (B, (1+)Pnn, T, 4)
                xT_input_seq = target_future_norm_xT

        # flatten: (B, (1+)Pnn, F)
        xT_input_flat: torch.Tensor = xT_input_seq.reshape(B, one_or_Pnn, -1)

        return xT_input_flat

    def _extract_cur_future_from_score(
        self,
        score_flat: torch.
        Tensor,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        target_current_xyyaw: torch.Tensor,  # (B, Pnn, 4)
        batch_size: int,
        one_or_Pnn: int,
    ) -> torch.Tensor:
        """훈련 모드에서 DiT 출력(flat)을 (B, Pnn, 1+T, 4) 형태로 바꾼다.

        - use_past_dit_input / use_current_input 설정 조합을 그대로 반영합니다.
        - 최종적으로 Decoder가 리턴하는 score는 항상 (B, Pnn, 1+T, 4) 형태가 되도록 맞춥니다.

        Args:
            score_flat: DiT 출력(flat).
                shape: (B, Pnn, F_out)
            target_current_xyyaw: 현재 상태(정규화).
                shape: (B, Pnn, 4)

        Returns:
            score_cur_future:
                shape: (B, Pnn, 1+T, 4)
        """
        B: int = batch_size
        if self.config.use_past_dit_input:
            # score_flat: (B, Pnn, (time_len + future_len) * 4)
            score_cur_future = score_flat.reshape(
                B,
                one_or_Pnn,
                self.config.time_len + self._future_len,
                4,
            )
            # 마지막 (1+T)만 사용: (B, Pnn, 1+T, 4)
            score_cur_future = score_cur_future[:, :,
                                                -(1 + self._future_len):, :]
            return score_cur_future

        if self.config.use_current_input:
            # score_cur_future: (B, Pnn, 1+T, 4)
            score_cur_future = score_flat.reshape(B, one_or_Pnn,
                                                  1 + self._future_len, 4)
            return score_cur_future

        # use_current_input=False 인 경우: DiT 출력은 미래만 (B,Pnn,T,4)
        score_future = score_flat.reshape(B, one_or_Pnn, self._future_len,
                                          4)  # (B,Pnn,T,4)
        score_cur_future = torch.cat(
            [target_current_xyyaw.unsqueeze(2), score_future],
            dim=2,
        )  # (B,Pnn,1+T,4)
        return score_cur_future

    def _append_training_feasible_outputs(
            self,
            decoder_training_output: Dict[str, torch.Tensor],
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

        decoder_training_output["integrated_trajectory"] = integrated_trajectory
        decoder_training_output[
            "control_constraint_diff"] = self.dit.norm_dit_returns.control_constraint_diff  # (B, (1+)Pnn, T, 3)

    def _assert_past_cur_valid_mask(
        self,
        valid_bpt: torch.Tensor,
        context: str = "savgol_filter_for_control_past_cur",
    ) -> None:
        """과거~현재(valid_bpt)의 유효 마스크가 행마다 0*1* (단조 증가)인지 검증.

        Args:
            valid_bpt: (B, Pnn, T1) bool
            context: 에러 메시지용 위치 정보

        Raises:
            ValueError: 1→0 전이가 하나라도 있으면(단조 증가 위반) 예외
        """
        # <추가하자>
        assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 이어야 합니다."
        B, Pnn, T1 = valid_bpt.shape
        v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)

        has_10 = (d < 0).any(dim=1)  # 1→0 전이(단조 증가 위반)
        if has_10.any():
            bad_idx = torch.nonzero(has_10, as_tuple=False).flatten()
            max_show = min(int(bad_idx.numel()), 8)
            sample = bad_idx[:max_show].tolist()
            b_list = [(i // Pnn) for i in sample]
            p_list = [(i % Pnn) for i in sample]
            raise ValueError(
                f"[{context}] past_cur 유효 마스크는 0*1* 형태여야 합니다(단조 증가). "
                f"1→0 전이가 감지되었습니다. 오류 row 수={int(bad_idx.numel())}, "
                f"예시 (b,p)={list(zip(b_list, p_list))}.")

    def _assert_cur_future_valid_mask(
            self,
            valid_bpt: torch.Tensor,
            context: str = "savgol_filter_for_control") -> None:
        """유효 마스크가 행마다 True*False* (단조 감소)인지 검증.

        Args:
            valid_bpt: (B, Pnn, T1) bool, 시간 축 마지막.
            context: 에러 메시지에 표시할 호출 위치 문자열.

        Raises:
            ValueError: 0→1 전이가 하나라도 발견되면(내부 구멍 또는 선행 무효 후 유효)
        """
        assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 여야 합니다."
        B, Pnn, T1 = valid_bpt.shape
        v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
        has_01 = (d > 0).any(dim=1)  # 0→1 전이 여부
        if has_01.any():
            bad_idx = torch.nonzero(has_01, as_tuple=False).flatten()
            # 가독성을 위해 일부만 표시
            max_show = min(int(bad_idx.numel()), 8)
            bad_idx_sample = bad_idx[:max_show].tolist()
            # (b,p) 인덱스 매핑
            b_list = [(i // Pnn) for i in bad_idx_sample]
            p_list = [(i % Pnn) for i in bad_idx_sample]
            raise ValueError(
                f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*). \n"
                f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.numel())},  \n"
                f"example (b,p)={list(zip(b_list, p_list))}.  \n"
                f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
            )

    def _get_target_past_cur_future_valid(
        self,
        target_past_current_mask: torch.
        Tensor,  # (B, (1+)Pnn, (1+)time_len) True=무효  True=빈 슬롯(무효 에이전트)
        target_future_valid: torch.Tensor,  #  (B, (1+) Pnn, future_len) bool
    ) -> torch.Tensor:  # (B, (1+)Pnn, time_len + future_len)
        target_past_current_valid = ~target_past_current_mask  # [B, (1+)pnn, time_len]  True=유효 에이전트
        target_past_cur_future_valid = torch.cat(
            [target_past_current_valid, target_future_valid],
            dim=-1)  # [B, (1+)pnn, time_len + future_len] bool
        return target_past_cur_future_valid

    def _reshape_xt_with_current_state(
        self,
        xt: torch.Tensor,  # shape: (B, Pnn, flattened_dim)
        batch_size: int,
        one_or_Pnn: int,
        target_agents_past: torch.Tensor,  # shape: (B, Pnn, time_len, 11)
        target_current_xyyaw: torch.Tensor  # shape: (B, Pnn, 4)
    ) -> torch.Tensor:
        """샘플 xt 를 (B, Pnn, _, 4) 모양으로 펼치고
        '과거~현재 상태 프레임'을 맨 앞에 올바르게 넣어주는 메서드.
        """
        if self.config.use_past_dit_input:
            # xt_reshaped: (B, Pnn, time_len+T, 4)
            xt_reshaped = xt.reshape(
                batch_size,
                one_or_Pnn,
                self.config.time_len + self._future_len,
                4,
            )
            # 과거~현재 상태 주입
            xt_reshaped[:, :, :self.config.
                        time_len, :] = target_agents_past[:, :, :, 0:4]
        else:
            if self.config.use_current_input:
                # xt_reshaped: (B, Pnn, 1+T, 4)
                xt_reshaped = xt.reshape(
                    batch_size,
                    one_or_Pnn,
                    1 + self._future_len,
                    4,
                )
                xt_reshaped[:, :,
                            0, :] = target_current_xyyaw  # 현재 상태 주입 (B,Pnn,4)
            else:
                # xt_reshaped: (B, Pnn, T, 4)
                xt_reshaped = xt.reshape(
                    batch_size,
                    one_or_Pnn,
                    self._future_len,
                    4,
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
                cond_last_pos_norm:
                    (B, (1+)Pnn, 4)
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

        # cond_last_pos_norm: (B, (1+)Pnn, 4)
        cond_last_pos_norm: Optional[torch.Tensor] = inputs.get(
            "cond_last_pos_norm", None)
        if cond_last_pos_norm is None:
            cond_last_pos_norm = target_agents_past.new_full(
                (batch_size, one_or_Pnn, 4),
                float("nan"),
            )
        cond_last_pos_norm = _cast_like(cond_last_pos_norm, target_agents_past)

        return (
            target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            cond_last_pos_norm,  # (B, (1+)Pnn, 4)
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
    ]:
        """ encoder_outputs dict 에서 디코더가 쓰는 텐서들을 꺼낸다.

        Args:
            encoder_outputs: Encoder 단계에서 만들어진 출력 dict.

        Returns:
            scene_encoding_token: (B, token_num, D) 장면 토큰.
            scene_encoding_token_mask: (B, token_num) bool 마스크.
        """
        scene_encoding_token: torch.Tensor = encoder_outputs[
            "encoding"]  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor = encoder_outputs[
            "encoding_mask"]  # (B, token_num) bool
        return (
            scene_encoding_token,
            scene_encoding_token_mask,
        )

    def _sample_inference_noise(
        self,
        target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
        batch_size: int,
        one_or_Pnn: int,
    ) -> torch.Tensor:
        """추론 모드에서 시작점으로 쓸 미래 노이즈 궤적을 만든다.

        Returns:
            noise:
                미래 프레임 노이즈.
                shape: (B, Pnn, T, 4)
        """
        B: int = batch_size
        noise: torch.Tensor = target_current_xyyaw.new_empty(
            (B, one_or_Pnn, self._future_len, 4),).normal_(
                0.0, self.config.eval_temperature)  # (B, (1+)Pnn, T, 4)
        return noise

    def _build_inference_xT_from_noise(
        self,
        noise: torch.Tensor,  # (B,(1+)Pnn,T,4)
        target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
    ) -> torch.Tensor:  # (B, one_or_Pnn, (time_len+T)*4) or (B, one_or_Pnn, (1+T)*4) or (B, one_or_Pnn, T*4)
        """샘플링 시작점 xT(flat)를 만든다.


        Returns:
            xT:
                shape:
                  - use_past_dit_input=True  -> (B, one_or_Pnn, (time_len+T)*4)
                  - use_current_input=True  -> (B, one_or_Pnn, (1+T)*4)
                  - use_current_input=False -> (B, one_or_Pnn, T*4)
        """
        target_current_xyyaw = target_agents_past[:, :,
                                                  -1, :4]  # (B, (1+)Pnn, 4)
        B, one_or_Pnn = noise.shape[:2]
        if self.config.use_past_dit_input:
            # XT: (B, one_or_Pnn, time_len+T, 4)
            xT: torch.Tensor = torch.cat(
                [
                    target_agents_past[:, :, :, :
                                       4],  # (B, one_or_Pnn, time_len, 4)
                    noise,  # (B, one_or_Pnn, T, 4)
                ],
                dim=2,
            ).reshape(B, one_or_Pnn, -1)  # (B, one_or_Pnn, (time_len+T)*4)
        else:
            if self.config.use_current_input:
                # Expected size 1024 but got size 2048 for tensor number 1 in the list.
                xT: torch.Tensor = torch.cat(
                    [
                        target_current_xyyaw[:, :,
                                             None, :],  # (B, one_or_Pnn, 1, 4) # 1024
                        noise,  # (B, one_or_Pnn, T, 4) # 2048
                    ],
                    dim=2,
                ).reshape(B, one_or_Pnn, -1)  # (B, one_or_Pnn, (1+T)*4)
                return xT

            xT = noise.reshape(B, one_or_Pnn, -1)  # (B, one_or_Pnn, T*4)
        return xT

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
        if x_seq.dim() != 4 or int(x_seq.shape[-1]) != 4:
            raise ValueError("x_seq는 (B,P,S,4) 이어야 합니다. "
                             f"got shape={tuple(x_seq.shape)}")
        if target_past_cur_future_valid.dim() != 3:
            raise ValueError(
                "target_past_cur_future_valid는 (B,P,T_all) 이어야 합니다. "
                f"got shape={tuple(target_past_cur_future_valid.shape)}")

        B, P, S, _ = x_seq.shape
        if int(target_past_cur_future_valid.shape[0]) != int(B) or int(
                target_past_cur_future_valid.shape[1]) != int(P):
            raise ValueError(
                "B,P 축이 서로 맞아야 합니다. "
                f"x_seq(B,P)=({B},{P}), "
                f"valid(B,P)=({int(target_past_cur_future_valid.shape[0])},{int(target_past_cur_future_valid.shape[1])})"
            )

        T_all: int = int(target_past_cur_future_valid.shape[-1])
        if T_all < int(S):
            raise ValueError("valid 마스크의 시간 길이가 x_seq보다 짧습니다. "
                             f"T_all={T_all}, S={int(S)}")

        # 뒤쪽 S칸이 x_seq의 시간축과 대응(현재+미래 / 미래만 / 과거+미래 모두 대응)
        valid_sliced: torch.Tensor = _to_bool_mask(
            target_past_cur_future_valid[..., -int(S):]).to(
                device=x_seq.device, dtype=torch.bool)  # (B,P,S)

        # (B,P,S,4)에서 무효 칸을 0으로
        x_seq_out: torch.Tensor = x_seq.masked_fill(~valid_sliced.unsqueeze(-1),
                                                    0.0)
        return x_seq_out

    def _mask_invalid_timesteps_in_flat_xyyaw(
            self,
            x_flat: torch.Tensor,  # (B, P, F) where F%4==0
            target_past_cur_future_valid: torch.Tensor,  # (B, P, T_all)
    ) -> torch.Tensor:
        """flat 형태(F=시간*4) 텐서에서 무효 시간 칸을 0으로 정리합니다.

        Args:
            x_flat (torch.Tensor):
                shape: (B, P, F), F는 4의 배수
            target_past_cur_future_valid (torch.Tensor):
                shape: (B, P, T_all), True=유효 / False=무효

        Returns:
            torch.Tensor:
                shape: (B, P, F), 무효 칸은 0
        """
        if x_flat.dim() != 3:
            raise ValueError(
                f"x_flat must be (B,P,F), got {tuple(x_flat.shape)}")
        F: int = int(x_flat.shape[-1])
        if (F % 4) != 0:
            raise ValueError(
                f"x_flat last dim must be multiple of 4. got F={F}")

        B, P, _ = x_flat.shape
        S: int = int(F // 4)

        x_seq = x_flat.reshape(B, P, S, 4)  # (B,P,S,4)
        x_seq = self._mask_invalid_timesteps_to_zero(
            x_seq=x_seq,
            target_past_cur_future_valid=target_past_cur_future_valid,
        )
        return x_seq.reshape(B, P, F)

    def _initial_state_constraint(
            self,
            xt: torch.
        Tensor,  # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
            t: torch.Tensor,  # (B) or (B,future_len)
            step: int,
            *,
            batch_size: int,  # DONE
            one_or_Pnn: int,  # DONE
            target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
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
        xt_sequence: torch.Tensor = self._reshape_xt_with_current_state(
            xt=xt,
            batch_size=B,
            one_or_Pnn=one_or_Pnn,
            target_agents_past=target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
        )

        # (옵션) feasible 결과와 섞기
        if self.config.use_feasible_blend:
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
        xt_sequence = self._project_future_yaw_to_unit_circle(
            xt_sequence=xt_sequence,  # (B, (1+)Pnn, _, 4)
        )

        # ✅ 추가: 무효(패딩) 타임스텝을 항상 0으로 강제
        xt_sequence = self._mask_invalid_timesteps_to_zero(
            x_seq=xt_sequence,  # (B,P,S,4)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B,P,T_all)
        )

        # 다시 flatten: (B, Pnn, F)
        return xt_sequence.reshape(B, one_or_Pnn, -1)

    def _build_inference_correcting_xt_fn(
        self,
        batch_size: int,
        one_or_Pnn: int,
        target_agents_past: torch.Tensor,  # (B, Pnn, time_len, 11)
        target_past_cur_future_valid: torch.Tensor,  # (B, Pnn, time_len_total)
    ) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
        """샘플링 루프가 요구하는 시그니처(xt,t,step) 형태의 보정 함수를 만든다.

        내부적으로는 self._initial_state_constraint 를 사용하고,
        batch/마스크/현재상태 같은 '고정 정보'는 partial로 미리 묶어서 전달합니다.

        Args:
            batch_size: B
            one_or_Pnn: Pnn
            target_current_xyyaw: (B, Pnn, 4)
            target_past_cur_future_valid: (B, Pnn, time_len_total)

        Returns:
            correcting_xt_fn:
                callable(xt, t, step) -> xt_out
                xt: (B, Pnn, F), xt_out: (B, Pnn, F)
        """
        target_current_xyyaw = target_agents_past[:, :,
                                                  -1, :4]  # (B, (1+)Pnn, 4)
        return partial(
            self._initial_state_constraint,
            batch_size=batch_size,
            one_or_Pnn=one_or_Pnn,
            target_agents_past=target_agents_past,
            target_current_xyyaw=target_current_xyyaw,
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

    def _build_amortized_full_time_for_flattened_trajectory(
            self,
            t_tau: torch.Tensor,  # (B, future_len)
            reference_flat: torch.Tensor,  # (B, Pnn, F)
    ) -> torch.Tensor:
        """flat 궤적 모양(F=프레임수*4)에 맞는 '프레임별 시간값'을 만든다.

        규칙
        ----
        - 과거/현재 프레임(있다면): 시간값을 0으로 둡니다.
        - 미래 프레임: t_tau를 그대로 붙입니다.

        Args:
            t_tau (torch.Tensor): 미래 프레임 시간값.
                - shape: (B, future_len)
            reference_flat (torch.Tensor): flat 궤적 텐서(길이 맞추기 기준).
                - shape: (B, Pnn, F)  (F는 4의 배수)

        Returns:
            torch.Tensor: 전체 프레임 시간값.
                - shape: (B, total_len)
                  * total_len = time_len + future_len (use_past_dit_input=True)
                  * total_len = 1 + future_len       (use_current_input=True)
                  * total_len = future_len           (use_current_input=False)
        """
        if reference_flat.dim() != 3:
            raise ValueError(
                f"reference_flat must be 3D (B,Pnn,F). got {tuple(reference_flat.shape)}"
            )

        B: int = int(reference_flat.shape[0])
        flat_dim: int = int(reference_flat.shape[-1])
        if flat_dim % 4 != 0:
            raise ValueError(
                f"reference_flat last dim must be multiple of 4. got F={flat_dim}"
            )

        total_len: int = int(flat_dim // 4)
        future_len: int = int(self._future_len)

        if self.config.use_past_dit_input:
            expected_total_len: int = int(self.config.time_len + future_len)
        elif self.config.use_current_input:
            expected_total_len = int(1 + future_len)
        else:
            expected_total_len = int(future_len)

        if total_len != expected_total_len:
            raise ValueError(
                "flat 길이가 config와 맞지 않습니다. "
                f"total_len={total_len}, expected={expected_total_len}, F={flat_dim}"
            )

        prefix_len: int = int(total_len - future_len)
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")

        if prefix_len == 0:
            # t_full: (B, future_len)
            return t_tau.to(dtype=torch.float32, device=reference_flat.device)

        # t_prefix: (B, prefix_len)
        t_prefix: torch.Tensor = torch.zeros(
            (B, prefix_len),
            device=reference_flat.device,
            dtype=torch.float32,
        )
        # t_full: (B, total_len)
        t_full: torch.Tensor = torch.cat(
            [t_prefix, t_tau.to(torch.float32)], dim=1)
        return t_full

    def _compute_amortized_sigma2_over_alpha_flat(
        self,
        t_full: torch.Tensor,  # (B, total_len)
        batch_size: int,
        one_or_Pnn: int,
        reference_tensor_for_device: torch.Tensor,
    ) -> torch.Tensor:
        """프레임별 시간값(t_full)로 (노이즈크기^2 / 원래값비율) 텐서를 만든다.

        목적
        ----
        amortized 1-step에서 guidance를 x0_pred에 반영할 때
        프레임마다 다른 '노이즈 크기'가 있으므로,
        프레임별로 다른 스케일을 곱해 주기 위한 텐서입니다.

        구현 방식(팩트)
        --------------
        - Decoder의 amortized 노이즈 섞기에서 self.sde.marginal_prob(...)를 쓰고 있으므로,
          여기서도 같은 SDE로 "원래값비율(=mean 계수)"과 "노이즈 크기(=std)"를 구합니다.
        - x_dummy를 1로 두면 marginal_prob의 mean이 곧 "원래값비율"이 됩니다.

        Args:
            t_full (torch.Tensor): 전체 프레임 시간값.
                - shape: (B, total_len)
            batch_size (int): B
            one_or_Pnn (int): Pnn
            reference_tensor_for_device (torch.Tensor): device 기준 텐서.
                - shape: 임의

        Returns:
            torch.Tensor: flat 스케일 텐서.
                - shape: (B, Pnn, total_len*4)
                - dtype: float32
        """
        B: int = int(batch_size)
        Pnn: int = int(one_or_Pnn)
        total_len: int = int(t_full.shape[1])

        # t_full_f32: (B, total_len)
        t_full_f32: torch.Tensor = t_full.to(
            device=reference_tensor_for_device.device,
            dtype=torch.float32,
        )

        # x_dummy: (B, Pnn, total_len, 4)
        x_dummy: torch.Tensor = torch.ones(
            (B, Pnn, total_len, 4),
            device=reference_tensor_for_device.device,
            dtype=torch.float32,
        )

        # mean_dummy: (B, Pnn, total_len, 4)  (x_dummy==1 이므로 "원래값비율" 역할)
        # std:        (B, 1, total_len, 1) 또는 (B, Pnn, total_len, 1) 등 (브로드캐스트 가능)
        mean_dummy, std = self.sde.marginal_prob(x_dummy, t_full_f32)

        # alpha_safe: (B, Pnn, total_len, 4)
        alpha_safe: torch.Tensor = mean_dummy.clamp_min(1e-6)

        # sigma2_over_alpha: (B, Pnn, total_len, 4)
        sigma2_over_alpha: torch.Tensor = (std**2) / alpha_safe

        # t==0인 프레임(과거/현재)은 guidance 스케일을 0으로 강제
        # t_mask: (B, 1, total_len, 1)
        t_mask: torch.Tensor = (t_full_f32 > 0.0).to(torch.float32).view(
            B, 1, total_len, 1)
        sigma2_over_alpha = sigma2_over_alpha * t_mask  # (B, Pnn, total_len, 4)

        # flat: (B, Pnn, total_len*4)
        sigma2_over_alpha_flat: torch.Tensor = sigma2_over_alpha.reshape(
            B, Pnn, total_len * 4)
        return sigma2_over_alpha_flat

    def _compute_classifier_cond_grad_for_guidance(
        self,
        xT_flat: torch.Tensor,  # (B, Pnn, F)
        t_tau: torch.Tensor,  # (B,)
        classifier_kwargs: Dict[str, object],
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """guidance가 xT를 어떤 방향으로 움직이고 싶어 하는지(미분 값)를 계산합니다.

        이 함수는 "guidance 함수가 낸 점수(또는 로그값)"를 xT에 대해 미분해서,
        xT를 어느 방향으로 바꾸면 그 값이 커지는지(=기울기)를 구합니다.

        주의 사항(팩트)
        -------------
        - torch의 미분 계산을 사용하므로, 일반 no_grad 추론보다 연산량이 늘어납니다.
        - guidance 함수 내부에서 모델을 추가로 부르면 비용이 더 늘 수 있습니다.
          (이건 guidance_fn 구현에 달려 있어 여기서는 확정할 수 없습니다.)

        Args:
            xT_flat (torch.Tensor):
                노이즈가 섞인 입력(flat).
                shape: (B, Pnn, F)
            t_tau (torch.Tensor):
                미래 프레임별 시간값.
                shape: (B, future_len)
            classifier_kwargs (Dict[str, object]):
                guidance 함수에 넘길 추가 정보.
            condition (Optional[torch.Tensor]):
                guidance 함수의 cond 인자. 기본 None.

        Returns:
            torch.Tensor:
                xT_flat과 같은 모양의 미분 값.
                shape: (B, Pnn, F)
                dtype: float32
        """
        if self._guidance_fn is None:
            raise ValueError(
                "self._guidance_fn is None but cond_grad was requested.")

        # t_tau_f32: (B,future_len))
        t_tau_f32: torch.Tensor = t_tau.to(device=xT_flat.device,
                                           dtype=torch.float32)

        with torch.inference_mode(False):
            with torch.enable_grad():
                # x_in: (B, Pnn, F)
                x_in: torch.Tensor = xT_flat.clone().detach().requires_grad_(
                    True)

                log_prob: torch.Tensor = self._guidance_fn(
                    x_in,
                    t_tau_f32,
                    condition,
                    **classifier_kwargs,
                )

                # cond_grad: (B, Pnn, F)
                cond_grad: torch.Tensor = torch.autograd.grad(
                    log_prob.sum(),
                    x_in,
                    retain_graph=False,
                    create_graph=False,
                )[0]

        return cond_grad.to(dtype=torch.float32)

    def _apply_classifier_guidance_for_amortized_one_step(
        self,
        xT_flat: torch.Tensor,  # (B, Pnn, F)
        x0_pred_flat: torch.Tensor,  # (B, Pnn, F)
        t_tau: torch.Tensor,  # (B, future_len)
        classifier_kwargs: Dict[str, object],
        guidance_scale: float,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """amortized 1-step 출력(x0_pred)에 classifier guidance를 반영한다.

        구현(팩트)
        ----------
        dpm_solver_pytorch.py의 classifier guidance는 "노이즈 예측"을 다음처럼 바꿉니다.

            noise_guided = noise - guidance_scale * sigma * cond_grad

        이 프로젝트에서 DiT는 x0를 직접 예측(model_type="x_start")하므로,
        위 식을 x0 쪽으로 정리하면 아래처럼 됩니다.

            x0_guided = x0_pred + guidance_scale * (sigma^2 / alpha) * cond_grad

        amortized에서는 프레임마다 시간값이 다르므로,
        sigma^2/alpha를 프레임별로 계산해 flat 텐서에 맞춰 곱합니다.

        Args:
            xT_flat (torch.Tensor): noisy 입력(flat).
                - shape: (B, Pnn, F)
            x0_pred_flat (torch.Tensor): DiT가 예측한 결과(flat).
                - shape: (B, Pnn, F)
            t_tau (torch.Tensor): 미래 프레임 시간값.
                - shape: (B, future_len)
            classifier_kwargs (Dict[str, object]): guidance 함수에 넘길 추가 정보.
            guidance_scale (float): guidance 강도. 0이면 아무 변화가 없습니다.
            condition (Optional[torch.Tensor]): guidance 함수의 cond 인자. 기본 None.

        Returns:
            torch.Tensor: guidance가 반영된 결과(flat).
                - shape: (B, Pnn, F)
                - dtype: float32
        """
        if self._guidance_fn is None:
            return x0_pred_flat
        if float(guidance_scale) == 0.0:
            return x0_pred_flat

        B: int = int(xT_flat.shape[0])
        Pnn: int = int(xT_flat.shape[1])

        # t_full: (B, total_len)
        t_full: torch.Tensor = self._build_amortized_full_time_for_flattened_trajectory(
            t_tau=t_tau,  # (B, future_len)
            reference_flat=xT_flat,  #  (B, Pnn, F)
        )

        # sigma2_over_alpha_flat: (B, Pnn, F)
        sigma2_over_alpha_flat: torch.Tensor = self._compute_amortized_sigma2_over_alpha_flat(
            t_full=t_full,
            batch_size=B,
            one_or_Pnn=Pnn,
            reference_tensor_for_device=xT_flat,
        )

        # cond_grad: (B, Pnn, F)
        cond_grad: torch.Tensor = self._compute_classifier_cond_grad_for_guidance(
            xT_flat=xT_flat,
            t_tau=t_tau,
            classifier_kwargs=classifier_kwargs,
            condition=condition,
        )

        # x0_guided: (B, Pnn, F)
        x0_guided: torch.Tensor = x0_pred_flat.to(torch.float32) + (
            float(guidance_scale) * sigma2_over_alpha_flat * cond_grad)
        return x0_guided

    def _build_classifier_kwargs_for_guidance(
        self,
        *,
        target_agents_past: Optional[torch.Tensor],
        scene_encoding_token: torch.Tensor,  # (B, token_num, D)
        scene_encoding_token_mask: torch.Tensor,  # (B, token_num)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, time_len_total)
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """guidance 함수가 필요로 하는 부가 정보를 한 번에 구성합니다.

        이 함수는
            DPM-Solver 경로(diffusion_steps>1) 와
            amortized 1-step 경로(diffusion_steps==1) 에서
        **같은 형태의 classifier_kwargs**를 쓰도록 만들기 위한 공통 유틸입니다.
        Returns:
            Dict[str, object]:
                GuidanceWrapper가 받는 **classifier_kwargs** dict.
        """
        model_condition: Dict[str, Any] = {
            "target_agents_past": target_agents_past,
            "target_past_cur_future_valid": target_past_cur_future_valid,
            "cross_c": scene_encoding_token,
            "cross_mask": scene_encoding_token_mask,
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

    def _run_dpm_sampler_for_inference(
        self,
        xT: torch.Tensor,
        target_agents_past: Optional[torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        target_past_cur_future_valid: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        correcting_xt_fn: Callable[[torch.Tensor, torch.Tensor, int],
                                   torch.Tensor],
        diffusion_steps: int,
    ) -> torch.Tensor:
        """dpm_sampler(또는 amortized 1-step)을 통해 최종 샘플 x0(flat)을 얻는다."""
        classifier_kwargs: Dict[
            str, object] = self._build_classifier_kwargs_for_guidance(
                target_agents_past=target_agents_past,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                target_past_cur_future_valid=target_past_cur_future_valid,
                inputs=inputs,
            )

        # -----------------------------
        # diffusion_steps > 1 (기존 DPM-Solver 경로)
        # -----------------------------
        if diffusion_steps > 1:
            #  (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
            x0: torch.Tensor = dpm_sampler(
                self.dit,
                xT.float(),
                diffusion_steps=diffusion_steps,
                other_model_params={
                    "target_agents_past":
                        target_agents_past,
                    "target_past_cur_future_valid":
                        target_past_cur_future_valid,
                    "cross_c":
                        scene_encoding_token,
                    "cross_mask":
                        scene_encoding_token_mask,
                },
                dpm_solver_params={
                    "correcting_xt_fn": correcting_xt_fn,
                },
                model_wrapper_params={
                    "classifier_fn":
                        self._guidance_fn,
                    "classifier_kwargs":
                        classifier_kwargs,
                    "guidance_scale":
                        self.config.guidance_scale,
                    "guidance_type": ("classifier" if self._guidance_fn
                                      is not None else "uncond"),
                },
            )
            return x0

        # -----------------------------
        # diffusion_steps == 1 (amortized 1-step)
        # -----------------------------
        assert diffusion_steps == 1, "diffusion_steps must be >= 1"

        B: int = int(xT.shape[0])
        xT_f32: torch.Tensor = xT.float()  # (B, Pnn, F)

        # t_tau: (B, future_len)
        t_tau: torch.Tensor = self._build_amortized_t_tau(
            batch_size=B,
            reference_tensor_for_device=xT_f32,
        )

        # low_t_mask: (B,)  amortized에서는 feasible을 항상 실행시키기 위한 마스크(기존 의도 유지)
        low_t_mask: torch.Tensor = torch.ones((B,),
                                              dtype=torch.bool,
                                              device=xT.device)

        # (1) 모델 1회 호출: x0_pred (flat)
        # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
        x0_pred: torch.Tensor = self.dit(
            target_input_norm_xT=xT_f32,  # (B, Pnn, F)
            diffusion_time=t_tau,  # (B, future_len)
            target_agents_past=target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            target_past_cur_future_valid=target_past_cur_future_valid,
            # (B, (1+)Pnn, time_len_total)
            cross_c=scene_encoding_token,  # (B, token_num, D)
            cross_mask=scene_encoding_token_mask,  # (B, token_num)
            low_t_mask=low_t_mask,  # (B,)
        )

        # (2) guidance를 x0_pred에 반영 (선택)
        if self._guidance_fn is not None and float(
                self.config.guidance_scale) != 0.0:
            # guidance_fn이 low_t_mask를 쓰는 경우를 위해 전달(너의 기존 의도 유지)
            classifier_kwargs["low_t_mask"] = low_t_mask

            x0_pred = self._apply_classifier_guidance_for_amortized_one_step(
                xT_flat=xT_f32,  # (B,Pnn,F)
                x0_pred_flat=x0_pred,  # (B,Pnn,F)
                t_tau=t_tau,  # (B,future_len)
                classifier_kwargs=classifier_kwargs,
                guidance_scale=self.config.guidance_scale,
                condition=None,
            )

        # (3) correcting_xt_fn을 x0 공간에서 1회 적용 (선택)
        if correcting_xt_fn is not None:
            x0_pred = correcting_xt_fn(x0_pred, t_tau, 0)

        _require_finite("amortized_one_step_x0", x0_pred)
        return x0_pred

    def _reshape_inference_x0_to_sequence(
            self,
            x0: torch.
        Tensor,  # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
    ) -> torch.Tensor:
        """샘플링 결과 x0(flat)를 (B, (1+)Pnn, 1+T, 4) 형태로 복원한다.
        Returns:
            x0_seq:
                shape: (B, (1+)Pnn, 1+T, 4)
        """
        batch_size, one_or_Pnn = x0.shape[:2]
        B: int = batch_size
        if self.config.use_past_dit_input:
            x0_seq = x0.reshape(
                B,
                one_or_Pnn,
                self.config.time_len + self._future_len,
                4,
            )
            x0_seq = x0_seq[:, :, -(1 + self._future_len):, :]  # (B,Pnn,1+T,4)
            return x0_seq
        if self.config.use_current_input:
            assert x0.shape == (B, one_or_Pnn, (1 + self._future_len) * 4)
            x0_seq = x0.reshape(B, one_or_Pnn, -1, 4)  # (B,Pnn,1+T,4)
            return x0_seq

        assert x0.shape == (B, one_or_Pnn, self._future_len * 4)
        x0_seq = torch.cat(
            [
                target_current_xyyaw.unsqueeze(2),  # (B,Pnn,1,4)
                x0.reshape(B, one_or_Pnn, -1, 4),  # (B,Pnn,T,4)
            ],
            dim=2,
        )  # (B,Pnn,1+T,4)
        return x0_seq

    def _append_inference_feasible_outputs(
            self,
            return_norm_dict: Dict[str, torch.Tensor],  #
            target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
    ) -> None:
        """추론 모드에서 feasible 출력(integrated_trajectory)을 outputs에 추가한다.
        Returns:
            None
        """
        if not self.config.use_feasible:
            return

        # integrated_trajectory: (B, (1+)Pnn, T, 4)  (정규화 상태)
        integrated_trajectory: torch.Tensor = self.dit.norm_dit_returns.integrated_trajectory
        integrated_trajectory = torch.cat(
            [target_current_xyyaw.unsqueeze(2), integrated_trajectory],
            dim=2,
        )  # (B, (1+)Pnn, 1+T, 4)
        return_norm_dict["integrated_trajectory"] = integrated_trajectory

    def _forward_training_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        target_agents_past: torch.Tensor,  # (B, (1+)Pnn, time_len, 11)
        target_past_cur_future_valid: torch.Tensor,
        batch_size: int,
        one_or_Pnn: int,
    ) -> Dict[str, torch.Tensor]:
        """훈련 모드에서 한 배치에 대해 decoder 를 한 번 돌린다."""
        decoder_training_output: Dict[str, torch.Tensor] = {}

        B: int = batch_size
        diffusion_time: torch.Tensor = inputs[
            "diffusion_time"]  # (B,) or (B, future_len)

        # 1) DiT 입력 준비 (flatten + 목표점 주입 옵션)
        """ xT_input_flat
        (B, (1+)Pnn, (time_len+future_len)*4) or 
        (B, (1+)Pnn, (1+future_len)*4) or 
        (B, (1+)Pnn, future_len*4)
        """
        xT_input_flat = self._build_training_dit_inputs(
            inputs=inputs,
            target_agents_past=target_agents_past,  # (B, (1+)Pnn, time_len, 11)
        )

        # ✅ 추가: DiT 입력에서도 무효 타임스텝 0 처리(실수로 downstream에서 쓰여도 안전)
        xT_input_flat = self._mask_invalid_timesteps_in_flat_xyyaw(
            x_flat=xT_input_flat,
            target_past_cur_future_valid=target_past_cur_future_valid,
        )

        # ✅ (B) diffusion_time: 미리 GPU float32로 맞춤 (DiT 내부 .to()가 no-op 되도록)
        diffusion_time = _prepare_diffusion_time_for_dit(
            diffusion_time=diffusion_time,  # (B,) or (B,future_len)
            reference_tensor=xT_input_flat,  # device 기준
        )

        # ✅ (C) low_t_mask도 bool + 같은 device로 (필요하면만) 변환
        low_t_mask_in: torch.Tensor = inputs["low_t_mask"]  # (B,) bool/0-1
        low_t_mask: torch.Tensor = _prepare_bool_mask_on_device(
            low_t_mask_in,
            device=xT_input_flat.device,
        )  # (B,) bool

        # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        score_flat: torch.Tensor = self.dit(
            target_input_norm_xT=
            xT_input_flat,  # (B, (1+)Pnn, F) #  F 에서 시간 길이는 time_len + future_len 또는 1 + future_len 또는 future_len
            diffusion_time=diffusion_time,  # (B,)  or (B, future_len)
            target_agents_past=target_agents_past,  # # (B, (1+)Pnn, time_len, 11)
            target_past_cur_future_valid=target_past_cur_future_valid,
            # (B, (1+)Pnn, time_len+future_len)
            cross_c=scene_encoding_token,  # (B, token_num, D)
            cross_mask=scene_encoding_token_mask,  # (B, token_num)
            low_t_mask=low_t_mask,
        )
        _require_finite("decoder_dit_output", score_flat)
        # (B, (1+)Pnn, 4)
        target_current_xyyaw = target_agents_past[:, :, -1, :4]
        # 3) (B,(1+)Pnn,F_out) -> (B,(1+)Pnn,1+T,4)
        score_cur_future: torch.Tensor = self._extract_cur_future_from_score(
            score_flat=
            score_flat,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            batch_size=B,
            one_or_Pnn=one_or_Pnn,
        )
        decoder_training_output[
            "score"] = score_cur_future  # (B, (1+)Pnn, 1+T, 4)
        """ decoder_training_output
        score : (B, (1+)Pnn, 1+T, 4)
        "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
        "control_constraint_diff" : (B, (1+)Pnn, T, 3)
        """
        # 4) feasible 출력 추가(옵션)
        self._append_training_feasible_outputs(
            decoder_training_output=decoder_training_output,
            target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
        )

        return decoder_training_output

    def _get_noise_trajectory_from_prev_trajectory(
        self,
        rollout_time_chunk_size: int,
        random_noise: torch.Tensor,
    ) -> torch.Tensor:
        """이전 step의 버퍼(self._x0_for_amortized_inference)로부터 다음 step의 noise_trajectory를 만듭니다.

        변경점(핵심)
        ----------
        - 기존: torch.randn_like(...)로 내부에서 랜덤 생성
        - 변경: 외부에서 전달된 random_noise를 사용 (cand_idx 기반 seed로 재현 가능)

        Args:
            rollout_time_chunk_size (int):
                이번에 실행(execute)한 시간 길이(gap).
            random_noise (torch.Tensor):
                표준정규 랜덤 텐서.
                shape: (B, (1+)Pnn, future_len, 4)

        Returns:
            torch.Tensor:
                noise_trajectory: (B, (1+)Pnn, future_len, 4)
        """
        if self._x0_for_amortized_inference is None:
            raise RuntimeError("self._x0_for_amortized_inference is None.")

        if not isinstance(random_noise, torch.Tensor):
            raise TypeError("random_noise must be torch.Tensor.")

        if tuple(random_noise.shape) != tuple(
                self._x0_for_amortized_inference.shape):
            raise ValueError(
                "random_noise shape가 self._x0_for_amortized_inference와 같아야 합니다. "
                f"random_noise={tuple(random_noise.shape)}, "
                f"buffer={tuple(self._x0_for_amortized_inference.shape)}")

        # (B, (1+)Pnn, future_len, 4)
        x0_shifted = torch.zeros_like(self._x0_for_amortized_inference)
        B = int(x0_shifted.shape[0])

        # 앞쪽으로 당기기 (rollout_time_chunk_size 만큼)
        if int(rollout_time_chunk_size) > 0:
            x0_shifted[:, :, :-int(rollout_time_chunk_size), :] = \
                self._x0_for_amortized_inference[
                    :, :, int(rollout_time_chunk_size):, :]

        # batch_diffusion_time: (B, future_len)
        batch_diffusion_time: torch.Tensor = self.t_tau.unsqueeze(0).repeat(
            B, 1).to(device=self._x0_for_amortized_inference.device)

        mean, std = self.sde.marginal_prob(x0_shifted, batch_diffusion_time)

        # ✅ 외부에서 받은 랜덤 사용
        random_noise = _ensure_tensor_on_ref(random_noise,
                                             self._x0_for_amortized_inference)

        noise_trajectory: torch.Tensor = mean + std * random_noise
        return noise_trajectory

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

    def _update_amortized_future_buffer_origin(
            self,
            rollout_time_chunk_size: int,
            target_future_valid: torch.Tensor,  # (B, (1+)Pnn, future_len)
    ) -> None:
        """amortized 추론에서 재사용하는 '미래 버퍼'의 좌표 기준을 다음 스텝 기준으로 맞춥니다.

        이 함수가 필요한 이유
        -------------------
        평가/롤아웃 코드에서는 매 스텝(또는 chunk)마다 입력 전체를 "새 ego 기준(0,0)"으로 바꿉니다.
        그런데 Decoder 안의 미래 버퍼(self._x0_for_amortized_inference)는 다음 스텝에도 재사용되므로,
        이 버퍼도 똑같이 기준을 바꾸지 않으면,
        다음 스텝에서 "과거/현재(새 기준)"과 "미래 버퍼(옛 기준)"가 섞일 수 있습니다.

        처리 순서(중요)
        -------------
        self._x0_for_amortized_inference 는 모델이 쓰는 스케일(정규화된 값)로 저장되어 있습니다.
        이동/회전은 실제 단위에서 하는 것이 안전하므로 아래 순서를 지킵니다.

          1) 버퍼를 state_normalizer.inverse 로 실제 단위로 되돌립니다.
          2) rollout_time_chunk_size 시점의 ego 포즈를 기준점으로 삼습니다.
          3) 버퍼 전체를 같은 기준으로 이동/회전합니다.
          4) 다시 state_normalizer 로 모델 입력 스케일로 맞춥니다.
          5) (cos, sin)이 길이 1이 되도록 한 번 더 정리합니다.

        """
        if self._x0_for_amortized_inference is None:
            return

        # buffer_norm: (B, (1+)Pnn, future_len, 4)
        buffer_norm: torch.Tensor = self._x0_for_amortized_inference

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

        self._x0_for_amortized_inference = buffer_norm_new

    def _forward_inference_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        scene_encoding_token: torch.Tensor,
        scene_encoding_token_mask: torch.Tensor,
        target_agents_past: torch.Tensor,
        # (B, (1+)Pnn, time_len, 11)
        target_past_cur_future_valid: torch.Tensor,
        batch_size: int,
        one_or_Pnn: int,
    ) -> Dict[str, torch.Tensor]:
        """ return_norm_dict
        "score" : (B, (1+)Pnn, 1+T, 4)
        "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
        """
        with torch.no_grad():
            target_current_xyyaw = target_agents_past[:, :,
                                                      -1, :4]  # (B,(1+)Pnn,4)
            return_norm_dict: Dict[str, torch.Tensor] = {}
            B: int = batch_size
            # 1) 시작 노이즈 생성 (미래만)
            # ✅ (변경) 외부에서 만든 noise를 반드시 입력으로 받는다.
            # noise: (B,(1+)Pnn,T,4)
            """
            noise_trajectory 가 들어온 경우
                1) use_amortized_diffusion 가 False인 경우
                2) use_amortized_diffusion 가 True인 경우이면서 첫번쨰 샘플링인 경우
            
            noise_trajectory 가 들어오지 않은 경우
                1) use_amortized_diffusion 가 True이면서 첫 스텝이 아닌 경우
            """
            noise_trajectory = inputs.get("inference_noise", None)
            if self.config.use_amortized_diffusion:
                if noise_trajectory is not None:  # 첫번쨰 샘플링
                    self._x0_for_amortized_inference = None
                    # noise_trajectory: (B,(1+)Pnn,T,4)
                    noise_trajectory = self._get_inference_noise_from_inputs(
                        inputs=inputs,
                        batch_size=int(B),
                        one_or_Pnn=int(one_or_Pnn),
                        target_current_xyyaw=target_current_xyyaw,
                    )
                    diffusion_steps = 16
                else:  # 첫 스텝이 아닌 경우
                    diffusion_steps = 1
                    assert self._x0_for_amortized_inference is not None, (
                        "When using amortized diffusion during inference, "
                        "if inference_noise is not provided, "
                        "self._x0_for_amortized_inference must be set.")

                    rollout_time_chunk_size = inputs["rollout_time_chunk_size"]
                    rollout_time_chunk_size_int = int(
                        rollout_time_chunk_size[0].item())

                    # ✅ cand_idx 기반 seed로 만든 랜덤을 inputs에서 받아 사용
                    amortized_random_noise = self._get_amortized_random_noise_from_inputs(
                        inputs=inputs,
                        batch_size=int(B),
                        one_or_Pnn=int(one_or_Pnn),
                        target_current_xyyaw=target_current_xyyaw,
                    )

                    noise_trajectory = self._get_noise_trajectory_from_prev_trajectory(
                        rollout_time_chunk_size=rollout_time_chunk_size_int,
                        random_noise=amortized_random_noise,
                    )
            else:  # 기존 DPM-Solver 경로
                assert self._x0_for_amortized_inference is None, (
                    "self._x0_for_amortized_inference should be None when not using amortized diffusion."
                )
                diffusion_steps = 10
                if noise_trajectory is None:
                    raise ValueError(
                        "inference_noise must be provided "
                        "in inputs when not using amortized diffusion.")
                    # noise_trajectory: torch.Tensor = self._sample_inference_noise(
                    #     target_current_xyyaw=
                    #     target_current_xyyaw,  # (B,(1+)Pnn,4)
                    #     batch_size=B,
                    #     one_or_Pnn=one_or_Pnn,
                    # )  # (B,(1+)Pnn,T,4)
                else:
                    noise_trajectory: torch.Tensor = self._get_inference_noise_from_inputs(
                        inputs=inputs,
                        batch_size=int(B),
                        one_or_Pnn=int(one_or_Pnn),
                        target_current_xyyaw=target_current_xyyaw,
                    )  # (B,(1+)Pnn,T,4)

            # 2) xT(flat) 생성
            # (B, (1+)Pnn, (time_len+T)*4) or (B, (1+)Pnn, (1+T)*4) or (B, Pnn, T*4)
            xT: torch.Tensor = self._build_inference_xT_from_noise(
                noise=noise_trajectory,  # (B,(1+)Pnn,T,4)
                target_agents_past=target_agents_past,  # (B,(1+)Pnn,time_len,11)
            )  # (B,Pnn,F)
            # ✅ 추가: 샘플링 시작점부터 무효 타임스텝 0 처리
            xT = self._mask_invalid_timesteps_in_flat_xyyaw(
                x_flat=xT,
                target_past_cur_future_valid=target_past_cur_future_valid,
            )

            # 4) 샘플링 중 보정 함수 구성
            correcting_xt_fn = self._build_inference_correcting_xt_fn(
                batch_size=B,
                one_or_Pnn=one_or_Pnn,
                target_agents_past=target_agents_past,  # (B,Pnn,time_len,11)
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B,Pnn,time_total)
            )

            # 5) dpm_sampler 실행
            # x0: (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
            x0: torch.Tensor = self._run_dpm_sampler_for_inference(
                xT=
                xT,  # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
                target_agents_past=target_agents_past,  # (B,Pnn,time_len,11)
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                target_past_cur_future_valid=target_past_cur_future_valid,
                inputs=inputs,
                correcting_xt_fn=correcting_xt_fn,
                diffusion_steps=diffusion_steps,
            )

            # dtype 맞춤(기존 로직 유지)
            x0 = x0.to(xT.dtype)

            # 6) x0_seq_norm: (B, (1+)Pnn, 1+T, 4)
            x0_seq_norm: torch.Tensor = self._reshape_inference_x0_to_sequence(
                x0=
                x0,  # (B, Pnn, (time_len+T)*4) or (B, Pnn, (1+T)*4) or (B, Pnn, T*4)
                target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            )
            # self.dit.norm_dit_returns.integrated_trajectory : (B, (1+)Pnn, T, 4) 정규화 상태
            if self.config.use_amortized_diffusion:
                # self._x0_for_amortized_inference = x0_seq_norm[:, :,
                #                                                1:, :].detach(
                #                                                )  # (B, 1+Pnn, T, 4)
                integrated = self.dit.norm_dit_returns.integrated_trajectory.to(
                    device=target_current_xyyaw.device,
                    dtype=target_current_xyyaw.dtype,
                ).detach()

                self._x0_for_amortized_inference = integrated  # (B, 1+Pnn, T, 4)
                rollout_time_chunk_size = inputs["rollout_time_chunk_size"]
                rollout_time_chunk_size_int = int(
                    rollout_time_chunk_size[0].item())
                # target_future_valid: (B, (1+)Pnn, T)
                target_future_valid = target_past_cur_future_valid[:, :, -self.
                                                                   _future_len:].detach(
                                                                   )
                # ✅ 미래 버퍼도 다음 스텝 기준 좌표로 맞추기
                self._update_amortized_future_buffer_origin(
                    rollout_time_chunk_size=rollout_time_chunk_size_int,
                    target_future_valid=target_future_valid,
                )

            # 8) feasible 출력(옵션)
            # return_norm_dict["integrated_trajectory"] : (B, (1+)Pnn, 1+T, 4)
            self._append_inference_feasible_outputs(
                return_norm_dict=return_norm_dict,
                target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            )
            return_norm_dict["score"] = x0_seq_norm  # (B, (1+)Pnn, 1+T, 4)
            """ return_norm_dict
            "score" : (B, (1+)Pnn, 1+T, 4)
            "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
            """
            return return_norm_dict

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
            Dict[str, torch.Tensor]: 최소한 "score" 키를 가지며,
            설정에 따라 "integrated_trajectory",
            "control_constraint_diff" 도 포함될 수 있다.
        """
        (
            target_agents_past,  # (B, (1+)Pnn, time_len, 11)
            cond_last_pos_norm,  # (B, (1+)Pnn, 4)
            target_past_cur_future_valid,  # (B, (1+)Pnn, time_len + future_len)
            batch_size,
            one_or_Pnn,
        ) = self._prepare_target_trajectories_and_masks(inputs)
        (
            scene_encoding_token,  # (B, token_num, D)
            scene_encoding_token_mask,  # (B, token_num)
        ) = self._unpack_encoder_outputs(encoder_outputs=encoder_outputs,)
        # ✅ (A) cross_c / (C) cross_mask 를 DiT 호출 전에 미리 정리
        # - training에서 작은 dtype 계산을 쓰는 경우(bf16/fp16), 여기서 맞춰 두면
        #   DiT 내부의 _cast_like(cross_c, x)가 복사 없이 끝납니다.
        scene_encoding_token, scene_encoding_token_mask = _prepare_cross_inputs_for_dit(
            cross_c=scene_encoding_token,  # (B, token_num, D)
            cross_mask=scene_encoding_token_mask,  # (B, token_num)
            reference_tensor=target_agents_past,  # device/dtype 기준
        )

        if self.training:
            return self._forward_training_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                target_agents_past=
                target_agents_past,  # (B, (1+)Pnn, time_len, 11)
                target_past_cur_future_valid=target_past_cur_future_valid,
                batch_size=batch_size,
                one_or_Pnn=one_or_Pnn,
            )
        else:
            """ return_norm_dict
            "score" : (B, (1+)Pnn, 1+T, 4)
            "integrated_trajectory" : (B, (1+)Pnn, 1+T, 4)
            """
            return self._forward_inference_mode(
                inputs=inputs,
                scene_encoding_token=scene_encoding_token,
                scene_encoding_token_mask=scene_encoding_token_mask,
                target_agents_past=target_agents_past,
                # (B, (1+)Pnn, time_len, 11)
                target_past_cur_future_valid=target_past_cur_future_valid,
                batch_size=batch_size,
                one_or_Pnn=one_or_Pnn,
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
        self.norm_dit_returns = None
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
        # if self.config.use_amortized_diffusion:
        output_dim_pre = int(output_dim / 4 * 6)
        # else:
        #     output_dim_pre = output_dim
        self.preproj = Mlp(
            in_features=
            output_dim_pre,  # x, y, cos(yaw), sin(yaw), noising timestep, validity
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

    def preproj_varlen(
            self,
            target_input_norm_xT: torch.Tensor,  # (B, (1+)Pnn, F=_*6)
            target_current_mask: torch.Tensor,  # (B, (1+)Pnn) True=pad(무효 에이전트)
    ) -> torch.Tensor:
        """pre-proj MLP 를 유효 에이전트 토큰에만 적용하는 전처리 함수.

        마스크를 이용해 유효 토큰만 펼친 뒤 MLP 를 통과시키고,
        다시 배치 모양으로 되돌립니다.


        Returns:
            torch.Tensor:
                pre-proj 후 토큰.
                shape: (B, Pnn, D)
        """
        B, Pnn, F = target_input_norm_xT.shape  # (B, Pnn, F)
        # unpad_input 은 True=유효 이므로 반전 필요
        attention_mask = (~target_current_mask).to(torch.bool)  # (B, Pnn)

        res = unpad_input(target_input_norm_xT, attention_mask)
        # x_unpad: (T_total, F), indices: (T_total,)
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        else:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        if x_unpad.numel() == 0:
            D_out = self.preproj.fc2.out_features
            zeros = target_input_norm_xT.new_zeros((B, Pnn, D_out))
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

    @staticmethod
    def _unpad_input_with_valid_mask(
        x: torch.Tensor,  # (B, L, C)
        valid_mask: torch.Tensor  # (B, L) True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """flash_attn.bert_padding.unpad_input 결과를 버전 차이(4/5개 반환)와 무관하게 정리합니다.

        Args:
            x (torch.Tensor): (B, L, C) 입력
            valid_mask (torch.Tensor): (B, L) True=유효 토큰

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - x_unpad: (T, C)
                - indices: (T,)
                - cu_seqlens: (B+1,) int32
                - max_seqlen: int
        """
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.to(torch.bool)

        res = unpad_input(x, valid_mask)
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, _ = res
        else:
            raise RuntimeError(
                f"unexpected unpad_input return size: {len(res)}")

        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen_int = int(max_seqlen)
        return x_unpad, indices, cu_seqlens, max_seqlen_int

    def preproj_varlen_packed(
        self,
        target_input_norm_xT: torch.Tensor,  # (B, P, F)
        target_current_mask: torch.Tensor,  # (B, P) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """pre-proj를 유효 에이전트 토큰만 대상으로 수행하고, packed(T,D)로 반환합니다.

        Args:
            target_input_norm_xT (torch.Tensor): (B, P, F)
            target_current_mask (torch.Tensor): (B, P) True=무효(패딩)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - x_unpad: (T, D)
                - indices: (T,)
                - cu_seqlens: (B+1,) int32
                - max_seqlen: int
        """
        if target_current_mask.dtype != torch.bool:
            target_current_mask = target_current_mask.to(torch.bool)

        valid_mask = (~target_current_mask)  # (B, P) True=valid
        xT_unpad, indices, cu_seqlens, max_seqlen = self._unpad_input_with_valid_mask(
            target_input_norm_xT, valid_mask)

        if xT_unpad.numel() == 0 or max_seqlen == 0:
            # 파라미터 터치(DDP unused 방지용)
            touch = (self.preproj.fc1.weight.view(-1)[:1].sum() +
                     (self.preproj.fc1.bias.view(-1)[:1].sum()
                      if self.preproj.fc1.bias is not None else 0) +
                     self.preproj.fc2.weight.view(-1)[:1].sum() +
                     (self.preproj.fc2.bias.view(-1)[:1].sum()
                      if self.preproj.fc2.bias is not None else 0)) * 0.0
            D_out = int(self.preproj.fc2.out_features)
            return xT_unpad.new_zeros(
                (0, D_out)) + touch, indices, cu_seqlens, max_seqlen

        x_unpad = self.preproj(xT_unpad)  # (T, D)
        return x_unpad, indices, cu_seqlens, max_seqlen

    def _build_cross_kv_cache(
            self,
            cross_c: torch.Tensor,  # (B, Lk, D)
            cross_mask: torch.Tensor,  # (B, Lk) True=pad
    ) -> FlashAttnKVCache:
        """scene 토큰(cross_c)을 한 번만 unpad해서 KV 캐시로 만듭니다."""
        if cross_mask.dtype != torch.bool:
            cross_mask = cross_mask.to(torch.bool)

        kv_valid = (~cross_mask)  # (B, Lk) True=valid
        kv_unpad, _, cu_k, max_k = self._unpad_input_with_valid_mask(
            cross_c, kv_valid)
        return FlashAttnKVCache(
            kv_unpad=kv_unpad,
            cu_seqlens_k=cu_k,
            max_seqlen_k=int(max_k),
        )

    def _select_tensors_for_feasible_projection(
        self,
        x: torch.
        Tensor,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        target_current_xyyaw: torch.Tensor,  # (B, (1+)Pnn, 4)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, 1+past_len+future_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FeasibleProjector에 넣기 전, 텐서의 '그래디언트 유지/차단' 방식을 정한다.

        - feasible_grad_to_dit=True:
            학습이 가능한 경로를 유지하기 위해 x/near_current를 그대로 사용하고 float32로만 바꿉니다.
        - feasible_grad_to_dit=False:
            FeasibleProjector 쪽으로는 학습 신호가 흐르지 않게 detach한 뒤 float32로 바꿉니다.

        Returns:
            x_for_feasible:
                shape: (B, (1+)Pnn, F_out), dtype=float32
            target_current_xyyaw_for_feasible:
                shape: (B, (1+)Pnn, 4), dtype=float32
            target_past_cur_future_valid_for_feasible:
                shape: (B, (1+)Pnn, time_len_total)
        """
        if self.config.feasible_grad_to_dit:
            x_for_feasible = x.float()
            target_current_xyyaw_for_feasible = target_current_xyyaw.float()
            target_past_cur_future_valid_for_feasible = target_past_cur_future_valid
        else:
            x_for_feasible = x.detach().float()
            target_current_xyyaw_for_feasible = target_current_xyyaw.detach(
            ).float()
            target_past_cur_future_valid_for_feasible = target_past_cur_future_valid.detach(
            )
        return (
            x_for_feasible,
            target_current_xyyaw_for_feasible,
            target_past_cur_future_valid_for_feasible,
        )

    def _build_diffusion_trajectory_for_feasible_projection(
            self,
            x_for_feasible: torch.Tensor,  # (B, (1+)Pnn, F_out)
            target_current_xyyaw_for_feasible: torch.
        Tensor,  # (B,(1+)Pnn,4) float32
    ) -> torch.Tensor:
        """FeasibleProjector 입력으로 쓸 (현재+미래) 궤적 텐서를 만든다.

        이 함수는 x(flat)를 (B, Pnn, 1+T, 4)로 바꾸고,
        첫 프레임(현재)을 원본 데이터로 로 맞춰줍니다.

        Returns:
            diffusion_trajectory:
                shape: (B, Pnn, 1+T, 4)
        """
        B, one_or_Pnn = x_for_feasible.shape[:2]

        # diffusion_trajectory: (B, Pnn, 1+T, 4)
        if self.config.use_past_dit_input:
            # x_for_feasible: (B, Pnn, (past_len + 1+T)*4) 같은 형태
            # diffusion_trajectory: (B, Pnn, 1+T, 4)
            diffusion_trajectory = (x_for_feasible.reshape(
                B, one_or_Pnn, -1, 4).contiguous()[:, :,
                                                   -(self._future_len + 1):, :]
                                   )  # (B, Pnn, 1+T, 4)
            diffusion_trajectory[:, :, 0, :] = target_current_xyyaw_for_feasible
            return diffusion_trajectory.contiguous()

        if self.config.use_current_input:
            diffusion_trajectory = x_for_feasible.reshape(
                B, one_or_Pnn, -1, 4).contiguous()  # (B,Pnn,1+T,4)
            diffusion_trajectory[:, :, 0, :] = target_current_xyyaw_for_feasible
            return diffusion_trajectory.contiguous()

        # use_current_input=False: x_for_feasible는 미래만 (B,Pnn,T*4)
        # diffusion_trajectory: (B, Pnn, 1+T, 4)
        diffusion_trajectory = torch.cat(
            [
                target_current_xyyaw_for_feasible.unsqueeze(
                    2),  # (B,(1+)Pnn,1,4)
                x_for_feasible.reshape(B, one_or_Pnn, -1, 4),  # (B,Pnn,T,4)
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
        assert diffusion_time.ndim == 1, \
            f"diffusion_time must be (B,), got {tuple(diffusion_time.shape)}"
        t_threshold: float = float(self.config.feasible_learn_noise_thresh)
        if not self.config.use_direct_loss:
            t_threshold = 1.0
        low_t_mask: torch.Tensor = (diffusion_time <= t_threshold)  # (B,)
        return low_t_mask

    def _maybe_replace_x_with_integrated_trajectory_for_eval(
            self,
            x: torch.Tensor,  # (B, Pnn, F_out)
            integrated_trajectory: torch.Tensor,  # (B, Pnn, T, 4)
            target_current_xyyaw_for_feasible: torch.Tensor,  # (B, Pnn, 4)
    ) -> torch.Tensor:
        """평가 모드에서 integrated_trajectory를 최종 출력으로 쓸지 결정한다.

        기존 규칙(그대로 유지):
            - training 모드이거나 direct loss 사용 시: x 그대로 반환
            - 평가 모드 + direct loss 미사용 시:
                * past_dit_input: x의 과거+현재 부분 + integrated_trajectory를 이어붙여 반환
                * current_input: 현재 + integrated_trajectory
                * current_input=False: integrated_trajectory만 반환


        Returns:
            x_out:
                최종 출력(flat).
                shape: (B, Pnn, F_out)
        """
        if self.training or self.config.use_direct_loss:
            return x
        batch_size, one_or_Pnn = x.shape[:2]
        B: int = batch_size

        if self.config.use_past_dit_input:
            # x의 과거+현재 부분만 유지한 뒤, 미래를 integrated_trajectory로 교체
            x_past_cur = (
                x.reshape(B, one_or_Pnn, -1,
                          4).contiguous()[:, :, :(self.config.time_len), :]
            )  # (B, Pnn, time_len, 4)

            x_new = torch.cat(
                [x_past_cur, integrated_trajectory],
                dim=2,
            )  # (B, Pnn, past_len+1+T, 4)

            return x_new.reshape(B, one_or_Pnn, -1)

        if self.config.use_current_input:
            x_new = torch.cat(
                [
                    target_current_xyyaw_for_feasible.unsqueeze(
                        2),  # (B,(1+)Pnn,1,4)
                    integrated_trajectory,  # (B,Pnn,T,4)
                ],
                dim=2,
            )  # (B,Pnn,1+T,4)
            return x_new.reshape(B, one_or_Pnn, -1)

        # use_current_input=False: 미래만 반환
        return integrated_trajectory.reshape(B, one_or_Pnn, -1)  # (B,Pnn,T*4)

    @property
    def model_type(self):
        return self._model_type

    def _run_dit_core_with_pram_v2(
        self,
        target_input_norm_xT: torch.Tensor,
        diffusion_time: torch.Tensor,
        cross_c: torch.Tensor,
        cross_mask: torch.Tensor,
        target_current_11_dim: torch.Tensor,
        target_current_valid: torch.Tensor,
    ) -> torch.Tensor:
        """DiT 본체(프리프로젝션 + PRAM-v2 블록 + 최종 투영)를 한 번 수행합니다.

        핵심 변경점:
          - 에이전트 토큰은 시작에 1번만 unpad하여 (T,D) packed로 블록 전체를 수행
          - scene 토큰 KV는 시작에 1번만 unpad하여 모든 블록에서 재사용
          - 블록별 PRAM 모듈레이션은 depth 전체를 한 번에 계산하고, 블록에서는 슬라이스만 사용
          - masked_fill은 최종 출력에서 1번만 수행
        """
        device_type: str = target_input_norm_xT.device.type

        target_current_mask = (~target_current_valid.to(torch.bool)
                              )  # (B, P) True=pad
        B, one_or_Pnn, _ = target_input_norm_xT.shape
        depth: int = int(len(self.blocks))

        # 1) pre-proj (packed)
        with profile_block(
                "DiT.preproj_varlen_packed",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            x_unpad, agent_indices, cu_q, max_q = self.preproj_varlen_packed(
                target_input_norm_xT=target_input_norm_xT,  # (B,P,F)
                target_current_mask=_to_bool_mask(target_current_mask),  # (B,P)
            )

        # dtype/device 정렬 기준은 x_unpad로
        # (packed이 비어있을 때도 dtype/device는 target_input_norm_xT와 같게 유지)
        ref_for_dtype = x_unpad if x_unpad.numel() > 0 else target_input_norm_xT

        # 2) cross 입력 정렬 + KV 캐시(1회)
        cross_c = _cast_like(cross_c, ref_for_dtype)
        cross_mask = _to_bool_mask(cross_mask).to(device=ref_for_dtype.device)

        with profile_block(
                "DiT.cross_kv_unpad_once",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            kv_cache = self._build_cross_kv_cache(cross_c=cross_c,
                                                  cross_mask=cross_mask)

        # 3) timestep embedding
        with profile_block(
                "DiT._get_time_embedding",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            t_embedding: torch.Tensor = self._get_time_embedding(
                diffusion_time, ref=ref_for_dtype)

        # 4) state_token_in (B,P,D)
        target_current_mask = _to_bool_mask(target_current_mask).to(
            device=ref_for_dtype.device)

        with profile_block(
                "DiT.pram_v2_state_token_encoder",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            state_token_in: torch.Tensor = self.pram_v2_state_token_encoder(
                target_cur_norm=target_current_11_dim.to(
                    dtype=ref_for_dtype.dtype, device=ref_for_dtype.device),
                target_current_mask=target_current_mask,
            )

        # 5) composer + time modulation
        with profile_block(
                "DiT.pram_v2_composer",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            composer_out = self.pram_v2_composer(
                state_token_in=state_token_in,
                target_current_mask=target_current_mask,
            )

        with profile_block(
                "DiT.pram_v2_time_mod",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            time_out = self.pram_v2_time_mod(t_embedding)

        # 6) PRAM 모듈레이션을 depth 전체에 대해 1번에 계산 (packed 기준)
        with profile_block(
                "DiT.pram_v2_modulations_packed_all_blocks",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            if agent_indices.dtype != torch.long:
                agent_indices = agent_indices.to(torch.long)

            P: int = int(one_or_Pnn)
            H: int = int(state_token_in.shape[-1])
            T: int = int(agent_indices.numel())

            # indices = b*P + p 이므로, 배치 인덱스는 indices//P
            batch_indices = (agent_indices // P).to(torch.long)  # (T,)

            # base (B,P,H) -> (T,H)
            ds_base = composer_out.delta_scale_base.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).reshape(B * P, H)
            sh_base = composer_out.shift_base.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).reshape(B * P, H)
            lg_base = composer_out.logit_gate_base.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).reshape(B * P, H)

            ds_base_p = ds_base.index_select(0, agent_indices)  # (T,H)
            sh_base_p = sh_base.index_select(0, agent_indices)  # (T,H)
            lg_base_p = lg_base.index_select(0, agent_indices)  # (T,H)

            # time (B,1,H) -> (B,H) -> (T,H) (배치별로 동일)
            ds_time_b = time_out.delta_scale_time.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).squeeze(1)  # (B,H)
            sh_time_b = time_out.shift_time.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).squeeze(1)  # (B,H)
            lg_time_b = time_out.logit_gate_time.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).squeeze(1)  # (B,H)

            ds_time_p = ds_time_b.index_select(0, batch_indices)  # (T,H)
            sh_time_p = sh_time_b.index_select(0, batch_indices)  # (T,H)
            lg_time_p = lg_time_b.index_select(0, batch_indices)  # (T,H)

            # path scalars: (depth,3)
            k_s = self.pram_v2_block_path_scalars.k_s.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).view(depth, 3, 1, 1)
            k_sh = self.pram_v2_block_path_scalars.k_sh.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).view(depth, 3, 1, 1)
            k_g = self.pram_v2_block_path_scalars.k_g.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).view(depth, 3, 1, 1)
            beta_g = self.pram_v2_block_path_scalars.beta_g.to(
                dtype=ref_for_dtype.dtype,
                device=ref_for_dtype.device).view(depth, 3, 1, 1)

            # (T,H) -> (1,1,T,H)
            ds_base_p = ds_base_p.view(1, 1, T, H)
            sh_base_p = sh_base_p.view(1, 1, T, H)
            lg_base_p = lg_base_p.view(1, 1, T, H)
            ds_time_p = ds_time_p.view(1, 1, T, H)
            sh_time_p = sh_time_p.view(1, 1, T, H)
            lg_time_p = lg_time_p.view(1, 1, T, H)

            # 최종 packed 모듈레이션: (depth,3,T,H)
            ds_packed_all = ds_time_p + k_s * ds_base_p
            sh_packed_all = sh_time_p + k_sh * sh_base_p
            gate_packed_all = torch.sigmoid(lg_time_p + k_g * lg_base_p +
                                            beta_g)

        # 7) 블록 스택: packed로만 수행
        with profile_block(
                "DiT.blocks_total_packed",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            for block_index, block in enumerate(self.blocks):
                with profile_block(
                        f"DiT.block_packed[{block_index}]",
                        enabled=self.config.profile_feasible,
                        device_type=device_type,
                ):
                    pram_mods = {
                        "SA":
                            ModulationTriplet(
                                delta_scale=ds_packed_all[block_index, 0],
                                shift=sh_packed_all[block_index, 0],
                                gate=gate_packed_all[block_index, 0],
                            ),
                        "FFN":
                            ModulationTriplet(
                                delta_scale=ds_packed_all[block_index, 1],
                                shift=sh_packed_all[block_index, 1],
                                gate=gate_packed_all[block_index, 1],
                            ),
                        "CA":
                            ModulationTriplet(
                                delta_scale=ds_packed_all[block_index, 2],
                                shift=sh_packed_all[block_index, 2],
                                gate=gate_packed_all[block_index, 2],
                            ),
                    }

                    x_unpad = block.forward_packed(
                        x_unpad=x_unpad,  # (T,D)
                        cu_seqlens_q=cu_q,  # (B+1,)
                        max_seqlen_q=max_q,  # int
                        cross_kv_cache=kv_cache,  # cached KV
                        pram_v2_modulations=pram_mods,  # packed mods
                    )

        # 8) blocks 출력(hidden)을 (B,P,H)로 1회 pad 복원(Feasible용 저장 포함)
        x_hidden: torch.Tensor = pad_input(x_unpad, agent_indices, B,
                                           one_or_Pnn)  # (B,P,H)

        if getattr(self.config, "feasible_grad_to_dit", False):
            self.final_hidden_tokens = x_hidden.float()
        else:
            self.final_hidden_tokens = x_hidden.detach().clone().float()

        # 9) PRAM-v2 최종 레이어(기존 함수 사용)
        with profile_block(
                "DiT.apply_pram_v2_final_layer",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            x_out = apply_pram_v2_final_layer(
                x=x_hidden,
                composer_out=composer_out,
                time_out=time_out,
                final_norm=self.pram_v2_final_norm,
                out_proj=self.pram_v2_out_proj,
                target_current_mask=target_current_mask,
                final_scalars=(
                    self.pram_v2_final_scale_scalar,
                    self.pram_v2_final_shift_scalar,
                ),
            )

        # ✅ masked_fill은 최종 1회만
        x_out = x_out.masked_fill(target_current_mask.unsqueeze(-1), 0.0)
        return x_out

    def _forward_score_branch(
            self,
            x: torch.
        Tensor,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
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
        x: torch.
        Tensor,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        low_t_mask: Optional[torch.Tensor],  # (B,)
        diffusion_time: torch.Tensor,  # (B,)
        target_agents_past: torch.Tensor,  # (B, (1+)Pnn, past_len, 11)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, 1+past_len+future_len)
    ) -> torch.Tensor:  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        """model_type 이 'x_start' 인 경우 출력 텐서를 만드는 함수."""
        target_past_11_dim = target_agents_past[:, :, :
                                                -1, :]  # (B, (1+)Pnn, past_len, 11)
        target_current_xyyaw = target_agents_past[:, :,
                                                  -1, :4]  # (B, (1+)Pnn, 4)
        target_class_one_hot = target_agents_past[:, :, -1,
                                                  8:11]  # (B, (1+)Pnn, 3)

        # 기본 경로: feasible 을 쓰지 않을 때
        if not self.config.use_feasible:
            return x

        # 1) feasible에 넣을 텐서 준비(그래디언트 유지/차단 포함)
        (
            x_for_feasible,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            target_current_xyyaw_for_feasible,  # (B,(1+)Pnn,4) float32
            target_past_cur_future_valid_for_feasible,  # (B,(1+)Pnn,time_len_total)
        ) = self._select_tensors_for_feasible_projection(
            x=x,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            target_current_xyyaw=target_current_xyyaw,  # (B, (1+)Pnn, 4)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, (1+)Pnn, 1+past_len+future_len)
        )

        # 2) diffusion_trajectory : (현재+미래) 궤적 텐서 만들기: (B,(1+)Pnn,1+T,4)
        diffusion_trajectory: torch.Tensor = self._build_diffusion_trajectory_for_feasible_projection(
            x_for_feasible=
            x_for_feasible,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            target_current_xyyaw_for_feasible=
            target_current_xyyaw_for_feasible,  # (B,(1+)Pnn,4) float32
        )

        # 3) low-t 마스크 계산
        if low_t_mask is None:
            low_t_mask: torch.Tensor = self._compute_feasible_low_t_mask(
                diffusion_time=diffusion_time,  # (B,)
            )  # (B,)

        # 4) FeasibleProjector 실행(dit_returns 갱신)
        self._feasible_projection(
            diffusion_trajectory=diffusion_trajectory,  # (B,(1+)Pnn,1+T,4)
            target_past_11_dim=target_past_11_dim,  # (B,(1+)Pnn,past_len,11)
            target_class_one_hot=target_class_one_hot,  # (B,(1+)Pnn,3)
            target_past_cur_future_valid=
            target_past_cur_future_valid_for_feasible,  # (B,(1+)Pnn,time_len_total)
            low_t_mask=low_t_mask,  # (B,)
        )
        """
        # integrated_trajectory : (B, (1+)Pnn, future_len, 4)
        # control_constraint_diff : (B, (1+)Pnn, future_len, 3)
        """
        # 5) 평가 모드 + use_direct_loss = False에서
        # integrated_trajectory로 출력 교체(옵션, 기존 로직 유지)
        # x_out: (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        x_out: torch.Tensor = self._maybe_replace_x_with_integrated_trajectory_for_eval(
            x=x,
            integrated_trajectory=self.norm_dit_returns.
            integrated_trajectory,  # (B, (1+)Pnn, future_len, 4)
            target_current_xyyaw_for_feasible=target_current_xyyaw_for_feasible,
        )
        return x_out

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
        target_input_norm_xT: torch.Tensor,  # (B, (1+)Pnn, F=_*4)
        diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, (1+)Pnn, past_len+1+future_len)
    ) -> torch.Tensor:  # (B, (1+)Pnn, F=_*6)
        """
        target_cur_future_norm_xT : (B, (1+)Pnn, _ * 4) -> (B, (1+)Pnn, _ * 6)

        5: x, y, cos, sin 에서 diffusion noise time step  + validity 추가

        diffusion noise time step
            참고로 1+future_len 은 현재+미래. 현재에는 노이즈를 추가하지 않을 것이므로, 0으로 채움
            미래 future_len 에 대해서는, diffusion_time 에 따라 노이즈 time step을 채움
        """
        diffusion_time = diffusion_time.to(device=target_input_norm_xT.device)
        # target_input_norm_xT: (B, (1+)Pnn, _ * 4) -> (B, (1+)Pnn, _, 4)
        target_input_norm_xT = target_input_norm_xT.reshape(
            target_input_norm_xT.shape[0],
            target_input_norm_xT.shape[1],
            -1,
            4,
        )  # (B, (1+)Pnn, T_any, 4)

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
        validity = target_past_cur_future_valid[:, :,
                                                -T_any:]  # (B,P,T_any) bool
        validity = validity.to(device=target_input_norm_xT.device)  # 안전
        validity_f = validity.to(dtype=target_input_norm_xT.dtype).unsqueeze(
            -1)  # (B,P,T_any,1) float
        diffusion_time_full = diffusion_time_full * validity_f  # (B,P,T_any,1)
        diffusion_time_full = _cast_like(diffusion_time_full,
                                         target_input_norm_xT)
        """
        target_input_norm_xT : (B, (1+)Pnn, past_cur_time_len + future_len, 4) 
            -> (B, (1+)Pnn, past_cur_time_len + future_len, 6)
        diffusion_time_full : (B, (1+)Pnn, past_cur_time_len + future_len, 1)
        validity : (B, (1+)Pnn, past_cur_time_len + future_len, 1)
        """
        target_input_norm_xT = torch.cat(
            [
                target_input_norm_xT,  # (B, (1+)Pnn, past_cur_time_len + future_len, 4)
                diffusion_time_full,  # (B, (1+)Pnn, past_cur_time_len + future_len, 1)
                validity_f,  # (B, (1+)Pnn, past_cur_time_len + future_len, 1)
            ],
            dim=-1,
        )  # (B, (1+)Pnn, past_cur_time_len + future_len, 6)
        target_input_norm_xT = target_input_norm_xT.reshape(
            B,
            P,
            -1,
        )  # (B, (1+)Pnn, _ * 6)
        return target_input_norm_xT

    def forward(
        self,
        target_input_norm_xT: torch.
        Tensor,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        #  F 에서 시간 길이는 time_len + future_len 또는 1 + future_len 또는 future_len
        diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        target_agents_past: torch.
        Tensor,  # (B, (1+)Pnn, time_len(=past_len+1), 11)
        target_past_cur_future_valid: torch.Tensor,
        # (B, (1+)Pnn, 1+past_len+future_len) bool
        cross_c: torch.Tensor,  # (B, token_num, D)
        cross_mask: torch.Tensor,  # (B, token_num)
        low_t_mask: Optional[torch.Tensor] = None,  # (B,) bool
    ) -> torch.Tensor:  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
        """DiT 전체 forward 를 수행하는 진입점.

        순서 개요:
            1) 과거+현재+미래 유효 마스크에서 현재 에이전트 마스크를 만든다.
            2) pre-proj + PRAM-v2 블록을 통과해 기본 출력 x 를 만든다.
            3) model_type 에 따라 "score" 또는 "x_start" 분기 처리한다.
               - "x_start" 인 경우, use_feasible 설정에 따라
                 FeasibleProjector 기반 보정을 추가로 수행한다.
        Returns:
            torch.Tensor:
                model_type 에 따른 최종 출력.
                - "score": score(x_t) (B, Pnn, F)
                - "x_start": x_start 또는 feasible 보정 후 궤적 (B, Pnn, F)
        """
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
            # if self.config.use_amortized_diffusion:
            #   target_input_norm_xT: (B, (1+)Pnn, (time_len+ T) *6)
            #   or (B, (1+)Pnn, T*6)
            #   or (B, (1+)Pnn, (1+T)*6)
            with profile_block(
                    "DiT._apply_diffusion_timestep_and_validity",
                    enabled=self.config.profile_feasible,
                    device_type=device_type,
            ):
                target_input_norm_xT = self._apply_diffusion_timestep_and_validity(
                    target_input_norm_xT, diffusion_time,
                    target_past_cur_future_valid)
            # x:  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            x: torch.Tensor = self._run_dit_core_with_pram_v2(
                target_input_norm_xT=target_input_norm_xT,  #
                diffusion_time=diffusion_time,  # (B,) or (B, future_len)
                cross_c=cross_c,
                cross_mask=cross_mask,
                target_current_11_dim=target_current_11_dim,  # (B, (1+)Pnn, 11)
                target_current_valid=target_current_valid,  # (B, (1+)Pnn)
            )

        # model_type 분기
        if self._model_type == "score":
            raise NotImplementedError("Score model type is not implemented.")
            # return self._forward_score_branch(
            #     x=x,  # x:  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
            #     diffusion_time=diffusion_time,
            # )
        elif self._model_type == "x_start":
            # x 를 대부분의 경우에 그대로 반환
            return self._forward_x_start_branch(
                x=x,  # (B, (1+)Pnn, (time_len+ T) *4) or (B, (1+)Pnn, T*4) or (B, (1+)Pnn, (1+T)*4)
                low_t_mask=low_t_mask,  # (B,)
                diffusion_time=diffusion_time,
                target_agents_past=
                target_agents_past,  # (B, (1+)Pnn, time_len, 11)
                target_past_cur_future_valid=target_past_cur_future_valid,
            )
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    # : stride 기반 down/up 샘플링을 통합한 새 파이프라인
    def _feasible_projection_core(
            self,
            diffusion_trajectory: torch.Tensor,  # (B, (1+)Pnn, 1+future_len, 4)
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
        device_type: str = diffusion_trajectory.device.type
        use_profile: bool = bool(getattr(self.config, "profile_feasible",
                                         False))

        # Feasible 파트는 모두 FP32로 고정
        with torch.autocast(device_type=device_type, enabled=False):
            diffusion_trajectory = diffusion_trajectory.float(
            )  # (B, Pnn, 1+T, 4)
            target_class_one_hot = target_class_one_hot.float()  # (B, Pnn, 3)
            target_past_cur_future_valid = target_past_cur_future_valid.to(
                torch.bool)  # (B, Pnn, time_len+future_len)
            if target_past_11_dim is not None:
                target_past_11_dim = target_past_11_dim.float(
                )  # (B, Pnn, past_len, 11)

            B, Pnn, one_future_len, _ = diffusion_trajectory.shape
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
                data=diffusion_trajectory,  # (B, Pnn, 1+T, 4)
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
                unnorm_near_past_xyyaw_stride,  # (B, Pnn, past_len_ds, 4) or None
                near_past_cur_future_valid_stride,  # (B, Pnn, past_len_ds+1+T_ds) bool
                diffusion_trajectory_stride_norm,  # (B, Pnn, 1+T_ds, 4)
                near_past_xyyaw_stride_norm,  # (B, Pnn, past_len_ds, 4) or None
                past_len_ds,  # int
                future_len_ds,  # int (T_ds)
            ) = self.feasible_projector.build_downsampled_feasible_inputs(
                diffusion_trajectory=diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                target_past=target_past_11_dim,  # (B, Pnn, past_len, 11) or None
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
                    near_past_cur_future_valid_stride,
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
                unnorm_seg_body_control_stride = self.feasible_projector.compute_midpoint_controls(
                    unnorm_diffusion_trajectory_stride,  # (B, Pnn, 1+T_ds, 4)
                    unnorm_near_past_xyyaw_stride,
                    # (B, Pnn, past_len_ds, 4) or None
                    unnorm_points_world_control_stride,
                    # (B, Pnn, point_len_ds, 3)
                    near_past_cur_future_valid_stride,
                    # (B, Pnn, past_len_ds+1+T_ds) bool
                    point_len_inputs=point_len_inputs_stride,
                )

            # --- (3) 관측 정규화 → FeasibleProjector 네트워크(TCN) 보정 ---
            temp_dict = {"seg_body_control": unnorm_seg_body_control_stride}
            norm_temp_dict = self.config.observation_normalizer(temp_dict)
            seg_body_control_stride = norm_temp_dict[
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
                    touch = self._ddp_touch_feasible_projector_params(
                        device=seg_body_control_stride_ref.device,
                        dtype=seg_body_control_stride_ref.dtype,
                    )
                    seg_body_control_stride_ref = seg_body_control_stride_ref + touch

            seg_body_control_stride_ref = seg_body_control_stride_ref.float()
            norm_temp_dict = {"seg_body_control": seg_body_control_stride_ref}
            temp_dict = self.config.observation_normalizer.inverse(
                norm_temp_dict)
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
            target_cur_future_valid = target_past_cur_future_valid[:, :, -(
                1 + future_len):]  # (B, Pnn, 1+future_len) bool

            with profile_block(
                    "feasible.filter_and_integrate",
                    enabled=use_profile,
                    device_type=device_type,
            ):
                """무효 구간(점이 무효한 구간)은 0.0으로 출력됩니다."""
                (
                    unnorm_integrated_trajectory,  # (B, Pnn, future_len, 4)
                    unnorm_control_constraint_diff,  # (B, Pnn, future_len, 3)
                ) = self.feasible_projector.filter_and_integrate(
                    unnorm_near_current_state,  # (B, Pnn, 4)
                    target_cur_future_valid,  # (B, Pnn, 1+future_len) bool
                    unnorm_fut_seg_body_control,  # (B, Pnn, future_len, 3)
                    target_class_one_hot,  # (B, Pnn, 3)
                )
            # 정규화해서 DiTReturns 로 저장
            target_future_valid = target_cur_future_valid[:, :,
                                                          1:]  # (B, Pnn, future_len) bool
            integrated_trajectory = self.config.state_normalizer(
                data=unnorm_integrated_trajectory,
                valid_mask=target_future_valid)  # (B, Pnn, future_len, 4)

            temp_dict = {"seg_body_control": unnorm_control_constraint_diff}
            norm_temp_dict = self.config.observation_normalizer(temp_dict)
            control_constraint_diff = norm_temp_dict[
                "seg_body_control"]  # (B, Pnn, future_len, 3)
            control_constraint_diff = control_constraint_diff.masked_fill(
                ~target_future_valid.unsqueeze(-1), 0.0)

            self.norm_dit_returns = DiTReturns(
                integrated_trajectory=
                integrated_trajectory,  # (B, (1+)Pnn, future_len, 4)
                control_constraint_diff=
                control_constraint_diff,  # (B, (1+)Pnn, future_len, 3)
            )

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
            diffusion_trajectory: torch.Tensor,  # (B, (1+)Pnn, 1+future_len, 4)
            target_past_11_dim: torch.Tensor,
            # (B, (1+)Pnn, past_len, 11) 또는 None
            target_class_one_hot: torch.Tensor,  # (B, (1+)Pnn, 3)
            target_past_cur_future_valid: torch.Tensor,
            # (B, (1+)Pnn, time_len(=1+past_len) + future_len) bool
            low_t_mask: torch.Tensor,  # (B, )
    ) -> None:
        B, Pnn, one_future_len, _ = diffusion_trajectory.shape
        future_len = one_future_len - 1
        device = diffusion_trajectory.device

        # [B] -> bool 로 정리
        low_t_mask = low_t_mask.to(device=device)
        low_t_mask_bool = low_t_mask if low_t_mask.dtype == torch.bool else (
            low_t_mask > 0.5)

        # 기본값:
        #   - integrated_trajectory: 원 궤적(x_start) 그대로 (t=1..future_len)
        #   - control_constraint_diff: 전부 0
        # diffusion_trajectory 는 정규화 상태라고 가정
        base_integrated = diffusion_trajectory[:, :, 1:, :].detach(
        )  # (B,Pnn,future_len,4)
        base_constraint = diffusion_trajectory.new_zeros(
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

            self.norm_dit_returns = DiTReturns(
                integrated_trajectory=integrated_all,  # (B,Pnn,future_len,4)
                control_constraint_diff=constraint_all,  # (B,Pnn,future_len,3)
            )
            return

        # ---- active subset만 projector 파이프라인 실행 ----
        self._feasible_projection_core(
            diffusion_trajectory[active_idx],
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

        self.norm_dit_returns = DiTReturns(
            integrated_trajectory=integrated_all,  # (B,Pnn,future_len,4)
            control_constraint_diff=constraint_all,  # (B,Pnn,future_len,3)
        )
