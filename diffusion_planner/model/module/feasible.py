import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union, TypedDict, Optional
from scipy.signal import savgol_filter  # type: ignore
import math
import numpy as np
import torch.nn.functional as F
# feasible.py 내부, FeasibleProjector 클래스에 추가/대체 ----------------------
import torch
from typing import Tuple

Number = Union[float, int]
ArrayLike = Union[np.ndarray, List[Number], Tuple[Number, ...]]
import torch
from torch import Tensor

torch.set_printoptions(sci_mode=False, precision=6)

from typing import Tuple
# feasible.py 최상단 import 근처
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any

# <추가하자>
from typing import NamedTuple

# decoder.py 또는 feasible.py 상단 import 근처에 추가
import time
from contextlib import contextmanager
from typing import Iterator


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


# <추가하자>
class PointLenInputs(NamedTuple):
    unnorm_points_xyyaw: torch.Tensor  # (B,Pnn,point_len,4)
    points_valid: torch.Tensor  # (B,Pnn,point_len)  bool
    past_cur_valid: Optional[torch.Tensor]  # (B,Pnn,past_len+1) bool or None
    cur_future_valid: torch.Tensor  # (B,Pnn,1+future_len) bool
    past_len: int
    point_len: int


# =========================
# [NEW] 하이퍼/제약 파라미터 컨테이너
# =========================
@dataclass
class _ConstraintHParams:
    """제약/적분 하이퍼파라미터(상수 모음)."""
    dt: float
    eps: float
    # 필요: STE용 밴드폭 η (권장 초기값)
    eta_slip: float = 0.07  # S0
    eta_speed: float = 0.05  # S1
    eta_inc: float = 0.10  # S2
    eta_yaw: float = 0.05  # S3
    eta_fric: float = 0.05  # S4


def _shape4(
    B: int, Pnn: int, T: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B,Pnn,T,4) 분해 저장용 4개 버퍼를 (x,y,cos,sin) 순서로 생성.

    returns:
            각 (B,Pnn,T) shape 텐서
    """
    zeros = lambda: torch.zeros((B, Pnn, T), device=device, dtype=dtype)
    return zeros(), zeros(), zeros(), zeros()  # x, y, cos, sin


class ActorClass(Enum):
    """행위자(객체) 종류."""
    PEDESTRIAN = "Pedestrian"
    BICYCLE = "Bicycle"
    CAR = "Car"


@dataclass(frozen=True)
class DynamicLimits:
    """행위자 클래스별 동역학 한계치.

    모든 내부 단위는 SI(미터, 초, 라디안) 기준.

    Attributes:
        v_max_mps: 최대 속도 [m/s]
        v_max_kmph: 최대 속도 [km/h] (편의상 함께 보관)
        a_max_mps2: 최대 종가속도(가감속 절대치) [m/s^2]
        alpha_max_radps2: 최대 요각가속도 [rad/s^2]
        a_lat_max_mps2: 최대 횡가속도 [m/s^2]
        R_min_m: 최소 선회 반경 [m]
        omega_max_abs_radps: 최대 요각속도 절대치 [rad/s]
    """
    v_max_mps: float
    v_max_kmph: float
    a_max_mps2: float
    alpha_max_radps2: float
    a_lat_max_mps2: float
    R_min_m: float
    omega_max_abs_radps: float

    def as_dict(self) -> Dict[str, float]:
        """딕셔너리 형태로 반환."""
        return asdict(self)


class FeasibleProjector(nn.Module):

    def __init__(self,
                 config,
                 hidden_dim: int,
                 use_feasible_dl: bool = True,
                 use_feasible_filter: bool = True):
        """Control Correction Network 본체 모듈 정의.

        Notes:
            - 트렁크 은닉 차원 H는 입력으로만 알 수 있으므로,
              trunk 압축기(Compressor)는 첫 forward에서 지연 초기화합니다.
        """
        super().__init__()
        self.config = config
        self.use_feasible_dl = use_feasible_dl
        self.use_feasible_filter = use_feasible_filter
        # --- [NEW] Savitzky–Golay 커널 캐시(LRU) ---
        # key: (W, polyorder, deriv_order, dt, dtype, device)
        # val: torch.Tensor of shape (1, 1, W)
        # [추가 요망] SG 위치별(one‑sided/중앙) 가중치 캐시(LRU)
        # key: (W, polyorder, deriv_order, m, dt, dtype, device)  → val: (W,) weights
        self.enable_profile: bool = False
        self.use_batch_integration = self.config.use_batch_integration  # 필요
        self.constraints_h_params = _ConstraintHParams(
            dt=0.1,
            eps=1e-6,
            # 필요: η 초기값(S0~S4)
            eta_slip=0.07,
            eta_speed=0.05,
            eta_inc=0.10,
            eta_yaw=0.05,
            eta_fric=0.05,
        )

        # ---- 동역학 한계(기존) ----
        self.constraints: Dict[ActorClass, DynamicLimits] = {
            ActorClass.PEDESTRIAN:
                DynamicLimits(
                    v_max_mps=8.0,
                    v_max_kmph=28.8,
                    a_max_mps2=6.0,
                    alpha_max_radps2=8.0,
                    a_lat_max_mps2=6.0,
                    R_min_m=0.00001,
                    omega_max_abs_radps=3.6,
                ),
            ActorClass.BICYCLE:
                DynamicLimits(
                    v_max_mps=20.0,
                    v_max_kmph=72.0,
                    a_max_mps2=6.0,
                    alpha_max_radps2=3.0,
                    a_lat_max_mps2=8.5,
                    R_min_m=1.20,
                    omega_max_abs_radps=1.5,
                ),
            ActorClass.CAR:
                DynamicLimits(
                    v_max_mps=55.6,
                    v_max_kmph=200.0,
                    a_max_mps2=8.0,
                    alpha_max_radps2=2.5,
                    a_lat_max_mps2=9.0,
                    R_min_m=4.50,
                    omega_max_abs_radps=0.9,
                ),
        }

        # ------------------------------
        # 아키텍처 하이퍼파라미터(고정 폭)
        # ------------------------------
        self._Dx: int = 12  # state encoder 출력 채널 (prev/fut 각각)
        self._Du: int = 8  # control adapter 출력 채널
        self._Dc: int = 16  # trunk compressor 출력 채널
        self._Din: int = self._Dx * 2 + self._Du + self._Dc  # 16+16+32+64=192
        self._C: int = self._Din  # 메인 채널 폭(192)
        self._eps: float = 1e-6

        # [추가 필요] L_integration 경로 차단용 플래그 (state, u_base detach)
        self.detach_u_for_ctrl_losses: bool = True  # 필요

        # ------------------------------
        # Encoders (시간축 보존, 채널만 변환)
        # ------------------------------
        # if self.use_feasible_dl:
        # 상태 인코더(현재 노드 X_prev: (x,y,cos,sin))
        self.state_prev_encoder = nn.Sequential(
            nn.LayerNorm(4),  # (B,Pnn,segment_len,4)
            nn.Linear(4, self._Dx),  #: #4 → 16
            nn.GELU(),
            nn.Linear(self._Dx, self._Dx),  #: 16 → 16 (= self._Dx)
        )
        self.state_fut_encoder = nn.Sequential(
            nn.LayerNorm(4),  # (B,Pnn,segment_len,4)
            nn.Linear(4, self._Dx),  #: 4 → 16
            nn.GELU(),
            nn.Linear(self._Dx, self._Dx),  # : 16 → 16 (= self._Dx)
        )
        self.control_adapter = nn.Sequential(
            nn.LayerNorm(3),  # (B,Pnn,segment_len,3)
            nn.Linear(3, self._Du),  # : 3 → 8 (= self._Du)
            nn.GELU(),
        )

        # (B,Pnn,H) -> (B,Pnn,_Dc=8) 로 trunk 압축
        self.trunk_compressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3 * self._Dc),  #: hidden_dim → 64 (중간 폭 축소)
            nn.GELU(),
            nn.Linear(3 * self._Dc, self._Dc),  #: 64 → 8 (= self._Dc)
        )

        # ------------------------------
        # Stem (채널 정렬)
        # ------------------------------
        self.stem_norm = nn.LayerNorm(self._Din)
        self.stem_fc = nn.Linear(self._Din, self._C)
        self.stem_act = nn.GELU()

        # ------------------------------
        # TCN 4블록: depthwise(7) + dilation {1,2,4,8} + 1x1
        # ------------------------------
        self._kernel_size: int = 7
        self._dilations: List[int] = [1, 2]  #: 4블록→2블록
        self.tcn_depth = len(self._dilations)  # : 현재는 2
        self.tcn_pre_lns = nn.ModuleList(  #: 블록 수만큼 LayerNorm
            [nn.LayerNorm(self._C) for _ in range(self.tcn_depth)])

        # depthwise conv (C 채널, groups=C)
        def _same_pad(k: int, d: int) -> int:
            return d * (k // 2)

        self.tcn_dw = nn.ModuleList([
            nn.Conv1d(in_channels=self._C,
                      out_channels=self._C,
                      kernel_size=self._kernel_size,
                      padding=_same_pad(self._kernel_size, d),
                      dilation=d,
                      groups=self._C,
                      bias=False) for d in self._dilations
        ])
        # pointwise 1x1 (시간축 보존, 채널 결합)
        self.tcn_linear = nn.ModuleList(
            [nn.Linear(self._C, self._C) for _ in range(self.tcn_depth)])

        # ------------------------------
        # Head & Gate
        # ------------------------------
        # 잔차 초안 ΔU_raw
        self.head = nn.Sequential(
            nn.Linear(self._C, self._C),
            nn.GELU(),
            nn.Linear(self._C, 3),
        )
        # 마지막 Linear 0-init → 초기엔 U_ref ≈ U_base
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        gate_hidden_dim = 32
        # 소프트 게이트 s = softplus(MLP_g(Z_s))
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self._C),
            nn.Linear(self._C, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        # gate 초기 스케일 s0 설정(보수적으로)
        s0 = 0.05
        b_init = math.log(math.exp(float(s0)) - 1.0)  # softplus^{-1}(s0)
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(b_init)

    def _infer_past_future_lengths_for_downsample(
            self,
            diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, time_len)
    ) -> Tuple[int, int, int]:
        """다운샘플링에 사용할 과거 길이와 미래 길이를 계산합니다.

        Args:
            diffusion_trajectory: 현재+미래 궤적. shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid: 과거~현재~미래 유효 마스크. shape: (B, Pnn, time_len)

        Returns:
            past_len: 과거 노드 개수.
            future_len: 미래 노드 개수(T).
            time_len: 전체 타임라인 길이.
        """
        B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
        mask_B, mask_Pnn, time_len = target_past_cur_future_valid.shape

        if (mask_B, mask_Pnn) != (B, Pnn):
            raise ValueError(
                "[FeasibleProjector] diffusion_trajectory 와 target_past_cur_future_valid 의 "
                f"(B,Pnn)이 다릅니다: (B,Pnn)=({B},{Pnn}), (mask_B,mask_Pnn)=({mask_B},{mask_Pnn})"
            )

        future_len: int = int(one_plus_T - 1)
        past_len: int = int(time_len - (1 + future_len))
        if past_len < 0:
            raise ValueError(
                f"[FeasibleProjector] time_len={time_len} 이(가) 1+future_len={1 + future_len} 보다 작습니다."
            )

        if time_len != past_len + 1 + future_len:
            raise ValueError(
                "[FeasibleProjector] time_len 과 (past_len + 1 + future_len) 가 일치하지 않습니다."
            )

        return past_len, future_len, time_len

    def _decide_use_past_for_downsample(
            self,
            past_len: int,
            target_past: Optional[
                torch.Tensor],  # shape: (B, Pnn, past_len, 11) or None
            unnorm_near_past_xyyaw: Optional[
                torch.Tensor],  # shape: (B, Pnn, past_len, 4) or None
    ) -> bool:
        """과거 구간을 다운샘플에 포함할지 여부를 결정합니다.

        - config.use_past_for_feasible 가 False 이면 과거를 사용하지 않습니다.
        - past_len 이 0 이면 자동으로 과거를 사용하지 않습니다.
        - 과거를 쓰겠다고 했는데 텐서가 None 이면 에러를 냅니다.
        """
        use_past: bool = bool(
            getattr(self.config, "use_past_for_feasible", True))

        if use_past and past_len <= 0:
            # 과거를 쓰기로 했지만 실제 과거가 없으면 자동으로 끕니다.
            use_past = False

        if use_past:
            if target_past is None or unnorm_near_past_xyyaw is None:
                raise ValueError(
                    "[FeasibleProjector] use_past_for_feasible=True 인데 "
                    "target_past / unnorm_near_past_xyyaw 가 None 입니다.")

        return use_past

    def _build_stride_indices_with_past(
        self,
        past_len: int,
        future_len: int,
        stride_step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, int]:
        """과거~현재~미래 전체 타임라인에서 사용할 다운샘플 index 를 생성합니다.

        현재 시점을 기준으로 앞뒤로 stride_step 간격으로 고르고,
        항상 현재 시점과 마지막 미래 시점을 포함합니다.

        Returns:
            sample_idx_full: 전체 타임라인 기준 index. shape: (K,)
            past_len_ds: 다운샘플 후 과거 노드 개수.
            future_len_ds: 다운샘플 후 미래 노드 개수.
        """
        idx_cur_full: int = past_len
        idx_last_full: int = past_len + future_len

        if stride_step == 1:
            sample_idx_full = torch.arange(
                0,
                idx_last_full + 1,
                device=device,
                dtype=torch.long,
            )  # shape: (time_len,)
        else:
            # 현재 index 기준으로 과거 / 미래를 대칭으로 선택
            back_rev = torch.arange(
                idx_cur_full,
                -1,
                -stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (K_back,)
            back = torch.flip(back_rev, dims=(0,))  # 과거 방향 오름차순

            forward = torch.arange(
                idx_cur_full + stride_step,
                idx_last_full + 1,
                stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (K_fwd,)

            sample_idx_full = torch.cat([back, forward], dim=0)  # shape: (K,)

        # 현재 index 가 정확히 한 번만 포함되는지 검사
        current_pos_tensor = (sample_idx_full == idx_cur_full).nonzero(
            as_tuple=False).view(-1)
        if current_pos_tensor.numel() != 1:
            raise ValueError("[FeasibleProjector] 다운샘플 index 계산 중 현재 시점이 "
                             "정확히 한 번 포함되지 않았습니다.")

        current_pos: int = int(current_pos_tensor.item())
        past_len_ds: int = current_pos
        future_len_ds: int = int(sample_idx_full.numel() - 1 - past_len_ds)

        return sample_idx_full, past_len_ds, future_len_ds

    def _downsample_with_past(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        target_past: torch.Tensor,  # shape: (B, Pnn, past_len, 11)
        target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, past_len+1+T)
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        unnorm_near_past_xyyaw: torch.Tensor,  # shape: (B, Pnn, past_len, 4)
        sample_idx_full: torch.Tensor,  # shape: (K,)
        past_len_ds: int,
        future_len_ds: int,
    ) -> Tuple[
            torch.
            Tensor,  # unnorm_diffusion_trajectory_stride # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # unnorm_near_past_xyyaw_stride # shape: (B, Pnn, past_len_ds, 4) or None
            torch.
            Tensor,  # near_past_cur_future_valid_stride # shape: (B, Pnn, past_len_ds+1+T_ds)
            torch.
            Tensor,  # diffusion_trajectory_stride_norm # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # near_past_xyyaw_stride_norm # shape: (B, Pnn, past_len_ds, 4) or None
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """과거~현재~미래 전체 타임라인 기준으로 stride 다운샘플을 적용합니다."""
        # 역정규화 포인트 결합: [과거, 현재+미래]
        unnorm_points_all = torch.cat(
            [unnorm_near_past_xyyaw, unnorm_diffusion_trajectory],
            dim=2,
        )  # shape: (B, Pnn, past_len+1+T, 4)

        points_valid_all = target_past_cur_future_valid.to(
            torch.bool)  # shape: (B, Pnn, past_len+1+T)

        # stride 적용
        unnorm_points_stride = unnorm_points_all[:, :,
                                                 sample_idx_full, :]  # shape: (B, Pnn, past_len_ds+1+T_ds, 4)
        valid_stride = points_valid_all[:, :,
                                        sample_idx_full]  # shape: (B, Pnn, past_len_ds+1+T_ds)

        if past_len_ds > 0:
            unnorm_near_past_xyyaw_stride: Optional[
                torch.Tensor] = unnorm_points_stride[:, :, :past_len_ds, :]
        else:
            unnorm_near_past_xyyaw_stride = None

        unnorm_diffusion_trajectory_stride = unnorm_points_stride[:, :,
                                                                  past_len_ds:, :]  # shape: (B, Pnn, 1+T_ds, 4)

        # 정규화 포인트도 동일 index 로 다운샘플
        near_past_xyyaw = target_past[..., :4]  # shape: (B, Pnn, past_len, 4)
        points_norm_all = torch.cat(
            [near_past_xyyaw, diffusion_trajectory],
            dim=2,
        )  # shape: (B, Pnn, past_len+1+T, 4)

        points_norm_stride = points_norm_all[:, :,
                                             sample_idx_full, :]  # shape: (B, Pnn, past_len_ds+1+T_ds, 4)

        if past_len_ds > 0:
            near_past_xyyaw_stride_norm: Optional[
                torch.Tensor] = points_norm_stride[:, :, :past_len_ds, :]
        else:
            near_past_xyyaw_stride_norm = None

        diffusion_trajectory_stride_norm = points_norm_stride[:, :,
                                                              past_len_ds:, :]  # shape: (B, Pnn, 1+T_ds, 4)

        near_past_cur_future_valid_stride = valid_stride  # shape: (B, Pnn, past_len_ds+1+T_ds)

        return (
            unnorm_diffusion_trajectory_stride,
            unnorm_near_past_xyyaw_stride,
            near_past_cur_future_valid_stride,
            diffusion_trajectory_stride_norm,
            near_past_xyyaw_stride_norm,
            past_len_ds,
            future_len_ds,
        )

    def _build_stride_indices_without_past(
        self,
        future_len: int,
        stride_step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """현재~미래 구간(1+T)만 사용하는 경우의 다운샘플 index 를 생성합니다.

        Returns:
            sample_idx_local: 현재~미래 구간에서 사용할 index. shape: (1+T_ds,)
            future_len_ds: 다운샘플 후 미래 노드 개수(T_ds).
        """
        total_points: int = 1 + future_len  # 현재 포함

        if stride_step == 1:
            sample_idx_local = torch.arange(
                0,
                total_points,
                device=device,
                dtype=torch.long,
            )  # shape: (1+T,)
        else:
            sample_idx_local = torch.arange(
                0,
                total_points,
                stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (1+T_ds,)

            # 마지막 index(T)는 반드시 포함
            if int(sample_idx_local[-1].item()) != future_len:
                raise ValueError(
                    "[FeasibleProjector] (use_past_for_feasible=False) 환경에서 "
                    f"stride_step={stride_step} 가 future_len={future_len} 을 정확히 나누지 못했습니다. "
                    "config.feasible_stride_dt 를 조정해 주세요.")

        future_len_ds: int = int(sample_idx_local.numel() - 1)
        return sample_idx_local, future_len_ds

    def _downsample_without_past(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        target_past_cur_future_valid: torch.Tensor,  # shape: (B, Pnn, time_len)
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        sample_idx_local: torch.Tensor,  # shape: (1+T_ds,)
        future_len_ds: int,
        future_len: int,
    ) -> Tuple[
            torch.Tensor,  # unnorm_diffusion_trajectory_stride
            Optional[torch.Tensor],  # unnorm_near_past_xyyaw_stride
            torch.Tensor,  # near_past_cur_future_valid_stride
            torch.Tensor,  # diffusion_trajectory_stride_norm
            Optional[torch.Tensor],  # near_past_xyyaw_stride_norm
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """과거를 사용하지 않고 현재~미래 구간만 stride 다운샘플합니다."""
        B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
        if one_plus_T != 1 + future_len:
            raise ValueError(
                f"[FeasibleProjector] diffusion_trajectory.shape[2]={one_plus_T} "
                f"!= 1 + future_len={1 + future_len}")

        # 현재~미래 유효 마스크만 추출
        cur_future_valid = target_past_cur_future_valid[:, :, -(
            1 + future_len):].to(torch.bool)  # shape: (B, Pnn, 1+future_len)

        # 역정규화 궤적 다운샘플
        unnorm_diffusion_trajectory_stride = unnorm_diffusion_trajectory[:, :,
                                                                         sample_idx_local, :]  # shape: (B, Pnn, 1+T_ds, 4)
        unnorm_near_past_xyyaw_stride: Optional[torch.Tensor] = None

        # 정규화 궤적 다운샘플
        diffusion_trajectory_stride_norm = diffusion_trajectory[:, :,
                                                                sample_idx_local, :]  # shape: (B, Pnn, 1+T_ds, 4)
        near_past_xyyaw_stride_norm: Optional[torch.Tensor] = None

        # 유효 마스크도 동일 index 로 다운샘플
        near_past_cur_future_valid_stride = cur_future_valid[:, :,
                                                             sample_idx_local]  # shape: (B, Pnn, 1+T_ds)

        past_len_ds: int = 0

        return (
            unnorm_diffusion_trajectory_stride,
            unnorm_near_past_xyyaw_stride,
            near_past_cur_future_valid_stride,
            diffusion_trajectory_stride_norm,
            near_past_xyyaw_stride_norm,
            past_len_ds,
            future_len_ds,
        )

    def build_downsampled_feasible_inputs(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4) 정규화 현재+미래
        target_past: Optional[
            torch.Tensor],  # shape: (B, Pnn, past_len, 11) 또는 None
        target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, time_len=1+past_len+T) bool
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        unnorm_near_past_xyyaw: Optional[
            torch.Tensor],  # shape: (B, Pnn, past_len, 4) 또는 None
        stride_step: int,
    ) -> Tuple[
            torch.
            Tensor,  # unnorm_diffusion_trajectory_stride # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # unnorm_near_past_xyyaw_stride # shape: (B, Pnn, past_len_ds, 4) or None
            torch.
            Tensor,  # near_past_cur_future_valid_stride # shape: (B, Pnn, past_len_ds+1+T_ds)
            torch.
            Tensor,  # diffusion_trajectory_stride_norm # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # near_past_xyyaw_stride_norm # shape: (B, Pnn, past_len_ds, 4) or None
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """FeasibleProjector용 다운샘플링 궤적/마스크를 구성하는 메인 함수.

        흐름:
            1) 과거 길이(past_len), 미래 길이(future_len)를 계산합니다.
            2) config + past_len 으로 과거 사용 여부(use_past)를 정합니다.
            3) use_past 에 따라 stride index 를 만들고,
            4) 정규화/역정규화 궤적 + 유효 마스크에 같은 index 를 적용합니다.
        """
        device: torch.device = diffusion_trajectory.device

        # 1) 과거/미래 길이 계산
        past_len, future_len, _ = self._infer_past_future_lengths_for_downsample(
            diffusion_trajectory=diffusion_trajectory,  # shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # shape: (B, Pnn, 1+past_len+T)
        )

        # 2) config + past_len 으로 과거 사용 여부 결정
        use_past: bool = self._decide_use_past_for_downsample(
            past_len=past_len,
            target_past=target_past,
            unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
        )

        if use_past:
            assert target_past is not None
            assert unnorm_near_past_xyyaw is not None

            # 3-a) 과거+현재+미래 전체 타임라인용 stride index 계산
            """
            sample_idx_full: shape (K,)
            past_len_ds: int
            future_len_ds: int
            """
            (sample_idx_full, past_len_ds,
             future_len_ds) = self._build_stride_indices_with_past(
                 past_len=past_len,
                 future_len=future_len,
                 stride_step=stride_step,
                 device=device,
             )

            # 4-a) 전체 타임라인 기준으로 다운샘플 적용
            return self._downsample_with_past(
                diffusion_trajectory=diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                target_past=target_past,  # (B, Pnn, past_len, 11)
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, Pnn, time_len)
                unnorm_diffusion_trajectory=
                unnorm_diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                unnorm_near_past_xyyaw=
                unnorm_near_past_xyyaw,  # (B, Pnn, past_len, 4)
                sample_idx_full=sample_idx_full,  # (K,)
                past_len_ds=past_len_ds,  # int
                future_len_ds=future_len_ds,  # int
            )

        # 3-b) 과거 미사용: 현재~미래(1+T) 구간만 사용하는 stride index 계산
        (sample_idx_local,
         future_len_ds) = self._build_stride_indices_without_past(
             future_len=future_len,
             stride_step=stride_step,
             device=device,
         )

        # 4-b) 현재~미래 구간만 다운샘플 적용
        return self._downsample_without_past(
            diffusion_trajectory=diffusion_trajectory,
            target_past_cur_future_valid=target_past_cur_future_valid,
            unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
            sample_idx_local=sample_idx_local,
            future_len_ds=future_len_ds,
            future_len=future_len,
        )

    def upsample_future_controls_from_stride(
        self,
        unnorm_seg_body_control_stride: torch.
        Tensor,  # (B, Pnn, segment_len_ds, 3)
        past_len_ds: int,
        future_len_ds: int,
        future_len_full: int,
        stride_step: int,
    ) -> torch.Tensor:
        """서브샘플링된 세그먼트 제어를 원래 future_len 길이로 선형 업샘플링합니다.

        Args:
            unnorm_seg_body_control_stride:
                (B, Pnn, segment_len_ds, 3)
                과거~현재~미래 또는 현재~미래 전체에 대한 서브샘플링 세그먼트 제어.
            past_len_ds:
                다운샘플 후 과거 세그먼트 개수.
                (use_past_for_feasible=False 인 경우 0)
            future_len_ds:
                다운샘플 후 미래 세그먼트 개수.
            future_len_full:
                원래 미래 세그먼트 개수(T). (예: 80)
            stride_step:
                시간 index 기준 다운샘플링 간격.

        Returns:
            (B, Pnn, future_len_full, 3)
                현재~미래 구간(base dt 해상도)의 제어 시퀀스.
        """
        B, Pnn, segment_len_ds, _ = unnorm_seg_body_control_stride.shape

        # 미래 부분만 추출
        start_future_idx: int = int(past_len_ds)
        if segment_len_ds < start_future_idx + future_len_ds:
            raise ValueError(
                "[FeasibleProjector] segment_len_ds 와 (past_len_ds, future_len_ds) 조합이 일치하지 않습니다."
            )
        # (B, Pnn, future_len_ds, 3)
        coarse_future_control = unnorm_seg_body_control_stride[:, :,
                                                               start_future_idx:
                                                               start_future_idx +
                                                               future_len_ds, :]

        # stride_step == 1 이면 다운샘플링이 없으므로 그대로 사용
        if stride_step == 1:
            if future_len_ds != future_len_full:
                raise ValueError(
                    "[FeasibleProjector] stride_step==1 인데 "
                    f"future_len_ds={future_len_ds} 와 future_len_full={future_len_full} 이 다릅니다."
                )
            return coarse_future_control  # (B, Pnn, future_len_full, 3)

        # stride_step > 1: 선형 업샘플링 필요
        B_flat: int = B * Pnn
        _, _, n_coarse, _ = coarse_future_control.shape
        if n_coarse != future_len_ds:
            raise ValueError(
                "[FeasibleProjector] coarse_future_control 길이와 future_len_ds 가 일치하지 않습니다."
            )

        if n_coarse <= 1:
            # coarse 값이 1개뿐이면 모든 future 시점에 동일한 제어를 복사
            return coarse_future_control.expand(B, Pnn, future_len_full, 3)

        # (B*Pnn, 3, n_coarse) 로 변환 후 1D linear interpolate
        controls_flat = coarse_future_control.reshape(
            B_flat, n_coarse, 3).permute(0, 2, 1)  # (B*Pnn, 3, n_coarse)

        # align_corners=True:
        #   index 0 ↔ 첫 미래 세그먼트, index n_coarse-1 ↔ 마지막 미래 세그먼트
        upsampled_flat = F.interpolate(
            controls_flat,  # (B*Pnn, 3, n_coarse)
            size=future_len_full,
            mode="linear",
            align_corners=True,
        )  # (B*Pnn, 3, future_len_full)

        upsampled = upsampled_flat.permute(0, 2, 1).reshape(
            B, Pnn, future_len_full, 3)  # (B, Pnn, future_len_full, 3)
        return upsampled

    def get_feasible_stride_params(
        self,
        future_len: int,
    ) -> Tuple[int, float, int, int]:
        """FeasibleProjector용 다운샘플링 간격과 SG 윈도 길이를 계산합니다.

        Args:
            future_len: 미래 노드 개수 T. (예: 80)

        Returns:
            stride_step: 정수 스트라이드 (1이면 다운샘플링 없음).
            dt_for_savgol: SG 필터에 넘길 샘플 간 시간 간격 [초].
            max_window_len_xy: x,y 좌표에 사용할 최대 윈도 길이(샘플 수).
            max_window_len_yaw: yaw에 사용할 최대 윈도 길이(샘플 수).
        """

        # FeasibleProjector 내부에서 사용하는 base dt (원래 타임스텝, 예: 0.1s)
        base_dt: float = float(self.constraints_h_params.dt)

        # 사용자가 원하는 다운샘플 간격(초). 없으면 base_dt 그대로 사용.
        desired_dt: float = float(
            getattr(self.config, "feasible_stride_dt", base_dt))
        # base_dt 보다 작게 들어오면 의미가 없으니 최소 base_dt로 클램프
        if desired_dt < base_dt:
            desired_dt = base_dt

        # index 기준 스트라이드 = 원하는 시간 간격 / base_dt
        stride_step: int = max(1, int(round(desired_dt / base_dt)))
        dt_for_savgol: float = base_dt * float(stride_step)

        # "보고 싶은 시간 길이"를 초 단위로 고정해 두고,
        # stride에 맞게 샘플 개수를 다시 계산한다.
        default_window_xy: int = int(
            getattr(self.config, "feasible_sg_max_window_len_xy", 11))
        default_window_yaw: int = int(
            getattr(self.config, "feasible_sg_max_window_len_yaw", 7))
        window_time_xy: float = base_dt * float(default_window_xy)
        window_time_yaw: float = base_dt * float(default_window_yaw)

        max_window_len_xy: int = max(1,
                                     int(round(window_time_xy / dt_for_savgol)))
        max_window_len_yaw: int = max(
            1, int(round(window_time_yaw / dt_for_savgol)))

        # SG 필터 특성상 홀수 길이 강제
        if max_window_len_xy % 2 == 0:
            max_window_len_xy += 1
        if max_window_len_yaw % 2 == 0:
            max_window_len_yaw += 1

        # 너무 큰 창은 실제 시퀀스 길이보다 약간만 크게 제한
        max_allow_window: int = future_len + 1
        max_window_len_xy = min(max_window_len_xy, max_allow_window)
        max_window_len_yaw = min(max_window_len_yaw, max_allow_window)

        # 현재 설계에서는 "현재 → 마지막 미래 노드"가 항상 포함되고
        # 그 사이가 등간격이 되어야 하므로
        # stride_step 이 future_len 을 정확히 나누지 못하면 에러로 알려준다.
        if future_len % stride_step != 0:
            raise ValueError(
                "[FeasibleProjector] future_len="
                f"{future_len} 이(가) stride_step={stride_step} 로 나누어 떨어지지 않습니다. "
                "config.feasible_stride_dt 를 조정해서 future_len % stride_step == 0 이 되도록 해 주세요."
            )

        return stride_step, dt_for_savgol, max_window_len_xy, max_window_len_yaw

    # ----------------------------
    # [NEW] 시간축 전체 배치로 S0/S1/S3 제약 적용 (S2는 미사용)
    # ----------------------------
    def _apply_constraints_batch(
        self,
        vx_b_raw: torch.Tensor,  # (B, Pnn, T)  바디 기준 x속도
        vy_b_raw: torch.Tensor,  # (B, Pnn, T)  바디 기준 y속도
        omega_raw: torch.Tensor,  # (B, Pnn, T)  요각속도
        key_to_limit_bp: Dict[str, torch.Tensor],
        #   v_max/a_max/.../is_nonholonomic, 각 (B, Pnn)
        hp: _ConstraintHParams,
        slip_epsilon: float = 0.10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """S0, S1, S3 제약을 시간축 전체에 한 번에 적용한다.

        - 입력 속도/각속도 시퀀스 (B, Pnn, T)에 대해
          같은 규칙(S0/S1/S3)을 모든 시간에 동시에 적용한다.
        - S2(가속도/각가속도 증분 제한)는 여기서는 아예 사용하지 않는다.
        """
        if not self.use_feasible_filter:
            # 필터를 끄면 그대로 통과
            return vx_b_raw, vy_b_raw, omega_raw

        # (S0) 비홀로노믹: y방향 속도 거의 0으로 유지
        vx_after, vy_after = self._apply_S0_nonholonomic_ste(
            vx_b=vx_b_raw,  # (B,Pnn,T)
            vy_b=vy_b_raw,  # (B,Pnn,T)
            slip_epsilon=slip_epsilon,
            eta=hp.eta_slip,
            eps=hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],  # (B,Pnn)
        )

        # (S1) 최대 속도 제한
        vx_after, vy_after = self._apply_S1_speed_limit_ste(
            vx_b=vx_after,  # (B,Pnn,T)
            vy_b=vy_after,  # (B,Pnn,T)
            v_max=key_to_limit_bp["v_max"],  # (B,Pnn)
            eta=hp.eta_speed,
            eps=hp.eps,
        )

        # (S3) 속도-연동 요각속도 제한
        omega_after = self._apply_S3_omega_clip_ste(
            vx_b=vx_after,  # (B,Pnn,T)
            vy_b=vy_after,  # (B,Pnn,T)
            omega=omega_raw,  # (B,Pnn,T)
            a_lat_max=key_to_limit_bp["a_lat_max"],  # (B,Pnn)
            R_min=key_to_limit_bp["R_min"],  # (B,Pnn)
            omega_abs_max=key_to_limit_bp["omega_abs_max"],  # (B,Pnn)
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],  # (B,Pnn)
            eta=hp.eta_yaw,
            eps=hp.eps,
        )

        return vx_after, vy_after, omega_after  # 모두 (B,Pnn,T)

    # ----------------------------
    # [NEW] 시간축 전체 배치 중점 적분 (cumsum 기반)
    # ----------------------------
    def _integrate_midpoint_batch(
        self,
        unnorm_near_current_state: torch.
        Tensor,  # (B, Pnn, 4)  [x0, y0, cos0, sin0]
        vx_b_seq: torch.Tensor,  # (B, Pnn, T)  바디 기준 x속도 시퀀스
        vy_b_seq: torch.Tensor,  # (B, Pnn, T)  바디 기준 y속도 시퀀스
        omega_seq: torch.Tensor,  # (B, Pnn, T)  요각속도 시퀀스
        hp: _ConstraintHParams,
    ) -> Dict[str, torch.Tensor]:
        """시간축 전체에 대해 '중점 적분'을 한 번에 수행한다.

        - 각 구간 k 에 대해,
          yaw_k 에서 시작해서 yaw_k + w_k*dt 까지 회전한다고 보고
          중간 각(yaw_mid)을 이용해 세계 좌표 속도를 계산한다.
        - 이 세계 좌표 속도를 dt만큼 계속 더해 가며 위치를 만든다.
        - 모든 계산을 시간축에 대해 cumsum 으로 처리하므로,
          python for 루프 없이 한 번에 연산한다.
        """
        B, Pnn, T = vx_b_seq.shape  # T = future_len
        device = vx_b_seq.device
        dtype = vx_b_seq.dtype

        dt: float = hp.dt
        eps: float = hp.eps

        # 초기 위치/자세 (노드 0)
        x0 = unnorm_near_current_state[..., 0]  # (B,Pnn)
        y0 = unnorm_near_current_state[..., 1]  # (B,Pnn)
        cos0 = unnorm_near_current_state[..., 2]  # (B,Pnn)
        sin0 = unnorm_near_current_state[..., 3]  # (B,Pnn)

        # yaw0 (라디안) 복원
        yaw0 = torch.atan2(sin0, cos0)  # (B,Pnn)

        # 각속도 적분: Δθ_k = w_k * dt
        dtheta_seq = omega_seq * dt  # (B,Pnn,T)

        # 누적합: sum_{j<=k} Δθ_j
        dtheta_prefix = torch.cumsum(dtheta_seq, dim=2)  # (B,Pnn,T)

        # 각 구간 시작 각도 yaw_k = yaw0 + sum_{j<k} Δθ_j  (exclusive cumsum)
        zero_pad = torch.zeros_like(dtheta_seq[..., :1])  # (B,Pnn,1)
        dtheta_exclusive = torch.cat(
            [zero_pad, dtheta_prefix[..., :-1]],
            dim=2,
        )  # (B,Pnn,T)
        yaw_start = yaw0.unsqueeze(-1) + dtheta_exclusive  # (B,Pnn,T)

        # 중점/종단 각도
        yaw_mid = yaw_start + 0.5 * dtheta_seq  # (B,Pnn,T)
        yaw_next = yaw_start + dtheta_seq  # (B,Pnn,T)

        cos_mid = torch.cos(yaw_mid)  # (B,Pnn,T)
        sin_mid = torch.sin(yaw_mid)  # (B,Pnn,T)

        # 세계 기준 중점 속도
        vwx_mid = cos_mid * vx_b_seq - sin_mid * vy_b_seq  # (B,Pnn,T)
        vwy_mid = sin_mid * vx_b_seq + cos_mid * vy_b_seq  # (B,Pnn,T)

        # dt 만큼 이동량
        dx_seq = vwx_mid * dt  # (B,Pnn,T)
        dy_seq = vwy_mid * dt  # (B,Pnn,T)

        # 누적합으로 위치 만들기
        x_cumsum = torch.cumsum(dx_seq, dim=2)  # (B,Pnn,T)
        y_cumsum = torch.cumsum(dy_seq, dim=2)  # (B,Pnn,T)

        x_next = x0.unsqueeze(-1) + x_cumsum  # (B,Pnn,T)
        y_next = y0.unsqueeze(-1) + y_cumsum  # (B,Pnn,T)

        # 최종 yaw(k+1) → cos/sin
        cos_next = torch.cos(yaw_next)  # (B,Pnn,T)
        sin_next = torch.sin(yaw_next)  # (B,Pnn,T)

        # 수치 오차 보정용 정규화
        norm_cs = torch.sqrt(cos_next * cos_next + sin_next * sin_next +
                             eps)  # (B,Pnn,T)
        cos_next = cos_next / norm_cs
        sin_next = sin_next / norm_cs

        key_to_all_states: Dict[str, torch.Tensor] = {
            "x_next": x_next,  # (B,Pnn,T)
            "y_next": y_next,  # (B,Pnn,T)
            "cos_next": cos_next,  # (B,Pnn,T)
            "sin_next": sin_next,  # (B,Pnn,T)
            "vx_after": vx_b_seq,  # (B,Pnn,T)
            "vy_after": vy_b_seq,  # (B,Pnn,T)
            "omega_after": omega_seq,  # (B,Pnn,T)
        }
        return key_to_all_states

    def _sg_build_design_matrix_and_gram(
        self,
        window_length: int,
        polyorder: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Savitzky–Golay 미분에 필요한 시간 관련 텐서들을 만든다.

        이 함수가 하는 일은 크게 두 가지다.

        1) 시간 특징 텐서(time_feature) 만들기
           - 한 윈도 안에 들어가는 점의 개수를 window_length = W 라고 하자. (예: W=11)
           - 윈도 안의 각 점에 대해, "중심에서 얼마나 떨어져 있는지" 시간을 만든다.
             예를 들어 W=5, dt=0.1 이면
                 인덱스 k = 0,1,2,3,4 에 대해
                 중심은 k=2 이고,
                 시간 값 τ_k 는 다음과 같이 잡는다.
                     k=0 → τ = -2*dt
                     k=1 → τ = -1*dt
                     k=2 → τ =  0
                     k=3 → τ = +1*dt
                     k=4 → τ = +2*dt
           - 그리고 각 위치마다 다음과 같은 값들을 한 줄로 모은다.
                 [1, τ, τ², τ³, ..., τ^P]
             여기서 P = polyorder 이다. (예: polyorder=2 이면 [1, τ, τ²])
           - 이렇게 해서 얻는 텐서가 time_feature 이다.
             shape: (W, P+1)
               - 첫 번째 축: 윈도 안에서의 위치(시간 순서)
               - 두 번째 축: [1, τ, τ², ...] 항들

        2) 시간 특징 쌍 텐서(time_feature_pair) 미리 계산하기
           - Savitzky–Golay 미분을 할 때는
             time_feature 를 가지고 여러 번 곱셈과 덧셈을 반복해서 쓰게 된다.
           - 매번 같은 곱셈을 반복하지 않기 위해,
             time_feature[row] 의 각 항들끼리 곱해 둔 값을 미리 계산해서 모아 둔다.
             이 값은 나중에 “작은 (P+1)×(P+1) 행렬”들을 만들 때 바로 재사용된다.
           - 이렇게 모아 둔 텐서가 time_feature_pair 이다.
             shape: (W, P+1, P+1)
               - 첫 번째 축: 윈도 안에서의 위치
               - 두 번째/세 번째 축: time_feature 의 각 열 조합에 대한 곱

        Args:
            window_length (int):
                한 번에 보는 윈도 길이 W (항상 홀수여야 한다. 예: 5, 7, 11 ...)
            polyorder (int):
                τ, τ², τ³ ... 를 몇 제곱까지 사용할지 (다항식 차수)
            dt (float):
                샘플 간 시간 간격 (초 단위)
            device (torch.device):
                결과 텐서를 놓을 디바이스 (cpu, cuda 등)
            dtype (torch.dtype):
                결과 텐서의 데이터 타입 (torch.float32, torch.float16 등)

        Returns:
            time_feature (torch.Tensor):
                shape: (W, P+1)
                윈도 안 각 위치에 대해 [1, τ, τ², ...] 값을 쌓아 놓은 텐서.

            time_feature_pair (torch.Tensor):
                shape: (W, P+1, P+1)
                time_feature 에서 나오는 값들끼리의 곱을 미리 모아 둔 텐서.
                나중에 (N,T) 전체에 대해 작은 (P+1,P+1) 행렬을 빠르게 만들 때 쓴다.
        """
        W: int = int(window_length)
        P_plus_one: int = int(polyorder) + 1

        # τ_k: (W,) = [-half*dt, ..., 0, ..., +half*dt]
        half_window: int = W // 2
        tau_k: Tensor = (torch.arange(W, device=device, dtype=dtype) -
                         float(half_window)) * float(dt)  # (W,)

        # design_matrix: (W, P+1)  = [1, τ, τ², ...]
        basis_list: List[Tensor] = [torch.ones_like(tau_k)]  # (W,)
        for degree in range(1, P_plus_one):
            basis_list.append(tau_k**degree)  # (W,)
        design_matrix: Tensor = torch.stack(basis_list, dim=-1)  # (W, P+1)

        # gram_per_k: (W, P+1, P+1)  = Φ_k^T Φ_k (k별 outer product)
        design_row: Tensor = design_matrix.unsqueeze(-1)  # (W, P+1, 1)
        design_col: Tensor = design_matrix.unsqueeze(-2)  # (W, 1, P+1)
        gram_per_k: Tensor = design_row * design_col  # (W, P+1, P+1)

        # power_per_k: (W, P+1)  = Φ_k (b 계산에 직접 사용)
        power_per_k: Tensor = design_matrix  # (W, P+1)

        return design_matrix, gram_per_k, power_per_k

    # <추가하자>
    def _prepare_points_and_masks(
            self,
            unnorm_diffusion_trajectory: torch.Tensor,  # (B,Pnn,1+future_len,4)
            unnorm_near_past_xyyaw: Optional[
                torch.Tensor],  # (B,Pnn,past_len,4) or None
            target_past_cur_future_valid: torch.
        Tensor,  # (B,Pnn,past_len+1+future_len) bool
    ) -> PointLenInputs:
        """(1) 과거/현재/미래 포인트 결합
           (2) past_cur / cur_future 마스크 분할
           (3) 단조성 검증


        return PointLenInputs(
            unnorm_points_xyyaw,  # (B,Pnn,point_len,4)
            points_valid, # (B,Pnn,point_len)  bool
            past_cur_valid, # (B,Pnn,past_len+1) bool or None
            cur_future_valid, # (B,Pnn,1+future_len) bool
            past_len, # int
            point_len
                if unnorm_near_past_xyyaw is None:
                    = 1 + future_len
                else:
                    = past_len + 1 + future_len
        """
        B, Pnn, T1_fut, C = unnorm_diffusion_trajectory.shape
        assert C == 4, "xyyaw 마지막 채널은 4여야 합니다."
        Bv, Pnnv, total_time_len = target_past_cur_future_valid.shape
        assert (Bv, Pnnv) == (B, Pnn), "valid 마스크 (B,Pnn) 불일치"
        if total_time_len < T1_fut:  # T1_fut = 1 + future_len
            raise ValueError("valid 마스크 길이가 (1+future_len)보다 짧습니다.")
        if unnorm_near_past_xyyaw is None:
            past_len = 0
            # point_len = 1 + future_len
            unnorm_points_xyyaw = unnorm_diffusion_trajectory  # (B,Pnn,1+future_len,4)
            # target_past_cur_future_valid: (B,Pnn,past_len+1+future_len)
            past_cur_valid = None  # 과거~현재 마스크 없음
            cur_future_valid = target_past_cur_future_valid[:, :, -T1_fut:].to(
                torch.bool)  # (B,Pnn,1+future_len)
            points_valid = cur_future_valid  # (B,Pnn,1+future_len)
            # 현재~미래: True*False* 검증
            self._assert_cur_future_valid_mask(
                cur_future_valid,
                context="savgol_filter_for_control_cur_future")
        else:
            assert unnorm_near_past_xyyaw.shape[:2] == (B, Pnn)
            assert unnorm_near_past_xyyaw.shape[-1] == 4
            past_len = int(unnorm_near_past_xyyaw.shape[2])
            time_len = past_len + 1
            # 포인트 결합
            # point_len = past_len + 1 + future_len
            unnorm_points_xyyaw = torch.cat(
                [unnorm_near_past_xyyaw, unnorm_diffusion_trajectory],
                dim=2)  # (B,Pnn,past_len+1+future_len,4)
            # 마스크 분리
            all_valid = target_past_cur_future_valid.to(
                torch.bool)  # (B,Pnn,past_len+1+future_len)
            past_cur_valid = all_valid[..., :time_len]  # (B,Pnn,past_len+1)
            cur_future_valid = all_valid[..., past_len:]  # (B,Pnn,1+future_len)
            points_valid = all_valid  # (B,Pnn,point_len)
            # 검증: 과거~현재(0*1*), 현재~미래(1*0*)
            self._assert_past_cur_valid_mask(
                past_cur_valid, context="savgol_filter_for_control_past_cur")
            self._assert_cur_future_valid_mask(
                cur_future_valid,
                context="savgol_filter_for_control_cur_future")

        point_len = int(unnorm_points_xyyaw.shape[2])
        """
        point_len
            if unnorm_near_past_xyyaw is None:
                = 1 + future_len
            else:
                = past_len + 1 + future_len
        """

        return PointLenInputs(
            unnorm_points_xyyaw=unnorm_points_xyyaw,  # (B,Pnn,point_len,4)
            points_valid=points_valid,  # (B,Pnn,point_len)  bool
            past_cur_valid=past_cur_valid,  # (B,Pnn,past_len+1) bool or None
            cur_future_valid=cur_future_valid,  # (B,Pnn,1+future_len) bool
            past_len=past_len,  # int
            point_len=point_len,  # int
        )

    # =========================================================
    # [추가 필요] STE 유틸 (공통)
    # =========================================================
    @staticmethod
    def _smoothstep_quintic01(t: torch.Tensor) -> torch.Tensor:
        """[0,1]→[0,1], C² smoothstep: 10t^3 - 15t^4 + 6t^5."""
        return t**3 * (10 - 15 * t + 6 * t * t)

    @staticmethod
    def _ste_band_weight(r: torch.Tensor, eta: float) -> torch.Tensor:
        """w(r): inside=1, band=1-φ, outside=0."""
        eta = float(max(0.0, eta))
        if eta == 0.0:
            return (r <= 1.0).to(r.dtype)
        r_lo, r_hi = 1.0 - eta, 1.0 + eta
        t = ((r - r_lo) / (2.0 * eta)).clamp(0.0, 1.0)
        phi = FeasibleProjector._smoothstep_quintic01(t)
        w_mid = 1.0 - phi
        w = torch.where(r <= r_lo, torch.ones_like(r),
                        torch.where(r >= r_hi, torch.zeros_like(r), w_mid))
        return w

    @staticmethod
    def _ste_scalar_clip(x: torch.Tensor, limit: torch.Tensor, eta: float,
                         eps: float) -> torch.Tensor:
        """forward=hard clamp, backward=band-weighted identity."""
        limit = limit.to(dtype=x.dtype, device=x.device)
        y_hard = x.clamp(-limit, limit)
        r = x.abs() / (limit + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * x + (1.0 - w) * x.detach()
        y = y_sur + (y_hard - y_sur).detach()
        return y

    @staticmethod
    def _ste_increment_vec(dv: torch.Tensor, limit: torch.Tensor, eta: float,
                           eps: float) -> torch.Tensor:
        """증분(벡터노름) STE-clip.

        dv: (B, Pnn, 2)
        limit: (B, Pnn)
        """
        limit = limit.to(dtype=dv.dtype,
                         device=dv.device).unsqueeze(-1)  # (B,Pnn,1)
        norm = torch.linalg.norm(dv, dim=-1,
                                 keepdim=True).clamp_min(eps)  # (B,Pnn,1)
        s_hard = torch.clamp(limit / norm, max=1.0)  # (B,Pnn,1)
        dv_h = s_hard * dv  # (B,Pnn,2)
        r = (norm.squeeze(-1)) / (limit.squeeze(-1) + eps)
        w = FeasibleProjector._ste_band_weight(r, eta).unsqueeze(-1)
        dv_sur = w * dv + (1 - w) * dv.detach()
        return dv_sur + (dv_h - dv_sur).detach()

    @staticmethod
    def _ste_yawrate_clip(omega_raw: torch.Tensor, allow: torch.Tensor,
                          eta: float, eps: float) -> torch.Tensor:
        """S3용: forward hard(±allow), backward는 r=|raw|/allow(detached)로 밴드 가중.

        omega_raw: (B, Pnn)
        allow: (B, Pnn)
        """
        allow = allow.to(dtype=omega_raw.dtype,
                         device=omega_raw.device)  # (B,Pnn)
        y_hard = omega_raw.clamp(-allow, allow)
        r = omega_raw.abs() / (allow.detach() + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * omega_raw + (1 - w) * omega_raw.detach()
        y = y_sur + (y_hard - y_sur).detach()

        return y

    @staticmethod
    def _ste_friction_scale(ax: torch.Tensor, ay: torch.Tensor,
                            ax_max: torch.Tensor, ay_max: torch.Tensor,
                            eta: float, eps: float) -> torch.Tensor:
        """S4용 타원 스케일: forward s_hard=min(1,1/r), backward s≈w(r)."""
        ax_max = ax_max.to(dtype=ax.dtype, device=ax.device)
        ay_max = ay_max.to(dtype=ay.dtype, device=ay.device)
        r = torch.sqrt((ax / (ax_max + eps))**2 + (ay / (ay_max + eps))**2 +
                       eps)
        s_hard = torch.clamp(1.0 / r, max=1.0)
        w = FeasibleProjector._ste_band_weight(r, eta)
        s_sur = w  # inside=1, band∈(0,1), outside=0
        s = s_sur + (s_hard - s_sur).detach()
        return s

    @staticmethod
    def _weighted_avg_two(
        a: torch.Tensor,
        b: torch.Tensor,
        a_valid: torch.Tensor,
        b_valid: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """두 텐서를 동일 shape에서 가중 평균합니다.

        Args:
            a (torch.Tensor): 좌측 값 (... 혹은 (..., D))
            b (torch.Tensor): 우측 값 (... 혹은 (..., D))
            a_valid (torch.Tensor): 좌측 가중치 (... 또는 (...,1))
            b_valid (torch.Tensor): 우측 가중치 (... 또는 (...,1))
            eps (float): 0으로 나눔 방지용 epsilon

        Returns:
            torch.Tensor: 가중 평균 텐서 (a,b와 동일 shape)
        """
        denom = (a_valid + b_valid).clamp_min(eps)
        return (a_valid * a + b_valid * b) / denom

    @staticmethod
    def _world_to_body(
        vx_w: torch.Tensor,
        vy_w: torch.Tensor,
        cos_yaw: torch.Tensor,
        sin_yaw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """세계 기준 속도를 몸체 기준 속도로 회전합니다.

        수식:
            v^b = R(-ψ) v^w,  R(-ψ) = [[cosψ, sinψ], [-sinψ, cosψ]]

        Args:
            vx_w (torch.Tensor): 세계 x축 속도 [..., T]
            vy (torch.Tensor):   세계 y축 속도 [..., T]
            cos_yaw (torch.Tensor): [..., T] cos(ψ)
            sin_yaw (torch.Tensor): [..., T] sin(ψ)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (v_x^b, v_y^b), shape 동일
        """
        vxb = cos_yaw * vx_w + sin_yaw * vy_w
        vyb = -sin_yaw * vx_w + cos_yaw * vy_w
        return vxb, vyb

    # <추가하자>
    def _prepare_midpoint_inputs(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        unnorm_near_past_xyyaw: Optional[
            torch.Tensor],  # (B, Pnn, past_len, 4) or None
        unnorm_points_world_control: torch.Tensor,  # (B, Pnn, point_len, 3)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len)+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """중점 제어 계산을 위해
        - 좌표 시퀀스(x, y, cos, sin)
        - 월드 프레임 제어(v_x^w, v_y^w, w)
        - 유효 마스크
        를 하나의 공통 타임라인 기준(point_len)으로 정리합니다.

        Returns:
            unnorm_points_xyyaw: (B, Pnn, point_len, 4)
            unnorm_points_world_control_aligned: (B, Pnn, point_len, 3)
            points_valid: (B, Pnn, point_len)  bool
            point_len: int
        """
        # savgol_filter_for_control과 동일한 규칙으로 포인트/마스크 정리
        point_len_inputs: PointLenInputs = self._prepare_points_and_masks(
            unnorm_diffusion_trajectory=
            unnorm_diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
            unnorm_near_past_xyyaw=
            unnorm_near_past_xyyaw,  # (B, Pnn, past_len, 4) or None
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, Pnn, time_len(=1+past_len)+future_len)
        )
        unnorm_points_xyyaw = point_len_inputs.unnorm_points_xyyaw  # (B, Pnn, point_len, 4)
        points_valid = point_len_inputs.points_valid  # (B, Pnn, point_len)
        B, Pnn, point_len, _ = unnorm_points_xyyaw.shape

        B_c, P_c, point_len_control, _ = unnorm_points_world_control.shape
        assert point_len_control == point_len, \
            f"포인트 개수(point_len={point_len})와 제어 시퀀스 길이(point_len_control={point_len_control})가 다릅니다."

        return unnorm_points_xyyaw, unnorm_points_world_control, points_valid, point_len

    # <추가하자>
    # <추가하자>
    def compute_midpoint_controls(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,
        # (B, Pnn, 1+future_len, 4) = [x, y, cos, sin] (현재~미래 구간)
        unnorm_near_past_xyyaw: Optional[torch.Tensor],
        # (B, Pnn, past_len, 4) = [x, y, cos, sin] (과거 구간) 또는 None
        unnorm_points_world_control: torch.Tensor,
        # (B, Pnn, point_len, 3) = [v_x^w, v_y^w, w]  (savgol_filter_for_control 출력)
        target_past_cur_future_valid: torch.Tensor,
        # (B, Pnn, time_len(=1+past_len) + future_len) bool  # 과거~현재~미래 노드 유효 마스크
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)  [v_x^b, v_y^b, w]_mid
        """구간 [t_k, t_{k+1})마다 중점(midpoint) 제어 [v_x^b, v_y^b, w]를 계산한다.

        segment_len = point_len - 1 이고,
        point_len 은 `_prepare_midpoint_inputs`가 반환하는 값에 따라
        - past 없음:  point_len = 1 + future_len → segment_len = future_len
        - past 있음: point_len = past_len + 1 + future_len → segment_len = past_len + future_len
        로 결정된다.
        """
        # 1) 포인트/제어/마스크를 하나의 타임라인 기준으로 정리
        (
            unnorm_points_xyyaw,  # (B, Pnn, point_len, 4)
            unnorm_points_world_control_aligned,  # (B, Pnn, point_len, 3)
            points_valid,  # (B, Pnn, point_len) bool
            point_len,  # int
        ) = self._prepare_midpoint_inputs(
            unnorm_diffusion_trajectory=
            unnorm_diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
            unnorm_near_past_xyyaw=
            unnorm_near_past_xyyaw,  # (B, Pnn, past_len, 4) or None
            unnorm_points_world_control=
            unnorm_points_world_control,  # (B, Pnn, point_len, 3)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, Pnn, time_len(=1+past_len) + future_len)
        )

        if point_len < 2:
            raise ValueError("포인트 개수가 2개 미만이면 중점 제어를 계산할 수 없습니다.")

        eps = 1e-6

        # 2) 노드 / 제어 성분 분解
        (
            cos_all,  # (B, Pnn, point_len)
            sin_all,  # (B, Pnn, point_len)
            v_x_all,  # (B, Pnn, point_len)
            v_y_all,  # (B, Pnn, point_len)
            omega_all  # (B, Pnn, point_len)
        ) = self._split_midpoint_nodes_and_controls(
            unnorm_points_xyyaw=unnorm_points_xyyaw,  # (B, Pnn, point_len, 4)
            unnorm_points_world_control=
            unnorm_points_world_control_aligned,  # (B, Pnn, point_len, 3)
        )

        # 3) 세그먼트 유효 마스크(시작/끝/구간) 계산
        start_valid, end_valid, seg_valid = self._build_midpoint_segment_valid_masks(
            points_valid=points_valid,  # (B, Pnn, point_len) bool
            value_dtype=v_x_all.dtype,
        )  # 모두 (B, Pnn, segment_len)

        # 4) 세계 프레임 중점 속도/각속도 계산
        v_x_mid_w, v_y_mid_w, omega_mid = self._compute_midpoint_world_values(
            v_x_all=v_x_all,  # (B, Pnn, point_len)
            v_y_all=v_y_all,  # (B, Pnn, point_len)
            omega_all=omega_all,  # (B, Pnn, point_len)
            start_valid=start_valid,  # (B, Pnn, segment_len)
            end_valid=end_valid,  # (B, Pnn, segment_len)
            eps=eps,
        )  # (B, Pnn, segment_len) 각각

        # 5) 중간 yaw(cos/sin) 계산
        cos_mid, sin_mid = self._compute_midpoint_yaw_from_cos_sin(
            cos_all=cos_all,  # (B, Pnn, point_len)
            sin_all=sin_all,  # (B, Pnn, point_len)
            start_valid=start_valid,  # (B, Pnn, segment_len)
            end_valid=end_valid,  # (B, Pnn, segment_len)
            eps=eps,
        )  # (B, Pnn, segment_len) 각각

        # 6) 세계 → 바디 프레임 회전 + 무효 구간 마스킹
        unnorm_seg_body_control = self._rotate_midpoint_world_to_body_and_apply_mask(
            v_x_mid_world=v_x_mid_w,  # (B, Pnn, segment_len)
            v_y_mid_world=v_y_mid_w,  # (B, Pnn, segment_len)
            omega_mid=omega_mid,  # (B, Pnn, segment_len)
            cos_mid=cos_mid,  # (B, Pnn, segment_len)
            sin_mid=sin_mid,  # (B, Pnn, segment_len)
            seg_valid=seg_valid,  # (B, Pnn, segment_len)
        )  # (B, Pnn, segment_len, 3)
        return unnorm_seg_body_control

    # <추가하자>
    def _build_midpoint_segment_valid_masks(
        self,
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        value_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """중점 제어 계산용 세그먼트 유효 마스크를 생성합니다.

        Args:
            points_valid: (B, Pnn, point_len) bool
                각 노드(x,y,cos,sin)가 유효한지 여부.
            value_dtype: torch.dtype
                v_x, v_y 등과 맞추기 위한 dtype (float32/float16 등)

        Returns:
            start_valid: (B, Pnn, segment_len)  # 왼쪽 끝 노드 유효 여부 (float)
            end_valid:   (B, Pnn, segment_len)  # 오른쪽 끝 노드 유효 여부 (float)
            seg_valid:   (B, Pnn, segment_len)  # 두 끝 모두 유효한 세그먼트(0/1 float)
        """
        # points_valid: (B, Pnn, point_len),  segment_len = point_len - 1
        valid_bool = points_valid.to(torch.bool)
        start_valid_bool = valid_bool[..., :-1]  # (B, Pnn, segment_len)
        end_valid_bool = valid_bool[..., 1:]  # (B, Pnn, segment_len)
        seg_valid_bool = start_valid_bool & end_valid_bool  # (B, Pnn, segment_len)

        start_valid = start_valid_bool.to(value_dtype)
        end_valid = end_valid_bool.to(value_dtype)
        seg_valid = seg_valid_bool.to(value_dtype)

        return start_valid, end_valid, seg_valid

    # <추가하자>
    def _rotate_midpoint_world_to_body_and_apply_mask(
            self,
            v_x_mid_world: torch.Tensor,  # (B, Pnn, segment_len)
            v_y_mid_world: torch.Tensor,  # (B, Pnn, segment_len)
            omega_mid: torch.Tensor,  # (B, Pnn, segment_len)
            cos_mid: torch.Tensor,  # (B, Pnn, segment_len)
            sin_mid: torch.Tensor,  # (B, Pnn, segment_len)
            seg_valid: torch.Tensor,  # (B, Pnn, segment_len)  float {0,1}
    ) -> torch.Tensor:
        """세계 속도 중점 + 중간 yaw → 바디 프레임 중점 제어로 변환하고, 무효 구간은 0으로 마스킹한다.

        Returns:
            unnorm_seg_body_control: (B, Pnn, segment_len, 3)
        """
        vxb_mid, vyb_mid = self._world_to_body(
            v_x_mid_world,  # (B, Pnn, segment_len)
            v_y_mid_world,  # (B, Pnn, segment_len)
            cos_mid,  # (B, Pnn, segment_len)
            sin_mid,  # (B, Pnn, segment_len)
        )

        vxb_mid = vxb_mid * seg_valid
        vyb_mid = vyb_mid * seg_valid
        omega_mid = omega_mid * seg_valid

        unnorm_seg_body_control = torch.stack(
            [vxb_mid, vyb_mid, omega_mid],
            dim=-1,
        )  # (B, Pnn, segment_len, 3)

        return unnorm_seg_body_control

    # <추가하자>
    def _compute_midpoint_yaw_from_cos_sin(
        self,
        cos_all: torch.Tensor,  # (B, Pnn, point_len)
        sin_all: torch.Tensor,  # (B, Pnn, point_len)
        start_valid: torch.Tensor,  # (B, Pnn, segment_len)
        end_valid: torch.Tensor,  # (B, Pnn, segment_len)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 끝점의 cos/sin과 유효 마스크로 중간 yaw의 cos/sin을 계산한다.

        Returns:
            cos_mid: (B, Pnn, segment_len)
            sin_mid: (B, Pnn, segment_len)
        """
        cos_start = cos_all[..., :-1]  # (B, Pnn, segment_len)
        cos_end = cos_all[..., 1:]  # (B, Pnn, segment_len)
        sin_start = sin_all[..., :-1]  # (B, Pnn, segment_len)
        sin_end = sin_all[..., 1:]  # (B, Pnn, segment_len)

        # cos_mid: (B, Pnn, segment_len)
        cos_mid = self._weighted_avg_two(
            cos_start,  # (B, Pnn, segment_len)
            cos_end,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        # sin_mid: (B, Pnn, segment_len)
        sin_mid = self._weighted_avg_two(
            sin_start,  # (B, Pnn, segment_len)
            sin_end,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        norm = (cos_mid * cos_mid + sin_mid * sin_mid).clamp_min(eps).sqrt()
        cos_mid = cos_mid / norm
        sin_mid = sin_mid / norm

        return cos_mid, sin_mid

    # <추가하자>
    def _compute_midpoint_world_values(
        self,
        v_x_all: torch.Tensor,  # (B, Pnn, point_len)
        v_y_all: torch.Tensor,  # (B, Pnn, point_len)
        omega_all: torch.Tensor,  # (B, Pnn, point_len)
        start_valid: torch.Tensor,  # (B, Pnn, segment_len)
        end_valid: torch.Tensor,  # (B, Pnn, segment_len)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """세계 프레임에서 중점 v_x^w, v_y^w, w 를 계산한다.

        Returns:
            v_x_mid_w: (B, Pnn, segment_len)
            v_y_mid_w: (B, Pnn, segment_len)
            omega_mid: (B, Pnn, segment_len)
        """
        # 구간 양 끝점 값
        v_x_start_w = v_x_all[..., :-1]  # (B, Pnn, segment_len)
        v_x_end_w = v_x_all[..., 1:]  # (B, Pnn, segment_len)
        v_y_start_w = v_y_all[..., :-1]  # (B, Pnn, segment_len)
        v_y_end_w = v_y_all[..., 1:]  # (B, Pnn, segment_len)

        omega_start = omega_all[..., :-1]  # (B, Pnn, segment_len)
        omega_end = omega_all[..., 1:]  # (B, Pnn, segment_len)

        # v_x_mid_w: (B, Pnn, segment_len)
        v_x_mid_w = self._weighted_avg_two(
            v_x_start_w,  # (B, Pnn, segment_len)
            v_x_end_w,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        # v_y_mid_w: (B, Pnn, segment_len)
        v_y_mid_w = self._weighted_avg_two(
            v_y_start_w,  # (B, Pnn, segment_len)
            v_y_end_w,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  #
            eps=eps)  # (B, Pnn, segment_len)

        # omega_mid: (B, Pnn, segment_len)
        omega_mid = self._weighted_avg_two(
            omega_start,  # (B, Pnn, segment_len)
            omega_end,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        return v_x_mid_w, v_y_mid_w, omega_mid

    # ------------------------------
    # 내부: mask‑aware depthwise conv (정규화 합성곱)
    # ------------------------------
    def _depthwise_conv_masked(
        self,
        seq_bptc: torch.Tensor,  # (B,Pnn,T,C)
        seg_mask_bpt: torch.Tensor,  # (B,Pnn,T), float(0/1)
        block_idx: int,
    ) -> torch.Tensor:
        """정규화 depthwise conv.

        Args:
            seq_bptc: (B,Pnn,T,C) 입력(마스크가 적용된 값 권장)
            seg_mask_bpt: (B,Pnn,T) 0/1 마스크
            block_idx: 블록 인덱스(0..3)

        Returns:
            (B,Pnn,T,C) 동일 길이 출력
        """
        B, Pnn, T, C = seq_bptc.shape
        conv = self.tcn_dw[block_idx]
        # (B*Pnn, C, T)
        x = seq_bptc.permute(0, 1, 3, 2).reshape(B * Pnn, C, T)
        m = seg_mask_bpt.reshape(B * Pnn, 1, T)  # (B*Pnn,1,T)

        # 분자: (x * m) 에 depthwise conv
        y_num = conv(x * m)  # (B*Pnn, C, T)

        # 분모: mask에 ones-kernel
        k = conv.kernel_size[0]
        pad = conv.padding[0]
        dil = conv.dilation[0]
        ones = torch.ones((1, 1, k), device=x.device, dtype=x.dtype)
        den = F.conv1d(m, ones, padding=pad, dilation=dil)  # (B*Pnn,1,T)
        y = y_num / den.clamp_min(self._eps)  # 브로드캐스트로 채널 공용 분모

        # (B,Pnn,T,C)
        y = y.view(B, Pnn, C, T).permute(0, 1, 3, 2).contiguous()
        return y

    # =========================================================
    # [A] 마스크/입력 전처리
    # =========================================================
    def _build_segment_mask(self, near_cur_future_valid):
        valid = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        seg_valid = (valid[..., :-1] & valid[..., 1:])  # (B,Pnn,T)  # ← AND
        seg_mask = seg_valid.to(torch.float32)
        seg_mask_1 = seg_mask.unsqueeze(-1)
        return seg_mask, seg_mask_1

    def _split_prev_fut(
            self,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+T, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """현재/미래 노드로 분리.

        Args:
            diffusion_trajectory (torch.Tensor): (B, Pnn, 1+T, 4) = [x, y, cos, sin]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_prev: (B, Pnn, T, 4)  ← t_k
                - x_fut : (B, Pnn, T, 4)  ← t_{k+1}
        """
        x_prev = diffusion_trajectory[..., :-1, :]  # (B,Pnn,T,4)
        x_fut = diffusion_trajectory[..., 1:, :]  # (B,Pnn,T,4)
        return x_prev, x_fut

    def _normalize_cos_sin(
        self,
        xycs: torch.Tensor,  # (B, Pnn, T, 4)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """(cos, sin)을 단위원으로 재투영(수치 안전).

        Args:
            xycs (torch.Tensor): (B, Pnn, T, 4) = [x, y, cos, sin]
            eps (float): 0 나눗셈 방지용 epsilon

        Returns:
            torch.Tensor: (B, Pnn, T, 4)  재투영된 텐서
        """
        cs = xycs[..., 2:4]
        norm = torch.linalg.norm(cs, dim=-1, keepdim=True).clamp_min(eps)
        out = xycs.clone()
        out[..., 2:4] = cs / norm
        return out

    # =========================================================
    # [B] 피처 인코딩 (prev/fut/u_base/trunk)
    # =========================================================
    def _features_from_inputs(
            self,
            x_prev: torch.Tensor,  # (B, Pnn, segment_len, 4)
            x_fut: torch.Tensor,  # (B, Pnn, segment_len, 4)
            u_base: torch.Tensor,  # (B, Pnn, segment_len, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
            points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len)
    ) -> torch.Tensor:
        """입력을 통일 피처 Z_in으로 변환.

        파이프라인:
            prev/fut 인코딩(16) + control 어댑터(32) + trunk 압축(64) → concat(192)

        Args:
            x_prev (torch.Tensor): (B, Pnn, segment_len, 4)
            x_fut (torch.Tensor): (B, Pnn, segment_len, 4)
            u_base (torch.Tensor): (B, Pnn, segment_len, 3)
            dit_final_hidden_tokens (torch.Tensor): (B, Pnn, H)

        Returns:
            torch.Tensor: (B, Pnn, segment_len, 192)  = Z_in
        """
        # 함수 본문 초입에 추가
        valid = points_valid.to(x_prev.dtype)  # (B,Pnn,1+segment_len)   요망!
        start_valid = valid[..., :-1].unsqueeze(
            -1)  # (B,Pnn,segment_len,1)   요망!
        end_valid = valid[..., 1:].unsqueeze(-1)  # (B,Pnn,segment_len,1)   요망!

        # prev/fut 노드 유효성으로 입력 자체 0화 (LN/Linear 이전 차단)
        x_prev = x_prev * start_valid
        x_fut = x_fut * end_valid

        # 세그먼트 유효성(AND)로 u_base 0화
        seg_mask, seg_mask_1 = self._build_segment_mask(points_valid)
        seg_mask_1 = seg_mask_1.to(u_base.dtype)

        if self.detach_u_for_ctrl_losses:
            u_base_in = (u_base.detach()) * seg_mask_1
        else:
            u_base_in = u_base * seg_mask_1

        feat_prev = self.state_prev_encoder(x_prev)  # (B,Pnn,segment_len,16)
        feat_fut = self.state_fut_encoder(x_fut)  # (B,Pnn,segment_len,16)
        feat_u = self.control_adapter(u_base_in)  # (B,Pnn,segment_len,32)
        # dit_final_hidden_tokens: (B,Pnn,H)
        # trunk: (B,Pnn,64)
        trunk = self.trunk_compressor(dit_final_hidden_tokens)
        trunk_rep = trunk.unsqueeze(2).expand(-1, -1, x_prev.size(2),
                                              -1)  # (B,Pnn,segment_len,64)

        Z_in = torch.cat([feat_prev, feat_fut, feat_u, trunk_rep],
                         dim=-1)  # (B,Pnn,segment_len,192)
        return Z_in

    def _split_midpoint_nodes_and_controls(
        self,
        unnorm_points_xyyaw: torch.Tensor,  # (B, Pnn, point_len, 4)
        unnorm_points_world_control: torch.Tensor,  # (B, Pnn, point_len, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """중점 제어 계산 전, 포인트 단위 성분을 분해한다.

        Returns:
            cos_all:   (B, Pnn, point_len)
            sin_all:   (B, Pnn, point_len)
            v_x_all:   (B, Pnn, point_len)
            v_y_all:   (B, Pnn, point_len)
            omega_all: (B, Pnn, point_len)
        """
        cos_all = unnorm_points_xyyaw[..., 2]  # (B, Pnn, point_len)
        sin_all = unnorm_points_xyyaw[..., 3]  # (B, Pnn, point_len)

        v_x_all = unnorm_points_world_control[..., 0]  # (B, Pnn, point_len)
        v_y_all = unnorm_points_world_control[..., 1]  # (B, Pnn, point_len)
        omega_all = unnorm_points_world_control[..., 2]  # (B, Pnn, point_len)

        return cos_all, sin_all, v_x_all, v_y_all, omega_all

    # =========================================================
    # [C] Stem / TCN 백본
    # =========================================================
    def _prepare_tcn_input(
        self,
        Z_in: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """Stem: LayerNorm → 1×1 Linear → GELU (시간 길이 보존)

        Args:
            Z_in (torch.Tensor): (B, Pnn, T, 192)
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 192)  (무효 시점 0-클램프 적용)
        """
        Z_s = self.stem_act(self.stem_fc(self.stem_norm(Z_in)))  # (B,Pnn,T,192)
        Z_s = Z_s * seg_mask_1
        return Z_s

    def _run_tcn(
        self,
        Z_s: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask: torch.Tensor,  # (B, Pnn, T)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """TCN 4블록 실행: [Pre-LN → depthwise(mask‑aware) → GELU → 1×1 → Residual] ×4

        Args:
            Z_s (torch.Tensor): (B, Pnn, T, 192)  Stem 출력
            seg_mask (torch.Tensor): (B, Pnn, T)  0/1
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 192)
        """
        Z = Z_s
        for block_idx in range(self.tcn_depth):
            # Z_ln: (B,Pnn,T,192)
            Z_ln = self.tcn_pre_lns[block_idx](Z) * seg_mask_1  # Pre-LN + mask
            # Y: (B,Pnn,T,192)
            Y = self._depthwise_conv_masked(Z_ln, seg_mask,
                                            block_idx)  # mask-aware
            Y = F.gelu(Y)
            Y = self.tcn_linear[block_idx](Y)  # pointwise 1×1
            Y = Y * seg_mask_1  # [추가] 블록 내부 중간단계에서도 0 고정
            Z = (Z + Y) * seg_mask_1  # Residual + mask
        return Z  # (B,Pnn,T,192)

    # =========================================================
    # [D] Head & Gate / 결합
    # =========================================================
    def _predict_delta_u(
        self,
        Z_tcn: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """잔차 제어 ΔU 산출: Head(초안) + softplus 게이트 스케일.

        ΔU = softplus(MLP_g(Z_s)) ⊙ tanh(Head(Z_tcn))

        Args:
            Z_tcn (torch.Tensor): (B, Pnn, T, 192)  TCN 출력
            Z_s (torch.Tensor): (B, Pnn, T, 192)  Stem 출력(게이트 입력)
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 3)  (무효 시점은 0)
        """
        delta_u_raw = self.head(Z_tcn)  # (B,Pnn,T,3)
        gate_scale = F.softplus(self.gate_mlp(Z_tcn.detach()))  # (B,Pnn,T,3)
        delta_u = gate_scale * torch.tanh(delta_u_raw)  # (B,Pnn,T,3)
        delta_u = delta_u * seg_mask_1  # 무효 시점 보정 0
        return delta_u

    def _build_per_agent_limits(
        self,
        near_class_one_hot: torch.Tensor,
        # (B,Pnn,3) 0:vehicle(CAR),1:ped,2:bicycle
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """클래스별 스칼라 제약치를 (B,Pnn) 텐서로 확장.

        Args:
            near_class_one_hot: (B, Pnn, 3)
             one‑hot (0: vehicle, 1: pedestrian, 2: bicycle)

        Returns:
            Dict[str, Tensor]:
            v_max/a_max/alpha_max/a_lat_max/R_min/omega_abs_max/a_x_max/a_y_max, is_nonholonomic
                * 모두 (B,Pnn) 모양
        """
        # 순서 주의: [vehicle(CAR), pedestrian, bicycle]
        car: DynamicLimits = self.constraints[ActorClass.CAR]
        ped: DynamicLimits = self.constraints[ActorClass.PEDESTRIAN]
        bic: DynamicLimits = self.constraints[ActorClass.BICYCLE]

        # 클래스별 상수 → 길이 3 텐서
        def cvec(getattr_name: str) -> torch.Tensor:
            vals = torch.tensor([
                getattr(car, getattr_name),
                getattr(ped, getattr_name),
                getattr(bic, getattr_name)
            ],
                                device=device,
                                dtype=dtype)  # (3,)
            # (B,Pnn,3) @ (3,) -> (B,Pnn)
            return (near_class_one_hot * vals).sum(dim=-1)

        v_max_bp = cvec("v_max_mps")
        a_max_bp = cvec("a_max_mps2")
        alpha_max_bp = cvec("alpha_max_radps2")
        a_lat_max_bp = cvec("a_lat_max_mps2")
        R_min_bp = cvec("R_min_m")
        omega_abs_bp = cvec("omega_max_abs_radps")

        # 마찰원 반경: 종/횡 가속 한계
        a_x_max_bp = a_max_bp
        a_y_max_bp = a_lat_max_bp

        # 보행자는 S0(비홀로노믹) 비활성, 자전거/차는 활성
        # near_class_one_hot[..., 1] 이 pedestrian
        is_ped = near_class_one_hot[..., 1] > 0.5  # (B,Pnn) bool
        is_nonholonomic = ~is_ped

        return dict(
            v_max=v_max_bp,  # (B,Pnn)
            a_max=a_max_bp,  # (B,Pnn)
            alpha_max=alpha_max_bp,  # (B,Pnn)
            a_lat_max=a_lat_max_bp,  # (B,Pnn)
            R_min=R_min_bp,  # (B,Pnn)
            omega_abs_max=omega_abs_bp,  # (B,Pnn)
            a_x_max=a_x_max_bp,  # (B,Pnn)
            a_y_max=a_y_max_bp,  # (B,Pnn)
            is_nonholonomic=is_nonholonomic,  # (B,Pnn) bool
        )

    # ----------------------------
    # [MOD] (S3) 옆가속/최소R/절대|w| (중점 속도 사용)
    # ----------------------------

    # ----------------------------
    # [NEW] 입력 분해 및 버퍼 초기화
    # ----------------------------
    def _split_controls(
        self,
        unnorm_cur_future_seg_body_control: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """제어 텐서를 성분별로 분리.

        Args:
            unnorm_cur_future_seg_body_control (torch.Tensor): (B, Pnn, T, 3) [vxb, vyb, w]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (vxb_raw, vyb_raw, omega_raw), 모두 (B,Pnn,T)
        """
        vx_b_raw = unnorm_cur_future_seg_body_control[..., 0]
        vy_b_raw = unnorm_cur_future_seg_body_control[..., 1]
        omega_raw = unnorm_cur_future_seg_body_control[..., 2]
        return vx_b_raw, vy_b_raw, omega_raw

    def _init_integration_buffers(
        self,
        B: int,
        Pnn: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """적분 산출/중간 결과 버퍼 생성."""
        x_next, y_next, cos_next, sin_next = _shape4(B, Pnn, T, device, dtype)
        vx_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        vy_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        omega_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        return {
            "x_next": x_next,
            "y_next": y_next,
            "cos_next": cos_next,
            "sin_next": sin_next,
            "vx_after": vx_after,
            "vy_after": vy_after,
            "omega_after": omega_after,
        }

    # === 아래 메서드들을 FeasibleProjector 클래스 "내부"에 추가하세요. ===

    # ----------------------------
    # [UTIL] 각도/벡터 보조 함수들
    # ----------------------------

    @staticmethod
    def _compute_mid_heading_from_cos_sin(
            cos_yaw_k: torch.Tensor,  # (...,)
            sin_yaw_k: torch.Tensor,  # (...,)
            half_delta_theta: torch.Tensor,  # (...,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """중점(head)의 cos/sin 계산: θ_mid = θ_k + 0.5 * Δθ.

        Args:
            cos_yaw_k (torch.Tensor): (...,) cos(θ_k)
            sin_yaw_k (torch.Tensor): (...,) sin(θ_k)
            half_delta_theta (torch.Tensor): (...,) 0.5 * Δθ

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos_mid, sin_mid)  (둘 다 (...,))
        """
        c = torch.cos(half_delta_theta)
        s = torch.sin(half_delta_theta)
        cos_mid = cos_yaw_k * c - sin_yaw_k * s
        sin_mid = sin_yaw_k * c + cos_yaw_k * s
        return cos_mid, sin_mid

    @staticmethod
    def _advance_heading_cos_sin(
        cos_yaw_k: torch.Tensor,  # (...,)
        sin_yaw_k: torch.Tensor,  # (...,)
        delta_theta: torch.Tensor,  # (...,)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """θ_{k+1} = θ_k + Δθ 를 (cos, sin)로 직접 업데이트하고 정규화.

        Args:
            cos_yaw_k (torch.Tensor): (...,) cos(θ_k)
            sin_yaw_k (torch.Tensor): (...,) sin(θ_k)
            delta_theta (torch.Tensor): (...,) Δθ
            eps (float): 수치 안정 epsilon

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos_{k+1}, sin_{k+1})
        """
        c = torch.cos(delta_theta)
        s = torch.sin(delta_theta)
        cos_next = cos_yaw_k * c - sin_yaw_k * s
        sin_next = sin_yaw_k * c + cos_yaw_k * s
        norm = torch.sqrt(cos_next * cos_next + sin_next * sin_next + eps)
        return cos_next / norm, sin_next / norm

    # =========================================================
    # [S0~S4] 제약 적용: **STE 버전** (forward=hard, backward=surrogate)
    # =========================================================
    def _apply_S0_nonholonomic_ste(
        self,
        vx_b: torch.Tensor,  # (B,Pnn)
        vy_b: torch.Tensor,  # (B,Pnn)
        slip_epsilon: float,
        eta: float,  # 0.07
        eps: float,
        is_nonholonomic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S0: v_y hard-clip(+STE). 보행자 제외."""
        limit = torch.full_like(vy_b, float(slip_epsilon))  # (B,Pnn)
        vy_new = self._ste_scalar_clip(vy_b, limit, eta, eps)  # (B,Pnn)

        # ★ 추가: 시간축이 있을 때 에이전트 마스크를 시간축으로 확장
        if is_nonholonomic.dim() == vy_b.dim() - 1:
            # is_nonholonomic: (B,Pnn) -> (B,Pnn,1) -> (B,Pnn,T)
            is_nonholonomic = is_nonholonomic.unsqueeze(-1).expand_as(vy_b)
        vy_out = torch.where(is_nonholonomic, vy_new, vy_b)
        return vx_b, vy_out

    def _apply_S1_speed_limit_ste(
            self, vx_b: torch.Tensor, vy_b: torch.Tensor, v_max: torch.Tensor,
            eta: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """라디얼(벡터) STE-clip."""
        v_max = v_max.to(dtype=vx_b.dtype, device=vx_b.device)
        # 추가하자: 시간축(T)이 있는 경우 v_max 를 (B,Pnn,1) 로 확장해 T에 브로드캐스트
        if v_max.dim() == vx_b.dim() - 1:
            v_max = v_max.unsqueeze(-1)  # (B,Pnn,1)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        s_hard = torch.clamp(v_max / speed.clamp_min(eps), max=1.0)  # (B,Pnn)
        vx_h, vy_h = s_hard * vx_b, s_hard * vy_b  # (B,Pnn)
        r = speed / (v_max + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        vx_sur, vy_sur = w * vx_b + (1 - w) * vx_b.detach(), w * vy_b + (
            1 - w) * vy_b.detach()
        vx_out = vx_sur + (vx_h - vx_sur).detach()
        vy_out = vy_sur + (vy_h - vy_sur).detach()
        return vx_out, vy_out

    def _apply_S2_accel_alpha_limits_ste(
            self, vx_b_prev: torch.Tensor, vy_b_prev: torch.Tensor,
            omega_prev: torch.Tensor, vx_b: torch.Tensor, vy_b: torch.Tensor,
            omega: torch.Tensor, a_max: torch.Tensor, alpha_max: torch.Tensor,
            dt: float, eta: float,
            eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 선형 속도 증분(벡터)
        dv = torch.stack([vx_b - vx_b_prev, vy_b - vy_b_prev],
                         dim=-1)  # (B,Pnn,2)
        dv_limit = (a_max * dt).to(dtype=dv.dtype, device=dv.device)  # (B,Pnn)
        # dv_ste: (B,Pnn,2)
        dv_ste = self._ste_increment_vec(dv, dv_limit, eta, eps)
        vx_b_new = vx_b_prev + dv_ste[..., 0]  # (B,Pnn)
        vy_b_new = vy_b_prev + dv_ste[..., 1]  # (B,Pnn)
        # 각속도 증분(스칼라)
        dw = omega - omega_prev  # (B,Pnn)
        dw_limit = (alpha_max * dt).to(dtype=dw.dtype,
                                       device=dw.device)  # (B,Pnn)
        dw_ste = self._ste_scalar_clip(dw, dw_limit, eta, eps)  # (B,Pnn)
        w_new = omega_prev + dw_ste
        return vx_b_new, vy_b_new, w_new

    # [추가 요망] (S3: 속도-연동 각속도 한계 — w clip, no slip angle)
    def _apply_S3_omega_clip_ste(
        self,
        vx_b: torch.Tensor,  # (B,Pnn)
        vy_b: torch.Tensor,  # (B,Pnn)
        omega: torch.Tensor,  # (B,Pnn)
        a_lat_max: torch.Tensor,  # (B,Pnn)
        R_min: torch.Tensor,  # (B,Pnn)
        omega_abs_max: torch.Tensor,  # (B,Pnn)
        is_nonholonomic: torch.Tensor,  # (B,Pnn)  True: 차/자전거, False: 보행자
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """(S3) β/Δβ 없이, 속도-연동 w 한계로 직접 clip.

        w_allow (차/자전거) = min( a_lat_max/|v|, |v|/R_min, w_abs_max )
        w_allow (보행자)     = min( a_lat_max/|v|, w_abs_max )   # 제자리 회전 허용
        """
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        if a_lat_max.dim() == speed.dim() - 1:
            a_lat_max = a_lat_max.unsqueeze(-1)  # (B,Pnn,1)
            R_min = R_min.unsqueeze(-1)  # (B,Pnn,1)
            omega_abs_max = omega_abs_max.unsqueeze(-1)  # (B,Pnn,1)
        allow_lat = a_lat_max / (speed + eps)  # (B,Pnn)
        allow_R = speed / (R_min + eps)  # (B,Pnn)
        allow_abs = omega_abs_max

        # 비홀로노믹이면 R_min 항 포함, 보행자는 제외
        allow_nonh = torch.minimum(torch.minimum(allow_lat, allow_R), allow_abs)
        allow_holo = torch.minimum(allow_lat, allow_abs)

        # 추가하자: is_nonholonomic 을 시간축으로 확장
        if is_nonholonomic.dim() == allow_nonh.dim() - 1:
            is_nonholonomic = is_nonholonomic.unsqueeze(-1).expand_as(
                allow_nonh)

        allow = torch.where(is_nonholonomic, allow_nonh, allow_holo)  # (B,Pnn)

        # forward: hard clamp, backward: band-weighted surrogate (allow는 detach)
        # omega_new: (B,Pnn)
        omega_new = self._ste_yawrate_clip(omega, allow, eta=eta, eps=eps)
        return omega_new

    # [추가 요망] (S4: β 미사용, Δv와 w를 동일 스케일 s로 동시 축소)
    def _apply_S4_friction_circle_ste(
        self,
        vx_b_prev: torch.Tensor,
        vy_b_prev: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        a_x_max: torch.Tensor,
        a_y_max: torch.Tensor,  # (B,Pnn)
        dt: float,
        eta: float,
        eps: float,
        use_ax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(S4) (a_x/a_x,max)^2+(a_y/a_y,max)^2≤1 를 만족하도록
        Δv와 w를 동일 스케일 s∈(0,1]로 동시 축소.
           a_x ≈ Δv/dt,  a_y ≈ v_mid * w_k
        """
        # [추가하세요!] 바디 프레임 기준 중간 속도/증분 (부호 보존)
        vxb_mid = 0.5 * (vx_b_prev + vx_b_k)  # (B,Pnn)
        vyb_mid = 0.5 * (vy_b_prev + vy_b_k)  # (B,Pnn)
        v_mid = torch.sqrt(vxb_mid * vxb_mid + vyb_mid * vyb_mid +
                           eps)  # (B,Pnn)

        dvx = vx_b_k - vx_b_prev  # (B,Pnn)
        dvy = vy_b_k - vy_b_prev  # (B,Pnn)
        if not use_ax:
            dvx = torch.zeros_like(dvx)  # k=0 등에서는 종가속 항 비활성
        ax_est = dvx / dt  # (B,Pnn)  ← 부호 보존 종가속도
        ay_est = v_mid * omega_k  # (B,Pnn)  ← 곡률 유도 횡가속도
        # ay_est = ay_est + (dvy / dt)
        s = self._ste_friction_scale(ax_est,
                                     ay_est,
                                     a_x_max,
                                     a_y_max,
                                     eta=eta,
                                     eps=eps)  # (B,Pnn)
        # [추가하세요!] Δv 벡터 자체와 ω를 같은 비율로 축소 → 방향/부호 그대로 유지
        dv = torch.stack([dvx, dvy], dim=-1)  # (B,Pnn,2)
        dv_sc = s.unsqueeze(-1) * dv  # (B,Pnn,2)
        vx_b_new = vx_b_prev + dv_sc[..., 0]  # (B,Pnn)
        vy_b_new = vy_b_prev + dv_sc[..., 1]  # (B,Pnn)
        omega_new = s * omega_k  # (B,Pnn)
        return vx_b_new, vy_b_new, omega_new

    # ----------------------------
    # [NEW] 한 스텝: 제약 S0~S4 적용(모두 STE 버전 호출)
    # ----------------------------
    def _apply_constraints_step(
        self,
        vx_b_prev: torch.Tensor,  # (B,Pnn)
        vy_b_prev: torch.Tensor,  # (B,Pnn)
        omega_prev: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,  # (B,Pnn)
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        hp: _ConstraintHParams,
        key_to_limit_bp: Dict[str, torch.Tensor],
        slip_epsilon: float = 0.20,
        apply_S2: bool = True,
        apply_S4_ax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ key_to_limit_bp
        Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn)

        v_max/a_max/alpha_max/a_lat_max/R_min
        /omega_abs_max/a_x_max/a_y_max, is_nonholonomic

        self.constraints_h_params = _ConstraintHParams(
            dt=0.1,
            eps=1e-6,
             필요: η 초기값(S0~S4)
            eta_slip=0.07,
            eta_speed=0.05,
            eta_inc=0.10,
            eta_yaw=0.05,
            eta_fric=0.05,
        )
        """
        # (S0)
        vx_b_k, vy_b_k = self._apply_S0_nonholonomic_ste(
            vx_b_k,  # (B,Pnn)
            vy_b_k,  # (B,Pnn)
            slip_epsilon,
            hp.eta_slip,  # 0.07
            hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],  # (B,Pnn)
        )
        # (S1)
        vx_b_k, vy_b_k = self._apply_S1_speed_limit_ste(
            vx_b_k, vy_b_k, key_to_limit_bp["v_max"], hp.eta_speed, hp.eps)
        # # (S2)
        if apply_S2:
            vx_b_k, vy_b_k, omega_k = self._apply_S2_accel_alpha_limits_ste(
                vx_b_prev, vy_b_prev, omega_prev, vx_b_k, vy_b_k, omega_k,
                key_to_limit_bp["a_max"], key_to_limit_bp["alpha_max"], hp.dt,
                hp.eta_inc, hp.eps)
        # (S3)
        omega_k = self._apply_S3_omega_clip_ste(
            vx_b=vx_b_k,
            vy_b=vy_b_k,
            omega=omega_k,
            a_lat_max=key_to_limit_bp["a_lat_max"],
            R_min=key_to_limit_bp["R_min"],
            omega_abs_max=key_to_limit_bp["omega_abs_max"],
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
            eta=hp.eta_yaw,
            eps=hp.eps)
        # (S4)
        # vx_b_k, vy_b_k, omega_k = self._apply_S4_friction_circle_ste(
        #     vx_b_prev,
        #     vy_b_prev,
        #     vx_b_k,
        #     vy_b_k,
        #     omega_k,
        #     key_to_limit_bp["a_x_max"],
        #     key_to_limit_bp["a_y_max"],
        #     hp.dt,
        #     hp.eta_fric,
        #     hp.eps,
        #     use_ax=apply_S4_ax)
        return vx_b_k, vy_b_k, omega_k

    # ----------------------------
    # [NEW] 한 스텝: 중점 적분 (변경 없음)
    # ----------------------------
    def _integrate_midpoint_step(
        self,
        x_k: torch.Tensor,  # (B,Pnn)
        y_k: torch.Tensor,  # (B,Pnn)
        cos_yaw_k: torch.Tensor,  # (B,Pnn)
        sin_yaw_k: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,  # (B,Pnn)
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        hp: _ConstraintHParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_dtheta = 0.5 * omega_k * hp.dt
        # cos_mid, sin_mid: (B,Pnn)
        cos_mid, sin_mid = self._compute_mid_heading_from_cos_sin(
            cos_yaw_k, sin_yaw_k, half_dtheta)
        vwx_mid = cos_mid * vx_b_k - sin_mid * vy_b_k
        vwy_mid = sin_mid * vx_b_k + cos_mid * vy_b_k
        x_k1 = x_k + vwx_mid * hp.dt  # (B,Pnn)
        y_k1 = y_k + vwy_mid * hp.dt  # (B,Pnn)
        dtheta = omega_k * hp.dt  # (B,Pnn)
        cos_yaw_k1, sin_yaw_k1 = self._advance_heading_cos_sin(cos_yaw_k,
                                                               sin_yaw_k,
                                                               dtheta,
                                                               eps=hp.eps)
        return x_k1, y_k1, cos_yaw_k1, sin_yaw_k1

    # ----------------------------
    # [NEW] 결과 조립(+마스크/차이)
    # ----------------------------
    def _assemble_outputs(
            self,
            key_to_all_states: Dict[str, torch.Tensor],
            vx_b_raw: torch.Tensor,  # (B,Pnn,T)
            vy_b_raw: torch.Tensor,  # (B,Pnn,T)
            omega_raw: torch.Tensor,  # (B,Pnn,T)
            near_cur_future_valid: torch.Tensor,  # (B,Pnn,1+T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """시점별 유효성(노드/구간 마스크)을 반영해 출력 텐서를 구성합니다.

        Args:
            key_to_all_states: 적분 및 제약 후 상태/제어 모음.
                - x_next / y_next / cos_next / sin_next: (B,Pnn,T)  ← 노드(t=1..T)
                - vx_after / vy_after / omega_after:     (B,Pnn,T)  ← 구간 평균 제어
            vx_b_raw, vy_b_raw, omega_raw: (B,Pnn,T)  필터 입력 제어(비정규화)
            near_cur_future_valid: (B,Pnn,1+T)  노드 유효성. t=0(현재) 포함.

        Returns:
            Tuple:
                - unnorm_integrated_trajectory: (B,Pnn,future_len,4)  # 노드 기반 → mask_node[...,1:]로 마스킹
                - unnorm_control_constraint_diff: (B,Pnn,future_len,3)  # 구간 기반 → mask_interval로 마스킹
        """
        # --- 상태 조립(노드 기반) ---
        x_next, y_next = key_to_all_states["x_next"], key_to_all_states[
            "y_next"]
        cos_next, sin_next = key_to_all_states["cos_next"], key_to_all_states[
            "sin_next"]
        unnorm_integrated_trajectory = torch.stack(
            [x_next, y_next, cos_next, sin_next], dim=-1)  # (B,Pnn,T,4)

        # --- 제어 차이(구간 기반, detach로 필터 역전파 차단) ---
        unnorm_control_constraint_diff = torch.stack(
            [
                key_to_all_states["vx_after"].detach() - vx_b_raw,
                key_to_all_states["vy_after"].detach() - vy_b_raw,
                key_to_all_states["omega_after"].detach() - omega_raw,
            ],
            dim=-1,  # (B,Pnn,T,3)
        )

        # --- 시점별 마스크 구성 ---
        mask_node = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        mask_state = mask_node[..., 1:]  # (B,Pnn,T)    # 노드 t=1..T
        mask_interval = mask_node[..., :-1] & mask_node[
            ..., 1:]  # (B,Pnn,T)    # 구간 AND

        # --- 마스킹 적용 ---
        unnorm_integrated_trajectory = unnorm_integrated_trajectory.masked_fill(
            ~mask_state.unsqueeze(-1), 0.0)
        unnorm_control_constraint_diff = unnorm_control_constraint_diff.masked_fill(
            ~mask_interval.unsqueeze(-1), 0.0)
        return unnorm_integrated_trajectory, unnorm_control_constraint_diff

    # ----------------------------
    # [UTIL] 속도/증분 soft & hard 클립
    # ----------------------------

    # ==========================================
    # 4) S2 (증분 제한) soft clip 수정
    # ==========================================

    # ============================
    # [REFACTORED] 본체: Filter + Integrate
    # ============================
    def _filter_and_integrate_sequential(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
            unnorm_cur_future_seg_body_control: torch.
        Tensor,  # (B, Pnn, future_len, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # self._assert_cur_future_valid_mask(near_cur_future_valid,
        #                                    context="filter_and_integrate")
        B, Pnn, future_len, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype
        if future_len == 0:
            raise ValueError("future_len=0: 적분할 미래 세그먼트가 없습니다.")
            # 맨 앞에 추가
        """ key_to_limit_bp
        Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn) 

        v_max/a_max/alpha_max/a_lat_max/R_min
        /omega_abs_max/a_x_max/a_y_max, is_nonholonomic
        """
        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot, device=device, dtype=dtype)
        # (B,Pnn,future_len)
        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(
            unnorm_cur_future_seg_body_control)
        """ key_to_all_states
        x_next / y_next / cos_next / sin_next: (B,Pnn,future_len)
        vx_after / vy_after / omega_after: (B,Pnn,future_len)
        """
        key_to_all_states: Dict[str,
                                torch.Tensor] = self._init_integration_buffers(
                                    B, Pnn, future_len, dtype, device)

        x_k = unnorm_near_current_state[..., 0]  # (B,Pnn)
        y_k = unnorm_near_current_state[..., 1]  # (B,Pnn)
        cos_yaw_k = unnorm_near_current_state[..., 2]  # (B,Pnn)
        sin_yaw_k = unnorm_near_current_state[..., 3]  # (B,Pnn)
        vx_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)  # (B,Pnn)
        vy_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)  # (B,Pnn)
        omega_prev = torch.zeros((B, Pnn), device=device,
                                 dtype=dtype)  # (B,Pnn)

        for k in range(future_len):
            apply_S2_k = (k > 0)
            apply_S4_ax_k = (k > 0)

            # (B,Pnn)
            vx_b_k, vy_b_k, yaw_rate_k = vx_b_raw[..., k], vy_b_raw[
                ..., k], omega_raw[..., k]
            """ key_to_limit_bp
            Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn) 

            v_max/a_max/alpha_max/a_lat_max/R_min
            /omega_abs_max/a_x_max/a_y_max, is_nonholonomic
            """
            # [STE] S0~S4
            if self.use_feasible_filter:
                vx_b_k, vy_b_k, yaw_rate_k = self._apply_constraints_step(
                    vx_b_prev,  # (B,Pnn)
                    vy_b_prev,  # (B,Pnn)
                    omega_prev,  # (B,Pnn)
                    vx_b_k,  # (B,Pnn)
                    vy_b_k,  # (B,Pnn)
                    yaw_rate_k,  # (B,Pnn)
                    hp=self.constraints_h_params,  # _ConstraintHParams
                    key_to_limit_bp=
                    key_to_limit_bp,  # Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn)
                    slip_epsilon=0.1,
                    apply_S2=apply_S2_k,
                    apply_S4_ax=apply_S4_ax_k,
                )
            x_k1, y_k1, cos_k1, sin_k1 = self._integrate_midpoint_step(
                x_k, y_k, cos_yaw_k, sin_yaw_k, vx_b_k, vy_b_k, yaw_rate_k,
                self.constraints_h_params)

            key_to_all_states["x_next"][..., k] = x_k1  # (B,Pnn)
            key_to_all_states["y_next"][..., k] = y_k1  # (B,Pnn)
            key_to_all_states["cos_next"][..., k] = cos_k1  # (B,Pnn)
            key_to_all_states["sin_next"][..., k] = sin_k1  # (B,Pnn)
            key_to_all_states["vx_after"][..., k] = vx_b_k  # (B,Pnn)
            key_to_all_states["vy_after"][..., k] = vy_b_k  # (B,Pnn)
            key_to_all_states["omega_after"][..., k] = yaw_rate_k  # (B,Pnn)

            x_k, y_k, cos_yaw_k, sin_yaw_k = x_k1, y_k1, cos_k1, sin_k1
            vx_b_prev, vy_b_prev, omega_prev = vx_b_k, vy_b_k, yaw_rate_k
        """ key_to_all_states
        x_next / y_next / cos_next / sin_next: (B,Pnn,future_len)
        vx_after / vy_after / omega_after: (B,Pnn,future_len)

        vx_b_raw: (B,Pnn,future_len)
        vy_b_raw: (B,Pnn,future_len)
        omega_raw: (B,Pnn,future_len)

        near_cur_future_valid: (B,Pnn,1+future_len) bool
        """
        return self._assemble_outputs(key_to_all_states, vx_b_raw, vy_b_raw,
                                      omega_raw, near_cur_future_valid)

    # 지우개: 아래 기존 filter_and_integrate 본문 전체를 대체합니다.
    # def filter_and_integrate(...):
    #     (기존 step-by-step for 루프 구현)
    #     ...

    # ----------------------------
    # [MOD] 경로 선택용 래퍼
    # ----------------------------
    def filter_and_integrate(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
            unnorm_cur_future_seg_body_control: torch.
        Tensor,  # (B, Pnn, future_len, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter + Integrate 최상위 래퍼.

        - 공통으로 유효 마스크 형태를 한 번 확인하고,
        - 설정(self.use_batch_integration)에 따라
            * False: 기존 step-by-step + S2 버전 사용
            * True : 시간축 완전 배치 + S2 미사용 버전 사용
        """

        self._assert_cur_future_valid_mask(
            near_cur_future_valid,
            context="filter_and_integrate",
        )

        if not self.use_batch_integration:
            # 기존 방식 유지 (S2 포함, for 루프)
            return self._filter_and_integrate_sequential(
                unnorm_near_current_state=unnorm_near_current_state,
                near_cur_future_valid=near_cur_future_valid,
                unnorm_cur_future_seg_body_control=
                unnorm_cur_future_seg_body_control,
                near_class_one_hot=near_class_one_hot,
            )
        # 추가하자: 시간축 완전 배치 버전 (S2 미사용)
        unnorm_integrated_trajectory, unnorm_control_constraint_diff = self._filter_and_integrate_batch(
            unnorm_near_current_state=unnorm_near_current_state,
            near_cur_future_valid=near_cur_future_valid,
            unnorm_cur_future_seg_body_control=
            unnorm_cur_future_seg_body_control,
            near_class_one_hot=near_class_one_hot,
        )
        return unnorm_integrated_trajectory, unnorm_control_constraint_diff

    # ----------------------------
    # [NEW PATH] 시간축 완전 배치 버전 (S2 미사용)
    # ----------------------------
    def _filter_and_integrate_batch(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
            unnorm_cur_future_seg_body_control: torch.
        Tensor,  # (B, Pnn, future_len, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """시간축 전체를 한 번에 처리하는 배치 버전.

        - S2(가속/각가속 증분 제한)는 사용하지 않는다.
        - S0/S1/S3만 vx,vy,omega 시퀀스에 배치로 적용한다.
        - yaw 및 위치는 cumsum 기반 중점 적분으로 계산한다.
        """
        B, Pnn, future_len, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype

        if future_len == 0:
            raise ValueError("future_len=0: 적분할 미래 세그먼트가 없습니다.")

        # per-agent 제한값 (v_max, a_lat_max, R_min, ...)
        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot,
            device=device,
            dtype=dtype,
        )

        # (B,Pnn,future_len)
        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(
            unnorm_cur_future_seg_body_control)

        # 시간축 전체에 S0/S1/S3 배치 적용 (S2는 미사용)
        """ 3개 모두 (B,Pnn,future_len) 반환"""
        vx_b_after, vy_b_after, omega_after = self._apply_constraints_batch(
            vx_b_raw=vx_b_raw,  # (B,Pnn,T)
            vy_b_raw=vy_b_raw,  # (B,Pnn,T)
            omega_raw=omega_raw,  # (B,Pnn,T)
            key_to_limit_bp=key_to_limit_bp,
            hp=self.constraints_h_params,
            slip_epsilon=0.1,
        )

        # 중점 적분을 시간축 전체에 대해 배치로 수행
        key_to_all_states = self._integrate_midpoint_batch(
            unnorm_near_current_state=unnorm_near_current_state,  # (B,Pnn,4)
            vx_b_seq=vx_b_after,  # (B,Pnn,T)
            vy_b_seq=vy_b_after,  # (B,Pnn,T)
            omega_seq=omega_after,  # (B,Pnn,T)
            hp=self.constraints_h_params,
        )

        return self._assemble_outputs(
            key_to_all_states=key_to_all_states,
            vx_b_raw=vx_b_raw,
            vy_b_raw=vy_b_raw,
            omega_raw=omega_raw,
            near_cur_future_valid=near_cur_future_valid,
        )

    # ================================================================
    # [REFACTOR] Savitzky–Golay 유틸들 (모두 torch-only, 미분 가능)
    # ================================================================

    # feasible.py 내 FeasibleProjector 클래스 안에 추가
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
                f"[{context}] near_cur_future_valid가 행 단위 단조 감소(True*False*) 가정에 위배됩니다. "
                f"0→1 전이가 감지되었습니다. 오류 row 수={int(bad_idx.numel())}, "
                f"예시 (b,p)={list(zip(b_list, p_list))}. "
                f"내부 구멍(1→0→1)이나 선행 무효 후 유효(0→1)는 허용되지 않습니다.")

    # 지우개: 예전 `_compute_world_linear_velocity_via_sg` 구현은
    #        x, y를 각각 단일 채널 SG에 넣어 두 번 호출하던 코드입니다.
    #        해당 본문 전체를 지우고 아래 새 구현으로 교체하세요.

    def _compute_world_linear_velocity_via_sg(
        self,
        x: torch.Tensor,  # (B, Pnn, point_len)
        y: torch.Tensor,  # (B, Pnn, point_len)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_len_xy: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x,y 위치 값이 시간에 따라 어떻게 움직이는지 부드러운 속도로 바꿔 주는 함수.

        - 단순 차분이 아니라, 주변 여러 점을 한꺼번에 보고 "부드러운 기울기"를 추정한다.
        - x와 y 두 축을 한 번에 처리해서, 계산량을 줄이도록 설계했다.

        Args:
            x: (B, Pnn, point_len)
                각 객체의 시간별 x 좌표.
            y: (B, Pnn, point_len)
                각 객체의 시간별 y 좌표.
            points_valid: (B, Pnn, point_len) bool
                해당 시점에 위치가 실제로 존재하는지(True/False) 표시.
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                Savitzky–Golay 다항식 차수.
            max_window_len_xy: int
                x,y에 사용할 최대 창 길이.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - v_x^w: (B, Pnn, point_len)  x축 방향 세계 속도
                - v_y^w: (B, Pnn, point_len)  y축 방향 세계 속도
        """
        # : x,y를 한 번에 채널로 묶어서 SG 한 번만 호출
        positions_stack = torch.stack(
            [x, y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)

        velocities_stack = self._sg_derivative_multi_for_points(
            sequences_points=positions_stack,  # (B,Pnn,point_len,2)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_xy,
        )  # (B, Pnn, point_len, 2)

        v_x_world = velocities_stack[..., 0]  # (B, Pnn, point_len)
        v_y_world = velocities_stack[..., 1]  # (B, Pnn, point_len)

        return v_x_world, v_y_world

    # ================================================================
    # 본 기능: 위치→세계속도, yaw→요레이트, 그리고 body 회전
    # ================================================================
    def savgol_filter_for_control(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,
        # (B,Pnn,1+future_len,4) = [x, y, cos, sin]
        unnorm_near_past_xyyaw: Optional[torch.Tensor],
        # (B,Pnn,past_len,4) or None
        target_past_cur_future_valid: torch.Tensor,
        # (B,Pnn,past_len+1+future_len) bool
        *,
        dt: float = 0.1,
        polyorder: int = 2,
        max_window_len_xy: int = 11,
        max_window_len_yaw: int = 7,
    ) -> torch.Tensor:
        """Savitzky–Golay(마스크 인지)로 **세계속도/각속도**를 추정하여 반환.

        Returns:
            unnorm_points_world_control:
                - past가 없으면: (B,Pnn,1+future_len,3)  = [v_x^w, v_y^w, w]
                - past가 있으면: (B,Pnn,past_len+1+future_len,3)
        """
        # 1) 포인트/마스크 준비 및 검증
        point_len_inputs: PointLenInputs = self._prepare_points_and_masks(
            unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
            # (B,Pnn,1+future_len,4)
            unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
            # (B,Pnn,past_len,4) or None
            target_past_cur_future_valid=target_past_cur_future_valid,
            # (B,Pnn,past_len+1+future_len) bool
        )
        unnorm_points_xyyaw = point_len_inputs.unnorm_points_xyyaw  # (B,Pnn,point_len,4)
        points_valid = point_len_inputs.points_valid  # (B,Pnn,point_len) bool
        B, Pnn, point_len, _ = unnorm_points_xyyaw.shape

        # 2) 분해
        x = unnorm_points_xyyaw[..., 0]  # (B,Pnn,point_len)
        y = unnorm_points_xyyaw[..., 1]  # (B,Pnn,point_len)
        cos_y = unnorm_points_xyyaw[..., 2]  # (B,Pnn,point_len)
        sin_y = unnorm_points_xyyaw[..., 3]  # (B,Pnn,point_len)

        # 3) SG-미분 (x/y → v_x^w, v_y^w) & (cos/sin → w)
        # v_x: (B,Pnn,point_len)
        # v_y: (B,Pnn,point_len)
        v_x, v_y = self._compute_world_linear_velocity_via_sg(
            x=x,  # (B,Pnn,point_len)
            y=y,  # (B,Pnn,point_len)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_len_xy=max_window_len_xy,
        )
        # yaw_rate: (B,Pnn,point_len)
        yaw_rate = self._compute_yaw_rate_via_sg(
            cos_y=cos_y,  # (B,Pnn,point_len)
            sin_y=sin_y,  # (B,Pnn,point_len)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_len_yaw=max_window_len_yaw,
        )

        # 4) 마스킹·스택 후 반환
        unnorm_points_world_control = self._mask_and_stack_world_controls(
            v_x=v_x, v_y=v_y, yaw_rate=yaw_rate,
            points_valid=points_valid)  # (B,Pnn,point_len,3)
        return unnorm_points_world_control

    def _compute_yaw_rate_via_sg(
        self,
        cos_y: torch.Tensor,  # (B, Pnn, point_len)
        sin_y: torch.Tensor,  # (B, Pnn, point_len)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_len_yaw: int,
    ) -> torch.Tensor:
        """cos(ψ), sin(ψ) 시퀀스로부터 각속도 ψ̇를 부드럽게 추정하는 함수.

        - 먼저 cos, sin을 함께 SG에 넣어 두 값의 시간 변화량을 구한다.
        - 그 뒤, 회전 운동의 관계식을 이용해 ψ̇를 계산한다.
        - cos, sin 을 다시 단위원으로 정리해 수치적인 오류가 쌓이지 않도록 한다.

        Args:
            cos_y: (B, Pnn, point_len)
                각 시점의 cos(ψ) 값.
            sin_y: (B, Pnn, point_len)
                각 시점의 sin(ψ) 값.
            points_valid: (B, Pnn, point_len) bool
                해당 시점에 자세 정보가 실제로 존재하는지(True/False).
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                Savitzky–Golay 다항식 차수.
            max_window_len_yaw: int
                yaw에 사용할 최대 창 길이.

        Returns:
            torch.Tensor: (B, Pnn, point_len)
                각 시점의 부드럽게 추정된 yaw_rate(ψ̇).
        """
        batch_size, num_neighbors, point_len = cos_y.shape  # (B,Pnn,T)

        # : cos,sin을 하나의 2채널 시퀀스로 묶어서 SG 한 번만 호출
        yaw_unit_stack = torch.stack(
            [cos_y, sin_y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)

        yaw_unit_derivative = self._sg_derivative_multi_for_points(
            sequences_points=yaw_unit_stack,  # (B,Pnn,point_len,2)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_yaw,
        )  # (B, Pnn, point_len, 2)

        dcos = yaw_unit_derivative[..., 0]  # (B, Pnn, point_len)
        dsin = yaw_unit_derivative[..., 1]  # (B, Pnn, point_len)

        # : cos,sin을 다시 단위원으로 투영해서 수치 오차를 줄인다.
        cos_sin_stack = torch.stack(
            [cos_y, sin_y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)
        norm = torch.linalg.norm(
            cos_sin_stack,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)  # (B, Pnn, point_len, 1)

        cos_unit = (cos_sin_stack[..., 0:1] / norm).squeeze(
            -1)  # (B,Pnn,point_len)
        sin_unit = (cos_sin_stack[..., 1:2] / norm).squeeze(
            -1)  # (B,Pnn,point_len)

        # : ψ̇ = (cos * d(sin) - sin * d(cos)) / (cos² + sin²)
        denom = (cos_unit * cos_unit + sin_unit * sin_unit).clamp_min(
            1e-6)  # (B,Pnn,T)
        yaw_rate = (cos_unit * dsin - sin_unit * dcos) / denom  # (B,Pnn,T)

        # 무효 시점은 0으로 정리
        yaw_rate = torch.where(
            points_valid,
            yaw_rate,
            torch.zeros_like(yaw_rate),
        )  # (B,Pnn,T)

        return yaw_rate

    # <추가하자>
    def _infer_forward_lengths_and_validate_base(
        self,
        target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len) + future_len)
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor]:
        """forward 용 기본 길이/shape 및 최소 검증을 수행한다.

        Returns:
            B: int
            Pnn: int
            past_len: int
            future_len: int
            segment_len: int
            valid_all: (B, Pnn, 1+past_len+future_len) bool
            cur_future_valid: (B, Pnn, 1+future_len) bool
        """
        B, Pnn, total_time_len = target_past_cur_future_valid.shape
        B2, Pnn2, one_plus_future_len, _ = diffusion_trajectory.shape

        future_len = one_plus_future_len - 1  # 현재 이후 미래 segment 개수
        if future_len <= 0:
            raise ValueError("[forward] diffusion_trajectory 길이가 2 미만입니다. "
                             "최소 (현재, 미래1) 두 노드가 필요합니다.")
        if (B2, Pnn2) != (B, Pnn):
            raise ValueError(
                "[_infer_forward_lengths_and_validate_base] "
                "near_past_cur_future_valid과 diffusion_trajectory의 "
                f"batch/agent 축이 다릅니다: "
                f"(B,Pnn)=({B},{Pnn}), (B2,Pnn2)=({B2},{Pnn2})")
        past_len = total_time_len - (1 + future_len)

        if past_len < 0:
            raise ValueError(
                f"[forward] 마스크 길이가 너무 짧습니다: total_time_len={total_time_len}, future_len={future_len}"
            )
        # past_len 은 '과거 노드 개수' (현재 포함 전까지)
        # <추가하자> 전체 길이 일관성 체크
        if total_time_len != past_len + 1 + future_len:
            raise ValueError(
                f"[forward] target_past_cur_future_valid.shape[2]={total_time_len} "
                f"!= past_len+1+future_len={past_len + 1 + future_len}")

        if (B2, Pnn2) != (B, Pnn):
            raise ValueError(
                f"배치/에이전트 축 불일치: (B,Pnn)=({B},{Pnn}), (B2,Pnn2)=({B2},{Pnn2})")

        # seg_body_control 이 들고 있는 segment 개수 = segment_len
        _, _, segment_len, _ = seg_body_control.shape

        # 노드 기준 전체 유효 마스크 (bool)
        valid_all = target_past_cur_future_valid.to(
            torch.bool)  # (B, Pnn, past_len+1+future_len)

        # 현재~미래 구간만 따로 떼서 1*0* 패턴 검증용(cur_future_valid)
        cur_future_valid = valid_all[..., past_len:]  # (B, Pnn, 1+future_len)
        self._assert_cur_future_valid_mask(cur_future_valid,
                                           context="forward_cur_future_valid")

        return B, Pnn, past_len, future_len, segment_len, valid_all, cur_future_valid

    # <추가하자>
    def _validate_forward_past_mask_if_needed(
            self,
            past_len: int,
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            valid_all: torch.Tensor,  # (B, Pnn, past_len+1+future_len) bool
    ) -> None:
        """과거 구간을 실제로 사용할 때만 0*1* 패턴 검증."""
        if past_len > 0 and near_past_xyyaw is not None:
            past_cur_valid = valid_all[..., :past_len +
                                       1]  # (B, Pnn, 1+past_len)
            self._assert_past_cur_valid_mask(past_cur_valid,
                                             context="forward_past_cur_valid")

    # <추가하자>
    def _build_forward_points_without_past(
            self,
            future_len: int,
            segment_len: int,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거 포인트를 사용하지 않는 경우의 points/valid 구성."""
        expected_segment_len = future_len
        if segment_len != expected_segment_len:
            raise ValueError("[forward] near_past_xyyaw is None 인 경우, "
                             f"seg_body_control 의 segment_len={segment_len} 이 "
                             f"future_len={future_len} 과 다릅니다.")

        points_trajectory = diffusion_trajectory  # (B, Pnn, 1+future_len, 4)
        points_valid = cur_future_valid  # (B, Pnn, 1+future_len)
        return points_trajectory, points_valid

    # <추가하자>
    def _build_forward_points_with_past(
        self,
        past_len: int,
        future_len: int,
        segment_len: int,
        near_past_xyyaw: torch.Tensor,  # (B, Pnn, past_len, 4)
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        valid_all: torch.Tensor,  # (B, Pnn, 1+past_len+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거~현재~미래 포인트를 모두 사용하는 경우의 points/valid 구성."""
        if near_past_xyyaw.shape[2] != past_len:
            raise ValueError(
                "[forward] near_past_xyyaw 의 past_len 이 유효 마스크에서 계산한 "
                f"past_len={past_len} 과 다릅니다. "
                f"(near_past_xyyaw.shape[2]={near_past_xyyaw.shape[2]})")

        expected_segment_len = past_len + future_len
        if segment_len != expected_segment_len:
            raise ValueError(
                "[forward] near_past_xyyaw 가 있을 때 seg_body_control 의 "
                f"segment_len={segment_len} 이 "
                f"past_len+future_len={expected_segment_len} 과 다릅니다.")

        points_trajectory = torch.cat(
            [near_past_xyyaw, diffusion_trajectory],
            dim=2)  # (B, Pnn, past_len + 1 + future_len, 4)

        if points_trajectory.shape[2] != (1 + expected_segment_len):
            raise ValueError(
                "[forward] points_trajectory 길이(과거+현재+미래)가 "
                f"1+segment_len={1 + expected_segment_len} 과 일치하지 않습니다. "
                f"(실제 길이={points_trajectory.shape[2]})")

        # 네트워크 입장에서는 '사용 가능한 모든 segment' 를 보고 싶으므로
        # 전체 타임라인 마스크를 그대로 넘긴다. (0*1*0* 패턴 허용)
        points_valid = valid_all  # (B, Pnn, past_len+1+future_len)
        return points_trajectory, points_valid

    # <추가하자>
    def _prepare_forward_points_and_mask(
            self,
            target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len) + future_len)
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward 에서 사용할 포인트 궤적과 노드 유효 마스크를 준비한다.

        Args:
            target_past_cur_future_valid:
                (B, Pnn, time_len(=1+past_len) + future_len) bool
                = [과거 0..past_len-1, 현재, 미래 1..future_len] 노드 유효 마스크.
            diffusion_trajectory:
                (B, Pnn, 1+future_len, 4) = [x, y, cos, sin]
                현재~미래 구간 포인트.
            near_past_xyyaw:
                None 이면 과거 포인트를 사용하지 않는 모드.
                Tensor 이면 (B, Pnn, past_len, 4) 로 과거 포인트를 포함.
            seg_body_control:
                (B, Pnn, segment_len, 3) = [v_x^b, v_y^b, w]_seg.
                segment_len 은
                  - near_past_xyyaw is None  → future_len
                  - near_past_xyyaw not None → past_len + future_len

        Returns:
            points_trajectory:
                (B, Pnn, 1+segment_len, 4)
                - 과거 없음: diffusion_trajectory (현재~미래)
                - 과거 있음: [near_past_xyyaw, diffusion_trajectory] concat (과거~현재~미래)
            points_valid:
                (B, Pnn, 1+segment_len) bool
                - 과거 없음: 현재~미래 부분(cur_future_valid)
                - 과거 있음: 과거~현재~미래 전체(target_past_cur_future_valid)
        """
        (
            B,  # int
            Pnn,  # int
            past_len,  # int
            future_len,  # int
            segment_len,  # int
            valid_all,  # (B, Pnn, 1+past_len+future_len)
            cur_future_valid,  # (B, Pnn, 1+future_len)
        ) = self._infer_forward_lengths_and_validate_base(
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, Pnn, time_len(=1+past_len) + future_len)
            diffusion_trajectory=
            diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
        )

        # 과거 마스크(0*1*)는 실제로 과거 포인트를 사용할 때만 검증
        self._validate_forward_past_mask_if_needed(
            past_len=past_len,  # int
            near_past_xyyaw=near_past_xyyaw,  # (B, Pnn, past_len, 4) or None
            valid_all=valid_all,  # (B, Pnn, 1+past_len+future_len) bool
        )

        # 분기: 과거 포인트 사용 여부
        if near_past_xyyaw is None:
            """
            points_trajectory: (B, Pnn, 1+future_len, 4)
            points_valid: (B, Pnn, 1+future_len)
            """
            (points_trajectory, points_valid
            ) = self._build_forward_points_without_past(
                future_len=future_len,
                segment_len=segment_len,
                diffusion_trajectory=
                diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
                cur_future_valid=cur_future_valid,  # (B, Pnn, 1+future_len) bool
            )
        else:
            """
            points_trajectory : (B, Pnn, past_len + 1 + future_len, 4)
            points_valid : (B, Pnn, past_len+ 1 +future_len)
            """
            (points_trajectory,
             points_valid) = self._build_forward_points_with_past(
                 past_len=past_len,
                 future_len=future_len,
                 segment_len=segment_len,
                 near_past_xyyaw=near_past_xyyaw,  # (B, Pnn, past_len, 4)
                 diffusion_trajectory=
                 diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
                 valid_all=valid_all,  # (B, Pnn, 1+past_len+future_len)
             )

        # 최종 shape 일관성 체크
        if points_valid.shape[2] != points_trajectory.shape[2]:
            raise ValueError(
                "[forward] points_trajectory 와 points_valid 의 time 축 길이가 다릅니다. "
                f"points_trajectory.shape[2]={points_trajectory.shape[2]}, "
                f"points_valid.shape[2]={points_valid.shape[2]}.")

        return points_trajectory, points_valid

    # <추가하자>
    def _mask_and_stack_world_controls(
            self,
            v_x: torch.Tensor,  # (B,Pnn,point_len)
            v_y: torch.Tensor,  # (B,Pnn,point_len)
            yaw_rate: torch.Tensor,  # (B,Pnn,point_len)
            points_valid: torch.Tensor,  # (B,Pnn,point_len)  bool
    ) -> torch.Tensor:
        """무효 시점 0 마스킹 후 [v_x^w, v_y^w, w] 스택."""
        v_x = torch.where(points_valid, v_x, torch.zeros_like(v_x))
        v_y = torch.where(points_valid, v_y, torch.zeros_like(v_y))
        yaw_rate = torch.where(points_valid, yaw_rate,
                               torch.zeros_like(yaw_rate))
        return torch.stack([v_x, v_y, yaw_rate], dim=-1)  # (B,Pnn,point_len,3)

    # [!추가하자!]
    def _build_forward_segment_mask_and_active_flat(
        self,
        points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len) bool
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[
            torch.Tensor,  # seg_mask: (B, Pnn, segment_len)
            torch.Tensor,  # seg_mask_1: (B, Pnn, segment_len, 1)
            torch.Tensor,  # seg_mask_flat: (B*Pnn, segment_len)
            torch.Tensor,  # seg_mask_1_flat: (B*Pnn, segment_len, 1)
            torch.Tensor,  # active_flat: (B*Pnn,)
            int,  # B
            int,  # Pnn
            int,  # segment_len
            int,  # B_Pnn
    ]:
        """forward용 세그먼트 마스크와 '유효 이웃 플래그'를 생성합니다.

        - 세그먼트 마스크는 양 끝 노드가 모두 유효(True)인 구간만 1.0입니다.
        - active_flat은 (batch, neighbor)마다 '한 segment라도 유효한지'를 나타내는 플래그입니다.

        Args:
            points_valid: (B, Pnn, 1+segment_len) bool
                노드 단위 유효 마스크.
            seg_body_control: (B, Pnn, segment_len, 3)
                세그먼트 단위 제어 시퀀스 [v_x^b, v_y^b, w].

        Returns:
            seg_mask: (B, Pnn, segment_len) float32
            seg_mask_1: (B, Pnn, segment_len, 1) float32
            seg_mask_flat: (B*Pnn, segment_len) float32
            seg_mask_1_flat: (B*Pnn, segment_len, 1) float32
            active_flat: (B*Pnn,) bool
                각 (batch, neighbor) 슬롯별로 '한 개 이상 유효 segment가 있으면' True.
            B: 배치 크기
            Pnn: neighbor 슬롯 수
            segment_len: 세그먼트 길이
            B_Pnn: B * Pnn (flatten된 row 수)
        """
        B, Pnn, segment_len, _ = seg_body_control.shape
        seg_mask, seg_mask_1 = self._build_segment_mask(points_valid)
        B_Pnn = B * Pnn
        seg_mask_flat = seg_mask.view(B_Pnn, segment_len)
        seg_mask_1_flat = seg_mask_1.view(B_Pnn, segment_len, 1)
        active_flat = seg_mask_flat.any(dim=-1)  # (B*Pnn,)
        return (
            seg_mask,
            seg_mask_1,
            seg_mask_flat,
            seg_mask_1_flat,
            active_flat,
            B,
            Pnn,
            segment_len,
            B_Pnn,
        )

    # [!추가하자!]
    def _gather_forward_active_subset(
        self,
        points_trajectory: torch.Tensor,  # (B, Pnn, 1+segment_len, 4)
        points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len)
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
        seg_mask_flat: torch.Tensor,  # (B*Pnn, segment_len)
        seg_mask_1_flat: torch.Tensor,  # (B*Pnn, segment_len, 1)
        dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
        active_flat: torch.Tensor,  # (B*Pnn,)
        B_Pnn: int,
        segment_len: int,
    ) -> Tuple[
            torch.Tensor,  # active_indices: (N_active,)
            torch.
            Tensor,  # points_trajectory_active: (N_active, 1, 1+segment_len, 4)
            torch.Tensor,  # points_valid_active: (N_active, 1, 1+segment_len)
            torch.
            Tensor,  # seg_body_control_active: (N_active, 1, segment_len, 3)
            torch.Tensor,  # seg_mask_active: (N_active, 1, segment_len)
            torch.Tensor,  # seg_mask_1_active: (N_active, 1, segment_len, 1)
            torch.Tensor,  # dit_tokens_active: (N_active, 1, H)
    ]:
        """유효 segment가 있는 이웃들만 골라 (N_active, 1, ·) 형태로 모읍니다.
        """
        # (B*Pnn,) -> (N_active,)
        active_indices = active_flat.nonzero(as_tuple=False).squeeze(
            -1)  # (N_active,)

        points_trajectory_flat = points_trajectory.view(
            B_Pnn, 1 + segment_len, 4)  # (B*Pnn, 1+segment_len, 4)
        points_valid_flat = points_valid.view(
            B_Pnn, 1 + segment_len)  # (B*Pnn, 1+segment_len)
        seg_body_control_flat = seg_body_control.view(
            B_Pnn, segment_len, 3)  # (B*Pnn, segment_len, 3)
        dit_tokens_flat = dit_final_hidden_tokens.view(B_Pnn, -1)  # (B*Pnn, H)

        points_trajectory_active = points_trajectory_flat[
            active_indices].unsqueeze(1)  # (N_active, 1, 1+segment_len, 4)
        points_valid_active = points_valid_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, 1+segment_len)
        seg_body_control_active = seg_body_control_flat[
            active_indices].unsqueeze(1)  # (N_active, 1, segment_len, 3)
        seg_mask_active = seg_mask_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, segment_len)
        seg_mask_1_active = seg_mask_1_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, segment_len, 1)
        dit_tokens_active = dit_tokens_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, H)

        return (
            active_indices,
            points_trajectory_active,
            points_valid_active,
            seg_body_control_active,
            seg_mask_active,
            seg_mask_1_active,
            dit_tokens_active,
        )

    # [!추가하자!]
    def _run_feasible_tcn_for_active_subset(
            self,
            points_trajectory_active: torch.
        Tensor,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active: torch.Tensor,  # (N_active, 1, 1+segment_len)
            seg_body_control_active: torch.
        Tensor,  # (N_active, 1, segment_len, 3)
            seg_mask_active: torch.Tensor,  # (N_active, 1, segment_len)
            seg_mask_1_active: torch.Tensor,  # (N_active, 1, segment_len, 1)
            dit_tokens_active: torch.Tensor,  # (N_active, 1, H)
    ) -> torch.Tensor:  # (N_active, segment_len, 3)
        """유효 이웃(active subset)에 대해서만 TCN 기반 ΔU를 계산합니다.

        Args:
            points_trajectory_active: (N_active, 1, 1+segment_len, 4)
                active 이웃들의 포인트 궤적.
            points_valid_active: (N_active, 1, 1+segment_len) bool
                포인트 유효 마스크.
            seg_body_control_active: (N_active, 1, segment_len, 3)
                세그먼트 제어 [v_x^b, v_y^b, w].
            seg_mask_active: (N_active, 1, segment_len)
                세그먼트 마스크(0/1).
            seg_mask_1_active: (N_active, 1, segment_len, 1)
                세그먼트 마스크(채널용).
            dit_tokens_active: (N_active, 1, H)
                디퓨전 트렁크 은닉.

        Returns:
            delta_u_active: (N_active, segment_len, 3)
                각 active 이웃에 대한 ΔU.
        """
        x_prev_active, x_fut_active = self._split_prev_fut(
            points_trajectory_active)  # (N_active, 1, segment_len, 4) each
        x_prev_active = self._normalize_cos_sin(x_prev_active)
        x_fut_active = self._normalize_cos_sin(x_fut_active)

        Z_in_active = self._features_from_inputs(
            x_prev=x_prev_active,  # (N_active, 1, segment_len, 4)
            x_fut=x_fut_active,  # (N_active, 1, segment_len, 4)
            u_base=seg_body_control_active,  # (N_active, 1, segment_len, 3)
            dit_final_hidden_tokens=dit_tokens_active,  # (N_active, 1, H)
            points_valid=points_valid_active,  # (N_active, 1, 1+segment_len)
        )  # (N_active, 1, segment_len, 192)

        Z_s_active = self._prepare_tcn_input(
            Z_in_active, seg_mask_1_active)  # (N_active, 1, segment_len, 192)
        Z_tcn_active = self._run_tcn(
            Z_s_active, seg_mask_active,
            seg_mask_1_active)  # (N_active, 1, segment_len, 192)

        delta_u_active = self._predict_delta_u(
            Z_tcn_active, seg_mask_1_active)  # (N_active, 1, segment_len, 3)
        delta_u_active = delta_u_active.squeeze(1)  # (N_active, segment_len, 3)
        return delta_u_active

    # [!추가하자!]
    def _scatter_forward_delta_u(
        self,
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
        delta_u_active: torch.Tensor,  # (N_active, segment_len, 3)
        active_indices: torch.Tensor,  # (N_active,)
        B: int,
        Pnn: int,
        segment_len: int,
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)
        """active 이웃들에서 계산한 ΔU를 전체 (B,Pnn,segment_len,3) 텐서로 되돌립니다.

        - inactive 이웃의 ΔU는 0으로 두고,
        - active 이웃 위치에만 ΔU를 채운 뒤,
        - 최종적으로 U_ref = U_base + ΔU 를 구성합니다.

        Args:
            seg_body_control: (B, Pnn, segment_len, 3)
                베이스 제어 U_base.
            delta_u_active: (N_active, segment_len, 3)
                active 이웃들의 ΔU.
            active_indices: (N_active,)
                flatten된 row 인덱스 (0..B*Pnn-1).
            B: 배치 크기.
            Pnn: neighbor 슬롯 수.
            segment_len: 세그먼트 길이.

        Returns:
            u_ref: (B, Pnn, segment_len, 3)
                보정된 제어.
        """
        B_Pnn = B * Pnn
        seg_body_control_flat = seg_body_control.view(
            B_Pnn, segment_len, 3)  # (B*Pnn, segment_len, 3)
        delta_u_flat = torch.zeros_like(
            seg_body_control_flat)  # (B*Pnn, segment_len, 3)
        delta_u_flat[active_indices] = delta_u_active  # active 위치만 채움

        delta_u = delta_u_flat.view(B, Pnn, segment_len,
                                    3)  # (B, Pnn, segment_len, 3)
        u_ref = seg_body_control + delta_u  # (B, Pnn, segment_len, 3)
        return u_ref

    def forward(
            self,
            target_past_cur_future_valid: torch.Tensor,
            # (B, Pnn, time_len(=1+past_len) + future_len) bool
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)
        """Control Correction Network 전체 forward 경로.

        - 과거/현재/미래 포인트와 유효 마스크를 정리한 뒤,
        - 세그먼트 마스크를 만들고,
        - 유효 segment가 하나라도 있는 이웃들(active subset)에 대해서만
          TCN 기반 ΔU를 계산한 후,
        - 전체 (B,Pnn,segment_len,3) 제어 텐서로 다시 scatter 합니다.

        Args:
            target_past_cur_future_valid:
                (B, Pnn, time_len(=1+past_len) + future_len) bool
                과거~현재~미래 노드 유효 마스크.
            diffusion_trajectory:
                (B, Pnn, 1+future_len, 4) = [x, y, cos, sin]
                현재~미래 포인트 궤적.
            near_past_xyyaw:
                None 이면 현재~미래만 사용 (segment_len = future_len).
                Tensor 이면 (B, Pnn, past_len, 4) 로 과거~현재~미래 전체 사용
                (segment_len = past_len + future_len).
            seg_body_control:
                (B, Pnn, segment_len, 3)  = [v_x^b, v_y^b, w]_seg.
            dit_final_hidden_tokens:
                (B, Pnn, H)  디퓨전 트렁크 최종 은닉.

        Returns:
            u_ref: (B, Pnn, segment_len, 3)
                입력 seg_body_control 에 대한 보정 제어.
        """
        if not self.use_feasible_dl:
            return seg_body_control

        # 1) forward에서 사용할 포인트 궤적 + 노드 유효 마스크 준비
        #    points_trajectory: (B, Pnn, 1+segment_len, 4)
        #    points_valid:      (B, Pnn, 1+segment_len) bool
        points_trajectory, points_valid = self._prepare_forward_points_and_mask(
            target_past_cur_future_valid=target_past_cur_future_valid,
            diffusion_trajectory=diffusion_trajectory,
            near_past_xyyaw=near_past_xyyaw,
            seg_body_control=seg_body_control,
        )

        # 2) 세그먼트 마스크 및 active 플래그 생성
        (
            seg_mask,  # (B, Pnn, segment_len)
            seg_mask_1,  # (B, Pnn, segment_len, 1)
            seg_mask_flat,  # (B*Pnn, segment_len)
            seg_mask_1_flat,  # (B*Pnn, segment_len, 1)
            active_flat,  # (B*Pnn,)
            B,
            Pnn,
            segment_len,
            B_Pnn,
        ) = self._build_forward_segment_mask_and_active_flat(
            points_valid=points_valid,  # (B, Pnn, 1+segment_len) bool
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
        )

        # 2-1) 유효 segment가 하나도 없는 경우: 그대로 반환
        if not active_flat.any():
            return seg_body_control

        # 3) active 이웃 subset만 모으기
        (
            active_indices,  # (N_active,)
            points_trajectory_active,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active,  # (N_active, 1, 1+segment_len)
            seg_body_control_active,  # (N_active, 1, segment_len, 3)
            seg_mask_active,  # (N_active, 1, segment_len)
            seg_mask_1_active,  # (N_active, 1, segment_len, 1)
            dit_tokens_active,  # (N_active, 1, H)
        ) = self._gather_forward_active_subset(
            points_trajectory=points_trajectory,  # (B, Pnn, 1+segment_len, 4)
            points_valid=points_valid,  # (B, Pnn, 1+segment_len) bool
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
            seg_mask_flat=seg_mask_flat,  # (B*Pnn, segment_len)
            seg_mask_1_flat=seg_mask_1_flat,  # (B*Pnn, segment_len, 1)
            dit_final_hidden_tokens=dit_final_hidden_tokens,  # (B, Pnn, H)
            active_flat=active_flat,  # (B*Pnn,)
            B_Pnn=B_Pnn,
            segment_len=segment_len,
        )

        # 4) active subset에 대해서만 TCN + Head 수행해 ΔU 계산
        # (N_active, segment_len, 3)
        delta_u_active = self._run_feasible_tcn_for_active_subset(
            points_trajectory_active=
            points_trajectory_active,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active=
            points_valid_active,  # (N_active, 1, 1+segment_len)
            seg_body_control_active=
            seg_body_control_active,  # (N_active, 1, segment_len, 3)
            seg_mask_active=seg_mask_active,  # (N_active, 1, segment_len)
            seg_mask_1_active=seg_mask_1_active,  # (N_active, 1, segment_len, 1)
            dit_tokens_active=dit_tokens_active,  # (N_active, 1, H)
        )

        # 5) ΔU를 전체 (B,Pnn,segment_len,3) 텐서로 scatter 후 U_ref 반환
        u_ref = self._scatter_forward_delta_u(
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
            delta_u_active=delta_u_active,  # (N_active, segment_len, 3)
            active_indices=active_indices,  # (N_active,)
            B=B,
            Pnn=Pnn,
            segment_len=segment_len,
        )
        return u_ref

    # ================================================================
    # [추가] 멀티 채널용 Savitzky–Golay 미분 유틸
    # ================================================================

    def _savgol_finite_difference_multi(
        self,
        sequence_multi_channel: torch.Tensor,  # (N, T, C)
        valid_mask_bT: torch.Tensor,  # (N, T) bool
        dt: float,
    ) -> torch.Tensor:
        """여러 값이 한 줄에 모여 있을 때, mask 를 고려해서
        가장 단순한 방식으로 시간 변화량을 구하는 함수.

        - 중심 시점이 invalid 이면 항상 0.
        - 중심이 valid 인 경우에만 이웃을 보고,
          - 양옆 모두 valid → 중앙 차분
          - 한쪽만 valid → 해당 방향 전/후방 차분
          - 양옆 모두 invalid → 0
        """
        num_rows, sequence_length, num_channels = sequence_multi_channel.shape  # (N, T, C)
        derivative_fd = torch.zeros_like(sequence_multi_channel)  # (N, T, C)

        if sequence_length <= 1:
            # 길이 0~1이면 변화량을 정의하기 애매하므로 그냥 0 유지
            return derivative_fd

        valid = valid_mask_bT.to(torch.bool)  # (N, T)

        # --- 길이 2인 특수 케이스 ---
        if sequence_length == 2:
            center_pair_valid = (valid[:, 0] & valid[:, 1]).unsqueeze(
                -1)  # (N,1)
            diff = (sequence_multi_channel[:, 1, :] -
                    sequence_multi_channel[:, 0, :]) / dt  # (N, C)
            diff = diff * center_pair_valid  # invalid 포함되면 0
            derivative_fd[:, 0, :] = diff
            derivative_fd[:, 1, :] = diff
            return derivative_fd

        # --- 길이 >= 3인 일반 케이스 ---

        # 1) 맨 앞(t=0): 앞은 없고, 오른쪽(t=1)만 본다.
        center0 = valid[:, 0]
        right0 = valid[:, 1]
        edge0_valid = (center0 & right0).unsqueeze(-1)  # (N,1)
        diff0 = (sequence_multi_channel[:, 1, :] -
                 sequence_multi_channel[:, 0, :]) / dt  # (N,C)
        derivative_fd[:, 0, :] = diff0 * edge0_valid

        # 2) 맨 뒤(t=T-1): 오른쪽은 없고, 왼쪽(t=T-2)만 본다.
        centerL = valid[:, sequence_length - 1]
        leftL = valid[:, sequence_length - 2]
        edgeL_valid = (centerL & leftL).unsqueeze(-1)  # (N,1)
        diffL = (
            sequence_multi_channel[:, sequence_length - 1, :] -
            sequence_multi_channel[:, sequence_length - 2, :]) / dt  # (N,C)
        derivative_fd[:, sequence_length - 1, :] = diffL * edgeL_valid

        # 3) 내부 구간(t=1..T-2): 중앙/전방/후방 차분 조합
        center = valid[:, 1:-1]  # (N, T-2)
        left = valid[:, :-2]  # (N, T-2)
        right = valid[:, 2:]  # (N, T-2)

        both = center & left & right
        only_left = center & left & (~right)
        only_right = center & (~left) & right
        # center 가 False 인 곳은 위 셋 다 False → derivative 0 유지

        both_f = both.unsqueeze(-1).to(
            sequence_multi_channel.dtype)  # (N,T-2,1)
        only_left_f = only_left.unsqueeze(-1).to(sequence_multi_channel.dtype)
        only_right_f = only_right.unsqueeze(-1).to(sequence_multi_channel.dtype)

        # 중앙 차분: (t+1 - t-1) / (2dt)
        central_diff = (sequence_multi_channel[:, 2:, :] -
                        sequence_multi_channel[:, :-2, :]) / (2.0 * dt
                                                             )  # (N,T-2,C)
        # 후방 차분: (t - t-1) / dt
        backward_diff = (sequence_multi_channel[:, 1:-1, :] -
                         sequence_multi_channel[:, :-2, :]) / dt  # (N,T-2,C)
        # 전방 차분: (t+1 - t) / dt
        forward_diff = (sequence_multi_channel[:, 2:, :] -
                        sequence_multi_channel[:, 1:-1, :]) / dt  # (N,T-2,C)

        derivative_fd[:, 1:-1, :] = (both_f * central_diff +
                                     only_left_f * backward_diff +
                                     only_right_f * forward_diff)

        return derivative_fd  # (N, T, C)

    def _savgol_select_window_length(
        self,
        sequence_length: int,
        max_window_length: int,
    ) -> int:
        """실제로 사용할 Savitzky–Golay 창 길이를 결정하는 함수.

        - 전체 길이보다 긴 창은 쓸 수 없으니 자동으로 줄인다.
        - SG 특성상 홀수 길이가 필요하므로 짝수이면 1 줄여서 홀수로 바꾼다.

        Args:
            sequence_length: int
                시퀀스 길이 T.
            max_window_length: int
                설정해 둔 최대 창 길이.

        Returns:
            int: 실제로 사용할 창 길이(0 이상).
                 0이면 SG를 적용하지 않고 유한 차분만 사용한다는 뜻으로 쓸 수 있다.
        """
        window_length = int(min(max_window_length, sequence_length))
        if window_length <= 0:
            return 0
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= 0:
            window_length = 1
        return window_length

    def _savgol_build_mask_and_count_multi(
        self,
        valid_mask_bT: torch.Tensor,  # (N, T) bool
        window_length: int,
        value_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """유효 마스크로부터, 각 시점 주변 창에 실제로 데이터가 몇 개 있는지 계산하는 함수.

        - 중심 시점마다 길이 W짜리 창을 슬라이딩하면서,
          그 안에서 유효(True)한 샘플 수를 센다.

        Args:
            valid_mask_bT: (N, T) bool
                각 시점 값이 실제로 존재하는지(True/False)를 나타내는 마스크.
            window_length: int
                SG 창 길이 W.
            value_dtype: torch.dtype
                float32/float16 등, 계산에 사용할 실수 타입.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mask_window: (N, T, W)  창 안에서의 유효 마스크(0/1 float)
                - valid_count: (N, T)     창 안 유효 샘플 개수
        """
        num_rows, sequence_length = valid_mask_bT.shape  # (N, T)
        half_window = window_length // 2

        valid_float = valid_mask_bT.to(dtype=value_dtype)  # (N, T)
        valid_padded = F.pad(
            valid_float,
            (half_window, half_window),
            mode="constant",
            value=0.0,
        )  # (N, T + 2*half_window)

        mask_window = valid_padded.unfold(
            dimension=-1,
            size=window_length,
            step=1,
        )  # (N, T, W)

        valid_count = mask_window.sum(dim=-1)  # (N, T)

        return mask_window, valid_count  # (N,T,W), (N,T)

    def _savgol_build_A_all_from_mask_multi(
            self,
            mask_window: torch.Tensor,  # (N, T, W)
            gram_per_k: torch.Tensor,  # (W, P+1, P+1)
    ) -> torch.Tensor:
        """창별 시간 정보(gram_per_k)와 마스크를 이용해
        각 시점에서 쓸 작은 행렬 A를 한 번에 만드는 함수.

        Args:
            mask_window: (N, T, W)
                각 시점 t 기준으로 주변 W개 위치가 유효한지(0/1) 나타내는 값.
            gram_per_k: (W, P+1, P+1)
                시간 기저 Φ_k에 대한 Φ_k^T Φ_k 값을 위치별로 쌓아둔 텐서.

        Returns:
            torch.Tensor: (N, T, P+1, P+1)
                각 (row, t)에서 사용할 작은 행렬 A.
        """
        num_rows, sequence_length, window_length = mask_window.shape  # (N, T, W)
        _, dim_p1, _ = gram_per_k.shape  # (W, P+1, P+1)

        mask_expanded = mask_window.unsqueeze(-1).unsqueeze(
            -1)  # (N, T, W, 1, 1)
        gram_expanded = gram_per_k.view(1, 1, window_length, dim_p1,
                                        dim_p1)  # (1, 1, W, P+1, P+1)

        A_all = (mask_expanded * gram_expanded).sum(dim=2)  # (N, T, P+1, P+1)
        return A_all

    def _savgol_build_b_all_multi(
        self,
        sequence_multi_channel: torch.Tensor,  # (N, T, C)
        mask_window: torch.Tensor,  # (N, T, W)
        power_per_k: torch.Tensor,  # (W, P+1)
        window_length: int,
    ) -> torch.Tensor:
        """여러 채널 값과 시간 기저를 이용해, 각 위치에서 쓸 작은 벡터 b를 한 번에 만드는 함수.

        Args:
            sequence_multi_channel: (N, T, C)
                시간에 따라 변하는 값들이 C개 채널로 모여 있는 텐서.
            mask_window: (N, T, W)
                각 시점 주변 창에 어떤 위치가 유효한지(0/1) 나타내는 값.
            power_per_k: (W, P+1)
                시간 기저 Φ_k = [1, τ, τ^2, ...] 를 위치별로 모아둔 텐서.
            window_length: int
                창 길이 W.

        Returns:
            torch.Tensor: (N, T, C, P+1)
                각 (row, t, channel)에 대해 작은 벡터 b.
        """
        num_rows, sequence_length, num_channels = sequence_multi_channel.shape  # (N,T,C)
        half_window = window_length // 2
        P_plus_one = power_per_k.shape[-1]

        # (1) 시퀀스를 창 단위로 펼치기
        sequence_flat = sequence_multi_channel.permute(0, 2, 1).contiguous()
        sequence_flat = sequence_flat.reshape(num_rows * num_channels,
                                              sequence_length)  # (N*C, T)

        sequence_padded = F.pad(sequence_flat, (half_window, half_window),
                                mode="constant",
                                value=0.0)
        sequence_window_flat = sequence_padded.unfold(dimension=-1,
                                                      size=window_length,
                                                      step=1)  # (N*C, T, W)

        sequence_window = sequence_window_flat.view(
            num_rows, num_channels, sequence_length,
            window_length).permute(0, 2, 3, 1)  # (N, T, W, C)

        # (2) 마스크를 곱해 유효 위치만 남기기
        weighted_values = sequence_window * mask_window.unsqueeze(
            -1)  # (N, T, W, C)
        weighted_values_ex = weighted_values.unsqueeze(-1)  # (N, T, W, C, 1)

        # (3) 시간 기저 Φ_k 를 채널마다 곱해서 b를 만들기
        power_expanded = power_per_k.view(1, 1, window_length, 1,
                                          P_plus_one)  # (1, 1, W, 1, P+1)

        b_all = (weighted_values_ex * power_expanded).sum(
            dim=2)  # (N, T, C, P+1)
        return b_all

    def _savgol_solve_multi(
        self,
        A_all: torch.Tensor,  # (N, T, P+1, P+1)
        b_all: torch.Tensor,  # (N, T, C, P+1)
        valid_count: torch.Tensor,  # (N, T)
        valid_center_mask: torch.Tensor,  # (N, T) bool
        fd_derivative: torch.Tensor,  # (N, T, C)
        polyorder: int,
        regularization_epsilon: float,
    ) -> torch.Tensor:
        """창 기반 LS를 사용할 수 있는 위치에서만 SG 미분을 계산하고,
        나머지는 유한 차분 결과를 그대로 활용하는 함수.

        Args:
            A_all: (N, T, P+1, P+1)
                각 위치에서 사용할 작은 행렬 A.
            b_all: (N, T, C, P+1)
                각 위치/채널에서 사용할 작은 벡터 b.
            valid_count: (N, T)
                창 안 유효 샘플 개수.
            valid_center_mask: (N, T) bool
                중심 시점 자체가 유효한지 여부.
            fd_derivative: (N, T, C)
                미리 계산해 둔 유한 차분 결과.
            polyorder: int
                다항식 차수 P.
            regularization_epsilon: float
                A 행렬에 더해 줄 작은 값(수치 안정용).

        Returns:
            torch.Tensor: (N, T, C)
                최종 SG 미분 결과. 중심이 무효인 위치는 0으로 채운다.
        """
        num_rows, sequence_length, dim_p1, _ = A_all.shape  # (N, T, P+1, P+1)
        _, _, num_channels, _ = b_all.shape  # (N, T, C, P+1)

        min_samples = int(polyorder) + 2

        # 기본값은 유한 차분으로 깔아두고, 좋은 위치만 SG로 덮어쓰기
        derivative_out = fd_derivative.clone()  # (N, T, C)

        # SG를 적용할 수 있는 위치(데이터가 충분하고, 중심이 유효한 곳)
        good_mask = (valid_count >= min_samples) & valid_center_mask  # (N, T)
        # [추가] 맨 앞/맨 뒤 몇 개 시점은 SG를 아예 쓰지 않음 (FD/0만 사용)
        edge_margin: int = 2  # 필요하면 2로 늘려도 됨
        if sequence_length > 2 * edge_margin:
            good_mask[:, :edge_margin] = False
            good_mask[:, -edge_margin:] = False
        if not good_mask.any():
            # SG로 풀 곳이 하나도 없으면, 유한 차분 + 중심 마스크만 적용하고 반환
            derivative_out = torch.where(
                valid_center_mask.unsqueeze(-1),
                derivative_out,
                torch.zeros_like(derivative_out),
            )
            return derivative_out

        # good 위치만 flatten 해서 batched solve
        good_flat_idx = good_mask.view(-1).nonzero(as_tuple=False).squeeze(
            -1)  # (M,)

        A_flat = A_all.view(-1, dim_p1, dim_p1)  # (N*T, P+1, P+1)
        A_good = A_flat[good_flat_idx]  # (M, P+1, P+1)

        b_flat = b_all.view(-1, num_channels, dim_p1)  # (N*T, C, P+1)
        b_good = b_flat[good_flat_idx].permute(0, 2, 1)  # (M, P+1, C)

        identity_matrix = torch.eye(
            dim_p1,
            device=A_good.device,
            dtype=A_good.dtype,
        ).unsqueeze(0)  # (1, P+1, P+1)

        A_good_reg = A_good + regularization_epsilon * identity_matrix  # (M, P+1, P+1)

        # 다채널(C개)을 한 번에 푸는 배치 선형 시스템
        coefficients_good = torch.linalg.solve(
            A_good_reg,
            b_good,
        )  # (M, P+1, C)

        first_derivative_good = coefficients_good[:, 1, :]  # (M, C)

        derivative_out_flat = derivative_out.view(-1, num_channels)  # (N*T, C)
        derivative_out_flat[good_flat_idx] = first_derivative_good
        derivative_out = derivative_out_flat.view(num_rows, sequence_length,
                                                  num_channels)  # (N, T, C)

        # 중심이 유효하지 않은 위치는 최종적으로 0으로 처리
        derivative_out = torch.where(
            valid_center_mask.unsqueeze(-1),
            derivative_out,
            torch.zeros_like(derivative_out),
        )

        return derivative_out  # (N, T, C)

    def _savgol_derivative_masked_multi_torch(
        self,
        seq_bTC: torch.Tensor,  # (N, T, C)
        valid_bT: torch.Tensor,  # (N, T) bool
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """여러 채널에 대해, 마스크를 고려한 Savitzky–Golay 1차 미분을 한 번에 계산하는 함수.

        전체 흐름:
            1) 먼저 모든 채널에 대해 유한 차분 결과를 만든다.
            2) 실제로 사용할 창 길이 W를 결정한다.
            3) 유효 마스크로부터 창별(mask_window) 유효 샘플 수(valid_count)를 구한다.
            4) 시간 기저(Φ_k)와 Gram(Φ_k^TΦ_k)을 만든다.
            5) A_all, b_all을 한 번에 만든다.
            6) 데이터가 충분한 위치만 작은 선형 시스템을 풀어서 SG 미분값을 구하고,
               나머지는 유한 차분 결과를 그대로 쓴다.

        Args:
            seq_bTC: (N, T, C)
                여러 채널로 묶인 시퀀스 값.
            valid_bT: (N, T) bool
                각 시점 유효 여부.
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                다항식 차수.
            max_window_length: int
                사용할 수 있는 최대 창 길이.

        Returns:
            torch.Tensor: (N, T, C)
                각 채널에 대한 SG 1차 미분 결과.
        """
        if seq_bTC.numel() == 0:
            return seq_bTC

        num_rows, sequence_length, num_channels = seq_bTC.shape  # (N, T, C)
        device = seq_bTC.device
        dtype = seq_bTC.dtype

        # 0) 유한 차분 기본값
        fd_derivative = self._savgol_finite_difference_multi(
            sequence_multi_channel=seq_bTC,  # (N,T,C)
            valid_mask_bT=valid_bT,  # (N,T) bool
            dt=dt,
        )  # (N, T, C)

        # 1) 창 길이 선택
        window_length = self._savgol_select_window_length(
            sequence_length=sequence_length,
            max_window_length=max_window_length,
        )

        if window_length == 0:
            # SG를 전혀 쓰지 못하는 상황: 유한 차분 결과에 중심 마스크만 씌워 반환
            return torch.where(
                valid_bT.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )

        # 2) 마스크 기반 창/유효 개수 계산
        mask_window, valid_count = self._savgol_build_mask_and_count_multi(
            valid_mask_bT=valid_bT,
            window_length=window_length,
            value_dtype=dtype,
        )  # (N,T,W), (N,T)

        # 3) 시간 기저 및 Gram, power 계산
        _, gram_per_k, power_per_k = self._sg_build_design_matrix_and_gram(
            window_length=window_length,
            polyorder=polyorder,
            dt=dt,
            device=device,
            dtype=dtype,
        )  # gram_per_k: (W,P+1,P+1), power_per_k: (W,P+1)

        # 4) A_all, b_all 계산
        A_all = self._savgol_build_A_all_from_mask_multi(
            mask_window=mask_window,  # (N,T,W)
            gram_per_k=gram_per_k,  # (W,P+1,P+1)
        )  # (N,T,P+1,P+1)

        b_all = self._savgol_build_b_all_multi(
            sequence_multi_channel=seq_bTC,  # (N,T,C)
            mask_window=mask_window,  # (N,T,W)
            power_per_k=power_per_k,  # (W,P+1)
            window_length=window_length,
        )  # (N,T,C,P+1)

        # 5) LS를 적용할 수 있는 위치만 SG를 쓰고, 나머지는 유한 차분 유지
        valid_center_mask = valid_bT.to(torch.bool)  # (N,T)

        derivative_out = self._savgol_solve_multi(
            A_all=A_all,  # (N,T,P+1,P+1)
            b_all=b_all,  # (N,T,C,P+1)
            valid_count=valid_count,  # (N,T)
            valid_center_mask=valid_center_mask,  # (N,T)
            fd_derivative=fd_derivative,  # (N,T,C)
            polyorder=polyorder,
            regularization_epsilon=float(getattr(self, "_eps", 1e-6)),
        )  # (N,T,C)

        return derivative_out  # (N,T,C)

    @classmethod
    def loss_weights_by_progress(cls, progress: float,
                                 args: Any) -> Tuple[float, float, float]:
        """손실 가중치 스케줄러.

        Args:
            progress (float): 전체 학습 진행도 p∈[0,1]. 전역 스텝 기반 권장.

        Returns:
            Tuple[float, float, float]: (w_direct, w_integration, w_constraint)
        """
        p = float(max(0.0, min(1.0, progress)))
        # piecewise-linear for integration weight
        if p <= args.p_sat:
            w_int = args.w_int_min + (args.w_int_max -
                                      args.w_int_min) * (p / args.p_sat)
        else:
            w_int = args.w_int_max
        return args.w_dir, w_int, args.w_const

    # ================================================================
    # [추가] (B,Pnn,point_len, C) 형태를 멀티 채널 SG에 넘겨주는 헬퍼
    # ================================================================
    def _sg_derivative_multi_for_points(
        self,
        sequences_points: torch.Tensor,  # (B, Pnn, point_len, C)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """(B, Pnn, point_len, C) 모양의 데이터를
        멀티 채널 SG 미분 함수에 넘기기 좋게 펴고 다시 되돌리는 함수.

        Args:
            sequences_points: (B, Pnn, point_len, C)
                예: x,y 좌표나 cos,sin 값을 채널로 묶은 텐서.
            points_valid: (B, Pnn, point_len) bool
                각 포인트가 실제로 있는지 나타내는 마스크.
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                다항식 차수.
            max_window_length: int
                최대 창 길이.

        Returns:
            torch.Tensor: (B, Pnn, point_len, C)
                각 채널별 SG 1차 미분 결과.
        """
        batch_size, num_neighbors, point_len, num_channels = sequences_points.shape  # (B,Pnn,T,C)

        sequence_flat = sequences_points.reshape(
            batch_size * num_neighbors,
            point_len,
            num_channels,
        )  # (B*Pnn, T, C)
        valid_flat = points_valid.reshape(
            batch_size * num_neighbors,
            point_len,
        )  # (B*Pnn, T)

        derivative_flat = self._savgol_derivative_masked_multi_torch(
            seq_bTC=sequence_flat,  # (B*Pnn, T, C)
            valid_bT=valid_flat,  # (B*Pnn, T)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_length,
        )  # (B*Pnn, T, C)

        derivative_points = derivative_flat.reshape(
            batch_size,
            num_neighbors,
            point_len,
            num_channels,
        )  # (B,Pnn,T,C)
        return derivative_points

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
