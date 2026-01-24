import math
from typing import Dict, Any, Tuple, Optional, Iterator, List
from contextlib import contextmanager
import torch
from diffusion_planner.model.module.feasible import FeasibleProjector
import torch.nn.functional as F


class DataStatistics:

    def __init__(self, config):
        self.config = config
        self.config.hist_print_upper_percentile = 0.99
        self.config.hist_print_lower_percentile = 0.01
        self.feasible_projector = FeasibleProjector(
            self.config, self.config.hidden_dim, self.config.use_feasible_dl,
            self.config.use_feasible_filter)
        """ (vehicle, pedestrian, bicycle)
        v_b_y_max
            - 0.1 m/s 단위로 통계 히스토그램 그리자 (0~0.1, 0.1~0.2, 0.2~0.3, ..., 29.9~30.0, ...)
        v_max
            - 3m/s 단위로 통계 히스토그램 그리자 (0~3, 3~6, 6~9, 9~12, 12~15, 15~18, 18~21, 21~24, 24~27, 27~30, ...)
        a_max
            - 1m/s^2 단위로 통계 히스토그램 그리자 (0~2, 2~4, 4~6, ..., 28~30, ...)
        alpha_max
            - 1 rad/s^2 단위로 통계 히스토그램 그리자 (0~1, 1~2, 2~3, ..., 18~19, 19~20, ...)
        a_lat_max
            - 1m/s^2 단위로 통계 히스토그램 그리자 (0~2, 2~4, 4~6, ..., 28~30, ...)
            - v_abs * omega_abs 으로 계산
        r_min
            - 1m 단위로 통계 히스토그램 그리자 (0~1, 1~2, 2~3, ..., 98~99, 99~100, ...)
            - v_abs / omega_abs 으로 계산
        omega_max
            - 0.3 rad/s 단위로 통계 히스토그램 그리자 (0~0.3, 0.3~0.6, 0.6~0.9, ..., 8.7~9.0, 9.0~9.3, ...)
        """

    def _auto_tune_stats_params_from_sg_windows(
        self,
        max_window_len_xy: int,
        max_window_len_yaw: int,
    ) -> None:
        """SG 창 길이에 맞춰 통계용 edge_trim/min_valid_len을 자동으로 보정합니다.

        목적:
            - SG는 창 길이 W에서 양 끝 W//2 구간이 중앙 방식이 아니어서 값이 튈 수 있습니다.
            - 특히 2차 변화율(가속도/각가속도)은 끝부분이 더 민감합니다.
            - 그래서 window 길이에 기반해 edge_trim을 충분히 크게 만들고,
              잘라낸 뒤에도 최소한의 유효 구간이 남도록 min_valid_len을 올립니다.

        규칙:
            half_xy  = max_window_len_xy  // 2
            half_yaw = max_window_len_yaw // 2
            stats_edge_trim >= max(2, max(half_xy, half_yaw) + 1)
            stats_min_valid_len >= 2*stats_edge_trim + 3
            stats_min_valid_len_for_accel >= 2*stats_edge_trim + 3

        Args:
            max_window_len_xy: x,y에 사용하는 SG 창 길이(샘플 수, 홀수).
            max_window_len_yaw: yaw에 사용하는 SG 창 길이(샘플 수, 홀수).

        Returns:
            None. self.config의 통계 파라미터를 갱신합니다.
        """
        w_xy = max(1, int(max_window_len_xy))
        w_yaw = max(1, int(max_window_len_yaw))

        half_xy = w_xy // 2
        half_yaw = w_yaw // 2

        recommended_edge_trim = max(2, max(half_xy, half_yaw) + 1)

        cur_edge_trim = int(getattr(self.config, "stats_edge_trim", 2))
        edge_trim = max(cur_edge_trim, recommended_edge_trim)

        min_len_required = 2 * edge_trim + 3

        cur_min_valid_len = int(getattr(self.config, "stats_min_valid_len", 5))
        cur_min_valid_len_for_accel = int(
            getattr(self.config, "stats_min_valid_len_for_accel", 7)
        )

        self.config.stats_edge_trim = int(edge_trim)
        self.config.stats_min_valid_len = int(max(cur_min_valid_len, min_len_required))
        self.config.stats_min_valid_len_for_accel = int(
            max(cur_min_valid_len_for_accel, min_len_required)
        )

    def do_data_statistics(self, inputs: Dict[str, Any]):
        """ inputs
        neighbor_agents_past : (B, agent_num, time_len, 11)
        neighbor_future_gt_3_dim : (B, agent_num, future_len, 3)

        neighbor_agents_past_is_valid : (B, agent_num, time_len)
        neighbor_future_gt_is_valid : (B, agent_num, future_len)

        """

        neighbor_agents_past = inputs[
            "neighbor_agents_past"]  # (B, agent_num, time_len, 11)
        neighbor_future_gt_3_dim = inputs[
            "neighbor_future_gt_3_dim"]  # (B, agent_num, future_len, 3)

        future_len = neighbor_future_gt_3_dim.shape[2]

        assert self.config.feasible_stride_dt == 0.1, \
            "DataStatistics assumes feasible_stride_dt == 0.1s"
        (
            stride_step,  # int # 정수 스트라이드 (1이면 다운샘플링 없음).
            dt_for_savgol,  # float #  SG 필터에 넘길 샘플 간 시간 간격 [초].
            max_window_len_xy,  # int # x,y 좌표에 사용할 최대 윈도 길이(샘플 수).
            max_window_len_yaw,  # int # yaw에 사용할 최대 윈도 길이(샘플 수).
        ) = self.feasible_projector.get_feasible_stride_params(future_len)

        # [추가] SG 창 길이에 맞춰 통계 파라미터 자동 보정
        self._auto_tune_stats_params_from_sg_windows(
            max_window_len_xy=max_window_len_xy,
            max_window_len_yaw=max_window_len_yaw,
        )


        neighbor_agents_current = neighbor_agents_past[:, :,
                                                       -1, :]  # (B, agent_num, 11)
        # (vehicle, pedestrian, bicycle)
        neighbor_agents_type = neighbor_agents_current[:, :, 8:
                                                       11]  # (B, agent_num, 3)

        neighbor_agents_past_4_dim = neighbor_agents_past[:, :, :, 0:
                                                          4]  # (B, agent_num, time_len, 4)
        # 4 : x, y, cos(heading), sin(heading)

        # neighbor_future_gt_3_dim: (x,y, heading) -> neighbor_future_gt_4_dim: (x,y, cos(heading), sin(heading))
        neighbor_future_gt_heading = neighbor_future_gt_3_dim[:, :, :, 2:
                                                              3]  # (B, agent_num, future_len, 1)
        neighbor_future_gt_cos_heading = torch.cos(
            neighbor_future_gt_heading)  # (B, agent_num, future_len, 1)
        neighbor_future_gt_sin_heading = torch.sin(
            neighbor_future_gt_heading)  # (B, agent_num, future_len, 1)
        neighbor_future_gt_4_dim = torch.cat(
            [
                neighbor_future_gt_3_dim[:, :, :, 0:2],
                neighbor_future_gt_cos_heading, neighbor_future_gt_sin_heading
            ],
            dim=-1)  # (B, agent_num, future_len, 4)

        # (B, agent_num, time_len + future_len, 4)
        neighbor_agents_past_cur_future_4_dim = torch.cat(
            [neighbor_agents_past_4_dim, neighbor_future_gt_4_dim], dim=2)
        ####################################
        neighbor_agents_past_is_valid = inputs[
            "neighbor_agents_past_is_valid"]  # (B, agent_num, time_len)
        neighbor_future_gt_is_valid = inputs[
            "neighbor_future_gt_is_valid"]  # (B, agent_num, future_len)

        neighbor_agents_past_cur_future_is_valid = torch.cat(
            [neighbor_agents_past_is_valid, neighbor_future_gt_is_valid],
            dim=2)  # (B, agent_num, time_len + future_len)
        ####################################
        # diffusion_trajectory: (B, agent_num, 1+future_len, 4)
        diffusion_trajectory = neighbor_agents_past_cur_future_4_dim[::, :, -(
            1 + future_len):, :]
        # past_xyyaw: (B, agent_num, past_len(=time_len -1), 4)
        past_xyyaw = neighbor_agents_past_4_dim[:, :, :-1, :]
        # (B,Pnn,past_len+1+future_len,3) [v_x^w, v_y^w, ω]
        points_world_control = self._savgol_filter_for_control_fp32(
            diffusion_trajectory=diffusion_trajectory,  # (B, agent_num, 1+future_len, 4)
            past_xyyaw=past_xyyaw,                      # (B, agent_num, past_len, 4)
            past_cur_future_valid=neighbor_agents_past_cur_future_is_valid,  # (B,agent_num,time_len+future_len)
            dt=dt_for_savgol,
            polyorder=2,
            max_window_len_xy=max_window_len_xy,
            max_window_len_yaw=max_window_len_yaw,
        )
        # (B, Pnn,  past_len + future_len, 3)
        # [v_x^b, v_y^b, ω]_mid
        seg_body_control = \
            self.feasible_projector.compute_midpoint_controls(
                diffusion_trajectory,  # (B, Pnn, 1+T_ds, 4)
                past_xyyaw,
                # (B, Pnn, past_len_ds, 4) or None
                points_world_control,  # (B, Pnn, point_len_ds, 3)
                neighbor_agents_past_cur_future_is_valid,
                # (B, Pnn, past_len_ds+1+T_ds) bool
            )
        # segment valid: 노드 유효(점) 마스크의 양 끝 AND
        node_valid = neighbor_agents_past_cur_future_is_valid.to(
            torch.bool)  # (B, agent_num, time_len+future_len)
        seg_valid = (node_valid[:, :, :-1] & node_valid[:, :, 1:]
                    )  # (B, agent_num, past_len+future_len)
        """
        abs_v_y_b_max : (B, agent_num)
        abs_v_max : (B, agent_num)
        abs_omega_max : (B, agent_num)
        abs_a_y_lat_max : (B, agent_num)
        r_min : (B, agent_num)
        """
        (abs_v_y_b_max, abs_v_max, abs_omega_max, abs_a_y_lat_max, r_min,
         seg_ctrl_stats_valid_ba) = \
            self._get_seg_body_control_statistics(seg_body_control, seg_valid,
                                                  neighbor_agents_type)

        self._set_seg_body_control_statistics(
            abs_v_y_b_max, abs_v_max, abs_omega_max, abs_a_y_lat_max, r_min,
            neighbor_agents_type,
            values_valid_ba=seg_ctrl_stats_valid_ba,
        )


        # =========================================================
        # accel, angular_accel
        # =========================================================

        # 무효점 처리 "강검증": 0*1*0* 형태(구멍 없음)만 허용
        self._assert_mask_no_hole_0_1_0(seg_valid,
                                        context="do_data_statistics.seg_valid")

        # [NEW] (x,y,cos,sin) 2차 곡선 기반으로 accel/alpha를 직접 계산
        # points_xycs: (B, agent_num, point_len, 4)
        points_xycs = neighbor_agents_past_cur_future_4_dim  # (B, agent_num, time_len+future_len, 4)
        points_valid = node_valid  # (B, agent_num, time_len+future_len) bool

        seg_body_accel, seg_body_angular_accel, seg_speed_ok = self._compute_seg_body_accel_and_angular_accel_via_poly2(
            points_xycs=points_xycs,
            points_world_control=points_world_control,
            points_valid=points_valid,
            neighbor_agents_type=neighbor_agents_type,  # ✅ 추가
            dt=dt_for_savgol,
            max_window_len_xy=max_window_len_xy,
            max_window_len_yaw=max_window_len_yaw,
            polyorder=2,
        )

        (seg_body_accel_max, seg_body_angular_accel_max, seg_ctrl2_stats_valid_ba) = \
            self._get_seg_body_control_2_statistics(
                seg_body_accel,
                seg_body_angular_accel,
                seg_valid,
                seg_speed_ok=seg_speed_ok,
            )


        self._set_seg_body_control_2_statistics(
            seg_body_accel_max,
            seg_body_angular_accel_max,
            neighbor_agents_type,
            values_valid_ba=seg_ctrl2_stats_valid_ba,
        )

    def _get_seg_body_control_2_statistics(
        self,
        seg_body_accel: torch.Tensor,         # (B, agent_num, total_len)
        seg_body_angular_accel: torch.Tensor, # (B, agent_num, total_len)
        seg_valid: torch.Tensor,              # (B, agent_num, total_len) bool
        *,
        seg_speed_ok: Optional[torch.Tensor] = None,  # (B, agent_num, total_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """가속도/각가속도 통계를 '안정적으로' 계산합니다.

        핵심 의도:
            1) a_max(선형 가속도)는 저속/정지 구간에서 값이 튀기 쉬우므로,
               speed < v_stop 인 세그먼트는 max 계산 대상에서 제외합니다.
            2) alpha_max(각가속도)는 위 speed 마스크를 강제로 걸지 않고,
               기존 유효 구간(seg_valid) 기준으로만 계산합니다.
            3) 가장자리(edge_trim)와 너무 짧은 유효 구간(min_valid_len)은 기존 규칙대로 제외합니다.

        Args:
            seg_body_accel: (B, agent_num, T)
                세그먼트 단위 선형 가속도 값(여기서는 크기 기반).
            seg_body_angular_accel: (B, agent_num, T)
                세그먼트 단위 각가속도 값.
            seg_valid: (B, agent_num, T) bool
                양 끝 노드가 유효한 세그먼트 마스크.
            seg_speed_ok: (B, agent_num, T) bool 또는 None
                True인 세그먼트만 a_max 계산에 포함합니다.
                None이면 speed 조건을 적용하지 않습니다.

        Returns:
            seg_body_accel_max         : (B, agent_num) float32
            seg_body_angular_accel_max : (B, agent_num) float32
            stats_valid_ba             : (B, agent_num) bool
        """
        if seg_body_accel.dim() != 3 or seg_body_angular_accel.dim() != 3:
            raise ValueError(
                f"seg_body_accel/seg_body_angular_accel must be (B,agent,T). "
                f"accel={tuple(seg_body_accel.shape)}, ang={tuple(seg_body_angular_accel.shape)}"
            )
        if seg_body_accel.shape != seg_body_angular_accel.shape:
            raise ValueError("seg_body_accel and seg_body_angular_accel shape mismatch.")
        if seg_valid.shape != seg_body_accel.shape:
            raise ValueError(
                f"seg_valid must match accel shape. seg_valid={tuple(seg_valid.shape)}, accel={tuple(seg_body_accel.shape)}"
            )

        if seg_speed_ok is not None:
            if seg_speed_ok.shape != seg_body_accel.shape:
                raise ValueError(
                    f"seg_speed_ok must match (B,agent,T). "
                    f"seg_speed_ok={tuple(seg_speed_ok.shape)}, accel={tuple(seg_body_accel.shape)}"
                )

        edge_trim = int(getattr(self.config, "stats_edge_trim", 2))
        min_valid_len = int(getattr(self.config, "stats_min_valid_len_for_accel", 7))

        seg_valid_stats, agent_ok = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )

        # a_max 계산 마스크: seg_valid_stats + (옵션) 저속 제외 마스크
        seg_valid_stats_for_accel = seg_valid_stats
        if seg_speed_ok is not None:
            seg_valid_stats_for_accel = seg_valid_stats_for_accel & seg_speed_ok.to(torch.bool)


        seg_body_accel_max, ok1 = self._masked_abs_max(seg_body_accel, seg_valid_stats_for_accel)
        seg_body_angular_accel_max, ok2 = self._masked_abs_max(seg_body_angular_accel, seg_valid_stats_for_accel)

        stats_valid_ba = agent_ok & ok1 & ok2
        return seg_body_accel_max, seg_body_angular_accel_max, stats_valid_ba


    def _get_seg_body_control_statistics(
            self,
            seg_body_control: torch.Tensor,  # (B, agent_num, total_len, 3)
            seg_valid: torch.Tensor,  # (B, agent_num, total_len) bool
            neighbor_agents_type: torch.Tensor,
            # (B, agent_num, 3)  (vehicle, pedestrian, bicycle)
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """속도/각속도 관련 통계를 '안정적으로' 계산합니다.

        Returns:
            abs(v_y^b)_max : (B, agent_num) float32
            abs(v)_max     : (B, agent_num) float32
            abs(ω)_max     : (B, agent_num) float32
            abs(a_y_lat)_max : (B, agent_num) float32
            R_min          : (B, agent_num) float32
            stats_valid_ba : (B, agent_num) bool
        """
        if seg_body_control.dim() != 4:
            raise ValueError(
                f"seg_body_control must be (B,agent,T,3). got={tuple(seg_body_control.shape)}")
        if seg_valid.shape[:3] != seg_body_control.shape[:3]:
            raise ValueError(
                f"seg_valid shape must match seg_body_control[:3]. "
                f"seg_valid={tuple(seg_valid.shape)}, seg_body_control={tuple(seg_body_control.shape)}"
            )
        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[
            :2] != seg_body_control.shape[:2] or neighbor_agents_type.shape[
            2] != 3:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3). got={tuple(neighbor_agents_type.shape)}, "
                f"expected (B,agent,3)=({seg_body_control.shape[0]},{seg_body_control.shape[1]},3)"
            )

        # (2)(3) 파라미터
        edge_trim = int(getattr(self.config, "stats_edge_trim", 2))
        min_valid_len = int(getattr(self.config, "stats_min_valid_len", 5))

        seg_valid_stats, agent_ok = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),  # (B,agent,T)
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )  # seg_valid_stats: (B,agent,T), agent_ok: (B,agent)

        # 값 분해
        v_y_b = seg_body_control[:, :, :, 1]  # (B,agent,T)
        omega = seg_body_control[:, :, :, 2]  # (B,agent,T)

        v = torch.norm(seg_body_control[:, :, :, 0:2], dim=-1)  # (B,agent,T)
        a_y_lat = v * omega.abs()  # (B,agent,T)
        r = v / (omega.abs() + 1e-6)  # (B,agent,T)

        abs_v_y_b_max, ok1 = self._masked_abs_max(v_y_b, seg_valid_stats)
        abs_v_max, ok2 = self._masked_abs_max(v, seg_valid_stats)
        abs_omega_max, ok3 = self._masked_abs_max(omega, seg_valid_stats)
        abs_a_y_lat_max, ok4 = self._masked_abs_max(a_y_lat, seg_valid_stats)

        # ================================
        # [핵심 변경] r_min 계산에 저속 구간 제외 마스크 추가
        # - 보행자: 속도 조건 적용 안 함(항상 포함)
        # - 차/자전거: v > v_min 인 구간만 포함
        # ================================
        v_min_car = float(
            getattr(self.config, "stats_r_min_v_min_car_mps", 1.5))
        v_min_bicycle = float(
            getattr(self.config, "stats_r_min_v_min_bicycle_mps", 0.5))

        is_car = (neighbor_agents_type[..., 0] > 0.5)  # (B,agent)
        is_ped = (neighbor_agents_type[..., 1] > 0.5)  # (B,agent)
        is_bicycle = (neighbor_agents_type[..., 2] > 0.5)  # (B,agent)

        # (B,agent) : 차면 v_min_car, 자전거면 v_min_bicycle, 보행자는 0
        v_min_ba = (is_car.to(v.dtype) * v_min_car) + (
                    is_bicycle.to(v.dtype) * v_min_bicycle)

        # (B,agent,T) : 보행자는 항상 True, 차/자전거는 v > v_min
        speed_ok_for_r = is_ped.unsqueeze(-1) | (v > v_min_ba.unsqueeze(-1))

        # r_min에만 적용되는 최종 마스크
        r_valid_stats = seg_valid_stats & speed_ok_for_r  # (B,agent,T)

        r_min, ok5 = self._masked_min(r, r_valid_stats, invalid_fill=1.0e6)

        stats_valid_ba = agent_ok & ok1 & ok2 & ok3 & ok4 & ok5  # (B,agent)

        return abs_v_y_b_max, abs_v_max, abs_omega_max, abs_a_y_lat_max, r_min, stats_valid_ba

    # =========================================================
# [NEW] Mask 검증/SG 미분 유틸 (DataStatistics 내부 구현)
# =========================================================

    # =========================================================
    # [NEW] 통계 안정화 + FP32 SG 유틸
    # =========================================================

    @staticmethod
    @contextmanager
    def _autocast_disabled(device: torch.device) -> Iterator[None]:
        """PyTorch의 '자동 정밀도 변경(autocast)'을 잠시 끕니다.

        - 학습 중 mixed precision이 켜져 있으면, 일부 연산이 float16/bfloat16로 바뀔 수 있습니다.
        - SG 미분 내부의 작은 선형 계산은 낮은 정밀도에서 흔들릴 수 있어서,
          이 블록 안에서는 강제로 float32 연산이 되도록 보장합니다.

        Args:
            device: 텐서가 올라가 있는 디바이스(cpu/cuda)

        Yields:
            None
        """
        if hasattr(torch, "autocast"):
            with torch.autocast(device_type=device.type, enabled=False):
                yield
            return

        if device.type == "cuda" and hasattr(torch.cuda, "amp"):
            with torch.cuda.amp.autocast(enabled=False):
                yield
            return

        yield

    @staticmethod
    def _need_fp32_for_sg(dtype: torch.dtype) -> bool:
        """낮은 정밀도(float16/bfloat16)에서 SG 계산이 흔들릴 수 있어 fp32 사용 여부를 결정합니다."""
        return dtype in (torch.float16, torch.bfloat16)

    def _savgol_filter_for_control_fp32(
        self,
        diffusion_trajectory: torch.Tensor,          # (B, agent_num, 1+future_len, 4)
        past_xyyaw: Optional[torch.Tensor],          # (B, agent_num, past_len, 4) or None
        past_cur_future_valid: torch.Tensor,         # (B, agent_num, past_len+1+future_len)
        *,
        dt: float,
        polyorder: int,
        max_window_len_xy: int,
        max_window_len_yaw: int,
    ) -> torch.Tensor:
        """세계 속도/각속도 추정(SG)을 float32로 수행해 안정성을 높입니다.

        Returns:
            (B, agent_num, point_len, 3)
            - dtype은 diffusion_trajectory dtype으로 복원합니다.
        """
        force_fp32 = bool(getattr(self.config, "stats_force_fp32_sg", True))
        if (not force_fp32) or (not self._need_fp32_for_sg(diffusion_trajectory.dtype)):
            return self.feasible_projector.savgol_filter_for_control(
                diffusion_trajectory,
                past_xyyaw,
                past_cur_future_valid,
                dt=float(dt),
                polyorder=int(polyorder),
                max_window_len_xy=int(max_window_len_xy),
                max_window_len_yaw=int(max_window_len_yaw),
            )

        in_dtype = diffusion_trajectory.dtype
        device = diffusion_trajectory.device
        past_xyyaw_fp32 = None if past_xyyaw is None else past_xyyaw.to(torch.float32)

        with self._autocast_disabled(device):
            out_fp32 = self.feasible_projector.savgol_filter_for_control(
                diffusion_trajectory.to(torch.float32),  # (B,agent,1+T,4)
                past_xyyaw_fp32,                         # (B,agent,past_len,4) or None
                past_cur_future_valid,                   # (B,agent,time_len)
                dt=float(dt),
                polyorder=int(polyorder),
                max_window_len_xy=int(max_window_len_xy),
                max_window_len_yaw=int(max_window_len_yaw),
            )  # (B,agent,point_len,3), float32

        return out_fp32.to(in_dtype)

    def _sg_derivative_multi_for_points_fp32(
        self,
        sequences_points: torch.Tensor,  # (B, agent_num, T, C)
        points_valid: torch.Tensor,      # (B, agent_num, T) bool
        *,
        dt: float,
        polyorder: int,
        max_window_length: int,
        derivative_order: int,
    ) -> torch.Tensor:
        """SG 미분을 float32로 수행해 수치 스파이크를 줄입니다.

        Args:
            sequences_points: (B, agent_num, T, C)
            points_valid: (B, agent_num, T) bool
            derivative_order: 1=속도, 2=가속도

        Returns:
            (B, agent_num, T, C)  dtype은 입력 dtype으로 복원합니다.
        """
        force_fp32 = bool(getattr(self.config, "stats_force_fp32_sg", True))
        if (not force_fp32) or (not self._need_fp32_for_sg(sequences_points.dtype)):
            return self.feasible_projector._sg_derivative_multi_for_points(
                sequences_points=sequences_points,
                points_valid=points_valid,
                dt=float(dt),
                polyorder=int(polyorder),
                max_window_length=int(max_window_length),
                derivative_order=int(derivative_order),
            )

        in_dtype = sequences_points.dtype
        device = sequences_points.device

        with self._autocast_disabled(device):
            out_fp32 = self.feasible_projector._sg_derivative_multi_for_points(
                sequences_points=sequences_points.to(torch.float32),  # (B,agent,T,C)
                points_valid=points_valid,                            # (B,agent,T)
                dt=float(dt),
                polyorder=int(polyorder),
                max_window_length=int(max_window_length),
                derivative_order=int(derivative_order),
            )  # (B,agent,T,C), float32

        return out_fp32.to(in_dtype)

    def _build_seg_valid_for_stats(
        self,
        seg_valid: torch.Tensor,  # (B, agent_num, T) bool
        *,
        edge_trim: int,
        min_valid_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """세그먼트 유효 마스크에서 '양 끝 N개'를 제외한 통계용 마스크를 만듭니다.

        목적:
            - 가장자리(처음/끝)는 SG가 아닌 단순 차분(FD)이 섞일 가능성이 높고,
              이 구간이 max/min을 튀게 만들 수 있습니다.
            - 그래서 통계는 중앙 구간만 보고 계산합니다.
            - 유효 길이가 너무 짧으면 통계에서 제외합니다.

        Args:
            seg_valid: (B, agent_num, T) bool
            edge_trim: 유효 구간의 양끝에서 제외할 세그먼트 수
            min_valid_len: 원래 seg_valid 기준 유효 세그먼트 수가 이 값보다 작으면 제외

        Returns:
            seg_valid_stats: (B, agent_num, T) bool
            agent_stats_valid: (B, agent_num) bool
        """
        if seg_valid.dim() != 3:
            raise ValueError(f"seg_valid must be (B,agent,T). got={tuple(seg_valid.shape)}")

        B, A, T = seg_valid.shape
        if T <= 0:
            return seg_valid.to(torch.bool), torch.zeros((B, A), device=seg_valid.device, dtype=torch.bool)

        edge_trim_i = max(0, int(edge_trim))
        min_valid_len_i = max(1, int(min_valid_len))

        valid = seg_valid.to(torch.bool)     # (B,A,T)
        orig_len = valid.sum(dim=2)          # (B,A)
        has_any = orig_len > 0              # (B,A)

        if edge_trim_i == 0:
            seg_valid_stats = valid
            eff_len = orig_len
        else:
            valid_i8 = valid.to(torch.int8)  # (B,A,T)
            start_idx = torch.argmax(valid_i8, dim=2)  # (B,A)

            rev = torch.flip(valid_i8, dims=(2,))
            last_from_end = torch.argmax(rev, dim=2)  # (B,A)
            end_idx = (T - 1) - last_from_end         # (B,A)

            start_idx = torch.where(has_any, start_idx, torch.zeros_like(start_idx))
            end_idx = torch.where(has_any, end_idx, torch.full_like(end_idx, -1))

            start_trim = (start_idx + edge_trim_i).clamp(min=0, max=T)      # (B,A)
            end_trim = (end_idx - edge_trim_i).clamp(min=-1, max=T - 1)     # (B,A)

            t = torch.arange(T, device=seg_valid.device).view(1, 1, T)      # (1,1,T)
            in_range = (t >= start_trim.unsqueeze(-1)) & (t <= end_trim.unsqueeze(-1))  # (B,A,T)

            seg_valid_stats = valid & in_range
            eff_len = seg_valid_stats.sum(dim=2)  # (B,A)

        # (3) 유효 길이가 짧으면 통계 제외
        agent_stats_valid = has_any & (orig_len >= min_valid_len_i) & (eff_len > 0)  # (B,A)

        # 제외 대상은 완전히 차단
        seg_valid_stats = seg_valid_stats & agent_stats_valid.unsqueeze(-1)

        return seg_valid_stats, agent_stats_valid

    @staticmethod
    def _masked_abs_max(
        values_bat: torch.Tensor,  # (B, agent_num, T)
        mask_bat: torch.Tensor,    # (B, agent_num, T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """mask=True인 곳만 보고 |values|의 최대값을 계산합니다."""
        if values_bat.dim() != 3 or mask_bat.dim() != 3 or values_bat.shape != mask_bat.shape:
            raise ValueError(
                f"values/mask must be same shape (B,agent,T). values={tuple(values_bat.shape)}, mask={tuple(mask_bat.shape)}"
            )

        v = values_bat.abs().to(torch.float32)  # (B,A,T)
        v = v.masked_fill(~mask_bat, float("-inf"))
        max_ba = v.max(dim=2).values            # (B,A)
        valid_ba = torch.isfinite(max_ba)       # (B,A)
        max_safe = torch.where(valid_ba, max_ba, torch.zeros_like(max_ba))
        return max_safe, valid_ba

    @staticmethod
    def _masked_min(
        values_bat: torch.Tensor,  # (B, agent_num, T)
        mask_bat: torch.Tensor,    # (B, agent_num, T) bool
        *,
        invalid_fill: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """mask=True인 곳만 보고 values의 최소값을 계산합니다."""
        if values_bat.dim() != 3 or mask_bat.dim() != 3 or values_bat.shape != mask_bat.shape:
            raise ValueError(
                f"values/mask must be same shape (B,agent,T). values={tuple(values_bat.shape)}, mask={tuple(mask_bat.shape)}"
            )

        v = values_bat.to(torch.float32)        # (B,A,T)
        v = v.masked_fill(~mask_bat, float("inf"))
        min_ba = v.min(dim=2).values            # (B,A)
        valid_ba = torch.isfinite(min_ba)       # (B,A)

        min_safe = torch.where(
            valid_ba,
            min_ba,
            torch.full_like(min_ba, float(invalid_fill)),
        )
        return min_safe, valid_ba

    def _get_stats_accel_v_stop_per_agent(
            self,
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
    ) -> torch.Tensor:
        """agent 타입별로 a_max 계산에 사용할 v_stop(저속 제외 기준)을 만듭니다.

        목적:
            - 정지/초저속에서는 (x,y) 미분이 작은 흔들림에도 커질 수 있어,
              a_max가 노이즈 때문에 과하게 커질 수 있습니다.
            - 그래서 "속도가 충분히 있을 때만" a_max를 뽑도록 세그먼트 마스크를 만드는데,
              그 기준 속도(v_stop)를 차/보행자/자전거별로 다르게 둡니다.

        설정값 우선순위:
            1) stats_accel_v_stop_vehicle_mps / stats_accel_v_stop_pedestrian_mps / stats_accel_v_stop_bicycle_mps
            2) (위 3개가 모두 없으면) stats_accel_v_stop_mps 를 모든 타입에 동일 적용
            3) (전부 없으면) 기본값 사용: vehicle=0.5, pedestrian=0.1, bicycle=0.3

        Args:
            neighbor_agents_type: (B, agent_num, 3)
                [vehicle, pedestrian, bicycle] 원-핫(또는 그에 준하는 값).

        Returns:
            v_stop_ba: (B, agent_num) float32
                각 agent별 v_stop 값.
        """
        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[
            -1] != 3:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3). got={tuple(neighbor_agents_type.shape)}"
            )

        # 타입 마스크: (B, agent_num)
        is_vehicle = neighbor_agents_type[..., 0] > 0.5
        is_ped = neighbor_agents_type[..., 1] > 0.5
        is_bicycle = neighbor_agents_type[..., 2] > 0.5

        v_vehicle_cfg = getattr(self.config, "stats_accel_v_stop_vehicle_mps",
                                None)
        v_ped_cfg = getattr(self.config, "stats_accel_v_stop_pedestrian_mps",
                            None)
        v_bic_cfg = getattr(self.config, "stats_accel_v_stop_bicycle_mps", None)
        v_all_cfg = getattr(self.config, "stats_accel_v_stop_mps", None)

        # per-class 설정이 "하나라도" 있으면 per-class 모드로 간주
        has_any_per_class = (v_vehicle_cfg is not None) or (
                    v_ped_cfg is not None) or (v_bic_cfg is not None)

        if has_any_per_class:
            v_vehicle = float(v_vehicle_cfg) if v_vehicle_cfg is not None else (
                float(v_all_cfg) if v_all_cfg is not None else 0.5)
            v_ped = float(v_ped_cfg) if v_ped_cfg is not None else (
                float(v_all_cfg) if v_all_cfg is not None else 0.1)
            v_bic = float(v_bic_cfg) if v_bic_cfg is not None else (
                float(v_all_cfg) if v_all_cfg is not None else 0.3)
        else:
            if v_all_cfg is not None:
                v_vehicle = float(v_all_cfg)
                v_ped = float(v_all_cfg)
                v_bic = float(v_all_cfg)
            else:
                v_vehicle, v_ped, v_bic = 0.5, 0.1, 0.3

        # 음수 방지(0이면 사실상 "저속 제외 안 함")
        v_vehicle = max(0.0, v_vehicle)
        v_ped = max(0.0, v_ped)
        v_bic = max(0.0, v_bic)

        dtype = torch.float32
        v_stop_ba = (
                is_vehicle.to(dtype) * v_vehicle +
                is_ped.to(dtype) * v_ped +
                is_bicycle.to(dtype) * v_bic
        )  # (B, agent_num)

        return v_stop_ba

    def _compute_seg_body_accel_and_angular_accel_via_poly2(
            self,
            points_xycs: torch.Tensor,  # (B, agent_num, point_len, 4)
            points_world_control: torch.Tensor,  # (B, agent_num, point_len, 3)
            points_valid: torch.Tensor,  # (B, agent_num, point_len) bool
            *,
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
            dt: float,
            max_window_len_xy: int,
            max_window_len_yaw: int,
            polyorder: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(x,y,cos,sin) 시퀀스로부터 세그먼트 가속도/각가속도를 계산합니다.

        변경 의도(요청사항 반영):
            1) 선형 가속도는 (ax, ay)의 크기 sqrt(ax^2 + ay^2)를 사용합니다.
            2) 정지/초저속 구간은 a_max 계산에서 제외할 수 있도록 seg_speed_ok를 함께 만듭니다.
               - v_stop은 vehicle/ped/bicycle별로 다르게 적용합니다.

        Args:
            points_xycs: (B, agent_num, point_len, 4)
            points_world_control: (B, agent_num, point_len, 3)
            points_valid: (B, agent_num, point_len) bool
            neighbor_agents_type: (B, agent_num, 3)
            dt: float
            max_window_len_xy: int
            max_window_len_yaw: int
            polyorder: int

        Returns:
            seg_body_accel: (B, agent_num, point_len-1)
            seg_body_angular_accel: (B, agent_num, point_len-1)
            seg_speed_ok: (B, agent_num, point_len-1) bool
        """
        eps = 1.0e-6

        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[
            :2] != points_valid.shape[:2] or neighbor_agents_type.shape[
            -1] != 3:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3) and match points_valid[:2]. "
                f"type={tuple(neighbor_agents_type.shape)}, points_valid={tuple(points_valid.shape)}"
            )

        # (B, agent_num, point_len) bool 보장
        points_valid_bool = points_valid.to(torch.bool)

        # -------------------------
        # (1) (x,y) -> 2차 변화율 (a_x, a_y)
        # -------------------------
        xy = points_xycs[..., 0:2]  # (B, agent_num, point_len, 2)

        dd_xy = self._sg_derivative_multi_for_points_fp32(
            sequences_points=xy,
            points_valid=points_valid_bool,
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_xy,
            derivative_order=2,
        )  # (B, agent_num, point_len, 2)

        a_x_w = dd_xy[..., 0]  # (B, agent_num, point_len)
        a_y_w = dd_xy[..., 1]  # (B, agent_num, point_len)

        # [핵심] 가속도 크기 (분모 제거)
        accel_mag_node = torch.sqrt(
            a_x_w * a_x_w + a_y_w * a_y_w + eps)  # (B, agent_num, point_len)
        accel_mag_node = torch.where(points_valid_bool, accel_mag_node,
                                     torch.zeros_like(accel_mag_node))

        # 속도(저속 제외 마스크용)
        v_x_w = points_world_control[..., 0]  # (B, agent_num, point_len)
        v_y_w = points_world_control[..., 1]  # (B, agent_num, point_len)
        speed_node = torch.sqrt(
            v_x_w * v_x_w + v_y_w * v_y_w)  # (B, agent_num, point_len)

        # 타입별 v_stop: (B, agent_num)
        v_stop_ba = self._get_stats_accel_v_stop_per_agent(
            neighbor_agents_type).to(
            device=speed_node.device, dtype=speed_node.dtype
        )  # (B, agent_num)

        # 노드 기준 speed_ok: (B, agent_num, point_len)
        speed_ok_node = points_valid_bool & (
                    speed_node >= v_stop_ba.unsqueeze(-1))

        # 세그먼트는 양 끝 노드가 모두 speed_ok 여야 True: (B, agent_num, segment_len)
        seg_speed_ok = speed_ok_node[..., :-1] & speed_ok_node[..., 1:]

        # 노드 -> 세그먼트(중간값): (B, agent_num, segment_len)
        seg_body_accel = 0.5 * (
                    accel_mag_node[..., :-1] + accel_mag_node[..., 1:])

        # 저속 세그먼트는 0으로 정리(통계에서 제외될 거지만, 값도 깔끔히)
        seg_body_accel = torch.where(seg_speed_ok, seg_body_accel,
                                     torch.zeros_like(seg_body_accel))

        # -------------------------
        # (2) (cos,sin) -> yaw 2차 변화율 (angular_accel)
        # -------------------------
        cos_y = points_xycs[..., 2]  # (B, agent_num, point_len)
        sin_y = points_xycs[..., 3]  # (B, agent_num, point_len)

        yaw_unit_stack = torch.stack([cos_y, sin_y],
                                     dim=-1)  # (B, agent_num, point_len, 2)

        dd_yaw_unit = self._sg_derivative_multi_for_points_fp32(
            sequences_points=yaw_unit_stack,
            points_valid=points_valid_bool,
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_yaw,
            derivative_order=2,
        )  # (B, agent_num, point_len, 2)

        d2cos = dd_yaw_unit[..., 0]  # (B, agent_num, point_len)
        d2sin = dd_yaw_unit[..., 1]  # (B, agent_num, point_len)

        norm = torch.sqrt(cos_y * cos_y + sin_y * sin_y).clamp_min(
            eps)  # (B, agent_num, point_len)
        cos_u = cos_y / norm
        sin_u = sin_y / norm
        denom = (cos_u * cos_u + sin_u * sin_u).clamp_min(eps)

        angular_accel_node = (
                                         cos_u * d2sin - sin_u * d2cos) / denom  # (B, agent_num, point_len)
        angular_accel_node = torch.where(points_valid_bool, angular_accel_node,
                                         torch.zeros_like(angular_accel_node))

        seg_body_angular_accel = 0.5 * (
                    angular_accel_node[..., :-1] + angular_accel_node[
                ..., 1:])  # (B, agent_num, segment_len)

        return seg_body_accel, seg_body_angular_accel, seg_speed_ok

    @staticmethod
    def _assert_mask_no_hole_0_1_0(mask_bpt: torch.Tensor,
                                   context: str) -> None:
        """마스크가 시간축에서 0*1*0* 형태(중간 구멍 없음)인지 강하게 검증합니다.

        Args:
            mask_bpt: (B, agent_num, T) bool
                시간축 유효 여부 마스크.
            context: 에러 메시지에 표시할 위치 문자열.

        Raises:
            ValueError:
                - 1→0→1 같은 "구멍"이 있으면 예외를 발생시킵니다.
                - (0→1) 전이가 2번 이상이거나, (1→0) 전이가 2번 이상이어도 예외.
        """
        if mask_bpt.dim() != 3:
            raise ValueError(
                f"[{context}] mask must be (B,agent,T), got shape={tuple(mask_bpt.shape)}"
            )

        B, A, T = mask_bpt.shape
        v = mask_bpt.reshape(-1, T).to(torch.int8)  # (B*A, T)
        if T <= 1:
            return

        d = v[:, 1:] - v[:, :-1]  # (B*A, T-1), -1/0/+1
        rises = (d > 0).sum(dim=1)  # 0->1
        falls = (d < 0).sum(dim=1)  # 1->0

        bad = (rises > 1) | (falls > 1)
        if bad.any():
            idx = torch.nonzero(bad, as_tuple=False).flatten()
            show = idx[:min(int(idx.numel()), 8)].tolist()
            b_list = [i // A for i in show]
            a_list = [i % A for i in show]
            raise ValueError(
                f"[{context}] mask violates 0*1*0* (has multiple transitions). "
                f"bad_rows={int(idx.numel())}, examples(b,agent)={list(zip(b_list, a_list))}"
            )

        # rise가 1번이고 fall도 1번이면, rise가 fall보다 앞에 있어야(0...1...0) 정상
        # rise/fall 위치를 찾아서 비교
        need_check = (rises == 1) & (falls == 1)
        if need_check.any():
            d_need = d[need_check]  # (M, T-1)
            rise_pos = torch.argmax((d_need > 0).to(torch.int64),
                                    dim=1)  # 첫 rise 위치
            fall_pos = torch.argmax((d_need < 0).to(torch.int64),
                                    dim=1)  # 첫 fall 위치
            bad_order = rise_pos >= fall_pos
            if bad_order.any():
                # 원래 (B*A) 인덱스로 되돌려 샘플 표시
                need_idx = torch.nonzero(need_check, as_tuple=False).flatten()
                bad_idx = need_idx[torch.nonzero(bad_order,
                                                 as_tuple=False).flatten()]
                show = bad_idx[:min(int(bad_idx.numel()), 8)].tolist()
                b_list = [i // A for i in show]
                a_list = [i % A for i in show]
                raise ValueError(
                    f"[{context}] mask has a hole pattern like 1->0->1 (rise after fall). "
                    f"bad_rows={int(bad_idx.numel())}, examples(b,agent)={list(zip(b_list, a_list))}"
                )





    # =========================================================
    # [NEW] 히스토그램 누적 유틸 (DataStatistics 내부)
    # =========================================================
    def _ensure_hist_state(self) -> None:
        """통계 누적에 필요한 내부 저장소를 준비합니다.

        저장하는 것:
            - 히스토그램 counts: 분포(그래프) 확인용
            - 원본 대표값 목록: p1/p99 같은 퍼센트 지점 값을 "원본 값"으로 정확히 계산하기 위함

        Returns:
            None
        """
        if not hasattr(self, "_histograms"):
            # metric_name -> {
            #   "edges": (num_bins+1,),
            #   "counts": (3, num_bins),
            #   "bin_width": float,
            #   "max_edge": float,
            # }
            self._histograms: Dict[str, Dict[str, Any]] = {}

        if not hasattr(self, "_hist_max_values"):
            # metric_name -> (3,) 타입별(vehicle/ped/bic) 참고용 최대값(원하면 제거 가능)
            self._hist_max_values: Dict[str, torch.Tensor] = {}

        if not hasattr(self, "_hist_raw_values"):
            # metric_name -> [class0_list, class1_list, class2_list]
            # 각 class_list는 (N,) 형태의 CPU 텐서들을 append 해둔 리스트입니다.
            self._hist_raw_values: Dict[str, List[List[torch.Tensor]]] = {}

        if not hasattr(self, "_hist_draw_round"):
            self._hist_draw_round: int = 0


    def _ensure_metric_histogram(
            self,
            metric_name: str,
            *,
            bin_width: float,
            max_edge: float,
            device: torch.device,
    ) -> None:
        """특정 지표(metric)의 히스토그램/원본값 누적 버퍼를 1회만 생성합니다.

        Args:
            metric_name: 지표 이름(예: "v_max")
            bin_width: 구간 폭
            max_edge: 마지막 경계값(예: 30.0)
            device: edges/counts를 올려둘 device

        Returns:
            None
        """
        self._ensure_hist_state()

        if metric_name in self._histograms:
            # device가 달라졌으면(드문 케이스), 기존 버퍼를 새 device로 옮김
            cur_counts: torch.Tensor = self._histograms[metric_name]["counts"]
            if cur_counts.device != device:
                self._histograms[metric_name]["edges"] = \
                self._histograms[metric_name]["edges"].to(device)
                self._histograms[metric_name]["counts"] = \
                self._histograms[metric_name]["counts"].to(device)
                if metric_name in self._hist_max_values:
                    self._hist_max_values[metric_name] = self._hist_max_values[
                        metric_name].to(device)

            # [NEW] 원본값 저장 버퍼가 없으면 생성
            if metric_name not in self._hist_raw_values:
                self._hist_raw_values[metric_name] = [[], [], []]
            return

        num_bins = int(round(float(max_edge) / float(bin_width)))
        num_bins = max(1, num_bins)

        edges = (torch.arange(num_bins + 1, device=device, dtype=torch.float32)
                 * float(bin_width))  # (num_bins+1,)
        counts = torch.zeros((3, num_bins), device=device,
                             dtype=torch.long)  # (3, num_bins)

        self._histograms[metric_name] = {
            "edges": edges,
            "counts": counts,
            "bin_width": float(bin_width),
            "max_edge": float(max_edge),
        }
        self._hist_max_values[metric_name] = torch.full(
            (3,), float("-inf"), device=device, dtype=torch.float32
        )

        # [NEW] 원본 대표값 저장(퍼센트 계산용)
        self._hist_raw_values[metric_name] = [[], [], []]

    def _accumulate_metric_per_batch(
            self,
            metric_name: str,
            values_ba: torch.Tensor,  # (B, agent_num)
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
            *,
            bin_width: float,
            max_edge: float,
            ignore_above: Optional[float] = None,
            values_valid_ba: Optional[torch.Tensor] = None,
            # (B, agent_num) bool
            reduce: str = "max",
    ) -> None:
        """배치 1회 호출에서, scene 단위 대표값을 누적합니다.

        - 기본은 scene마다 "최댓값(max)"을 대표로 누적합니다.
        - r_min 같은 경우는 의미상 "최솟값(min)"을 대표로 누적해야 하므로 reduce="min"을 사용합니다.
        - 퍼센트 지점(p1/p99 등)은 counts 근사가 아니라, 여기서 모은 원본 대표값으로 계산합니다.

        Args:
            metric_name: 지표 이름
            values_ba: (B, agent_num)
            neighbor_agents_type: (B, agent_num, 3)
            bin_width: 히스토그램 bin 폭
            max_edge: 히스토그램 최대 경계
            ignore_above: 이 값 이상은 제외(None이면 미사용)
            values_valid_ba: (B, agent_num) bool. False인 agent는 제외
            reduce: "max" 또는 "min"

        Returns:
            None
        """
        if values_ba.dim() != 2:
            raise ValueError(
                f"values_ba must be (B,agent_num). got={tuple(values_ba.shape)}")
        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[
            :2] != values_ba.shape:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent_num,3) and match values_ba. "
                f"got type={tuple(neighbor_agents_type.shape)}, values={tuple(values_ba.shape)}"
            )
        if values_valid_ba is not None and values_valid_ba.shape != values_ba.shape:
            raise ValueError(
                f"values_valid_ba must match values_ba shape. "
                f"valid={tuple(values_valid_ba.shape)}, values={tuple(values_ba.shape)}"
            )

        reduce_mode = str(reduce).lower().strip()
        if reduce_mode not in ("max", "min"):
            raise ValueError(f"reduce must be 'max' or 'min'. got={reduce}")

        device = values_ba.device
        self._ensure_metric_histogram(
            metric_name,
            bin_width=float(bin_width),
            max_edge=float(max_edge),
            device=device,
        )

        hist = self._histograms[metric_name]
        edges: torch.Tensor = hist["edges"]  # (num_bins+1,)
        counts: torch.Tensor = hist["counts"]  # (3, num_bins)
        num_bins = int(counts.shape[1])
        max_edge_f = float(hist["max_edge"])

        values = values_ba.detach()  # (B, agent_num)
        type_mask = (
                    neighbor_agents_type.detach() > 0.5)  # (B, agent_num, 3) bool
        valid_mask_ba = None if values_valid_ba is None else values_valid_ba.detach().to(
            torch.bool)  # (B,agent)

        if torch.is_floating_point(values):
            fill_min = torch.finfo(values.dtype).min
            fill_max = torch.finfo(values.dtype).max
        else:
            fill_min = torch.iinfo(values.dtype).min
            fill_max = torch.iinfo(values.dtype).max

        fill_val = fill_min if reduce_mode == "max" else fill_max
        eps = 1.0e-6

        for c in range(3):
            # mask_ba: (B, agent_num)
            mask_ba = type_mask[:, :, c]

            # agent 단위 유효성 반영
            if valid_mask_ba is not None:
                mask_ba = mask_ba & valid_mask_ba

            if ignore_above is not None:
                mask_ba = mask_ba & (values < float(ignore_above))

            # scene 내에 해당 타입 agent가 "유효하게" 존재하는지
            has_b = mask_ba.any(dim=1)  # (B,)

            masked = values.masked_fill(~mask_ba, fill_val)  # (B, agent_num)

            if reduce_mode == "max":
                per_b = masked.max(dim=1).values  # (B,)
            else:
                per_b = masked.min(dim=1).values  # (B,)

            # NaN/Inf는 통계에서 제외
            has_b = has_b & torch.isfinite(per_b)  # (B,)

            # 히스토그램 누적 (has_b==False는 weights=0이므로 영향 없음)
            per_b_safe = torch.where(has_b, per_b,
                                     torch.zeros_like(per_b))  # (B,)
            v32 = per_b_safe.to(torch.float32).clamp(min=0.0,
                                                     max=max_edge_f - eps)  # (B,)

            idx = torch.bucketize(v32, edges, right=False) - 1  # (B,)
            idx = idx.clamp(min=0, max=num_bins - 1).to(torch.int64)

            weights = has_b.to(torch.long)  # (B,)

            add = torch.zeros((num_bins,), device=device, dtype=torch.long)
            add.scatter_add_(0, idx, weights)
            counts[c] += add

            # [NEW] 원본 대표값 저장(퍼센트 계산용): (N,) CPU
            raw = per_b.detach().to(torch.float32)
            raw = raw[has_b].cpu()
            if raw.numel() > 0:
                self._hist_raw_values[metric_name][c].append(raw)

            # 참고용 최대값(원하면 제거 가능)
            per_b_valid = per_b.to(torch.float32).masked_fill(~has_b,
                                                              float("-inf"))
            call_max = per_b_valid.max()
            self._hist_max_values[metric_name][c] = torch.maximum(
                self._hist_max_values[metric_name][c],
                call_max,
            )

    # =========================================================
    # [IMPLEMENT] _set_seg_body_control_2_statistics
    # =========================================================
    def _set_seg_body_control_2_statistics(
        self,
        seg_body_accel_max: torch.Tensor,          # (B, agent_num)
        seg_body_angular_accel_max: torch.Tensor,  # (B, agent_num)
        neighbor_agents_type: torch.Tensor,        # (B, agent_num, 3)
        *,
        values_valid_ba: Optional[torch.Tensor] = None,  # (B, agent_num) bool
    ) -> None:
        """a_max/alpha_max 히스토그램 누적. values_valid_ba가 있으면 그 agent는 제외합니다."""
        self._accumulate_metric_per_batch(
            metric_name="a_max",
            values_ba=seg_body_accel_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
            values_valid_ba=values_valid_ba,
        )

        self._accumulate_metric_per_batch(
            metric_name="alpha_max",
            values_ba=seg_body_angular_accel_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=20.0,
            values_valid_ba=values_valid_ba,
        )



    # =========================================================
    # [IMPLEMENT] _set_seg_body_control_statistics
    # =========================================================
    def _set_seg_body_control_statistics(
            self,
            abs_v_y_b_max: torch.Tensor,  # (B, agent_num)
            abs_v_max: torch.Tensor,  # (B, agent_num)
            abs_omega_max: torch.Tensor,  # (B, agent_num)
            abs_a_y_lat_max: torch.Tensor,  # (B, agent_num)
            r_min: torch.Tensor,  # (B, agent_num)
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
            *,
            values_valid_ba: Optional[torch.Tensor] = None,
            # (B, agent_num) bool
    ) -> None:
        """v/omega 관련 히스토그램 누적. values_valid_ba가 있으면 그 agent는 제외합니다."""
        self._accumulate_metric_per_batch(
            metric_name="v_b_y_max",
            values_ba=abs_v_y_b_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.1,
            max_edge=30.0,
            values_valid_ba=values_valid_ba,
            reduce="max",
        )

        self._accumulate_metric_per_batch(
            metric_name="v_max",
            values_ba=abs_v_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=3.0,
            max_edge=30.0,
            values_valid_ba=values_valid_ba,
            reduce="max",
        )

        self._accumulate_metric_per_batch(
            metric_name="a_lat_max",
            values_ba=abs_a_y_lat_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
            values_valid_ba=values_valid_ba,
            reduce="max",
        )

        # [FIX] r_min은 scene 대표값을 "최솟값"으로 누적해야 의미가 맞습니다.
        self._accumulate_metric_per_batch(
            metric_name="r_min",
            values_ba=r_min,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=100.0,
            ignore_above=1.0e5,
            values_valid_ba=values_valid_ba,
            reduce="min",
        )

        self._accumulate_metric_per_batch(
            metric_name="omega_max",
            values_ba=abs_omega_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.3,
            max_edge=9.3,
            values_valid_ba=values_valid_ba,
            reduce="max",
        )

    @staticmethod
    def _percentile_value_from_raw(
            values_1d: torch.Tensor,  # (N,) CPU/any
            q: float,
            *,
            mode: str,
    ) -> float:
        """원본 값(1D)에서 퍼센트 지점 값을 정확히 계산합니다.

        중요한 점:
            - 히스토그램 bin counts로 근사하지 않습니다.
            - 실제로 관측된 값들만 정렬해서, 그 중 하나를 선택합니다.
              (그래서 "bin 끝값 같은 가짜 소수점"이 나오지 않습니다.)

        Args:
            values_1d: (N,) 형태의 값들. NaN/Inf는 자동으로 제외합니다.
            q: 0~1 사이 (예: 0.99)
            mode:
                - "higher": 상위 퍼센트에서 보수적으로 큰 쪽 값을 선택
                - "lower": 하위 퍼센트에서 보수적으로 작은 쪽 값을 선택
                - "nearest": 가장 가까운 위치의 값을 선택

        Returns:
            퍼센트 지점 값(float). 데이터가 없으면 NaN.
        """
        v = values_1d.detach().to(torch.float32).reshape(-1)  # (N,)
        v = v[torch.isfinite(v)]  # (N_finite,)
        if v.numel() == 0:
            return float("nan")

        q_f = float(q)
        q_f = max(0.0, min(1.0, q_f))

        v_sorted = torch.sort(v).values  # (N,)
        n = int(v_sorted.numel())

        if q_f <= 0.0:
            return float(v_sorted[0].item())
        if q_f >= 1.0:
            return float(v_sorted[n - 1].item())

        pos = q_f * float(n - 1)

        m = str(mode).lower().strip()
        if m == "higher":
            idx = int(math.ceil(pos))
        elif m == "lower":
            idx = int(math.floor(pos))
        elif m == "nearest":
            idx = int(round(pos))
        else:
            raise ValueError(
                f"mode must be one of 'higher','lower','nearest'. got={mode}")

        idx = max(0, min(idx, n - 1))
        return float(v_sorted[idx].item())




    def draw_histograms(self) -> Dict[str, Dict[str, Any]]:
        """누적된 히스토그램 데이터를 정리하고, p1/p99를 원본 값으로 정확히 계산해 출력/표시합니다.

        변경점(요청사항 반영):
            - p_low/p_high를 counts 기반 근사로 구하지 않습니다.
            - 배치마다 누적해 둔 원본 대표값으로 p_low/p_high를 계산합니다.
            - r_min은 누적 자체가 scene 최솟값 기준으로 들어오므로,
              이름과 의미가 일치합니다.

        Returns:
            hist_data: metric_name -> {
                "edges": (num_bins+1,) CPU tensor
                "counts": (3, num_bins) CPU tensor
                "percent": (3, num_bins) CPU tensor (0~100)
                "total_per_class": (3,) CPU tensor  (counts 기준)
                "raw_total_per_class": (3,) CPU tensor (원본 대표값 개수 = p계산에 실제 사용)
                "max_per_class": (3,) CPU tensor (참고용)
                "p_low_per_class": (3,) CPU tensor  (원본 값 기반)
                "p_high_per_class": (3,) CPU tensor (원본 값 기반)
                "bin_width": float
                "max_edge": float
            }
        """
        self._ensure_hist_state()

        if len(self._histograms) == 0:
            print("[draw_histograms] 누적된 통계가 없습니다.")
            return {}

        # ---- 퍼센트 설정 ----
        q_high = float(
            getattr(self.config, "hist_print_upper_percentile", 0.99))
        q_high = max(0.0, min(1.0, q_high))

        q_low_cfg = getattr(self.config, "hist_print_lower_percentile", None)
        if q_low_cfg is None:
            q_low = 1.0 - q_high
        else:
            q_low = float(q_low_cfg)
        q_low = max(0.0, min(1.0, q_low))

        high_pct = q_high * 100.0
        low_pct = q_low * 100.0

        class_names = ["vehicle", "pedestrian", "bicycle"]

        hist_data: Dict[str, Dict[str, Any]] = {}
        for metric_name, hist in self._histograms.items():
            edges = hist["edges"].detach().cpu()  # (num_bins+1,)
            counts = hist["counts"].detach().cpu()  # (3, num_bins)
            totals = counts.sum(dim=1)  # (3,)

            denom = totals.clamp(min=1).to(torch.float32)  # (3,)
            percent = counts.to(torch.float32) / denom.unsqueeze(
                -1) * 100.0  # (3, num_bins)

            # 참고용(원하면 제거 가능)
            max_vals = self._hist_max_values.get(
                metric_name, torch.full((3,), float("nan"))
            ).detach().cpu()  # (3,)

            # ---- (A) 원본 대표값 기반 p_low/p_high ----
            raw_lists = self._hist_raw_values.get(metric_name, [[], [], []])

            raw_totals_list: List[int] = []
            p_low_list: List[float] = []
            p_high_list: List[float] = []

            for c in range(3):
                if len(raw_lists[c]) == 0:
                    raw_totals_list.append(0)
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    continue

                raw_cat = torch.cat(raw_lists[c], dim=0).to(
                    torch.float32)  # (N,)
                raw_cat = raw_cat[torch.isfinite(raw_cat)]
                n_raw = int(raw_cat.numel())
                raw_totals_list.append(n_raw)

                if n_raw <= 0:
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    continue

                v_low = self._percentile_value_from_raw(raw_cat, q_low,
                                                        mode="lower")
                v_high = self._percentile_value_from_raw(raw_cat, q_high,
                                                         mode="higher")
                p_low_list.append(v_low)
                p_high_list.append(v_high)

            hist_data[metric_name] = {
                "edges": edges,
                "counts": counts,
                "percent": percent,
                "total_per_class": totals,  # counts 기준
                "raw_total_per_class": torch.tensor(raw_totals_list,
                                                    dtype=torch.long),  # 원본값 기준
                "max_per_class": max_vals,  # 참고용
                "p_low_per_class": torch.tensor(p_low_list,
                                                dtype=torch.float32),
                "p_high_per_class": torch.tensor(p_high_list,
                                                 dtype=torch.float32),
                "bin_width": float(hist["bin_width"]),
                "max_edge": float(hist["max_edge"]),
            }

        # ---- 출력 ----
        print(
            "========== robust scalar stats (from RAW values percentiles) ==========")
        for metric_name in sorted(hist_data.keys()):
            totals_raw = hist_data[metric_name]["raw_total_per_class"]  # (3,)
            p_low = hist_data[metric_name]["p_low_per_class"]  # (3,)
            p_high = hist_data[metric_name]["p_high_per_class"]  # (3,)

            msg_parts = []
            for i, cls in enumerate(class_names):
                n_i = int(totals_raw[i].item())
                if n_i <= 0 or (not torch.isfinite(p_high[i])):
                    msg_parts.append(f"{cls}=NA(N={n_i})")
                    continue

                lo = float(p_low[i].item()) if torch.isfinite(
                    p_low[i]) else float("nan")
                hi = float(p_high[i].item())

                msg_parts.append(
                    f"{cls}:p{low_pct:.3g}={lo:.6g},p{high_pct:.3g}={hi:.6g}(N={n_i})"
                )

            print(f"{metric_name}: " + " | ".join(msg_parts))

        # ---- 히스토그램 이미지 저장 + p_low/p_high 수직선 ----
        import os
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None:
            out_dir = getattr(self.config, "histogram_output_dir", "histograms")
            os.makedirs(out_dir, exist_ok=True)

            self._hist_draw_round += 1
            round_id = int(self._hist_draw_round)

            for metric_name in sorted(hist_data.keys()):
                edges = hist_data[metric_name]["edges"]  # CPU
                percent = hist_data[metric_name]["percent"]  # CPU (3, num_bins)
                counts = hist_data[metric_name]["counts"]  # CPU (3, num_bins)
                bin_width = float(hist_data[metric_name]["bin_width"])

                centers = ((edges[:-1] + edges[
                    1:]) * 0.5).tolist()  # (num_bins,)

                p_low = hist_data[metric_name]["p_low_per_class"]  # CPU (3,)
                p_high = hist_data[metric_name]["p_high_per_class"]  # CPU (3,)
                totals_raw = hist_data[metric_name][
                    "raw_total_per_class"]  # CPU (3,)

                for c, cls_name in enumerate(class_names):
                    total = int(totals_raw[c].item())
                    if total <= 0:
                        continue

                    y = percent[c].tolist()  # (num_bins,)

                    fig = plt.figure()
                    plt.bar(centers, y, width=bin_width * 0.9)
                    plt.ylim(0.0, 100.0)
                    plt.xlabel(metric_name)
                    plt.ylabel("Percent(%)")
                    plt.title(f"{metric_name} / {cls_name} (N={total})")

                    hi = float(p_high[c].item())
                    if hi == hi:  # NaN 체크
                        plt.axvline(hi, linestyle="--", linewidth=1.5)

                    lo = float(p_low[c].item())
                    if lo == lo:
                        plt.axvline(lo, linestyle=":", linewidth=1.5)

                    plt.tight_layout()
                    save_name = f"{metric_name}_{cls_name}_r{round_id:04d}.png"
                    fig.savefig(os.path.join(out_dir, save_name), dpi=150)
                    plt.close(fig)

        # 누적 초기화
        self._histograms.clear()
        self._hist_max_values.clear()
        self._hist_raw_values.clear()

        return hist_data
