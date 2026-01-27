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
        neighbor_future_gt_11_dim : (B, agent_num, future_len, 11)

        neighbor_agents_past_is_valid : (B, agent_num, time_len)
        neighbor_future_gt_is_valid : (B, agent_num, future_len)

        """

        neighbor_agents_past = inputs[
            "neighbor_agents_past"]  # (B, agent_num, time_len, 11)
        neighbor_future_gt_11_dim = inputs[
            "neighbor_future_gt_11_dim"]  # (B, agent_num, future_len, 11)

        future_len = neighbor_future_gt_11_dim.shape[2]

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
        neighbor_future_gt_heading = neighbor_future_gt_11_dim[:, :, :, 2:
                                                              3]  # (B, agent_num, future_len, 1)
        neighbor_future_gt_cos_heading = torch.cos(
            neighbor_future_gt_heading)  # (B, agent_num, future_len, 1)
        neighbor_future_gt_sin_heading = torch.sin(
            neighbor_future_gt_heading)  # (B, agent_num, future_len, 1)
        neighbor_future_gt_4_dim = torch.cat(
            [
                neighbor_future_gt_11_dim[:, :, :, 0:2],
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
        points_valid = neighbor_agents_past_cur_future_is_valid.to(
            torch.bool)  # (B, agent_num, time_len+future_len)
        seg_valid = (points_valid[:, :, :-1] & points_valid[:, :, 1:]
                    )  # (B, agent_num, past_len+future_len)
        # 속도(저속 제외 마스크용)
        v_x_w = points_world_control[..., 0]  # (B, agent_num, point_len)
        v_y_w = points_world_control[..., 1]  # (B, agent_num, point_len)

        speed_of_points = torch.sqrt(
            v_x_w * v_x_w + v_y_w * v_y_w)  # (B, agent_num, point_len)

        # 타입별 v_stop: (B, agent_num)
        v_stop_per_agent = self._get_stats_accel_v_stop_per_agent(
            neighbor_agents_type).to(
            device=speed_of_points.device, dtype=speed_of_points.dtype
        )  # (B, agent_num)
        # 노드 기준 speed_ok: (B, agent_num, point_len)
        not_low_speed_points = points_valid & (
                    speed_of_points >= v_stop_per_agent.unsqueeze(-1))
        # 세그먼트는 양 끝 노드가 모두 speed_ok 여야 True: (B, agent_num, segment_len)
        not_low_speed_seg = not_low_speed_points[..., :-1] & not_low_speed_points[..., 1:]



        """
        abs_v_y_b_max : (B, agent_num)
        abs_v_max : (B, agent_num)
        abs_omega_max : (B, agent_num)
        abs_a_y_lat_max : (B, agent_num)
        r_min : (B, agent_num)
        seg_ctrl_stats_valid_ba: (B, agent_num)
        """
        # =========================================================
        # [NEW] v_b_y / v / a_lat / omega / r : agent-시간축 기반 완화 극값 3개 누적
        # =========================================================
        self._accumulate_seg_body_control_statistics_robust(
            seg_body_control=seg_body_control,     # (B, agent_num, T, 3)
            seg_valid=seg_valid,                   # (B, agent_num, T)
            not_low_speed_seg=not_low_speed_seg,   # (B, agent_num, T)
            neighbor_agents_type=neighbor_agents_type,  # (B, agent_num, 3)
        )


        # =========================================================
        # accel, angular_accel
        # =========================================================

        # 무효점 처리 "강검증": 0*1*0* 형태(구멍 없음)만 허용
        self._assert_mask_no_hole_0_1_0(seg_valid,
                                        context="do_data_statistics.seg_valid")

        # [NEW] (x,y,cos,sin) 2차 곡선 기반으로 accel/alpha를 직접 계산
        # seg_body_accel: (B, agent_num, total_len)
        # seg_body_angular_accel: (B, agent_num, total_len)
        seg_body_accel, seg_body_angular_accel = self._compute_seg_body_accel_and_angular_accel_via_poly2(
            neighbor_agents_past_cur_future_4_dim=neighbor_agents_past_cur_future_4_dim, # (B, agent_num, time_len+future_len, 4)
            not_low_speed_seg=not_low_speed_seg,
            points_valid=points_valid,
            neighbor_agents_type=neighbor_agents_type,  # ✅ 추가
            dt=dt_for_savgol,
            max_window_len_xy=max_window_len_xy,
            max_window_len_yaw=max_window_len_yaw,
            polyorder=2,
        )
        """
        seg_body_accel_max : (B, agent_num)
        seg_body_angular_accel_max : (B, agent_num)
        seg_ctrl2_stats_valid_ba : (B, agent_num)
        """
        # =========================================================
        # [NEW] a / alpha : agent-시간축 기반 완화 극값 3개 누적
        # =========================================================
        self._accumulate_seg_body_control_2_statistics_robust(
            seg_body_accel=seg_body_accel,                 # (B, agent_num, T)
            seg_body_angular_accel=seg_body_angular_accel, # (B, agent_num, T)
            seg_valid=seg_valid,                           # (B, agent_num, T)
            not_low_speed_seg=not_low_speed_seg,           # (B, agent_num, T)
            neighbor_agents_type=neighbor_agents_type,     # (B, agent_num, 3)
        )


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
            v_stop_per_agent: (B, agent_num) float32
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

        v_vehicle = self.config.stats_accel_v_stop_vehicle_mps
        v_ped = self.config.stats_accel_v_stop_pedestrian_mps
        v_bic = self.config.stats_accel_v_stop_bicycle_mps

        dtype = torch.float32
        v_stop_per_agent = (
                is_vehicle.to(dtype) * v_vehicle +
                is_ped.to(dtype) * v_ped +
                is_bicycle.to(dtype) * v_bic
        )  # (B, agent_num)

        return v_stop_per_agent

    def _compute_seg_body_accel_and_angular_accel_via_poly2(
            self,
            neighbor_agents_past_cur_future_4_dim: torch.Tensor,  # (B, agent_num, point_len, 4)
            not_low_speed_seg: torch.Tensor,  # (B, agent_num, point_len-1) bool
            points_valid: torch.Tensor,  # (B, agent_num, point_len) bool
            *,
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
            dt: float,
            max_window_len_xy: int,
            max_window_len_yaw: int,
            polyorder: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(x,y,cos,sin) 시퀀스로부터 세그먼트 가속도/각가속도를 계산합니다.

        변경 의도(요청사항 반영):
            1) 선형 가속도는 (ax, ay)의 크기 sqrt(ax^2 + ay^2)를 사용합니다.
            2) 정지/초저속 구간은 a_max 계산에서 제외할 수 있도록 seg_speed_ok를 함께 만듭니다.
               - v_stop은 vehicle/ped/bicycle별로 다르게 적용합니다.

        Args:
            neighbor_agents_past_cur_future_4_dim: (B, agent_num, point_len, 4)
            not_low_speed_seg:
            points_valid: (B, agent_num, point_len) bool
            neighbor_agents_type: (B, agent_num, 3)
            dt: float
            max_window_len_xy: int
            max_window_len_yaw: int
            polyorder: int

        Returns:
            seg_body_accel: (B, agent_num, point_len-1)
            seg_body_angular_accel: (B, agent_num, point_len-1)
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

        # -------------------------
        # (1) (x,y) -> 2차 변화율 (a_x, a_y)
        # -------------------------
        xy = neighbor_agents_past_cur_future_4_dim[..., 0:2]  # (B, agent_num, point_len, 2)

        dd_xy = self._sg_derivative_multi_for_points_fp32(
            sequences_points=xy,
            points_valid=points_valid,
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
        accel_mag_node = torch.where(points_valid, accel_mag_node,
                                     torch.zeros_like(accel_mag_node))


        # 노드 -> 세그먼트(중간값): (B, agent_num, segment_len)
        seg_body_accel = 0.5 * (
                    accel_mag_node[..., :-1] + accel_mag_node[..., 1:])

        # 저속 세그먼트는 0으로 정리(통계에서 제외될 거지만, 값도 깔끔히)
        seg_body_accel = torch.where(not_low_speed_seg, seg_body_accel,
                                     torch.zeros_like(seg_body_accel))

        # -------------------------
        # (2) (cos,sin) -> yaw 2차 변화율 (angular_accel)
        # -------------------------
        cos_y = neighbor_agents_past_cur_future_4_dim[..., 2]  # (B, agent_num, point_len)
        sin_y = neighbor_agents_past_cur_future_4_dim[..., 3]  # (B, agent_num, point_len)

        yaw_unit_stack = torch.stack([cos_y, sin_y],
                                     dim=-1)  # (B, agent_num, point_len, 2)

        dd_yaw_unit = self._sg_derivative_multi_for_points_fp32(
            sequences_points=yaw_unit_stack,
            points_valid=points_valid,
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
        angular_accel_node = torch.where(points_valid, angular_accel_node,
                                         torch.zeros_like(angular_accel_node))

        seg_body_angular_accel = 0.5 * (
                    angular_accel_node[..., :-1] + angular_accel_node[
                ..., 1:])  # (B, agent_num, segment_len)

        # seg_body_angular_accel 도 저속 세그먼트는 0으로 정리
        seg_body_angular_accel = torch.where(
            not_low_speed_seg,
            seg_body_angular_accel,
            torch.zeros_like(seg_body_angular_accel)
        )
        return seg_body_accel, seg_body_angular_accel

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

        변경 요약:
            - do_data_statistics가 배치마다 호출되므로, 배치별 원본값을 계속 저장하면 메모리가 계속 늘어납니다.
            - 그래서 원본값(raw) 저장은 제거하고,
              히스토그램 카운트(counts)만 누적해 메모리를 거의 고정시킵니다.

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
            # metric_name -> (3,) 타입별 참고용 최대값
            self._hist_max_values: Dict[str, torch.Tensor] = {}

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
        """특정 지표(metric)의 히스토그램 버퍼를 1회만 생성합니다.

        변경 요약:
            - 원본값(raw) 저장 버퍼를 만들지 않습니다.
            - counts/edges만 유지하므로 메모리 사용량이 거의 고정됩니다.

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
            # device가 달라졌으면 기존 버퍼를 새 device로 옮김
            cur_counts: torch.Tensor = self._histograms[metric_name]["counts"]
            if cur_counts.device != device:
                self._histograms[metric_name]["edges"] = \
                    self._histograms[metric_name]["edges"].to(device)
                self._histograms[metric_name]["counts"] = \
                    self._histograms[metric_name]["counts"].to(device)
                if metric_name in self._hist_max_values:
                    self._hist_max_values[metric_name] = self._hist_max_values[metric_name].to(device)
            return

        num_bins = int(round(float(max_edge) / float(bin_width)))
        num_bins = max(1, num_bins)

        edges = (torch.arange(num_bins + 1, device=device, dtype=torch.float32)
                 * float(bin_width))  # (num_bins+1,)
        counts = torch.zeros((3, num_bins), device=device, dtype=torch.long)  # (3, num_bins)

        self._histograms[metric_name] = {
            "edges": edges,
            "counts": counts,
            "bin_width": float(bin_width),
            "max_edge": float(max_edge),
        }

        self._hist_max_values[metric_name] = torch.full(
            (3,), float("-inf"), device=device, dtype=torch.float32
        )



    def _accumulate_metric_from_time_series(
        self,
        metric_name: str,
        values_bat: torch.Tensor,            # (B, agent_num, T)
        mask_bat: torch.Tensor,              # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
        *,
        bin_width: float,
        max_edge: float,
        ignore_above: Optional[float] = None,
    ) -> None:
        """시간별 값 전체를 타입별로 모아 히스토그램 카운트만 누적합니다.

        변경 요약:
            - 원본값(raw)을 저장하지 않습니다. (CPU 메모리 누적 제거)
            - counts만 누적하므로 메모리 사용량이 거의 고정됩니다.
            - 퍼센트(p99.9 등)는 draw_histograms에서 counts로부터 구간 단위로 근사 계산합니다.

        Args:
            metric_name: 지표 이름
            values_bat: (B, agent_num, T) 시간별 값
            mask_bat: (B, agent_num, T) bool, True인 값만 포함
            neighbor_agents_type: (B, agent_num, 3) 타입 정보
            bin_width: 히스토그램 구간 폭
            max_edge: 히스토그램 최대 경계
            ignore_above: 너무 큰 값 제외(예: r에서 분모가 거의 0인 경우)

        Returns:
            None
        """
        if values_bat.dim() != 3:
            raise ValueError(f"values_bat must be (B,agent,T). got={tuple(values_bat.shape)}")
        if mask_bat.shape != values_bat.shape:
            raise ValueError(
                f"mask_bat must match values_bat shape. mask={tuple(mask_bat.shape)}, values={tuple(values_bat.shape)}"
            )
        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[:2] != values_bat.shape[:2] or neighbor_agents_type.shape[-1] != 3:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3) and match values_bat[:2]. "
                f"type={tuple(neighbor_agents_type.shape)}, values={tuple(values_bat.shape)}"
            )

        device = values_bat.device
        self._ensure_metric_histogram(
            metric_name=metric_name,
            bin_width=float(bin_width),
            max_edge=float(max_edge),
            device=device,
        )

        hist = self._histograms[metric_name]
        edges: torch.Tensor = hist["edges"]    # (num_bins+1,)
        counts: torch.Tensor = hist["counts"]  # (3, num_bins)
        num_bins = int(counts.shape[1])

        max_edge_f = float(hist["max_edge"])
        eps = 1.0e-6

        values = values_bat.detach()
        mask = mask_bat.detach().to(torch.bool)
        type_mask = (neighbor_agents_type.detach() > 0.5)  # (B,agent,3) bool

        # NaN/Inf 제외
        mask = mask & torch.isfinite(values)

        # 너무 큰 값 제외 옵션
        if ignore_above is not None:
            mask = mask & (values < float(ignore_above))

        for c in range(3):
            # (B,agent,T)
            m = mask & type_mask[:, :, c].unsqueeze(-1)

            vals = values[m].to(torch.float32)  # (N,)
            if vals.numel() == 0:
                continue

            # 히스토그램 누적
            v32 = vals.clamp(min=0.0, max=max_edge_f - eps)
            idx = torch.bucketize(v32, edges, right=False) - 1
            idx = idx.clamp(min=0, max=num_bins - 1).to(torch.int64)

            add = torch.zeros((num_bins,), device=device, dtype=torch.long)
            add.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            counts[c] += add

            # 참고용 최대값
            self._hist_max_values[metric_name][c] = torch.maximum(
                self._hist_max_values[metric_name][c],
                vals.max(),
            )



    def _accumulate_seg_body_control_statistics_robust(
        self,
        seg_body_control: torch.Tensor,        # (B, agent_num, T, 3)
        seg_valid: torch.Tensor,              # (B, agent_num, T) bool
        not_low_speed_seg: torch.Tensor,       # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,    # (B, agent_num, 3)
    ) -> None:
        """v_b_y / v / a_lat / omega / r 을 '시간별 값 전체'로 누적합니다.

        핵심 변경:
            - agent마다 3개 대표값을 뽑지 않습니다.
            - 유효한 시간 구간(mask=True) 안의 값을 전부 타입별로 모아서 히스토그램/원본값을 누적합니다.
            - draw_histograms의 퍼센트 출력 방식은 그대로 유지됩니다.

        Args:
            seg_body_control: (B, agent_num, T, 3)
                [v_x^b, v_y^b, omega] 형태의 세그먼트 값.
            seg_valid: (B, agent_num, T) bool
                유효한 세그먼트 마스크.
            not_low_speed_seg: (B, agent_num, T) bool
                저속이 아닌 세그먼트 마스크(필요한 지표에서만 사용).
            neighbor_agents_type: (B, agent_num, 3)
                타입 정보.

        Returns:
            None
        """
        edge_trim = int(self.config.stats_edge_trim)
        min_valid_len = int(self.config.stats_min_valid_len)

        # seg_valid_stats: (B,agent,T)  (edge_trim/min_valid_len 적용된 마스크)
        seg_valid_stats, _ = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )

        not_low_speed_seg = not_low_speed_seg.to(torch.bool)  # (B,agent,T)

        # 값 분해: (B,agent,T)
        v_y_b = seg_body_control[..., 1]
        omega = seg_body_control[..., 2]
        v = torch.norm(seg_body_control[..., 0:2], dim=-1)
        a_lat = v * omega.abs()
        r = v / (omega.abs() + 1e-6)

        # 기존 의도 유지: 저속 구간은 0으로 정리해서 누적(분포에서 저속 노이즈 완화)
        v_y_b_for_stats = torch.where(not_low_speed_seg, v_y_b, torch.zeros_like(v_y_b))
        a_lat_for_stats = torch.where(not_low_speed_seg, a_lat, torch.zeros_like(a_lat))

        # v_b_y_max: (실제론 시간별 |v_y_b| 분포 누적)
        self._accumulate_metric_from_time_series(
            metric_name="v_b_y_max",
            values_bat=v_y_b_for_stats.abs(),     # (B,agent,T)
            mask_bat=seg_valid_stats,             # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.1,
            max_edge=30.0,
        )

        # v_max: (시간별 speed 분포 누적)
        self._accumulate_metric_from_time_series(
            metric_name="v_max",
            values_bat=v,                         # (B,agent,T)
            mask_bat=seg_valid_stats,             # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=3.0,
            max_edge=30.0,
        )

        # a_lat_max: (시간별 a_lat 분포 누적)
        self._accumulate_metric_from_time_series(
            metric_name="a_lat_max",
            values_bat=a_lat_for_stats,           # (B,agent,T)
            mask_bat=seg_valid_stats,             # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
        )

        # omega_max: 저속 구간은 제외해서 누적(기존 의도 유지)
        omega_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="omega_max",
            values_bat=omega.abs(),               # (B,agent,T)
            mask_bat=omega_mask,                  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.3,
            max_edge=9.3,
        )

        # r_min: 저속 구간 제외 + 너무 큰 값 제외
        r_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="r_min",
            values_bat=r,                         # (B,agent,T)
            mask_bat=r_mask,                      # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=100.0,
            ignore_above=1.0e5,
        )
    def _accumulate_seg_body_control_2_statistics_robust(
        self,
        seg_body_accel: torch.Tensor,             # (B, agent_num, T)
        seg_body_angular_accel: torch.Tensor,     # (B, agent_num, T)
        seg_valid: torch.Tensor,                  # (B, agent_num, T) bool
        not_low_speed_seg: torch.Tensor,          # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,       # (B, agent_num, 3)
    ) -> None:
        """a / alpha 를 '시간별 값 전체'로 누적합니다.

        Args:
            seg_body_accel: (B, agent_num, T)
                시간별 가속도 크기 값.
            seg_body_angular_accel: (B, agent_num, T)
                시간별 각가속도 값.
            seg_valid: (B, agent_num, T) bool
                유효 세그먼트 마스크.
            not_low_speed_seg: (B, agent_num, T) bool
                저속이 아닌 세그먼트 마스크(a 쪽에서 사용).
            neighbor_agents_type: (B, agent_num, 3)
                타입 정보.

        Returns:
            None
        """
        edge_trim = int(self.config.stats_edge_trim)
        min_valid_len = int(self.config.stats_min_valid_len)

        seg_valid_stats, _ = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )  # (B,agent,T)

        # a_max: 저속 제외(기존 의도 유지)
        accel_mask = seg_valid_stats & not_low_speed_seg.to(torch.bool)
        self._accumulate_metric_from_time_series(
            metric_name="a_max",
            values_bat=seg_body_accel.to(torch.float32),
            mask_bat=accel_mask,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
        )

        # alpha_max: abs 기준, seg_valid_stats 기준(저속 구간 값은 0이라면 자연히 0으로 쌓임)
        self._accumulate_metric_from_time_series(
            metric_name="alpha_max",
            values_bat=seg_body_angular_accel.abs().to(torch.float32),
            mask_bat=seg_valid_stats,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=20.0,
        )


    @staticmethod
    def _percentile_values_from_histogram(
        edges: torch.Tensor,        # (num_bins+1,) CPU float
        counts_1d: torch.Tensor,    # (num_bins,) CPU long
        qs: List[float],
        *,
        mode: str,
    ) -> List[float]:
        """히스토그램 카운트로 퍼센트 지점 값을 '구간 단위'로 근사 계산합니다.

        핵심:
            - 원본값을 전부 저장하지 않기 때문에, 정확한 값 대신 bin(구간) 기반 근사값을 사용합니다.
            - mode="higher": 보수적으로 큰 쪽(오른쪽 경계값) 사용
            - mode="lower" : 보수적으로 작은 쪽(왼쪽 경계값) 사용

        Args:
            edges: (num_bins+1,) bin 경계값
            counts_1d: (num_bins,) bin별 개수
            qs: 0~1 사이 퍼센트(예: 0.999)
            mode: "higher" or "lower" or "nearest"

        Returns:
            qs 길이만큼의 값 리스트. 데이터가 없으면 nan.
        """
        m = str(mode).lower().strip()
        if m not in ("higher", "lower", "nearest"):
            raise ValueError(f"mode must be 'higher','lower','nearest'. got={mode}")

        counts64 = counts_1d.to(torch.int64)
        num_bins = int(counts64.numel())
        total = int(counts64.sum().item())
        if total <= 0 or num_bins <= 0:
            return [float("nan")] * len(qs)

        cum = torch.cumsum(counts64, dim=0)  # (num_bins,)

        out: List[float] = []
        for q in qs:
            q_f = float(q)
            q_f = max(0.0, min(1.0, q_f))

            if q_f <= 0.0:
                out.append(float(edges[0].item()))
                continue
            if q_f >= 1.0:
                out.append(float(edges[-1].item()))
                continue

            target = int(math.ceil(q_f * total))
            target = max(1, min(target, total))

            idx = int(torch.searchsorted(
                cum,
                torch.tensor(target, dtype=cum.dtype),
                right=False
            ).item())
            idx = max(0, min(idx, num_bins - 1))

            if m == "higher":
                out.append(float(edges[idx + 1].item()))
            elif m == "lower":
                out.append(float(edges[idx].item()))
            else:
                out.append(float(((edges[idx] + edges[idx + 1]) * 0.5).item()))

        return out



    def draw_histograms(self) -> Dict[str, Dict[str, Any]]:
        """누적된 히스토그램을 정리하고, 여러 퍼센트 기준 한계값을 출력/그림에 표시합니다.

        변경 요약:
            - 원본값(raw)을 저장하지 않으므로,
              퍼센트 값(p99.9 등)은 히스토그램 counts 누적분포로 '구간 단위' 근사 계산합니다.
            - 메모리 사용량은 거의 고정됩니다.

        Returns:
            hist_data: metric_name -> 각종 히스토그램/퍼센트/퍼센타일 정보
        """
        self._ensure_hist_state()

        if len(self._histograms) == 0:
            print("[draw_histograms] 누적된 통계가 없습니다.")
            return {}

        q_high = float(getattr(self.config, "hist_print_upper_percentile", 0.99))
        q_high = max(0.0, min(1.0, q_high))

        q_low_cfg = getattr(self.config, "hist_print_lower_percentile", None)
        if q_low_cfg is None:
            q_low = 1.0 - q_high
        else:
            q_low = float(q_low_cfg)
        q_low = max(0.0, min(1.0, q_low))

        class_names = ["vehicle", "pedestrian", "bicycle"]

        high_p_list = [99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 96.0, 95.0]
        low_p_list_r = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        hist_data: Dict[str, Dict[str, Any]] = {}
        for metric_name, hist in self._histograms.items():
            edges = hist["edges"].detach().cpu()     # (num_bins+1,)
            counts = hist["counts"].detach().cpu()   # (3, num_bins)
            totals = counts.sum(dim=1)               # (3,)

            denom = totals.clamp(min=1).to(torch.float32)
            percent = counts.to(torch.float32) / denom.unsqueeze(-1) * 100.0

            max_vals = self._hist_max_values.get(
                metric_name, torch.full((3,), float("nan"))
            ).detach().cpu()

            # metric별 퍼센트 목록/모드
            if metric_name == "r_min":
                limit_pcts_actual = low_p_list_r
                limit_mode = "lower"
            else:
                limit_pcts_actual = high_p_list
                limit_mode = "higher"

            limit_qs = [float(p) / 100.0 for p in limit_pcts_actual]

            raw_totals_list: List[int] = []
            p_low_list: List[float] = []
            p_high_list: List[float] = []
            limit_values_all: List[List[float]] = []

            for c in range(3):
                n = int(totals[c].item())
                raw_totals_list.append(n)

                if n <= 0:
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    limit_values_all.append([float("nan")] * len(limit_qs))
                    continue

                # p_low/p_high (근사)
                v_low = self._percentile_values_from_histogram(
                    edges=edges,
                    counts_1d=counts[c],
                    qs=[q_low],
                    mode="lower",
                )[0]
                v_high = self._percentile_values_from_histogram(
                    edges=edges,
                    counts_1d=counts[c],
                    qs=[q_high],
                    mode="higher",
                )[0]
                p_low_list.append(v_low)
                p_high_list.append(v_high)

                # 요청 퍼센트 값들(근사)
                vals_c = self._percentile_values_from_histogram(
                    edges=edges,
                    counts_1d=counts[c],
                    qs=limit_qs,
                    mode=limit_mode,
                )
                limit_values_all.append(vals_c)

            hist_data[metric_name] = {
                "edges": edges,
                "counts": counts,
                "percent": percent,
                "total_per_class": totals,
                # raw 저장을 안 하므로, N은 "히스토그램에 누적된 개수"로 대체
                "raw_total_per_class": torch.tensor(raw_totals_list, dtype=torch.long),
                "max_per_class": max_vals,
                "p_low_per_class": torch.tensor(p_low_list, dtype=torch.float32),
                "p_high_per_class": torch.tensor(p_high_list, dtype=torch.float32),
                "limit_percentiles": list(limit_pcts_actual),
                "limit_values_per_class": torch.tensor(limit_values_all, dtype=torch.float32),  # (3,8)
                "bin_width": float(hist["bin_width"]),
                "max_edge": float(hist["max_edge"]),
            }

        # ===== 출력 =====
        print("========== robust limits (agent-level, softened extremes) ==========")
        metric_order = sorted(hist_data.keys())

        for j, p_hi in enumerate(high_p_list):
            p_r = low_p_list_r[j]
            print(f"[p{p_hi:g}] (r_min은 p{p_r:g} 값 사용)")

            for metric_name in metric_order:
                totals_raw = hist_data[metric_name]["raw_total_per_class"]     # (3,)
                limit_vals = hist_data[metric_name]["limit_values_per_class"] # (3,8)

                parts: List[str] = []
                for i, cls in enumerate(class_names):
                    n_i = int(totals_raw[i].item())
                    if n_i <= 0:
                        parts.append(f"{cls}=NA(N={n_i})")
                        continue

                    v = float(limit_vals[i, j].item())
                    if v == v:
                        parts.append(f"{cls}={v:.6g}(N={n_i})")
                    else:
                        parts.append(f"{cls}=nan(N={n_i})")

                print(f"  {metric_name}: " + " | ".join(parts))

        # ===== 그림 =====
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
                edges = hist_data[metric_name]["edges"]
                percent = hist_data[metric_name]["percent"]
                bin_width = float(hist_data[metric_name]["bin_width"])

                centers = ((edges[:-1] + edges[1:]) * 0.5).tolist()

                totals_raw = hist_data[metric_name]["raw_total_per_class"]
                limit_vals = hist_data[metric_name]["limit_values_per_class"]

                for c, cls_name in enumerate(class_names):
                    total = int(totals_raw[c].item())
                    if total <= 0:
                        continue

                    y = percent[c].tolist()

                    fig = plt.figure()
                    plt.bar(centers, y, width=bin_width * 0.9)
                    plt.ylim(0.0, 100.0)
                    plt.xlabel(metric_name)
                    plt.ylabel("Percent(%)")
                    plt.title(f"{metric_name} / {cls_name} (N={total})")

                    for j, p_hi in enumerate(high_p_list):
                        x = float(limit_vals[c, j].item())
                        if x == x:
                            plt.axvline(x, linestyle="--", linewidth=1.2, label=f"p{p_hi:g}")

                    plt.legend(fontsize="x-small", ncol=2, loc="upper right")

                    plt.tight_layout()
                    save_name = f"{metric_name}_{cls_name}_r{round_id:04d}.png"
                    fig.savefig(os.path.join(out_dir, save_name), dpi=150)
                    plt.close(fig)

        # 누적 초기화
        self._histograms.clear()
        self._hist_max_values.clear()

        return hist_data

