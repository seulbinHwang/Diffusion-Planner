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


    def _select_robust_extremes_per_agent(
        self,
        values_bat: torch.Tensor,  # (B, agent_num, T)
        mask_bat: torch.Tensor,    # (B, agent_num, T) bool
        *,
        reduce: str,
        top_k: int = 7,
        pick_n: int = 3,
        fallback_value: Optional[float] = None,
        fallback_valid_ba: Optional[torch.Tensor] = None,  # (B, agent_num) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """agent별 시간축 값에서 '완화된 극값' pick_n개를 뽑습니다.

        의도:
            - 그냥 max/min 1개는 튀는 값(노이즈)에 너무 민감할 수 있습니다.
            - 그래서 각 agent의 시간축 값들 중
              - max 계열: 상위 top_k개 중 가운데 pick_n개
              - min 계열: 하위 top_k개 중 가운데 pick_n개
              를 선택합니다.

        동작 요약:
            1) mask=True 인 위치만 후보로 둡니다.
            2) reduce="max"면 상위 top_k개를 뽑고(큰 값 기준),
               reduce="min"면 하위 top_k개를 뽑습니다(작은 값 기준).
            3) 그 top_k개(또는 가능한 개수) 안에서 가운데 pick_n개를 고릅니다.

        Args:
            values_bat: (B, agent_num, T)
                agent별 시간축 값.
            mask_bat: (B, agent_num, T) bool
                값이 유효한 위치만 True.
            reduce: "max" 또는 "min"
            top_k: 기본 7 (요청사항)
            pick_n: 기본 3 (요청사항)
            fallback_value:
                - None이면: 유효 값이 pick_n개 미만인 agent는 "무효" 처리합니다.
                - 값이 있으면: 유효 값이 pick_n개 미만인 agent는 fallback_value로 채웁니다.
            fallback_valid_ba: (B, agent_num) bool
                fallback을 쓸 때, 이 마스크가 True인 agent는 "유효"로 간주할지 결정합니다.

        Returns:
            selected_baK: (B, agent_num, pick_n) float32
                agent별 선택된 값 pick_n개.
            selected_valid_ba: (B, agent_num) bool
                선택이 정상적으로 되었는지(또는 fallback으로 유효 처리했는지).
        """
        if values_bat.dim() != 3 or mask_bat.dim() != 3 or values_bat.shape != mask_bat.shape:
            raise ValueError(
                f"values/mask must be same shape (B,agent,T). values={tuple(values_bat.shape)}, mask={tuple(mask_bat.shape)}"
            )

        B, A, T = values_bat.shape

        reduce_mode = str(reduce).lower().strip()
        if reduce_mode not in ("max", "min"):
            raise ValueError(f"reduce must be 'max' or 'min'. got={reduce}")

        values = values_bat.to(torch.float32)     # (B,A,T)
        mask = mask_bat.to(torch.bool)            # (B,A,T)

        if reduce_mode == "max":
            fill = float("-inf")
            largest = True
        else:
            fill = float("inf")
            largest = False

        # 마스크 밖은 -inf/inf로 채워 topk에서 자동 배제되게 함
        masked = values.masked_fill(~mask, fill)  # (B,A,T)

        k = int(min(max(1, int(top_k)), T))        # top_k는 고정이지만 T보다 클 수는 없음
        pick = int(max(1, int(pick_n)))

        # topk 추출 (max면 큰 값 topk, min이면 작은 값 topk)
        top_vals = torch.topk(masked, k=k, dim=2, largest=largest, sorted=True).values  # (B,A,k)

        finite = torch.isfinite(top_vals)          # (B,A,k)
        valid_count = finite.sum(dim=2)            # (B,A)
        ok = valid_count >= pick                   # (B,A)

        # 유효 개수(valid_count) 안에서 가운데 pick개 선택
        # 예) k=7이면 valid_count=7 -> start=2 -> 3,4,5번째(0-index 2,3,4)
        start = ((valid_count - pick) // 2).clamp(min=0)  # (B,A)

        base = torch.arange(pick, device=values_bat.device).view(1, 1, pick)  # (1,1,pick)
        idx = (start.unsqueeze(-1) + base).clamp(min=0, max=k - 1).to(torch.long)  # (B,A,pick)

        selected = torch.gather(top_vals, dim=2, index=idx)  # (B,A,pick)

        if fallback_value is not None:
            fallback = torch.full_like(selected, float(fallback_value))  # (B,A,pick)
            selected = torch.where(ok.unsqueeze(-1), selected, fallback)

            if fallback_valid_ba is None:
                selected_valid = ok
            else:
                fb = fallback_valid_ba.to(torch.bool)
                selected_valid = ok | ((~ok) & fb)
        else:
            # 무효 agent는 NaN으로 채워서 이후 isfinite 필터에서 빠지게 함
            selected = torch.where(ok.unsqueeze(-1), selected, torch.full_like(selected, float("nan")))
            selected_valid = ok

        return selected, selected_valid

    def _accumulate_metric_from_agent_samples(
        self,
        metric_name: str,
        samples_baK: torch.Tensor,           # (B, agent_num, K)
        neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
        *,
        bin_width: float,
        max_edge: float,
        samples_valid_ba: torch.Tensor,      # (B, agent_num) bool
    ) -> None:
        """agent 샘플들을 타입별로 모아 히스토그램/원본값을 누적합니다.

        여기서 "샘플"은 scene 대표값 1개가 아니라,
        agent마다 뽑아낸 대표 극값 K개(예: 3개)입니다.

        Args:
            metric_name: 지표 이름
            samples_baK: (B, agent_num, K)
                agent별 샘플 K개.
            neighbor_agents_type: (B, agent_num, 3)
                [vehicle, pedestrian, bicycle] 타입 마스크(원-핫에 준하는 값).
            bin_width: 히스토그램 bin 폭
            max_edge: 히스토그램 최대 경계
            samples_valid_ba: (B, agent_num) bool
                True인 agent만 샘플을 통계에 포함합니다.

        Returns:
            None
        """
        if samples_baK.dim() != 3:
            raise ValueError(f"samples_baK must be (B,agent,K). got={tuple(samples_baK.shape)}")
        if neighbor_agents_type.dim() != 3 or neighbor_agents_type.shape[:2] != samples_baK.shape[:2]:
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3) and match samples_baK[:2]. "
                f"type={tuple(neighbor_agents_type.shape)}, samples={tuple(samples_baK.shape)}"
            )
        if samples_valid_ba.shape != samples_baK.shape[:2]:
            raise ValueError(
                f"samples_valid_ba must be (B,agent). got={tuple(samples_valid_ba.shape)}"
            )

        device = samples_baK.device
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

        type_mask = (neighbor_agents_type.detach() > 0.5)      # (B,agent,3) bool
        valid_ba = samples_valid_ba.detach().to(torch.bool)    # (B,agent)

        for c in range(3):
            # agent 단위로 타입 + 유효성 반영
            agent_mask = type_mask[:, :, c] & valid_ba         # (B,agent)

            # 샘플 단위 마스크: (B,agent,K)
            sample_mask = agent_mask.unsqueeze(-1) & torch.isfinite(samples_baK)

            vals = samples_baK[sample_mask].to(torch.float32)  # (N,)
            if vals.numel() == 0:
                continue

            # 원본 샘플 저장(퍼센트 계산용)
            self._hist_raw_values[metric_name][c].append(vals.detach().cpu())

            # 히스토그램 누적
            v32 = vals.clamp(min=0.0, max=max_edge_f - eps)      # (N,)
            idx = torch.bucketize(v32, edges, right=False) - 1   # (N,)
            idx = idx.clamp(min=0, max=num_bins - 1).to(torch.int64)

            add = torch.zeros((num_bins,), device=device, dtype=torch.long)
            add.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            counts[c] += add

            # 참고용 최대값
            self._hist_max_values[metric_name][c] = torch.maximum(
                self._hist_max_values[metric_name][c],
                vals.max(),
            )

    def _accumulate_metric_agent_robust_extremes(
        self,
        metric_name: str,
        values_bat: torch.Tensor,            # (B, agent_num, T)
        mask_bat: torch.Tensor,              # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
        *,
        bin_width: float,
        max_edge: float,
        reduce: str,
        ignore_above: Optional[float] = None,
        base_valid_ba: Optional[torch.Tensor] = None,     # (B,agent) bool
        fallback_value: Optional[float] = None,
        fallback_valid_ba: Optional[torch.Tensor] = None, # (B,agent) bool
        top_k: int = 7,
        pick_n: int = 3,
    ) -> None:
        """시간축 값 -> (상/하위 7개 중 가운데 3개) -> 타입별 누적까지 한 번에 처리합니다.

        Args:
            metric_name: 지표 이름
            values_bat: (B, agent_num, T)
            mask_bat: (B, agent_num, T) bool
            neighbor_agents_type: (B, agent_num, 3)
            bin_width, max_edge: 히스토그램 설정
            reduce: "max" 또는 "min"
            ignore_above: 값이 너무 큰 경우(예: r에서 분모가 거의 0) 제외용
            base_valid_ba: (B,agent) bool
                False인 agent는 통계에서 제외합니다.
            fallback_value/fallback_valid_ba:
                유효 값이 pick_n개 미만일 때 처리 방식.
            top_k/pick_n: 기본 (7,3)

        Returns:
            None
        """
        if ignore_above is not None:
            mask_bat = mask_bat & (values_bat < float(ignore_above))

        selected_baK, selected_ok_ba = self._select_robust_extremes_per_agent(
            values_bat=values_bat,
            mask_bat=mask_bat,
            reduce=reduce,
            top_k=int(top_k),
            pick_n=int(pick_n),
            fallback_value=fallback_value,
            fallback_valid_ba=fallback_valid_ba,
        )

        if base_valid_ba is not None:
            selected_ok_ba = selected_ok_ba & base_valid_ba.to(torch.bool)

        self._accumulate_metric_from_agent_samples(
            metric_name=metric_name,
            samples_baK=selected_baK,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=float(bin_width),
            max_edge=float(max_edge),
            samples_valid_ba=selected_ok_ba,
        )

    def _accumulate_seg_body_control_statistics_robust(
        self,
        seg_body_control: torch.Tensor,        # (B, agent_num, T, 3)
        seg_valid: torch.Tensor,              # (B, agent_num, T) bool
        not_low_speed_seg: torch.Tensor,       # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,    # (B, agent_num, 3)
    ) -> None:
        """v_b_y / v / a_lat / omega / r 을 'agent 시간축 완화 극값 3개' 방식으로 누적합니다."""
        edge_trim = int(self.config.stats_edge_trim)
        min_valid_len = int(self.config.stats_min_valid_len)

        seg_valid_stats, agent_stats_valid = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )  # (B,agent,T), (B,agent)

        not_low_speed_seg = not_low_speed_seg.to(torch.bool)  # (B,agent,T)

        # 값 분해
        v_y_b = seg_body_control[..., 1]  # (B,agent,T)
        omega = seg_body_control[..., 2]  # (B,agent,T)

        v = torch.norm(seg_body_control[..., 0:2], dim=-1)  # (B,agent,T)
        a_lat = v * omega.abs()                             # (B,agent,T)
        r = v / (omega.abs() + 1e-6)                        # (B,agent,T)

        # 저속 구간 정리(기존 코드 의도 유지)
        v_y_b_for_stats = torch.where(not_low_speed_seg, v_y_b, torch.zeros_like(v_y_b))  # (B,agent,T)
        a_lat_for_stats = torch.where(not_low_speed_seg, a_lat, torch.zeros_like(a_lat)) # (B,agent,T)

        # v_b_y_max: 상위7개 중 가운데3개 (abs 기준)
        self._accumulate_metric_agent_robust_extremes(
            metric_name="v_b_y_max",
            values_bat=v_y_b_for_stats.abs(),     # (B,agent,T)
            mask_bat=seg_valid_stats,             # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.1,
            max_edge=30.0,
            reduce="max",
            base_valid_ba=agent_stats_valid,
        )

        # v_max
        self._accumulate_metric_agent_robust_extremes(
            metric_name="v_max",
            values_bat=v,                          # (B,agent,T)
            mask_bat=seg_valid_stats,              # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=3.0,
            max_edge=30.0,
            reduce="max",
            base_valid_ba=agent_stats_valid,
        )

        # a_lat_max
        self._accumulate_metric_agent_robust_extremes(
            metric_name="a_lat_max",
            values_bat=a_lat_for_stats,            # (B,agent,T)
            mask_bat=seg_valid_stats,              # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
            reduce="max",
            base_valid_ba=agent_stats_valid,
        )

        # omega_max: 저속 구간은 mask에서 제외(기존 코드 의도 유지)
        omega_mask = seg_valid_stats & not_low_speed_seg  # (B,agent,T)
        # 예외적으로, omega_mask에서 유효값이 너무 적으면 0으로 채워 유효 처리(기존 코드와 비슷한 취지)
        self._accumulate_metric_agent_robust_extremes(
            metric_name="omega_max",
            values_bat=omega.abs(),                # (B,agent,T)
            mask_bat=omega_mask,                   # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.3,
            max_edge=9.3,
            reduce="max",
            base_valid_ba=agent_stats_valid,
            fallback_value=0.0,
            fallback_valid_ba=agent_stats_valid,
        )

        # r_min: 하위7개 중 가운데3개
        r_mask = seg_valid_stats & not_low_speed_seg  # (B,agent,T)
        self._accumulate_metric_agent_robust_extremes(
            metric_name="r_min",
            values_bat=r,                          # (B,agent,T)
            mask_bat=r_mask,                       # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=100.0,
            reduce="min",
            ignore_above=1.0e5,
            base_valid_ba=agent_stats_valid,
        )

    def _accumulate_seg_body_control_2_statistics_robust(
        self,
        seg_body_accel: torch.Tensor,             # (B, agent_num, T)
        seg_body_angular_accel: torch.Tensor,     # (B, agent_num, T)
        seg_valid: torch.Tensor,                  # (B, agent_num, T) bool
        not_low_speed_seg: torch.Tensor,          # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,       # (B, agent_num, 3)
    ) -> None:
        """a / alpha 를 'agent 시간축 완화 극값 3개' 방식으로 누적합니다."""
        edge_trim = int(self.config.stats_edge_trim)
        min_valid_len = int(self.config.stats_min_valid_len)

        seg_valid_stats, agent_stats_valid = self._build_seg_valid_for_stats(
            seg_valid=seg_valid.to(torch.bool),
            edge_trim=edge_trim,
            min_valid_len=min_valid_len,
        )  # (B,agent,T), (B,agent)

        # a_max: 저속 제외를 한 번 더 확실히 적용(원하면 seg_valid_stats만 써도 됩니다)
        accel_mask = seg_valid_stats & not_low_speed_seg.to(torch.bool)  # (B,agent,T)

        self._accumulate_metric_agent_robust_extremes(
            metric_name="a_max",
            values_bat=seg_body_accel.to(torch.float32),  # (B,agent,T)
            mask_bat=accel_mask,                          # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
            reduce="max",
            base_valid_ba=agent_stats_valid,
        )

        # alpha_max: abs 기준, seg_valid_stats 기준
        self._accumulate_metric_agent_robust_extremes(
            metric_name="alpha_max",
            values_bat=seg_body_angular_accel.abs().to(torch.float32),  # (B,agent,T)
            mask_bat=seg_valid_stats,                                   # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=20.0,
            reduce="max",
            base_valid_ba=agent_stats_valid,
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
        """누적된 히스토그램 데이터를 정리하고, 타입별 '한계값'을 여러 퍼센트로 출력합니다.

        현재 누적 기준:
            - scene 대표값 1개가 아니라,
            - agent별 시간축에서 뽑은 '완화된 극값 3개' 샘플을 누적합니다.

        출력 규칙(요청사항):
            - v_b_y_max / v_max / a_lat_max / omega_max / a_max / alpha_max:
                상위 P = 99.9 / 99.5 / 99 / 98.5 / 98 / 97 / 96 / 95
            - r_min:
                하위 P = 0.1 / 0.5 / 1 / 1.5 / 2 / 3 / 4 / 5

        Returns:
            hist_data: metric_name -> {
                "edges": (num_bins+1,) CPU tensor
                "counts": (3, num_bins) CPU tensor
                "percent": (3, num_bins) CPU tensor (0~100)
                "total_per_class": (3,) CPU tensor  (counts 기준)
                "raw_total_per_class": (3,) CPU tensor (원본 샘플 개수)
                "max_per_class": (3,) CPU tensor (참고용)
                "p_low_per_class": (3,) CPU tensor  (config 기반)
                "p_high_per_class": (3,) CPU tensor (config 기반)
                "limit_percentiles": List[float]
                "limit_values_per_class": (3, P) CPU tensor
                "bin_width": float
                "max_edge": float
            }
        """
        self._ensure_hist_state()

        if len(self._histograms) == 0:
            print("[draw_histograms] 누적된 통계가 없습니다.")
            return {}

        # (그림에 그릴 p_low/p_high는 기존 config를 그대로 사용)
        q_high = float(getattr(self.config, "hist_print_upper_percentile", 0.99))
        q_high = max(0.0, min(1.0, q_high))

        q_low_cfg = getattr(self.config, "hist_print_lower_percentile", None)
        if q_low_cfg is None:
            q_low = 1.0 - q_high
        else:
            q_low = float(q_low_cfg)
        q_low = max(0.0, min(1.0, q_low))

        class_names = ["vehicle", "pedestrian", "bicycle"]

        # 요청하신 출력 퍼센트(%) 목록
        high_p_list = [99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 96.0, 95.0]
        low_p_list_r = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        hist_data: Dict[str, Dict[str, Any]] = {}
        for metric_name, hist in self._histograms.items():
            edges = hist["edges"].detach().cpu()     # (num_bins+1,)
            counts = hist["counts"].detach().cpu()   # (3, num_bins)
            totals = counts.sum(dim=1)               # (3,)

            denom = totals.clamp(min=1).to(torch.float32)   # (3,)
            percent = counts.to(torch.float32) / denom.unsqueeze(-1) * 100.0  # (3, num_bins)

            max_vals = self._hist_max_values.get(
                metric_name, torch.full((3,), float("nan"))
            ).detach().cpu()

            raw_lists = self._hist_raw_values.get(metric_name, [[], [], []])

            raw_totals_list: List[int] = []
            p_low_list: List[float] = []
            p_high_list: List[float] = []

            # 출력용 limit 퍼센트/모드 결정
            if metric_name == "r_min":
                limit_pcts = low_p_list_r
                limit_mode = "lower"
            else:
                limit_pcts = high_p_list
                limit_mode = "higher"

            limit_values_all: List[List[float]] = []  # (3, P)

            for c in range(3):
                if len(raw_lists[c]) == 0:
                    raw_totals_list.append(0)
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    limit_values_all.append([float("nan")] * len(limit_pcts))
                    continue

                raw_cat = torch.cat(raw_lists[c], dim=0).to(torch.float32)  # (N,)
                raw_cat = raw_cat[torch.isfinite(raw_cat)]
                n_raw = int(raw_cat.numel())
                raw_totals_list.append(n_raw)

                if n_raw <= 0:
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    limit_values_all.append([float("nan")] * len(limit_pcts))
                    continue

                # 그림용 p_low/p_high (config 기반)
                v_low = self._percentile_value_from_raw(raw_cat, q_low, mode="lower")
                v_high = self._percentile_value_from_raw(raw_cat, q_high, mode="higher")
                p_low_list.append(v_low)
                p_high_list.append(v_high)

                # 요청하신 여러 퍼센트 limit 값
                vals_c: List[float] = []
                for p in limit_pcts:
                    q = float(p) / 100.0
                    vals_c.append(self._percentile_value_from_raw(raw_cat, q, mode=limit_mode))
                limit_values_all.append(vals_c)

            hist_data[metric_name] = {
                "edges": edges,
                "counts": counts,
                "percent": percent,
                "total_per_class": totals,
                "raw_total_per_class": torch.tensor(raw_totals_list, dtype=torch.long),
                "max_per_class": max_vals,
                "p_low_per_class": torch.tensor(p_low_list, dtype=torch.float32),
                "p_high_per_class": torch.tensor(p_high_list, dtype=torch.float32),
                "limit_percentiles": list(limit_pcts),
                "limit_values_per_class": torch.tensor(limit_values_all, dtype=torch.float32),  # (3,P)
                "bin_width": float(hist["bin_width"]),
                "max_edge": float(hist["max_edge"]),
            }

        # ---- 출력 ----
        print("========== robust limits (agent-level, softened extremes) ==========")
        for metric_name in sorted(hist_data.keys()):
            totals_raw = hist_data[metric_name]["raw_total_per_class"]          # (3,)
            limit_pcts = hist_data[metric_name]["limit_percentiles"]            # List[float]
            limit_vals = hist_data[metric_name]["limit_values_per_class"]       # (3,P)

            msg_parts = []
            for i, cls in enumerate(class_names):
                n_i = int(totals_raw[i].item())
                if n_i <= 0:
                    msg_parts.append(f"{cls}=NA(N={n_i})")
                    continue

                vals_i = limit_vals[i]  # (P,)
                parts_i = []
                for j, p in enumerate(limit_pcts):
                    v = float(vals_i[j].item())
                    if v == v:  # NaN 체크
                        parts_i.append(f"p{p:g}={v:.6g}")
                    else:
                        parts_i.append(f"p{p:g}=nan")

                msg_parts.append(f"{cls}(" + ",".join(parts_i) + f",N={n_i})")

            print(f"{metric_name}: " + " | ".join(msg_parts))

        # ---- 히스토그램 이미지 저장(기존 유지: config 기반 p_low/p_high만 수직선) ----
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
                edges = hist_data[metric_name]["edges"]      # CPU
                percent = hist_data[metric_name]["percent"]  # CPU (3, num_bins)
                bin_width = float(hist_data[metric_name]["bin_width"])

                centers = ((edges[:-1] + edges[1:]) * 0.5).tolist()  # (num_bins,)

                p_low = hist_data[metric_name]["p_low_per_class"]    # CPU (3,)
                p_high = hist_data[metric_name]["p_high_per_class"]  # CPU (3,)
                totals_raw = hist_data[metric_name]["raw_total_per_class"]  # CPU (3,)

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
                    if hi == hi:
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
