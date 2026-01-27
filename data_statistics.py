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
        # =========================================================
        # 히스토그램 설정 (클래스 순서: vehicle, pedestrian, bicycle)
        # - bin_width: bin 간격
        # - max_edge_per_class: (vehicle, pedestrian, bicycle) 순서의 최대 x 범위
        # =========================================================
        self._hist_spec: Dict[str, Dict[str, Any]] = {
            "v_max": {
                "bin_width": 1.0,
                "max_edge_per_class": (45.0, 7.5, 20.0),# (vehicle, pedestrian, bicycle)
            },
            "a_max": {
                "bin_width": 0.2,
                "max_edge_per_class": (13.0, 7.0, 7.0),# (vehicle, pedestrian, bicycle)
            },
            "alpha_max": {
                "bin_width": 0.2,
                "max_edge_per_class": (3.5, 19.0, 9.0),# (vehicle, pedestrian, bicycle)
            },
            "a_lat_max": {
                "bin_width": 0.1,
                "max_edge_per_class": (6.0, 3.0, 5.0), # (vehicle, pedestrian, bicycle)
            },
            "r_min": {
                "bin_width": 0.1,
                "max_edge_per_class": (5.0, 0.5, 3.0),# (vehicle, pedestrian, bicycle)
            },
            "omega_max": {
                "bin_width": 0.1,
                "max_edge_per_class": (2.5, 4.5, 2.5),# (vehicle, pedestrian, bicycle)
            },
            "v_b_y_max": {
                "bin_width": 0.05,
                "max_edge_per_class": (2.5, 1.5, 2.5),# (vehicle, pedestrian, bicycle)
            },
        }

    def _get_histogram_spec(
        self,
        metric_name: str,
    ) -> Tuple[float, Tuple[float, float, float]]:
        """지표(metric)에 대한 히스토그램 설정을 반환합니다.

        Args:
            metric_name: 지표 이름. 예) "v_max", "a_max", "alpha_max", "a_lat_max",
                "r_min", "omega_max", "v_b_y_max"

        Returns:
            bin_width: float
                bin 간격.
            max_edge_per_class: (vehicle, pedestrian, bicycle)
                클래스별 최대 x 범위.
        """
        if not hasattr(self, "_hist_spec") or metric_name not in self._hist_spec:
            raise ValueError(f"Unknown metric_name={metric_name}. _hist_spec에 정의가 필요합니다.")

        spec = self._hist_spec[metric_name]
        bin_width = float(spec["bin_width"])
        max_edge_per_class = spec["max_edge_per_class"]
        if (not isinstance(max_edge_per_class, (tuple, list))) or len(max_edge_per_class) != 3:
            raise ValueError(
                f"{metric_name}.max_edge_per_class must be (vehicle,ped,bic). got={max_edge_per_class}"
            )
        return bin_width, (float(max_edge_per_class[0]), float(max_edge_per_class[1]), float(max_edge_per_class[2]))

    def _get_metric_axis_label(self, metric_name: str) -> str:
        """히스토그램 x축 라벨(표시 이름)을 metric에 맞게 반환합니다.

        주의:
            - 현재 히스토그램은 "max/min 결과"가 아니라,
              시간(세그먼트)별 값들을 그대로 누적한 분포입니다.
            - 그래서 metric_name이 *_max, r_min 형태여도,
              x축 라벨은 실제로 쌓인 값의 의미(단위 포함)로 표시합니다.

        Args:
            metric_name: 지표 이름 (예: "v_max", "a_max", ...)

        Returns:
            x_label: x축에 표시할 문자열
        """
        label_map: Dict[str, str] = {
            "v_max": "v (m/s)",
            "a_max": "a (m/s^2)",
            "alpha_max": "|alpha| (rad/s^2)",
            "a_lat_max": "a_lat = v*|omega| (m/s^2)",
            "omega_max": "|omega| (rad/s)",
            "v_b_y_max": "|v_y^b| (m/s)",
            "r_min": "r = v/|omega| (m)",
        }
        return label_map.get(metric_name, metric_name)

    def _ensure_metric_histogram(
        self,
        metric_name: str,
        *,
        bin_width: float,
        max_edge_per_class: Tuple[float, float, float],
        device: torch.device,
    ) -> None:
        """특정 지표(metric)의 히스토그램 버퍼를 1회만 생성합니다. (클래스별 x범위 지원)

        - 기존 구조는 metric마다 edges/counts를 1개만 두었지만,
          여기서는 (vehicle/ped/bic)별로 max_edge가 다를 수 있으므로
          edges/counts를 클래스별로 따로 만듭니다.

        저장 형태:
            self._histograms[metric_name] = {
                "edges_per_class": [edges0, edges1, edges2]  # 각 (num_bins+1,)
                "counts_per_class": [cnt0, cnt1, cnt2]      # 각 (num_bins,)
                "bin_width": float
                "max_edge_per_class": (3,) tuple[float]
            }

        Args:
            metric_name: 지표 이름
            bin_width: bin 간격
            max_edge_per_class: (vehicle, pedestrian, bicycle) 최대 x 범위
            device: 텐서 디바이스
        """
        self._ensure_hist_state()

        if metric_name in self._histograms:
            # device가 달라졌으면 기존 버퍼를 새 device로 옮김
            edges_list: List[torch.Tensor] = self._histograms[metric_name]["edges_per_class"]
            counts_list: List[torch.Tensor] = self._histograms[metric_name]["counts_per_class"]
            if edges_list[0].device != device:
                self._histograms[metric_name]["edges_per_class"] = [e.to(device) for e in edges_list]
                self._histograms[metric_name]["counts_per_class"] = [c.to(device) for c in counts_list]
                if metric_name in self._hist_max_values:
                    self._hist_max_values[metric_name] = self._hist_max_values[metric_name].to(device)
            return

        bw = float(bin_width)
        max_edges = [float(max_edge_per_class[0]), float(max_edge_per_class[1]), float(max_edge_per_class[2])]

        edges_per_class: List[torch.Tensor] = []
        counts_per_class: List[torch.Tensor] = []

        for m in max_edges:
            # ceil로 bin 개수 결정 (max_edge가 bin_width의 배수가 아니어도 커버)
            num_bins = int(math.ceil(m / bw))
            num_bins = max(1, num_bins)

            edges = (torch.arange(num_bins + 1, device=device, dtype=torch.float32) * bw)
            # 마지막 edge는 요청 max_edge로 "딱 맞춤" (마지막 bin 폭이 조금 달라도 OK)
            edges[-1] = float(m)

            counts = torch.zeros((num_bins,), device=device, dtype=torch.long)
            edges_per_class.append(edges)
            counts_per_class.append(counts)

        self._histograms[metric_name] = {
            "edges_per_class": edges_per_class,
            "counts_per_class": counts_per_class,
            "bin_width": float(bw),
            "max_edge_per_class": (max_edges[0], max_edges[1], max_edges[2]),
        }

        self._hist_max_values[metric_name] = torch.full(
            (3,), float("-inf"), device=device, dtype=torch.float32
        )


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
        neighbor_future_gt_4_dim = neighbor_future_gt_11_dim[:, :, : , 0:4]  # (B, agent_num, future_len, 4)

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
        accel_mag_node = torch.sqrt(a_x_w * a_x_w + a_y_w * a_y_w)  # (B, agent_num, point_len)
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




    def _accumulate_metric_from_time_series(
        self,
        metric_name: str,
        values_bat: torch.Tensor,            # (B, agent_num, T)
        mask_bat: torch.Tensor,              # (B, agent_num, T) bool
        neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
        *,
        ignore_above: Optional[float] = None,
    ) -> None:
        """시간별 값 전체를 타입별로 모아 히스토그램 카운트만 누적합니다. (클래스별 x범위)

        Args:
            metric_name: 지표 이름
            values_bat: (B, agent_num, T)
            mask_bat: (B, agent_num, T) bool
            neighbor_agents_type: (B, agent_num, 3)
            ignore_above: 너무 큰 값 제외(예: r에서 분모가 거의 0인 경우)
        """
        if values_bat.dim() != 3:
            raise ValueError(f"values_bat must be (B,agent,T). got={tuple(values_bat.shape)}")
        if mask_bat.shape != values_bat.shape:
            raise ValueError(
                f"mask_bat must match values_bat shape. mask={tuple(mask_bat.shape)}, values={tuple(values_bat.shape)}"
            )
        if (neighbor_agents_type.dim() != 3
                or neighbor_agents_type.shape[:2] != values_bat.shape[:2]
                or neighbor_agents_type.shape[-1] != 3):
            raise ValueError(
                f"neighbor_agents_type must be (B,agent,3) and match values_bat[:2]. "
                f"type={tuple(neighbor_agents_type.shape)}, values={tuple(values_bat.shape)}"
            )

        # 스펙(요청한 bin_width / max_edge) 가져오기
        bin_width, max_edge_per_class = self._get_histogram_spec(metric_name)

        device = values_bat.device
        self._ensure_metric_histogram(
            metric_name=metric_name,
            bin_width=bin_width,
            max_edge_per_class=max_edge_per_class,
            device=device,
        )

        hist = self._histograms[metric_name]
        edges_list: List[torch.Tensor] = hist["edges_per_class"]    # len=3, each (num_bins+1,)
        counts_list: List[torch.Tensor] = hist["counts_per_class"]  # len=3, each (num_bins,)
        max_edges: Tuple[float, float, float] = hist["max_edge_per_class"]

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

            edges_c = edges_list[c]
            counts_c = counts_list[c]
            num_bins = int(counts_c.numel())
            max_edge_c = float(max_edges[c])

            # 히스토그램 누적 (0~max_edge 범위로 clamp)
            v32 = vals.clamp(min=0.0, max=max_edge_c - eps)
            idx = torch.bucketize(v32, edges_c, right=False) - 1
            idx = idx.clamp(min=0, max=num_bins - 1).to(torch.int64)

            add = torch.zeros((num_bins,), device=device, dtype=torch.long)
            add.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            counts_c += add

            # 참고용 최대값
            self._hist_max_values[metric_name][c] = torch.maximum(
                self._hist_max_values[metric_name][c],
                vals.max(),
            )

    @staticmethod
    def _draw_wave_break_mark(ax, *, y_axes: float) -> None:
        """축 경계에 '물결' 형태의 끊김 표시를 그립니다.

        Args:
            ax: matplotlib axis
            y_axes: 축 좌표계에서 y 위치 (0.0 또는 1.0)
        """
        try:
            import numpy as np
        except Exception:
            np = None

        # 좌/우에 작은 물결을 두 개 그립니다.
        span = 0.08
        amp = 0.015
        waves = 3

        def _plot_one_side(x_start: float) -> None:
            if np is None:
                # numpy가 없으면 간단한 지그재그로 대체
                xs = [x_start, x_start + span * 0.33, x_start + span * 0.66, x_start + span]
                ys = [y_axes, y_axes + amp, y_axes - amp, y_axes + amp]
            else:
                xs = np.linspace(x_start, x_start + span, 200)
                phase = np.linspace(0.0, 2.0 * math.pi * waves, xs.size)
                ys = y_axes + amp * np.sin(phase)
            ax.plot(xs, ys, transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)

        _plot_one_side(-0.02)         # left
        _plot_one_side(1.0 - span + 0.02)  # right

    @staticmethod
    def _save_histogram_percent_broken_y(
        *,
        plt,
        centers: List[float],
        heights_percent: List[float],
        widths: List[float],
        x_label: str,
        title: str,
        vlines: List[Tuple[float, str]],
        out_path: str,
        y_break: float = 0.5,
    ) -> None:
        """0~y_break 구간은 크게, y_break~100 구간은 작게 보이도록 히스토그램을 저장합니다.

        Args:
            plt: matplotlib.pyplot
            centers: x 위치 (len=N)
            heights_percent: y 값(%) (len=N)
            widths: 막대 폭 (len=N)
            x_label: x축 라벨
            title: 제목
            vlines: [(x, label), ...] 수직선들
            out_path: 저장 경로
            y_break: 끊김 기준 (기본 10%)
        """
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.05)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        for ax in (ax_top, ax_bot):
            ax.bar(centers, heights_percent, width=widths)
            for x, label in vlines:
                ax.axvline(x, linestyle="--", linewidth=1.2, label=label)

        ax_bot.set_ylim(0.0, float(y_break))
        ax_top.set_ylim(float(y_break), 100.0)

        # x tick은 아래만 보이게
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(axis="x", which="both", bottom=False)

        # 끊김 구간 표시(물결)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        DataStatistics._draw_wave_break_mark(ax_top, y_axes=0.0)
        DataStatistics._draw_wave_break_mark(ax_bot, y_axes=1.0)

        ax_bot.set_xlabel(x_label)
        ax_bot.set_ylabel("Percent(%)")
        ax_top.set_title(title)

        # legend는 위쪽에만(중복 방지)
        if len(vlines) > 0:
            ax_top.legend(fontsize="x-small", ncol=2, loc="upper right")

        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _accumulate_seg_body_control_statistics_robust(
            self,
            seg_body_control: torch.Tensor,  # (B, agent_num, T, 3)
            seg_valid: torch.Tensor,  # (B, agent_num, T) bool
            not_low_speed_seg: torch.Tensor,  # (B, agent_num, T) bool
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
    ) -> None:
        """v_b_y / v / a_lat / omega / r 을 '시간별 값 전체'로 누적합니다.

        변경점(중요):
            - a_lat_max는 저속 구간을 0으로 만들어 포함시키지 않고,
              저속 구간을 mask에서 제외합니다.
            - 즉, a_lat_max의 누적 마스크는 (seg_valid_stats & not_low_speed_seg) 입니다.

        Args:
            seg_body_control: (B, agent_num, T, 3)
                [v_x^b, v_y^b, omega] 세그먼트 중간값 제어.
            seg_valid: (B, agent_num, T) bool
                노드 양 끝이 유효한 세그먼트 마스크.
            not_low_speed_seg: (B, agent_num, T) bool
                저속이 아닌 세그먼트 마스크.
            neighbor_agents_type: (B, agent_num, 3)
                [vehicle, pedestrian, bicycle] 타입 정보.

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

        not_low_speed_seg = not_low_speed_seg.to(torch.bool)  # (B,agent,T)

        v_y_b = seg_body_control[..., 1]  # (B,agent,T)
        omega = seg_body_control[..., 2]  # (B,agent,T)

        v = torch.norm(seg_body_control[..., 0:2], dim=-1)  # (B,agent,T)
        a_lat = v * omega.abs()  # (B,agent,T)
        r = v / (omega.abs() + 1e-6)  # (B,agent,T)

        # v_b_y는 저속 구간을 "제외"해서 누적(기존 코드와 동일한 의미 유지)
        v_b_y_mask = seg_valid_stats & not_low_speed_seg

        self._accumulate_metric_from_time_series(
            metric_name="v_b_y_max",
            values_bat=v_y_b.abs(),  # (B,agent,T)
            mask_bat=v_b_y_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
        )

        # v는 저속 포함해도 큰 문제 없어서 seg_valid_stats만 사용(기존 유지)
        self._accumulate_metric_from_time_series(
            metric_name="v_max",
            values_bat=v,  # (B,agent,T)
            mask_bat=seg_valid_stats,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
        )

        # [FIX] a_lat_max: 저속 구간을 0으로 포함시키지 않고 mask에서 제외
        a_lat_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="a_lat_max",
            values_bat=a_lat,  # (B,agent,T)
            mask_bat=a_lat_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
        )

        # omega_max: 저속 제외(기존 유지)
        omega_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="omega_max",
            values_bat=omega.abs(),  # (B,agent,T)
            mask_bat=omega_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
        )

        # r_min: 저속 제외 + 큰 값 제외(기존 유지)
        r_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="r_min",
            values_bat=r,  # (B,agent,T)
            mask_bat=r_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
            ignore_above=1.0e5,
        )

    def _accumulate_seg_body_control_2_statistics_robust(
            self,
            seg_body_accel: torch.Tensor,  # (B, agent_num, T)
            seg_body_angular_accel: torch.Tensor,  # (B, agent_num, T)
            seg_valid: torch.Tensor,  # (B, agent_num, T) bool
            not_low_speed_seg: torch.Tensor,  # (B, agent_num, T) bool
            neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
    ) -> None:
        """a / alpha 를 '시간별 값 전체'로 누적합니다.

        변경점(중요):
            - alpha_max도 저속 구간을 0으로 포함시키지 않고,
              저속 구간을 mask에서 제외합니다.
            - 즉, alpha_max의 누적 마스크는 (seg_valid_stats & not_low_speed_seg) 입니다.

        Args:
            seg_body_accel: (B, agent_num, T)
                세그먼트 기준 가속도 크기(시간별).
            seg_body_angular_accel: (B, agent_num, T)
                세그먼트 기준 각가속도(시간별).
            seg_valid: (B, agent_num, T) bool
                유효 세그먼트 마스크.
            not_low_speed_seg: (B, agent_num, T) bool
                저속이 아닌 세그먼트 마스크.
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

        not_low_speed_seg = not_low_speed_seg.to(torch.bool)

        # a_max: 저속 제외(기존 유지)
        accel_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="a_max",
            values_bat=seg_body_accel.to(torch.float32),  # (B,agent,T)
            mask_bat=accel_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
        )

        # [FIX] alpha_max: 저속 제외(기존은 seg_valid_stats만 써서 0이 포함될 수 있었음)
        alpha_mask = seg_valid_stats & not_low_speed_seg
        self._accumulate_metric_from_time_series(
            metric_name="alpha_max",
            values_bat=seg_body_angular_accel.abs().to(torch.float32),
            # (B,agent,T)
            mask_bat=alpha_mask,  # (B,agent,T)
            neighbor_agents_type=neighbor_agents_type,
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

        현재 구조(클래스별 x범위):
            - metric마다 vehicle/pedestrian/bicycle의 x축 최대 범위가 달라서
              edges/counts를 클래스별로 따로 갖습니다.
            - 그래서 draw_histograms에서도 클래스별 edges/counts를 사용해
              퍼센트(%)와 퍼센트 지점 값들을 계산합니다.

        Returns:
            hist_data: metric_name -> {
                "edges_per_class": [edges_v, edges_p, edges_b]          # 각 (num_bins+1,)
                "counts_per_class": [counts_v, counts_p, counts_b]      # 각 (num_bins,)
                "percent_per_class": [percent_v, percent_p, percent_b]  # 각 (num_bins,)
                "raw_total_per_class": (3,) long
                "max_per_class": (3,) float
                "p_low_per_class": (3,) float
                "p_high_per_class": (3,) float
                "limit_percentiles": list[float]
                "limit_values_per_class": (3, K) float
                "bin_width": float
                "max_edge_per_class": (3,) float
            }
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
            # ===== 클래스별 edges/counts 가져오기 (CPU로) =====
            if "edges_per_class" not in hist or "counts_per_class" not in hist:
                raise ValueError(
                    f"[draw_histograms] metric={metric_name} is not class-specific histogram. "
                    f"expected keys: edges_per_class / counts_per_class"
                )

            edges_list_gpu: List[torch.Tensor] = hist["edges_per_class"]
            counts_list_gpu: List[torch.Tensor] = hist["counts_per_class"]

            if len(edges_list_gpu) != 3 or len(counts_list_gpu) != 3:
                raise ValueError(
                    f"[draw_histograms] metric={metric_name} edges/counts must have length 3. "
                    f"got edges={len(edges_list_gpu)}, counts={len(counts_list_gpu)}"
                )

            edges_per_class: List[torch.Tensor] = [e.detach().cpu() for e in edges_list_gpu]
            counts_per_class: List[torch.Tensor] = [c.detach().cpu() for c in counts_list_gpu]

            totals_list: List[int] = [int(c.sum().item()) for c in counts_per_class]
            totals = torch.tensor(totals_list, dtype=torch.long)  # (3,)

            # percent_per_class: 각 (num_bins,) 0~100
            percent_per_class: List[torch.Tensor] = []
            for c in range(3):
                denom = float(max(totals_list[c], 1))
                percent_per_class.append(counts_per_class[c].to(torch.float32) / denom * 100.0)

            # 참고용 최대값(각 클래스별로 1개)
            max_vals = self._hist_max_values.get(
                metric_name, torch.full((3,), float("nan"), device=edges_list_gpu[0].device)
            ).detach().cpu()  # (3,)

            # metric별 "한계 퍼센트" 목록
            if metric_name == "r_min":
                limit_pcts_actual = low_p_list_r
                limit_mode = "lower"
            else:
                limit_pcts_actual = high_p_list
                limit_mode = "higher"

            limit_qs = [float(p) / 100.0 for p in limit_pcts_actual]  # len=8

            p_low_list: List[float] = []
            p_high_list: List[float] = []
            limit_values_all: List[List[float]] = []

            for c in range(3):
                n = totals_list[c]
                if n <= 0:
                    p_low_list.append(float("nan"))
                    p_high_list.append(float("nan"))
                    limit_values_all.append([float("nan")] * len(limit_qs))
                    continue

                edges_c = edges_per_class[c]    # (num_bins+1,)
                counts_c = counts_per_class[c]  # (num_bins,)

                # p_low / p_high (근사)
                v_low = self._percentile_values_from_histogram(
                    edges=edges_c,
                    counts_1d=counts_c,
                    qs=[q_low],
                    mode="lower",
                )[0]
                v_high = self._percentile_values_from_histogram(
                    edges=edges_c,
                    counts_1d=counts_c,
                    qs=[q_high],
                    mode="higher",
                )[0]

                p_low_list.append(v_low)
                p_high_list.append(v_high)

                # 요청 퍼센트 값들(근사)
                vals_c = self._percentile_values_from_histogram(
                    edges=edges_c,
                    counts_1d=counts_c,
                    qs=limit_qs,
                    mode=limit_mode,
                )
                limit_values_all.append(vals_c)

            hist_data[metric_name] = {
                "edges_per_class": edges_per_class,
                "counts_per_class": counts_per_class,
                "percent_per_class": percent_per_class,
                "total_per_class": totals,
                "raw_total_per_class": totals.clone(),  # raw 저장이 없으니 totals로 대체
                "max_per_class": max_vals,
                "p_low_per_class": torch.tensor(p_low_list, dtype=torch.float32),
                "p_high_per_class": torch.tensor(p_high_list, dtype=torch.float32),
                "limit_percentiles": list(limit_pcts_actual),
                "limit_values_per_class": torch.tensor(limit_values_all, dtype=torch.float32),  # (3,8)
                "bin_width": float(hist.get("bin_width", float("nan"))),
                "max_edge_per_class": tuple(hist.get("max_edge_per_class", (float("nan"), float("nan"), float("nan")))),
            }

        # ===== 출력 =====
        print("========== robust limits (agent-time, softened extremes) ==========")
        metric_order = sorted(hist_data.keys())

        for j, p_hi in enumerate(high_p_list):
            p_r = low_p_list_r[j]
            print(f"[p{p_hi:g}] (r_min은 p{p_r:g} 값 사용)")

            for metric_name in metric_order:
                totals_raw = hist_data[metric_name]["raw_total_per_class"]      # (3,)
                limit_vals = hist_data[metric_name]["limit_values_per_class"]  # (3,8)

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
                edges_list = hist_data[metric_name]["edges_per_class"]         # len=3
                percent_list = hist_data[metric_name]["percent_per_class"]     # len=3
                totals_raw = hist_data[metric_name]["raw_total_per_class"]     # (3,)
                limit_vals = hist_data[metric_name]["limit_values_per_class"] # (3, 8)
                limit_pcts = hist_data[metric_name]["limit_percentiles"]       # len=8

                for c, cls_name in enumerate(class_names):
                    total = int(totals_raw[c].item())
                    if total <= 0:
                        continue

                    edges_c = edges_list[c]                # (num_bins+1,)
                    y_c = percent_list[c].tolist()         # (num_bins,)
                    centers = ((edges_c[:-1] + edges_c[1:]) * 0.5).tolist()
                    widths = (edges_c[1:] - edges_c[:-1]).tolist()
                    widths = [float(w) * 0.9 for w in widths]

                    vlines: List[Tuple[float, str]] = []
                    for jj, p in enumerate(limit_pcts):
                        x = float(limit_vals[c, jj].item())
                        if x == x:
                            vlines.append((x, f"p{p:g}"))

                    save_name = f"{metric_name}_{cls_name}_r{round_id:04d}.png"
                    out_path = os.path.join(out_dir, save_name)

                    x_label = self._get_metric_axis_label(metric_name)

                    DataStatistics._save_histogram_percent_broken_y(
                        plt=plt,
                        centers=centers,
                        heights_percent=y_c,
                        widths=widths,
                        x_label=x_label,
                        title=f"{x_label} / {cls_name} (N={total})",
                        vlines=vlines,
                        out_path=out_path,
                        y_break=0.5,
                    )

        # 누적 초기화
        self._histograms.clear()
        self._hist_max_values.clear()

        return hist_data
