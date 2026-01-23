from typing import Dict, Any, Tuple, Optional

import torch
from diffusion_planner.model.module.feasible import FeasibleProjector


class DataStatistics:

    def __init__(self, config):
        self.config = config
        self.feasible_projector = FeasibleProjector(
            self.config, self.config.hidden_dim, self.config.use_feasible_dl,
            self.config.use_feasible_filter)

        # SG(사비츠키-골레이)용 캐시: (window_len, polyorder, dt, dtype, device) -> (gram_per_k, power_per_k)
        self._sg_cache: Dict[Tuple[int, int, float, torch.dtype, str],
                             Tuple[torch.Tensor, torch.Tensor]] = {}
        self._sg_reg_eps: float = 1e-6
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

    def do_data_statistics(self, inputs: Dict[str, Any],
                           outputs: Dict[str, torch.Tensor]):
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

        (
            stride_step,  # int # 정수 스트라이드 (1이면 다운샘플링 없음).
            dt_for_savgol,  # float #  SG 필터에 넘길 샘플 간 시간 간격 [초].
            max_window_len_xy,  # int # x,y 좌표에 사용할 최대 윈도 길이(샘플 수).
            max_window_len_yaw,  # int # yaw에 사용할 최대 윈도 길이(샘플 수).
        ) = self.feasible_projector.get_feasible_stride_params(future_len)

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
        points_world_control = (
            self.feasible_projector.savgol_filter_for_control(
                diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
                past_xyyaw,
                #  (B, agent_num, past_len(=time_len -1), 4)
                neighbor_agents_past_cur_future_is_valid,
                # (B,Pnn,past_len+1+future_len) bool
                dt=dt_for_savgol,
                polyorder=2,
                max_window_len_xy=max_window_len_xy,
                max_window_len_yaw=max_window_len_yaw,
            ))
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
        (abs_v_y_b_max, abs_v_max, abs_omega_max,  abs_a_y_lat_max, r_min
        ) = self._get_seg_body_control_statistics(seg_body_control, seg_valid)
        self._set_seg_body_control_statistics(abs_v_y_b_max, abs_v_max,
                                              abs_omega_max, abs_a_y_lat_max, r_min,
                                              neighbor_agents_type)

        # (B, agent_num, past_len + future_len)
        seg_body_v = torch.norm(seg_body_control[:, :, :, 0:2], dim=-1)
        # (B, agent_num, past_len + future_len)
        seg_body_omega = seg_body_control[:, :, :, 2]
        # =========================================================
        # seg_body_v, seg_body_omega -> accel, angular_accel
        # =========================================================


        # 무효점 처리 "강검증": 0*1*0* 형태(구멍 없음)만 허용
        self._assert_mask_no_hole_0_1_0(seg_valid,
                                        context="do_data_statistics.seg_valid")

        # v, omega의 1차 미분 = accel, angular_accel
        # (B, agent_num, past_len + future_len)
        seg_body_accel = self._savgol_first_derivative_masked_1d(
            seq_bat=seg_body_v,  # (B, agent_num, T)
            valid_bat=seg_valid,  # (B, agent_num, T)
            dt=float(dt_for_savgol),
            polyorder=2,
            max_window_length=int(max_window_len_xy),
        )
        # (B, agent_num, past_len + future_len)
        seg_body_angular_accel = self._savgol_first_derivative_masked_1d(
            seq_bat=seg_body_omega,  # (B, agent_num, T)
            valid_bat=seg_valid,  # (B, agent_num, T)
            dt=float(dt_for_savgol),
            polyorder=2,
            max_window_length=int(max_window_len_yaw),
        )
        (seg_body_accel_max, seg_body_angular_accel_max) \
            = self._get_seg_body_control_2_statistics(seg_body_accel, seg_body_angular_accel, seg_valid)
        self._set_seg_body_control_2_statistics(seg_body_accel_max, seg_body_angular_accel_max,
                                              neighbor_agents_type)

    def _get_seg_body_control_2_statistics(self,
                                           seg_body_accel: torch.Tensor,  # (B, agent_num, past_len + future_len)
                                           seg_body_angular_accel: torch.Tensor,  # (B, agent_num, past_len + future_len)
                                           seg_valid: torch.Tensor,  # (B, agent_num, past_len + future_len)
                                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ seg_body_accel: (B, agent_num, past_len + future_len)
            seg_body_angular_accel: (B, agent_num, past_len + future_len)
        returns:
            seg_body_accel_max : (B, agent_num)
            seg_body_angular_accel_max : (B, agent_num)
        """
        B, agent_num, total_len = seg_body_accel.shape
        # seg_body_accel_max
        seg_body_accel = seg_body_accel * seg_valid.to(seg_body_accel.dtype)
        seg_body_accel_max, _ = torch.max(torch.abs(seg_body_accel),
                                         dim=2)  # (B, agent_num)
        # seg_body_angular_accel_max
        seg_body_angular_accel = seg_body_angular_accel * seg_valid.to(
            seg_body_angular_accel.dtype)
        seg_body_angular_accel_max, _ = torch.max(torch.abs(
            seg_body_angular_accel), dim=2)  # (B, agent_num)
        return seg_body_accel_max, seg_body_angular_accel_max

    def _set_seg_body_control_2_statistics(self,
        seg_body_accel_max: torch.Tensor,  # (B, agent_num)
        seg_body_angular_accel_max: torch.Tensor,  # (B, agent_num)
        neighbor_agents_type: torch.Tensor  # (B, agent_num, 3) # (vehicle, pedestrian, bicycle)
                                           ):
        """
        a_max
            - 1m/s^2 단위로 통계 히스토그램 그리자 (0~2, 2~4, 4~6, ..., 28~30, ...)
        alpha_max
            - 1 rad/s^2 단위로 통계 히스토그램 그리자 (0~1, 1~2, 2~3, ..., 18~19, 19~20, ...)

        _set_seg_body_control_2_statistics 가 여러번 계속 호출될 수 있으므로,
        누적 통계로 관리할 수 있도록 코딩 부탁.

        배치 별 최댓값을 구해서 통계 값에 반영하는 방식으로 부탁. (에이전트 별 데이터를 쌓는게 아니라)


        결과적으로는 모든 _set_seg_body_control_2_statistics 의 호출이 끝났을 떄,
            a_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            alpha_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            해서 총 3 * 2 개의 히스토그램 그림이 완성되도록 부탁.
        그림 그리는 함수는 나중에 따로 구현할 예정이므로, 여기서는 히스토그램 데이터 누적만 잘 처리해주시면 됨.

        데이터가 쌓여서, 메모리가 방대해지지 않는 방식으로 코딩 부탁.

        그리고 추가로, vehicle, pedestrian, bicycle 별로 각각,
        a_max / alpha_max 의 스칼라 최대값(모든 agent중 최댓값) 도
        별도로 기록해주시면 감사하겠습니다. (역대 모든 배치에서의 최댓값들 기록 (배치별 최대값 말고, 지금까지 모든 배치 중 최댓값 1개))
        . 구간별 값을 기록하는게 아니라 정확한 수치 최대값 하나만 기록)

        히스토그램에는 세로축이 빈도수가 아니라 , % 비율(0~100%)로 표시되도록 고려 부탁드립니다. (히스토그램 그리는 함수에서 처리 예정)
        """

        pass

    def _get_seg_body_control_statistics(
        self, seg_body_control: torch.Tensor, # (B, agent_num, past_len + future_len, 3)
            seg_valid: torch.Tensor, # (B, agent_num, past_len + future_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ seg_body_control: (B, agent_num, past_len + future_len, 3)
            [v_x^b, v_y^b, ω]_mid

        returns:
            abs(v_y^b)_max : (B, agent_num)
            abs(v)_max : (B, agent_num)
            abs(ω)_max : (B, agent_num)
            abs(a_y_lat)_max : (B, agent_num)
            R_min : (B, agent_num)
        """
        B, agent_num, total_len, _ = seg_body_control.shape

        v_y_b = seg_body_control[:, :, :, 1]  # (B, agent_num, total_len)
        omega = seg_body_control[:, :, :, 2]  # (B, agent_num, total_len)

        abs_v_y_b_max, _ = torch.max(torch.abs(v_y_b), dim=2)  # (B, agent_num)
        v = torch.norm(seg_body_control[:, :, :, 0:2],
                       dim=-1)  # (B, agent_num, total_len)
        abs_v_max, _ = torch.max(torch.abs(v), dim=2)  # (B, agent_num)
        abs_omega_max, _ = torch.max(torch.abs(omega), dim=2)  # (B, agent_num)
        # abs_a_y_lat_max
        a_y_lat = v * torch.abs(omega)  # (B, agent_num, total_len)
        abs_a_y_lat_max, _ = torch.max(torch.abs(a_y_lat), dim=2)  # (B, agent_num)
        # R_min
        r_min = v / torch.abs(omega + 1e-6) # (B, agent_num, total_len)
        # apply seg_valid at r_min
        r_min = r_min * seg_valid.to(r_min.dtype) + (1.0e6) * (
            1 - seg_valid.to(r_min.dtype))  # invalid인 곳은 매우 큰 값으로 대체
        r_min, _ = torch.min(r_min, dim=2)  # (B, agent_num)
        return abs_v_y_b_max, abs_v_max, abs_omega_max, abs_a_y_lat_max, r_min

    def _set_seg_body_control_statistics(
        self,
        abs_v_y_b_max: torch.Tensor,  # (B, agent_num)
        abs_v_max: torch.Tensor,  # (B, agent_num)
        abs_omega_max: torch.Tensor,  # (B, agent_num)
        abs_a_y_lat_max: torch.Tensor,  # (B, agent_num)
        r_min: torch.Tensor,  # (B, agent_num)
        neighbor_agents_type: torch.
        Tensor  # (B, agent_num, 3) # (vehicle, pedestrian, bicycle)
    ):
        """
        v_b_y_max
            - 0.1 m/s 단위로 통계 히스토그램 그리자 (0~0.1, 0.1~0.2, 0.2~0.3, ..., 29.9~30.0, ...)
        v_max
            - 3m/s 단위로 통계 히스토그램 그리자 (0~3, 3~6, 6~9, 9~12, 12~15, 15~18, 18~21, 21~24, 24~27, 27~30, ...)
        a_lat_max
            - 1m/s^2 단위로 통계 히스토그램 그리자 (0~2, 2~4, 4~6, ..., 28~30, ...)
            - v_abs * omega_abs 으로 계산
        r_min
            - 1m 단위로 통계 히스토그램 그리자 (0~1, 1~2, 2~3, ..., 98~99, 99~100, ...)
            - v_abs / omega_abs 으로 계산
        omega_max
            - 0.3 rad/s 단위로 통계 히스토그램 그리자 (0~0.3, 0.3~0.6, 0.6~0.9, ..., 8.7~9.0, 9.0~9.3, ...)

        _get_seg_body_control_statistics 가 여러번 계속 호출될 수 있으므로,
        누적 통계로 관리할 수 있도록 코딩 부탁.

        배치 별 최댓값을 구해서 통계 값에 반영하는 방식으로 부탁. (에이전트 별 데이터를 쌓는게 아니라)


        결과적으로는 모든 _get_seg_body_control_statistics 의 호출이 끝났을 떄,
            v_b_y_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            v_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            a_lat_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            r_min 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            omega_max 에 vehicle, pedestrian, bicycle 각각에 대한 히스토그램 총 3개 그림을,
            해서 총 3 * 5 개의 히스토그램 그림이 완성되도록 부탁.
        그림 그리는 함수는 나중에 따로 구현할 예정이므로, 여기서는 히스토그램 데이터 누적만 잘 처리해주시면 됨.

        데이터가 쌓여서, 메모리가 방대해지지 않는 방식으로 코딩 부탁.

        그리고 추가로, vehicle, pedestrian, bicycle 별로 각각,
        v_b_y_max / v_max / a_lat_max / r_min / omega_max 의 스칼라 최대값(모든 agent중 최댓값) 도
        별도로 기록해주시면 감사하겠습니다. (역대 모든 배치에서의 최댓값들 기록 (배치별 최대값 말고, 지금까지 모든 배치 중 최댓값 1개))
        . 구간별 값을 기록하는게 아니라 정확한 수치 최대값 하나만 기록)

        히스토그램에는 세로축이 빈도수가 아니라 , % 비율(0~100%)로 표시되도록 고려 부탁드립니다. (히스토그램 그리는 함수에서 처리 예정)
        """

        pass

    def draw_histograms(self):
        """모든 배치에 대한 통계가 누적된 후, 히스토그램을 그립니다.

        호출 시점:
            - 모든 배치에 대한 데이터 처리가 끝난 후, epoch 단위로 호출됩니다.

        히스토그램 그리기:
            - 각 통계 항목별로 vehicle, pedestrian, bicycle 에 대한 히스토그램을 그립니다.
            - 히스토그램의 세로축은 빈도수가 아닌 % 비율(0~100%)로 표시됩니다.
            - 히스토그램은 별도의 시각화 도구나 라이브러리를 사용하여 그릴 수 있습니다.

        메모리 관리:
            - 히스토그램을 그린 후에는 누적된 통계 데이터를 초기화하여 메모리를 관리합니다.

        추가로, 각 항목별로 vehicle, pedestrian, bicycle 에 대한 스칼라 최대값도 출력합니다.
        """

        pass

# =========================================================
# [NEW] Mask 검증/SG 미분 유틸 (DataStatistics 내부 구현)
# =========================================================

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

    @staticmethod
    def _finite_difference_1d_masked(
        seq_nt: torch.Tensor,  # (N, T)
        valid_nt: torch.Tensor,  # (N, T) bool
        dt: float,
    ) -> torch.Tensor:
        """마스크를 고려한 1D 유한차분(기본 미분값)입니다.

        규칙(중심 시점 기준):
            - 중심이 invalid면 항상 0
            - 중심이 valid일 때만 이웃을 사용
              * 양옆 valid면 중앙차분
              * 한쪽만 valid면 그 방향 차분
              * 양옆 모두 invalid면 0

        Args:
            seq_nt: (N, T)
            valid_nt: (N, T) bool
            dt: float

        Returns:
            (N, T) 미분값
        """
        N, T = seq_nt.shape
        out = torch.zeros_like(seq_nt)

        if T <= 1:
            return out

        valid = valid_nt.to(torch.bool)

        if T == 2:
            both = (valid[:, 0] & valid[:, 1])
            diff = (seq_nt[:, 1] - seq_nt[:, 0]) / float(dt)
            diff = diff * both.to(seq_nt.dtype)
            out[:, 0] = diff
            out[:, 1] = diff
            return out

        # t=0 (forward)
        v0 = valid[:, 0] & valid[:, 1]
        out[:, 0] = (
            (seq_nt[:, 1] - seq_nt[:, 0]) / float(dt)) * v0.to(seq_nt.dtype)

        # t=T-1 (backward)
        vL = valid[:, -1] & valid[:, -2]
        out[:, -1] = (
            (seq_nt[:, -1] - seq_nt[:, -2]) / float(dt)) * vL.to(seq_nt.dtype)

        # middle (1..T-2)
        c = valid[:, 1:-1]
        l = valid[:, :-2]
        r = valid[:, 2:]

        both = c & l & r
        only_l = c & l & (~r)
        only_r = c & (~l) & r

        both_f = both.to(seq_nt.dtype)
        only_l_f = only_l.to(seq_nt.dtype)
        only_r_f = only_r.to(seq_nt.dtype)

        central = (seq_nt[:, 2:] - seq_nt[:, :-2]) / (2.0 * float(dt))
        backward = (seq_nt[:, 1:-1] - seq_nt[:, :-2]) / float(dt)
        forward = (seq_nt[:, 2:] - seq_nt[:, 1:-1]) / float(dt)

        out[:,
            1:-1] = both_f * central + only_l_f * backward + only_r_f * forward
        return out

    @staticmethod
    def _select_odd_window_length(sequence_length: int,
                                  max_window_length: int) -> int:
        """SG에 쓸 실제 window length를 고릅니다(홀수 강제).

        Args:
            sequence_length: T
            max_window_length: 최대 window

        Returns:
            홀수 window 길이(최소 1)
        """
        w = int(min(sequence_length, int(max_window_length)))
        if w <= 0:
            return 1
        if w % 2 == 0:
            w -= 1
        return max(1, w)

    def _get_sg_gram_and_power(
        self,
        window_length: int,
        polyorder: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SG 계산에 필요한 (gram_per_k, power_per_k)를 캐시로 관리합니다.

        Args:
            window_length: W (홀수)
            polyorder: P
            dt: 샘플 간 시간 간격
            device: 텐서 디바이스
            dtype: 텐서 dtype

        Returns:
            gram_per_k: (W, P+1, P+1)
            power_per_k: (W, P+1)
        """
        dt_key = float(round(float(dt), 8))
        cache_key = (int(window_length), int(polyorder), dt_key, dtype,
                     str(device))

        hit = self._sg_cache.get(cache_key, None)
        if hit is not None:
            return hit

        W = int(window_length)
        P1 = int(polyorder) + 1
        half = W // 2

        tau = (torch.arange(W, device=device, dtype=dtype) -
               float(half)) * float(dt)  # (W,)
        power_cols = [torch.ones_like(tau)]
        for k in range(1, P1):
            power_cols.append(tau**k)
        power_per_k = torch.stack(power_cols, dim=-1)  # (W, P+1)

        gram_per_k = power_per_k.unsqueeze(-1) * power_per_k.unsqueeze(
            -2)  # (W, P+1, P+1)

        self._sg_cache[cache_key] = (gram_per_k, power_per_k)
        return gram_per_k, power_per_k

    def _savgol_first_derivative_masked_1d(
        self,
        seq_bat: torch.Tensor,  # (B, agent_num, T)
        valid_bat: torch.Tensor,  # (B, agent_num, T) bool
        *,
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """무효점을 절대 섞지 않고, 1차 미분(변화율)을 계산합니다.

        처리 방식(핵심만):
            1) 기본값은 "마스크 인지 유한차분"으로 만듭니다.
            2) 그 다음, 주변 여러 점을 한꺼번에 보는 SG(다항식 맞추기)로
               더 안정적인 미분을 낼 수 있는 위치만 골라서 덮어씁니다.
            3) 중심 시점이 invalid이면 결과는 무조건 0입니다.

        Args:
            seq_bat: (B, agent_num, T)
            valid_bat: (B, agent_num, T) bool
            dt: float
            polyorder: int (권장 2)
            max_window_length: int (홀수로 자동 조정됨)

        Returns:
            (B, agent_num, T) 1차 미분 결과
        """
        if seq_bat.dim() != 3:
            raise ValueError(
                f"seq_bat must be (B,agent,T), got shape={tuple(seq_bat.shape)}"
            )
        if valid_bat.shape != seq_bat.shape:
            raise ValueError(
                f"valid_bat shape must match seq_bat. {tuple(valid_bat.shape)} vs {tuple(seq_bat.shape)}"
            )

        B, A, T = seq_bat.shape
        N = B * A
        device = seq_bat.device
        dtype = seq_bat.dtype

        seq_nt = seq_bat.reshape(N, T)  # (N, T)
        valid_nt = valid_bat.reshape(N, T).to(torch.bool)  # (N, T)

        # 1) 기본 미분(마스크 인지)
        fd = self._finite_difference_1d_masked(seq_nt=seq_nt,
                                               valid_nt=valid_nt,
                                               dt=float(dt))  # (N, T)

        # T가 너무 짧으면 여기서 끝
        if T <= 2:
            out = torch.where(valid_nt, fd, torch.zeros_like(fd))
            return out.reshape(B, A, T)

        # 2) SG 준비
        window_length = self._select_odd_window_length(T,
                                                       int(max_window_length))
        min_samples = int(polyorder) + 2
        if window_length < min_samples:
            out = torch.where(valid_nt, fd, torch.zeros_like(fd))
            return out.reshape(B, A, T)

        half = window_length // 2

        # mask_window: (N, T, W), valid_count: (N, T)
        valid_float = valid_nt.to(dtype=dtype)
        valid_padded = F.pad(valid_float, (half, half),
                             mode="constant",
                             value=0.0)  # (N, T+2*half)
        mask_window = valid_padded.unfold(dimension=-1,
                                          size=window_length,
                                          step=1)  # (N, T, W)
        valid_count = mask_window.sum(dim=-1)  # (N, T)

        gram_per_k, power_per_k = self._get_sg_gram_and_power(
            window_length=window_length,
            polyorder=int(polyorder),
            dt=float(dt),
            device=device,
            dtype=dtype,
        )  # gram: (W,P+1,P+1), power: (W,P+1)

        P1 = int(polyorder) + 1

        # A_all: (N, T, P+1, P+1)
        A_all = (mask_window.unsqueeze(-1).unsqueeze(-1) *
                 gram_per_k.view(1, 1, window_length, P1, P1)).sum(dim=2)

        # b_all: (N, T, P+1)
        seq_padded = F.pad(seq_nt, (half, half), mode="constant",
                           value=0.0)  # (N, T+2*half)
        seq_window = seq_padded.unfold(dimension=-1, size=window_length,
                                       step=1)  # (N, T, W)
        weighted = seq_window * mask_window  # (N, T, W)

        b_all = (weighted.unsqueeze(-1) *
                 power_per_k.view(1, 1, window_length, P1)).sum(
                     dim=2)  # (N, T, P+1)

        # 3) "SG로 풀어도 되는 위치"만 선택
        good_mask = valid_nt & (valid_count >= float(min_samples))  # (N, T)

        # 너무 양 끝은(수치 흔들림 방지) SG를 쓰지 않고 FD로 둠
        edge_margin = 2
        if T > 2 * edge_margin:
            good_mask[:, :edge_margin] = False
            good_mask[:, -edge_margin:] = False

        out = fd.clone()  # (N, T)

        if good_mask.any():
            idx = good_mask.reshape(-1).nonzero(as_tuple=False).squeeze(
                -1)  # (M,)

            A_flat = A_all.reshape(-1, P1, P1)[idx]  # (M, P1, P1)
            b_flat = b_all.reshape(-1, P1)[idx]  # (M, P1)

            I = torch.eye(P1, device=device,
                          dtype=dtype).unsqueeze(0)  # (1, P1, P1)
            A_reg = A_flat + float(self._sg_reg_eps) * I  # (M, P1, P1)

            # coeff: (M, P1, 1) -> (M, P1)
            coeff = torch.linalg.solve(A_reg, b_flat.unsqueeze(-1)).squeeze(-1)
            sg_deriv = coeff[:, 1]  # 1차항 계수 = 1차 미분값 (M,)

            out_flat = out.reshape(-1)
            out_flat[idx] = sg_deriv
            out = out_flat.reshape(N, T)

        # 4) 중심이 invalid인 곳은 무조건 0
        out = torch.where(valid_nt, out, torch.zeros_like(out))
        return out.reshape(B, A, T)

    # =========================================================
    # [NEW] 히스토그램 누적/그리기 유틸 (DataStatistics 내부)
    # =========================================================

    def _ensure_hist_state(self) -> None:
        """히스토그램 누적에 필요한 내부 저장소를 준비합니다.

        - 배치가 여러 번 들어와도, 값 전체를 저장하지 않습니다.
        - 대신 "구간별 개수(counts)"만 누적해서 메모리가 커지지 않게 합니다.

        Returns:
            None
        """
        if not hasattr(self, "_histograms"):
            # metric_name -> {"edges": (num_bins+1,), "counts": (3,num_bins), ...}
            self._histograms: Dict[str, Dict[str, Any]] = {}

        if not hasattr(self, "_hist_max_values"):
            # metric_name -> (3,)  타입별(veh/ped/bic) 역대 최대값
            self._hist_max_values: Dict[str, torch.Tensor] = {}

        if not hasattr(self, "_hist_draw_round"):
            # draw_histograms 호출 횟수(파일명 중복 방지용)
            self._hist_draw_round: int = 0


    def _ensure_metric_histogram(self, metric_name: str, *, bin_width: float,
                                max_edge: float) -> None:
        """특정 지표(metric)의 히스토그램 버퍼를 1회만 생성합니다.

        Args:
            metric_name: 지표 이름(예: "v_max")
            bin_width: 구간 폭(예: 0.1)
            max_edge: 마지막 경계값(예: 30.0이면 마지막 구간은 29.9~30.0)

        내부에 생성되는 텐서:
            - edges: (num_bins+1,)  구간 경계값들
            - counts: (3, num_bins)  타입별 누적 개수 (0:vehicle, 1:pedestrian, 2:bicycle)

        Returns:
            None
        """
        self._ensure_hist_state()

        if metric_name in self._histograms:
            return

        num_bins = int(round(float(max_edge) / float(bin_width)))
        num_bins = max(1, num_bins)

        # edges: (num_bins+1,)
        edges = torch.arange(num_bins + 1,
                             dtype=torch.float32) * float(bin_width)
        # counts: (3, num_bins)
        counts = torch.zeros((3, num_bins), dtype=torch.long)

        self._histograms[metric_name] = {
            "edges": edges,
            "counts": counts,
            "bin_width": float(bin_width),
            "max_edge": float(max_edge),
        }
        self._hist_max_values[metric_name] = torch.zeros((3,), dtype=torch.float32)


    def _add_values_to_histogram(self, metric_name: str, class_index: int,
                                 values_1d: torch.Tensor) -> None:
        """값(1차원)을 히스토그램 구간에 맞춰 누적합니다.

        - values_1d는 저장하지 않고, "구간별 개수"만 올립니다.
        - 값이 범위를 벗어나면, 히스토그램 메모리가 커지지 않도록
          [0, max_edge] 범위로 잘라서 마지막 구간에 모이도록 합니다.

        Args:
            metric_name: 지표 이름
            class_index: 0(vehicle), 1(pedestrian), 2(bicycle)
            values_1d: (N,) 값 목록

        Returns:
            None
        """
        if values_1d.numel() == 0:
            return

        hist = self._histograms[metric_name]
        edges: torch.Tensor = hist["edges"]  # (num_bins+1,)
        counts: torch.Tensor = hist["counts"]  # (3, num_bins)
        max_edge = float(hist["max_edge"])
        num_bins = int(counts.shape[1])

        # CPU float32로 옮겨서 가볍게 처리 (N이 매우 작아야 함: 배치당 타입별 1개)
        v = values_1d.detach().to(torch.float32).cpu()  # (N,)

        # max_edge에 정확히 걸리면 마지막 구간 밖으로 나갈 수 있어 아주 조금 빼줌
        eps = 1.0e-6
        v = v.clamp(min=0.0, max=max_edge - eps)

        # 각 값이 속한 구간 인덱스 계산
        idx = torch.bucketize(v, edges, right=False) - 1  # (N,)
        idx = idx.clamp(min=0, max=num_bins - 1).to(torch.int64)

        add = torch.bincount(idx, minlength=num_bins)  # (num_bins,)
        counts[class_index] += add


    def _accumulate_metric_by_batch_max(
        self,
        metric_name: str,
        values_ba: torch.Tensor,  # (B, agent_num)
        neighbor_agents_type: torch.Tensor,  # (B, agent_num, 3)
        *,
        bin_width: float,
        max_edge: float,
        ignore_above: Optional[float] = None,
    ) -> None:
        """배치(현재 호출 1회) 기준으로 타입별 최댓값 1개만 누적합니다.

        핵심 규칙:
            - 에이전트별 값을 전부 쌓지 않습니다.
            - 현재 배치에서, 타입별로 (B, agent_num) 전체를 통틀어 최댓값 1개를 구합니다.
            - 그 최댓값 1개만 히스토그램에 누적합니다.
            - 동시에 "역대(지금까지) 최대값"도 타입별로 갱신합니다.

        Args:
            metric_name: 지표 이름
            values_ba: (B, agent_num)
            neighbor_agents_type: (B, agent_num, 3)  (vehicle, pedestrian, bicycle) 원-핫 또는 0/1
            bin_width: 히스토그램 구간 폭
            max_edge: 히스토그램 마지막 경계값
            ignore_above: 이 값보다 큰 값은 "통계에서 제외" (예: r_min의 1e6 같은 비정상/센티넬 방지용)

        Returns:
            None
        """
        self._ensure_metric_histogram(metric_name,
                                      bin_width=float(bin_width),
                                      max_edge=float(max_edge))

        # type_mask: (B, agent_num, 3) bool
        type_mask = neighbor_agents_type.detach() > 0.5
        values = values_ba.detach()  # (B, agent_num)

        if torch.is_floating_point(values):
            fill_val = float(torch.finfo(values.dtype).min)
        else:
            fill_val = -1e9

        for c in range(3):
            # mask: (B, agent_num)
            mask = type_mask[:, :, c]

            if ignore_above is not None:
                mask = mask & (values < float(ignore_above))

            if not bool(mask.any()):
                continue

            # 마스크 바깥은 아주 작은 값으로 채워 max에 영향을 못 주게 함
            masked = values.masked_fill(~mask, fill_val)  # (B, agent_num)
            batch_max = masked.max()  # scalar

            batch_max_f = float(batch_max.item())
            if batch_max_f > float(self._hist_max_values[metric_name][c].item()):
                self._hist_max_values[metric_name][c] = batch_max_f

            # 히스토그램에는 배치당 1개만 누적
            self._add_values_to_histogram(metric_name, c, batch_max.view(1))


    # =========================================================
    # [IMPLEMENT] _set_seg_body_control_2_statistics
    # =========================================================
    def _set_seg_body_control_2_statistics(
        self,
        seg_body_accel_max: torch.Tensor,  # (B, agent_num)
        seg_body_angular_accel_max: torch.Tensor,  # (B, agent_num)
        neighbor_agents_type: torch.Tensor  # (B, agent_num, 3)
    ) -> None:
        """a_max / alpha_max 히스토그램 누적(배치별 최댓값 기반) + 역대 최대값 기록.

        구현 방식(요청사항 반영):
            - 이 함수가 여러 번 호출돼도 누적 통계가 유지됩니다.
            - "에이전트별 값 전체"를 쌓지 않습니다.
            - 현재 배치에서 타입별(vehicle/ped/bic) 최댓값 1개만 뽑아 히스토그램에 넣습니다.
            - 동시에, 지금까지 모든 배치에서의 타입별 "역대 최대값"도 1개씩 유지합니다.

        히스토그램 구간:
            - a_max: 0~2, 2~4, ..., 28~30  (구간 폭 2, 최대 30)
            - alpha_max: 0~1, 1~2, ..., 19~20 (구간 폭 1, 최대 20)

        Args:
            seg_body_accel_max: (B, agent_num)
            seg_body_angular_accel_max: (B, agent_num)
            neighbor_agents_type: (B, agent_num, 3)

        Returns:
            None
        """
        # a_max
        self._accumulate_metric_by_batch_max(
            metric_name="a_max",
            values_ba=seg_body_accel_max,  # (B, agent_num)
            neighbor_agents_type=neighbor_agents_type,  # (B, agent_num, 3)
            bin_width=2.0,
            max_edge=30.0,
        )

        # alpha_max
        self._accumulate_metric_by_batch_max(
            metric_name="alpha_max",
            values_ba=seg_body_angular_accel_max,  # (B, agent_num)
            neighbor_agents_type=neighbor_agents_type,  # (B, agent_num, 3)
            bin_width=1.0,
            max_edge=20.0,
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
    ) -> None:
        """v_b_y_max / v_max / a_lat_max / r_min / omega_max 누적 통계 관리.

        구현 방식(요청사항 반영):
            - 이 함수가 여러 번 호출돼도 누적 통계가 유지됩니다.
            - "에이전트별 값 전체"를 쌓지 않습니다.
            - 현재 배치(현재 호출 1회)에서 타입별 최댓값 1개만 히스토그램에 넣습니다.
            - 동시에, 지금까지 모든 배치에서의 타입별 "역대 최대값"도 기록합니다.

        히스토그램 구간:
            - v_b_y_max: 0~0.1, 0.1~0.2, ..., 29.9~30.0  (구간 폭 0.1, 최대 30)
            - v_max: 0~3, 3~6, ..., 27~30  (구간 폭 3, 최대 30)
            - a_lat_max: 0~2, 2~4, ..., 28~30 (구간 폭 2, 최대 30)
            - r_min: 0~1, 1~2, ..., 99~100 (구간 폭 1, 최대 100)
            - omega_max: 0~0.3, 0.3~0.6, ..., 9.0~9.3 (구간 폭 0.3, 최대 9.3)

        주의:
            - r_min은 계산 과정에서 매우 큰 값(예: 1e6)이 들어갈 수 있어,
              ignore_above=1e5를 두어 비정상적으로 큰 값은 통계에서 제외합니다.

        Args:
            abs_v_y_b_max: (B, agent_num)
            abs_v_max: (B, agent_num)
            abs_omega_max: (B, agent_num)
            abs_a_y_lat_max: (B, agent_num)
            r_min: (B, agent_num)
            neighbor_agents_type: (B, agent_num, 3)

        Returns:
            None
        """
        self._accumulate_metric_by_batch_max(
            metric_name="v_b_y_max",
            values_ba=abs_v_y_b_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.1,
            max_edge=30.0,
        )

        self._accumulate_metric_by_batch_max(
            metric_name="v_max",
            values_ba=abs_v_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=3.0,
            max_edge=30.0,
        )

        self._accumulate_metric_by_batch_max(
            metric_name="a_lat_max",
            values_ba=abs_a_y_lat_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=2.0,
            max_edge=30.0,
        )

        self._accumulate_metric_by_batch_max(
            metric_name="r_min",
            values_ba=r_min,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=1.0,
            max_edge=100.0,
            ignore_above=1.0e5,
        )

        self._accumulate_metric_by_batch_max(
            metric_name="omega_max",
            values_ba=abs_omega_max,
            neighbor_agents_type=neighbor_agents_type,
            bin_width=0.3,
            max_edge=9.3,
        )


    # =========================================================
    # [IMPLEMENT] draw_histograms
    # =========================================================
    def draw_histograms(self) -> None:
        """모든 배치에 대한 통계가 누적된 후, 히스토그램을 그립니다.

        구현 방식:
            - 누적된 counts를 이용해, 각 구간의 비율(%)을 계산하여 막대 그래프로 저장합니다.
            - 저장 후에는 누적 통계를 초기화해서 메모리를 관리합니다.
            - 추가로, 각 지표/타입별 역대 최대값도 출력합니다.

        저장 위치:
            - 기본: "./histograms"
            - config에 histogram_output_dir 속성이 있으면 그 값을 사용합니다.

        Returns:
            None
        """
        self._ensure_hist_state()

        if len(self._histograms) == 0:
            print("[draw_histograms] 누적된 통계가 없습니다.")
            return

        import os
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("draw_histograms를 사용하려면 matplotlib가 필요합니다.") from e

        out_dir = getattr(self.config, "histogram_output_dir", "histograms")
        os.makedirs(out_dir, exist_ok=True)

        self._hist_draw_round += 1
        round_id = int(self._hist_draw_round)

        class_names = ["vehicle", "pedestrian", "bicycle"]
        metric_order = [
            "v_b_y_max",
            "v_max",
            "a_lat_max",
            "r_min",
            "omega_max",
            "a_max",
            "alpha_max",
        ]

        # 히스토그램 저장
        for metric_name in metric_order:
            if metric_name not in self._histograms:
                continue

            hist = self._histograms[metric_name]
            edges: torch.Tensor = hist["edges"]  # (num_bins+1,)
            counts: torch.Tensor = hist["counts"]  # (3, num_bins)
            bin_width = float(hist["bin_width"])

            # 막대 중심값(가로축): (num_bins,)
            centers = ((edges[:-1] + edges[1:]) * 0.5).tolist()

            for c, cls_name in enumerate(class_names):
                total = int(counts[c].sum().item())
                if total <= 0:
                    continue

                # 세로축을 %로 변환 (0~100)
                percent = (counts[c].to(torch.float32) / float(total) *
                           100.0).tolist()

                fig = plt.figure()
                plt.bar(centers, percent, width=bin_width * 0.9)
                plt.ylim(0.0, 100.0)
                plt.xlabel(metric_name)
                plt.ylabel("Percent(%)")
                plt.title(f"{metric_name} / {cls_name} (N={total})")
                plt.tight_layout()

                save_name = f"{metric_name}_{cls_name}_r{round_id:04d}.png"
                save_path = os.path.join(out_dir, save_name)
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

        # 역대 최대값 출력
        print("========== scalar max (all batches so far) ==========")
        for metric_name in metric_order:
            if metric_name not in self._hist_max_values:
                continue
            mv = self._hist_max_values[metric_name]  # (3,)
            msg = ", ".join([
                f"{class_names[i]}={float(mv[i].item()):.6g}" for i in range(3)
            ])
            print(f"{metric_name}: {msg}")

        # 메모리 관리: 누적 통계 초기화
        self._histograms.clear()
    self._hist_max_values.clear()
