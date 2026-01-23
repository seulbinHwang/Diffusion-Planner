from typing import Dict, Any, Tuple
import torch
from diffusion_planner.model.module.feasible import FeasibleProjector


class DataStatistics:

    def __init__(self, config):
        self.config = config
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
        r_min
            - 1m 단위로 통계 히스토그램 그리자 (0~1, 1~2, 2~3, ..., 98~99, 99~100, ...)
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
        """
        abs_v_y_b_max : (B, agent_num)
        abs_v_max : (B, agent_num)
        abs_omega_max : (B, agent_num)
        """
        (abs_v_y_b_max, abs_v_max, abs_omega_max
        ) = self._get_seg_body_control_statistics(seg_body_control)


        # (B, agent_num, past_len + future_len)
        seg_body_v = torch.norm(seg_body_control[:, :, :, 0:2], dim=-1)
        # (B, agent_num, past_len + future_len)
        seg_body_omega = seg_body_control[:, :, :, 2]
        """
        TODO : apply savgol filter to seg_body_v and seg_body_omega
        to get acceleration and angular acceleration.
        
        # ASSUMPTION: 
            1. using same dt
            2. using window length that you want (your reasonable one)
            3. NEVER consider stride_step > 1. stride_step == 1 always.
        
        # IMPORTANT:
            연산 소요 속도를 짧게 가져가도록 코딩 부탁.
        
        WANTED OUTPUTS:
        seg_body_accel : (B, agent_num, past_len + future_len)
        seg_body_angular_accel : (B, agent_num, past_len + future_len)
        """

    def _get_seg_body_control_statistics(
        self, seg_body_control: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ seg_body_control: (B, agent_num, past_len + future_len, 3)
            [v_x^b, v_y^b, ω]_mid

        returns:
            abs(v_y^b)_max : (B, agent_num)
            abs(v)_max : (B, agent_num)
            abs(ω)_max : (B, agent_num)
        """
        B, agent_num, total_len, _ = seg_body_control.shape

        v_y_b = seg_body_control[:, :, :, 1]  # (B, agent_num, total_len)
        omega = seg_body_control[:, :, :, 2]  # (B, agent_num, total_len)

        abs_v_y_b_max, _ = torch.max(torch.abs(v_y_b), dim=2)  # (B, agent_num)
        v = torch.norm(seg_body_control[:, :, :, 0:2],
                       dim=-1)  # (B, agent_num, total_len)
        abs_v_max, _ = torch.max(torch.abs(v), dim=2)  # (B, agent_num)
        abs_omega_max, _ = torch.max(torch.abs(omega), dim=2)  # (B, agent_num)

        return abs_v_y_b_max, abs_v_max, abs_omega_max
