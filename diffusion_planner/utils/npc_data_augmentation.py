from idlelib.debugger_r import DictProxy

import torch
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple, Union, cast, Dict

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

NUM_REFINE = 20
REFINE_HORIZON = 2.0
TIME_INTERVAL = 0.1


def normalize_angle(angle):
    """
    각도를 [-π, π] 범위로 정규화합니다.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class NPCStatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    def __init__(
        self,
        low: List[float] = [
            -0., -0.75, -0.35, -1, -0.5, -0., -0., -0., -0., -0.
        ],
        high: List[float] = [0., 0.75, 0.35, 1, 0.5, 0., 0., 0., 0., 0.],
        augment_prob: float = 0.5,
        normalize=True,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """
        Initialize the augmentor,
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        """
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._device = torch.device(device)
        self._low = torch.tensor(low).to(self._device)
        self._high = torch.tensor(high).to(self._device)
        self._wheel_base = get_pacifica_parameters().wheel_base

        self.refine_horizon = REFINE_HORIZON
        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL

        T = REFINE_HORIZON + TIME_INTERVAL
        self.coeff_matrix = torch.linalg.inv(
            torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0], [1, T, T**2, T**3, T**4, T**5],
                          [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                          [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3]],
                         device=device,
                         dtype=torch.float32))
        # TIME_INTERVAL: 0.1, REFINE_HORIZON: 2.0, NUM_REFINE: 20
        a = torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(
            1)  # (20, 1) # [0.1, 0.2, ..., 2.0]
        b = torch.arange(6).unsqueeze(0)  # (1,6) # [[0, 1, 2, 3, 4, 5]]
        self.t_matrix = torch.pow(a, b).to(device=device)  # shape (B, N+1)

    def __call__(self, inputs: Dict[str, torch.Tensor],
                 neighbors_future_all: torch.Tensor,
                 args) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            inputs:
                - neighbor_agents_past: (B, agent_num, time_len, 11)
            neighbors_future_all: (B, agent_num, future_len, 3)
        Returns:
            inputs:
                - neighbor_agents_past: (B, agent_num, time_len, 11)
            neighbors_future: (B, Pnn, future_len, 3)

        """
        neighbor_agents_past = inputs[
            'neighbor_agents_past']  # (B, agent_num, time_len, 11)
        near_current_states = neighbor_agents_past[:, :
                                                   args.predicted_neighbor_num,
                                                   -1, :]  # (B, pnn, 11)
        # near_current_wrt_self: (B, pnn, 11)
        near_current_wrt_self = self.convert_npc_state_to_self_frame(
            near_current_states)
        batch_ = near_current_wrt_self.shape[0]
        near_current_vx_wrt_self = near_current_wrt_self[...,
                                                         4]  # x축 속도 # (B, pnn)
        # aug_flags: (B, pnn)
        aug_flags = self.generate_aug_flag(batch_, args.predicted_neighbor_num,
                                           self._augment_prob,
                                           near_current_vx_wrt_self)
        # aug_near_current_wrt_self: (B, Pnn, 11)
        aug_near_current_wrt_self = self.augment(near_current_wrt_self,
                                                    aug_flags)
        near_current_xyyaw = near_current_states[:, :, :4]  # (B, pnn, 4)
        # aug_near_current: (B, pnn, 11)
        aug_near_current = self.convert_npc_state_self_to_ego(
            near_current_xyyaw, # (B, Pnn, 4)
            aug_near_current_wrt_self # (B, Pnn, 11)
        )
        """
        aug_near_current: (B, Pnn, 11)
        neighbor_agents_past: (B, agent_num, time_len, 11)
        neighbors_future_all: (B, agent_num, future_len, 3) -> (B, agent_num, future_len, 4)
        aug_flags: (B, Pnn) (bool)
        """
        aug_near_current, neighbor_agents_past, neighbors_future_all, aug_flags = \
            self.reorder_neighbors_after_augmentation(
            aug_near_current, neighbor_agents_past, neighbors_future_all,
            aug_flags)
        ##################
        neighbor_agents_past = self.refine_neighbor_past_trajectories(
            aug_flags, neighbor_agents_past
        )
        inputs["neighbor_agents_past"] = neighbor_agents_past
        # neighbors_future_all: (B, agent_num, future_len, 4) -> (B, agent_num, future_len, 3)
        neighbors_future_all = self.interpolation_future_trajectory(
            aug_near_current, neighbors_future_all, aug_flags)
        # return neighbors_future : (B, Pnn, future_len, 3)
        neighbors_future = neighbors_future_all[:, :args.predicted_neighbor_num,
                                                    :, :]
        return inputs, neighbors_future


    def refine_neighbor_past_trajectories(self,
            aug_flag: Tensor,  # (B, Pnn)
            ordered_neighbor_agents_past: Tensor,
            # (B, agent_num, time_len, 11)
    ) -> Tensor:
        """
        Quintic 5차 스플라인으로 **과거 2초(20step)** NPC 궤적을 보정한다.
        - 현재 프레임(t=0s)은 그대로 두고, t=−2s~−0.1s 전 구간을 재생성
        - 시작·끝 점의 **위치·속도·가속도·yaw‑rate** 경계조건을 정확히 만족

        Args:
            aug_flag (Tensor): (B,Pnn)
                각 배치의 Pnn개 예측 대상(agent)별 augment 여부.
            ordered_neighbor_agents_past (Tensor): (B,agent_num,time_len,11)
                과거 20step(–2s→–0.1s)+현재 프레임 순서의 궤적.

        Returns:
            neighbor_agents_past: (B,agent_num,time_len,11)
                `ordered_neighbor_agents_past` 의 복사본에 보정이 적용된 결과.
        """
        # ───────────────── 기본 파라미터 ─────────────────
        B, agent_num, time_len, D = ordered_neighbor_agents_past.shape
        Pnn = aug_flag.shape[1]
        device, dtype = ordered_neighbor_agents_past.device, ordered_neighbor_agents_past.dtype
        steps: int = time_len - 1  # 20
        T: float = self.refine_horizon  # 2.0

        # ───────────────── 1. 대상 마스크 계산 ─────────────────
        mask_flat: Tensor = aug_flag.clone().reshape(-1)  # (B * Pnn)

        if mask_flat.sum() == 0:
            # 보정 대상이 없으면 원본 그대로 반환
            return ordered_neighbor_agents_past.clone()

        # ───────────────── 2. 데이터 전개 ─────────────────
        near_agents_past = ordered_neighbor_agents_past[:,
                           :Pnn]  # (B, Pnn, time_len, 11)
        near_agents_past = near_agents_past.reshape(-1, time_len,
                                                    D)  # (B·Pnn, time_len, 11)

        near_current = near_agents_past[:, -1]  # (B·Pnn, 11)   (t = 0s)
        near_past = near_agents_past[:, :-1]  # (B·Pnn, time_len - 1, 11)

        sel_idx = mask_flat.nonzero(as_tuple=True)[0]  # (M,)  보정 대상 인덱스
        M: int = sel_idx.size(0)

        aug_near_current: Tensor = near_current[sel_idx]  # (M, 11)
        aug_near_past: Tensor = near_past[sel_idx]  # (M, time_len - 1, 11)

        # ───────────────── 3. Quintic 경계조건 ─────────────────
        # 현재(t=0)
        x0, y0 = aug_near_current[:, 0], aug_near_current[:, 1]  # (M,)
        cos0, sin0 = aug_near_current[:, 2], aug_near_current[:, 3]
        theta0 = torch.atan2(sin0, cos0)  # (M,)

        v0 = torch.norm(aug_near_current[:, 4:6],
                        dim=-1)  # (M,) speed magnitude
        # 가속도 벡터 (t=-0.1 → t=0)
        a0_vec = (aug_near_current[:, 4:6] - aug_near_past[:, -1,
                                             4:6]) / self.time_interval
        a0 = torch.norm(a0_vec, dim=-1)  # (M,)

        theta_prev = torch.atan2(aug_near_past[:, -1, 3],
                                 aug_near_past[:, -1, 2])  # (M,)
        omega0 = self.normalize_angle(theta0 - theta_prev) / self.time_interval  # (M,)

        # 과거 첫 프레임(t = –2s, index 0)
        xT, yT = aug_near_past[:, 0, 0], aug_near_past[:, 0, 1]
        cosT, sinT = aug_near_past[:, 0, 2], aug_near_past[:, 0, 3]
        thetaT = torch.atan2(sinT, cosT)

        vT = torch.norm(aug_near_past[:, 0, 4:6], dim=-1)  # (M,)
        aT_vec = (aug_near_past[:, 1, 4:6] - aug_near_past[:, 0,
                                             4:6]) / self.time_interval
        aT = torch.norm(aT_vec, dim=-1)

        theta_next = torch.atan2(aug_near_past[:, 1, 3], aug_near_past[:, 1, 2])
        omegaT = self.normalize_angle(theta_next - thetaT) / self.time_interval

        # Boundary vectors (M, 6)
        sx = torch.stack([
            x0,
            v0 * torch.cos(theta0),
            a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
            xT,
            vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
        ], dim=-1)

        sy = torch.stack([
            y0,
            v0 * torch.sin(theta0),
            a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
            yT,
            vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
        ], dim=-1)

        # ───────────────── 4. Quintic 계수 ─────────────────
        A_inv = torch.tensor(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0],
             [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
             [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
             [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]],
            device=device, dtype=dtype).inverse()  # (6, 6)

        ax_coef = (A_inv @ sx.unsqueeze(-1)).squeeze(-1)  # (M, 6)
        ay_coef = (A_inv @ sy.unsqueeze(-1)).squeeze(-1)  # (M, 6)

        # ───────────────── 5. 궤적 샘플링 (t = -2 … -0.1) ─────────────────
        t_vec = torch.linspace(self.time_interval, T, steps, device=device,
                               dtype=dtype)  # (time_len - 1,) 0.1…2.0
        T_mat = torch.stack([t_vec ** i for i in range(6)],
                            dim=-1)  # (time_len - 1, 6)

        # 양수 방향으로 생성 후 flip → 과거→최근 순서
        traj_x = (T_mat @ ax_coef.T).T.flip(dims=[1])  # (M, time_len - 1)
        traj_y = (T_mat @ ay_coef.T).T.flip(dims=[1])  # (M, time_len - 1)

        # ────────── 5‑①. 속도·헤딩 (forward 차분) ──────────
        # pos_x/y 에 현재(t=0) 위치 붙여 forward diff
        pos_x = torch.cat([traj_x, x0.unsqueeze(1)], dim=1)  # (M, time_len)
        pos_y = torch.cat([traj_y, y0.unsqueeze(1)], dim=1)  # (M, time_len)

        dx = pos_x[:, 1:] - pos_x[:, :-1]  # (M, time_len - 1)
        dy = pos_y[:, 1:] - pos_y[:, :-1]  # (M, time_len - 1)

        vx = dx / self.time_interval  # (M, time_len - 1)
        vy = dy / self.time_interval
        heading = torch.atan2(dy, dx)  # (M, time_len - 1)
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)

        # ────────── 5‑②. new_seg 구성 (M, time_len - 1, 6) ──────────
        new_seg = torch.zeros((M, steps, 6), device=device, dtype=dtype)
        new_seg[..., 0] = traj_x
        new_seg[..., 1] = traj_y
        new_seg[..., 2] = cos_h
        new_seg[..., 3] = sin_h
        new_seg[..., 4] = vx
        new_seg[..., 5] = vy

        # ───────────────── 6. near_past 덮어쓰기 (전 구간) ─────────────────
        near_past[sel_idx, :, :6] = new_seg  # t = −2 … −0.1 전부 교체

        # ───────────────── 7. 원본 텐서 복원 ─────────────────
        near_agents_past[sel_idx, :-1, :6] = near_past[sel_idx, :, :6]
        near_agents_past = near_agents_past.reshape(B, Pnn, time_len, D)

        neighbor_agents_past = ordered_neighbor_agents_past.clone()
        neighbor_agents_past[:, :Pnn] = near_agents_past
        return neighbor_agents_past


    def interpolation_future_trajectory(
        self,
        aug_near_current: torch.Tensor,
        neighbors_future_all: torch.Tensor,
        aug_flags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quintic 스플라인으로 **NPC 미래 궤적** 앞 `20` step(0.1s 간격, 2s 구간)을
        부드럽게 재보간한다.

        Args:
            aug_near_current (Tensor):
                증강된 현재 NPC 상태.
                **Shape** – ``(B, Pnn, 11)``
                ``[x, y, cosθ, sinθ, v_x, v_y, a_x, a_y, steer, yaw_rate, padding]``
            neighbors_future_all (Tensor):
                보간 전 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 4)``
                ``[x, y, cosθ, sinθ]`` × future_len
            aug_flags (Tensor):
                보간 수행 여부 플래그.
                **Shape** – ``(B, Pnn,)`` bool

        Returns:
            Tensor:
                ``neighbors_future_all`` –
                보간이 적용된 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 3)``
        """
        # aug_near_current: (B, Pnn, 11) -> (B * Pnn, 11)
        Pnn = aug_near_current.shape[1]  # predicted_neighbor_num
        aug_near_current = aug_near_current.reshape(-1,
                                                    aug_near_current.shape[-1])
        aug_flags = aug_flags.reshape(-1)  # (B * Pnn,)
        neighbors_future_all = neighbors_future_all.to(self._device)
        aug_near_current = aug_near_current.to(self._device)  # (B * Pnn, 11)
        aug_flags = aug_flags.to(self._device)
        # neighbors_future_all: (B, agent_num, future_len, 4)
        # I want to make neighbors_future_all from neighbors_future_all
        # (B, agent_num, future_len, 4) -> (B * agent_num, future_len, 4)  -> (B * agent_num, future_len, 3)
        b, agent_num, future_len, four_dim = neighbors_future_all.shape
        near_future = neighbors_future_all[:, :Pnn, :, :].clone(
        )  # (B, Pnn, future_len, 4)
        near_future = near_future.reshape(b * Pnn, future_len,
                                          four_dim)  # (B * Pnn, future_len, 4)

        # (B * Pnn, future_len, 4) -> (B * Pnn, future_len, 3)
        cos_ = near_future[:, :, 2]  # cos # (B * Pnn, future_len)
        sin_ = near_future[:, :, 3]  # sin # (B * Pnn, future_len)
        near_future = torch.cat(
            [
                near_future[:, :, :2],  # x, y
                torch.atan2(sin_, cos_)[..., None],  # heading
            ],
            dim=-1)  # (B * Pnn, future_len, 3)
        near_future = near_future.to(self._device)

        refine_P = self.num_refine  # 20
        dt = self.time_interval  # 0.1
        T = self.refine_horizon  # 2.0
        B_n_Pnn = aug_near_current.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        x0, y0, theta0, v0 = (
            aug_near_current[:, 0],  # (B_n_Pnn,)
            aug_near_current[:, 1],
            torch.atan2(
                (near_future[:, int(refine_P / 2), 1] - aug_near_current[:, 1]),
                (near_future[:, int(refine_P / 2), 0] -
                 aug_near_current[:, 0])),
            torch.norm(aug_near_current[:, 4:6], dim=-1),
        )
        first_fut_point_vel = (near_future[:, 1, 0:2] -
                               near_future[:, 0, 0:2]) / dt  # (B * Pnn, 2)
        a0_vec = (first_fut_point_vel - aug_near_current[:, 4:6]) / dt
        a0 = torch.norm(a0_vec, dim=-1)
        omega0 = self.normalize_angle(near_future[:, 0, 2] - theta0) / dt

        xT, yT, thetaT, vT, aT, omegaT = (
            near_future[:, refine_P,
                        0], near_future[:, refine_P,
                                        1], near_future[:, refine_P, 2],
            torch.norm(
                near_future[:, refine_P, :2] - near_future[:, refine_P - 1, :2],
                dim=-1) / dt,
            torch.norm(near_future[:, refine_P, :2] -
                       2 * near_future[:, refine_P - 1, :2] +
                       near_future[:, refine_P - 2, :2],
                       dim=-1) / dt**2,
            self.normalize_angle(near_future[:, refine_P, 2] -
                                 near_future[:, refine_P - 1, 2]) / dt)

        # Boundary conditions
        sx = torch.stack([
            x0, v0 * torch.cos(theta0), a0 * torch.cos(theta0) -
            v0 * torch.sin(theta0) * omega0, xT, vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT
        ],
                         dim=-1)

        sy = torch.stack([
            y0, v0 * torch.sin(theta0), a0 * torch.sin(theta0) +
            v0 * torch.cos(theta0) * omega0, yT, vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT
        ],
                         dim=-1)

        # ───────────────── 4. Quintic 계수 ─────────────────
        A_inv = torch.tensor(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0],
             [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
             [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
             [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]],
            device=device, dtype=dtype).inverse()  # (6, 6)

        ax_coef = (A_inv @ sx.unsqueeze(-1)).squeeze(-1)  # (M, 6)
        ay_coef = (A_inv @ sy.unsqueeze(-1)).squeeze(-1)  # (M, 6)

        # ───────────────── 5. 궤적 샘플링 (t = -2 … -0.1) ─────────────────
        t_vec = torch.linspace(self.time_interval, T, steps, device=device,
                               dtype=dtype)  # (time_len - 1,) 0.1…2.0
        T_mat = torch.stack([t_vec ** i for i in range(6)],
                            dim=-1)  # (time_len - 1, 6)

        # 양수 방향으로 생성 후 flip → 과거→최근 순서
        traj_x = (T_mat @ ax_coef.T).T.flip(dims=[1])  # (M, time_len - 1)
        traj_y = (T_mat @ ay_coef.T).T.flip(dims=[1])  # (M, time_len - 1)

        # ────────── 5‑①. 속도·헤딩 (forward 차분) ──────────
        # pos_x/y 에 현재(t=0) 위치 붙여 forward diff
        pos_x = torch.cat([traj_x, x0.unsqueeze(1)], dim=1)  # (M, time_len)
        pos_y = torch.cat([traj_y, y0.unsqueeze(1)], dim=1)  # (M, time_len)

        dx = pos_x[:, 1:] - pos_x[:, :-1]  # (M, time_len - 1)
        dy = pos_y[:, 1:] - pos_y[:, :-1]  # (M, time_len - 1)

        vx = dx / self.time_interval  # (M, time_len - 1)
        vy = dy / self.time_interval
        heading = torch.atan2(dy, dx)  # (M, time_len - 1)
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)

        # ────────── 5‑②. new_seg 구성 (M, time_len - 1, 6) ──────────
        new_seg = torch.zeros((M, steps, 6), device=device, dtype=dtype)
        new_seg[..., 0] = traj_x
        new_seg[..., 1] = traj_y
        new_seg[..., 2] = cos_h
        new_seg[..., 3] = sin_h
        new_seg[..., 4] = vx
        new_seg[..., 5] = vy

        # ───────────────── 6. near_past 덮어쓰기 (전 구간) ─────────────────
        near_past[sel_idx, :, :6] = new_seg  # t = −2 … −0.1 전부 교체

        # ───────────────── 7. 원본 텐서 복원 ─────────────────
        near_agents_past[sel_idx, :-1, :6] = near_past[sel_idx, :, :6]
        near_agents_past = near_agents_past.reshape(B, Pnn, time_len, D)

        neighbor_agents_past = ordered_neighbor_agents_past.clone()
        neighbor_agents_past[:, :Pnn] = near_agents_past
        return neighbor_agents_past


    def interpolation_future_trajectory(
        self,
        aug_near_current: torch.Tensor,
        neighbors_future_all: torch.Tensor,
        aug_flags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quintic 스플라인으로 **NPC 미래 궤적** 앞 `20` step(0.1s 간격, 2s 구간)을
        부드럽게 재보간한다.

        Args:
            aug_near_current (Tensor):
                증강된 현재 NPC 상태.
                **Shape** – ``(B, Pnn, 11)``
                ``[x, y, cosθ, sinθ, v_x, v_y, a_x, a_y, steer, yaw_rate, padding]``
            neighbors_future_all (Tensor):
                보간 전 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 4)``
                ``[x, y, cosθ, sinθ]`` × future_len
            aug_flags (Tensor):
                보간 수행 여부 플래그.
                **Shape** – ``(B, Pnn,)`` bool

        Returns:
            Tensor:
                ``neighbors_future_all`` –
                보간이 적용된 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 3)``
        """
        # aug_near_current: (B, Pnn, 11) -> (B * Pnn, 11)
        Pnn = aug_near_current.shape[1]  # predicted_neighbor_num
        aug_near_current = aug_near_current.reshape(-1,
                                                    aug_near_current.shape[-1])
        aug_flags = aug_flags.reshape(-1)  # (B * Pnn,)
        neighbors_future_all = neighbors_future_all.to(self._device)
        aug_near_current = aug_near_current.to(self._device)  # (B * Pnn, 11)
        aug_flags = aug_flags.to(self._device)
        # neighbors_future_all: (B, agent_num, future_len, 4)
        # I want to make neighbors_future_all from neighbors_future_all
        # (B, agent_num, future_len, 4) -> (B * agent_num, future_len, 4)  -> (B * agent_num, future_len, 3)
        b, agent_num, future_len, four_dim = neighbors_future_all.shape
        near_future = neighbors_future_all[:, :Pnn, :, :].clone(
        )  # (B, Pnn, future_len, 4)
        near_future = near_future.reshape(b * Pnn, future_len,
                                          four_dim)  # (B * Pnn, future_len, 4)

        # (B * Pnn, future_len, 4) -> (B * Pnn, future_len, 3)
        cos_ = near_future[:, :, 2]  # cos # (B * Pnn, future_len)
        sin_ = near_future[:, :, 3]  # sin # (B * Pnn, future_len)
        near_future = torch.cat(
            [
                near_future[:, :, :2],  # x, y
                torch.atan2(sin_, cos_)[..., None],  # heading
            ],
            dim=-1)  # (B * Pnn, future_len, 3)
        near_future = near_future.to(self._device)

        refine_P = self.num_refine  # 20
        dt = self.time_interval  # 0.1
        T = self.refine_horizon  # 2.0
        B_n_Pnn = aug_near_current.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        x0, y0, theta0, v0 = (
            aug_near_current[:, 0],  # (B_n_Pnn,)
            aug_near_current[:, 1],
            torch.atan2(
                (near_future[:, int(refine_P / 2), 1] - aug_near_current[:, 1]),
                (near_future[:, int(refine_P / 2), 0] -
                 aug_near_current[:, 0])),
            torch.norm(aug_near_current[:, 4:6], dim=-1),
        )
        first_fut_point_vel = (near_future[:, 1, 0:2] -
                               near_future[:, 0, 0:2]) / dt  # (B * Pnn, 2)
        a0_vec = (first_fut_point_vel - aug_near_current[:, 4:6]) / dt
        a0 = torch.norm(a0_vec, dim=-1)
        omega0 = self.normalize_angle(near_future[:, 0, 2] - theta0) / dt

        xT, yT, thetaT, vT, aT, omegaT = (
            near_future[:, refine_P,
                        0], near_future[:, refine_P,
                                        1], near_future[:, refine_P, 2],
            torch.norm(
                near_future[:, refine_P, :2] - near_future[:, refine_P - 1, :2],
                dim=-1) / dt,
            torch.norm(near_future[:, refine_P, :2] -
                       2 * near_future[:, refine_P - 1, :2] +
                       near_future[:, refine_P - 2, :2],
                       dim=-1) / dt**2,
            self.normalize_angle(near_future[:, refine_P, 2] -
                                 near_future[:, refine_P - 1, 2]) / dt)

        # Boundary conditions
        sx = torch.stack([
            x0, v0 * torch.cos(theta0), a0 * torch.cos(theta0) -
            v0 * torch.sin(theta0) * omega0, xT, vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT
        ],
                         dim=-1)

        sy = torch.stack([
            y0, v0 * torch.sin(theta0), a0 * torch.sin(theta0) +
            v0 * torch.cos(theta0) * omega0, yT, vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT
        ],
                         dim=-1)

        # ───────────────── 4. Quintic 계수 ─────────────────
        A_inv = torch.tensor(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0],
             [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
             [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
             [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]],
            device=device, dtype=dtype).inverse()  # (6, 6)

        ax_coef = (A_inv @ sx.unsqueeze(-1)).squeeze(-1)  # (M, 6)
        ay_coef = (A_inv @ sy.unsqueeze(-1)).squeeze(-1)  # (M, 6)

        # ───────────────── 5. 궤적 샘플링 (t = -2 … -0.1) ─────────────────
        t_vec = torch.linspace(self.time_interval, T, steps, device=device,
                               dtype=dtype)  # (time_len - 1,) 0.1…2.0
        T_mat = torch.stack([t_vec ** i for i in range(6)],
                            dim=-1)  # (time_len - 1, 6)

        # 양수 방향으로 생성 후 flip → 과거→최근 순서
        traj_x = (T_mat @ ax_coef.T).T.flip(dims=[1])  # (M, time_len - 1)
        traj_y = (T_mat @ ay_coef.T).T.flip(dims=[1])  # (M, time_len - 1)

        # ────────── 5‑①. 속도·헤딩 (forward 차분) ──────────
        # pos_x/y 에 현재(t=0) 위치 붙여 forward diff
        pos_x = torch.cat([traj_x, x0.unsqueeze(1)], dim=1)  # (M, time_len)
        pos_y = torch.cat([traj_y, y0.unsqueeze(1)], dim=1)  # (M, time_len)

        dx = pos_x[:, 1:] - pos_x[:, :-1]  # (M, time_len - 1)
        dy = pos_y[:, 1:] - pos_y[:, :-1]  # (M, time_len - 1)

        vx = dx / self.time_interval  # (M, time_len - 1)
        vy = dy / self.time_interval
        heading = torch.atan2(dy, dx)  # (M, time_len - 1)
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)

        # ────────── 5‑②. new_seg 구성 (M, time_len - 1, 6) ──────────
        new_seg = torch.zeros((M, steps, 6), device=device, dtype=dtype)
        new_seg[..., 0] = traj_x
        new_seg[..., 1] = traj_y
        new_seg[..., 2] = cos_h
        new_seg[..., 3] = sin_h
        new_seg[..., 4] = vx
        new_seg[..., 5] = vy

        # ───────────────── 6. near_past 덮어쓰기 (전 구간) ─────────────────
        near_past[sel_idx, :, :6] = new_seg  # t = −2 … −0.1 전부 교체

        # ───────────────── 7. 원본 텐서 복원 ─────────────────
        near_agents_past[sel_idx, :-1, :6] = near_past[sel_idx, :, :6]
        near_agents_past = near_agents_past.reshape(B, Pnn, time_len, D)

        neighbor_agents_past = ordered_neighbor_agents_past.clone()
        neighbor_agents_past[:, :Pnn] = near_agents_past
        return neighbor_agents_past


    def interpolation_future_trajectory(
        self,
        aug_near_current: torch.Tensor,
        neighbors_future_all: torch.Tensor,
        aug_flags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quintic 스플라인으로 **NPC 미래 궤적** 앞 `20` step(0.1s 간격, 2s 구간)을
        부드럽게 재보간한다.

        Args:
            aug_near_current (Tensor):
                증강된 현재 NPC 상태.
                **Shape** – ``(B, Pnn, 11)``
                ``[x, y, cosθ, sinθ, v_x, v_y, a_x, a_y, steer, yaw_rate, padding]``
            neighbors_future_all (Tensor):
                보간 전 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 4)``
                ``[x, y, cosθ, sinθ]`` × future_len
            aug_flags (Tensor):
                보간 수행 여부 플래그.
                **Shape** – ``(B, Pnn,)`` bool

        Returns:
            Tensor:
                ``neighbors_future_all`` –
                보간이 적용된 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 3)``
        """
        # aug_near_current: (B, Pnn, 11) -> (B * Pnn, 11)
        Pnn = aug_near_current.shape[1]  # predicted_neighbor_num
        aug_near_current = aug_near_current.reshape(-1,
                                                    aug_near_current.shape[-1])
        aug_flags = aug_flags.reshape(-1)  # (B * Pnn,)
        neighbors_future_all = neighbors_future_all.to(self._device)
        aug_near_current = aug_near_current.to(self._device)  # (B * Pnn, 11)
        aug_flags = aug_flags.to(self._device)
        # neighbors_future_all: (B, agent_num, future_len, 4)
        # I want to make neighbors_future_all from neighbors_future_all
        # (B, agent_num, future_len, 4) -> (B * agent_num, future_len, 4)  -> (B * agent_num, future_len, 3)
        b, agent_num, future_len, four_dim = neighbors_future_all.shape
        near_future = neighbors_future_all[:, :Pnn, :, :].clone(
        )  # (B, Pnn, future_len, 4)
        near_future = near_future.reshape(b * Pnn, future_len,
                                          four_dim)  # (B * Pnn, future_len, 4)

        # (B * Pnn, future_len, 4) -> (B * Pnn, future_len, 3)
        cos_ = near_future[:, :, 2]  # cos # (B * Pnn, future_len)
        sin_ = near_future[:, :, 3]  # sin # (B * Pnn, future_len)
        near_future = torch.cat(
            [
                near_future[:, :, :2],  # x, y
                torch.atan2(sin_, cos_)[..., None],  # heading
            ],
            dim=-1)  # (B * Pnn, future_len, 3)
        near_future = near_future.to(self._device)

        refine_P = self.num_refine  # 20
        dt = self.time_interval  # 0.1
        T = self.refine_horizon  # 2.0
        B_n_Pnn = aug_near_current.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B_n_Pnn, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        x0, y0, theta0, v0 = (
            aug_near_current[:, 0],  # (B_n_Pnn,)
            aug_near_current[:, 1],
            torch.atan2(
                (near_future[:, int(refine_P / 2), 1] - aug_near_current[:, 1]),
                (near_future[:, int(refine_P / 2), 0] -
                 aug_near_current[:, 0])),
            torch.norm(aug_near_current[:, 4:6], dim=-1),
        )
        first_fut_point_vel = (near_future[:, 1, 0:2] -
                               near_future[:, 0, 0:2]) / dt  # (B * Pnn, 2)
        a0_vec = (first_fut_point_vel - aug_near_current[:, 4:6]) / dt
        a0 = torch.norm(a0_vec, dim=-1)
        omega0 = self.normalize_angle(near_future[:, 0, 2] - theta0) / dt

        xT, yT, thetaT, vT, aT, omegaT = (
            near_future[:, refine_P,
                        0], near_future[:, refine_P,
                                        1], near_future[:, refine_P, 2],
            torch.norm(
                near_future[:, refine_P, :2] - near_future[:, refine_P - 1, :2],
                dim=-1) / dt,
            torch.norm(near_future[:, refine_P, :2] -
                       2 * near_future[:, refine_P - 1, :2] +
                       near_future[:, refine_P - 2, :2],
                       dim=-1) / dt**2,
            self.normalize_angle(near_future[:, refine_P, 2] -
                                 near_future[:, refine_P - 1, 2]) / dt)

        # Boundary conditions
        sx = torch.stack([
            x0, v0 * torch.cos(theta0), a0 * torch.cos(theta0) -
            v0 * torch.sin(theta0) * omega0, xT, vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT
        ],
                         dim=-1)

        sy = torch.stack([
            y0, v0 * torch.sin(theta0), a0 * torch.sin(theta0) +
            v0 * torch.cos(theta0) * omega0, yT, vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT
        ],
                         dim=-1)

        ax_coef = (A @ sx[:, :, None]).squeeze(-1)  # B_n_Pnn, 6, 1
        ay_coef = (A @ sy[:, :, None]).squeeze(-1)  # B_n_Pnn, 6, 1

        traj_x = M_t @ ax_coef  # (B_n_Pnn, P, 1)
        traj_y = M_t @ ay_coef  # (B_n_Pnn, P, 1)
        a = torch.atan2(traj_y[:, :1, 0] - y0.unsqueeze(-1),
                        traj_x[:, :1, 0] - x0.unsqueeze(-1))  # (B_n_Pnn, 1)
        b = torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0],
                        traj_x[:, 1:, 0] - traj_x[:, :-1, 0])  # (B_n_Pnn, P-1)
        traj_heading = torch.cat([a, b], dim=1)  # (B_n_Pnn, P)
        traj_cos = torch.cos(traj_heading)  # (B_n_Pnn, P)
        traj_sin = torch.sin(traj_heading)
        # (B_n_Pnn, P, 4)
        refined_result = torch.stack(
            [
                traj_x[:, :, 0],  # x
                traj_y[:, :, 0],  # y
                traj_cos,  # cos
                traj_sin,  # sin
            ],
            dim=-1)  # (B_n_Pnn, P=20, 4)
        """
        `refined_result`(20 step)을
        `ordered_neighbors_interpolated_future[:, :, 1:21]` 영역에 복사한다.
        보간 대상은 `aug_flags == True` 인 샘플에 한정한다.
        """

        near_future[aug_flags, :refine_P, :] = refined_result[
            aug_flags]  # 보간 결과 적용 # (B * Pnn, 20, 4)
        # near_future: (B * Pnn, 20, 4) -> (B, Pnn, 20, 4)
        near_future = near_future.reshape(b, Pnn, refine_P, four_dim)
        # neighbors_future_all 에 near_future 을 반영한다.
        neighbors_future_all[:, :Pnn, :refine_P, :] = near_future
        # neighbors_future_all: (B, agent_num, future_len, 4) -> (B, agent_num, future_len, 3)
        cos_yaw = neighbors_future_all[:, :, :, 2]  # (B, agent_num, future_len)
        sin_yaw = neighbors_future_all[:, :, :, 3]  # (B, agent_num, future_len)
        # clamp를 사용하여 atan2의 입력값이 0에 가까워지는 것을 방지
        eps = torch.finfo(cos_yaw.dtype).eps
        cos_yaw = cos_yaw.clamp(min=eps)
        yaw = torch.atan2(sin_yaw, cos_yaw)  # (B, agent_num, future_len)
        neighbors_future_all = torch.cat(
            [
                neighbors_future_all[:, :, :, :2],  # x, y
                yaw[..., None],  # heading
            ],
            dim=-1) # (B, agent_num, future_len, 3)
        return neighbors_future_all

    @staticmethod
    def convert_npc_state_to_self_frame(near_current_states: Tensor) -> Tensor:
        """
        NPC 상태 텐서를 **각 NPC 자신의 로컬 좌표계**로 변환한다.

        Args:
            near_current_states (Tensor):
                - shape: **(B, pnn, 11)**

        Returns:
            near_current_wrt_self
                - shape: **(B, pnn, 11)**
                - 각 NPC의 상태를 **자기 자신을 원점·정면으로 둔 프레임**으로 표현
                  - x, y → 0
                  - cos(yaw) → 1, sin(yaw) → 0
                  - 속도는 아래 회전식으로 변환
                      * v_x' = cos(phi) * v_x + sin(phi) * v_y
                      * v_y' = -sin(phi) * v_x + cos(phi) * v_y
                  - width, length, 클래스 원‑핫은 그대로 유지

        """
        if near_current_states.ndim != 3 or near_current_states.size(-1) != 11:
            raise ValueError(
                f"near_current_states must have shape (B, Pnn, 11); got "
                f"{near_current_states.shape}")

        # (B, pnn, 11) → 복사본 생성 (grad 보존)
        near_current_wrt_self: Tensor = near_current_states.clone()

        # ──────────────────────────────────────────────────────────
        # 1. heading 정보
        # ──────────────────────────────────────────────────────────
        cos_phi: Tensor = near_current_states[..., 2]  # (B, pnn)
        sin_phi: Tensor = near_current_states[..., 3]  # (B, pnn)

        # ──────────────────────────────────────────────────────────
        # 2. 속도 회전
        # ──────────────────────────────────────────────────────────
        vx: Tensor = near_current_states[..., 4]  # (B, pnn)
        vy: Tensor = near_current_states[..., 5]  # (B, pnn)

        # 회전 행렬 R(-phi) 적용 (요소별 연산, 브로드캐스팅)
        vx_rot: Tensor = vx * cos_phi + vy * sin_phi
        vy_rot: Tensor = -vx * sin_phi + vy * cos_phi

        # ──────────────────────────────────────────────────────────
        # 3. 결과 텐서 채우기
        # ──────────────────────────────────────────────────────────
        near_current_wrt_self[..., 0] = 0.0  # x'
        near_current_wrt_self[..., 1] = 0.0  # y'
        near_current_wrt_self[..., 2] = 1.0  # cos(0)
        near_current_wrt_self[..., 3] = 0.0  # sin(0)
        near_current_wrt_self[..., 4] = vx_rot
        near_current_wrt_self[..., 5] = vy_rot
        # width(6), length(7), class one‑hot(8:11) 는 그대로 유지

        return near_current_wrt_self

    @staticmethod
    def generate_aug_flag(B: int, predicted_neighbor_num: int,
                          augment_prob: float,
                          near_current_vx_wrt_self: Tensor) -> Tensor:
        """
        배치별로 랜덤하게 augment 여부 플래그를 생성합니다.

        Args:
            B (int): 배치 크기.
            predicted_neighbor_num (int): 배치당 에이전트 수.
            augment_prob (float): 배치별로 augment 비율 (0.0부터 1.0 사이).
                `round(predicted_neighbor_num * augment_prob)` 개수만큼 선택합니다.
            near_current_vx_wrt_self: (B, predicted_neighbor_num) 크기의 텐서로,
                각 에이전트의 x축 속도를 나타냅니다. (좌표계는 에이전트 기준)

        Returns:
            torch.BoolTensor: 각 에이전트의 augment 수행 여부를 담은 2D BoolTensor.
                - shape: (B , predicted_neighbor_num,)
                - True  → 해당 위치 에이전트에 augment를 수행
                - False → 해당 위치 에이전트에 augment를 수행하지 않음
        """
        device = near_current_vx_wrt_self.device
        # 0) 입력값 검증
        if augment_prob < 0.0 or augment_prob > 1.0:
            raise ValueError("augment_prob는 0.0 이상 1.0 이하이어야 합니다.")

        # 1) 배치당 선택할 agent 개수
        k = int(round(predicted_neighbor_num * augment_prob))

        # 2) 극단 case 처리
        if k <= 0:
            return torch.zeros(B , predicted_neighbor_num, dtype=torch.bool, device=device)
        if k >= predicted_neighbor_num:
            return torch.ones(B , predicted_neighbor_num, dtype=torch.bool, device=device)

        # 3) (B, N) 크기의 균등 가중치 행렬 생성
        weights = torch.ones((B, predicted_neighbor_num), device=device)

        # 4) 배치별로 k개 인덱스 샘플링 (replacement=False)
        #    결과 shape: (B, k)
        sampled_idx = torch.multinomial(weights, k, replacement=False)

        # 5) 빈 Bool 플래그 텐서 생성
        aug_flags = torch.zeros((B, predicted_neighbor_num), dtype=torch.bool, device=device)

        # 6) 배치 인덱스와 결합해 True 할당
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
        aug_flags[batch_idx, sampled_idx] = True

        # 7) near_current_vx_wrt_self 값의 절댓값이 2.0 이하인 경우는 augment를 수행하지 않음.
        aug_flags = aug_flags & (torch.abs(near_current_vx_wrt_self) >= 2.0
                                )  # (B, N) 크기의 텐서와 논리곱 수행

        return aug_flags

    def augment(
            self,
            near_current_wrt_self: torch.
        Tensor,  # (B, predicted_neighbor_num, 11)
            aug_flags: torch.Tensor,  # (B, predicted_neighbor_num)
    ) -> torch.Tensor:
        """
        near_current_wrt_self: Tensor # (B, predicted_neighbor_num, 11)
            11: x, y, cos(yaw), sin(yaw), vx, vy, width, length, one-hot (3)
        Returns:
            aug_near_current_wrt_self: Tensor # (B, predicted_neighbor_num, 11)
        """
        near_current_wrt_self = near_current_wrt_self.to(self._device)
        aug_flags = aug_flags.to(self._device)

        B, predicted_neighbor_num, dim_ = near_current_wrt_self.shape
        B_n_pnn = B * predicted_neighbor_num
        temp_dim = len(self._low)  # 10
        random_tensor = torch.rand(B_n_pnn,
                                   temp_dim).to(self._device)  # (B_n_pnn, 10)
        scaled_random_tensor = self._low + (
            self._high - self._low) * random_tensor  # # (B_n_pnn, 10)
        temp_near_current_wrt_self = torch.zeros(
            (B, predicted_neighbor_num, temp_dim), dtype=torch.float32).to(
                self._device)  # (B, predicted_neighbor_num, 10)
        temp_near_current_wrt_self[:, :, 3:] = near_current_wrt_self[:, :, 4:]
        # temp_near_current_wrt_self: (B, predicted_neighbor_num, 10) -> (B_n_pnn, 10)
        temp_near_current_wrt_self = temp_near_current_wrt_self.reshape(
            B_n_pnn, temp_dim)  # (B_n_pnn, 10)
        aug_temp_near_current_wrt_self = temp_near_current_wrt_self + scaled_random_tensor  # (B_n_pnn, 10)
        # TODO: vx를 0 이상으로 제한하면, 후진하는 차량에 대한 대응력을 학습할 수 없게 됩니다.
        # vx를 0 이상으로 제한
        aug_temp_near_current_wrt_self[:, 3] = torch.max(
            aug_temp_near_current_wrt_self[:, 3],
            torch.tensor(0.0, device=aug_temp_near_current_wrt_self.device))
        # aug_temp_near_current_wrt_self: (B_n_pnn, 10) -> (B, predicted_neighbor_num, 10)
        aug_temp_near_current_wrt_self = aug_temp_near_current_wrt_self.reshape(
            B, predicted_neighbor_num,
            temp_dim)  # (B, predicted_neighbor_num, 10)

        aug_near_current_wrt_self = near_current_wrt_self.clone(
        )  # (B, predicted_neighbor_num, 11)
        # aug_flags가 True인 경우에만 업데이트. # aug_flags: (B, predicted_neighbor_num)
        # 안전성을 위해 aug_flags의 shape과 타입을 검증
        if aug_flags.dtype != torch.bool:
            raise TypeError(f"aug_flags must be torch.bool, got {aug_flags.dtype}")

        if aug_flags.shape != (B, predicted_neighbor_num):
            raise ValueError(f"aug_flags shape mismatch: expected {(B, predicted_neighbor_num)}, got {aug_flags.shape}")

        aug_flag_3d = aug_flags.unsqueeze(-1)  # (B, Pnn, 1)

        # 1) x, y
        aug_near_current_wrt_self[..., :2] = torch.where(
            aug_flag_3d, aug_temp_near_current_wrt_self[..., :2],
            aug_near_current_wrt_self[..., :2])

        # 2) cos, sin
        yaw = aug_temp_near_current_wrt_self[..., 2]  # (B,Pnn)
        cos_new, sin_new = yaw.cos(), yaw.sin()
        aug_near_current_wrt_self[..., 2:4] = torch.where(
            aug_flag_3d, torch.stack([cos_new, sin_new], dim=-1),
            aug_near_current_wrt_self[..., 2:4])

        # 3) vx ~ class‑one‑hot
        aug_near_current_wrt_self[..., 4:11] = torch.where(
            aug_flag_3d, aug_temp_near_current_wrt_self[..., 3:],
            aug_near_current_wrt_self[..., 4:11])
        return aug_near_current_wrt_self


    @staticmethod
    def convert_npc_state_self_to_ego(
            near_current_xyyaw: Tensor,
            aug_near_current_wrt_self: Tensor,
    ) -> Tensor:
        """
        NPC 상태를 **각 NPC 자신의 로컬 좌표계**에서 **ego 좌표계**로 변환한다.

        Args:
            near_current_xyyaw (Tensor):
                - shape: (B, pnn, 4)
                - 4차원 구성
                  0. x_0  [m]  ┐
                  1. y_0  [m]  │ NPC 원점·heading을 ego 프레임으로 표현
                  2. cos(phi_0)│
                  3. sin(phi_0)┘
            aug_near_current_wrt_self (Tensor):
                - shape: (B, pnn, 11)
                - 11차원 구성
                  0. x_local  [m]
                  1. y_local  [m]
                  2. cos(yaw_local)
                  3. sin(yaw_local)
                  4. v_x_local [m/s]
                  5. v_y_local [m/s]
                  6. width  [m]
                  7. length [m]
                  8–10. one-hot(class)


        Returns:
            aug_near_current:
                - shape: (B, pnn, 11)
                - aug_near_current_wrt_self 를 ego 프레임으로 변환한 결과
                  (width, length, class one-hot 은 그대로 유지)

        Raises:
            ValueError: 입력 차원이 (B, pnn, 11)·(B, pnn, 4) 형식이 아닐 때.
        """
        # ──────────────────── 입력 검증 ────────────────────
        if near_current_xyyaw.ndim != 3 or near_current_xyyaw.size(-1) != 4:
            raise ValueError(
                f"'near_current_xyyaw' must have shape (B, pnn, 4); "
                f"got {near_current_xyyaw.shape}")
        if aug_near_current_wrt_self.ndim != 3 or aug_near_current_wrt_self.size(
                -1) != 11:
            raise ValueError(
                f"'aug_near_current_wrt_self' must have shape (B, pnn, 11); "
                f"got {aug_near_current_wrt_self.shape}")

        if aug_near_current_wrt_self.shape[:2] != near_current_xyyaw.shape[:2]:
            raise ValueError(
                "batch/agent 차원이 서로 다릅니다: "
                f"{aug_near_current_wrt_self.shape[:2]} vs {near_current_xyyaw.shape[:2]}"
            )

        # ───────────────── dtype·device 일치 ─────────────────
        device, dtype = aug_near_current_wrt_self.device, aug_near_current_wrt_self.dtype
        near_current_xyyaw = near_current_xyyaw.to(device=device, dtype=dtype)

        # ───────────────── 필드 분리 (shape) ─────────────────
        # self-frame 값
        x_loc: Tensor = aug_near_current_wrt_self[..., 0]  # (B, pnn)
        y_loc: Tensor = aug_near_current_wrt_self[..., 1]  # (B, pnn)
        cos_loc: Tensor = aug_near_current_wrt_self[..., 2]  # (B, pnn)
        sin_loc: Tensor = aug_near_current_wrt_self[..., 3]  # (B, pnn)
        vx_loc: Tensor = aug_near_current_wrt_self[..., 4]  # (B, pnn)
        vy_loc: Tensor = aug_near_current_wrt_self[..., 5]  # (B, pnn)

        # ego-frame에서 본 NPC 원점·heading
        x0: Tensor = near_current_xyyaw[..., 0]  # (B, pnn)
        y0: Tensor = near_current_xyyaw[..., 1]  # (B, pnn)
        cos0: Tensor = near_current_xyyaw[..., 2]  # (B, pnn)
        sin0: Tensor = near_current_xyyaw[..., 3]  # (B, pnn)

        # ───────────────── 위치 변환 ─────────────────
        # [x_ego, y_ego] = R(phi0) · [x_loc, y_loc] + [x0, y0]
        x_ego: Tensor = cos0 * x_loc - sin0 * y_loc + x0  # (B, pnn)
        y_ego: Tensor = sin0 * x_loc + cos0 * y_loc + y0  # (B, pnn)

        # ───────────────── heading 변환 ─────────────────
        # cos(phi0 + philoc) / sin(phi0 + philoc)
        cos_ego: Tensor = cos0 * cos_loc - sin0 * sin_loc  # (B, pnn)
        sin_ego: Tensor = sin0 * cos_loc + cos0 * sin_loc  # (B, pnn)

        # ───────────────── 속도 변환 ─────────────────
        vx_ego: Tensor = cos0 * vx_loc - sin0 * vy_loc  # (B, pnn)
        vy_ego: Tensor = sin0 * vx_loc + cos0 * vy_loc  # (B, pnn)

        # ───────────────── 결과 구성 ─────────────────
        aug_near_current: Tensor = aug_near_current_wrt_self.clone()  # (B, pnn, 11), non-inplace

        aug_near_current[..., 0] = x_ego
        aug_near_current[..., 1] = y_ego
        aug_near_current[..., 2] = cos_ego
        aug_near_current[..., 3] = sin_ego
        aug_near_current[..., 4] = vx_ego
        aug_near_current[..., 5] = vy_ego
        # width(6), length(7), class one-hot(8:10) 는 그대로 유지

        return aug_near_current

    @staticmethod
    def reorder_neighbors_after_augmentation(
            aug_near_current: torch.Tensor,
            neighbor_agents_past: torch.Tensor,
            neighbors_future_all: torch.Tensor,
            aug_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        증강(augmentation)‑된 **가까운 Pnn대 NPC**의 현재 위치를 과거·미래 시계열에 반영하고,
        *모든* 주변 ``agents_num``대 NPC를 **새로운 거리 순서**(ego → NPC 거리 오름차순)로
        재정렬

        Args:
            aug_near_current (Tensor):
                증강 후 가까운 이웃의 현재 상태
                **Shape** – ``(B, Pnn, 11)``
                ``[x, y, cosθ, sinθ, v_x, v_y, width, length, one‑hot(3)]``
            neighbor_agents_past (Tensor):
                증강 전 거리순으로 정렬된 과거~현재 궤적
                **Shape** – ``(B, agents_num, time_len, 11)``
            neighbors_future_all (Tensor):
                증강 전 거리순으로 정렬된 미래 궤적(현재 제외)
                **Shape** – ``(B, agents_num, future_len, 3)``
            aug_flags (Tensor):
                `aug_near_current` 각 NPC별 증강 여부
                **Shape** – ``(B, Pnn)``  (bool)


        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            0. aug_near_current
                증강된 현재 프레임 반영
                ``(B, Pnn, 11)``


            1. **neighbor_agents_past** –
               증강된 현재 프레임 반영 + 새 거리순 정렬
               ``(B, agents_num, time_len, 11)``

            2. **neighbors_future_all** –
               새 거리순 정렬된 미래 궤적
               ``(B, agents_num, future_len, 4)``

            3. **aug_flags** –
               새 거리순에서 앞 `Pnn` NPC에 대한 증강 여부
               ``(B, Pnn)``  (bool)
        """
        # ───── 기본 정보 및 안전장치 ──────────────────────────────────────────
        B, Pnn, _ = aug_near_current.shape
        _, agents_num, time_len, _ = neighbor_agents_past.shape
        _, _, future_len, _ = neighbors_future_all.shape
        # neighbors_future_all: (B, agents_num, future_len, 3) -> (B, agents_num, future_len, 4)
        yaw_ = neighbors_future_all[..., 2]  # (B, agents_num, future_len)
        cos_yaw = torch.cos(yaw_)  # (B, agents_num, future_len)
        sin_yaw = torch.sin(yaw_)  # (B, agents_num, future_len)
        future_4ch = torch.cat(
            [neighbors_future_all[..., :2], cos_yaw[..., None],
                sin_yaw[..., None]], dim=-1) # (B, agents_num, future_len, 4)


        if Pnn > agents_num:
            raise ValueError(f"Pnn({Pnn}) must be ≤ agents_num({agents_num})")

        device = neighbor_agents_past.device
        aug_near_current = aug_near_current.to(device)
        aug_flags = aug_flags.to(device)

        # ───── 1. 현재 프레임에 증강 결과 반영 ────────────────────────────────
        past_updated = neighbor_agents_past.clone()  # (B, agent_num, T, 11)
        mask = aug_flags[:, :Pnn].to(device)  # (B, Pnn)
        # True인 위치에만 덮어쓰기
        past_updated[:, :Pnn, -1, :] = torch.where(
            mask.unsqueeze(-1),
            aug_near_current,
            past_updated[:, :Pnn, -1, :]
        )

        # ───── 2. 신규 거리 계산 (ego → NPC) ────────────────────────────────
        cur_xy: torch.Tensor = past_updated[:, :, -1, :2]  # (B, agent_num, 2)
        dist: torch.Tensor = torch.linalg.norm(cur_xy, dim=-1)  # (B, agent_num)
        order_idx: torch.Tensor = dist.argsort(dim=1)  # (B, agent_num)

        # ───── 3. 텐서 재정렬 (take_along_dim 사용으로 메모리 효율화) ─────────────────
        # 과거 궤적 재정렬
        # 4) gather를 이용한 재정렬
        idx_past = order_idx[:, :, None, None].expand(-1, -1, time_len, 11) # (B, agent_num, time_len, 11)
        neighbor_agents_past = past_updated.gather(dim=1, index=idx_past)

        idx_fut = order_idx[:, :, None, None].expand(-1, -1, future_len, 4) # (B, agent_num, future_len, 4)
        neighbors_future_all = future_4ch.gather(dim=1, index=idx_fut)

        # 5) augment 플래그 재정렬
        flag_full = torch.zeros((B, agents_num), dtype=torch.bool, device=device)
        flag_full[:, :Pnn] = aug_flags
        flag_reordered = flag_full.gather(dim=1, index=order_idx)
        aug_flags = flag_reordered[:, :Pnn]

        # 6) 재정렬 후 최신 상태 추출
        aug_near_current = neighbor_agents_past[:, :Pnn, -1, :]

        return aug_near_current, neighbor_agents_past, neighbors_future_all, aug_flags
