from idlelib.debugger_r import DictProxy

import torch
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple, Union, cast, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

NUM_REFINE = 20
REFINE_HORIZON = 2.0
TIME_INTERVAL = 0.1



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
        self.count = 0
        self.refine_horizon = REFINE_HORIZON
        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL
        T = - REFINE_HORIZON
        self.A_inv_const = torch.tensor(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0],
             [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
             [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
             [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]],
             device=device, dtype=torch.float32).inverse()  # (6, 6)
        T = REFINE_HORIZON + TIME_INTERVAL

        self.coeff_matrix = torch.linalg.inv(
            torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0], [1, T, T**2, T**3, T**4, T**5],
                          [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                          [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3]],
                         device=device,
                         dtype=torch.float32))
        ##### # 0.1 , 2.0, 20
        t_vec = torch.linspace(self.time_interval, REFINE_HORIZON, NUM_REFINE, device=device,
                               dtype=torch.float32)  # (time_len - 1,) 0.1,,, 2.0
        # i want to make -2.0 , -1.9, ..., -0.1
        t_veec_2 = torch.linspace(-REFINE_HORIZON, -self.time_interval,
                                    NUM_REFINE,
                                    device=device,
                                    dtype=torch.float32)
        self.T_mat = torch.stack([t_veec_2 ** i for i in range(6)],
                            dim=-1)  # (time_len - 1, 6)

        # TIME_INTERVAL: 0.1, REFINE_HORIZON: 2.0, NUM_REFINE: 20
        a = torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(
            1)  # (20, 1) # [0.1, 0.2, ..., 2.0]
        b = torch.arange(6).unsqueeze(0)  # (1,6) # [[0, 1, 2, 3, 4, 5]]
        self.t_matrix = torch.pow(a, b).to(device=device)  # shape (B, N+1)

    def _debug_visualize_states(
            self,
            neighbor_agents_past: torch.Tensor,  # (B, agent_num, time_len, 11)
            neighbors_future_all: torch.Tensor,  # (B, agent_num, future_len, 3)
            aug_near_current: torch.Tensor,  # (B, Pnn, 11)
            aug_flags: torch.Tensor,  # (B, Pnn)  (bool)
            batch_idx: int = 0,
            save_path: str = "debug_vis.png",
    ) -> None:
        """
        • 배치 `batch_idx` 만 시각화
        • 빨간색 : 원본 NPC 궤적/상태
        • 파란색 : 증강된 현재 NPC
        • 과거/미래 궤적은 `aug_flags==True` & **실제 존재** 에이전트만 그림
        • 패딩(전부 0) 슬롯은 자동으로 무시
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import math, numpy as np

        # ── 텐서 → NumPy ───────────────────────────────
        past = neighbor_agents_past[batch_idx].cpu().numpy()  # (N, T, 11)
        fut = neighbors_future_all[batch_idx].cpu().numpy()  # (N, F, 3)
        aug = aug_near_current[batch_idx].cpu().numpy()  # (Pnn, 11)
        flags = aug_flags[batch_idx].cpu().numpy().astype(bool)  # (Pnn,)

        agent_num, T, _ = past.shape
        F = fut.shape[1]

        # ── ① 유효 슬롯 마스크 ──────────────────────────
        valid_agents = (np.abs(past[:, -1, :]).sum(axis=1) > 1e-6)  # (N,) # bool
        valid_augents = (np.abs(aug).sum(axis=1) > 1e-6)  # (Pnn,) # bool

        # flags & 존재 여부를 모두 만족하는 인덱스
        sel_idx = np.where(flags & valid_augents)[0] # (Pnn,)

        # ── ② Figure 초기화 ─────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

        def _rect_corners(x, y, yaw, w, l):
            hw, hl = w / 2.0, l / 2.0
            local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
            R = np.array([[math.cos(yaw), -math.sin(yaw)],
                          [math.sin(yaw), math.cos(yaw)]])
            return local @ R.T + np.array([x, y])

        # ── ③ 과거 궤적 & heading (빨간색) ───────────────
        for i in sel_idx:  # flags==True & 존재
            for t in range(T - 1):
                x, y = past[i, t, 0:2]
                cos_, sin_ = past[i, t, 2:4]
                ax.plot(x, y, "o", color="red", ms=2)
                ax.arrow(x, y, cos_, sin_,
                         head_width=0.05, head_length=0.1,
                         color="red", linewidth=0.5,
                         length_includes_head=True)

        # ── ④ 미래 궤적 & heading (빨간색) ───────────────
        for i in sel_idx:
            for f in range(F):
                x, y, yaw = fut[i, f]
                ax.plot(x, y, "o", color="red", ms=2)
                ax.arrow(x, y, math.cos(yaw), math.sin(yaw),
                         head_width=0.05, head_length=0.1,
                         color="red", linewidth=0.5,
                         length_includes_head=True)

        # ── ⑤ 현재 프레임 (빨간 사각형) ──────────────────
        for i in np.where(valid_agents)[0]:
            x, y = past[i, -1, 0:2]
            cos_, sin_ = past[i, -1, 2:4]
            yaw = math.atan2(sin_, cos_)
            w, l = past[i, -1, 6:8]
            rect = Polygon(_rect_corners(x, y, yaw, w, l),
                           closed=True, fill=False,
                           edgecolor="red", linewidth=1.0)
            ax.add_patch(rect)
            ax.arrow(x, y, math.cos(yaw), math.sin(yaw),
                     head_width=0.1, head_length=0.2,
                     color="red", linewidth=1.0,
                     length_includes_head=True)

        # ── ⑥ 증강 현재 프레임 (파란 사각형) ──────────────
        for k in np.where(valid_augents)[0]:
            x, y = aug[k, 0:2]
            cos_, sin_ = aug[k, 2:4]
            yaw = math.atan2(sin_, cos_)
            w, l = aug[k, 6:8]
            style = '--' if flags[k] else '-'
            rect = Polygon(_rect_corners(x, y, yaw, w, l),
                           closed=True, fill=False,
                           edgecolor="blue", linestyle=style, linewidth=1.0)
            ax.add_patch(rect)
            ax.arrow(x, y, math.cos(yaw), math.sin(yaw),
                     head_width=0.1, head_length=0.2,
                     color="blue", linewidth=1.0,
                     length_includes_head=True)

            # 필요 시 속도벡터(주석 해제)
            # if flags[k]:
            #     vx0, vy0 = past[k, -1, 4], past[k, -1, 5]
            #     ax.arrow(x, y, vx0, vy0,
            #              head_width=0.1, head_length=0.2,
            #              color="red", linewidth=1.0,
            #              length_includes_head=True)
            #     vx1, vy1 = aug[k, 4], aug[k, 5]
            #     ax.arrow(x, y, vx1, vy1,
            #              head_width=0.1, head_length=0.2,
            #              color="blue", linewidth=1.0,
            #              length_includes_head=True)

        # ── ⑦ 마무리 ────────────────────────────────────
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    def normalize_angle(
        self, angle: Union[np.ndarray,
                           torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (angle + np.pi) % (2 * np.pi) - np.pi


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
        near_agents_current = neighbor_agents_past[:, :
                                                   args.predicted_neighbor_num,
                                                   -1, :]  # (B, pnn, 11)
        # near_current_wrt_self: (B, pnn, 11)
        near_current_wrt_self = self.convert_near_current_from_ego_to_self(
            near_agents_current)
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
        near_current_xyyaw = near_agents_current[:, :, :4]  # (B, pnn, 4)
        # aug_near_current: (B, pnn, 11)
        aug_near_current = self.convert_near_current_from_self_to_ego(
            near_current_xyyaw, # (B, Pnn, 4)
            aug_near_current_wrt_self # (B, Pnn, 11)
        )
        self._debug_visualize_states(
            neighbor_agents_past.clone().detach(),
            neighbors_future_all.clone().detach(),
            aug_near_current.clone().detach(),
            aug_flags.clone().detach(),
            batch_idx=0,
            save_path=f"debug_vis_{self.count}.png"  # 필요 시 경로/파일명 변경
        )
        """
        neighbor_agents_past: (B, agent_num, time_len, 11)
        neighbors_future_all: (B, agent_num, future_len, 3) -> (B, agent_num, future_len, 4)
        aug_flags: (B, agent_num) (bool)
        """
        neighbor_agents_past, neighbors_future_all, aug_flags = \
            self.reorder_neighbors_after_augmentation(
            aug_near_current, neighbor_agents_past, neighbors_future_all,
            aug_flags)
        neighbors_future_all_dim_three = torch.zeros(
            neighbors_future_all.shape[0],
            neighbors_future_all.shape[1],
            neighbors_future_all.shape[2],
            3,
            device=neighbors_future_all.device,
            dtype=neighbors_future_all.dtype
        )
        # neighbors_future_all_dim_three: (B, agent_num, future_len, 3)
        neighbors_future_all_dim_three[:, :, :, 0:2] = neighbors_future_all[
            :, :, :, 0:2]
        neighbors_future_all_dim_three[:, :, :, 2] = torch.atan2(
            neighbors_future_all[:, :, :, 3],
            neighbors_future_all[:, :, :, 2])
        aug_near_current_new = neighbor_agents_past[:,
            :args.predicted_neighbor_num, -1, :] # (B, Pnn, 11)
        aug_flags_new = aug_flags[:, :args.predicted_neighbor_num]  # (B, Pnn)
        self._debug_visualize_states(
            neighbor_agents_past.clone().detach(),
            neighbors_future_all_dim_three.clone().detach(),
            aug_near_current_new.clone().detach(),
            aug_flags_new.clone().detach(),
            batch_idx=0,
            save_path=f"debug_vis_{self.count}_revised.png"  # 필요 시 경로/파일명 변경
        )
        ##################
        neighbor_agents_past = self.refine_neighbor_past_trajectories(
            aug_flags, neighbor_agents_past
        )
        inputs["neighbor_agents_past"] = neighbor_agents_past
        # neighbors_future_all: (B, agent_num, future_len, 4) -> (B, agent_num, future_len, 3)
        neighbor_agents_current = neighbor_agents_past[:, :, -1, :]  # (B, agent_num, 11)
        neighbors_future_all = self.interpolation_future_trajectory(
            neighbor_agents_current, neighbors_future_all, aug_flags)
        # return neighbors_future : (B, Pnn, future_len, 3)
        neighbors_future = neighbors_future_all[:, :args.predicted_neighbor_num,
                                                    :, :]
        self.count += 1
        return inputs, neighbors_future


    def refine_neighbor_past_trajectories(self,
            aug_flags: Tensor,  # (B, agent_num)
            ordered_neighbor_agents_past: Tensor,
            # (B, agent_num, time_len, 11)
    ) -> Tensor:
        """
        Quintic 5차 스플라인으로 **과거 2초(20step)** NPC 궤적을 보정한다.
        - 현재 프레임(t=0s)은 그대로 두고, t=−2s~−0.1s 전 구간을 재생성
        - 시작·끝 점의 **위치·속도·가속도·yaw‑rate** 경계조건을 정확히 만족

        Args:
            aug_flags (Tensor): (B,agent_num)
                각 배치의 Pnn개 예측 대상(agent)별 augment 여부.
            ordered_neighbor_agents_past (Tensor): (B,agent_num,time_len,11)
                과거 20step(–2s→–0.1s)+현재 프레임 순서의 궤적.

        Returns:
            neighbor_agents_past: (B,agent_num,time_len,11)
                `ordered_neighbor_agents_past` 의 복사본에 보정이 적용된 결과.
        """
        # ───────────────── 기본 파라미터 ─────────────────
        B, agent_num, time_len, D = ordered_neighbor_agents_past.shape
        device, dtype = ordered_neighbor_agents_past.device, ordered_neighbor_agents_past.dtype
        T: float = self.refine_horizon  # 2.0

        # ───────────────── 1. 대상 마스크 계산 ─────────────────
        mask_flat: Tensor = aug_flags.clone().reshape(-1)  # (B * agent_num)

        if mask_flat.sum() == 0:
            # 보정 대상이 없으면 원본 그대로 반환
            return ordered_neighbor_agents_past.clone()

        # ───────────────── 2. 데이터 전개 ─────────────────
        neighbor_agents_past = ordered_neighbor_agents_past.reshape(-1, time_len,
                                                    D)  # (B·agent_num, time_len, 11)

        neighbor_current = neighbor_agents_past[:, -1]  # (B·agent_num, 11)   (t = 0s)
        neighbor_past = neighbor_agents_past[:, :-1]  # (B·agent_num, time_len - 1, 11)

        sel_idx = mask_flat.nonzero(as_tuple=True)[0]  # (M,)  보정 대상 인덱스
        M: int = sel_idx.size(0)

        aug_neighbor_current: Tensor = neighbor_current[sel_idx]  # (M, 11)
        aug_neighbor_past: Tensor = neighbor_past[sel_idx]  # (M, time_len - 1, 11)

        # ───────────────── 3. Quintic 경계조건 ─────────────────
        # 현재(t=0)
        x0, y0 = aug_neighbor_current[:, 0], aug_neighbor_current[:, 1]  # (M,)
        cos0, sin0 = aug_neighbor_current[:, 2], aug_neighbor_current[:, 3]
        theta0 = torch.atan2(sin0, cos0)  # (M,)

        v0 = torch.norm(aug_neighbor_current[:, 4:6],
                        dim=-1)  # (M,) speed magnitude
        # 가속도 벡터 (t=-0.1 → t=0)
        a0_vec = (aug_neighbor_current[:, 4:6] - aug_neighbor_past[:, -1,
                                             4:6]) / self.time_interval
        a0 = torch.norm(a0_vec, dim=-1)  # (M,)

        theta_prev = torch.atan2(aug_neighbor_past[:, -1, 3],
                                 aug_neighbor_past[:, -1, 2])  # (M,)
        omega0 = self.normalize_angle(theta0 - theta_prev) / self.time_interval  # (M,)

        # 과거 첫 프레임(t = –2s, index 0)
        xT, yT = aug_neighbor_past[:, 0, 0], aug_neighbor_past[:, 0, 1]
        cosT, sinT = aug_neighbor_past[:, 0, 2], aug_neighbor_past[:, 0, 3]
        thetaT = torch.atan2(sinT, cosT)

        vT = torch.norm(aug_neighbor_past[:, 0, 4:6], dim=-1)  # (M,)
        aT_vec = (aug_neighbor_past[:, 1, 4:6] - aug_neighbor_past[:, 0,
                                             4:6]) / self.time_interval
        aT = torch.norm(aT_vec, dim=-1)

        theta_next = torch.atan2(aug_neighbor_past[:, 1, 3], aug_neighbor_past[:, 1, 2])
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
        # coeff_matrix: (6, 6) # sx, sy: (M, 6) -> (M, 6, 1)
        # (6, 6) @ (M, 6, 1) -> (M, 6, 1) -> (M, 6)
        ax_coef = (self.A_inv_const @ sx.unsqueeze(-1)).squeeze(-1)  # (M, 6)
        ay_coef = (self.A_inv_const @ sy.unsqueeze(-1)).squeeze(-1)  # (M, 6)

        # ───────────────── 5. 궤적 샘플링 (t = -2 … -0.1) ─────────────────

        # 양수 방향으로 생성 후 flip → 과거→최근 순서
        traj_x = (self.T_mat @ ax_coef.T).T  # (M, time_len - 1)
        traj_y = (self.T_mat @ ay_coef.T).T  # (M, time_len - 1)

        # ────────── 5‑①. 속도·헤딩 (forward 차분) ──────────
        # pos_x/y 에 현재(t=0) 위치 붙여 forward diff
        pos_x = torch.cat([traj_x, x0.unsqueeze(1)], dim=1)  # (M, time_len)
        pos_y = torch.cat([traj_y, y0.unsqueeze(1)], dim=1)  # (M, time_len)

        dx = pos_x[:, 1:] - pos_x[:, :-1]  # (M, time_len - 1)
        dy = pos_y[:, 1:] - pos_y[:, :-1]  # (M, time_len - 1)

        vx = dx / self.time_interval  # (M, time_len - 1)
        vy = dy / self.time_interval
        # 0으로 나누는 경우 방지하여 heading 구하기
        # ── 정지‑구간 안전 처리 ────────────────────────────────
        heading_raw = torch.atan2(dy, dx)  # (M, steps)
        mask_zero = (dx.abs() + dy.abs()) < 1e-6  # True → dx=dy≈0

        heading = heading_raw.clone()
        heading[:, 0] = torch.where(mask_zero[:, 0], thetaT,
                                    heading_raw[:, 0])

        # forward‑fill : 1~steps‑1
        for t in range(1, heading.shape[1]):
            heading[:, t] = torch.where(mask_zero[:, t], heading[:, t - 1],
                                        heading_raw[:, t])
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)

        # ────────── 5‑②. new_seg 구성 (M, time_len - 1, 6) ──────────
        new_seg = torch.zeros((M, time_len - 1, 6), device=device, dtype=dtype)
        new_seg[..., 0] = traj_x
        new_seg[..., 1] = traj_y
        new_seg[..., 2] = cos_h
        new_seg[..., 3] = sin_h
        new_seg[..., 4] = vx
        new_seg[..., 5] = vy
        # 첫 경계 조건

        # ───────────────── 6. neighbor_past 덮어쓰기 (전 구간) ─────────────────
        # (B·agent_num, time_len - 1, 11)
        neighbor_past[sel_idx, :, :6] = new_seg  # t = −2 … −0.1 전부 교체

        # ───────────────── 7. 원본 텐서 복원 ─────────────────
        # (B·agent_num, time_len, 11)
        neighbor_agents_past[sel_idx, :-1, :6] = neighbor_past[sel_idx, :, :6]
        neighbor_agents_past = neighbor_agents_past.reshape(B, agent_num, time_len, D)

        return neighbor_agents_past


    def interpolation_future_trajectory(
        self,
        neighbor_agents_current: torch.Tensor,
        neighbors_future_all: torch.Tensor,
        aug_flags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quintic 스플라인으로 **NPC 미래 궤적** 앞 `20` step(0.1s 간격, 2s 구간)을
        부드럽게 재보간한다.

        Args:
            neighbor_agents_current (Tensor):
                증강된 현재 NPC 상태.
                **Shape** – ``(B, agent_num, 11)``
                ``[x, y, cosθ, sinθ, v_x, v_y, a_x, a_y, steer, yaw_rate, padding]``
            neighbors_future_all (Tensor):
                보간 전 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 4)``
                ``[x, y, cosθ, sinθ]`` × future_len
            aug_flags (Tensor):
                보간 수행 여부 플래그.
                **Shape** – ``(B, agent_num,)`` bool

        Returns:
            Tensor:
                ``neighbors_future_all`` –
                보간이 적용된 미래 궤적.
                **Shape** – ``(B, agent_num, future_len, 3)``
        """
        # neighbor_agents_current: (B, agent_num, 11) -> (B * agent_num, 11)
        neighbor_agents_current = neighbor_agents_current.reshape(-1,
                                                    neighbor_agents_current.shape[-1])
        aug_flags = aug_flags.reshape(-1)  # (B * agent_num,)
        aug_flags = aug_flags.to(self._device)
        if aug_flags.sum() == 0:
            # If no augmentation is needed, return the original future trajectory
            # neighbors_future_all: (B, agent_num, future_len, 4) -> (B , agent_num, future_len, 3)
            neighbors_future_cos = neighbors_future_all[..., 2]  # (B, agent_num, future_len)
            neighbors_future_sin = neighbors_future_all[..., 3]  # (B, agent_num, future_len)
            neighbors_future_all = torch.cat(
                [
                    neighbors_future_all[..., :2],  # x, y
                    torch.atan2(neighbors_future_sin, neighbors_future_cos)[..., None],  # heading
                ],
                dim=-1)
            return neighbors_future_all
        sel_idx = aug_flags.nonzero(as_tuple=True)[0]  # (M,)  보정 대상 인덱스
        neighbors_future_all = neighbors_future_all.to(self._device)
        neighbor_agents_current = neighbor_agents_current.to(self._device)  # (B * agent_num, 11)
        aug_neighbor_agents_current = neighbor_agents_current[sel_idx]  # (M, 11)

        # neighbors_future_all: (B, agent_num, future_len, 4)
        # (B, agent_num, future_len, 4) -> (B * agent_num, future_len, 4)  -> (B * agent_num, future_len, 3)
        batch_, agent_num, future_len, four_dim = neighbors_future_all.shape
        neighbor_future = neighbors_future_all.reshape(batch_ * agent_num, future_len,
                                          four_dim)  # (B * agent_num, future_len, 4)
        aug_neighbor_future = neighbor_future[sel_idx]  # (M, future_len, 4)

        # (M, future_len, 4) -> (M, future_len, 3)
        cos_ = aug_neighbor_future[:, :, 2]  # cos # (M, future_len)
        sin_ = aug_neighbor_future[:, :, 3]  # sin # (M, future_len)
        aug_neighbor_future = torch.cat(
            [
                aug_neighbor_future[:, :, :2],  # x, y
                torch.atan2(sin_, cos_)[..., None],  # heading
            ],
            dim=-1)  # (M, future_len, 3)
        
        aug_neighbor_future = aug_neighbor_future.to(self._device)

        refine_P = self.num_refine  # 20
        dt = self.time_interval  # 0.1
        M = aug_neighbor_agents_current.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(M, -1, -1)
        # (M, 6, 6)
        A = self.coeff_matrix.unsqueeze(0).expand(M, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        x0, y0, theta0, v0 = (
            aug_neighbor_agents_current[:, 0],  # (M,)
            aug_neighbor_agents_current[:, 1],
            torch.atan2(
                (aug_neighbor_future[:, int(refine_P / 2), 1] - aug_neighbor_agents_current[:, 1]),
                (aug_neighbor_future[:, int(refine_P / 2), 0] -
                 aug_neighbor_agents_current[:, 0])),
            torch.norm(aug_neighbor_agents_current[:, 4:6], dim=-1),
        )
        first_fut_point_vel = (aug_neighbor_future[:, 1, 0:2] -
                               aug_neighbor_future[:, 0, 0:2]) / dt  # (M, 2)
        a0_vec = (first_fut_point_vel - aug_neighbor_agents_current[:, 4:6]) / dt
        a0 = torch.norm(a0_vec, dim=-1)
        omega0 = self.normalize_angle(aug_neighbor_future[:, 0, 2] - theta0) / dt

        xT, yT, thetaT, vT, aT, omegaT = (
            aug_neighbor_future[:, refine_P,
                        0], aug_neighbor_future[:, refine_P,
                                        1], aug_neighbor_future[:, refine_P, 2],
            torch.norm(
                aug_neighbor_future[:, refine_P, :2] - aug_neighbor_future[:, refine_P - 1, :2],
                dim=-1) / dt,
            torch.norm(aug_neighbor_future[:, refine_P, :2] -
                       2 * aug_neighbor_future[:, refine_P - 1, :2] +
                       aug_neighbor_future[:, refine_P - 2, :2],
                       dim=-1) / dt**2,
            self.normalize_angle(aug_neighbor_future[:, refine_P, 2] -
                                 aug_neighbor_future[:, refine_P - 1, 2]) / dt)

        # Boundary conditions
        sx = torch.stack([
            x0, v0 * torch.cos(theta0), a0 * torch.cos(theta0) -
            v0 * torch.sin(theta0) * omega0, xT, vT * torch.cos(thetaT),
            aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT
        ],
                         dim=-1) # (M, 6)

        sy = torch.stack([
            y0, v0 * torch.sin(theta0), a0 * torch.sin(theta0) +
            v0 * torch.cos(theta0) * omega0, yT, vT * torch.sin(thetaT),
            aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT
        ],
                         dim=-1)
        # (M, 6, 6) @ (M, 6, 1) -> (M, 6, 1)
        ax_coef = A @ sx[:, :, None]  # M, 6, 1
        ay_coef = A @ sy[:, :, None]  # M, 6, 1
        # M_t: (M, P, 6) @ ax_coef: (M, 6, 1) -> (M, P, 1)
        traj_x = M_t @ ax_coef  # (M, P, 1)
        traj_y = M_t @ ay_coef  # (M, P, 1)
        a = torch.atan2(traj_y[:, :1, 0] - y0.unsqueeze(-1),
                        traj_x[:, :1, 0] - x0.unsqueeze(-1))  # (M, 1)
        b = torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0],
                        traj_x[:, 1:, 0] - traj_x[:, :-1, 0])  # (M, P-1)
        traj_heading = torch.cat([a, b], dim=1)  # (M, P)


        traj_cos = torch.cos(traj_heading)  # (M, P)
        traj_sin = torch.sin(traj_heading)
        # (M, P, 4)
        refined_result = torch.stack(
            [
                traj_x[:, :, 0],  # x
                traj_y[:, :, 0],  # y
                traj_cos,  # cos
                traj_sin,  # sin
            ],
            dim=-1)  # (M, P=20, 4)
        """
        `refined_result`(20 step)을
        `ordered_neighbors_interpolated_future[:, :, 0:20]` 영역에 복사한다.
        보간 대상은 `aug_flags == True` 인 샘플에 한정한다.
        """
        neighbor_future[sel_idx, :refine_P, :] = refined_result  
        # neighbor_future: (B * agent_num, future_len, 4) -> (B, agent_num, future_len, 4)

        neighbor_future = neighbor_future.reshape(batch_, agent_num, future_len, four_dim)
        # neighbor_future: (B, agent_num, future_len, 4) -> (B, agent_num, future_len, 3)
        cos_yaw = neighbor_future[:, :, :, 2]  # (B, agent_num, future_len)
        sin_yaw = neighbor_future[:, :, :, 3]  # (B, agent_num, future_len)
        yaw = torch.atan2(sin_yaw, cos_yaw)  # (B, agent_num, future_len)
        neighbor_future = torch.cat(
            [
                neighbor_future[:, :, :, :2],  # x, y
                yaw[..., None],  # heading
            ],
            dim=-1).to(self._device) # (B, agent_num, future_len, 3)
        return neighbor_future

    @staticmethod
    def convert_near_current_from_ego_to_self(
            near_agents_current: torch.Tensor
    ) -> torch.Tensor:
        """
        NPC 상태 텐서를 **각 NPC 자신의 로컬 좌표계**로 변환한다.
        (패딩‧빈 슬롯은 그대로 0 을 유지)

        Args
        ----
        near_agents_current : Tensor
            - shape: **(B, pnn, 11)**
            - 뒤쪽 슬롯은 모두 0 으로 패딩될 수 있음.

        Returns
        -------
        Tensor
            - shape: **(B, pnn, 11)**
            - 실제 NPC(0 이 아닌 슬롯)만 원점·정면 기준 좌표로 변환
            - 패딩 슬롯(전부 0)은 변환 없이 그대로 유지
        """
        if near_agents_current.ndim != 3 or near_agents_current.size(-1) != 11:
            raise ValueError(
                f"'near_agents_current' must have shape (B, pnn, 11); "
                f"got {near_agents_current.shape}"
            )

        # (B, pnn, 11) → 복사본 생성 (grad 보존)
        near_current_wrt_self: torch.Tensor = near_agents_current.clone()

        # ───── 0. 실제‑NPC 마스크 (전부 0? → False) ──────────────────────
        # shape: (B, pnn)
        valid_mask = near_agents_current.abs().sum(dim=-1) > 0
        if not valid_mask.any():
            # 모든 슬롯이 0 이면 그대로 반환
            return near_current_wrt_self

        mask = valid_mask.unsqueeze(-1)  # (B, pnn, 1)  브로드캐스트용

        # ───── 1. heading 정보 ──────────────────────────────────────────
        cos_phi = near_agents_current[..., 2]  # (B, pnn)
        sin_phi = near_agents_current[..., 3]

        # ───── 2. 속도 회전 (R(-phi)) ──────────────────────────────────
        vx = near_agents_current[..., 4]
        vy = near_agents_current[..., 5]

        vx_rot = vx * cos_phi + vy * sin_phi
        vy_rot = -vx * sin_phi + vy * cos_phi

        # ───── 3. 실제‑NPC 위치만 덮어쓰기 ──────────────────────────────
        # x', y', cos, sin
        near_current_wrt_self[..., 0:4] = torch.where(
            mask.expand_as(near_current_wrt_self[..., 0:4]),
            torch.tensor([0.0, 0.0, 1.0, 0.0],
                         device=near_agents_current.device,
                         dtype=near_agents_current.dtype)
            .view(1, 1, 4),  # 상수 벡터 (0,0,1,0)
            near_current_wrt_self[..., 0:4]
        )

        # vx', vy'
        near_current_wrt_self[..., 4:6] = torch.where(
            mask, torch.stack([vx_rot, vy_rot], dim=-1), near_current_wrt_self[..., 4:6]
        )

        # width(6), length(7), class‑one‑hot(8:11) 는 그대로 유지
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
        k = int(round(predicted_neighbor_num * augment_prob + 0.00001))

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
        # batch_idx: (B, k)
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
    def convert_near_current_from_self_to_ego(
        near_current_xyyaw: Tensor,            # (B, pnn, 4)
        aug_near_current_wrt_self: Tensor,     # (B, pnn, 11)
    ) -> Tensor:
        """
        NPC 상태를 **자기 프레임 → ego 프레임** 으로 변환하되,
        zero‑padding 슬롯은 그대로 둔다.
        """
        # ─── (1) 입력 검증 ──────────────────────────────
        if near_current_xyyaw.ndim != 3 or near_current_xyyaw.size(-1) != 4:
            raise ValueError("'near_current_xyyaw' shape must be (B,pnn,4)")
        if aug_near_current_wrt_self.ndim != 3 or aug_near_current_wrt_self.size(-1) != 11:
            raise ValueError("'aug_near_current_wrt_self' shape must be (B,pnn,11)")
        if near_current_xyyaw.shape[:2] != aug_near_current_wrt_self.shape[:2]:
            raise ValueError("batch/pnn 차원이 다릅니다.")

        # ─── (2) dtype·device 통일 ─────────────────────
        device, dtype = aug_near_current_wrt_self.device, aug_near_current_wrt_self.dtype
        near_current_xyyaw = near_current_xyyaw.to(device=device, dtype=dtype)

        # ─── (3) 유효‑마스크(valid_mask) 생성 ───────────
        # padding 슬롯은 벡터 전체가 0 → sum == 0
        valid_mask = (
            near_current_xyyaw.abs().sum(dim=-1) > 0
        ) & (
            aug_near_current_wrt_self.abs().sum(dim=-1) > 0
        )                                         # (B, pnn) bool
        mask_exp = valid_mask.unsqueeze(-1)        # (B, pnn, 1)  브로드캐스트용

        # ─── (4) 로컬→ego 변환 (모든 슬롯 계산) ──────────
        # self‑frame 값
        x_loc, y_loc = aug_near_current_wrt_self[..., 0], aug_near_current_wrt_self[..., 1]
        cos_loc, sin_loc = aug_near_current_wrt_self[..., 2], aug_near_current_wrt_self[..., 3]
        vx_loc,  vy_loc  = aug_near_current_wrt_self[..., 4], aug_near_current_wrt_self[..., 5]

        # ego‑frame 기준 NPC 원점·heading
        x0, y0 = near_current_xyyaw[..., 0], near_current_xyyaw[..., 1]
        cos0, sin0 = near_current_xyyaw[..., 2], near_current_xyyaw[..., 3]

        # 위치
        x_ego = cos0 * x_loc - sin0 * y_loc + x0
        y_ego = sin0 * x_loc + cos0 * y_loc + y0
        # heading
        cos_ego = cos0 * cos_loc - sin0 * sin_loc
        sin_ego = sin0 * cos_loc + cos0 * sin_loc
        # 속도
        vx_ego = cos0 * vx_loc - sin0 * vy_loc
        vy_ego = sin0 * vx_loc + cos0 * vy_loc

        # ─── (5) 결과 조립 (유효 슬롯만 덮어쓰기) ────────
        aug_near_current = aug_near_current_wrt_self.clone()      # (B, pnn, 11)

        aug_near_current[..., 0] = torch.where(mask_exp.squeeze(-1), x_ego, aug_near_current[..., 0])
        aug_near_current[..., 1] = torch.where(mask_exp.squeeze(-1), y_ego, aug_near_current[..., 1])
        aug_near_current[..., 2] = torch.where(mask_exp.squeeze(-1), cos_ego, aug_near_current[..., 2])
        aug_near_current[..., 3] = torch.where(mask_exp.squeeze(-1), sin_ego, aug_near_current[..., 3])
        aug_near_current[..., 4] = torch.where(mask_exp.squeeze(-1), vx_ego, aug_near_current[..., 4])
        aug_near_current[..., 5] = torch.where(mask_exp.squeeze(-1), vy_ego, aug_near_current[..., 5])
        # width(6), length(7), class one‑hot(8:10)은 그대로 유지

        return aug_near_current


    @staticmethod
    def reorder_neighbors_after_augmentation(
        aug_near_current: torch.Tensor,        # (B, Pnn, 11)
        neighbor_agents_past: torch.Tensor,   # (B, agent_num, time_len, 11)
        neighbors_future_all: torch.Tensor,   # (B, agent_num, future_len, 3)
        aug_flags: torch.Tensor,              # (B, Pnn) (bool)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - 증강된 Pnn대 NPC의 현재 상태를 과거·미래 시계열에 반영한 뒤
        - ***실제로 존재하는*** NPC(전부 0 이 아님)만 거리순(ego→NPC)으로 정렬
        - 0‑패딩 슬롯(가짜 NPC)은 항상 배열 뒤쪽으로 유지
        - 증강 여부 플래그도 동일한 새 순서로 재정렬
        """
        # ─── 0. 기본 정보 ──────────────────────────────────────────
        B, Pnn, _ = aug_near_current.shape
        _, agent_num, time_len, _ = neighbor_agents_past.shape
        _, _, future_len, _ = neighbors_future_all.shape
        device   = neighbor_agents_past.device
        dtype    = neighbor_agents_past.dtype

        if Pnn > agent_num:
            raise ValueError(f"Pnn({Pnn}) must be ≤ agent_num({agent_num})")

        # dtype 정합성
        aug_near_current = aug_near_current.to(dtype=dtype, device=device)
        aug_flags        = aug_flags.to(device=device)

        # ─── 1. 현재 프레임에 증강 결과 반영 (Pnn 앞 슬롯만) ─────────────
        past_updated = neighbor_agents_past.clone()                # (B,N,T,11)
        past_updated[:, :Pnn, -1, :] = torch.where(
            aug_flags.unsqueeze(-1),                               # (B,Pnn,1)
            aug_near_current,                                      # (B,Pnn,11)
            past_updated[:, :Pnn, -1, :],
        )

        # ─── 2. “실제 NPC” 마스크 생성  (전부 0 ➜ False) ───────────────
        # 현재 프레임(-1) 전체 11‑D 벡터가 완전히 0 이면 가짜‑슬롯
        valid_mask: torch.Tensor = (past_updated[:, :, -1, :].abs().sum(dim=-1) > 0)  # (B,N)

        # ─── 3. 거리 계산 & 무한대 패널티 부여 ───────────────────────
        cur_xy = past_updated[:, :, -1, :2]                                      # (B,N,2)
        dist   = torch.linalg.norm(cur_xy, dim=-1)                               # (B,N)
        big_val = torch.tensor(float("inf"), device=device, dtype=dtype)
        dist = torch.where(valid_mask, dist, big_val)                            # 가짜 슬롯 → inf

        # 〈유효‑NPC 오름차순, 그다음 가짜‑NPC〉
        order_idx = dist.argsort(dim=1)                                          # (B,N)

        # ─── 4‑A. 과거 궤적 재배열 ───────────────────────────────────
        idx = order_idx.unsqueeze(-1).unsqueeze(-1)                              # (B,N,1,1)
        neighbor_agents_past = torch.take_along_dim(
            past_updated, idx.expand(-1, -1, time_len, 11), dim=1)               # (B,N,T,11)

        # ─── 4‑B. 미래 궤적(3→4ch) 변환 후 재배열 ───────────────────
        neighbors_future_all = neighbors_future_all.to(dtype=dtype, device=device)
        yaw = neighbors_future_all[..., 2]                                       # (B,N,F)
        future_4ch = torch.cat(
            [neighbors_future_all[..., :2],       # x,y
             torch.cos(yaw).unsqueeze(-1),
             torch.sin(yaw).unsqueeze(-1)],       # cos,sin
            dim=-1)                                                                    # (B,N,F,4)

        neighbors_future_all = torch.take_along_dim(
            future_4ch, idx.expand(-1, -1, future_len, 4), dim=1)                # (B,N,F,4)

        # ─── 5. aug_flags 새 순서로 재정렬 & 가짜‑NPC 위치엔 False ───
        flag_full = torch.zeros((B, agent_num), dtype=torch.bool, device=device) # (B,N)
        flag_full[:, :Pnn] = aug_flags                                           # 증강표시 주입
        # 가짜 슬롯엔 무조건 False
        flag_full = flag_full & valid_mask

        flag_reordered = torch.take_along_dim(flag_full, order_idx, dim=1)       # (B,N)

        return neighbor_agents_past, neighbors_future_all, flag_reordered