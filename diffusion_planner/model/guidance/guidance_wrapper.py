from typing import List
import torch

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.guidance.collision import collision_guidance_fn
from diffusion_planner.model.guidance.feasible_gui import feasible_guidance_fn

N = 1
sde = VPSDE_linear()


class GuidanceWrapper:

    def __init__(self):
        self._guidance_fns = [feasible_guidance_fn
                             ]  # collision_guidance_fn 안쓸거임

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        This function is a wrapper for the guidance functions in the model.

        kwargs
            "model": self.dit,
            "model_condition":
            {
                "cross_c":
                    scene_encoding_token,
                "ego_fut_global":
                    ego_fut_global,
                "near_agents_route_lane_emb":
                    near_agents_route_lane_emb,
                "near_past_cur_future_valid":
                    near_past_cur_future_valid,  # [B, pnn, time_len(=1+past_len) + future_len] bool # 과거-현재-미래
                "cross_mask":
                    scene_encoding_token_mask,
                "route_known_mask":
                    route_known_mask,
                "near_class_one_hot":
                    near_class_one_hot,
                "near_current_xyyaw":
                    near_current_xyyaw,
                "near_past":
                    near_past,  # [B, pnn, past_len=(time_len - 1), 11]
            },
            "inputs": inputs,
            "observation_normalizer": self._observation_normalizer,
            "state_normalizer": self._state_normalizer
        """
        energy = 0

        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]
        config = kwargs["config"]

        # x_fix : (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
        # x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = model(x_in, t_input, **model_condition) - x_in

        assert x_fix.requires_grad, \
            " GuidanceWrapper 입력이 x_in에 대한 gradient를 가지지 않습니다."
        feasible_returns = model.dit_returns
        kwargs[
            "integrated_trajectory"] = feasible_returns.integrated_trajectory  # (B,Pnn,T,4)
        kwargs[
            "control_constraint_diff"] = feasible_returns.control_constraint_diff  # (B,Pnn,T,3)
        # x_fix : (B, Pnn, T, 4) or (B, Pnn, (1+T), 4)
        x_fix = x_fix.reshape(B, P, -1, 4)
        if config.use_current_input:
            x_fix[:, :, 0] = 0.0
        # x_dit : (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
        x_dit = x_in + x_fix.reshape(B, P, -1)

        # x_dit : (B, Pnn, T, 4) or (B, Pnn, (1+T), 4)
        # x_dit = state_normalizer.inverse(x_dit.reshape(B, P, -1, 4))
        x_dit = x_dit.reshape(B, P, -1, 4)  # 정규화된 상태 그대로 guidance_fn 에 넘김
        assert x_dit.requires_grad, \
            "GuidanceWrapper 출력이 x_dit에 대한 gradient를 가지지 않습니다."
        # x_dit에 의존하는 0 텐서 (B,)
        # TODO: 안 써서, 주석 처리함
        # kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])

        for guidance_fn in self._guidance_fns:
            energy += guidance_fn(x_dit, t_input, cond, **kwargs)

        assert not torch.isnan(energy).any()

        return energy
