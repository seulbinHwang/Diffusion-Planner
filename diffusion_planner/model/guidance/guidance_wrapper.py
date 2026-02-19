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
        TODO
2) GuidanceWrapper는 무조건 (...,4)로 reshape 함 → pose_based=False면 바로 터짐

GuidanceWrapper에서 x_fix = x_fix.reshape(B,P,-1,4) / x_dit = x_dit.reshape(B,P,-1,4)를 고정으로 함.

그러면 pose_based=False(3차원 control)에서는 shape이 3의 배수라서 바로 에러 날 거야.

3) model_type="v"일 때 GuidanceWrapper의 “의미”가 어긋날 수 있음

GuidanceWrapper는 x_dit를 사실상 model(x_in, t) 출력으로 만들고(= x_in + (model(x_in,t)-x_in)),

그걸 “정규화된 pose 궤적”처럼 reshape해서 feasible_guidance_fn(x_dit, ...)에 넣고 있어.

그런데 네 DiT는

model_type="x_start"면 model(...) 출력이 x0라서 괜찮을 수 있는데,

model_type="v"면 model(...) 출력이 v(속도 파라미터) 이고, 그걸 pose처럼 취급하면 guidance 의미가 깨질 가능성이 커.

→ 설계안에서 너가 강조한 “x0_base를 기반으로 마지막 정리” 관점과도 어긋나.
(v를 쓰는 경우라면, guidance가 참조할 “x_dit”를 model.diffusion_trajectory_flat(= x0) 쪽으로 잡는 게 더 일관돼.)

        """
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
        feasible_returns = model.norm_dit_returns
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
