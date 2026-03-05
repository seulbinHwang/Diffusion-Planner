from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from diffusion_planner.model.guidance.safety_guidance import safety_guidance_fn


class GuidanceWrapper:
    """여러 guidance 함수를 한 번에 호출하기 위한 래퍼입니다.

    핵심 규칙(중요)
    - DPM-Solver classifier 경로에서는 첫 입력 x_in이 "현재 샘플(x_t)"입니다.
      그래서 안전 점수는 반드시 모델이 만든 "복원 결과(x0_pred)"로 계산해야 합니다.
    - model_wrapper가 kwargs["x0_pred"]를 넘겨주면 그걸 그대로 씁니다.
    - amortized 1-step처럼 x0_pred가 없으면, 여기서 모델을 다시 호출해 x0_pred를 만들고 씁니다.
    """

    def __init__(self) -> None:
        # ✅ feasible_guidance_fn 제거, safety_guidance만 사용
        self._guidance_fns: List[Any] = [safety_guidance_fn]

    def _get_x0_pred_flat(
        self,
        x_in: torch.Tensor,  # (B,P,F)
        t_input: torch.Tensor,  # (B,) 또는 (B,T)도 가능(Decoder가 넘기면)
        *,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """guidance 계산에 사용할 x0_pred(flat)를 확보합니다.

        우선순위:
        1) kwargs["x0_pred"] 가 있으면 그대로 사용
        2) 없으면 model(x_in, diffusion_time, ...)을 다시 호출해서 model.diffusion_sequence_flat을 사용

        Args:
            x_in (torch.Tensor): 현재 샘플(flat). shape: (B,P,F)
            t_input (torch.Tensor): 대표 시간. shape: (B,)
            kwargs (Dict[str,Any]): model/model_condition/config 등이 들어 있음

        Returns:
            torch.Tensor: x0_pred(flat), shape: (B,P,F)
        """
        x0_pred = kwargs.get("x0_pred", None)
        if isinstance(x0_pred, torch.Tensor):
            return x0_pred

        model = kwargs.get("model", None)
        model_condition = kwargs.get("model_condition", None)
        if model is None or not isinstance(model_condition, dict):
            raise ValueError(
                "GuidanceWrapper: x0_pred가 없는데 model/model_condition을 찾을 수 없습니다."
            )

        # ✅ amortized 1-step에서는 Decoder가 (B,T)인 t_tau를 넘겨줄 수 있음
        diffusion_time_for_guidance = kwargs.get("diffusion_time_for_guidance", None)
        if isinstance(diffusion_time_for_guidance, torch.Tensor):
            diffusion_time = diffusion_time_for_guidance
        else:
            diffusion_time = t_input

        diffusion_time = diffusion_time.to(device=x_in.device, dtype=torch.float32)

        # (B,T) 시간을 쓰면, DiT 내부 feasible 쪽에서 (B,) low_t_mask가 필요할 수 있어 안전하게 제공
        low_t_mask = kwargs.get("low_t_mask_for_guidance", None)
        if low_t_mask is None and diffusion_time.dim() == 2:
            B = int(x_in.shape[0])
            low_t_mask = torch.ones((B,), device=x_in.device, dtype=torch.bool)

        # 모델 재호출(grad 흐름 유지)
        _ = model(
            x_in,
            diffusion_time,
            **model_condition,
            low_t_mask=low_t_mask,
        )

        x0_new = getattr(model, "diffusion_sequence_flat", None)
        if not isinstance(x0_new, torch.Tensor):
            raise RuntimeError("GuidanceWrapper: model.diffusion_sequence_flat을 얻지 못했습니다.")
        return x0_new

    def __call__(
        self,
        x_in: torch.Tensor,  # (B,P,F)  (DPM-Solver에서는 x_t)
        t_input: torch.Tensor,  # (B,)
        cond: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """guidance 점수(스칼라)를 (B,)로 반환합니다.

        Args:
            x_in (torch.Tensor): 현재 샘플(flat). shape: (B,P,F)
            t_input (torch.Tensor): 시간. shape: (B,)
            cond (Optional[torch.Tensor]): 사용 안 함
            **kwargs: model/model_condition/state_normalizer/config/inputs/x0_pred 등

        Returns:
            torch.Tensor: (B,) 점수 텐서
        """
        if x_in.dim() != 3:
            raise ValueError(f"GuidanceWrapper: x_in must be (B,P,F). got {tuple(x_in.shape)}")
        if t_input.dim() != 1:
            raise ValueError(f"GuidanceWrapper: t_input must be (B,). got {tuple(t_input.shape)}")

        # ✅ 핵심: 안전 점수는 x0_pred로 계산
        x0_pred_flat = self._get_x0_pred_flat(x_in=x_in, t_input=t_input, kwargs=kwargs)

        # safety_guidance_fn이 kwargs["x0_pred"]를 우선 사용하도록 같이 넣어둠
        local_kwargs = dict(kwargs)
        local_kwargs["x0_pred"] = x0_pred_flat

        B = int(x_in.shape[0])
        energy = x_in.new_zeros((B,), dtype=torch.float32)

        for fn in self._guidance_fns:
            out = fn(
                x0_pred_flat,  # 첫 인자도 x0로 맞춤
                t_input,
                cond,
                **local_kwargs,
            )
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"GuidanceWrapper: guidance fn must return Tensor, got {type(out)}")
            energy = energy + out.to(torch.float32)

        if torch.isnan(energy).any():
            raise RuntimeError("GuidanceWrapper: energy에 NaN이 있습니다.")
        return energy