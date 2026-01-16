from typing import Dict
import torch
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


def dpm_sampler(model: torch.nn.Module,
                x_T,
                other_model_params: Dict = {},
                diffusion_steps=10,
                noise_schedule_params: Dict = {},
                model_wrapper_params: Dict = {},
                dpm_solver_params: Dict = {},
                sample_params: Dict = {}):

    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(schedule='linear',
                                             **noise_schedule_params)
        """ DPM-Solver
            DPM-Solver는 내부적으로 model_fn(x, t_continuous)만 알고 호출합니다.
            그런데 실제 DiT는 (x, t_input, **조건들)을 받아야 하죠.
            model_wrapper는 그 사이를 연결해줍니다.
        """
        """ model_fn
                diffusion_planner의 출력값(회복 궤적) 을 활용해, 
                noise 를 출력하는 함수입니다.
            DPM-Solver 가 호출할 함수임
            만약 아래의 **model_wrapper_params 중 guidance_type 값이 "uncond" 이면
            
                다른 말로는, Decoder._guidance_fn 가 None 이면 = noise_pred_fn
                model_fn = dpm.noise_pred_fn(x, t_continuous, cond=None) 함수
                    t_input = t_continuous
                    output = model(x, t_input, **model_kwargs)
                    return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(
                        sigma_t, x.dim()) = 노이즈 형테의 예측값
            guidance_type 값이 "classifier" 이면, (다른 말로는 Decoder._guidance_fn 가 있으면)
                cond_grad = cond_grad_fn(x, t_input)
                return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
                    즉, cond_grad 을 통해 노이즈 예측값을 조정함
        """
        if diffusion_steps == 1:
            order = 1
            method = "singlestep_fixed"
            denoise_to_zero = False
        else:
            order = 2
            method = "multistep"
            denoise_to_zero = True
        model_fn = dpm.model_wrapper(
            model,  # use your noise prediction model here
            noise_schedule,
            model_type=model.model_type,  # or "x_start" or "v" or "score"
            model_kwargs=other_model_params,
            **model_wrapper_params)
        """ dpm_solver
        """
        dpm_solver = dpm.DPM_Solver(
            model_fn, # noise 를 출력하는 함수입니다.
            noise_schedule,
            algorithm_type="dpmsolver++",
            **dpm_solver_params)  #= {"correcting_xt_fn": correcting_xt_fn }

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(x_T,
                                       steps=diffusion_steps, # 10
                                       order=order,
                                       skip_type="logSNR",
                                       method=method,
                                       denoise_to_zero=denoise_to_zero, # 마지막에 한번 더 x0로 정리 하겠다.
                                       **sample_params)

    return sample_dpm
