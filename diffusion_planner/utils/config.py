import json
import torch

from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer


class Config:
    
    def __init__(
            self,
            args_file,
            guidance_fn
    ):
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
        """
        ego_current_state : 10
        neighbor_agents_past : 11 
        static_objects : 10
        lanes : 12
        lanes_speed_limit : 1
        route_lanes : 12
        route_lanes_speed_limit : 1
        """
        self.guidance_fn = guidance_fn