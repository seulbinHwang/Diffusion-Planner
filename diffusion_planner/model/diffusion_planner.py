import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from typing import Iterator


class Diffusion_Planner(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)
        # 현재 학습 Stage (1, 2, 3). 기본은 Stage1.
        self._training_stage: int = 1

    @property
    def training_stage(self) -> int:
        """현재 학습 Stage 번호를 돌려줍니다.

        Returns:
            int: 1, 2, 3 중 하나.
        """
        return self._training_stage

    def set_stage(self, stage: int) -> None:
        """학습 Stage를 설정하고, 로컬 인코더 얼림 여부를 바꿉니다.

        Stage 규칙:
            - 1: Encoder/Decoder 전부 학습.
            - 2: 로컬 인코더(Group A)만 얼리고 나머지는 학습.
            - 3: 다시 전부 학습. (학습률 차이는 optimizer에서 처리)
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"stage는 1, 2, 3 중 하나여야 합니다. 받은 값: {stage}")

        self._training_stage = int(stage)

        # 기본값: 전 파라미터 학습 가능
        for param in self.parameters():
            param.requires_grad_(True)

        # Stage2: 로컬 인코더만 freeze
        if stage == 2:
            encoder_core: Encoder = self.encoder.encoder
            if hasattr(encoder_core, "iter_encoder_local_parameters"):
                for param in encoder_core.iter_encoder_local_parameters():
                    param.requires_grad_(False)

    def iter_group_encoder_local_parameters(self) -> Iterator[nn.Parameter]:
        """로컬 인코더(Group A) 파라미터 이터레이터를 돌려줍니다."""
        encoder_core: Encoder = self.encoder.encoder
        if hasattr(encoder_core, "iter_encoder_local_parameters"):
            for param in encoder_core.iter_encoder_local_parameters():
                # param: (out_dim, in_dim) 또는 (dim,)
                yield param

    def iter_group_encoder_global_parameters(self) -> Iterator[nn.Parameter]:
        """글로벌 인코더(Group B) 파라미터 이터레이터를 돌려줍니다."""
        encoder_core: Encoder = self.encoder.encoder
        if hasattr(encoder_core, "iter_encoder_global_parameters"):
            for param in encoder_core.iter_encoder_global_parameters():
                # param: (out_dim, in_dim) 또는 (dim,)
                yield param

    def iter_group_decoder_parameters(self) -> Iterator[nn.Parameter]:
        """디코더(Group C) 파라미터 이터레이터를 돌려줍니다.

        Decoder 전체(DiT + PRAM + Feasible projector)의 파라미터가 포함됩니다.
        """
        for param in self.decoder.decoder.parameters():
            # param: (out_dim, in_dim) 또는 (dim,)
            yield param

    @property
    def sde(self):
        return self.decoder.decoder.sde

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs


class Diffusion_Planner_Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding MLP:

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs


class Diffusion_Planner_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        return

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return decoder_outputs
