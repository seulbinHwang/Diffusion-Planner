import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder


class Diffusion_Planner(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

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
