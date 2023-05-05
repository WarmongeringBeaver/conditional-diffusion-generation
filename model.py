"""UNConditional Unet2D model.

Shamelessly copied from HF's excellent course on diffusion models.
"""

import torch
from diffusers.models import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from torch import nn


class ClassConditionedUnet(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor | int,
    ) -> UNet2DOutput:
        ### Shape of images:
        bs, _, w, h = images.shape

        if type(timesteps) == torch.Tensor:
            assert (
                timesteps.shape[0] == bs
            ), f"Batch size of images and timesteps must match: images.shape = {images.shape}, timesteps.shape = {timesteps.shape}"

        ### Forward pass
        # images is shape (bs, 3, image_size, image_size);

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(images, timesteps)
        # .sample: (bs, 3, image_size, image_size)
