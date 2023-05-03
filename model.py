"""Conditional Unet2D model."""

import torch
from torch import nn
from diffusers.models import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes, class_emb_dim, image_size):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + class_emb_dim,
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
    def forward(self, x, t, class_labels) -> UNet2DOutput:
        # Shape of x:
        bs, _, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dinemsion
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(
            bs, class_cond.shape[1], w, h
        )
        # x is shape (bs, 3, image_size, image_size);
        # class_cond is now (bs, class_emb_dim, image_size, image_size)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)
        # (bs, 3+class_emb_dim, image_size, image_size)

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t)  # .sample: (bs, 3, image_size, image_size)
