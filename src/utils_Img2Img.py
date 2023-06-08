from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import normaltest
from torch import Tensor

from src.pipeline_conditional_ddim import ConditionialDDIMPipeline


def check_Gaussianity(latent: Tensor):
    print("Checking Gausianity...")
    _, axes = plt.subplots(nrows=1, ncols=len(latent), figsize=(16, 3))
    for idx, itm in enumerate(latent):
        axes[idx].hist(itm.cpu().numpy().flatten(), bins=100, range=(-3, 3))
        print(
            f"latent {idx}: mean={itm.mean().item()}, std={itm.std().item()}", end="; ")
        _, p = normaltest(itm.cpu().numpy().flatten())
        print(
            f"2-sided Χ² probability for the normality hypothesis: {p}"
        )
    plt.show()


def tensor_to_PIL(tensor: Tensor, pipeline: ConditionialDDIMPipeline):
    assert tensor.ndim == 4
    img_to_show = tensor.clone().detach()
    img_to_show = (img_to_show / 2 + 0.5).clamp(0, 1)
    img_to_show = img_to_show.cpu().permute(0, 2, 3, 1).numpy()
    img_to_show = pipeline.numpy_to_pil(img_to_show)
    if type(img_to_show) == list and len(img_to_show) == 1:
        return img_to_show[0]
    return img_to_show


def print_grid(list_PIL_images: list[Image], nb_img_per_row: int = 5, titles=None):
    if titles is not None:
        assert len(titles) == len(list_PIL_images)
    num_images = len(list_PIL_images)
    nrows = ceil(num_images / nb_img_per_row)
    _, axes = plt.subplots(
        nrows=nrows, ncols=nb_img_per_row, figsize=(16, 4*nrows)
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i in range(num_images):
        row_nb = i // nb_img_per_row
        col_nb = i % nb_img_per_row
        axes[row_nb, col_nb].imshow(list_PIL_images[i])
        axes[row_nb, col_nb].axis('off')
        if titles is not None:
            axes[row_nb, col_nb].set_title(titles[i])
    plt.tight_layout()
    plt.show()
