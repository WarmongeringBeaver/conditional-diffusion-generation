# Load local dataset with hugging face Datasets library

from io import TextIOBase
from math import ceil
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms
from utils import header_print

# TODO's:
# - clean data loading (in particular, make a clear link between class index and class name.....)


def load_BBBC021_comp_conc_nice_phen(root_data_dir: str, selected_ds_names: list[str]):
    """Loads the BBBC021_comp_conc_nice_phen dataset."""
    header_print("Loading datasets from folder " + root_data_dir)
    header_print("Selected datasets: " + str(selected_ds_names) + "\n")
    ds_list = []
    for ds_name in selected_ds_names:
        ds_path = Path(root_data_dir, ds_name)
        ds = load_dataset(
            "imagefolder",
            data_dir=ds_path.as_posix(),
            cache_dir="./.HF_cache",
            drop_labels=False,
        )["train"]
        ds_list.append(ds)

    full_dataset = torch.utils.data.ConcatDataset(ds_list)

    return full_dataset


def preprocess_dataset(
    full_dataset: torch.utils.data.dataset.ConcatDataset,
    args: dict,
    logfile: TextIOBase,
):
    # resize to square images of size image_size x image_size
    image_size = args["image_size"]
    logfile.write(f"image_size: {image_size}\n")

    # on-the-fly data cleaning / augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )
    # Map to (-1, 1) because of th SILU activation function used (by default) in the UNet2DModel

    def transform(class_idx):
        def transform_for_this_class(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            classes = torch.tensor([class_idx] * len(examples))
            return {"images": images, "classes": classes}

        return transform_for_this_class

    for class_idx, dataset in enumerate(full_dataset.datasets):
        dataset.set_transform(transform(class_idx))

    # create a dataloader to serve up the transformed images in batches
    batch_size = args["batch_size"]
    logfile.write(f"batch_size: {batch_size}\n\n")
    nb_batches = ceil(len(full_dataset) / batch_size)

    dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader, nb_batches
