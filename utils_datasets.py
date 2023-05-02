# Load local dataset with hugging face Datasets library

import os
from io import TextIOBase
from math import ceil
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from pytorch_fid.fid_score import save_fid_stats
from torchvision import transforms
from utils import header_print

# TODO's:
# - clean data loading (in particular, make a clear link between class index and class name.....)
# - remove legacy code for HF toy dataset


def load_BBBC021_comp_conc_nice_phen(root_data_dir: str, selected_ds_names: list[str]):
    """Loads the BBBC021_comp_conc_nice_phen dataset."""
    header_print("Loading datasets from folder " + root_data_dir)
    header_print("Selected datasets: " + str(selected_ds_names) + "\n")
    ds_list = []
    for ds_name in selected_ds_names:
        ds_path = Path(root_data_dir, ds_name)
        ds = load_dataset(
            "imagefolder", data_dir=ds_path, cache_dir="./.HF_cache", drop_labels=False
        )["train"]
        ds_list.append(ds)

    full_dataset = torch.utils.data.ConcatDataset(ds_list)

    return full_dataset


def precompute_dataset_fid_statistics(
    root_data_dir: str,
    selected_ds_names: "list[str]",
    Inception_feat_dim: int,
    logfile,
    fid_batch_size: int = 50,
    device: str = "cuda",
    num_workers: int = 8,
    force_recompute: str = "False",
):
    """Computes the FID statistics for the selected datasets.
    
    Saves the statistics in the `datasets_stats` folder, in a file named:
        ``<dataset_name>_feat_dim_<Inception_feat_dim>.npz``.

    Arguments:
    ----------
    - `root_data_dir`: `str`\\
    The root directory where each dataset is stored in its own subfolder.
    - `selected_ds_names`: `list[str]`\\
    The names of the datasets loaded for the experiment and for which to compute the FID statistics.
    - `Inception_feat_dim`: `int`\\
    The dimension of the InceptionV3 features. Will change the FIDs!
    - `force_recompute`: `str`\\
    STRING for now, because of parsing...
    """
    header_print("Precomputing FID statistics for loaded datasets\n")
    os.makedirs("./datasets_stats", exist_ok=True)
    for ds_name in selected_ds_names:
        stat_file_path = Path(
            "./datasets_stats",
            ds_name + "_feat_dim_" + str(Inception_feat_dim) + ".npz",
        )
        if stat_file_path.exists() and force_recompute != "True":
            msg = f"File {stat_file_path} already exists. Skipping precomputation."
            logfile.write(msg + "\n")
            print(msg)
            continue
        ds_path = Path(root_data_dir, ds_name)
        save_fid_stats(
            (ds_path.as_posix(), stat_file_path.as_posix()),
            fid_batch_size,
            device,
            Inception_feat_dim,
            num_workers,
        )


def load_full_dataset_from_HF(dataset_id: str, logfile):
    header_print("Loading dataset " + dataset_id + " from Hugging Face Datasets")
    logfile.write(f"dataset_id: {dataset_id}\n")
    ds_global: DatasetDict = load_dataset(dataset_id, "full", cache_dir="./data")
    train_dataset = ds_global["train"]
    test_dataset = ds_global["test"]
    validation_dataset = ds_global["validation"]
    # concatenate all data
    full_dataset = torch.utils.data.ConcatDataset(
        [train_dataset, test_dataset, validation_dataset]
    )
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
    logfile.write(f"batch_size: {batch_size}\n")
    nb_batches = ceil(len(full_dataset) / batch_size)

    dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader, nb_batches
