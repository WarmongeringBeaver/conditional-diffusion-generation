from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def setup_dataset(args, logger):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    elif args.use_pytorch_loader:
        dataset = ImageFolder(
            root=Path(args.train_data_dir, "train").as_posix(),
            transform=lambda x: augmentations(x.convert("RGB")),
            target_transform=lambda y: torch.tensor(y).long(),
        )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # map to [-1, 1] for SiLU
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        class_labels = examples["label"]
        return {"images": images, "class_labels": class_labels}

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of classes: {len(dataset.classes)}")

    if not args.use_pytorch_loader:
        dataset.set_transform(transform_images)

    return dataset, len(dataset.classes)
