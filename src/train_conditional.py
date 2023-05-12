"""Conditional DDIM generation."""

import argparse
import inspect
import logging
import os
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import Optional

import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch_fidelity
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from conditional_ddim import ConditionialDDIMPipeline
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

import diffusers
from diffusers import DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version  # check_min_version,
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDIM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--use_pytorch_loader",
        default=False,
        action="store_true",
        help="Whether to use the PyTorch ImageFolder loader instead of the HF Dataset loader. Usefull for folder symlinks...",
    )
    parser.add_argument(
        "--nb_classes",
        type=int,
        required=True,
        help="The number of classes in the dataset.",
    )  # TODO: infer from the dataset!
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written. Will be used as the WandB project name!",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=100,
        help="How often to save images during training.",
    )
    parser.add_argument(
        "--nb_generated_images",
        type=int,
        default=1000,
        help="How many images to generate. Only the first batch will be logged and images not logged will be lost.",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=100,
        help="How often to save the model during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo",
        action="store_true",
        help="Whether or not to create a private repository.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    # parser.add_argument(
    #     "--logging_dir",
    #     type=str,
    #     default="logs",
    #     help=(
    #         "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
    #         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    #     ),
    # )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddim_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddim_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    return args


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    ### Checks
    assert args.use_pytorch_loader, "Only PyTorch loader is supported for now."
    assert args.logger == "wandb", "Only W&B logger is supported for now."

    # logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=args.output_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training."
            )

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

        # Initialise your wandb run, passing wandb parameters and any config information
        accelerator.init_trackers(
            project_name=args.output_dir,
            config=args,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for model in models:
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token
            )

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Create a temporary folder to save the generated images during training.
        # Used for metrics computations; a small number of these (eval_batch_size) is logged
        image_generation_tmp_save_folder = Path(".tmp_image_generation_folder")

        # Create a folder to save the *full* pipeline
        full_pipeline_save_folder = Path(args.output_dir, "full_pipeline_save")
        os.makedirs(full_pipeline_save_folder, exist_ok=True)

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type=None,  # = nn.Embedding...
            num_class_embeds=args.nb_classes,
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDIMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.ddim_num_steps,
            beta_schedule=args.ddim_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.ddim_num_steps,
            beta_schedule=args.ddim_beta_schedule,
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

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
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        class_labels = examples["label"]
        # TODO: change this to the actual class labels
        return {"images": images, "class_labels": class_labels}

    logger.info(f"Dataset size: {len(dataset)}")

    if not args.use_pytorch_loader:
        dataset.set_transform(transform_images)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    nb_eval_batches = ceil(args.nb_generated_images / args.eval_batch_size)
    actual_eval_batch_sizes = [args.eval_batch_size] * (nb_eval_batches - 1)
    actual_eval_batch_sizes += [args.nb_generated_images % args.eval_batch_size]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            chckpnts_dir = Path(args.output_dir, "checkpoints")
            if not Path.exists(chckpnts_dir):
                logger.warning(
                    f"No 'checkpoints' directory found in output_dir {args.output_dir}; creating one."
                )
                os.makedirs(chckpnts_dir)
            dirs = os.listdir(chckpnts_dir)
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = Path(chckpnts_dir, dirs[-1]).as_posix() if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("_")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            if args.use_pytorch_loader:
                clean_images = batch[0]
                class_labels = batch[1]
            else:
                clean_images = batch["images"]
                class_labels = batch["class_labels"]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps, class_labels).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise)
                    # TODO: weight according to timesteps
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod,
                        timesteps,
                        (clean_images.shape[0], 1, 1, 1),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}"
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # time to save a checkpoint!
                    if accelerator.is_main_process:
                        save_path = Path(
                            args.output_dir, "checkpoints", f"checkpoint_{global_step}"
                        )
                        accelerator.save_state(save_path.as_posix())
                        logger.info(f"Checkpointed step {global_step} at {save_path}")
                        # Delete old checkpoints if needed
                        checkpoints_list = os.listdir(
                            Path(args.output_dir, "checkpoints")
                        )
                        nb_checkpoints = len(checkpoints_list)
                        if nb_checkpoints > args.checkpoints_total_limit:
                            to_del = sorted(
                                checkpoints_list, key=lambda x: int(x.split("_")[1])
                            )[: -args.checkpoints_total_limit]
                            if len(to_del) > 1:
                                logger.warning(
                                    "More than 1 checkpoint to delete? Previous delete must have failed..."
                                )
                            for dir in to_del:
                                rmtree(Path(args.output_dir, "checkpoints", dir))

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Generate sample images for visual inspection
            if (
                epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1
            ) and epoch > 0:
                progress_bar = tqdm(
                    total=nb_eval_batches * args.nb_classes,
                    desc="Generating images",
                )
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = ConditionialDDIMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )
                pipeline.set_progress_bar_config(disable=True)

                # set manual seed in order to observe the "same" images
                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                # -> generate args.nb_generated_images in batches *per class*
                for label_idx, label in enumerate(range(args.nb_classes)):
                    # clean image_generation_tmp_save_folder (it's per-class)
                    if os.path.exists(image_generation_tmp_save_folder):
                        rmtree(image_generation_tmp_save_folder)
                    os.makedirs(image_generation_tmp_save_folder, exist_ok=False)

                    # get class name
                    class_name = dataset.classes[label]

                    # pretty bar
                    postfix_str = (
                        f"Current class: {class_name} ({label_idx+1}/{args.nb_classes})"
                    )
                    progress_bar.set_postfix_str(postfix_str)

                    # loop over eval batches
                    for batch_idx in range(nb_eval_batches):
                        actual_bs = actual_eval_batch_sizes[batch_idx]
                        class_labels = torch.full(
                            (actual_bs,), label, device=pipeline.device
                        ).long()
                        images = pipeline(
                            class_labels,
                            generator=generator,
                            batch_size=actual_bs,
                            num_inference_steps=args.ddim_num_inference_steps,
                            output_type="numpy",
                        ).images

                        # save images to disk, writting over the previous ones
                        images_pil = pipeline.numpy_to_pil(images)
                        for idx, img in enumerate(images_pil):
                            tot_idx = args.eval_batch_size * batch_idx + idx
                            img.save(
                                Path(image_generation_tmp_save_folder, f"{tot_idx}.png")
                            )

                        # denormalize the images and save to logger if first batch
                        # (first batch only to prevent "logger overflow")
                        if batch_idx == 0:
                            images_processed = (images * 255).round().astype("uint8")

                            if args.logger == "tensorboard":
                                if is_accelerate_version(">=", "0.17.0.dev0"):
                                    tracker = accelerator.get_tracker(
                                        "tensorboard", unwrap=True
                                    )
                                else:
                                    tracker = accelerator.get_tracker("tensorboard")
                                tracker.add_images(
                                    "test_samples",
                                    images_processed.transpose(0, 3, 1, 2),
                                    epoch,
                                )
                            elif args.logger == "wandb":
                                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                                accelerator.get_tracker("wandb").log(
                                    {
                                        f"generated_samples/{class_name}": [
                                            wandb.Image(img) for img in images_processed
                                        ],
                                        "epoch": epoch,
                                    },
                                    step=global_step,
                                )
                        progress_bar.update()

                    # Compute metrics
                    metrics_dict = torch_fidelity.calculate_metrics(
                        input1=args.train_data_dir + "/train" + f"/{class_name}",
                        input2=image_generation_tmp_save_folder.as_posix(),
                        cuda=True,
                        batch_size=args.eval_batch_size,
                        # isc=True,
                        fid=True,
                        # kid=True,
                        # ppl=True,
                        verbose=False,
                        cache_root=".fidelity_cache",
                        input1_cache_name=f"{class_name}",  # forces caching
                        rng_seed=42,
                    )
                    for metric_name in metrics_dict.keys():
                        accelerator.get_tracker("wandb").log(
                            {
                                f"{metric_name}/{class_name}": metrics_dict[
                                    metric_name
                                ],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                if args.use_ema:
                    ema_model.restore(unet.parameters())

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = ConditionialDDIMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(full_pipeline_save_folder)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
