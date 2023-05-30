import inspect
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

import wandb
from args_parser import parse_args
from unet_2d import UNet2DModel
from utils_dataset import setup_dataset
from utils_misc import (
    args_checker,
    create_repo_structure,
    setup_logger,
    setup_xformers_memory_efficient_attention,
)
from utils_training import (
    checkpoint_model,
    generate_samples_and_compute_metrics,
    get_training_setup,
    perform_training_epoch,
    resume_from_checkpoint,
)

logger = get_logger(__name__, log_level="INFO")


def main(args):
    # ------------------------- Checks -------------------------
    args_checker(args)

    # ----------------------- Accelerator ----------------------
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=args.output_dir,
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # -------------------------- WandB -------------------------
    wandb_project_name = args.output_dir.lstrip("experiments/")
    logger.info(f"Logging to project {wandb_project_name}")
    accelerator.init_trackers(
        project_name=wandb_project_name,
        config=args,
    )

    # Make one log on every process with the configuration for debugging.
    setup_logger(logger, accelerator)

    # ------------------- Repository scruture ------------------
    (
        image_generation_tmp_save_folder,
        full_pipeline_save_folder,
        repo,
    ) = create_repo_structure(args, accelerator)

    # -------------------------- Model -------------------------
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
    else:
        ema_model = None

    if args.enable_xformers_memory_efficient_attention:
        setup_xformers_memory_efficient_attention(model, logger)

    # track gradients
    if accelerator.is_main_process:
        wandb.watch(model)

    # --------------------- Noise scheduler --------------------
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDIMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.ddim_num_steps,
            beta_schedule=args.ddim_beta_schedule,
            prediction_type=args.prediction_type,
            beta_start=args.ddim_beta_start,
            beta_end=args.ddim_beta_end,
        )
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.ddim_num_steps,
            beta_schedule=args.ddim_beta_schedule,
        )

    # ------------------------ Optimizer -----------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------- Dataset ------------------------
    dataset = setup_dataset(args, logger)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ----------------- Learning rate scheduler -----------------
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # ------------------ Distributed compute  ------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # --------------------- Training setup ---------------------
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    first_epoch = 0
    global_step = 0
    resume_step = 0

    (
        num_update_steps_per_epoch,
        tot_nb_eval_batches,
        actual_eval_batch_sizes_for_this_process,
    ) = get_training_setup(args, accelerator, train_dataloader, logger, dataset)

    # ----------------- Resume from checkpoint -----------------
    if args.resume_from_checkpoint:
        first_epoch, resume_step = resume_from_checkpoint(
            args, logger, accelerator, num_update_steps_per_epoch
        )

    # ---------------------- Seeds & RNGs ----------------------
    rng = np.random.default_rng()  # TODO: seed this

    # ---------------------- Training loop ---------------------
    for epoch in range(first_epoch, args.num_epochs):
        # Training epoch
        global_step = perform_training_epoch(
            model,
            num_update_steps_per_epoch,
            accelerator,
            epoch,
            train_dataloader,
            args,
            first_epoch,
            resume_step,
            noise_scheduler,
            rng,
            global_step,
            optimizer,
            lr_scheduler,
            ema_model,
            logger,
        )

        # Generate sample images for visual inspection & metrics computation
        if (
            epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1
        ) and epoch > 0:
            generate_samples_and_compute_metrics(
                tot_nb_eval_batches,
                args,
                accelerator,
                model,
                ema_model,
                noise_scheduler,
                image_generation_tmp_save_folder,
                dataset,
                actual_eval_batch_sizes_for_this_process,
                args.guidance_factor,
                epoch,
                global_step,
            )

        if (
            accelerator.is_main_process
            and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1)
            and epoch != 0
        ):
            checkpoint_model(
                accelerator,
                model,
                args,
                ema_model,
                noise_scheduler,
                full_pipeline_save_folder,
                repo,
                epoch,
            )

        # do not start new epoch before generation & checkpointing is done
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
