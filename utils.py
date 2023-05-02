"""Utilities for unconditional_diffusion_generation.{ipynb,py}

`_manual_image_gen` and `_numpy_to_pil` were shamelessly stolen from HF.
"""

import ast
import os
import pickle
import sys
from argparse import ArgumentParser
from io import TextIOBase
from math import ceil
from pathlib import Path
from warnings import warn

import numpy as np
import tomli
import torch
import torchvision
from diffusers import DDIMPipeline, DDIMScheduler
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths

# TODO's:
# - take parameters from config (& log them) for compute_FIDs
# - remove CL parsing and simply directly give args as in-code python dict


@torch.no_grad()
def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


@torch.no_grad()
def save_grid_images(images, size, filename) -> None:
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    # 10 is the spacing between images
    width = size * len(images) + (len(images) - 1) * 10
    output_im = Image.new("RGB", (width, size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size + i * 10, 0))
    output_im.save(filename)


def parse_wrapper(timestamp_str: str) -> dict:
    """Parses args from `config.toml`, or falls back to command line."""
    # Define parser
    desc = "Experiments results are saved in `experiments/run_<timestamp>."
    desc += " Reads arguments from `config.toml` if present, else from command line."
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "--run_desc",
        type=str,
        default="",
        help="Will be added to the run folder name: keep it short!",
    )
    parser.add_argument("--root_data_dir", type=str, required=True)
    parser.add_argument("--selected_datasets", type=ast.literal_eval, required=True)
    parser.add_argument(
        "--save_every", type=int, default=100, help="Save every SAVE_EVERY epochs"
    )
    parser.add_argument(
        "--generate_every",
        type=int,
        default=10,
        help="Generate & compute FID every GENERATE_EVERY epochs",
    )
    hlp_msg = "Number of generated images every GENERATE_EVERY epoch. "
    hlp_msg += "Important for the FID computation reliability. Authors recommend at least 10,000! "
    hlp_msg += "See https://github.com/bioinf-jku/TTUR for original TF implementation."
    parser.add_argument(
        "--nb_generated_images",
        help="Number of generated images every GENERATE_EVERY epoch. Important for the FID computation reliability!",
        type=int,
        required=True,
    )
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_train_timesteps", type=int, required=True)
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--beta_start", type=float, required=True)
    parser.add_argument("--beta_end", type=float, required=True)
    parser.add_argument(
        "--beta_schedule",
        type=str,
        help="One of `linear`, `squaredcos_cap_v2` or `scaled_linear`",
        required=True,
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        help="Number of epochs to train for",
        required=True,
    )
    parser.add_argument(
        "--compile",
        help="Compile the model?",
        type=str,
        choices=["True", "False"],
        default="False",
    )
    parser.add_argument("--Inception_feat_dim", type=int, required=True)
    parser.add_argument(
        "--FID_recompute",
        help="Recompute FID statistics?",
        type=str,
        choices=["True", "False"],
        default="False",
    )
    parser.add_argument(
        "--class_emb_size", type=int, required=True, help="Size of the class embedding"
    )
    ## Catch --help flag from CL before reading file
    if len(sys.argv) >= 2 and sys.argv[1] in ["--help", "-h"]:
        parser.parse_args(["--help"])
    # Try to read from file
    configfile_path = "config.toml"
    if os.path.isfile(configfile_path):
        header_print("Reading config from config.toml...")
        with open(configfile_path, "rb") as f:
            config: dict = tomli.load(f)
            config_list = []
            for key, value in config.items():
                config_list += [f"--{key}", str(value)]
            args: dict = vars(parser.parse_args(config_list))
    # If no config file, read from command line
    else:
        my_warn("No config file found (`config.toml`), taking command-line arguments")
        args: dict = vars(parser.parse_args())
    return args


@torch.no_grad()
def save_loss_plot(
    losses_per_epoch: "list[float]",
    loss_func,
    this_experiment_folder: str,
    window_width: int = 20,
) -> None:
    plt.style.use("ggplot")
    save_folder = this_experiment_folder + "/outputs"
    current_nb_epochs = len(losses_per_epoch)
    # raw
    plt.plot(
        list(range(1, current_nb_epochs + 1)),
        losses_per_epoch,
        label="instantaneous",
    )
    # moving average
    cumsum_vec = np.cumsum(np.insert(losses_per_epoch, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(
        list(range(window_width, current_nb_epochs + 1)),
        ma_vec,
        label=f"running average",
    )
    plt.xlabel("iteration")
    plt.ylabel(f"average {loss_func._get_name()}")
    plt.legend()
    # write image to file
    save_folder = this_experiment_folder + "/outputs"
    plt.savefig(save_folder + "/loss_plot.png", bbox_inches="tight", dpi=300)
    # write logscale image to file
    plt.yscale("log")
    plt.savefig(save_folder + "/loss_log_plot.png", bbox_inches="tight", dpi=300)
    plt.close()
    # save list of loss values
    np.save(save_folder + "/losses.npy", losses_per_epoch)


@torch.no_grad()
def save_current_model(
    model,
    noise_scheduler: DDIMScheduler,
    this_experiment_folder: str,
    epoch: int,
    logfile: TextIOBase,
    save_method: str = "torch",
) -> None:
    save_folder = Path(this_experiment_folder, "checkpoints", f"epoch_{epoch}")
    # some trouble with onnx...
    match save_method:
        case "torch":
            os.makedirs(save_folder)
            torch.save(model.state_dict(), Path(save_folder, "unet.pt"))
            logfile.write(f"Saved model at epoch {epoch} in {save_folder}\n")
        case "hugging_face":
            image_pipe = DDIMPipeline(unet=model, scheduler=noise_scheduler)
            image_pipe.save_pretrained(save_folder)
            logfile.write(f"Saved pipeline at epoch {epoch} in {save_folder}\n")
        case _:
            raise ValueError(f"Unknown save method {save_method}")


@torch.no_grad()
def generate_samples(
    model,
    noise_scheduler: DDIMScheduler,
    device: str,
    args: dict,
    this_experiment_folder: str,
    epoch: int,
    iterator,
    logfile: TextIOBase,
):
    """Conditional generation."""
    # copy training scheduler to ensure no leakage to the training config
    scheduler: DDIMScheduler = DDIMScheduler.from_config(noise_scheduler.config)
    # set timestep to the number of *inference* steps
    scheduler.set_timesteps(args["num_inference_steps"])

    # generate samples iterating over datasets
    for ds_nb, ds_name in enumerate(args["selected_datasets"]):
        # the class is this dataset idx
        classes = torch.tensor([ds_nb] * args["batch_size"])
        postfix_str = f"Generating samples: dataset {ds_nb+1}/{len(args['selected_datasets'])} | batch "
        # save generated samples to a epoch/dataset–specific folder
        save_folder = Path(
            this_experiment_folder,
            "outputs",
            f"epoch_{epoch}_generated_images",
            ds_name,
        )
        os.makedirs(save_folder)
        # batch generation to avoid (D)OOM
        nb_inference_batches = ceil(args["nb_generated_images"] / args["batch_size"])
        # nb_samples_to_generate = [<batch_size>, ..., <batch_size>, <what's left>]
        nb_samples_to_generate = [args["batch_size"]] * (nb_inference_batches - 1)
        nb_samples_to_generate += [args["nb_generated_images"] % args["batch_size"]]
        for b in range(nb_inference_batches):
            postfix_str = postfix_str[: postfix_str.find("batch") + 6]
            postfix_str += f"{b+1}/{nb_inference_batches}"
            iterator.set_postfix_str(postfix_str)
            # generation (cannot use DDIM Pipeline as it is unconditional!)
            actual_batch_size = nb_samples_to_generate[b]
            images = _manual_image_gen(
                actual_batch_size,
                args["image_size"],
                device,
                model,
                scheduler,
                classes[:actual_batch_size],
            )
            for idx, img in enumerate(images):
                tot_idx = b * args["batch_size"] + idx
                img.save(Path(save_folder, f"{tot_idx}.png"))
        logfile.write(
            f"Generated {args['nb_generated_images']} samples at epoch {epoch} in {save_folder}\n"
        )


@torch.no_grad()
def _manual_image_gen(
    actual_batch_size,
    image_size,
    device,
    model,
    scheduler: DDIMScheduler,
    classes,
) -> list:
    image = torch.randn(
        (
            actual_batch_size,
            3,
            image_size,
            image_size,
        ),
        device=device,
        dtype=model.dtype,
    )
    classes = classes.to(device)
    for t in range(len(scheduler.timesteps)):
        # 1. predict noise model_output
        model_output = model(image, t, classes).sample

        # 2. predict previous mean of image x_t-1 and add variance depending on eta
        # eta corresponds to sigma in paper and should be between [0, 1]
        # DDIM ⟺ sigma (eta here) = 0 (the default)
        # do x_t -> x_t-1
        image = scheduler.step(
            model_output,
            t,
            image,
        ).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = _numpy_to_pil(image)
    return image


def _numpy_to_pil(images: np.ndarray) -> list:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


@torch.no_grad()
def compute_FIDs(
    args: dict,
    this_experiment_folder: str,
    epoch: int,
    device: str,
    FIDs_per_epoch: list[dict[str, float]],
) -> None:
    """Computes and saves FID scores & plots for each dataset.

    From the Authors (https://github.com/bioinf-jku/TTUR#fr%C3%A9chet-inception-distance-fid):
        IMPORTANT: The number of samples to calculate the Gaussian statistics
        (mean and covariance) should be greater than the dimension of the coding layer.
    Also:
        We recommend using a minimum sample size of 10,000 [!!] to calculate the FID
        otherwise the true FID of the generator is underestimated.
    """
    FIDs_per_epoch.append({})
    # compute FID for each dataset
    for ds_name in args["selected_datasets"]:
        precomputed_stats_path = Path(
            "datasets_stats",
            ds_name + "_feat_dim_" + str(args["Inception_feat_dim"]) + ".npz",
        )
        generated_ds_path = Path(
            this_experiment_folder,
            "outputs",
            f"epoch_{epoch}_generated_images",
            ds_name,
        )
        fid_value = calculate_fid_given_paths(
            (generated_ds_path.as_posix(), precomputed_stats_path.as_posix()),
            batch_size=args["batch_size"],
            device=device,
            dims=args["Inception_feat_dim"],
            num_workers=8,
        )
        FIDs_per_epoch[-1][ds_name] = fid_value
    # pickle FID score & plot
    _save_FID_score(FIDs_per_epoch, this_experiment_folder, args["generate_every"])


def _save_FID_score(
    FIDs_per_epoch: list[dict[str, float]],
    this_experiment_folder: str,
    generate_every: int,
    window_width: int = 10,
) -> None:
    """Saves the FID score & plot between generated datasets and precomputed statistics."""
    # refresh live plot of FID score
    plt.style.use("ggplot")
    save_folder = this_experiment_folder + "/outputs"
    current_nb_epochs = len(FIDs_per_epoch) * generate_every
    # load past FID values & update plot
    for ds_name in FIDs_per_epoch[0].keys():
        FIDs_for_this_ds = [
            FIDs_per_epoch[t][ds_name] for t in range(len(FIDs_per_epoch))
        ]
        plt.plot(
            list(range(1, current_nb_epochs + 1, generate_every)),
            FIDs_for_this_ds,
            label=ds_name,
        )
        # # add moving average -> BROKEN
        # if len(FIDs_for_this_ds) >= window_width:
        #     cumsum_vec = np.cumsum(np.insert(FIDs_for_this_ds, 0, 0))
        #     ma_vec = (
        #         cumsum_vec[window_width:] - cumsum_vec[:-window_width]
        #     ) / window_width
        #     plt.plot(
        #         list(range(window_width, current_nb_epochs + 1, generate_every)),
        #         ma_vec,
        #         label=f"{ds_name} (running average)",
        #     )
    plt.xlabel("iteration")
    plt.ylabel("FID score")
    plt.legend()
    # write image to file
    plt.savefig(save_folder + "/FID_plot.png", bbox_inches="tight", dpi=300)
    plt.close()
    # save list of FID values
    with open(save_folder + "/FID_values.pickle", "wb") as handle:
        pickle.dump(FIDs_per_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)


def my_warn(msg: str) -> None:
    warn(
        f"\n  \033[93m{msg}\033[0m",
        stacklevel=3,
    )


def header_print(msg: str) -> None:
    print(f"\033[1m\n====> {msg}\033[0m")
