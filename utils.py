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
import torch_fidelity
import torchvision
from diffusers import DDIMPipeline, DDIMScheduler
from matplotlib import pyplot as plt
from PIL import Image

# TODO's:
# - take parameters from config (& log them) for compute_metrics
# - directly give args as in-code python dict?


########################################################################################
################################# Visualization helpers ################################
########################################################################################
def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def save_grid_images(images, size, filename) -> None:
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    # 10 is the spacing between images
    width = size * len(images) + (len(images) - 1) * 10
    output_im = Image.new("RGB", (width, size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size + i * 10, 0))
    output_im.save(filename)


########################################################################################
################################### Arguments parsing ##################################
########################################################################################
def define_config_file_parser() -> ArgumentParser:
    desc = "Experiments results are saved in `experiments/run_<timestamp>."
    desc += " Reads the following arguments from the passed TOML config file path (and NOT from CL; TODO: clean this)."
    configfile_parser = ArgumentParser(description=desc)
    configfile_parser.add_argument(
        "--run_desc",
        type=str,
        default="",
        help="Will be added to the run folder name: keep it short!",
    )
    configfile_parser.add_argument("--root_data_dir", type=str, required=True)
    configfile_parser.add_argument(
        "--selected_datasets", type=ast.literal_eval, required=True
    )
    configfile_parser.add_argument(
        "--save_every", type=int, default=100, help="Save every SAVE_EVERY epochs"
    )
    configfile_parser.add_argument(
        "--generate_every",
        type=int,
        default=10,
        help="Generate & compute metrics every GENERATE_EVERY epochs",
    )
    hlp_msg = "Number of generated images every GENERATE_EVERY epoch. "
    hlp_msg += "Important for the FID computation reliability. Authors recommend at least 10,000! "
    hlp_msg += "See https://github.com/bioinf-jku/TTUR for original TF implementation."
    configfile_parser.add_argument(
        "--nb_generated_images",
        help=hlp_msg,
        type=int,
        required=True,
    )
    configfile_parser.add_argument("--image_size", type=int, required=True)
    configfile_parser.add_argument("--batch_size", type=int, required=True)
    configfile_parser.add_argument("--num_train_timesteps", type=int, required=True)
    configfile_parser.add_argument("--num_inference_steps", type=int, required=True)
    configfile_parser.add_argument("--beta_start", type=float, required=True)
    configfile_parser.add_argument("--beta_end", type=float, required=True)
    configfile_parser.add_argument(
        "--beta_schedule",
        type=str,
        help="One of `linear`, `squaredcos_cap_v2` or `scaled_linear`",
        required=True,
    )
    configfile_parser.add_argument(
        "--nb_epochs",
        type=int,
        help="Number of epochs to train for",
        required=True,
    )
    configfile_parser.add_argument(
        "--compile",
        help="Compile the model?",
        type=str,
        choices=["True", "False"],
        default="False",
    )
    return configfile_parser


def parse_wrapper() -> dict:
    """Parses args from a configuration file path passed through CL."""
    # Define parser
    configfile_parser = define_config_file_parser()
    # Catch --help flag from CL before reading file
    if len(sys.argv) >= 2 and sys.argv[1] in ["--help", "-h"]:
        configfile_parser.parse_args(["--help"])
    # Get config file path from CL
    cl_parser = ArgumentParser()
    cl_parser.add_argument("configfile_path", type=str)
    configfile_path = cl_parser.parse_args().configfile_path
    # Read config file
    header_print(f"Reading config from {configfile_path}...")
    with open(configfile_path, "rb") as f:
        config: dict = tomli.load(f)
        config_list = []
        for key, value in config.items():
            config_list += [f"--{key}", str(value)]
        args: dict = vars(configfile_parser.parse_args(config_list))
    return args


########################################################################################
###################################### Generation ######################################
########################################################################################
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
) -> list:
    image = torch.randn(
        (
            actual_batch_size,
            3,
            image_size,
            image_size,
        ),
        device=device,
        dtype=model.dtype if hasattr(model, "dtype") else None,
    )
    for t in range(len(scheduler.timesteps)):
        # 1. predict noise model_output
        model_output = model(image, t).sample

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


########################################################################################
################################ Model & Metrics saving ################################
########################################################################################
def save_loss_plot(
    losses_per_epoch: list[float],
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
    # plt.yscale("log")
    # plt.savefig(save_folder + "/loss_log_plot.png", bbox_inches="tight", dpi=300)
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
def compute_metrics(
    args: dict,
    this_experiment_folder: str,
    epoch: int,
    metrics_per_epoch: list[dict],
) -> None:
    """Computes and saves metrics & plots for each dataset.

    Nota: from the Authors of FID:
        IMPORTANT: The number of samples to calculate the Gaussian statistics
        (mean and covariance) should be greater than the dimension of the coding layer.
    Also:
        We recommend using a minimum sample size of 10,000 [!!] to calculate the FID
        otherwise the true FID of the generator is underestimated.
    See https://github.com/bioinf-jku/TTUR#fr%C3%A9chet-inception-distance-fid.
    """
    metrics_per_epoch.append({})
    for ds_name in args["selected_datasets"]:
        generated_images_folder = Path(
            this_experiment_folder,
            "outputs",
            f"epoch_{epoch}_generated_images",
            ds_name,
        )
        dataset_folder = Path(args["root_data_dir"], ds_name)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=dataset_folder.as_posix(),
            input2=generated_images_folder.as_posix(),
            cuda=True,
            batch_size=args["batch_size"],
            isc=True,
            fid=True,
            # kid=True,
            # ppl=True,
            verbose=False,
            cache_root=".fidelity_cache",
            input1_cache_name=f"{ds_name}",  # forces caching
            rng_seed=42,
        )
        metrics_per_epoch[-1][ds_name] = metrics_dict
    # pickle FID score & plot
    _save_and_plot_metrics(
        metrics_per_epoch, this_experiment_folder, args["generate_every"]
    )


def _save_and_plot_metrics(
    metrics_per_epoch: list[dict],
    this_experiment_folder: str,
    generate_every: int,
    # window_width: int = 10,
) -> None:
    """Saves the metrics dict & plot the metrics.
    
    Arguments
    ---------
    - metrics_per_epoch: list of dicts\\
        Each dict containing the metrics for each dataset; structure:
        
        ```
        metrics_per_epoch[epoch][dataset_name][metric_name] = metric_value
        ```
        
        where `metric_name` comes from `torch_fidelity`.
    """
    # refresh live plot for each metric score
    plt.style.use("ggplot")
    save_folder = this_experiment_folder + "/outputs"
    current_nb_epochs = len(metrics_per_epoch) * generate_every
    # load past metrics values & update plot for each metric
    computed_metric_names = next(iter(metrics_per_epoch[0].items()))[1].keys()
    # one plot per metric
    for metric_name in computed_metric_names:
        # all datasets are compared on the same plot
        for ds_name in metrics_per_epoch[0].keys():
            this_metric_for_this_ds = [
                metrics_per_epoch[t][ds_name][metric_name]
                for t in range(len(metrics_per_epoch))
            ]
            plt.plot(
                list(range(1, current_nb_epochs + 1, generate_every)),
                this_metric_for_this_ds,
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
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(f"{metric_name} for each dataset over epochs")
        # write image to file
        plt.savefig(
            save_folder + f"/{metric_name}_plot.png", bbox_inches="tight", dpi=400
        )
        plt.close()
    # save metrics_per_epoch
    with open(save_folder + "/metrics_per_epoch.pickle", "wb") as handle:
        pickle.dump(metrics_per_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)


########################################################################################
##################################### Miscellaneous ####################################
########################################################################################
def my_warn(msg: str) -> None:
    warn(
        f"\n  \033[93m{msg}\033[0m",
        stacklevel=3,
    )


def header_print(msg: str) -> None:
    print(f"\033[1m\n====> {msg}\033[0m")
