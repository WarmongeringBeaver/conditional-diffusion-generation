"""
Conditional DDIM training script.
"""

import os
import time

import torch
from diffusers import DDIMScheduler
from diffusers.models import UNet2DModel
from torch import nn
from tqdm import tqdm
from tqdm.auto import trange
from utils import (
    compute_FIDs,
    generate_samples,
    header_print,
    my_warn,
    parse_wrapper,
    save_current_model,
    save_loss_plot,
)
from utils_datasets import (
    load_BBBC021_comp_conc_nice_phen,
    precompute_dataset_fid_statistics,
    preprocess_dataset,
)

# TODO's:
# - Adapt loss to unbalanced classes
# - use HF Accelerate to train on multiple GPUs
# - remove rogue prints about Qt (is it matplotlib?) and rogue tqdm bar (pytorch_fid)
#       note: prints appear juste before rogue tqdm bar, maybe both are caused by pytorch_fid?
#       -> requires PR on GH... => shamelessly steal & modify code in local? :)
# - profile!
# - save performance statistics (time, memory, etc.) along training
# - implement other metrics: -> pytorch-fidelity
#   - IS
#   - Perceptual Distance
#   - ..?
#   - Classification accuracy if Anis has a pretrained classifier?
# - try velocity model
# - use mixed precision
# - try gradient accumulation if memory bound
# - use learning rate scheduler (at least pass lr as arg)
# - allow to choose compile strategy in args (max-autotune, etc.)
# - use proper logging
# - use a proper VCS at some point...
# - log torch summary for better visualization
# - use tensorboard or W&B for better viz


########################################################################################
######################################## Startup #######################################
########################################################################################
## Run ID (= UTC timestamp)
timestamp = time.gmtime()
timestamp_str: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", timestamp)

## Args
args: dict = parse_wrapper(timestamp_str)

## Folder structure
# experiments
#   |- run_<timestamp>
#       |- checkpoints
#       |- outputs
this_experiment_folder = (
    f"experiments/run_{timestamp_str}"
    + ("_" * (args["run_desc"] != ""))
    + args["run_desc"].replace(" ", "_")
)
header_print("Saving experiment results to folder " + this_experiment_folder)
os.makedirs(this_experiment_folder + "/checkpoints")
os.makedirs(this_experiment_folder + "/outputs")

## Logfile
logfilename = f"log_{timestamp_str}.txt"

# check if file exists
logfile_path = this_experiment_folder + "/" + logfilename
if os.path.isfile(logfile_path):
    err_msg = f"Log file {logfilename} already exists. Please specify a different log file name."
    raise RuntimeError(err_msg)
logfile = open(logfile_path, "x", buffering=1)
# buffering=1: flush every new line; allows logging in real time
logfile.write(f"timestamp: {timestamp_str}\n")
logfile.write(f"root_data_dir: {args['root_data_dir']}\n")
logfile.write(f"selected_datasets: {args['selected_datasets']}\n")

## Check CUDA availability
if not torch.cuda.is_available():
    err = "Pytorch reports no GPU available!"
    logfile.write(err + "\n")
    raise RuntimeError(err)
device = "cuda"
logfile.write(f"device: {device}\n")

## Misc CUDA optimizations
torch.set_float32_matmul_precision("high")

## Dataset
# full_dataset = load_full_dataset_from_HF(
#     "keremberke/blood-cell-object-detection", logfile
# )
full_dataset = load_BBBC021_comp_conc_nice_phen(
    args["root_data_dir"], args["selected_datasets"]
)
dataloader, nb_batches = preprocess_dataset(full_dataset, args, logfile)

## Compute FID stats
precompute_dataset_fid_statistics(
    args["root_data_dir"],
    args["selected_datasets"],
    args["Inception_feat_dim"],
    logfile,
    force_recompute=args["FID_recompute"],
)

########################################################################################
######################################### Model ########################################
########################################################################################
## Scheduler
# choices for `DDIMScheduler` are `linear`, `squaredcos_cap_v2` and `scaled_linear`.
num_train_timesteps = args["num_train_timesteps"]
beta_start = args["beta_start"]
beta_end = args["beta_end"]
beta_schedule = args["beta_schedule"]
# TODO: understand why cosine schedule is called that way...

noise_scheduler = DDIMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_start=beta_start,
    beta_end=beta_end,
    beta_schedule=beta_schedule,
)
logfile.write(f"noise_scheduler: {noise_scheduler}\n")
logfile.write(f"num_inference_steps: {args['num_inference_steps']}\n")

## Model
# UNet-like architecture with 4 down and upsampling blocks with self-attention down the U.
num_classes = len(args["selected_datasets"])
model = UNet2DModel(
    sample_size=args["image_size"],  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
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
        "UpBlock2D",  # a retorch_dtype=torch.float16gular ResNet upsampling block
    ),
    class_embed_type=None,  # None = nn.Embedding (...)
    num_class_embeds=len(args["selected_datasets"]),
)
model.to(device)

# the model is compiled (will only work with `torch>=2.0.0`); this will take quite some time at first pass
# (`args["compile"] == "True"` is ugly but works with config file)
if hasattr(torch, "compile") and args["compile"] == "True":
    tqdm.write("\nModel will be compiled! First call will take a while...")
    model = torch.compile(model, mode="reduce-overhead")
    # reduce-overhead for quick compile
else:
    msg = "Skipping compilation, either because torch.compile is not available"
    msg += " or because --compile='False' was specified."
    my_warn(msg)
    logfile.write(msg + "\n")
logfile.write(f"model: {model}\n")

## Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
logfile.write(f"optimizer: {optimizer}\n")

## Loss
loss_func = nn.MSELoss()
logfile.write(f"loss_func: {loss_func}\n")

########################################################################################
######################################## Learning ######################################
########################################################################################
nb_epochs = args["nb_epochs"]
logfile.write(f"nb_epochs: {nb_epochs}\n")

# keep track of losses and FIDs along training
losses_per_epoch: list[float] = []
FIDs_per_epoch: list[dict[str, float]] = []

header_print("Learning...\n")
iterator = trange(nb_epochs, desc="Epoch")
for epoch in iterator:
    losses_per_epoch.append(0)  # init loss value for this new epoch

    for step, batch in enumerate(dataloader):
        iterator.set_postfix_str(f"Gradient descent: batch {step+1}/{nb_batches}")
        clean_images = batch["images"].to(device)
        classes = batch["classes"].to(device)

        # Sample Gaussian noise to add to the images
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]  # actual batch size

        # Sample a random timestep for each image
        # (thus "double" stochasticity)
        timesteps = torch.randint(
            0, num_train_timesteps, (bs,), device=clean_images.device, dtype=torch.int32
        ).to(device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps,
        )

        # Get the model prediction (not the noisy image, but the noise itself)
        noise_pred = model(noisy_images, timesteps, classes).sample

        # Compute the loss
        loss = loss_func(noise_pred, noise)

        # Compute gradients
        loss.backward(loss)

        # Save loss
        loss_value = float(loss.item())
        # running mean of batch losses for this epoch
        losses_per_epoch[-1] += (loss_value - losses_per_epoch[-1]) / (step + 1)

        # Update model parameters
        optimizer.step()
        optimizer.zero_grad()

    # print loss
    iterator.set_postfix_str(f"Saving loss plot...")
    save_loss_plot(losses_per_epoch, loss_func, this_experiment_folder)

    # save current model
    if epoch % args["save_every"] == 0 and (epoch != 0):
        iterator.set_postfix_str(f"Saving model...")
        save_current_model(
            model, noise_scheduler, this_experiment_folder, epoch, logfile
        )

    # generate samples & compute FID
    if epoch % args["generate_every"] == 0:
        # generate samples and write them to disk
        model.eval()
        generate_samples(
            model,
            noise_scheduler,
            device,
            args,
            this_experiment_folder,
            epoch,
            iterator,
            logfile,
        )
        # compute FID
        iterator.set_postfix_str(f"Computing FID...")
        compute_FIDs(args, this_experiment_folder, epoch, device, FIDs_per_epoch)
        model.train()


## Save final pipeline if needed
if (nb_epochs - 1) % args["save_every"] != 0:
    print(f"Saving final model...")
    save_current_model(
        model, noise_scheduler, this_experiment_folder, nb_epochs - 1, logfile
    )


## Close logfile
logfile.close()

header_print("Script terminated successfully!")
