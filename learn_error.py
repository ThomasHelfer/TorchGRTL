import sys
import os
import argparse

import torch
import pandas as pd
import numpy as np
import time
import shutil
import yaml

import wandb
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import TensorDataset, DataLoader


from GeneralRelativity.Utils import (
    get_box_format,
    TensorDict,
    cut_ghosts,
    keys,
    keys_all,
)
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from pyinterpx.Interpolation import *
from GeneralRelativity.TensorAlgebra import compute_christoffel
from GeneralRelativity.Constraints import constraint_equations

from SuperResolution.models import SuperResolution3DNet
from SuperResolution.losses import (
    Hamiltonian_loss,
    Hamiltonian_and_momentum_loss,
    Hamiltonian_and_momentum_loss_boundary_condition,
)
from SuperResolution.utils import load_config, copy_config_file

time_stamp = int(time.time())


def main():
    default_job_id = "local_run"
    # Parse the arguments
    slurm_job_id = os.getenv("SLURM_JOB_ID", default_job_id)

    folder_name = f"Run{slurm_job_id}"

    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
        # sys.exit(1)  # Exit the program with a non-zero exit code to indicate an error

    else:
        print(f"Folder '{folder_name}' already exists.")

    # copy code into folder for reproducibility
    shutil.copy("learn_error.py", folder_name)
    print(f"Copied script to '{folder_name}'.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Executing the model on :", device)

    torch.manual_seed(3)
    np.random.seed(6)
    writer = SummaryWriter(f"{folder_name}")

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="Path to the configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize wandb
    wandb.init(project="TorchGRTL", config=config)
    # Log hyperparameters
    wandb.config.update(config)

    # Copy the configuration file to the tracking directory
    copy_config_file(args.config, folder_name)

    # Create an empty file with the wandb run name
    run_name = wandb.run.name
    run_name_file_path = os.path.join(folder_name, run_name)
    with open(run_name_file_path, "w") as f:
        pass

    # Access configuration variables
    ADAMsteps = config["ADAMsteps"]
    n_steps = config["n_steps"]
    res_level = config["res_level"]
    scaling_factor = config["scaling_factor"]
    factor = config["factor"]
    filenamesX = config["filenamesX"].format(res_level=res_level)
    restart = config["restart"]
    file_path = config["file_path"]
    lambda_fac = config["lambda_fac"]
    kernel_size = config["kernel_size"]
    padding = config["padding"]
    num_layers = config["num_layers"]
    nonlinearity = config["nonlinearity"]
    masking_percentage = config["masking_percentage"]
    mask_type = config["mask_type"]
    write_out_freq = config["write_out_freq"]
    downsample = config["downsample"]
    align_corners = config["align_corners"]

    print(f"lambda_fac {type(scaling_factor)}")

    num_varsX = 25
    dataX = get_box_format(filenamesX, num_varsX)
    # Cutting out extra values added for validation
    dataX = dataX[:, :, :, :, :25]

    plt.imshow(
        dataX[0, 8, :, :, 0], cmap="viridis"
    )  # 'viridis' is a colormap, you can choose others like 'plasma', 'inferno', etc.
    plt.colorbar()  # Add a colorbar to show the scale
    plt.title("2D Array Plot")
    plt.savefig("testarray.png")
    # Instantiate the model
    net = SuperResolution3DNet(
        factor,
        scaling_factor=scaling_factor,
        num_layers=num_layers,
        kernel_size=kernel_size,
        padding=padding,
        nonlinearity=nonlinearity,
        masking_percentage=masking_percentage,
        mask_type=mask_type,
        align_corners=align_corners,
    ).to(torch.double)

    # Create a random 3D low-resolution input tensor (batch size, channels, depth, height, width)
    input_tensor = torch.randn(
        1,
        dataX.shape[4],
        dataX.shape[1] // downsample,
        dataX.shape[2] // downsample,
        dataX.shape[3] // downsample,
    ).to(
        torch.double
    )  # Adjust dimensions as needed
    # Forward pass to obtain the high-resolution output
    output_tensor, _ = net(input_tensor)
    print("mean", torch.mean(output_tensor))

    # Check the shape of the output
    print("Input Shape:", dataX.shape)
    print("Output Shape:", output_tensor.shape)

    diff = (dataX.shape[-2] - output_tensor.shape[-1]) // 2
    print(f"diff {diff}")

    # global step counter
    counter = 0

    losses_train = []
    losses_val = []
    losses_val_interp = []
    steps_val = []

    optimizerBFGS = torch.optim.LBFGS(
        net.parameters(), lr=config["lr_bfgs"]
    )  # Use LBFGS sometimes, it really does do magic sometimes, though its a bit of a diva
    optimizerADAM = torch.optim.Adam(net.parameters(), lr=config["lr_adam"])

    # Define the ratio for the split (e.g., 80% train, 20% test)
    train_ratio = config["train_ratio"]
    test_ratio = 1 - train_ratio

    # Calculate the number of samples for each split
    num_samples = len(dataX)
    num_train = int(train_ratio * num_samples)
    num_test = num_samples - num_train

    # Permute data to put the channel as the second dimension (N, C, H, W, D)
    dataX = dataX.permute(0, 4, 1, 2, 3)

    # Create a dataset from tensors
    dataset = TensorDataset(dataX)

    # Split the dataset into training and testing datasets
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
    batch_size = config["batch_size"]

    # Create DataLoader for batching -- in case data gets larger
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if restart and os.path.exists(file_path):
        net.load_state_dict(torch.load(file_path))

    # oneoverdx = 64.0 / 16.0
    oneoverdx = (64.0 * 2**res_level) / 512.0 * float(factor) / float(downsample)
    print(f"dx {1.0/oneoverdx}")
    if config["loss"] == "Ham":
        my_loss = Hamiltonian_loss(oneoverdx)
    elif config["loss"] == "Ham_mom":
        my_loss = Hamiltonian_and_momentum_loss(oneoverdx)
    elif config["loss"] == "Ham_mom_boundary_simple":
        my_loss = Hamiltonian_and_momentum_loss_boundary_condition(oneoverdx)
    elif config["loss"] == "L1":
        my_loss = torch.nn.L1Loss()
        # raise Value error if downsample == factor
        if downsample != factor:
            raise ValueError("L1 loss not implemented for downsample == factor")

    net.train()
    net.to(device)
    net.to(torch.double)

    # For validation error
    L1Loss = torch.nn.L1Loss()
    ham_loss = Hamiltonian_and_momentum_loss(oneoverdx)

    print("training")
    pbar = trange(n_steps)
    for i in pbar:
        total_loss_train = 0
        for (y_batch,) in train_loader:
            net.train()
            batchcounter = 0
            # for X_batch, y_batch in train_loader:
            y_batch = y_batch.to(device)
            X_batch = y_batch[:, :, ::downsample, ::downsample, ::downsample].clone()
            y_batch = y_batch[
                :, :25, diff - 1 : -diff - 1, diff - 1 : -diff - 1, diff - 1 : -diff - 1
            ]
            batchcounter += 1

            # This is needed for LBFGS
            def closure():
                if torch.is_grad_enabled():
                    optimizerBFGS.zero_grad()
                y_pred, y_interp = net(X_batch)

                loss_train = my_loss(y_pred, y_batch)
                if loss_train.requires_grad:
                    loss_train.backward()
                return loss_train

            # doing some ADAM first to warm up, sometimes BFGS fuckes up if you start too early
            if counter < ADAMsteps:
                y_pred, y_interp = net(X_batch)

                loss_train = my_loss(y_pred, y_batch)
                optimizerADAM.zero_grad()
                loss_train.backward()
                optimizerADAM.step()
                # print(f'ADAM {batchcounter}')

            else:
                optimizerBFGS.step(closure)
                # print(f'BFGS {batchcounter}')

            loss_train = closure()
            total_loss_train += loss_train.item()

        # Calculate the average training loss
        average_loss_train = total_loss_train / len(train_loader)
        # Log the average training loss
        wandb.log({"loss/train": average_loss_train, "step": counter})
        # Log the average training loss
        writer.add_scalar("loss/train", average_loss_train, counter)
        losses_train.append(average_loss_train)
        if np.isnan(average_loss_train):
            print("we got nans")

        # Validation

        if counter % 1 == 0:
            with torch.no_grad():
                net.eval()
                total_loss_val = 0.0
                interp_val = 0.0
                L1Loss_val = 0.0
                L1Loss_val_interp = 0.0
                loss_hard_base = 0.0
                Ham_loss = 0.0
                Ham_loss_interp = 0.0
                for (y_val_batch,) in test_loader:
                    # for X_val_batch, y_val_batch in test_loader:
                    # Transfer batch to GPU
                    y_val_batch = y_val_batch.to(device)
                    X_val_batch = y_val_batch[
                        :, :, ::downsample, ::downsample, ::downsample
                    ].clone()
                    y_val_batch = y_val_batch[
                        :,
                        :25,
                        diff - 1 : -diff - 1,
                        diff - 1 : -diff - 1,
                        diff - 1 : -diff - 1,
                    ]
                    y_val_pred, y_val_interp = net(X_val_batch)
                    loss_val = my_loss(y_val_pred, y_val_batch)
                    total_loss_val += loss_val.item()
                    interp_val += my_loss(y_val_interp, y_val_batch).item()
                    if downsample == factor:
                        L1Loss_val += L1Loss(
                            y_val_pred[:, 0, :, :, :], y_val_batch[:, 0, :, :, :]
                        )
                        L1Loss_val_interp += L1Loss(
                            y_val_interp[:, 0, :, :, :], y_val_batch[:, 0, :, :, :]
                        )
                        loss_hard_base += ham_loss(y_val_batch, None)

                        if config["loss"] == "L1":
                            Ham_loss_interp += ham_loss(y_val_interp, None)
                            Ham_loss += ham_loss(y_val_pred, None)
                # Calculate the average loss
                average_loss_val = total_loss_val / len(test_loader)
                average_interp_val = interp_val / len(test_loader)
                losses_val_interp.append(average_interp_val)
                losses_val.append(average_loss_val)
                steps_val.append(counter)
                writer.add_scalar("loss/test", loss_val.item(), counter)
                wandb.log(
                    {
                        "loss/val": average_loss_val,
                        "loss/val_interp": average_interp_val,
                        "step": counter,
                    }
                )
                if downsample == factor:
                    wandb.log(
                        {
                            "loss/val_hard_baseline": loss_hard_base / len(test_loader),
                            "L1Loss/val_interp_baseline": L1Loss_val_interp
                            / len(test_loader),
                            "L1Loss/val": L1Loss_val / len(test_loader),
                            "step": counter,
                        }
                    )
                    if config["loss"] == "L1":
                        wandb.log(
                            {
                                "Hamloss/val": Ham_loss / len(test_loader),
                                "Hamloss/interp_val": Ham_loss_interp
                                / len(test_loader),
                                "step": counter,
                            }
                        )

        if counter % write_out_freq == 0:
            # Writing out network and scaler
            torch.save(
                net.state_dict(),
                f"{folder_name}/model_epoch_counter_{counter:010d}_data_time_{time_stamp}.pth",
            )
        # Advancing global counter
        counter += 1

    # Plotting shit at the end
    plt.figure(figsize=(9, 6))
    print(f"final val loss {losses_val[-1]} relative {losses_val_interp[-1]}")
    plt.plot(np.array(losses_train), label="Train")
    plt.plot(steps_val, np.array(losses_val), label="Val", linewidth=0.5)
    plt.plot(steps_val, np.array(losses_val_interp), label="baseline", linewidth=0.5)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{folder_name}/training.png")
    plt.close()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
