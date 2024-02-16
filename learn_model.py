import torch
import pandas as pd
import numpy as np
import time
import sys
import os
import argparse
import shutil
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from GeneralRelativity.Utils import (
    get_box_format,
    TensorDict,
    cut_ghosts,
    keys,
    keys_all,
)
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2, FOR3, FOR4
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from pyinterpx.Interpolation import *
from GeneralRelativity.TensorAlgebra import (
    compute_christoffel,
    compute_trace,
    compute_christoffel_fast,
    raise_all,
)
from GeneralRelativity.CCZ4Geometry import compute_ricci
from GeneralRelativity.Constraints import constraint_equations

from torch.utils.data import TensorDataset, DataLoader

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
    shutil.copy("learn_model.py", folder_name)
    print(f"Copied script to '{folder_name}'.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Executing the model on :", device)

    torch.manual_seed(1)
    np.random.seed(4)
    writer = SummaryWriter(f"{folder_name}")

    # Loading small testdata
    filenamesX = (
        "/home/thelfer1/scr4_tedwar42/thelfer1/data_gen_binary/outputXdata_level1_*"
    )
    num_varsX = 100
    dataX = get_box_format(filenamesX, num_varsX)
    # Cutting out extra values added for validation
    dataX = dataX[:, :, :, :, :25]

    class SuperResolution3DNet(torch.nn.Module):
        def __init__(self):
            super(SuperResolution3DNet, self).__init__()

            # Encoder
            # The encoder consists of two 3D convolutional layers.
            # The first conv layer expands the channel size from 25 to 64.
            # The second conv layer further expands the channel size from 64 to 128.
            # ReLU activation functions are used for non-linearity.
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv3d(25, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
            )

            # Decoder
            # The decoder uses a transposed 3D convolution (or deconvolution) to upsample the feature maps.
            # The channel size is reduced from 128 back to 64.
            # A final 3D convolution reduces the channel size back to the original size of 25.
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(64, 25, kernel_size=3, padding=1),
            )

        def forward(self, x):
            # Forward pass of the network
            # x is the input tensor

            # Save the original input for later use
            tmp = x

            # Apply the encoder
            x = self.encoder(x)

            # Apply the decoder
            x = self.decoder(x)

            # Reusing the input data for faster learning
            # Here, every 2nd element in the spatial dimensions of x is replaced by the corresponding element in the original input.
            # This is a form of skip connection, which helps in retaining high-frequency details from the input.

            x[:, :, ::2, ::2, ::2] = tmp[:, :, ...]

            return x

    # Instantiate the model
    net = SuperResolution3DNet().to(torch.double)

    # global step counter
    counter = 0

    losses_train = []
    losses_val = []
    steps_val = []

    optimizerBFGS = torch.optim.LBFGS(
        net.parameters(), lr=0.1
    )  # Use LBFGS sometimes, it really does do magic sometimes, though its a bit of a diva
    optimizerADAM = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Define the ratio for the split (e.g., 80% train, 20% test)
    train_ratio = 0.8
    test_ratio = 1 - train_ratio

    # Calculate the number of samples for each split
    num_samples = len(dataX)
    num_train = int(train_ratio * num_samples)
    num_test = num_samples - num_train

    train_torch = dataX[:num_train].permute(0, 4, 1, 2, 3).to(device)
    test_torch = dataX[num_train:].permute(0, 4, 1, 2, 3).to(device)

    batch_size = 5

    # Create DataLoader for batching -- in case data gets larger
    train_loader = DataLoader(
        dataset=TensorDataset(train_torch),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(test_torch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Magical loss coming from General Relativity
    class Hamiltonian_loss:
        def __init__(self, oneoverdx: float):
            self.oneoverdx = oneoverdx

        def __call__(self, output: torch.tensor, dummy: torch.tensor) -> torch.tensor:
            # For learning we need shape (batch,channel,x,y,z), however TorchGRTL works with (batch,x,y,z,channel), thus the permute
            output = output.permute(0, 2, 3, 4, 1)
            # cutting ghosts off, otherwise you will run into problems later
            dataXcut = cut_ghosts(output)

            # creating dict with values
            vars = TensorDict(dataXcut, keys)
            # creating dict with derivatives
            d1 = TensorDict(diff1(output, oneoverdx), keys)
            d2 = TensorDict(diff2(output, oneoverdx**2), keys)
            # calculating variables required for constraints
            h_UU = torch.inverse(vars["h"])
            chris = compute_christoffel(d1["h"], h_UU)
            # Computing Constraints
            out = constraint_equations(vars, d1, d2, h_UU, chris)
            loss = torch.mean(out["Ham"] * out["Ham"])
            return loss

    # load model in case you restart form checkpoint
    restart = False
    file_path = "model_epoch_counter_0000004000_data_time_1706644608.pth"
    if restart and os.path.exists(file_path):
        net.load_state_dict(torch.load(file_path))

    oneoverdx = 64.0 / 16.0
    my_loss = Hamiltonian_loss(oneoverdx)

    # Note: it will slow down signficantly with BFGS steps, they are 10x slower, just be aware!
    ADAMsteps = (
        1000  # Will perform # steps of ADAM steps and then switch over to BFGS-L
    )
    n_steps = 1000  # Total amount of steps

    net.train()
    net.to(device)

    my_loss = torch.nn.L1Loss()
    print("training")
    pbar = trange(n_steps)
    for i in pbar:
        total_loss_train = 0
        for (y_batch,) in train_loader:
            batchcounter = 0
            # for X_batch, y_batch in train_loader:
            y_batch = y_batch.to(device)
            X_batch = y_batch[:, :, ::2, ::2, ::2].clone()
            y_batch = y_batch[:, :25, ...]
            batchcounter += 1

            # This is needed for LBFGS
            def closure():
                if torch.is_grad_enabled():
                    optimizerBFGS.zero_grad()
                y_pred = net(X_batch)

                loss_train = my_loss(y_pred, y_batch)
                if loss_train.requires_grad:
                    loss_train.backward()
                return loss_train

            # doing some ADAM first to warm up, sometimes BFGS fuckes up if you start too early
            if counter < ADAMsteps:
                y_pred = net(X_batch)

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
        writer.add_scalar("loss/train", average_loss_train, counter)
        losses_train.append(average_loss_train)
        if np.isnan(average_loss_train):
            print("we got nans")

        # Validation

        if counter % 4 == 0:
            with torch.no_grad():
                total_loss_val = (
                    0.0  # Initialize a variable to accumulate the total loss
                )
                for (y_val_batch,) in test_loader:
                    # for X_val_batch, y_val_batch in test_loader:
                    # Transfer batch to GPU
                    y_val_batch = y_val_batch.to(device)
                    X_val_batch = y_val_batch[:, :, ::2, ::2, ::2].clone()
                    y_val_batch = y_val_batch[:, :25]
                    y_val_pred = net(X_val_batch)
                    loss_val = my_loss(y_val_pred, y_val_batch)
                    total_loss_val += loss_val.item()
                # Calculate the average loss
                average_loss_val = total_loss_val / len(test_loader)
                losses_val.append(average_loss_val)
                steps_val.append(counter)
                writer.add_scalar("loss/test", loss_val.item(), counter)
        if counter % 1000 == 0:
            # Writing out network and scaler
            torch.save(
                net.state_dict(),
                f"{folder_name}/model_epoch_counter_{counter:010d}_data_time_{time_stamp}.pth",
            )
        # Advancing global counter
        counter += 1
    # print(steps_val)
    # Plotting shit at the end
    plt.figure(figsize=(9, 6))
    plt.plot(np.array(losses_train), label="Train")
    plt.plot(
        steps_val, np.array(losses_val), label="Val with Relative loss", linewidth=0.5
    )
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{folder_name}/training.png")
    plt.close()

    writer.flush()
    writer.close()

    # Get comparison with classical methods

    # Interpolation compared to what is used typically in codes ( we interpolate between 6 values with polynomials x^i y^k z^k containing powers up to 3)
    points = 6
    power = 3
    channels = 25
    shape = X_batch.shape
    interpolation = interp(points, power, channels, False, True, torch.double)
    ghosts = int(math.ceil(points / 2))
    shape_higher_order = (shape[-1] - 2 * ghosts) * 2 + 2

    y_interpolated = interpolation(X_batch.detach().cpu()).detach().to(torch.double)
    y_interpolated_buffer = torch.zeros_like(y_batch.detach())
    diff = (y_batch.shape[-1] - y_interpolated.shape[-1]) // 2
    y_interpolated_buffer[:, :, diff:-diff, diff:-diff, diff:-diff] = y_interpolated

    box = 0
    channel = 0
    slice = 9
    # Note we remove some part of the grid as the interpolation needs space
    max_val = (
        torch.max(
            y_batch[box, channel, diff - 1 : -diff - 1, diff - 1 : -diff - 1, slice - 1]
        )
        .cpu()
        .numpy()
    )
    min_val = (
        torch.min(
            y_batch[box, channel, diff - 1 : -diff - 1, diff - 1 : -diff - 1, slice - 1]
        )
        .cpu()
        .numpy()
    )
    net.eval()
    y_pred = net(X_batch.detach())

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot ground truth
    axes[0].set_title("Ground Truth")
    im0 = axes[0].imshow(
        y_batch[box, channel, diff - 1 : -diff - 1, diff - 1 : -diff - 1, slice - 1]
        .cpu()
        .numpy(),
        vmin=min_val,
        vmax=max_val,
        cmap="viridis",
    )
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")

    # Plot Neural Network
    axes[1].set_title("Neural Network")
    im1 = axes[1].imshow(
        y_pred[box, channel, diff - 1 : -diff - 1, diff - 1 : -diff - 1, slice - 1]
        .detach()
        .cpu()
        .numpy(),
        vmin=min_val,
        vmax=max_val,
        cmap="viridis",
    )
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")

    # Plot Interpolation
    axes[2].set_title("Interpolation")
    im2 = axes[2].imshow(
        y_interpolated[box, channel, :, :, slice - diff],
        vmin=min_val,
        vmax=max_val,
        cmap="viridis",
    )
    axes[2].set_xlabel("X-axis")
    axes[2].set_ylabel("Y-axis")

    # Add colorbars
    cbar0 = fig.colorbar(im0, ax=axes[0])
    # cbar0.set_label('Values')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    # cbar1.set_label('Values')
    cbar2 = fig.colorbar(im2, ax=axes[2])
    # cbar2.set_label('Values')
    plt.tight_layout()
    plt.savefig(folder_name + "/comparison2d.png")
    plt.close()

    box = 0
    channel = 0
    slice = 5

    net.eval()
    y_pred = net(X_batch.detach())

    plt.plot(
        y_batch[box, channel, diff - 1 : -diff - 1, slice - 1, slice - 1]
        .detach()
        .cpu()
        .numpy(),
        label="ground truth",
    )
    plt.plot(
        y_pred[box, channel, diff - 1 : -diff - 1, slice - 1, slice - 1]
        .detach()
        .cpu()
        .numpy(),
        label="neural network ",
    )
    plt.plot(
        y_interpolated[box, channel, :, slice - diff, slice - diff]
        .detach()
        .cpu()
        .numpy(),
        label="interpolation ",
        linestyle=":",
        alpha=0.6,
    )
    plt.legend()
    plt.savefig(folder_name + "/comparison1d.png")
    plt.close()

    box = 0
    channel = 0
    slice = 5

    net.eval()
    y_pred = net(X_batch.detach())

    plt.plot(
        np.abs(
            y_batch[box, channel, diff - 1 : -diff - 1, slice - 1, slice - 1]
            .detach()
            .cpu()
            .numpy()
            - y_pred[box, channel, diff - 1 : -diff - 1, slice - 1, slice - 1]
            .detach()
            .cpu()
            .numpy()
        ),
        label="neural network residual ",
    )
    plt.plot(
        np.abs(
            y_batch[box, channel, diff - 1 : -diff - 1, slice - 1, slice - 1]
            .detach()
            .cpu()
            .numpy()
            - y_interpolated[box, channel, :, slice - diff, slice - diff]
            .detach()
            .cpu()
            .numpy()
        ),
        label="interpolation residual",
        linestyle=":",
        alpha=0.6,
    )
    plt.yscale("log")
    plt.legend()
    plt.savefig(folder_name + "/residual.png")
    plt.close()

    # Calculate L2Ham performance
    my_loss = Hamiltonian_loss(oneoverdx)

    net.eval()
    y_pred = net(X_batch.detach())

    with open(folder_name + "/Metric_data.txt", "a") as file:
        file.write(
            f"Reference data L2 Ham {my_loss(y_batch[:, :, diff-1:-diff-1, diff-1:-diff-1, diff-1:-diff-1], torch.tensor([])).detach().cpu().numpy()}\n"
        )
        file.write(
            f"Neural Network L2 Ham {my_loss(y_pred[:, :, diff-1:-diff-1, diff-1:-diff-1, diff-1:-diff-1], torch.tensor([])).detach().cpu().numpy()}\n"
        )
        file.write(
            f"Interpolation L2 Ham  {my_loss(y_interpolated, torch.tensor([])).detach().numpy()}\n"
        )
        file.write("--------------------\n")

    # Calculate L1 performance
    my_loss = torch.nn.L1Loss()

    net.eval()
    y_pred = net(X_batch.detach())

    with open(folder_name + "/Metric_data.txt", "a") as file:
        file.write(
            f"L1 loss Neural Network {my_loss(y_pred[:, :, diff-1:-diff-1, diff-1:-diff-1, diff-1:-diff-1].cpu(), y_batch[:, :, diff-1:-diff-1, diff-1:-diff-1, diff-1:-diff-1].cpu())}\n"
        )
        file.write(
            f"L1 loss interpolation {my_loss(y_interpolated.cpu(), y_batch[:, :, diff-1:-diff-1, diff-1:-diff-1, diff-1:-diff-1].cpu())}\n"
        )


if __name__ == "__main__":
    main()
