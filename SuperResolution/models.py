import os
import re
import yaml
from typing import Tuple, Dict, Any

from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pyinterpx.Interpolation import interp
from GeneralRelativity.Utils import (
    get_box_format,
    TensorDict,
    cut_ghosts,
    keys,
    keys_all,
)


class SuperResolution3DNet(torch.nn.Module):
    def __init__(
        self, factor, scaling_factor, kernel_size=3, padding="same", nonlinearity="relu"
    ):
        super(SuperResolution3DNet, self).__init__()
        self.points = 6
        self.power = 3
        self.channels = 25
        self.kernel_size = kernel_size
        self.padding = padding
        self.interpolation = interp(
            num_points=self.points,
            max_degree=self.power,
            num_channels=self.channels,
            learnable=False,
            align_corners=True,
            factor=factor,
            dtype=torch.double,
        )
        self.scaling_factor = scaling_factor

        if nonlinearity == "relu":
            self.nonlinearity = torch.nn.ReLU()
        elif nonlinearity == "gelu":
            self.nonlinearity = torch.nn.GELU()

        # Encoder
        # The encoder consists of two 3D convolutional layers.
        # The first conv layer expands the channel size from 25 to 64.
        # The second conv layer further expands the channel size from 64 to 128.
        # ReLU activation functions are used for non-linearity.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(25, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.nonlinearity,
            torch.nn.Conv3d(64, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.nonlinearity,
            torch.nn.Conv3d(64, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.nonlinearity,
            torch.nn.Conv3d(64, 25, kernel_size=self.kernel_size, padding=self.padding),
            self.nonlinearity,
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
        # Reusing the input data for faster learning

        x = self.interpolation(x)
        tmp = x.clone()

        x = x + self.encoder(tmp) * self.scaling_factor
        return x, tmp


def check_performance(
    net: torch.nn.Module,
    datafolder: str,
    my_loss: torch.nn.Module,
    device: torch.device,
    batchsize: int = 50,
) -> Tuple[float, float]:
    """
    Evaluates the performance of the neural network on the dataset.

    Args:
        net (torch.nn.Module): The neural network model to evaluate.
        datafolder (str): Path to the folder containing the data.
        my_loss (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device to perform the computations on (e.g., 'cpu' or 'cuda').
        batchsize (int, optional): The batch size for data loading. Defaults to 50.

    Returns:
        Tuple[float, float]: The total validation loss and the interpolation validation loss.
    """
    num_vars = 25

    # Load data in the required format
    dataX = get_box_format(datafolder, num_vars)

    # Cut out extra values added for validation (if any)
    dataX = dataX[:, :, :, :, :25]

    # Rearrange dimensions to match the expected input format for the network
    dataX = dataX.permute(0, 4, 1, 2, 3)

    # Create a dataset and a data loader
    dataset = TensorDataset(dataX)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    # Move the model to the specified device (CPU or GPU)
    net.to(device)
    net.eval()  # Set the model to evaluation mode

    print("Evaluating model")

    # Initialize variables to accumulate losses
    total_loss_val = 0.0
    interp_val = 0.0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches in the data loader
        for (X_val_batch,) in tqdm(data_loader):
            # Move the batch to the specified device
            X_val_batch = X_val_batch.to(device)

            # Forward pass: get predictions from the network
            y_val_pred, y_val_interp = net(X_val_batch)

            # Compute the loss for predictions
            loss_val = my_loss(y_val_pred)
            total_loss_val += loss_val.item()

            # Compute the loss for interpolations
            interp_val += my_loss(
                y_val_interp,
            ).item()

    # Return the accumulated losses
    return total_loss_val, interp_val


def load_model(directory_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load the SuperResolution3DNet model from the checkpoint with the largest index
    and read the configuration from the config.yaml file.

    Args:
        directory_path (str): Path to the directory containing the checkpoint files and config.yaml.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: The loaded model and configuration dictionary.
    """
    # Pattern to match the checkpoint files
    pattern = r"model_epoch_counter_(\d+)_data_time_\d+\.pth"

    # List to store the checkpoint file names and their indices
    checkpoints = []

    # Path to the config.yaml file
    config_file_path = os.path.join(directory_path, "config.yaml")
    print(f"Config file path: {config_file_path}")

    # Read Yaml config file
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
    else:
        raise FileNotFoundError(f"Config file not found at path: {config_file_path}")

    # Iterate through the files in the directory to find checkpoint files
    for filename in os.listdir(directory_path):
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            checkpoints.append((index, filename))

    # Initialize the model with parameters from the config
    factor = config["factor"]
    net = SuperResolution3DNet(factor, config["scaling_factor"]).to(torch.double)

    # Find the checkpoint with the largest index
    if checkpoints:
        largest_index_checkpoint = max(checkpoints, key=lambda x: x[0])
        largest_checkpoint_file = largest_index_checkpoint[1]
    else:
        raise FileNotFoundError("No checkpoint files found.")

    # Load the model state if restarting
    if config["restart"] and os.path.exists(largest_checkpoint_file):
        net.load_state_dict(
            torch.load(os.path.join(directory_path, largest_checkpoint_file))
        )
    else:
        print(
            f"No restart or checkpoint file not found at path: {largest_checkpoint_file}"
        )

    return net, config
