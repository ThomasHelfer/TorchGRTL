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


def create_mask(tensor, zero_percentage):
    """
    Create a mask for a tensor such that a given percentage of elements are set to zero.

    Args:
        tensor (torch.Tensor): The input tensor for which the mask is created.
        zero_percentage (float): The percentage of elements to set to zero (between 0 and 1).

    Returns:
        torch.Tensor: A mask with the same shape as the input tensor, containing zeros and ones.
    """
    # Get the shape and device of the input tensor
    tensor_shape = tensor.shape
    device = tensor.device

    # Determine the total number of elements
    num_elements = tensor.numel()

    # Determine the number of elements to set to zero
    num_zeros = int(num_elements * zero_percentage)

    # Create a mask with the specified percentage of zeros
    mask = torch.ones(num_elements, device=device)
    mask[:num_zeros] = 0

    # Shuffle the mask
    mask = mask[torch.randperm(num_elements)]

    # Reshape the mask to the shape of the tensor
    mask = mask.reshape(tensor_shape)

    return mask


class SuperResolution3DNet(torch.nn.Module):
    def __init__(
        self,
        factor,
        scaling_factor,
        num_layers=2,
        kernel_size=3,
        padding="same",
        nonlinearity="relu",
        masking_percentage=0.1,
        mask_type="random",
    ):
        super(SuperResolution3DNet, self).__init__()
        self.points = 6
        self.power = 3
        self.channels = 25
        self.kernel_size = kernel_size
        self.padding = padding
        self.mask_type = mask_type
        if scaling_factor > 1:
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
        self.num_layers = num_layers
        self.masking_percentage = masking_percentage

        if nonlinearity == "relu":
            self.nonlinearity = torch.nn.ReLU()
        elif nonlinearity == "gelu":
            self.nonlinearity = torch.nn.GELU()

        ## lower res
        layers = []
        in_channels = 25
        out_channels = 25
        hidden_channels = 64
        # First layer
        layers.append(
            nn.Conv3d(
                in_channels, hidden_channels, kernel_size=kernel_size, padding=padding
            )
        )
        layers.append(self.nonlinearity)

        # Hidden layers
        for i in range(self.num_layers):
            layers.append(
                nn.Conv3d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(self.nonlinearity)

        # Last layer
        layers.append(
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # Reusing the input data for faster learning
        if self.scaling_factor > 1:
            x = self.interpolation(x)
        tmp = x.clone()
        if self.training:
            if self.mask_type == "random":
                mask = create_mask(tmp, self.masking_percentage)
            elif self.mask_type == "checkers":
                mask = torch.ones_like(tmp)
                ratio = int(1.0 / self.masking_percentage)
                mask[:, ::ratio, ::ratio, ::ratio] = 0
            else:
                # throw error
                raise ValueError(
                    "Invalid mask type. Please use 'random' or 'checkers'."
                )
        else:
            mask = torch.ones_like(tmp)

        x = x + mask * self.encoder(tmp) * self.scaling_factor
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
