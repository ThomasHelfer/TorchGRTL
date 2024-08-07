import re
import glob
import os
import yaml
from typing import Tuple, Dict, Any

from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np

from pyinterpx.Interpolation import interp
from GeneralRelativity.Utils import get_box_format
from SuperResolution.losses import (
    Hamiltonian_loss,
    Hamiltonian_and_momentum_loss,
    Hamiltonian_and_momentum_loss_boundary_condition,
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
        align_corners=True,
    ):
        super(SuperResolution3DNet, self).__init__()
        self.points = 6
        self.power = 3
        self.channels = 25
        self.kernel_size = kernel_size
        self.padding = padding
        self.mask_type = mask_type
        self.factor = factor
        if factor > 1:
            self.interpolation = interp(
                num_points=self.points,
                max_degree=self.power,
                num_channels=self.channels,
                learnable=False,
                align_corners=align_corners,
                factor=factor,
                dtype=torch.double,
            )
        self.align_corners = align_corners
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
        if self.factor > 1:
            x = self.interpolation(x)
        tmp = x.clone()
        if self.training:
            if self.mask_type == "random":
                mask = create_mask(tmp, self.masking_percentage)
            elif self.mask_type == "checkers":
                mask = torch.ones_like(tmp)
                if self.masking_percentage != 0:
                    ratio = int(1.0 / self.masking_percentage)
                    random_shift = np.random.randint(0, ratio)
                    mask[
                        :, random_shift::ratio, random_shift::ratio, random_shift::ratio
                    ] = 0
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
    config: Dict[str, Any],
    batchsize: int = 50,
) -> Tuple[float, float]:
    """
    Evaluates the performance of the neural network on the dataset.

    Args:
        net (torch.nn.Module): The neural network model to evaluate.
        datafolder (str): Path to the folder containing the data.
        my_loss (torch.nn.Module): The loss function used for evaluation.
        batchsize (int, optional): The batch size for data loading. Defaults to 50.

    Returns:
        Tuple[float, float]: The total validation loss and the interpolation validation loss.
    """
    res_level = config["res_level"]
    downsample = config["downsample"]
    factor = config["factor"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # oneoverdx = 64.0 / 16.0
    oneoverdx = (64.0 * 2**res_level) / 512.0 * float(factor) / float(downsample)
    print(f"dx {1.0/oneoverdx}")

    my_loss = Hamiltonian_and_momentum_loss(oneoverdx)

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
        for (y_val_batch,) in tqdm(data_loader):
            # Move the batch to the specified device
            y_val_batch = y_val_batch.to(device)
            X_val_batch = y_val_batch[
                :, :, ::downsample, ::downsample, ::downsample
            ].clone()

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # List to store the checkpoint file names and their indices
    checkpoints = []

    # Path to the config.yaml file
    yaml_files = glob.glob(os.path.join(directory_path, "*.yaml"))

    if yaml_files:
        # If a YAML file is found, use the first one
        config_file_path = yaml_files[0]
    else:
        print("No YAML file found in the directory.")
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

    # Pattern to match the checkpoint files
    pattern = r"model_epoch_counter_(\d+)_data_time_\d+\.pth"

    # List to store the checkpoint file names and their indices
    checkpoints = []

    # Iterate through the files in the directory to find checkpoint files
    for filename in os.listdir(directory_path):
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            checkpoints.append((index, filename))

    # Find the checkpoint with the largest index
    if checkpoints:
        largest_index_checkpoint = max(checkpoints, key=lambda x: x[0])
        largest_checkpoint_file = largest_index_checkpoint[1]
        print(largest_index_checkpoint[1])
    else:
        raise FileNotFoundError("No checkpoint files found.")

    path_to_largest_checkpoint_file = os.path.join(
        directory_path, largest_checkpoint_file
    )
    print(device)
    # Load the model state if restarting
    if os.path.exists(path_to_largest_checkpoint_file):
        net.load_state_dict(
            torch.load(
                path_to_largest_checkpoint_file, map_location=torch.device(device)
            )
        )
        print(f"loaded model from {path_to_largest_checkpoint_file}")
    else:
        print(
            f"No restart or checkpoint file not found at path: {path_to_largest_checkpoint_file}"
        )

    return net, config


def calculate_test_loss(
    net: torch.nn.Module, config: Dict[str, Any], name: str
) -> Dict[str, float]:
    """
    Loads data, processes it, and computes various metrics for a given network and configuration.

    Parameters:
    net (torch.nn.Module): The neural network model to be evaluated.
    config (Dict[str, Any]): A dictionary containing configuration parameters.
    name (str): The name of the current configuration.

    Returns:
    Dict[str, float]: A dictionary containing various computed metrics.
    """
    train_ratio = config["train_ratio"]
    res_level = config["res_level"]
    filenamesX = config["filenamesX"].format(res_level=res_level)
    downsample = config["downsample"]
    factor = config["factor"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # For validation error
    L1Loss = torch.nn.L1Loss()
    ham_loss = Hamiltonian_and_momentum_loss(oneoverdx)

    num_varsX = 25
    dataX = get_box_format(filenamesX, num_varsX)
    # Cutting out extra values added for validation
    dataX = dataX[:, :, :, :, :25]

    # Calculate the number of samples for each split
    num_samples = len(dataX)
    num_train = int(train_ratio * num_samples)
    num_test = num_samples - num_train

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

    diff = (dataX.shape[-2] - output_tensor.shape[-1]) // 2

    dataX = dataX.permute(0, 4, 1, 2, 3)

    # Create a dataset from tensors
    dataset = TensorDataset(dataX)

    # Split the dataset into training and testing datasets
    train_dataset, test_dataset = random_split(
        dataset, [num_train, num_test], generator=torch.Generator().manual_seed(3)
    )
    batch_size = config["batch_size"]

    # Create DataLoader for batching -- in case data gets larger
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    net.to(device)
    net.to(torch.double)

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
            if diff != 0:
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

        N = len(test_loader)
        metrics = {
            "config_name": name,  # Example of saving config name
            "total_loss_val": total_loss_val / N,
            "interp_val": interp_val / N,
            "L1Loss_val": L1Loss_val / N,
            "L1Loss_val_interp": L1Loss_val_interp / N,
            "loss_hard_base": loss_hard_base / N,
            "Ham_loss": Ham_loss / N,
            "Ham_loss_interp": Ham_loss_interp / N,
        }
    return metrics
