import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import math
import time
from typing import Tuple
import torch.nn as nn


def print_grid_lay_out(
    interp_point: np.ndarray = np.array([0.5, 0.5, 0.5]),
    num_points: int = 6,
    max_degree: int = 4,
):
    half = int(np.floor(float(max_degree) / 2.0))
    # Generate 3D meshgrid for coarse grid points
    coarse_grid_x, coarse_grid_y, coarse_grid_z = np.meshgrid(
        np.arange(0 - half, num_points - half),
        np.arange(0 - half, num_points - half),
        np.arange(0 - half, num_points - half),
    )
    coarse_grid_points_index = np.vstack(
        [coarse_grid_x.ravel(), coarse_grid_y.ravel(), coarse_grid_z.ravel()]
    ).T

    vecvals, grid_points_index = calculate_stencils(
        interp_point, num_points, max_degree
    )

    # testplotting the different weights
    fig = plt.figure()
    ax = fig.add_subplot()
    norm = (vecvals - min(vecvals)) / (max(vecvals) + abs(min(vecvals)))
    zeros_z = np.where(coarse_grid_points_index[:, 2] == 0)
    ax.scatter(
        coarse_grid_points_index[zeros_z, 0],
        coarse_grid_points_index[zeros_z, 1],
        c=norm[zeros_z] * 100,
        cmap="jet",
    )
    ax.scatter(interp_point[0], interp_point[1], label="interpolation point", c="black")
    plt.legend()
    plt.savefig("layout_interpolation_grid.png")
    plt.close()


def calculate_stencils(
    interp_point: np.ndarray = np.array([0.5, 0.5, 0.5]),
    num_points: int = 6,
    max_degree: int = 3,
) -> (np.ndarray, np.ndarray):
    """
    Calculate interpolation stencils for a given point in a 3D grid.

    This function computes the coefficients for polynomial interpolation in a 3D grid.
    It uses a least squares approach to find the coefficients that best fit the data
    points in the grid.

    The coordinates of the grid are the following for num_points = 6
     [[(-2,-2) (-2,-1) (-2,0) (-2,1) (-2,2) (-2,3) ]
      [(-1,-2) (-1,-1) (-1,0) (-1,1) (-1,2) (-1,3) ]
      [( 0,-2) ( 0,-1) ( 0,0) ( 0,1) ( 0,2) ( 0,3) ]
      [( 1,-2) ( 1,-1) ( 1,0) ( 1,1) ( 1,2) ( 1,3) ]
      [( 2,-2) ( 2,-1) ( 2,0) ( 2,1) ( 2,2) ( 2,3) ]
      [( 3,-2) ( 3,-1) ( 3,0) ( 3,1) ( 3,2) ( 3,3) ]]
    and the input vector points ask what point you would like
    to obtain. The exact center of this grid is (0.5,0.5,0.5).

    Parameters:
    interp_point (np.ndarray): The 3D point where interpolation is to be performed.
    num_points (int): Number of points along each axis in the grid.
    max_degree (int): Maximum degree of the polynomial used for interpolation.

    Returns:
    tuple: A tuple containing two np.ndarrays. The first array contains the coefficients
           of the interpolation polynomial. The second array contains the indices of the
           coarse grid points used for interpolation.
    """

    dx = 1
    # Shift index to center around zero
    half = int(np.floor(float(max_degree) / 2.0))

    # Generate 3D meshgrid for coarse grid points
    coarse_grid_x, coarse_grid_y, coarse_grid_z = np.meshgrid(
        np.arange(0 - half, num_points - half),
        np.arange(0 - half, num_points - half),
        np.arange(0 - half, num_points - half),
    )
    coarse_grid_points_index = np.vstack(
        [coarse_grid_x.ravel(), coarse_grid_y.ravel(), coarse_grid_z.ravel()]
    ).T
    coarse_grid_points = dx * coarse_grid_points_index
    # Evaluate the sinusoidal function on the coarse grid points

    # Define the fine grid point where you want to interpolate the value
    interp_point = dx * interp_point

    # Prepare the matrix of powers for the coarse grid points
    # Including the constant term where i = j = k = 0
    powers = [
        (i, j, k)
        for i in range(max_degree + 1)
        for j in range(max_degree + 1)
        for k in range(max_degree + 1)
    ]

    # Calculate the matrix of basis functions evaluations on the coarse grid
    coarse_matrix = np.array(
        [
            [x**i * y**j * z**k for (i, j, k) in powers]
            for x, y, z in coarse_grid_points
        ]
    )

    # Compute the normal matrix A^TA and the right-hand side A^Tb
    normal_matrix = coarse_matrix.T @ coarse_matrix
    rhs = coarse_matrix.T

    # Invert normal matrix and solve for coefficients
    normal_matrix = np.linalg.inv(normal_matrix)
    # Solve the normal equations for the coefficients
    coefficients = normal_matrix @ rhs

    # Vector of basis functions evaluations at the interpolation point
    fine_vector = np.array(
        [
            interp_point[0] ** i * interp_point[1] ** j * interp_point[2] ** k
            for (i, j, k) in powers
        ]
    )
    # Calculate coefficients for the interpolated point
    vecvals = fine_vector @ (coefficients)

    return vecvals, coarse_grid_points_index


class interp:
    def __init__(
        self,
        num_points: int = 6,
        max_degree: int = 3,
        num_channels: int = 1,
        learnable: bool = False,
        align_grids_with_lower_dim_values: bool = False,
    ):
        """
        Initialize the Interp class.

        Parameters:
        num_points (int): Number of points to use in interpolation.
        max_degree (int): The maximum degree of the polynomial used in interpolation.
        num_channels (int): Number of channels in the input tensor.
        learnable (bool): If True, the interpolation parameters are learnable.
        align_grids_with_lower_dim_values (bool): If True, aligns grid points with lower-dimensional values.
        """
        self.num_points = num_points
        self.max_degree = max_degree
        self.vecvals_array = []  # Vector values for interpolation
        self.grid_points_index_array = []  # Grid points indices for interpolation

        # Define fixed values for grid alignment
        if align_grids_with_lower_dim_values:
            values = [0.0, 0.50]
        else:
            values = [0.25, 0.75]
        self.relative_positions = np.array(list(itertools.product(values, repeat=3)))

        # Calculate vector values and grid points indices
        for interp_point in self.relative_positions:
            vecvals, grid_points_index = calculate_stencils(
                interp_point, num_points, max_degree
            )
            vecvals[np.abs(vecvals) < 1e-10] = 0
            self.vecvals_array.append(vecvals.tolist())
            self.grid_points_index_array.append(grid_points_index.tolist())

        # Compute relative indices for interpolated array

        if align_grids_with_lower_dim_values:
            self.relative_index_for_interpolated_array = np.int8(
                np.round(self.relative_positions + 0.5)
            )
        else:
            self.relative_index_for_interpolated_array = np.int8(
                np.round(self.relative_positions)
            )

        # Convert lists to tensors for faster computation
        self.grid_points_index_array = torch.tensor(self.grid_points_index_array)
        self.vecvals_array = torch.tensor(self.vecvals_array)

        ## Initialize the kernel for convolution

        self.conv_layers = []

        # Iterate over the displacement, weight, and relative position information
        for (
            displacements,
            weights,
        ) in zip(self.grid_points_index_array, self.vecvals_array):
            kernel_size = self.num_points  # Size of the convolutional kernel

            # Create a convolutional kernel with zeros
            kernel = torch.zeros(
                (num_channels, 1, kernel_size, kernel_size, kernel_size)
            )

            # Find the minimum index for displacements to adjust kernel indexing
            min_index = torch.min(displacements)

            # Populate the kernel with weights according to displacements
            for displacement, weight in zip(displacements, weights):
                index = (
                    displacement - min_index
                )  # Adjust index based on minimum displacement
                kernel[:, :, index[0], index[1], index[2]] = weight

            conv_layer = nn.Conv3d(num_channels, num_channels, kernel_size)
            conv_layer.weight = nn.Parameter(kernel)
            conv_layer.bias = nn.Parameter(torch.zeros(num_channels))
            conv_layer.weight.requires_grad = learnable
            conv_layer.bias.requires_grad = learnable
            self.conv_layers.append(conv_layer)

    def __call__(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the interpolation on the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor to interpolate.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the interpolated tensor and the position tensor.
        """

        # Calculate the number of ghost points based on num_points
        # Ghost points are used for padding or handling edges during interpolation
        ghosts = int(math.ceil(self.num_points / 2))
        shape = tensor.shape

        # Initialize the tensor for storing interpolation results
        # The output tensor will have modified spatial dimensions based on the number of ghost points
        interpolation = torch.zeros(
            shape[0],  # batch size
            shape[1],  # number of channels
            (shape[2] - 2 * ghosts) * 2 + 2,  # modified x dimension
            (shape[3] - 2 * ghosts) * 2 + 2,  # modified y dimension
            (shape[4] - 2 * ghosts) * 2 + 2,  # modified z dimension
        )

        # Initialize a tensor to store positions
        # This tensor keeps track of the positions in the interpolated space
        position = torch.zeros(
            (shape[2] - 2 * ghosts) * 2 + 2,  # x dimension
            (shape[3] - 2 * ghosts) * 2 + 2,  # y dimension
            (shape[4] - 2 * ghosts) * 2 + 2,  # z dimension
            3,  # 3D coordinates
        )

        for (
            displacements,
            weights,
            relative_index,
            relative_position,
            conv_layer,
        ) in zip(
            self.grid_points_index_array,
            self.vecvals_array,
            self.relative_index_for_interpolated_array,
            self.relative_positions,
            self.conv_layers,
        ):
            convoluted_tensor = conv_layer(tensor)

            # Update the interpolation tensor with the convolution results
            # This is done selectively based on the relative index
            interpolation[
                :,
                :,
                relative_index[0] :: 2,
                relative_index[1] :: 2,
                relative_index[2] :: 2,
            ] = convoluted_tensor

        return interpolation

    def get_postion(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the position of the interpolated points.

        Parameters:
        tensor (torch.tensor): The input tensor to interpolate.

        Returns:
        torch.tensor: The interpolated tensor.
        """

        # Calculate the number of ghost points based on num_points
        ghosts = int(math.ceil(self.num_points / 2))
        shape = tensor.shape

        # Initialize a tensor to store positions
        position = torch.zeros(
            (shape[2] - 2 * ghosts) * 2 + 2,
            (shape[3] - 2 * ghosts) * 2 + 2,
            (shape[4] - 2 * ghosts) * 2 + 2,
            3,
        )

        # Perform interpolation
        for i in range(ghosts - 2, shape[2] - ghosts - 1):
            for j in range(ghosts - 2, shape[3] - ghosts - 1):
                for k in range(ghosts - 2, shape[4] - ghosts - 1):
                    index_for_input_array = torch.tensor([i, j, k])
                    for (
                        displacements,
                        weights,
                        relative_index,
                        relative_position,
                    ) in zip(
                        self.grid_points_index_array,
                        self.vecvals_array,
                        self.relative_index_for_interpolated_array,
                        self.relative_positions,
                    ):
                        result = 0

                        # Writing results to the interpolated array
                        ind = 2 * (index_for_input_array - (ghosts - 2)) + (
                            relative_index
                        )
                        # This array gives the position of the interpolated point in the interpolated array relative to the input array
                        position[ind[0], ind[1], ind[2]] = (
                            index_for_input_array + relative_position
                        )

        return position

    def non_vector_implementation(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the interpolation on the given tensor.

        Parameters:
        tensor (torch.tensor): The input tensor to interpolate.

        Returns:
        torch.tensor: The interpolated tensor.
        """

        # Calculate the number of ghost points based on num_points
        ghosts = int(math.ceil(self.num_points / 2))
        shape = tensor.shape

        # Initialize the tensor for storing interpolation results
        interpolation = torch.zeros(
            shape[0],
            shape[1],
            (shape[2] - 2 * ghosts) * 2 + 2,
            (shape[3] - 2 * ghosts) * 2 + 2,
            (shape[4] - 2 * ghosts) * 2 + 2,
        )

        # Initialize a tensor to store positions
        position = torch.zeros(
            (shape[2] - 2 * ghosts) * 2 + 2,
            (shape[3] - 2 * ghosts) * 2 + 2,
            (shape[4] - 2 * ghosts) * 2 + 2,
            3,
        )

        # Perform interpolation
        for i in range(ghosts - 2, shape[2] - ghosts - 1):
            for j in range(ghosts - 2, shape[3] - ghosts - 1):
                for k in range(ghosts - 2, shape[4] - ghosts - 1):
                    index_for_input_array = torch.tensor([i, j, k])
                    for (
                        displacements,
                        weights,
                        relative_index,
                        relative_position,
                    ) in zip(
                        self.grid_points_index_array,
                        self.vecvals_array,
                        self.relative_index_for_interpolated_array,
                        self.relative_positions,
                    ):
                        result = 0
                        for displacement, weight in zip(displacements, weights):
                            # Ensure indices are scalar values
                            index = index_for_input_array + displacement
                            result += (
                                weight * tensor[:, :, index[0], index[1], index[2]]
                            )

                        # Writing results to the interpolated array
                        ind = 2 * (index_for_input_array - (ghosts - 2)) + (
                            relative_index
                        )
                        interpolation[:, :, ind[0], ind[1], ind[2]] = result
                        # This array gives the position of the interpolated point in the interpolated array relative to the input array
                        position[ind[0], ind[1], ind[2]] = (
                            index_for_input_array + relative_position
                        )

        return interpolation, position

    def sinusoidal_function(self, x, y, z):
        """
        A sinusoidal function of three variables x, y, and z.
        """
        return np.sin(x) * np.sin(y) * np.sin(z)

    def plot_grid_position(self):
        length = 10
        dx = 0.01
        x = torch.rand(2, 1, length, length, length)
        input_positions = torch.zeros(length, length, length, 3)
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                for k in range(x.shape[4]):
                    input_positions[i, j, k] = torch.tensor([i, j, k])
                    pos = dx * np.array([i, j, k])
                    x[:, :, i, j, k] = self.sinusoidal_function(*pos)
        time1 = time.time()
        interpolated = self(x)
        print(f"Time taken for interpolation: {(time.time() - time1):.2f} sec")
        positions = self.get_postion(x)
        ghosts = int(math.ceil(6 / 2))
        # Scatter plot
        plt.scatter(
            input_positions[:, :, 4, 0],
            input_positions[:, :, 4, 1],
            label="Input",
            color="blue",
            marker="o",
        )
        plt.scatter(
            positions[:, :, 4, 0],
            positions[:, :, 4, 1],
            label="Interpolated",
            color="red",
            marker="x",
        )

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
        plt.xticks(input_positions[:, 0, 4, 0])
        plt.yticks(input_positions[0, :, 4, 1])
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.savefig(f"interpolation_grid.png")
        plt.close()


if __name__ == "__main__":
    print_grid_lay_out()
    interpolation = interp(6, 3)
    interpolation.plot_grid_position()
