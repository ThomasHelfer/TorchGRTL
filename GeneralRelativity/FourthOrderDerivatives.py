import numpy as np
from tqdm.auto import tqdm, trange
import glob
import torch


def diff1(tensor: torch.Tensor, one_over_dx: float) -> torch.Tensor:
    """
    Compute the first derivative along all spatial dimensions (x, y, z) for a PyTorch tensor
    and return a single tensor of shape (batchsize, x-4, y-4, z-4, num_variables, 3).
    No padding is applied, so the derivative tensor will be smaller than the input tensor.

    :param tensor: Input tensor of shape (batchsize, x, y, z, num_variables).
    :param one_over_dx: Inverse of the grid spacing.
    :return: Tensor of shape (batchsize, x-4, y-4, z-4, num_variables, 3) containing derivatives.
    """
    weight_far = torch.tensor(8.333333333333333e-2)
    weight_near = torch.tensor(6.666666666666667e-1)

    derivatives = []

    for dim in range(1, 4):  # Loop over spatial dimensions (x, y, z)
        # Calculate derivatives without padding

        derivative = (
            weight_far * tensor.narrow(dim, 0, tensor.size(dim) - 4)
            - weight_near * tensor.narrow(dim, 1, tensor.size(dim) - 4)
            + weight_near * tensor.narrow(dim, 3, tensor.size(dim) - 4)
            - weight_far * tensor.narrow(dim, 4, tensor.size(dim) - 4)
        )

        indices = [x for x in [1, 2, 3] if x != dim]
        for ind in indices:
            derivative = derivative.narrow(ind, 2, tensor.size(ind) - 4)

        derivatives.append(derivative * one_over_dx)

    # Concatenate derivatives along a new dimension
    all_derivatives = torch.stack(derivatives, dim=-1)

    return all_derivatives


def mixed_diff2_tensor(
    tensor: torch.tensor, i: int, j: int, one_over_dx2: float
) -> torch.tensor:
    """
    Calculate the mixed second derivative for a PyTorch tensor of shape (batches, x, y, z, variables)
    along the specified spatial dimensions (i, j).

    :param tensor: Input tensor of shape (batches, x, y, z, variables).
    :param i: First spatial dimension for mixed derivative (0 for x, 1 for y, 2 for z).
    :param j: Second spatial dimension for mixed derivative (0 for x, 1 for y, 2 for z).
    :param one_over_dx2: Inverse of the square of the grid spacing.
    :return: Tensor containing mixed second derivatives.
    """

    weight_far_far = 6.94444444444444444444e-3
    weight_near_far = 5.55555555555555555556e-2
    weight_near_near = 4.44444444444444444444e-1

    # Adjust indices for the spatial dimensions (add 1 because first dimension is batch)
    dim1 = i + 1
    dim2 = j + 1

    # Calculate mixed second derivatives
    mixed_derivative = (
        weight_far_far
        * tensor.narrow(dim1, 0, tensor.size(dim1) - 4).narrow(
            dim2, 0, tensor.size(dim2) - 4
        )
        - weight_near_far
        * tensor.narrow(dim1, 0, tensor.size(dim1) - 4).narrow(
            dim2, 1, tensor.size(dim2) - 4
        )
        + weight_near_far
        * tensor.narrow(dim1, 0, tensor.size(dim1) - 4).narrow(
            dim2, 3, tensor.size(dim2) - 4
        )
        - weight_far_far
        * tensor.narrow(dim1, 0, tensor.size(dim1) - 4).narrow(
            dim2, 4, tensor.size(dim2) - 4
        )
        - weight_near_far
        * tensor.narrow(dim1, 1, tensor.size(dim1) - 4).narrow(
            dim2, 0, tensor.size(dim2) - 4
        )
        + weight_near_near
        * tensor.narrow(dim1, 1, tensor.size(dim1) - 4).narrow(
            dim2, 1, tensor.size(dim2) - 4
        )
        - weight_near_near
        * tensor.narrow(dim1, 1, tensor.size(dim1) - 4).narrow(
            dim2, 3, tensor.size(dim2) - 4
        )
        + weight_near_far
        * tensor.narrow(dim1, 1, tensor.size(dim1) - 4).narrow(
            dim2, 4, tensor.size(dim2) - 4
        )
        + weight_near_far
        * tensor.narrow(dim1, 3, tensor.size(dim1) - 4).narrow(
            dim2, 0, tensor.size(dim2) - 4
        )
        - weight_near_near
        * tensor.narrow(dim1, 3, tensor.size(dim1) - 4).narrow(
            dim2, 1, tensor.size(dim2) - 4
        )
        + weight_near_near
        * tensor.narrow(dim1, 3, tensor.size(dim1) - 4).narrow(
            dim2, 3, tensor.size(dim2) - 4
        )
        - weight_near_far
        * tensor.narrow(dim1, 3, tensor.size(dim1) - 4).narrow(
            dim2, 4, tensor.size(dim2) - 4
        )
        - weight_far_far
        * tensor.narrow(dim1, 4, tensor.size(dim1) - 4).narrow(
            dim2, 0, tensor.size(dim2) - 4
        )
        + weight_near_far
        * tensor.narrow(dim1, 4, tensor.size(dim1) - 4).narrow(
            dim2, 1, tensor.size(dim2) - 4
        )
        - weight_near_far
        * tensor.narrow(dim1, 4, tensor.size(dim1) - 4).narrow(
            dim2, 3, tensor.size(dim2) - 4
        )
        + weight_far_far
        * tensor.narrow(dim1, 4, tensor.size(dim1) - 4).narrow(
            dim2, 4, tensor.size(dim2) - 4
        )
    )

    # Final calculation multiplied by one_over_dx2
    mixed_derivative *= one_over_dx2

    indices = [x for x in [1, 2, 3] if (x != dim1) and (x != dim2)]
    for ind in indices:
        mixed_derivative = mixed_derivative.narrow(ind, 2, tensor.size(ind) - 4)

    return mixed_derivative


def diff2_multidim(tensor: torch.tensor, i: int, one_over_dx2: float) -> torch.tensor:
    """
    Calculate the second derivative using a finite difference method for a multi-dimensional
    PyTorch tensor along the specified spatial dimension.

    :param tensor: Input tensor of shape (batch, x, y, z, num_vars).
    :param i: Index representing the spatial dimension (0 for x, 1 for y, 2 for z).
    :param one_over_dx2: Inverse of the square of the grid spacing.
    :return: Tensor of the same shape as the input tensor containing the second derivative along the specified dimension.
    """
    weight_far = 8.33333333333333333333e-2
    weight_near = 1.33333333333333333333e0
    weight_local = 2.50000000000000000000e0

    # Determine the spatial dimension to calculate the derivative
    dim = i + 1  # Adjusting for batch dimension

    ## Pad the tensor for boundary conditions
    # padded_tensor = torch.nn.functional.pad(tensor, pad=(0, 0, 0, 0, 2, 2, 2, 2), mode='replicate')
    """
        data_t weight_far = 8.33333333333333333333e-2;
        data_t weight_near = 1.33333333333333333333e+0;
        data_t weight_local = 2.50000000000000000000e+0;

        return (-weight_far * in[idx - 2 * stride] +
                weight_near * in[idx - stride] - weight_local * in[idx] +
                weight_near * in[idx + stride] -
                weight_far * in[idx + 2 * stride]) *
               m_one_over_dx2;
    """
    # Calculate the second derivative
    second_derivative = (
        -weight_far * tensor.narrow(dim, 0, tensor.size(dim) - 4)
        + weight_near * tensor.narrow(dim, 1, tensor.size(dim) - 4)
        - weight_local * tensor.narrow(dim, 2, tensor.size(dim) - 4)
        + weight_near * tensor.narrow(dim, 3, tensor.size(dim) - 4)
        - weight_far * tensor.narrow(dim, 4, tensor.size(dim) - 4)
    )

    indices = [x for x in [1, 2, 3] if x != dim]
    for ind in indices:
        second_derivative = second_derivative.narrow(ind, 2, tensor.size(ind) - 4)

    return second_derivative * one_over_dx2


def diff2(tensor: torch.tensor, one_over_dx2: float) -> torch.tensor:
    """
    Compute the matrix of second derivatives for a PyTorch tensor.

    :param tensor: Input tensor of shape [batch, x, y, z, num_var].
    :param one_over_dx2: Inverse of the square of the grid spacing.
    :return: Tensor of shape [batch, x, y, z, num_var, i, j] where i, j are the derivative directions.
    """

    batch_size, x_dim, y_dim, z_dim, num_var = tensor.shape
    num_dims = 3  # Number of spatial dimensions (x, y, z)

    # Initialize an output tensor to store the derivatives
    derivative_tensor = torch.zeros(
        (batch_size, x_dim - 4, y_dim - 4, z_dim - 4, num_var, num_dims, num_dims),
        dtype=tensor.dtype,
        device=tensor.device,
    )

    # Iterate over all combinations of spatial dimensions
    for i in range(num_dims):
        for j in range(num_dims):
            if i == j:
                # Diagonal elements: second derivative along the same dimension
                second_derivative = diff2_multidim(tensor, i, one_over_dx2)
            else:
                # Off-diagonal elements: mixed second derivative
                second_derivative = mixed_diff2_tensor(tensor, i, j, one_over_dx2)

            # Assign the computed derivative to the appropriate slice of the output tensor
            derivative_tensor[..., i, j] = second_derivative

    return derivative_tensor
