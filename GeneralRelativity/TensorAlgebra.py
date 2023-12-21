import numpy as np
from tqdm.auto import tqdm, trange
import glob
import torch


def compute_christoffel(d1_metric: torch.tensor, h_UU: torch.tensor) -> torch.tensor:
    """
    Computes the Christoffel symbols of the first kind (LLL) and second kind (ULL)
    for a given metric and its inverse.

    Parameters:
    d1_metric (np.ndarray): Derivative of the metric tensor, shape [batch, x, y, z, i, j, dx].
    h_UU (np.ndarray): Inverse of the metric tensor, shape [batch, x, y, z, i, j, dx].

    Returns:
    Tuple of np.ndarray: Two arrays representing the Christoffel symbols LLL and ULL, each of shape [batch, x, y, z, i, j, k].
    """

    # Initialize the output arrays
    shape = d1_metric.shape[:-1] + (
        d1_metric.shape[-2],
    )  # shape is [batch, x, y, z, i, j, k]
    LLL = torch.zeros(shape, dtype=d1_metric.dtype)
    ULL = torch.zeros(shape, dtype=d1_metric.dtype)

    # Compute Christoffel symbols of the first kind (LLL)
    #         out.LLL[i][j][k] = 0.5 * (d1_metric[j][i][k] + d1_metric[k][i][j] -
    #                              d1_metric[j][k][i]);
    for i in range(3):
        for j in range(3):
            for k in range(3):
                LLL[..., i, j, k] = 0.5 * (
                    +d1_metric[..., j, i, k]
                    + d1_metric[..., k, i, j]
                    - d1_metric[..., j, k, i]
                )

    # Compute Christoffel symbols of the second kind
    #         FOR1(l) { out.ULL[i][j][k] += h_UU[i][l] * out.LLL[l][j][k]; }
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    ULL[..., i, j, k] += h_UU[..., i, l] * LLL[..., l, j, k]

    return LLL, ULL


def compute_christoffel_fast(
    d1_metric: torch.tensor, h_UU: torch.tensor
) -> torch.tensor:
    """
    Computes the Christoffel symbols of the first kind (LLL) and second kind (ULL)
    for a given metric and its inverse using PyTorch. This version uses only torch function and is a bit faster, but less readable.

    Parameters:
    d1_metric (torch.Tensor): Derivative of the metric tensor, shape [batch, x, y, z, i, j, dx].
    h_UU (torch.Tensor): Inverse of the metric tensor, shape [batch, x, y, z, i, j].

    Returns:
    Tuple of torch.Tensor: Two tensors representing the Christoffel symbols LLL and ULL,
                            each of shape [batch, x, y, z, i, j, dx].
    """

    # Initialize the output tensors
    # batch, x, y, z, i, j, dx = d1_metric.shape
    LLL = torch.zeros_like(d1_metric)
    ULL = torch.zeros_like(LLL)

    # Compute Christoffel symbols of the first kind (LLL)
    # Adjusting indices and dimensions for proper computation
    #  out.LLL[i][j][k] = 0.5 * (d1_metric[j][i][k] + d1_metric[k][i][j] - d1_metric[j][k][i]);
    test = (d1_metric).clone()
    LLL = 0.5 * (
        test.permute(0, 1, 2, 3, 4, 5, 6)
        + test.permute(0, 1, 2, 3, 4, 6, 5)
        - d1_metric.permute(0, 1, 2, 3, 6, 5, 4)
    )

    # Compute Christoffel symbols of the second kind (ULL)
    # Corrected Einstein summation for tensor contraction
    # Note: 'ijklmn->ijklm' aligns the dimensions correctly
    # Compute Christoffel symbols of the second kind
    #         FOR1(l) { out.ULL[i][j][k] += h_UU[i][l] * out.LLL[l][j][k]; }
    ULL = torch.einsum("bxzyil,bxzyijk->bxzyijk", h_UU, LLL)

    return LLL, ULL
