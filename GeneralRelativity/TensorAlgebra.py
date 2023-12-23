import numpy as np
from tqdm.auto import tqdm, trange
import glob
import torch
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2, FOR3, FOR4


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

    chris = {
        "LLL": torch.zeros(d1_metric.shape, dtype=d1_metric.dtype),
        "ULL": torch.zeros(d1_metric.shape, dtype=d1_metric.dtype),
    }

    # Compute Christoffel symbols of the first kind (LLL)
    #         out.LLL[i][j][k] = 0.5 * (d1_metric[j][i][k] + d1_metric[k][i][j] -
    #                              d1_metric[j][k][i]);
    for i, j, k in FOR3():
        chris["LLL"][..., i, j, k] = 0.5 * (
            +d1_metric[..., j, i, k] + d1_metric[..., k, i, j] - d1_metric[..., j, k, i]
        )

    # Compute Christoffel symbols of the second kind
    #         FOR1(l) { out.ULL[i][j][k] += h_UU[i][l] * out.LLL[l][j][k]; }
    for i, j, k, l in FOR4():
        chris["ULL"][..., i, j, k] += h_UU[..., i, l] * chris["LLL"][..., l, j, k]

    return chris


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
    chris = {
        "LLL": torch.zeros(d1_metric.shape, dtype=d1_metric.dtype),
        "ULL": torch.zeros(d1_metric.shape, dtype=d1_metric.dtype),
    }

    # Compute Christoffel symbols of the first kind (LLL)
    # Adjusting indices and dimensions for proper computation
    #  out.LLL[i][j][k] = 0.5 * (d1_metric[j][i][k] + d1_metric[k][i][j] - d1_metric[j][k][i]);
    test = (d1_metric).clone()
    chris["LLL"] = 0.5 * (
        test.permute(0, 1, 2, 3, 4, 5, 6)
        + test.permute(0, 1, 2, 3, 4, 6, 5)
        - d1_metric.permute(0, 1, 2, 3, 6, 5, 4)
    )

    # Compute Christoffel symbols of the second kind (ULL)
    # Corrected Einstein summation for tensor contraction
    # Note: 'ijklmn->ijklm' aligns the dimensions correctly
    # Compute Christoffel symbols of the second kind
    #         FOR1(l) { out.ULL[i][j][k] += h_UU[i][l] * out.LLL[l][j][k]; }
    chris["ULL"] = torch.einsum("bxzyil,bxzyijk->bxzyijk", h_UU, chris["LLL"])

    return chris


def compute_trace(tensor_LL, inverse_metric):
    """
    Computes the trace of a 2-Tensor with lower indices given an inverse metric.

    Args:
        tensor_LL (torch.Tensor): The 2-Tensor with lower indices.
        inverse_metric (torch.Tensor): The inverse metric tensor.

    Returns:
        float: The trace of the tensor.
    """
    trace = 0.0
    for i, j in FOR2():
        trace += inverse_metric[..., i, j] * tensor_LL[..., i, j]
    return trace


def raise_all(tensor_L: torch.Tensor, inverse_metric: torch.Tensor) -> torch.Tensor:
    """
    Raises the index of a covector (tensor with a lower index) using the inverse metric.

    Args:
        tensor_L (torch.Tensor): The covector (tensor with a lower index) to be raised.
        inverse_metric (torch.Tensor): The inverse metric tensor.

    Returns:
        torch.Tensor: The resulting tensor with the index raised.
    """
    tensor_U = torch.zeros_like(tensor_L)
    for i, j in FOR2():
        tensor_U[..., i] += inverse_metric[..., i, j] * tensor_L[..., j]
    return tensor_U
