import torch
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from GeneralRelativity.Utils import (
    get_box_format,
    TensorDict,
    cut_ghosts,
    keys,
    keys_all,
)
import os
import sys


def test_TensorDict():
    """
    Test function to validate the TensorDict data structure used for handling tensors.

    This function performs several checks:
    1. It verifies the integrity and correctness of the TensorDict structure
    2. It checks the symmetry properties of the 'h' and 'A' tensors and their derivatives, which are
       expected to be symmetric.

    The function assumes the existence of a specific data structure and certain keys within the TensorDict
    (e.g., 'chi', 'h', 'A', 'shift') and uses these to perform the validations. The test data is read from
    files located in a 'TestData' directory relative to the script's location.

    Assertions are used to validate the conditions, with a failure indicating a problem in the TensorDict
    implementation or the data it contains.
    """

    # Define the path to the test data files for variable X
    filenamesX = os.path.dirname(__file__) + "/TestData/Xdata_level0_step*"

    # Number of variables in the data
    num_varsX = 104

    # Read the data in a box format
    dataX = get_box_format(filenamesX, num_varsX)

    # Tolerance for comparison
    tol = 1e-6

    # Compute the differential value
    oneoverdx = 64.0 / 4.0

    # Prepare the data and compute derivatives using TensorDict
    dataXcut = cut_ghosts(dataX)
    vars = TensorDict(dataXcut, keys_all)
    d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
    d2 = TensorDict(diff2(dataX, oneoverdx**2), keys_all)

    # Spot check if Tensordict is using right data
    assert torch.mean(torch.abs(vars["chi"] - dataXcut[..., 0])) == 0

    # Check symmetry of metric
    for i in range(3):
        for j in range(i, 3):
            assert (
                torch.mean(torch.abs(vars["h"][..., i, j] - vars["h"][..., j, i])) == 0
            )
            assert (
                torch.mean(torch.abs(vars["A"][..., i, j] - vars["A"][..., j, i])) == 0
            )
            assert (
                torch.mean(torch.abs(d1["h"][..., i, j, :] - d1["h"][..., j, i, :]))
                == 0
            )
            assert (
                torch.mean(torch.abs(d1["A"][..., i, j, :] - d1["A"][..., j, i, :]))
                == 0
            )
            assert (
                torch.mean(
                    torch.abs(d2["h"][..., i, j, :, :] - d2["h"][..., j, i, :, :])
                )
                == 0
            )
            assert (
                torch.mean(
                    torch.abs(d2["A"][..., i, j, :, :] - d2["A"][..., j, i, :, :])
                )
                == 0
            )
    # Check symmetry of vector
    for i in range(3):
        assert torch.mean(torch.abs(vars["shift"][..., i] - dataXcut[..., 19 + i])) == 0


if __name__ == "__main__":
    test_TensorDict()
