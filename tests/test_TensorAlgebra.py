import torch
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from GeneralRelativity.Utils import (
    get_box_format,
    TensorDict,
    cut_ghosts,
    keys,
    keys_all,
)
from GeneralRelativity.TensorAlgebra import (
    compute_christoffel,
    compute_christoffel_fast,
)
import os
import sys


def test_chris():
    """
    Test function to validate the computation of Christoffel symbols.

    This function reads tensor data from files, computes Christoffel symbols using two different
    implementations (compute_christoffel and compute_christoffel_fast), and then compares the results
    to ensure they are consistent with each other. It also checks the symmetry property of the Christoffel
    symbols. Assertions are used to ensure that the differences are within a specified tolerance.
    """
    # Define the path to the test data files for variable X
    filenamesX = os.path.dirname(__file__) + "/TestData/Xdata_level0_step*"

    # Number of variables in the data
    num_varsX = 104

    # Read the data in a box format
    dataX = get_box_format(filenamesX, num_varsX)

    # Tolerance for comparison
    tol = 1e-10

    # Compute the differential value
    oneoverdx = 64.0 / 4.0

    # Prepare the data and compute derivatives using TensorDict
    vars = TensorDict(cut_ghosts(dataX), keys_all)
    d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
    h_UU = torch.inverse(vars["h"])
    chris = compute_christoffel(d1["h"], h_UU)
    chris_2nd_implementation = compute_christoffel_fast(d1["h"], h_UU)

    # Compare two versions of Christoffel symbols
    assert torch.mean(torch.abs(chris["LLL"] - chris_2nd_implementation["LLL"])) < tol
    assert torch.mean(torch.abs(chris["ULL"] - chris_2nd_implementation["ULL"])) < tol

    # Check symmetry of Christoffel symbols
    for i in range(3):
        for j in range(i, 3):
            assert (
                torch.mean(
                    torch.abs(chris["ULL"][..., i, j])
                    - torch.abs(chris["ULL"][..., j, i])
                )
                == 0
            )
            assert (
                torch.mean(
                    torch.abs(chris_2nd_implementation["ULL"][..., i, j])
                    - torch.abs(chris_2nd_implementation["ULL"][..., j, i])
                )
                == 0
            )


if __name__ == "__main__":
    test_chris()
