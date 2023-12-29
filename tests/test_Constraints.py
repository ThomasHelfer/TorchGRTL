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
from GeneralRelativity.Constraints import constraint_equations
import os
import sys


def test_Constraints():
    """
    Test the Constraints class for correctness.

    This function tests the Constraints class by comparing its output with precomputed data from GRTL code ().
    It reads test data for variable X from a specified directory, computes the necessary
    derivatives, and checks if the output of the Constraints class is within a specified
    tolerance level of the expected values.

    The test passes if the computed values are sufficiently close to the expected values,
    within a specified tolerance level.
    """
    # Define the path to the test data files for variable X
    filenamesX = os.path.dirname(__file__) + "/TestData/Xdata_level0_step*"

    # Number of variables in the data
    num_varsX = 104

    # Read the data in a box format
    dataX = get_box_format(filenamesX, num_varsX)

    # Tolerance for comparison
    tol = 1e-11

    # Compute the differential value
    oneoverdx = 64.0 / 4.0

    # Prepare the data and compute derivatives using TensorDict
    vars = TensorDict(cut_ghosts(dataX), keys_all)
    d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
    d2 = TensorDict(diff2(dataX, oneoverdx**2), keys_all)
    h_UU = torch.inverse(vars["h"])
    chris = compute_christoffel(d1["h"], h_UU)
    out = constraint_equations(vars, d1, d2, h_UU, chris)

    assert (torch.mean(torch.abs(out["Ham"] - vars["Ham"]))) < tol
    assert (torch.mean(torch.abs(out['Mom'][...,0]-vars['Mom1']))) < tol
    assert (torch.mean(torch.abs(out['Mom'][...,1]-vars['Mom2']))) < tol
    assert (torch.mean(torch.abs(out['Mom'][...,2]-vars['Mom3']))) < tol

if __name__ == "__main__":
    test_Constraints()
