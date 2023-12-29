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


def test_Constraints():
    # Define the path to the test data files for variable X
    filenamesX = os.path.dirname(__file__) + "/TestData/Xdata_level0_step*"

    # Number of variables in the data
    num_varsX = 104

    # Read the data in a box format
    dataX = get_box_format(filenamesX, num_varsX)

    # Tolerance for comparison
    tol = 1e-12

    # Compute the differential value
    oneoverdx = 64.0 / 4.0

    # Prepare the data and compute derivatives using TensorDict
    vars = TensorDict(cut_ghosts(dataX), keys_all)
    d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
    h_UU = torch.inverse(vars["h"])
    chris = compute_christoffel(d1["h"], h_UU)

    assert (f"{torch.mean(torch.abs(out['Ham']-vars['Ham']))}") < tol


if __name__ == "__main__":
    test_Constraints()
