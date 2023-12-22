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


def test_compare_diff_with_reference():
    """
    Test function to compare the calculated first and second derivatives with reference data.

    This function reads test data for a specific variable from a set of files, computes the first
    and second derivatives using predefined functions (diff1, diff2), and then asserts that the
    computed derivatives are within a specified tolerance of the reference derivative data.
    """
    # Define the path to the test data files for variable X
    filenamesX = os.path.dirname(__file__) + "/TestData/Xdata_level0_step*"

    # Number of variables in the data
    num_varsX = 100

    # Read the data in a box format
    dataX = get_box_format(filenamesX, num_varsX)

    # Tolerance for comparison
    tol = 1e-6

    # Compute the differential value
    oneoverdx = 64.0 / 4.0

    # Prepare the data and compute derivatives using TensorDict
    vars = TensorDict(cut_ghosts(dataX), keys_all)
    d1 = TensorDict(diff1(dataX, oneoverdx), keys_all)
    d2 = TensorDict(diff2(dataX, oneoverdx**2), keys_all)

    # Test first derivative against reference data
    assert (torch.mean(torch.abs(vars["dx_chi"] - d1["chi"][..., 0]))).numpy() < tol
    assert (torch.mean(torch.abs(vars["dy_chi"] - d1["chi"][..., 1]))).numpy() < tol
    assert (torch.mean(torch.abs(vars["dz_chi"] - d1["chi"][..., 2]))).numpy() < tol
    # test second derivative against reference data
    assert (
        torch.mean(
            torch.abs(d1["dx_chi"])[..., 0] - torch.abs(d2["chi"])[..., 0, 0]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dx_chi"])[..., 1] - torch.abs(d2["chi"])[..., 0, 1]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dx_chi"])[..., 2] - torch.abs(d2["chi"])[..., 0, 2]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dy_chi"])[..., 0] - torch.abs(d2["chi"])[..., 1, 0]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dy_chi"])[..., 1] - torch.abs(d2["chi"])[..., 1, 1]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dy_chi"])[..., 2] - torch.abs(d2["chi"])[..., 1, 2]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dz_chi"])[..., 0] - torch.abs(d2["chi"])[..., 2, 0]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dz_chi"])[..., 1] - torch.abs(d2["chi"])[..., 2, 1]
        ).numpy()
        < tol
    )
    assert (
        torch.mean(
            torch.abs(d1["dz_chi"])[..., 2] - torch.abs(d2["chi"])[..., 2, 2]
        ).numpy()
        < tol
    )


if __name__ == "__main__":
    test_compare_diff_with_reference()
