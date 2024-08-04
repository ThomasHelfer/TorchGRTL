import torch
import torch.nn as nn

from GeneralRelativity.Utils import (
    TensorDict,
    cut_ghosts,
    keys,
)
from GeneralRelativity.FourthOrderDerivatives import diff1, diff2
from pyinterpx.Interpolation import *
from GeneralRelativity.TensorAlgebra import (
    compute_christoffel,
)
from GeneralRelativity.CCZ4Geometry import compute_ricci
from GeneralRelativity.Constraints import constraint_equations


#  Magical loss coming from General Relativity
class Hamiltonian_loss:
    def __init__(self, oneoverdx: float):
        self.oneoverdx = oneoverdx

    def __call__(
        self, output: torch.tensor, dummy: torch.tensor = torch.tensor([])
    ) -> torch.tensor:
        # For learning we need shape (batch,channel,x,y,z), however TorchGRTL works with (batch,x,y,z,channel), thus the permute
        output = output.permute(0, 2, 3, 4, 1)

        # cutting ghosts off, otherwise you will run into problems later
        dataXcut = cut_ghosts(output)

        # creating dict with values
        vars = TensorDict(dataXcut, keys)
        # creating dict with derivatives
        d1 = TensorDict(diff1(output, self.oneoverdx), keys)
        d2 = TensorDict(diff2(output, self.oneoverdx**2), keys)
        # calculating variables required for constraints
        h_UU = torch.inverse(vars["h"])
        chris = compute_christoffel(d1["h"], h_UU)
        # Computing Constraints
        out = constraint_equations(vars, d1, d2, h_UU, chris)
        loss = torch.mean(out["Ham"] * out["Ham"])
        return loss


class Hamiltonian_and_momentum_loss:
    def __init__(self, oneoverdx: float):
        self.oneoverdx = oneoverdx

    def __call__(
        self, output: torch.tensor, dummy: torch.tensor = torch.tensor([])
    ) -> torch.tensor:
        # For learning we need shape (batch,channel,x,y,z), however TorchGRTL works with (batch,x,y,z,channel), thus the permute
        output = output.permute(0, 2, 3, 4, 1)

        # cutting ghosts off, otherwise you will run into problems later
        dataXcut = cut_ghosts(output)

        # creating dict with values
        vars = TensorDict(dataXcut, keys)
        # creating dict with derivatives
        d1 = TensorDict(diff1(output, self.oneoverdx), keys)
        d2 = TensorDict(diff2(output, self.oneoverdx**2), keys)
        # calculating variables required for constraints
        h_UU = torch.inverse(vars["h"])
        chris = compute_christoffel(d1["h"], h_UU)
        # Computing Constraints
        out = constraint_equations(vars, d1, d2, h_UU, chris)
        loss = torch.mean(out["Ham"] * out["Ham"]) + torch.mean(out["Mom"] * out["Mom"])
        return loss


class Hamiltonian_and_momentum_loss:
    def __init__(self, oneoverdx: float):
        self.oneoverdx = oneoverdx

    def __call__(
        self, output: torch.tensor, interpolated: torch.tensor = torch.tensor([])
    ) -> torch.tensor:
        # For learning we need shape (batch,channel,x,y,z), however TorchGRTL works with (batch,x,y,z,channel), thus the permute
        output = output.permute(0, 2, 3, 4, 1)

        # cutting ghosts off, otherwise you will run into problems later
        dataXcut = cut_ghosts(output)

        # creating dict with values
        vars = TensorDict(dataXcut, keys)
        # creating dict with derivatives
        d1 = TensorDict(diff1(output, self.oneoverdx), keys)
        d2 = TensorDict(diff2(output, self.oneoverdx**2), keys)
        # calculating variables required for constraints
        h_UU = torch.inverse(vars["h"])
        chris = compute_christoffel(d1["h"], h_UU)
        # Computing Constraints
        out = constraint_equations(vars, d1, d2, h_UU, chris)
        loss = torch.mean(out["Ham"] * out["Ham"]) + torch.mean(out["Mom"] * out["Mom"])
        return loss


class Hamiltonian_and_momentum_loss_boundary_condition:
    def __init__(self, oneoverdx: float):
        self.oneoverdx = oneoverdx
        self.criterion = nn.MSELoss()

    def __call__(
        self, output: torch.tensor, target: torch.tensor = torch.tensor([])
    ) -> torch.tensor:
        # For learning we need shape (batch,channel,x,y,z), however TorchGRTL works with (batch,x,y,z,channel), thus the permute
        output = output.permute(0, 2, 3, 4, 1)
        # Get the middle index
        D = output.size(2)
        middle_idx = D // 2

        # cutting ghosts off, otherwise you will run into problems later
        dataXcut = cut_ghosts(output)

        # creating dict with values
        vars = TensorDict(dataXcut, keys)
        # creating dict with derivatives
        d1 = TensorDict(diff1(output, self.oneoverdx), keys)
        d2 = TensorDict(diff2(output, self.oneoverdx**2), keys)
        # calculating variables required for constraints
        h_UU = torch.inverse(vars["h"])
        chris = compute_christoffel(d1["h"], h_UU)

        # Extract the middle pixel values
        if target is not None:
            input_middle_pixel = output[:, middle_idx, middle_idx, middle_idx, :]
            target_middle_pixel = target[:, :, middle_idx, middle_idx, middle_idx]
        # Computing Constraints
        out = constraint_equations(vars, d1, d2, h_UU, chris)
        loss = torch.mean(out["Ham"] * out["Ham"]) + torch.mean(out["Mom"] * out["Mom"])
        if target is not None:
            loss += 10 * self.criterion(input_middle_pixel, target_middle_pixel)
        return loss
