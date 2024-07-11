import torch

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


from SuperResolution.models import SuperResolution3DNet, check_performance


#  Magical loss coming from General Relativity
class Hamiltonian_loss:
    def __init__(self, oneoverdx: float):
        self.oneoverdx = oneoverdx

    def __call__(self, output: torch.tensor) -> torch.tensor:
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
        hamloss = torch.mean(out["Ham"] * out["Ham"])
        loss = hamloss
        return loss
