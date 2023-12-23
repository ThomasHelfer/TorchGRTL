import torch
from typing import Dict
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2
from GeneralRelativity.TensorAlgebra import compute_trace
from GeneralRelativity.CCZ4Geometry import compute_ricci


def constraint_equations(
    vars: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
    h_UU: torch.Tensor,
    chris: Dict[str, torch.Tensor],
    m_cosmological_constant: float = 0.0,
    GR_SPACEDIM: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Compute the Hamiltonian and momentum constraints.

    Args:
        vars (dict): Dictionary of tensor variables.
        d1 (dict): Dictionary of first derivatives of tensor variables.
        d2 (dict): Dictionary of second derivatives of tensor variables.
        h_UU (torch.Tensor): Inverse metric tensor.
        chris (dict): Dictionary containing Christoffel symbols.
        m_cosmological_constant (float): Cosmological constant (default is 0).
        GR_SPACEDIM (int): Spatial dimension (default is 4).

    Returns:
        dict: Dictionary containing the Hamiltonian and momentum constraints.
    """

    out = {}
    ricci = compute_ricci(vars, d1, d2, h_UU, chris)

    A_UU = raise_all(vars["A"], h_UU)
    tr_A2 = compute_trace(vars["A"], A_UU)

    out["Ham"] = (
        ricci["scalar"]
        + (GR_SPACEDIM - 1.0) * vars["K"] * vars["K"] / GR_SPACEDIM
        - tr_A2
    )
    out["Ham"] -= 2 * m_cosmological_constant

    out["Ham_abs_terms"] = (
        torch.abs(ricci["scalar"])
        + torch.abs(tr_A2)
        + torch.abs((GR_SPACEDIM - 1.0) * vars["K"] * vars["K"] / GR_SPACEDIM)
    )
    out["Ham_abs_terms"] += 2.0 * torch.abs(m_cosmological_constant)

    if "Mom" in vars or "Mom_abs_terms" in vars:
        covd_A = torch.zeros_like(vars["A"])
        for i, j, k in FOR3():
            covd_A[..., i, j, k] = d1["A"][..., j, k, i]
            for l in FOR1():
                covd_A[..., i, j, k] -= (
                    chris["ULL"][..., l, i, j] * vars["A"][..., l, k]
                    + chris["ULL"][..., l, i, k] * vars["A"][..., l, j]
                )

        out["Mom"] = torch.zeros((GR_SPACEDIM,))
        out["Mom_abs_terms"] = torch.zeros((GR_SPACEDIM,))
        for i in FOR1():
            out["Mom"][..., i] = -(GR_SPACEDIM - 1.0) * d1["K"][..., i] / GR_SPACEDIM
            out["Mom_abs_terms"][..., i] = torch.abs(out["Mom"][..., i])

        covd_A_term = torch.zeros((GR_SPACEDIM,))
        d1_chi_term = torch.zeros((GR_SPACEDIM,))
        chi_regularised = torch.clamp(vars["chi"], min=1e-6)
        for i, j, k in FOR3():
            covd_A_term[..., i] += h_UU[..., j, k] * covd_A[..., k, j, i]
            d1_chi_term[..., i] -= (
                GR_SPACEDIM
                * h_UU[..., j, k]
                * vars["A"][..., i, j]
                * d1["chi"][..., k]
                / (2 * chi_regularised)
            )

        for i in FOR1():
            out["Mom"][..., i] += covd_A_term[..., i] + d1_chi_term[..., i]
            out["Mom_abs_terms"][..., i] += torch.abs(covd_A_term[..., i]) + torch.abs(
                d1_chi_term[..., i]
            )

    return out
