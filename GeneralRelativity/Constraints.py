import torch
from typing import Dict, List
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2, FOR3
from GeneralRelativity.TensorAlgebra import compute_trace, raise_all
from GeneralRelativity.CCZ4Geometry import compute_ricci


def constraint_equations(
    vars: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
    h_UU: torch.Tensor,
    chris: Dict[str, torch.Tensor],
    GR_SPACEDIM: int = 3,
    cosmological_constant: float = 0,
) -> Dict[str, torch.Tensor]:
    """
    Calculates the constraint equations in the context of general relativity.

    Args:
        vars (Dict[str, torch.Tensor]): Dictionary containing variables.
            Expected to have keys like 'A', 'K', 'chi', etc.
        d1 (Dict[str, torch.Tensor]): Dictionary containing first derivatives.
        d2 (Dict[str, torch.Tensor]): Dictionary containing second derivatives.
        h_UU (torch.Tensor): The inverse metric tensor.
        chris (Dict[str, torch.Tensor]): Dictionary containing Christoffel symbols,
            expected to have keys like 'ULL'.
        GR_SPACEDIM (int, optional): The spatial dimension of the general relativity
            calculations. Defaults to 4.
        cosmological_constant (float, optional): The cosmological constant. Defaults to 0.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the computed constraint equations.
            Keys include 'Ham', 'Ham_abs_terms', 'Mom', 'Mom_abs_terms'.
    """
    dtype = vars["chi"].dtype

    out = {
        "Ham": torch.zeros_like(vars["chi"], dtype=dtype),
        "Ham_abs_terms": torch.zeros_like(vars["chi"], dtype=dtype),
        "Mom": torch.zeros_like(vars["shift"], dtype=dtype),
        "Mom_abs_terms": torch.zeros_like(vars["shift"], dtype=dtype),
    }

    # auto ricci = CCZ4Geometry::compute_ricci(vars, d1, d2, h_UU, chris);
    # auto A_UU = TensorAlgebra::raise_all(vars.A, h_UU);
    # data_t tr_A2 = TensorAlgebra::compute_trace(vars.A, A_UU);
    ricci = compute_ricci(vars, d1, d2, h_UU, chris, GR_SPACEDIM)
    A_UU = raise_all(vars["A"], h_UU)
    tr_A2 = compute_trace(vars["A"], A_UU)
    # out.Ham = ricci.scalar +
    #     (GR_SPACEDIM - 1.) * vars.K * vars.K / GR_SPACEDIM - tr_A2;
    # out.Ham -= 2 * m_cosmological_constant;
    out["Ham"] = (
        ricci["scalar"] + (GR_SPACEDIM - 1) * vars["K"] ** 2 / GR_SPACEDIM - tr_A2
    )
    out["Ham"] -= 2 * cosmological_constant

    #    out.Ham_abs_terms =
    #        abs(ricci.scalar) + abs(tr_A2) +
    #        abs((GR_SPACEDIM - 1.) * vars.K * vars.K / GR_SPACEDIM);
    #    out.Ham_abs_terms += 2.0 * abs(m_cosmological_constant);
    out["Ham_abs_terms"] = (
        torch.abs(ricci["scalar"])
        + torch.abs(tr_A2)
        + torch.abs((GR_SPACEDIM - 1) * vars["K"] ** 2 / GR_SPACEDIM)
    )
    out["Ham_abs_terms"] += 2.0 * abs(cosmological_constant)

    #    Tensor<2, data_t> covd_A[CH_SPACEDIM];
    #    FOR3(i, j, k)
    #    {
    covd_A = torch.zeros_like(d1["A"])
    for i, j, k in FOR3():
        #  covd_A[i][j][k] = d1.A[j][k][i];
        #    FOR1(l)
        #    {
        covd_A[..., i, j, k] = d1["A"][..., j, k, i]
        for l in FOR1():
            #  covd_A[i][j][k] += -chris.ULL[l][i][j] * vars.A[l][k] -
            #                       chris.ULL[l][i][k] * vars.A[l][j];
            covd_A[..., i, j, k] -= (
                chris["ULL"][..., l, i, j] * vars["A"][..., l, k]
                - chris["ULL"][..., l, i, k] * vars["A"][..., l, j]
            )

    for i in FOR1():
        #   out.Mom[i] = -(GR_SPACEDIM - 1.) * d1.K[i] / GR_SPACEDIM;
        #   out.Mom_abs_terms[i] = abs(out.Mom[i]);
        out["Mom"][..., i] = -(GR_SPACEDIM - 1.0) * d1["K"][..., i] / GR_SPACEDIM
        out["Mom_abs_terms"][..., i] = torch.abs(out["Mom"][..., i])
    #   Tensor<1, data_t> covd_A_term = 0.0;
    #   Tensor<1, data_t> d1_chi_term = 0.0;
    #   const data_t chi_regularised = simd_max(1e-6, vars.chi);
    covd_A_term = torch.zeros_like(d1["chi"], dtype=dtype)
    d1_chi_term = torch.zeros_like(d1["chi"], dtype=dtype)
    chi_regularised = torch.maximum(torch.tensor(1e-6), vars["chi"])
    for i, j, k in FOR3():
        #    covd_A_term[i] += h_UU[j][k] * covd_A[k][j][i];
        #    d1_chi_term[i] += -GR_SPACEDIM * h_UU[j][k] * vars.A[i][j] *
        #                      d1.chi[k] / (2 * chi_regularised);
        covd_A_term[..., i] += h_UU[..., j, k] * covd_A[..., k, j, i]
        d1_chi_term[..., i] += (
            -GR_SPACEDIM
            * h_UU[..., j, k]
            * vars["A"][..., i, j]
            * d1["chi"][..., k]
            / (2 * chi_regularised)
        )

    for i in FOR1():
        #  out.Mom[i] += covd_A_term[i] + d1_chi_term[i];
        #    out.Mom_abs_terms[i] += abs(covd_A_term[i]) + abs(d1_chi_term[i]);
        out["Mom"][..., i] += covd_A_term[..., i] + d1_chi_term[..., i]
        out["Mom_abs_terms"][..., :, i] += torch.abs(covd_A_term[..., i]) + torch.abs(
            d1_chi_term[..., i]
        )

    return out
