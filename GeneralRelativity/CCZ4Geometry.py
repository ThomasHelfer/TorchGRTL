from typing import Dict
from GeneralRelativity.DimensionDefinitions import FOR1, FOR2
from GeneralRelativity.TensorAlgebra import compute_trace
import torch


def compute_ricci_Z(
    vars: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
    h_UU: torch.Tensor,
    chris: Dict[str, torch.Tensor],
    Z_over_chi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute the Ricci tensor Z using the provided variables, derivatives, and Christoffel symbols.

    Args:
        vars (dict): Dictionary of tensor variables.
        d1 (dict): Dictionary of first derivatives of tensor variables.
        d2 (dict): Dictionary of second derivatives of tensor variables.
        h_UU (torch.Tensor): Inverse metric tensor.
        chris (dict): Dictionary containing ULL and LLL Christoffel symbols.
        Z_over_chi (torch.Tensor): Tensor Z divided by chi.

    Returns:
        dict: Dictionary containing the components of the Ricci tensor Z.
    """
    out = {"LL": torch.zeros_like(vars["h"]), "scalar": 0}

    GR_SPACEDIM = 4
    boxtildechi = 0

    covdtilde2chi = torch.zeros_like(vars["h"])
    #    FOR2(k, l)
    #    {
    #        covdtilde2chi[k][l] = d2.chi[k][l];
    #        FOR1(m) { covdtilde2chi[k][l] -= chris.ULL[m][k][l] * d1.chi[m]; }
    #    }
    for k, l in FOR2():
        covdtilde2chi[..., k, l] = d2["chi"][..., k, l]
        for m in FOR1():
            covdtilde2chi[..., k, l] -= chris["ULL"][..., m, k, l] * d1["chi"][..., m]
    #    FOR2(k, l) { boxtildechi += h_UU[k][l] * covdtilde2chi[k][l]; }
    for k, l in FOR2():
        boxtildechi += h_UU[..., k, l] * covdtilde2chi[..., k, l]
    # data_t dchi_dot_dchi = 0;
    #    {
    #        FOR2(m, n) { dchi_dot_dchi += h_UU[m][n] * d1.chi[m] * d1.chi[n]; }
    #    }
    dchi_dot_dchi = torch.zeros_like(vars["chi"][...])
    for m, n in FOR2():
        dchi_dot_dchi += h_UU[..., m, n] * d1["chi"][..., m] * d1["chi"][..., n]

    # FOR2(i, j)
    # {
    for i, j in FOR2():
        #    data_t ricci_tilde = 0;
        #    FOR1(k)
        #    {
        ricci_tilde = 0
        for k in FOR1():
            #    ricci_tilde += 0.5 * (vars.h[k][i] * d1.Gamma[k][j] +
            #                          vars.h[k][j] * d1.Gamma[k][i]);
            #    ricci_tilde += 0.5 * (vars.Gamma[k] - 2 * Z_over_chi[k]) *
            #                   (chris.LLL[i][j][k] + chris.LLL[j][i][k]);
            ricci_tilde += 0.5 * (
                vars["h"][..., k, i] * d1["Gamma"][..., k, j]
                + vars["h"][..., k, j] * d1["Gamma"][..., k, i]
            )
            ricci_tilde += (
                0.5
                * (vars["Gamma"][..., k] - 2 * Z_over_chi[..., k])
                * (chris["LLL"][..., i, j, k] + chris["LLL"][..., j, i, k])
            )
            # FOR1(l)
            # {
            for l in FOR1():
                #  ricci_tilde -= 0.5 * h_UU[k][l] * d2.h[i][j][k][l];
                ricci_tilde -= 0.5 * h_UU[..., k, l] * d2["h"][..., i, j, k, l]
                #  FOR1(m)
                #   {
                for m in FOR1():
                    # ricci_tilde +=
                    #        h_UU[l][m] *
                    #        (chris.ULL[k][l][i] * chris.LLL[j][k][m] +
                    #         chris.ULL[k][l][j] * chris.LLL[i][k][m] +
                    #         chris.ULL[k][i][m] * chris.LLL[k][l][j]);
                    ricci_tilde += h_UU[..., l, m] * (
                        chris["ULL"][..., k, l, i] * chris["LLL"][..., j, k, m]
                        + chris["ULL"][..., k, l, j] * chris["LLL"][..., i, k, m]
                        + chris["ULL"][..., k, i, m] * chris["LLL"][..., k, l, j]
                    )

        # data_t ricci_chi =
        #    0.5 * ((GR_SPACEDIM - 2) * covdtilde2chi[i][j] +
        #    vars.h[i][j] * boxtildechi -
        #    ((GR_SPACEDIM - 2) * d1.chi[i] * d1.chi[j] +
        #    GR_SPACEDIM * vars.h[i][j] * dchi_dot_dchi) /
        #    (2 * vars.chi));
        ricci_chi = 0.5 * (
            (GR_SPACEDIM - 2) * covdtilde2chi[..., i, j]
            + vars["h"][..., i, j] * boxtildechi
            - (
                (GR_SPACEDIM - 2) * d1["chi"][..., i] * d1["chi"][..., j]
                + GR_SPACEDIM * vars["h"][..., i, j] * dchi_dot_dchi
            )
            / (2 * vars["chi"])
        )
        # data_t z_terms = 0;
        # FOR1(k)
        # {
        z_terms = 0
        for k in FOR1():
            # z_terms +=
            # Z_over_chi[k] *
            # (vars.h[i][k] * d1.chi[j] + vars.h[j][k] * d1.chi[i] -
            # vars.h[i][j] * d1.chi[k] + d1.h[i][j][k] * vars.chi);
            z_terms += Z_over_chi[..., k] * (
                vars["h"][..., i, k] * d1["chi"][..., j]
                + vars["h"][..., j, k] * d1["chi"][..., i]
                - vars["h"][..., i, j] * d1["chi"][..., k]
                + d1["h"][..., i, j, k] * vars["chi"]
            )
        #     out.LL[i][j] =
        #        (ricci_chi + vars.chi * ricci_tilde + z_terms) / vars.chi;
        out["LL"][..., i, j] = (ricci_chi + vars["chi"] * ricci_tilde + z_terms) / vars[
            "chi"
        ]
    # out.scalar = vars.chi * TensorAlgebra::compute_trace(out.LL, h_UU);
    out["scalar"] = vars["chi"] * compute_trace(out["LL"], h_UU)

    return out


def compute_ricci(
    vars: Dict[str, torch.Tensor],
    d1: Dict[str, torch.Tensor],
    d2: Dict[str, torch.Tensor],
    h_UU: torch.Tensor,
    chris: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute the Ricci tensor using the provided variables, derivatives, and Christoffel symbols.

    Args:
        vars (dict): Dictionary of tensor variables.
        d1 (dict): Dictionary of first derivatives of tensor variables.
        d2 (dict): Dictionary of second derivatives of tensor variables.
        h_UU (torch.Tensor): Inverse metric tensor.
        chris (dict): Dictionary containing Christoffel symbols.

    Returns:
        dict: Dictionary containing the components of the Ricci tensor.
    """
    Z0 = torch.zeros_like(torch.zeros_like(d1["chi"]))
    return compute_ricci_Z(vars, d1, d2, h_UU, chris, Z0)
