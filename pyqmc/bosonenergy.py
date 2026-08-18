"""Mean-field and bosonic kinetic terms for ABVMC / ABCDMC.

For DFT mean field, the local XC potential at walker positions uses libxc
``vrho`` (``eval_xc(...)[1][0]``). That is the full KS potential for LDA.
For GGA (PBE), it is the local spin density potential only; the GGA
``vsigma`` / ∇·(vσ ∇ρ) contribution that appears in PySCF's Vxc matrix is
not included. This matches the ABVMC formulation in Eq. 21 of
doi: 10.1063/5.0155513.
"""

import numpy as np
from pyscf.dft import libxc, numint

SUPPORTED_XC = ("LDA,VWN", "PBE,PBE", "HF")

# PySCF xctype and AO derivative order for each XC string.
XC_KIND = {
    "LDA,VWN": ("LDA", 0),
    "PBE,PBE": ("GGA", 1),
}


def _normalize_xc(xc):
    xc = xc.replace(" ", "")
    if xc not in SUPPORTED_XC:
        raise ValueError(f"Unsupported xc={xc!r}; expected one of {SUPPORTED_XC}")
    return xc


def eval_vrho(mol, dm, xc, coords, spin=1):
    """Local spin-resolved vrho from libxc at ``coords``."""
    xc = _normalize_xc(xc)
    if xc == "HF":
        raise ValueError("HF has no libxc vrho")

    xctype, deriv = XC_KIND[xc]
    ao = numint.eval_ao(mol, coords, deriv=deriv)
    rho_up = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
    rho_dn = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
    vrho = np.asarray(libxc.eval_xc(xc, (rho_up, rho_dn), spin=spin)[1][0])
    if vrho.ndim == 1:
        vrho = np.stack([vrho, vrho], axis=1)
    return vrho


def get_vxc(configs, mol, dm, nelec, xc):
    """Sum libxc vrho over electrons for each walker configuration."""
    nconf, nelec_cfg, _ = configs.configs.shape
    nup = nelec[0]
    if nelec_cfg != sum(nelec):
        raise ValueError("configs electron count inconsistent with mf_inputs['nelec']")

    coords = configs.configs.reshape(-1, 3)
    vrho = eval_vrho(mol, dm, xc, coords, spin=1)
    vrho = vrho.reshape(nconf, nelec_cfg, 2)

    spin_idx = np.array([int(e >= nup) for e in range(nelec_cfg)])
    return np.sum([vrho[:, i, spin_idx[i]] for i in range(nelec_cfg)], axis=0)


def dft_energy(mf_inputs, configs):
    """
    Returns the KS related terms in Eq. 21 in doi: 10.1063/5.0155513.

    Returns:
        v_mf: V_H + V_XC summed over electrons (per walker)
        ecorr: sum of occupied KS eigenvalues (E_0^MF)
        saved_results: dict with vj, vxc when applicable
    """
    nconf, nelec, _ = configs.configs.shape
    xc = _normalize_xc(mf_inputs["xc"])
    nup_dn = mf_inputs["nelec"]
    mo_energy = mf_inputs["mo_energy"]
    mo_occ = mf_inputs["mo_occ"]
    mol = mf_inputs["mol"]
    dm = mf_inputs["dm"]

    def get_vj(configs):
        dm_total = dm[0] + dm[1]
        r = configs.configs.reshape(-1, 3)
        vj_all = np.einsum("pij,ij->p", mol.intor("int1e_grids", grids=r), dm_total)
        return vj_all.reshape(nconf, nelec).sum(axis=1)

    if xc != "HF":
        vj = get_vj(configs)
        vxc = get_vxc(configs, mol, dm, nup_dn, xc)
        ecorr = np.sum(mo_energy * mo_occ)
        v_mf = vj + vxc
        saved_results = {"vj": vj, "vxc": vxc}
    else:
        v_mf = np.zeros(nconf)
        ecorr = np.sum(mo_energy * mo_occ)
        V_eff_ao = mf_inputs["veff"]
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            ao_value = numint.eval_ao(mol, configs.configs[:, e, :])
            v_mf = np.einsum("gp, pq, gq -> g", ao_value, V_eff_ao[s], ao_value)
        saved_results = {}

    return v_mf, ecorr, saved_results


def boson_kinetic(configs, wf):
    """
    Returns the jastrow laplacian (lap_j) and the bosonic drift (drift_b) terms
    in Eq. 21 in doi: 10.1063/5.0155513.
    """
    nconf, nelec, _ = configs.configs.shape

    has_jastrow = True
    try:
        wave_functions = wf.wf_factors
    except AttributeError:
        has_jastrow = False
        wave_functions = [wf]

    jastrow_wf = None
    boson_wf = None
    from pyqmc import bosonslater
    from pyqmc import jastrowspin

    for wave in wave_functions:
        if isinstance(wave, bosonslater.BosonWF):
            boson_wf = wave
        if isinstance(wave, jastrowspin.JastrowSpin):
            jastrow_wf = wave

    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    if has_jastrow:
        for e in range(nelec):
            grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            lap_j += -0.5 * (lap_je.real + np.sum(grad_je.real**2, axis=0))
            grad_b = boson_wf.gradient(e, configs.electron(e))
            drift_b -= np.einsum("di,di->i", grad_je, grad_b)
            grad = np.sum([grad_je, grad_b], axis=0)
            grad2 += np.sum(np.abs(grad) ** 2, axis=0)
    return lap_j, drift_b, grad2
