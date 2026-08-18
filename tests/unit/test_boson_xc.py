"""Validate ABVMC XC potentials (LDA and PBE local vrho)."""

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import libxc, numint

from pyqmc.bosonenergy import dft_energy, eval_vrho, get_vxc
from pyqmc.mc import initial_guess
from pyqmc.xc_analytic import analytic_lda, analytic_pbe, gga_spin_density, pbe_vx, pbe_vc


def _run_uks(atom, spin, charge, xc, basis="cc-pvdz"):
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis, unit="angstrom", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.make_rdm1()


def _mf_inputs(mol, mf, dm, xc):
    ao = numint.eval_ao(mol, mf.grids.coords, deriv=0)
    rho = np.einsum("pi,ij,pj->p", ao, dm[0] + dm[1], ao)
    return dict(
        xc=xc,
        nelec=mf.nelec,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        mol=mol,
        dm=dm,
        grids=mf.grids,
        rho=rho,
    )


def _max_rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
def test_dft_energy_vxc_matches_libxc(xc):
    mol, mf, dm = _run_uks("He 0 0 0", spin=0, charge=0, xc=xc)
    configs = initial_guess(mol, 8)
    mf_inputs = _mf_inputs(mol, mf, dm, xc)

    vxc_code = dft_energy(mf_inputs, configs)[2]["vxc"]

    nconf, nelec, _ = configs.configs.shape
    vrho = eval_vrho(mol, dm, xc, configs.configs.reshape(-1, 3)).reshape(nconf, nelec, 2)
    spin = np.array([int(e >= mf.nelec[0]) for e in range(nelec)])
    vxc_ref = np.sum([vrho[:, i, spin[i]] for i in range(nelec)], axis=0)

    assert _max_rel(vxc_code, vxc_ref) < 1e-12


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
def test_get_vxc_matches_eval_vrho(xc):
    mol, mf, dm = _run_uks("Li 0 0 0", spin=1, charge=0, xc=xc)
    configs = initial_guess(mol, 6)
    vxc = get_vxc(configs, mol, dm, mf.nelec, xc)

    nconf, nelec, _ = configs.configs.shape
    vrho = eval_vrho(mol, dm, xc, configs.configs.reshape(-1, 3)).reshape(nconf, nelec, 2)
    spin = np.array([int(e >= mf.nelec[0]) for e in range(nelec)])
    vxc_ref = np.sum([vrho[:, i, spin[i]] for i in range(nelec)], axis=0)

    assert _max_rel(vxc, vxc_ref) < 1e-12


def test_analytic_lda_matches_libxc():
    mol, mf, dm = _run_uks("He 0 0 0", spin=0, charge=0, xc="LDA,VWN")
    coords = mf.grids.coords[:: max(1, len(mf.grids.coords) // 100)]
    _, _, a_a, a_b = analytic_lda(mol, dm, coords)
    vrho = eval_vrho(mol, dm, "LDA,VWN", coords)
    assert _max_rel(a_a, vrho[:, 0]) < 1e-5
    assert _max_rel(a_b, vrho[:, 1]) < 1e-5


def test_analytic_pbe_matches_libxc():
    mol, mf, dm = _run_uks("He 0 0 0", spin=0, charge=0, xc="PBE,PBE")
    coords = mf.grids.coords[:: max(1, len(mf.grids.coords) // 100)]
    ra, rb, ga, gb, gtot = gga_spin_density(mol, dm, coords)
    vx_a, vx_b = pbe_vx(ra, rb, ga, gb)
    vc_a, vc_b = pbe_vc(ra, rb, gtot)

    ao = numint.eval_ao(mol, coords, deriv=1)
    ru = numint.eval_rho(mol, ao, dm[0], xctype="GGA")
    rd = numint.eval_rho(mol, ao, dm[1], xctype="GGA")
    vx = np.asarray(libxc.eval_xc("GGA_X_PBE", (ru, rd), spin=1)[1][0])
    vc = np.asarray(libxc.eval_xc("GGA_C_PBE", (ru, rd), spin=1)[1][0])

    assert _max_rel(vx_a, vx[:, 0]) < 1e-12
    assert _max_rel(vx_b, vx[:, 1]) < 1e-12
    assert _max_rel(vc_a, vc[:, 0]) < 1e-5
    assert _max_rel(vc_b, vc[:, 1]) < 1e-5

    _, _, ap_a, ap_b = analytic_pbe(mol, dm, coords)
    vrho = eval_vrho(mol, dm, "PBE,PBE", coords)
    assert _max_rel(ap_a, vrho[:, 0]) < 1e-5
    assert _max_rel(ap_b, vrho[:, 1]) < 1e-5
