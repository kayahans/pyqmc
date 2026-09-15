"""Parity tests for boson MF AO → ρ → vrho (numba vs pyscf numint)."""

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import numint

pytest.importorskip("numba")

from pyqmc.observables import bosonenergy


def _uks(atom, spin, charge, xc, basis="sto-3g"):
    # sto-3g avoids SP hybrid shells that AtomicOrbitalEvaluator cannot pack yet.
    mol = gto.M(
        atom=atom, spin=spin, charge=charge, basis=basis, unit="angstrom", verbose=0
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.make_rdm1()


def _coords(mol, n=40, seed=0):
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    near = atom_coords[rng.integers(0, len(atom_coords), size=n // 2)]
    near = near + 0.05 * rng.standard_normal(near.shape)
    far = 2.0 * rng.standard_normal((n - near.shape[0], 3))
    return np.vstack([near, far])


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize(
    "system",
    [
        ("He 0 0 0", 0, 0),
        ("Li 0 0 0", 1, 0),
        ("Be 0 0 0", 1, 1),  # Be+ (3 e−)
        ("Ne 0 0 0", 0, 0),
    ],
)
def test_numba_ao_rho_vrho_vs_pyscf(xc, system):
    atom, spin, charge = system
    mol, mf, dm = _uks(atom, spin, charge, xc)
    coords = _coords(mol)
    xctype, deriv = bosonenergy.XC_KIND[xc.replace(" ", "")]

    ao_pyscf = numint.eval_ao(mol, coords, deriv=deriv)
    ao_numba = bosonenergy.eval_ao(
        mol, coords, deriv=deriv, evaluate_mf_with="numba"
    )
    assert ao_numba.shape == ao_pyscf.shape
    assert np.amax(np.abs(ao_numba - ao_pyscf)) < 3e-5

    for s in (0, 1):
        rho_ref = numint.eval_rho(mol, ao_pyscf, dm[s], xctype=xctype)
        rho_numba = bosonenergy.eval_rho_from_ao(ao_numba, dm[s], xctype=xctype)
        assert rho_numba.shape == rho_ref.shape
        assert np.amax(np.abs(rho_numba - rho_ref)) < 1e-4

    vrho_pyscf = bosonenergy.eval_vrho(
        mol, dm, xc, coords, evaluate_mf_with="pyscf"
    )
    mf_inputs = {"mol": mol, "dm": dm, "xc": xc}
    bosonenergy.prepare_mf_evaluator(mf_inputs, evaluate_mf_with="numba")
    vrho_numba = bosonenergy.eval_vrho(
        mol, dm, xc, coords, mf_inputs=mf_inputs
    )
    assert vrho_numba.shape == vrho_pyscf.shape
    assert np.amax(np.abs(vrho_numba - vrho_pyscf)) < 1e-5


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
def test_get_vxc_numba_vs_pyscf(xc):
    mol, mf, dm = _uks("He 0 0 0", 0, 0, xc)
    nconf, nelec = 5, sum(mf.nelec)
    rng = np.random.default_rng(1)
    configs_arr = 0.5 * rng.standard_normal((nconf, nelec, 3))

    class _Configs:
        def __init__(self, configs):
            self.configs = configs

    configs = _Configs(configs_arr)
    vxc_pyscf = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, xc, evaluate_mf_with="pyscf"
    )
    mf_inputs = {"mol": mol, "dm": dm, "xc": xc, "nelec": mf.nelec}
    bosonenergy.prepare_mf_evaluator(mf_inputs, evaluate_mf_with="numba")
    vxc_numba = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, xc, mf_inputs=mf_inputs
    )
    assert vxc_numba.shape == (nconf,)
    assert np.amax(np.abs(vxc_numba - vxc_pyscf)) < 1e-5
