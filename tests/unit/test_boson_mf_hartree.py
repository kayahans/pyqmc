"""Parity tests for analytic Hartree Vj (mf_hartree) vs PySCF int1e_grids."""

import numpy as np
import pytest
from pyscf import dft, gto

pytest.importorskip("numba")

from pyqmc.observables import bosonenergy, mf_hartree


def _uks(atom, spin, charge, xc="LDA,VWN", basis="sto-3g"):
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


def _coords(mol, n=30, seed=0):
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    near = atom_coords[rng.integers(0, len(atom_coords), size=n // 2)]
    near = near + 0.05 * rng.standard_normal(near.shape)
    far = 2.0 * rng.standard_normal((n - near.shape[0], 3))
    return np.vstack([near, far])


@pytest.mark.parametrize(
    "system",
    [
        ("He 0 0 0", 0, 0, "sto-3g"),
        ("Li 0 0 0", 1, 0, "sto-3g"),
        ("Be 0 0 0", 1, 1, "sto-3g"),  # Be+
        ("Ne 0 0 0", 0, 0, "sto-3g"),
        ("Ne 0 0 0", 0, 0, "cc-pvdz"),  # includes d
        ("H 0 0 0; H 0 0 0.74", 0, 0, "sto-3g"),
    ],
)
def test_mf_hartree_vj_vs_pyscf(system):
    atom, spin, charge, basis = system
    mol, mf, dm = _uks(atom, spin, charge, basis=basis)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol)
    v_ref = mf_hartree.eval_vj_pyscf(mol, dm_total, coords, chunk_size=16)
    v = mf_hartree.eval_vj_points(mol, dm_total, coords, chunk_size=16)
    assert v.shape == v_ref.shape
    assert np.amax(np.abs(v - v_ref)) < 1e-10


def test_get_vj_numba_vs_pyscf():
    mol, mf, dm = _uks("He 0 0 0", 0, 0)
    nconf, nelec = 5, sum(mf.nelec)
    rng = np.random.default_rng(1)
    configs_arr = 0.5 * rng.standard_normal((nconf, nelec, 3))

    class _Configs:
        def __init__(self, configs):
            self.configs = configs

    configs = _Configs(configs_arr)
    vj_pyscf = bosonenergy.get_vj(configs, mol, dm, evaluate_mf_with="pyscf")
    mf_inputs = {
        "mol": mol,
        "dm": dm,
        "xc": "LDA,VWN",
        "nelec": mf.nelec,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }
    bosonenergy.prepare_mf_evaluator(mf_inputs, evaluate_mf_with="numba")
    vj_numba = bosonenergy.get_vj(configs, mol, dm, mf_inputs=mf_inputs)
    assert vj_numba.shape == (nconf,)
    assert np.amax(np.abs(vj_numba - vj_pyscf)) < 1e-10


def test_dft_energy_vj_numba_vs_pyscf():
    mol, mf, dm = _uks("Li 0 0 0", 1, 0, xc="LDA,VWN")
    nconf, nelec = 4, sum(mf.nelec)
    rng = np.random.default_rng(2)
    configs_arr = 0.4 * rng.standard_normal((nconf, nelec, 3))

    class _Configs:
        def __init__(self, configs):
            self.configs = configs

    configs = _Configs(configs_arr)
    base = {
        "mol": mol,
        "dm": dm,
        "xc": "LDA,VWN",
        "nelec": mf.nelec,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }
    mf_pyscf = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_pyscf, evaluate_mf_with="pyscf")
    v_p, _, saved_p = bosonenergy.dft_energy(mf_pyscf, configs)

    mf_numba = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_numba, evaluate_mf_with="numba")
    v_n, _, saved_n = bosonenergy.dft_energy(mf_numba, configs)

    assert np.amax(np.abs(saved_n["vj"] - saved_p["vj"])) < 1e-10
    assert np.amax(np.abs(v_n - v_p)) < 1e-8
