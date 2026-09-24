"""RI / density-fitted Hartree potential vs on-the-fly pyscf oracle.

All molecular systems use ccECP (``ecp='ccecp'``) with a matching valence basis.
RI Vj is approximate; tolerances reflect default ``make_auxbasis`` quality.
Vxc for ``evaluate_mf_with='ri'`` still uses the numba AO → ρ → libxc path.
"""

import copy

import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc.observables import bosonenergy, mf_hartree, mf_ri_hartree

# Absolute error (Ha) vs exact int1e_grids for default auxbasis.
TOL_VJ = 1e-6
TOL_VXC = 1e-6
TOL_DFT = 5e-3  # summed Vj+Vxc over electrons / walkers

_ECP = "ccecp"
_BASIS = "ccecp-cc-pvdz"


class _Configs:
    def __init__(self, configs):
        self.configs = configs


def _uks(
    atom,
    spin,
    charge,
    xc="LDA,VWN",
    basis=_BASIS,
    ecp=_ECP,
    **scf_kwargs,
):
    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        ecp=ecp,
        unit="angstrom",
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    for key, value in scf_kwargs.items():
        setattr(mf, key, value)
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.make_rdm1()


def _coords(mol, n=24, seed=0):
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    near = atom_coords[rng.integers(0, len(atom_coords), size=n // 2)]
    near = near + 0.05 * rng.standard_normal(near.shape)
    far = 2.0 * rng.standard_normal((n - near.shape[0], 3))
    return np.vstack([near, far])


def _walker_configs(mol, mf, nconf=3, seed=1, scale=0.4):
    nelec = sum(mf.nelec)
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    configs = np.zeros((nconf, nelec, 3))
    for i in range(nconf):
        centers = atom_coords[rng.integers(0, len(atom_coords), size=nelec)]
        configs[i] = centers + scale * rng.standard_normal((nelec, 3))
    return _Configs(configs)


def _assert_close(a, b, tol):
    err = np.amax(np.abs(np.asarray(a) - np.asarray(b)))
    assert np.isfinite(err)
    assert err < tol, f"max|err|={err} tol={tol}"


@pytest.mark.parametrize(
    "system",
    [
        ("He 0 0 0", 0, 0, {}),
        ("Li 0 0 0", 1, 0, {}),
        ("Ne 0 0 0", 0, 0, {}),
    ],
)
def test_ri_vj_light_atoms_vs_pyscf(system):
    atom, spin, charge, scf_kwargs = system
    mol, mf, dm = _uks(atom, spin, charge, **scf_kwargs)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol)
    v_ex = mf_hartree.eval_vj_pyscf(mol, dm_total, coords)
    ev = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)
    v_ri = ev(coords)
    _assert_close(v_ri, v_ex, TOL_VJ)


@pytest.mark.parametrize(
    "atom,spin,charge,scf_kwargs",
    [
        ("Sc 0 0 0", 1, 0, {"level_shift": 0.2}),
        ("Fe 0 0 0", 4, 0, {"level_shift": 0.5, "damp": 0.5, "max_cycle": 150}),
    ],
)
def test_ri_vj_occupied_d_vs_pyscf(atom, spin, charge, scf_kwargs):
    mol, mf, dm = _uks(atom, spin, charge, **scf_kwargs)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol, n=16, seed=2)
    v_ex = mf_hartree.eval_vj_pyscf(mol, dm_total, coords)
    ev = mf_ri_hartree.RIHartreePotentialEvaluator(
        mol, dm_total, auxbasis="def2-universal-jkfit"
    )
    v_ri = ev(coords)
    _assert_close(v_ri, v_ex, TOL_VJ)


def test_ri_vj_self_consistent_with_fitted_density():
    """RI Vj matches Σ_P c_P φ_P rebuilt from the same fit (sanity, not vs exact)."""
    mol, mf, dm = _uks("Ne 0 0 0", 0, 0)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol, n=16, seed=5)
    ev = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total, auxbasis="def2-universal-jkfit")
    v1 = ev(coords)
    v2 = mf_ri_hartree.eval_vj_ri(ev.auxmol, ev.c_aux, coords, chunk_size=8)
    _assert_close(v1, v2, 1e-12)


def test_ri_get_vj_and_vxc_through_bosonenergy():
    mol, mf, dm = _uks("He 0 0 0", 0, 0)
    configs = _walker_configs(mol, mf)
    mf_inputs = {
        "mol": mol,
        "dm": dm,
        "nelec": mf.nelec,
        "xc": "LDA,VWN",
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }
    bosonenergy.prepare_mf_evaluator(mf_inputs, evaluate_mf_with="ri")
    assert isinstance(
        mf_inputs["vj_evaluator"], mf_ri_hartree.RIHartreePotentialEvaluator
    )

    vj_ri = bosonenergy.get_vj(configs, mol, dm, mf_inputs=mf_inputs)
    vj_ex = bosonenergy.get_vj(configs, mol, dm, evaluate_mf_with="pyscf")
    _assert_close(vj_ri, vj_ex, TOL_DFT)

    vxc_ri = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, "LDA,VWN", mf_inputs=mf_inputs
    )
    vxc_ex = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, "LDA,VWN", evaluate_mf_with="pyscf"
    )
    _assert_close(vxc_ri, vxc_ex, TOL_VXC)


def test_ri_dft_energy_vs_pyscf():
    mol, mf, dm = _uks("He 0 0 0", 0, 0)
    configs = _walker_configs(mol, mf)
    base = {
        "mol": mol,
        "dm": dm,
        "nelec": mf.nelec,
        "xc": "LDA,VWN",
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }
    mf_p = copy.deepcopy(base)
    bosonenergy.prepare_mf_evaluator(mf_p, evaluate_mf_with="pyscf")
    v_p, ecorr_p, saved_p = bosonenergy.dft_energy(mf_p, configs)

    mf_r = copy.deepcopy(base)
    bosonenergy.prepare_mf_evaluator(mf_r, evaluate_mf_with="ri")
    v_r, ecorr_r, saved_r = bosonenergy.dft_energy(mf_r, configs)

    assert ecorr_p == ecorr_r
    _assert_close(v_r, v_p, TOL_DFT)
    _assert_close(saved_r["vj"], saved_p["vj"], TOL_DFT)
    _assert_close(saved_r["vxc"], saved_p["vxc"], TOL_VXC)
    assert "ri" in bosonenergy.SUPPORTED_EVALUATE_MF
