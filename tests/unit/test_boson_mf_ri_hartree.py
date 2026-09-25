"""RI / density-fitted Hartree vs on-the-fly pyscf (``int1e_grids``).

All systems use ccECP + ``ccecp-cc-pvdz``. RI is approximate: residuals are
DF fit error, **not** implementation noise. Empirically (default
``make_auxbasis``):

* absolute |ΔV_H| peaks **near nuclei** (r ≲ 0.3 Bohr);
* light atoms ~1e-5–3e-4 Ha; Fe/Sc can reach ~1e-3–1e-2 Ha near the nucleus;
* ``autoaux`` often tightens light atoms / Fe; ``def2-universal-jkfit`` helps
  bare Sc/Fe but can worsen He/Ne — check per molecule;
* Vxc for ``evaluate_mf_with='ri'`` still uses numba AO → ρ → libxc (machine
  match to pyscf within 1e-6).

Do **not** expect 1e-6 vs exact Hartree; that is the wrong target for RI.
"""

import copy

import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc.observables import bosonenergy, mf_hartree, mf_ri_hartree

# Absolute Ha vs int1e_grids for default make_auxbasis (pointwise).
TOL_VJ_LIGHT = 5e-4  # He / Li / Ne
TOL_VJ_TM = 1e-2  # Sc / Fe near-nucleus worst case with make_auxbasis
TOL_VXC = 1e-6
# Summed Vj over electrons per walker (QMC-relevant).
TOL_WALKER_LIGHT = 2e-4
TOL_WALKER_TM = 1e-2
TOL_DFT = 5e-3

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
    mf.max_cycle = 120
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


def _radial_coords(mol, r_lo, r_hi, n=24, seed=0):
    """Points with distance to a random nucleus in [r_lo, r_hi] (Bohr)."""
    rng = np.random.default_rng(seed)
    atoms = mol.atom_coords()
    rs = rng.uniform(r_lo, r_hi, n)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    centers = atoms[rng.integers(0, len(atoms), n)]
    return centers + dirs * rs[:, None]


def _walker_configs(mol, mf, nconf=3, seed=1, scale=0.4):
    nelec = sum(mf.nelec)
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    configs = np.zeros((nconf, nelec, 3))
    for i in range(nconf):
        centers = atom_coords[rng.integers(0, len(atom_coords), size=nelec)]
        configs[i] = centers + scale * rng.standard_normal((nelec, 3))
    return _Configs(configs)


def _assert_close(a, b, tol, label=""):
    err = np.amax(np.abs(np.asarray(a) - np.asarray(b)))
    assert np.isfinite(err)
    msg = f"max|err|={err} tol={tol}"
    if label:
        msg = f"{label}: {msg}"
    assert err < tol, msg


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
    v_ri = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)(coords)
    _assert_close(v_ri, v_ex, TOL_VJ_LIGHT)


@pytest.mark.parametrize(
    "atom,spin,charge,scf_kwargs",
    [
        ("Sc 0 0 0", 1, 0, {"level_shift": 0.2}),
        ("Fe 0 0 0", 4, 0, {"level_shift": 0.5, "damp": 0.5}),
    ],
)
def test_ri_vj_occupied_d_vs_pyscf(atom, spin, charge, scf_kwargs):
    mol, mf, dm = _uks(atom, spin, charge, **scf_kwargs)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol, n=16, seed=2)
    v_ex = mf_hartree.eval_vj_pyscf(mol, dm_total, coords)
    # Default make_auxbasis: TM near-nucleus residuals can be ~1e-3–1e-2.
    v_ri = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)(coords)
    _assert_close(v_ri, v_ex, TOL_VJ_TM, label="make_auxbasis")
    # autoaux typically tighter for Fe; still allow TM budget.
    v_aa = mf_ri_hartree.RIHartreePotentialEvaluator(
        mol, dm_total, auxbasis="autoaux"
    )(coords)
    _assert_close(v_aa, v_ex, TOL_VJ_TM, label="autoaux")


def test_ri_error_larger_near_nucleus_than_far():
    """Document spatial structure: absolute |ΔVj| peaks near the nucleus."""
    mol, mf, dm = _uks("Ne 0 0 0", 0, 0)
    dm_total = dm[0] + dm[1]
    ev = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)

    near = _radial_coords(mol, 0.02, 0.15, n=32, seed=10)
    far = _radial_coords(mol, 3.0, 8.0, n=32, seed=11)
    d_near = np.abs(ev(near) - mf_hartree.eval_vj_pyscf(mol, dm_total, near))
    d_far = np.abs(ev(far) - mf_hartree.eval_vj_pyscf(mol, dm_total, far))

    assert d_near.max() > d_far.max(), (
        f"expected larger near-nucleus abs error; "
        f"near max={d_near.max():.3e} far max={d_far.max():.3e}"
    )
    # Relative errors stay modest even near the nucleus.
    v_near = mf_hartree.eval_vj_pyscf(mol, dm_total, near)
    rel_near = d_near / np.maximum(np.abs(v_near), 1e-8)
    assert rel_near.max() < 1e-3


def test_ri_autoaux_improves_fe_near_nucleus():
    """For Fe, autoaux cuts near-nucleus abs error vs default make_auxbasis."""
    mol, mf, dm = _uks(
        "Fe 0 0 0", 4, 0, level_shift=0.5, damp=0.5
    )
    dm_total = dm[0] + dm[1]
    near = _radial_coords(mol, 0.02, 0.15, n=24, seed=7)
    v_ex = mf_hartree.eval_vj_pyscf(mol, dm_total, near)

    e_def = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)
    e_aa = mf_ri_hartree.RIHartreePotentialEvaluator(
        mol, dm_total, auxbasis="autoaux"
    )
    err_def = np.max(np.abs(e_def(near) - v_ex))
    err_aa = np.max(np.abs(e_aa(near) - v_ex))
    assert err_aa < err_def, (
        f"autoaux should tighten Fe near-nucleus; "
        f"make_aux={err_def:.3e} autoaux={err_aa:.3e}"
    )
    assert err_aa < 2e-3


def test_ri_vj_self_consistent_with_fitted_density():
    """RI Vj matches Σ_P c_P φ_P rebuilt from the same fit (sanity)."""
    mol, mf, dm = _uks("Ne 0 0 0", 0, 0)
    dm_total = dm[0] + dm[1]
    coords = _coords(mol, n=16, seed=5)
    ev = mf_ri_hartree.RIHartreePotentialEvaluator(mol, dm_total)
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
    _assert_close(vj_ri, vj_ex, TOL_WALKER_LIGHT)

    vxc_ri = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, "LDA,VWN", mf_inputs=mf_inputs
    )
    vxc_ex = bosonenergy.get_vxc(
        configs, mol, dm, mf.nelec, "LDA,VWN", evaluate_mf_with="pyscf"
    )
    _assert_close(vxc_ri, vxc_ex, TOL_VXC)


@pytest.mark.parametrize(
    "system,tol",
    [
        (("He 0 0 0", 0, 0, {}), TOL_WALKER_LIGHT),
        (("Ne 0 0 0", 0, 0, {}), TOL_WALKER_LIGHT),
        (("Fe 0 0 0", 4, 0, {"level_shift": 0.5, "damp": 0.5}), TOL_WALKER_TM),
    ],
)
def test_ri_walker_summed_vj(system, tol):
    """QMC-relevant: |Σ_e ΔV_H| per walker vs pyscf."""
    atom, spin, charge, scf_kwargs = system
    mol, mf, dm = _uks(atom, spin, charge, **scf_kwargs)
    configs = _walker_configs(mol, mf, nconf=8, seed=3)
    mf_inputs = {"mol": mol, "dm": dm, "mf_ri_auxbasis": None}
    bosonenergy.prepare_mf_evaluator(mf_inputs, evaluate_mf_with="ri")
    vj_ri = bosonenergy.get_vj(configs, mol, dm, mf_inputs=mf_inputs)
    vj_ex = bosonenergy.get_vj(configs, mol, dm, evaluate_mf_with="pyscf")
    _assert_close(vj_ri, vj_ex, tol)


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


def test_resolve_auxbasis_aliases():
    mol, _, _ = _uks("He 0 0 0", 0, 0)
    a0 = mf_ri_hartree.resolve_auxbasis(mol, None)
    a1 = mf_ri_hartree.resolve_auxbasis(mol, "make_auxbasis")
    assert a0 == a1
    aa = mf_ri_hartree.resolve_auxbasis(mol, "autoaux")
    assert aa is not None
    # Named basis string still passes through
    assert mf_ri_hartree.resolve_auxbasis(mol, "weigend") == "weigend"
