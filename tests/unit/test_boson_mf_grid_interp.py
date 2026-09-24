"""Grid + scattered-interpolation MF potential vs on-the-fly pyscf oracle.

All molecular systems use ccECP (``ecp='ccecp'``) with a matching valence basis.
"""

import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc.observables import bosonenergy, mf_grid_interp, mf_hartree

# Uniform absolute error (Ha) vs on-the-fly pyscf for every grid-MF assertion.
TOL_ABS = 1e-1
# Node recovery for ScatteredScalarInterpolator (same sample points).
TOL_NODE = 1e-6

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
    mf.conv_tol = 1e-10
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


def _assert_close(a, b, tol=TOL_ABS):
    err = np.amax(np.abs(np.asarray(a) - np.asarray(b)))
    assert np.isfinite(err)
    assert err < tol


@pytest.mark.parametrize("method", list(mf_grid_interp.SUPPORTED_SCATTERED_METHODS))
def test_scattered_interpolator_reproduces_samples(method):
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((40, 3))
    vals = np.sin(pts[:, 0]) + 0.3 * pts[:, 1]
    interp = mf_grid_interp.ScatteredScalarInterpolator(pts, vals, method=method)
    _assert_close(interp(pts), vals, tol=TOL_NODE)


@pytest.mark.parametrize("method", ["nearest", "linear", "rbf"])
@pytest.mark.parametrize(
    "system",
    [
        ("He 0 0 0", 0, 0),
        ("Li 0 0 0", 1, 0),
        ("Ne 0 0 0", 0, 0),
    ],
    ids=["He", "Li", "Ne"],
)
def test_grid_mf_light_atoms_vs_pyscf(method, system):
    """Dense-grid table vs on-the-fly pyscf for light ccECP atoms."""
    atom, spin, charge = system
    mol, mf, dm = _uks(atom, spin, charge)
    assert mol._ecp
    coords = _coords(mol, n=20, seed=2)
    dm_total = dm[0] + dm[1]

    ev = mf_grid_interp.GridMFPotentialEvaluator(
        mol, dm, xc="LDA,VWN", grid_level=7, method=method, chunk_size=64
    )
    vj_ref = mf_hartree.eval_vj_pyscf(mol, dm_total, coords, chunk_size=64)
    vxc_ref = bosonenergy.eval_vrho(
        mol, dm, "LDA,VWN", coords, evaluate_mf_with="pyscf"
    )
    _assert_close(ev.eval_vj_points(coords), vj_ref)
    _assert_close(ev.eval_vxc_points(coords), vxc_ref)


@pytest.mark.parametrize("method", ["nearest", "linear", "rbf"])
@pytest.mark.parametrize("l", [0, 1, 2])
def test_grid_mf_pure_l_vj(method, l):
    """Identity-DM pure shell Vj vs int1e_grids (H + ccECP, hand shell)."""
    mol = gto.M(
        atom="H 0 0 0",
        basis={"H": [[l, [0.8, 0.7], [0.2, 0.3]]]},
        ecp=_ECP,
        cart=False,
        spin=1,
        charge=0,
        unit="Bohr",
        verbose=0,
    )
    assert mol._ecp
    eye = np.eye(mol.nao)
    dm = np.stack([0.5 * eye, 0.5 * eye], axis=0)
    coords = _coords(mol, n=16, seed=l + 3)
    dm_total = dm[0] + dm[1]

    ev = mf_grid_interp.GridMFPotentialEvaluator(
        mol, dm, xc="LDA,VWN", grid_level=7, method=method, chunk_size=32
    )
    vj_ref = mf_hartree.eval_vj_pyscf(mol, dm_total, coords, chunk_size=32)
    _assert_close(ev.eval_vj_points(coords), vj_ref)


@pytest.mark.parametrize("method", ["nearest", "linear"])
@pytest.mark.parametrize(
    "atom,spin,charge,scf_kwargs",
    [
        ("Sc 0 0 0", 1, 0, {}),
        ("Fe 0 0 0", 4, 0, {"level_shift": 0.5}),
        ("Sc 0 0 0; H 0 0 1.78", 0, 0, {}),
    ],
    ids=["Sc", "Fe", "ScH"],
)
def test_grid_mf_occupied_d_vs_pyscf(method, atom, spin, charge, scf_kwargs):
    """Occupied-d ccECP systems vs int1e_grids / eval_vrho."""
    mol, mf, dm = _uks(atom, spin, charge, **scf_kwargs)
    assert mol._ecp
    coords = _coords(mol, n=16, seed=4)
    dm_total = dm[0] + dm[1]

    ev = mf_grid_interp.GridMFPotentialEvaluator(
        mol, dm, xc="LDA,VWN", grid_level=5, method=method, chunk_size=64
    )
    """End-to-end ``evaluate_mf_with='grid'`` through ``dft_energy`` (He ccECP)."""
    mol, mf, dm = _uks("He 0 0 0", 0, 0)
    assert mol._ecp
    configs = _walker_configs(mol, mf, nconf=2)
    base = {
        "mol": mol,
        "dm": dm,
        "xc": "LDA,VWN",
        "nelec": mf.nelec,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
        "mf_grid_level": 6,
        "mf_grid_method": "nearest",
    }

    mf_p = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_p, evaluate_mf_with="pyscf")
    v_p, ecorr_p, saved_p = bosonenergy.dft_energy(mf_p, configs)

    mf_g = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_g, evaluate_mf_with="grid")
    v_g, ecorr_g, saved_g = bosonenergy.dft_energy(mf_g, configs)

    assert ecorr_g == pytest.approx(ecorr_p)
    assert mf_g["grid_mf_evaluator"] is not None
    _assert_close(saved_g["vj"], saved_p["vj"])
    _assert_close(saved_g["vxc"], saved_p["vxc"])
    _assert_close(v_g, v_p)


def test_grid_imports():
    assert hasattr(mf_grid_interp, "GridMFPotentialEvaluator")
    assert "grid" in bosonenergy.SUPPORTED_EVALUATE_MF
