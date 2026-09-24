"""Parity tests for analytic Hartree Vj (mf_hartree) vs PySCF int1e_grids."""

import numpy as np
import pytest
from pyscf import dft, gto

pytest.importorskip("numba")

from pyqmc.observables import bosonenergy, mf_hartree


class _Configs:
    def __init__(self, configs):
        self.configs = configs


def _uks(atom, spin, charge, xc="LDA,VWN", basis="sto-3g", ecp=None, **scf_kwargs):
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


def _coords(mol, n=30, seed=0):
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    near = atom_coords[rng.integers(0, len(atom_coords), size=n // 2)]
    near = near + 0.05 * rng.standard_normal(near.shape)
    far = 2.0 * rng.standard_normal((n - near.shape[0], 3))
    return np.vstack([near, far])


def _angular_population(mol, dm, angmom):
    """Trace of ``dm`` on AOs whose shell label is this angular momentum."""
    idx = []
    for i, lab in enumerate(mol.ao_labels()):
        shell = lab.split()[-1]
        if len(shell) >= 2 and shell[1] == angmom:
            idx.append(i)
    if not idx:
        return 0.0
    return float(np.trace(dm[np.ix_(idx, idx)]))


def _assert_vj_parity(mol, dm, coords, tol=1e-10):
    """Analytic Vj vs chunked ``int1e_grids``, including the class API.

    The free function and ``HartreePotentialEvaluator`` use different chunk
    sizes from the oracle so a short coordinate list still crosses a chunk
    boundary on at least one path.
    """
    v_ref = mf_hartree.eval_vj_pyscf(mol, dm, coords, chunk_size=16)
    v = mf_hartree.eval_vj_points(mol, dm, coords, chunk_size=5)
    assert v.shape == v_ref.shape == (coords.shape[0],)
    assert np.amax(np.abs(v - v_ref)) < tol
    v_cls = mf_hartree.HartreePotentialEvaluator(mol, dm, chunk_size=7)(coords)
    assert v_cls.shape == v_ref.shape
    assert np.amax(np.abs(v_cls - v_ref)) < tol


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
    _assert_vj_parity(mol, dm[0] + dm[1], _coords(mol))


@pytest.mark.parametrize("cart", [False, True], ids=["sph", "cart"])
@pytest.mark.parametrize("l", range(5))
def test_pure_l_shell_identity_dm(l, cart):
    """One contracted shell on H, identity DM, cartesian and spherical.

    Libcint's radial factor for s/p differs from d/f/g. An identity density
    weights every component, so a bad high-L norm shows up against int1e_grids.
    """
    mol = gto.M(
        atom="H 0 0 0",
        basis={"H": [[l, [0.8, 0.7], [0.2, 0.3]]]},
        cart=cart,
        spin=1,
        charge=0,
        unit="Bohr",
        verbose=0,
    )
    dm = np.eye(mol.nao)
    _assert_vj_parity(mol, dm, _coords(mol, n=20, seed=l + 10 * int(cart)))


@pytest.mark.parametrize(
    "atom,spin,charge,basis,ecp,scf_kwargs",
    [
        ("Sc 0 0 0", 1, 0, "sto-3g", None, {}),
        # Open 3d shell: plain DIIS stalls; damping plus a level shift converges.
        ("Fe 0 0 0", 4, 0, "sto-3g", None, {"level_shift": 0.3, "damp": 0.4}),
        ("Sc 0 0 0; H 0 0 1.78", 0, 0, "sto-3g", None, {}),
        ("Sc 0 0 0; H 0 0 1.78", 0, 0, "ccecp-cc-pvdz", "ccecp", {}),
    ],
    ids=["Sc-sto3g", "Fe-sto3g", "ScH-sto3g", "ScH-ccecp-cc-pvdz"],
)
def test_occupied_d_vj_vs_int1e_grids(atom, spin, charge, basis, ecp, scf_kwargs):
    """UKS LDA densities with an occupied d shell vs int1e_grids."""
    mol, mf, dm = _uks(
        atom, spin, charge, basis=basis, ecp=ecp, **scf_kwargs
    )
    dm_total = dm[0] + dm[1]
    # Guard the setup: sto-3g ScH has about 0.3 electrons on Sc 3d.
    assert _angular_population(mol, dm_total, "d") > 0.2
    _assert_vj_parity(mol, dm_total, _coords(mol, seed=3))


def test_get_vj_numba_vs_pyscf():
    mol, mf, dm = _uks("He 0 0 0", 0, 0)
    nconf, nelec = 5, sum(mf.nelec)
    rng = np.random.default_rng(1)
    configs = _Configs(0.5 * rng.standard_normal((nconf, nelec, 3)))
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


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize(
    "system",
    [
        ("He 0 0 0", 0, 0),
        ("Li 0 0 0", 1, 0),
        ("Ne 0 0 0", 0, 0),
    ],
)
def test_dft_energy_numba_vs_pyscf(xc, system):
    """Full MF local energy (Vj + Vxc) numba vs pyscf oracle.

    Tolerance goal: no mHa-scale rigid shift (~1e-5 Ha/electron).
    """
    atom, spin, charge = system
    mol, mf, dm = _uks(atom, spin, charge, xc=xc)
    nconf, nelec = 4, sum(mf.nelec)
    rng = np.random.default_rng(2)
    configs = _Configs(0.4 * rng.standard_normal((nconf, nelec, 3)))
    base = {
        "mol": mol,
        "dm": dm,
        "xc": xc,
        "nelec": mf.nelec,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }

    mf_pyscf = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_pyscf, evaluate_mf_with="pyscf")
    v_p, ecorr_p, saved_p = bosonenergy.dft_energy(mf_pyscf, configs)

    mf_numba = dict(base)
    bosonenergy.prepare_mf_evaluator(mf_numba, evaluate_mf_with="numba")
    v_n, ecorr_n, saved_n = bosonenergy.dft_energy(mf_numba, configs)

    assert ecorr_n == pytest.approx(ecorr_p)
    assert np.amax(np.abs(saved_n["vj"] - saved_p["vj"])) < 1e-10
    # Per-electron XC / total MF potential (Ha)
    assert np.amax(np.abs(saved_n["vxc"] - saved_p["vxc"])) / nelec < 1e-5
    assert np.amax(np.abs(v_n - v_p)) / nelec < 1e-5
