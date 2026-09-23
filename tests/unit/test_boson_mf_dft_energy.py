"""Parity tests for bosonic MF local-energy terms (numba vs pyscf)."""

import copy

import numpy as np
import pytest
from pyscf import dft, gto

pytest.importorskip("numba")

from pyqmc.observables import bosonenergy
from pyqmc.observables.bosonaccumulators import ABQMCEnergyAccumulator


def _uks(atom, spin, charge, xc, basis="sto-3g", ecp=None):
    """Build a UKS molecule for MF parity tests.

    Requirements for AtomicOrbitalEvaluator:
    - spherical AOs (cart=False)
    - no SP hybrid shells (avoid Pople 6-31G)
    - segmented contractions only (coeff columns == 2); general-contraction
      all-electron Dunning shells (shape Nx3+) cannot be packed yet
    - angular momentum <= 5 (aug-cc-pvqz max l = 4)
    """
    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        ecp=ecp,
        cart=False,
        unit="angstrom",
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.make_rdm1()


def _mf_inputs(mol, mf, dm, xc):
    return {
        "mol": mol,
        "dm": dm,
        "xc": xc,
        "nelec": mf.nelec,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
    }


class _Configs:
    def __init__(self, configs):
        self.configs = configs


def _walker_configs(mol, mf, nconf=5, seed=1, scale=0.5):
    nelec = sum(mf.nelec)
    rng = np.random.default_rng(seed)
    atom_coords = mol.atom_coords()
    configs = np.zeros((nconf, nelec, 3))
    for i in range(nconf):
        centers = atom_coords[rng.integers(0, len(atom_coords), size=nelec)]
        configs[i] = centers + scale * rng.standard_normal((nelec, 3))
    return _Configs(configs)


# (atom, spin, charge, basis, ecp)
#
# General contraction: some Dunning all-electron shells store several AOs on
# the same primitives as columns [exp, c1, c2, ...] (shape N x 3+). The numba
# AtomicOrbitalEvaluator only packs segmented shells [exp, c] (N x 2), so
# all-electron aug-cc-pvXz for Li/Be/Ne cannot be used yet. He AE qz is
# segmented-only and packs fine. ccECP valence bases are also segmented.
#
# Ne has no ccecp-aug-cc-* set in PySCF; use non-augmented ccecp-cc-pvtz.
_SYSTEMS = [
    # All-electron sto-3g
    ("He 0 0 0", 0, 0, "sto-3g", None),
    ("Li 0 0 0", 1, 0, "sto-3g", None),
    ("Be 0 0 0", 1, 1, "sto-3g", None),  # Be+
    ("Ne 0 0 0", 0, 0, "sto-3g", None),
    # All-electron aug-cc-pvqz (He only; see general-contraction note above)
    ("He 0 0 0", 0, 0, "aug-cc-pvqz", None),
    # ccECP + matching valence basis
    ("Li 0 0 0", 1, 0, "ccecp-aug-cc-pvqz", "ccecp"),
    ("Be 0 0 0", 1, 1, "ccecp-aug-cc-pvqz", "ccecp"),  # Be+
    ("B 0 0 0", 0, 1, "ccecp-aug-cc-pvqz", "ccecp"),  # B+
    ("Ne 0 0 0", 0, 0, "ccecp-cc-pvtz", "ccecp"),  # no ccecp-aug-cc-* for Ne
]


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize("system", _SYSTEMS)
def test_dft_energy_numba_vs_pyscf(xc, system):
    atom, spin, charge, basis, ecp = system
    mol, mf, dm = _uks(atom, spin, charge, xc, basis=basis, ecp=ecp)
    configs = _walker_configs(mol, mf)
    base = _mf_inputs(mol, mf, dm, xc)

    mf_pyscf = copy.deepcopy(base)
    bosonenergy.prepare_mf_evaluator(mf_pyscf, evaluate_mf_with="pyscf")
    v_p, ecorr_p, saved_p = bosonenergy.dft_energy(mf_pyscf, configs)

    mf_numba = copy.deepcopy(base)
    bosonenergy.prepare_mf_evaluator(mf_numba, evaluate_mf_with="numba")
    v_n, ecorr_n, saved_n = bosonenergy.dft_energy(mf_numba, configs)

    assert saved_p.keys() == saved_n.keys() == {"vj", "vxc"}
    dvj = np.amax(np.abs(saved_n["vj"] - saved_p["vj"]))
    dvxc = np.amax(np.abs(saved_n["vxc"] - saved_p["vxc"]))
    dvmf = np.amax(np.abs(v_n - v_p))
    print(f"dvj={dvj:.3e}  dvxc={dvxc:.3e}  dvmf={dvmf:.3e}")
    assert dvj < 1e-12
    assert dvxc < 1e-12
    assert dvmf < 1e-12
    assert ecorr_n == ecorr_p


def test_abqmc_accumulator_mf_backend_parity():
    """ABQMCEnergyAccumulator prepare path yields matching dft_energy terms."""
    mol, mf, dm = _uks(
        "Li 0 0 0",
        1,
        0,
        "LDA,VWN",
        basis="ccecp-aug-cc-pvqz",
        ecp="ccecp",
    )
    configs = _walker_configs(mol, mf, nconf=4, seed=2)
    base = _mf_inputs(mol, mf, dm, "LDA,VWN")

    acc_p = ABQMCEnergyAccumulator(
        copy.deepcopy(base), evaluate_mf_with="pyscf"
    )
    acc_n = ABQMCEnergyAccumulator(
        copy.deepcopy(base), evaluate_mf_with="numba"
    )

    v_p, ecorr_p, saved_p = bosonenergy.dft_energy(acc_p.mf_inputs, configs)
    v_n, ecorr_n, saved_n = bosonenergy.dft_energy(acc_n.mf_inputs, configs)

    assert acc_p.mf_inputs["evaluate_mf_with"] == "pyscf"
    assert acc_n.mf_inputs["evaluate_mf_with"] == "numba"
    
    assert np.amax(np.abs(saved_n["vj"] - saved_p["vj"])) < 1e-11
    assert np.amax(np.abs(saved_n["vxc"] - saved_p["vxc"])) < 1e-11
    assert np.amax(np.abs(v_n - v_p)) < 1e-11
    assert ecorr_n == ecorr_p
    assert "vj" in saved_n and "vxc" in saved_n
