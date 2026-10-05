"""Tests for vectorized CI determinant excitation counting / filtering helpers."""

import numpy as np

from pyqmc.wf.bosonslater import (
    _count_excitations_vectorized,
    _orbital_degeneracy_group_ids,
)


def _count_excitations_ref(occ_excited, occ_ground, deg_group_ids):
    """Reference matching vectorized logic (set ops + group matching)."""
    occ_e = set(np.asarray(occ_excited).ravel())
    occ_g = set(np.asarray(occ_ground).ravel())
    exc_new = occ_e - occ_g
    exc_rem = occ_g - occ_e
    if not exc_new and not exc_rem:
        return 0
    matched = 0
    groups = {}
    for orb in exc_new | exc_rem:
        groups.setdefault(int(deg_group_ids[orb]), []).append(orb)
    for orbs in groups.values():
        n_new = sum(1 for o in orbs if o in exc_new)
        n_rem = sum(1 for o in orbs if o in exc_rem)
        matched += min(n_new, n_rem)
    return len(exc_new) - matched


def test_degeneracy_group_ids_contiguous():
    e = np.array([0.0, 0.0, 1.0, 1.0 + 1e-8, 2.0])
    g = _orbital_degeneracy_group_ids(e, deg_tol=1e-6)
    assert g[0] == g[1]
    assert g[2] == g[3]
    assert g[4] != g[2]
    assert len(np.unique(g)) == 3


def test_count_excitations_ground_single_double_deg_swap():
    # MOs: 0,1 occupied ground; 2,3 are degenerate virtuals; 4 higher
    mo_e = np.array([-1.0, -0.5, 0.1, 0.1, 1.0])
    deg = _orbital_degeneracy_group_ids(mo_e)
    ground = np.array([0, 1])

    occ = np.array(
        [
            [0, 1],  # ground
            [0, 2],  # single
            [2, 3],  # double
            [0, 3],  # single into other degenerate virtual
        ]
    )
    got = _count_excitations_vectorized(occ, ground, deg)
    ref = np.array(
        [_count_excitations_ref(o, ground, deg) for o in occ], dtype=np.int64
    )
    assert np.array_equal(got, ref)
    assert got[0] == 0
    assert got[1] == 1
    assert got[2] == 2
    assert got[3] == 1

    # Degenerate swap: replace 2 with 3 when both occupied somehow —
    # ground [0,2] vs excited [0,3] with 2,3 degenerate → 0 excitations
    ground2 = np.array([0, 2])
    occ_swap = np.array([[0, 3]])
    got_swap = _count_excitations_vectorized(occ_swap, ground2, deg)
    assert got_swap[0] == 0
    assert _count_excitations_ref(occ_swap[0], ground2, deg) == 0


def test_count_excitations_many_dets_matches_ref():
    rng = np.random.default_rng(0)
    n_mo, n_occ, n_dets = 20, 4, 500
    mo_e = np.sort(rng.normal(size=n_mo))
    # Force a few degeneracies
    mo_e[5] = mo_e[4]
    mo_e[11] = mo_e[10]
    deg = _orbital_degeneracy_group_ids(mo_e)
    ground = np.sort(rng.choice(n_mo, size=n_occ, replace=False))
    occ = np.vstack(
        [np.sort(rng.choice(n_mo, size=n_occ, replace=False)) for _ in range(n_dets)]
    )
    occ[0] = ground
    got = _count_excitations_vectorized(occ, ground, deg)
    ref = np.array(
        [_count_excitations_ref(o, ground, deg) for o in occ], dtype=np.int64
    )
    assert np.array_equal(got, ref)


def test_filter_determinants_string_path_matches_large_ci():
    """String-table occupations must match large_ci + binary_to_occ (tol=-1)."""
    from pyscf import gto, scf, mcscf, fci
    from pyqmc.wf.determinant_tools import binary_to_occ
    from pyqmc.wf.bosonslater import filter_determinants_from_ci

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.kernel()
    ncore = mc.ncore
    deters = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)

    for emax in (0.5, "doubles", "1.0,doubles"):
        dets, saved = filter_determinants_from_ci(
            mc, mf.mo_energy, emax, print_report=False, use_symm=False
        )
        for i, (w, (a, b)) in enumerate(dets):
            ind = int(saved["sorted_mask_indices"][i])
            assert abs(float(w) - float(deters[ind][0])) < 1e-12
            assert a == binary_to_occ(deters[ind][1], ncore)[0]
            assert b == binary_to_occ(deters[ind][2], ncore)[0]


def test_filter_determinants_accepts_flat_ci():
    """HDF5 dumps often store CI as a 1D vector of length na*nb."""
    from pyscf import gto, scf, mcscf
    from pyqmc.wf.bosonslater import filter_determinants_from_ci

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.kernel()
    ci2d = np.asarray(mc.ci).copy()
    mc.ci = ci2d.ravel()  # flat, as loaded from checkpoint
    dets_flat, saved_flat = filter_determinants_from_ci(
        mc, mf.mo_energy, "doubles", print_report=False, use_symm=False
    )
    mc.ci = ci2d
    dets_2d, saved_2d = filter_determinants_from_ci(
        mc, mf.mo_energy, "doubles", print_report=False, use_symm=False
    )
    assert len(dets_flat) == len(dets_2d)
    assert np.allclose(
        saved_flat["sorted_filtered_energies"], saved_2d["sorted_filtered_energies"]
    )
    for (w1, o1), (w2, o2) in zip(dets_flat, dets_2d):
        assert abs(float(w1) - float(w2)) < 1e-12
        assert o1 == o2
