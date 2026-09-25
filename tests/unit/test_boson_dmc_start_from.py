"""Eq population_snapshots → stats start_from assembly."""

import os
import uuid

import h5py
import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc import bosonrecipes
from pyqmc.method import bosondmc


def test_snapshot_schedule_equidistant():
    assert bosondmc.snapshot_schedule(100, 10) == set(range(9, 100, 10))
    assert bosondmc.snapshot_schedule(4, 2) == {1, 3}
    assert bosondmc.snapshot_schedule(10, None) == set()


def test_branch_nconfig_out():
    from pyqmc.configurations.coord import OpenConfigs

    rng = np.random.default_rng(0)
    configs = OpenConfigs(rng.standard_normal((8, 2, 3)))
    weights = np.ones(8)
    configs, weights, info = bosondmc.branch(configs, weights, nconfig_out=16)
    assert configs.configs.shape[0] == 16
    assert weights.shape == (16,)
    assert np.allclose(weights, weights[0])


@pytest.mark.slow
def test_population_snapshots_start_from(tmp_path):
    """Tiny He ABDMC: eq with K=2 snapshots → stats with 2× walkers, block=0."""
    mol = gto.M(atom="He 0. 0. 0.", basis="sto-3g", unit="bohr", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "LDA,VWN"
    dft_chk = str(tmp_path / "he_uks.hdf5")
    mf.chkfile = dft_chk
    mf.kernel()

    eq_hdf = str(tmp_path / f"eq_{uuid.uuid4().hex}.hdf5")
    stats_hdf = str(tmp_path / f"stats_{uuid.uuid4().hex}.hdf5")
    vmc_hdf = str(tmp_path / f"vmc_{uuid.uuid4().hex}.hdf5")

    n_eq = 8
    n_snap = 2
    n_stat = n_eq * n_snap
    jastrow_kws = {"ion_cusp": False, "na": 0, "nb": 0}

    bosonrecipes.ABDMC(
        dft_chk,
        eq_hdf,
        nconfig=n_eq,
        nblocks=4,
        nsteps_per_block=1,
        tstep=0.05,
        population_snapshots=n_snap,
        load_parameters=False,
        jastrow_kws=jastrow_kws,
        xc="LDA,VWN",
        evaluate_mf_with="numba",
        vmc_options={
            "nblocks": 2,
            "nsteps_per_block": 2,
            "tstep": 0.5,
            "hdf_file": vmc_hdf,
            "accumulators": ["energy"],
        },
        verbose=False,
    )

    with h5py.File(eq_hdf, "r") as f:
        assert "population_snapshots" in f
        g = f["population_snapshots"]
        assert int(g.attrs["n_written"]) == n_snap
        assert g["configs"].shape == (n_snap, n_eq, mol.nelectron, 3)

    bosonrecipes.ABDMC(
        dft_chk,
        stats_hdf,
        nconfig=n_stat,
        nblocks=2,
        nsteps_per_block=1,
        tstep=0.05,
        start_from=eq_hdf,
        load_parameters=False,
        jastrow_kws=jastrow_kws,
        xc="LDA,VWN",
        evaluate_mf_with="numba",
        verbose=False,
    )

    with h5py.File(stats_hdf, "r") as f:
        assert f["block"][0] == 0
        assert f["configs"].shape[0] == n_stat
        w = np.asarray(f["weights"])
        assert w.shape == (n_stat,)
        assert np.allclose(w, w[0])

    # continue_from + existing hdf_file still errors
    with pytest.raises(RuntimeError, match="already exists"):
        bosonrecipes.ABDMC(
            dft_chk,
            stats_hdf,
            nconfig=n_stat,
            nblocks=3,
            start_from=eq_hdf,
            load_parameters=False,
            jastrow_kws=jastrow_kws,
            xc="LDA,VWN",
            evaluate_mf_with="numba",
            verbose=False,
        )

    for path in (eq_hdf, stats_hdf, vmc_hdf, dft_chk):
        if os.path.isfile(path):
            os.remove(path)
