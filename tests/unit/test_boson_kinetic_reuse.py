"""ABCDMC kinetic cache must match direct boson_kinetic."""

import numpy as np
from pyscf import dft, gto

from pyqmc import bosonrecipes
from pyqmc.observables import bosonenergy


def _he_setup(tmp_path):
    mol = gto.M(atom="He 0. 0. 0.", basis="sto-3g", unit="bohr", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "LDA,VWN"
    dft_chk = str(tmp_path / "he.hdf5")
    mf.chkfile = dft_chk
    mf.kernel()
    assert mf.converged

    wf, configs, acc = bosonrecipes.initialize_boson_qmc_objects(
        dft_chk,
        nconfig=8,
        load_parameters=None,
        jastrow_kws={"ion_cusp": False, "na": 1, "nb": 1},
        seed=1,
        xc="LDA,VWN",
        evaluate_mf_with="numba",
        accumulators=["abc_dmc_excitations"],
        use_symm=False,
        initial_guess_r=1.0,
    )
    # Avoid pathological far-away walkers from spherical guess (grads underflow)
    rng = np.random.default_rng(0)
    configs.configs[:] = 0.5 * rng.standard_normal(configs.configs.shape)
    wf.recompute(configs)
    return wf, configs, acc


def test_abcdmc_kinetic_cache_matches_direct(tmp_path):
    wf, configs, acc = _he_setup(tmp_path)
    bosonenergy.clear_boson_kinetic_cache(wf)

    lap_j0, drift_b0, grad20 = bosonenergy.boson_kinetic(configs, wf)

    abcdmc = acc["abc_dmc_excitations"]
    _ = abcdmc(configs, wf)

    lap_j1, drift_b1, grad21 = bosonenergy.boson_kinetic(configs, wf)

    assert np.allclose(lap_j1, lap_j0, rtol=1e-10, atol=1e-10)
    assert np.allclose(drift_b1, drift_b0, rtol=1e-10, atol=1e-10)
    assert np.allclose(grad21, grad20, rtol=1e-10, atol=1e-10)

    # Cache is one-shot; second call recomputes and still matches
    lap_j2, drift_b2, grad22 = bosonenergy.boson_kinetic(configs, wf)
    assert np.allclose(lap_j2, lap_j0, rtol=1e-10, atol=1e-10)
    assert np.allclose(drift_b2, drift_b0, rtol=1e-10, atol=1e-10)
    assert np.allclose(grad22, grad20, rtol=1e-10, atol=1e-10)


def test_energy_total_unchanged_with_abcdmc_first(tmp_path):
    wf, configs, acc = _he_setup(tmp_path)
    en_acc = acc["energy"]
    abcdmc = acc["abc_dmc_excitations"]

    bosonenergy.clear_boson_kinetic_cache(wf)
    e_direct = en_acc(configs, wf)

    bosonenergy.clear_boson_kinetic_cache(wf)
    _ = abcdmc(configs, wf)
    e_reuse = en_acc(configs, wf)

    for k in ("ka", "kb", "grad2", "ke", "total"):
        assert np.allclose(e_reuse[k], e_direct[k], rtol=1e-10, atol=1e-10)
