"""Tests for the direct pure-diffusion estimator in ABCDMC."""

import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc import bosonrecipes
from pyqmc.configurations.coord import OpenConfigs
from pyqmc.method import bosondmc
from pyqmc.observables import bosonaccumulators


class _FakeBosonWF:
    """Two-basis-function model with analytically controlled gradients."""

    num_det = 2
    dtype = float
    _det_prod_filter = None

    def recompute(self, configs):
        self._configs = configs.configs.copy()
        nconf = len(configs.configs)
        return np.ones(nconf), np.zeros(nconf)

    def value(self):
        nconf = len(self._configs)
        return np.ones(nconf), np.zeros(nconf)

    def value_dets(self):
        nconf = len(self._configs)
        values = np.broadcast_to(np.array([1.0, 2.0]), (nconf, 2))
        return np.ones_like(values), np.log(values)

    def gradient_dets(self, e, epos):
        nconf = len(epos.configs)
        loggrad_phi_n = np.zeros((2, 3, nconf))
        loggrad_phi_n[0, 0, :] = 1.0
        loggrad_phi_n[1, 0, :] = 2.0
        loggrad_b = np.zeros((3, nconf))
        return loggrad_phi_n, loggrad_b

    def gradient_laplacian_dets(self, e, epos):
        loggrad_phi_n, loggrad_b = self.gradient_dets(e, epos)
        lap_phi_n = np.zeros((len(epos.configs), 2))
        return lap_phi_n, loggrad_phi_n, loggrad_b

    def laplacian(self, e, epos, **kwargs):
        return np.zeros(len(epos.configs))


class _FakeJastrow:
    dtype = float

    def recompute(self, configs):
        self._configs = configs.configs.copy()
        nconf = len(configs.configs)
        return np.ones(nconf), np.zeros(nconf)

    def gradient_laplacian(self, e, epos):
        nconf = len(epos.configs)
        grad = np.zeros((3, nconf))
        grad[0, :] = 3.0
        return grad, np.zeros(nconf)


class _FakeProductWF:
    dtype = float

    def __init__(self):
        self.wf_factors = [_FakeBosonWF(), _FakeJastrow()]

    def recompute(self, configs):
        for factor in self.wf_factors:
            factor.recompute(configs)
        nconf = len(configs.configs)
        return np.ones(nconf), np.zeros(nconf)

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        return None


def _fake_accumulator(delta_method, diffusion_probe="direct"):
    acc = bosonaccumulators.ABCDMCMatrixAccumulator(
        mf_inputs={},
        system_params={"dtype": float},
        delta_method=delta_method,
        diffusion_probe=diffusion_probe,
        use_symm=False,
    )
    acc._boson_wf_type = _FakeBosonWF
    acc._jastrow_wf_type = _FakeJastrow
    return acc


def test_diffusion_probe_adds_d_to_g_without_mutating_production(
    monkeypatch,
):
    nconf = 2
    tstep = 0.25
    configs = OpenConfigs(np.zeros((nconf, 1, 3)))
    wf = _FakeProductWF()
    wf.recompute(configs)
    configs_before = configs.configs.copy()
    production_values_before = wf.wf_factors[0]._configs.copy()

    displacement = np.zeros_like(configs.configs)
    displacement[:, 0, 0] = 0.5  # velocity_x = 0.5 / 0.25 = 2

    def fixed_normal(*, scale, size):
        assert scale == pytest.approx(np.sqrt(tstep))
        assert size == configs.configs.shape
        return displacement.copy()

    monkeypatch.setattr(bosonaccumulators.np.random, "normal", fixed_normal)

    acc = _fake_accumulator("diffusion")
    result = acc(configs, wf, tstep=tstep)

    # At both R and R_tilde, phi=(1,2) and grad(phi)=(1,4) along x.
    # G uses grad(log(Phi_B Psi_T))=3 and D uses velocity=2, and the probe
    # adds D, so delta = outer(phi, grad phi) * (3+2).
    expected_delta = np.array([[5.0, 20.0], [10.0, 40.0]])
    expected_ovlp = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert np.allclose(result["delta"], expected_delta[None, :, :])
    assert np.allclose(result["ovlp"], expected_ovlp[None, :, :])

    assert np.array_equal(configs.configs, configs_before)
    assert np.array_equal(wf.wf_factors[0]._configs, production_values_before)
    probe_boson = acc._probe_wf.wf_factors[0]
    assert np.allclose(probe_boson._configs, configs_before + displacement)


def test_richardson_probe_cancels_constant_gradient_and_keeps_production(
    monkeypatch,
):
    nconf = 2
    tstep = 0.25
    configs = OpenConfigs(np.zeros((nconf, 1, 3)))
    wf = _FakeProductWF()
    wf.recompute(configs)
    configs_before = configs.configs.copy()
    production_values_before = wf.wf_factors[0]._configs.copy()

    z = np.zeros_like(configs.configs)
    z[:, 0, 0] = 1.5

    def fixed_normal(*, loc=0.0, scale=1.0, size=None):
        assert scale == pytest.approx(1.0)
        assert size == configs.configs.shape
        return z.copy()

    monkeypatch.setattr(bosonaccumulators.np.random, "normal", fixed_normal)

    acc = _fake_accumulator("diffusion", diffusion_probe="richardson")
    result = acc(configs, wf, tstep=tstep)

    # Constant gradients make every odd score cancel, so D_ex = 0 and
    # delta reduces to G = outer(phi, grad phi) * grad(log Phi_B Psi_T).
    expected_delta = np.array([[3.0, 12.0], [6.0, 24.0]])
    assert np.allclose(result["delta"], expected_delta[None, :, :])
    assert np.array_equal(configs.configs, configs_before)
    assert np.array_equal(wf.wf_factors[0]._configs, production_values_before)
    probe_boson = acc._probe_wf.wf_factors[0]
    last_displacement = -np.sqrt(0.5 * tstep) * z
    assert np.allclose(probe_boson._configs, configs_before + last_displacement)


def test_delta_method_validation_and_ibp_smoke():
    with pytest.raises(ValueError, match="delta_method"):
        _fake_accumulator("unknown")
    with pytest.raises(ValueError, match="diffusion_probe"):
        _fake_accumulator("diffusion", diffusion_probe="unknown")

    configs = OpenConfigs(np.zeros((2, 1, 3)))
    wf = _FakeProductWF()
    wf.recompute(configs)

    diffusion = _fake_accumulator("diffusion")
    with pytest.raises(ValueError, match="positive finite tstep"):
        diffusion(configs, wf)

    ibp = _fake_accumulator("ibp")
    result = ibp(configs, wf)
    assert result["delta"].shape == (2, 2, 2)
    assert result["ovlp"].shape == (2, 2, 2)


def test_real_boson_wf_diffusion_smoke(tmp_path):
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", unit="bohr", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "LDA,VWN"
    dft_chk = str(tmp_path / "he_diffusion.hdf5")
    mf.chkfile = dft_chk
    mf.kernel()
    assert mf.converged

    wf, configs, accumulators = bosonrecipes.initialize_boson_qmc_objects(
        dft_chk,
        nconfig=4,
        load_parameters=None,
        jastrow_kws={"ion_cusp": False, "na": 1, "nb": 1},
        seed=1,
        xc="LDA,VWN",
        evaluate_mf_with="numba",
        accumulators=["abc_dmc_excitations"],
        delta_method="diffusion",
        use_symm=False,
        initial_guess_r=1.0,
    )
    rng = np.random.default_rng(4)
    configs.configs[:] = 0.25 * rng.standard_normal(configs.configs.shape)
    wf.recompute(configs)
    configs_before = configs.configs.copy()
    value_before = wf.value()

    result = accumulators["abc_dmc_excitations"](configs, wf, tstep=0.05)

    assert result["delta"].shape == (4, 1, 1)
    assert result["ovlp"].shape == (4, 1, 1)
    assert np.array_equal(configs.configs, configs_before)
    value_after = wf.value()
    assert np.array_equal(value_after[0], value_before[0])
    assert np.array_equal(value_after[1], value_before[1])


class _EnergyAccumulator:
    def __call__(self, configs, wf):
        return {"total": np.zeros(len(configs.configs))}


class _SpyABCDMCAccumulator:
    delta_method = "diffusion"

    def __init__(self):
        self.tsteps = []

    def __call__(self, configs, wf, tstep=None):
        self.tsteps.append(tstep)
        return {"delta": np.array([1.0, 3.0])}


def test_dmc_passes_tstep_and_uses_normal_post_update_weights(monkeypatch):
    configs = OpenConfigs(np.zeros((2, 1, 3)))
    wf = _FakeProductWF()
    weights = np.ones(2)
    spy = _SpyABCDMCAccumulator()

    def no_move(wf, configs, tstep, e):
        return (
            configs.electron(e),
            np.ones(len(configs.configs), dtype=bool),
            np.ones(len(configs.configs)),
            None,
        )

    monkeypatch.setattr(bosondmc, "propose_drift_diffusion", no_move)
    monkeypatch.setattr(
        bosondmc, "get_V2", lambda configs, wf, energydat: np.zeros(2)
    )
    target_multiplier = np.array([1.0, 2.0])
    target_s = np.log(target_multiplier) / 0.1
    monkeypatch.setattr(
        bosondmc,
        "compute_S",
        lambda *args, **kwargs: target_s,
    )

    result, _, final_weights = bosondmc.dmc_propagate(
        wf,
        configs,
        weights,
        tstep=0.1,
        branchcut_start=10.0,
        e_trial=0.0,
        e_est=0.0,
        nsteps=1,
        accumulators={
            "energy": _EnergyAccumulator(),
            bosonaccumulators.ABCDMC_ACC_KEY: spy,
        },
    )

    assert spy.tsteps == [pytest.approx(0.1)]
    assert np.allclose(final_weights, target_multiplier)
    # The current configurations carry post-update weights (1,2):
    # (1*1 + 2*3)/(1+2) = 7/3.
    assert result["abc_dmc_excitationsdelta"] == pytest.approx(7.0 / 3.0)
