"""Compare the one-point diffusion probe with the four-point Richardson probe.

Both probes consume one ``np.random.normal`` draw per evaluation. Reseeding
before each method gives them the same Gaussian steps. The test function is

    psi = 1 + A x + B x^2 + C x^3

for which the diffusion score has the expansion D0 + c1 tau + c2 tau^2.
Richardson cancels c1 tau. A one-determinant Slater wavefunction is not used:
psi_n = Phi_n/Phi_B is identically 1, so both probes return zero.
"""

import numpy as np
from pyqmc.configurations.coord import OpenConfigs
from pyqmc.observables.bosonaccumulators import ABCDMCMatrixAccumulator


# psi = 1 + A x + B x^2 + C x^3.
# E[D] = D0 + D1 tau + D2 tau^2, with D0 = A^2 + 2B.
_A = 0.15
_B = 0.2
_C = 0.02
_D0 = _A**2 + 2.0 * _B
_D1 = 6.0 * _B**2 + 12.0 * _A * _C
_D2 = 45.0 * _C**2


class _PolynomialBosonWF:
    num_det = 1
    dtype = float
    _det_prod_filter = None

    def recompute(self, configs):
        self._configs = np.array(configs.configs, copy=True)
        nconf = len(self._configs)
        return np.ones(nconf), np.zeros(nconf)

    def value(self):
        nconf = len(self._configs)
        return np.ones(nconf), np.zeros(nconf)

    def value_dets(self):
        x = self._configs[:, 0, 0]
        psi = 1.0 + _A * x + _B * x * x + _C * x**3
        values = psi[:, np.newaxis]
        return np.ones_like(values), np.log(values)

    def gradient_dets(self, e, epos):
        x = epos.configs[:, 0]
        psi = 1.0 + _A * x + _B * x * x + _C * x**3
        dpsi = _A + 2.0 * _B * x + 3.0 * _C * x * x
        nconf = len(x)
        loggrad_phi_n = np.zeros((1, 3, nconf))
        loggrad_phi_n[0, 0, :] = dpsi / psi
        return loggrad_phi_n, np.zeros((3, nconf))


class _ZeroJastrow:
    dtype = float

    def recompute(self, configs):
        nconf = len(configs.configs)
        return np.ones(nconf), np.zeros(nconf)

    def gradient_laplacian(self, e, epos):
        nconf = len(epos.configs)
        return np.zeros((3, nconf)), np.zeros(nconf)


class _Product:
    dtype = float

    def __init__(self, boson, jastrow):
        self.wf_factors = [boson, jastrow]

    def recompute(self, configs):
        for factor in self.wf_factors:
            factor.recompute(configs)


def _accumulator(diffusion_probe):
    acc = ABCDMCMatrixAccumulator(
        mf_inputs={},
        system_params={"dtype": float},
        delta_method="diffusion",
        diffusion_probe=diffusion_probe,
        use_symm=False,
    )
    acc._boson_wf_type = _PolynomialBosonWF
    acc._jastrow_wf_type = _ZeroJastrow
    return acc


def _mean_and_se(acc, configs, wf, tau, nprobe, seed):
    """Average delta over probes. The seed fixes the Gaussian stream."""
    np.random.seed(seed)
    samples = np.empty(nprobe)
    for i in range(nprobe):
        samples[i] = np.mean(acc(configs, wf, tstep=tau)["delta"].real)
    stderr = samples.std(ddof=1) / np.sqrt(nprobe)
    return float(samples.mean()), float(stderr)


def compare_probes(nconf=64, nprobe=200, seed=17):
    configs = OpenConfigs(np.zeros((nconf, 1, 3)))
    wf = _Product(_PolynomialBosonWF(), _ZeroJastrow())
    wf.recompute(configs)
    direct = _accumulator("direct")
    richardson = _accumulator("richardson")
    taus = (0.5, 0.05, 0.005)

    print("psi = 1 + 0.15 x + 0.2 x^2 + 0.02 x^3, walkers fixed at the origin.")
    print(f"tau -> 0 limit of D is {_D0:.6f}.")
    print(
        f"{'tau':>8} {'direct':>16} {'richardson':>16} "
        f"{'analytic direct':>16} {'analytic rich.':>16} {'|d-r|':>10}"
    )
    rows = []
    for tau in taus:
        d_mean, d_se = _mean_and_se(direct, configs, wf, tau, nprobe, seed)
        r_mean, r_se = _mean_and_se(richardson, configs, wf, tau, nprobe, seed)
        analytic_direct = _D0 + _D1 * tau + _D2 * tau**2
        analytic_rich = _D0 - 0.5 * _D2 * tau**2
        rows.append((tau, d_mean, d_se, r_mean, r_se, analytic_direct, analytic_rich))
        print(
            f"{tau:8.4f} {d_mean:10.5f}±{d_se:.4f} {r_mean:10.5f}±{r_se:.4f} "
            f"{analytic_direct:16.5f} {analytic_rich:16.5f} {abs(d_mean - r_mean):10.5f}"
        )

    tau_large, d_large, _, r_large, r_se_large, _, analytic_rich_large = rows[0]
    tau_small, d_small, d_se_small, r_small, r_se_small, _, _ = rows[-1]
    assert abs(r_large - analytic_rich_large) < 4.0 * r_se_large + 1e-3
    assert abs(d_large - _D0) > abs(r_large - _D0)
    small_gap = abs(d_small - r_small)
    small_se = np.sqrt(d_se_small**2 + r_se_small**2)
    large_gap = abs(d_large - r_large)
    assert small_gap < large_gap
    assert small_gap < 4.0 * small_se
    print(
        f"At tau={tau_large}, |direct - limit|={abs(d_large - _D0):.4f} and "
        f"|richardson - limit|={abs(r_large - _D0):.4f}."
    )
    print(
        f"At tau={tau_small}, |direct - richardson|={small_gap:.4f} "
        f"against a combined stderr of {small_se:.4f}."
    )
    return rows


if __name__ == "__main__":
    compare_probes()
