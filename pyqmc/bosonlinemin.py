import numpy as np
import pyqmc.gpu as gpu
import scipy
import h5py
import os

from bosonmc import abvmc
from bosonslater import BosonWF
from bosonjastrowspin import BosonJastrowSpin


def sr_update(pgrad, Sij, step, eps=0.1):
    invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step  # / np.linalg.norm(v)


def sd_update(pgrad, Sij, step, eps=0.1):
    return -pgrad * step  # / np.linalg.norm(pgrad)


def sr12_update(pgrad, Sij, step, eps=0.1):
    invSij = scipy.linalg.sqrtm(np.linalg.inv(Sij + eps * np.eye(Sij.shape[0])))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step  # / np.linalg.norm(v)


def opt_hdf(hdf_file, data, attr, configs, parameters):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
                hdf.create_group("wf")
                for k, it in parameters.items():
                    hdf.create_dataset("wf/" + k, data=gpu.asnumpy(it))
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)
            for k, it in parameters.items():
                hdf["wf/" + k][...] = gpu.asnumpy(it.copy())


def polyfit_relative(xfit, yfit, degree):
    p = np.polyfit(xfit, yfit, degree)
    ypred = np.polyval(p, xfit)
    resid = (ypred - yfit) ** 2
    relative_error = np.var(resid) / np.var(yfit)
    return p, relative_error


def stable_fit(xfit, yfit, tolerance=1e-2):
    """Fit a line and quadratic to xfit and yfit.
    1. If the linear fit is as good as the quadriatic, choose the lower endpoint.
    2. If the curvature is positive, estimate the minimum x value.
    3. If the lowest yfit is less than the new guess, use that xfit instead.

    :parameter list xfit: scalar step sizes along line
    :parameter list yfit: estimated energies at xfit points
    :parameter float tolerance: how good the quadratic fit needs to be
    :returns: estimated x-value of minimum
    :rtype: float
    """
    steprange = np.max(xfit)
    minstep = np.min(xfit)
    a = np.argmin(yfit)
    pq, relative_errq = polyfit_relative(xfit, yfit, 2)
    pl, relative_errl = polyfit_relative(xfit, yfit, 1)

    if relative_errl / relative_errq < 2:  # If a linear fit is about as good..
        if pl[0] < 0:
            est_min = steprange
        else:
            est_min = minstep
        out_y = np.polyval(pl, est_min)
    elif relative_errq < tolerance and pq[0] > 0:  # If quadratic fit is good
        est_min = -pq[1] / (2 * pq[0])
        if est_min > steprange:
            est_min = steprange
        if est_min < minstep:
            est_min = minstep
        out_y = np.polyval(pq, est_min)
    else:
        est_min = xfit[a]
        out_y = yfit[a]
    if (
        out_y > yfit[a]
    ):  # If min(yfit) has a lower energy than the guess, use it instead
        est_min = xfit[a]
    return est_min


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.2,
    max_iterations=30,
    warmup_options=None,
    vmcoptions=None,
    lmoptions=None,
    update=sr_update,
    update_kws=None,
    verbose=False,
    npts=5,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Optimizes energy by determining gradients with stochastic reconfiguration
        and minimizing the energy along gradient directions using correlated sampling.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter pgrad_acc: A PGradAccumulator-like object
    :parameter float steprange: How far to search in the line minimization
    :parameter int warmup: number of steps to use for vmc warmup
    :parameter int max_iterations: (maximum) number of steps in the gradient descent
    :parameter dict vmcoptions: a dictionary of options for the vmc method
    :parameter dict lmoptions: a dictionary of options for the lm method
    :parameter update: A function that generates a parameter change
    :parameter update_kws: Any keywords
    :parameter int npts: number of points to fit to in each line minimization
    :parameter boolean verbose: print output if True
    :return: optimized wave function, optimization data
    """
    if vmcoptions is None:
        vmcoptions = {}
    vmcoptions.update({"verbose": verbose})
    if lmoptions is None:
        lmoptions = {}
    if update_kws is None:
        update_kws = {}
    if warmup_options is None:
        warmup_options = dict(nblocks=1, nsteps_per_block=100)
    if "tstep" not in warmup_options and "tstep" in vmcoptions:
        warmup_options["tstep"] = vmcoptions["tstep"]
    assert npts >= 3, f"linemin npts={npts}; need npts >= 3 for correlated sampling"
    iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            coords.load_hdf(hdf)
    else:  # not restarting -- VMC warm up period
        if verbose:
            print("starting warmup")
        # _, coords = abvmc(
        #     wf,
        #     coords,
        #     accumulators={},
        #     client=client,
        #     npartitions=npartitions,
        #     **warmup_options,
        # )
        if verbose:
            print("finished warmup", flush=True)

    # Attributes for linemin
    attr = dict(max_iterations=max_iterations, npts=npts, steprange=steprange)

    def gradient_energy_function(x, coords):
        newparms = pgrad_acc.transform.deserialize(wf, x)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        df, coords = abvmc(
            wf,
            coords,
            accumulators={"pgrad": pgrad_acc},
            client=client,
            npartitions=npartitions,
            **vmcoptions,
        )
        en = np.real(np.mean(df["pgradtotal"], axis=0))
        # en_err = np.std(df["pgradtotal"], axis=0) / np.sqrt(df["pgradtotal"].shape[0])
        var = np.sqrt(1./(df["pgradtotal"].shape[0]-1)*np.sum(df['pgradtotal']**2-np.mean(df['pgradtotal'])**2))
        ratio = abs(var/en)
        sigma = np.std(df["pgradtotal"], axis=0) * np.sqrt(np.mean(df["nconfig"]))
        dpH = np.mean(df["pgraddpH"], axis=0)
        dp = np.mean(df["pgraddppsi"], axis=0)
        dpdp = np.mean(df["pgraddpidpj"], axis=0)
        grad = 2 * np.real(dpH - en * dp)
        Sij = np.real(dpdp - np.einsum("i,j->ij", dp, dp))

        if np.any(np.isnan(grad)):
            for nm, quant in {"dpH": dpH, "dp": dp, "en": en}.items():
                print(nm, quant)
            raise ValueError("NaN detected in derivatives")

        return coords, grad, Sij, en, var, sigma, ratio
    
    x0 = pgrad_acc.transform.serialize_parameters(wf.parameters)
    
    df = []
    # Gradient descent cycles
    for it in range(max_iterations):
        # Calculate gradient accurately
        # print('it', it, '='*20)
        # print('x0', x0)
        coords, pgrad, Sij, en, var, sigma, ratio = gradient_energy_function(x0, coords)
        # print('en', en, 'en_err', en_err)
        # print('pgrad', pgrad)
        step_data = {}
        step_data["energy"] = en
        step_data["var"] = var
        step_data["ratio"] = ratio
        step_data["x"] = x0
        step_data["pgradient"] = pgrad
        step_data["iteration"] = it + iteration_offset
        step_data["nconfig"] = coords.configs.shape[0]

        if verbose:
            print("descent en", en, var, " estimated sigma ", sigma)
            print("descent |grad|", np.linalg.norm(pgrad), flush=True)
            # print(pgrad)
            # print('a', wf.parameters.data['wf2']['acoeff'])
            # print('b', wf.parameters.data['wf2']['bcoeff'])

        xfit = []
        yfit = []
        # Calculate samples to fit.
        # include near zero in the fit, and go backwards as well
        # We don't use the above computed value because we are
        # doing correlated sampling.
        steps = np.linspace(-steprange / (npts - 2), steprange, npts)
        params = [x0 + update(pgrad, Sij, step, **update_kws) for step in steps]
        # print('params', params[-2])
        step_data['params'] = params[-2]
        if client is None:
            stepsdata = correlated_compute_boson(wf, coords, params, pgrad_acc)

        stepsdata["weight"] = (
            stepsdata["weight"] / np.mean(stepsdata["weight"], axis=1)[:, np.newaxis]
        )
        en = np.real(np.mean(stepsdata["total"] * stepsdata["weight"], axis=1))
        yfit.extend(en)
        xfit.extend(steps)
        est_min = stable_fit(xfit, yfit)
        # print('est_min', est_min)
        x0 += update(pgrad, Sij, est_min, **update_kws)
        step_data["tau"] = xfit
        step_data["yfit"] = yfit
        step_data["est_min"] = est_min

        opt_hdf(
            hdf_file, step_data, attr, coords, pgrad_acc.transform.deserialize(wf, x0)
        )
        df.append(step_data)

    newparms = pgrad_acc.transform.deserialize(wf, x0)
    for k in newparms:
        wf.parameters[k] = newparms[k]

    return wf, df



def correlated_compute_boson(wf, configs, params, pgrad_acc):
    """
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    :parameter wf: wave function object
    :parameter configs: (nconf, nelec, 3) array
    :parameter params: (nsteps, nparams) array
        list of arrays of parameters (serialized) at each step
    :parameter pgrad_acc: PGradAccumulator
    :returns: a single dict with indices [parameter, values]

    """

    data = []

    jastrow_wf = None

    for wave in wf.wf_factors:
        if isinstance(wave, BosonJastrowSpin):
            jastrow_wf = wave    
    psi0 = jastrow_wf.recompute(configs)[1]  # recompute gives det

    current_state = np.random.get_state()
    for p in params:
        np.random.set_state(current_state)
        newparms = pgrad_acc.transform.deserialize(wf, p)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        psi = jastrow_wf.recompute(configs)[1]  # recompute gives det
        rawweights = (psi/psi0)**2
        df = pgrad_acc.enacc(configs, wf)
        df["weight"] = rawweights
        data.append(df)
    data_ret = {}
    for k in data[0].keys():
        data_ret[k] = np.asarray([d[k] for d in data])
    return data_ret


