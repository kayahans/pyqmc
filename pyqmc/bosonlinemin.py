import numpy as np
import pyqmc.gpu as gpu
import scipy
import h5py
import os
import pyqmc

from pyqmc.bosonmc import abvmc
from pyqmc import bosonslater


def np_pretty_print(nparray):
    c = '\n'
    with np.printoptions(formatter={'all': lambda x: f'{x:10.4g}'}, linewidth=150):
        c += nparray.__str__()
    return c

from scipy.sparse.linalg import cg

def get_sr_update_function(method="sr"):
    """Return the appropriate update function based on the method name.
    
    Args:
        method (str): Name of the update method. Options are:
            - "sr": Standard stochastic reconfiguration update
            - "sr_cg": SR update using conjugate gradient solver
            - "sd": Steepest descent update 
            - "sr12": SR update using square root of inverse S matrix
            
    Returns:
        function: The corresponding update function
    """
    update_functions = {
        "sr": sr_update,
        "sr_cg": sr_update_cg,
        "sr_adaptive": sr_update_adaptive,
        "sr_svd": sr_update_svd,
        "sr_gradient_based": sr_update_gradient_based,
        "sd": sd_update,
        "sr12": sr12_update
    }
    
    if method not in update_functions:
        raise ValueError(f"Unknown SR update method: {method}. Valid options are: {list(update_functions.keys())}")
        
    return update_functions[method]

def sr_update_cg(pgrad, Sij, step, eps=0.1, atol=1e-4, maxiter=100):
    Sij_reg = Sij + eps * np.eye(Sij.shape[0])
    v, info = cg(Sij_reg, pgrad, atol=atol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError("CG did not converge")
    return -v * step

def sr_update_adaptive(pgrad, Sij, step, eps_min=1e-4, eps_max=0.1, cond_threshold=1e3):
    Sij_reg = Sij + eps_min * np.eye(Sij.shape[0])
    cond = np.linalg.cond(Sij_reg)
    eps = eps_min * (cond / cond_threshold) if cond > cond_threshold else eps_min
    eps = min(eps, eps_max)
    invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step

def sr_update_svd(pgrad, Sij, step, min_eigval=1e-6):
    eigvals, eigvecs = np.linalg.eigh(Sij)
    mask = eigvals > min_eigval
    invSij = (eigvecs[:, mask] / eigvals[mask]) @ eigvecs[:, mask].T
    v = np.einsum("ij,j->i", invSij, pgrad)
    svd_step = -v*step
    return svd_step

def sr_update_gradient_based(pgrad, Sij, step, eps_min=1e-4, eps_max=0.1, grad_threshold=1.0):
    grad_norm = np.linalg.norm(pgrad)
    eps = eps_min * (grad_norm / grad_threshold) if grad_norm > grad_threshold else eps_min
    eps = min(eps, eps_max)
    invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step

# def sr_update_energy_based(pgrad, Sij, step, eps_min=1e-4, eps_max=0.1, energy_increase_threshold=1e-3):
#     # Assume energy_old and energy_new are available
#     energy_change = energy_new - energy_old
#     eps = eps_min * (abs(energy_change) / energy_increase_threshold) if energy_change > energy_increase_threshold else eps_min
#     eps = min(eps, eps_max)
#     invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
#     v = np.einsum("ij,j->i", invSij, pgrad)
#     return -v * step

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

def stable_fit(xfit, yfit, tolerance=1e-2, steprange=0.2, nblocks=1, min_step=0.01, step_factor=2, pgrad_prev=None, pgrad=None):
    # """Fit a line and quadratic to xfit and yfit.
    
    # The function handles several cases:
    # 1. If linear fit is better than quadratic fit (relative_errl/relative_errq < 2):
    #    - If slope is negative (pl[0] < 0), take maximum step
    #    - If slope is positive (pl[0] > 0), take minimum step
    # 2. If quadratic fit is better:
    #    - If quadratic coefficient is positive (convex), find minimum
    #    - If quadratic coefficient is negative (concave), reduce step range and increase nblocks
    # 3. If neither fit is good, use the point with minimum y value
    
    # Args:
    #     xfit: scalar step sizes along line
    #     yfit: estimated energies at xfit points
    #     tolerance: how good the quadratic fit needs to be
    #     steprange: current step range
    #     nblocks: current number of blocks
        
    # Returns:
    #     tuple: (estimated x-value of minimum, new step range, new nblocks)
    # """

    # minstep = np.min(xfit)
    # a = np.argmin(yfit)
    # pq, relative_errq = polyfit_relative(xfit, yfit, 2)
    # pl, relative_errl = polyfit_relative(xfit, yfit, 1)
    
    # # Default values for step range and nblocks
    # new_steprange = steprange
    # new_nblocks = nblocks
    
    # if relative_errl / relative_errq < 2:  # Linear fit is better
    #     print('linear fit is better')
    #     if pl[0] < 0:  # Negative slope - take maximum step
    #         est_min = steprange
    #     else:  # Positive slope - take minimum step
    #         est_min = minstep
    #     out_y = np.polyval(pl, est_min)
    # elif relative_errq < tolerance:  # Quadratic fit is good
    #     print('quadratic fit is good')
    #     if pq[0] > 0:  # Convex fit - find minimum
    #         print('convex fit')
    #         est_min = -pq[1] / (2 * pq[0])
    #         if est_min > steprange:
    #             est_min = steprange
    #         if est_min < minstep:
    #             est_min = minstep
    #         out_y = np.polyval(pq, est_min)
    #     else:  # Concave fit - reduce step range and increase nblocks
    #         print('concave fit')
    #         est_min = 0.0 # reject this step and try again
    #         out_y = -np.inf
    #         # Reduce step range more aggressively and increase nblocks more
    #         new_steprange = steprange * 0.95  # More aggressive reduction
    #         new_nblocks = int(nblocks * 1.2)  # More aggressive increase in blocks
    # else:  # Neither fit is good
    #     est_min = 0.0 # reject this step and try again
    #     out_y = -np.inf
    #     # If fit is bad, also reduce step range and increase blocks
    #     new_steprange = steprange * 0.1
    #     new_nblocks = nblocks 
        
    # if out_y > yfit[a]:  # If min(yfit) has lower energy than guess, use it
    #     est_min = xfit[a]
    
    # pq, relative_errq = polyfit_relative(xfit, yfit, 2)
    from scipy.interpolate import CubicSpline
    try:
        cs = CubicSpline(xfit, yfit)
        xdense = np.linspace(xfit[0], xfit[-1], 100)
        ydense = cs(xdense)
        dense_ind = np.argmin(ydense)
        est_min = xdense[dense_ind]
        if est_min < min_step:
            new_steprange = min_step
        else:
            new_steprange = np.abs(est_min)*step_factor        
    except:
        est_min = xfit[np.argmin(yfit)]
    
    if est_min < 0.0:
        est_min = 0.0
        
    new_steprange = steprange
    new_nblocks = nblocks
    # if np.linalg.norm(pgrad_prev) == 0.0:
    #     new_steprange = steprange
    # else:
    #     if np.linalg.norm(pgrad-pgrad_prev) < 0.01:
    #         new_nblocks = nblocks * 2
            

    return est_min, new_steprange, new_nblocks


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.1,
    stderr_weight=0.95,
    correlated_reference_wfs=None,
    max_iterations=30,
    warmup_options=None,
    vmcoptions=None,
    lmoptions=None,
    update="sr_svd",
    update_kws=None,
    verbose=False,
    npts=10,
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
    update = get_sr_update_function(update)

    if vmcoptions is None:
        vmcoptions = {}
    vmcoptions.update({"verbose": verbose})
    if "nblocks" not in vmcoptions:
        vmcoptions["nblocks"] = 10
    if lmoptions is None:
        lmoptions = {}
    if update_kws is None:
        update_kws = {}
    if warmup_options is None:
        warmup_options = dict(nblocks=1, nsteps_per_block=100)
    if "tstep" not in warmup_options and "tstep" in vmcoptions:
        warmup_options["tstep"] = vmcoptions["tstep"]
    assert npts >= 3, f"linemin npts={npts}; need npts >= 3 for correlated sampling"

    if correlated_reference_wfs is None:
        correlated_reference_wfs = [0, 1]
    # Add diagnostic tracking
    diagnostic_data = {
        'iterations': [],
        'energies': [],
        'energy_errors': [],
        'gradient_norms': [],
        'parameter_changes': [],
        'weight_stats': [],
        'step_sizes': [],
        'line_search': {
            'xfit': [],
            'yfit': [],
            'est_min': []
        },
        'sr_params': {
            'eps': [],
            'Sij_condition': [],
            'Sij_diag': [],
            'gradient_components': [],
            'est_min': [],
            'steprange': [],
            'Sij': [],  # Full S matrix
            'dp': [],   # Parameter derivatives
            'dpdp': [], # Parameter derivative products
            'dpH': []   # Energy derivatives
        }
    }

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
            print("starting ABVMC warmup")  
            _, coords = abvmc(
                wf,
                coords,
                accumulators={'energy': pgrad_acc.enacc},
                client=client,
                npartitions=npartitions,
                **warmup_options,
            )
            print("finished ABVMC warmup", flush=True)

    # Attributes for linemin
    attr = dict(max_iterations=max_iterations, npts=npts, steprange=steprange)

    def gradient_energy_function(x, coords):
        newparms = pgrad_acc.transform.deserialize(wf, x)
        if verbose:
            c = ''
            for key, value in newparms.items():
                c += f'{key}({value.flatten().shape[0]} elements): {np_pretty_print(value.flatten())}\n'
            print('Wavefunction parameters: \n', c)

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
        total_energy = df["pgradtotal"]
        en = np.real(np.mean(total_energy, axis=0))
        var = np.sqrt(1./(total_energy.shape[0]-1)*np.sum(total_energy**2-np.mean(total_energy)**2))
        ratio = abs(var/en)
        sigma = np.std(total_energy, axis=0) * np.sqrt(np.mean(df["nconfig"]))
        dpH = np.mean(df["pgraddpH"], axis=0)
        dp = np.mean(df["pgraddppsi"], axis=0)
        dpdp = np.mean(df["pgraddpidpj"], axis=0)
        grad = 2 * np.real(dpH - en * dp)
        Sij = np.real(dpdp - np.einsum("i,j->ij", dp, dp))
        saved_results = {}
        for k in df.keys():
            saved_results[k] = np.mean(df[k], axis=0)


        if np.any(np.isnan(grad)):
            for nm, quant in {"dpH": dpH, "dp": dp, "en": en}.items():
                print(nm, quant)
            raise ValueError("NaN detected in derivatives")

        return coords, grad, Sij, en, var, sigma, ratio, saved_results, total_energy
    
    x0 = pgrad_acc.transform.serialize_parameters(wf.parameters)
    
    df = []
    # Gradient descent cycles
    pgrad_prev = x0 * 0.0
    for it in range(max_iterations):
        # Calculate gradient accurately
        print('it', it, 'starting ' + '='*20)
        print('steprange', steprange)
        print('nblocks', vmcoptions['nblocks'])
        coords, pgrad, Sij, en, en_err, sigma, ratio, saved_results, total_energy = gradient_energy_function(x0, coords)
        # Track diagnostics
        diagnostic_data['iterations'].append(it)
        diagnostic_data['energies'].append(en)
        diagnostic_data['energy_errors'].append(en_err)
        diagnostic_data['gradient_norms'].append(np.linalg.norm(pgrad))

        # Track SR parameters
        diagnostic_data['sr_params']['eps'].append(update_kws.get('eps', 0.1))
        diagnostic_data['sr_params']['Sij_condition'].append(np.linalg.cond(Sij))
        diagnostic_data['sr_params']['Sij_diag'].append(np.diag(Sij))
        diagnostic_data['sr_params']['gradient_components'].append(pgrad)
        diagnostic_data['sr_params']['steprange'].append(steprange)
        diagnostic_data['sr_params']['Sij'].append(Sij)
        diagnostic_data['sr_params']['dp'].append(saved_results['pgraddppsi'])
        diagnostic_data['sr_params']['dpdp'].append(saved_results['pgraddpidpj'])
        diagnostic_data['sr_params']['dpH'].append(saved_results['pgraddpH'])

        if verbose:
            print(f'pgrad: {pgrad.shape} {np_pretty_print(pgrad)}')
            print(f'Sij: {Sij.shape} {np_pretty_print(np.diag(Sij))}')
            print(f'en: {en}')
            print(f'en_err: {en_err}')
            print(f'sigma: {sigma}')
            print(f'ratio: {ratio}')

        step_data = {}
        step_data["energy"] = en
        
        step_data["ka"] = saved_results["pgradka"]
        step_data["kb"] = saved_results["pgradkb"]
        step_data["ee"] = saved_results["pgradee"]
        step_data["ei"] = saved_results["pgradei"]
        step_data["vj"] = saved_results["pgradvj"]
        step_data["vxc"] = saved_results["pgradvxc"]
        step_data["corr"] = saved_results["pgradcorr"]
        step_data["energy_error"] = en_err
        step_data["ratio"] = ratio
        step_data["x"] = x0
        step_data["pgradient"] = pgrad
        step_data["iteration"] = it + iteration_offset
        step_data["nconfig"] = coords.configs.shape[0]

        if verbose:
            print("descent en", en, en_err, " estimated sigma ", sigma)
            print("descent |grad|", np.linalg.norm(pgrad), flush=True)

        xfit = []
        yfit = []
        steps = np.linspace(-steprange / (npts - 2), steprange, npts)
        params = [x0 + update(pgrad, Sij, step, **update_kws) for step in steps]
        
        if client is None:
            stepsdata = correlated_compute_boson(wf, coords, params, pgrad_acc, correlated_reference_wfs)
        else:
            stepsdata = correlated_compute_boson_parallel(wf, coords, params, pgrad_acc, client, npartitions, correlated_reference_wfs)
        

        # Track weight statistics
        weights = stepsdata["weight"]
        weights = weights / np.mean(weights, axis=1, keepdims=True)
        stepsdata["weight"] = weights

        weight_mean = np.mean(weights, axis=1)
        weight_std = np.std(weights, axis=1)
        weight_max = np.max(weights, axis=1)
        diagnostic_data['weight_stats'].append({
            'mean': weight_mean,
            'std': weight_std,
            'max': weight_max
        })

        stepsdata["weight"] = (
            stepsdata["weight"] / np.mean(stepsdata["weight"], axis=1)[:, np.newaxis]
        )
        en = np.real(np.mean(stepsdata["total"] * stepsdata["weight"], axis=1))
        en_std = np.std(stepsdata["total"], axis=1)

        # Exclude ka and kb from correlated sampling. 
        # total_mk = stepsdata["total"] - stepsdata["ka"] - stepsdata["ke"]
        # en = np.real(np.mean(total_mk * stepsdata["weight"], axis=1))

        # yfit.extend(en)
        yfit.extend(en + stderr_weight*en_std)
        xfit.extend(steps)
        
        # Get current nblocks from vmcoptions
        current_nblocks = vmcoptions.get('nblocks', 1)
        est_min, new_steprange, new_nblocks = stable_fit(xfit, yfit,steprange=steprange, nblocks=current_nblocks, pgrad_prev=pgrad_prev, pgrad=pgrad)
        
        if est_min < 0 or est_min >= steprange:
            print('est_min is out of range, stabilizing optimization')
            # Stabilize optimization by taking the midpoint of the step range
            est_min = steprange / 2
            print('stabilized est_min to', est_min)

        # Update step range and nblocks if needed
        if new_steprange != steprange:
            steprange = new_steprange
            if verbose:
                print(f"Adjusting step range to {steprange}")
        
        if new_nblocks != current_nblocks:
            vmcoptions['nblocks'] = new_nblocks
            if verbose:
                print(f"Adjusting nblocks to {new_nblocks}")
        
        # est_min = ks_fit(xfit, en_std, steps)
        
        # Track line search data
        diagnostic_data['line_search']['xfit'].append(np.array(xfit))
        diagnostic_data['line_search']['yfit'].append(np.array(yfit))
        diagnostic_data['line_search']['est_min'].append(est_min)
        
        dx = update(pgrad, Sij, est_min, **update_kws)
        
        # Track parameter changes with more detail
        param_change = np.linalg.norm(dx)
        relative_change = np.abs(dx / (np.abs(x0) + 1e-10))
        max_relative_change = np.max(relative_change)
        diagnostic_data['parameter_changes'].append({
            'absolute': param_change,
            'max_relative': max_relative_change,
            'relative_changes': relative_change,
            'dx': dx
        })
        diagnostic_data['step_sizes'].append(est_min)

        x0 += dx
        step_data["tau"] = xfit
        step_data["x0"] = x0
        step_data["yfit"] = yfit
        step_data["est_min"] = est_min

        x0_deserialized = pgrad_acc.transform.deserialize(wf, x0)
        if verbose:
            c = ''
            for key, value in x0_deserialized.items():
                c += f'{key}({value.flatten().shape[0]} elements): {value.flatten()}\n'
            print('Wavefunction parameters: \n', c)
            print('Change in parameters: ', np_pretty_print(dx))
            print('x0', np_pretty_print(x0))
            print('x_fit', np_pretty_print(np.array(xfit)))
            print('y_fit', np_pretty_print(np.array(yfit)))
            print('est_min', est_min)
            
            # Plot diagnostics
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 15))
            
            # Energy plot
            plt.subplot(3, 3, 1)
            plt.errorbar(diagnostic_data['iterations'], diagnostic_data['energies'], 
                        yerr=diagnostic_data['energy_errors'], fmt='o-')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.title('Energy vs Iteration')
            
            # Gradient norm plot
            plt.subplot(3, 3, 2)
            plt.plot(diagnostic_data['iterations'], diagnostic_data['gradient_norms'], 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm vs Iteration')
            
            # Parameter changes plot
            plt.subplot(3, 3, 3)
            param_changes = [d['absolute'] for d in diagnostic_data['parameter_changes']]
            plt.plot(diagnostic_data['iterations'], param_changes, 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Parameter Change Norm')
            plt.title('Parameter Changes vs Iteration')
            
            # Weight statistics plot
            plt.subplot(3, 3, 4)
            # weight_means = [np.mean(d['mean']) for d in diagnostic_data['weight_stats']]
            # weight_stds = [np.mean(d['std']) for d in diagnostic_data['weight_stats']]
            # plt.errorbar(diagnostic_data['iterations'], weight_means, yerr=weight_stds, fmt='o-')
            # plt.xlabel('Iteration')
            # plt.ylabel('Mean Weight')
            # plt.title('Weight Statistics vs Iteration')

            plt.plot(total_energy, '-o')
            plt.xlabel('Step')
            plt.ylabel('Total Energy')
            plt.title('Total Energy vs Step')
            # SR parameters plots
            plt.subplot(3, 3, 5)
            plt.plot(diagnostic_data['iterations'], diagnostic_data['sr_params']['eps'], 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('SR eps')
            plt.title('SR Regularization Parameter')

            plt.subplot(3, 3, 6)
            plt.plot(diagnostic_data['iterations'], diagnostic_data['sr_params']['Sij_condition'], 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Condition Number')
            plt.title('S Matrix Condition Number')

            plt.subplot(3, 3, 7)
            plt.plot(diagnostic_data['iterations'], diagnostic_data['line_search']['est_min'], 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Estimated Minimum')
            plt.title('Line Search Minimum')

            plt.subplot(3, 3, 8)
            max_rel_changes = [d['max_relative'] for d in diagnostic_data['parameter_changes']]
            plt.plot(diagnostic_data['iterations'], max_rel_changes, 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Max Relative Change')
            plt.title('Maximum Relative Parameter Change')

            plt.subplot(3, 3, 9)
            plt.plot(diagnostic_data['iterations'], diagnostic_data['sr_params']['steprange'], 'o-')
            plt.xlabel('Iteration')
            plt.ylabel('Step Range')
            plt.title('Line Search Range')

            plt.tight_layout()
            plt.savefig(f'optimization_diagnostics_iter_{it}.png')
            plt.close()

            # Plot energy components
            energy_components = ['ka', 'kb', 'grad2', 'ke', 'ee', 'corr', 'ei', 'ii', 'total', 'vj', 'vxc']
            n_components = len(energy_components)
            n_cols = 4
            n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division
            
            plt.figure(figsize=(20, 5*n_rows))
            for i, component in enumerate(energy_components):
                plt.subplot(n_rows, n_cols, i + 1)
                e_comp_mean = np.mean(stepsdata[component]*stepsdata["weight"], axis=1)
                e_comp_std = np.std(stepsdata[component]*stepsdata["weight"], axis=1)
                plt.errorbar(xfit,  e_comp_mean, yerr=e_comp_std, fmt='o-', label=component)
                plt.xlabel('Iteration')
                plt.ylabel('Energy')
                plt.title(f'{component} vs Iteration')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'energy_components_iter_{it}.png')
            plt.close()

            # Plot line search data for current iteration
            plt.figure(figsize=(10, 6))
            plt.plot(xfit, yfit, 'o-', label='Energy points')
            plt.axvline(x=est_min, color='r', linestyle='--', label='Estimated minimum')
            plt.xlabel('Step size')
            plt.ylabel('Energy')
            plt.title(f'Line Search (Iteration {it})')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'line_search_iter_{it}.png')
            plt.close()

            # Print detailed step size information
            print("\nStep Size Calculation Details:")
            print(f"SR eps: {update_kws.get('eps', 0.1)}")
            print(f"S matrix condition number: {np.linalg.cond(Sij):.2e}")
            print(f"Line search range: {steprange}")
            print(f"Estimated minimum step: {est_min}")
            print(f"Maximum relative parameter change: {max_relative_change:.2e}")
            print(f"Parameter change norm: {param_change:.2e}")
            print(f"Gradient norm: {np.linalg.norm(pgrad):.2e}")
            print("="*50)

        opt_hdf(
            hdf_file, step_data, attr, coords, x0_deserialized
        )
        df.append(step_data)
        pgrad_prev = pgrad        
        print('it', it, ' finished ' + '='*20)

    # Save detailed diagnostic data
    # if hdf_file is not None:
    #     with h5py.File(hdf_file, "a") as hdf:
    #         if "diagnostics" not in hdf:
    #             diag_grp = hdf.create_group("diagnostics")
    #         else:
    #             diag_grp = hdf["diagnostics"]
            
    #         # Save basic arrays
    #         for key in ['iterations', 'energies', 'energy_errors', 'gradient_norms', 'step_sizes']:
    #             if key in diagnostic_data:
    #                 if key in diag_grp:
    #                     del diag_grp[key]
    #                 diag_grp.create_dataset(key, data=np.array(diagnostic_data[key]))
            
    #         # Save parameter changes
    #         if "parameter_changes" in diag_grp:
    #             del diag_grp["parameter_changes"]
    #         param_grp = diag_grp.create_group("parameter_changes")
    #         for i, change in enumerate(diagnostic_data['parameter_changes']):
    #             iter_grp = param_grp.create_group(f"iteration_{i}")
    #             for key, value in change.items():
    #                 iter_grp.create_dataset(key, data=value)
            
    #         # Save weight statistics
    #         if "weight_stats" in diag_grp:
    #             del diag_grp["weight_stats"]
    #         weight_grp = diag_grp.create_group("weight_stats")
    #         for i, stats in enumerate(diagnostic_data['weight_stats']):
    #             iter_grp = weight_grp.create_group(f"iteration_{i}")
    #             for key, value in stats.items():
    #                 iter_grp.create_dataset(key, data=value)
            
    #         # Save SR parameters
    #         if "sr_params" in diag_grp:
    #             del diag_grp["sr_params"]
    #         sr_grp = diag_grp.create_group("sr_params")
    #         for key in ['eps', 'Sij_condition', 'Sij_diag', 'gradient_components', 'est_min', 'steprange']:
    #             if key in diagnostic_data['sr_params']:
    #                 sr_grp.create_dataset(key, data=np.array(diagnostic_data['sr_params'][key]))
            
    #         # Save full matrices and vectors
    #         matrices_grp = sr_grp.create_group("matrices")
    #         for i in range(len(diagnostic_data['sr_params']['Sij'])):
    #             iter_grp = matrices_grp.create_group(f"iteration_{i}")
    #             iter_grp.create_dataset("Sij", data=diagnostic_data['sr_params']['Sij'][i])
    #             iter_grp.create_dataset("dp", data=diagnostic_data['sr_params']['dp'][i])
    #             iter_grp.create_dataset("dpdp", data=diagnostic_data['sr_params']['dpdp'][i])
    #             iter_grp.create_dataset("dpH", data=diagnostic_data['sr_params']['dpH'][i])

    #         # Save line search data
    #         if "line_search" in diag_grp:
    #             del diag_grp["line_search"]
    #         line_search_grp = diag_grp.create_group("line_search")
    #         for i in range(len(diagnostic_data['line_search']['xfit'])):
    #             iter_grp = line_search_grp.create_group(f"iteration_{i}")
    #             iter_grp.create_dataset("xfit", data=diagnostic_data['line_search']['xfit'][i])
    #             iter_grp.create_dataset("yfit", data=diagnostic_data['line_search']['yfit'][i])
    #             iter_grp.create_dataset("est_min", data=diagnostic_data['line_search']['est_min'][i])

    newparms = pgrad_acc.transform.deserialize(wf, x0)
    for k in newparms:
        wf.parameters[k] = newparms[k]

    return wf, df


def correlated_compute_boson_parallel(wf, configs, params, pgrad_acc, client, npartitions, correlated_reference_wfs):
    config = configs.split(npartitions)
    runs = [
        client.submit(correlated_compute_boson, wf, conf, params, pgrad_acc, correlated_reference_wfs)
        for conf in config
    ]
    allresults = [r.result() for r in runs]
    block_avg = {}
    for k in allresults[0].keys():
        block_avg[k] = np.hstack([res[k] for res in allresults])
    return block_avg


def correlated_compute_boson(wf, configs, params, pgrad_acc, ref_wfs = [0,1]):
    """
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    :parameter wf: wave function object
    :parameter configs: (nconf, nelec, 3) array
    :parameter params: (nsteps, nparams) array
        list of arrays of parameters (serialized) at each step
    :parameter pgrad_acc: PGradAccumulator
    :returns: a single dict with indices [parameter, values]

    """
    
    # data = []

    # jastrow_wf = None

    # for wave in wf.wf_factors:
    #     if isinstance(wave, pyqmc.jastrowspin.JastrowSpin):
    #         jastrow_wf = wave    
    
    # psi0 = jastrow_wf.recompute(configs)[1]  # recompute gives det

    # current_state = np.random.get_state()
    # for p in params:
    #     np.random.set_state(current_state)
    #     newparms = pgrad_acc.transform.deserialize(wf, p)
    #     for k in newparms:
    #         wf.parameters[k] = newparms[k]
    #     psi = jastrow_wf.recompute(configs)[1]  # recompute gives det
    #     rawweights = (psi/psi0)**2
    #     df = pgrad_acc.enacc(configs, wf)
    #     df["weight"] = rawweights
    #     data.append(df)
    # data_ret = {}
    # for k in data[0].keys():
    #     data_ret[k] = np.asarray([d[k] for d in data])
    # return data_ret


    # data = []
    # psi0 = wf.recompute(configs)[1]  # recompute gives det

    # current_state = np.random.get_state()
    # for p in params:
    #     np.random.set_state(current_state)
    #     newparms = pgrad_acc.transform.deserialize(wf, p)
    #     for k in newparms:
    #         wf.parameters[k] = newparms[k]
    #     psi = wf.recompute(configs)[1]  # recompute gives det
    #     rawweights = np.exp(2 * (psi - psi0))  # convert from log(|psi|) to |psi|**2
    #     df = pgrad_acc.enacc(configs, wf)
    #     df["weight"] = rawweights
    #     data.append(df)
    # data_ret = {}
    # for k in data[0].keys():
    #     data_ret[k] = np.asarray([d[k] for d in data])
    

    data = []
    current_state = np.random.get_state()
    psi = np.zeros((len(params), len(configs.configs)))
    for i, p in enumerate(params):
        np.random.set_state(current_state)
        newparms = pgrad_acc.transform.deserialize(wf, p)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        psi[i] = wf.recompute(configs)[1]  # recompute gives logdet
        df = pgrad_acc.enacc(configs, wf)
        data.append(df)

    data_ret = {}
    for k in data[0].keys():
        data_ret[k] = np.asarray([d[k] for d in data])

    ref = np.amax(psi, axis=0)
    psirel = np.exp(2 * (psi - ref))
    rho = np.mean([psirel[i] for i in ref_wfs], axis=0)
    data_ret["weight"] = psirel / rho    
    
    return data_ret
