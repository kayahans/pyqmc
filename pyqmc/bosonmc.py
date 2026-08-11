# MIT License
# 
# Copyright (c) 2019-2024 The PyQMC Developers
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import h5py
import logging
import copy


def limdrift(g, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    :parameter g: a [nconf,ndim] vector
    :parameter cutoff: the maximum magnitude
    :returns: The vector with the cutoff applied.
    """
    tot = np.linalg.norm(g, axis=1)
    mask = tot > cutoff
    g[mask, :] = cutoff * g[mask, :] / tot[mask, np.newaxis]
    return g


def boson_vmc_file(hdf_file, data, attr, configs):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)

def boson_vmc_worker(wf, configs, tstep, nsteps, accumulators):
    """
    Run VMC for nsteps.

    :return: a dictionary of averages from each accumulator.
    """
    nconf, nelec, _ = configs.configs.shape
    block_avg = {}
    wf.tstep = tstep
    
    # Pre-allocate arrays for better performance
    gauss = np.empty((nconf, 3))
    grad = np.empty((nconf, 3))
    new_grad = np.empty((nconf, 3))
    wf.recompute(configs) 

    for _ in range(nsteps):
        acc = 0.0
        # wf.curr_config = copy.deepcopy(configs)
        # wf.accept_array = np.zeros((nelec, nconf))
        
        for e in range(nelec):
            # Propose move
            # _, val_old = wf.recompute(configs)
            # wf_new = copy.deepcopy(wf) # TODO: check if this is correct 

            g, _, _ = wf.gradient_value(e, configs.electron(e))
            grad[:] = limdrift(np.real(g.T))
            
            # Generate random moves
            gauss[:] = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            # Compute reverse move
            g, ks_ratio, saved = wf.gradient_value(e, newcoorde)
            new_grad[:] = limdrift(np.real(g.T))
            
            # Compute acceptance ratio (Lucas Wagner's implementation)
            # forward = np.sum(gauss**2, axis=1)
            # backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)
            # t_prob_old = np.exp(1 / (2 * tstep) * (forward - backward))
            
            # Detailed balance/Metropolis-Hastings proof
            # q(y, x) = G(x, y; dt) * Psi^2(y)/ G(y, x; dt) * Psi^2(x)
            # G(x, y; dt) = exp(-1/(2dt) (x-y -dt(grad(x))^2)
            # y = x + dt grad(x) + gauss
            # x - y = -dt grad(x) - gauss
            # G(x, y; dt)/G(y, x; dt) = exp(-1/(2dt) (x-y -dt(grad(y))^2 - (y-x -dt(grad(x))^2))
            #                         = exp(-1/(2dt) (x-y -dt(grad(y))^2 - (x-y +dt(grad(x))^2))
            #                         = exp(-1/(2dt) (-2(x-y)dt[grad(x)+grad(y) + dt^2(grad(y)^2-grad(x)^2))
            #                         = exp(((x-y)[grad(x)+grad(y) - dt/2(grad(y)^2-grad(x)^2))
            #                         = exp([grad(x)+grad(y)]*[(x-y) - dt/2(grad(y)-grad(x))])
            #                         = exp([grad(x)+grad(y)]*[-dt grad(x) -gauss - dt/2(grad(y) - grad(x))]
            #                         = exp([grad(x)+grad(y)]*[-gauss - dt/2(grad(y) + grad(x))])
            # grad_sum = grad(x) + grad(y)
            #                         = exp(-[grad_sum]*[gauss + dt/2(grad_sum)])
            # When the sign in the exponent is negative, it agrees with the previous implementation

            # Current implementation
            grad_sum = grad + new_grad
            t_prob = np.exp(-np.sum(grad_sum * (gauss + tstep/2 * grad_sum), axis=1))

            # newcoord = copy.deepcopy(configs)
            # newcoord.configs[:,e,:] = newcoorde.configs
            # _, val_new = wf.value_configs(newcoord)
            # ratio = np.exp(2*(val_new-val_old)) * t_prob 
            # accept = ratio > np.random.rand(nconf)
            ratio = ks_ratio * t_prob
            accept = ratio > np.random.rand(nconf)
            # Restore wave function (not needed with value_configs)
            # wf.recompute(configs) # TODO: check if this is correct 

            # Update configuration and wave function
            configs.move(e, newcoorde, accept)
            wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved)
            acc += np.mean(accept) / nelec
            # option 1 no electon resolution wf.accept_array += accept.astype(float)/nelec 
            # option 2 e resolution wf.accept_array[e] += accept.astype(float)
        # Rolling average on step
        for k, accumulator in accumulators.items():
            dat = accumulator.avg(configs, wf)
            for m, res in dat.items():
                if k + m not in block_avg:
                    block_avg[k + m] = res / nsteps
                else:
                    block_avg[k + m] += res / nsteps
        block_avg["acceptance"] = acc
        
    return block_avg, configs

def abvmc_parallel(
    wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
):
    """
    Run VMC in parallel using distributed computing.
    
    Args:
        wf: Wave function object
        configs: Configuration object
        tstep: Time step
        nsteps_per_block: Number of steps per block
        accumulators: Dictionary of accumulators
        client: Distributed computing client
        npartitions: Number of partitions for parallel processing
        
    Returns:
        block_avg: Dictionary of averaged results
        configs: Updated configuration object
    """
    # Split configurations into partitions
    config = configs.split(npartitions)
    
    # Submit all tasks at once for better parallelization
    runs = [
        client.submit(boson_vmc_worker, wf, conf, tstep, nsteps_per_block, accumulators)
        for conf in config
    ]
    
    allresults = list(zip(*[r.result() for r in runs]))
    
    # Join configurations
    configs.join(allresults[1])
    
    # Calculate weights for averaging
    confweight = np.array([len(c.configs) for c in config], dtype=float)
    confweight /= np.mean(confweight) * npartitions
    
    # Combine results with weights
    block_avg = {}
    for k in allresults[0][0].keys():
        block_avg[k] = np.sum(
            [res[k] * w for res, w in zip(allresults[0], confweight)], axis=0
        )
    
    return block_avg, configs

# def abvmc_parallel(
#     wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
# ):
#     config = configs.split(npartitions)
#     runs = [
#         client.submit(boson_vmc_worker, wf, conf, tstep, nsteps_per_block, accumulators)
#         for conf in config
#     ]
#     allresults = list(zip(*[r.result() for r in runs]))
#     configs.join(allresults[1])
#     confweight = np.array([len(c.configs) for c in config], dtype=float)
#     confweight /= np.mean(confweight) * npartitions
#     block_avg = {}
#     for k in allresults[0][0].keys():
#         block_avg[k] = np.sum(
#             [res[k] * w for res, w in zip(allresults[0], confweight)], axis=0
#         )
#     return block_avg, configs


def check_convergence(param_array, convergence_threshold):
    """
    Check convergence of parameters using reblocking to handle autocorrelation.
    
    Args:
        param_array: Array of parameter values with shape (nblocks, ...)
        convergence_threshold: Threshold for convergence
        
    Returns:
        bool: True if converged, False otherwise
    """
    def reblock_data(data, nblocks):
        """Helper function to reblock data into nblocks"""
        n = len(data)
        if nblocks > n:
            return data
        block_size = n // nblocks
        return np.array([np.mean(data[i:i+block_size], axis=0) for i in range(0, n, block_size)])
    
    def find_optimal_block_size(data):
        """Find optimal block size using the error in error method"""
        n = len(data)
        max_blocks = min(100, n // 2)  # Limit maximum blocks to avoid too small blocks
        block_sizes = [n // i for i in range(2, max_blocks + 1)]
        
        errors = []
        for size in block_sizes:
            blocks = reblock_data(data, n // size)
            error = np.std(blocks, axis=0) / np.sqrt(len(blocks) - 1)
            errors.append(error)
        
        # Find where error in error stabilizes
        error_diffs = np.diff(errors, axis=0)
        if len(error_diffs) == 0:
            return n // 2
        
        # Find first point where error difference is small
        for i, diff in enumerate(error_diffs):
            if np.all(np.abs(diff) < 1e-10):
                return block_sizes[i]
        
        return block_sizes[-1]  # Return largest block size if no stabilization found
    
    if len(param_array.shape) == 1:
        # 1D array case
        optimal_blocks = reblock_data(param_array, find_optimal_block_size(param_array))
        mean_val = np.mean(optimal_blocks)
        std_val = np.std(optimal_blocks) / np.sqrt(len(optimal_blocks) - 1)
        print(mean_val, std_val, param_array.shape, optimal_blocks.shape)
        if std_val == 0.0:
            result = False

        # Handle values close to zero
        if abs(mean_val) < convergence_threshold:
            result = std_val < convergence_threshold
        else:
            # Use relative error for non-zero values
            result = std_val / abs(mean_val) < convergence_threshold
        return result, optimal_blocks.shape, (mean_val, std_val)
    
    elif len(param_array.shape) == 2:        
        # Find optimal block size for each component
        optimal_blocks = []
        for i in range(param_array.shape[1]):
            opt_blocks = find_optimal_block_size(param_array[:, i])
            component_blocks = reblock_data(param_array[:, i], opt_blocks)
            optimal_blocks.append(component_blocks)
        
        optimal_blocks = np.array(optimal_blocks).T
        array_mean = np.mean(optimal_blocks, axis=0)
        array_std = np.std(optimal_blocks, axis=0) / np.sqrt(len(optimal_blocks) - 1)
        
        # Handle each component separately
        result = True
        for mean, std in zip(array_mean, array_std):
            if std == 0.0:
                result = False
            if abs(mean) < convergence_threshold:
                if std >= convergence_threshold:
                    result = False
            else:
                if std / abs(mean) >= convergence_threshold:
                    result = False
        return result, optimal_blocks.shape, (array_mean, array_std)
    else:
        raise ValueError(f"Unexpected array shape: {param_array.shape}")


def abvmc(
    wf,
    configs,
    tstep=0.5,
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    blockoffset=0,
    accumulators=None,
    verbose=False,
    hdf_file=None,
    continue_from=None,
    client=None,
    npartitions=None,
    converged_parameter = None, 
    convergence_threshold = 1e-3,
):
    """Run a Monte Carlo sample of a given wave function.

    :parameter wf: trial wave function for VMC
    :type wf: a PyQMC wave-function-like object
    :parameter configs: (nconfig, nelec, 3) - initial electron coordinates to start calculation.
    :type configs: PyQMC configs object
    :parameter float tstep: Time step for move proposals. Only affects efficiency.
    :parameter int nblocks: Number of VMC blocks to run. If a calculation is continued (either from continue_from or from using the same hdf_file as a previous call), nblocks includes the blocks from previous calls; i.e., nblocks is the total number of blocks run over all the calls to vmc.
    :parameter int nsteps_per_block: Number of steps to run per block
    :parameter int nsteps: (Deprecated) Number of steps to run, maps to nblocks = nsteps, nsteps_per_block = 1
    :parameter int blockoffset: If continuing a run, what to start the block numbering at. The calculation will stop when the block number reaches nblocks.
    :parameter accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
    :parameter boolean verbose: Print out step information
    :parameter str hdf_file: Hdf_file to store vmc output.
    :parameter str continue_from: Hdf_file to continue vmc calculation from.
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :returns: (df,configs)
       df: A list of dictionaries nstep long that contains all results from the accumulators. These are averaged across all walkers.

       configs: The final coordinates from this calculation.
    :rtype: list of dictionaries, pyqmc.coord.Configs

    """
    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    if accumulators is None:
        accumulators = {}
        if verbose:
            print("WARNING: running ABVMC with no accumulators")

    # Restart
    if continue_from is None:
        continue_from = hdf_file
    elif not os.path.isfile(continue_from):
        raise RuntimeError("cannot continue from {0}; the file does not exist!")
    elif hdf_file is not None and os.path.isfile(hdf_file):
        raise RuntimeError(
            "continue_from is not None but hdf_file={0} already exists! Delete or rename {0} and try again.".format(
                hdf_file
            )
        )
    if continue_from is not None and os.path.isfile(continue_from):
        with h5py.File(continue_from, "r") as hdf:
            if "configs" in hdf.keys():
                blockoffset = hdf["block"][-1] + 1
                configs.load_hdf(hdf)
                if verbose:
                    print(
                        f"Restarting calculation {continue_from} from block {blockoffset}"
                    )
    # Print simulation parameters
    print(f"ABVMC simulation parameters:")
    print(f"Number of walkers: {configs.configs.shape[0]}")
    print(f"tstep: {tstep}")
    print(f"nblocks: {nblocks}")
    print(f"nsteps_per_block: {nsteps_per_block}")
    print(f"nsteps: {nsteps}")
    print(f"blockoffset: {blockoffset}")
    print(f"converged_parameter: {converged_parameter}")
    print(f"convergence_threshold: {convergence_threshold}")
    print(f"Using accumulators: {accumulators}")
    
    def vmc_run(wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions, block, df):
        if verbose:
            print(f"-", end="", flush=True)
        if client is None:
            block_avg, configs = boson_vmc_worker(
                wf, configs, tstep, nsteps_per_block, accumulators
            )
        else:
            block_avg, configs = abvmc_parallel(
                wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
            )
        # Append blocks
        block_avg["block"] = block
        block_avg["nconfig"] = nsteps_per_block * configs.configs.shape[0]
        boson_vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
        
    df = []
    block = 0
    if blockoffset >= nblocks:
        logging.warning(f"blockoffset {blockoffset} >= nblocks {nblocks}; no steps will be run.")
        block = blockoffset
    
    if converged_parameter is not None:
        print(f"Checking convergence for {converged_parameter} with threshold {convergence_threshold}")
        converged = False
        min_blocks = 10  # Minimum number of blocks before checking convergence
        plot = True
        if verbose:
            if plot:
                import matplotlib.pyplot as plt
                plt.ion()  # Turn on interactive mode
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
                fig.suptitle(f'Convergence of {converged_parameter}')

                # Initialize lists to store data for plotting
                blocks = []
                means = []
                stds = []
                block_sizes = []
                
                # Create initial empty plots
                line1, = ax1.plot([], [], 'b-o', label='Mean')
                line2, = ax2.plot([], [], 'r-o', label='Standard Error')
                line3, = ax3.plot([], [], 'g-o', label='Optimal Block Size')
                
                # Set up the axes
                ax1.set_xlabel('Block')
                ax1.set_ylabel('Value')
                ax1.set_title(f'{converged_parameter} Value')
                ax1.grid(True)
                ax1.legend()
                
                ax2.set_xlabel('Block')
                ax2.set_ylabel('Standard Error')
                ax2.set_title(f'Standard Error of {converged_parameter}')
                ax2.grid(True)
                ax2.legend()

                ax3.set_xlabel('Block')
                ax3.set_ylabel('Block Size')
                ax3.set_title('Optimal Block Size')
                ax3.grid(True)
                ax3.legend()

        last_check_convergence = block
        while not converged:
            vmc_run(wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions, block, df)
            block += 1
            
            if block >= min_blocks:  # Only check convergence after minimum blocks
                param_array = np.asarray([d[converged_parameter] for d in df])
                
                if block > 2*last_check_convergence:
                    print(f"Checking convergence for {converged_parameter} at block {block} when last checked at {last_check_convergence}")
                    converged, optimal_block_sizes, (mean_val, std_val) = check_convergence(param_array, convergence_threshold)
                    last_check_convergence = block
                    if verbose:
                        if len(param_array.shape) == 1:
                            print(f"Mean: {mean_val:.6f}")
                            print(f"Std: {std_val:.6f}")
                            print(f"Optimal block size: {optimal_block_sizes}")
                        else:
                            print(f"Mean: {mean_val}")
                            print(f"Std: {std_val}")
                            print(f"Optimal block sizes: {optimal_block_sizes}")
                        if plot:
                            # Update data lists
                            blocks.append(block)
                            means.append(mean_val)
                            stds.append(std_val)
                            block_sizes.append(optimal_block_sizes[0])  # For 1D arrays


                            # Update the plots
                            line1.set_data(blocks, means)
                            line2.set_data(blocks, stds)
                            line3.set_data(blocks, block_sizes)
                            
                            # Adjust axis limits
                            ax1.relim()
                            ax1.autoscale_view()
                            ax2.relim()
                            ax2.autoscale_view()
                            ax3.relim()
                            ax3.autoscale_view()
                            
                            # Draw and pause to update the plot
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            plt.pause(0.1)  # Small pause to allow the plot to update

    else:
        for block in range(blockoffset, nblocks):
            vmc_run(wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions, block, df)
        
        

    if verbose:
        print("vmc done")

    df_return = {}
    if len(df) > 0:
        for k in df[0].keys():
            df_return[k] = np.asarray([d[k] for d in df])
    return df_return, configs
