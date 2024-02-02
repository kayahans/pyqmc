# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import h5py


def initial_guess(mol, nconfig, r=1.0, seed = None):
    """Generate an initial guess by distributing electrons near atoms
    proportional to their charge.

    assign electrons to atoms based on atom charges
    assign the minimum number first, and assign the leftover ones randomly
    this algorithm chooses atoms *with replacement* to assign leftover electrons

    :parameter mol: A PySCF-like molecule object. Should have atom_charges(), atom_coords(), and nelec
    :parameter nconfig: How many configurations to generate.
    :parameter r: How far from the atoms to distribute the electrons
    :returns: (nconfig,nelectrons,3) array of electron positions randomly distributed near the atoms.
    :rtype: ndarray

    """
    from pyqmc.coord import OpenConfigs, PeriodicConfigs
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    wts = mol.atom_charges()
    wts = wts / np.sum(wts)
    
    for s in [0, 1]:
        neach = np.array(
            np.floor(mol.nelec[s] * wts), dtype=int
        )  # integer number of elec on each atom
        nleft = (
            mol.nelec[s] * wts - neach
        )  # fraction of electron unassigned on each atom
        nassigned = np.sum(neach)  # number of electrons assigned
        totleft = int(mol.nelec[s] - nassigned)  # number of electrons not yet assigned
        ind0 = s * mol.nelec[0]
        epos[:, ind0 : ind0 + nassigned, :] = np.repeat(
            mol.atom_coords(), neach, axis=0
        )  # assign core electrons
        if totleft > 0:
            bins = np.cumsum(nleft) / totleft
            inds = np.argpartition(
                rng.random((nconfig, len(wts))), totleft, axis=1
            )[:, :totleft]
            epos[:, ind0 + nassigned : ind0 + mol.nelec[s], :] = mol.atom_coords()[
                inds
            ]  # assign remaining electrons
    epos += r * rng.randn(*epos.shape)  # random shifts from atom positions
    if hasattr(mol, "a"):
        epos = PeriodicConfigs(epos, mol.lattice_vectors())
    else:
        epos = OpenConfigs(epos)
    return epos

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


def vmc_file(hdf_file, data, attr, configs):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)


def vmc_worker(wf, configs, tstep, nsteps, accumulators, bosonic=False):
    """
    Run VMC for nsteps.

    :return: a dictionary of averages from each accumulator.
    """
    nconf, nelec, _ = configs.configs.shape
    block_avg = {}
    wf.recompute(configs)
    for i in range(nsteps):
        acc = 0.0
        
        for e in range(nelec):
            # Propose move
            g1, _, _ = wf.gradient_value(e, configs.electron(e))
            
            grad = - limdrift(np.real(g1.T))
            rng = np.random #.RandomState(i)
            gauss = rng.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            index = 30
            # print('='*30, '\n', 'wf_inv1', wf._inverse[0][index])

            # Compute reverse move
            g2, new_val, saved = wf.gradient_value(e, newcoorde, configs=configs)
            new_grad = - limdrift(np.real(g2.T))
            forward = np.sum(gauss**2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)
            
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            ratio = np.abs(new_val) ** 2 * t_prob

            accept = ratio > np.random.rand(nconf)
            
            # Update wave function
            configs.move(e, newcoorde, accept)
            wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved)
            acc += np.mean(accept) / nelec
        # print(" ")
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

def vmc_parallel(
    wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
):
    config = configs.split(npartitions)
    runs = [
        client.submit(vmc_worker, wf, conf, tstep, nsteps_per_block, accumulators)
        for conf in config
    ]
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[1])
    confweight = np.array([len(c.configs) for c in config], dtype=float)
    confweight /= np.mean(confweight) * npartitions
    block_avg = {}
    for k in allresults[0][0].keys():
        block_avg[k] = np.sum(
            [res[k] * w for res, w in zip(allresults[0], confweight)], axis=0
        )
    return block_avg, configs


def vmc(
    wf,
    configs,
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    tstep=0.5,
    accumulators=None,
    verbose=False,
    stepoffset=0,
    hdf_file=None,
    continue_from=None,
    client=None,
    npartitions=None,
):
    """Run a Monte Carlo sample of a given wave function.

    :parameter wf: trial wave function for VMC
    :type wf: a PyQMC wave-function-like class
    :parameter configs: Initial electron coordinates
    :type configs: PyQMC configs object
    :parameter int nblocks: Number of VMC blocks to run
    :parameter int nsteps_per_block: Number of steps to run per block
    :parameter int nsteps: (Deprecated) Number of steps to run, maps to nblocks = 1, nsteps_per_block = nsteps
    :parameter float tstep: Time step for move proposals. Only affects efficiency.
    :parameter accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
    :parameter boolean verbose: Print out step information
    :parameter int stepoffset: If continuing a run, what to start the step numbering at.
    :parameter str hdf_file: Hdf_file to store vmc output.
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
            print("WARNING: running VMC with no accumulators")

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
                stepoffset = hdf["block"][-1] + 1
                configs.load_hdf(hdf)
                if verbose:
                    print(
                        f"Restarting calculation {continue_from} from step {stepoffset}"
                    )

    df = []

    for block in range(nblocks):
        if verbose:
            print(f"-", end="", flush=True)
        # import pdb
        # pdb.set_trace()            
        if client is None:
            block_avg, configs = vmc_worker(
                wf, configs, tstep, nsteps_per_block, accumulators
            )
        else:
            block_avg, configs = vmc_parallel(
                wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
            )
        # Append blocks
        block_avg["block"] = stepoffset + block
        block_avg["nconfig"] = nsteps_per_block * configs.configs.shape[0]
        vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
    if verbose:
        print("vmc done")

    df_return = {}
    for k in df[0].keys():
        df_return[k] = np.asarray([d[k] for d in df])
    return df_return, configs


#kayahan edited below
def abvmc(
    wf,
    configs,
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    tstep=0.5,
    accumulators=None,
    verbose=False,
    stepoffset=0,
    hdf_file=None,
    continue_from=None,
    client=None,
    npartitions=None,
):
    """Run a Monte Carlo sample of a given wave function.

    :parameter wf: trial wave function for VMC
    :type wf: a PyQMC wave-function-like class
    :parameter configs: Initial electron coordinates
    :type configs: PyQMC configs object
    :parameter int nblocks: Number of VMC blocks to run
    :parameter int nsteps_per_block: Number of steps to run per block
    :parameter int nsteps: (Deprecated) Number of steps to run, maps to nblocks = 1, nsteps_per_block = nsteps
    :parameter float tstep: Time step for move proposals. Only affects efficiency.
    :parameter accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
    :parameter boolean verbose: Print out step information
    :parameter int stepoffset: If continuing a run, what to start the step numbering at.
    :parameter str hdf_file: Hdf_file to store vmc output.
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
            print("WARNING: running VMC with no accumulators")

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
                stepoffset = hdf["block"][-1] + 1
                configs.load_hdf(hdf)
                if verbose:
                    print(
                        f"Restarting calculation {continue_from} from step {stepoffset}"
                    )

    df = []

    for block in range(nblocks):
        if verbose:
            print(f"-", end="", flush=True)
        if client is None:
            block_avg, configs = abvmc_worker(
                wf, configs, tstep, nsteps_per_block, accumulators
            )
        else:
            print("Parallel not yet implemented")
            exit()
        # Append blocks
        block_avg["block"] = stepoffset + block
        block_avg["nconfig"] = nsteps_per_block * configs.configs.shape[0]
        vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
    if verbose:
        print("vmc done")

    df_return = {}
    for k in df[0].keys():
        df_return[k] = np.asarray([d[k] for d in df])
    return df_return, configs

def abvmc_worker(wf, configs, tstep, nsteps, accumulators):
    """
    Run ABVMC for nsteps.

    :return: a dictionary of averages from each accumulator.

    Reuses much of the code in vmc_worker

    """
    
    nconf, nelec, _ = configs.configs.shape
    block_avg = {}
    wf.recompute(configs)
    # recompute calculates the wavefunction and fully calculates the inverse
    for i in range(nsteps):
        acc = 0.0
        
        for e in range(nelec):
            # Propose move
            g1, _, _ = wf.gradient_value(e, configs.electron(e))
            _, psi1, _ = wf.value()

            # g1 is the forward x,y,z gradient
            # val is the value of the A(R)^-1 * A(R) here, so it is always 1 in the forward step
            # A is defined as \psi = det(A(R)), where A is the matrix constructing the slater determinant
            # if \psi is a single determinant wavefunction
            # So the charge density is n(R) = \sum(A(R))**2 
            
            # lng1 = gradient of ln(\psi)
            # Below a bit hacky way to avoid properly renormalizing 
            # the \psi to generate steps and calculate accept/recept ratios
            # directly using the bosonic wavefunction.
            # Here, we calculate the derivative of ln(\psi)=lng1, so that
            # we cam use the vmc_worker code here with no modification
            lng1 = g1/np.tile(psi1, (3,1))
            grad = - limdrift(np.real(lng1.T))
            rng = np.random #.RandomState(i)
            gauss = rng.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)
            
            index = 30
            # print('='*30, '\n', 'wf_inv1', wf._inverse[0][index])
            
            # Now compute reverse move
            # For the backwards move, configs are passed to gradient_value.
            g2, new_val, saved = wf.gradient_value(e, newcoorde, configs=configs) 

            psi2 = saved['psi']          

            # g2 is the backward x,y,z gradient
            # new_val here is A(R)^-1 * A(R'), because the A(R)^-1 is not updated with R' yet
            # Therefore the np.abs(new_val) below becomes equal to \sum(A(R')/A(R))**2 

            lng2 = g2/np.tile(psi2, (3,1))
            new_grad = -limdrift(np.real(lng2.T))
            forward = np.sum(gauss**2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)
            
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            ratio = np.abs(new_val) ** 2 * t_prob

            accept = ratio > np.random.rand(nconf)
            # print('wf_inv2', wf._inverse[0][index])
            # print(
            #     #   i, '\n',
            #     #   e, '\n',
                  
            #     #   val1[index], '\n',
            #     #   np.log(val2[index]), '\n',
            #     #   np.log(valj[index]), '\n',
            #       lng1.T[index], '\n',
            #       newcoorde.configs[index], '\n',
            #     #   val[index], '\n',
            #     #   grad[index], '\n',
            #     #   gauss[index], '\n',
            #     #   tt[index], '\n',
            #     #   tstep, '\n',
            #     #   configs.electron(e).configs[index], '\n',
            #     #   newcoorde.configs[index], '\n',
            #     # new_val[index], '\n',
                
            #       lng2.T[index], '\n',
            #       new_grad[index], '\n',
            #       t_prob[index], '\n',
            #       ratio[index], '\n', 
            #       new_val[index], '\n'

            #     #   accept[0], 
            #     #   gauss[0], 
            #       )
            # from sys import exit
            # exit()
            # Update wave function
            configs.move(e, newcoorde, accept)
            wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved['values'])
            # update internals runs the sherman morrison update in block
            # this updates the determinant inverse based on the electronic update
            # without fully updating the inverse matrix. 
            acc += np.mean(accept) / nelec
        # print(" ")
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

def fixed_initial_guess(mol, nconfig, r=1.0):


    """
    Ideally works for H2 molecule with H-H aligned over z-axis and H1 at origin, H2 at [0,0,2]
    :parameter mol: A PySCF-like molecule object. Should have atom_charges(), atom_coords(), and nelec
    :parameter nconfig: How many configurations to generate.
    :parameter r: How far from the atoms to distribute the electrons
    :returns: (nconfig,nelectrons,3) array of electron positions randomly distributed near the atoms.
    :rtype: ndarray

    """
    from pyqmc.coord import OpenConfigs, PeriodicConfigs

    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    wts = mol.atom_charges()
    wts = wts / np.sum(wts)
    ind0 = 0
    for s in [0, 1]:
        neach = np.array(
            np.floor(mol.nelec[s] * wts), dtype=int
        )  # integer number of elec on each atom
        nleft = (
            mol.nelec[s] * wts - neach
        )  # fraction of electron unassigned on each atom
        nassigned = np.sum(neach)  # number of electrons assigned
        totleft = int(mol.nelec[s] - nassigned)  # number of electrons not yet assigned
        # 
        max = 3
        min = -1 
        if ind0 > 0:
            max = -10
            min = max
        epos[:, ind0, :] = np.linspace([-0.1,-0.1+ind0,min], [-0.1,-0.1+ind0,max], num=nconfig)
        ind0 += 1
    #     np.repeat(
    #         mol.atom_coords(), neach, axis=0
    #     )  # assign core electrons
    #     if totleft > 0:
    #         bins = np.cumsum(nleft) / totleft
    #         inds = np.argpartition(
    #             np.random.random((nconfig, len(wts))), totleft, axis=1
    #         )[:, :totleft]
    #         epos[:, ind0 + nassigned : ind0 + mol.nelec[s], :] = mol.atom_coords()[
    #             inds
    #         ]  # assign remaining electrons

    # epos += r * np.random.randn(*epos.shape)  # random shifts from atom positions
    # epos = np.linspace(0, 2, num=nconfig)
    # print(epos)
    if hasattr(mol, "a"):
        epos = PeriodicConfigs(epos, mol.lattice_vectors())
    else:
        epos = OpenConfigs(epos)
    
    return epos