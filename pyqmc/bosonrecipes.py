import os
import pyqmc
import numpy as np
from pyqmc import bosonwftools
import pyqmc.pyscftools as pyscftools
import pyqmc.supercell as supercell
import h5py
import pandas as pd
from pyqmc import bosonmc
from pyqmc import bosonlinemin
from pyqmc import bosondmc
from pyqmc import wftools

from pyqmc import bosonaccumulators

def ABOPTIMIZE(
    dft_checkfile: str,
    output: str,
    nconfig: int = 1000,
    ci_checkfile:str|None=None,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws = {"ion_cusp":False, 'na':0},
    slater_kws:  list|None = None,
    det_emax: float|None=None,
    xc: str = 'LDA,VWN',
    use_symm = False,
    initial_guess_r = 10.0,
    njastrow = 2,
    opt_options: list|None = None,
    opt_method: str = "linemin",
    use_dft_density = False,
    **linemin_kws,
):
    """Auxiliary Boson wavefunction Slater Jastrow optimization

    Args:
        dft_checkfile (str): dft chk filename
        output (str): output chk filename 
        nconfig (int, optional): number of configurations. Defaults to 1000.
        load_parameters (str, optional): load wavefunction parameters from a chk file. Defaults to None.
        S (_type_, optional): _description_. Defaults to None.
        jastrow_kws (list | None, optional): _description_. Defaults to None.
        slater_kws (list | None, optional): _description_. Defaults to None.
        opt_method (str, optional): Optimization method to use. Options are "linemin". Defaults to "linemin".

    Raises:
        RuntimeError: _description_
    """


    anchors = None
    target_root = None
    nodal_cutoff = 1e-3

    linemin_kws["hdf_file"] = output
    if load_parameters is not None and output is not None and os.path.isfile(output):
        raise RuntimeError(
            "load_parameters is not None and output={0} already exists! Delete or rename {0} and try again.".format(
                output
            )
        )
    if target_root is None and anchors is not None:
        target_root = len(anchors)
    else:
        target_root = 0

    wf, configs, acc = initialize_boson_qmc_objects(
        dft_checkfile,
        opt_wf = True,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        det_emax=det_emax,
        opt_options=opt_options,
        xc=xc,
        use_symm=use_symm,
        initial_guess_r=initial_guess_r,
        njastrow=njastrow,
        use_dft_density=use_dft_density,
    )
    if anchors is None:
        if opt_method == "linemin":
            wf, df = bosonlinemin.line_minimization(wf, configs, acc, **linemin_kws)
        else:
            raise ValueError(f"Unknown optimization method: {opt_method}. Valid options are 'linemin'")
            
    return wf, df

def ABVMC(
    dft_checkfile: str,
    output: str,
    nconfig=1000,
    ci_checkfile: str|None=None,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws = {"ion_cusp":False, 'na':0},
    slater_kws:  list|None = None,
    accumulators: list|None = None,
    seed: int|None=None,
    det_emax: float|None=None,
    warmup_options = None,
    xc: str = 'LDA,VWN',
    use_symm = False,
    initial_guess_r = 15.0,
    njastrow = 2,
    **vmc_kws,
):
    """Auxiliary Boson VMC recipe

    Args:
        dft_checkfile (str): dft chk filename
        output (str): output chk filename 
        nconfig (int, optional): number of configurations. Defaults to 1000.
        ci_checkfile (str | None, optional): CI chkfile. Defaults to None.
        load_parameters (str | None, optional): load wavefunction parameters from a chk file. Defaults to None.
        S (_type_, optional): _description_. Defaults to None.
        jastrow_kws (list | None, optional): _description_. Defaults to None.
        slater_kws (list | None, optional): _description_. Defaults to None.
        accumulators (list | None, optional): List of accumulators. Defaults to None.
    """
    vmc_kws["hdf_file"] = output
    print("Running ABVMC")
    print('Statistical accumulators:', accumulators)
    wf, configs, acc = initialize_boson_qmc_objects(
        dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators,
        seed=seed,
        det_emax=det_emax,
        xc=xc,
        use_symm=use_symm,
        initial_guess_r=initial_guess_r,
        njastrow=njastrow,
    )
    if warmup_options is None:
        warmup_options = dict(nblocks=0, tstep=0.5, accumulators=None)
    
    if warmup_options['nblocks'] > 0:
        print('Running warmup')
        try:
            num_det = wf.num_det
            wf_dtype = wf.dtype
        except:
            for wave in wf.wf_factors:
                try:
                    num_det = wave.num_det
                    wf_dtype = wave.dtype
                except:
                    pass
        system_params = {
            'nconf': configs.configs.shape[0],
            'ndets': num_det,
            'nelec': configs.configs.shape[1],
            'dtype': wf_dtype
        }

        if warmup_options['accumulators'] is not None:
            warmup_acc = {}
            possible_accumulators = {
                             'energy':bosonaccumulators.ABQMCEnergyAccumulator(wf.mf_inputs),
                             'ab_vmc_excitations':bosonaccumulators.ABVMCMatrixAccumulator(wf.mf_inputs, use_symm=use_symm),
                             'abc_dmc_excitations':bosonaccumulators.ABCDMCMatrixAccumulator(wf.mf_inputs, system_params, use_symm=use_symm),
                             'density':bosonaccumulators.DensityAccumulator(),
                             'radial_density':bosonaccumulators.RadialDensityAccumulator()}
            print('Warmup accumulators:', warmup_options['accumulators'])
            for acc_name in warmup_options['accumulators']:
                warmup_acc[acc_name] = possible_accumulators[acc_name]
            warmup_options['accumulators'] = warmup_acc
        _, configs = bosonmc.abvmc(
                wf,
                configs,
                **warmup_options
        )
    print('Warmup complete')
    print('Running VMC')
    bosonmc.abvmc(wf, configs, accumulators=acc, **vmc_kws)
    return wf, configs, acc

def ABDMC(
    dft_checkfile: str,
    output: str,
    nconfig=1000,
    ci_checkfile: str|None=None,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws: dict|None = None,
    slater_kws:  dict|None = None,
    accumulators: list|None = None,
    seed: int|None=None,
    det_emax: float|None=None,
    xc: str = 'LDA,VWN',
    use_symm = False,
    initial_guess_r = 15.0,
    use_dft_density = False,
    **dmc_kws,
):  
    """Auxiliary Boson DMC recipe

    Args:
        dft_checkfile (str): dft chk filename
        output (str): output chk filename 
        nconfig (int, optional): number of configurations. Defaults to 1000.
        ci_checkfile (str | None, optional): CI chkfile. Defaults to None.
        load_parameters (str | None, optional): load wavefunction parameters from a chk file. Defaults to None.
        S (_type_, optional): _description_. Defaults to None.
        jastrow_kws (list | None, optional): _description_. Defaults to None.
        slater_kws (list | None, optional): _description_. Defaults to None.
        accumulators (list | None, optional): List of accumulators. Defaults to None.
    """    
    dmc_kws["hdf_file"] = output
    print("Running ABDMC")
    wf, configs, acc = initialize_boson_qmc_objects(
        dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators,
        seed=seed,
        det_emax=det_emax,
        xc=xc,
        use_dft_density=use_dft_density,
        use_symm=use_symm,
        initial_guess_r=initial_guess_r,
    )
    # Extract VMC options from DMC keyword arguments if present, otherwise return None
    vmc_options = dmc_kws.pop('vmc_options', None)
    if vmc_options is not None and vmc_options['accumulators'] is not None:
        vmc_acc = {}
        possible_accumulators = {
                            'energy':bosonaccumulators.ABQMCEnergyAccumulator(wf.mf_inputs),
                            'ab_vmc_excitations':bosonaccumulators.ABVMCMatrixAccumulator(wf.mf_inputs, use_symm=use_symm),
                            'density':bosonaccumulators.DensityAccumulator(),
                            'radial_density':bosonaccumulators.RadialDensityAccumulator()}
        print('DMC accumulators:', vmc_options['accumulators'])
        for acc_name in vmc_options['accumulators']:
            vmc_acc[acc_name] = possible_accumulators[acc_name]
        vmc_options['accumulators'] = vmc_acc
        dmc_kws['vmc_options'] = vmc_options
    bosondmc.rundmc(wf, configs, accumulators=acc, **dmc_kws)

def create_pyscf_grid(mol, level=4):
    """
    Create PySCF DFT grid for molecular system.
    
    PySCF grids are atomic-centered and optimized for molecular calculations:
    - Higher density near nuclei where orbitals vary rapidly
    - Sparser in regions far from atoms
    - Comes with integration weights for proper normalization
    
    Args:
        mol: PySCF molecule object
        level: Grid level (1-9), higher = more accurate but slower
               3 = good default, 5 = high accuracy, 1 = coarse
        
    Returns:
        coords: Grid coordinates (n_points, 3)
        weights: Integration weights (n_points,)
    """
    print(f"\nCreating PySCF DFT grid...")
    print(f"  Grid level: {level}")
    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    
    coords = grids.coords
    weights = grids.weights
    
    print(f"  Total grid points: {len(coords)}")
    print(f"  Grid point range (Bohr):")
    print(f"    X: [{np.min(coords[:,0]):.2f}, {np.max(coords[:,0]):.2f}]")
    print(f"    Y: [{np.min(coords[:,1]):.2f}, {np.max(coords[:,1]):.2f}]")
    print(f"    Z: [{np.min(coords[:,2]):.2f}, {np.max(coords[:,2]):.2f}]")
    print(f"  Total weight (should ≈ volume): {np.sum(weights):.2f}")
    
    return coords, weights

def calculate_density_on_grid(mf, coords, weights, frozen=1, ncas=6, nelecas=(4,1), 
                              ecut=None, chunk_size=50000):
    """
    Calculate multi-determinant probability density on PySCF grid points.
    
    Memory-optimized: uses orbital weighting and chunked processing to avoid
    large intermediates (n_points × n_det × n_elec).
    
    Args:
        mf: Mean field object from PySCF
        coords: Grid coordinates (n_points, 3)
        weights: Integration weights (n_points,)
        frozen: Number of frozen (core) orbitals
        ncas: Number of active space orbitals
        nelecas: Number of active electrons (n_alpha, n_beta)
        ecut: Energy cutoff for determinant selection (if None, use all)
        chunk_size: Grid points per chunk for memory efficiency (default: 50000)
        
    Returns:
        density: 1D array of probability density |Ψ|² at grid points
    """
    from itertools import combinations
    print(f"\nCalculating density on grid...")
    print(f"  Frozen orbitals: {frozen}")
    print(f"  Active space: NCAS={ncas}, NELECAS={nelecas}")
    if ecut is not None:
        print(f"  Energy cutoff: {ecut} Hartree")
    
    # Get molecular orbital coefficients
    mo_coeff = mf.mo_coeff  # Shape: (2, n_ao, n_mo) for UHF/UKS
    n_mo_needed = frozen + ncas
    mo_coeff_up = mo_coeff[0][:, :n_mo_needed]  # Only orbitals we need
    mo_coeff_dn = mo_coeff[1][:, :n_mo_needed]
    
    # Generate determinants
    up_orbs = dn_orbs = np.arange(ncas) + frozen
    up_det = list(combinations(up_orbs, nelecas[0]))
    dn_det = list(combinations(dn_orbs, nelecas[1]))
    
    # Include frozen orbitals in determinants
    frozen_array = list(range(frozen))
    up_det = [np.array(frozen_array + list(x)) for x in up_det]
    dn_det = [np.array(frozen_array + list(x)) for x in dn_det]
    
    # Apply energy cutoff if specified
    if ecut is not None:
        mo_energy = mf.mo_energy
        up_det_arr = np.array(up_det)
        dn_det_arr = np.array(dn_det)
        up_energies = np.sum(mo_energy[0][up_det_arr], axis=1)
        dn_energies = np.sum(mo_energy[1][dn_det_arr], axis=1)
        det_mf_energies = up_energies[:, np.newaxis] + dn_energies[np.newaxis, :]
        det_mf_energies -= np.min(det_mf_energies)
        mask = np.argwhere(det_mf_energies < ecut)
        
        # Filter determinants
        up_det_filtered = [up_det[ij[0]] for ij in mask]
        dn_det_filtered = [dn_det[ij[1]] for ij in mask]
    else:
        # Use all determinants
        up_det_filtered = [up for up in up_det for dn in dn_det]
        dn_det_filtered = [dn for up in up_det for dn in dn_det]
    
    n_det = len(up_det_filtered)
    print(f"  Number of determinants: {n_det}")

    # Orbital weights: density = sum_i (count_i/n_det) * |phi_i|^2
    # Avoids huge (n_points, n_det, n_elec) intermediate
    orb_weights_up = np.zeros(n_mo_needed)
    orb_weights_dn = np.zeros(n_mo_needed)
    for up, dn in zip(up_det_filtered, dn_det_filtered):
        for i in up:
            orb_weights_up[i] += 1
        for j in dn:
            orb_weights_dn[j] += 1
    orb_weights_up /= n_det
    orb_weights_dn /= n_det

    # Chunked processing to limit peak memory
    n_points = len(coords)
    density = np.zeros(n_points)
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        coords_chunk = coords[start:end]
        
        ao_value = mf.mol.eval_gto('GTOval_sph', coords_chunk)
        mo_up = np.dot(ao_value, mo_coeff_up)  # (chunk, n_mo_needed)
        mo_dn = np.dot(ao_value, mo_coeff_dn)
        
        # density = sum_i weight_i * |phi_i|^2
        density[start:end] = (
            np.dot(mo_up ** 2, orb_weights_up) +
            np.dot(mo_dn ** 2, orb_weights_dn)
        )
    
    print(f"  Density range: [{np.min(density):.6e}, {np.max(density):.6e}]")
    
    # Calculate integrated density using weights
    integrated_density = np.sum(density * weights)
    print(f"  Integrated density (∫ρ dV): {integrated_density:.6f}")
    print(f"    (Should be close to total # electrons for proper normalization)")
    
    return density

def generate_walker_configs(density, coords, weights, n_walkers, n_electrons, seed=None):
    """
    Generate walker configurations by sampling from the density on PySCF grid.
    
    Each walker contains n_electrons, and the distribution of electrons 
    across all walkers matches the probability density.
    
    Args:
        density: 1D array of probability density at grid points
        coords: Grid coordinates (n_points, 3)
        weights: Integration weights (n_points,)
        n_walkers: Number of walkers to generate
        n_electrons: Number of electrons per walker
        seed: Random seed for reproducibility
        
    Returns:
        walker_configs: Array of shape (n_walkers, n_electrons, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nGenerating walker configurations...")
    print(f"  Number of walkers: {n_walkers}")
    print(f"  Electrons per walker: {n_electrons}")
    
    # Normalize density to get probability distribution
    # Weight the density by the integration weights
    weighted_density = density * weights
    weighted_density = np.maximum(weighted_density, 0)  # Ensure non-negative
    total_density = np.sum(weighted_density)
    
    if total_density <= 0:
        raise ValueError("Total weighted density is zero or negative!")
    
    prob_dist = weighted_density / total_density
    
    # Build CDF for sampling
    cdf = np.cumsum(prob_dist)
    
    # Total number of electron positions to sample
    total_samples = n_walkers * n_electrons
    
    # Sample indices from CDF
    u = np.random.rand(total_samples)
    indices = np.searchsorted(cdf, u)
    
    # Get coordinates for sampled indices
    sampled_coords = coords[indices]
    
    # Add small random jitter based on local grid spacing
    # Estimate local grid spacing from nearest neighbors
    # For simplicity, use a small fixed jitter (0.1 Bohr)
    jitter_scale = 0.1  # Bohr
    jitter = jitter_scale * (2 * np.random.rand(total_samples, 3) - 1)
    sampled_coords += jitter
    
    # Reshape into walker configurations
    walker_configs = sampled_coords.reshape(n_walkers, n_electrons, 3)
    
    print(f"  Walker configs shape: {walker_configs.shape}")
    print(f"  Position range (Bohr):")
    print(f"    X: [{np.min(walker_configs[:,:,0]):.2f}, {np.max(walker_configs[:,:,0]):.2f}]")
    print(f"    Y: [{np.min(walker_configs[:,:,1]):.2f}, {np.max(walker_configs[:,:,1]):.2f}]")
    print(f"    Z: [{np.min(walker_configs[:,:,2]):.2f}, {np.max(walker_configs[:,:,2]):.2f}]")
    
    return walker_configs



def initial_guess(mol, nconfig, r=None, seed = None, use_dft_density=False, mf = None, ncas = None, nelecas = None, frozen = 0):
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
    if use_dft_density:
        coords, weights = create_pyscf_grid(mol, level=9)
        density = calculate_density_on_grid(mf, coords, weights, frozen=frozen, ncas=ncas, nelecas=nelecas, ecut=None)
        epos = generate_walker_configs(density, coords, weights, nconfig, np.sum(mol.nelec), seed=seed)
    else:
        if r == None:
            r = 15.0
        print("Initializing guess with r = ", r)
        
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

    show_radial_profile = False
    if show_radial_profile:
        import matplotlib.pyplot as plt
        from pyqmc.coord import OpenConfigs
        radial_density = bosonaccumulators.RadialDensityAccumulator()
        results = radial_density(epos, None)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(results['r'], results['radial_density'])
        axs[1].plot(results['r'], results['int_density'])
        plt.show()
        
    return epos


def initialize_boson_qmc_objects(
    dft_checkfile,
    nconfig=1000,
    load_parameters=None,
    ci_checkfile=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    accumulators=None,
    opt_wf=False,
    seed = None,
    det_emax = None,
    use_dft_density = False,
    initial_guess_r = 10.0,
    use_symm = False,
    xc = 'LDA,VWN',
    opt_options = None,
    njastrow = 2
):  
    
    target_root=0
    nodal_cutoff=1e-3    

    if ci_checkfile is None:
        mol, mf = pyscftools.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyscftools.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        if not hasattr(mc.ci, "shape") or len(mc.ci.shape) == 3:
            mc.fci = mc.ci
            # print('Selecting target CI root #', target_root)
            mc.ci = mc.ci[target_root]
    
    # Try to load mf_inputs from checkfile first (new format)
    mf_inputs = pyscftools.load_mf_inputs_from_hdf5(dft_checkfile, mol=mol)
    
    if mf_inputs is None:
        # Fall back to old method: construct mf_inputs from PySCF objects
        # Remove any spaces from xc string
        xc = xc.replace(" ", "")
        available_xc = ['LDA,VWN','PBE,PBE','HF']
        mf_inputs = {}
        if xc not in available_xc:
            raise ValueError(f"xc={xc} not in available_xc={available_xc}")

        try:
            mf_inputs['dm'] = mf.make_rdm1()
        except:
            print("WARNING: mf.make_rdm1() is not available, cannot use DFT as Mean Field")

        rho, grids = bosonaccumulators.calculate_mf_density(mol, mf_inputs['dm'])

        mf_inputs.update({'xc':xc,
                     'mol':mf.mol,
                     'nelec': mf.nelec,
                     'mo_energy': mf.mo_energy,
                     'mo_occ': mf.mo_occ, 
                     'grids': grids, 
                     'rho' : rho })
    else:
        # mf_inputs loaded from checkfile
        # Ensure xc parameter takes precedence if provided and different
        xc = xc.replace(" ", "")
        if xc != mf_inputs.get('xc', '').replace(" ", ""):
            print(f"Warning: xc parameter ({xc}) differs from checkfile ({mf_inputs.get('xc', 'N/A')}). Using parameter value.")
            mf_inputs['xc'] = xc
        
        # Validate xc
        available_xc = ['LDA,VWN','PBE,PBE','HF']
        if mf_inputs['xc'] not in available_xc:
            raise ValueError(f"xc={mf_inputs['xc']} from checkfile not in available_xc={available_xc}")
        
        # Ensure mol is set correctly (should already be set by load_mf_inputs_from_hdf5)
        if 'mol' not in mf_inputs:
            mf_inputs['mol'] = mol
        
        print("Loaded mf_inputs from checkfile")
    
    if jastrow_kws == None:
        jastrow_kws = dict()
    
    # if "ion_cusp" in jastrow_kws.keys():
    #     if jastrow_kws["ion_cusp"] != False:
    #         print("WARNING: ion_cusp = True is not the default behavior")
    # else:
    #     print("WARNING: Using ion_cusp = False as default")
    #     jastrow_kws["ion_cusp"] = True
    

    if S is not None:
        mol = supercell.get_supercell(mol, np.asarray(S))
    # Use when testing HF
    if load_parameters is False:
        wf, to_opt = bosonwftools.generate_boson_wf(
            mol, mf, mc=mc, jastrow = None, jastrow_kws=jastrow_kws, slater_kws=slater_kws, det_emax=det_emax, use_symm=use_symm
        )
        num_det = wf.num_det
        wf_dtype = wf.dtype
    else:
        if njastrow == 2:
            wf, to_opt = bosonwftools.generate_boson_wf(
                mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws, det_emax=det_emax, use_symm=use_symm
            )
        elif njastrow == 3:
            from pyqmc.wftools import generate_jastrow, generate_jastrow3
            wf, to_opt = bosonwftools.generate_boson_wf(
                mol, mf, mc=mc, jastrow = [generate_jastrow, generate_jastrow3], jastrow_kws=jastrow_kws, slater_kws=slater_kws, det_emax=det_emax, use_symm=use_symm
            )
        if load_parameters is not None:
            print('Loading WF parameters from', load_parameters)
            wftools.read_wf(wf, load_parameters)    
        for wave in wf.wf_factors:
            try:
                num_det = wave.num_det
                wf_dtype = wave.dtype
            except:
                pass

    if opt_options is not None:
        allowed_opt_options = ['only_acoeff', 'only_bcoeff']
        for opt_option in opt_options:
            if opt_option in allowed_opt_options:
                if opt_option == 'only_acoeff':
                    to_opt['bcoeff'] = False
                elif opt_option == 'only_bcoeff':
                    to_opt['acoeff'] = False
            else:
                raise ValueError(f"Unknown opt_option: {opt_option}")
    
    
    if use_dft_density:
        print('Using DFT density guess')
    else:
        print('Using spherical guess')

    if mc is not None:   
        configs = initial_guess(mol, nconfig, r=initial_guess_r, seed=seed, use_dft_density=use_dft_density, mf=mf, ncas = mc.ncas, nelecas = mc.nelecas, frozen = mc.ncore)
    else:
        configs = initial_guess(mol, nconfig, r=initial_guess_r, seed=seed, use_dft_density=use_dft_density, mf=mf)
    
    system_params = {
        'nconf': configs.configs.shape[0],
        'ndets': num_det,
        'nelec': configs.configs.shape[1],
        'dtype': wf_dtype
    }

    acc = {}
    acc['energy'] = bosonaccumulators.ABQMCEnergyAccumulator(mf_inputs)
    
    possible_accumulators = {'ab_vmc_excitations':bosonaccumulators.ABVMCMatrixAccumulator(mf_inputs, use_symm=use_symm),
                             'abc_dmc_excitations':bosonaccumulators.ABCDMCMatrixAccumulator(mf_inputs, system_params, use_symm=use_symm),
                             'density':bosonaccumulators.DensityAccumulator(),
                             'radial_density':bosonaccumulators.RadialDensityAccumulator()}
    if accumulators is not None and len(accumulators) > 0:
        for acc_name in accumulators:
            if acc_name not in possible_accumulators:
                raise ValueError(f"Accumulator {acc_name} not found in possible accumulators")
            else:
                acc[acc_name] = possible_accumulators[acc_name]
                acc['energy'].__dict__.update(mf_inputs)
                print(f"Using accumulator: {acc_name}")
        
    if opt_wf is True:
        mf.xc = xc
        acc = bosonaccumulators.boson_gradient_generator(
            mf, wf, to_opt, nodal_cutoff=nodal_cutoff
        )

    # Bind MF inputs
    wf.mf_inputs = mf_inputs

    return wf, configs, acc

def read_abvmc(fname):
    with h5py.File(fname) as f:
        print(f.keys())
        keys = ['energytotal', 'energyee', 'energyei', 'energyke', 'energyvxc']
        d = dict()
        for k in keys:
            d[k] = f[k][...]
        return pd.DataFrame(d)
        
def read_abopt(fname):
    with h5py.File(fname) as f:
        return pd.DataFrame(
            {
                "energy": f["energy"][...],
                "iteration": f["iteration"][...],
                "var": f["var"][...],
                "ratio": f["ratio"][...],
                "fname": [fname] * len(f["energy"]),
            }
        )        