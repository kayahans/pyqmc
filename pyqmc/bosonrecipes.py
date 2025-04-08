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
        # accumulators=bosonaccumulators,
        det_emax=det_emax,
        xc=xc,
    )
    if anchors is None:
        wf, df = bosonlinemin.line_minimization(wf, configs, acc, **linemin_kws)
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
    nwarmup: int = 0,
    dtwarmup: float|None=None,
    xc: str = 'LDA,VWN',
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
    )
    
    if nwarmup > 0:
        # First equilibration
        # Reused keywords
        eq_keywords = ['verbose', 'hdf_file', 'nsteps_per_block', 'client', 'npartitions']
        eq_tags = {}
        for kw in eq_keywords:
            if kw in vmc_kws.keys():
                if kw == 'hdf_file':
                    eq_tags[kw] = 'eq_'+vmc_kws[kw]
                else:
                    eq_tags[kw] = vmc_kws[kw]
        _, configs = bosonmc.abvmc(
                wf,
                configs,
                nblocks = nwarmup,
                tstep   = dtwarmup,
                **eq_tags
        )
        
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
    )
    bosondmc.rundmc(wf, configs, accumulators=acc, **dmc_kws)

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
    xc = 'LDA,VWN',
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

    available_xc = ['LDA,VWN', 'HF']
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
            mol, mf, mc=mc, jastrow = None, jastrow_kws=jastrow_kws, slater_kws=slater_kws, det_emax=det_emax
        )
    else:
        wf, to_opt = bosonwftools.generate_boson_wf(
            mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws, det_emax=det_emax
        )
        if load_parameters is not None:
            print('Loading WF parameters from', load_parameters)
            wftools.read_wf(wf, load_parameters)    

    
    print('Using spherical guess')
    configs = initial_guess(mol, nconfig,seed=seed)

    acc = {}
    acc['energy'] = bosonaccumulators.ABQMCEnergyAccumulator(mf_inputs)

    possible_accumulators = {'ab_vmc_excitations':bosonaccumulators.ABVMCMatrixAccumulator(), 
                             'ab_dmc_excitations':bosonaccumulators.ABDMCMatrixAccumulator(),
                             'abc_dmc_excitations':bosonaccumulators.ABCDMCMatrixAccumulator()}
    if accumulators is not None and len(accumulators) > 0:
        for acc_name in accumulators:
            if acc_name not in possible_accumulators:
                raise ValueError(f"Accumulator {acc_name} not found in possible accumulators")
            else:
                acc[acc_name] = possible_accumulators[acc_name]
                acc['energy'].__dict__.update(mf_inputs)
                print(f"Using accumulator {acc_name}")
        
    if opt_wf is True:
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