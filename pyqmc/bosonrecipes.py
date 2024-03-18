import os
import numpy as np
import bosonwftools
import pyqmc.pyscftools as pyscftools
import pyqmc.supercell as supercell
import h5py
import pandas as pd
import mc
import linemin
# import bosonmc
import wftools
import pyqmc
import bosonaccumulators

def ABOPTIMIZE(
    dft_checkfile: str,
    output: str,
    nconfig: int = 1000,
    ci_checkfile:str|None=None,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws: list|None = None,
    slater_kws:  list|None = None,
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
        accumulators=bosonaccumulators,
    )
    if anchors is None:
        wf, df = linemin.line_minimization(wf, configs, acc, **linemin_kws)
    # else:
    #     wfs = []
    #     for i, a in enumerate(anchors):
    #         wfs.append(
    #             initialize_qmc_objects(
    #                 dft_checkfile,
    #                 ci_checkfile=ci_checkfile,
    #                 load_parameters=a,
    #                 S=S,
    #                 jastrow_kws=jastrow_kws,
    #                 slater_kws=slater_kws,
    #                 target_root=i,
    #             )[0]
    #         )
    #     # wfs = [wftools.read_wf(copy.deepcopy(wf), a) for a in anchors]
    #     wfs.append(wf)
    #     optimize_ortho.optimize_orthogonal(wfs, configs, acc, **linemin_kws)
    return wf, df

def ABOPTIMIZE2(
    dft_checkfile: str,
    output: str,
    nconfig: int = 1000,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws: list|None = None,
    slater_kws:  list|None = None,
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


    ci_checkfile = None
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
        accumulators=bosonaccumulators,
    )

    if anchors is None:
        wf, df = linemin.line_minimization(wf, configs, acc, **linemin_kws)
    # else:
    #     wfs = []
    #     for i, a in enumerate(anchors):
    #         wfs.append(
    #             initialize_qmc_objects(
    #                 dft_checkfile,
    #                 ci_checkfile=ci_checkfile,
    #                 load_parameters=a,
    #                 S=S,
    #                 jastrow_kws=jastrow_kws,
    #                 slater_kws=slater_kws,
    #                 target_root=i,
    #             )[0]
    #         )
    #     # wfs = [wftools.read_wf(copy.deepcopy(wf), a) for a in anchors]
    #     wfs.append(wf)
    #     optimize_ortho.optimize_orthogonal(wfs, configs, acc, **linemin_kws)
    return wf, df


def ABVMC(
    dft_checkfile: str,
    output: str,
    nconfig=1000,
    ci_checkfile: str|None=None,
    load_parameters: str|None=None,
    S=None,
    jastrow_kws: list|None = None,
    slater_kws:  list|None = None,
    accumulators: list|None = None,
    seed: int|None=None,
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
    )
    # bosonmc.abvmc(wf, configs, accumulators=acc, **vmc_kws)
    mc.vmc(wf, configs, accumulators=acc, **vmc_kws)

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
    import abdmc
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
    )
    abdmc.runabdmc(wf, configs, accumulators=acc, **dmc_kws)

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
            mc.ci = mc.ci[target_root]

    if S is not None:
        mol = supercell.get_supercell(mol, np.asarray(S))
    
    # Use when testing HF
    if load_parameters is False:
        wf, to_opt = bosonwftools.generate_boson_wf(
            mol, mf, mc=mc, jastrow = None, jastrow_kws=jastrow_kws, slater_kws=slater_kws
        )
    else:
        wf, to_opt = bosonwftools.generate_boson_wf(
            mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws
        )
    if load_parameters is not None:
        wftools.read_wf(wf, load_parameters)    
    print('Using spherical guess')
    configs = initial_guess(mol, nconfig,seed=seed)
    if opt_wf:
        accumulators = bosonaccumulators.boson_gradient_generator(
            mf, wf, to_opt, nodal_cutoff=nodal_cutoff
        )
    else:
        accumulators = {}
        if slater_kws is not None and "twist" in slater_kws.keys():
            twist = slater_kws["twist"]
        else:
            twist = 0
        accumulators['energy'] = bosonaccumulators.ABQMCEnergyAccumulator(mf)
        accumulators['excitations'] = bosonaccumulators.ABVMCMatrixAccumulator(mf, mc)
        # acc = generate_accumulators(mol, mf, twist=twist, **accumulators)
    return wf, configs, accumulators

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