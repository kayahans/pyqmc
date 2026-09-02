#!/usr/bin/env python3
import os
import time
import copy
import concurrent.futures
import multiprocessing as mp
from mpi4py.futures import MPIPoolExecutor
import mpi4py.MPI
import yaml
import h5py

import pyqmc.gpu as gpu
import pyqmc.pyscftools as pyscftools
from pyqmc import bosonwftools
from pyqmc import bosonrecipes

# Wavefunction generation
serial = False
continue_qmc = True
reuse = True
# To profile the code 
# profile_boson_dmc: true
# profile_abcdmc_print_every: 10

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

basis_name = config['basis_name']
ecp = config.get('ecp', None)
checkfile_dir = config.get('checkfile_dir', '.')
symm_tag = config['symm_tag']
det_emax = config['det_emax']
dtwarmup = config.get('dtwarmup', 0.5)
dmc_eq_dt = config.get('dmc_eq_dt', 0.02)
dmc_stat_dt = config.get('dmc_stat_dt', 0.01)
dmc_stat_nstep = config.get('dmc_stat_nstep', 50000)
vmc_nstep = config.get('vmc_nstep', 5000)
xc = config.get('xc', 'PBE, PBE')
ion_cusp = config['ion_cusp']
na = config['na']
nb = config['nb']
rcut = config['rcut']
# exclude_core = config.get('exclude_core', False)

from pyqmc.bosonaccumulators import configure_boson_dmc_profiling

configure_boson_dmc_profiling(
    enabled=config.get("profile_boson_dmc"),
    print_every=config.get("profile_abcdmc_print_every"),
)

def print_config():
    print("Printing configuration...")
    print("--------------------------------")
    print(f"dtwarmup: {dtwarmup}")
    print(f"dmc_eq_dt: {dmc_eq_dt}")
    print(f"dmc_stat_dt: {dmc_stat_dt}")
    print(f"vmc_nstep: {vmc_nstep}")
    print(f"xc: {xc}")
    print(f"ion_cusp: {ion_cusp}")
    print(f"na: {na}")
    print(f"nb: {nb}")
    print(f"rcut: {rcut}")
    # print(f"exclude_core: {exclude_core}")
    print(f"det_emax: {det_emax}")
    print(f"symm_tag: {symm_tag}")
    print(f"basis_name: {basis_name}")
    print(f"checkfile_dir: {checkfile_dir}")
    print(f"reuse: {reuse}")
    print(f"continue_qmc: {continue_qmc}")
    print(f"serial: {serial}")
    print("Configuration printed")
    print("--------------------------------")



def write_mock_opt_hdf5(
    dft_checkfile,
    ci_checkfile,
    output_hdf5,
    jastrow_kws,
    det_emax,
    use_symm=True,
):
    """Build Boson×Jastrow WF (same recipe as ABDMC) and save ``wf/*`` parameters for read_wf.

    Layout matches bosonlinemin / wftools.read_wf: group ``wf`` with one dataset per
    parameter key.
    """
    target_root = 0
    mol, mf, mc = pyscftools.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
    if not hasattr(mc.ci, "shape") or len(mc.ci.shape) == 3:
        mc.fci = mc.ci
        mc.ci = mc.ci[target_root]

    wf, _ = bosonwftools.generate_boson_wf(
        mol,
        mf,
        mc=mc,
        jastrow_kws=jastrow_kws,
        slater_kws={},
        det_emax=det_emax,
        use_symm=use_symm,
    )
    with h5py.File(output_hdf5, "w") as hdf:
        hdf.create_group("wf")
        for k in wf.parameters.keys():
            hdf.create_dataset("wf/" + k, data=gpu.asnumpy(wf.parameters[k]))


# Multithreading setup
def setup_parallel_environment():
    """Setup environment for parallel execution"""
    # Get number of threads from environment or use CPU count
    nthreads = int(os.environ.get('OMP_NUM_THREADS', mp.cpu_count()))

    # Set NumPy to use multiple threads
    os.environ['OMP_NUM_THREADS'] = str(nthreads)
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)

    # # Configure NumPy to use multiple threads
    # np.set_num_threads(nthreads)

    print(f"Running with {nthreads} threads")
    # print(f"NumPy threads: {np.get_num_threads()}")

    return nthreads

# Setup parallel environment

if __name__=="__main__":
    # Start overall timing
    start_time = time.time()
    print(f"=== Starting QMC calculation at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print_config()
    nthreads = setup_parallel_environment()
    comm = mpi4py.MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    npartitions= comm.Get_size()-1
    # npartitions = 8

    # 1. CI/DFT calculations
    # Load configuration from YAML file


    atom_name = config['atom_name']
    ncas = config['ncas']
    nelecas = tuple(config['nelecas'])

    # Format strings for different checkpoint files
    file_template = (
        '{}_atom_basis_{}_diffuse_S0P0D0_v1{}.hdf5 '  # scf
        '{}_ci_atom_basis_{}_diffuse_S0P0D0{}.hdf5 ' # ci
        '{}_opt_cas_{}_nelecas_{}_{}.hdf5 '  # opt
        '{}_dmc_eq_cas_{}_nelecas_{}_{}.hdf5 '  # dmc eq
        '{}_dmc_cas_{}_nelecas_{}_{}.hdf5'  # dmc
    )

    # Format with parameters
    files_str = file_template.format(
        atom_name, basis_name, symm_tag,
        atom_name, basis_name, symm_tag, # For scf and ci files
        atom_name, ncas, nelecas[0], nelecas[1],  # For opt file
        atom_name, ncas, nelecas[0], nelecas[1],  # For dmc eq file
        atom_name, ncas, nelecas[0], nelecas[1]   # For dmc file
    )

    # Split into individual filenames
    scf_checkfile, ci_checkfile, opt_checkfile, abvmc_eq_checkfile, abvmc_checkfile = files_str.split()
    scf_checkfile = os.path.join(checkfile_dir, scf_checkfile)
    ci_checkfile = os.path.join(checkfile_dir, ci_checkfile)
    for path in (scf_checkfile, ci_checkfile):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    print(f"SCF checkpoint: {scf_checkfile}")
    print(f"CI checkpoint:  {ci_checkfile}")
    # dtwarmup = 0.1
    vmc_equilibrium = {
        'nconfig': 1280,
        'nsteps_per_block': 1,
        'nblocks': vmc_nstep,
        'tstep': dtwarmup,
        'hdf_file': 'warmup_vmc.hdf5',
        'accumulators': ['energy', 'radial_density'],
        # 'exclude_core': exclude_core,
    }
    dmc_equilibrium = {
        'nconfig': vmc_equilibrium['nconfig'],
        'nblocks': 5000,
        'tstep': dmc_eq_dt,
        'hdf_file': 'warmup_dmc.hdf5',
        'accumulators': ['energy',],
        # 'exclude_core': exclude_core,
    }
    dmc_statistics = {
        'nconfig': 12800,
        'nblocks': dmc_stat_nstep,
        'tstep': dmc_stat_dt,
        # 'exclude_core': exclude_core,
    }
    
    # 2. Fixed Jastrow (e-e cusp only: na=nb=0); write mock opt HDF5 instead of ABOPTIMIZE
    jastrow_kws = {"ion_cusp": False, "na": na, "nb": nb, "rcut": rcut, "init_type": "zero"}
    if not reuse:
        # for fname in [opt_checkfile]:
        #     if os.path.isfile(fname):
        #         os.remove(fname)
        print("RUNNING ABVMC OPTIMIZATION with single determinant")
        num_int = 20
        aboptimize_params = {
                            'ci_checkfile'   : ci_checkfile,
                            'max_iterations' : num_int, 
                            'jastrow_kws'    : jastrow_kws,
                            'verbose'        : True,                            
                            'nconfig'        : 12800,
                            'det_emax'       : det_emax,
                            'use_symm'       : True,
                            'use_dft_density': True,
                            'opt_method'     : 'linemin',
                            # 'vmcoptions'     : {'nsteps_per_block': 10, 'nblocks': 1000, 'tstep': 0.3, 'converged_parameter': 'pgradtotal', 'convergence_threshold': 1e-5},
                            'vmcoptions'     : {'nsteps_per_block': 10, 'nblocks': 10, 'tstep': dtwarmup},
                            'warmup_options' : vmc_equilibrium
                            }
        
        # Start timing for optimization
        opt_start_time = time.time()
        print(f"=== Starting Jastrow optimization at {time.strftime('%H:%M:%S')} ===")
        
        if serial:
            print('Using Serial code')
            bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                    opt_checkfile, 
                                    **aboptimize_params)
        else:
            print('Using Parallel code')
            # with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
            with MPIPoolExecutor(max_workers=npartitions) as client:
                bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                opt_checkfile, 
                                client          = client,
                                npartitions     = npartitions,
                                **aboptimize_params)  
        
        # End timing for optimization
        opt_end_time = time.time()
        opt_duration = opt_end_time - opt_start_time
        print(f"=== Jastrow optimization completed in {opt_duration:.2f} seconds ({opt_duration/60:.2f} minutes) ===")
    else:
        print('Reusing jastrows')
        if not os.path.isfile(opt_checkfile):
            print("Writing mock opt checkpoint (fixed Jastrow, no line minimization)")
            opt_start_time = time.time()
            write_mock_opt_hdf5(
                scf_checkfile,
                ci_checkfile,
                opt_checkfile,
                jastrow_kws,
                det_emax,
                use_symm=True,
            )
            opt_end_time = time.time()
            print(f"=== Mock opt file written to {opt_checkfile} in {opt_end_time - opt_start_time:.2f} s ===")
                
    opt_analyze_parameters = {
        'scf_checkfile' : scf_checkfile,
        'ci_checkfile' : ci_checkfile,
        'verbose' : True,
        'jastrow_kws' : jastrow_kws,
        'load_parameters' : opt_checkfile,
        'tstep' : 0.3,
        'nconfig' : dmc_equilibrium['nconfig'],
    }
    # from analysis import analyze_jastrows, analyze_opt

    # if os.path.isfile(opt_checkfile):
    #     analyze_opt(opt_checkfile)
    #     analyze_jastrows(opt_analyze_parameters)
    # else:
    #     print('No jastrow optimization results found')
    # exit()
    # 3. ABVMC
    if not continue_qmc:
        for fname in [abvmc_checkfile]:
            if os.path.isfile(fname):
                os.remove(fname)
    # Equilibrium DMC
    vmc_eq_noconfig  = vmc_equilibrium.copy()
    vmc_eq_noconfig.pop('nconfig')
    abdmc_params_eq = {
                    'ci_checkfile' : ci_checkfile,
                    'verbose' : True,
                    'jastrow_kws'    : jastrow_kws,
                    'tstep'   : dmc_equilibrium['tstep'],
                    'nconfig' : dmc_equilibrium['nconfig'],
                    'nblocks' : dmc_equilibrium['nblocks'],
                    'vmc_options' : vmc_eq_noconfig,
                    'nsteps_per_block' : 1,
                    'load_parameters' : opt_checkfile,
                    'initial_guess_r' : 10,
                    'xc'             : xc,
                    'use_dft_density' : True,
                    # 'exclude_core': exclude_core,
                    # load_parameters = False,
	                'use_symm' : True,
                    'det_emax' : det_emax}

    # Accumulation DMC
    abdmc_params = copy.deepcopy(abdmc_params_eq)
    abdmc_params['accumulators'] = ['abc_dmc_excitations']
    abdmc_params['vmc_options']['hdf_file'] = abvmc_eq_checkfile
    abdmc_params['nconfig'] = dmc_statistics['nconfig']
    abdmc_params['nblocks'] = dmc_statistics['nblocks']
    abdmc_params['tstep'] = dmc_statistics['tstep']
    abdmc_params['nsteps_per_block'] = 20

    # Start timing for ABVMC
    abdmc_start_time = time.time()
    print(f"=== Starting ABDMC EQUILIBRIUM calculation at {time.strftime('%H:%M:%S')} ===")
    if serial:
        print('Using Serial code')
        bosonrecipes.ABDMC(scf_checkfile,
                        abvmc_eq_checkfile,
                        **abdmc_params_eq,
                        )

    else:
        print('Using Parallel code')
        with MPIPoolExecutor(max_workers=npartitions) as client:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
            bosonrecipes.ABDMC(scf_checkfile,
                            abvmc_eq_checkfile,
                            client = client,
                            npartitions=npartitions,
                            **abdmc_params_eq)

    # End timing for ABVMC
    abdmc_end_time = time.time()
    abdmc_duration = abdmc_end_time - abdmc_start_time
    print(f"=== ABDMC EQUILIBRIUM calculation completed in {abdmc_duration:.2f} seconds ({abdmc_duration/60:.2f} minutes) ===")

    abdmc_start_time = time.time()
    print(f"=== Starting ABDMC STATISTICS calculation at {time.strftime('%H:%M:%S')} ===")
    if serial:
        print('Using Serial code')
        
        bosonrecipes.ABDMC(scf_checkfile,
                        abvmc_checkfile,
                        **abdmc_params,
                        )

    else:
        print('Using Parallel code')
        with MPIPoolExecutor(max_workers=npartitions) as client:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
            bosonrecipes.ABDMC(scf_checkfile,
                            abvmc_checkfile,
                            client = client,
                            npartitions=npartitions,
                            **abdmc_params)

    # End timing for ABVMC
    abdmc_end_time = time.time()
    abdmc_duration = abdmc_end_time - abdmc_start_time
    print(f"=== ABDMC STATISTICS calculation completed in {abdmc_duration:.2f} seconds ({abdmc_duration/60:.2f} minutes) ===")

    # End overall timing
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"=== Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes) ===")
    print(f"=== Calculation completed at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

