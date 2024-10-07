#!/usr/bin/env python
import os
import concurrent.futures
import recipes
import bosonrecipes
# Wavefunction generation
from scf_runs import run_lda_he, run_lda_li, run_lda_be, run_lda_b, run_lda_c, run_lda_n





if __name__=="__main__":
    ncore = 16
    
    # 1. DFT calculations
    # scf_checkfile, opt_checkfile, mf_lda = run_lda_li()
    # ci_checkfile = None
    # # 1. CI calculation
    from scf_runs import run_lda_li, run_casci, run_lda_h2, run_lda_he, run_lda_carbon, run_lda_be
    # scf_checkfile, _, mf_lda = run_lda_h2()
    scf_checkfile, _, mf_lda = run_lda_he()
    # scf_checkfile, _, mf_lda = run_lda_li()
    # scf_checkfile, _, mf_lda = run_lda_be()
    # scf_checkfile, _, mf_lda = run_lda_carbon()

    ci_checkfile, mc, opt_checkfile, abvmc_checkfile = run_casci(scf_checkfile, nroots=2, ncas = 9, nelecas=(1, 1))
    
    print(opt_checkfile, abvmc_checkfile)
    
    # 2. Boson Jastrow optimization
    reuse = True
    jastrow_kws = {"ion_cusp":False, "na":0}
    
    if not reuse:
        for fname in [opt_checkfile]:
            if os.path.isfile(fname):
                os.remove(fname)
        print("RUNNING ABVMC OPTIMIZATION")
        num_int = 6
        nconfig = 1000
        serial = False
        if serial:
            bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                    opt_checkfile, 
                                    ci_checkfile   = ci_checkfile,
                                    max_iterations = num_int, 
                                    jastrow_kws    = jastrow_kws,
                                    verbose        = True,                            
                                    nconfig        = nconfig)  
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
                bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                opt_checkfile, 
                                ci_checkfile = ci_checkfile,
                                max_iterations=num_int, 
                                verbose=True,  
                                load_parameters = None, 
                                jastrow_kws    = jastrow_kws,
                                client = client, 
                                npartitions=ncore,                            
                                nconfig=nconfig)
        # Jastrow optimization results
        df = recipes.read_opt(opt_checkfile)
        print(df)

    # 3. ABVMC
    for fname in [abvmc_checkfile]:
        if os.path.isfile(fname):
            os.remove(fname)

    nconfig = 1000
    nblocks = 200
    tstep = 0.3

    serial = False
    if serial:
        print('Using Serial code')
        bosonrecipes.ABVMC(scf_checkfile, 
                        abvmc_checkfile, 
                        ci_checkfile = ci_checkfile,
                        verbose = True,  
                        jastrow_kws    = jastrow_kws,
                        tstep   = tstep,
                        nconfig = nconfig,
                        nblocks = nblocks,
                        accumulators = ['excitations'],
                        nsteps_per_block = 20,
                        load_parameters = opt_checkfile
                        )
                        
    else:
        print('Using Parallel code')
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
            bosonrecipes.ABVMC(scf_checkfile, 
                            abvmc_checkfile, 
                            ci_checkfile = ci_checkfile,
                            verbose = True,  
                            jastrow_kws    = jastrow_kws,
                            tstep   = tstep,
                            nconfig = nconfig,
                            nblocks = nblocks,
                            nsteps_per_block = 20,
                            load_parameters = opt_checkfile, 
                            accumulators = ['excitations'],
                            client = client, 
                            npartitions=ncore)

# with h5py.File("sj.hdf5") as f:
#     print("keys", list(f.keys()))
#     print("wave function parameters", list(f['wf'].keys()))
#     ee_j = f['energy'][...]   
#     x = f['x'][...]
#     yfit = f['yfit'][...]
#     # pgrad = f['pgradient'][...]   
#     print(f['wf/wf2acoeff'][()])
#     print(f['wf/wf2bcoeff'][()])

