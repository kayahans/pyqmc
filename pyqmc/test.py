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
    from scf_runs import run_lda_li, run_casci, run_lda_h2
    scf_checkfile, opt_checkfile, mf_lda = run_lda_h2()
    # scf_checkfile, opt_checkfile, mf_lda = run_lda_li()
    ci_checkfile, mc = run_casci(scf_checkfile, nroots=2, ncas =3)
    
    # for fname in ['sj.hdf5']:
    #     if os.path.isfile(fname):
    #         os.remove(fname)
    # print("RUNNING VMC OPTIMIZATION")
    # with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
    #     recipes.OPTIMIZE(scf_checkfile, "sj.hdf5", 
    #                     max_iterations=6, 
    #                     nconfig=1000, 
    #                     verbose=True,
    #                     jastrow_kws={"ion_cusp":False},
    #                     client = client, 
    #                     npartitions=ncore
    #                     )


    # 2. Boson Jastrow optimization
    abvmcopt_file = 'abvmc-j-opt.hdf5'
    reuse = False
    if not reuse:
        for fname in [abvmcopt_file]:
            if os.path.isfile(fname):
                os.remove(fname)
        print("RUNNING ABVMC OPTIMIZATION")
        num_int = 1
        
        serial = False
        if serial:
            bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                    abvmcopt_file, 
                                    ci_checkfile   = ci_checkfile,
                                    max_iterations = num_int, 
                                    jastrow_kws    = {"ion_cusp":False},
                                    verbose        = True,                            
                                    nconfig        = 1000)  
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
                bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                                abvmcopt_file, 
                                ci_checkfile = ci_checkfile,
                                max_iterations=1, 
                                verbose=True,  
                                load_parameters = None, 
                                # nblocks=nblocks, 
                                # nsteps_per_block=nsteps_per_block,
                                # tstep= tstep, 
                                jastrow_kws={"ion_cusp":False},
                                client = client, 
                                npartitions=ncore,                            
                                nconfig=1000)
        # Jastrow optimization results
        df = recipes.read_opt(abvmcopt_file)
        print(df)

    # 3. ABVMC
    abvmc_file = 'abvmc.hdf5'
    for fname in [abvmc_file]:
        if os.path.isfile(fname):
            os.remove(fname)
    
    serial = True
    if serial:
        print('Using Serial code')
        bosonrecipes.ABVMC(scf_checkfile, 
                        abvmc_file, 
                        ci_checkfile = ci_checkfile,
                        verbose = True,  
                        jastrow_kws={"ion_cusp":False},
                        tstep   = 0.3,
                        nconfig = 1000,
                        nblocks = 100,
                        nsteps_per_block = 20,
                        load_parameters = abvmcopt_file
                        )
                        
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
            bosonrecipes.ABVMC(scf_checkfile, 
                            abvmc_file, 
                            ci_checkfile = ci_checkfile,
                            verbose = True,  
                            jastrow_kws={"ion_cusp":False},
                            tstep   = 0.3,
                            nconfig = 10000,
                            nblocks = 1000,
                            nsteps_per_block = 20,
                            # load_parameters = abvmcopt_file, 
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

