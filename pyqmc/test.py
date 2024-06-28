#!/usr/bin/env python
import os
import concurrent.futures
import recipes
import bosonrecipes
# Wavefunction generation
from scf_runs import run_lda_he, run_lda_li, run_lda_be, run_lda_b, run_lda_c, run_lda_n





if __name__=="__main__":
    scf_checkfile, opt_checkfile, mf_lda = run_lda_li()
    ncore = 16
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


    # Boson Jastrow optimization
    abvmc_file = 'abvmc-j-single.hdf5'
    for fname in [abvmc_file]:
        if os.path.isfile(fname):
            os.remove(fname)
    print("RUNNING ABVMC OPTIMIZATION")

    # bosonrecipes.ABOPTIMIZE(scf_checkfile, 
    #                         abvmc_file, 
    #                         max_iterations=6, 
    #                         verbose=True,                            
    #                         nconfig=1000)  
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
        bosonrecipes.ABOPTIMIZE(scf_checkfile, 
                        abvmc_file, 
                        max_iterations=6, 
                        verbose=True,  
                        load_parameters = None, 
                        # nblocks=nblocks, 
                        # nsteps_per_block=nsteps_per_block,
                        # tstep= tstep, 
                        # jastrow_kws={"ion_cusp":False},
                        client = client, 
                        npartitions=ncore,                            
                        nconfig=1000)
    # Jastrow optimization results
    df = recipes.read_opt(abvmc_file)
    print(df)

# with h5py.File("sj.hdf5") as f:
#     print("keys", list(f.keys()))
#     print("wave function parameters", list(f['wf'].keys()))
#     ee_j = f['energy'][...]   
#     x = f['x'][...]
#     yfit = f['yfit'][...]
#     # pgrad = f['pgradient'][...]   
#     print(f['wf/wf2acoeff'][()])
#     print(f['wf/wf2bcoeff'][()])

