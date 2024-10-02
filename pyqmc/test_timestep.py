#!/usr/bin/env python
import concurrent
import bosonrecipes
# Wavefunction generation
from scf_runs import run_lda_he, run_lda_li, run_lda_be, run_lda_b, run_lda_c, run_lda_n
import h5py
import numpy as np
import matplotlib.pyplot as plt

def run_ABVMC(scf_checkfile, 
              abvmc_checkfile, 
              ci_checkfile, 
              tstep = 0.1, 
              nconfig=1000, 
              nblocks=100, 
              nsteps_per_block=20, 
              load_parameters=None, 
              jastrow_kws = {"ion_cusp":False},
              serial = True, 
              ncore = None):
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
                        nsteps_per_block = nsteps_per_block,
                        load_parameters = load_parameters
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
                            nsteps_per_block = nsteps_per_block,
                            load_parameters = load_parameters, 
                            client = client, 
                            npartitions=ncore)


if __name__=="__main__":

    # 1. DFT calculations
    # scf_checkfile, opt_checkfile, mf_lda = run_lda_li()
    # ci_checkfile = None
    # # 1. CI calculation
    from scf_runs import run_lda_li, run_casci, run_lda_h2, run_lda_he, run_lda_carbon
    # scf_checkfile, _, mf_lda = run_lda_h2()
    scf_checkfile, _, mf_lda = run_lda_he()
    # scf_checkfile, _, mf_lda = run_lda_li()
    # scf_checkfile, _, mf_lda = run_lda_carbon()

    ci_checkfile, mc, opt_checkfile, abvmc_checkfile = run_casci(scf_checkfile, nroots=2, ncas =3, nelecas=(1, 1))
    
    print(opt_checkfile, abvmc_checkfile)
    t = [0.001, 0.02, 0.1, 0.3, 1]
    dt_discard = 100
    e_t_abvmc = []
    e_t_abvmc_err = []
    ab_e_dict = {}
    for dt in t:
        print('timestep = ', dt)
        rootname = abvmc_checkfile.split('.hdf5')[0]
        abvmc_checkfile_dt = rootname+'_{}.hdf5'.format(dt)
        run_ABVMC(scf_checkfile, 
              abvmc_checkfile_dt, 
              ci_checkfile, 
              tstep = dt, 
              nblocks = 310,
              serial = False, 
              ncore = 16)
            
        with h5py.File(abvmc_checkfile_dt) as f:
            print(f.keys())
            etot_t = f['energytotal'][...] 
        ab_e_dict[dt]=etot_t
        for e in [etot_t]:
            e_m = np.mean(e[dt_discard:])
            e_d = np.sqrt(np.var(e[dt_discard:]))
            print(e_m, e_d)
            e_t_abvmc.append(e_m)
            e_t_abvmc_err.append(e_d)
    import pdb
    pdb.set_trace()

    # for i in t:
    #     plt.plot(ab_e_dict[i], label=str(i))
    plt.errorbar(np.arange(len(e_t_abvmc)), e_t_abvmc, yerr=e_t_abvmc_err)
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('Energy (Ha)')
    # plt.ylim((-37.8, -37.6))
    plt.show()
    