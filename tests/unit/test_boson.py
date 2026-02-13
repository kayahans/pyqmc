import pyqmc.api as pyq
import pyqmc.bosonwftools as bosonwftools
import pyqmc.wftools as wftools
import pytest 
import os
import numpy as np
from pyqmc.mc import initial_guess
from pyqmc.bosonslater import BosonWF

def erase_file(fname):
    try:
        os.remove(fname)
    except:
        pass
    

@pytest.mark.boson
def test_boson_wf(H2_ccecp_casci_s0):
    '''Boson and Slater wavefunctions should have the same parameters'''
    mol, mf, mc = H2_ccecp_casci_s0

    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc)

    parameters = wfs.parameters.keys()
    parameters_boson = wfb.parameters.keys()

    for param in parameters:
        assert param in parameters_boson

    for param in parameters:
        assert wfs.parameters[param] == wfb.parameters[param]
        
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_wf_li(Li_ccecp_casci_s1):
    '''Boson and Slater wavefunctions should have the same parameters'''
    mol, mf, mc = Li_ccecp_casci_s1

    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc)

    parameters = wfs.parameters.keys()
    parameters_boson = wfb.parameters.keys()

    for param in parameters:
        assert param in parameters_boson

    for param in parameters:
        assert wfs.parameters[param] == wfb.parameters[param]
        
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_wf_value(H2_ccecp_uhf):
    '''With a single determinant, the auxiliary boson and Slater wavefunctions should have the same value'''
    mol, mf = H2_ccecp_uhf
    mc = None
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc)
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    wfs.recompute(configs)
    assert np.allclose(wfb.value(), wfs.value())
    


@pytest.mark.boson
def test_boson_derivatives(H2_ccecp_uhf):
    '''The derivatives of the single determinant auxiliary boson and Slater wavefunctions should be the same'''
    mol, mf = H2_ccecp_uhf
    mc = None
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc)
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    wfs.recompute(configs)
    e = 0
    epos = configs.electron(e)
    wfb_grad = wfb.gradient(e, epos)
    wfs_grad = wfs.gradient(e, epos)
    assert np.allclose(wfb_grad, wfs_grad)
    
    wfb_grad_v, wfb_val, wfb_saved = wfb.gradient_value(e, epos)
    wfs_grad_v, wfs_val, wfs_saved = wfs.gradient_value(e, epos)

    assert np.allclose(wfb_grad_v, wfb_grad)
    assert np.allclose(wfs_grad_v, wfs_grad)
    
    # assert np.allclose(wfb_val, wfs_val) (This does not have to be true)

    wfb_lap = wfb.laplacian(e, epos)
    wfs_lap = wfs.laplacian(e, epos)
    assert np.allclose(wfb_lap, wfs_lap)

@pytest.mark.boson_slow
def test_boson_abvmc_timestep_convergence_he_lda():
    '''For an AB-HF calculation, the total energy should converge to the same value for different timesteps'''
    from pyqmc.bosonrecipes import ABVMC
    # import matplotlib.pyplot as plt
    
    def run_scf(chkfile):
        erase_file(chkfile)
        from pyscf import gto, dft
        mol = gto.M(
            atom="He 0 0. 0.0", basis="aug-ccpvqz",  unit="bohr", spin = 0
        )
        # mf = scf.UHF(mol)
        mf = dft.UKS(mol)
        mf.chkfile = chkfile
        mf.xc = 'LDA,VWN'
        mf.kernel()
        return mf

    def read_abvmc_energies(fname):
        import h5py
        with h5py.File(fname, 'r') as f:
            energies = f['energytotal'][:]
        return energies

    def within_3_std(mean_i, std_i, ref_mean, ref_std):
        if mean_i > ref_mean:
            assert mean_i - 3*std_i < ref_mean + 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'
        else:
            assert mean_i + 3*std_i > ref_mean - 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'

    # Try parallellization
    try:
        import concurrent.futures
        import os
        npartitions = int(os.cpu_count() + 4)
        client = concurrent.futures.ProcessPoolExecutor(max_workers=npartitions)
    except:
        client = None
        npartitions = 1
    #end try 


    dft_checkfile = 'he_scf.hdf5'
    mf = run_scf(dft_checkfile)

    e_results = []
    dt_list = [1, 0.3, 0.1, 0.03, 0.01]
    discard = 100
    for dt in dt_list:
        abvmc_filename = f'he_abvmc_{dt}.hdf5'
        erase_file(abvmc_filename)

        wf, configs, acc = ABVMC(
            dft_checkfile=dft_checkfile,
            output=abvmc_filename,
            nconfig=100,
            tstep=dt,
            nblocks=200,
            nsteps_per_block=10,
            load_parameters=False, 
            seed = 1,
            client = client,
            npartitions = npartitions,
            xc = 'LDA,VWN',
        )
        e = read_abvmc_energies(abvmc_filename)
        e_results.append(e[discard:])
        # plt.plot(e, label=f'dt={dt}')
        
    erase_file(dft_checkfile)
    for dt in dt_list:
        abvmc_filename = f'he_abvmc_{dt}.hdf5'
        erase_file(abvmc_filename)
    
    e_ref = e_results[0]  
    ref_mean = np.mean(e_ref)
    ref_std = np.std(e_ref)
    for i, dt_i in enumerate(dt_list[1:]):
        mean_i = np.mean(e_results[i+1])
        std_i = np.std(e_results[i+1])
        within_3_std(mean_i, std_i, ref_mean, ref_std)


@pytest.mark.boson_slow
def test_boson_abvmc_timestep_convergence_li_lda():
    '''For an AB-HF calculation, the total energy should converge to the same value for different timesteps'''
    from pyqmc.bosonrecipes import ABVMC
    import matplotlib.pyplot as plt
    
    def run_scf(chkfile):
        erase_file(chkfile)
        from pyscf import gto, dft
        mol = gto.M(
            atom="Li 0 0. 0.0", basis="aug-ccpvqz",  unit="bohr", spin = 1
        )
        mf = dft.UKS(mol)
        mf.chkfile = chkfile
        mf.xc = 'LDA,VWN'
        mf.kernel()
        return mf

    def read_abvmc_energies(fname):
        import h5py
        with h5py.File(fname, 'r') as f:
            energies = f['energytotal'][:]
        return energies

    def within_3_std(mean_i, std_i, ref_mean, ref_std):
        if mean_i > ref_mean:
            assert mean_i - 3*std_i < ref_mean + 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'
        else:
            assert mean_i + 3*std_i > ref_mean - 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'

    # Try parallellization
    try:
        import concurrent.futures
        import os
        npartitions = int(os.cpu_count() + 4)
        client = concurrent.futures.ProcessPoolExecutor(max_workers=npartitions)
    except:
        client = None
        npartitions = 1
    #end try 

    dft_checkfile = 'li_scf.hdf5'
    mf = run_scf(dft_checkfile)

    e_results = []
    dt_list = [1, 0.3, 0.1, 0.03, 0.01]
    discard = 100
    for dt in dt_list:
        abvmc_filename = f'li_abvmc_{dt}.hdf5'
        erase_file(abvmc_filename)

        wf, configs, acc = ABVMC(
            dft_checkfile=dft_checkfile,
            output=abvmc_filename,
            nconfig=100,
            tstep=dt,
            nblocks=200,
            nsteps_per_block=10,
            load_parameters=False, 
            seed = 1,
            client = client,
            npartitions = npartitions,
            xc = 'LDA,VWN',
        )
        e = read_abvmc_energies(abvmc_filename)
        e_results.append(e[discard:])
        plt.plot(e, label=f'dt={dt}')
    
    if client is not None:
        client.shutdown()
    
    plt.legend()
    plt.show()
    erase_file(dft_checkfile)
    for dt in dt_list:
        abvmc_filename = f'li_abvmc_{dt}.hdf5'
        erase_file(abvmc_filename)
    
    e_ref = e_results[0]  
    ref_mean = np.mean(e_ref)
    ref_std = np.std(e_ref)
    for i, dt_i in enumerate(dt_list[1:]):
        mean_i = np.mean(e_results[i+1])
        std_i = np.std(e_results[i+1])
        within_3_std(mean_i, std_i, ref_mean, ref_std)

# @pytest.mark.boson_new
# def test_pyscf_energies(Li_ccecp_casci_s1):
#     '''This test only uses pyscf, mainly to confirm their implementation of the Fock matrix'''
#     mol, mf, _ = Li_ccecp_casci_s1
#     from pyscf import dft
#     # Define a well converged grid using pyscf 
#     grids = dft.gen_grid.Grids(mol)
#     grids.build()
#     weights = grids.weights
    
#     # Fock matrix 
#     fock = mf.get_fock()
    

#     mo_coeff_up = mf.mo_coeff[0]
#     mo_coeff_down = mf.mo_coeff[1]
#     mo_occ_up = mf.mo_occ[0]
#     mo_occ_down = mf.mo_occ[1]

#     occ_orbs_up = mo_coeff_up[:, mo_occ_up > 0.]
#     occ_orbs_down = mo_coeff_down[:, mo_occ_down > 0.]

#     fock_up = np.einsum('ui,uv,vi->', occ_orbs_up, fock[0], occ_orbs_up)
#     fock_down = np.einsum('ui,uv,vi->', occ_orbs_down, fock[1], occ_orbs_down)
#     fock_energy = fock_up + fock_down
      
#     eigenvalue_sum = np.sum(mf.mo_energy[mf.mo_occ > 0.])

#     assert np.allclose(eigenvalue_sum, fock_energy), 'Fock energy is not the same as the eigenvalue sum'
    
    
# @pytest.mark.boson_new
# def test_boson_mf_energy(Li_ccecp_casci_s1):
#     '''Confirm that the kinetic energy density obtained from two ways are the same'''
#     mol, mf, _ = Li_ccecp_casci_s1

#     configs = initial_guess(mol, 10)
#     # We only need to evaluate on a grid of points
#     nconf, nelec, ndim = configs.configs.shape
#     nup_dn = mol.nelec
#     for e in range(nelec):
#         s = int(e >= nup_dn[0])
#         ao_value = numint.eval_ao(mol, configs.configs[:,e,:])

# TODO: @pytest.mark.boson_new
# def test_boson_abvmc_timestep_convergence_hf():
#     '''Confirm that the kinetic energy density obtained from two ways are the same'''
#     from pyqmc.bosonrecipes import ABVMC
#     # import matplotlib.pyplot as plt
    
#     def erase_file(fname):
#         try:
#             os.remove(fname)
#         except:
#             pass
    
#     def run_scf(chkfile):
#         erase_file(chkfile)
#         from pyscf import gto, scf, dft
#         mol = gto.M(
#             atom="He 0 0. 0.0", basis="aug-ccpvqz",  unit="bohr", spin = 0
#         )
#         # mf = scf.UHF(mol)
#         mf = scf.UHF(mol)
#         mf.chkfile = chkfile
#         mf.kernel()
#         return mf

#     def read_abvmc_energies(fname):
#         import h5py
#         with h5py.File(fname, 'r') as f:
#             energies = f['energytotal'][:]
#         return energies
    
#     dft_checkfile = 'he_scf.hdf5'
#     mf = run_scf(dft_checkfile)

#     e_results = []
#     dt_list = [1, 0.3, 0.1, 0.03, 0.01]
#     discard = 100
#     for dt in dt_list:
#         abvmc_filename = f'he_abvmc_{dt}.hdf5'
#         erase_file(abvmc_filename)

#         wf, configs, acc = ABVMC(
#             dft_checkfile=dft_checkfile,
#             output=abvmc_filename,
#             nconfig=1000,
#             tstep=dt,
#             nblocks=300,
#             nsteps_per_block=10,
#             load_parameters=False, 
#             seed = 1,
#         )
#         e = read_abvmc_energies(abvmc_filename)
#         e_results.append(e[discard:])
#         # plt.plot(e, label=f'dt={dt}')
        
#     # plt.legend()
#     # plt.show()
#     erase_file(dft_checkfile)
#     for dt in dt_list:
#         abvmc_filename = f'he_abvmc_{dt}.hdf5'
#         erase_file(abvmc_filename)
    
#     e_ref = e_results[0]  
#     from scipy.stats import ttest_rel
#     for i, dt_i in enumerate(dt_list[1:]):
#         t, p = ttest_rel(e_results[i+1], e_ref)
#         # print(np.mean(e_results[i+1]), np.mean(e_ref), p)
#         assert p < 0.05, f'dt={dt_i} is significantly different from dt={dt_list[0]}'


# @pytest.mark.boson_new
# def test_boson_local_energy(Li_ccecp_casci_s1):
#     '''The local energy of the single determinant auxiliary boson and 
#     Slater wavefunctions should be the same with single determinant and no jastrow factor
#     Explicityly exclude jastrows'''
#     mol, mf, _ = Li_ccecp_casci_s1
#     dm = mf.make_rdm1()
#     mf.dm = dm

#     mc = None
#     wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, jastrow=None, mc=mc)
#     wfs, to_opt = wftools.generate_wf(mol, mf, jastrow=None, mc=mc)
#     configs = initial_guess(mol, 10)
#     wfb.recompute(configs)  
#     wfs.recompute(configs)
#     e = 0
#     from pyqmc.bosonaccumulators import ABQMCEnergyAccumulator
#     from pyqmc.accumulators import EnergyAccumulator
#     epos = configs.electron(e)

#     ab_acc = ABQMCEnergyAccumulator(mf)
#     ab_results = ab_acc(configs, wfb)
#     ab_ke = ab_results['ke']
#     ab_ka = ab_results['ka']
#     ab_kb = ab_results['kb']
#     ab_vh = ab_results['vh']
#     ab_vxc = ab_results['vxc']
#     ab_ecorr = ab_results['corr']
#     ab_grad2 = ab_results['grad2']
#     ab_ee = ab_results['ee']
#     ab_ei = ab_results['ei']
    
#     ab_ii = ab_results['ii']
#     ab_total = ab_results['total']
    
#     acc = EnergyAccumulator(mol)
#     results = acc(configs, wfs)
#     ke = results['ke']
#     ee = results['ee']
#     ei = results['ei']
#     grad2 = results['grad2']
#     _, _, ii = acc.coulomb.energy(configs)
#     # ecp = results['ecp']
#     total = results['total']
#     import pdb; pdb.set_trace()
#     assert np.allclose(ab_ee, ee), 'Electron-electron energies are not the same'
    
#     assert np.allclose(ab_ei, ei), 'Electron-ion energies are not the same'
#     # assert np.allclose(ab_ecp, ecp), 'ECP energies are not the same'
#     assert np.allclose(ab_ii, ii), 'Ion-ion energies are not the same'
#     assert np.allclose(ab_ke, 0), 'AB-HF kinetic energies must be zero'
#     assert np.allclose(ab_ka, 0), 'AB-HF kinetic energies contributions A must be zero'
#     assert np.allclose(ab_kb, 0), 'AB-HF kinetic energies contributions B must be zero'
#     assert np.allclose(ab_grad2, grad2), 'Grad2 energies are not the same'
#     assert np.allclose(ab_total, total), 'Total energies are not the same'
    
@pytest.mark.boson
def test_boson_derivatives_li(Li_ccecp_casci_s1):
    '''The derivatives of the single determinant auxiliary boson and Slater wavefunctions should be the same'''
    mol, mf, mc = Li_ccecp_casci_s1
    mc = None
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc)
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    wfs.recompute(configs)
    e = 0
    epos = configs.electron(e)
    wfb_grad = wfb.gradient(e, epos)
    wfs_grad = wfs.gradient(e, epos)
    assert np.allclose(wfb_grad, wfs_grad)
    
    wfb_grad_v, wfb_val, wfb_saved = wfb.gradient_value(e, epos)
    wfs_grad_v, wfs_val, wfs_saved = wfs.gradient_value(e, epos)

    assert np.allclose(wfb_grad_v, wfb_grad)
    assert np.allclose(wfs_grad_v, wfs_grad)
    
    # assert np.allclose(wfb_val, wfs_val) (This does not have to be true)

    wfb_lap = wfb.laplacian(e, epos)
    wfs_lap = wfs.laplacian(e, epos)
    assert np.allclose(wfb_lap, wfs_lap)
    

@pytest.mark.boson
def test_boson_dets_value_singlet(H2_ccecp_casci_s0):
    r'''Given \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
    Check that \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}'''
    mol, mf, mc = H2_ccecp_casci_s0
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break   
    
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    e = 0
    epos = configs.electron(e)
    wfb_val_dets = wfb.value_dets()[1]
    wfb_val = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    phi_b = 1./2 * np.log(np.einsum('d, id->i', det_coeff,np.exp(2*wfb_val_dets) ))
    assert np.allclose(phi_b, wfb_val)
    erase_file('hmf.hdf5')
    
@pytest.mark.boson
def test_boson_dets_value_triplet(H2_ccecp_casci_s2):
    r'''Given \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
    Check that \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}'''
    mol, mf, mc = H2_ccecp_casci_s2
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break   
    
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    e = 0
    epos = configs.electron(e)
    wfb_val_dets = wfb.value_dets()[1]
    wfb_val = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    phi_b = 1./2 * np.log(np.einsum('d, id->i', det_coeff,np.exp(2*wfb_val_dets) ))
    assert np.allclose(phi_b, wfb_val)
    erase_file('hmf.hdf5')
    
@pytest.mark.boson
def test_boson_dets_grad_singlet(H2_ccecp_casci_s0):
    r'''Given \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
    Check that ∇\Phi_B = \sum_{n}{\frac{\Phi_n^2}{\Phi_B}∇log(\Phi_n)}'''
    mol, mf, mc = H2_ccecp_casci_s0
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)

    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break

    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    e = 0
    epos = configs.electron(e)
    wfb_grad_dets, wfb_grad = wfb.gradient_dets(e, epos)
    # wfb_grad = wfb.gradient(e, epos)

    dv = wfb.value_dets()[1]
    v = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), wfb_grad_dets)
    assert np.allclose(gc, wfb_grad)
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_dets_grad_triplet(H2_ccecp_casci_s2):
    r'''Given \Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
    Check that ∇\Phi_B = \sum_{n}{\frac{\Phi_n^2}{\Phi_B}∇log(\Phi_n)}'''
    mol, mf, mc = H2_ccecp_casci_s2
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)

    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break

    configs = initial_guess(mol, 10)
    wfb.recompute(configs)  
    e = 0
    epos = configs.electron(e)
    wfb_grad_dets, wfb_grad = wfb.gradient_dets(e, epos)
    # wfb_grad = wfb.gradient(e, epos)

    dv = wfb.value_dets()[1]
    v = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), wfb_grad_dets)
    assert np.allclose(gc, wfb_grad)
    erase_file('hmf.hdf5')
    
@pytest.mark.boson
def test_boson_gradient_analytical_vs_numerical(H2_ccecp_casci_s0):
    r'''For an N-electron system, where N-1 electrons are fixed, and the Nth electron is moved on a line
    the gradient of the wavefunction can be calculated analytically, and numerically (using np.gradient)'''
    mol, mf, mc = H2_ccecp_casci_s0
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break

    from pyqmc.coord import OpenConfigs
    nconfig = 10000
    # electron 0 is fixed at -0.1, -0.1, -2
    # electron 1 is moved from -0.1, -0.1, -1 to -0.1, -0.1, 1
    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    epos[:, 0, :] = np.linspace([-0.1,-0.1,-2], [-0.1,-0.1,-2], num=nconfig)
    epos[:, 1, :] = np.linspace([-0.1,-0.1,1], [-0.1,-0.1,2], num=nconfig)
    configs = OpenConfigs(epos)
    e = 1
    epos = configs.electron(e)

    _, wfb_value = wfb.recompute(configs)
    wfb_grad = wfb.gradient(e, epos)

    num_grad = np.gradient(wfb_value)
    wfb_grad_z = wfb_grad[2]
    dz = configs.configs[1]-configs.configs[0]
    dz = dz[dz!=0][0]
    assert np.allclose(wfb_grad_z, num_grad/dz, rtol=1e-4)
    erase_file('hmf.hdf5')        

@pytest.mark.boson
def test_boson_gradient_analytical_vs_numerical_triplet(H2_ccecp_casci_s2):
    r'''For an N-electron system, where N-1 electrons are fixed, and the Nth electron is moved on a line
    the gradient of the wavefunction can be calculated analytically, and numerically (using np.gradient)'''
    mol, mf, mc = H2_ccecp_casci_s2
    wfbj, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    # Dont use jastrow factor
    for wf in wfbj.wf_factors:
        if isinstance(wf, BosonWF):
            wfb = wf
            break

    from pyqmc.coord import OpenConfigs
    nconfig = 10000
    # electron 0 is fixed at -0.1, -0.1, -2
    # electron 1 is moved from -0.1, -0.1, -1 to -0.1, -0.1, 1
    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    epos[:, 0, :] = np.linspace([-0.1,-0.1,-2], [-0.1,-0.1,-2], num=nconfig)
    epos[:, 1, :] = np.linspace([-0.1,-0.1,1], [-0.1,-0.1,2], num=nconfig)
    configs = OpenConfigs(epos)
    e = 1
    epos = configs.electron(e)

    _, wfb_value = wfb.recompute(configs)
    wfb_grad = wfb.gradient(e, epos)

    num_grad = np.gradient(wfb_value)
    wfb_grad_z = wfb_grad[2]
    dz = configs.configs[1]-configs.configs[0]
    dz = dz[dz!=0][0]
    assert np.allclose(wfb_grad_z, num_grad/dz, rtol=1e-4)    
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_jastrow_gradient_analytical_vs_numerical(H2_ccecp_casci_s0):
    '''For an N-electron system, where N-1 electrons are fixed, and the Nth electron is moved on a line
    the gradient of the wavefunction can be calculated analytically, and numerically (using np.gradient)'''
    mol, mf, mc = H2_ccecp_casci_s0
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    from pyqmc.coord import OpenConfigs
    nconfig = 10000
    # electron 0 is fixed at -0.1, -0.1, -2
    # electron 1 is moved from -0.1, -0.1, -1 to -0.1, -0.1, 1
    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    epos[:, 0, :] = np.linspace([-0.1,-0.1,-2], [-0.1,-0.1,-2], num=nconfig)
    epos[:, 1, :] = np.linspace([-0.1,-0.1,1], [-0.1,-0.1,2], num=nconfig)
    configs = OpenConfigs(epos)
    e = 1
    epos = configs.electron(e)

    _, wfb_value = wfb.recompute(configs)
    wfb_grad = wfb.gradient(e, epos)

    num_grad = np.gradient(wfb_value)
    wfb_grad_z = wfb_grad[2]
    dz = configs.configs[1]-configs.configs[0]
    dz = dz[dz!=0][0]
    assert np.allclose(wfb_grad_z, num_grad/dz, rtol=1e-4)
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_jastrow_gradient_analytical_vs_numerical_triplet(H2_ccecp_casci_s2):
    '''For an N-electron system, where N-1 electrons are fixed, and the Nth electron is moved on a line
    the gradient of the wavefunction can be calculated analytically, and numerically (using np.gradient)'''
    mol, mf, mc = H2_ccecp_casci_s2
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    
    from pyqmc.coord import OpenConfigs
    nconfig = 10000
    # electron 0 is fixed at -0.1, -0.1, -2
    # electron 1 is moved from -0.1, -0.1, -1 to -0.1, -0.1, 1
    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    epos[:, 0, :] = np.linspace([-0.1,-0.1,-2], [-0.1,-0.1,-2], num=nconfig)
    epos[:, 1, :] = np.linspace([-0.1,-0.1,1], [-0.1,-0.1,2], num=nconfig)
    configs = OpenConfigs(epos)
    e = 1
    epos = configs.electron(e)

    _, wfb_value = wfb.recompute(configs)
    wfb_grad = wfb.gradient(e, epos)

    num_grad = np.gradient(wfb_value)
    wfb_grad_z = wfb_grad[2]
    dz = configs.configs[1]-configs.configs[0]
    dz = dz[dz!=0][0]
    assert np.allclose(wfb_grad_z, num_grad/dz, rtol=1e-4) 
    erase_file('hmf.hdf5')   

@pytest.mark.boson
def test_boson_aboptimize(H2_ccecp_casci_s2):
    # TODO: Optimization test is working, but different from line_minimization in place. 
    # Understand why they give different results.
    '''Test that the auxiliary boson wavefunction is the same as the ab-vmc wavefunction'''
    mol, mf, mc = H2_ccecp_casci_s2
    dm = mf.make_rdm1()
    mf.dm = dm
    
    from pyqmc.bosonlinemin import line_minimization
    from pyqmc.bosonaccumulators import boson_gradient_generator
    nconfig = 1000
    
    configs = initial_guess(mol, nconfig)
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc)
    acc = boson_gradient_generator(
            mf, wfb, to_opt, nodal_cutoff=1e-3
    )
        
    linemin_kws = {'max_iterations': 2}
    
    wf, df = line_minimization(wfb, configs, acc, **linemin_kws)
    erase_file('hmf.hdf5')

@pytest.mark.boson
def test_boson_pgradient(H2_ccecp_casci_s2):
    r'''Partial derivatives of jastrow factor from auxiliary boson and Slater wavefunctions 
    should be identical. AB-VMC has no e-i terms, thus the jastrow_kws below.'''
    
    mol, mf, mc = H2_ccecp_casci_s2
    jastrow_kws = {"ion_cusp":False, "na":0, "nb":3}
    wfb, to_opt = bosonwftools.generate_boson_wf(mol, mf, mc=mc, jastrow_kws=jastrow_kws)
    wfs, to_opt = wftools.generate_wf(mol, mf, mc=mc, jastrow_kws=jastrow_kws)
    
    configs = initial_guess(mol, 10)
    wfb.recompute(configs)
    wfs.recompute(configs)
    
    pgradb = wfb.pgradient()
    pgrads = wfs.pgradient()
    
    for wfb_key, wfb_val in pgradb.items():
        wfs_val = pgrads[wfb_key]
        assert np.allclose(wfb_val, wfs_val)
    
    erase_file('hmf.hdf5')
    
if __name__ == "__main__":
    test_boson_wf()
    test_boson_wf_value()
    test_boson_derivatives()
    test_boson_dets_value_singlet()
    test_boson_dets_value_triplet()
    test_boson_dets_grad_singlet()
    test_boson_dets_grad_triplet()
    test_boson_gradient_analytical_vs_numerical()


# TODO: @pytest.mark.boson
# def test_boson_abvmc_timestep_convergence_he_hf():
#     '''For an AB-HF calculation, the total energy should converge to the same value for different timesteps'''
#     from pyqmc.bosonrecipes import ABVMC
#     import matplotlib.pyplot as plt
    
#     def run_scf(chkfile):
#         erase_file(chkfile)
#         from pyscf import gto, scf
#         mol = gto.M(
#             atom="He 0 0. 0.0", basis="aug-ccpvqz",  unit="bohr", spin = 0
#         )
#         mf = scf.UHF(mol)
#         mf.chkfile = chkfile
#         mf.kernel()
#         return mf

#     def read_abvmc_energies(fname):
#         import h5py
#         with h5py.File(fname, 'r') as f:
#             energies = f['energytotal'][:]
#         return energies

#     def within_3_std(mean_i, std_i, ref_mean, ref_std):
#         if mean_i > ref_mean:
#             assert mean_i - 3*std_i < ref_mean + 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'
#         else:
#             assert mean_i + 3*std_i > ref_mean - 3*ref_std, f'dt={dt_i} is significantly different from dt={dt_list[0]}'

#     # Try parallellization
#     # try:
#     #     import concurrent.futures
#     #     import os
#     #     npartitions = int(os.cpu_count() + 4)
#     #     client = concurrent.futures.ProcessPoolExecutor(max_workers=npartitions)
#     # except:
#     client = None
#     npartitions = 1
#     #end try 


#     dft_checkfile = 'he_scf.hdf5'
#     mf = run_scf(dft_checkfile)

#     e_results = []
#     dt_list = [1, 0.3, 0.1, 0.03, 0.01]
#     discard = 100
#     for dt in dt_list:
#         abvmc_filename = f'he_abvmc_{dt}.hdf5'
#         erase_file(abvmc_filename)

#         wf, configs, acc = ABVMC(
#             dft_checkfile=dft_checkfile,
#             output=abvmc_filename,
#             nconfig=100,
#             tstep=dt,
#             nblocks=200,
#             nsteps_per_block=10,
#             load_parameters=False, 
#             seed = 1,
#             client = client,
#             npartitions = npartitions,
#             xc = 'HF',
#         )
#         e = read_abvmc_energies(abvmc_filename)
#         e_results.append(e[discard:])
#         plt.plot(e, label=f'dt={dt}')
        
#     erase_file(dft_checkfile)
#     for dt in dt_list:
#         abvmc_filename = f'he_abvmc_{dt}.hdf5'
#         erase_file(abvmc_filename)
    
#     plt.legend()
#     plt.show()
#     import pdb; pdb.set_trace()
#     e_ref = e_results[0]  
#     ref_mean = np.mean(e_ref)
#     ref_std = np.std(e_ref)
#     for i, dt_i in enumerate(dt_list[1:]):
#         mean_i = np.mean(e_results[i+1])
#         std_i = np.std(e_results[i+1])
#         within_3_std(mean_i, std_i, ref_mean, ref_std)    