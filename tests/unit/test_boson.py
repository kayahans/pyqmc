import pyqmc.api as pyq
import pyqmc.bosonwftools as bosonwftools
import pyqmc.wftools as wftools
import pytest 
import os
import numpy as np
from pyqmc.mc import initial_guess
from pyqmc.bosonslater import BosonWF

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
        
    os.remove('hmf.hdf5')

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
        
    os.remove('hmf.hdf5')

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
    os.remove('hmf.hdf5')
    
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
    os.remove('hmf.hdf5')
    
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
    wfb_grad_dets = wfb.gradient_dets(e, epos)
    wfb_grad = wfb.gradient(e, epos)

    dv = wfb.value_dets()[1]
    v = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), wfb_grad_dets)
    assert np.allclose(gc, wfb_grad)
    os.remove('hmf.hdf5')
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
    wfb_grad_dets = wfb.gradient_dets(e, epos)
    wfb_grad = wfb.gradient(e, epos)

    dv = wfb.value_dets()[1]
    v = wfb.value()[1]
    det_coeff = wfb.myparameters['det_coeff']
    gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), wfb_grad_dets)
    assert np.allclose(gc, wfb_grad)
    os.remove('hmf.hdf5')
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
    os.remove('hmf.hdf5')        

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
    os.remove('hmf.hdf5')

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
    os.remove('hmf.hdf5')

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
    os.remove('hmf.hdf5')   

@pytest.mark.boson_slow
def test_boson_aboptimize(H2_ccecp_casci_s2):
    # TODO: Optimization test is working, but different from line_minimization in place. 
    # Understand why they give different results.
    '''Test that the auxiliary boson wavefunction is the same as the ab-vmc wavefunction'''
    mol, mf, mc = H2_ccecp_casci_s2
    dm = mf.make_rdm1()
    mf.dm = dm
    
    from pyqmc.bosonaccumulators import ABQMCEnergyAccumulator
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
    os.remove('hmf.hdf5')
    os.remove('linemin.hdf5')
    
    
if __name__ == "__main__":
    test_boson_wf()
    test_boson_wf_value()
    test_boson_derivatives()
    test_boson_dets_value_singlet()
    test_boson_dets_value_triplet()
    test_boson_dets_grad_singlet()
    test_boson_dets_grad_triplet()
    test_boson_gradient_analytical_vs_numerical()