import numpy as np
from pyscf.dft import numint, libxc

#kayahan added below
def dft_energy(mol, dm, mo_energy, mo_occ, configs, nup_dn):
    '''
    Returns the KS related terms in  Eq. 21 in doi: 10.1063/5.0155513. 
    MF is assumed to be LDA ('LDA, VWN'), therefore, for another input DFT functional, 
    this may not work as intended.
    Returns: 
        vj: Electrostatic potential
        vxc: XC potential
        ecorr: sum of the occupied KS eigenvalues (E_0^MF)
    '''
    xc = 'LDA,VWN'
    mol = mol
    nconf, nelec, ndim = configs.configs.shape
    #Hartree potential
    dm = dm
    vj = np.zeros(nconf)
    #Eigenvalue sum
    ecorr = np.sum(mo_energy*mo_occ) 
    #Vxc potential
    vxc = np.zeros(nconf)
    # _ = mf.energy_tot() # Why do we need this? TODO: check later, disable now
    for e in range(nelec):
        s = int(e >= nup_dn[0])
        ao_value = numint.eval_ao(mol, configs.configs[:,e,:])
        rho_u = numint.eval_rho(mol, ao_value, dm[0], xctype='LDA')
        rho_d = numint.eval_rho(mol, ao_value, dm[1], xctype='LDA')
        excd, vxcs  = libxc.eval_xc(xc, np.array([rho_u, rho_d]), spin=1)[:2]
        vxc += vxcs[0][:,s]
        vj += np.einsum('pij,sij->p', mol.intor('int1e_grids', grids=configs.configs[:,e,:]), dm)
    return vj, vxc, ecorr

def boson_kinetic(configs, wf):
    '''
    Returns the jastrow laplacian (lap_j) and the bosonic drift (drift_b) terms 
    in Eq. 21 in doi: 10.1063/5.0155513. 
    '''
    nconf, nelec, ndim = configs.configs.shape
    ke = np.zeros(nconf)
    has_jastrow = True
    try:
        wave_functions = wf.wf_factors
    except:
        has_jastrow = False
        wave_functions = [wf]
    jastrow_wf = None
    boson_wf = None
    from bosonslater import BosonWF
    from jastrowspin import JastrowSpin
    for wave in wave_functions:
        if isinstance(wave, BosonWF):
            boson_wf = wave
        if isinstance(wave, JastrowSpin):
            jastrow_wf = wave
    
    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    if has_jastrow:
        # If no jastrows (HF), then these terms are zero
        for e in range(nelec):
            _, val_je = jastrow_wf.value()            
            grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            # Convert to exp form of jastrow gradients from the jastrow log wavefunction
            # If \Psi_J = exp(J)
            # \frac{\nabla{e^{U(r)}}}{e^{U(r)}} = {\nabla^2}U(r) + {\nabla}U(r) \cdot {\nabla}U(r)}
            # If \Psi_J = exp(-J)
            # \frac{\nabla{e^{-U(r)}}}{e^{-U(r)}} = - [{\nabla^2}U(r) - {\nabla}U(r) \cdot {\nabla}U(r)}]
            # import pdb
            # pdb.set_trace()
            lap_j += -0.5 * (lap_je.real + np.sum((grad_je.real)**2, axis=0))
            # lap_j += -0.5 * lap_je.real
            # lap_j += 0.5  * (lap_je.real + np.einsum("di,di->i",grad_j,np.conjugate(grad_j)))
            grad_b = boson_wf.gradient(e, configs.electron(e))
            grad = wf.gradient(e, configs.electron(e))
            grad2 += np.sum(np.abs(grad) ** 2, axis=0)
            drift_b += np.einsum("di,di->i", -grad_je, grad_b)
        # ke = lap_j + drift_b
    return lap_j, drift_b, grad2

# def boson_drift(configs, wf):
#     # TODO: Check where this is used
#     nconf, nelec, ndim = configs.configs.shape
#     ke = np.zeros(nconf)
#     grad2 = np.zeros(nconf)
#     for e in range(nelec):
#         grad, lap = wf.gradient_laplacian(e, configs.electron(e))
#         ke += -0.5 * lap.real
#         grad2 += np.sum(np.abs(grad) ** 2, axis=0)
#     return ke, grad2