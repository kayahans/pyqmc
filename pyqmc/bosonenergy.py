import numpy as np
from pyscf.dft import numint, libxc

#kayahan added below
def dft_energy(mf_inputs, configs):
    '''
    Returns the KS related terms in  Eq. 21 in doi: 10.1063/5.0155513. 
    MF is assumed to be LDA ('LDA, VWN'), therefore, for another input DFT functional, 
    this may not work as intended.
    Returns: 
        vj: Electrostatic potential
        vxc: XC potential
        ecorr: sum of the occupied KS eigenvalues (E_0^MF)
    '''
    nconf, nelec, ndim = configs.configs.shape
    v_mf = 0
    xc = mf_inputs['xc']
    nup_dn = mf_inputs['nelec']
    mo_energy = mf_inputs['mo_energy']
    mo_occ = mf_inputs['mo_occ']
    mol = mf_inputs['mol']
    dm = mf_inputs['dm']
    # grids = mf_inputs['grids']
    # rho = mf_inputs['rho']

    def get_vj(configs):
        vj = 0
        dm_total = dm[0] + dm[1]
        for e in range(nelec):
            # Fast (x10^3)
            r = configs.configs[:,e,:]
            vj += np.einsum('pij,ij->p', mol.intor('int1e_grids', grids=r), dm_total)
            # Slow 
            # for i, r in enumerate(configs.configs[:,e,:]):
                # distances = np.linalg.norm(grids.coords - r, axis=1)
                # mask = distances > 1e-2 # Do not include grids that are very close
                # vj[i] += np.sum(rho[mask] / distances[mask] * grids.weights[mask])    
        return vj

    def get_vxc(configs):
        vxc = 0
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            r = configs.configs[:,e,:]
            ao = numint.eval_ao(mol, r, deriv=0)
            rho_up = np.einsum('pi,ij,pj->p', ao, dm[0], ao)
            rho_down = np.einsum('pi,ij,pj->p', ao, dm[1], ao)
            _, vxcs, _, _  = libxc.eval_xc(xc, np.array([rho_up, rho_down]), spin = len(nup_dn)-1)
            vxc += vxcs[0][:,s]
        return vxc
    
    if xc == 'LDA,VWN':
        vj = get_vj(configs)
        vxc = get_vxc(configs)
        ecorr = np.sum(mo_energy*mo_occ) 
        # Older code for reference
        # vj = np.zeros(nconf)
        # vxc = np.zeros(nconf)
        # for e in range(nelec):
        #     s = int(e >= nup_dn[0])
        #     ao_value = numint.eval_ao(mol, configs.configs[:,e,:])
        #     # rho_u = numint.eval_rho(mol, ao_value, dm[0], xctype='LDA')
        #     # rho_d = numint.eval_rho(mol, ao_value, dm[1], xctype='LDA')
        #     rho_u = np.einsum('pi,ij,pj->p', ao_value, dm[0], ao_value)
        #     rho_d = np.einsum('pi,ij,pj->p', ao_value, dm[0], ao_value)

        #     excd, vxcs  = libxc.eval_xc(xc, np.array([rho_u, rho_d]), spin=1)[:2]
        #     vxc += vxcs[0][:,s]
        #     vj += np.einsum('pij,sij->p', mol.intor('int1e_grids', grids=configs.configs[:,e,:]), dm)
        # #end for 
        v_mf = vj + vxc
        saved_results = {'vj': vj, 'vxc': vxc}
    elif xc == 'HF':
        v_mf = np.zeros(nconf)
        ecorr = np.sum(mo_energy*mo_occ) 
        V_eff_ao = mf_inputs['veff']
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            ao_value = numint.eval_ao(mol, configs.configs[:,e,:])
            v_mf = np.einsum('gp, pq, gq -> g', ao_value, V_eff_ao[s], ao_value)
        saved_results = {}
        
    return v_mf, ecorr, saved_results

def boson_kinetic(configs, wf):
    '''
    Returns the jastrow laplacian (lap_j) and the bosonic drift (drift_b) terms 
    in Eq. 21 in doi: 10.1063/5.0155513. 
    '''
    nconf, nelec, _ = configs.configs.shape
    
    has_jastrow = True
    try:
        wave_functions = wf.wf_factors
    except:
        has_jastrow = False
        wave_functions = [wf]
    
    jastrow_wf = None
    boson_wf = None
    from pyqmc import bosonslater
    from pyqmc import jastrowspin
    # from pyqmc.bosonslater import BosonWF
    # from pyqmc.jastrowspin import JastrowSpin
    for wave in wave_functions:
        if isinstance(wave, bosonslater.BosonWF):
            boson_wf = wave
        if isinstance(wave, jastrowspin.JastrowSpin):
            jastrow_wf = wave
    
    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    if has_jastrow:
        # If no jastrows (HF), then these terms are zero
        for e in range(nelec):
            grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            lap_j += -0.5 * lap_je.real
            grad_b = boson_wf.gradient(e, configs.electron(e))
            drift_b += np.einsum("di,di->i", grad_je, grad_b)
            grad = wf.gradient(e, configs.electron(e))
            grad2 += np.sum(np.abs(grad) ** 2, axis=0)
        # ke = lap_j + drift_b
    return lap_j, drift_b, grad2

# def boson_kinetic(configs, wf):
#     '''
#     Returns the jastrow laplacian (lap_j) and the bosonic drift (drift_b) terms 
#     in Eq. 21 in doi: 10.1063/5.0155513. 
#     '''
#     nconf, nelec, ndim = configs.configs.shape
#     ke = np.zeros(nconf)
#     has_jastrow = True
#     try:
#         wave_functions = wf.wf_factors
#     except:
#         has_jastrow = False
#         wave_functions = [wf]
#     jastrow_wf = None
#     boson_wf = None
#     from bosonslater import BosonWF
#     from jastrowspin import JastrowSpin
#     for wave in wave_functions:
#         if isinstance(wave, BosonWF):
#             boson_wf = wave
#         if isinstance(wave, JastrowSpin):
#             jastrow_wf = wave
    
#     lap_j = np.zeros(nconf)
#     drift_b = np.zeros(nconf)
#     grad2 = np.zeros(nconf)
#     if has_jastrow:
#         # If no jastrows (HF), then these terms are zero
#         for e in range(nelec):
#             _, val_je = jastrow_wf.value()            
#             grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
#             # Convert to exp form of jastrow gradients from the jastrow log wavefunction
#             # If \Psi_J = exp(J)
#             # \frac{\nabla{e^{U(r)}}}{e^{U(r)}} = {\nabla^2}U(r) + {\nabla}U(r) \cdot {\nabla}U(r)}
#             # If \Psi_J = exp(-J)
#             # \frac{\nabla{e^{-U(r)}}}{e^{-U(r)}} = - [{\nabla^2}U(r) - {\nabla}U(r) \cdot {\nabla}U(r)}]
#             # import pdb
#             # pdb.set_trace()
#             lap_j += -0.5 * (lap_je.real + np.sum((grad_je.real)**2, axis=0))
#             # lap_j += -0.5 * lap_je.real
#             # lap_j += 0.5  * (lap_je.real + np.einsum("di,di->i",grad_j,np.conjugate(grad_j)))
#             grad_b = boson_wf.gradient(e, configs.electron(e))
#             drift_b += np.einsum("di,di->i", -grad_je, grad_b)
#             grad = wf.gradient(e, configs.electron(e))
#             grad2 += np.sum(np.abs(grad) ** 2, axis=0)
#         # ke = lap_j + drift_b
#     return lap_j, drift_b, grad2

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