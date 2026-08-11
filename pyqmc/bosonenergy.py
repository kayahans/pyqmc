import numpy as np
from pyscf.dft import numint, libxc


def dft_energy(mf_inputs, configs):
    '''
    Returns the KS related terms in  Eq. 21 in doi: 10.1063/5.0155513. 
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
    grids = mf_inputs['grids']
    rho = mf_inputs['rho']

    def get_vj(configs):
        dm_total = dm[0] + dm[1]
        r = configs.configs.reshape(-1, 3)
        vj_all = np.einsum('pij,ij->p', mol.intor('int1e_grids', grids=r), dm_total)
        vj = vj_all.reshape(configs.configs.shape[0], nelec).sum(axis=1)
        return vj

    def get_vxc(configs, xc_func='LDA,VWN'):
        if xc_func == 'LDA,VWN':
            xctype = 'LDA'
            deriv = 0
        elif xc_func == 'PBE,PBE':
            xctype = 'GGA'
            deriv = 1
        else:
            raise ValueError(f"Unsupported XC functional: {xc_func}")

        r = configs.configs.reshape(-1, 3)
        s = np.array([int(e >= nup_dn[0]) for e in range(nelec)])
        ao_value = numint.eval_ao(mol, r, deriv=deriv)
        rho_up = numint.eval_rho(mol, ao_value, dm[0], xctype=xctype)
        rho_down = numint.eval_rho(mol, ao_value, dm[1], xctype=xctype)
        vxcs = libxc.eval_xc(xc_func, (rho_up, rho_down), spin=len(nup_dn)-1)[1][0]
        vxcs = vxcs.reshape(nconf, nelec, -1)
        vxc = np.sum([vxcs[:, i, s[i]] for i in range(nelec)], axis=0)
        return vxc

    if xc is not None and xc != 'HF':
        vj = get_vj(configs)
        vxc = get_vxc(configs, xc_func=xc)
        ecorr = np.sum(mo_energy * mo_occ)
        v_mf = vj + vxc
        saved_results = {'vj': vj, 'vxc': vxc}
    elif xc == 'HF':
        v_mf = np.zeros(nconf)
        ecorr = np.sum(mo_energy * mo_occ)
        V_eff_ao = mf_inputs['veff']
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            ao_value = numint.eval_ao(mol, configs.configs[:, e, :])
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
    except AttributeError:
        has_jastrow = False
        wave_functions = [wf]

    jastrow_wf = None
    boson_wf = None
    from pyqmc import bosonslater
    from pyqmc import jastrowspin
    for wave in wave_functions:
        if isinstance(wave, bosonslater.BosonWF):
            boson_wf = wave
        if isinstance(wave, jastrowspin.JastrowSpin):
            jastrow_wf = wave

    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    if has_jastrow:
        for e in range(nelec):
            grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            lap_j += -0.5 * (lap_je.real + np.sum(grad_je.real**2, axis=0))
            grad_b = boson_wf.gradient(e, configs.electron(e))
            drift_b -= np.einsum("di,di->i", grad_je, grad_b)
            grad = np.sum([grad_je, grad_b], axis=0)
            grad2 += np.sum(np.abs(grad) ** 2, axis=0)
    return lap_j, drift_b, grad2
