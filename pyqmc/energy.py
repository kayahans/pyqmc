import numpy as np
import pyqmc.eval_ecp as eval_ecp
import pyqmc.distance as distance
from pyscf.dft import numint, libxc, gen_grid

class OpenCoulomb:
    def __init__(self, mol):
        self.mol = mol
        self.ii_energy = ii_energy(self.mol)

    def energy(self, configs):
        return ee_energy(configs), ei_energy(self.mol, configs), self.ii_energy


def ee_energy(configs):
    ne = configs.configs.shape[1]
    if ne == 1:
        return np.zeros(configs.configs.shape[0])
    ee = np.zeros(configs.configs.shape[0])
    ee, ij = configs.dist.dist_matrix(configs.configs)
    ee = np.linalg.norm(ee, axis=2)
    return np.sum(1.0 / ee, axis=1)


def ei_energy(mol, configs):
    ei = 0.0
    for c, coord in zip(mol.atom_charges(), mol.atom_coords()):
        delta = configs.configs - coord[np.newaxis, np.newaxis, :]
        deltar = np.sqrt(np.sum(delta**2, axis=2))
        ei += -c * np.sum(1.0 / deltar, axis=1)
    return ei

def ii_energy(mol):
    d = distance.RawDistance()
    rij, ij = d.dist_matrix(mol.atom_coords()[np.newaxis, :, :])
    if len(ij) == 0:
        return np.array([0.0])
    rij = np.linalg.norm(rij, axis=2)[0, :]
    c = mol.atom_charges()
    return sum(c[i] * c[j] / r for (i, j), r in zip(ij, rij))


def kinetic(configs, wf):
    nconf, nelec, ndim = configs.configs.shape
    ke = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    for e in range(nelec):
        grad, lap = wf.gradient_laplacian(e, configs.electron(e))
        ke += -0.5 * lap.real
        grad2 += np.sum(np.abs(grad) ** 2, axis=0)
    return ke, grad2

#kayahan added below
def dft_energy(mf, configs, nup_dn):
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
    mol = mf.mol
    nconf, nelec, ndim = configs.configs.shape
    #Hartree potential
    dm = mf.make_rdm1()
    vj = np.zeros(nconf)
    #Eigenvalue sum
    ecorr = np.sum(mf.mo_energy*mf.mo_occ) 
    #Vxc potential
    vxc = np.zeros(nconf)
    _ = mf.energy_tot()
    dm = mf.make_rdm1()
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
    Returns the jastrow laplacian and the bosonic drift terms in Eq. 21 in doi: 10.1063/5.0155513. 
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
    from bosonwf import BosonWF
    from jastrowspin import BosonJastrowSpin
    for wave in wave_functions:
        if isinstance(wave, BosonWF):
            boson_wf = wave
        if isinstance(wave, BosonJastrowSpin):
            jastrow_wf = wave
    
    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    if has_jastrow:
        for e in range(nelec):
            grad_j, lap = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            # Convert to exp form of jastrow gradients from the jastrow log wavefunction
            # \frac{\nabla{e^{-U(r)}}}{e^{-U(r)}} = {\nabla^2}U(r) + ({\nabla}U(r))^2
            lap_j += -0.5  * (lap.real + np.einsum("di,di->i",grad_j,np.conjugate(grad_j)))
            grad_b = boson_wf.gradient(e, configs.electron(e))
            drift_b += np.einsum("di,di->i", -grad_j, grad_b)/boson_wf.value()[1]
            # TODO: check if division is required
        # ke = lap_j + drift_b
    return lap_j , drift_b

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