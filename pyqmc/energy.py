import numpy as np
import pyqmc.eval_ecp as eval_ecp
import pyqmc.distance as distance
from pyscf.dft import numint, libxc

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

def vxc_energy(mf, configs):
    xc = 'LDA,VWN'
    mol = mf.mol
    nconf, nelec, ndim = configs.configs.shape
    vxc = np.zeros(nconf)
    for s in [0,1]:
        ao_value = numint.eval_ao(mol, configs.configs[:,s,:])
        dm = mf.make_rdm1()
        rho = numint.eval_rho(mol, ao_value, dm, xctype='LDA')
        exc, vxcs = libxc.eval_xc(xc, rho)[:2]
        vxc += vxcs[0]
    return vxc

def boson_kinetic(configs, wf):
    nconf, nelec, ndim = configs.configs.shape
    ke = np.zeros(nconf)
    wave_functions = wf.wf_factors
    jastrow_wf = None
    boson_wf = None
    import pdb
    
    from bosonwf import BosonWF
    from pyqmc.jastrowspin import JastrowSpin
    for wave in wave_functions:
        if isinstance(wave, BosonWF):
            boson_wf = wave
        if isinstance(wave, JastrowSpin):
            jastrow_wf = wave
    
    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    pdb.set_trace()
    for e in range(nelec):
        grad_j, lap = jastrow_wf.gradient_laplacian(e, configs.electron(e))
        lap_j += -0.5  * lap.real
        grad_b = boson_wf.gradient(e, configs.electron(e))
        drift_b += np.einsum("di,di->i", grad_j, grad_b) #dim,config
        # TODO: check if division is required
    ke = lap_j + drift_b
    return ke

def boson_drift(configs, wf):
    nconf, nelec, ndim = configs.configs.shape
    ke = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    for e in range(nelec):
        grad, lap = wf.gradient_laplacian(e, configs.electron(e))
        ke += -0.5 * lap.real
        grad2 += np.sum(np.abs(grad) ** 2, axis=0)
    return ke, grad2