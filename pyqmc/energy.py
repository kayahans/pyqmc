import numpy as np
import pyqmc.eval_ecp as eval_ecp
import pyqmc.distance as distance
from pyscf.dft import numint

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

def vxc_energy(mf, box, configs):
    # Will be tested on LDA only
    mol = mf.mol
    grids = mf.grids
    dm = mf.make_rdm1()
    nelec, ex, vx = numint.nr_vxc(mol, grids, mf.xc, dm)
    ao_value = numint.eval_ao(mol, configs)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    c0 = numint._dot_ao_dm(mol, ao_value, vx, None, shls_slice, ao_loc)
    rho = numint._contract_rho(ao_value, c0)
    rhod = rho.reshape((box.ys, box.xs, box.zs), order='C')
    return rhod
