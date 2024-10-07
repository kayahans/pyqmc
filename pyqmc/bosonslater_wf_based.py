import numpy as np
import pyqmc.gpu as gpu
import warnings
import pyscftools
import copy
from wftools import generate_slater

def sherman_morrison_row(e, inv, vec):
    tmp = np.einsum("ek,ekj->ej", vec, inv)
    ratio = tmp[:, e]
    inv_ratio = inv[:, :, e] / ratio[:, np.newaxis]
    invnew = inv - np.einsum("ki,kj->kij", inv_ratio, tmp)
    invnew[:, :, e] = inv_ratio
    return ratio, invnew


def get_complex_phase(x):
    return x / np.abs(x)


class JoinParameters:
    """
    This class provides a dict-like interface that actually references
    other dictionaries in the background.
    If keys collide, then the first dictionary that matches the key will be returned.
    However, some bad things may happen if you have colliding keys.
    """

    def __init__(self, dicts):
        self.data = {}
        self.data = dicts

    def find_i(self, idx):
        for i, d in enumerate(self.data):
            if idx in d:
                return i

    def __setitem__(self, idx, value):
        i = self.find_i(idx)
        self.data[i][idx] = value

    def __getitem__(self, idx):
        i = self.find_i(idx)
        return self.data[i][idx]

    def __delitem__(self, idx):
        i = self.find_i(idx)
        del self.data[i][idx]

    def __iter__(self):
        for d in self.data:
            yield from d.keys()

    def __len__(self):
        return sum(len(i) for i in self.data)

    def items(self):
        for d in self.data:
            yield from d.items()

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        for d in self.data:
            yield from d.keys()

    def values(self):
        for d in self.data:
            yield from d.values()


def sherman_morrison_ms(e, inv, vec):
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    ratio = tmp[:, :, e]
    inv_ratio = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    invnew = inv - np.einsum("kdi,kdj->kdij", inv_ratio, tmp)
    invnew[:, :, :, e] = inv_ratio
    return ratio, invnew

def compute_boson_value(updets, dndets, det_coeffs):
    """
    Given the up and down determinant values, safely compute the total log wave function.
    """
    upref = gpu.cp.amax(updets[1]).real
    dnref = gpu.cp.amax(dndets[1]).real
    logvals = 2*(updets[1] - upref + dndets[1] - dnref)
    wf_val = gpu.cp.einsum("d,id->i", det_coeffs, gpu.cp.exp(logvals))
    
    wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
    wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + 2*(upref + dnref))
    return gpu.asnumpy(wf_sign), gpu.asnumpy(wf_logval)
class BosonWF:

    def __init__(self, mol, mf, mc=None, tol=None, twist=None, determinants=None, eval_gto_precision=None):
        """
        Create Bosonic wavefunction
        Args:
            mol (_type_): A Mole object
            mf (_type_): a pyscf mean-field object
            mc (_type_, optional): a pyscf multiconfigurational object. Supports HCI and CAS. Defaults to None.
            tol (_type_, optional): smallest determinant weight to include in the wave function. Defaults to None.
            twist (_type_, optional): the twist of the calculation. Defaults to None.
            determinants (_type_, optional): A list of determinants suitable to pass into create_packed_objects. Defaults to None.

            You cannot pass both mc/tol and determinants.
        """
        self.tol = -1 if tol is None else tol
        self._mol = mol

        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            ncore = mc.ncore
            if not hasattr(ncore, "__len__"):
                ncore = [ncore, ncore]
            self._nelec = (mc.nelecas[0] + ncore[0], mc.nelecas[1] + ncore[1])
        else:
            self._nelec = mol.nelec
        self.eval_gto_precision = eval_gto_precision
        
        self.wfs = []
        self.num_det = mc.ci.shape[0] * mc.ci.shape[1]

        for i in range(mc.ci.shape[0]):
            for j in range(mc.ci.shape[1]):
                mc0 = copy.copy(mc)
                mc0.ci = mc.ci * 0
                mc0.ci[i, j] = 1 # zero all coefficients except one, this is probably quite inefficient, but should work for now.
                wf0, _ = generate_slater(mol, mf, mc=mc0, optimize_determinants=False)
                self.wfs.append(wf0)

        self.myparameters = {}
        (
            det_coeff,
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = pyscftools.orbital_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants, tol=self.tol, eval_gto_precision=self.eval_gto_precision
        )

        # Use constant weight 
        self.myparameters["det_coeff"] = np.ones(self.num_det)/self.num_det
        # self.myparameters["det_coeff"] = det_coeff
        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        self.dtype = float

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        for wf in self.wfs:
            _ = wf.recompute(configs)
        return self.value()

    
    def value(self):
        value, real_values, _ = self.value_all()
        sign = np.ones(value.shape)
        return sign, value

    def value_all(self):
        # value returns ln(\Phi_B)=ln(\sqrt{\sum_n\Phi_n^2})=1/2*ln\sum_n{exp(2*ln(\Phi_n))})
        # ln(\Phi_B)=ln(\sqrt{1/N * \sum_n\Phi_n^2})=1/2*ln[1/N *\sum_n{exp(2*ln(\Phi_n))}]) 
        values = []
        signs = []
        num_wf = len(self.wfs)
        ci = 1./num_wf
        for wf in self.wfs:
            wf_phase, wf_value = wf.value()
            values.append(np.exp(2*wf_value))
            signs.append(wf_phase)
        
        real_values = gpu.asnumpy(np.array(values))
        signs = gpu.asnumpy(np.array(signs))

        value = 1./2 * np.log(ci * np.einsum('di->i',real_values))
        return value, real_values, signs


    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array
        which allows us to update only certain walkers"""

        for wf in self.wfs:
            wf.updateinternals(e, epos, configs, mask=mask, saved_values=saved_values)

    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        #returns \nabla ln(\Phi_B)=\frac{\nabla \Phi_B}{\Phi_B}
        #= \frac{\sum{\nabla \Phi_n*\Phi_n}}{\Phi_B^2}
        #= \frac{\sum{\Phi_n^2 * \nabla ln(\Phi_n)}}{\Phi_B^2}
        #= \frac{\sum{exp(2*ln(\Phi_n)) * \nabla ln(\Phi_n)}}{exp(2*ln(\Phi_B))}
        grads = []
        num_wf = len(self.wfs)
        ci = 1./num_wf
        for wf in self.wfs: # over n 
            wf_grad = wf.gradient(e, epos)
            grads.append(wf_grad)
        grads = gpu.asnumpy(np.array(grads))
        
        value, real_values, _ = self.value_all()
        grad = ci * np.einsum('di, dgi, i->gi', real_values, grads, 1./np.exp(2*value))
        
        return grad
    
    def gradient_value(self, e, epos):
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        grads = []
        ratios = []
        num_wf = len(self.wfs)
        ci = 1./num_wf        
        for wf in self.wfs: # over n 
            wf_grad, wf_ratio, _ = wf.gradient_value(e, epos)
            grads.append(wf_grad)
            ratios.append(wf_ratio)
        grads = gpu.asnumpy(np.array(grads))

        value, real_values, _ = self.value_all()
        grad = ci * np.einsum('di, dgi, i->gi', real_values, grads, 1./np.exp(2*value))

        ratios = gpu.asnumpy(np.array(ratios))
        # import pdb
        # pdb.set_trace()
        # ratiosf = np.ones(ratios.shape[1])
        ratiosf = np.mean(ratios, axis=0)

        # ratiosf = np.sqrt(np.sum(ratios**2, axis=0))
        # ratiosf = np.average(ratios, weights=1./real_values, axis=0)
        # ratiosf = np.sqrt(np.average(ratios, weights=real_values, axis=0))/np.exp(value)
        # ratiosf = np.average(ratios, weights=real_values, axis=0)/np.exp(2*value)
        # import pdb
        # pdb.set_trace()
        # ratiosf = np.ones(ratiosf.shape)/2
        return grad, ratiosf, (aograd[:, 0], mograd[0])


    def pgradient(self):
        # Has not been fully implemented
        d = {}
        
        # value, real_values, signs = self.value_all()
        
        # pgradients = []
        # for wf in self.wfs:
        #     pgradients.append(wf.pgradient())
        
        # real_value = np.exp(2*value)
        # d["det_coeff"] = 1./2 * real_values**2 / real_value

        return d