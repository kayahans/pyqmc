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
        
        self.num_det = mc.ci.shape[0] * mc.ci.shape[1]

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
        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        iscomplex = self.orbitals.mo_dtype == complex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if iscomplex else float

        self.get_phase = get_complex_phase if iscomplex else gpu.cp.sign

    def recompute(self, configs):
        """This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""

        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos("GTOval_sph", configs)
        self._aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            self._dets.append(
                gpu.cp.asarray(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]

            is_zero = np.sum(np.abs(self._dets[s][0]) < 1e-16)
            compute = np.isfinite(self._dets[s][1])
            if is_zero > 0:
                warnings.warn(
                    f"A wave function is zero. Found this proportion: {is_zero/nconf}"
                )
                # print(configs.configs[])
                print(f"zero {is_zero/np.prod(compute.shape)}")
            self._inverse.append(gpu.cp.zeros(mo_vals.shape, dtype=mo_vals.dtype))
            for d in range(compute.shape[1]):
                self._inverse[s][compute[:, d], d, :, :] = gpu.cp.linalg.inv(
                    mo_vals[compute[:, d], d, :, :]
                )
            # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """Update any internals given that electron e moved to epos. mask is a Boolean array
        which allows us to update only certain walkers"""

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = np.ones(epos.configs.shape[0], dtype=bool)
        is_zero = np.sum(np.isinf(self._dets[s][1]))
        if is_zero:
            warnings.warn(
                "Found a zero in the wave function. Recomputing everything. This should not happen often."
            )
            self.recompute(configs)
            return

        eeff = e - s * self._nelec[0]
        if saved_values is None:
            ao = self.orbitals.aos("GTOval_sph", epos, mask)
            self._aovals[:, mask, e, :] = ao
            mo = self.orbitals.mos(ao, s)
        else:
            ao, mo = saved_values
            self._aovals[:, mask, e, :] = ao[:, mask]
            mo = mo[mask]
        mo_vals = mo[:, self._det_occup[s]]
        det_ratio, self._inverse[s][mask, :, :, :] = sherman_morrison_ms(
            eeff, self._inverse[s][mask, :, :, :], mo_vals
        )
        self._dets[s][0, mask, :] *= self.get_phase(det_ratio)
        self._dets[s][1, mask, :] += gpu.cp.log(gpu.cp.abs(det_ratio))

    def value(self):
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]

        upref = gpu.cp.amax(updets[1]).real
        dnref = gpu.cp.amax(dndets[1]).real
        det_coeff = self.myparameters['det_coeff']
        logvals = 2*(updets[1] - upref + dndets[1] - dnref)
        wf_val = gpu.cp.einsum("d, id->i", det_coeff, gpu.cp.exp(logvals))

        wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
        wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + 2*(upref + dnref))        
        return wf_sign, wf_logval

    def value_dets(self, test = False):
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]

        # upref = gpu.cp.amax(updets[1]).real
        # dnref = gpu.cp.amax(dndets[1]).real
        # det_coeff = self.myparameters['det_coeff']
        
        # logvals = (updets[1] - upref + dndets[1] - dnref)
        # wf_val = gpu.cp.einsum("id->di", gpu.cp.exp(logvals))

        # wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
        # wf_logval = np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + (upref + dnref))        
        
        wf_logval = (updets[1] + dndets[1])
        wf_sign = updets[0] * dndets[0]

        if test:
            det_coeff = self.myparameters['det_coeff']
            tol = 1E-15
            phi_b = 1./2 * np.log(np.einsum('d, di->i', det_coeff,np.exp(2*wf_logval) ))
            assert ((np.abs(phi_b - self.value()[1]) < tol).all())
        return wf_sign, wf_logval

    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        #returns \nabla ln(\Phi_B)=\frac{\nabla \Phi_B}{\Phi_B}
        #= \frac{\sum{\nabla \Phi_n*\Phi_n}}{\Phi_B^2}
        #= \frac{\sum{\Phi_n^2 * \nabla ln(\Phi_n)}}{\Phi_B^2}
        #= \frac{\sum{exp(2*ln(\Phi_n)) * \nabla ln(\Phi_n)}}{exp(2*ln(\Phi_B))}
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        det_coeff = self.myparameters['det_coeff']
        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )

        jacobid = jacobi[..., self._det_map[s]]
        jacobid = jacobid[1:]/jacobid[0]

        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )
        grad = numer / denom
        grad[~np.isfinite(grad)] = 0.0
        return grad
    
    def gradient_value(self, e, epos):
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        det_coeff = self.myparameters['det_coeff']
        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )

        jacobid = jacobi[..., self._det_map[s]]
        jacobid = jacobid[1:]/jacobid[0]

        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )
        derivatives = numer / denom
        derivatives[~np.isfinite(derivatives)] = 0.0
        values = derivatives[0]
        values[~np.isfinite(values)] = 1.0
        return derivatives, values, (aograd[:, 0], mograd[0])

    def gradient_dets(self, e, epos, test=False):
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        det_coeff = self.myparameters['det_coeff']
        jac =  gpu.cp.einsum(
            "ei...d->dei...",
            jacobi[..., self._det_map[s]],
        )
        grads = jac[:, 1:, :]

        if test:
            dv = self.value_dets()[1]
            v = self.value()[1]
            gc = np.einsum('d, di,dei->ei', det_coeff, np.exp(2*(dv-v)), grads)
            assert((gc == self.gradient(e, epos)).all())
        return grads

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