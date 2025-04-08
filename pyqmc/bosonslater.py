import numpy as np
import pyqmc.gpu as gpu
import warnings
import pyqmc
import copy
from pyqmc.wftools import generate_slater
import h5py
import time
report_timer = False
def timer_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        wrapper.total_time += duration
        wrapper.total_calls += 1
        if wrapper.total_calls % 1 == 0 and report_timer:
            print(f'Spent {(wrapper.total_time):.4f}s in function {(wrapper.total_calls)} calls to {func.__name__!r}') 
        return result
    wrapper.total_calls = 0
    wrapper.total_time = 0
    return wrapper

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

    def __init__(self, mol, mf, mc=None, tol=None, twist=None, determinants=None, eval_gto_precision=None, det_emax = None):
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
            ncore = (0,0)
            self._nelec = mol.nelec
        self.eval_gto_precision = eval_gto_precision
        try:
            self.num_det = mc.ci.shape[0] * mc.ci.shape[1]
        except:
            self.num_det = 1

        self.myparameters = {}
        (
            self.myparameters["det_coeff"],
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = pyqmc.orbitals.choose_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants, tol=self.tol
        )

        self.det_info_file = 'det_info.hdf5'
        self.hmf_file      = 'hmf.hdf5'

        
        if self.num_det > 1:
            self.filter_determinants(det_emax, mf.mo_energy, ncore)
            self.get_hmf(mf.mo_energy, ncore)
        else:
            print('Using only one determinant')

        # Use constant weight 
        # self.myparameters["det_coeff"] = np.ones(self.num_det)/self.num_det
        self.myparameters["det_coeff"] = np.ones(self.num_det)
        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        iscomplex = self.orbitals.mo_dtype == complex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if iscomplex else float

        self.get_phase = get_complex_phase if iscomplex else gpu.cp.sign
        
    def get_hmf(self, mo_energies, ncore):
        mask_up = np.array(self._det_occup[0]) + ncore[0]
        mask_dn = np.array(self._det_occup[1]) + ncore[1]

        if len(mo_energies.shape) == 1:
            if np.sum(mask_up) == 0:
                raise ValueError("No occupied orbitals for up spin in RHF")
            else:
                up_energies = np.sum(mo_energies[mask_up], axis=1)
                dn_energies = np.sum(mo_energies[mask_dn], axis=1)
        else:
            if np.sum(mask_up) == 0:
                up_energies = np.zeros(self._det_map[0].shape)
            else:
                up_energies = np.sum(mo_energies[0][mask_up], axis=1)
            if np.sum(mask_dn) == 0:
                dn_energies = np.zeros(self._det_map[1].shape)
            else:
                dn_energies = np.sum(mo_energies[1][mask_dn], axis=1)
        total_energies = up_energies[self._det_map[0]] + dn_energies[self._det_map[1]]
        hf = h5py.File(self.hmf_file, 'w')
        hf.create_dataset('hmf', data=total_energies)
        self.hmf = np.diag(total_energies)
        hf.close()
    
    def filter_determinants(self, emax, mo_energies, ncore, print_info = True):
        
        determinants_filtered = False
        print("Filtering determinants, energy units are in Hartree")
        if isinstance(emax, float):
            assert emax > 0, "Emax must be positive for energy based determinant filtering"
            determinants_filtered = True
            up_energies = np.sum(mo_energies[0][self._det_occup[0]+ncore[0]], axis=1)
            dn_energies = np.sum(mo_energies[1][self._det_occup[1]+ncore[1]], axis=1)
            total_energies = up_energies[self._det_map[0]] + dn_energies[self._det_map[1]]
            min_energy = np.min(total_energies)
            if print_info:
                info_string = "Eigenvalues: " + ' '.join([str(np.round(x - min_energy, 3)) for x in np.sort(total_energies)])
                print(info_string)
            emax = emax + min_energy
            print("Determinants being filtered with emax + min eigenvalue", emax)
            mask = total_energies < emax
            temp_det_map = self._det_map[np.row_stack((mask, mask))]
            num_init_dets = len(self._det_map[0])
            unused_temp_det_map = self._det_map[np.row_stack((~mask, ~mask))]
            det_map_shape = np.array(temp_det_map.shape)
            num_used_dets = int(det_map_shape[0]/2)

            det_map = temp_det_map.reshape(2, num_used_dets)
            unused_det_map = unused_temp_det_map.reshape(2, num_init_dets-num_used_dets)
            print('Min eigenvalue', np.round(np.min(total_energies), 3))
            print('Max eigenvalue', np.round(np.max(total_energies), 3))
        elif isinstance(emax, int):
            assert emax > 0 and emax <= 100, "Emax must be between 0 and 100 for percentage based determinant filtering"
            determinants_filtered = True
            percentile = emax
            print("Determinants being filtered with percentage ", percentile)
            up_energies = np.sum(mo_energies[0][self._det_occup[0]+ncore[0]], axis=1)
            dn_energies = np.sum(mo_energies[1][self._det_occup[1]+ncore[1]], axis=1)
            total_energies = up_energies[self._det_map[0]] + dn_energies[self._det_map[1]]
            emax = np.percentile(total_energies, percentile)
            if print_info:
                energy_range = np.max(total_energies) - np.min(total_energies)
                info_string = "Eigenvalues percentiles: " + ' '.join([str(np.round((x - np.min(total_energies))/energy_range, 3)) for x in np.sort(total_energies)])
                print(info_string)

            mask = total_energies < emax
            temp_det_map = self._det_map[np.row_stack((mask, mask))]
            num_init_dets = len(self._det_map[0])
            unused_temp_det_map = self._det_map[np.row_stack((~mask, ~mask))]
            det_map_shape = np.array(temp_det_map.shape)
            num_used_dets = int(det_map_shape[0]/2)

            det_map = temp_det_map.reshape(2, num_used_dets)
            unused_det_map = unused_temp_det_map.reshape(2, num_init_dets-num_used_dets)
            print('Min eigenvalue', np.round(np.min(total_energies), 3))
            print('Max eigenvalue', np.round(np.max(total_energies), 3))
        elif emax == 'singles' or emax == 'doubles':
            determinants_filtered = True
            up_ground = self._det_occup[0][0]
            dn_ground = self._det_occup[1][0]
            up_num_exc = np.array([np.setdiff1d(x, up_ground).shape[0] for x in self._det_occup[0]])
            dn_num_exc = np.array([np.setdiff1d(x, dn_ground).shape[0] for x in self._det_occup[1]])
            tot_exc = up_num_exc[self._det_map[0]] + dn_num_exc[self._det_map[1]]
            if emax == 'singles':
                mask = tot_exc < 2
            elif emax == 'doubles':
                mask = tot_exc < 3
            tot_used_exc = tot_exc[mask]
            det_map = self._det_map[np.row_stack((mask, mask))].reshape(2, -1)
            unused_det_map = self._det_map[np.row_stack((~mask, ~mask))].reshape(2, -1)
            num_init_dets = len(self._det_map[0])
            det_map_shape = np.array(det_map.shape)
            num_used_dets = int(det_map_shape[1])
            print('Det excitations', tot_used_exc)
            self._tot_used_exc = tot_used_exc
        else:
            num_used_dets = len(self._det_map[0])
            print('Used # of determinants', num_used_dets)

        if determinants_filtered:
            self._det_map_orig = copy.deepcopy(self._det_map)
            self._det_map_mask = mask
            self._det_map = det_map
            self.num_det = num_used_dets
            print('Initial # of determinants', num_init_dets)
            print('Filtered # of determinants', num_init_dets-num_used_dets)
            print('Used # of determinants', num_used_dets)
            if num_used_dets == 0:
                raise ValueError("No determinants left after filtering")
            hf = h5py.File(self.det_info_file, 'w')
            hf.create_dataset('det_map_orig', data=self._det_map_orig)
            hf.create_dataset('det_map_mask', data=self._det_map_mask)
            hf.create_dataset('det_map',      data=self._det_map)
            hf.close()
        
    @timer_func
    def recompute(self, configs):
        r"""This computes the value from scratch. Returns the logarithm of the wave function as
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

    @timer_func
    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        r"""Update any internals given that electron e moved to epos. mask is a Boolean array
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
    
    @timer_func
    def value(self):
        r"""Returns the logarithmic value of the bosonic wavefunction: log(\Phi_B)

        Returns:
            sign, logval: sign and logatithmic value of the bosonic wavefunction
        """
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
    
    @timer_func
    def value_dets(self, test = False):
        r"""Returns logarithmic values of all Slater determinants used to form bosonic wavefunction

        Args:
            test (bool, optional): Calculates the value of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            sign, logval: sign and logatithmic value of each wavefunction
        """
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]

        wf_logval = (updets[1] + dndets[1])
        wf_sign = updets[0] * dndets[0]

        if test:
            det_coeff = self.myparameters['det_coeff']
            tol = 1E-12
            phi_b = 1./2 * np.log(np.einsum('d, id->i', det_coeff,np.exp(2*wf_logval) ))
            try:
                assert ((np.abs(phi_b - self.value()[1]) < tol).all())
            except:
                print('value_dets error', np.max(np.abs(phi_b - self.value()[1])))
                      
        return wf_sign, wf_logval
    
    @timer_func
    def gradient(self, e, epos):
        r"""Compute the gradient of the log wave function ∇log(Psi_B) 
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
        # import pdb
        # pdb.set_trace()
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
    
    def gradient_laplacian(self, e, epos):
        grad = self.gradient(e, epos)
        lap = self.laplacian(e, epos)
        return grad, lap
    
    @ timer_func
    def laplacian(self, e, epos):
        r"""Returns ∇²(Phi_B)/Phi_B of bosonic wave function for electron e at position epos
        Returns array of shape (nconfigs,)
        \[
        \nabla^2 \Phi_B = \frac{\sum_l \left( \nabla \Phi_l \cdot \nabla \Phi_l + \Phi_l \nabla^2 \Phi_l \right)}{\Phi_B} 
        - \frac{\left( \sum_l \Phi_l \nabla \Phi_l \right)^2}{\Phi_B^3}.
        \]
        # The Laplacian of the bosonic wave function (Phi_B) divided by Phi_B is:
        #
        # 1. First term: Sum over determinants l of:
        #    - (gradient of Phi_l)·(gradient of Phi_l)  [dot product of gradients]
        #    - plus (Phi_l)·(Laplacian of Phi_l)
        #    All divided by Phi_B squared
        #
        # 2. Second term: Subtract
        #    - The square of (sum of Phi_l times gradient of Phi_l)
        #    - Divided by Phi_B 4th power
        #
        # This implements the quotient rule for second derivatives of the bosonic wave function
        """

        # import pdb; pdb.set_trace()
        lap_n = self.laplacian_dets(e, epos) # ∇²(Phi_n)
        # Get value of determinants
        phase_n, logval_n = self.value_dets()   # phase(Phi_n), log(Phi_n)
        val_n = phase_n * np.nan_to_num(np.exp(logval_n)) # Phi_n
        # Get gradient of determinants
        loggrad_n = self.gradient_dets(e, epos) # ∇log(Phi_n) # large

        # Get value of bosonic wavefunction
        phase_b, logval_b = self.value() # phase(Phi_B), log(Phi_B)
        val_b = phase_b * np.nan_to_num(np.exp(logval_b)) # Phi_B

        # Calculate ∇²(Phi_B)/Phi_B
        # First term: Sum over determinants l of: (gradient of Phi_l)·(gradient of Phi_l)  [dot product of gradients]
        
        grad_phi_l = np.einsum('nxc, cn->nxc', loggrad_n, val_n)
        
        lap_b1 = np.einsum('nxc, nxc->c', grad_phi_l, grad_phi_l)
        lap_b1 += np.einsum('cn, cn->c', val_n, lap_n)
        lap_b1 /= val_b**2
        # Second term: Minus the square of (sum of Phi_l times gradient of Phi_l)
        lap_b2 = np.einsum('cn, nxc->cx', val_n, grad_phi_l)
        lap_b2 = np.einsum('cx, cx->c', lap_b2, lap_b2)
        lap_b2 /= val_b**4
        lap_b = lap_b1 - lap_b2
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(val_b,lap_b)
        # plt.show()
        # import pdb; pdb.set_trace()
        return lap_b
    
    
    def gradient_value(self, e, epos):
        r"""Returns the ∇log(Phi_B) gradient of bosonic wavefunction and its log value log(Phi_B)
        Phi_B is defined in eq. 4, Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
        Returns array of shape (nconfigs, 3) and (nconfigs,)"""

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
    
    @timer_func
    def gradient_dets(self, e, epos, test=False):
        r"""Returns the ∇log(Phi_l) gradient of each slater determinant forming the bosonic wavefunction
        Phi_l is defined in eq. 14, psi_l = Phi_l/Phi_B

        Args:
            e (_type_): electron index
            epos (_type_): electron coordinates
            test (bool, optional): Calculates the gradient of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            gradient: [# of determinants, cartesian(3), nconfigs]
        """

        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = np.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        # Removed detcoeff and ref values
        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * np.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
            )
        )

        numer = np.einsum(
            "ei...d,di->edi...",
            ratios[..., self._det_map[s]],
            det_array,
        )
        
        # denom has the sum of Multideterminant WF, not needed
        grads = numer[1:] / numer[0]
        grads = np.einsum('edi->dei', grads)

        if test:
            tol = 1E-6
            dv = self.value_dets()[1]
            v = self.value()[1]
            det_coeff = self.myparameters['det_coeff']
            gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), grads)
            try:
                assert ((np.abs(gc - self.gradient(e, epos)) < tol).all())
                print('gradient_dets test passed')
            except:
                print('gradient_dets error', np.max(np.abs(gc - self.gradient(e, epos))))
            exit()
        return grads
    
    def laplacian_dets(self, e, epos, test=False):
        r"""Returns laplacian ∇²(Phi_l) of each slater determinant forming the bosonic wavefunction
        Phi_l is defined in eq. 14, psi_l = Phi_l/Phi_B

        Args:
            e (_type_): electron index
            epos (_type_): electron coordinates
            test (bool, optional): Calculates the laplacian of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            laplacian: [# of determinants, nconfigs]
        """
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao_val = ao[:, 0, :, :]
        ao_lap = gpu.cp.sum(ao[:, [4, 7, 9], :, :], axis=1)
        mo_lap_vals = gpu.cp.stack(
            [self.orbitals.mos(x, s)[..., self._det_occup[s]] for x in [ao_val, ao_lap]]
        )

        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mo_lap_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                # - upref
                # - dnref
            )
        )
        
        # det_coeff = self.myparameters['det_coeff']
        numer = gpu.cp.einsum(
            "ei...d,di->ei...d",
            jacobi[..., self._det_map[s]],
            # det_coeff,
            det_array,
        )
        # denom = np.sum(numer[0], axis=1)
        # lap = np.einsum('id, i->id', numer[1], 1./denom)
        
        lap = numer[1]

        # np.sum(numer[0], axis=1) should be the same as denom in laplacian @ slater.py
        # If want to return ∇²(Psi_n), return numer[1]
        # np.einsum('ie, i->ie',numer[1], 1./np.sum(numer[0], axis=1)) returns ∇²(Psi_n)/\sum(Psi_n)
        # For testing against slater laplacian, return np.einsum('ie, i->ie',numer[1], 1./np.sum(numer[0], axis=1))
        # We don't need to evaluate numer[0], if we want ∇²(Psi_n) 
        return lap
        

    def pgradient(self):
        # Not implemented
        d = {}
        return d