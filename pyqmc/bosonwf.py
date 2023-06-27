import numpy as np
import pyqmc.gpu as gpu
import determinant_tools as determinant_tools
import pyqmc.orbitals
import warnings


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


class BosonWF:
    r"""
    """

    def __init__(self, mol, mf, tol=None):
        """
        """
        from wftools import generate_slater
        self.tol = -1 if tol is None else tol
        self._mol = mol
        self._nelec = mol.nelec
        self.nmax = 1 
        wf1, to_opt1 = generate_slater(mol, mf)
        self.slater_dets = [wf1]

    def recompute(self, configs):
        """This computes the value from scratch"""
        nconf, nelec, ndim = configs.configs.shape
        self._dets = []
        self._detsq = []
        self._inverse = []
        for det_ind, slater_det in enumerate(self.slater_dets):
            aos = slater_det.orbitals.aos("GTOval_sph", configs)
            slater_det._aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
            self._dets.append([])
            self._detsq.append([])
            self._inverse.append([])
            for s in [0, 1]:                
                begin = slater_det._nelec[0] * s
                end = slater_det._nelec[0] + slater_det._nelec[1] * s
                mo = slater_det.orbitals.mos(slater_det._aovals[:, :, begin:end, :], s)
                mo_vals = gpu.cp.swapaxes(mo[:, :, slater_det._det_occup[s]], 1, 2)
                det = np.linalg.det(mo_vals)
                self._dets[det_ind].append(det)
                conj = np.conjugate(det)
                self._detsq[det_ind].append(np.multiply(det, conj)) # Assuming real
                compute = np.isfinite(self._dets[det_ind][s])
                self._inverse[det_ind].append(gpu.cp.zeros(mo_vals.shape, dtype=mo_vals.dtype))
                for d in range(compute.shape[1]):
                    self._inverse[det_ind][s][compute[:, d], d, :, :] = gpu.cp.linalg.inv(
                        mo_vals[compute[:, d], d, :, :]
                    )
        self._configs = configs
        self._dets = gpu.cp.array(self._dets)
        self._detsq = gpu.cp.array(self._detsq)
        self._inverse = gpu.cp.array(self._inverse)
        
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
        """
        """
        wf_val = 0
        # import pdb
        # pdb.set_trace()
        for det_ind, slater_det in enumerate(self.slater_dets):

            updets = self._dets[det_ind][0][:, [0]]
            dndets = self._dets[det_ind][1][:, [0]]
            cmm = gpu.cp.einsum("ci,ci->ci", updets, dndets)
            cmm2 = gpu.cp.einsum("ci,ci->ci", cmm, cmm)
            wf_val += np.nan_to_num(cmm2)
        wf_val_sqrt = np.sqrt(wf_val)
        return wf_val_sqrt

    def _testrow(self, e, vec, mask=None, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin
        if mask is None:
            mask = [True] * vec.shape[0]

        ratios = gpu.cp.einsum(
            "i...dj,idj...->i...d",
            vec,
            self._inverse[s][mask][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, mask][:, self._det_map[0]]
            * self._dets[1][0, mask][:, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, mask][:, self._det_map[0]]
                + self._dets[1][1, mask][:, self._det_map[1]]
                - upref
                - dnref
            )
        ).T
        numer = gpu.cp.einsum(
            "i...d,d,di->i...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )
        denom = gpu.cp.einsum(
            "d,di->i...",
            self.parameters["det_coeff"],
            det_array,
        )

        if len(numer.shape) == 2:
            denom = denom[:, gpu.cp.newaxis]
        return numer / denom

    def _testrowderiv(self, e, vec, spin=None):
        """vec is a nconfig,nmo vector which replaces row e"""
        s = int(e >= self._nelec[0]) if spin is None else spin

        ratios = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            vec,
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
                - upref
                - dnref
            )
        )
        numer = gpu.cp.einsum(
            "ei...d,d,di->ei...",
            ratios[..., self._det_map[s]],
            self.parameters["det_coeff"],
            det_array,
        )
        denom = gpu.cp.einsum(
            "d,di->i...",
            self.parameters["det_coeff"],
            det_array,
        )
        # curr_val = self.value()

        if len(numer.shape) == 3:
            denom = denom[gpu.cp.newaxis, :, gpu.cp.newaxis]
        return numer / denom

    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i
        of spin s in determinant det"""

        return gpu.cp.einsum(
            "ij...,ij->i...", vec, self._inverse[s][:, det, i, :], optimize="greedy"
        )

    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)

        mograd_vals = mograd[:, :, self._det_occup[s]]

        ratios = self._testrowderiv(e, mograd_vals)
        return gpu.asnumpy(ratios[1:] / ratios[0])

    def gradient_value(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        self._grad_nom = []
        import pdb
        ratio = 0
        for det_ind, slater_det in enumerate(self.slater_dets):
            d_ao = slater_det.orbitals.aos("GTOval_sph_deriv1", epos)
            d_mo = slater_det.orbitals.mos(d_ao, s)
            d_mo_vals = d_mo[:, :, slater_det._det_occup[s]]    
            pdb.set_trace()
            ratios = gpu.cp.einsum(
                "eidj,idj->eid",
                d_mo_vals,
                self._inverse[det_ind, s, ..., e - s * self._nelec[0]],
            )
            det_up_i = self._dets[det_ind][0]
            det_dn_i = self._dets[det_ind][1]
            ratio += gpu.cp.einsum("id,id,id,id,eid->ei", det_dn_i, det_dn_i, det_up_i, det_up_i, ratios)
        
        val = self.value()
        derivatives = ratio[1:]/val[:,0] #ratio[0]
        
        derivatives[~np.isfinite(derivatives)] = 0.0
        values = ratios[0]
        values[~np.isfinite(values)] = 1.0
        import matplotlib.pyplot as plt
        z = epos.configs[:, 2]
        plt.plot(z,2*derivatives[2], '-ob', label='der_pyqmc')
        
        plt.plot(z,np.gradient(val[:,0], z), '-sk', label='der_num')
        plt.plot(z,val*10, '-or', label='value*10')
        plt.legend()
        plt.xlabel('Z coordinate')
        plt.show()
        
        pdb.set_trace()
        return derivatives, values, val


    def laplacian(self, e, epos):
        """Compute the laplacian Psi/ Psi."""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao_val = ao[:, 0, :, :]
        ao_lap = gpu.cp.sum(ao[:, [4, 7, 9], :, :], axis=1)
        mos = gpu.cp.stack(
            [self.orbitals.mos(x, s)[..., self._det_occup[s]] for x in [ao_val, ao_lap]]
        )
        ratios = self._testrowderiv(e, mos)
        return gpu.asnumpy(ratios[1] / ratios[0])

    def gradient_laplacian(self, e, epos):
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        ao = gpu.cp.concatenate(
            [ao[:, 0:4, ...], ao[:, [4, 7, 9], ...].sum(axis=1, keepdims=True)], axis=1
        )
        mo = self.orbitals.mos(ao, s)
        mo_vals = mo[:, :, self._det_occup[s]]
        ratios = self._testrowderiv(e, mo_vals)
        ratios = gpu.asnumpy(ratios / ratios[:1])
        return ratios[1:-1], ratios[-1]

    def testvalue(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos"""
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        mo = self.orbitals.mos(ao, s)
        mo_vals = mo[..., self._det_occup[s]]
        if len(epos.configs.shape) > 2:
            mo_vals = mo_vals.reshape(
                -1, epos.configs.shape[1], mo_vals.shape[1], mo_vals.shape[2]
            )
        return gpu.asnumpy(self._testrow(e, mo_vals, mask)), (ao, mo)

    def testvalue_many(self, e, epos, mask=None):
        """return the ratio between the current wave function and the wave function if
        electron e's position is replaced by epos for each electron"""
        s = (e >= self._nelec[0]).astype(int)
        ao = self.orbitals.aos("GTOval_sph", epos, mask)
        ratios = gpu.cp.zeros((epos.configs.shape[0], e.shape[0]), dtype=self.dtype)
        for spin in [0, 1]:
            ind = s == spin
            mo = self.orbitals.mos(ao, spin)
            mo_vals = mo[..., self._det_occup[spin]]
            ratios[:, ind] = self._testrow(e[ind], mo_vals, mask, spin=spin)

        return gpu.asnumpy(ratios)

    def pgradient(self):
        """Compute the parameter gradient of Psi.
        Returns :math:`\partial_p \Psi/\Psi` as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.

        The wave function is given by ci Di, with an implicit sum

        We have two sets of parameters:

        Determinant coefficients:
        di psi/psi = Dui Ddi/psi

        Orbital coefficients:
        dj psi/psi = ci dj (Dui Ddi)/psi

        Let's suppose that j corresponds to an up orbital coefficient. Then
        dj (Dui Ddi) = (dj Dui)/Dui Dui Ddi/psi = (dj Dui)/Dui di psi/psi
        where di psi/psi is the derivative defined above.
        """
        d = {}

        # Det coeff
        curr_val = self.value()
        nonzero = curr_val[0] != 0.0

        # dets[spin][ (phase,log), configuration, determinant]
        dets = (
            self._dets[0][:, :, self._det_map[0]],
            self._dets[1][:, :, self._det_map[1]],
        )

        d["det_coeff"] = gpu.cp.zeros(dets[0].shape[1:], dtype=dets[0].dtype)
        d["det_coeff"][nonzero, :] = (
            dets[0][0, nonzero, :]
            * dets[1][0, nonzero, :]
            * gpu.cp.exp(
                dets[0][1, nonzero, :]
                + dets[1][1, nonzero, :]
                - gpu.cp.array(curr_val[1][nonzero, np.newaxis])
            )
            / gpu.cp.array(curr_val[0][nonzero, np.newaxis])
        )

        for s, parm in zip([0, 1], ["mo_coeff_alpha", "mo_coeff_beta"]):
            ao = self._aovals[
                :, :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]

            split, aos = self.orbitals.pgradient(ao, s)
            mos = gpu.cp.split(gpu.cp.arange(split[-1]), gpu.asnumpy(split).astype(int))
            # Compute dj Diu/Diu
            nao = aos[0].shape[-1]
            nconf = aos[0].shape[0]
            nmo = int(split[-1])
            deriv = gpu.cp.zeros(
                (len(self._det_occup[s]), nconf, nao, nmo), dtype=curr_val[0].dtype
            )
            for det, occ in enumerate(self._det_occup[s]):
                for ao, mo in zip(aos, mos):
                    for i in mo:
                        if i in occ:
                            col = occ.index(i)
                            deriv[det, :, :, i] = self._testcol(det, col, s, ao)

            # now we reduce over determinants
            d[parm] = gpu.cp.zeros(deriv.shape[1:], dtype=curr_val[0].dtype)
            for di, coeff in enumerate(self.parameters["det_coeff"]):
                whichdet = self._det_map[s][di]
                d[parm] += (
                    deriv[whichdet]
                    * coeff
                    * d["det_coeff"][:, di, np.newaxis, np.newaxis]
                )

        for k, v in d.items():
            d[k] = gpu.asnumpy(v)

        for k in list(d.keys()):
            if np.prod(d[k].shape) == 0:
                del d[k]
        return d
