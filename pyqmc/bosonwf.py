import numpy as np
import pyqmc.gpu as gpu
import determinant_tools as determinant_tools
import orbitals
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

    def __init__(self, mol, mf, mc=None, tol=None, twist=None, determinants=None):
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
        self.myparameters = {}
        (
            det_coeff,
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = orbitals.choose_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=determinants, tol=self.tol
        )
        num_det = len(det_coeff)
        # Use constant weight 
        self.myparameters["det_coeff"] = np.ones(num_det)/num_det

        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        iscomplex = self.orbitals.mo_dtype == complex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        self.dtype = complex if iscomplex else float

    def recompute(self,configs,det_zero_tol=1e-16):
        """Compute the value of wavefunction from scratch

        Args:
            configs (_type_): QMC configurations
            det_zero_tol (_type_, optional): zero tolerance for determinant. Defaults to 1e-16.

        Returns:
            _type_: _description_
        """
        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos("GTOval_sph", configs)
        self._aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        self._detsq = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            det = np.linalg.det(mo_vals)
            detsq = gpu.cp.asarray(det * np.conjugate(det), dtype=float)
            self._detsq.append(detsq)
            is_zero = np.sum(np.abs(self._detsq[s]) < det_zero_tol)
            compute = np.isfinite(self._detsq[s])
            zero_pct = is_zero/np.prod(compute.shape)
            if is_zero > 0 and zero_pct > 0.01:
                warnings.warn(
                    f"A wave function is zero. Found this proportion: {is_zero/nconf}"
                )
                print(f"zero {is_zero/np.prod(compute.shape)}")
            self._inverse.append(gpu.cp.zeros(mo_vals.shape, dtype=mo_vals.dtype))
            for d in range(compute.shape[1]):
                self._inverse[s][compute[:, d], d, :, :] = gpu.cp.linalg.inv(
                    mo_vals[compute[:, d], d, :, :]
                )
        self._configs = configs
        self._detsq = gpu.cp.array(self._detsq)
        # self._inverse = gpu.cp.array(self._inverse)
        return self.value()

    def updateinternals(self, e:int, epos, configs, mask: list=None, saved_values=None):
        """Update any internals given that electron e moved to epos. 
        Mask is a Boolean array which allows us to update only certain walkers.

        Args:
            e (int): electron index
            epos (_type_): "new" position of all electrons, array?
            configs (_type_): all electron configurations
            mask (list, optional): mask on . Defaults to None.
            saved_values (_type_, optional): _description_. Defaults to None.
        """

        s = int(e >= self._nelec[0]) # spin index
        if mask is None:
            mask = np.ones(epos.configs.shape[0], dtype=bool)
        is_zero = np.sum(np.isinf(self._detsq[s][1]))
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
        self._detsq[s][mask, :] *= gpu.cp.abs(det_ratio)**2
        # TODO(kayahans): Check this again 

    def value(self):
        """Returns the value of the bosonic wavefunction

        Returns:
            _type_: _description_
        """
        det_coeff = self.parameters["det_coeff"]
        updetsq = self._detsq[0][:, self._det_map[0]]
        dndetsq = self._detsq[1][:, self._det_map[1]]
        valsq = gpu.cp.einsum("id,id,d->i", updetsq, dndetsq, det_coeff)
        val = np.sqrt(valsq)
        sign = np.ones(val.shape[0]) # bosonic wavefunction always have sign +1 
        return sign, val
    
    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        grad, _, _ = self.gradient_value(e, epos)
        #TODO: gradient is shortcut, but check if using the original version has any computational advantage
        return grad

    def gradient_value(self, e, epos, configs=None):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        

        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        detsq_up = self._detsq[0][:, self._det_map[0]]
        detsq_dn = self._detsq[1][:, self._det_map[1]]
        
        det_coeff = self.parameters["det_coeff"]
        numer = gpu.cp.einsum("d,id,id,eid->ei",det_coeff, detsq_up, detsq_dn, jacobi[..., self._det_map[s]])
        if configs is not None:
            cf = configs.copy()
            cf.configs[:,e,:] = epos.configs
            sign, values = self.recompute(cf)
        else:
            sign, values = self.value()
        
        grad = numer[1:]/values
        # print("Grad max ", np.max(grad))
        grad[~np.isfinite(grad)] = 0.0
        values[~np.isfinite(values)] = 1.0
        return grad, values, (aograd[:, 0], mograd[0])
