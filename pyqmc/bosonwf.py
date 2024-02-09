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
        print('Number of determinants in the bosonic wavefunction=', num_det)
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
        # Logdets are not used outside this function unlike in the slater.py
        slogdets = []
        self._detsq = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            slogdet = gpu.cp.asarray(np.linalg.slogdet(mo_vals)) # Spin, (sign, val), nconf, [ndet_up, ndet_dn]
            
            det = gpu.cp.asarray(np.linalg.det(mo_vals))
            detsq = gpu.cp.asarray(det * np.conjugate(det), dtype=float)            
            
            slogdets.append(slogdet)
            self._detsq.append(detsq)
            
            is_zero = np.sum(np.abs(slogdets[s][0]) < det_zero_tol)
            compute = np.isfinite(slogdets[s][1])
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
            # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    def value_updated(self, configs, det_zero_tol=1e-16):
        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos("GTOval_sph", configs)
        aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        detsq = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            det = np.linalg.det(mo_vals)
            detsq_s = gpu.cp.asarray(det * np.conjugate(det), dtype=float)
            detsq.append(detsq_s)
        det_coeff = self.parameters["det_coeff"]
        updetsq = detsq[0][:, self._det_map[0]]
        dndetsq = detsq[1][:, self._det_map[1]]
        valsq = gpu.cp.einsum("id,id,d->i", updetsq, dndetsq, det_coeff)
        val = np.sqrt(valsq)
        sign = np.ones(val.shape[0]) # bosonic wavefunction always have sign +1 
        return sign, val   

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
        """Returns the sign and value of the bosonic wavefunction

        Returns:
            _type_: _description_
        """
        det_coeff = self.parameters["det_coeff"] # C_d
        updetsq = self._detsq[0][:, self._det_map[0]] # |{D_{di}^{up}}|^2
        dndetsq = self._detsq[1][:, self._det_map[1]] # |{D_{di}^{dn}}|^2
        valsqd = gpu.cp.einsum("id,id->id", updetsq, dndetsq) # |{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2
        valsq = gpu.cp.einsum("d, id->i", det_coeff, valsqd)  # \sum_d{C_d*|{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2}
        
        val = np.sqrt(valsq) # \psi_{i}=\sqrt{\sum_d{C_d*|{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2}}
        # vald = np.sqrt(valsqd) # \psi_{id}=\sqrt{|{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2}
        sign = np.ones(val.shape[0]) # bosonic wavefunction always have sign +1 

        return sign, val
            
    def gradient(self, e, epos):
        """Compute the gradient of the log wave function
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        grad, _ = self.gradient_value(e, epos)
        #TODO: gradient is shortcut, but check if using the original version has any computational advantage
        return grad

    def gradient_value(self, e, epos, configs=None):
        """Compute the gradient of the bosonic wave function
        This is typically called in the block of VMC and DMC, where the inverse is not updated
        """
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos("GTOval_sph_deriv1", epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        
        # Derivative of the determinant is calculated using Jacobi's formula
        # https://en.wikipedia.org/wiki/Jacobi%27s_formula
        # https://courses.physics.illinois.edu/phys466/fa2016/projects/1999/team4/webpage/local_energy/node4.html
        
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        det_coeff = self.parameters["det_coeff"]
        
        updetsq = self._detsq[0][:, self._det_map[0]] # |{D_{di}^{up}}|^2
        dndetsq = self._detsq[1][:, self._det_map[1]] # |{D_{di}^{dn}}|^2
        valsqd = gpu.cp.einsum("id,id->id", updetsq, dndetsq) # |{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2
        vald = np.sqrt(valsqd) # \psi_{id}=\sqrt{|{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2}

        # Derivative of \Phi
        # \Phi' = \sqrt{\sum_i{\psi_i}^2}' = \sum{\psi' * \psi} / \Phi 
        # \psi = det(D_di), where D_di is the Slater determinant matrix
        # \psi' is calculated using the jacobi theorem described above
        # \psi x derivative, \psi_x = det(A)*tr[A^-1 * \del_x(A)] (x is a Cartesian axis)
        numer = gpu.cp.einsum("d,id,id,gid->gi",det_coeff, vald, vald, jacobi[..., self._det_map[s]])
        
        # values = \phi(R)
        if configs is not None:
            # calculate \phi with R'
            cf = configs.copy()
            cf.configs[:,e,:] = epos.configs
            sign, psi = self.value_updated(cf)        
        else:
            # calculate \phi with R
            sign, psi = self.value()        
        grad = gpu.cp.einsum("gi,i->gi", numer[1:], 1./psi)
        grad[~np.isfinite(grad)] = 0.0
        psi[~np.isfinite(psi)] = 1.0

        saved_values = {'values':(aograd[:, 0], mograd[0]),
                        'sign':sign,
                        'psi':psi}
        
        return grad, saved_values

    def gradient_laplacian(self, e, epos, configs = None):
        print('Not implemented')
        exit()            
        # s = int(e >= self._nelec[0])
        # ao = self.orbitals.aos("GTOval_sph_deriv2", epos)
        
        # aograd = ao[:, 1:4, :, :] #dx, dy, dz
        # aolap = ao[:, [9], :, :] #dxx, dyy, dzz
        # import pdb
        # pdb.set_trace()
        # molap = self.orbitals.mos(aolap, s)
        # molap_vals = molap[:, :, self._det_occup[s]]

        # mograd = self.orbitals.mos(aograd, s)
        # mograd_vals = mograd[:, :, self._det_occup[s]]

        # import pdb
        # pdb.set_trace()

        # jacobidel = gpu.cp.einsum(
        #     "ei...dj,idj...->ei...d",
        #     mograd_vals,
        #     self._inverse[s][..., e - s * self._nelec[0]],
        # )

        # jacobilap = gpu.cp.einsum(
        #     "ei...dj,idj...->ei...d",
        #     molap_vals,
        #     self._inverse[s][..., e - s * self._nelec[0]],
        # )

        # jacobidelsq = gpu.cp.einsum(
        #     "ei...dj,idj...,ei...dj,idj...->ei...d",
        #     mograd_vals,
        #     self._inverse[s][..., e - s * self._nelec[0]],
        #     mograd_vals,
        #     self._inverse[s][..., e - s * self._nelec[0]],
        # )

        # # For a single determinant, det(A)
        # # del_\alpha\beta (det(A)) = det(A) * {Tr[A^-1 del_\alpha A] * Tr[A^-1 del_\beta A] 
        # #                                      + Tr[A^-1 del_\alpha\beta A]
        # #                                      - Tr[A^-1 del_\alpha A A^-1 del_\beta A]}
        # # For \alpha = \beta
        # # del_\alpha\alpha (det(A)) = det(A) * {Tr[A^-1 del_\alpha A]^2
        # #                                      + Tr[A^-1 del_\alpha\alpha A]
        # #                                      - Tr[(A^-1 del_\alpha A)^2]}

        # # del_\alpha\alpha (det(A)) = det(A) * {tra+trb-trc}
        # # lap (det(A)) = det(A) * \sum_xyz{tra+trb-trc}

        # det_coeff = self.parameters["det_coeff"]
        
        # updetsq = self._detsq[0][:, self._det_map[0]] # |{D_{di}^{up}}|^2
        # dndetsq = self._detsq[1][:, self._det_map[1]] # |{D_{di}^{dn}}|^2
        # valsqd = gpu.cp.einsum("id,id->id", updetsq, dndetsq) # |{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2
        # vald = np.sqrt(valsqd) # \psi_{id}=\sqrt{|{D_{di}^{up}}|^2*|{D_{di}^{dn}}|^2}        

        # # values = \phi(R)
        # if configs is not None:
        #     # calculate \phi with R'
        #     cf = configs.copy()
        #     cf.configs[:,e,:] = epos.configs
        #     sign, psi = self.value_updated(cf)        
        # else:
        #     # calculate \phi with R
        #     sign, psi = self.value()   

        # tra = gpu.cp.einsum('eid,eid->eid',jacobidel[..., self._det_map[s]], jacobidel[..., self._det_map[s]])
        # trb = jacobilap[..., self._det_map[s]]
        # trc = jacobidelsq[..., self._det_map[s]]
        # lapd = gpu.cp.einsum('id, eid->id', vald, tra+trb) #-trc)

        # gradgd = gpu.cp.einsum("id,gid->gid",vald, jacobidel[..., self._det_map[s]])
        # grad_numer = gpu.cp.einsum("d,id,gid->gi",det_coeff, vald, gradgd)
        
        # grad = gpu.cp.einsum("gi,i->gi", grad_numer, 1./psi)
        
        # lap_numer_a1 = gpu.cp.einsum("id,id->id",lapd, vald)
        # lap_numer_a2 = gpu.cp.einsum("eid,eid->id",gradgd, gradgd)
        # lap_numer_a = gpu.cp.einsum("id,i,i->id", (lap_numer_a1 + lap_numer_a2),psi,psi)

        # #TODO here delPhiB is sum over all determinants
        # lap_numer_b = gpu.cp.einsum("d,id,id,id->id",det_coeff,lap_numer_a2,vald,vald)
        # lap_numer = lap_numer_a - lap_numer_b
        # invpsi=1./psi
        # # lap = gpu.cp.einsum("d,id,i,i,i->i", det_coeff, lap_numer,invpsi,invpsi,invpsi)
        # lap = gpu.cp.einsum("id->i", lapd)
        # saved_values = {'grad':grad,
        #                 'sign':sign,
        #                 'psi':psi,
        #                 'vald':vald,
        #                 'tra':tra,
        #                 'trb':trb,
        #                 'trc':trc,}
        
        return lap, saved_values


    def _testcol(self, det, i, s, vec):
        """vec is a nconfig,nmo vector which replaces column i
        of spin s in determinant det"""

        return gpu.cp.einsum(
            "ij...,ij->i...", vec, self._inverse[s][:, det, i, :], optimize="greedy"
        )


    def pgradient(self):
        """Compute the parameter gradient of Phi.
        Returns :math:`\partial_p \Phi` as a dictionary of numpy arrays,
        which correspond to the parameter dictionary.

        The wave function is given by PhiB = \sqrt(\sum(ci Di^2)), with an implicit sum

        We have two sets of parameters:

        Determinant coefficients:
        di PhiB = 1/2 (Dui Ddi)^2 / Phi

        Orbital coefficients assuming orbital corresponds to an up determinant:
        dj PhiB = ci (Dui Ddi)^2 tr[Dui^-1 dj(Dui)]/PhiB

        Using the Determinant coefficient expression

        dj PhiB = ci * 

        Let's suppose that j corresponds to an up orbital coefficient. Then
        dj (Dui Ddi) = (dj Dui)/Dui Dui Ddi/psi = (dj Dui)/Dui di psi/psi
        where di psi/psi is the derivative defined above.
        """
        d = {}

        # Determinant coefficients
        curr_val = self.value() #sign, val
        # detsq[spin][configuration, determinant]
        detsq = (
            self._detsq[0][:, self._det_map[0]],
            self._detsq[1][:, self._det_map[1]],
        )

        d["det_coeff"] = gpu.cp.zeros(detsq[0].shape[1:], dtype=detsq[0].dtype)
        #sgn(Dup)*sgn(Ddn)*exp(Dup+Ddn)/[sgn(psi)*exp(logval)]
        d["det_coeff"] = 1./2 * (
            detsq[0] * detsq[1] 
            ) / gpu.cp.array(curr_val[1][:, np.newaxis])
        
        # Orbital coefficients
        for s, parm in zip([0, 1], ["mo_coeff_alpha", "mo_coeff_beta"]):
            ao = self._aovals[
                :, :, s * self._nelec[0] : self._nelec[s] + s * self._nelec[0], :
            ]
            # Derivatives wrt molecular orbital coefficients 
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
                    * 2 * d["det_coeff"][:, di, np.newaxis, np.newaxis]
                )

        for k, v in d.items():
            d[k] = gpu.asnumpy(v)

        for k in list(d.keys()):
            if np.prod(d[k].shape) == 0:
                del d[k]

        return d