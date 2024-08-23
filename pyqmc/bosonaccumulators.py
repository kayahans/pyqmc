import numpy as np
import energy
import bosonenergy
import pyqmc.ewald as ewald
import copy

from accumulators import LinearTransform
import bosonslater
import jastrowspin
from boson_stochastic_reconfiguration import BosonStochasticReconfiguration
# from pyqmc.stochastic_reconfiguration import StochasticReconfiguration
from wftools import generate_wf

PGradTransform = BosonStochasticReconfiguration

def boson_gradient_generator(mf, wf, to_opt=None, nodal_cutoff=1e-3, **ewald_kwargs):
    return PGradTransform(
        ABQMCEnergyAccumulator(mf, **ewald_kwargs),
        LinearTransform(wf.parameters, to_opt),
        nodal_cutoff=nodal_cutoff,
    )

class ABQMCEnergyAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __init__(self, mf, **kwargs):
        self.mol = mf.mol
        self.dm = mf.dm
        self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        

        if hasattr(self.mol, "a"):
            self.coulomb = ewald.Ewald(self.mol, **kwargs)
        else:
            self.coulomb = energy.OpenCoulomb(self.mol, **kwargs)

    def __call__(self, configs, wf):
        ee, ei, ii = self.coulomb.energy(configs)
        try:
            nwf = len(wf.wf_factors)
        except:
            nwf = 1

        if nwf == 1:
            nup_dn = wf._nelec
        else:
            for wfi in wf.wf_factors:
                if isinstance(wfi, bosonslater.BosonWF):
                    nup_dn = wfi._nelec
        vh,vxc,ecorr = bosonenergy.dft_energy(self.mol, self.dm, self.mo_energy, self.mo_occ, configs, nup_dn)
        ke1, ke2, grad2 = bosonenergy.boson_kinetic(configs, wf)
        ke = ke1+ke2
        energies =  {
            "ka": ke1,
            "kb": ke2,
            "grad2": grad2,
            "ke": ke,
            "ee": ee,
            "ei": ee,
            "vh": vh,
            "vxc": vxc,
            "corr": np.ones(ee.shape)*ecorr,
            "ei": ei, # For debugging, ei is not used in ABQMC
            "ii":np.ones(ee.shape)*ii,
            # Eq. 21-22 in doi: 10.1063/5.0155513 is the electronic energy
            # Therefore ii term is added here
            # V_MF = V_H + V_XC (only supports LDA for now)
            # E_Corr is the sum of KS eigenvalues 
            "total": ke + ee - (vh + vxc) + ecorr + ii,
        }
        return energies 

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def var(self, configs, wf):
        return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}
    
    def keys(self):
        return set(["ke", "ee", "vxc", "ei", "total", "grad2"])

    def shapes(self):
        return {"ke": (), "ee": (), "vxc": (), "ei": (), "ecp": (), "total": (), "grad2": ()}

class ABVMCMatrixAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __call__(self, configs, wf):
        # nconf, nelec, _ = configs.configs.shape
        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        # import pdb
        # pdb.set_trace()
        boson_wf.recompute(configs)
        # updets = boson_wf._dets[0][:, :, boson_wf._det_map[0]]
        # dndets = boson_wf._dets[1][:, :, boson_wf._det_map[1]]
        
        # nup = np.einsum('ni, ni, nj, nj->nij', updets[0], np.exp(updets[1]), updets[0], np.exp(updets[1]))
        # ndn = np.einsum('ni, ni, nj, nj->nij', dndets[0], np.exp(dndets[1]), dndets[0], np.exp(dndets[1]))

        # psi_i = np.einsum('ni, ni, ni, ni->ni', updets[0], np.exp(updets[1]), dndets[0], np.exp(dndets[1]))
        # psi_ij = np.sqrt(np.einsum('ni, ni, nj, nj ->nij', psi_i, psi_i, psi_i, psi_i))
        # rho = np.sum(psi_i**2, axis=1)
        # rho = np.exp(wf.value()[1])**2

        # ovlp_ij  = nup * ndn 
        # ovlp_nom = np.einsum('nij, n->nij', ovlp_ij, rho)
        # ovlp_nom = ovlp_ij/psi_ij
        # ovlp_d1 = np.einsum('ni, ni, n->ni', psi_i, psi_i, rho)
        
        # Using different understanding of determinants (Incorrect)
        # Option 1 
        # updet_sign, updet_val = boson_wf._dets[0]
        # dndet_sign, dndet_val = boson_wf._dets[1]
        # Option 2 
        # updet_sign, updet_val = boson_wf._dets[0][:, :, boson_wf._det_map[0]]
        # dndet_sign, dndet_val = boson_wf._dets[1][:, :, boson_wf._det_map[1]]
        # sign, logval = boson_wf.value()
        # val = np.exp(logval)
        # rho = val**2 

        # nup = np.einsum('ni, ni, nj, nj->nij', updet_sign, np.exp(updet_val), updet_sign, np.exp(updet_val)) # ui * uj
        # ndn = np.einsum('ni, ni, nj, nj->nij', dndet_sign, np.exp(dndet_val), dndet_sign, np.exp(dndet_val)) # di * dj 
        # psi_i = np.einsum('ni, ni, ni, ni->ni', updet_sign, np.exp(updet_val), dndet_sign, np.exp(dndet_val)) # ui * di 

        # ovlp_ij = np.einsum('nij, nij, n ->nij', nup, ndn, 1./rho) # w(R) * (ui * uj) * (di * dj), w = 1 / rho

        # norm_i = np.einsum('ni, ni, n ->ni', psi_i, psi_i, 1./rho)

        # Option 3
        if boson_wf.ovlp is None:
            num_ao = boson_wf._aovals.shape[-1]

            det_dim_up = boson_wf._dets[0][1].shape[-1]
            det_dim_dn = boson_wf._dets[1][1].shape[-1]
            det_dim = det_dim_up * det_dim_dn

            occ_arr_shape = (2, det_dim, num_ao)
            occ_arr = np.zeros(occ_arr_shape)
            for i in range(det_dim):
                up_i = boson_wf._det_map[0,i]
                dn_i = boson_wf._det_map[1,i]
                occ_arr[0, i][boson_wf._det_occup[0][up_i]] = 1
                occ_arr[1, i][boson_wf._det_occup[1][dn_i]] = 1

            mf_ovlp = boson_wf.mf_ovlp
            ovlp = np.einsum('io, op, jp->ij',occ_arr[0], mf_ovlp, occ_arr[0])
            boson_wf.ovlp = ovlp
        else:
            ovlp = boson_wf.ovlp

        sign, logval = boson_wf.value()
        val = np.exp(logval)
        rho = val**2 

        updet_sign, updet_val = boson_wf._dets[0][:, :, boson_wf._det_map[0]]
        dndet_sign, dndet_val = boson_wf._dets[1][:, :, boson_wf._det_map[1]]
        nup_i = np.einsum('ni, ni->ni', updet_sign, np.exp(updet_val)) # ui
        ndn_i = np.einsum('ni, ni->ni', dndet_sign, np.exp(dndet_val)) # uj

        nup_ij = np.einsum('ni, ij, nj->nij', nup_i, ovlp, nup_i) # ui * uj
        ndn_ij = np.einsum('ni, ij, nj->nij', ndn_i, ovlp, ndn_i) # ui * uj

        ovlp_ij = np.einsum('nij, nij, n ->nij', nup_ij, ndn_ij, 1./rho) # w(R) * (ui * uj) * (di * dj), w = 1 / rho
        
        psi_i = np.einsum('ni, ni, ni, ni->ni', updet_sign, np.exp(updet_val), dndet_sign, np.exp(dndet_val)) # ui * di 
        norm_i = np.einsum('ni, ni, n ->ni', psi_i, psi_i, 1./rho)
        # import pdb
        # pdb.set_trace()
        # norm_ij = np.sqrt(np.einsum('i, j ->ij', norm_i, norm_i))

        # mat = ovlp_ij/norm_ij

        # print(boson_wf._det_map, updets.shape, dndets.shape, nup.shape, ndn.shape, ovlp.shape)
        # _ = [wfi.recompute(configs) for wfi in self.wfs]
        # _, wf_val = boson_wf.recompute(configs)
        # self.wfs = None
        # phase, log_vals = [
        #     np.nan_to_num(np.array(x)) for x in zip(*[wfi.value() for wfi in self.wfs])
        # ]
        # import pdb
        # pdb.set_trace()
        # psi = np.array(phase * np.nan_to_num(np.exp(log_vals)))
        
        
        
        
        
        # wf_val = np.nan_to_num(np.exp(wf_val))
        # wf_valr = boson_wf.value_real()

        # ovlp = np.einsum("ic,jc->cij", psi.conj(), psi / wf_val**2)
        # rho = np.mean(np.nan_to_num(np.exp(2 * (log_vals))), axis=0)
        
        # Test 1
        # ref = np.max(log_vals, axis=0)  # for numerical stability
        # rho = np.mean(np.nan_to_num(np.exp(2 * (log_vals - ref))), axis=0)
        # psi = phase * np.nan_to_num(np.exp(log_vals - ref)) 
        # ovlp2 = np.einsum("ic,jc->cij", psi.conj(), psi / rho)
        
        # Test 2 (last used)
        # ref = np.max(log_vals, axis=0)  # for numerical stability
        # rho = np.mean(np.nan_to_num(np.exp(2 * (wf_val - ref))), axis=0)
        # psi = phase * np.nan_to_num(np.exp(log_vals - ref)) 
        # ovlp = np.einsum("ic,jc->cij", psi.conj(), psi / rho)


        # Value 
        # updets = boson_wf._dets[0][:, :, boson_wf._det_map[0]]
        # dndets = boson_wf._dets[1][:, :, boson_wf._det_map[1]]
        # det_coeffs = boson_wf.parameters["det_coeff"]
        # upref = gpu.cp.amax(updets[1]).real
        # dnref = gpu.cp.amax(dndets[1]).real
        # logvals = 2*(updets[1] - upref + dndets[1] - dnref)
        # wf_vala = gpu.cp.einsum("d,id->i", det_coeffs, gpu.cp.exp(logvals))
        
        # wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_vala))
        # wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_vala)) + 2*(upref + dnref))

        # import pdb
        # pdb.set_trace()
        
        delta = ovlp_ij*0
        wf_val = ovlp_ij*0

        # delta = ovlp_nom*0
        # wf_val = ovlp_nom*0
        
        # sum over electrons for the gradient terms
        # numer_e = 0
        # from mc import limdrift
        # for e in range(nelec):  
        #     log_grads = [np.nan_to_num(limdrift(np.array(x).T).T) for x in [wfi.gradient_value(e, configs.electron(e))[0] for wfi in self.wfs]]
        #     grads = np.array([x[0]*x[1] for x in zip(log_vals, log_grads)])
        #     wf_grad, _ = boson_wf.gradient_value(e, configs.electron(e))
        #     j_grad, _ = jastrow_wf.gradient_value(e, configs.electron(e))
        #     numer1 = gpu.cp.einsum("c, dxc->cdx", wf_val, grads)
        #     numer2 = gpu.cp.einsum("xc, dc->cdx", wf_grad, psi)
        #     numer_e += gpu.cp.einsum("cdx, xc->dc", numer1-numer2, j_grad)

        # numer  = gpu.cp.einsum("nc, lc ->cnl", psi, numer_e)
        # delta  = gpu.cp.einsum("cnl, c ->cnl", numer, 1./wf_val)
        # results = {'delta':delta, 'ovlp_ij': ovlp_ij, 'psi_ij': psi_ij, 'ovlp_nom':ovlp_nom, 'wf_val': wf_val, 
        #         #    'ovlp_d1':ovlp_d1,
        #            }
        results = {'delta':delta, 'ovlp_ij': ovlp_ij, 'norm_i':norm_i}

        return results 

    def avg(self, configs, wf):
        results = self(configs, wf)
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def var(self, configs, wf):
        return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}
    
    def keys(self):
        return set(["matrix"])

    def shapes(self):
        return {"matrix": ()}
