import numpy as np
import energy
import bosonenergy
import pyqmc.ewald as ewald
import copy

from accumulators import LinearTransform
import bosonslater
import jastrowspin
from boson_stochastic_reconfiguration import BosonStochasticReconfiguration
from stochastic_reconfiguration import StochasticReconfiguration
from wftools import generate_wf

from bosonslater import timer_func

PGradTransform = BosonStochasticReconfiguration
# PGradTransform = StochasticReconfiguration

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
    
    @timer_func
    def __call__(self, configs, wf):
        ee, ei, ii = self.coulomb.energy(configs)
        try:
            nwf = len(wf.wf_factors)
        except:
            nwf = 1

        if nwf == 1:
            nup_dn = wf._nelec
        else:
            nup_dn = None
            for wfi in wf.wf_factors:
                if nup_dn is None:
                    try:
                        nup_dn = wfi._nelec
                    except:
                        pass
        vh,vxc,ecorr = bosonenergy.dft_energy(self.mol, self.dm, self.mo_energy, self.mo_occ, configs, nup_dn)
        ke1, ke2, grad2 = bosonenergy.boson_kinetic(configs, wf)
        # ke1 *= 0
        # ke2 *= 0
        ke = ke1+ke2
        energies =  {
            "ka": ke1,
            "kb": ke2,
            "grad2": grad2,
            "ke": ke,
            "ee": ee,
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
        # print(np.mean(ke1), np.mean(ke2), np.mean(ee), np.mean(vh), np.mean(vxc), np.mean(ecorr), np.mean(ei), np.mean(ii), np.mean(energies['total']))
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
    
    @timer_func
    def __call__(self, configs, wf):
        
        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        
        nconf, nelec, _ = configs.configs.shape

        phase, log_val = boson_wf.value() # Eq. 4
        val = phase * np.nan_to_num(np.exp(log_val)) #[c]
        
        phases, log_vals = boson_wf.value_dets()
        psis = phases * np.nan_to_num(np.exp(log_vals))
        psi_basis = np.einsum('cn, c->nc', psis, 1./val) # eq. 14
        # variant 1, using Acceptance from VMC
        # acc = copy.deepcopy(wf.accept_array)
        # facc = np.sum(acc, axis=0)/nelec
        # ovlp_ij = nconf /np.sum(facc) * np.einsum("lc,nc,c->cln", psi_basis.conj(), psi_basis, facc)

        # variant 2 do not use acceptance from VMC 
        ovlp_ij = np.einsum("lc,nc->cln", psi_basis.conj(), psi_basis)

        delta = 0
        
        for e in range(nelec):
            epos = configs.electron(e)
            grad_b_e = wf.gradient(e, epos)
            grad_n = boson_wf.gradient_dets(e, epos)
            grad_psi_basis = np.einsum('nc, nxc->nxc', psi_basis, grad_n-grad_b_e)
            grad_j = jastrow_wf.gradient(e, configs.electron(e))
            # variant 1 use acceptance from VMC
            # delta += nconf /np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, grad_j, grad_psi_basis, acc[e])
            # variant 2 do not use acceptance from VMC
            delta += np.einsum("nc,xc,lxc->cnl", psi_basis, grad_j, grad_psi_basis)

        results = {'delta':delta, 'ovlp_ij': ovlp_ij}
        return results 

    def avg(self, configs, wf):
        # results = self(configs, wf)
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def var(self, configs, wf):
        return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}
    
    def keys(self):
        return set(["matrix"])

    def shapes(self):
        return {"matrix": ()}

class ABDMCMatrixAccumulator:
    """Returns local energy of each configuration in a dictionary."""
    
    @timer_func
    def __call__(self, configs, wf):
        
        # First check if a component of the matrix element works
        # configs: current configs with accept/reject
        # wf.curr_config.configs: configs prior to accept/reject
        # wf.next_config.configs: configs with gaussian added only 
        nconf, nelec, _ = configs.configs.shape

        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        
        # boson_wf = wf

        # 1. Fernando's method
        ri = wf.curr_config.configs
        matel2 = [0, 0]
        extrapolate_timesteps = np.sort([0.05, 0.1]) # smaller first 
        assert(extrapolate_timesteps.shape[0] == 2) # 2-point extrapolation only 
        t1, t2 = extrapolate_timesteps
        # prefactors = [t2/(t2-t1), -t1/(t2-t1)]
        for ind, tstep in enumerate(extrapolate_timesteps):
            next_config = copy.deepcopy(configs)    
            for e in range(nelec):
                gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
                newcoordeg = wf.curr_config.configs[:, e, :] + gauss
                newcoordeg = wf.curr_config.make_irreducible(e, newcoordeg)
                next_config.move(e, newcoordeg, np.ones(nconf, dtype=bool))

            rf = next_config.configs
            drdt = -(rf-ri)/tstep

            wf.recompute(next_config)
            phase, log_val = wf.value() #log(\psi_BT)
            val = phase * np.nan_to_num(np.exp(log_val)) #\psi_BT

            phases, log_vals = boson_wf.value_dets() #log(\phi_n)
            psis = phases * np.nan_to_num(np.exp(log_vals))
            psi_basis_s = np.einsum('cn, c->nc', psis, 1./val) # eq. 14
            
            # acc = copy.deepcopy(wf.accept_array)
            # acc[acc<1.0] = 0
            for e in range(nelec):
                epos_s = next_config.electron(e)
                # grad_t_e_s = wf.gradient(e, epos_s) # \nabla{log(\Psi_B^T)}
                grad_b_e_s = boson_wf.gradient(e, epos_s) # \nabla{log(\Psi_B)}
                grad_n_s = boson_wf.gradient_dets(e, epos_s) # \nabla{log(\Phi_n)}
                grad_psi_basis_s = np.einsum('nc, nxc->nxc', psi_basis_s, grad_n_s-grad_b_e_s)
                gradf_s = -drdt[:,e,:] 
                # matel2 += np.einsum("nc,cx,lxc,c->cnl", psi_basis_s, gradf_s, grad_psi_basis_s, acc[e])
                matel2[ind] += np.einsum("nc,cx,lxc->cnl", psi_basis_s, gradf_s, grad_psi_basis_s)

        # # 2. Using configs from accept/reject
        wf.recompute(configs)
        phase, log_val = wf.value() #log(\psi_BT)
        val = phase * np.nan_to_num(np.exp(log_val)) #\psi_BT

        phases, log_vals = boson_wf.value_dets() #log(\phi_n)
        psis = phases * np.nan_to_num(np.exp(log_vals))
        psi_basis = np.einsum('cn, c->nc', psis, 1./val) # eq. 14

        ovlp_ij = np.einsum("lc,nc->cln", psi_basis.conj(), psi_basis)

        # import pdb
        # pdb.set_trace()
        matel = 0
        for e in range(nelec):
            epos = configs.electron(e)
            grad_t_e = wf.gradient(e, epos)
            grad_b_e = boson_wf.gradient(e, epos)
            grad_j_e = jastrow_wf.gradient(e, epos)
            grad_n = boson_wf.gradient_dets(e, epos)
            grad_psi_basis = np.einsum('nc, nxc->nxc', psi_basis, grad_n-grad_b_e)
            # \nabla(f_B) = \nabla(\psi_BT^2) = 2 * \nabla(log(\psi_BT)) * \psi_BT**2
            # gradf = -2 * grad_b_e # + grad_b_e + grad_t_e
            # matel += nconf/np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, gradf, grad_psi_basis, acc[e])
            # matel += np.einsum("nc,xc,lxc->cnl", psi_basis, gradf, grad_psi_basis)
            matel += np.einsum("nc,xc,lxc->cnl", psi_basis, -grad_j_e, grad_psi_basis)
            matel_temp = np.einsum("nc,xc,lxc->cnl", psi_basis, grad_b_e + grad_t_e, grad_psi_basis)
            matel2[0] += matel_temp
            matel2[1] += matel_temp
        
        # wf.recompute(configs)
        # Matel 2 is the statistical approach
        # Matel 1 is analytical (these are equal only when VMC is used)
        results = {
                    'matel':matel, 
                    'matel2_t1':matel2[0], 
                    'matel2_t2':matel2[1], 
                    'ovlp': ovlp_ij}
        return results 

    def avg(self, configs, wf):
        # results = self(configs, wf)
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def var(self, configs, wf):
        return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}
    
    def keys(self):
        return set(["matrix"])

    def shapes(self):
        return {"matrix": ()}
