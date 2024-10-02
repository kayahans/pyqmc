# After the discussion of how to calculate the dot product of gradients
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
        
        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        

        nconf, nelec, _ = configs.configs.shape
        log_grad_wfs = []
        for wfi in wf.wfs:
            wfi.recompute(configs)
            grad = 0
            for e in range(nelec):
                ge, _, _ = wfi.gradient_value(e, configs.electron(e))
                grad += ge
            log_grad_wfs.append(grad)
        log_grad_wfs = np.array(log_grad_wfs)

        phase, log_vals = [
            np.nan_to_num(np.array(x)) for x in zip(*[wfi.value() for wfi in wf.wfs])
        ]
        
        ref = np.max(log_vals, axis=0)  # for numerical stability
        rho = np.mean(np.nan_to_num(np.exp(2 * (log_vals - ref))), axis=0)
        psi = phase * np.nan_to_num(np.exp(log_vals - ref))
        ovlp_ij = np.einsum("ic,jc->cij", psi.conj(), psi / rho)
        
        # Delta 
        grad_wfs = np.einsum("idc,ic->idc", log_grad_wfs, psi)*np.exp(ref) # [det, 3=g, conf]
        
        jastrow_wf.recompute(configs)
        #option 1 
        jastrow_grad = np.sum(np.array([jastrow_wf.gradient(e, configs.electron(e)) for e in range(nelec)]), axis=0)
        #option 2 
        # jastrow_phase, jastrow_log_val = jastrow_wf.value()
        # jastrow_val = jastrow_phase * np.nan_to_num(np.exp(jastrow_log_val))

        # jastrow_log_grad = 0
        # for e in range(nelec):
        #     ge, _, _ = jastrow_wf.gradient_value(e, configs.electron(e))
        #     jastrow_log_grad += ge

        # jastrow_grad = np.einsum("dc,c->dc", jastrow_log_grad, jastrow_val)
        psibt_sign, psibt_logval = wf.value() # Eq. 4
        phib_sign, phib_logval = boson_wf.value() # Eq. 4
        phib_log_grad = np.sum(np.array([boson_wf.gradient(e, configs.electron(e)) for e in range(nelec)]), axis=0) # log gradient, phib'/phib [3=g, conf]
        

        # Ignore log instability for now
        psibt_val = psibt_sign * np.nan_to_num(np.exp(psibt_logval)) #[c]
        phib_val = phib_sign * np.nan_to_num(np.exp(phib_logval)) #[c]
        delta_inner = grad_wfs - np.einsum("gc, dc->dgc", phib_log_grad, psi)
        delta = np.einsum("c, c, lc, ngc, gc -> cln", psibt_val**2, 1./phib_val**2, psi, delta_inner, jastrow_grad)

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
