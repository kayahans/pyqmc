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

        for wfi in wf.wfs:
            wfi.recompute(configs)

        phib_sign, phib_logval = boson_wf.value() # Eq. 4
        phib_val = phib_sign * np.nan_to_num(np.exp(phib_logval)) #[c]
        
        psibt_sign, psibt_logval = wf.value() # Eq. 4
        psibt_val = psibt_sign * np.nan_to_num(np.exp(psibt_logval)) #[c]
        
        phase, log_vals = [
            np.nan_to_num(np.array(x)) for x in zip(*[wfi.value() for wfi in wf.wfs])
        ]
        
        psi = phase * np.nan_to_num(np.exp(log_vals))
        ovlp_ij = np.einsum("lc,nc, c->cln", psi.conj(), psi, (1./phib_val**2))
        # No jastrow variant
        # ovlp_ij = np.einsum("ic,jc->cij", psi.conj(), psi * (1./phib_val**2))
        # Older variant
        # ovlp_ij = np.einsum("ic,jc, c->cij", psi.conj(), psi, (psibt_val**2/phib_val**4))

        # Delta 
        # Eq. 34 
        # wfn_inner is below
        # \nabla\phi_n \cdot \nabla{J} = \frac{\nabla\Phi_n\Phi_B-\Phi_n\nabla\Phi_B}{\Phi_B^2} \cdot \nabla{J}
        # =\frac{{\nabla}[ln(\Phi_n)]\Phi_n\Phi_B-\Phi_n{\nabla}[ln(\Phi_B)]\Phi_B}{\Phi_B^2} \cdot \nabla{J}
        # =\frac{{\nabla}[ln(\Phi_n)]\Phi_n-\Phi_n{\nabla}[ln(\Phi_B)]}{\Phi_B} \cdot \nabla{J}
        # =\frac{\Phi_n}{\Phi_B}\{{\nabla}[ln(\Phi_n)]-{\nabla}[ln(\Phi_B)]\} \cdot \nabla{J}
        # where
        # \frac{\Phi_n}{\Phi_B} = e^{ln(\frac{\Phi_n}{\Phi_B})} = e^{ln(\Phi_n)-ln(\Phi_B)}
        wfn_inner = np.zeros(psi.shape)
        for ind, wfi in enumerate(wf.wfs):  
            wfi_sign, wfi_value = wfi.value()
            nb_ratio = wfi_sign * np.exp(wfi_value - phib_logval)
            grad = 0
            for e in range(nelec):
                grad_phi_n = wfi.gradient(e, configs.electron(e))
                grad_b = boson_wf.gradient(e, configs.electron(e))
                grad_j = jastrow_wf.gradient(e, configs.electron(e))
                grad += np.einsum("dc,dc->c", grad_phi_n - grad_b, grad_j)
            wfn_inner[ind] = np.einsum("c, c -> c", nb_ratio, grad)
        
        # {\Psi_B^T}^2\phi_l \nabla\phi_n \cdot \nabla{J}
        # Ignore log instability for now
        # delta = np.einsum("c, lc, nc -> cln", psibt_val, psi, wfn_inner)
        delta = np.einsum("lc, c, nc -> cln", psi, 1./phib_val, wfn_inner)
        # Older variant
        # delta = np.einsum("c, c, lc, nc -> cln", psibt_val**2, 1./phib_val**2, psi, wfn_inner)
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
