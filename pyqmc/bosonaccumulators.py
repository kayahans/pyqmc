import numpy as np
import pyqmc.gpu as gpu
import energy
import bosonenergy
import pyqmc.ewald as ewald
import pyqmc.eval_ecp as eval_ecp
import copy

from accumulators import LinearTransform
import bosonslater
# import bosonjastrowspin
from boson_stochastic_reconfiguration import StochasticReconfiguration
from wftools import generate_wf

PGradTransform = StochasticReconfiguration

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
        self.mf = mf

        if hasattr(self.mol, "a"):
            self.coulomb = ewald.Ewald(self.mol, **kwargs)
        else:
            self.coulomb = energy.OpenCoulomb(self.mol, **kwargs)

    def __call__(self, configs, wf):
        ee, ei, ii = self.coulomb.energy(configs)
        mf = self.mf
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
        vh,vxc,ecorr = bosonenergy.dft_energy(mf, configs, nup_dn)
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

# class ABVMCMatrixAccumulator:
#     """Returns local energy of each configuration in a dictionary."""

#     def __init__(self, mf, mc, **kwargs):
#         self.mol = mf.mol
#         self.mf = mf
#         self.mc = mc
#         wfs = []
#         for fci in self.mc.fci:
#             mci = copy.copy(self.mc)
#             mci.ci = fci
#             wfi, _ = generate_wf(self.mol, self.mf, mc=mci, jastrow=None)
#             wfs.append(wfi)
#         self.wfs = wfs

#     def __call__(self, configs, wf):
#         nconf, nelec, _ = configs.configs.shape
#         wave_functions = wf.wf_factors
#         for wave in wave_functions:
#             if isinstance(wave, BosonWF):
#                 boson_wf = wave
#             if isinstance(wave, bosonjastrowspin.BosonJastrowSpin):
#                 jastrow_wf = wave        

#         _ = [wfi.recompute(configs) for wfi in self.wfs]

#         phase, log_vals = [
#             np.nan_to_num(np.array(x)) for x in zip(*[wfi.value() for wfi in self.wfs])
#         ]
#         psi = np.array(phase * np.nan_to_num(np.exp(log_vals)))
#         _ = boson_wf.recompute(configs)
#         _, wf_val = boson_wf.value()
#         ovlp = np.einsum("ic,jc->cij", psi.conj(), psi / wf_val**2)
        
#         # sum over electrons for the gradient terms
#         numer_e = 0
#         from mc import limdrift
#         for e in range(nelec):  
#             log_grads = [np.nan_to_num(limdrift(np.array(x).T).T) for x in [wfi.gradient_value(e, configs.electron(e))[0] for wfi in self.wfs]]
#             grads = np.array([x[0]*x[1] for x in zip(log_vals, log_grads)])
#             wf_grad, _ = boson_wf.gradient_value(e, configs.electron(e))
#             j_grad, _ = jastrow_wf.gradient_value(e, configs.electron(e))
#             numer1 = gpu.cp.einsum("c, dxc->cdx", wf_val, grads)
#             numer2 = gpu.cp.einsum("xc, dc->cdx", wf_grad, psi)
#             numer_e += gpu.cp.einsum("cdx, xc->dc", numer1-numer2, j_grad)

#         numer  = gpu.cp.einsum("nc, lc ->cnl", psi, numer_e)
#         delta  = gpu.cp.einsum("cnl, c ->cnl", numer, 1./wf_val)

#         results = {'delta':delta, 'ovlp':ovlp, 'wf_val': wf_val, }

#         return results 

#     def avg(self, configs, wf):
#         results = self(configs, wf)
#         return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

#     def var(self, configs, wf):
#         return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

#     def has_nonlocal_moves(self):
#         return self.mol._ecp != {}
    
#     def keys(self):
#         return set(["matrix"])

#     def shapes(self):
#         return {"matrix": ()}
