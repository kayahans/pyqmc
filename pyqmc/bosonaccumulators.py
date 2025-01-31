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

def get_psi_basis(boson_wf):
    """Calculate the basis functions for the bosonic wave function.

    This function computes the basis functions Phi_l/Phi_B used in the bosonic wave function
    expansion, where Phi_l are the individual Slater determinants and Phi_B is the total 
    bosonic trial wave function.

    Args:
        boson_wf: A BosonWF object representing the bosonic wave function

    Returns:
        ndarray: Array of shape (ndet, nconfig) containing the basis functions ψᵢ/ψ_BT,
                where ndet is the number of determinants and nconfig is the number of 
                configurations. Each element [i,c] gives the ratio of determinant i to 
                the total wave function for configuration c.

    Notes:
        - Uses equations 4 and 14 from the reference paper
        - Handles numerical stability by using log values and nan_to_num
        - The basis functions sum to 1 for each configuration np.mean(psi_basis**2)=1.0
    """
    phase, log_val = boson_wf.value() # log(Phi_B) Eq. 4
    val = phase * np.nan_to_num(np.exp(log_val)) #Phi_B
    
    phases, log_vals = boson_wf.value_dets() #log(Phi_l)
    psis = phases * np.nan_to_num(np.exp(log_vals)) # Phi_l
    psi_basis = np.einsum('cn, c->nc', psis, 1./val) # Phi_l/Phi_B, eq. 14
    return psi_basis

class ABVMCMatrixAccumulator:
    """Accumulator for matrix quantities in Auxiliary Boson Variational Monte Carlo (ABVMC) calculations.
    
    This class computes and accumulates matrices needed for ABVMC calculations, specifically:
    - Overlap matrices between different basis states
    - Gradient-based quantities incorporating both bosonic and Jastrow correlation components
    - Delta matrix is defined in eq. 34. 
    - It should be used with VMC calculations
    - If used with DMC calculations, otherwise the \Psi_B_T term will be f_B, will be a mixed estimator
    - However in that case, when DMC branching is disabled, the ABVMC is expected to be equal to ABCDMC. 

    Methods
    -------
    __call__(configs, wf)
        Compute matrices for given configurations and wavefunction.
        
        Args:
            configs: Object containing electron configurations
            wf: Wavefunction object containing both bosonic and Jastrow components
            
        Returns:
            dict: Contains 'delta' and 'ovlp' matrices
                - delta: Gradient-based quantity incorporating both wavefunction components
                - ovlp: Overlap matrices between basis states
    
    Notes
    -----
    The class expects the wavefunction object to contain both a BosonWF and 
    JastrowSpin component, which are used to compute various gradients and
    matrix elements needed in the ABVMC calculation.
    """

    @timer_func
    def __call__(self, configs, wf):
        
        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        
        nconf, nelec, ndets = configs.configs.shape
        psi_n = get_psi_basis(boson_wf)
        
        # variant 1, using Acceptance from VMC
        # acc = copy.deepcopy(wf.accept_array)
        # facc = np.sum(acc, axis=0)/nelec
        # ovlp_ij = nconf /np.sum(facc) * np.einsum("lc,nc,c->cln", psi_basis.conj(), psi_basis, facc)
        # variant 2 do not use acceptance from VMC 
        ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n)

        delta = 0
        
        for e in range(nelec):
            epos = configs.electron(e)
            # grad_b_e = wf.gradient(e, epos) ## Jan 31, 2025
            log_grad_b_e = boson_wf.gradient(e, epos)
            log_grad_n = boson_wf.gradient_dets(e, epos)
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, log_grad_n-log_grad_b_e)
            grad_j = jastrow_wf.gradient(e, configs.electron(e))
            # variant 1 use acceptance from VMC
            # delta += nconf /np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, grad_j, grad_psi_basis, acc[e])
            # variant 2 do not use acceptance from VMC
            delta += np.einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)

        results = {'delta':delta, 'ovlp': ovlp_ij}
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



        for e in range(nelec):
            # Get position of electron e
            epos_s = configs.electron(e)
            # ∇log(Phi_n) 
            loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
            # ∇log(Psi_B) eq. 4
            loggrad_b = boson_wf.gradient(e, epos_s) 
            # ∇Psi_n = ∇(Phi_n/Phi_B)
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b)  
            grad_j = jastrow_wf.gradient(e, epos_s)
            matel += np.einsum('lc, nxc, nxc->cln', psi_n, grad_j, grad_psi_n)

class ABCDMCMatrixAccumulator:
    """Accumulator for computing matrix elements in Auxiliary-field Boson Diffusion Monte Carlo (ABCDMC).
    
    Specifically calculates:
    1. Overlap matrices between different basis states
    2. Matrix elements involving kinetic and potential energy terms
    
    The calculation includes:
    - Wavefunctions ratios
    - Gradients and Laplacians of both bosonic and trial wavefunctions
    - Integration by parts terms for the kinetic energy
    
    
    Methods
    -------
    __call__(configs, wf)
        Compute matrix elements for given configurations and wavefunction.
        
        Parameters
        ----------
        configs : object
            Contains electron configurations with shape (nconf, nelec, ndets)
        wf : object
            Wavefunction object containing both BosonWF and JastrowSpin components
            
        Returns
        -------
        dict
            'matel': Matrix elements including kinetic and potential terms
            'ovlp': Overlap matrices between basis states
            
    Notes
    -----
    The implementation follows quantum Monte Carlo formalism where:
    - Φ_B is the bosonic wavefunction
    - Φ_n are the determinant components
    - Ψ_BT is the Slater Jastrow trial wavefunction (eq. 4)
    
    The matrix elements are computed using:
    1. Gradient terms: ∇Ψ_n = ∇(Φ_n/Φ_B)
    2. Laplacian terms: ∇²(Φ_n/Φ_B)
    3. Integration by parts for the kinetic energy terms
    """    
    
    @timer_func
    def __call__(self, configs, wf):
        
        nconf, nelec, ndets = configs.configs.shape

        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        
        psi_n = get_psi_basis(boson_wf) # Phi_l/Phi_B
        ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n)

        # phase_fb, logval_fb = wf.value() # log(Psi_BT)
        # val_fb = phase_fb * np.nan_to_num(np.exp(logval_fb)) # Psi_BT

        phase_phi_b, logval_phi_b = boson_wf.value() # log(Phi_B)
        val_phi_b = phase_phi_b * np.nan_to_num(np.exp(logval_phi_b)) # Phi_B

        # Matrix element by integration parts on the ∇f_B term of the eq. 23 
        # For integration by parts see eq. 17 in the reference paper
        matel = np.einsum('lc, ln, nc->cln', psi_n, boson_wf.hmf, psi_n) # Psi_l * H * Psi_n
        # phases, log_vals = boson_wf.value_dets() #log(Phi_l)
        # psis = phases * np.nan_to_num(np.exp(log_vals)) # Phi_l
        
        for e in range(nelec):
            # Get position of electron e
            epos_s = configs.electron(e)

            # ∇log(Phi_n) 
            loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
            # ∇log(Psi_B) eq. 4
            loggrad_b = boson_wf.gradient(e, epos_s) 
            # ∇Psi_n = ∇(Phi_n/Phi_B)
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b)  
            # ∇²(Psi_B)/Psi_B
            lap_phi_b = boson_wf.laplacian(e, epos_s)      
            # ∇²Phi_n
            lap_phi_n = boson_wf.laplacian_dets(e, epos_s) 
            # ∇log(Psi_B^T)
            loggrad_psi_bt = wf.gradient(e, epos_s)

            # lap_psi_n: (eq. before eq. 23)
            # ∇²(Phi_n/Phi_B) = [∇²(Phi_n)*Phi_B - Phi_n*∇²(Phi_B)]/(Phi_B^2) 
            #                   - 2*∇(Phi_B)·∇(Phi_n/Phi_B)/Phi_B
            lap_psi_n  = np.einsum('cn, c->cn', lap_phi_n, 1./val_phi_b) # ∇²(Phi_n)/Phi_B
            lap_psi_n -= np.einsum('c, nc->cn', lap_phi_b, psi_n) # -∇²(Phi_B)/Phi_B * Psi_n or -∇²(Phi_B)/Phi_B^2 * Phi_n 
            lap_psi_n -= 2 * np.einsum('xc, nxc->cn', loggrad_b, grad_psi_n) # - 2*∇(log(Psi_B))*∇(Psi_n)

            matel += np.einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n) # ∇Psi_l \dot ∇Psi_n 
            matel += np.einsum('lc, cn->cln', psi_n, lap_psi_n) # Psi_l * ∇²Psi_n
            matel += np.einsum('lc, xc, nxc->cln', psi_n, loggrad_b + loggrad_psi_bt, grad_psi_n) # Psi_l * [∇(log(Phi_B)) + ∇(log(Psi_BT))] \dot ∇Psi_n        

        results = {'matel':matel, 
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


class ABDMCMatrixAccumulator:
    """Accumulator for computing matrix elements in Auxiliary Boson Diffusion Monte Carlo.
    
    Based on eq. 18. 
    
    Note: Currently missing the EB-VB term in the matrix element calculation.
    
    
    Methods
    -------
    __call__(configs, wf)
        Compute matrix elements for given configurations and wavefunction.
        
        Parameters
        ----------
        configs : object
            Contains electron configurations with shape (nconf, nelec, ndim)
        wf : object
            Wavefunction object containing BosonWF component
            
        Returns
        -------
        dict
            'matel': Matrix elements including kinetic terms
                    Shape: (nconf, ndet, ndet)
            'ovlp': Overlap matrices between basis states
                   Shape: (nconf, ndet, ndet)
    
    """

    @timer_func
    def __call__(self, configs, wf):

        nconf, nelec, _ = configs.configs.shape

        wave_functions = wf.wf_factors
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            
        psi_n = get_psi_basis(boson_wf) # Phi_l/Phi_B
        ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n)

        # Matrix element by integration parts on the ∇f_B term of the eq. 23 
        # For integration by parts see eq. 17 in the reference paper
        matel = 0
        
        for e in range(nelec):
            # Get position of electron e
            epos_s = configs.electron(e)

            # ∇²Phi_n
            lap_phi_n = boson_wf.laplacian_dets(e, epos_s) 
            # ∇log(Phi_n) 
            loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
            # ∇log(Psi_B) eq. 4
            loggrad_b = boson_wf.gradient(e, epos_s) 
            # ∇Psi_n = ∇(Phi_n/Phi_B)
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b)  
            # ∇log(Psi_B^T)
            loggrad_psi_bt = wf.gradient(e, epos_s)

            matel += 1./2 * np.einsum('lc, cn->cln', psi_n, lap_phi_n) # Psi_l * ∇²Psi_n
            matel += 0 # Missing EB-VB term 
            matel -= np.einsum('lc, nxc, nxc->cln', psi_n, grad_psi_n-loggrad_psi_bt, grad_psi_n)  # Check indices

        results = {'matel':matel, 
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




# class ABDMCMatrixAccumulator:
#     """Returns local energy of each configuration in a dictionary."""
    
#     @timer_func
#     def __call__(self, configs, wf):
        
#         # First check if a component of the matrix element works
#         # configs: current configs with accept/reject
#         # wf.curr_config.configs: configs prior to accept/reject
#         # wf.next_config.configs: configs with gaussian added only 
#         nconf, nelec, _ = configs.configs.shape

#         wave_functions = wf.wf_factors
#         for wave in wave_functions:
#             if isinstance(wave, bosonslater.BosonWF):
#                 boson_wf = wave
#             if isinstance(wave, jastrowspin.JastrowSpin):
#                 jastrow_wf = wave        
        
#         # boson_wf = wf
#         matel2 = 0
#         # 1. Fernando's method
#         # ri = wf.curr_config.configs
#         # matel2 = [0, 0, 0, 0]
#         # extrapolate_timesteps = np.sort([0.01, 0.02, 0.05, 0.1]) # smaller first 
#         # assert(extrapolate_timesteps.shape[0] == 4) # 4-point extrapolation
#         # # t1, t2 = extrapolate_timesteps
#         # # prefactors = [t2/(t2-t1), -t1/(t2-t1)]
#         # for ind, tstep in enumerate(extrapolate_timesteps):
#         #     next_config = copy.deepcopy(configs)    
#         #     for e in range(nelec):
#         #         gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
#         #         newcoordeg = wf.curr_config.configs[:, e, :] + gauss
#         #         newcoordeg = wf.curr_config.make_irreducible(e, newcoordeg)
#         #         next_config.move(e, newcoordeg, np.ones(nconf, dtype=bool))

#         #     rf = next_config.configs
#         #     drdt = -(rf-ri)/tstep

#         #     wf.recompute(next_config)
#         #     psi_basis_s = get_psi_basis(boson_wf)
            
#         #     # acc = copy.deepcopy(wf.accept_array)
#         #     # acc[acc<1.0] = 0
#         #     for e in range(nelec):
#         #         epos_s = next_config.electron(e)
#         #         # grad_t_e_s = wf.gradient(e, epos_s) # \nabla{log(\Psi_B^T)}
#         #         grad_b_e_s = boson_wf.gradient(e, epos_s) # \nabla{log(\Psi_B)}
#         #         grad_n_s = boson_wf.gradient_dets(e, epos_s) # \nabla{log(\Phi_n)}
#         #         grad_psi_basis_s = np.einsum('nc, nxc->nxc', psi_basis_s, grad_n_s-grad_b_e_s)
#         #         gradf_s = -drdt[:,e,:] 
#         #         # matel2 += np.einsum("nc,cx,lxc,c->cnl", psi_basis_s, gradf_s, grad_psi_basis_s, acc[e])
#         #         matel2[ind] += np.einsum("nc,cx,lxc->cnl", psi_basis_s, gradf_s, grad_psi_basis_s)


#         # 2. Integration by parts
#         psi_n = get_psi_basis(boson_wf)
        
#         # This code calculates derivatives needed for quantum Monte Carlo calculations
#         # It loops over each electron and computes gradients and laplacians of wavefunctions
#         phase_fb, logval_fb = wf.value()
#         val_fb = phase_fb * np.nan_to_num(np.exp(logval_fb)) #\psi_BT

#         for e in range(nelec):
#             # Get position of electron e
#             epos_s = configs.electron(e)
            
#             # Calculate derivatives of the bosonic wavefunction Psi_B
#             grad_b = boson_wf.gradient(e, epos_s)      # Gradient: ∇log(Psi_B) 
#             lap_b = boson_wf.laplacian(e, epos_s)      # Laplacian: ∇²(Psi_B)/Psi_B
            
#             # # Calculate derivatives of each individual Slater determinant Phi_n that makes up Psi_B
#             grad_phi_n = boson_wf.gradient_dets(e, epos_s)  # ∇log(Phi_n) for each determinant
#             lap_phi_n = boson_wf.laplacian_dets(e, epos_s)  # ∇²Phi_n/Phi_n for each determinant
            
             # Gradient of ratio using product rule:
             # ∇(Phi_n/Psi_B) = (Phi_n/Psi_B) * (∇log(Phi_n) - ∇log(Psi_B))
#             grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, grad_phi_n-grad_b)
            
#             # Laplacian of ratio using product and chain rules:
#             # ∇²(Phi_n/Psi_B) = (Phi_n/Psi_B) * [∇²log(Phi_n) - ∇²log(Psi_B) - (∇log(Phi_n) - ∇log(Psi_B))²]
#             import pdb; pdb.set_trace()
#             lap_psi_n = np.einsum('nc, nc->nc', psi_n, lap_phi_n.T-lap_b)  # First term
#             lap_psi_n -= np.einsum('nxc, nxc->nc', grad_psi_n, grad_phi_n-grad_b)  # Second term
            
#             matel2 -= np.einsum("nc,c,lc->cnl", psi_n, val_fb, lap_psi_n)
#             matel2 -= np.einsum("nxc,c,lxc->cnl", grad_psi_n, val_fb, grad_psi_n)
            
#             # grad_t_e_s = wf.gradient(e, epos_s) # \nabla{log(\Psi_B^T)}
            
            
#         #     lap_n_s = boson_wf.laplacian_dets(e, epos_s) # \nabla^2{log(\Phi_n)}
#         #     grad_psi_basis_s = np.einsum('nc, nxc->nxc', psi_basis_s, grad_n_s-grad_b_e_s)
            
#             # matel2[ind] += np.einsum("nc,cx,lxc->cnl", psi_basis_s, gradf_s, grad_psi_basis_s)


#         # Use configs from accept/reject
#         wf.recompute(configs)
#         psi_basis = get_psi_basis(boson_wf)
#         ovlp_ij = np.einsum("lc,nc->cln", psi_basis.conj(), psi_basis)

#         matel = 0
#         for e in range(nelec):
#             epos = configs.electron(e)
#             grad_t_e = wf.gradient(e, epos)
#             grad_b_e = boson_wf.gradient(e, epos)
#             grad_j_e = jastrow_wf.gradient(e, epos)
#             grad_n = boson_wf.gradient_dets(e, epos)
#             grad_psi_basis = np.einsum('nc, nxc->nxc', psi_basis, grad_n-grad_b_e)
#             # \nabla(f_B) = \nabla(\psi_BT^2) = 2 * \nabla(log(\psi_BT)) * \psi_BT**2
#             # gradf = -2 * grad_b_e # + grad_b_e + grad_t_e
#             # matel += nconf/np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, gradf, grad_psi_basis, acc[e])
#             # matel += np.einsum("nc,xc,lxc->cnl", psi_basis, gradf, grad_psi_basis)
#             matel += np.einsum("nc,xc,lxc->cnl", psi_basis, grad_j_e, grad_psi_basis)
#             matel_temp = np.einsum("nc,xc,lxc->cnl", psi_basis, grad_b_e + grad_t_e, grad_psi_basis)
#             matel2 += matel_temp
#             # matel2[0] += matel_temp
#             # matel2[1] += matel_temp
#             # matel2[2] += matel_temp
#             # matel2[3] += matel_temp
        
#         # wf.recompute(configs)
#         # Matel 2 is the statistical approach
#         # Matel 1 is analytical (these are equal only when VMC is used)
#         results = {
#                     'matel':matel, 
#                     # 'matel2_t1':matel2[0], 
#                     # 'matel2_t2':matel2[1], 
#                     # 'matel2_t3':matel2[2], 
#                     # 'matel2_t4':matel2[3], 
#                     'ovlp': ovlp_ij}
#         return results 

#     def avg(self, configs, wf):
#         # results = self(configs, wf)
#         return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

#     def var(self, configs, wf):
#         return {k: np.sqrt(np.abs(it**2 - np.mean(it, axis=0)**2)) for k, it in self(configs, wf).items()}

#     def has_nonlocal_moves(self):
#         return self.mol._ecp != {}
    
#     def keys(self):
#         return set(["matrix"])

#     def shapes(self):
#         return {"matrix": ()}
