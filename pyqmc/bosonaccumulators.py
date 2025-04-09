import numpy as np
from pyqmc import energy
from pyqmc import bosonenergy
import pyqmc.ewald as ewald
import copy

from pyqmc.accumulators import LinearTransform
from pyqmc import bosonslater
from pyqmc import jastrowspin


from pyqmc.bosonslater import timer_func

from pyqmc.accumulators import PGradTransform

def calculate_mf_density(mol, dm):
    """
    Calculate electron density on a grid given density matrix
    
    Args:
        mol: PySCF Mole object
        dm: Density matrix (2D or 3D array, based on AO basis)
    
    Returns:
        coords: Grid coordinates (N x 3 array)
        rho: Electron density at each point (array of length N)
        weights: Grid weights for integration
    """
    # Create a grid
    try:
        from pyscf import dft
    except:
        raise ImportError("pyscf is not installed")
    
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3  # Can be adjusted for accuracy vs. speed (1-9)
    grids.build()
    
    # Get grid coordinates and weights
    coords = grids.coords
    weights = grids.weights
    
    # Evaluate AO values on the grid
    ao_value = dft.numint.eval_ao(mol, coords)
    
    # Calculate density at each point
    # rho(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
    if len(dm.shape) == 3:
        # UHF case - dm is (dm_a, dm_b)
        dm_up = dm[0]
        dm_dn = dm[1]
        rho_up = np.einsum('pi,ij,pj->p', ao_value, dm_up, ao_value)
        rho_dn = np.einsum('pi,ij,pj->p', ao_value, dm_dn, ao_value)
        rho = rho_up + rho_dn
    else:
        raise ValueError("RHF case not implemented")
        # Once implemented, uncomment the following line
        # RHF case - dm is single matrix
        # rho = np.einsum('pi,ij,pj->p', ao_value, dm, ao_value)    
    print("Total number of electrons in Mean Field method (numerical integration):", 
        np.sum(rho * weights))  # Should be close to the total number of electrons        
    return rho, grids

def boson_gradient_generator(mf, wf, to_opt=None, nodal_cutoff=1e-3, **ewald_kwargs):
    mf_inputs = {}
    try:
        mf_inputs['dm'] = mf.make_rdm1()
    except:
        print("WARNING: mf.make_rdm1() is not available, cannot use DFT as Mean Field")

    rho, grids = calculate_mf_density(mf.mol, mf_inputs['dm'])

    mf_inputs.update({'xc':'LDA,VWN',
                 'mol':mf.mol,
                 'nelec': mf.nelec,
                 'mo_energy': mf.mo_energy,
                 'mo_occ': mf.mo_occ, 
                 'grids': grids, 
                 'rho' : rho })

    return PGradTransform(
        ABQMCEnergyAccumulator(mf_inputs, **ewald_kwargs),
        LinearTransform(wf.parameters, to_opt),
        nodal_cutoff=nodal_cutoff,
    )

class ABQMCEnergyAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __init__(self, mf_inputs, **kwargs):
        try:
            self.mol = mf_inputs['mol']
        except:
            import pdb; pdb.set_trace()
            
        self.mf_inputs = mf_inputs
        
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
        v_mf, ecorr, saved_results = bosonenergy.dft_energy(self.mf_inputs, configs)
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
            "corr": np.ones(ee.shape)*ecorr,
            "ei": ei, # For debugging, ei is not used in ABQMC
            "ii":np.ones(ee.shape)*ii,
            # Eq. 21-22 in doi: 10.1063/5.0155513 is the electronic energy
            # Therefore ii term is added here
            # V_MF = V_H + V_XC (only supports LDA for now)
            # E_Corr is the sum of KS eigenvalues 
            "total": ke + ee - (v_mf) + ecorr + ii,
        }
        if len(saved_results.keys()) > 0:
            energies.update(saved_results)
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
        grad_j = 0
        for e in range(nelec):
            epos = configs.electron(e)
            # grad_b_e = wf.gradient(e, epos) ## Jan 31, 2025
            log_grad_b_e = boson_wf.gradient(e, epos)
            log_grad_n = boson_wf.gradient_dets(e, epos)
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, log_grad_n-log_grad_b_e)
            grad_j = jastrow_wf.gradient(e, epos)

            # variant 1 use acceptance from VMC
            # delta += nconf /np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, grad_j, grad_psi_basis, acc[e])
            # variant 2 do not use acceptance from VMC
            delta += np.einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)
            # print('VMC', e, np.sum(grad_j), np.sum(grad_psi_n), np.sum(psi_n), np.sum(delta), delta[0,0,0],)

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
        
        nconf, nelec, nx = configs.configs.shape

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
        delta = 0
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
            jgrad          = jastrow_wf.gradient(e, epos_s)
            # print('DMC-J', e, np.sum(jgrad))
            
            # lap_psi_n: (eq. before eq. 23)
            # ∇²(Phi_n/Phi_B) = [∇²(Phi_n)*Phi_B - Phi_n*∇²(Phi_B)]/(Phi_B^2) 
            #                   - 2*∇(Phi_B)·∇(Phi_n/Phi_B)/Phi_B
            lap_psi_n  = np.einsum('cn, c->cn', lap_phi_n, 1./val_phi_b) # ∇²(Phi_n)/Phi_B
            lap_psi_n -= np.einsum('c, nc->cn', lap_phi_b, psi_n) # -∇²(Phi_B)/Phi_B * Psi_n or -∇²(Phi_B)/Phi_B^2 * Phi_n 
            lap_psi_n -= 2 * np.einsum('xc, nxc->cn', loggrad_b, grad_psi_n) # - 2*∇(log(Psi_B))*∇(Psi_n)

            delta1 = np.einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n) # ∇Psi_l \dot ∇Psi_n 
            delta2 = np.einsum('lc, cn->cln', psi_n, lap_psi_n) # Psi_l * ∇²Psi_n
            
            delta3 = np.einsum('lc, xc, nxc->cln', psi_n, loggrad_b + loggrad_psi_bt, grad_psi_n) # Psi_l * [∇(log(Phi_B)) + ∇(log(Psi_BT))] \dot ∇Psi_n        
            # delta4 = np.einsum('lc, xc, nxc->cln', psi_n, -2 * loggrad_b, grad_psi_n)
            # delta6 = np.einsum('lc, xc, nxc->cln', psi_n, -2 * loggrad_psi_bt, grad_psi_n)
            # delta5 = delta1 + delta2
            delta += delta1 + delta2 + delta3
            # print('DMC', e, np.sum(grad_psi_n), np.sum(psi_n), np.sum(delta1), np.sum(delta2), np.sum(delta3), np.sum(delta), delta[0,0,0])
            # print()
            # ndets = lap_phi_n.shape[1]
            # print(e,ndets)
            # for i in range(ndets):
            #     a = np.sum(delta5, axis=0)[i,i]
            #     b = np.sum(delta6, axis=0)[i,i]
            #     print(a, b, b/a)
            # import pdb; pdb.set_trace()
            matel += delta
            
        # exit()
        results = {'matel':matel, 
                   'delta': delta,
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




