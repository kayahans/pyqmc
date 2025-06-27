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

def calculate_radial_orbital_densities(mol, dm, mo_coeff):
    """
    Calculate radial orbital densities for each orbital in the mean field object
    """
    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    coords = grids.coords
    weights = grids.weights
    ao_value = dft.numint.eval_ao(mol, coords)
    mo_coeff = np.einsum('pi,ij,pj->p', ao_value, mo_coeff, ao_value)
    # Calculate densit  y at each point for molecular 
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
        # rho = np.einsum('pi,ij,pj->p', ao_value, dm, ao_value)    
    
    return rho, coords, weights

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
    def __init__(self, mf_inputs, **kwargs):
        self.en_acc = ABQMCEnergyAccumulator(mf_inputs, **kwargs)
        
    @timer_func
    def __call__(self, configs, wf, use_symm = False):
        
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
        # en_acc = self.en_acc(configs, wf)
        # # eb0 = en_acc['total'] - en_acc['corr']
        # mean_eb0 = np.mean(eb0, axis=0)*np.ones_like(eb0)

        # delta = np.einsum('lc, n, nc->cln', psi_n, np.diag(boson_wf.hmf), psi_n) 
        delta = 0
        # delta1 = 0
        # delta2 = 0
        for e in range(nelec):
            # Get position of electron e
            epos_s = configs.electron(e)

            # # 1. \Phi_l\Phi_n terms
            # lap_phi_n = boson_wf.laplacian_dets(e, epos_s)  # ∇²(Phi_n)/Phi_n
            # lap_phi_b = boson_wf.laplacian(e, epos_s)      # ∇²(Psi_B)/Psi_B
            # # delta1b = np.einsum('lc, c, nc->cln', psi_n, mean_eb0, psi_n) 
            # delta1c = np.einsum('lc, cn, nc->cln', psi_n, lap_phi_n, psi_n) 
            # delta1d = -np.einsum('lc, c, nc->cln', psi_n, lap_phi_b, psi_n) 
            # # delta1 = delta1b.copy()
            # delta1 += delta1c + delta1d

            loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
            loggrad_b = boson_wf.gradient(e, epos_s) # ∇log(Psi_B) eq. 4
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n - loggrad_b)  

            grad_j = jastrow_wf.gradient(e, epos_s)
            delta += np.einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)
            
        # delta += delta1 + delta2

        # delta = 0
        # grad_j = 0
        # for e in range(nelec):
        #     epos = configs.electron(e)
        #     # grad_b_e = wf.gradient(e, epos) ## Jan 31, 2025
        #     loggrad_phi_n = boson_wf.gradient_dets(e, epos) 
        #     loggrad_b = boson_wf.gradient(e, epos) # ∇log(Psi_B) eq. 4
        #     grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n - loggrad_b)  

        #     grad_j = jastrow_wf.gradient(e, epos)
        #     delta += np.einsum('lc, n, nc->cln', psi_n, np.diag(boson_wf.hmf), psi_n) 
        #     # variant 1 use acceptance from VMC
        #     # delta += nconf /np.sum(acc[e]) * np.einsum("nc,xc,lxc, c ->cnl", psi_basis, grad_j, grad_psi_basis, acc[e])
        #     # variant 2 do not use acceptance from VMC
        #     delta += np.einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)
        #     # print('VMC', e, np.sum(grad_j), np.sum(grad_psi_n), np.sum(psi_n), np.sum(delta), delta[0,0,0],)

        # Disable this for now, but post-process
        # if use_symm:
        #     symm_mask = boson_wf._det_prod_filter
        #     # If symm mask is True, then keep the calculated values, otherwise set to zero
        #     delta = np.einsum('cln, ln->cln', delta, symm_mask)
        #     ovlp_ij = np.einsum('cln, ln->cln', ovlp_ij, symm_mask)
            
        results = {'delta':delta, 
                #    'delta1': delta1,
                #    'delta2': delta2,
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
    """Accumulator for computing matrix elements in Auxiliary-field Boson Corrected Diffusion Monte Carlo (ABCDMC).
    
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
    def __init__(self, mf_inputs, **kwargs):
        self.en_acc = ABQMCEnergyAccumulator(mf_inputs, **kwargs)
        
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
        en_acc = self.en_acc(configs, wf)
        # import pdb; pdb.set_trace()
        # eb0 = en_acc['total'] - en_acc['corr']
        # mean_eb0 = np.mean(eb0)*np.ones_like(eb0)
        
        # delta1_hmf = np.einsum('lc, n, nc->cln', psi_n, np.diag(boson_wf.hmf), psi_n) 
        # delta1_eb0 = np.einsum('lc, c, nc->cln', psi_n, mean_eb0, psi_n) 
        # delta = delta1_hmf.copy()
        # delta += delta1_eb0
        delta = 0 
        delta1 = 0
        delta2 = 0
        delta3 = 0

        for e in range(nelec):
            # Get position of electron e
            epos_s = configs.electron(e)

            # 1. \Phi_l\Phi_n terms
            lap_phi_n = boson_wf.laplacian_dets(e, epos_s).copy()  # ∇²(Phi_n)/Phi_n
            lap_phi_b = boson_wf.laplacian(e, epos_s).copy()      # ∇²(Psi_B)/Psi_B
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            
            # delta1b_e = np.einsum('lc, c, nc->cln', psi_n, mean_eb0, psi_n) 
            delta1c_e = np.einsum('lc, cn, nc->cln', psi_n, lap_phi_n, psi_n) 
            delta1d_e = -np.einsum('lc, c, nc->cln', psi_n, lap_phi_b, psi_n) 
            # delta1 = delta1b_e + delta1c_e + delta1d_e
            delta1 += delta1c_e + delta1d_e

            # Debug prints
            # assert np.allclose(delta1, delta1a + delta1b + delta1c + delta1d)

            loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
            loggrad_b = boson_wf.gradient(e, epos_s) # ∇log(Psi_B) eq. 4
            grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n - loggrad_b)  
            # 2. \Phi_l∇\Phi_B terms
            
            # loggrad_psi_bt = wf.gradient(e, epos_s) # ∇log(Psi_BT) eq. 4
            # delta2 += np.einsum('lc, xc, nxc->cln', psi_n, -loggrad_b + loggrad_psi_bt, grad_psi_n) # Psi_l * [∇(log(Phi_B)) + ∇(log(Psi_BT))] \dot ∇Psi_n        
            delta2 += np.einsum('lc, xc, nxc->cln', psi_n, jastrow_wf.gradient(e, epos_s), grad_psi_n) 

            # 3. ∇\Phi_l∇\Phi_n terms (No terms)
            delta3 += np.einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n) # Psi_l * [∇(log(Phi_B)) + ∇(log(Psi_BT))] \dot ∇Psi_n        
            
        # import matplotlib.pyplot as plt
        # plt.plot(np.diag(np.mean(delta1a, axis=0)), '-o', label='1a')
        # plt.plot(np.diag(np.mean(delta1b, axis=0)), '-o', label='1b')
        # plt.plot(np.diag(np.mean(delta1c, axis=0)), '-o', label='1c')
        # plt.plot(np.diag(np.mean(delta1d, axis=0)), '-o', label='1d')
        # plt.plot(np.diag(np.mean(delta1c+delta1d, axis=0)), '-o', label='1c+1d')
        # plt.plot(np.diag(np.mean(delta, axis=0)), '-o', label='total')
        # plt.legend()
        # plt.show()
        delta = delta1 + delta2 + delta3
        results = {'delta': delta,
                   'delta1': delta1,
                   'delta2': delta2,
                   'delta3': delta3,
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

class ABCDMCMatrixAccumulator_old:
    """Accumulator for computing matrix elements in Auxiliary-field Boson Corrected Diffusion Monte Carlo (ABCDMC).
    
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
            # matel += delta
            
        # exit()
        results = { #'matel':matel, 
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

# class ABDMCMatrixAccumulator:
#     """Accumulator for computing matrix elements in Auxiliary Boson Diffusion Monte Carlo.
    
#     Based on eq. 18. 
    
#     Note: Currently missing the EB-VB term in the matrix element calculation.
    
    
#     Methods
#     -------
#     __call__(configs, wf)
#         Compute matrix elements for given configurations and wavefunction.
        
#         Parameters
#         ----------
#         configs : object
#             Contains electron configurations with shape (nconf, nelec, ndim)
#         wf : object
#             Wavefunction object containing BosonWF component
            
#         Returns
#         -------
#         dict
#             'matel': Matrix elements including kinetic terms
#                     Shape: (nconf, ndet, ndet)
#             'ovlp': Overlap matrices between basis states
#                    Shape: (nconf, ndet, ndet)
    
#     """

#     @timer_func
#     def __call__(self, configs, wf):

#         nconf, nelec, _ = configs.configs.shape

#         wave_functions = wf.wf_factors
#         for wave in wave_functions:
#             if isinstance(wave, bosonslater.BosonWF):
#                 boson_wf = wave
            
#         psi_n = get_psi_basis(boson_wf) # Phi_l/Phi_B
#         ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n)

#         # Matrix element by integration parts on the ∇f_B term of the eq. 23 
#         # For integration by parts see eq. 17 in the reference paper
#         matel = 0
        
#         for e in range(nelec):
#             # Get position of electron e
#             epos_s = configs.electron(e)

            
#             lap_phi_n = boson_wf.laplacian_dets(e, epos_s) # ∇²Phi_n/Phi_n
#             # ∇log(Phi_n) 
#             loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
#             # ∇log(Psi_B) eq. 4
#             loggrad_b = boson_wf.gradient(e, epos_s) 
#             # ∇Psi_n = ∇(Phi_n/Phi_B)
#             grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b)  
#             # ∇log(Psi_B^T)
#             loggrad_psi_bt = wf.gradient(e, epos_s)

#             matel += 1./2 * np.einsum('lc, cn->cln', psi_n, lap_phi_n) # Psi_l * ∇²Psi_n
#             matel += 0 # Missing EB-VB term 
#             matel -= np.einsum('lc, nxc, nxc->cln', psi_n, grad_psi_n-loggrad_psi_bt, grad_psi_n)  # Check indices

#         results = {'matel':matel, 
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

# class ABDMCMatrixAccumulator_old:
#     """Accumulator for computing matrix elements in Auxiliary Boson Diffusion Monte Carlo.
    
#     Based on eq. 18. 
    
#     Note: Currently missing the EB-VB term in the matrix element calculation.
    
    
#     Methods
#     -------
#     __call__(configs, wf)
#         Compute matrix elements for given configurations and wavefunction.
        
#         Parameters
#         ----------
#         configs : object
#             Contains electron configurations with shape (nconf, nelec, ndim)
#         wf : object
#             Wavefunction object containing BosonWF component
            
#         Returns
#         -------
#         dict
#             'matel': Matrix elements including kinetic terms
#                     Shape: (nconf, ndet, ndet)
#             'ovlp': Overlap matrices between basis states
#                    Shape: (nconf, ndet, ndet)
    
#     """

#     @timer_func
#     def __call__(self, configs, wf):

#         nconf, nelec, _ = configs.configs.shape

#         wave_functions = wf.wf_factors
#         for wave in wave_functions:
#             if isinstance(wave, bosonslater.BosonWF):
#                 boson_wf = wave
            
#         psi_n = get_psi_basis(boson_wf) # Phi_l/Phi_B
#         ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n)

#         # Matrix element by integration parts on the ∇f_B term of the eq. 23 
#         # For integration by parts see eq. 17 in the reference paper
#         matel = 0
        
#         for e in range(nelec):
#             # Get position of electron e
#             epos_s = configs.electron(e)

#             # ∇²Phi_n
#             lap_phi_n = boson_wf.laplacian_dets(e, epos_s) 
#             # ∇log(Phi_n) 
#             loggrad_phi_n = boson_wf.gradient_dets(e, epos_s) 
#             # ∇log(Psi_B) eq. 4
#             loggrad_b = boson_wf.gradient(e, epos_s) 
#             # ∇Psi_n = ∇(Phi_n/Phi_B)
#             grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b)  
#             # ∇log(Psi_B^T)
#             loggrad_psi_bt = wf.gradient(e, epos_s)

#             matel += 1./2 * np.einsum('lc, cn->cln', psi_n, lap_phi_n) # Psi_l * ∇²Psi_n
#             matel += 0 # Missing EB-VB term 
#             matel -= np.einsum('lc, nxc, nxc->cln', psi_n, grad_psi_n-loggrad_psi_bt, grad_psi_n)  # Check indices

#         results = {'matel':matel, 
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



class DensityAccumulator:
    """Accumulates electron density in bins.
    
    This accumulator computes the electron density by binning electron positions
    into a grid. The density is normalized by the bin volume and number of electrons.
    
    Args:
        bins (tuple): Number of bins in each dimension (nx, ny, nz)
        range (tuple): Range of coordinates to bin in each dimension ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    """
    
    def __init__(self, bins=(50, 50, 50), range=None):
        self.bins = bins
        self.range = range
        
    def __call__(self, configs, wf):
        """Compute density for each configuration.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'density' array of shape (nconf, *bins)
        """
        nconf, nelec, _ = configs.configs.shape
        
        # Reshape configs to (nconf*nelec, 3) for histogram
        positions = configs.configs.reshape(-1, 3)
        
        # Compute histogram for each configuration
        density = np.zeros((nconf, *self.bins))
        for i in range(nconf):
            start = i * nelec
            end = (i + 1) * nelec
            hist, _ = np.histogramdd(
                positions[start:end],
                bins=self.bins,
                range=self.range,
                density=True
            )
            density[i] = hist
            
        return {"density": density}
    
    def avg(self, configs, wf):
        """Compute average density across configurations.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'density' array of shape (*bins)
        """
        results = self(configs, wf)
        return {k: np.mean(v, axis=0) for k, v in results.items()}
    
    def var(self, configs, wf):
        """Compute variance of density across configurations.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'density' array of shape (*bins)
        """
        results = self(configs, wf)
        return {k: np.var(v, axis=0) for k, v in results.items()}
    
    def keys(self):
        """Return set of keys in the accumulator results."""
        return set(["density"])
    
    def shapes(self):
        """Return shapes of arrays in the accumulator results."""
        return {"density": self.bins}

class RadialDensityAccumulator:
    """Accumulates radial electron density.
    
    This accumulator computes the radial electron density by binning electron
    distances from the origin into radial shells. The density is normalized by
    the shell volume and number of electrons.
    
    Args:
        nbins (int): Number of radial bins
        rmax (float): Maximum radius to consider
        rmin (float): Minimum radius to consider (default: 0)
        center (array-like): Center point for radial distance calculation (default: origin)
    """
    
    def __init__(self, nbins=400, rmax=40.0, rmin=0.0, center=None):
        self.nbins = nbins
        self.rmax = rmax
        self.rmin = rmin
        self.center = np.zeros(3) if center is None else np.array(center)
        
        # Pre-compute bin edges and volumes
        self.bin_edges = np.linspace(rmin, rmax, nbins + 1)
        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        
        # Volume of each spherical shell
        self.shell_volumes = 4/3 * np.pi * (self.bin_edges[1:]**3 - self.bin_edges[:-1]**3)
        
    def __call__(self, configs, wf):
        """Compute radial density for each configuration.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'radial_density' array of shape (nconf, nbins)
        """
        nconf, nelec, _ = configs.configs.shape
        
        # Reshape configs to (nconf*nelec, 3) for distance calculation
        positions = configs.configs.reshape(-1, 3)
        
        # Calculate distances from center
        distances = np.sqrt(np.sum((positions - self.center)**2, axis=1))
        
        # Compute histogram for each configuration
        hist, _ = np.histogram(
            distances,
            bins=self.bin_edges,
            density=True
        )
        results = {'r': self.bin_centers,
                   'int_density': hist, 
                   'radial_density': hist / self.shell_volumes}
        return results
    
    def avg(self, configs, wf):
        """Compute average radial density across configurations.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'radial_density' array of shape (nbins,)
        """
        results = self(configs, wf)
        # return {k: np.mean(v, axis=0) for k, v in results.items()}
        return {k: v for k, v in results.items()}
    
    def var(self, configs, wf):
        """Compute variance of radial density across configurations.
        
        Args:
            configs: Electron configurations
            wf: Wave function object
            
        Returns:
            dict: Contains 'radial_density' array of shape (nbins,)
        """
        results = self(configs, wf)
        return {k: np.var(v, axis=0) for k, v in results.items()}
    
    def keys(self):
        """Return set of keys in the accumulator results."""
        return set(["radial_density"])
    
    def shapes(self):
        """Return shapes of arrays in the accumulator results."""
        return {"radial_density": (self.nbins,)}
    
    def get_radial_points(self):
        """Return the radial points (bin centers) for plotting.
        
        Returns:
            array: Radial points where density is evaluated
        """
        return self.bin_centers

def test_addition_difference():
    """Test function to demonstrate the difference between in-place and new array addition."""
    # Create some test arrays
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.array([[9.0, 10.0], [11.0, 12.0]])
    
    # Method 1: In-place addition
    result1 = a.copy()
    result1 += b
    result1 += c
    
    # Method 2: New array addition
    result2 = a + b + c
    
    print("Original arrays:")
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    print("\nResult from in-place addition (result1):")
    print(result1)
    print("\nResult from new array addition (result2):")
    print(result2)
    print("\nAre they equal?", np.array_equal(result1, result2))
    print("Max difference:", np.max(np.abs(result1 - result2)))
    print("Mean difference:", np.mean(np.abs(result1 - result2)))

# Add this line to run the test
if __name__ == "__main__":
    test_addition_difference()




