#!/usr/bin/env python3
"""
Run DFT calculation for Li atom with diffuse basis set and save results.

Based on test.ipynb examples, uses S=4, P=8, D=11 diffuse functions.
Saves: xc name, mol parameters, nelec, mo_energy, mo_occ, grids, and rho (charge density).
"""

import numpy as np
import json
from pyscf import gto, scf, dft, symm
from scipy.linalg import eigh
import pickle
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
from scipy.special import comb 

# ===========================================================================
# Generate Diffuse Basis Functions
# ===========================================================================

# ===========================================================================
# Main Calculation
# ===========================================================================

def infer_ecp(basis_set, ecp=None):
    """Return ECP name from config or infer from a -PP basis set name."""
    if ecp:
        return ecp
    if basis_set.endswith('-PP'):
        return basis_set[4:] if basis_set.startswith('aug-') else basis_set
    return None

def read_config_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    ATOM = config['atom']
    BASES = config['basis_sets']
    SPIN = config['spin']
    XC_FUNCTIONAL = config.get('xc_functional', 'LDA, VWN')
    NCAS = config['ncas']
    NELECAS = config['nelecas'] 
    CHARGE = config.get('charge', 0)
    RATIO = config.get('ratio', 2)
    N_FUNCS = config['n_funcs']
    SYMM_RKS = config.get('symm_rks', False)
    ORB_SYMM = config.get('orb_symm', False)
    ECP = config.get('ecp')
    return ATOM, BASES, XC_FUNCTIONAL, SPIN, NCAS, NELECAS, CHARGE, RATIO, N_FUNCS, SYMM_RKS, ORB_SYMM, ECP

def generate_diffuse_basis(atom, base_basis, ratio=2.5, n_funcs=4, angular_momenta=None, verbose=False):
    """
    Generate diffuse basis functions by extending a base basis set.
    """
    if isinstance(base_basis, str):
        base = gto.basis.load(base_basis, atom)
    else:
        base = base_basis
    
    ang_num_to_letter = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I'}
    ang_letter_to_num = {v: k for k, v in ang_num_to_letter.items()}
    
    # Find minimum exponent for each angular momentum
    min_exp = {}
    for shell in base:
        ang_num = shell[0]
        ang_letter = ang_num_to_letter.get(ang_num, str(ang_num))
        for prim in shell[1:]:
            if isinstance(prim[0], (int, float)):
                exp = float(prim[0])
                if ang_letter not in min_exp or exp < min_exp[ang_letter]:
                    min_exp[ang_letter] = exp
    
    # Handle n_funcs as int or dict
    if isinstance(n_funcs, dict):
        n_funcs_dict = n_funcs
        if angular_momenta is None:
            angular_momenta = sorted(n_funcs_dict.keys(), key=lambda x: ang_letter_to_num.get(x, 99))
    else:
        if angular_momenta is None:
            angular_momenta = sorted(min_exp.keys(), key=lambda x: ang_letter_to_num.get(x, 99))
        n_funcs_dict = {ang: n_funcs for ang in angular_momenta}
    
    diffuse_basis = []
    
    if verbose:
        print("Generating diffuse basis functions:")
    
    for ang in angular_momenta:
        if ang not in min_exp:
            if verbose:
                print(f"  Warning: {ang} channel not in base basis, skipping")
            continue
        
        n_ang = n_funcs_dict.get(ang, 0)
        if n_ang == 0:
            continue
            
        ang_num = ang_letter_to_num[ang]
        alpha_0 = min_exp[ang] / ratio
        
        if verbose:
            print(f"  {ang} channel: {n_ang} functions")
        
        for i in range(n_ang):
            alpha = alpha_0 / (ratio ** i)
            diffuse_basis.append([ang_num, [alpha, 1.0]])
            
            if verbose:
                r_rms = np.sqrt(3 / (4 * alpha))
                print(f"    α = {alpha:.7f}, <r²>^½ = {r_rms:.1f} Bohr")
    
    return diffuse_basis

def compute_orbital_r2_all(mol, mo_coeff):
    """
    Compute <r²> for all MOs.
    
    Parameters
    ----------
    mol : gto.Mole
        Molecule object
    mo_coeff : array
        MO coefficients (nao, nmo)
        
    Returns
    -------
    r2_array : array
        Array of <r²> values for each orbital
    """
    r2_ao = mol.intor('int1e_r2')
    # Transform all MOs at once: C^T S C for each column
    r2_array = np.diag(mo_coeff.T @ r2_ao @ mo_coeff)
    return r2_array

def filter_mos_with_degeneracy(mol, mo_energy, mo_occ, mo_coeff, mo_energy_tol = 1e-6, exclude_orbitals = None, max_n = None):
    """
    Filter MO energies and occupations to only include the valence orbitals.
    """
    orbital_labels = []
    orbital_groups = defaultdict(list)

    if len(mo_energy) == 2:
        is_uks = True
        mo_energy_alpha = mo_energy[0]
        mo_occ_alpha = mo_occ[0]
        mo_energy_beta = mo_energy[1]
        mo_occ_beta = mo_occ[1]
        r2_dz_alpha = compute_orbital_r2_all(mol, mo_coeff[0])
        r_rms_dz_alpha = np.sqrt(np.maximum(0, r2_dz_alpha))
        r2_dz_beta = compute_orbital_r2_all(mol, mo_coeff[1])
        r_rms_dz_beta = np.sqrt(np.maximum(0, r2_dz_beta))
        alpha_indices = []
        beta_indices = []
    else:
        is_uks = False
        mo_energy_alpha = mo_energy
        mo_occ_alpha = mo_occ
        r2_dz_alpha = compute_orbital_r2_all(mol, mo_coeff)
        r_rms_dz_alpha = np.sqrt(np.maximum(0, r2_dz_alpha))
        alpha_indices = []
    
    def find_degenerate_group(energies, start_idx, tol):
        """Find all orbitals degenerate with the one at start_idx."""
        group = [start_idx]
        ref_energy = energies[start_idx]
        for j in range(start_idx + 1, len(energies)):
            if abs(energies[j] - ref_energy) < tol:
                group.append(j)
            else:
                break
        return group
    
    def get_orbital_type(deg_size):
        """Map degeneracy size to orbital type."""
        type_map = {1: 's', 3: 'p', 5: 'd', 7: 'f', 9: 'g'}
        return type_map.get(deg_size, 'unknown')
    
    def process_orbitals(energies, spin_name, label_counter, labeled, label_dict, exclude_orbitals = None, max_n = None):
        """Process orbitals for a given spin (alpha or beta)."""
        indices = []
        n_orbs = len(energies)
        i = 0
        
        while i < n_orbs:
            if labeled[i]:
                i += 1
                continue
                
            # Find degenerate group
            degenerate = find_degenerate_group(energies, i, mo_energy_tol)
            deg_size = len(degenerate)
            
            # Determine orbital type
            orb_type = get_orbital_type(deg_size)
            if orb_type == 'unknown':
                print(f"Unknown orbital type for {spin_name}: degeneracy={deg_size}, indices={degenerate}")
                orb_type = 's'
                #exit()
            
            # Get label
            n = label_counter[orb_type]
            label = f"{n}{orb_type}"
            label_counter[orb_type] += 1
            
            # Process all orbitals in this degenerate group
            for idx in degenerate:
                if orb_type not in exclude_orbitals and n <= max_n:
                    indices.append(idx)
                label_dict[idx] = label
                orbital_labels.append(label)
                orbital_groups[label].append(idx)
                labeled[idx] = True
            
            # Move to next unlabeled orbital
            i = degenerate[-1] + 1
        print(f'Selected orbitals: {len(indices)} out of {n_orbs}')    
        print(f'Orbitals_indices: {indices}')
        return indices
    
    # Initialize tracking arrays
    n_orbs = len(mo_energy_alpha)
    labeled_alpha = np.zeros(n_orbs, dtype=bool)
    label_counter_alpha = {'s': 1, 'p': 2, 'd': 3, 'f': 4, 'g': 5}
    label_dict_alpha = {}
    
    if is_uks:
        labeled_beta = np.zeros(n_orbs, dtype=bool)
        label_counter_beta = {'s': 1, 'p': 2, 'd': 3, 'f': 4, 'g': 5}
        label_dict_beta = {}
    
    # Process alpha orbitals
    alpha_indices = process_orbitals(
        mo_energy_alpha, 
        'α', label_counter_alpha, labeled_alpha, label_dict_alpha, exclude_orbitals = exclude_orbitals, max_n = max_n
    )
    
    # Process beta orbitals separately (if UKS)
    if is_uks:
        beta_indices = process_orbitals(
            mo_energy_beta, 
            'β', label_counter_beta, labeled_beta, label_dict_beta, exclude_orbitals = exclude_orbitals, max_n = max_n
        )
    
    # Print results
    print("\nLabeling orbitals:")
    print(f"{'Spin':>6} {'Indicator':>8} {'Idx':>4} {'Energy':>12} {'<r²>^½':>10} {'Occ':>6} {'Label':>8}")
    print("-" * 60)
    
    # Print alpha orbitals
    for idx in range(n_orbs):
        label = label_dict_alpha.get(idx, 'N/A')
        if idx not in alpha_indices:
            indicator = 'excluded'
            tab = '    '
        else:
            indicator = 'selected'
            tab = ''
        print(f"{tab}{'α':>6} {indicator:>8} {idx:>4} {mo_energy_alpha[idx]:>12.6f} {r_rms_dz_alpha[idx]:>10.2f} "
              f"{mo_occ_alpha[idx]:>6.2f} {label:>8}")
    
    # Print beta orbitals (if UKS)
    if is_uks:
        print("\nBeta orbitals:")
        for idx in range(n_orbs):
            if idx not in beta_indices:
                indicator = 'excluded'
                tab = '    '
            else:
                indicator = 'selected'
                tab = ''
            label = label_dict_beta.get(idx, 'N/A')
            print(f"{tab}{'β':>6} {indicator:>8} {idx:>4} {mo_energy_beta[idx]:>12.6f} {r_rms_dz_beta[idx]:>10.2f} "
                  f"{mo_occ_beta[idx]:>6.2f} {label:>8}")
        indices = (alpha_indices, beta_indices)
    else:
        indices = np.array(alpha_indices)
    return indices

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
    grids.level = 5  # Can be adjusted for accuracy vs. speed (1-9)
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

def calculate_ci_shape(ncas, nelecas):
    """
    Calculate the shape of the CI vector for a CAS space.
    
    Parameters
    ----------
    ncas : int
        Number of active orbitals
    nelecas : tuple or int
        Number of active electrons. If tuple (na, nb), unrestricted.
        If int, restricted (total electrons).
        
    Returns
    -------
    ndet : int
        Number of determinants in the CI space
    """
    if isinstance(nelecas, (list, tuple)):
        # Unrestricted: (na, nb)
        na, nb = nelecas
        # Number of ways to place na electrons in ncas orbitals (alpha)
        ndet_alpha = int(comb(ncas, na, exact=True))
        # Number of ways to place nb electrons in ncas orbitals (beta)
        ndet_beta = int(comb(ncas, nb, exact=True))
        ndet = ndet_alpha * ndet_beta
    else:
        # Restricted: total electrons
        ndet = int(comb(ncas, nelecas, exact=True))
    
    return ndet

def construct_cas(mo_energy, 
                  mo_occ, 
                  mo_coeff, 
                  indices, 
                  ncas, 
                  nelecas, 
                  nelec,
                  ncore,
                  mf = None,
                  scf_chkfile = None,
                  ci_chkfile = None):
    is_uks = False
    ncas_is_larger = False
    ncas_is_smaller = False
    if isinstance(indices, (list, tuple)) and len(indices) == 2:
        is_uks = True
        # Convert to numpy arrays if needed
        alpha_idx = np.asarray(indices[0])
        beta_idx = np.asarray(indices[1])
        
        # For UKS: mo_coeff is a list [mo_coeff_alpha, mo_coeff_beta]
        # Each is shape (nao, nmo), so we need to select columns
        mo_coeff_alpha = mo_coeff[0][:, alpha_idx].copy()
        mo_coeff_beta = mo_coeff[1][:, beta_idx].copy()
        mo_occ_alpha = mo_occ[0][alpha_idx].copy()
        mo_occ_beta = mo_occ[1][beta_idx].copy()
        mo_energy_alpha = mo_energy[0][alpha_idx].copy()
        mo_energy_beta = mo_energy[1][beta_idx].copy()
        
        # Validate ncas matches number of selected orbitals
        n_alpha = len(alpha_idx)
        n_beta = len(beta_idx)
        if 'ncore' not in locals():
            ncore = nelec[0] - nelecas[0]
        
        if n_alpha < ncas or n_beta < ncas:
            # If selected orbitals are fewer than ncas, adjust ncas
            print(f"Warning: ncas={ncas} but selected {n_alpha} alpha and {n_beta} beta orbitals")
            ncas_is_larger = True
            # Set ncas to the minimum of (n_alpha - ncore, n_beta - ncore)
            ncas_alpha = n_alpha - ncore
            ncas_beta = n_beta - ncore
            ncas = min(ncas_alpha, ncas_beta)
            print(f"Adjusting ncas to {ncas} (min of alpha: {ncas_alpha}, beta: {ncas_beta})")
        elif n_alpha > ncas or n_beta > ncas:
            # If selected orbitals are more than ncas, take only first ncas orbitals
            print(f"Warning: ncas={ncas} but selected {n_alpha} alpha and {n_beta} beta orbitals")
            ncas_is_smaller = True
            # Trim to first ncas orbitals
            nmo = ncas + ncore
            mo_coeff_alpha = mo_coeff_alpha[:, :nmo]
            mo_coeff_beta = mo_coeff_beta[:, :nmo]
            mo_occ_alpha = mo_occ_alpha[:nmo]
            mo_occ_beta = mo_occ_beta[:nmo]
            mo_energy_alpha = mo_energy_alpha[:nmo]
            mo_energy_beta = mo_energy_beta[:nmo]
            print(f"Trimming to first {nmo} orbitals")

    else:
        # Convert to numpy array if needed
        idx = np.asarray(indices)
        
        # For RKS: mo_coeff is a 2D array (nao, nmo), select columns
        mo_coeff_alpha = mo_coeff[:, idx].copy()
        mo_occ_alpha = mo_occ[idx].copy()
        mo_energy_alpha = mo_energy[idx].copy()
        
        # Validate ncas matches number of selected orbitals
        n_selected = len(idx)
        if 'ncore' not in locals():
            ncore = nelec - nelecas
        if n_selected < ncas:
            # If selected orbitals are fewer than ncas, adjust ncas
            print(f"Warning: ncas={ncas} but selected {n_selected} orbitals")
            ncas_is_larger = True
            ncas = n_selected - ncore
            print(f"Adjusting ncas to {ncas} (n_selected: {n_selected} - ncore: {ncore})")
        elif n_selected > ncas:
            # If selected orbitals are more than ncas, take only first ncas orbitals
            print(f"Warning: ncas={ncas} but selected {n_selected} orbitals")
            ncas_is_smaller = True
            nmo = ncas + ncore
            # Trim to first ncas orbitals
            mo_coeff_alpha = mo_coeff_alpha[:, :nmo]
            mo_occ_alpha = mo_occ_alpha[:nmo]
            mo_energy_alpha = mo_energy_alpha[:nmo]
            print(f"Trimming to first {nmo} orbitals")
    
    mf_trimmed = mf.copy()
    if is_uks: 
        mf_trimmed.mo_coeff = (mo_coeff_alpha, mo_coeff_beta)
        mf_trimmed.mo_occ = np.stack([mo_occ_alpha, mo_occ_beta], axis=0)
        mf_trimmed.mo_energy = np.stack([mo_energy_alpha, mo_energy_beta], axis=0)
    else:
        # For RKS, mo_coeff is a 2D array
        mf_trimmed.mo_coeff = mo_coeff_alpha
        mf_trimmed.mo_occ = mo_occ_alpha
        mf_trimmed.mo_energy = mo_energy_alpha
    
    # Assert that mf_trimmed values are not the same as mf
    def _arrays_differ(a, b):
        # Compare numpy arrays or tuples of arrays
        try:
            import numpy as np
            if isinstance(a, tuple) and isinstance(b, tuple):
                return any(_arrays_differ(ax, bx) for ax, bx in zip(a, b))
            return not np.allclose(a, b)
        except:
            return True

    # assert _arrays_differ(mf_trimmed.mo_coeff, mf.mo_coeff), "mf_trimmed.mo_coeff is identical to mf.mo_coeff"
    # assert _arrays_differ(mf_trimmed.mo_occ, mf.mo_occ), "mf_trimmed.mo_occ is identical to mf.mo_occ"
    # assert _arrays_differ(mf_trimmed.mo_energy, mf.mo_energy), "mf_trimmed.mo_energy is identical to mf.mo_energy"

    if ncas_is_larger or ncas_is_smaller:
        print(f"Final ncas: {ncas}")
    
    ndet = calculate_ci_shape(ncas, nelecas)
    print(f"Number of determinants: {ndet}")
    norm_factor = 1.0 / np.sqrt(ndet)
    ci_vector = np.full(ndet, norm_factor, dtype=np.float64)

    dm_trimmed = mf_trimmed.make_rdm1()
    mol_trimmed = mf_trimmed.mol
    rho, grids = calculate_mf_density(mol_trimmed, dm_trimmed)
    
    mf_trimmed.xc = mf_trimmed.xc.upper()
    if mf_trimmed.xc == 'LDA,VWN':
        deriv = 0
    elif mf_trimmed.xc == 'PBE,PBE':
        deriv = 1
    else:
        raise ValueError(f"Unsupported XC functional: {mf_trimmed.xc}")
    
    mf_inputs = {
        'xc': mf_trimmed.xc,
        'deriv': deriv,
        'nelec': mf_trimmed.nelec,
        'mo_energy': mf_trimmed.mo_energy,
        'mo_occ': mf_trimmed.mo_occ,
        'mo_coeff': mf_trimmed.mo_coeff,
        'dm': dm_trimmed,
        'rho': rho,
        'grids': grids,
    }
    if scf_chkfile is not None:    
        with h5py.File(scf_chkfile, 'w') as f:
            f['mol'] = mol_trimmed.dumps()
            scf_grp = f.create_group('scf')
            
            # Store SCF data (matching li.hdf5 structure)
            scf_grp.create_dataset('e_tot', data=mf_trimmed.e_tot)
            scf_grp.create_dataset('mo_coeff', data=mf_trimmed.mo_coeff)
            scf_grp.create_dataset('mo_energy', data=mf_trimmed.mo_energy)
            scf_grp.create_dataset('mo_occ', data=mf_trimmed.mo_occ)    
            mf_inputs_grp = f.create_group("mf_inputs")
            for key, value in mf_inputs.items():
                if hasattr(value, "dtype") and hasattr(value, "shape"):
                    mf_inputs_grp.create_dataset(key, data=value)
                elif isinstance(value, (str, int, float)):
                    mf_inputs_grp.attrs[key] = value
                elif value is None:
                    mf_inputs_grp.attrs[key] = "None"
                else:
                    # For objects like molecule, store as string or repr
                    try:
                        mf_inputs_grp.attrs[key] = str(value)
                    except Exception:
                        mf_inputs_grp.attrs[key] = repr(value)

    if ci_chkfile is not None:
        from pyscf import mcscf
        mc = mcscf.casci.CASCI(mf_trimmed, ncas, nelecas)
        assert (ncore == mc.ncore), "ncore mismatch"
        
        with h5py.File(ci_chkfile, "a") as f:
            f.create_group("ci")
            f["ci/ncas"] = ncas
            f["ci/ncore"] = ncore
            f["ci/nelecas"] = list(nelecas) if isinstance(nelecas, (list, tuple)) else [nelecas]
            f["ci/fci"] = ci_vector
            f["ci/ci"] = ci_vector
            f["ci/mo_coeff"] = mf_trimmed.mo_coeff
            f["ci/mf_mo_energy"] = mf_trimmed.mo_energy
            f["ci/mo_occ"] = mf_trimmed.mo_occ
            print("Available output from CI file:", f["ci"].keys())
            print(f"CI vector shape: {ci_vector.shape}")
            print(f"CI vector norm: {np.linalg.norm(ci_vector):.6f}")
    
    print("CI file created:", ci_chkfile)
    return ci_chkfile

def write_scf_to_h5(mf, scf_file):
    with h5py.File(scf_file, 'w') as f:
        f['mol'] = mf.mol.dumps()
        scf_grp = f.create_group('scf')
        scf_grp.create_dataset('e_tot', data=mf.e_tot)
        scf_grp.create_dataset('mo_coeff', data=mf.mo_coeff)
        scf_grp.create_dataset('mo_energy', data=mf.mo_energy)
        scf_grp.create_dataset('mo_occ', data=mf.mo_occ)
    return scf_file

def print_mo_energies(mo_energies_by_basis):
    BASES = list(mo_energies_by_basis.keys())
# Print MO energies table at the end
    print("\n" + "=" * 70)
    print("MO Energies Comparison")
    print("=" * 70)
    
    # Determine if we have UKS (alpha and beta) or RKS
    is_uks = any('beta' in energies for energies in mo_energies_by_basis.values())
    if is_uks:
        # Print header for UKS
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set + ' (α)':>20} {basis_set + ' (β)':>20}"
        print(header)
        print("-" * (6 + 40 * len(BASES)))
        
        # Print each MO energy
        # n_mos = min(20, max(len(e['alpha']) for e in mo_energies_by_basis.values()))
        n_mos = max(len(e['alpha']) for e in mo_energies_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                energies = mo_energies_by_basis.get(basis_set, {})
                alpha_energy = energies.get('alpha', [0])[i] if i < len(energies.get('alpha', [])) else 'N/A'
                beta_energy = energies.get('beta', [0])[i] if i < len(energies.get('beta', [])) else 'N/A'
                if isinstance(alpha_energy, (int, float)):
                    row += f"{alpha_energy:>20.6f}"
                else:
                    row += f"{str(alpha_energy):>20}"
                if isinstance(beta_energy, (int, float)):
                    row += f"{beta_energy:>20.6f}"
                else:
                    row += f"{str(beta_energy):>20}"
            print(row)
    else:
        # Print header for RKS
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set:>20}"
        print(header)
        print("-" * (6 + 20 * len(BASES)))
        
        # Print each MO energy
        # n_mos = min(20, max(len(e['alpha']) for e in mo_energies_by_basis.values()))
        n_mos = max(len(e['alpha']) for e in mo_energies_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                energies = mo_energies_by_basis.get(basis_set, {})
                energy = energies.get('alpha', [0])[i] if i < len(energies.get('alpha', [])) else 'N/A'
                if isinstance(energy, (int, float)):
                    row += f"{energy:>20.6f}"
                else:
                    row += f"{str(energy):>20}"
            print(row)
    
    print("=" * 70)

def print_r_rms_values(r2_by_basis):
    BASES = list(r2_by_basis.keys())
    # Print r2 values table at the end
    print("\n" + "=" * 70)
    print("<r²>^½ Values Comparison")
    print("=" * 70)
    
    # Determine if we have UKS (alpha and beta) or RKS
    is_uks = any('beta' in r2_data for r2_data in r2_by_basis.values())
    if is_uks:
        # Print header for UKS
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set + ' (α)':>20} {basis_set + ' (β)':>20}"
        print(header)
        print("-" * (6 + 40 * len(BASES)))
        
        # Print each r2 value
        # n_mos = min(20, max(len(e['alpha']) for e in r2_by_basis.values()))
        n_mos = max(len(e['alpha']) for e in r2_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                r2_data = r2_by_basis.get(basis_set, {})
                alpha_r2 = r2_data.get('alpha', [0])[i] if i < len(r2_data.get('alpha', [])) else 'N/A'
                beta_r2 = r2_data.get('beta', [0])[i] if i < len(r2_data.get('beta', [])) else 'N/A'
                if isinstance(alpha_r2, (int, float)):
                    row += f"{alpha_r2:>20.6f}"
                else:
                    row += f"{str(alpha_r2):>20}"
                if isinstance(beta_r2, (int, float)):
                    row += f"{beta_r2:>20.6f}"
                else:
                    row += f"{str(beta_r2):>20}"
            print(row)
    else:
        # Print header for RKS
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set:>20}"
        print(header)
        print("-" * (6 + 20 * len(BASES)))
        
        # Print each r2 value
        # n_mos = min(20, max(len(e['alpha']) for e in r2_by_basis.values()))
        n_mos = max(len(e['alpha']) for e in r2_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                r2_data = r2_by_basis.get(basis_set, {})
                r2_value = r2_data.get('alpha', [0])[i] if i < len(r2_data.get('alpha', [])) else 'N/A'
                if isinstance(r2_value, (int, float)):
                    row += f"{r2_value:>20.6f}"
                else:
                    row += f"{str(r2_value):>20}"
            print(row)
    
    print("=" * 70)

def run_calculation():
    """Run DFT calculation and save results."""
    
    # Parameters
    ATOM, BASES, XC_FUNCTIONAL, SPIN, NCAS, NELECAS, CHARGE, RATIO, N_FUNCS, SYMM_RKS, ORB_SYMM, ECP = read_config_json('config.json')
    
    print("=" * 70)
    print("Atom DFT Calculation with Diffuse Basis")
    print("=" * 70)
    print(f"Atom: {ATOM}")
    print(f"Base basis: {BASES}")
    if ECP:
        print(f"ECP: {ECP}")
    print(f"XC functional: {XC_FUNCTIONAL}")
    print(f"Diffuse functions: S={N_FUNCS['S']}, P={N_FUNCS['P']}, D={N_FUNCS['D']}")
    print(f"Ratio: {RATIO}")
    print("=" * 70)
    diffuse_string = f'S{N_FUNCS["S"]}P{N_FUNCS["P"]}D{N_FUNCS["D"]}'
    
    # Dictionary to store MO energies and r2 values for each basis set
    mo_energies_by_basis = {}
    kinetic_energies_by_basis = {}
    r2_by_basis = {}
    
    # Generate diffuse basis
    for basis_set in BASES:    
        print("\nGenerating diffuse basis...")
        diffuse_basis = generate_diffuse_basis(
            ATOM, basis_set,
            ratio=RATIO,
            n_funcs=N_FUNCS,
            verbose=True
        )
        # import pdb; pdb.set_trace()
        base_basis = gto.basis.load(basis_set, ATOM)
        #base_basis = gto.uncontract(gto.basis.load(basis_set, ATOM))
        full_basis = base_basis + diffuse_basis
    
        # Build molecule
        print("\nBuilding molecule...")
        ecp_name = infer_ecp(basis_set, ECP)
        mol_kwargs = dict(
            atom=f'{ATOM} 0 0 0',
            basis={ATOM: full_basis},
            charge=CHARGE,
            spin=SPIN,
            symmetry=ORB_SYMM,
            verbose=0,
        )
        if ecp_name:
            mol_kwargs['ecp'] = {ATOM: ecp_name}
            print(f"Using ECP: {ecp_name}")
        mol = gto.M(**mol_kwargs)


        print(f"Molecule created: nao={mol.nao}, nelec={mol.nelectron}")
    
        # Run UKS calculation
        print("\nRunning UKS calculation...")
        mf = scf.UKS(mol)
        mf.xc = XC_FUNCTIONAL
        mf.conv_tol = 1e-9
        mf.verbose = 4
        mf = scf.addons.frac_occ(mf)
        mf.kernel()
        ncore = mol.nelec[0] - NELECAS[0]
        print(f'UKS energy: {mf.e_tot:.6f} Ha')
        if SYMM_RKS:
            S = mol.intor('int1e_ovlp')
            dm_uks = mf.make_rdm1()
            dm_alpha = dm_uks[0]
            dm_beta = dm_uks[1]
            dm_total = dm_alpha + dm_beta
            mf_rks = dft.ROKS(mol)
            mf_rks.xc = XC_FUNCTIONAL
            mf_rks.verbose = 0
            h1e = mf_rks.get_hcore(mol)
            veff_rks = mf_rks.get_veff(mol, dm_total)
            
            if veff_rks.ndim == 3:
                fock_rks = h1e + 0.5 * (veff_rks[0] + veff_rks[1])
            else:
                fock_rks = h1e + veff_rks
            
            mo_occ_alpha = mf.mo_occ[0]
            mo_occ_beta = mf.mo_occ[1]
            mo_occ_rks = mo_occ_alpha + mo_occ_beta
            mo_energy_rks, mo_coeff_rks = eigh(fock_rks, S)
            mo_coeff_alpha = mo_coeff_rks
            mo_coeff_beta = mo_coeff_rks
            # dm_rks = mf_rks.make_rdm1(mo_coeff_rks, mo_occ_rks)
            # tot_electrons =  np.trace(S @ dm_rks[0]) + np.trace(S @ dm_rks[1])
            dm_rks = np.einsum('pi,i,qi->pq', mo_coeff_rks, mo_occ_rks, mo_coeff_rks)
            tot_electrons =  np.trace(S @ dm_rks)
            print(f"Total electrons after AOC calculation: {tot_electrons}, expected {mol.nelectron}")
            e_tot_rks = mf_rks.energy_tot(dm=dm_rks)

            if ORB_SYMM:
                print(f"Symmetrizing orbitals using {ORB_SYMM} symmetry ...")
                print(f'Energy prior to symmetrization: {e_tot_rks:.6f} Ha')
                symm_coeff = symm.symmetrize_space(mol, mo_coeff_rks)
                
                def find_degenerate_groups(energies, tol=1e-6):
                    """Return list of (start_idx, end_idx) for each degenerate group."""
                    groups = []
                    i = 0
                    while i < len(energies):
                        j = i + 1
                        while j < len(energies) and abs(energies[j] - energies[i]) < tol:
                            j += 1
                        if j > i + 1:  # degenerate group (2+ orbitals)
                            groups.append((i, j))
                        i = j
                    return groups

                symm_coeff = mo_coeff_rks.copy()
                for start, end in find_degenerate_groups(mo_energy_rks):
                    idx = np.arange(start, end)
                    symm_block = symm.symmetrize_space(mol, mo_coeff_rks[:, idx])
                    irrep_ids = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, symm_block)
                    print(f"Symmetrized degenerate group [{start}:{end}], Irrep IDs: {irrep_ids}")
                    symm_coeff[:, idx] = symm_block
                    print(f"Symmetrized degenerate group [{start}:{end}], E={mo_energy_rks[start]:.6f}")

                # Symmetrize occupied and virtual separately, then accumulate
                # occ_idx = np.where(mo_occ_rks > 0)[0]
                # virt_idx = np.where(mo_occ_rks <= 0)[0]
                # symm_coeff = np.zeros_like(mo_coeff_rks)
                # if len(occ_idx) > 0:
                #     symm_coeff[:, occ_idx] = symm.symmetrize_space(mol, mo_coeff_rks[:, occ_idx])
                #     print(f"Symmetrized occupied orbitals: {len(occ_idx)}")
                # if len(virt_idx) > 0:
                #     symm_coeff[:, virt_idx] = symm.symmetrize_space(mol, mo_coeff_rks[:, virt_idx])
                #     print(f"Symmetrized virtual orbitals: {len(virt_idx)}")

                # Plot overlap <mo_coeff_rks|symm_coeff> and orthogonality symm_coeff.T @ S @ symm_coeff
                overlap = mo_coeff_rks.T @ S @ symm_coeff
                ortho = symm_coeff.T @ S @ symm_coeff  # should be I for orthonormal orbitals
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                im0 = axes[0, 0].imshow(np.abs(overlap), aspect='auto', cmap='viridis', vmin=0, vmax=1)
                axes[0, 0].set_xlabel('symm_coeff orbital')
                axes[0, 0].set_ylabel('mo_coeff_rks orbital')
                axes[0, 0].set_title(r'$|\langle$ mo_coeff_rks $|$ symm_coeff $\rangle_S|$')
                plt.colorbar(im0, ax=axes[0, 0], label='|overlap|')
                axes[0, 1].plot(np.diag(overlap), 'o-', label='diagonal')
                axes[0, 1].axhline(0, color='gray', ls=':')
                axes[0, 1].set_xlabel('Orbital index')
                axes[0, 1].set_ylabel('Overlap')
                axes[0, 1].set_title('Diagonal overlap (old↔new)')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                im1 = axes[1, 0].imshow(ortho, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=1.1)
                axes[1, 0].set_xlabel('symm_coeff orbital')
                axes[1, 0].set_ylabel('symm_coeff orbital')
                axes[1, 0].set_title(r'symm_coeff$^T$ S symm_coeff (orthogonality)')
                plt.colorbar(im1, ax=axes[1, 0], label='overlap')
                axes[1, 1].plot(np.diag(ortho), 'o-', label='diagonal')
                axes[1, 1].plot(np.diag(ortho) - 1, 'x--', label='diagonal - 1')
                axes[1, 1].axhline(0, color='gray', ls=':')
                axes[1, 1].set_xlabel('Orbital index')
                axes[1, 1].set_ylabel('Overlap')
                axes[1, 1].set_title('Orthogonality: diag (should=1), off-diag (should=0)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                plt.tight_layout()
                plt.savefig('overlap_symm_vs_orig.png', dpi=150)
                print('Saved overlap_symm_vs_orig.png')
                print(f'Orthogonality: max|diag-1|={np.max(np.abs(np.diag(ortho)-1)):.2e}, max|off-diag|={np.max(np.abs(ortho - np.diag(np.diag(ortho)))):.2e}')
                plt.close()

                from pyscf.tools import cubegen
                for i in range(9):
                    cubegen.orbital(mol, f'old_orb_{i}.cube', mo_coeff_rks[:, i])
                    cubegen.orbital(mol, f'new_orb_{i}.cube', symm_coeff[:, i])
                print(mo_energy_rks)

                mo_coeff_alpha = symm_coeff.copy()
                mo_coeff_beta = symm_coeff.copy()
                # dm_rks = mf_rks.make_rdm1(symm_coeff, mo_occ_rks)
                dm_rks = np.einsum('pi,i,qi->pq', symm_coeff, mo_occ_rks, symm_coeff)
                tot_electrons =  np.trace(S @ dm_rks)
                print(f"Total electrons after orbital symmetrization: {tot_electrons}, expected {mol.nelectron}")
                e_tot_rks = mf_rks.energy_tot(dm=dm_rks)
                print(f'Energy after orbital symmetrization: {e_tot_rks:.6f} Ha')


            print(f"ROKS orbital energies computed: {len(mo_energy_rks)} orbitals")
            print(f"  Lowest energy: {mo_energy_rks[0]:.6f} Ha")
            print(f"  Highest energy: {mo_energy_rks[-1]:.6f} Ha")
            mo_energy = (mo_energy_rks, mo_energy_rks)
            mo_occ = (mo_occ_alpha, mo_occ_beta)
            mo_coeff = (mo_coeff_alpha, mo_coeff_beta)
            e_tot = e_tot_rks
        else:
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ
            mo_coeff = mf.mo_coeff
            e_tot = mf.e_tot
        T = mol.intor_symmetric('int1e_kin')
        mo_coeff = mf.mo_coeff
        
        # Print an analysis on MO states
        if len(mo_energy) == 2:
            mo_energies_by_basis[basis_set] = {
                'alpha': mo_energy[0],
                'beta': mo_energy[1]
            }
            ek_alpha = np.einsum('pi,pq,qi->i', mo_coeff[0], T, mo_coeff[0])   
            ek_beta = np.einsum('pi,pq,qi->i', mo_coeff[1], T, mo_coeff[1])   
            kinetic_energies_by_basis[basis_set] = {
                'alpha': ek_alpha,
                'beta': ek_beta
            }
            # Compute and store r2 values for UKS
            r2_alpha = compute_orbital_r2_all(mol, mo_coeff[0])
            r_rms_alpha = np.sqrt(np.maximum(0, r2_alpha))
            r2_beta = compute_orbital_r2_all(mol, mo_coeff[1])
            r_rms_beta = np.sqrt(np.maximum(0, r2_beta))
            r2_by_basis[basis_set] = {
                'alpha': r_rms_alpha,
                'beta': r_rms_beta
            }
        else:
            mo_energies_by_basis[basis_set] = {
                'alpha': mo_energy if hasattr(mo_energy, '__len__') else [mo_energy]
            }
            ek = np.einsum('pi,pq,qi->i', mo_coeff, T, mo_coeff)
            kinetic_energies_by_basis[basis_set] = {
                'alpha': ek
            }
            # Compute and store r2 values for RKS
            r2_values = compute_orbital_r2_all(mol, mo_coeff)
            r_rms_values = np.sqrt(np.maximum(0, r2_values))
            r2_by_basis[basis_set] = {
                'alpha': r_rms_values if hasattr(r_rms_values, '__len__') else [r_rms_values]
            }

        scf_file = f'{ATOM.lower()}_atom_basis_{basis_set}_diffuse_{diffuse_string}_v0.hdf5'
        if SYMM_RKS:
            trimmed_scf_file = f'{ATOM.lower()}_atom_basis_{basis_set}_diffuse_{diffuse_string}_v1_symm.hdf5'
        else:
            trimmed_scf_file = f'{ATOM.lower()}_atom_basis_{basis_set}_diffuse_{diffuse_string}_v1.hdf5'
        write_scf_to_h5(mf, scf_file)

        indices = filter_mos_with_degeneracy(
            mol,
            mo_energy,
            mo_occ,
            mo_coeff,
            exclude_orbitals=['f', 'g'],
            max_n=10
        )
        
        if SYMM_RKS:
            ci_chkfile = f'{ATOM.lower()}_ci_atom_basis_{basis_set}_diffuse_{diffuse_string}_symm.hdf5'
        else:
            ci_chkfile = f'{ATOM.lower()}_ci_atom_basis_{basis_set}_diffuse_{diffuse_string}.hdf5'
        
        construct_cas(
            mo_energy,
            mo_occ,
            mo_coeff,
            indices,
            NCAS,
            NELECAS,
            mol.nelec,
            ncore,
            mf = mf,
            scf_chkfile=trimmed_scf_file,
            ci_chkfile=ci_chkfile
        )
        
        # Plot CAS states
        if SYMM_RKS:
            cas_plot_filename = f'{ATOM.lower()}_cas_states_basis_{basis_set}_diffuse_{diffuse_string}_symm.png'
        else:
            cas_plot_filename = f'{ATOM.lower()}_cas_states_basis_{basis_set}_diffuse_{diffuse_string}.png'
        plot_cas_states(ci_chkfile, plot_filename=cas_plot_filename) #, max_states=20)
        
    print_kinetic_energies(kinetic_energies_by_basis)
    print_mo_energies(mo_energies_by_basis)
    print_r_rms_values(r2_by_basis)
    
def print_kinetic_energies(kinetic_energies_by_basis):
    BASES = list(kinetic_energies_by_basis.keys())
    print("\n" + "=" * 70)
    print("Kinetic Energy Comparison")
    print("=" * 70)

    # Same layout as print_mo_energies: rows = MO index, cols = each basis (α and β if UKS)
    is_uks = any('beta' in energies for energies in kinetic_energies_by_basis.values())
    if is_uks:
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set + ' (α)':>20} {basis_set + ' (β)':>20}"
        print(header)
        print("-" * (6 + 40 * len(BASES)))

        n_mos = max(len(e['alpha']) for e in kinetic_energies_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                energies = kinetic_energies_by_basis.get(basis_set, {})
                alpha_ke = energies.get('alpha', [0])[i] if i < len(energies.get('alpha', [])) else 'N/A'
                beta_ke = energies.get('beta', [0])[i] if i < len(energies.get('beta', [])) else 'N/A'
                if isinstance(alpha_ke, (int, float, np.floating)):
                    row += f"{alpha_ke:>20.6f}"
                else:
                    row += f"{str(alpha_ke):>20}"
                if isinstance(beta_ke, (int, float, np.floating)):
                    row += f"{beta_ke:>20.6f}"
                else:
                    row += f"{str(beta_ke):>20}"
            print(row)
    else:
        header = f"{'MO #':<6}"
        for basis_set in BASES:
            header += f"{basis_set:>20}"
        print(header)
        print("-" * (6 + 20 * len(BASES)))

        n_mos = max(len(e['alpha']) for e in kinetic_energies_by_basis.values())
        for i in range(n_mos):
            row = f"{i+1:<6}"
            for basis_set in BASES:
                energies = kinetic_energies_by_basis.get(basis_set, {})
                ke = energies.get('alpha', [0])[i] if i < len(energies.get('alpha', [])) else 'N/A'
                if isinstance(ke, (int, float, np.floating)):
                    row += f"{ke:>20.6f}"
                else:
                    row += f"{str(ke):>20}"
            print(row)

    print("=" * 70)

# ===========================================================================
# Helper Functions to Load Saved Data
# ===========================================================================

def load_molecule_from_h5(h5_file):
    """
    Reconstruct PySCF molecule object from saved HDF5 file.
    
    Parameters
    ----------
    h5_file : str
        Path to HDF5 file
        
    Returns
    -------
    mol : gto.Mole
        Reconstructed molecule object
    """
    import h5py
    
    with h5py.File(h5_file, 'r') as f:
        mol_grp = f['mol']
        basis_grp = mol_grp['basis']
        
        # Load basis list
        basis_bytes = basis_grp['basis_list'][:].tobytes()
        basis = pickle.loads(basis_bytes)
        
        # Reconstruct molecule
        mol = gto.M(
            atom=mol_grp.attrs['atom'],
            basis=basis,
            charge=mol_grp.attrs['charge'],
            spin=mol_grp.attrs['spin'],
            symmetry=mol_grp.attrs['symmetry'],
            verbose=0
        )
    
    return mol


def load_basis_params_from_h5(h5_file):
    """
    Load basis generation parameters from HDF5 file.
    These can be used to regenerate the basis using generate_diffuse_basis().
    
    Parameters
    ----------
    h5_file : str
        Path to HDF5 file
        
    Returns
    -------
    params : dict
        Dictionary with 'base_basis', 'ratio', 'n_funcs', 'angular_momenta'
    """
    import h5py
    
    with h5py.File(h5_file, 'r') as f:
        basis_params_grp = f['mol/basis/generation_params']
        
        angular_momenta = basis_params_grp.attrs['angular_momenta'].split(',')
        n_funcs = {
            'S': basis_params_grp.attrs['n_funcs_S'],
            'P': basis_params_grp.attrs['n_funcs_P'],
            'D': basis_params_grp.attrs['n_funcs_D'],
        }
        
        params = {
            'base_basis': basis_params_grp.attrs['base_basis'],
            'ratio': basis_params_grp.attrs['ratio'],
            'n_funcs': n_funcs,
            'angular_momenta': angular_momenta,
        }
    
    return params


# ===========================================================================
# Plotting Functions
# ===========================================================================

def load_mo_energies_from_h5(h5_file):
    """
    Load MO energies and occupations from HDF5 file.
    Prefers RKS energies if available (from symmetrized UKS calculation).
    
    Parameters
    ----------
    h5_file : str
        Path to HDF5 file
        
    Returns
    -------
    data : dict
        Dictionary with 'mo_energy_rks', 'mo_energy_alpha', 'mo_energy_beta', 
        'mo_occ_alpha', 'mo_occ_beta', and optionally 'mo_coeff_rks'
    """
    import h5py
    
    with h5py.File(h5_file, 'r') as f:
        # Prefer RKS energies if available
        mo_energy_rks = f['mo_energy_rks'][:]
        mo_energy_alpha = f['mo_energy_alpha'][:]
        mo_energy_beta = f['mo_energy_beta'][:]
        data = {
            'mo_energy_rks': mo_energy_rks,
            'mo_energy_alpha': mo_energy_alpha,
            'mo_energy_beta': mo_energy_beta,
        }
        data['mo_occ_alpha'] = f['mo_occ_alpha'][:]
        data['mo_occ_beta'] = f['mo_occ_beta'][:]
        
        # Add calculation type if available
        if 'calculation_type' in f.attrs:
            data['calculation_type'] = f.attrs['calculation_type']
    
    return data


def compute_casci_energies_from_mo(mo_energy, ncas, nelecas, ncore=0):
    """
    Compute CASCI state energies from MO energies in mean field approximation.
    
    The energy of each determinant is the sum of occupied MO energies.
    
    Parameters
    ----------
    mo_energy : array or tuple
        MO energies. For UKS: (mo_energy_alpha, mo_energy_beta), for RKS: array
    ncas : int
        Number of active orbitals
    nelecas : tuple or int
        Number of active electrons. If tuple (na, nb), unrestricted.
        If int, restricted (total electrons).
    ncore : int
        Number of core orbitals (not included in CAS)
        
    Returns
    -------
    cas_energies : array
        Sorted CASCI state energies
    determinants : list
        List of determinants, each as (alpha_occ, beta_occ) or (occ,) for restricted
    """
    from itertools import combinations
    
    # Handle different input formats
    if isinstance(mo_energy, (list, tuple)) and len(mo_energy) == 2:
        # UKS case: tuple/list of arrays
        mo_energy_alpha = np.asarray(mo_energy[0])
        mo_energy_beta = np.asarray(mo_energy[1])
        is_uks = True
    else:
        # Convert to numpy array
        mo_energy = np.asarray(mo_energy)
        
        if mo_energy.ndim == 2 and mo_energy.shape[0] == 2:
            # UKS case: 2D array with shape (2, norb)
            mo_energy_alpha = mo_energy[0]
            mo_energy_beta = mo_energy[1]
            is_uks = True
        else:
            # RKS case: 1D array
            mo_energy_alpha = mo_energy.flatten()
            mo_energy_beta = mo_energy_alpha
            is_uks = False
    
    if isinstance(nelecas, (list, tuple)):
        na, nb = nelecas
    else:
        na = nb = nelecas // 2
        is_uks = False
    
    # Active orbital indices (after ncore)
    active_orbitals = list(range(ncore, ncore + ncas))
    
    # Get MO energies for active orbitals
    active_energies_alpha = mo_energy_alpha[active_orbitals]
    active_energies_beta = mo_energy_beta[active_orbitals] if is_uks else active_energies_alpha
    
    # Generate all possible determinants
    cas_energies = []
    determinants = []
    
    if is_uks:
        # Unrestricted: generate all combinations of alpha and beta occupations
        for alpha_occ in combinations(range(ncas), na):
            for beta_occ in combinations(range(ncas), nb):
                # Compute energy as sum of occupied MO energies
                energy = (np.sum(active_energies_alpha[list(alpha_occ)]) + 
                         np.sum(active_energies_beta[list(beta_occ)]))
                cas_energies.append(energy)
                determinants.append((list(alpha_occ), list(beta_occ)))
    else:
        # Restricted: generate all combinations
        for occ in combinations(range(ncas), na + nb):
            energy = np.sum(active_energies_alpha[list(occ)])
            cas_energies.append(energy)
            determinants.append(list(occ))
    
    # Sort by energy
    sorted_indices = np.argsort(cas_energies)
    cas_energies = np.array(cas_energies)[sorted_indices]
    determinants = [determinants[i] for i in sorted_indices]
    
    return cas_energies, determinants


def characterize_excitation(determinant, ground_determinant, ncas, mo_labels=None):
    """
    Characterize an excitation by comparing with ground state determinant.
    
    Parameters
    ----------
    determinant : list or tuple
        Current determinant (alpha_occ, beta_occ) or (occ,)
    ground_determinant : list or tuple
        Ground state determinant
    ncas : int
        Number of active orbitals
    mo_labels : list, optional
        Labels for MOs (e.g., ['1s', '2s', '2p', '3s', '3p'])
        
    Returns
    -------
    str
        Description of the excitation (e.g., "1 2s to 3p excitation")
    """
    if mo_labels is None:
        # Generate default labels based on orbital index
        mo_labels = []
        n_orb = 0
        for n in range(1, 10):
            for l_letter in ['s', 'p', 'd', 'f']:
                if l_letter == 's':
                    mo_labels.append(f"{n}{l_letter}")
                    n_orb += 1
                elif l_letter == 'p':
                    for _ in range(3):
                        mo_labels.append(f"{n}{l_letter}")
                        n_orb += 1
                        if n_orb >= ncas:
                            break
                    if n_orb >= ncas:
                        break
                elif l_letter == 'd':
                    for _ in range(5):
                        mo_labels.append(f"{n}{l_letter}")
                        n_orb += 1
                        if n_orb >= ncas:
                            break
                    if n_orb >= ncas:
                        break
                if n_orb >= ncas:
                    break
            if n_orb >= ncas:
                break
        # Pad if needed
        while len(mo_labels) < ncas:
            mo_labels.append(f"orb{len(mo_labels)}")
    
    # Handle UKS vs RKS
    if isinstance(determinant, tuple) and len(determinant) == 2:
        # UKS: (alpha_occ, beta_occ)
        det_alpha = set(determinant[0])
        det_beta = set(determinant[1])
        gs_alpha = set(ground_determinant[0])
        gs_beta = set(ground_determinant[1])
        
        # Find differences
        alpha_removed = gs_alpha - det_alpha
        alpha_added = det_alpha - gs_alpha
        beta_removed = gs_beta - det_beta
        beta_added = det_beta - gs_beta
        
        transitions = []
        for orb_from in alpha_removed:
            for orb_to in alpha_added:
                transitions.append(f"α {mo_labels[orb_from]}→{mo_labels[orb_to]}")
        for orb_from in beta_removed:
            for orb_to in beta_added:
                transitions.append(f"β {mo_labels[orb_from]}→{mo_labels[orb_to]}")
        
        if not transitions:
            return "Ground state"
        return ", ".join(transitions)
    else:
        # RKS: single list
        det_occ = set(determinant)
        gs_occ = set(ground_determinant)
        
        removed = gs_occ - det_occ
        added = det_occ - gs_occ
        
        transitions = []
        for orb_from in removed:
            for orb_to in added:
                transitions.append(f"{mo_labels[orb_from]}→{mo_labels[orb_to]}")
        
        if not transitions:
            return "Ground state"
        return ", ".join(transitions)


def plot_cas_states(ci_chkfile, plot_filename='cas_states.png', max_states=50):
    """
    Plot CAS state energies computed from MO energies with excitation characterizations.
    
    Parameters
    ----------
    ci_chkfile : str
        Path to CI checkpoint file
    plot_filename : str
        Output filename for the plot
    max_states : int
        Maximum number of states to plot
    """
    import h5py
    
    with h5py.File(ci_chkfile, 'r') as f:
        if 'ci' not in f:
            print(f"Error: 'ci' group not found in {ci_chkfile}")
            return
        
        ci_grp = f['ci']
        ncas = int(ci_grp['ncas'][()])
        ncore = int(ci_grp['ncore'][()])
        nelecas = ci_grp['nelecas'][:]
        if len(nelecas) == 2:
            nelecas = tuple(nelecas)
        else:
            nelecas = int(nelecas[0])
        
        # Load MO energies
        if 'mf_mo_energy' in ci_grp:
            mo_energy = np.array(ci_grp['mf_mo_energy'][:])
        else:
            print("Error: MO energies not found in CI file")
            return
    
    # Compute CASCI energies from MO energies
    print(f"Computing CASCI energies from MO energies (ncas={ncas}, nelecas={nelecas}, ncore={ncore})...")
    cas_energies, determinants = compute_casci_energies_from_mo(mo_energy, ncas, nelecas, ncore)
    
    # Normalize to ground state
    cas_energies_normalized = cas_energies - cas_energies[0]
    
    # Limit number of states
    n_states = min(len(cas_energies_normalized), max_states)
    cas_energies_plot = cas_energies_normalized[:n_states]
    determinants_plot = determinants[:n_states]
    
    # Characterize excitations
    ground_det = determinants[0]
    # excitation_labels = []
    # for i, det in enumerate(determinants_plot):
    #     if i == 0:
    #         label = "Ground state"
    #     else:
    #         label = "" #characterize_excitation(det, ground_det, ncas)
    #     excitation_labels.append(label)
    
    # Create the plot
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot energies
    state_indices = np.arange(n_states)
    ax.plot(state_indices, cas_energies_plot, 'o-', markersize=8, linewidth=2, 
            color='blue', label='CASCI States (from MO energies)')
    
    # # Add labels for each state
    # for i, (energy, label) in enumerate(zip(cas_energies_plot, excitation_labels)):
    #     # Truncate long labels
    #     display_label = label if len(label) < 40 else label[:37] + "..."
    #     ax.text(i, energy, f'  {i}\n  {display_label}', fontsize=8, 
    #             verticalalignment='bottom' if i % 2 == 0 else 'top',
    #             horizontalalignment='left',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('State Index', fontsize=12)
    ax.set_ylabel('Energy (Ha, relative to ground state)', fontsize=12)
    ax.set_title(f'CASCI State Energies from MO Energies\n(ncas={ncas}, nelecas={nelecas}, ncore={ncore})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"CAS state plot saved to: {plot_filename}")
    print(f"Total states computed: {len(cas_energies)}")
    print(f"States plotted: {n_states}")


if __name__ == '__main__':

    run_calculation()
        
    # Example: How to load the data back
    print("\n" + "=" * 70)
    print("Example: Loading saved data")
    print("=" * 70)
    print("\nTo reconstruct molecule:")
    print("  from run_atom_diffuse import load_molecule_from_h5")
    print("  mol = load_molecule_from_h5('li_atom_diffuse_S4P8D11.h5')")
    print("\nTo get basis generation parameters:")
    print("  from run_atom_diffuse import load_basis_params_from_h5")
    print("  params = load_basis_params_from_h5('li_atom_diffuse_S4P8D11.h5')")
    print("  # Then regenerate: generate_diffuse_basis('Li', **params)")
    print("\nTo plot MO energies from saved file:")
    print("  from run_atom_diffuse import load_mo_energies_from_h5, plot_mo_energies")
    print("  data = load_mo_energies_from_h5('li_atom_diffuse_S4P8D11.h5')")
    print("  plot_mo_energies(data['mo_energy_alpha'], data['mo_energy_beta'],")
    print("                   data['mo_occ_alpha'], data['mo_occ_beta'])")

