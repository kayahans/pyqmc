#!/usr/bin/env python 
# from adjustText import adjust_text
import h5py
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob
import json
import os
import re
from scipy import stats
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import linear_sum_assignment
import warnings
from datetime import datetime
import contextlib
import shutil
import tempfile
import yaml
from typing import Any, Dict, List, Sequence, Tuple

warnings.filterwarnings('ignore')
# Find files containing both '_dmc_' and '_nelecas_' in the current directory
DISCARD_STEPS = 5000
BLOCK_SIZE = 2000


@contextlib.contextmanager
def hdf5_snapshot_read(filepath, snapshot=True):
    """
    Read HDF5 without holding an HDF5 lock on *filepath* while live writers append.

    When snapshot=True (default), copy via plain ``shutil.copy2`` (no h5py open on
    the source), read arrays from the private temp copy, then delete it immediately.
    """
    temp_path = None
    read_path = filepath
    if snapshot:
        fd, temp_path = tempfile.mkstemp(suffix='.hdf5', prefix='plot_dmc_')
        os.close(fd)
        try:
            shutil.copy2(filepath, temp_path)
        except Exception:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
            raise
        read_path = temp_path
        print(f'HDF5 snapshot: copied {filepath} -> {temp_path}')

    prev_lock = os.environ.get('HDF5_USE_FILE_LOCKING')
    try:
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        with h5py.File(read_path, 'r') as f:
            yield f
    finally:
        if prev_lock is None:
            os.environ.pop('HDF5_USE_FILE_LOCKING', None)
        else:
            os.environ['HDF5_USE_FILE_LOCKING'] = prev_lock
        if temp_path and os.path.isfile(temp_path):
            os.remove(temp_path)


def load_mo_cbs_errors_mo_energies_with_cbs(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse ``mo_energies_with_cbs.txt`` and return CBS-ε(DZ) corrections (Ha) for α/β
    on each spatial MO, indexed 0..n_mo-1 in PySCF MO order (file column ``MO #`` is 1-based).

    Returns
    -------
    err_alpha, err_beta : (n_mo,) float arrays, zeros for MO indices not present in the file
    mo_numbers : 1D array of MO# values actually read from the file
    """
    from_file: Dict[int, Tuple[float, float]] = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or 'MO #' in line or '---' in line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                mo1 = int(parts[0])
            except ValueError:
                continue
            try:
                eps_a = float(parts[-2])
                eps_b = float(parts[-1])
            except (IndexError, ValueError):
                continue
            if mo1 < 1:
                continue
            from_file[mo1] = (eps_a, eps_b)

    if not from_file:
        raise ValueError(f"No MO rows parsed from {path}")

    n_mo = max(from_file.keys())
    err_alpha = np.zeros(n_mo, dtype=np.float64)
    err_beta = np.zeros(n_mo, dtype=np.float64)
    for mo1, (a, b) in from_file.items():
        err_alpha[mo1 - 1] = a
        err_beta[mo1 - 1] = b
    mo_numbers = np.array(sorted(from_file.keys()), dtype=np.int32)
    return err_alpha, err_beta, mo_numbers


def _nmo_mf(mf) -> int:
    c = mf.mo_coeff
    if c is None:
        return 0
    if c.ndim == 2:
        return c.shape[1]
    return c[0].shape[1]


def get_cbs_shifts_for_dets(
    n_det: int,
    config_path: str,
    mo_cbs_path: str,
    ground_det_index: int = 0,
) -> np.ndarray:
    """
    Per-determinant CBS-ε(DZ) shifts in the same det order as
    ``filter_determinants_from_ci`` / ``sorted_mask_indices``.

    Relative to the ground (reference) determinant ``ground_det_index``: for each
    spin, orbitals in the ref but not the target act as *holes* (minus ε), and
    orbitals in the target but not the ref act as *particles* (plus ε)::

        shift = Σ_{j in G_α\\D_α} (-err_α[j]) + Σ_{j in D_α\\G_α} (+err_α[j])
              + (same for β)
    The reference row has zero shift. ``ground_det_index`` should match the
    min-mean-field det index (e.g. ``argmin(hmf_diag)``) when that is the GS.
    """
    from pyqmc import pyscftools
    from pyqmc.bosonslater import filter_determinants_from_ci, binary_to_occ
    from pyscf import fci

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    atom_name = config['atom_name']
    basis_name = config['basis_name']
    symm_tag = config['symm_tag']
    det_emax = config['det_emax']
    file_template = (
        '{}_atom_basis_{}_diffuse_S0P0D0_v1{}.hdf5 '
        '{}_ci_atom_basis_{}_diffuse_S0P0D0{}.hdf5'
    )
    scf_chk, ci_chk = file_template.format(
        atom_name, basis_name, symm_tag, atom_name, basis_name, symm_tag
    ).split()

    mol, mf, mc = pyscftools.recover_pyscf(scf_chk, ci_checkfile=ci_chk)

    mo_energies = getattr(mf, 'mo_energy', None)
    if mo_energies is None:
        inp = pyscftools.load_mf_inputs_from_hdf5(scf_chk, mol=mol)
        if inp is not None and 'mo_energy' in inp:
            mo_energies = inp['mo_energy']
    if mo_energies is None:
        fock_mo = (
            mf.mo_coeff.conj().T @ mf.get_fock(dm=mf.make_rdm1()) @ mf.mo_coeff
        )
        mo_energies = np.diag(fock_mo).real

    _, saved = filter_determinants_from_ci(
        mc, mo_energies, det_emax, mol=mol, mf=mf
    )
    ncore = mc.ncore if hasattr(mc, 'ncore') else 0
    deters_orig = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
    alpha_occ = np.array([binary_to_occ(x[1], ncore)[0] for x in deters_orig])
    beta_occ = np.array([binary_to_occ(x[2], ncore)[0] for x in deters_orig])
    sorted_mask_indices = saved['sorted_mask_indices']
    occupations = [
        (alpha_occ[i], beta_occ[i]) for i in sorted_mask_indices
    ]

    if len(occupations) != n_det:
        raise ValueError(
            f'CBS shift: determinant count mismatch — PySCF/CI has {len(occupations)} '
            f'determinants but DMC hmf/delta has {n_det}. Check config, det_emax, and H5 order.'
        )
    if ground_det_index < 0 or ground_det_index >= len(occupations):
        raise ValueError(
            f'CBS shift: ground_det_index={ground_det_index} out of range for '
            f'len(occupations)={len(occupations)}'
        )

    err_a_file, err_b_file, _ = load_mo_cbs_errors_mo_energies_with_cbs(mo_cbs_path)
    nmo = _nmo_mf(mf)
    if len(err_a_file) < nmo:
        err_alpha = np.zeros(nmo, dtype=np.float64)
        err_beta = np.zeros(nmo, dtype=np.float64)
        err_alpha[: len(err_a_file)] = err_a_file
        err_beta[: len(err_b_file)] = err_b_file
    else:
        err_alpha = err_a_file[:nmo].copy()
        err_beta = err_b_file[:nmo].copy()

    g_a = set(int(x) for x in occupations[ground_det_index][0])
    g_b = set(int(x) for x in occupations[ground_det_index][1])

    def _signed_spin_shift(set_ref: set, set_det: set, err_vec: np.ndarray) -> Tuple[float, int]:
        t = 0.0
        o = 0
        for j in set_ref - set_det:
            jj = int(j)
            if jj < 0 or jj >= nmo:
                o += 1
            else:
                t -= err_vec[jj]
        for j in set_det - set_ref:
            jj = int(j)
            if jj < 0 or jj >= nmo:
                o += 1
            else:
                t += err_vec[jj]
        return t, o

    shifts = np.empty(n_det, dtype=np.float64)
    hmf_local = np.empty(n_det, dtype=np.float64)
    
    oob = 0
    for k, (occ_a, occ_b) in enumerate(occupations):
        d_a = set(int(x) for x in occ_a)
        d_b = set(int(x) for x in occ_b)
        s_a, oa = _signed_spin_shift(g_a, d_a, err_alpha)
        s_b, ob = _signed_spin_shift(g_b, d_b, err_beta)
        oob += oa + ob
        shifts[k] = s_a + s_b
        hmf_local[k] = np.sum(mf.mo_energy[0][occ_a]) + np.sum(mf.mo_energy[1][occ_b])
    if oob:
        print(
            f'Warning: {oob} α/β index lookups were out of bounds for nmo={nmo} '
            f'(CBS excitation shift / hole-particle).'
        )
    return shifts


def plot_energies(energies):
    total_energy = energies['total']
    ee = energies['ee']
    ei = energies['ei']
    ka = energies['ka']
    kb = energies['kb']
    vj = energies['vj']
    vxc = energies['vxc']
    #eb0 = energies['eb0']

    # Prepare energy data with labels, colors, and linestyles
    energy_data = [
        ('Total Energy', total_energy, 'k', '-'),
        ('e-e', ee, 'r', '-'),
        ('e-i', ei, 'b', '-'),
        ('kinetic_a', ka, 'g', '-'),
        ('kinetic_b', kb, 'y', '-'),
        ('Hartree', vj, 'c', '-'),
        ('XC', vxc, 'm', '-'),
        ('Hartree + XC', vj + vxc, 'k', '--'),
        #('eb0', eb0, 'k', ':'),
    ]
    print("Total number of steps: ", total_energy.shape)
    discard = min(DISCARD_STEPS, int(total_energy.shape[0]/5))
    print(f'Discarding {discard} steps')
    # Calculate mean and std for each energy
    energy_stats = {}
    for label, data, color, linestyle in energy_data:
        mean = np.mean(data[discard:])
        std = np.std(data[discard:])
        energy_stats[label] = {'mean': mean, 'std': std, 'data': data, 'color': color, 'linestyle': linestyle}
    
    # Find largest standard deviation
    largest_std = max(stat['std'] for stat in energy_stats.values())
    
    # Calculate delta_y = 4 * largest_std
    delta_y = 9 * largest_std
    
    # Create subplots - one for each energy
    n_energies = 9
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()  # Flatten to easily index 0..8
    
    # Plot each energy in its own subplot
    for idx, (label, data, color, linestyle) in enumerate(energy_data):
        ax = axes[idx]
        stats = energy_stats[label]
        mean = stats['mean']
        std = stats['std']
        
        # Plot the energy
        ax.plot(data, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8, label=label)
        ax.axvline(x=discard, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        # Add mean line
        ax.axhline(y=mean, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Mean')
        
        # Set y-axis limits: miny = mean - 2*std, maxy = mean + 2*std
        # But ensure delta_y is the same for all plots
        miny = mean - 2 * std
        maxy = mean + 2 * std
        
        # Adjust to ensure delta_y is consistent
        current_delta = maxy - miny
        if current_delta < delta_y:
            # Expand symmetrically around mean
            expansion = (delta_y - current_delta) / 2
            miny = mean - 2 * std - expansion
            maxy = mean + 2 * std + expansion
        
        ax.set_ylim(miny, maxy)
        
        # Add text with mean and std
        text_str = f'Mean: {mean:.6f}, Std: {std:.6f}'
        ax.text(0.02, 0.1, text_str, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Energy', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Energy Contributions (Individual Plots)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('energy_contributions.png', dpi=150, bbox_inches='tight')
    plt.close()


# def plot_energies(energies):
#     total_energy = energies['total']
#     ee = energies['ee']
#     ei = energies['ei']
#     ka = energies['ka']
#     kb = energies['kb']
#     vj = energies['vj']
#     vxc = energies['vxc']
#     eb0 = energies['eb0']

#     # Create 2x1 subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

#     # First subplot
#     ax1.plot(total_energy, 'k-', label='Total Energy')
#     ax1.plot(ee, 'r-', label='e-e')

#     ax1.plot(ka, 'g-', label='kinetic_a')
#     ax1.plot(kb, 'y-', label='kinetic_b')
#     ax1.plot(vj, 'c-', label='Hartree')
#     ax1.plot(vxc, 'm-', label='XC')
#     ax1.plot(vj + vxc, 'k--', label='Hartree + XC')
#     ax1.legend()
#     ax1.set_title('Energy Contributions')
#     ax1.set_xlabel('Step')
#     ax1.set_ylabel('Energy')
#     ax1.grid(True, alpha=0.3)

#     # Second subplot
#     ax2.plot(ei, 'b-', label='e-i')
#     ax2.plot(eb0, 'k--', label='eb0')
#     ax2.legend()
#     ax2.text(0.05, 0.95, f'Last value of E-I: {ei[-1]:.4f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
#     ax2.set_title('E-I and EB_0 Energies')
#     ax2.set_xlabel('Step') 
#     ax2.set_ylabel('Energy')
#     ax2.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('energy_contributions.png')
#     plt.close()

#     num_steps = total_energy.shape[0]

def get_data(
    file_name,
    discard_steps=None,
    factor=3.0,
    hmf_file='hmf.hdf5',
    cbs_mo_file='wfs/mo_energies_with_cbs.txt',
    config_path='config.yaml',
    apply_cbs_correction=True,
    hdf5_snapshot=True,
):
    with hdf5_snapshot_read(file_name, snapshot=hdf5_snapshot) as f:
        total_energy = f['energytotal'][:]
        ee = f['energyee'][:] # e-e 
        ei = f['energyei'][:] # e-i
        ka = f['energyka'][:] # k-a
        kb = f['energykb'][:] # k-b
        vj = f['energyvj'][:] # Hartree
        vxc = f['energyvxc'][:] # XC

        energies = {
            'total': total_energy,
            'ee': ee,
            'ei': ei,
            'ka': ka,
            'kb': kb,
            'vj': vj,
            'vxc': vxc,
            #'eb0': ka + kb + ee - (vj + vxc),
        }

        delta_vmc = f['abc_dmc_excitationsdelta'][:]  # Shape: (L, M, N)
        ovlp_vmc = f['abc_dmc_excitationsovlp'][:]
    full_data = {
        'delta': delta_vmc,
        'ovlp': ovlp_vmc,
    }
    num_steps = delta_vmc.shape[0]
    print(f'File name: {file_name}')
    print(f'Total number of steps: {num_steps}')
    if discard_steps is None:
        discard_steps = np.min([int(num_steps/3), DISCARD_STEPS])
    print(f'Discard steps: {discard_steps}')
    num_det = delta_vmc.shape[1]
    print(f'Number of determinants: {num_det}')

    with hdf5_snapshot_read(hmf_file, snapshot=hdf5_snapshot) as hmf_f:
        hmf_diag = np.array(hmf_f['hmf'][:])

    cbs_info: Dict[str, Any] = {
        'applied': False,
        'iref': None,
        'max_abs_delta': 0.0,
        'message': '',
    }
    if apply_cbs_correction and os.path.isfile(cbs_mo_file) and os.path.isfile(config_path):
        try:
            iref = int(np.argmin(hmf_diag))
            shift = get_cbs_shifts_for_dets(
                num_det, config_path, cbs_mo_file, ground_det_index=iref
            )
            hmf_diag = hmf_diag + shift
            cbs_info['applied'] = True
            cbs_info['iref'] = iref
            cbs_info['shift'] = np.array(shift, dtype=np.float64, copy=True)
            cbs_info['max_abs_delta'] = float(np.max(np.abs(shift)))
            cbs_info['message'] = (
                'CBS-ε(DZ) on diagonal: hole/particle vs occ at min-mean-field det (iref); '
                'ref det shift is zero by construction'
            )
            print(
                f'CBS correction: applied, iref={iref}, '
                f"max|Δdiag|={cbs_info['max_abs_delta']:.6f} Ha"
            )
        except Exception as e:
            cbs_info['message'] = str(e)
            print(f'Warning: CBS-ε(DZ) correction skipped: {e}')
    else:
        if not apply_cbs_correction:
            cbs_info['message'] = 'disabled by flag'
        elif not os.path.isfile(cbs_mo_file):
            cbs_info['message'] = f'missing {cbs_mo_file}'
        else:
            cbs_info['message'] = f'missing {config_path}'

    hmf = np.diag(hmf_diag)
    
    # mean_delta = np.mean(delta_vmc, axis=0)
    # std_delta = np.std(delta_vmc, axis=0)
    def simple_reblock(data, block_size=BLOCK_SIZE):
        """
        Simple reblocking method to compute mean and std accounting for autocorrelation.
        
        Parameters
        ----------
        data : array
            Data array of shape (n_steps, ...)
        block_size : int, optional
            Block size. If None, uses sqrt(n_steps) as default.
            
        Returns
        -------
        mean : array
            Reblocked mean
        std : array
            Reblocked standard error (std of block means)
        """
        n_steps = data.shape[0]
        # Determine block size
        if block_size is None:
            # Default: use sqrt of number of steps
            # block_size = max(1, int(np.sqrt(n_steps)))
            block_size = max(BLOCK_SIZE, int(np.sqrt(n_steps)))
            block_size = int(min(BLOCK_SIZE, n_steps/4))
        # Ensure block_size doesn't exceed 1/4 of data length
        
        
        # Number of complete blocks
        n_blocks = n_steps // block_size
        
        if n_blocks < 2:
            # If we can't form at least 2 blocks, fall back to simple statistics
            print(f"Warning: Only {n_blocks} block(s) possible, using simple statistics")
            return np.mean(data, axis=0), np.std(data, axis=0) / np.sqrt(n_steps), np.mean(data, axis=0), 1
        
        # Reshape data into blocks: (n_blocks, block_size, ...)
        data_blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size, *data.shape[1:])
        
        # Compute block averages
        block_means = np.mean(data_blocks, axis=1)  # Shape: (n_blocks, ...)
        
        # Overall mean (average of block means)
        mean = np.mean(block_means, axis=0)
        
        # Standard error: std of block means, scaled by sqrt(n_blocks - 1)
        std = np.std(block_means, axis=0, ddof=1)
        
        return mean, std, block_means, block_size
    
    # Apply reblocking to delta and ovlp (after discarding initial steps)
    # REBLOCKING CONVERGENCE
    data_steps = num_steps - discard_steps
    print(f'Steps used for reblocking: {data_steps}')
    if data_steps > 4000:
        block_sizes = [500, 1000, 2000]
    else:
        block_sizes = [int(data_steps/4)]
    for block_size in block_sizes:
        try:
            mean_delta, std_delta, block_means_delta, block_size_delta = simple_reblock(delta_vmc[discard_steps:], block_size)
            mean_ovlp, std_ovlp, block_means_ovlp, block_size_ovlp = simple_reblock(ovlp_vmc[discard_steps:], block_size)
            plt.figure(figsize=(10, 10))
            # Leave-one-block-out jackknife: eigenvalues from (n*mean - block_k)/(n-1)
            # last_delta = np.mean(block_means_delta[-1], axis=0)
            # last_ovlp = np.mean(block_means_ovlp[-1], axis=0)
            last_eigs = eigenpairs_delta_sinv_plus_hmf(mean_delta, mean_ovlp, hmf)['eigv']
            n_jk = block_means_delta.shape[0]
            eigs_block = []
            if n_jk >= 2:
                nm1 = float(n_jk - 1)
                for k in range(n_jk):
                    delta_k = (n_jk * mean_delta - block_means_delta[k]) / nm1
                    ovlp_k = (n_jk * mean_ovlp - block_means_ovlp[k]) / nm1
                    delta_filter = std_delta > factor * np.abs(mean_delta)
                    ovlp_filter = std_ovlp > factor * np.abs(mean_ovlp)
                    # print(f'delta_filter: {delta_filter.sum()}')
                    # print(f'ovlp_filter: {ovlp_filter.sum()}')
                    delta_k[delta_filter] = 0
                    ovlp_k[ovlp_filter] = 0
                    eigs_i = eigenpairs_delta_sinv_plus_hmf(delta_k, ovlp_k, hmf, ref_eigv=last_eigs)['de']
                    eigs_block.append(eigs_i)
            else:
                eigs_block.append(
                    eigenpairs_delta_sinv_plus_hmf(mean_delta, mean_ovlp, hmf, ref_eigv=last_eigs)['de']
                )

            eigs_block = np.array(eigs_block)
            level_columns = [f'Level_{level}' for level in range(eigs_block.shape[1])]
            eigs_block_df = pd.DataFrame(np.real(eigs_block), columns=level_columns)
            eigs_block_df.insert(0, 'jackknife_index', np.arange(len(eigs_block_df)))
            eigs_block_df.to_csv(f'eigs_block_{block_size}.csv', index=False)
            for level in range(eigs_block.shape[1]):
                plt.plot(
                    np.arange(len(eigs_block)),
                    np.real(eigs_block[:, level]),
                    '-o',
                    label=f'Level {level}',
                )
            plt.xlabel('Jackknife index k (block k left out of mean)')
            plt.ylabel('Eigenvalue (shifted)')
            plt.title(
                f'Jackknife eigenvalues (block size = {block_size}, n = {n_jk})'
            )
            plt.ylim(0, 1.1)
            plt.legend()
            plt.tight_layout()

            # state_1 = eigs_block[:,1:4]
            # state_2 = eigs_block[:,4]
            # state_3 = eigs_block[:,5:8]
            # print(f'Mean and std of eigs_block at block size {block_size}: ')
            # print(f'State 1: {np.mean(state_1).real}, {np.std(state_1).real}, {np.mean(state_1, axis=0).real}')
            # print(f'State 2: {np.mean(state_2).real}, {np.std(state_2).real}, {np.mean(state_2, axis=0).real}')
            # print(f'State 3: {np.mean(state_3).real}, {np.std(state_3).real}, {np.mean(state_3, axis=0).real}')
            plt.savefig(f'eigs_block_{block_size}.png')

            assert block_size_delta == block_size_ovlp, "Block sizes for delta and ovlp are different"
        except:
            print(f'Error in reblocking for block size {block_size}')
    # Print reblocking info
    print(f'Reblocking: block_size={block_size_delta}, n_blocks={data_steps // block_size_delta}')
    delta_mask = std_delta > factor * np.abs(mean_delta)
    filtered_mean_delta = mean_delta.copy()
    filtered_mean_delta[delta_mask] = 0
    delta = {
        'mean': mean_delta,
        'std': std_delta,
        'filtered_mean': filtered_mean_delta,
        'block_means': block_means_delta,
        'filter_mask': delta_mask,
    }
    # mean_ovlp = np.mean(ovlp_vmc, axis=0)
    # std_ovlp = np.std(ovlp_vmc, axis=0)
    ovlp_mask = std_ovlp > factor * np.abs(mean_ovlp)
    filtered_mean_ovlp = mean_ovlp.copy()
    filtered_mean_ovlp[ovlp_mask] = 0
    ovlp = {
        'mean': mean_ovlp,
        'std': std_ovlp,
        'filtered_mean': filtered_mean_ovlp,
        'block_means': block_means_ovlp,
        'filter_mask': ovlp_mask,
    }
    n_blocks_reblock = int(data_steps // block_size_delta)
    step_info = {
        'n_total': int(num_steps),
        'n_discard': int(discard_steps),
        'n_used': int(data_steps),
        'n_blocks': n_blocks_reblock,
        'block_size': int(block_size_delta),
        'cbs': cbs_info,
    }
    return delta, ovlp, hmf, energies, full_data, step_info

def ordered_matrices(delta, ovlp, hmf):
    indices = np.argsort(np.diag(hmf))
    if np.all(indices == np.arange(len(indices))):
        print("hmf is already ordered")

    ix = np.ix_(indices, indices)
    delta_mean = delta['mean'][ix]
    delta_std = delta['std'][ix]
    delta_filtered = delta['filtered_mean'][ix]
    ovlp_mean = ovlp['mean'][ix]
    ovlp_std = ovlp['std'][ix]
    ovlp_filtered = ovlp['filtered_mean'][ix]

    # Reorder block_means and apply filter mask for block bootstrap
    delta_block_means = None
    ovlp_block_means = None
    if 'block_means' in delta and 'filter_mask' in delta:
        delta_mask_reordered = delta['filter_mask'][ix]
        ovlp_mask_reordered = ovlp['filter_mask'][ix]
        # block_means shape: (n_blocks, M, N)
        delta_bm_ordered = delta['block_means'][:, indices, :][:, :, indices]
        ovlp_bm_ordered = ovlp['block_means'][:, indices, :][:, :, indices]
        # Apply filter: zero entries where std > factor * |mean|
        delta_block_means = np.where(delta_mask_reordered[np.newaxis, :, :], 0, delta_bm_ordered)
        ovlp_block_means = np.where(ovlp_mask_reordered[np.newaxis, :, :], 0, ovlp_bm_ordered)

    delta = {
        'mean': delta_mean,
        'std': delta_std,
        'filtered_mean': delta_filtered,
        'block_means': delta_block_means,
    }
    ovlp = {
        'mean': ovlp_mean,
        'std': ovlp_std,
        'filtered_mean': ovlp_filtered,
        'block_means': ovlp_block_means,
    }
    hmf = hmf[ix]
    return delta, ovlp, hmf

def symmetrize_overlap(S):
    """Force real overlap matrix symmetric (Hermitian in real arithmetic)."""
    return 0.5 * (S + S.T)



def eigenpairs_delta_sinv_plus_hmf(
    delta_i,
    ovlp_i,
    hmf,
    regularization=1e-10,
    ref_eigv=None,
    near_deg_tol=1e-2,
):
    sinv = robust_matrix_inverse(ovlp_i, regularization=regularization)
    h = np.matmul(sinv, delta_i) + hmf
    eig, eigv = np.linalg.eig(h)
    order = np.lexsort((eig.imag, eig.real))
    if ref_eigv is not None and ref_eigv.shape == eigv.shape:
        eig_sorted = eig[order]
        eigv_sorted = eigv[:, order]
        overlap = np.abs(np.matmul(eigv_sorted.conj().T, ref_eigv))
        block_order = []
        start = 0
        while start < len(eig_sorted):
            end = start + 1
            while end < len(eig_sorted) and np.abs(eig_sorted[end].real - eig_sorted[end - 1].real) < near_deg_tol:
                end += 1
            if end - start == 1:
                block_order.append(start)
            else:
                local_overlap = overlap[start:end, start:end]
                rows, cols = linear_sum_assignment(-local_overlap)
                local_rank = rows[np.argsort(cols)] + start
                block_order.extend(local_rank.tolist())
            start = end
        order = order[np.array(block_order, dtype=int)]

    eig = eig[order]
    ref = np.min(eig.real)
    if np.any(eig.imag > 1e-6):
        print("Warning: Eigenvalues are complex")
        # print(eig.imag)
    eigv = eigv[:, order]
    de = eig - ref
    results = {
        'eig': eig,
        'eigv': eigv,
        'de': de,
    }
    return results

def robust_matrix_inverse(matrix, regularization=1e-10):
    """Robust matrix inversion with regularization for ill-conditioned matrices."""
    # Add small regularization to diagonal
    regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
    
    try:
        # Try direct inversion first
        return np.linalg.inv(regularized_matrix)
    except np.linalg.LinAlgError:
        # If that fails, use pseudo-inverse
        print("Warning: Using pseudo-inverse due to ill-conditioned matrix")
        return np.linalg.pinv(regularized_matrix)

def parse_nist_configuration(config_string):
    """
    Parse a NIST configuration string (e.g., "1s^2 2s^1") into orbital occupations.
    Returns a dictionary mapping orbital labels to occupation numbers.
    """
    from collections import defaultdict
    
    if not config_string or config_string == 'N/A':
        return {}
    
    occ_dict = defaultdict(int)
    
    # Pattern to match orbital labels with superscripts: e.g., "1s^2", "2p^1"
    pattern = r'(\d+)([spdfgh])\^?(\d+)'
    matches = re.findall(pattern, config_string)
    
    for match in matches:
        n, l, occ = match
        label = f"{n}{l}"
        occ_dict[label] += int(occ)
    
    return dict(occ_dict)

def is_nist_single_excitation(nist_config, ground_config):
    """
    Check if a NIST configuration represents a single excitation.
    Compares orbital occupations to ground state.
    """
    nist_occ = parse_nist_configuration(nist_config)
    gs_occ = parse_nist_configuration(ground_config)
    
    if not nist_occ or not gs_occ:
        return None  # Can't determine
    
    # Count electrons that changed orbitals
    all_orbitals = set(list(nist_occ.keys()) + list(gs_occ.keys()))
    num_changes = 0
    
    for orb in all_orbitals:
        nist_count = nist_occ.get(orb, 0)
        gs_count = gs_occ.get(orb, 0)
        diff = abs(nist_count - gs_count)
        num_changes += diff
    
    # Single excitation means exactly 2 electrons moved (one from, one to)
    return num_changes == 2

def _nist_degeneracy_to_count(deg) -> int:
    """
    Map a NIST degeneracy field to the number of replicated levels (microstates).
    Strings like ``3/9`` use the numerator (3), consistent with
    ``generate_eig_convergence_report`` expansion.
    """
    if deg is None:
        return 1
    if isinstance(deg, str):
        s = deg.strip()
        if not s:
            return 1
        if "/" in s:
            try:
                return int(s.split("/")[0].strip())
            except ValueError:
                return 1
        try:
            return int(s)
        except ValueError:
            return 1
    try:
        return int(deg)
    except (TypeError, ValueError):
        return 1


def read_nist_data(filename):
    """
    Read NIST data from a report text file. Only section

        1. EXCITATIONS ACCESSIBLE IN CASCI WITH NIST ENERGIES

    is parsed; parsing stops at section 2 (e.g. "2. EXCITATIONS ALLOWED BY CASCI BUT NOT IN NIST")
    or any later numbered section.

    Expected table columns: Configuration | Term | Degeneracy | Excitation (eV) | Excitation (Ha)
    Optional trailing parenthetical after Ha is ignored.

    Returns:
        list of dicts with keys: 'energy' (Ha), 'energy_ev', 'degeneracy', 'term_symbol',
        'configuration'
    """
    if filename is None:
        filename = 'Li_sz_0p5.txt'

    section1_re = re.compile(
        r'1\.\s*EXCITATIONS ACCESSIBLE IN CASCI WITH NIST ENERGIES',
        re.IGNORECASE,
    )
    numbered_section_re = re.compile(r'^\s*(\d+)\.\s+')

    def _is_rule_line(s):
        t = s.strip()
        return len(t) >= 3 and len(set(t)) == 1 and t[0] in '-='

    def _parse_row(line):
        """Parse one data row from section 1; return dict or None."""
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('!'):
            return None
        if _is_rule_line(s):
            return None
        # Split on 2+ spaces or tabs (table columns)
        parts = [p for p in re.split(r'\s{2,}|\t', s) if p]
        if len(parts) < 5:
            return None
        # Drop trailing non-numeric fields (e.g. parenthetical "(L=1, S=0.5, ...)")
        while len(parts) >= 2:
            try:
                ha = float(parts[-1])
                ev = float(parts[-2])
                break
            except ValueError:
                parts.pop()
        else:
            return None
        if len(parts) < 5:
            return None
        deg = parts[-3]
        term = parts[-4]
        configuration = ' '.join(parts[:-4])
        return {
            'energy': ha,
            'energy_ev': ev,
            'degeneracy': deg,
            'term_symbol': term,
            'configuration': configuration,
        }

    excitations = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            in_section = False
            for line in f:
                if not in_section:
                    if section1_re.search(line):
                        in_section = True
                    continue

                # Inside section 1 only: stop at section 2, 3, ... (numbered headings)
                msec = numbered_section_re.match(line)
                if msec and int(msec.group(1)) >= 2:
                    break

                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.upper().startswith('CONFIGURATION') and 'EXCITATION' in stripped.upper():
                    continue
                if _is_rule_line(stripped):
                    continue

                row = _parse_row(line)
                if row is not None:
                    excitations.append(row)

        n_levels = len(excitations)
        n_states = sum(_nist_degeneracy_to_count(exc.get("degeneracy")) for exc in excitations)
        print(
            f"Read {n_levels} excitations from {filename} (section 1 only); "
            f"total NIST states (sum of degeneracies) = {n_states}"
        )
        return excitations

    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []
        
def find_files(filename = None):
    if filename is not None:
        file_name = filename
    else:
        cwd = os.getcwd()
        dmc_files = glob.glob(f'{cwd}/*_dmc_cas_*_nelecas_*.hdf5')

        if dmc_files:
            print("Found DMC files:")
            for f in dmc_files:
                print(f)
            # Use the first matching file
            file_name = dmc_files[0]
        else:
            print("No DMC files found matching pattern")
            exit()

    print(f'Will analyze {file_name}')
    return file_name

def extract_system_info():
    """
    Extract system information from directory path and HDF5 files.
    Returns a dictionary with atom, basis, CAS info, etc.
    """
    info = {
        'atom': 'Unknown',
        'basis': 'Unknown',
        'cas_type': 'Unknown',
        'ncas': 'Unknown',
        'nelecas': 'Unknown',
        'directory': os.getcwd()
    }
    
    # Try to extract from directory path
    cwd = os.getcwd()
    path_parts = cwd.split('/')
    
    # Look for atom info (e.g., "9-F" or similar patterns)
    for part in path_parts:
        if '-' in part and len(part) <= 5:
            # Could be atom identifier like "9-F"
            if part.split('-')[-1].isalpha() and len(part.split('-')[-1]) <= 2:
                info['atom'] = part.split('-')[-1]
        # Look for basis set (e.g., "vqz", "vdz", "vtz")
        if 'vqz' in part.lower() or 'vdz' in part.lower() or 'vtz' in part.lower():
            info['basis'] = part
        # Look for CAS type
        if 'casci' in part.lower():
            info['cas_type'] = 'CASCI'
        elif 'casscf' in part.lower():
            info['cas_type'] = 'CASSCF'
    
    # Try to extract from HDF5 files
    try:
        # Look for CI checkfile
        ci_files = glob.glob('*_ci_*.hdf5') + glob.glob('*ci*.hdf5')
        if ci_files:
            with hdf5_snapshot_read(ci_files[0], snapshot=False) as f:
                if 'ci' in f:
                    if 'ncas' in f['ci']:
                        info['ncas'] = int(f['ci/ncas'][()])
                    if 'nelecas' in f['ci']:
                        nelecas = f['ci/nelecas'][:]
                        info['nelecas'] = tuple(nelecas) if len(nelecas) == 2 else int(nelecas[0])
        
        # Look for SCF checkfile for basis info
        scf_files = glob.glob('*_atom_basis_*.hdf5') + glob.glob('*scf*.hdf5')
        if scf_files:
            # Try to extract basis from filename
            for fname in scf_files:
                if 'basis' in fname:
                    parts = fname.split('_')
                    for i, part in enumerate(parts):
                        if 'basis' in part.lower() and i + 1 < len(parts):
                            info['basis'] = parts[i + 1]
                            break
    except Exception as e:
        print(f"Warning: Could not extract info from HDF5 files: {e}")
    
    return info

# Max rows in CALCULATED vs NIST table (expanded microstate indices 0..N-1; same as NIST eigenvalues).
NIST_COMPARISON_CAP = 100

def generate_eig_convergence_report(results, hmf, nist_data_sorted, system_info, det_list, 
                                     report_filename='eig_convergence_report.txt', step_info=None,
                                     nist_file_order=False):
    """
    Generate a comprehensive text report of eigenvalue convergence data.

    """
    with open(report_filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("EIGENVALUE CONVERGENCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # System Information
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Atom:                    {system_info['atom']}\n")
        f.write(f"Basis Set:               {system_info['basis']}\n")
        f.write(f"CAS Type:                 {system_info['cas_type']}\n")
        f.write(f"CAS Size (ncas):          {system_info['ncas']}\n")
        f.write(f"Active Electrons:          {system_info['nelecas']}\n")
        if step_info is not None:
            f.write(f"DMC steps (total):         {step_info['n_total']}\n")
            f.write(f"DMC discard (equil.):     {step_info['n_discard']}\n")
            f.write(f"DMC steps (averaging):     {step_info['n_used']}\n")
            f.write(
                f"Reblocking:                {step_info['n_blocks']} blocks × "
                f"{step_info['block_size']} steps/block\n"
            )
        f.write(f"Working Directory:        {system_info['directory']}\n")
        f.write(f"Report Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if step_info is not None and 'cbs' in step_info:
            c = step_info['cbs']
            f.write("CBS-ε(DZ) on HMF diagonal: ")
            if c.get('applied'):
                f.write('yes\n')
                f.write(
                    f"  (ref det index iref = {c.get('iref')}, "
                    f"max|Δdiag| = {c.get('max_abs_delta', 0):.6e} Ha)\n"
                )
            else:
                f.write("no\n")
            if c.get('message') and not c.get('applied'):
                f.write(f"  reason: {c['message']}\n")
        f.write("\n")
        
        # HMF Eigenvalues
        f.write("HMF (MEAN FIELD) EIGENVALUES\n")
        f.write("-" * 80 + "\n")
        eig_hmf = np.sort(np.diag(hmf))
        eig_hmf_normalized = eig_hmf - eig_hmf[0]
        f.write(f"Total HMF eigenvalues:    {len(eig_hmf)}\n")
        f.write(f"Ground state energy:      {eig_hmf[0]:.10f} Ha\n")
        f.write("\nHMF Eigenvalues (relative to ground state, Ha):\n")
        f.write(f"{'Index':<8} {'Energy (Ha)':<20} {'Energy (eV)':<20}\n")
        f.write("-" * 48 + "\n")
        for idx, e in enumerate(eig_hmf):
            f.write(f"{idx:<8} {e:<20.10f} {e * 27.211386245988:<20.10f}\n")
        f.write("\n")
        
        # Results for each determinant size
        f.write("EIGENVALUE CONVERGENCE BY DETERMINANT SIZE\n")
        f.write("-" * 80 + "\n")
        for det_size in sorted(results.keys()):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Determinant Size: {det_size + 1}\n")
            f.write(f"{'=' * 80}\n")
            
            result = results[det_size]
            eig = np.array(result['eigenvalues'])
            eig_normalized = np.array(result['eigenvalues_normalized'])
            eig_std = np.array(result['eigenvalues_std']) if 'eigenvalues_std' in result else None
            
            f.write(f"Total eigenvalues:        {len(eig)}\n")
            f.write(f"Ground state energy:       {eig[0]:.10f} Ha\n")
            f.write(f"Sum of overlap diagonal:  {np.sum(np.diag(np.array(result['ovlp']))):.10f}\n")
            f.write("\n")
            
            f.write("Eigenvalues (relative to ground state, Ha):\n")
            has_errors = eig_std is not None
            if has_errors:
                f.write(f"{'Index':<8} {'Energy (Ha)':<20} {'Error (Ha)':<15} {'Energy (eV)':<20}\n")
                f.write("-" * 63 + "\n")
                for idx, e in enumerate(eig_normalized):
                    err = eig_std[idx] if idx < len(eig_std) else 0
                    f.write(f"{idx:<8} {e:<20.10f} {err:<15.6e} {e * 27.211386245988:<20.10f}\n")
            else:
                f.write(f"{'Index':<8} {'Energy (Ha)':<20} {'Energy (eV)':<20}\n")
                f.write("-" * 48 + "\n")
                for idx, e in enumerate(eig_normalized):
                    f.write(f"{idx:<8} {e:<20.10f} {e * 27.211386245988:<20.10f}\n")
            f.write("\n")
            
            # Matrix statistics
            ovlp = np.array(result['ovlp'])
            delta = np.array(result['delta'])
            sigma = np.array(result['sigma'])
            h = np.array(result['h'])
            
            f.write("Matrix Statistics:\n")
            f.write(f"  Overlap matrix (S):\n")
            f.write(f"    Shape:                 {ovlp.shape}\n")
            f.write(f"    Trace:                 {np.trace(ovlp):.10f}\n")
            f.write(f"    Diagonal sum:          {np.sum(np.diag(ovlp)):.10f}\n")
            f.write(f"    Condition number:      {np.linalg.cond(ovlp):.2e}\n")
            f.write(f"  Delta matrix:\n")
            f.write(f"    Shape:                 {delta.shape}\n")
            f.write(f"    Trace:                 {np.trace(delta):.10f}\n")
            f.write(f"    Max abs value:         {np.max(np.abs(delta)):.10f}\n")
            f.write(f"  Sigma matrix (S^-1 * Delta):\n")
            f.write(f"    Shape:                 {sigma.shape}\n")
            f.write(f"    Trace:                 {np.trace(sigma):.10f}\n")
            f.write(f"    Max abs value:         {np.max(np.abs(sigma)):.10f}\n")
            f.write(f"  Hamiltonian (HMF + Sigma):\n")
            f.write(f"    Shape:                 {h.shape}\n")
            f.write(f"    Trace:                 {np.trace(h):.10f}\n")
            f.write(f"    Ground state:          {np.min(np.diag(h)):.10f} Ha\n")
            f.write("\n")
        
        # NIST Reference Data
        if nist_data_sorted is not None and len(nist_data_sorted) > 0:
            f.write("NIST REFERENCE DATA\n")
            f.write("-" * 80 + "\n")
            nist_energies = np.array([state['energy'] for state in nist_data_sorted])
            nist_energies_normalized = nist_energies - nist_energies[0]
            n_nist_levels = len(nist_data_sorted)
            n_nist_microstates = sum(
                _nist_degeneracy_to_count(exc.get("degeneracy", 1)) for exc in nist_data_sorted
            )
            f.write(f"NIST levels (unique terms):     {n_nist_levels}\n")
            f.write(f"NIST microstates (sum of Deg): {n_nist_microstates}\n")
            nist_order_label = (
                "nist_data.txt section 1 (file order)"
                if nist_file_order
                else "ascending NIST energy"
            )
            f.write(f"NIST level ordering:            {nist_order_label}\n")
            f.write(
                f"(Comparison table below uses microstate indices; each level is replicated by its Deg column.)\n"
            )
            f.write(f"Ground state energy:      {nist_energies[0]:.10f} Ha\n")
            f.write("\n")
            f.write("NIST Energies (relative to ground state, Ha):\n")
            f.write(f"{'Index':<8} {'Energy (Ha)':<20} {'Energy (eV)':<20} {'Term':<15} {'Config':<20} {'Deg':<8}\n")
            f.write("-" * 91 + "\n")
            for idx, exc in enumerate(nist_data_sorted):
                e = nist_energies_normalized[idx]
                term = exc.get('term_symbol', 'N/A')
                config = exc.get('configuration', 'N/A')
                deg = exc.get('degeneracy', 'N/A')
                f.write(f"{idx:<8} {e:<20.10f} {e * 27.211386245988:<20.10f} "
                       f"{str(term):<15} {str(config):<20} {str(deg):<8}\n")
            f.write("\n")
        
        # Comparison Table
        if nist_data_sorted is not None and len(nist_data_sorted) > 0:
            f.write("COMPARISON: CALCULATED vs NIST\n")
            f.write("-" * 80 + "\n")
            # Compare largest determinant size with NIST
            if results:
                max_det_size = max(results.keys())
                result = results[max_det_size]
                eig_calc = np.array(result['eigenvalues_normalized'])
                calc_std = (
                    np.array(result['eigenvalues_std'], dtype=float)
                    if 'eigenvalues_std' in result
                    else None
                )
                has_calc_err = calc_std is not None and len(calc_std) > 0
                
                # Expand NIST data to account for degeneracy
                # Each state is replicated by its degeneracy number
                nist_energies_expanded = []
                nist_data_expanded = []
                for exc in nist_data_sorted:
                    energy = exc['energy']
                    energy_normalized = energy - nist_data_sorted[0]['energy']
                    degeneracy = _nist_degeneracy_to_count(exc.get("degeneracy", 1))

                    # Replicate this state by its degeneracy
                    for _ in range(degeneracy):
                        nist_energies_expanded.append(energy_normalized)
                        nist_data_expanded.append(exc)
                
                nist_energies_expanded = np.array(nist_energies_expanded)
                
                n_compare = min(len(eig_calc), len(nist_energies_expanded), NIST_COMPARISON_CAP)
                idx_hi = max(0, n_compare - 1)
                f.write(
                    f"Comparing first {n_compare} microstates (NIST expanded by degeneracy; "
                    f"row indices 0..{idx_hi}; cap {NIST_COMPARISON_CAP}); "
                    f"largest determinant size {max_det_size + 1}:\n"
                )
                if has_calc_err:
                    f.write(
                        f"{'State':<8} {'Calc (Ha)':<18} {'σ_calc (Ha)':<14} {'NIST (Ha)':<18} "
                        f"{'Δ (Ha)':<18} {'Δ (eV)':<18} {'NIST Term':<15} {'NIST Deg':<10}\n"
                    )
                    f.write("-" * 128 + "\n")
                else:
                    f.write(
                        f"{'State':<8} {'Calculated (Ha)':<20} {'NIST (Ha)':<20} "
                        f"{'Difference (Ha)':<20} {'Difference (eV)':<20} {'NIST Term':<15} {'NIST Deg':<10}\n"
                    )
                    f.write("-" * 113 + "\n")
                for i in range(n_compare):
                    calc_e = eig_calc[i] if i < len(eig_calc) else 'N/A'
                    sig_e = (
                        float(calc_std[i])
                        if has_calc_err and i < len(calc_std)
                        else None
                    )
                    nist_e = nist_energies_expanded[i] if i < len(nist_energies_expanded) else 'N/A'
                    nist_term = nist_data_expanded[i].get('term_symbol', 'N/A') if i < len(nist_data_expanded) else 'N/A'
                    nist_deg = nist_data_expanded[i].get('degeneracy', 'N/A') if i < len(nist_data_expanded) else 'N/A'
                    
                    if isinstance(calc_e, (int, float)) and isinstance(nist_e, (int, float)):
                        diff = calc_e - nist_e
                        if has_calc_err:
                            sig_str = f"{sig_e:.6e}" if sig_e is not None else "N/A"
                            f.write(
                                f"{i:<8} {calc_e:<18.10f} {sig_str:<14} {nist_e:<18.10f} "
                                f"{diff:<18.10f} {diff * 27.211386245988:<18.10f} "
                                f"{str(nist_term):<15} {str(nist_deg):<10}\n"
                            )
                        else:
                            f.write(
                                f"{i:<8} {calc_e:<20.10f} {nist_e:<20.10f} "
                                f"{diff:<20.10f} {diff * 27.211386245988:<20.10f} "
                                f"{str(nist_term):<15} {str(nist_deg):<10}\n"
                            )
                    else:
                        if has_calc_err:
                            sig_str = f"{sig_e:.6e}" if sig_e is not None else "N/A"
                            f.write(
                                f"{i:<8} {str(calc_e):<18} {sig_str:<14} {str(nist_e):<18} "
                                f"{'N/A':<18} {'N/A':<18} {str(nist_term):<15} {str(nist_deg):<10}\n"
                            )
                        else:
                            f.write(
                                f"{i:<8} {str(calc_e):<20} {str(nist_e):<20} {'N/A':<20} {'N/A':<20} "
                                f"{str(nist_term):<15} {str(nist_deg):<10}\n"
                            )
                f.write("\n")
                report_dir = os.path.dirname(os.path.abspath(report_filename))

        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Determinant sizes tested:  {sorted(det_list)}\n")
        total_dets = hmf.shape[0]
        f.write(f"Total number of determinants: {total_dets}\n")
        f.write(f"Report file:               {report_filename}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report saved to: {report_filename}")

def _compute_det_prod_filter(mol, mf, symm_data, occupations):
    """
    Compute the determinant symmetry mask (ndets, ndets) where mask[l,n]=True
    iff matrix element <l|O|n> can be nonzero (same total irrep for both spins).

    Args:
        mol: pyscf Mole object
        mf: pyscf mean-field object (for mo_coeff)
        symm_data: dict from BosonWF.symm_utils with 'matrix', 'irrep_to_idx'
        occupations: list of (occ_up, occ_dn) pairs; each is list/array of orbital indices

    Returns:
        (ndets, ndets) boolean array
    """
    from pyscf import symm

    prod_matrix = symm_data["matrix"]
    irrep_to_idx = symm_data["irrep_to_idx"]
    idx_to_irrep = {v: k for k, v in irrep_to_idx.items()}

    mo_coeff = mf.mo_coeff
    if len(mo_coeff.shape) == 2:
        mo_up = mo_coeff
        mo_dn = mo_coeff
    else:
        mo_up = mo_coeff[0]
        mo_dn = mo_coeff[1]

    up_orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_up)
    down_orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_dn)

    ndets = len(occupations)

    def get_prod(occ, orbsym):
        prod = 0
        for orb in occ:
            irrep_name = orbsym[orb]
            idx = irrep_to_idx.get(irrep_name)
            if idx is None:
                raise ValueError(f"Orbital irrep {irrep_name} not in character table")
            prod = prod_matrix[prod, idx]
        return prod

    det_prod = []
    det_prod_up = np.zeros(ndets, dtype=int)
    det_prod_dn = np.zeros(ndets, dtype=int)
    for i in range(ndets):
        occ_up, occ_dn = occupations[i]
        det_prod_up[i] = get_prod(occ_up, up_orbsym)
        det_prod_dn[i] = get_prod(occ_dn, down_orbsym)
        det_prod.append(idx_to_irrep[prod_matrix[det_prod_up[i], det_prod_dn[i]]])
    return det_prod

def plot_eig_convergence(delta, ovlp, hmf, nist_data, plot_evolution=True, plot_filename='eig_convergence.png',
                         use_filtered=True, show_error_bars=True, n_mc_samples=1000, filter_factor=10.0, no_plot=False,
                         step_info=None, cbs_shifts_1d=None, nist_file_order=False):
    """
    Plot eigenvalue convergence with optional error bars.
    Uses block bootstrap when block_means available (accounts for autocorrelation);
    falls back to Monte Carlo propagation when only mean/std available.
    
    Parameters
    ----------
    delta, ovlp : array or dict
        If dict: must contain 'filtered_mean' or 'mean'; 'block_means' for block bootstrap,
        or 'std' for Monte Carlo fallback.
    use_filtered : bool
        If True and dict passed, use filtered_mean; else use mean.
    show_error_bars : bool
        If True, compute error bars (block bootstrap preferred over Monte Carlo).
    n_mc_samples : int
        Number of Monte Carlo samples when block_means not available.
    filter_factor : float
        Factor for filter mask: entries with std > factor*|mean| are zeroed.
    cbs_shifts_1d : ndarray, optional
        Length ``ndet`` = ``hmf.shape[0]``, CBS-ε(DZ) hole/particle shift per
        determinant, in the **same row order** as ``hmf`` (after
        ``ordered_matrices``). If given, shown together with the sorted HMF
        reference curve on ``eig_convergence_*.png``.
    nist_file_order : bool
        If True, keep NIST levels in ``nist_data.txt`` section-1 file order.
        If False (default), sort levels by ascending NIST energy before
        normalization, plotting, and the comparison table.
    """
    # Handle both array and dict inputs
    block_keep_mask = None
    if isinstance(delta, dict) and isinstance(ovlp, dict):
        delta_block_means = delta.get('block_means', None)
        ovlp_block_means = ovlp.get('block_means', None)
        use_block_filtering = True
        if use_block_filtering and delta_block_means is not None and getattr(delta_block_means, "ndim", 0) == 3:
            from matplotlib.patches import Patch

            _k = 3.0
            _dbm = delta_block_means
            _obm = ovlp_block_means
            
            _dbm_mean = np.mean(_dbm, axis=0)
            _dbm_std = np.std(_dbm, axis=0, ddof=1)
            _dbm_diff = _dbm - _dbm_mean
            _dbm_filt = (np.abs(_dbm_diff) > _dbm_std * _k).any(axis=(1, 2))
            _bfilt = _dbm_filt
            _valid = _dbm_std > 0
            _nblk = _dbm.shape[0]
            # _obm_mean = np.mean(_obm, axis=0)
            # _obm_std = np.std(_obm, axis=0, ddof=1)
            # _obm_diff = _obm - _obm_mean
            # _obm_bfilt = (np.abs(_obm_diff) > _obm_std * _k).any(axis=(1, 2))

            # _bfilt = np.logical_or(_dbm_filt, _obm_bfilt)
            # _valid = np.logical_and(_dbm_std > 0, _obm_std > 0)
            # _nblk = _dbm.shape[0]

            _w = max(8.0, _nblk * 0.25)
            _fig_dbg, _ax_dbg = plt.subplots(figsize=(_w, 5))
            _rng = np.random.default_rng(0)
            for _i in range(_nblk):
                _vals = _dbm[_i][_valid].ravel()
                if _vals.size == 0:
                    continue
                _jit = _rng.uniform(-0.2, 0.2, size=_vals.size)
                _ax_dbg.scatter(
                    _i + _jit,
                    _vals,
                    s=12,
                    c=("tab:red" if _bfilt[_i] else "tab:blue"),
                    alpha=0.45,
                    edgecolors="none",
                )
            for _i in range(_nblk):
                if _bfilt[_i]:
                    _ax_dbg.axvspan(_i - 0.45, _i + 0.45, color="tab:red", alpha=0.08, zorder=0)
            _ref = _dbm_mean[_valid].ravel()
            if _ref.size:
                _ax_dbg.axhline(float(np.mean(_ref)), color="k", ls="--", lw=0.8, alpha=0.6)
            _ax_dbg.set_xlabel("block index")
            _ax_dbg.set_ylabel(r"$\Delta$ block mean (flattened, std>0 cells only)")
            _ax_dbg.set_xticks(range(_nblk))
            _ax_dbg.legend(
                handles=[
                    Patch(facecolor="tab:blue", alpha=0.5, label="kept"),
                    Patch(facecolor="tab:red", alpha=0.5, label="discarded"),
                ],
            )
            _ax_dbg.set_title("DEBUG: delta block means vs outlier filter")
            _fig_dbg.tight_layout()
            _dbg_path = os.path.join(os.getcwd(), "debug_block_filter_delta.png")
            _fig_dbg.savefig(_dbg_path, dpi=150)
            plt.close(_fig_dbg)
            print(f"DEBUG: block filter diagnostic saved to {_dbg_path}")
            # exit()
            _inv_bfilt = np.logical_not(_bfilt)
            block_keep_mask = _inv_bfilt
            num_kept = int(np.sum(_inv_bfilt))
            num_discarded = int(np.sum(_bfilt))
            print(
                f"Block outlier filter: kept {num_kept}/{_nblk} blocks "
                f"({num_discarded} discarded)"
            )
            delta_mean = np.mean(delta_block_means[_inv_bfilt], axis=0)
            delta_std = np.std(delta_block_means[_inv_bfilt], axis=0, ddof=1)
            ovlp_mean = np.mean(ovlp_block_means[_inv_bfilt], axis=0)
            ovlp_std = np.std(ovlp_block_means[_inv_bfilt], axis=0, ddof=1)
        # --- end DEBUG ---
        else:
            delta_mean = delta['filtered_mean'] if use_filtered else delta['mean']
            delta_std = delta.get('std', None)
            ovlp_mean = ovlp['filtered_mean'] if use_filtered else ovlp['mean']
            ovlp_std = ovlp.get('std', None)

        
    else:
        delta_mean = delta
        delta_std = None
        delta_block_means = None
        ovlp_mean = ovlp
        ovlp_std = None
        ovlp_block_means = None        

    if delta_block_means is not None and ovlp_block_means is not None:
        if block_keep_mask is not None:
            delta_block_means_boot = delta_block_means[block_keep_mask]
            ovlp_block_means_boot = ovlp_block_means[block_keep_mask]
        else:
            delta_block_means_boot = delta_block_means
            ovlp_block_means_boot = ovlp_block_means
    else:
        delta_block_means_boot = None
        ovlp_block_means_boot = None

    # Prefer block bootstrap when block_means available (n_blocks >= 2)
    use_block_bootstrap = (
        show_error_bars and delta_block_means_boot is not None and ovlp_block_means_boot is not None
        and delta_block_means_boot.shape[0] >= 2
    )
    run_mc = (
        show_error_bars and not use_block_bootstrap
        and delta_std is not None and ovlp_std is not None and n_mc_samples > 0
    )
    if use_block_bootstrap:
        n_blocks = delta_block_means_boot.shape[0]
        if block_keep_mask is not None:
            n_total = delta_block_means.shape[0]
            print(
                f"Block bootstrap error bars: {n_blocks} kept blocks "
                f"({n_total - n_blocks} discarded of {n_total})"
            )
        else:
            print(f"Block bootstrap error bars: {n_blocks} blocks")
    elif run_mc:
        delta_mask = delta_std > filter_factor * np.abs(delta_mean)
        ovlp_mask = ovlp_std > filter_factor * np.abs(ovlp_mean)
        print(f"Monte Carlo error bars: {n_mc_samples} samples")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    num_det = delta_mean.shape[0] - 1

    results = {}
    ndet = delta_mean.shape[0] - 1
    # Find unique values in the diagonal of hmf within tolerance of 0.001 Ha and use their indices
    # Keep only the largest (last) index for each unique value group
    hmf_diag = np.diag(hmf)
    tolerance = 0.001  # Ha
    # Since hmf is ordered, find where consecutive differences exceed tolerance
    # When diff[i] > tolerance, hmf_diag[i] is the last element of its group
    diffs = np.diff(hmf_diag)
    # Find indices where difference exceeds tolerance
    # These indices mark the last element of each unique value group (except the last group)
    boundary_indices = np.where(diffs > tolerance)[0]
    unique_indices = list(boundary_indices)  # These are the last indices of each group
    # Always include the last index of the array (last element of the last group)
    unique_indices.append(len(hmf_diag) - 1)
    det_list = sorted(list(set(unique_indices)))  # Use set to remove duplicates, then sort
    keep_first = 5
    keep_rest = 5
    if len(det_list) > 15:
        first_n = det_list[:keep_first]
        rest = det_list[keep_first:]
        # Sample 5 from rest with equal index spacing, including the last
        indices = np.linspace(0, len(rest) - 1, keep_rest, dtype=int)
        sampled = [rest[i] for i in indices]
        det_list = first_n + sampled
    # Ensure min_size and ndet are included
    if ndet not in det_list:
        det_list.append(ndet)
    det_list = sorted(det_list)
    print("det_list:", det_list)
    eigv_heatmaps = []
    ovlp_heatmaps = []
    for i in det_list:
        try:
            indices = np.ix_(range(i+1), range(i+1))
            sinv = robust_matrix_inverse(ovlp_mean[indices])
            print('Sum of ovlp diagonal: ', np.sum(np.diag(ovlp_mean[indices])))
            sigma = np.matmul(sinv, delta_mean[indices])
            h = hmf[indices] + sigma
            eig, eigv = np.linalg.eig(h)
            
            print('Largest complex component in eigenvalues: ', np.max(np.abs(eig.imag)))
            if np.max(np.abs(eig.imag)) > 1e-5:
                print(f"Warning: Large complex component in eigenvalues: max imaginary part = {np.max(np.abs(eig.imag)):.3e}")
            eig = eig.real
            # Sort eigenvalues and corresponding eigenvectors
            sort_idx = np.argsort(eig)
            eig = eig[sort_idx]
            eigv = eigv[:, sort_idx]
            eigv_heatmaps.append(eigv)
            ovlp_heatmaps.append(ovlp_mean[indices])
            # Use level shifting to make n lowest eigenvalues zero 
            # n here is equal to the number of ground state determinants in hmf
            e_tol = 1e-3 # tolerance for level shifting
            n = np.sum(np.diag(hmf)-np.diag(hmf)[0] < e_tol)
            eign = np.min(eig) - eig[:n]
            eigvn = eigv[:, :n]

            h_new = h + np.einsum('ij,jk,j->ik', eigvn, eigvn.conj().T, eign)
            eig_sh, eigv_sh = np.linalg.eig(h_new)

            eig_sh = eig_sh.real
            sort_idx = np.argsort(eig_sh)
            eig_sh = eig_sh[sort_idx]
            eigv_sh = eigv_sh[:, sort_idx]
            
            eig_normalized = eig - eig[0]
            # Error bars: block bootstrap (preferred) or Monte Carlo
            eig_std = None
            if use_block_bootstrap:
                eig_samples = []
                for b in range(delta_block_means_boot.shape[0]):
                    try:
                        delta_b = delta_block_means_boot[b][indices]
                        ovlp_b = ovlp_block_means_boot[b][indices]
                        sinv_b = robust_matrix_inverse(ovlp_b)
                        sigma_b = np.matmul(sinv_b, delta_b)
                        h_b = hmf[indices] + sigma_b
                        eig_b, _ = np.linalg.eig(h_b)
                        eig_b = np.sort(eig_b.real)
                        eig_b_norm = eig_b - eig_b[0]
                        eig_samples.append(eig_b_norm)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        continue
                if len(eig_samples) >= 2:
                    eig_samples = np.array(eig_samples)
                    n_blocks_used = len(eig_samples)
                    eig_std = np.std(eig_samples, axis=0, ddof=1) / np.sqrt(n_blocks_used)
            elif run_mc:
                eig_samples = []
                for _ in range(n_mc_samples):
                    delta_samp = np.where(delta_mask[indices], 0, delta_mean[indices] + delta_std[indices] * np.random.randn(*(delta_mean[indices].shape)))
                    ovlp_samp = np.where(ovlp_mask[indices], 0, ovlp_mean[indices] + ovlp_std[indices] * np.random.randn(*(ovlp_mean[indices].shape)))
                    try:
                        sinv_samp = robust_matrix_inverse(ovlp_samp)
                        sigma_samp = np.matmul(sinv_samp, delta_samp)
                        h_samp = hmf[indices] + sigma_samp
                        eig_samp, _ = np.linalg.eig(h_samp)
                        eig_samp = np.sort(eig_samp.real)
                        eig_samp_norm = eig_samp - eig_samp[0]
                        eig_samples.append(eig_samp_norm)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        continue
                if eig_samples:
                    eig_samples = np.array(eig_samples)
                    eig_std = np.std(eig_samples, axis=0, ddof=1)
            
            if eig_std is not None:
                x_vals = np.arange(len(eig_normalized))
                ax.errorbar(x_vals, eig_normalized, yerr=eig_std, fmt='-o', label=f'{i+1} dets',
                           markersize=4, linewidth=1.5, alpha=0.7, capsize=2, capthick=1)
            else:
                ax.plot(eig_normalized, '-o', label=f'{i+1} dets', markersize=4, linewidth=1.5, alpha=0.7)
            results[i] = {
                'ovlp': ovlp_mean[indices].tolist(),
                'delta': delta_mean[indices].tolist(),
                'h': h.tolist(),
                'h_shifted': h_new.tolist(),
                'sinv': sinv.tolist(),
                'sigma': sigma.tolist(),
                'eigenvalues': eig.tolist(),
                'eigenvalues_shifted': eig_sh.tolist(),
                'eigenvalues_normalized': eig_normalized.tolist(),
                'eigenvectors': eigv.tolist(),
            }
            if eig_std is not None:
                results[i]['eigenvalues_std'] = eig_std.tolist()
        except:
            print(f'Error for {i+1} determinants')
            continue         
    # Extract NIST reference energies if available
    
    nist_data_sorted = None
    if nist_data is not None and len(nist_data) > 0:
        if nist_file_order:
            nist_data_sorted = list(nist_data)
            print("\nNIST level ordering: nist_data.txt section 1 (file order)")
        else:
            nist_data_sorted = sorted(nist_data, key=lambda x: x['energy'])
            print("\nNIST level ordering: ascending energy")
        nist_energies = np.array([state['energy'] for state in nist_data_sorted])
        # Normalize to ground state (first row in ordered list)
        nist_energies_normalized = nist_energies - nist_energies[0]
        print(f"NIST reference data: {len(nist_data_sorted)} states")
        print(f"NIST energies (relative to ground state, Ha):")
        for i, exc in enumerate(nist_data_sorted):
            e = nist_energies_normalized[i]
            term = exc.get('term_symbol', '')
            config = exc.get('configuration', '')
            deg = exc.get('degeneracy', '')
            label_parts = [p for p in [term, config, f'g={deg}'] if p]
            label = ', '.join(label_parts) if label_parts else f'State {i+1}'
            print(f"  {i}: {e:.6f} Ha - {label}")
    
    # if delta_mean.shape[0] > 10:
    #     increments = int(delta_mean.shape[0]/10)
    # else:
    #     increments = 1
    
    diag = np.diag(hmf)
    order = np.argsort(diag)
    eig_hmf = diag[order]
    eig_hmf_normalized = eig_hmf - eig_hmf[0]
    ax.plot(
        np.arange(len(eig_hmf_normalized)),
        eig_hmf_normalized,
        '-o',
        label='HMF',
        markersize=5,
        linewidth=2,
        color='black',
    )
    if (
        cbs_shifts_1d is not None
        and getattr(cbs_shifts_1d, 'size', 0) == hmf.shape[0]
    ):
        cbs_sorted = np.asarray(cbs_shifts_1d, dtype=np.float64)[order]
        cbs_plot = cbs_sorted - cbs_sorted[0]
        ax.plot(
            np.arange(len(cbs_plot)),
            cbs_plot,
            '-s',
            label=r'CBS-$\varepsilon$(DZ) shift (diag)',
            markersize=4,
            linewidth=1.8,
            color='tab:green',
            alpha=0.9,
        )
    # Plot NIST reference energies if available
    xlim = ax.get_xlim()
    if xlim[1] > 100:
        xlim = (0, 50)
        ax.set_xlim(xlim)
        xlim = ax.get_xlim()
        deltaeig = eig_hmf_normalized[48] - eig_hmf_normalized[0]
        ymax = deltaeig * 1.1
        ylim = (0, ymax)
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()
        ymax = ylim[1]
    x_max = xlim[1]

    if nist_data_sorted is not None:
        # Get the x-axis limits for annotation positioning
        
        # Filter NIST states with normalized energies below max eigenvalue + 0.5
        max_eig_hmf_norm = np.max(eig_hmf_normalized)
        nist_filtered = [exc for exc in nist_data_sorted if (exc['energy'] - nist_data_sorted[0]['energy']) <= (max_eig_hmf_norm + 0.5)]
        
        # Enhanced filtering: exclude double excitations and high-n orbitals
        # First, get ground state configuration
        ground_config = nist_data_sorted[0].get('configuration', '') if nist_data_sorted else ''
        
        # Filter out double excitations and high-n orbitals
        excluded_configurations = ['d', '4s', '4p', '5s', '5p', '6s', '6p', '7s', '7p', '8s', '8p', '9s', '9p', 'f', 'g', 'h']
        nist_filtered_enhanced = []
        for exc in nist_filtered:
            config = exc.get('configuration', '')
            # Skip if contains excluded orbital types
            if any(exc_orb in config for exc_orb in excluded_configurations):
                continue
            # Check if it's a single excitation (if we can determine)
            if ground_config:
                is_single = is_nist_single_excitation(config, ground_config)
                if is_single is False:  # Explicitly a double or higher excitation
                    continue
            nist_filtered_enhanced.append(exc)
        nist_filtered = nist_filtered_enhanced
        # n_states_to_plot = len(nist_filtered)
        
        texts = []
        prev_energy = 0
        for exc in nist_filtered:
            energy = exc['energy']
            # Normalize to ground state (first NIST energy)
            energy_normalized = energy - nist_data_sorted[0]['energy']
            if energy_normalized < ymax:
                degeneracy = exc.get('degeneracy', '')
                term_symbol = exc.get('term_symbol', '')
                configuration = exc.get('configuration', '')
                
                # Draw horizontal dashed line for this excitation
                ax.axhline(y=energy_normalized, color='r', linestyle=':', alpha=0.5, linewidth=1)
                
                # Create annotation text with all available information
                annotation_parts = []
                if degeneracy:
                    annotation_parts.append(f'g={degeneracy}')
                if term_symbol:
                    annotation_parts.append(term_symbol)
                if configuration:
                    annotation_parts.append(configuration)
                annotation_text = ', '.join(annotation_parts) if annotation_parts else ''
                
                # Annotate on the right side of the plot
                if annotation_text:
                    if energy_normalized - prev_energy < 0.02 and prev_energy != 0:
                        energy_text = prev_energy + 0.02
                    else:
                        energy_text = energy_normalized
                    texts.append(ax.text(x_max + 0.05 * (x_max - xlim[0]), energy_text, 
                                        annotation_text, fontsize=8, color='red', alpha=0.7, 
                                        ha='left', va='center'))
                    prev_energy = energy_text
        
        # Use adjust_text to avoid overlapping annotations
        # if texts:
        #     adjust_text(texts, 
        #                x=[x_max + 0.05 * (x_max - xlim[0])] * len(texts),
        #                y=[exc['energy'] - nist_data_sorted[0]['energy'] for exc in nist_filtered],
        #                arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.5),
        #                only_move={'objects': 'x'})
        
        # Add legend entry for NIST reference
        ax.plot([], [], 'r:', alpha=0.5, linewidth=1, label='NIST reference')
    
    ax.set_xlabel('State Index', fontsize=12)
    ax.set_ylabel('Energy (Ha, relative to ground state)', fontsize=12)
    ax.set_title('Eigenvalue Convergence with Determinant Size', fontsize=14)
    # ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    if eigv_heatmaps:
        from pyqmc import pyscftools
        from pyqmc.bosonslater import filter_determinants_from_ci
        from pyqmc.bosonslater import BosonWF
        from pyscf import fci
        from pyqmc.bosonslater import binary_to_occ
        import yaml

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        atom_name = config['atom_name']
        basis_name = config['basis_name']
        symm_tag = config['symm_tag']
        det_emax = config['det_emax']
        file_template = (
            '{}_atom_basis_{}_diffuse_S0P0D0_v1{}.hdf5 '  # scf
            '{}_ci_atom_basis_{}_diffuse_S0P0D0{}.hdf5 ' # ci
        )
        scf_chk, ci_chk = file_template.format(atom_name, basis_name, symm_tag, atom_name, basis_name, symm_tag).split()

        mol, mf, mc = pyscftools.recover_pyscf(scf_chk, ci_checkfile=ci_chk)

        # MO energies: often on mf after load; else from optional mf_inputs group; else Fock diagonal
        mo_energies = getattr(mf, "mo_energy", None)
        if mo_energies is None:
            inp = pyscftools.load_mf_inputs_from_hdf5(scf_chk, mol=mol)
            if inp is not None and "mo_energy" in inp:
                mo_energies = inp["mo_energy"]
        if mo_energies is None:
            fock_mo = mf.mo_coeff.conj().T @ mf.get_fock(dm=mf.make_rdm1()) @ mf.mo_coeff
            mo_energies = np.diag(fock_mo).real


        dets, saved = filter_determinants_from_ci(
            mc, mo_energies, det_emax, mol=mol, mf=mf
        )
        symm_data = BosonWF.symm_utils(mol, mol.groupname)
        deters_orig = fci.addons.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=-1)
        ncore = mc.ncore if hasattr(mc, "ncore") else 0
        alpha_occ = np.array([binary_to_occ(x[1], ncore)[0] for x in deters_orig])
        beta_occ = np.array([binary_to_occ(x[2], ncore)[0] for x in deters_orig])

        sorted_mask_indices = saved['sorted_mask_indices']
        occupations = [(alpha_occ[i], beta_occ[i]) for i in sorted_mask_indices]

        det_symm = _compute_det_prod_filter(mol, mf, symm_data, occupations)
    
        for i, (eigv_i, ovlp_i) in enumerate(zip(eigv_heatmaps, ovlp_heatmaps)):
            fig, axs = plt.subplots(2, 1, figsize=(8, 12))       
            eigv_i[np.abs(eigv_i) < 0.1] = 0
            # eigv_csr = csr_matrix(eigv)
            # n_components, labels = connected_components(eigv_csr, directed=False)
            # new_order = np.argsort(labels)
            # eigv_block = eigv[new_order, :][:, new_order]
            # MO based density matrix 
            # Print eigv as a readable file
            # Save eigv (real part) as a text file for inspection
            eigv_txt_filename = f"eigv_{i+1}.txt"
            # Format: row per vector, columns as floats
            with open(eigv_txt_filename, "w") as f_eigv:
                for row in eigv_i.real:
                    row_str = " ".join(f"{val: .6f}" for val in row)
                    f_eigv.write(row_str + "\n")
            eigv_size = eigv_i.shape[0]
            sns.heatmap(eigv_i.real, annot=False, cmap='viridis', ax=axs[0])
            axs[0].set_title(f'Eigenvector Heatmap |Determinants={i+1} | Original')
            # axs[0].set_ticklabels(np.arange(eigv_size))
            axs[0].set_yticks(np.arange(eigv_size))
            axs[0].set_yticklabels(det_symm[:eigv_size], rotation=0)
       
            sns.heatmap(ovlp_i.real, annot=False, cmap='viridis', ax=axs[1])
            axs[1].set_title(f'Eigenvector Heatmap |Determinants={i+1} | Blocked')
            # axs[1].set_xticklabels(np.arange(eigv_size))
            axs[1].set_yticks(np.arange(eigv_size))
            axs[1].set_yticklabels(det_symm[:eigv_size], rotation=0)
            plt.savefig(f'eigv_{i+1}.png')
            plt.close()
    # Generate comprehensive text report
    system_info = extract_system_info()
    report_filename = plot_filename.replace('.png', '_report.txt')
    generate_eig_convergence_report(results, hmf, nist_data_sorted, system_info, det_list, 
                                     report_filename=report_filename, step_info=step_info,
                                     nist_file_order=nist_file_order)
    
    if plot_evolution:   
        for i in results.keys():
            fig, axes = plt.subplots(8, 2, figsize=(6, 24))
            fig.suptitle(f'{i+1} determinants')
            ovlp = np.array(results[i]['ovlp'])
            
            sns.heatmap(ovlp, annot=False, cmap='viridis', ax=axes[0, 0])
            axes[0, 0].set_title(r'$\mathcal{S}$')

            sns.heatmap(ovlp-np.diag(np.diag(ovlp)), annot=False, cmap='viridis', ax=axes[0, 1])
            axes[0, 1].set_title(r'$\mathcal{S} - \mathcal{S}_{diag}\mathcal{I}$')

            sinv = np.array(results[i]['sinv'])
            sns.heatmap(sinv, annot=False, cmap='viridis', ax=axes[1, 0])
            axes[1, 0].set_title(r'$\mathcal{S}^{-1}$')
            sns.heatmap(sinv-np.diag(np.diag(sinv)), annot=False, cmap='viridis', ax=axes[1, 1])
            axes[1, 1].set_title(r'$\mathcal{S}^{-1} - \mathcal{S}^{-1}_{diag}\mathcal{I}$')

            delta = np.array(results[i]['delta'])
            sns.heatmap(delta, annot=False, cmap='viridis', ax=axes[2, 0])
            axes[2, 0].set_title(r'$\Delta$')

            sns.heatmap(delta-np.diag(np.diag(delta)), annot=False, cmap='viridis', ax=axes[2, 1])
            axes[2, 1].set_title(r'$\Delta - \Delta_{diag}\mathcal{I}$')

            sigma = np.array(results[i]['sigma'])
            sns.heatmap(sigma, annot=False, cmap='viridis', ax=axes[3, 0])
            axes[3, 0].set_title(r'$\Sigma$')
            sns.heatmap(sigma-np.diag(np.diag(sigma)), annot=False, cmap='viridis', ax=axes[3, 1])
            axes[3, 1].set_title(r'$\Sigma - \Sigma_{diag}\mathcal{I}$')

            h = np.array(results[i]['h'])
            sns.heatmap(h, annot=False, cmap='viridis', ax=axes[4, 0])
            axes[4, 0].set_title(r'H')
            sns.heatmap(h-np.diag(np.diag(h)), annot=False, cmap='viridis', ax=axes[4, 1])
            axes[4, 1].set_title(r'H - H_{diag}\mathcal{I}$')


            axes[5, 0].plot(np.diag(results[i]['ovlp']), '-o', label=r'$\mathcal{S}$ diagonal')
            axes[5, 0].set_title(r'$\mathcal{S}$ diagonal')

            h_shifted = np.array(results[i]['h_shifted'])
            sns.heatmap(h_shifted.real - h.real, annot=False, cmap='viridis', ax=axes[5, 1])
            axes[5, 1].set_title(r'H shifted minus H ')

            axes[6, 0].plot(np.diag(results[i]['sigma']), '-o', label=f'{i}' + r'$\Sigma$ diagonal')
            axes[6, 0].set_title(r'$\Sigma$ diagonal')        
            
            shifted_excitation_e = np.array(results[i]['eigenvalues_shifted']) - np.array(results[i]['eigenvalues_shifted'])[0]
            axes[6, 1].plot(shifted_excitation_e, '-o', label=f'{i} eigenvalues shifted')
            axes[6, 1].set_title(r'Eigenvalues shifted')

            axes[7, 0].plot(np.diag(results[i]['h']), '-o', label=f'{i}H diagonal')
            axes[7, 0].set_title(r'H diagonal')

            axes[7, 1].plot(results[i]['eigenvalues'], '-o', label=f'{i} eigenvalues')
            axes[7, 1].set_title(r'Eigenvalues')


            plt.savefig(f'delta_ovlp_h_sinv_evolution_{i}.png')
            plt.close()

def plot_delta_ovlp(delta, ovlp):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    import seaborn as sns
    sns.heatmap(delta['mean'], annot=False, cmap='viridis', ax=axes[0, 0])
    axes[0, 0].set_title(r'$\Delta$ mean')
    sns.heatmap(delta['std'], annot=False, cmap='viridis', ax=axes[0, 1])
    axes[0, 1].set_title(r'$\Delta$ std')
    sns.heatmap(ovlp['mean'], annot=False, cmap='viridis', ax=axes[1, 0])
    axes[1, 0].set_title(r'Overlap mean')
    sns.heatmap(ovlp['std'], annot=False, cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title(r'Overlap std')
    plt.savefig('delta_and_ovlp.png')
    plt.close()

def plot_evolution(full_data, indices = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6)], discard = 1000, chunk_size=20):
    if indices is None:
        delta_shape = full_data['delta'].shape
        indices = [(i, i) for i in range(delta_shape[1])]

    # Split into chunks of chunk_size (default 20) per file
    for chunk_idx in range(0, len(indices), chunk_size):
        chunk = indices[chunk_idx:chunk_idx + chunk_size]
        num_indices = len(chunk)
        fig, axes = plt.subplots(num_indices, 2, figsize=(10, 4*num_indices))
        axes = np.atleast_2d(axes)

        for i, index in enumerate(chunk):
            delta = full_data['delta'][:, index[0], index[1]]
            ovlp = full_data['ovlp'][:, index[0], index[1]]
            axes[i, 0].plot(delta, '-o', label=r'$\Delta$')
            axes[i, 0].axvline(x=discard, color='gray', linestyle='--', alpha=0.5)
            axes[i, 0].axhline(y=np.mean(delta[discard:]), color='gray', linestyle='--', alpha=0.5)
            axes[i, 0].set_title(r'$\Delta$' + f'({index[0]}, {index[1]})')
            axes[i, 1].plot(ovlp, '-o', label=r'$\mathcal{S}$')
            axes[i, 1].axvline(x=discard, color='gray', linestyle='--', alpha=0.5)
            axes[i, 1].axhline(y=np.mean(ovlp[discard:]), color='gray', linestyle='--', alpha=0.5)
            axes[i, 1].set_title(r'$\mathcal{S}$' + f'({index[0]}, {index[1]})')

        suffix = f'_{chunk_idx // chunk_size}' if len(indices) > chunk_size else ''
        plt.savefig(f'evolution_delta_and_ovlp{suffix}.png')
        plt.close()

def get_discard(num_steps):
    """
    Determine number of initial steps to discard for equilibration.
    
    Parameters
    ----------
    num_steps : int or None
        Total number of steps in the simulation
        
    Returns
    -------
    int
        Number of steps to discard
    """
    if num_steps is None:
        return 500  # Default for unknown number of steps
    elif num_steps < 1000:
        return int(num_steps / 2)  # Discard half for short runs
    else:
        # For longer runs, discard 1/3 of steps, capped at 5000
        return min(int(num_steps / 3), DISCARD_STEPS)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot DMC results with optional plotting control.')
    parser.add_argument('--no-plot', action='store_true', help='If specified, disables all plotting functions (headless only).')
    parser.add_argument('--hmf-file', type=str, default='hmf.hdf5', help='Path to hmf.hdf5 (dataset "hmf").')
    parser.add_argument(
        '--cbs-mo-file',
        type=str,
        default='../wfs/mo_energies_with_cbs.txt',
        help='MO CBS-ε(DZ) table from basis extrapolation.',
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='PySCF/CI config for determinant ordering.'
    )
    parser.add_argument(
        '--no-cbs-correction',
        action='store_true',
        help='Do not add CBS-ε(DZ) diagonal correction to HMF.',
    )
    parser.add_argument(
        '--nist-file-order',
        action='store_true',
        help='Keep NIST levels in nist_data.txt section-1 file order (default: sort by energy).',
    )
    parser.add_argument(
        '--direct-hdf5',
        action='store_true',
        help='Open HDF5 in place (no temp snapshot). Only for finished runs; '
             'may block live DMC writers.',
    )
    args = parser.parse_args()
    file_name = find_files()
    
    delta_raw, ovlp_raw, hmf_raw, energies, full_data, step_info = get_data(
        file_name,
        hmf_file=args.hmf_file,
        cbs_mo_file=args.cbs_mo_file,
        config_path=args.config,
        apply_cbs_correction=not args.no_cbs_correction,
        hdf5_snapshot=not args.direct_hdf5,
    )
    discard_steps = get_discard(num_steps = full_data['delta'].shape[0])

    cbsi = step_info.get('cbs', {})
    cbs_shifts_1d = None
    if cbsi.get('applied') and cbsi.get('shift') is not None:
        _permutation = np.argsort(np.diag(hmf_raw))
        cbs_shifts_1d = np.asarray(cbsi['shift'], dtype=np.float64)[_permutation]

    delta, ovlp, hmf = ordered_matrices(delta_raw, ovlp_raw, hmf_raw)
    plot_energies(energies)
    # plot_delta_ovlp(delta, ovlp)
    nist_data = read_nist_data('./nist_data.txt')
    plot_eig_convergence(
        delta, ovlp, hmf, nist_data, plot_evolution=not args.no_plot,
        plot_filename='eig_convergence_filtered.png', use_filtered=True, show_error_bars=True,
        n_mc_samples=1000, step_info=step_info, cbs_shifts_1d=cbs_shifts_1d,
        nist_file_order=args.nist_file_order,
    )

    if not args.no_plot:
         plot_evolution(full_data, indices = None, discard = discard_steps)
    else:
         print('Plotting evolution is disabled.')
