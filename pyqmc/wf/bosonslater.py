import numpy as np
import pandas as pd
import pyqmc.gpu as gpu
import warnings
import pyqmc
import pyscf
import copy
from pyqmc.wftools import generate_slater
import h5py
import time
from scipy.sparse import lil_matrix

report_timer = False
def timer_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        wrapper.total_time += duration
        wrapper.total_calls += 1
        if wrapper.total_calls % 1 == 0 and report_timer:
            print(f'Spent {(wrapper.total_time):.4f}s in function {(wrapper.total_calls)} calls to {func.__name__!r}') 
        return result
    wrapper.total_calls = 0
    wrapper.total_time = 0
    return wrapper

def sherman_morrison_row(e, inv, vec):
    tmp = np.einsum("ek,ekj->ej", vec, inv)
    ratio = tmp[:, e]
    inv_ratio = inv[:, :, e] / ratio[:, np.newaxis]
    invnew = inv - np.einsum("ki,kj->kij", inv_ratio, tmp)
    invnew[:, :, e] = inv_ratio
    return ratio, invnew


def get_complex_phase(x):
    return x / np.abs(x)


class JoinParameters:
    """
    This class provides a dict-like interface that actually references
    other dictionaries in the background.
    If keys collide, then the first dictionary that matches the key will be returned.
    However, some bad things may happen if you have colliding keys.
    """

    def __init__(self, dicts):
        self.data = {}
        self.data = dicts

    def find_i(self, idx):
        for i, d in enumerate(self.data):
            if idx in d:
                return i

    def __setitem__(self, idx, value):
        i = self.find_i(idx)
        self.data[i][idx] = value

    def __getitem__(self, idx):
        i = self.find_i(idx)
        return self.data[i][idx]

    def __delitem__(self, idx):
        i = self.find_i(idx)
        del self.data[i][idx]

    def __iter__(self):
        for d in self.data:
            yield from d.keys()

    def __len__(self):
        return sum(len(i) for i in self.data)

    def items(self):
        for d in self.data:
            yield from d.items()

    def __repr__(self):
        return self.data.__repr__()

    def keys(self):
        for d in self.data:
            yield from d.keys()

    def values(self):
        for d in self.data:
            yield from d.values()


def sherman_morrison_ms(e, inv, vec):
    tmp = np.einsum("edk,edkj->edj", vec, inv)
    ratio = tmp[:, :, e]
    inv_ratio = inv[:, :, :, e] / ratio[:, :, np.newaxis]
    invnew = inv - np.einsum("kdi,kdj->kdij", inv_ratio, tmp)
    invnew[:, :, :, e] = inv_ratio
    return ratio, invnew

def compute_boson_value(updets, dndets, det_coeffs):
    """
    Given the up and down determinant values, safely compute the total log wave function.
    """
    upref = gpu.cp.amax(updets[1]).real
    dnref = gpu.cp.amax(dndets[1]).real
    logvals = 2*(updets[1] - upref + dndets[1] - dnref)
    wf_val = gpu.cp.einsum("d,id->i", det_coeffs, gpu.cp.exp(logvals))
    
    wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
    wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + 2*(upref + dnref))
    return gpu.asnumpy(wf_sign), gpu.asnumpy(wf_logval)


def _orbital_degeneracy_group_ids(mo_energies, deg_tol=1e-6):
    """Assign contiguous degeneracy-group ids from sorted MO energies."""
    e = np.asarray(mo_energies, dtype=np.float64).ravel()
    n = e.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(e, kind="mergesort")
    groups = np.empty(n, dtype=np.int64)
    gid = 0
    groups[order[0]] = 0
    for k in range(1, n):
        if abs(e[order[k]] - e[order[k - 1]]) >= deg_tol:
            gid += 1
        groups[order[k]] = gid
    return groups


def _count_excitations_vectorized(occ, occ_ground, deg_group_ids):
    """
    Count true excitations vs ``occ_ground`` for every row of ``occ``.

    Degenerate orbital swaps (same energy group) do not count as excitations.
    ``occ`` is (n_dets, n_occ); ``occ_ground`` is (n_occ,); ``deg_group_ids``
    is (n_orbs,) with an id per MO index.
    """
    occ = np.asarray(occ, dtype=np.int64)
    occ_ground = np.asarray(occ_ground, dtype=np.int64).ravel()
    if occ.ndim == 1:
        occ = occ.reshape(1, -1)
    n_dets = occ.shape[0]
    n_orbs = int(deg_group_ids.size)
    if n_orbs == 0:
        return np.zeros(n_dets, dtype=np.int64)

    max_needed = int(max(int(np.max(occ)), int(np.max(occ_ground)), 0)) + 1
    if max_needed > n_orbs:
        # Pad group ids if occupations reference orbitals beyond mo_energy length
        extra = np.arange(n_orbs, max_needed, dtype=np.int64) + (
            int(deg_group_ids.max()) + 1 if n_orbs else 0
        )
        deg_group_ids = np.concatenate([deg_group_ids, extra])
        n_orbs = max_needed

    ground_bool = np.zeros(n_orbs, dtype=bool)
    ground_bool[occ_ground] = True
    occ_bool = np.zeros((n_dets, n_orbs), dtype=bool)
    rows = np.arange(n_dets, dtype=np.int64)[:, None]
    occ_bool[rows, occ] = True

    is_new = occ_bool & ~ground_bool
    is_rem = ~occ_bool & ground_bool
    num_new = is_new.sum(axis=1)

    # Only groups that appear in ground or any excited occupation matter
    active = ground_bool | is_new.any(axis=0) | is_rem.any(axis=0)
    active_groups = np.unique(deg_group_ids[active])
    matched = np.zeros(n_dets, dtype=np.int64)
    for g in active_groups:
        in_g = deg_group_ids == g
        matched += np.minimum(is_new[:, in_g].sum(axis=1), is_rem[:, in_g].sum(axis=1))
    return num_new - matched


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

    det_prod = np.zeros(ndets, dtype=int)
    for i in range(ndets):
        occ_up, occ_dn = occupations[i]
        det_prod[i] = prod_matrix[
            get_prod(occ_up, up_orbsym), get_prod(occ_dn, down_orbsym)
        ]
    return det_prod[:, None] == det_prod[None, :]


def _cas_string_occupations(ncas, nelec_cas, ncore, nstrings):
    """
    Occupations for all CAS FCI strings of one spin, with core orbitals prepended.

    Returns
    -------
    occ : (nstrings, ncore + nelec_cas) int64
        Orbital indices including frozen core (same layout as binary_to_occ).
    """
    from pyscf.fci import cistring

    if nstrings == 0:
        return np.zeros((0, int(ncore) + int(nelec_cas)), dtype=np.int64)
    addrs = np.arange(nstrings, dtype=np.int64)
    strs = cistring.addrs2str(ncas, nelec_cas, addrs)
    occ_cas = cistring._strs2occslst(np.asarray(strs, dtype=np.int64), ncas).astype(
        np.int64
    )
    if ncore:
        core = np.arange(ncore, dtype=np.int64)
        return np.hstack(
            [np.broadcast_to(core, (nstrings, ncore)), occ_cas + ncore]
        )
    return occ_cas


def filter_determinants_from_ci(
    mc,
    mo_energies,
    det_emax,
    include_zeros=True,
    mol=None,
    mf=None,
    use_symm=False,
    energy_tol=1e-3,
    print_report=True,
    e_virt_max=None,
):
    """
    Filter determinants from a CI object based on energy criteria before processing.
    include_zeros: Whether to include zeros in the filtering
    
    Args:
        mc: pyscf multiconfigurational object (HCI, CAS, etc.)
        mo_energies: MO energies from mean field calculation
        det_emax: Energy threshold for filtering (float, int, 'singles', 'doubles', or 'energy,criteria')
        include_zeros: Whether to include zeros in the filtering
        e_virt_max: If set, reject any determinant that occupies an MO with
            energy strictly above this ceiling (Hartree). Use ``0.0`` to drop
            excitations into unbound (positive-energy) KS virtuals. The ground
            determinant is still restored when ``include_zeros`` is True.

    Returns:
        list: Filtered determinants in format suitable for orbital_evaluator_from_pyscf

    Notes:
        Dense CAS CI only (2D or flat 1D of size na*nb). Alpha/beta string
        occupations are decoded once (na+nb) and pair energies/excitations are
        broadcast; full N_det occupation lists are never built. Flat indices
        match ``large_ci(..., tol=-1)`` C-order.
    """
    from pyscf.fci import cistring
    from pyscf.fci.addons import _unpack_nelec

    if print_report:
        print("="*20 + "Filtering determinants start" + "="*20)
        print("Filtering determinants, energy units are in Hartree")

    ncore = mc.ncore if hasattr(mc, "ncore") else 0
    if hasattr(ncore, "__len__"):
        ncore_a, ncore_b = int(ncore[0]), int(ncore[1])
    else:
        ncore_a = ncore_b = int(ncore)

    n_a, n_b = _unpack_nelec(mc.nelecas)
    na = cistring.num_strings(mc.ncas, n_a)
    nb = cistring.num_strings(mc.ncas, n_b)
    n_dets = na * nb

    ci = np.asarray(mc.ci)
    if ci.ndim == 3:
        # Multi-root: use first root (caller usually selects target_root already)
        ci = ci[0]
    if ci.size != n_dets:
        raise ValueError(
            f"CI size {ci.size} does not match FCI dimension na*nb={na}*{nb}={n_dets} "
            f"(ncas={mc.ncas}, nelecas=({n_a},{n_b}))"
        )
    # Same reshape as pyscf.fci.addons.large_ci (handles flat hdf5 dumps)
    ci = ci.reshape(na, nb)

    # Decode α/β strings once — not all N_det pairs
    occ_a = _cas_string_occupations(mc.ncas, n_a, ncore_a, na)
    occ_b = _cas_string_occupations(mc.ncas, n_b, ncore_b, nb)

    if np.ndim(mo_energies) == 1:
        mo_energies = [mo_energies, mo_energies]
    mo_up = np.asarray(mo_energies[0], dtype=np.float64)
    mo_dn = np.asarray(mo_energies[1], dtype=np.float64)
    deg_up = _orbital_degeneracy_group_ids(mo_up)
    deg_dn = _orbital_degeneracy_group_ids(mo_dn)

    alpha_occ_ground = occ_a[0]
    beta_occ_ground = occ_b[0]
    E_a = np.sum(mo_up[occ_a], axis=1)  # (na,)
    E_b = np.sum(mo_dn[occ_b], axis=1)  # (nb,)
    total_energies = E_a[:, None] + E_b[None, :]  # (na, nb); never store pair occs
    ground_state_energy = float(total_energies[0, 0])
    near_ground = total_energies - ground_state_energy < 1e-6

    # Excitation counts on strings; pair total = broadcast sum
    exc_a_str = _count_excitations_vectorized(occ_a, alpha_occ_ground, deg_up)
    exc_b_str = _count_excitations_vectorized(occ_b, beta_occ_ground, deg_dn)
    up_num_exc = exc_a_str[:, None]
    dn_num_exc = exc_b_str[None, :]
    tot_exc = up_num_exc + dn_num_exc

    def pair_occ(flat_i):
        """Occupation arrays for flat C-order index into (na, nb)."""
        ia = int(flat_i) // nb
        ib = int(flat_i) % nb
        return occ_a[ia], occ_b[ib]

    # Apply filtering based on det_emax criteria
    option_text = ""
    if isinstance(det_emax, (float, np.floating)):
        assert det_emax > 0, "Emax must be positive for energy based determinant filtering"
        emax = float(det_emax) + ground_state_energy
        emin = float(np.min(total_energies))
        option_text = "Determinants being filtered with emax + min eigenvalue" + str(emax)
        mask = total_energies <= emax + energy_tol
        if include_zeros:
            mask = mask | near_ground

    elif isinstance(det_emax, int):
        assert det_emax > 0 and det_emax <= 100, "Emax must be between 0 and 100 for percentage based determinant filtering"
        percentile = det_emax
        option_text = "Determinants being filtered with percentage " + str(percentile)
        emax = float(np.percentile(total_energies, percentile))
        emin = float(np.min(total_energies))
        mask = total_energies <= emax + energy_tol
        if include_zeros:
            mask = mask | near_ground

    elif det_emax == 'singles' or det_emax == 'doubles':
        emax = float(np.max(total_energies))
        emin = float(np.min(total_energies))
        if det_emax == 'singles':
            mask = tot_exc < 2
        else:
            mask = tot_exc < 3
        if include_zeros:
            mask = mask | near_ground
        option_text = 'Det excitations' + str(tot_exc[mask])

    elif isinstance(det_emax, str) and ',' in det_emax:
        # Parse string of format "energy,criteria" e.g. "1.5,singles"
        # If the float portion has two energies " e.g. "1.0 1.5,singles", than we work inside the range of the two energies

        try:
            emax_energy, emax_criteria = det_emax.split(',')
            try:
                emin_energy, emax_energy = emax_energy.split(' ')
                emin_energy = float(emin_energy)
                emax_energy = float(emax_energy)
            except Exception:
                emin_energy = 0
                emax_energy = float(emax_energy)

            emax_criteria = emax_criteria.lower()
            option_text = "Determinants being filtered with energy range " + str(emin_energy) + " to " + str(emax_energy) + " and criteria " + str(emax_criteria)
            if emax_criteria not in ['singles', 'doubles', 'doubles_linked_singles']:
                raise ValueError("Criteria must be singles or doubles")
        except Exception as exc:
            raise ValueError("String format must be 'energy,criteria' where energy is a float and criteria is 'singles' or 'doubles'") from exc

        emax = emax_energy + ground_state_energy
        emin = emin_energy + ground_state_energy - 1E-6 # -1E-6 to avoid floating point issues
        energy_mask = (total_energies <= emax + energy_tol) & (
            total_energies > emin - energy_tol
        )

        if emax_criteria == 'singles' or emax_criteria == 'doubles':
            mask = energy_mask.copy()
            if emax_criteria == 'singles':
                mask &= tot_exc < 2
            else:
                mask &= tot_exc < 3
            if include_zeros:
                mask |= near_ground
        elif emax_criteria == 'doubles_linked_singles':
            import itertools

            def single_key_alpha(occ, occ_g):
                rem = set(occ_g) - set(occ)
                add = set(occ) - set(occ_g)
                if len(rem) != 1 or len(add) != 1:
                    return None
                return ("a", rem.pop(), add.pop())

            def single_key_beta(occ, occ_g):
                rem = set(occ_g) - set(occ)
                add = set(occ) - set(occ_g)
                if len(rem) != 1 or len(add) != 1:
                    return None
                return ("b", rem.pop(), add.pop())

            def double_is_product_of_allowed_singles(occ_a, occ_b, occ_ag, occ_bg, up_n, dn_n, allowed):
                if up_n == 1 and dn_n == 1:
                    ka, kb = single_key_alpha(occ_a, occ_ag), single_key_beta(occ_b, occ_bg)
                    return ka is not None and kb is not None and ka in allowed and kb in allowed

                if up_n == 2 and dn_n == 0:
                    rem = list(set(occ_ag) - set(occ_a))
                    add = list(set(occ_a) - set(occ_ag))
                    if len(rem) != 2 or len(add) != 2:
                        return False
                    for p1, p2 in itertools.permutations(add):
                        k1, k2 = ("a", rem[0], p1), ("a", rem[1], p2)
                        if k1 in allowed and k2 in allowed:
                            return True
                    return False

                if up_n == 0 and dn_n == 2:
                    rem = list(set(occ_bg) - set(occ_b))
                    add = list(set(occ_b) - set(occ_bg))
                    if len(rem) != 2 or len(add) != 2:
                        return False
                    for p1, p2 in itertools.permutations(add):
                        k1, k2 = ("b", rem[0], p1), ("b", rem[1], p2)
                        if k1 in allowed and k2 in allowed:
                            return True
                    return False

                return False

            single_energy_ok = (tot_exc == 1) & (total_energies <= emax + energy_tol)
            allowed_singles = set()
            for ia, ib in zip(*np.nonzero(single_energy_ok)):
                ua, ub = int(exc_a_str[ia]), int(exc_b_str[ib])
                if ua == 1 and ub == 0:
                    k = single_key_alpha(occ_a[ia], alpha_occ_ground)
                elif ua == 0 and ub == 1:
                    k = single_key_beta(occ_b[ib], beta_occ_ground)
                else:
                    continue
                if k is not None:
                    allowed_singles.add(k)
            mask = np.zeros((na, nb), dtype=bool)
            check = energy_mask | single_energy_ok | (near_ground if include_zeros else False)
            for ia, ib in zip(*np.nonzero(check)):
                te = int(tot_exc[ia, ib])
                if te == 0:
                    mask[ia, ib] = include_zeros
                elif te == 1:
                    mask[ia, ib] = single_energy_ok[ia, ib]
                elif te == 2 and energy_mask[ia, ib]:
                    mask[ia, ib] = double_is_product_of_allowed_singles(
                        occ_a[ia], occ_b[ib],
                        alpha_occ_ground, beta_occ_ground,
                        int(exc_a_str[ia]), int(exc_b_str[ib]), allowed_singles,
                    )

            if print_report:
                ag, bg = alpha_occ_ground, beta_occ_ground
                print("\nFiltered determinants (doubles_from_singles path):")
                print(
                    "  i | E (abs) | ΔE vs ground | class | α_exc β_exc | "
                    "α rem→add | β rem→add"
                )
                print("  " + "-" * 78)
                kept = np.flatnonzero(mask)
                kept = kept[np.argsort(total_energies.ravel()[kept])]
                for i in kept:
                    ia, ib = divmod(int(i), nb)
                    E_i = float(total_energies[ia, ib])
                    dE = E_i - ground_state_energy
                    te = int(tot_exc[ia, ib])
                    ua, ub = int(exc_a_str[ia]), int(exc_b_str[ib])
                    a_i, b_i = occ_a[ia], occ_b[ib]
                    a_rem = [int(x) for x in sorted(set(ag) - set(a_i))]
                    a_add = [int(x) for x in sorted(set(a_i) - set(ag))]
                    b_rem = [int(x) for x in sorted(set(bg) - set(b_i))]
                    b_add = [int(x) for x in sorted(set(b_i) - set(bg))]
                    if te == 0:
                        exc_class = "ground"
                    elif te == 1:
                        exc_class = "single"
                    elif te == 2:
                        exc_class = "double"
                    else:
                        exc_class = f"higher({te})"
                    a_part = f"{a_rem}→{a_add}" if a_rem or a_add else "—"
                    b_part = f"{b_rem}→{b_add}" if b_rem or b_add else "—"
                    print(
                        f"  {int(i):3d} | {E_i:11.6f} | {dE:12.6f} | {exc_class:7s} | "
                        f"{ua:1d} {ub:1d}     | {a_part:16s} | {b_part:16s}"
                    )

    else:
        # No filtering - return all determinants
        option_text = "No filtering"
        mask = np.ones((na, nb), dtype=bool)
        emax = float(np.max(total_energies))
        emin = float(np.min(total_energies))

    n_before_virt = int(np.sum(mask))
    if e_virt_max is not None:
        # Drop any α/β string that occupies an MO above the continuum ceiling.
        a_ok = np.max(mo_up[occ_a], axis=1) <= float(e_virt_max)
        b_ok = np.max(mo_dn[occ_b], axis=1) <= float(e_virt_max)
        virt_mask = a_ok[:, None] & b_ok[None, :]
        mask = mask & virt_mask
        if print_report:
            print(
                f"Virtual-energy ceiling e_virt_max={float(e_virt_max):.6f} Ha: "
                f"removed {n_before_virt - int(np.sum(mask))} determinants "
                f"occupying unbound MOs"
            )

    if include_zeros:
        mask = mask | near_ground
    filtered_energies = total_energies[mask]

    # Print report on filtered determinants
    if print_report:
        print(option_text)
        print("\nDeterminant Filtering Report:")
        print("-" * 50)
        print('Emax', np.round(emax, 3), np.round(emax-ground_state_energy, 3))
        print('Emin', np.round(emin, 3), np.round(emin-ground_state_energy, 3))
        print(f"Total determinants before filtering: {n_dets}")
        print(f"Determinants removed: {n_dets - int(np.sum(mask))}")
        print(f"Determinants remaining: {int(np.sum(mask))}")
        print('Min filtered eigenvalue', np.round(np.min(filtered_energies), 3), np.round(np.min(filtered_energies)-ground_state_energy, 3))
        print('Max filtered eigenvalue', np.round(np.max(filtered_energies), 3), np.round(np.max(filtered_energies)-ground_state_energy, 3))
    if np.sum(~mask) > 0:
        removed_energies = np.round(total_energies[~mask] - ground_state_energy, 3)
        unique_rel, counts = np.unique(removed_energies, return_counts=True)
        report = ' '.join([f'{e}(×{c})' for e, c in zip(unique_rel, counts)])
        if print_report:
            print('Unique removed eigenvalues (relative to ground, count):', report)

    if np.sum(mask) > 0:
        unrounded_energies = total_energies[mask]
        rounded_energy = np.round(unrounded_energies - ground_state_energy, 3)
        unique_rel, idx, counts = np.unique(rounded_energy, return_index=True, return_counts=True)
        report = ' '.join([f'{e}(×{c})' for e, c in zip(unique_rel, counts)])
        if print_report:
            print('Unique used eigenvalues (relative to ground, count):', report)
    else:
        print('No used determinants, exiting... ')
        exit()

    mask_indices = np.flatnonzero(mask)
    order = np.argsort(filtered_energies, kind="mergesort")
    sorted_mask_indices = mask_indices[order]
    sorted_filtered_energies = filtered_energies[order]

    # Materialize (weight, occupation) only for kept, energy-sorted dets
    ci_flat = ci.ravel()
    filtered_determinants = []
    occupations_kept = []
    for ind in sorted_mask_indices:
        a_occ, b_occ = pair_occ(ind)
        a_list, b_list = a_occ.tolist(), b_occ.tolist()
        filtered_determinants.append(
            (ci_flat[int(ind)], [a_list, b_list])
        )
        occupations_kept.append((a_occ, b_occ))
    saved = {
        "sorted_mask_indices": sorted_mask_indices,
        "sorted_filtered_energies": sorted_filtered_energies,
    }
    # Compute symmetry mask when requested (only for groups in CHARACTER_TABLE)
    if use_symm and mol is not None and mf is not None and mol.symmetry:
        try:
            symm_data = BosonWF.symm_utils(mol, mol.groupname)
            det_prod_filter = _compute_det_prod_filter(
                mol, mf, symm_data, occupations_kept
            )
            saved['det_prod_filter'] = det_prod_filter
        except ValueError as e:
            if "not in available groups" in str(e):
                pass  # Group (e.g. Dooh) not in CHARACTER_TABLE, skip symmetry mask
            else:
                raise
    return filtered_determinants, saved

class BosonWF:

    def __init__(self, mol, mf, 
                 mc=None, 
                 tol=None, 
                 twist=None, 
                 determinants=None, 
                 eval_gto_precision=None, 
                 det_emax = None,
                 e_virt_max = None,
                 use_symm = True, 
                 target_dtype = None):
        """
        Create Bosonic wavefunction
        Args:
            mol (_type_): A Mole object
            mf (_type_): a pyscf mean-field object
            mc (_type_, optional): a pyscf multiconfigurational object. Supports HCI and CAS. Defaults to None.
            tol (_type_, optional): smallest determinant weight to include in the wave function. Defaults to None.
            twist (_type_, optional): the twist of the calculation. Defaults to None.
            determinants (_type_, optional): A list of determinants suitable to pass into create_packed_objects. Defaults to None.

            You cannot pass both mc/tol and determinants.
        """
        self.tol = -1 if tol is None else tol
        self._mol = mol
        if mol.cart:
            self._gtoval = "GTOval_cart"
            self._gtoval_deriv1 = "GTOval_cart_deriv1"
            self._gtoval_deriv2 = "GTOval_cart_deriv2"
        else:
            self._gtoval = "GTOval_sph"
            self._gtoval_deriv1 = "GTOval_sph_deriv1"
            self._gtoval_deriv2 = "GTOval_sph_deriv2"
        if hasattr(mc, "nelecas"):
            # In case nelecas overrode the information from the molecule object.
            ncore = mc.ncore
            if not hasattr(ncore, "__len__"):
                ncore = [ncore, ncore]
            self._nelec = (mc.nelecas[0] + ncore[0], mc.nelecas[1] + ncore[1])
        else:
            ncore = (0,0)
            self._nelec = mol.nelec
        self.eval_gto_precision = eval_gto_precision
        
        try:
            self.num_det = int(np.asarray(mc.ci).size)
        except Exception:
            self.num_det = 1
        
        self.myparameters = {}
        
        # Check if we need to filter determinants before processing
        saved_filter = None
        if mol.symmetry and det_emax is not None and mc is not None:
            # Filter determinants first, then pass them to orbital_evaluator_from_pyscf
            filtered_determinants, saved_filter = filter_determinants_from_ci(
                mc,
                mf.mo_energy,
                det_emax,
                mol=mol,
                mf=mf,
                use_symm=use_symm,
                e_virt_max=e_virt_max,
            )
            self.num_det = len(filtered_determinants)

            if saved_filter is not None:
                self.saved_filter = saved_filter
        else:
            filtered_determinants = None
        (   _,
            self._det_occup,
            self._det_map,
            self.orbitals,
        ) = pyqmc.pyscftools.orbital_evaluator_from_pyscf(
            mol, mf, mc, twist=twist, determinants=filtered_determinants, tol=self.tol
        )

        self.det_info_file = 'det_info.hdf5'
        self.hmf_file      = 'hmf.hdf5'

        if mol.symmetry:
            try:
                self.symm_data = self.symm_utils(mol, mol.groupname)
                self.mo_coeff = mf.mo_coeff
            except ValueError as e:
                if "not in available groups" in str(e):
                    self.symm_data = None  # Group (e.g. Dooh) not in CHARACTER_TABLE
                    self.mo_coeff = mf.mo_coeff
                else:
                    raise
        else:
            self.symm_data = None

        # Set _det_prod_filter when use_symm and mol.symmetry
        if use_symm and mol.symmetry and self.symm_data is not None:
            if saved_filter is not None and 'det_prod_filter' in saved_filter:
                self._det_prod_filter = saved_filter['det_prod_filter']
            else:
                occupations = [
                    (self._det_occup[0][self._det_map[0, i]], self._det_occup[1][self._det_map[1, i]])
                    for i in range(self.num_det)
                ]
                self._det_prod_filter = _compute_det_prod_filter(mol, mf, self.symm_data, occupations)
        else:
            self._det_prod_filter = None

        if self.num_det > 1:
            if saved_filter is not None and 'sorted_filtered_energies' in saved_filter:
                self.get_hmf(saved_filter['sorted_filtered_energies'])
            else:
                raise ValueError('sorted_filtered_energies not found in saved_filter')
        else:
            print('Using only one determinant')
        
        
        # Use constant weight 
        self.myparameters["det_coeff"] = np.ones(self.num_det)
        self.parameters = JoinParameters([self.myparameters, self.orbitals.parameters])

        iscomplex = self.orbitals.mo_dtype == complex or bool(
            sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
        )
        if target_dtype is not None:
            self.dtype = complex if np.issubdtype(np.dtype(target_dtype), np.complexfloating) else float
        else:
            self.dtype = complex if iscomplex else float

        self.get_phase = get_complex_phase if iscomplex else gpu.cp.sign

    @staticmethod
    def direct_product_table(characters, irrep_to_idx=None):
        """
        Compute the direct product table for irreducible representations.
        Uses full character table so all irreps are included (not just mol.irrep subset).

        Parameters:
        characters (dict): A dictionary where keys are irrep labels and values are lists of characters
        irrep_to_idx (dict): Optional mapping from irrep name to index. If None, built from characters.keys().

        Returns:
        dict: table, matrix, irrep_to_idx, plot_data
        """
        irreps = list(characters.keys())
        if irrep_to_idx is None:
            irrep_to_idx = {irrep: i for i, irrep in enumerate(irreps)}
        print(irrep_to_idx)

        n_ops = len(list(characters.values())[0])
        n = len(characters.keys())
        dp_table = {}
        matrix = np.zeros((n, n), dtype=int)
        plot_data = np.zeros((n, n), dtype=np.dtype('<U10'))

        for i, irrep1 in enumerate(irreps):
            for j, irrep2 in enumerate(irreps):
                product = [characters[irrep1][k] * characters[irrep2][k] for k in range(n_ops)]
                for irrep in irreps:
                    projection = sum(product[k] * characters[irrep][k] for k in range(n_ops)) / n_ops
                    if abs(projection - 1.0) < 1e-10:
                        dp_table[(irrep1, irrep2)] = irrep
                        plot_data[i, j] = irrep
                        matrix[irrep_to_idx[irrep1], irrep_to_idx[irrep2]] = irrep_to_idx[irrep]
                        break

        return {"table": dp_table, "matrix": matrix, "irrep_to_idx": irrep_to_idx, "plot_data": plot_data}

    @staticmethod
    def symm_utils(mol, abel_group):
        """Given a molecule and its abelian group, returns direct product table.
        Uses full CHARACTER_TABLE so all irreps are included (handles operations in
        CHARACTER_TABLE that are not in mol.irrep).
        """
        from pyscf.symm.param import CHARACTER_TABLE as character_table
        available_groups = character_table.keys()
        if abel_group not in available_groups:
            raise ValueError(f"Group {abel_group} not in available groups {available_groups}")

        ct = character_table[abel_group]
        ct_dict = {}
        irrep_to_idx = {}
        for i, item in enumerate(ct):
            key = item[0]
            value = np.array(item[1:])
            ct_dict[key] = value
            irrep_to_idx[key] = i

        print('='*20+"Symmetry data"+"="*20)
        print("Using Symmetric MOs: ")
        print("Miller indices, irrep_ids, orb_shape")
        for s, i, c in zip(mol.irrep_name, mol.irrep_id, mol.symm_orb):
            print(s, i, c.shape)
        print(ct_dict.keys())

        characters = ct_dict
        irreps = list(characters.keys())
        n_ops = len(list(characters.values())[0])
        n = len(characters.keys())
        dp_table = {}
        matrix = np.zeros((n, n), dtype=int)
        plot_data = np.zeros((n, n), dtype=np.dtype('<U10'))

        for i, irrep1 in enumerate(irreps):
            for j, irrep2 in enumerate(irreps):
                product = [characters[irrep1][k] * characters[irrep2][k] for k in range(n_ops)]
                for irrep in irreps:
                    projection = sum(product[k] * characters[irrep][k] for k in range(n_ops)) / n_ops
                    if abs(projection - 1.0) < 1e-10:
                        dp_table[(irrep1, irrep2)] = irrep
                        plot_data[i, j] = irrep
                        matrix[irrep_to_idx[irrep1], irrep_to_idx[irrep2]] = irrep_to_idx[irrep]
                        break

        print("Direct product table (irrep_id):")
        print(matrix)
        df_plot = pd.DataFrame(plot_data, index=irreps, columns=irreps)
        print("Direct product table (irrep_name):")
        print(df_plot)
        print('='*20+"Symmetry data end"+"="*20)

        results = {
            "matrix": matrix,
            "irrep_to_idx": irrep_to_idx,
            "irrep_names": mol.irrep_name,
            "irrep_ids": mol.irrep_id
        }
        return results

    
    def get_hmf(self, sorted_filtered_energies):
        hf = h5py.File(self.hmf_file, 'w')
        hf.create_dataset('hmf', data=sorted_filtered_energies)
        self.hmf = np.diag(sorted_filtered_energies)
        if hasattr(self, 'saved_filter') and self.saved_filter is not None:
            if 'det_prod_filter' in self.saved_filter:
                hf.create_dataset('saved_filter/det_prod_filter', data=self.saved_filter['det_prod_filter'])
        hf.close()

    @timer_func
    def recompute(self, configs):
        r"""This computes the value from scratch. Returns the logarithm of the wave function as
        (phase,logdet). If the wf is real, phase will be +/- 1."""
        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos(self._gtoval, configs)
        self._aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        self._dets = []
        self._inverse = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(self._aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            self._dets.append(
                gpu.cp.asarray(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]
            is_zero = np.sum(np.abs(self._dets[s][0]) < 1e-16)
            compute = np.isfinite(self._dets[s][1])
            if is_zero > 0:
                warnings.warn(
                    f"A wave function is zero. Found this proportion: {is_zero/nconf}"
                )
                print(f"zero {is_zero/np.prod(compute.shape)}")
            self._inverse.append(gpu.cp.zeros(mo_vals.shape, dtype=mo_vals.dtype))
            for d in range(compute.shape[1]):
                self._inverse[s][compute[:, d], d, :, :] = gpu.cp.linalg.inv(
                    mo_vals[compute[:, d], d, :, :]
                )
            # spin, Nconf, [ndet_up, ndet_dn], nelec, nelec
        return self.value()

    @timer_func
    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        r"""Update any internals given that electron e moved to epos. mask is a Boolean array
        which allows us to update only certain walkers"""

        s = int(e >= self._nelec[0])
        if mask is None:
            mask = np.ones(epos.configs.shape[0], dtype=bool)
        is_zero = np.sum(np.isinf(self._dets[s][1]))
        if is_zero:
            warnings.warn(
                "Found a zero in the wave function. Recomputing everything. This should not happen often."
            )
            self.recompute(configs)
            return

        eeff = e - s * self._nelec[0]
        if saved_values is None:
            ao = self.orbitals.aos(self._gtoval, epos, mask)
            self._aovals[:, mask, e, :] = ao
            mo = self.orbitals.mos(ao, s)
        else:
            ao, mo = saved_values
            self._aovals[:, mask, e, :] = ao[:, mask]
            mo = mo[mask]
        mo_vals = mo[:, self._det_occup[s]]
        det_ratio, self._inverse[s][mask, :, :, :] = sherman_morrison_ms(
            eeff, self._inverse[s][mask, :, :, :], mo_vals
        )
        self._dets[s][0, mask, :] *= self.get_phase(det_ratio)
        self._dets[s][1, mask, :] += gpu.cp.log(gpu.cp.abs(det_ratio))
    
    @timer_func
    def value(self):
        r"""Returns the logarithmic value of the bosonic wavefunction: log(\Phi_B)

        Returns:
            sign, logval: sign and logatithmic value of the bosonic wavefunction
        """
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]

        upref = gpu.cp.amax(updets[1]).real
        dnref = gpu.cp.amax(dndets[1]).real
        det_coeff = self.myparameters['det_coeff']
        logvals = 2*(updets[1] - upref + dndets[1] - dnref)
        wf_val = gpu.cp.einsum("d, id->i", det_coeff, gpu.cp.exp(logvals))
        # wf_val = self.regularize(wf_val)
        
        wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
        wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + 2*(upref + dnref))        
        return wf_sign, wf_logval
    
    def value_configs(self, configs):
        r"""Returns the value of the bosonic wavefunction for a given configuration"""
        nconf, nelec, ndim = configs.configs.shape
        aos = self.orbitals.aos(self._gtoval, configs)
        aovals = aos.reshape(-1, nconf, nelec, aos.shape[-1])
        dets = []
        for s in [0, 1]:
            begin = self._nelec[0] * s
            end = self._nelec[0] + self._nelec[1] * s
            mo = self.orbitals.mos(aovals[:, :, begin:end, :], s)
            mo_vals = gpu.cp.swapaxes(mo[:, :, self._det_occup[s]], 1, 2)
            dets.append(
                gpu.cp.asarray(np.linalg.slogdet(mo_vals))
            )  # Spin, (sign, val), nconf, [ndet_up, ndet_dn]
            is_zero = np.sum(np.abs(dets[s][0]) < 1e-16)
            compute = np.isfinite(dets[s][1])
            if is_zero > 0:
                warnings.warn(
                    f"A wave function is zero. Found this proportion: {is_zero/nconf}"
                )
                print(f"zero {is_zero/np.prod(compute.shape)}")

        updets = dets[0][:, :, self._det_map[0]]
        dndets = dets[1][:, :, self._det_map[1]]

        upref = gpu.cp.amax(updets[1]).real
        dnref = gpu.cp.amax(dndets[1]).real
        det_coeff = self.myparameters['det_coeff']
        logvals = 2*(updets[1] - upref + dndets[1] - dnref)
        wf_val = gpu.cp.einsum("d, id->i", det_coeff, gpu.cp.exp(logvals))

        wf_sign = np.nan_to_num(wf_val / gpu.cp.abs(wf_val))
        wf_logval = 1./2 * np.nan_to_num(gpu.cp.log(gpu.cp.abs(wf_val)) + 2*(upref + dnref))        
        return wf_sign, wf_logval
    
    @timer_func
    def value_dets(self, test = False):
        r"""Returns logarithmic values (∇Phi_l/Phi_l) of all Slater determinants used to form bosonic wavefunction

        Args:
            test (bool, optional): Calculates the value of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            sign, logval: sign and logatithmic value of each wavefunction
        """
        updets = self._dets[0][:, :, self._det_map[0]]
        dndets = self._dets[1][:, :, self._det_map[1]]

        wf_logval = (updets[1] + dndets[1])
        wf_sign = updets[0] * dndets[0]

        if test:
            det_coeff = self.myparameters['det_coeff']
            tol = 1E-12
            phi_b = 1./2 * np.log(np.einsum('d, id->i', det_coeff,np.exp(2*wf_logval) ))
            try:
                assert ((np.abs(phi_b - self.value()[1]) < tol).all())
            except:
                print('value_dets error', np.max(np.abs(phi_b - self.value()[1])))
                      
        return wf_sign, wf_logval
    
    @timer_func
    def gradient(self, e, epos):
        r"""Compute the gradient of the log wave function ∇log(Psi_B) 
        Note that this can be called even if the internals have not been updated for electron e,
        if epos differs from the current position of electron e."""
        #returns \nabla ln(\Phi_B)=\frac{\nabla \Phi_B}{\Phi_B}
        #= \frac{\sum{\nabla \Phi_n*\Phi_n}}{\Phi_B^2}
        #= \frac{\sum{\Phi_n^2 * \nabla ln(\Phi_n)}}{\Phi_B^2}
        #= \frac{\sum{exp(2*ln(\Phi_n)) * \nabla ln(\Phi_n)}}{exp(2*ln(\Phi_B))}
        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos(self._gtoval_deriv1, epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        
        det_coeff = self.myparameters['det_coeff']
        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )
        jacobi0 = jacobi[0]
        res = np.finfo(jacobi0.dtype).resolution
        jacobi0[jacobi0 < res] = res
        jacobi[0] = jacobi0
        
        jacobid = jacobi[..., self._det_map[s]]
        jacobid = jacobid[1:]/jacobid[0]

        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )

        res = np.finfo(denom.dtype).resolution
        denom[denom < res] = res
        
        grad = numer / denom
        grad[~np.isfinite(grad)] = 0.0
        return grad
    
    def gradient_laplacian(self, e, epos):
        grad = self.gradient(e, epos)
        lap = self.laplacian(e, epos)
        return grad, lap
    
    @ timer_func
    def laplacian(self, e, epos, lap_phi_n = None, loggrad_phi_n = None, phi_n = None, phi_b = None):
        r"""Returns ∇²(Phi_B)/Phi_B of bosonic wave function for electron e at position epos
        Returns array of shape (nconfigs,)
        \[
        \nabla^2 \Phi_B = \frac{\sum_l \left( \nabla \Phi_l \cdot \nabla \Phi_l + \Phi_l \nabla^2 \Phi_l \right)}{\Phi_B} 
        - \frac{\left( \sum_l \Phi_l \nabla \Phi_l \right)^2}{\Phi_B^3}.
        \]
        # The Laplacian of the bosonic wave function (Phi_B) divided by Phi_B is:
        #
        # 1. First term: Sum over determinants l of:
        #    - (gradient of Phi_l)·(gradient of Phi_l)  [dot product of gradients]
        #    - plus (Phi_l)·(Laplacian of Phi_l)
        #    All divided by Phi_B squared
        #
        # 2. Second term: Subtract
        #    - The square of (sum of Phi_l times gradient of Phi_l)
        #    - Divided by Phi_B 4th power
        #
        # This implements the quotient rule for second derivatives of the bosonic wave function
        """

        if lap_phi_n is None:
            lap_phi_n = self.laplacian_dets(e, epos) # ∇²(Phi_n)
        
        if loggrad_phi_n is None:
            loggrad_phi_n, _ = self.gradient_dets(e, epos) # ∇log(Phi_n) # large

        if phi_n is None:
            phase_n, logval_n = self.value_dets()   # phase(Phi_n), log(Phi_n)
            phi_n = phase_n * np.nan_to_num(np.exp(logval_n)) # Phi_n
        
        if phi_b is None:
            phase_b, logval_b = self.value() # phase(Phi_B), log(Phi_B)
            phi_b = phase_b * np.nan_to_num(np.exp(logval_b)) # Phi_B

        

        # Calculate ∇²(Phi_B)/Phi_B
        # First term: Sum over determinants l of: (gradient of Phi_l)·(gradient of Phi_l)  [dot product of gradients]
        
        grad_phi_l = np.einsum('nxc, cn->nxc', loggrad_phi_n, phi_n)
        
        lap_b1 = np.einsum('nxc, nxc->c', grad_phi_l, grad_phi_l)
        # term below can be executed only once. 
        lap_b1 += np.einsum('cn, cn->c', phi_n**2, lap_phi_n) # Changed due to new lap_n definition in this commit
        lap_b1 /= phi_b**2
        # Second term: Minus the square of (sum of Phi_l times gradient of Phi_l)
        lap_b2 = np.einsum('cn, nxc->cx', phi_n, grad_phi_l)
        lap_b2 = np.einsum('cx, cx->c', lap_b2, lap_b2)
        lap_b2 /= phi_b**4
        lap_b = lap_b1 - lap_b2
        return lap_b
    
    @staticmethod
    def regularize(array, resolution=None):
        '''
        Regularize an array to resolution value to avoid division by zero.
        '''
        if resolution is None:
            resolution = np.finfo(array.dtype).resolution
        mask = np.abs(array) < resolution
        array_sign = 2 * (array[mask] >= 0) - 1
        array[mask] = resolution * array_sign
        return array

    def gradient_value(self, e, epos):
        r"""Returns the ∇log(Phi_B) gradient of bosonic wavefunction and its log value log(Phi_B)
        Phi_B is defined in eq. 4, Phi_B = \sqrt{\sum_{n}{\Phi_n^2}}
        Returns array of shape (nconfigs, 3) and (nconfigs,)"""

        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos(self._gtoval_deriv1, epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]
        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        det_coeff = self.myparameters['det_coeff']
        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )

        jacobid = jacobi[..., self._det_map[s]]

        # jacobid[0] = self.regularize(jacobid[0])

        ratio = np.einsum('d, di, id-> i', det_coeff, det_array**2, jacobid[0]**2)
        jacobid = jacobid[1:]/jacobid[0]

        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )
        # denom = self.regularize(denom)

        ratio =  ratio/denom
        derivatives = numer / denom
        derivatives[~np.isfinite(derivatives)] = 0.0
        # values = derivatives[0]
        # values[~np.isfinite(values)] = 1.0
        return derivatives, ratio, (aograd[:, 0], mograd[0])
    
    @timer_func
    def gradient_dets(self, e, epos, test=False):
        r"""Returns the ∇log(Phi_l) gradient of each slater determinant forming the bosonic wavefunction
        Phi_l is defined in eq. 14, psi_l = Phi_l/Phi_B

        Args:
            e (_type_): electron index
            epos (_type_): electron coordinates
            test (bool, optional): Calculates the gradient of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            gradient: [# of determinants, cartesian(3), nconfigs]
        """

        s = int(e >= self._nelec[0])
        aograd = self.orbitals.aos(self._gtoval_deriv1, epos)
        mograd = self.orbitals.mos(aograd, s)
        mograd_vals = mograd[:, :, self._det_occup[s]]

        jacobi = np.einsum(
            "ei...dj,idj...->ei...d",
            mograd_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        
        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real
        # Removed detcoeff and ref values
        det_array = (
            self._dets[0][0, :, self._det_map[0]] 
            * self._dets[1][0, :, self._det_map[1]] 
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )

        numer = np.einsum(
            "ei...d,di->edi...",
            jacobi[..., self._det_map[s]],
            det_array,
        )
        # numer[0] = self.regularize(numer[0])
        grads_n = numer[1:] / numer[0]
        grads_n = np.einsum('edi->dei', grads_n)

        # Calculate grad as well 
        det_coeff = self.myparameters['det_coeff']
        jacobid = jacobi[..., self._det_map[s]]

        jacobid[0] = self.regularize(jacobid[0])
        jacobid = jacobid[1:]/jacobid[0]
        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )
        # denom = self.regularize(denom)
        grad = numer / denom
        grad[~np.isfinite(grad)] = 0.0

        
        # det_array = (
        #     self._dets[0][0, :, self._det_map[0]]
        #     * self._dets[1][0, :, self._det_map[1]]
        #     * np.exp(
        #         self._dets[0][1, :, self._det_map[0]]
        #         + self._dets[1][1, :, self._det_map[1]]
        #     )
        # )
        # numer = np.einsum(
        #     "ei...d,di->edi...",
        #     ratios[..., self._det_map[s]],
        #     det_array,
        # )


        
        # denom has the sum of Multideterminant WF, not needed

        if test:
            tol = 1E-6
            dv = self.value_dets()[1]
            v = self.value()[1]
            det_coeff = self.myparameters['det_coeff']
            gc = np.einsum('d, id,dei->ei', det_coeff, np.exp(2*(dv-v[:, None])), grads_n)
            try:
                assert ((np.abs(gc - self.gradient(e, epos)) < tol).all())
                print('gradient_dets test passed')
            except:
                print('gradient_dets error', np.max(np.abs(gc - self.gradient(e, epos))))
            exit()
        return grads_n, grad
    
    def laplacian_dets(self, e, epos, test=False):
        r"""Returns laplacian ∇²(Phi_l)/Phi_l of each slater determinant forming the bosonic wavefunction
        Phi_l is defined in eq. 14, psi_l = Phi_l/Phi_B

        Args:
            e (_type_): electron index
            epos (_type_): electron coordinates
            test (bool, optional): Calculates the laplacian of bosonic wavefunction using values in this function.
                                   Defaults to False.

        Returns:
            laplacian: [# of determinants, nconfigs]
        """
        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos(self._gtoval_deriv2, epos)
        ao_val = ao[:, 0, :, :]
        ao_lap = ao[:, 4, :, :]  # orbitals.aos already returns compact [val,∇,∇²]
        mo_lap_vals = gpu.cp.stack(
            [self.orbitals.mos(x, s)[..., self._det_occup[s]] for x in [ao_val, ao_lap]]
        )

        jacobi = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mo_lap_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        # det_array = (
        #     self._dets[0][0, :, self._det_map[0]]
        #     * self._dets[1][0, :, self._det_map[1]]
        #     * gpu.cp.exp(
        #         self._dets[0][1, :, self._det_map[0]]
        #         + self._dets[1][1, :, self._det_map[1]]
        #         # - upref
        #         # - dnref
        #     )
        # )
        
        # # det_coeff = self.myparameters['det_coeff']
        # numer = gpu.cp.einsum(
        #     "ei...d,di->ei...d",
        #     jacobi[..., self._det_map[s]],
        #     # det_coeff,
        #     det_array,
        # )
        # # denom = np.sum(numer[0], axis=1)
        # # lap = np.einsum('id, i->id', numer[1], 1./denom)
        
        # lap = numer[1]/numer[0]

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )
        
        # det_coeff = self.myparameters['det_coeff']
        numer = gpu.cp.einsum(
            "ei...d,di->ei...d",
            jacobi[..., self._det_map[s]],
            # det_coeff,
            det_array,
        )
        # denom = np.sum(numer[0], axis=1)
        # lap = np.einsum('id, i->id', numer[1], 1./denom)
        
        lap = numer[1]/numer[0]

        # np.sum(numer[0], axis=1) should be the same as denom in laplacian @ slater.py
        # If want to return ∇²(Psi_n), return numer[1]
        # np.einsum('ie, i->ie',numer[1], 1./np.sum(numer[0], axis=1)) returns ∇²(Psi_n)/\sum(Psi_n)
        # For testing against slater laplacian, return np.einsum('ie, i->ie',numer[1], 1./np.sum(numer[0], axis=1))
        # We don't need to evaluate numer[0], if we want ∇²(Psi_n) 
        return lap
        


    @timer_func
    def gradient_laplacian_dets(self, e, epos):
        r"""Returns the ∇²(Phi_l), ∇log(Phi_l) and ∇log(Phi_B) of each slater determinant forming the bosonic wavefunction
        Phi_l is defined in eq. 14, psi_l = Phi_l/Phi_B

        Args:
            e (_type_): electron index
            epos (_type_): electron coordinates

        Returns:
            laplacian: [# of determinants, nconfigs]
            gradient: [# of determinants, cartesian(3), nconfigs]
            gradient_b: [cartesian(3), nconfigs]
        """

        s = int(e >= self._nelec[0])
        ao = self.orbitals.aos(self._gtoval_deriv2, epos)

        ao_val = ao[:, 0, :, :]
        ao_grad = ao[:, 0:4, :, :]
        mo_grad = self.orbitals.mos(ao_grad, s)
        mo_grad_vals = mo_grad[:, :, self._det_occup[s]]

        ao_lap = ao[:, 4, :, :]  # orbitals.aos already returns compact [val,∇,∇²]
        mo_lap_vals = gpu.cp.stack(
            [self.orbitals.mos(x, s)[..., self._det_occup[s]] for x in [ao_val, ao_lap]]
        )
        
        
        jacobi_grad = np.einsum(
            "ei...dj,idj...->ei...d",
            mo_grad_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )
        
        jacobi_lap = gpu.cp.einsum(
            "ei...dj,idj...->ei...d",
            mo_lap_vals,
            self._inverse[s][..., e - s * self._nelec[0]],
        )

        upref = gpu.cp.amax(self._dets[0][1]).real
        dnref = gpu.cp.amax(self._dets[1][1]).real

        det_array = (
            self._dets[0][0, :, self._det_map[0]]
            * self._dets[1][0, :, self._det_map[1]]
            * gpu.cp.exp(
                self._dets[0][1, :, self._det_map[0]]
                + self._dets[1][1, :, self._det_map[1]]
                - upref
                - dnref
            )
        )
        
        # Calculate grad_n
        numer = np.einsum(
            "ei...d,di->edi...",
            jacobi_grad[..., self._det_map[s]],
            det_array,
        )
        # numer[0] = self.regularize(numer[0])
        grad_n = numer[1:] / numer[0]
        grad_n = np.einsum('edi->dei', grad_n)

        # Calculate lap_n
        numer = gpu.cp.einsum(
            "ei...d,di->ei...d",
            jacobi_lap[..., self._det_map[s]],
            # det_coeff,
            det_array,
        )
        # denom = np.sum(numer[0], axis=1)
        # lap = np.einsum('id, i->id', numer[1], 1./denom)
        # numer[0] = self.regularize(numer[0])
        lap_n = numer[1]/numer[0]

        # Calculate grad_b as well 
        det_coeff = self.myparameters['det_coeff']
        jacobid = jacobi_grad[..., self._det_map[s]]
        # jacobid[0] = self.regularize(jacobid[0])
        jacobid = jacobid[1:]/jacobid[0]
        numer =  gpu.cp.einsum(
            "ei...d,d,di->ei...",
            jacobid,
            det_coeff,
            det_array**2
        )

        denom = gpu.cp.einsum(
            "d,di->i...",
            det_coeff,
            det_array**2
        )
        # denom = self.regularize(denom)
        grad_b = numer / denom
        grad_b[~np.isfinite(grad_b)] = 0.0

        return lap_n, grad_n, grad_b

    def pgradient(self):
        # Not implemented
        d = {}
        return d
