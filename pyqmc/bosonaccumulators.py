import os
import sys
import time
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

# --- optional boson DMC / ABCDMC wall-clock profiling (env or pyqmc.boson_profile_config) ---
from pyqmc import boson_profile_config as _bprof

ABCDMC_ACC_KEY = "abc_dmc_excitations"
ABQMC_ENERGY_ACC_KEY = "energy"


def configure_boson_dmc_profiling(enabled=None, print_every=None):
    """Set profiling from ``config.yaml`` (recommended on batch systems). See ``pyqmc.boson_profile_config``."""
    _bprof.configure(enabled=enabled, print_every=print_every)


def boson_dmc_profile_enabled():
    return _bprof.is_enabled()


def boson_dmc_profile_print_every():
    return _bprof.print_interval()


def bdmc_profile_worker_label():
    """Tag for this OS process (MPI rank if mpi4py is available, else pid)."""
    try:
        from mpi4py import MPI

        return f"[worker {MPI.COMM_WORLD.Get_rank()}]"
    except Exception:
        return f"[worker pid={os.getpid()}]"


def bdmc_profile_print(text):
    """Print profiling lines with a worker tag on every line (for log post-processing)."""
    tag = bdmc_profile_worker_label()
    for line in str(text).splitlines():
        msg = f"{tag} {line}"
        print(msg, flush=True)
        # Many batch systems capture stderr more reliably than worker stdout.
        print(msg, file=sys.stderr, flush=True)


_bdmc_w_prop = 0.0
_bdmc_w_hdf = 0.0
_bdmc_w_abcdmc = 0.0
_bdmc_step = 0
_bdmc_prof_banner_shown = False
_bdmc_rundmc_entry_announced = False

# Per-DMC-step wall time inside dmc_propagate excluding the ABCDMC accumulator call
# (same bucket as DMC_propagate_ex_ABCDMC). Keys are filled from bosondmc.dmc_propagate.
_dmc_prop_inner_secs = {}


def bdmc_profile_note_rundmc_start(client_is_parallel):
    """One line per process when rundmc runs with profiling (confirms env is visible here)."""
    global _bdmc_rundmc_entry_announced
    if not boson_dmc_profile_enabled() or _bdmc_rundmc_entry_announced:
        return
    _bdmc_rundmc_entry_announced = True
    mode = "parallel_client" if client_is_parallel else "serial"
    nrep = boson_dmc_profile_print_every()
    bdmc_profile_print(
        f"rundmc: profiling active on this process ({mode}); print_every={nrep}"
    )


def bdmc_profile_ensure_banner_once():
    global _bdmc_prof_banner_shown
    if _bdmc_prof_banner_shown or not boson_dmc_profile_enabled():
        return
    _bdmc_prof_banner_shown = True
    n = boson_dmc_profile_print_every()
    if n <= 0:
        extra = (
            "periodic reports off (set profile_abcdmc_print_every in config.yaml "
            "or PYQMC_PROFILE_ABCDMC_PRINT_EVERY=N)"
        )
    else:
        extra = f"periodic reports every {n} DMC steps"
    bdmc_profile_print(
        "--- pyqmc boson DMC profiling: ON "
        "(config: profile_boson_dmc / profile_abcdmc_print_every, or env PYQMC_PROFILE_*). "
        f"{extra}. ---"
    )


def bdmc_profile_add_prop(dt):
    global _bdmc_w_prop
    bdmc_profile_ensure_banner_once()
    _bdmc_w_prop += dt


def bdmc_profile_add_hdf(dt):
    global _bdmc_w_hdf
    _bdmc_w_hdf += dt


def bdmc_profile_add_abcdmc(dt):
    global _bdmc_w_abcdmc
    _bdmc_w_abcdmc += dt


def bdmc_profile_reset_window():
    global _bdmc_w_prop, _bdmc_w_hdf, _bdmc_w_abcdmc, _dmc_prop_inner_secs
    _bdmc_w_prop = _bdmc_w_hdf = _bdmc_w_abcdmc = 0.0
    _dmc_prop_inner_secs.clear()


def bdmc_prop_inner_add(key, dt):
    """Accumulate fine-grained time inside ``dmc_propagate`` (excl. ABCDMC accumulator)."""
    global _dmc_prop_inner_secs
    if not boson_dmc_profile_enabled() or dt <= 0.0:
        return
    _dmc_prop_inner_secs[key] = _dmc_prop_inner_secs.get(key, 0.0) + dt


def bdmc_prop_inner_mark(key, tmark):
    """Record wall time since ``tmark`` for ``key``; return new ``perf_counter()``."""
    if not boson_dmc_profile_enabled():
        return time.perf_counter()
    now = time.perf_counter()
    bdmc_prop_inner_add(key, now - tmark)
    return now


def _profile_bar(pct, width=18):
    n = int(round(width * min(100.0, max(0.0, pct)) / 100.0))
    return "#" * n + "." * (width - n)


def bdmc_profile_format_report(accumulators):
    """Build text for coarse timers + ABCDMC internal breakdown."""
    lines = []
    nwin = boson_dmc_profile_print_every()
    tot = _bdmc_w_prop + _bdmc_w_hdf + _bdmc_w_abcdmc
    lines.append(
        f"--- boson DMC profile (global step {_bdmc_step}, window ~{nwin} steps) ---"
    )
    if tot <= 0:
        lines.append("(no coarse time recorded this window)")
        return "\n".join(lines)
    for name, sec in (
        ("DMC_propagate_ex_ABCDMC", _bdmc_w_prop),
        ("HDF5_dmc_file", _bdmc_w_hdf),
        ("ABCDMCMatrixAccumulator", _bdmc_w_abcdmc),
    ):
        pct = 100.0 * sec / tot
        lines.append(f"  {name:28s} {sec:8.4f}s  {pct:5.1f}%  {_profile_bar(pct)}")
    inner_sum = sum(_dmc_prop_inner_secs.values())
    if _bdmc_w_prop > 0.0 and inner_sum > 0.0:
        lines.append(
            "  inside DMC_propagate_ex_ABCDMC (each line as % of that bucket; "
            "sums to ~100% if fully covered):"
        )
        display = dict(_dmc_prop_inner_secs)
        unacc = _bdmc_w_prop - inner_sum
        if unacc > 1e-9:
            display["_unaccounted"] = max(0.0, unacc)
        for k in sorted(display, key=lambda x: -display[x]):
            p = 100.0 * display[k] / _bdmc_w_prop
            lines.append(
                f"    {k:26s} {display[k]:8.4f}s  {p:5.1f}%  {_profile_bar(p)}"
            )
    en_acc = accumulators.get(ABQMC_ENERGY_ACC_KEY) if accumulators else None
    if en_acc is not None and hasattr(en_acc, "_prof_secs"):
        esecs = en_acc._prof_secs
        ein = sum(esecs.values())
        if ein > 0:
            lines.append(
                "  inside ABQMCEnergyAccumulator (__call__) "
                "(percents vs sum of these components):"
            )
            for k in sorted(esecs, key=lambda x: -esecs[x]):
                p = 100.0 * esecs[k] / ein
                lines.append(
                    f"    {k:22s} {esecs[k]:8.4f}s  {p:5.1f}%  {_profile_bar(p)}"
                )
    acc = accumulators.get(ABCDMC_ACC_KEY) if accumulators else None
    if acc is not None and hasattr(acc, "_prof_secs"):
        secs = acc._prof_secs
        inner = sum(secs.values())
        lines.append(
            f"  ABCDMC config: numba_installed={NUMBA_AVAILABLE} use_symm={getattr(acc, 'use_symm', '?')} "
            f"symm_mask_all_ones={getattr(acc, '_prof_symm_all_ones', '?')}"
        )
        if inner > 0:
            lines.append("  inside ABCDMC (__call__):")
            for k in sorted(secs, key=lambda x: -secs[x]):
                p = 100.0 * secs[k] / inner
                lines.append(
                    f"    {k:22s} {secs[k]:8.4f}s  {p:5.1f}%  {_profile_bar(p)}"
                )
    return "\n".join(lines)


def bdmc_profile_end_step(accumulators):
    global _bdmc_step
    if not boson_dmc_profile_enabled():
        return
    _bdmc_step += 1
    n = boson_dmc_profile_print_every()
    if n <= 0 or _bdmc_step % n != 0:
        return
    bdmc_profile_print(bdmc_profile_format_report(accumulators))
    bdmc_profile_reset_window()
    acc = accumulators.get(ABCDMC_ACC_KEY) if accumulators else None
    if acc is not None and hasattr(acc, "_prof_secs"):
        acc._prof_secs.clear()
    en_acc = accumulators.get(ABQMC_ENERGY_ACC_KEY) if accumulators else None
    if en_acc is not None and hasattr(en_acc, "_prof_secs"):
        en_acc._prof_secs.clear()


try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, fastmath=True)
    def _accumulate_delta_vmc_contributions_numba(
        delta_out,  # Output array to accumulate into (nconf, ndets, ndets)
        grad_psi_n,  # (ndets, 3, nconf) - already computed grad_psi_n
        psi_n,  # (ndets, nconf)
        grad_j,  # (3, nconf)
    ):
        """
        Accumulate delta contributions for VMC (ABVMC) for one electron.
        Implements: einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)
        
        Computes:
        delta += sum over x: psi_n[l, c] * grad_j[x, c] * grad_psi_n[n, x, c]
        """
        nconf, ndets = psi_n.shape[1], psi_n.shape[0]
        
        for c in range(nconf):  # Loop over configurations
            for l in range(ndets):  # Loop over determinant l
                for n in range(ndets):  # Loop over determinant n
                    # Sum over spatial dimensions x
                    for x in range(3):
                        delta_out[c, l, n] += psi_n[l, c] * grad_j[x, c] * grad_psi_n[n, x, c]

    @jit(nopython=True, cache=True, fastmath=True)
    def _accumulate_ovlp_ij_numba(
        ovlp_out,  # Output array (nconf, ndets, ndets)
        psi_n_conj,  # (ndets, nconf)
        psi_n,  # (ndets, nconf)
        symm_mask,  # (ndets, ndets)
    ):
        """
        Compute overlap matrix ovlp_ij[c,l,n] = psi_n_conj[l,c] * psi_n[n,c].
        Implements: einsum("lc,nc->cln", psi_n_conj, psi_n)
        Zeros elements where symm_mask[l,n] is False.
        """
        nconf, ndets = psi_n.shape[1], psi_n.shape[0]
        
        for l in range(ndets):
            for n in range(ndets):
                if symm_mask[l, n]:
                    for c in range(nconf):
                        ovlp_out[c, l, n] = psi_n_conj[l, c] * psi_n[n, c]
                else:
                    for c in range(nconf):
                        ovlp_out[c, l, n] = 0.0

    @jit(nopython=True, cache=True, fastmath=True)
    def _accumulate_delta_dmc_contributions_numba(
        delta_out,  # Output array to accumulate into (nconf, ndets, ndets)
        psi_n_conj,  # (ndets, nconf) - conjugated psi_n
        psi_n,  # (ndets, nconf)
        lap_phi_n,  # (ndets, nconf) from gradient_laplacian_dets
        lap_phi_b,  # (nconf,)
        loggrad_phi_n,  # (ndets, 3, nconf)
        loggrad_b,  # (3, nconf)
        grad_j,  # (3, nconf)
        symm_mask,  # (ndets, ndets)
    ):
        """
        Accumulate all delta contributions for one electron.
        Combines three einsum operations into a single numba-compiled loop.
        
        Computes:
        1. delta1c_e = einsum('lc, cn, nc->cln', psi_n_conj, lap_phi_n, psi_n)
        2. delta1d_e = -einsum('lc, c, nc->cln', psi_n_conj, lap_phi_b, psi_n)
        3. grad_psi_n = psi_n[:, np.newaxis, :] * (loggrad_phi_n - loggrad_b)
        4. delta2 = einsum('lc, xc, nxc->cln', psi_n_conj, grad_j, grad_psi_n)
        5. delta3 = einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n)
        """
        nconf, ndets = psi_n.shape[1], psi_n.shape[0]
        
        # Temporary array for grad_psi_n (ndets, 3, nconf)
        # We'll compute it on the fly to avoid allocation
        
        
        for l in range(ndets):  # Loop over determinant l
            for n in range(ndets):  # Loop over determinant n
                if symm_mask[l, n]:
                    for c in range(nconf):  # Loop over configurations
                        # 1. delta1c_e contribution: psi_n_conj[l,c] * lap_phi_n[n,c] * psi_n[n,c]
                        # lap_phi_n is (ndets, nconf) = (n, c) from gradient_laplacian_dets
                        delta_out[c, l, n] += psi_n_conj[l, c] * lap_phi_n[c, n] * psi_n[n, c]
                        
                        # 2. delta1d_e contribution: -psi_n_conj[l,c] * lap_phi_b[c] * psi_n[n,c]
                        delta_out[c, l, n] -= psi_n_conj[l, c] * lap_phi_b[c] * psi_n[n, c]
                        
                        # 3. Compute grad_psi_n contributions
                        # grad_psi_n[n, x, c] = psi_n[n, c] * (loggrad_phi_n[n, x, c] - loggrad_b[x, c])
                        for x in range(3):  # Loop over spatial dimensions
                            grad_psi_n_val = psi_n[n, c] * (loggrad_phi_n[n, x, c] - loggrad_b[x, c])
                            
                            # 4. delta2 contribution: psi_n_conj[l,c] * grad_j[x,c] * grad_psi_n[n,x,c]
                            delta_out[c, l, n] += psi_n_conj[l, c] * grad_j[x, c] * grad_psi_n_val
                            
                            # 5. delta3 contribution: grad_psi_n[l,x,c] * grad_psi_n[n,x,c]
                            grad_psi_n_l = psi_n[l, c] * (loggrad_phi_n[l, x, c] - loggrad_b[x, c])
                            delta_out[c, l, n] += grad_psi_n_l * grad_psi_n_val
                else:
                    for c in range(nconf):
                        delta_out[c, l, n] = 0.0
else:
    # Fallback to numpy einsum if numba is not available
    def _accumulate_ovlp_ij_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available. Install numba to use this optimization.")

    def _accumulate_delta_vmc_contributions_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available. Install numba to use this optimization.")
    
    def _accumulate_delta_dmc_contributions_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available. Install numba to use this optimization.")


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

def boson_gradient_generator(mf, wf, to_opt=None, nodal_cutoff=1e-3, **ewald_kwargs):
    mf_inputs = {}
    try:
        mf_inputs['dm'] = mf.make_rdm1()
    except:
        print("WARNING: mf.make_rdm1() is not available, cannot use DFT as Mean Field")

    rho, grids = calculate_mf_density(mf.mol, mf_inputs['dm'])
    mf_inputs.update({'xc':mf.xc,
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
            self.mf_inputs = mf_inputs
        except:
            if 'mol' in kwargs:
                self.mol = kwargs['mol']
                del kwargs['mol']
            else:
                raise ValueError("mol is not in mf_inputs or kwargs")

            if 'mf_inputs' in kwargs:
                self.mf_inputs = kwargs['mf_inputs']
                del kwargs['mf_inputs']
            else:
                raise ValueError("mf_inputs is not in mf_inputs or kwargs")
        if hasattr(self.mol, "a"):
            self.coulomb = ewald.Ewald(self.mol, **kwargs)
        else:
            self.coulomb = energy.OpenCoulomb(self.mol, **kwargs)
        self._prof_secs = {}

    def _abqmc_energy_prof_add(self, key, t0):
        if not boson_dmc_profile_enabled():
            return time.perf_counter()
        self._prof_secs[key] = self._prof_secs.get(key, 0.0) + (
            time.perf_counter() - t0
        )
        return time.perf_counter()

    @timer_func
    def __call__(self, configs, wf):
        do_prof = boson_dmc_profile_enabled()
        te = time.perf_counter() if do_prof else None

        ee, ei, ii = self.coulomb.energy(configs)
        if do_prof:
            te = self._abqmc_energy_prof_add("coulomb_energy", te)

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
        if do_prof:
            te = self._abqmc_energy_prof_add("nelec_resolve", te)

        v_mf, ecorr, saved_results = bosonenergy.dft_energy(self.mf_inputs, configs)
        if do_prof:
            te = self._abqmc_energy_prof_add("dft_energy", te)

        ke1, ke2, grad2 = bosonenergy.boson_kinetic(configs, wf)
        if do_prof:
            te = self._abqmc_energy_prof_add("boson_kinetic", te)

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
            'v_mf': v_mf,
            # Eq. 21-22 in doi: 10.1063/5.0155513 is the electronic energy
            # Therefore ii term is added here
            # V_MF = V_H + V_XC (LDA or PBE local vrho via bosonenergy)
            # E_Corr is the sum of KS eigenvalues 
            "total": ke + ee - (v_mf) + ecorr + ii,
        }
        if len(saved_results.keys()) > 0:
            energies.update(saved_results)
        if do_prof:
            self._abqmc_energy_prof_add("en_dict_build", te)
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

def get_psi_basis(boson_wf, phi_b = None, phi_n = None):
    """Calculate the basis functions for the bosonic wave function.

    This function computes the basis functions Phi_l/Phi_B used in the bosonic wave function
    expansion, where Phi_l are the individual Slater determinants and Phi_B is the total 
    bosonic trial wave function.

    Args:
        boson_wf: A BosonWF object representing the bosonic wave function
        phi_b: Optional pre-computed Phi_B array (nconf,). If None, will be computed.
        phi_n: Optional pre-computed Phi_n array (nconf, ndets). If None, will be computed.

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
    if phi_b is None:
        phase, log_val = boson_wf.value() # log(Phi_B) Eq. 4
        phi_b = phase * np.nan_to_num(np.exp(log_val)) #Phi_B
    
    if phi_n is None:
        phases, log_vals = boson_wf.value_dets() #log(Phi_l)
        phi_n = phases * np.nan_to_num(np.exp(log_vals)) # Phi_l
    
    # psi_basis = np.einsum('cn, c->nc', psis, 1./val) # Phi_l/Phi_B, eq. 14
    # # Optimized: use broadcasting instead of einsum for better performance
    psi_n = (phi_n / phi_b[:, np.newaxis]).T  # Phi_n/Phi_B, eq. 14
    return psi_n

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
    def __init__(self, mf_inputs, use_32bit=True, use_symm=False, **kwargs):
        """
        Args:
            mf_inputs: Mean field inputs
            use_32bit: If True, use float32/complex64 instead of float64/complex128
                       This halves memory usage and may improve performance
            use_symm: If True, apply symmetry mask to delta and ovlp when available
            **kwargs: Additional arguments passed to ABQMCEnergyAccumulator
        """
        self.en_acc = ABQMCEnergyAccumulator(mf_inputs, **kwargs)
        self.use_32bit = use_32bit
        self.use_symm = use_symm # Useful for numba
        if use_symm:
            self._symm_mask = None
        # Pre-allocate arrays to avoid repeated allocation
        # Will be resized on first call or if dimensions change
        self._delta = None
        self._delta_shape = None
        self._ovlp_ij = None
        self._ovlp_ij_shape = None
        
    @timer_func
    def __call__(self, configs, wf):
        wave_functions = wf.wf_factors
        boson_wf = None
        jastrow_wf = None
        for wave in wave_functions:
            if isinstance(wave, bosonslater.BosonWF):
                boson_wf = wave
            if isinstance(wave, jastrowspin.JastrowSpin):
                jastrow_wf = wave        
        
        if self._symm_mask is None:
            self._symm_mask = boson_wf._det_prod_filter
        
        nconf, nelec, _ = configs.configs.shape
        boson_value = boson_wf.value()
        phi_b = boson_value[0] * np.nan_to_num(np.exp(boson_value[1])) # Phi_B
        phi_n_value = boson_wf.value_dets()   # phase(Phi_n), log(Phi_n)
        phi_n = phi_n_value[0] * np.nan_to_num(np.exp(phi_n_value[1])) # Phi_n
        psi_n = get_psi_basis(boson_wf, phi_n=phi_n, phi_b=phi_b) # Phi_n/Phi_B
        ndets = psi_n.shape[0]
        
        # Determine target dtype: convert to 32-bit if requested (halves memory usage)
        # Convert EARLY to avoid wasteful 64-bit computations
        if self.use_32bit:
            if psi_n.dtype == np.complex128:
                target_dtype = np.complex64
            elif psi_n.dtype == np.float64:
                target_dtype = np.float32
            else:
                target_dtype = psi_n.dtype
            # Convert psi_n immediately - all subsequent operations will use 32-bit
            psi_n = psi_n.astype(target_dtype, copy=False)  # copy=False if possible
        else:
            target_dtype = psi_n.dtype
        
        # variant 1, using Acceptance from VMC
        # acc = copy.deepcopy(wf.accept_array)
        # facc = np.sum(acc, axis=0)/nelec
        # ovlp_ij = nconf /np.sum(facc) * np.einsum("lc,nc,c->cln", psi_basis.conj(), psi_basis, facc)
        # variant 2 do not use acceptance from VMC 
        # Optimized: reuse pre-allocated array for einsum output to avoid allocation overhead
        ovlp_ij_shape = (nconf, ndets, ndets)
        if self._ovlp_ij is None or self._ovlp_ij_shape != ovlp_ij_shape:
            # Allocate new array if first call or size changed
            self._ovlp_ij = np.zeros(ovlp_ij_shape, dtype=target_dtype)
            self._ovlp_ij_shape = ovlp_ij_shape
        else:
            self._ovlp_ij.fill(0)
        ovlp_ij = self._ovlp_ij
        
        # Use einsum with out parameter to write directly into pre-allocated array
        # psi_n is already in target_dtype, so no conversion needed
        np.einsum("lc,nc->cln", psi_n.conj(), psi_n, out=ovlp_ij, optimize='optimal')
        # ovlp_ij2 = psi_n.conj()[None, :, :] * psi_n[:, None, :]
        # en_acc = self.en_acc(configs, wf)
        # # eb0 = en_acc['total'] - en_acc['corr']
        # mean_eb0 = np.mean(eb0, axis=0)*np.ones_like(eb0)

        # delta = np.einsum('lc, n, nc->cln', psi_n, np.diag(boson_wf.hmf), psi_n) 
        # Optimized: reuse pre-allocated array if size matches, otherwise reallocate
        # Use same dtype as ovlp_ij for consistency
        delta_shape = (nconf, ndets, ndets)
        if self._delta is None or self._delta_shape != delta_shape:
            # Allocate new array if first call or size changed
            self._delta = np.zeros(delta_shape, dtype=target_dtype)
            self._delta_shape = delta_shape
        else:
            # Reuse existing array and reset to zero (faster than reallocating)
            self._delta.fill(0)
        delta = self._delta
        # delta = 0
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

            loggrad_phi_n, loggrad_b = boson_wf.gradient_dets(e, epos_s) 
            # loggrad_b = boson_wf.gradient(e, epos_s) # ∇log(Psi_B) eq. 4
            grad_j = -jastrow_wf.gradient(e, epos_s)
            # Convert gradients to target dtype if using 32-bit (avoids 64-bit intermediate computations)
            # Converting BEFORE operations is more efficient than converting after
            if self.use_32bit:
                if loggrad_phi_n.dtype != target_dtype:
                    loggrad_phi_n = loggrad_phi_n.astype(target_dtype, copy=False)
                if loggrad_b.dtype != target_dtype:
                    loggrad_b = loggrad_b.astype(target_dtype, copy=False)
                if grad_j.dtype != target_dtype:
                    grad_j = grad_j.astype(target_dtype, copy=False)
            
            # grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n - loggrad_b, optimize='optimal')  
            # Optimized: use broadcasting instead of einsum for better performance
            # gradient_dets returns (ndet, 3, nconf), so we need to add newaxis in the middle
            # psi_n is (ndet, nconf), so psi_n[:, np.newaxis, :] is (ndet, 1, nconf)
            # This broadcasts correctly with (ndet, 3, nconf) to give (ndet, 3, nconf)
            # All arrays are now in target_dtype, so operations stay in 32-bit
            grad_psi_n = psi_n[:, np.newaxis, :] * (loggrad_phi_n - loggrad_b)
            if NUMBA_AVAILABLE and not np.iscomplexobj(psi_n):
                _accumulate_delta_vmc_contributions_numba(
                    delta, 
                    grad_psi_n, 
                    psi_n, 
                    grad_j)
            else:
                delta += np.einsum("lc,xc,nxc->cln", psi_n, grad_j, grad_psi_n)
            # import pdb; pdb.set_trace()
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
    def __init__(self, mf_inputs, system_params, use_symm=False, **kwargs):
        """
        Args:
            mf_inputs: Mean field inputs
            system_params: Dict with 'dtype' and other system parameters
            use_symm: If True, apply symmetry mask to delta and ovlp when available
            **kwargs: Additional arguments passed to ABQMCEnergyAccumulator
        """
        # self.en_acc = ABQMCEnergyAccumulator(mf_inputs, **kwargs)
        self.use_symm = use_symm
        self._symm_mask = None  # Set from boson_wf._det_prod_filter on first call when use_symm
        self.dtype = system_params['dtype']
        self.memallocated = False
        self.nconf = None #system_params['nconf']
        self.ndets = None #system_params['ndets']
        self.nelec = None #system_params['nelec']
        self._ovlp_ij = None #np.zeros((self.nconf, self.ndets, self.ndets), dtype=self.dtype)
        self._delta = None #np.zeros((self.nconf, self.ndets, self.ndets), dtype=self.dtype)
        self._grad_psi_n = None #np.zeros((self.ndets, 3, self.nconf), dtype=self.dtype)        
        # self._phi_n = np.zeros((self.ndets, self.nconf), dtype=self.dtype)
        # self._phi_b = np.zeros((self.nconf,), dtype=self.dtype)
        # self._psi_n = np.zeros((self.ndets, self.nconf), dtype=self.dtype)
        if not hasattr(self, '_boson_wf_type'):
            self._boson_wf_type = bosonslater.BosonWF
            self._jastrow_wf_type = jastrowspin.JastrowSpin
        if NUMBA_AVAILABLE:
            print('Numba is available')
        self._prof_secs = {}

    def _prof_add(self, key, t0):
        self._prof_secs[key] = self._prof_secs.get(key, 0.0) + (time.perf_counter() - t0)
        return time.perf_counter()

    @timer_func
    def __call__(self, configs, wf, use_symm=None):
        do = boson_dmc_profile_enabled()
        if do:
            bdmc_profile_ensure_banner_once()
        t0 = time.perf_counter() if do else None
        for wave in wf.wf_factors:
            if isinstance(wave, self._boson_wf_type):
                boson_wf = wave
            if isinstance(wave, self._jastrow_wf_type):
                jastrow_wf = wave        
        
        if use_symm is None:
            use_symm = self.use_symm
        if self._symm_mask is None:
            self._symm_mask = getattr(boson_wf, '_det_prod_filter', None)

        boson_value = boson_wf.value() # phase(Phi_B), log(Phi_B)
        phi_b = boson_value[0] * np.nan_to_num(np.exp(boson_value[1])) # Phi_B

        phi_n_value = boson_wf.value_dets()   # phase(Phi_n), log(Phi_n)
        phi_n = phi_n_value[0] * np.nan_to_num(np.exp(phi_n_value[1])) # Phi_n
        
        psi_n = get_psi_basis(boson_wf, phi_n=phi_n, phi_b=phi_b) # Phi_n/Phi_B
        psi_n_conj = psi_n.conj()
        
        if not self.memallocated:
            # Stochastic comb ensures that these remain fixed. 
            self.nconf, self.nelec, _ = configs.configs.shape
            self.ndets = boson_wf.num_det
            self._ovlp_ij = np.zeros((self.nconf, self.ndets, self.ndets), dtype=self.dtype)
            self._delta = np.zeros((self.nconf, self.ndets, self.ndets), dtype=self.dtype)
            self._buf_delta = np.zeros((self.nconf, self.ndets, self.ndets), dtype=self.dtype)
            self._grad_psi_n = np.zeros((self.ndets, 3, self.nconf), dtype=self.dtype)
            self.memallocated = True

        if do:
            t0 = self._prof_add("values_psi", t0)

        symm_mask = (self._symm_mask if (use_symm and self._symm_mask is not None)
                     else np.ones((self.ndets, self.ndets), dtype=bool))
        if do:
            self._prof_symm_all_ones = bool(np.all(symm_mask))

        ovlp_ij = self._ovlp_ij
        if NUMBA_AVAILABLE and symm_mask is not None:
            _accumulate_ovlp_ij_numba(
                ovlp_ij,
                psi_n_conj,
                psi_n,
                symm_mask,
            )
            if do:
                t0 = self._prof_add("ovlp_numba", t0)
        else:
            np.einsum("lc,nc->cln", psi_n_conj, psi_n, out=ovlp_ij, optimize='optimal')
            if do:
                t0 = self._prof_add("ovlp_einsum", t0)
        
        delta = self._delta
        delta.fill(0.0)
        for e in range(self.nelec):
            # Get position of electron e
            epos_s = configs.electron(e)

            # All the terms that go into delta calculation
            # lap_phi_n = boson_wf.laplacian_dets(e, epos_s)  # ∇²(Phi_n)/Phi_n
            # loggrad_phi_n, loggrad_b = boson_wf.gradient_dets(e, epos_s)  #∇log(Phi_n) and ∇log(Psi_B) eq. 4
            if do:
                te = time.perf_counter()
            lap_phi_n, loggrad_phi_n, loggrad_b = boson_wf.gradient_laplacian_dets(e, epos_s)  #∇²(Phi_n) and ∇log(Phi_n)
            if do:
                te = self._prof_add("elec_grad_lap", te)
            if do:
                te = time.perf_counter()
            lap_phi_b = boson_wf.laplacian(e, epos_s, 
                                           lap_phi_n=lap_phi_n, 
                                           loggrad_phi_n=loggrad_phi_n, 
                                           phi_n=phi_n, 
                                           phi_b=phi_b)      
            if do:
                te = self._prof_add("elec_lap_b", te)
            if do:
                te = time.perf_counter()
            grad_j = jastrow_wf.gradient(e, epos_s)
            if do:
                te = self._prof_add("elec_jastrow", te)

            if do:
                te = time.perf_counter()
            if NUMBA_AVAILABLE and symm_mask is not None:
                _accumulate_delta_dmc_contributions_numba(
                    delta,
                    psi_n_conj,
                    psi_n,
                    lap_phi_n,
                    lap_phi_b,
                    loggrad_phi_n,
                    loggrad_b,
                    grad_j,
                    symm_mask,
                )
            else:
                np.einsum('lc, cn, nc->cln', psi_n, lap_phi_n, psi_n, out=self._buf_delta,  optimize='optimal')
                delta += self._buf_delta
                np.einsum('lc, c, nc->cln', psi_n, lap_phi_b, psi_n, out=self._buf_delta, optimize='optimal')
                delta -= self._buf_delta
                grad_psi_n = self._grad_psi_n
                np.multiply(psi_n[:, np.newaxis, :], loggrad_phi_n - loggrad_b, out=grad_psi_n)
                np.einsum('lc, xc, nxc->cln', psi_n, grad_j, grad_psi_n, out=self._buf_delta, optimize='optimal')
                delta += self._buf_delta
                np.einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n, out=self._buf_delta, optimize='optimal')
                delta += self._buf_delta
            if do:
                self._prof_add("delta_numba" if (NUMBA_AVAILABLE and symm_mask is not None) else "delta_einsum", te)
            
        results = {'delta': delta,
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

# class ABCDMCMatrixAccumulator_old:
#     """Accumulator for computing matrix elements in Auxiliary-field Boson Corrected Diffusion Monte Carlo (ABCDMC).
    
#     Specifically calculates:
#     1. Overlap matrices between different basis states
#     2. Matrix elements involving kinetic and potential energy terms
    
#     The calculation includes:
#     - Wavefunctions ratios
#     - Gradients and Laplacians of both bosonic and trial wavefunctions
#     - Integration by parts terms for the kinetic energy
    
    
#     Methods
#     -------
#     __call__(configs, wf)
#         Compute matrix elements for given configurations and wavefunction.
        
#         Parameters
#         ----------
#         configs : object
#             Contains electron configurations with shape (nconf, nelec, ndets)
#         wf : object
#             Wavefunction object containing both BosonWF and JastrowSpin components
            
#         Returns
#         -------
#         dict
#             'matel': Matrix elements including kinetic and potential terms
#             'ovlp': Overlap matrices between basis states
            
#     Notes
#     -----
#     The implementation follows quantum Monte Carlo formalism where:
#     - Φ_B is the bosonic wavefunction
#     - Φ_n are the determinant components
#     - Ψ_BT is the Slater Jastrow trial wavefunction (eq. 4)
    
#     The matrix elements are computed using:
#     1. Gradient terms: ∇Ψ_n = ∇(Φ_n/Φ_B)
#     2. Laplacian terms: ∇²(Φ_n/Φ_B)
#     3. Integration by parts for the kinetic energy terms
#     """    
    
#     @timer_func
#     def __call__(self, configs, wf):

#         wave_functions = wf.wf_factors
#         boson_wf = None
#         jastrow_wf = None
#         for wave in wave_functions:
#             if isinstance(wave, bosonslater.BosonWF):
#                 boson_wf = wave
#             if isinstance(wave, jastrowspin.JastrowSpin):
#                 jastrow_wf = wave        
        
#         psi_n = get_psi_basis(boson_wf) # Phi_l/Phi_B
#         # Optimized: use optimal einsum path for better performance with many determinants
#         ovlp_ij = np.einsum("lc,nc->cln", psi_n.conj(), psi_n, optimize='optimal')

#         # phase_fb, logval_fb = wf.value() # log(Psi_BT)
#         # val_fb = phase_fb * np.nan_to_num(np.exp(logval_fb)) # Psi_BT

#         phase_phi_b, logval_phi_b = boson_wf.value() # log(Phi_B)
#         val_phi_b = phase_phi_b * np.nan_to_num(np.exp(logval_phi_b)) # Phi_B

#         # Matrix element by integration parts on the ∇f_B term of the eq. 23 
#         # For integration by parts see eq. 17 in the reference paper
#         matel = np.einsum('lc, ln, nc->cln', psi_n, boson_wf.hmf, psi_n) # Psi_l * H * Psi_n
#         delta = 0
#         # phases, log_vals = boson_wf.value_dets() #log(Phi_l)
#         # psis = phases * np.nan_to_num(np.exp(log_vals)) # Phi_l
        
#         for e in range(self.nelec):
#             # Get position of electron e
#             epos_s = configs.electron(e)

#             # ∇log(Phi_n) 
#             loggrad_phi_n, loggrad_b = boson_wf.gradient_dets(e, epos_s) 
#             # ∇log(Psi_B) eq. 4
#             # loggrad_b = boson_wf.gradient(e, epos_s) 
#             # ∇Psi_n = ∇(Phi_n/Phi_B)
#             # grad_psi_n = np.einsum('nc, nxc->nxc', psi_n, loggrad_phi_n-loggrad_b, optimize='optimal')  
#             # Optimized: use broadcasting instead of einsum for better performance
#             # gradient_dets returns (ndet, 3, nconf), so we need to add newaxis in the middle
#             # psi_n is (ndet, nconf), so psi_n[:, np.newaxis, :] is (ndet, 1, nconf)
#             # This broadcasts correctly with (ndet, 3, nconf) to give (ndet, 3, nconf)
#             grad_psi_n = psi_n[:, np.newaxis, :] * (loggrad_phi_n - loggrad_b)
#             # ∇²(Psi_B)/Psi_B
#             lap_phi_b = boson_wf.laplacian(e, epos_s)      
#             # ∇²Phi_n
#             lap_phi_n = boson_wf.laplacian_dets(e, epos_s) 
#             # ∇log(Psi_B^T)
#             loggrad_psi_bt = wf.gradient(e, epos_s)
#             jgrad          = jastrow_wf.gradient(e, epos_s)
#             # print('DMC-J', e, np.sum(jgrad))
            
#             # lap_psi_n: (eq. before eq. 23)
#             # ∇²(Phi_n/Phi_B) = [∇²(Phi_n)*Phi_B - Phi_n*∇²(Phi_B)]/(Phi_B^2) 
#             #                   - 2*∇(Phi_B)·∇(Phi_n/Phi_B)/Phi_B
#             lap_psi_n  = np.einsum('cn, c->cn', lap_phi_n, 1./val_phi_b) # ∇²(Phi_n)/Phi_B
#             lap_psi_n -= np.einsum('c, nc->cn', lap_phi_b, psi_n) # -∇²(Phi_B)/Phi_B * Psi_n or -∇²(Phi_B)/Phi_B^2 * Phi_n 
#             lap_psi_n -= 2 * np.einsum('xc, nxc->cn', loggrad_b, grad_psi_n) # - 2*∇(log(Psi_B))*∇(Psi_n)
#             # Optimized: use broadcasting and optimal einsum paths
#             # lap_psi_n = (lap_phi_n / val_phi_b[:, np.newaxis])  # ∇²(Phi_n)/Phi_B
#             # lap_psi_n -= lap_phi_b[:, np.newaxis] * psi_n.T  # -∇²(Phi_B)/Phi_B * Psi_n or -∇²(Phi_B)/Phi_B^2 * Phi_n 
#             # lap_psi_n -= 2 * np.einsum('xc, nxc->cn', loggrad_b, grad_psi_n, optimize='optimal') # - 2*∇(log(Psi_B))*∇(Psi_n)

#             # Optimized: use optimal einsum paths
#             delta1 = np.einsum('lxc, nxc->cln', grad_psi_n, grad_psi_n, optimize='optimal') # ∇Psi_l \dot ∇Psi_n 
#             delta2 = np.einsum('lc, cn->cln', psi_n, lap_psi_n, optimize='optimal') # Psi_l * ∇²Psi_n
            
#             delta3 = np.einsum('lc, xc, nxc->cln', psi_n, loggrad_b + loggrad_psi_bt, grad_psi_n, optimize='optimal') # Psi_l * [∇(log(Phi_B)) + ∇(log(Psi_BT))] \dot ∇Psi_n        
#             # delta4 = np.einsum('lc, xc, nxc->cln', psi_n, -2 * loggrad_b, grad_psi_n)
#             # delta6 = np.einsum('lc, xc, nxc->cln', psi_n, -2 * loggrad_psi_bt, grad_psi_n)
#             # delta5 = delta1 + delta2
#             delta += delta1 + delta2 + delta3
#             # print('DMC', e, np.sum(grad_psi_n), np.sum(psi_n), np.sum(delta1), np.sum(delta2), np.sum(delta3), np.sum(delta), delta[0,0,0])
#             # print()
#             # ndets = lap_phi_n.shape[1]
#             # print(e,ndets)
#             # for i in range(ndets):
#             #     a = np.sum(delta5, axis=0)[i,i]
#             #     b = np.sum(delta6, axis=0)[i,i]
#             #     print(a, b, b/a)
#             # import pdb; pdb.set_trace()
#             # matel += delta
            
#         # exit()
#         results = { #'matel':matel, 
#                    'delta': delta,
#                    'ovlp': ovlp_ij}
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




