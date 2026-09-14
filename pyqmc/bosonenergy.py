"""Mean-field and bosonic kinetic terms for ABVMC / ABCDMC.

For DFT mean field, the local XC potential at walker positions uses libxc
``vrho`` (``eval_xc(...)[1][0]``). That is the full KS potential for LDA.
For GGA (PBE), it is the local spin density potential only; the GGA
``vsigma`` / ∇·(vσ ∇ρ) contribution that appears in PySCF's Vxc matrix is
not included. This matches the ABVMC formulation in Eq. 21 of
doi: 10.1063/5.0155513.

Optional real-space interpolation (``use_interpolation_mf``) tabulates the
fixed Hartree (Vj) and local XC (vrho) potentials on a Cartesian grid once
and evaluates them at walker positions via linear interpolation.

Grid construction can be parallelized over coordinate chunks with
``mf_interp_nworkers`` (ProcessPoolExecutor on the driver).
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pyscf.dft import libxc, numint
from scipy.interpolate import RegularGridInterpolator

SUPPORTED_XC = ("LDA,VWN", "PBE,PBE", "HF")

# PySCF xctype and AO derivative order for each XC string.
XC_KIND = {
    "LDA,VWN": ("LDA", 0),
    "PBE,PBE": ("GGA", 1),
}

# Defaults for optional MF potential interpolation (Bohr).
DEFAULT_MF_INTERP_SPACING = 0.15
DEFAULT_MF_INTERP_PADDING = 5.0
DEFAULT_MF_INTERP_CHUNK = 2048

# Process-pool worker state (set by initializer; not used on the driver).
_POOL_STATE = {}


def _normalize_xc(xc):
    xc = xc.replace(" ", "")
    if xc not in SUPPORTED_XC:
        raise ValueError(f"Unsupported xc={xc!r}; expected one of {SUPPORTED_XC}")
    return xc


def _force_single_blas_thread():
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _init_mf_interp_pool(mol, dm_total, dm, xc):
    """Initializer for ProcessPool workers building MF tables."""
    _force_single_blas_thread()
    _POOL_STATE["mol"] = mol
    _POOL_STATE["dm_total"] = dm_total
    _POOL_STATE["dm"] = dm
    _POOL_STATE["xc"] = xc


def _vj_chunk_worker(coords):
    mol = _POOL_STATE["mol"]
    dm_total = _POOL_STATE["dm_total"]
    ints = mol.intor("int1e_grids", grids=coords)
    return np.einsum("pij,ij->p", ints, dm_total)


def _vrho_chunk_worker(coords):
    return eval_vrho(
        _POOL_STATE["mol"],
        _POOL_STATE["dm"],
        _POOL_STATE["xc"],
        coords,
        spin=1,
    )


def eval_vrho(mol, dm, xc, coords, spin=1):
    """Local spin-resolved vrho from libxc at ``coords``."""
    xc = _normalize_xc(xc)
    if xc == "HF":
        raise ValueError("HF has no libxc vrho")

    xctype, deriv = XC_KIND[xc]
    ao = numint.eval_ao(mol, coords, deriv=deriv)
    rho_up = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
    rho_dn = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
    vrho = np.asarray(libxc.eval_xc(xc, (rho_up, rho_dn), spin=spin)[1][0])
    if vrho.ndim == 1:
        vrho = np.stack([vrho, vrho], axis=1)
    return vrho


def get_vxc(configs, mol, dm, nelec, xc):
    """Sum libxc vrho over electrons for each walker configuration."""
    nconf, nelec_cfg, _ = configs.configs.shape
    nup = nelec[0]
    if nelec_cfg != sum(nelec):
        raise ValueError("configs electron count inconsistent with mf_inputs['nelec']")

    coords = configs.configs.reshape(-1, 3)
    vrho = eval_vrho(mol, dm, xc, coords, spin=1)
    vrho = vrho.reshape(nconf, nelec_cfg, 2)

    spin_idx = np.array([int(e >= nup) for e in range(nelec_cfg)])
    return np.sum([vrho[:, i, spin_idx[i]] for i in range(nelec_cfg)], axis=0)


def get_vj(configs, mol, dm):
    """Hartree potential of total SCF density at electron positions, summed per walker."""
    nconf, nelec, _ = configs.configs.shape
    dm_total = dm[0] + dm[1]
    r = configs.configs.reshape(-1, 3)
    vj_all = _eval_vj_on_coords(mol, dm_total, r)
    return vj_all.reshape(nconf, nelec).sum(axis=1)


def _coord_chunks(coords, chunk_size):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    return [coords[i0:min(i0 + chunk_size, n)] for i0 in range(0, n, chunk_size)]


def _eval_vj_on_coords(
    mol,
    dm_total,
    coords,
    chunk_size=DEFAULT_MF_INTERP_CHUNK,
    nworkers=1,
    dm=None,
    xc=None,
):
    """Evaluate Vj(r) = Tr[dm · int1e_grids(r)] at many points, chunked for memory."""
    coords = np.asarray(coords, dtype=float)
    chunks = _coord_chunks(coords, chunk_size)
    if nworkers is None or int(nworkers) <= 1 or len(chunks) <= 1:
        out = np.empty(coords.shape[0], dtype=float)
        i0 = 0
        for chunk in chunks:
            ints = mol.intor("int1e_grids", grids=chunk)
            out[i0 : i0 + chunk.shape[0]] = np.einsum("pij,ij->p", ints, dm_total)
            i0 += chunk.shape[0]
        return out

    nworkers = min(int(nworkers), len(chunks))
    with ProcessPoolExecutor(
        max_workers=nworkers,
        initializer=_init_mf_interp_pool,
        initargs=(mol, dm_total, dm, xc),
    ) as pool:
        parts = list(pool.map(_vj_chunk_worker, chunks, chunksize=1))
    return np.concatenate(parts, axis=0)


def _eval_vrho_chunked(
    mol,
    dm,
    xc,
    coords,
    spin=1,
    chunk_size=DEFAULT_MF_INTERP_CHUNK,
    nworkers=1,
    dm_total=None,
):
    coords = np.asarray(coords, dtype=float)
    chunks = _coord_chunks(coords, chunk_size)
    if nworkers is None or int(nworkers) <= 1 or len(chunks) <= 1:
        return np.concatenate(
            [eval_vrho(mol, dm, xc, chunk, spin=spin) for chunk in chunks], axis=0
        )

    nworkers = min(int(nworkers), len(chunks))
    # spin is fixed to 1 in the pool worker (ABVMC path); keep serial if spin!=1
    if spin != 1:
        return np.concatenate(
            [eval_vrho(mol, dm, xc, chunk, spin=spin) for chunk in chunks], axis=0
        )

    with ProcessPoolExecutor(
        max_workers=nworkers,
        initializer=_init_mf_interp_pool,
        initargs=(mol, dm_total if dm_total is not None else dm[0] + dm[1], dm, xc),
    ) as pool:
        parts = list(pool.map(_vrho_chunk_worker, chunks, chunksize=1))
    return np.concatenate(parts, axis=0)


def cartesian_grid_axes(mol, spacing=DEFAULT_MF_INTERP_SPACING, padding=DEFAULT_MF_INTERP_PADDING):
    """Uniform Cartesian axes in Bohr enclosing the molecule plus padding."""
    atoms = np.asarray(mol.atom_coords(), dtype=float)
    lo = atoms.min(axis=0) - padding
    hi = atoms.max(axis=0) + padding
    axes = []
    for a, b in zip(lo, hi):
        n = max(int(np.ceil((b - a) / spacing)) + 1, 2)
        axes.append(np.linspace(a, b, n))
    return tuple(axes)


def mf_interp_grid_stats(
    mol,
    spacing=DEFAULT_MF_INTERP_SPACING,
    padding=DEFAULT_MF_INTERP_PADDING,
    dtype=np.float64,
):
    """Return grid shape, point count, and bytes for Vj + Vxc↑ + Vxc↓ tables."""
    x, y, z = cartesian_grid_axes(mol, spacing=spacing, padding=padding)
    shape = (x.size, y.size, z.size)
    n_points = int(np.prod(shape))
    itemsize = np.dtype(dtype).itemsize
    # Three scalar fields on the grid (Hartree + two spin XC channels)
    n_arrays = 3
    nbytes = n_points * itemsize * n_arrays
    return {
        "shape": shape,
        "n_points": n_points,
        "nbytes": nbytes,
        "dtype": np.dtype(dtype),
        "n_arrays": n_arrays,
    }


def format_bytes(nbytes):
    """Human-readable byte count."""
    nbytes = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(nbytes)} {unit}"
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} TiB"


def build_mf_potential_tables(
    mol,
    dm,
    xc,
    spacing=DEFAULT_MF_INTERP_SPACING,
    padding=DEFAULT_MF_INTERP_PADDING,
    chunk_size=DEFAULT_MF_INTERP_CHUNK,
    nworkers=1,
):
    """Tabulate Vj and spin-resolved Vxc on a Cartesian grid (Bohr).

    Parameters
    ----------
    nworkers : int
        Process-pool size for chunked Vj/Vxc evaluation. ``1`` is serial.
    """
    xc = _normalize_xc(xc)
    if xc == "HF":
        raise ValueError("MF interpolation is not supported for HF; use LDA/PBE")

    x, y, z = cartesian_grid_axes(mol, spacing=spacing, padding=padding)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    shape = (x.size, y.size, z.size)
    dm_total = dm[0] + dm[1]
    chunks = _coord_chunks(coords, chunk_size)
    nworkers = 1 if nworkers is None else max(int(nworkers), 1)

    t0 = time.perf_counter()
    if nworkers <= 1 or len(chunks) <= 1:
        vj = _eval_vj_on_coords(
            mol, dm_total, coords, chunk_size=chunk_size, nworkers=1
        ).reshape(shape)
        t_vj = time.perf_counter() - t0
        t0 = time.perf_counter()
        vrho = _eval_vrho_chunked(
            mol, dm, xc, coords, spin=1, chunk_size=chunk_size, nworkers=1
        )
        t_vxc = time.perf_counter() - t0
    else:
        nworkers = min(nworkers, len(chunks))
        try:
            with ProcessPoolExecutor(
                max_workers=nworkers,
                initializer=_init_mf_interp_pool,
                initargs=(mol, dm_total, dm, xc),
            ) as pool:
                vj_parts = list(pool.map(_vj_chunk_worker, chunks, chunksize=1))
                t_vj = time.perf_counter() - t0
                t0 = time.perf_counter()
                vrho_parts = list(pool.map(_vrho_chunk_worker, chunks, chunksize=1))
                t_vxc = time.perf_counter() - t0
            vj = np.concatenate(vj_parts, axis=0).reshape(shape)
            vrho = np.concatenate(vrho_parts, axis=0)
        except (PermissionError, OSError) as exc:
            print(
                f"Warning: ProcessPool MF interp build failed ({exc}); "
                "falling back to serial"
            )
            nworkers = 1
            vj = _eval_vj_on_coords(
                mol, dm_total, coords, chunk_size=chunk_size, nworkers=1
            ).reshape(shape)
            t_vj = time.perf_counter() - t0
            t0 = time.perf_counter()
            vrho = _eval_vrho_chunked(
                mol, dm, xc, coords, spin=1, chunk_size=chunk_size, nworkers=1
            )
            t_vxc = time.perf_counter() - t0

    vxc_up = vrho[:, 0].reshape(shape)
    vxc_dn = vrho[:, 1].reshape(shape)

    print(
        f"MF interp tables: grid={shape} ({coords.shape[0]} pts), "
        f"storage≈{format_bytes(coords.shape[0] * 8 * 3)}, "
        f"nworkers={nworkers}, Vj={t_vj:.2f}s, Vxc={t_vxc:.2f}s"
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "vj": vj,
        "vxc_up": vxc_up,
        "vxc_dn": vxc_dn,
        "xc": xc,
        "spacing": float(spacing),
        "padding": float(padding),
        "nworkers": nworkers,
    }


def _make_interpolator(axes, values):
    return RegularGridInterpolator(
        axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def attach_mf_interpolators(
    mf_inputs,
    spacing=None,
    padding=None,
    chunk_size=None,
    nworkers=None,
    force=False,
):
    """Build and store RegularGridInterpolators on ``mf_inputs`` for Vj / Vxc.

    Keys written:
      - mf_interp: dict with vj, vxc_up, vxc_dn interpolators and meta
      - use_interpolation_mf: True
    """
    if mf_inputs.get("mf_interp") is not None and not force:
        mf_inputs["use_interpolation_mf"] = True
        return mf_inputs

    xc = _normalize_xc(mf_inputs["xc"])
    if xc == "HF":
        raise ValueError(
            "use_interpolation_mf requires LDA/PBE; HF uses the AO V_eff path"
        )

    if spacing is None:
        spacing = mf_inputs.get("mf_interp_spacing", DEFAULT_MF_INTERP_SPACING)
    if padding is None:
        padding = mf_inputs.get("mf_interp_padding", DEFAULT_MF_INTERP_PADDING)
    if chunk_size is None:
        chunk_size = mf_inputs.get("mf_interp_chunk", DEFAULT_MF_INTERP_CHUNK)
    if nworkers is None:
        nworkers = mf_inputs.get("mf_interp_nworkers", 1)

    tables = build_mf_potential_tables(
        mf_inputs["mol"],
        mf_inputs["dm"],
        xc,
        spacing=spacing,
        padding=padding,
        chunk_size=chunk_size,
        nworkers=nworkers,
    )
    axes = (tables["x"], tables["y"], tables["z"])
    mf_inputs["mf_interp"] = {
        "vj": _make_interpolator(axes, tables["vj"]),
        "vxc_up": _make_interpolator(axes, tables["vxc_up"]),
        "vxc_dn": _make_interpolator(axes, tables["vxc_dn"]),
        "meta": {
            "xc": tables["xc"],
            "spacing": tables["spacing"],
            "padding": tables["padding"],
            "shape": tables["vj"].shape,
            "nworkers": tables["nworkers"],
        },
    }
    mf_inputs["mf_interp_spacing"] = tables["spacing"]
    mf_inputs["mf_interp_padding"] = tables["padding"]
    mf_inputs["mf_interp_nworkers"] = tables["nworkers"]
    mf_inputs["use_interpolation_mf"] = True
    return mf_inputs


def get_vj_interpolated(configs, interp_vj):
    """Sum interpolated Hartree potential over electrons per walker."""
    nconf, nelec, _ = configs.configs.shape
    r = configs.configs.reshape(-1, 3)
    return np.asarray(interp_vj(r), dtype=float).reshape(nconf, nelec).sum(axis=1)


def get_vxc_interpolated(configs, nelec, interp_vxc_up, interp_vxc_dn):
    """Sum interpolated spin-selected Vxc over electrons per walker."""
    nconf, nelec_cfg, _ = configs.configs.shape
    nup = nelec[0]
    if nelec_cfg != sum(nelec):
        raise ValueError("configs electron count inconsistent with mf_inputs['nelec']")

    r = configs.configs.reshape(-1, 3)
    v_up = np.asarray(interp_vxc_up(r), dtype=float).reshape(nconf, nelec_cfg)
    v_dn = np.asarray(interp_vxc_dn(r), dtype=float).reshape(nconf, nelec_cfg)
    spin_idx = np.array([int(e >= nup) for e in range(nelec_cfg)])
    parts = [v_dn[:, i] if spin_idx[i] else v_up[:, i] for i in range(nelec_cfg)]
    return np.sum(parts, axis=0)


def dft_energy(mf_inputs, configs):
    """
    Returns the KS related terms in Eq. 21 in doi: 10.1063/5.0155513.

    If ``mf_inputs['use_interpolation_mf']`` is true, Vj and Vxc are taken from
    pre-tabulated Cartesian interpolators (built on first use if missing).

    Returns:
        v_mf: V_H + V_XC summed over electrons (per walker)
        ecorr: sum of occupied KS eigenvalues (E_0^MF)
        saved_results: dict with vj, vxc when applicable
    """
    nconf, nelec, _ = configs.configs.shape
    xc = _normalize_xc(mf_inputs["xc"])
    nup_dn = mf_inputs["nelec"]
    mo_energy = mf_inputs["mo_energy"]
    mo_occ = mf_inputs["mo_occ"]
    mol = mf_inputs["mol"]
    dm = mf_inputs["dm"]
    use_interp = bool(mf_inputs.get("use_interpolation_mf", False))

    if xc != "HF":
        if use_interp:
            if mf_inputs.get("mf_interp") is None:
                attach_mf_interpolators(mf_inputs)
            inter = mf_inputs["mf_interp"]
            vj = get_vj_interpolated(configs, inter["vj"])
            vxc = get_vxc_interpolated(
                configs, nup_dn, inter["vxc_up"], inter["vxc_dn"]
            )
        else:
            vj = get_vj(configs, mol, dm)
            vxc = get_vxc(configs, mol, dm, nup_dn, xc)
        ecorr = np.sum(mo_energy * mo_occ)
        v_mf = vj + vxc
        saved_results = {"vj": vj, "vxc": vxc}
    else:
        if use_interp:
            raise ValueError(
                "use_interpolation_mf is not supported for HF mean field"
            )
        v_mf = np.zeros(nconf)
        ecorr = np.sum(mo_energy * mo_occ)
        V_eff_ao = mf_inputs["veff"]
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            ao_value = numint.eval_ao(mol, configs.configs[:, e, :])
            v_mf = np.einsum("gp, pq, gq -> g", ao_value, V_eff_ao[s], ao_value)
        saved_results = {}

    return v_mf, ecorr, saved_results


def boson_kinetic(configs, wf):
    """
    Returns the jastrow laplacian (lap_j) and the bosonic drift (drift_b) terms
    in Eq. 21 in doi: 10.1063/5.0155513.
    """
    nconf, nelec, _ = configs.configs.shape

    has_jastrow = True
    try:
        wave_functions = wf.wf_factors
    except AttributeError:
        has_jastrow = False
        wave_functions = [wf]

    jastrow_wf = None
    boson_wf = None
    from pyqmc import bosonslater
    from pyqmc import jastrowspin

    for wave in wave_functions:
        if isinstance(wave, bosonslater.BosonWF):
            boson_wf = wave
        if isinstance(wave, jastrowspin.JastrowSpin):
            jastrow_wf = wave

    lap_j = np.zeros(nconf)
    drift_b = np.zeros(nconf)
    grad2 = np.zeros(nconf)
    if has_jastrow:
        for e in range(nelec):
            grad_je, lap_je = jastrow_wf.gradient_laplacian(e, configs.electron(e))
            lap_j += -0.5 * (lap_je.real + np.sum(grad_je.real**2, axis=0))
            grad_b = boson_wf.gradient(e, configs.electron(e))
            drift_b -= np.einsum("di,di->i", grad_je, grad_b)
            grad = np.sum([grad_je, grad_b], axis=0)
            grad2 += np.sum(np.abs(grad) ** 2, axis=0)
    return lap_j, drift_b, grad2
