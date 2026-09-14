"""Validate optional Cartesian interpolation of MF Hartree / local XC potentials.

Covers He, Li, Ne and probes when interpolation error grows using fixed
walker geometries. Assertions use relative-to-field errors:

    |V_interp - V_ref| / (|V_ref| + eps)
"""

import numpy as np
import pytest
from pyscf import dft, gto

from pyqmc.bosonenergy import (
    attach_mf_interpolators,
    build_mf_potential_tables,
    dft_energy,
    get_vj,
    get_vxc,
)


# Cartesian table used for agreement tests (Bohr).
SPACING = 0.1
PADDING = 4.0
# Floor so relative error is defined when the reference field is near zero.
REL_EPS = 1e-3

# Atomic systems: (label, atom string, spin, charge, nup, ndn, Z)
ATOMS = [
    ("He", "He 0 0 0", 0, 0, 1, 1, 2),
    ("Li", "Li 0 0 0", 1, 0, 2, 1, 3),
    ("Ne", "Ne 0 0 0", 0, 0, 5, 5, 10),
]


def _run_uks(atom, spin, charge, xc, basis="cc-pvdz"):
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis, unit="angstrom", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.make_rdm1()


@pytest.fixture(scope="module")
def uks_cache():
    """Cache (mol, mf, dm) per (label, xc) across the module."""
    cache = {}

    def get(label, atom, spin, charge, xc):
        key = (label, xc)
        if key not in cache:
            cache[key] = _run_uks(atom, spin=spin, charge=charge, xc=xc)
        return cache[key]

    return get


@pytest.fixture(scope="module")
def interp_cache(uks_cache):
    """Cache interpolated mf_inputs per (label, xc, spacing, padding)."""
    cache = {}

    def get(label, atom, spin, charge, xc, spacing=SPACING, padding=PADDING):
        key = (label, xc, spacing, padding)
        if key not in cache:
            mol, mf, dm = uks_cache(label, atom, spin, charge, xc)
            cache[key] = (
                mol,
                mf,
                dm,
                _mf_inputs(
                    mol, mf, dm, xc, use_interpolation_mf=True, spacing=spacing, padding=padding
                ),
            )
        return cache[key]

    return get


def _mf_inputs(mol, mf, dm, xc, use_interpolation_mf=False, spacing=SPACING, padding=PADDING):
    mf_inputs = dict(
        xc=xc,
        nelec=mf.nelec,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        mol=mol,
        dm=dm,
        use_interpolation_mf=use_interpolation_mf,
        mf_interp_spacing=spacing,
        mf_interp_padding=padding,
    )
    if use_interpolation_mf:
        attach_mf_interpolators(mf_inputs, spacing=spacing, padding=padding)
    return mf_inputs


def _rel_err(approx, ref, eps=REL_EPS):
    """Pointwise |approx-ref| / (|ref| + eps)."""
    approx = np.asarray(approx, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return np.abs(approx - ref) / (np.abs(ref) + eps)


def _errors(approx, ref, eps=REL_EPS):
    """Absolute and relative error summary vs reference field ``ref``."""
    abs_err = np.abs(np.asarray(approx, dtype=float) - np.asarray(ref, dtype=float))
    rel = _rel_err(approx, ref, eps=eps)
    return {
        "mae": float(np.mean(abs_err)),
        "max": float(np.max(abs_err)),
        "mre": float(np.mean(rel)),
        "max_rel": float(np.max(rel)),
    }


class _Configs:
    """Minimal configs: ``configs`` array shaped (nconf, nelec, 3)."""

    def __init__(self, arr):
        self.configs = np.asarray(arr, dtype=float)


def _unit_vectors(n, rng):
    """Approximately uniform directions on the sphere."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _place_electrons(nup, ndn, radii_up, radii_dn, dirs_up, dirs_dn, origin=None):
    """Build one walker: (nelec, 3) with spin-blocked ordering (up then dn)."""
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    pos_up = origin + dirs_up * np.asarray(radii_up)[:, None]
    pos_dn = origin + dirs_dn * np.asarray(radii_dn)[:, None]
    return np.vstack([pos_up, pos_dn])


def _shell_radii(n):
    base = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8])
    if n <= len(base):
        return base[:n].copy()
    return np.concatenate([base, np.linspace(3.0, 4.0, n - len(base))])


def _jitter_radii(radii, rng, scale=0.05, rmin=0.05):
    """Multiplicative radius jitter; keep radii above ``rmin``."""
    radii = np.asarray(radii, dtype=float) * (1.0 + scale * rng.normal(size=len(radii)))
    return np.maximum(radii, rmin)


def build_regime_walkers(mol, nup, ndn, seed=0, nwalkers=100):
    """Ensemble of walkers probing distinct geometry regimes.

    Each regime returns configs shaped ``(nwalkers, nelec, 3)``. Walkers within a
    regime share the same qualitative constraints but vary in orientation and
    have small radius jitter.
    """
    nelec = nup + ndn
    origin = np.asarray(mol.atom_coords(), dtype=float).reshape(-1, 3).mean(axis=0)
    rng = np.random.default_rng(seed)
    r_up0 = _shell_radii(nup)
    r_dn0 = _shell_radii(ndn)

    def _one_valence_spread():
        return _place_electrons(
            nup,
            ndn,
            _jitter_radii(r_up0, rng, scale=0.08),
            _jitter_radii(r_dn0, rng, scale=0.08) if ndn else r_dn0,
            _unit_vectors(nup, rng),
            _unit_vectors(ndn, rng) if ndn else np.zeros((0, 3)),
            origin,
        )

    def _one_far_field():
        return _place_electrons(
            nup,
            ndn,
            _jitter_radii(r_up0 + 4.0, rng, scale=0.05, rmin=3.5),
            _jitter_radii(r_dn0 + 4.0, rng, scale=0.05, rmin=3.5) if ndn else r_dn0,
            _unit_vectors(nup, rng),
            _unit_vectors(ndn, rng) if ndn else np.zeros((0, 3)),
            origin,
        )

    def _one_near_nucleus():
        r_up = _jitter_radii(r_up0, rng, scale=0.08)
        r_dn = _jitter_radii(r_dn0, rng, scale=0.08) if ndn else r_dn0.copy()
        r_up[0] = 0.06 + 0.06 * rng.random()
        if ndn:
            r_dn[0] = 0.08 + 0.06 * rng.random()
        return _place_electrons(
            nup,
            ndn,
            r_up,
            r_dn,
            _unit_vectors(nup, rng),
            _unit_vectors(ndn, rng) if ndn else np.zeros((0, 3)),
            origin,
        )

    def _one_core_cluster():
        r_up = np.full(nup, 0.10) + 0.06 * rng.random(nup)
        r_dn = np.full(ndn, 0.12) + 0.06 * rng.random(ndn) if ndn else r_dn0
        axis = _unit_vectors(1, rng)[0]
        dirs_up = axis + 0.08 * rng.normal(size=(nup, 3))
        dirs_up /= np.linalg.norm(dirs_up, axis=1, keepdims=True)
        if ndn:
            dirs_dn = axis + 0.08 * rng.normal(size=(ndn, 3))
            dirs_dn /= np.linalg.norm(dirs_dn, axis=1, keepdims=True)
        else:
            dirs_dn = np.zeros((0, 3))
        return _place_electrons(nup, ndn, r_up, r_dn, dirs_up, dirs_dn, origin)

    def _one_valence_close_pair():
        axis = _unit_vectors(1, rng)[0]
        r0 = 2.3 + 0.4 * rng.random()
        r_up = np.full(nup, r0)
        r_dn = np.full(ndn, r0) if ndn else r_dn0.copy()
        dirs_up = np.tile(axis, (nup, 1))
        dirs_dn = np.tile(axis, (max(ndn, 1), 1))[:ndn] if ndn else np.zeros((0, 3))
        if nup > 1:
            r_up[1] = r0 + 0.03 + 0.04 * rng.random()
        if ndn:
            r_dn[0] = r0 + 0.05 + 0.04 * rng.random()
            dirs_dn = dirs_dn + 0.02 * rng.normal(size=(ndn, 3))
            dirs_dn /= np.linalg.norm(dirs_dn, axis=1, keepdims=True)
        return _place_electrons(nup, ndn, r_up, r_dn, dirs_up, dirs_dn, origin)

    def _one_close_near_nucleus():
        axis = _unit_vectors(1, rng)[0]
        r0 = 0.12 + 0.08 * rng.random()
        r_up = np.full(nup, r0)
        r_dn = np.full(ndn, r0) if ndn else r_dn0.copy()
        dirs_up = np.tile(axis, (nup, 1))
        dirs_dn = np.tile(axis, (max(ndn, 1), 1))[:ndn] if ndn else np.zeros((0, 3))
        if nup > 1:
            r_up[1] = r0 + 0.03 + 0.03 * rng.random()
        if ndn:
            r_dn[0] = r0 + 0.04 + 0.03 * rng.random()
            dirs_dn = dirs_dn + 0.03 * rng.normal(size=(ndn, 3))
            dirs_dn /= np.linalg.norm(dirs_dn, axis=1, keepdims=True)
        return _place_electrons(nup, ndn, r_up, r_dn, dirs_up, dirs_dn, origin)

    def _one_mixed_core_valence():
        r_up = _jitter_radii(r_up0, rng, scale=0.08)
        r_dn = _jitter_radii(r_dn0, rng, scale=0.08) if ndn else r_dn0.copy()
        r_up[: max(1, nup // 2)] = 0.12 + 0.08 * rng.random(max(1, nup // 2))
        if ndn:
            ncore_dn = max(1, ndn // 2)
            r_dn[:ncore_dn] = 0.14 + 0.08 * rng.random(ncore_dn)
        return _place_electrons(
            nup,
            ndn,
            r_up,
            r_dn,
            _unit_vectors(nup, rng),
            _unit_vectors(ndn, rng) if ndn else np.zeros((0, 3)),
            origin,
        )

    builders = {
        "valence_spread": _one_valence_spread,
        "far_field": _one_far_field,
        "near_nucleus": _one_near_nucleus,
        "core_cluster": _one_core_cluster,
        "valence_close_pair": _one_valence_close_pair,
        "close_near_nucleus": _one_close_near_nucleus,
        "mixed_core_valence": _one_mixed_core_valence,
    }

    regimes = {
        name: np.stack([builder() for _ in range(nwalkers)], axis=0)
        for name, builder in builders.items()
    }
    for name, cfg in regimes.items():
        assert cfg.shape == (nwalkers, nelec, 3), (name, cfg.shape)
    return regimes


def _compare_dft_energy(mol, mf, dm, xc, configs, interp_inputs=None, spacing=SPACING, padding=PADDING):
    direct = _mf_inputs(mol, mf, dm, xc, use_interpolation_mf=False)
    if interp_inputs is None:
        interp = _mf_inputs(
            mol, mf, dm, xc, use_interpolation_mf=True, spacing=spacing, padding=padding
        )
    else:
        interp = dict(interp_inputs)
        interp["nelec"] = mf.nelec
    v_d, _, s_d = dft_energy(direct, configs)
    v_i, _, s_i = dft_energy(interp, configs)
    return {
        "vj": _errors(s_i["vj"], s_d["vj"]),
        "vxc": _errors(s_i["vxc"], s_d["vxc"]),
        "v_mf": _errors(v_i, v_d),
    }


def _points_away_from_nuclei(mol, n=200, rmin=0.5, box=3.0, seed=0):
    rng = np.random.default_rng(seed)
    atoms = np.asarray(mol.atom_coords(), dtype=float)
    center = atoms.mean(axis=0)
    pts = []
    while len(pts) < n:
        p = center + rng.uniform(-box, box, size=3)
        if np.min(np.linalg.norm(atoms - p, axis=1)) >= rmin:
            pts.append(p)
    return np.asarray(pts)


def _inside_interp_box(mol, coords, padding=PADDING):
    """Keep only points inside the Cartesian table domain (avoid extrapolation)."""
    atoms = np.asarray(mol.atom_coords(), dtype=float)
    lo = atoms.min(axis=0) - padding
    hi = atoms.max(axis=0) + padding
    mask = np.all((coords >= lo) & (coords <= hi), axis=1)
    return coords[mask]


def _stratify_dft_grid(mol, grids, r_edges=(0.2, 0.5, 1.0, 2.0), padding=PADDING):
    """Split DFT quadrature coords by distance to nearest nucleus."""
    coords = _inside_interp_box(mol, grids.coords, padding=padding)
    atoms = np.asarray(mol.atom_coords(), dtype=float)
    r = np.min(
        np.linalg.norm(coords[:, None, :] - atoms[None, :, :], axis=2), axis=1
    )
    edges = (0.0,) + tuple(r_edges) + (np.inf,)
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r >= lo) & (r < hi)
        label = f"r=[{lo},{hi})"
        if np.any(mask):
            bins[label] = coords[mask]
    return bins


# Relative MRE ceilings (fraction of |V_ref|). Same scale for He/Li/Ne.
TOL_AWAY_MRE = {"vj": 5e-3, "vxc": 2e-2}
TOL_AWAY_MAX_REL = {"vj": 2e-2, "vxc": 5e-2}

TOL_RADIUS_MRE = {
    "r=[0.0,0.2)": {"vj": 5e-2, "vxc": 1e-1},
    "r=[0.2,0.5)": {"vj": 2e-2, "vxc": 5e-2},
    "r=[0.5,1.0)": {"vj": 8e-3, "vxc": 2e-2},
    "r=[1.0,2.0)": {"vj": 5e-3, "vxc": 1e-2},
    "r=[2.0,inf)": {"vj": 5e-3, "vxc": 1e-2},
}

# Soft ceilings for walker-summed v_mf mean relative error.
REGIME_TOL_V_MF_MRE = {
    "far_field": 2e-2,
    "valence_spread": 2e-2,
    "valence_close_pair": 2e-2,
    "mixed_core_valence": 5e-2,
    "near_nucleus": 8e-2,
    "close_near_nucleus": 1e-1,
    "core_cluster": 1.5e-1,
}

NUCLEAR_REGIMES = {
    "near_nucleus",
    "close_near_nucleus",
    "core_cluster",
    "mixed_core_valence",
}


# ---------------------------------------------------------------------------
# Basic agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize("label,atom,spin,charge,nup,ndn,Z", ATOMS)
def test_interpolation_mf_agrees_away_from_nuclei(
    xc, label, atom, spin, charge, nup, ndn, Z, interp_cache
):
    mol, mf, dm, interp = interp_cache(label, atom, spin, charge, xc)
    coords = _points_away_from_nuclei(mol, n=128, rmin=0.5, box=2.5, seed=1)
    configs = _Configs(coords.reshape(-1, 1, 3))
    nelec = (1, 0)

    interp = dict(interp)
    interp["nelec"] = nelec

    vj_d = get_vj(configs, mol, dm)
    vxc_d = get_vxc(configs, mol, dm, nelec, xc)
    _, _, saved_i = dft_energy(interp, configs)

    evj = _errors(saved_i["vj"], vj_d)
    evxc = _errors(saved_i["vxc"], vxc_d)
    assert evj["mre"] < TOL_AWAY_MRE["vj"], (label, xc, evj)
    assert evxc["mre"] < TOL_AWAY_MRE["vxc"], (label, xc, evxc)
    assert evj["max_rel"] < TOL_AWAY_MAX_REL["vj"], (label, xc, evj)
    assert evxc["max_rel"] < TOL_AWAY_MAX_REL["vxc"], (label, xc, evxc)


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize("label,atom,spin,charge,nup,ndn,Z", ATOMS)
def test_interpolation_mf_dft_grid_by_radius(
    xc, label, atom, spin, charge, nup, ndn, Z, interp_cache
):
    """Relative agreement on SCF quadrature points, stratified by nuclear distance."""
    mol, mf, dm, interp = interp_cache(label, atom, spin, charge, xc)
    bins = _stratify_dft_grid(mol, mf.grids)
    nelec = (1, 0)
    interp = dict(interp)
    interp["nelec"] = nelec

    report = {}
    for label_r, coords in bins.items():
        if coords.shape[0] > 400:
            coords = coords[:: max(1, coords.shape[0] // 400)]
        configs = _Configs(coords.reshape(-1, 1, 3))
        vj_d = get_vj(configs, mol, dm)
        vxc_d = get_vxc(configs, mol, dm, nelec, xc)
        _, _, saved_i = dft_energy(interp, configs)
        evj = _errors(saved_i["vj"], vj_d)
        evxc = _errors(saved_i["vxc"], vxc_d)
        report[label_r] = {"vj": evj, "vxc": evxc, "n": coords.shape[0]}
        tol = TOL_RADIUS_MRE[label_r]
        assert evj["mre"] < tol["vj"], (label, xc, label_r, evj, tol["vj"])
        assert evxc["mre"] < tol["vxc"], (label, xc, label_r, evxc, tol["vxc"])

    if "r=[0.0,0.2)" in report and "r=[2.0,inf)" in report:
        assert report["r=[0.0,0.2)"]["vj"]["mre"] >= report["r=[2.0,inf)"]["vj"]["mre"] * 0.5


# ---------------------------------------------------------------------------
# Fixed walker regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE,PBE"])
@pytest.mark.parametrize("label,atom,spin,charge,nup,ndn,Z", ATOMS)
def test_interpolation_mf_fixed_walker_regimes(
    xc, label, atom, spin, charge, nup, ndn, Z, interp_cache
):
    mol, mf, dm, interp = interp_cache(label, atom, spin, charge, xc)
    regimes = build_regime_walkers(mol, nup, ndn, seed=0)

    errors = {}
    for regime, cfg in regimes.items():
        err = _compare_dft_energy(mol, mf, dm, xc, _Configs(cfg), interp_inputs=interp)
        errors[regime] = err
        tol = REGIME_TOL_V_MF_MRE[regime]
        assert err["v_mf"]["mre"] < tol, (
            f"{label} {xc} {regime}: v_mf mre={err['v_mf']['mre']:.4e} "
            f"tol={tol:.4e} detail={err}"
        )

    worst = max(errors, key=lambda k: errors[k]["v_mf"]["mre"])
    assert worst in NUCLEAR_REGIMES, (
        label,
        xc,
        worst,
        {k: v["v_mf"]["mre"] for k, v in errors.items()},
    )

    far = errors["far_field"]["v_mf"]["mre"]
    nuclear_mres = [errors[r]["v_mf"]["mre"] for r in NUCLEAR_REGIMES if r in errors]
    assert max(nuclear_mres) >= far - 1e-9


@pytest.mark.parametrize("xc", ["LDA,VWN"])
@pytest.mark.parametrize("label,atom,spin,charge,nup,ndn,Z", ATOMS)
def test_interpolation_mf_regime_error_report(
    xc, label, atom, spin, charge, nup, ndn, Z, interp_cache, capsys
):
    """Print absolute + relative error table (pytest -s)."""
    mol, mf, dm, interp = interp_cache(label, atom, spin, charge, xc)
    regimes = build_regime_walkers(mol, nup, ndn, seed=0)
    rows = []
    for regime, cfg in regimes.items():
        err = _compare_dft_energy(mol, mf, dm, xc, _Configs(cfg), interp_inputs=interp)
        rows.append(
            (
                regime,
                err["vj"]["mae"],
                err["vj"]["mre"],
                err["vxc"]["mae"],
                err["vxc"]["mre"],
                err["v_mf"]["mae"],
                err["v_mf"]["mre"],
            )
        )
    rows.sort(key=lambda r: r[-1])  # by v_mf relative MRE
    nwalkers = next(iter(regimes.values())).shape[0]
    print(
        f"\nMF interp errors [{label}, {xc}, spacing={SPACING}, eps={REL_EPS}, "
        f"nwalkers={nwalkers}]:"
    )
    print(
        f"{'regime':22s} {'vj_mae':>10s} {'vj_mre':>10s} "
        f"{'vxc_mae':>10s} {'vxc_mre':>10s} {'vmf_mae':>10s} {'vmf_mre':>10s}"
    )
    for row in rows:
        print(
            f"{row[0]:22s} {row[1]:10.3e} {row[2]:10.3e} "
            f"{row[3]:10.3e} {row[4]:10.3e} {row[5]:10.3e} {row[6]:10.3e}"
        )
    worst = rows[-1][0]
    assert worst in NUCLEAR_REGIMES, worst


# ---------------------------------------------------------------------------
# Infra
# ---------------------------------------------------------------------------


def test_parallel_build_matches_serial():
    mol, mf, dm = _run_uks("He 0 0 0", spin=0, charge=0, xc="LDA,VWN")
    serial = build_mf_potential_tables(
        mol, dm, "LDA,VWN", spacing=0.4, padding=2.0, chunk_size=64, nworkers=1
    )
    parallel = build_mf_potential_tables(
        mol, dm, "LDA,VWN", spacing=0.4, padding=2.0, chunk_size=64, nworkers=2
    )
    assert np.allclose(serial["vj"], parallel["vj"], atol=1e-10, rtol=0)
    assert np.allclose(serial["vxc_up"], parallel["vxc_up"], atol=1e-10, rtol=0)
    assert np.allclose(serial["vxc_dn"], parallel["vxc_dn"], atol=1e-10, rtol=0)


def test_interpolation_mf_rejects_hf():
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "HF"
    mf.kernel()
    mf_inputs = dict(
        xc="HF",
        nelec=mf.nelec,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        mol=mol,
        dm=mf.make_rdm1(),
        use_interpolation_mf=True,
    )
    with pytest.raises(ValueError, match="HF"):
        attach_mf_interpolators(mf_inputs)


def test_dft_energy_lazy_builds_interpolators():
    mol, mf, dm = _run_uks("He 0 0 0", spin=0, charge=0, xc="LDA,VWN")
    regimes = build_regime_walkers(mol, 1, 1, seed=0)
    configs = _Configs(regimes["valence_spread"])
    mf_inputs = dict(
        xc="LDA,VWN",
        nelec=mf.nelec,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        mol=mol,
        dm=dm,
        use_interpolation_mf=True,
        mf_interp_spacing=0.12,
        mf_interp_padding=3.5,
    )
    assert mf_inputs.get("mf_interp") is None
    v_mf, _, saved = dft_energy(mf_inputs, configs)
    assert mf_inputs.get("mf_interp") is not None
    assert saved["vj"].shape == v_mf.shape
