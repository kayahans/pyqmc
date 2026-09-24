"""Tabulated MF potentials on a dense PySCF DFT grid + scattered interpolation.

Build once from the frozen SCF density:

1. ``pyscf.dft.gen_grid.Grids`` at a high ``level`` (dense atom-centered quadrature).
2. Evaluate Hartree ``V_H`` and spin-resolved libxc ``vrho`` on ``grids.coords``.
3. Interpolate to arbitrary electron positions with a scattered-data method.

This is an experimental speed/accuracy path for ABVMC. The DFT grid is a
quadrature mesh, not a lookup lattice; expect accuracy to depend strongly on
``grid_level`` and ``method``. Use on-the-fly ``pyscf`` / ``numba`` as the oracle.
"""

from __future__ import annotations

import numpy as np
from pyscf.dft import gen_grid, libxc, numint
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)
from scipy.spatial import cKDTree

from pyqmc.observables import mf_hartree

SUPPORTED_SCATTERED_METHODS = ("nearest", "linear", "rbf")


def build_dense_grids(mol, level=7):
    """PySCF atom-centered DFT grid (dense by default: ``level=7``)."""
    grids = gen_grid.Grids(mol)
    grids.level = int(level)
    grids.build()
    return grids


def _as_total_dm(dm):
    dm = np.asarray(dm)
    if dm.ndim == 3:
        return dm[0] + dm[1], dm
    return dm, None


def eval_vj_on_coords(mol, dm_total, coords, chunk_size=256):
    """Hartree potential on an arbitrary point set (PySCF ``int1e_grids``)."""
    return mf_hartree.eval_vj_pyscf(mol, dm_total, coords, chunk_size=chunk_size)


def eval_vxc_spin_on_coords(mol, dm, xc, coords, chunk_size=2048):
    """Spin-resolved local ``vrho`` (N, 2) on ``coords`` via numint + libxc."""
    xc = xc.replace(" ", "")
    dm = np.asarray(dm)
    if dm.ndim != 3:
        raise ValueError("eval_vxc_spin_on_coords expects UKS dm with shape (2, nao, nao)")
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    out = np.empty((coords.shape[0], 2), dtype=np.float64)
    # LDA vs GGA: AO derivative order
    if xc.upper().startswith("LDA") or xc in ("LDA,VWN", "LDA_VWN"):
        xctype, deriv = "LDA", 0
    else:
        xctype, deriv = "GGA", 1
    for i0 in range(0, coords.shape[0], chunk_size):
        chunk = coords[i0 : i0 + chunk_size]
        ao = numint.eval_ao(mol, chunk, deriv=deriv)
        rho_up = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
        rho_dn = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
        vrho = np.asarray(libxc.eval_xc(xc, (rho_up, rho_dn), spin=1)[1][0])
        if vrho.ndim == 1:
            vrho = np.stack([vrho, vrho], axis=1)
        out[i0 : i0 + chunk_size] = vrho
    return out


def _dedupe_grid_points(coords, *value_arrays, tol=1e-12):
    """Drop near-duplicate grid points (can appear at atom-sphere overlaps)."""
    coords = np.asarray(coords, dtype=np.float64)
    # Quantize for uniqueness
    key = np.round(coords / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    out_vals = [np.asarray(v)[idx] for v in value_arrays]
    return coords[idx], out_vals


class ScatteredScalarInterpolator:
    """Interpolate a scalar (or multi-column) field from scattered 3D samples."""

    def __init__(self, points, values, method="nearest", rbf_neighbors=64):
        method = method.lower()
        if method not in SUPPORTED_SCATTERED_METHODS:
            raise ValueError(
                f"method={method!r} not in {SUPPORTED_SCATTERED_METHODS}"
            )
        self.method = method
        self.points = np.asarray(points, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if self.points.shape[0] != self.values.shape[0]:
            raise ValueError("points/values length mismatch")
        if self.points.shape[0] < 4:
            raise ValueError("need at least 4 unique grid points for interpolation")

        self._nearest = NearestNDInterpolator(self.points, self.values)
        self._linear = None
        self._rbf = None
        if method == "linear":
            # fill_value nan → repaired with nearest outside the hull
            self._linear = LinearNDInterpolator(
                self.points, self.values, fill_value=np.nan
            )
        elif method == "rbf":
            n = self.points.shape[0]
            neighbors = min(int(rbf_neighbors), n - 1)
            self._rbf = RBFInterpolator(
                self.points,
                self.values,
                kernel="thin_plate_spline",
                neighbors=neighbors,
            )

    def __call__(self, coords):
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if self.method == "nearest":
            out = self._nearest(coords)
        elif self.method == "linear":
            out = self._linear(coords)
            bad = np.isnan(out).any(axis=1)
            if np.any(bad):
                out = np.array(out, copy=True)
                out[bad] = self._nearest(coords[bad])
        else:
            out = self._rbf(coords)
        if out.ndim == 1:
            return out
        if out.shape[1] == 1:
            return out[:, 0]
        return out


class GridMFPotentialEvaluator:
    """Frozen-SCF ``V_H`` / ``vrho`` tables on a dense DFT grid + scattered interp."""

    def __init__(
        self,
        mol,
        dm,
        xc="LDA,VWN",
        grid_level=7,
        method="nearest",
        chunk_size=256,
        rbf_neighbors=64,
    ):
        self.mol = mol
        self.xc = xc.replace(" ", "")
        self.grid_level = int(grid_level)
        self.method = method.lower()
        self.chunk_size = chunk_size

        dm_total, dm_spin = _as_total_dm(dm)
        if dm_spin is None:
            raise ValueError("GridMFPotentialEvaluator expects UKS dm (2, nao, nao)")
        self.dm = np.asarray(dm_spin, dtype=np.float64)

        grids = build_dense_grids(mol, level=self.grid_level)
        gcoords = np.asarray(grids.coords, dtype=np.float64)
        self.ngrids_raw = gcoords.shape[0]

        vj = eval_vj_on_coords(mol, dm_total, gcoords, chunk_size=chunk_size)
        vxc = eval_vxc_spin_on_coords(
            mol, self.dm, self.xc, gcoords, chunk_size=max(chunk_size, 512)
        )
        gcoords, (vj, vxc) = _dedupe_grid_points(gcoords, vj, vxc)
        self.grid_coords = gcoords
        self.ngrids = gcoords.shape[0]
        self.vj_grid = vj
        self.vxc_grid = vxc

        self.vj_interp = ScatteredScalarInterpolator(
            gcoords, vj, method=self.method, rbf_neighbors=rbf_neighbors
        )
        self.vxc_interp = ScatteredScalarInterpolator(
            gcoords, vxc, method=self.method, rbf_neighbors=rbf_neighbors
        )
        # KDTree kept for diagnostics / optional NN distance checks
        self._tree = cKDTree(gcoords)

    def eval_vj_points(self, coords):
        return np.asarray(self.vj_interp(coords), dtype=np.float64).reshape(-1)

    def eval_vxc_points(self, coords):
        """Return ``(N, 2)`` spin-resolved interpolated vrho."""
        out = np.asarray(self.vxc_interp(coords), dtype=np.float64)
        if out.ndim == 1:
            out = np.stack([out, out], axis=1)
        return out

    def eval_vj_sum(self, configs):
        nconf, nelec, _ = configs.configs.shape
        v = self.eval_vj_points(configs.configs.reshape(-1, 3))
        return v.reshape(nconf, nelec).sum(axis=1)

    def eval_vxc_sum(self, configs, nelec):
        nconf, nelec_cfg, _ = configs.configs.shape
        nup = nelec[0]
        if nelec_cfg != sum(nelec):
            raise ValueError("configs electron count inconsistent with nelec")
        vxc = self.eval_vxc_points(configs.configs.reshape(-1, 3)).reshape(
            nconf, nelec_cfg, 2
        )
        spin_idx = np.array([int(e >= nup) for e in range(nelec_cfg)])
        return np.sum([vxc[:, i, spin_idx[i]] for i in range(nelec_cfg)], axis=0)
