"""Analytic Hartree potential on grids: V_H(r) = Σ_{μν} D_{μν} (μν|r).

Implements the McMurchie–Davidson electrostatic potential of a GTO density,
matching ``mol.intor("int1e_grids")`` contracted with the density matrix.
Cartesian MD integrals are transformed to PySCF's spherical AO order via
``mol.cart2sph_coeff()``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from pyscf.gto.mole import gto_norm
from scipy.special import gamma, gammainc


def _cart_angles(l: int):
    """Cartesian (lx, ly, lz) tuples in PySCF order."""
    out = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            out.append((lx, ly, l - lx - ly))
    return out


def _normalize_contraction(l, exps, ctr):
    """Primitive coeffs matching libcint / ``eval_gto`` radial scaling.

    Start from PySCF ``gto_norm(l, exps) * ctr``. Libcint then multiplies s/p
    by the real spherical-harmonic factor ``sqrt((2l+1)/(4π))``; for d/f/g
    and higher that factor is already absorbed, so use bare ``gto_norm``.
    """
    exps = np.asarray(exps, dtype=np.float64)
    ctr = np.asarray(ctr, dtype=np.float64)
    c = ctr * np.atleast_1d(np.asarray(gto_norm(l, exps), dtype=np.float64))
    if l < 2:
        c = c * np.sqrt((2 * l + 1) / (4.0 * np.pi))
    return c


def boys_array(nmax: int, T: float | np.ndarray) -> np.ndarray:
    """Boys F_n(T) for n = 0 … nmax.

    Uses F_n(T) = (1/2) T^{-(n+1/2)} γ(n+1/2, T).
    ``T`` may be scalar or array; returns shape ``(nmax+1,)`` or ``(nmax+1, nT)``.
    """
    T = np.asarray(T, dtype=np.float64)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)
    out = np.empty((nmax + 1, T.size), dtype=np.float64)
    small = T < 1e-12
    large = ~small
    for n in range(nmax + 1):
        out[n, small] = 1.0 / (2 * n + 1)
        if np.any(large):
            # gammainc is regularized lower incomplete γ(s,x)/Γ(s)
            s = n + 0.5
            out[n, large] = 0.5 * np.power(T[large], -s) * gamma(s) * gammainc(s, T[large])
    if scalar:
        return out[:, 0]
    return out


@njit(cache=True, fastmath=True)
def _make_E(imax, jmax, Ax, Bx, alpha, beta, E):
    """Fill E[t, i, j] McMurchie–Davidson coefficients (1D)."""
    p = alpha + beta
    mu = alpha * beta / p
    Px = (alpha * Ax + beta * Bx) / p
    XPA = Px - Ax
    XPB = Px - Bx
    XAB = Ax - Bx
    nmax = imax + jmax
    for t in range(nmax + 1):
        for i in range(imax + 1):
            for j in range(jmax + 1):
                E[t, i, j] = 0.0
    E[0, 0, 0] = math.exp(-mu * XAB * XAB)
    for i in range(imax + 1):
        for j in range(jmax + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                for t in range(i + j + 1):
                    val = XPA * E[t, i - 1, j]
                    if t > 0:
                        val += E[t - 1, i - 1, j] / (2.0 * p)
                    if t + 1 <= nmax:
                        val += (t + 1) * E[t + 1, i - 1, j]
                    E[t, i, j] = val
            else:
                for t in range(i + j + 1):
                    val = XPB * E[t, i, j - 1]
                    if t > 0:
                        val += E[t - 1, i, j - 1] / (2.0 * p)
                    if t + 1 <= nmax:
                        val += (t + 1) * E[t + 1, i, j - 1]
                    E[t, i, j] = val
    return p, Px


def _R0_batch(tmax, umax, vmax, pexp, P, coords):
    """Hermite Coulomb R_{tuv}^{(0)} at many points. Returns (npts, tmax+1, umax+1, vmax+1)."""
    npts = coords.shape[0]
    nmax = tmax + umax + vmax
    RPC = P[None, :] - coords  # (npts, 3)
    T = pexp * np.einsum("pi,pi->p", RPC, RPC)
    Fn = boys_array(nmax, T)  # (nmax+1, npts)
    R = np.zeros((nmax + 1, npts, tmax + 1, umax + 1, vmax + 1), dtype=np.float64)
    for n in range(nmax + 1):
        R[n, :, 0, 0, 0] = ((-2.0 * pexp) ** n) * (2.0 * np.pi / pexp) * Fn[n]
    for t in range(tmax):
        for n in range(nmax - t):
            val = RPC[:, 0] * R[n + 1, :, t, 0, 0]
            if t > 0:
                val = val + t * R[n + 1, :, t - 1, 0, 0]
            R[n, :, t + 1, 0, 0] = val
    for u in range(umax):
        for t in range(tmax + 1):
            for n in range(nmax - t - u):
                val = RPC[:, 1] * R[n + 1, :, t, u, 0]
                if u > 0:
                    val = val + u * R[n + 1, :, t, u - 1, 0]
                R[n, :, t, u + 1, 0] = val
    for v in range(vmax):
        for t in range(tmax + 1):
            for u in range(umax + 1):
                for n in range(nmax - t - u - v):
                    val = RPC[:, 2] * R[n + 1, :, t, u, v]
                    if v > 0:
                        val = val + v * R[n + 1, :, t, u, v - 1]
                    R[n, :, t, u, v + 1] = val
    return R[0]


def pack_basis_hartree(mol):
    """Pack Cartesian AO primitives and spherical↔Cartesian transform."""
    aos = []
    max_l = 0
    for ish in range(mol.nbas):
        l = int(mol.bas_angular(ish))
        max_l = max(max_l, l)
        A = np.asarray(mol.atom_coord(mol.bas_atom(ish)), dtype=np.float64)
        exps = np.asarray(mol.bas_exp(ish), dtype=np.float64)
        ctr = np.asarray(mol.bas_ctr_coeff(ish), dtype=np.float64)
        for ic in range(ctr.shape[1]):
            c = _normalize_contraction(l, exps, ctr[:, ic])
            for ang in _cart_angles(l):
                aos.append(
                    {
                        "A": A,
                        "ang": ang,
                        "exps": exps,
                        "c": np.asarray(c, dtype=np.float64),
                    }
                )
    ncart = len(aos)
    if mol.cart:
        c2s = np.eye(ncart)
    else:
        c2s = np.asarray(mol.cart2sph_coeff(), dtype=np.float64)
        if c2s.shape[0] != ncart:
            raise ValueError(
                f"cart2sph size mismatch: packed {ncart} cartesian AOs, "
                f"coeff shape {c2s.shape}"
            )
    return {
        "aos": aos,
        "c2s": c2s,
        "ncart": ncart,
        "nao_sph": int(c2s.shape[1]),
        "max_l": max_l,
    }


def _dm_to_cart(dm, pack):
    c2s = pack["c2s"]
    dm = np.asarray(dm, dtype=np.float64)
    if dm.shape == (c2s.shape[0], c2s.shape[0]):
        return dm
    return c2s @ dm @ c2s.T


def density_to_hermite(dm, pack, dm_cutoff=1e-16, hermite_cutoff=1e-16):
    """Contract Cartesian DM into Hermite Gaussian sources for V_H."""
    dm_cart = _dm_to_cart(dm, pack)
    aos = pack["aos"]
    ncart = pack["ncart"]
    max_l = pack["max_l"]
    buckets = {}

    for i in range(ncart):
        ai = aos[i]
        Ai, angi, exi, ci = ai["A"], ai["ang"], ai["exps"], ai["c"]
        for j in range(ncart):
            dij = dm_cart[i, j]
            if abs(dij) < dm_cutoff:
                continue
            aj = aos[j]
            Aj, angj, exj, cj = aj["A"], aj["ang"], aj["exps"], aj["c"]
            for p, ap in enumerate(exi):
                for q, bq in enumerate(exj):
                    w = dij * ci[p] * cj[q]
                    if abs(w) < dm_cutoff:
                        continue
                    la0, la1, la2 = angi
                    lb0, lb1, lb2 = angj
                    Ex = np.zeros((la0 + lb0 + 1, max_l + 1, max_l + 1))
                    Ey = np.zeros((la1 + lb1 + 1, max_l + 1, max_l + 1))
                    Ez = np.zeros((la2 + lb2 + 1, max_l + 1, max_l + 1))
                    psum, Px = _make_E(la0, lb0, Ai[0], Aj[0], ap, bq, Ex)
                    _, Py = _make_E(la1, lb1, Ai[1], Aj[1], ap, bq, Ey)
                    _, Pz = _make_E(la2, lb2, Ai[2], Aj[2], ap, bq, Ez)
                    tmax, umax, vmax = la0 + lb0, la1 + lb1, la2 + lb2
                    key = (
                        round(Px, 12),
                        round(Py, 12),
                        round(Pz, 12),
                        round(psum, 12),
                        tmax,
                        umax,
                        vmax,
                    )
                    if key not in buckets:
                        buckets[key] = {
                            "P": np.array([Px, Py, Pz], dtype=np.float64),
                            "pexp": float(psum),
                            "H": np.zeros((tmax + 1, umax + 1, vmax + 1)),
                            "tmax": tmax,
                            "umax": umax,
                            "vmax": vmax,
                        }
                    H = buckets[key]["H"]
                    for t in range(tmax + 1):
                        for u in range(umax + 1):
                            for v in range(vmax + 1):
                                H[t, u, v] += (
                                    w
                                    * Ex[t, la0, lb0]
                                    * Ey[u, la1, lb1]
                                    * Ez[v, la2, lb2]
                                )

    sources = []
    for b in buckets.values():
        if np.max(np.abs(b["H"])) < hermite_cutoff:
            continue
        sources.append(b)
    return sources


def eval_vh_from_hermite(coords, sources, chunk_size=256):
    """Evaluate V_H at ``coords`` (N, 3) from Hermite sources."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(coords.shape[0], dtype=np.float64)
    if not sources:
        return out
    for i0 in range(0, coords.shape[0], chunk_size):
        chunk = coords[i0 : i0 + chunk_size]
        acc = np.zeros(chunk.shape[0], dtype=np.float64)
        for src in sources:
            R = _R0_batch(
                src["tmax"],
                src["umax"],
                src["vmax"],
                src["pexp"],
                src["P"],
                chunk,
            )
            acc += np.einsum("tuv,ptuv->p", src["H"], R)
        out[i0 : i0 + chunk_size] = acc
    return out


def eval_vj_points(mol, dm, coords, pack=None, sources=None, chunk_size=256):
    """V_H(r) = Σ_{μν} D_{μν} (μν|r) at ``coords``.

    ``dm`` is the spherical (or cartesian, if ``mol.cart``) AO density matrix.
    """
    if pack is None:
        pack = pack_basis_hartree(mol)
    if sources is None:
        sources = density_to_hermite(dm, pack)
    return eval_vh_from_hermite(coords, sources, chunk_size=chunk_size)


def eval_vj_pyscf(mol, dm, coords, chunk_size=256):
    """Oracle: chunked ``int1e_grids`` contraction."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    dm = np.asarray(dm, dtype=np.float64)
    out = np.empty(coords.shape[0], dtype=np.float64)
    for i0 in range(0, coords.shape[0], chunk_size):
        chunk = coords[i0 : i0 + chunk_size]
        ints = mol.intor("int1e_grids", grids=chunk)
        out[i0 : i0 + chunk_size] = np.einsum("pij,ij->p", ints, dm)
    return out


class HartreePotentialEvaluator:
    """Packed analytic V_H evaluator for a fixed SCF density matrix."""

    def __init__(self, mol, dm, chunk_size=256):
        self.mol = mol
        self.pack = pack_basis_hartree(mol)
        self.dm = np.asarray(dm, dtype=np.float64)
        self.sources = density_to_hermite(self.dm, self.pack)
        self.chunk_size = chunk_size

    def __call__(self, coords):
        return eval_vh_from_hermite(
            coords, self.sources, chunk_size=self.chunk_size
        )
