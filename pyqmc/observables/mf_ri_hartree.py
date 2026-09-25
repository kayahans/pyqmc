"""Resolution-of-identity (density-fitted) Hartree potential at arbitrary points.

Fit the SCF density once in an auxiliary basis,

    ρ(r) ≈ Σ_P c_P ξ_P(r),

then evaluate

    V_H(r) = Σ_P c_P ∫ ξ_P(r') / |r − r'| dr'

at electron positions via two-center Coulomb integrals between the aux
molecule and a ``fakemol_for_charges`` probe (same pattern as
``pyscf.tools.cubegen.mep``, but with the fitted coeff vector instead of the
AO density matrix).

Vxc is *not* density-fitted here; callers should keep using AO → ρ → libxc
(``numba`` / ``pyscf``) for the exchange–correlation piece.

Accuracy vs exact ``int1e_grids`` (empirical, ccECP + ccecp-cc-pvdz)
--------------------------------------------------------------------
RI never matches pyscf to machine precision; the residual is the DF fit error.

* **Largest absolute errors** are typically **near nuclei** (r ≲ 0.3 Bohr),
  where |V_H| is also largest. Relative errors there are often still ~1e-5–1e-4.
* **Far field** (r ≳ 2 Bohr) is usually quieter in absolute Ha.
* **Open d-shell / TM** (Fe, Sc) show the biggest residuals with the default
  aux (~1e-3–1e-2 Ha near the nucleus with ``make_auxbasis``).
* **Aux basis choice matters and is system-dependent**:
  ``make_auxbasis`` is a good default for light atoms; ``autoaux`` often
  tightens light atoms and Fe; ``def2-universal-jkfit`` can help bare Sc/Fe
  but can *worsen* He/Ne/ScH. Always check vs pyscf for the target molecule.
* For ABVMC, the relevant figure is often **summed |ΔV_H| over electrons per
  walker** (≈1e-5 He, ≈1e-4 Ne, ≈1e-3 Fe with default aux on typical samples).
"""

from __future__ import annotations

import numpy as np
from pyscf import df, gto
from pyscf.gto.mole import fakemol_for_charges

# String aliases accepted by ``resolve_auxbasis`` / config ``mf_ri_auxbasis``.
AUXBASIS_ALIASES = ("make_auxbasis", "autoaux", "autoabs", "aug_etb")


def _as_total_dm(dm):
    dm = np.asarray(dm, dtype=np.float64)
    if dm.ndim == 3:
        return dm[0] + dm[1]
    return dm


def resolve_auxbasis(mol, auxbasis=None):
    """Resolve ``auxbasis`` for DF fitting.

    ``None`` / ``\"make_auxbasis\"`` → ``pyscf.df.make_auxbasis(mol)``.
    ``\"autoaux\"`` / ``\"autoabs\"`` / ``\"aug_etb\"`` → matching PySCF generators.
    Any other value is passed through to ``df.make_auxmol`` (named basis string
    or basis dict).
    """
    if auxbasis is None or auxbasis == "make_auxbasis":
        return df.addons.make_auxbasis(mol)
    if isinstance(auxbasis, str):
        key = auxbasis.strip().lower().replace("-", "_")
        if key == "autoaux":
            return df.autoaux(mol)
        if key == "autoabs":
            return df.autoabs(mol)
        if key in ("aug_etb", "augetb"):
            return df.aug_etb(mol)
    return auxbasis


def _invert_j2c(j2c, thresh=1e-10):
    """Pseudo-inverse of the aux metric (P|Q) via eigendecomposition."""
    w, v = np.linalg.eigh(j2c)
    winv = np.where(w > thresh, 1.0 / w, 0.0)
    return (v * winv) @ v.T


def fit_ri_coefficients(mol, dm, auxbasis=None, j2c_thresh=1e-10):
    """Return ``(auxmol, c_aux)`` for ρ ≈ Σ_P c_P ξ_P.

    See ``resolve_auxbasis`` for accepted ``auxbasis`` values.
    """
    dm_total = _as_total_dm(dm)
    auxbasis = resolve_auxbasis(mol, auxbasis)
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    eri3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    j2c = auxmol.intor("int2c2e", hermi=1)
    rho_aux = np.einsum("ijP,ij->P", eri3c, dm_total, optimize=True)
    c_aux = _invert_j2c(j2c, thresh=j2c_thresh) @ rho_aux
    return auxmol, np.asarray(c_aux, dtype=np.float64)


def eval_vj_ri(auxmol, c_aux, coords, chunk_size=256):
    """Hartree potential from fitted aux coefficients at ``coords`` (N, 3)."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    c_aux = np.asarray(c_aux, dtype=np.float64).ravel()
    out = np.empty(coords.shape[0], dtype=np.float64)
    for i0 in range(0, coords.shape[0], chunk_size):
        chunk = coords[i0 : i0 + chunk_size]
        fakemol = fakemol_for_charges(chunk)
        # (naux, nchunk): Coulomb potential of each aux function at probes
        phi = gto.mole.intor_cross("int2c2e", auxmol, fakemol)
        out[i0 : i0 + chunk.shape[0]] = phi.T @ c_aux
    return out


class RIHartreePotentialEvaluator:
    """Frozen-density RI Hartree potential: fit once, evaluate many times."""

    def __init__(
        self,
        mol,
        dm,
        auxbasis=None,
        chunk_size=256,
        j2c_thresh=1e-10,
    ):
        self.mol = mol
        self.dm_total = _as_total_dm(dm)
        self.chunk_size = int(chunk_size)
        self.auxbasis = auxbasis
        self.auxmol, self.c_aux = fit_ri_coefficients(
            mol, self.dm_total, auxbasis=auxbasis, j2c_thresh=j2c_thresh
        )
        self.naux = self.auxmol.nao_nr()

    def __call__(self, coords):
        return eval_vj_ri(
            self.auxmol, self.c_aux, coords, chunk_size=self.chunk_size
        )
