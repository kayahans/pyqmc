"""Mean-field and bosonic kinetic terms for ABVMC / ABCDMC.

For DFT mean field, the local XC potential at walker positions uses libxc
``vrho`` (``eval_xc(...)[1][0]``). That is the full KS potential for LDA.
For GGA (PBE), it is the local spin density potential only; the GGA
``vsigma`` / ∇·(vσ ∇ρ) contribution that appears in PySCF's Vxc matrix is
not included. This matches the ABVMC formulation in Eq. 21 of
doi: 10.1063/5.0155513.

AO → ρ → vrho and Hartree Vj use ``evaluate_mf_with="numba"`` (packed
``AtomicOrbitalEvaluator`` + ``mf_hartree``), ``"pyscf"``
(``numint`` / ``int1e_grids`` oracle / fallback), ``"grid"`` (dense PySCF
DFT grid + scattered interpolation; see ``mf_grid_interp``), or ``"ri"``
(density-fitted Hartree via ``mf_ri_hartree``; Vxc still AO → ρ → libxc).
"""

import numpy as np
from pyscf.dft import libxc, numint

from pyqmc.observables import mf_hartree

SUPPORTED_XC = ("LDA,VWN", "PBE,PBE", "HF")
SUPPORTED_EVALUATE_MF = ("pyscf", "numba", "grid", "ri")

# PySCF xctype and AO derivative order for each XC string.
XC_KIND = {
    "LDA,VWN": ("LDA", 0),
    "PBE,PBE": ("GGA", 1),
}


def _normalize_xc(xc):
    xc = xc.replace(" ", "")
    if xc not in SUPPORTED_XC:
        raise ValueError(f"Unsupported xc={xc!r}; expected one of {SUPPORTED_XC}")
    return xc


def _normalize_evaluate_mf(evaluate_mf_with):
    if evaluate_mf_with not in SUPPORTED_EVALUATE_MF:
        raise ValueError(
            f"evaluate_mf_with={evaluate_mf_with!r} not recognized; "
            f"must be one of {SUPPORTED_EVALUATE_MF}"
        )
    return evaluate_mf_with


def prepare_mf_evaluator(mf_inputs, evaluate_mf_with="numba"):
    """Attach MF AO / Hartree / grid / RI backend choice to ``mf_inputs``.

    Packs the basis once via ``AtomicOrbitalEvaluator`` and builds a
    ``HartreePotentialEvaluator`` for the frozen SCF density when
    ``evaluate_mf_with="numba"``. For ``"grid"``, builds a
    ``GridMFPotentialEvaluator`` (dense DFT grid + scattered interp).
    For ``"ri"``, fits RI Hartree coefficients once (``RIHartreePotentialEvaluator``)
    and keeps a numba AO evaluator for Vxc. Optional ``mf_ri_auxbasis`` selects
    the auxiliary basis (default: ``pyscf.df.make_auxbasis``).
    Idempotent if already prepared with the same backend.
    """
    evaluate_mf_with = _normalize_evaluate_mf(evaluate_mf_with)
    prev = mf_inputs.get("evaluate_mf_with")
    if prev == evaluate_mf_with and (
        (
            evaluate_mf_with == "pyscf"
            and mf_inputs.get("grid_mf_evaluator") is None
            and mf_inputs.get("vj_evaluator") is None
        )
        or (
            evaluate_mf_with == "numba"
            and mf_inputs.get("ao_evaluator") is not None
            and mf_inputs.get("vj_evaluator") is not None
            and mf_inputs.get("grid_mf_evaluator") is None
        )
        or (
            evaluate_mf_with == "grid"
            and mf_inputs.get("grid_mf_evaluator") is not None
        )
        or (
            evaluate_mf_with == "ri"
            and mf_inputs.get("ao_evaluator") is not None
            and mf_inputs.get("vj_evaluator") is not None
            and mf_inputs.get("grid_mf_evaluator") is None
        )
    ):
        return mf_inputs

    mf_inputs["evaluate_mf_with"] = evaluate_mf_with
    if evaluate_mf_with == "numba":
        from pyqmc.wf.numba.gto import AtomicOrbitalEvaluator

        mol = mf_inputs["mol"]
        dm = mf_inputs["dm"]
        dm_total = dm[0] + dm[1] if np.asarray(dm).ndim == 3 else dm
        mf_inputs["ao_evaluator"] = AtomicOrbitalEvaluator(mol)
        mf_inputs["vj_evaluator"] = mf_hartree.HartreePotentialEvaluator(
            mol, dm_total
        )
        mf_inputs.pop("grid_mf_evaluator", None)
    elif evaluate_mf_with == "ri":
        from pyqmc.observables import mf_ri_hartree
        from pyqmc.wf.numba.gto import AtomicOrbitalEvaluator

        mol = mf_inputs["mol"]
        dm = mf_inputs["dm"]
        mf_inputs["ao_evaluator"] = AtomicOrbitalEvaluator(mol)
        mf_inputs["vj_evaluator"] = mf_ri_hartree.RIHartreePotentialEvaluator(
            mol,
            dm,
            auxbasis=mf_inputs.get("mf_ri_auxbasis"),
            chunk_size=int(mf_inputs.get("mf_ri_chunk_size", 256)),
        )
        mf_inputs.pop("grid_mf_evaluator", None)
    elif evaluate_mf_with == "grid":
        from pyqmc.observables import mf_grid_interp

        mol = mf_inputs["mol"]
        dm = mf_inputs["dm"]
        xc = _normalize_xc(mf_inputs.get("xc", "LDA,VWN"))
        grid_level = int(mf_inputs.get("mf_grid_level", 7))
        method = mf_inputs.get("mf_grid_method", "nearest")
        mf_inputs["grid_mf_evaluator"] = mf_grid_interp.GridMFPotentialEvaluator(
            mol,
            dm,
            xc=xc,
            grid_level=grid_level,
            method=method,
            chunk_size=int(mf_inputs.get("mf_grid_chunk_size", 256)),
            rbf_neighbors=int(mf_inputs.get("mf_grid_rbf_neighbors", 64)),
        )
        mf_inputs.pop("ao_evaluator", None)
        mf_inputs.pop("vj_evaluator", None)
    else:
        mf_inputs.pop("ao_evaluator", None)
        mf_inputs.pop("vj_evaluator", None)
        mf_inputs.pop("grid_mf_evaluator", None)
    return mf_inputs


def _mf_backend(
    mf_inputs=None,
    evaluate_mf_with=None,
    ao_evaluator=None,
    vj_evaluator=None,
    grid_mf_evaluator=None,
):
    if evaluate_mf_with is None:
        evaluate_mf_with = (
            mf_inputs.get("evaluate_mf_with", "numba") if mf_inputs else "numba"
        )
    evaluate_mf_with = _normalize_evaluate_mf(evaluate_mf_with)
    if ao_evaluator is None and mf_inputs is not None:
        ao_evaluator = mf_inputs.get("ao_evaluator")
    if vj_evaluator is None and mf_inputs is not None:
        vj_evaluator = mf_inputs.get("vj_evaluator")
    if grid_mf_evaluator is None and mf_inputs is not None:
        grid_mf_evaluator = mf_inputs.get("grid_mf_evaluator")
    if evaluate_mf_with in ("numba", "ri"):
        mol = mf_inputs["mol"] if mf_inputs is not None else None
        if ao_evaluator is None:
            if mol is None:
                raise ValueError(
                    f"{evaluate_mf_with} MF path requires ao_evaluator or "
                    "mf_inputs['mol']; call prepare_mf_evaluator(mf_inputs) first"
                )
            from pyqmc.wf.numba.gto import AtomicOrbitalEvaluator

            ao_evaluator = AtomicOrbitalEvaluator(mol)
            if mf_inputs is not None:
                mf_inputs["ao_evaluator"] = ao_evaluator
                mf_inputs["evaluate_mf_with"] = evaluate_mf_with
        if vj_evaluator is None:
            if mol is None or mf_inputs is None or "dm" not in mf_inputs:
                raise ValueError(
                    f"{evaluate_mf_with} MF path requires vj_evaluator or "
                    "mf_inputs['mol'/'dm']; call prepare_mf_evaluator(mf_inputs) first"
                )
            dm = mf_inputs["dm"]
            if evaluate_mf_with == "ri":
                from pyqmc.observables import mf_ri_hartree

                vj_evaluator = mf_ri_hartree.RIHartreePotentialEvaluator(
                    mol,
                    dm,
                    auxbasis=mf_inputs.get("mf_ri_auxbasis"),
                    chunk_size=int(mf_inputs.get("mf_ri_chunk_size", 256)),
                )
            else:
                dm_total = dm[0] + dm[1] if np.asarray(dm).ndim == 3 else dm
                vj_evaluator = mf_hartree.HartreePotentialEvaluator(mol, dm_total)
            mf_inputs["vj_evaluator"] = vj_evaluator
            mf_inputs["evaluate_mf_with"] = evaluate_mf_with
    elif evaluate_mf_with == "grid":
        if grid_mf_evaluator is None:
            if mf_inputs is None:
                raise ValueError(
                    "grid MF path requires grid_mf_evaluator or mf_inputs; "
                    "call prepare_mf_evaluator(mf_inputs, evaluate_mf_with='grid')"
                )
            prepare_mf_evaluator(mf_inputs, evaluate_mf_with="grid")
            grid_mf_evaluator = mf_inputs["grid_mf_evaluator"]
    return evaluate_mf_with, ao_evaluator, vj_evaluator, grid_mf_evaluator


def eval_ao(mol, coords, deriv=0, evaluate_mf_with="numba", ao_evaluator=None):
    """Evaluate AOs at ``coords`` (N, 3).

    Returns the same layout as ``pyscf.dft.numint.eval_ao``:
    ``(N, nao)`` for ``deriv=0``, ``(4, N, nao)`` for ``deriv=1``.

    ``"ri"`` and ``"grid"`` use the numba AO path (grid Vxc does not call this).
    """
    evaluate_mf_with = _normalize_evaluate_mf(evaluate_mf_with)
    if evaluate_mf_with == "pyscf":
        return numint.eval_ao(mol, coords, deriv=deriv)

    if ao_evaluator is None:
        from pyqmc.wf.numba.gto import AtomicOrbitalEvaluator

        ao_evaluator = AtomicOrbitalEvaluator(mol)
    if deriv == 0:
        return ao_evaluator.eval_gto("GTOval_sph", coords)
    if deriv == 1:
        return ao_evaluator.eval_gto("GTOval_sph_deriv1", coords)
    raise ValueError(f"deriv={deriv} not supported; expected 0 or 1")


def eval_rho_from_ao(ao, dm, xctype="LDA"):
    """Electron density from AO values (hermitian DM, matches ``numint.eval_rho``).

    LDA / HF: ``ao`` is ``(N, nao)``, returns ``(N,)``.
    GGA: ``ao`` is ``(4, N, nao)``, returns ``(4, N)`` with ``ρ`` and ``∇ρ``.
    """
    xctype = xctype.upper()
    if xctype in ("LDA", "HF"):
        c0 = ao @ dm
        return np.einsum("pi,pi->p", ao, c0)
    if xctype == "GGA":
        rho = np.empty((4, ao.shape[1]), dtype=ao.dtype)
        c0 = ao[0] @ dm
        rho[0] = np.einsum("pi,pi->p", ao[0], c0)
        for i in range(1, 4):
            # *2 for hermitian DM: ∇ρ = 2 Re(χ† D ∇χ)
            rho[i] = 2.0 * np.einsum("pi,pi->p", ao[i], c0)
        return rho
    raise ValueError(f"Unsupported xctype={xctype!r}; expected LDA, HF, or GGA")


def eval_vrho(
    mol,
    dm,
    xc,
    coords,
    spin=1,
    evaluate_mf_with=None,
    ao_evaluator=None,
    mf_inputs=None,
):
    """Local spin-resolved vrho from libxc at ``coords``."""
    xc = _normalize_xc(xc)
    if xc == "HF":
        raise ValueError("HF has no libxc vrho")

    evaluate_mf_with, ao_evaluator, _, grid_mf_evaluator = _mf_backend(
        mf_inputs, evaluate_mf_with, ao_evaluator
    )
    if evaluate_mf_with == "grid":
        return grid_mf_evaluator.eval_vxc_points(coords)

    xctype, deriv = XC_KIND[xc]

    if evaluate_mf_with == "pyscf":
        ao = numint.eval_ao(mol, coords, deriv=deriv)
        rho_up = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
        rho_dn = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
    else:
        # numba and ri share the packed AO → ρ path for Vxc
        ao = eval_ao(
            mol,
            coords,
            deriv=deriv,
            evaluate_mf_with="numba",
            ao_evaluator=ao_evaluator,
        )
        rho_up = eval_rho_from_ao(ao, dm[0], xctype=xctype)
        rho_dn = eval_rho_from_ao(ao, dm[1], xctype=xctype)

    vrho = np.asarray(libxc.eval_xc(xc, (rho_up, rho_dn), spin=spin)[1][0])
    if vrho.ndim == 1:
        vrho = np.stack([vrho, vrho], axis=1)
    return vrho


def get_vxc(configs, mol, dm, nelec, xc, evaluate_mf_with=None, ao_evaluator=None, mf_inputs=None):
    """Sum libxc vrho over electrons for each walker configuration."""
    nconf, nelec_cfg, _ = configs.configs.shape
    nup = nelec[0]
    if nelec_cfg != sum(nelec):
        raise ValueError("configs electron count inconsistent with mf_inputs['nelec']")

    evaluate_mf_with, ao_evaluator, _, grid_mf_evaluator = _mf_backend(
        mf_inputs, evaluate_mf_with, ao_evaluator
    )
    if evaluate_mf_with == "grid":
        return grid_mf_evaluator.eval_vxc_sum(configs, nelec)

    coords = configs.configs.reshape(-1, 3)
    vrho = eval_vrho(
        mol,
        dm,
        xc,
        coords,
        spin=1,
        evaluate_mf_with=evaluate_mf_with,
        ao_evaluator=ao_evaluator,
        mf_inputs=mf_inputs,
    )
    vrho = vrho.reshape(nconf, nelec_cfg, 2)

    spin_idx = np.array([int(e >= nup) for e in range(nelec_cfg)])
    return np.sum([vrho[:, i, spin_idx[i]] for i in range(nelec_cfg)], axis=0)


def get_vj(
    configs,
    mol,
    dm,
    evaluate_mf_with=None,
    vj_evaluator=None,
    mf_inputs=None,
    chunk_size=256,
):
    """Hartree potential of total SCF density at electron positions, summed per walker."""
    nconf, nelec, _ = configs.configs.shape
    dm_total = dm[0] + dm[1]
    r = configs.configs.reshape(-1, 3)

    evaluate_mf_with, _, vj_evaluator, grid_mf_evaluator = _mf_backend(
        mf_inputs, evaluate_mf_with, vj_evaluator=vj_evaluator
    )
    if evaluate_mf_with == "grid":
        vj_all = grid_mf_evaluator.eval_vj_points(r)
    elif evaluate_mf_with == "pyscf":
        vj_all = mf_hartree.eval_vj_pyscf(mol, dm_total, r, chunk_size=chunk_size)
    else:
        # numba (analytic MD) or ri (fitted aux Coulomb); both are callables
        if vj_evaluator is None:
            if evaluate_mf_with == "ri":
                from pyqmc.observables import mf_ri_hartree

                auxbasis = (
                    mf_inputs.get("mf_ri_auxbasis") if mf_inputs is not None else None
                )
                vj_evaluator = mf_ri_hartree.RIHartreePotentialEvaluator(
                    mol, dm_total, auxbasis=auxbasis, chunk_size=chunk_size
                )
            else:
                vj_evaluator = mf_hartree.HartreePotentialEvaluator(
                    mol, dm_total, chunk_size=chunk_size
                )
        vj_all = vj_evaluator(r)
    return vj_all.reshape(nconf, nelec).sum(axis=1)


def dft_energy(mf_inputs, configs):
    """
    Returns the KS related terms in Eq. 21 in doi: 10.1063/5.0155513.

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
    evaluate_mf_with, ao_evaluator, vj_evaluator, _ = _mf_backend(mf_inputs)

    if xc != "HF":
        vj = get_vj(
            configs,
            mol,
            dm,
            evaluate_mf_with=evaluate_mf_with,
            vj_evaluator=vj_evaluator,
            mf_inputs=mf_inputs,
        )
        vxc = get_vxc(
            configs,
            mol,
            dm,
            nup_dn,
            xc,
            evaluate_mf_with=evaluate_mf_with,
            ao_evaluator=ao_evaluator,
            mf_inputs=mf_inputs,
        )
        ecorr = np.sum(mo_energy * mo_occ)
        v_mf = vj + vxc
        saved_results = {"vj": vj, "vxc": vxc}
    else:
        v_mf = np.zeros(nconf)
        ecorr = np.sum(mo_energy * mo_occ)
        V_eff_ao = mf_inputs["veff"]
        for e in range(nelec):
            s = int(e >= nup_dn[0])
            ao_value = eval_ao(
                mol,
                configs.configs[:, e, :],
                deriv=0,
                evaluate_mf_with=(
                    "pyscf"
                    if evaluate_mf_with == "grid"
                    else ("numba" if evaluate_mf_with == "ri" else evaluate_mf_with)
                ),
                ao_evaluator=ao_evaluator,
            )
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
    from pyqmc.wf import bosonslater
    from pyqmc.wf import jastrowspin

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
