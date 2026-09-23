#!/usr/bin/env python3
"""
Plot Jastrow e–e and e–i contributions vs distance for each optimization iteration.

Requires an HDF5 file produced by line minimization that stores the serialized parameter
vector `x` each iteration (e.g. pyqmc `bosonlinemin.line_minimization`). The final
wave function parameters are also in the `wf/` group.

Usage
-----
  python plot_jastrow_iterations.py --recipe-json opt_analyze_parameters.json

  python plot_jastrow_iterations.py --hdf5 opt.hdf5 --dft scf.chk \\
      --ci ci.chk --nconfig 1000

Dependencies: pyqmc, pyscf, matplotlib, numpy, h5py
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Tuple

import h5py
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import pyqmc.gpu as gpu
import pyqmc.jastrowspin as jastrowspin
import pyqmc.recipes as recipes
from pyqmc.coord import OpenConfigs
from pyqmc.multiplywf import MultiplyWF


def get_ee_configs(
    config_shape,
    r: float = 10.0,
    rmax: float = 100.0,
    spin: int = 0,
    config_type: str = "ud",
):
    """Synthetic configs: two chosen electrons move along z; others sent to rmax."""
    nconfig = config_shape[0]
    nelec = config_shape[1]
    nelec_up = (nelec + spin) // 2
    nelec_down = (nelec - spin) // 2

    if config_type == "uu" and nelec_up >= 2:
        e_ind = [0, 1]
    elif config_type == "ud" and nelec_up >= 1 and nelec_down >= 1:
        e_ind = [0, nelec_up]
    elif config_type == "du" and nelec_up >= 1 and nelec_down >= 1:
        e_ind = [nelec_up, 0]
    elif config_type == "dd" and nelec_down >= 2:
        e_ind = [nelec_up, nelec_up + 1]
    else:
        return None, None

    epos = np.zeros((nconfig, nelec, 3)) - rmax
    z_grid = np.arange(-r, r, 2 * r / nconfig)
    first_ind = True
    for i in range(nelec):
        if i in e_ind:
            if first_ind:
                epos[:, i, 2] = 0
                first_ind = False
            else:
                epos[:, i, 2] = z_grid
        else:
            epos[:, i, 2] = rmax * (i + 1)

    return OpenConfigs(epos), z_grid


def get_ei_configs(
    config_shape,
    r: float = 10.0,
    rmax: float = 100.0,
    spin: int = 0,
    config_type: str = "u",
):
    """Synthetic configs: one electron scans along z toward ions at origin."""
    nconfig = config_shape[0]
    nelec = config_shape[1]
    nelec_up = (nelec + spin) // 2
    nelec_down = (nelec - spin) // 2

    if config_type == "u":
        e_ind = 0
    elif config_type == "d":
        e_ind = nelec_up
    else:
        raise ValueError(f"Invalid config_type: {config_type}")

    epos = np.zeros((nconfig, nelec, 3)) - rmax
    z_grid = np.arange(-r, r, 2 * r / nconfig)
    epos[:, e_ind, 0] = 0
    epos[:, e_ind, 1] = 0
    epos[:, e_ind, 2] = z_grid
    return OpenConfigs(epos), z_grid


def find_jastrow(wf) -> jastrowspin.JastrowSpin:
    if isinstance(wf, jastrowspin.JastrowSpin):
        return wf
    if isinstance(wf, MultiplyWF):
        for fac in wf.wf_factors:
            if isinstance(fac, jastrowspin.JastrowSpin):
                return fac
    raise TypeError(
        "Wave function has no JastrowSpin factor. "
        "Use a Slater–Jastrow / MultiplyWF workflow."
    )


def jastrow_uee_uei(j: jastrowspin.JastrowSpin) -> Tuple[np.ndarray, np.ndarray]:
    """E–e and e–i parts of log J after :meth:`JastrowSpin.recompute` has been called."""
    U_ee = gpu.cp.sum(j._bvalues * j.parameters["bcoeff"], axis=(2, 1))
    U_ei = gpu.cp.einsum("ijkl,jkl->i", j._avalues, j.parameters["acoeff"])
    return np.asarray(gpu.asnumpy(U_ee.real)), np.asarray(gpu.asnumpy(U_ei.real))


def apply_serialized(wf, transform, x_vec: np.ndarray) -> None:
    newp = transform.deserialize(wf, x_vec)
    for k in newp:
        wf.parameters[k] = newp[k]


def try_initialize_boson_qmc(
    dft_checkfile: str,
    nconfig: int,
    ci_checkfile: str | None,
    load_parameters: str | None,
    jastrow_kws: dict | None,
    det_emax: Any | None,
    extra_kw: dict,
):
    """Use bosonrecipes.initialize_boson_qmc_objects when available."""
    try:
        from pyqmc.bosonrecipes import initialize_boson_qmc_objects
    except ImportError:
        return None

    kw = dict(
        dft_checkfile=dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        jastrow_kws=jastrow_kws or {},
    )
    if det_emax is not None:
        kw["det_emax"] = det_emax
    kw.update(extra_kw)
    return initialize_boson_qmc_objects(**kw)


def build_wavefunction(
    dft_checkfile: str,
    nconfig: int,
    ci_checkfile: str | None,
    load_parameters: str | None,
    jastrow_kws: dict | None,
    det_emax: Any | None,
    use_boson: bool,
    extra_kw: dict,
) -> Tuple[Any, Any, Any]:
    """Return (wf, configs, acc) with acc.transform compatible with the optimization HDF5."""
    if use_boson:
        out = try_initialize_boson_qmc(
            dft_checkfile,
            nconfig,
            ci_checkfile,
            load_parameters,
            jastrow_kws,
            det_emax,
            extra_kw,
        )
        if out is not None:
            return out
        print(
            "Warning: --boson set but pyqmc.bosonrecipes could not be imported; "
            "falling back to recipes.initialize_qmc_objects.",
            file=sys.stderr,
        )

    allowed = ("S", "slater_kws", "target_root", "nodal_cutoff")
    init_kw = {k: v for k, v in extra_kw.items() if k in allowed}
    wf, configs, acc = recipes.initialize_qmc_objects(
        dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        jastrow_kws=jastrow_kws,
        opt_wf=True,
        **init_kw,
    )
    return wf, configs, acc


def count_iterations(hdf_path: str) -> int:
    with h5py.File(hdf_path, "r") as hdf:
        if "x" in hdf.keys():
            return int(hdf["x"].shape[0])
    return 1


def main():

    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    basis_name = config['basis_name']
    symm_tag = config['symm_tag']
    det_emax = config['det_emax']
    ion_cusp = config['ion_cusp']
    na = config['na']
    nb = config['nb']
    rcut = config['rcut']
    atom_name = config['atom_name']
    ncas = config['ncas']
    nelecas = tuple(config['nelecas'])



    dft = f'{atom_name}_atom_basis_{basis_name}_diffuse_S0P0D0_v1{symm_tag}.hdf5'  # scf
    ci = f'{atom_name}_ci_atom_basis_{basis_name}_diffuse_S0P0D0{symm_tag}.hdf5' # ci
    hdf_path = f'{atom_name}_opt_cas_{ncas}_nelecas_{nelecas[0]}_{nelecas[1]}.hdf5'  # opt

    jkw = {"ion_cusp":ion_cusp, "na":na, "nb":nb, "rcut":rcut, "init_type":"zero"}
    nconfig = 1000
    use_boson = True
    extra_kw = {'opt_wf' : True}
    
    wf, configs, acc = build_wavefunction(
        dft,
        nconfig,
        ci,
        load_parameters=hdf_path,
        jastrow_kws=jkw if jkw else None,
        det_emax=det_emax,
        use_boson=use_boson,
        extra_kw=extra_kw,
    )

    transform = acc.transform
    jast = find_jastrow(wf)

    with h5py.File(hdf_path, "r") as hdf:
        if "x" in hdf.keys():
            x_hist = np.asarray(hdf["x"])
            n_it = x_hist.shape[0]
            if transform.nparams and x_hist.shape[1] != transform.nparams:
                raise ValueError(
                    f"HDF5 x has {x_hist.shape[1]} params but transform expects {transform.nparams}. "
                    "Rebuild wf/acc with the same jastrow and to_opt as the optimization run."
                )
        else:
            x_hist = None
            n_it = 1
            print(
                "No dataset 'x' in HDF5 — plotting only the current wf/ parameters.",
                file=sys.stderr,
            )
    n_plot = n_it
    norm_it = Normalize(vmin=0, vmax=max(n_plot - 1, 1))

    nelec = jast._mol.nelec
    spin = int(nelec[0] - nelec[1])
    configs_shape = (configs.configs.shape[0], configs.configs.shape[1])

    fig_number = {"uu": 0, "ud": 1, "du": 2, "dd": 3, "u": 4, "d": 5}

    def plot_one_iteration(axs_flat, it: int, color) -> None:
        if x_hist is not None:
            apply_serialized(wf, transform, x_hist[it])

        for config_type in ["uu", "ud", "du", "dd"]:
            cfg_ee, x = get_ee_configs(configs_shape, r=rcut, config_type=config_type, spin=spin)
            iax = fig_number[config_type]
            if cfg_ee is None:
                continue
            jast.recompute(cfg_ee)
            U_ee, U_ei = jastrow_uee_uei(jast)
            axs_flat[iax].plot(x, U_ee, color=color, alpha=0.9, linewidth=1.2)
            # axs_flat[iax].plot(x, U_ei, color=color, alpha=0.55, linestyle="--", linewidth=1.0)
            axs_flat[iax].set_title(f"e–e scan ({config_type})")
            axs_flat[iax].set_xlabel(r"$r_{ee}$ (a.u.)")
            axs_flat[iax].set_ylabel(r"contribution to $\ln J$")

        for config_type in ["u", "d"]:
            cfg_ei, x = get_ei_configs(configs_shape, r=rcut, config_type=config_type, spin=spin)
            iax = fig_number[config_type]
            jast.recompute(cfg_ei)
            U_ee, U_ei = jastrow_uee_uei(jast)
            # axs_flat[iax].plot(x, U_ee, color=color, alpha=0.9, linewidth=1.2)
            axs_flat[iax].plot(x, U_ei, color=color, alpha=0.55, linestyle="--", linewidth=1.0)
            axs_flat[iax].set_title(f"e–i scan ({config_type})")
            axs_flat[iax].set_xlabel(r"$r_{ei}$ (a.u.)")
            axs_flat[iax].set_ylabel(r"contribution to $\ln J$")

    # if True:
    #     for it in range(n_plot):
    #         fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    #         axs = axs.flatten()
    #         c = mpl_cm.plasma(norm_it(it))
    #         plot_one_iteration(axs, it, c)
    #         for ax in axs:
    #             ax.axhline(0.0, color="k", linewidth=0.3, alpha=0.4)
    #         fig.suptitle(
    #             rf"Jastrow: $U_{{ee}}$ (solid), $U_{{ei}}$ (dashed) — iteration {it}"
    #         )
    #         plt.tight_layout()
    #         ppath = f"jastrow_ee_ei_iterations_it{it}.png"
    #         plt.savefig(ppath, dpi=300, bbox_inches="tight")
    #         plt.close(fig)
    #     print(f"Wrote {n_plot} figures with prefix {args.per_iteration_prefix}")
    #     return

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()
    for it in range(n_plot):
        c = mpl_cm.plasma(norm_it(it))
        plot_one_iteration(axs, it, c)

    for ax in axs:
        ax.axhline(0.0, color="k", linewidth=0.3, alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_it)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.6, label="iteration index")
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        r"Jastrow $U_{ee}$ (solid) vs $U_{ei}$ (dashed) contributions to $\ln J$ (colormap = iteration)"
    )
    plt.tight_layout()
    plt.savefig("jastrow_ee_ei_iterations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote jastrow_ee_ei_iterations.png")


if __name__ == "__main__":
    main()
