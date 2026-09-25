# MIT License
# 
# Copyright (c) 2019-2024 The PyQMC Developers
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import os
import time
import numpy as np
# import pyqmc.method.mc as mc
from pyqmc.method import mc
import sys
import h5py
import logging
import copy
import pyqmc.method.bosonmc as bosonmc

_bacc = None


def _bacc_mod():
    global _bacc
    if _bacc is None:
        from pyqmc.observables import bosonaccumulators as m

        _bacc = m
    return _bacc


from pyqmc.method import boson_profile_config as _bpf


def _bosondmc_prof_enabled():
    return _bpf.is_enabled()


_prof_parallel_client_warned = False
_dmc_propagate_mpi_prof_noted = False


def _warn_profile_parallel_client(client):
    """Profiling prints run inside worker dmc_propagate; driver logs often miss them."""
    global _prof_parallel_client_warned
    if (
        client is None
        or not _bosondmc_prof_enabled()
        or _prof_parallel_client_warned
    ):
        return
    _prof_parallel_client_warned = True
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    except Exception:
        pass
    _bacc_mod().bdmc_profile_print(
        "pyqmc boson DMC profiling: a parallel client is in use (e.g. MPIPoolExecutor). "
        "Banner and periodic profile lines are printed from worker ranks during "
        "dmc_propagate—they may not appear in this process's stdout. "
        "Use serial DMC (client=None) to see them here, or inspect worker/MPI logs. "
        "Enable via config.yaml (profile_boson_dmc / profile_abcdmc_print_every) or env PYQMC_PROFILE_*."
    )


def limdrift(g, tau, acyrus=0.25):
    """
    Use Cyrus Umrigar's algorithm to limit the drift near nodes.

    :parameter g: a [nconf,ndim] vector
    :parameter tau: time step
    :parameter acyrus: the maximum magnitude
    :returns: The vector with the cut off applied and multiplied by tau.
    """
    v2 = np.sum(g**2, axis=1)
    mask = v2 > 1e-8
    taueff = np.ones(v2.shape) * tau
    taueff[mask] = (np.sqrt(1 + 2 * tau * acyrus * v2[mask]) - 1) / (acyrus * v2[mask])
    return g * taueff[:, np.newaxis]


def get_V2(configs, wf, acc_out):
    if "grad2" in acc_out.keys():
        return acc_out["grad2"]

    nconfig, nelec = configs.configs.shape[0:2]
    v2 = np.zeros(nconfig)
    for e in range(nelec):
        v2 += np.sum(np.abs(wf.gradient(e, configs.electron(e))).T ** 2, axis=1)
    return v2


def propose_drift_diffusion(wf, configs, tstep, e):
    nconfig = configs.configs.shape[0]
    prof = _bosondmc_prof_enabled()
    bacc = _bacc_mod() if prof else None

    # _, val_old = wf.recompute(configs) # Kayahan added 
    # wf_new = copy.deepcopy(wf)         # Kayahan added (1)

    if prof:
        t0 = time.perf_counter()
    grad_e = wf.gradient(e, configs.electron(e))
    if prof:
        bacc.bdmc_prop_inner_add("propdd_wf_gradient", time.perf_counter() - t0)
        t0 = time.perf_counter()
    gradt = limdrift(np.real(grad_e.T), tstep)
    # np.random.seed(1)
    gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconfig, 3))
    eposnew = configs.configs[:, e, :] + gauss + gradt
    newepos = configs.make_irreducible(e, eposnew)
    if prof:
        bacc.bdmc_prop_inner_add(
            "propdd_forward_drift_draw_geom", time.perf_counter() - t0
        )
        t0 = time.perf_counter()

    # Compute reverse move
    # g, wfratio, saved = wf.gradient_value(e, newepos)
    g, ks_ratio, saved = wf.gradient_value(e, newepos)  # Kayahan modified (2)
    if prof:
        bacc.bdmc_prop_inner_add("propdd_wf_gradient_value", time.perf_counter() - t0)
        t0 = time.perf_counter()
    new_grad = limdrift(np.real(g.T), tstep)
    forward = np.sum(gauss**2, axis=1)
    backward = np.sum((gauss + gradt + new_grad) ** 2, axis=1)
    t_prob = np.exp(1 / (2 * tstep) * (forward - backward))

    # newcoord = copy.deepcopy(configs)           # Kayahan added     
    # newcoorde = newcoord.configs[:, e, :] + gauss + gradt # Kayahan added 
    # newcoorde = newcoord.make_irreducible(e, newcoorde) # Kayahan added 
    # newcoord.configs[:,e,:] = newcoorde.configs # Kayahan added 
    # # print(newcoord.configs[0])
    # _, val_new = wf_new.recompute(newcoord) # Kayahan added (3)
    # wfratio = np.exp((val_new-val_old)) # Kayahan added (4)
    # # Acceptance -- fixed-node: reject if wf changes sign
    # ratio = np.abs(wfratio) ** 2 * t_prob #(5) 
    
    if np.any(ks_ratio < 0):
        print("WARNING: Negative wf_ratio detected")
    ratio = np.abs(ks_ratio) * t_prob
    
    # if wf.dtype == float:             # Kayahan modified, no fixed node error
    #     ratio *= np.sign(wfratio)     # Kayahan modified, no fixed node error
    accept = ratio > np.random.rand(nconfig)
    r2 = np.sum((gauss + gradt) ** 2, axis=1)
    if prof:
        bacc.bdmc_prop_inner_add("propdd_metropolis_accept", time.perf_counter() - t0)

    return newepos, accept, r2, saved


def propose_tmoves(wf, configs, energy_accumulator, tstep, e):
    """
    No side effect calculation of t-moves

    Returns:
       new proposed positions
       probability of acceptance
       sum of weights
    """
    moves = energy_accumulator.nonlocal_tmoves(configs, wf, e, tstep)
    t_amplitudes = moves["ratio"] * moves["weight"]

    forward_probability = np.zeros_like(t_amplitudes)
    forward_probability[t_amplitudes > 0] = t_amplitudes[t_amplitudes > 0]
    norm = 1.0 + np.sum(forward_probability, axis=1)  # EQN 34

    def select_walker(array):
        r = np.random.rand()
        return np.searchsorted(array, r)

    cdf = np.cumsum(forward_probability / norm[:, np.newaxis], axis=1)
    selected_moves = np.apply_along_axis(select_walker, 1, cdf)
    move_selected = selected_moves < t_amplitudes.shape[1]

    newpos = np.zeros((norm.shape[0], 3))
    reverse_ratio = np.zeros((norm.shape[0]))
    backward_amplitudes = t_amplitudes.copy()
    for walker, move in enumerate(selected_moves):
        if move_selected[walker]:
            newpos[walker, :] = moves["configs"].configs[walker, move, :]
            reverse_ratio[walker] = 1.0 / moves["ratio"][walker, move]
            backward_amplitudes[walker, :] *= reverse_ratio[walker]
            # This is the move back to the original position
            backward_amplitudes[walker, move] = (
                reverse_ratio[walker] * moves["weight"][walker, move]
            )
        else:
            newpos[walker, :] = configs.configs[walker, e, :]
            reverse_ratio[walker] = 0.0

    newpos = configs.make_irreducible(e, newpos)

    backward_amplitudes[backward_amplitudes < 0] = 0.0
    back_norm = 1.0 + np.sum(backward_amplitudes, axis=1)
    acceptance = norm / back_norm
    acceptance[move_selected == False] = 0.0

    return newpos, move_selected, acceptance, np.sum(t_amplitudes)


def dmc_propagate(
    wf,
    configs,
    weights,
    tstep,
    branchcut_start,
    e_trial,
    e_est,
    nsteps=5,
    accumulators=None,
    ekey=("energy", "total"),
    no_branching=False,
):
    """
    Propagate DMC without branching

    :parameter wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators
    :parameter configs: Configs object, (nconfig, nelec, 3) - initial coordinates to start calculation.
    :parameter weights: (nconfig,) - initial weights to start calculation
    :parameter tstep: Time step for move proposals. Introduces time step error.
    :parameter nsteps: number of DMC steps to take
    :parameter accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.
    :parameter ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]
    :returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation

    """
    global _dmc_propagate_mpi_prof_noted
    assert accumulators is not None, "Need an energy accumulator for DMC"
    if _bosondmc_prof_enabled() and not _dmc_propagate_mpi_prof_noted:
        _dmc_propagate_mpi_prof_noted = True
        try:
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_size() > 1:
                _bacc_mod().bdmc_profile_print(
                    "dmc_propagate: profiling on this MPI rank (parallel DMC partition)."
                )
        except Exception:
            pass
    nconfig, nelec = configs.configs.shape[0:2]
    wf.recompute(configs)
    
    energy_acc = accumulators[ekey[0]](configs, wf)
    eloc = energy_acc[ekey[1]].real
    v2 = get_V2(configs, wf, energy_acc)
    df = []

    for _ in range(nsteps):
        b = _bacc_mod() if _bosondmc_prof_enabled() else None
        if b is not None:
            t_step0 = time.perf_counter()
        t_abcdmc = 0.0
        r2_accepted = np.zeros(nconfig)
        r2_proposed = np.zeros(nconfig)
        prob_acceptance = np.zeros(nconfig)
        # tmove_acceptance = np.zeros(nconfig)
        if b is not None:
            t_inner = time.perf_counter()

        # if accumulators[ekey[0]].has_nonlocal_moves():
        #     for e in range(nelec):  # T-moves
        #         newepos, mask, probability, ecp_totweight = propose_tmoves(
        #             wf, configs, accumulators[ekey[0]], tstep, e
        #         )
        #         accept = mask & (probability > np.random.rand(nconfig))
        #         configs.move(e, newepos, accept)
        #         wf.updateinternals(e, newepos, configs, mask=accept)
        #         tmove_acceptance += accept / nelec
        
        # wf.curr_config = copy.deepcopy(configs)

        for e in range(nelec):  # drift-diffusion
            newepos, accept, r2, saved = propose_drift_diffusion(wf, configs, tstep, e)
            if b is not None:
                # propose_drift_diffusion adds propdd_* keys; refresh mark for next section
                t_inner = time.perf_counter()
            configs.move(e, newepos, accept)
            if b is not None:
                t_inner = _bacc_mod().bdmc_prop_inner_mark("configs_move", t_inner)
            wf.updateinternals(e, newepos, configs, mask=accept, saved_values=saved)
            if b is not None:
                t_inner = _bacc_mod().bdmc_prop_inner_mark(
                    "wf_updateinternals", t_inner
                )
            r2_proposed += r2
            r2_accepted[accept] += r2[accept]
            prob_acceptance += accept / nelec
            if b is not None:
                t_inner = _bacc_mod().bdmc_prop_inner_mark(
                    "electron_r2_accept_stats", t_inner
                )

        # weights
        elocold = eloc.copy()
        v2old = v2.copy()
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("state_copy_eloc_v2", t_inner)
        energydat = accumulators[ekey[0]](configs, wf)
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("energy_accumulator", t_inner)
        eloc = energydat[ekey[1]].real

        tdamp = r2_accepted / r2_proposed
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("tdamp_eloc_extract", t_inner)
        v2 = get_V2(configs, wf, energydat)
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("get_V2", t_inner)

        Snew = compute_S(e_trial, e_est, branchcut_start, v2, tstep, eloc, nelec)
        Sold = compute_S(e_trial, e_est, branchcut_start, v2old, tstep, elocold, nelec)
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("compute_S", t_inner)
        if no_branching:
            wmult = 1
        else:
            wmult = np.exp(tstep * tdamp * (0.5 * Snew + 0.5 * Sold))
        weights *= wmult
        wavg = np.mean(weights)
        if b is not None:
            t_inner = _bacc_mod().bdmc_prop_inner_mark("weight_update", t_inner)
        # print(wavg)
        
        avg = {}
        t_accum_loop = time.perf_counter() if b is not None else None
        for k, accumulator in accumulators.items():
            if k == ekey[0]:
                dat = energydat
            elif b is not None and k == b.ABCDMC_ACC_KEY:
                t0a = time.perf_counter()
                dat = accumulator(configs, wf)
                t_abcdmc = time.perf_counter() - t0a
                b.bdmc_profile_add_abcdmc(t_abcdmc)
            else:
                dat = accumulator(configs, wf)
            for m, res in dat.items():
                avg[k + m] = np.einsum("...i,i...->...", weights, res) / (
                    nconfig * wavg
                )
        if b is not None:
            _bacc_mod().bdmc_prop_inner_add(
                "accumulators_avg_loop_excl_ABCDMC",
                time.perf_counter() - t_accum_loop - t_abcdmc,
            )
        avg["weight"] = wavg
        avg["acceptance"] = np.mean(prob_acceptance)
        # avg["tmove_acceptance"] = np.mean(tmove_acceptance)
        if b is not None:
            b.bdmc_profile_add_prop(time.perf_counter() - t_step0 - t_abcdmc)
            b.bdmc_profile_end_step(accumulators)
        df.append(avg)
    weight = np.asarray([d["weight"] for d in df])
    avg_weight = weight / np.mean(weight)
    df_ret = {
        k: np.mean([d[k] * w for d, w in zip(df, avg_weight)], axis=0)
        for k in df[0].keys()
    }

    df_ret["weight"] = np.mean(weight)

    return df_ret, configs, weights


def compute_S(e_trial, e_est, branchcut, v2, tau, eloc, nelec):
    r"""
    .. math:: S = E_T - E_{\rm est} + \frac{f_{\rm sat}(E_{\rm est} - E_L; c_{\rm branch})}{1 + \frac{v^2\tau}{n_{\rm elec}}}

    :math:`f_{\rm sat}(x; c)` is the saturation function: :math:`x` if :math:`|x| < c` and :math:`c{\rm sign}(x)` otherwise.
    """
    e_cut = e_est - eloc
    mask = np.abs(e_cut) > branchcut
    e_cut[mask] = branchcut * np.sign(e_cut[mask])
    denominator = 1 + (v2 * tau / nelec) ** 2

    return e_trial - e_est + e_cut / denominator


def dmc_propagate_parallel(wf, configs, weights, client, npartitions, *args, **kwargs):
    r"""Parallelizes calls to dmc_propagate by splitting configs

    If npartitions does not evenly divide nconfigs, we need to reweight the results based on the number of configs per parallel task.

    The final result should be equivalent to the non-parallelized case.
    The average weight :math:`w` and the weighted average of observables :math:`\langle O \rangle` are returned.
    Index :math:`i` refers to walker index.

    .. math::
        w = \sum_i w_i / n_{\rm config}
        \qquad\quad \langle O \rangle = \sum_i o_{i}  w_i / \sum_i w_i

    Split over parallel tasks, we need to reweight by number of walkers.
    The average weight :math:`w_p` and weighted average of observables :math:`\langle O\rangle_p` are returned from each task.

    .. math::
        w_p = \sum_j^{{\rm task}\, p} w_j / n_{{\rm config}, p}
        \qquad\quad \langle O \rangle_p = \frac{\sum_j^{{\rm task}\, p} o_{j}  w_j }{ \sum_j^{{\rm task}\, p} w_j }


    The total weight and total average (defined above) are computed from the task weights :math:`w_p` and task averages :math:`\langle O\rangle_p` as

    .. math::
        w = \sum_p w_p n_{{\rm config}, p} /  n_{\rm config},
        \qquad\quad \langle O \rangle = \frac{ \sum_p \langle O\rangle_p  \sum_j^{{\rm task}\, p} w_j }{ \sum_i w_i}.

    We can rewrite the weights using the equations above

    .. math::
        \langle O \rangle &= \frac{ \sum_p \langle O\rangle_p w_p  n_{{\rm config}, p}  }{ w n_{\rm config} }

        &= \sum_p \langle O\rangle_p \frac{w_p n_{{\rm config}, p}}{\sum_p w_p n_{{\rm config}, p}}


    By reweighting the task weights as :math:`\overline{w}_p = w_p n_{{\rm config}, p}`, we can omit the reweighting factor :math:`\frac{n_{{\rm config}, p}}{n_{\rm config}}` (that we use to collect parallel vmc).
    Instead, we use only the reweighting factor :math:`\overline{w}_p / \sum_p \overline{w}_p`

    .. math:: \langle O \rangle = \sum_p \langle O\rangle_p \frac{\overline{w}_p }{\sum_p \overline{w}_p }
    """

    config = configs.split(npartitions)
    weight = np.array_split(weights, npartitions)
    runs = [
        client.submit(dmc_propagate, wf, conf, wt, *args, **kwargs)
        for conf, wt in zip(config, weight)
    ]
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[1])
    weights = np.concatenate(allresults[2])
    confweight = np.array([len(c.configs) for c in config], dtype=float)
    weight = np.array([w["weight"] for w in allresults[0]]) * confweight
    weight_avg = weight / np.sum(weight)
    block_avg = {
        k: np.sum(
            [res[k] * ww for res, ww in zip(allresults[0], weight_avg)],
            axis=0,
        )
        for k in allresults[0][0].keys()
    }
    block_avg["weight"] = np.mean(weight)
    return block_avg, configs, weights


def branch(configs, weights, nconfig_out=None):
    """
    Perform branching on a set of walkers using the 'stochastic comb'

    Walkers are resampled with probability proportional to the weights, and the
    new weights are all set to be equal to the average weight.

    :parameter configs: (nconfig,nelec,3) walker coordinates
    :parameter weights: (nconfig,) walker weights
    :parameter nconfig_out: optional output population size (default: same as input)
    :returns: resampled walker configurations and weights all equal to average weight
    """
    weights = np.asarray(weights, dtype=np.float64).ravel()
    nconfig_in = configs.configs.shape[0]
    if nconfig_out is None:
        nconfig_out = nconfig_in
    nconfig_out = int(nconfig_out)
    if np.any(weights > 2.0):
        logging.warning("Some weights are larger than 2")
    probability = np.cumsum(weights)
    wtot = float(probability[-1])
    if wtot <= 0:
        raise ValueError("branch: total weight must be positive")

    base = np.random.rand() * wtot
    newinds = np.searchsorted(
        probability,
        (base + np.linspace(0, wtot, nconfig_out, endpoint=False)) % wtot,
    )
    unique, counts = np.unique(newinds, return_counts=True)

    configs.resample(newinds)
    weights = np.full(nconfig_out, wtot / nconfig_out, dtype=np.float64)
    return (
        configs,
        weights,
        {
            "max branches": np.max(counts),
            "Number of walkers killed": nconfig_in - unique.shape[0],
        },
    )


def population_snapshots_count(hdf_file):
    """Number of population snapshots already stored in ``hdf_file``."""
    if hdf_file is None or not os.path.isfile(hdf_file):
        return 0
    with h5py.File(hdf_file, "r") as hdf:
        if "population_snapshots" not in hdf:
            return 0
        return int(hdf["population_snapshots"].attrs.get("n_written", 0))


def snapshot_schedule(nblocks, n_snapshots, blockoffset=0, n_already=0):
    """Equidistant block indices for the next population snapshots to write.

    Fresh eq (``n_already=0``, ``blockoffset=0``): for ``nblocks=100``,
    ``n_snapshots=10`` → ``{9, 19, ..., 99}`` (0-based).

    Restart: if ``n_snapshots=10`` and ``n_already=2``, schedule ``10 - 2``
    snapshots equidistant over ``[blockoffset, nblocks - 1]`` (remaining eq
    blocks only, even if ``nblocks`` or ``blockoffset`` changed).
    """
    if n_snapshots is None:
        return set()
    n_snapshots = int(n_snapshots)
    n_already = int(n_already)
    if n_snapshots <= 0 or nblocks <= 0:
        return set()
    remaining = n_snapshots - n_already
    if remaining <= 0 or nblocks <= blockoffset:
        return set()
    blockoffset = int(blockoffset)
    if n_already == 0 and blockoffset == 0:
        first = max(0, int(nblocks) // n_snapshots - 1)
        if first >= nblocks:
            first = nblocks - 1
        blocks = np.linspace(first, nblocks - 1, remaining, dtype=int)
    else:
        blocks = np.linspace(blockoffset, nblocks - 1, remaining, dtype=int)
    return set(int(b) for b in np.unique(blocks))


def write_population_snapshot(
    hdf_file, n_slots, block, configs, weights, e_trial, e_est, esigma
):
    """Append one post-branch population into ``population_snapshots/``."""
    if hdf_file is None:
        return
    weights = np.asarray(weights, dtype=np.float64).ravel()
    with h5py.File(hdf_file, "a") as hdf:
        if "population_snapshots" not in hdf:
            g = hdf.create_group("population_snapshots")
            nwalk, nelec, _ = configs.configs.shape
            g.create_dataset("block", shape=(n_slots,), dtype=np.int64, fillvalue=-1)
            g.create_dataset(
                "configs",
                shape=(n_slots, nwalk, nelec, 3),
                dtype=configs.configs.dtype,
            )
            g.create_dataset("weights", shape=(n_slots, nwalk), dtype=np.float64)
            g.create_dataset("e_trial", shape=(n_slots,), dtype=np.float64)
            g.create_dataset("e_est", shape=(n_slots,), dtype=np.float64)
            g.create_dataset("esigma", shape=(n_slots,), dtype=np.float64)
            g.attrs["n_written"] = 0
        else:
            g = hdf["population_snapshots"]
        i = int(g.attrs.get("n_written", 0))
        if i > 0:
            existing = {int(g["block"][j]) for j in range(i)}
            if int(block) in existing:
                return
        if i >= n_slots:
            logging.warning(
                "population_snapshots already has %s entries; skipping block %s",
                n_slots,
                block,
            )
            return
        g["block"][i] = int(block)
        g["configs"][i] = configs.configs
        g["weights"][i] = weights
        g["e_trial"][i] = float(e_trial)
        g["e_est"][i] = float(e_est)
        g["esigma"][i] = float(esigma)
        g.attrs["n_written"] = i + 1


def assemble_population_from_snapshots(hdf_path, configs, nconfig_out=None):
    """Concat equidistant eq snapshots, equalize time-slice weight, comb to size.

    Returns ``(configs, weights, e_trial, e_est, esigma)``.
    """
    with h5py.File(hdf_path, "r") as hdf:
        if "population_snapshots" not in hdf:
            raise ValueError(
                f"start_from={hdf_path!r} has no population_snapshots group; "
                "re-run equilibrium with population_snapshots=K"
            )
        g = hdf["population_snapshots"]
        n_written = int(g.attrs.get("n_written", g["block"].shape[0]))
        if n_written <= 0:
            raise ValueError(f"population_snapshots in {hdf_path!r} is empty")
        snap_configs = np.asarray(g["configs"][:n_written])
        snap_weights = np.asarray(g["weights"][:n_written], dtype=np.float64)
        e_trial = float(g["e_trial"][n_written - 1])
        e_est = float(g["e_est"][n_written - 1])
        esigma = float(g["esigma"][n_written - 1])

    k, n_eq = snap_weights.shape
    w_parts = []
    for s in range(k):
        ws = snap_weights[s].copy()
        ssum = float(np.sum(ws))
        if ssum <= 0:
            raise ValueError(f"snapshot {s} has non-positive total weight")
        ws *= (1.0 / k) / ssum
        w_parts.append(ws)
    weights = np.concatenate(w_parts)
    configs.configs = snap_configs.reshape(k * n_eq, *snap_configs.shape[2:]).copy()
    if nconfig_out is None:
        nconfig_out = k * n_eq
    configs, weights, _info = branch(configs, weights, nconfig_out=nconfig_out)
    return configs, weights, e_trial, e_est, esigma


def dmc_file(hdf_file, data, attr, configs, weights):
    import pyqmc.method.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
            if "weights" not in hdf.keys():
                hdf.create_dataset("weights", weights.shape)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)
            hdf["weights"][:] = weights


def evaluate_energy_worker(configs, wf, en):
    wf.recompute(configs)
    return en(configs, wf)


def evaluate_energies(wf, configs, en, client, npartitions):
    if client is None:
        return evaluate_energy_worker(configs, wf, en)

    else:
        config = configs.split(npartitions)
        runs = [client.submit(evaluate_energy_worker, conf, wf, en) for conf in config]
        ret = {}
        data = [r.result() for r in runs]
        for k in data[0].keys():
            ret[k] = np.concatenate([d[k] for d in data])
        return ret


def rundmc(
    wf,
    configs,
    weights=None,
    tstep=0.01,
    nblocks=200,
    nsteps_per_block=5,
    blockoffset=0,
    accumulators=None,
    verbose=False,
    hdf_file=None,
    continue_from=None,
    start_from=None,
    population_snapshots=None,
    client=None,
    npartitions=None,
    ekey=("energy", "total"),
    vmc_options=None,
    branchcut_start=10,
    feedback=1.0,
    branchtime=None,
    stepoffset=None,
    nsteps=None,
    no_branching=False,
):
    """
    Run DMC

    :parameter wf: trial wave function for DMC. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators
    :type wf: a PyQMC wave-function-like object
    :parameter configs: (nconfig, nelec, 3) - initial electron coordinates to start calculation.
    :type configs: PyQMC configs object
    :parameter weights: (nconfig,) - initial weights to start calculation, defaults to uniform.
    :parameter float tstep: Time step for move proposals. Introduces time step error.
    :parameter int nblocks: number of DMC blocks to run; branching is performed at the end of each block. If a calculation is continued (either from continue_from or from using the same hdf_file as a previous call), nblocks includes the blocks from previous calls; i.e., nblocks is the total number of blocks run over all the calls to rundmc. For ``start_from``, ``nblocks`` is the number of blocks in this run only (indexing starts at 0).
    :parameter int nsteps_per_block: number of steps to take between branching; branching is performed at the end of each block
    :parameter int blockoffset: If continuing a run, what to start the block numbering at. The calculation will stop when the block number reaches nblocks.
    :parameter accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.
    :parameter boolean verbose: Print out step information
    :parameter str hdf_file: Hdf_file to store dmc output.
    :parameter str continue_from: Hdf_file to continue dmc calculation from (same-run restart; block indexing continues).
    :parameter str start_from: Equilibrium DMC HDF5 with ``population_snapshots``; assemble walkers, reset block index to 0, skip VMC. Mutually exclusive with ``continue_from``.
    :parameter int population_snapshots: If set, save this many equidistant post-branch populations into ``population_snapshots/`` for a later ``start_from`` stats run.
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :parameter ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]
    :parameter int branchcut_start: Used in computing weights. Recommended for "experts only".
    :parameter float feedback: Feedback strength for controlling normalization. Recommended for "experts only".
    :returns: (df,coords,weights)
      df: A list of dictionaries nblocks long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
    :rtype: list of dictionaries, pyqmc.configurations.coord.Configs, ndarray
    """
    # Deprecated backwards compatibility
    if branchtime is not None:
        logging.warning("rundmc kwarg `branchtime` is deprecated. Use `nsteps_per_block` instead. Overriding `nsteps_per_block` if given.")
        nsteps_per_block = branchtime
    if nsteps is not None:
        logging.warning("rundmc kwarg `nsteps` is deprecated. Use `nblocks` and `nsteps_per_block` instead. Overriding nblocks if given.")
        nblocks = nsteps // nsteps_per_block
    if stepoffset is not None:
        logging.warning("rundmc kwarg `stepoffset` is deprecated. Use `blockoffset` and `nsteps_per_block` instead. Overriding blockoffset if given.")
        blockoffset = stepoffset // nsteps_per_block

    if start_from is not None and continue_from is not None:
        raise ValueError("start_from and continue_from are mutually exclusive")

    # Don't continue onto a file that's already there.
    if (
        (continue_from is not None or start_from is not None)
        and hdf_file is not None
        and os.path.isfile(hdf_file)
    ):
        raise RuntimeError(
            f"continue_from/start_from is set but hdf_file={hdf_file} already exists! "
            f"Delete or rename {hdf_file} and try again."
        )

    # Restart if hdf_file is there (not used with start_from)
    if (
        continue_from is None
        and start_from is None
        and hdf_file is not None
        and os.path.isfile(hdf_file)
    ):
        continue_from = hdf_file

    if start_from is not None:
        nconfig_out = configs.configs.shape[0]
        configs, weights, e_trial, e_est, esigma = assemble_population_from_snapshots(
            start_from, configs, nconfig_out=nconfig_out
        )
        blockoffset = 0
        if verbose:
            print(
                f"Starting from population_snapshots in {start_from} "
                f"({configs.configs.shape[0]} walkers); block indexing resets to 0"
            )
        wf.recompute(configs)
    elif continue_from is not None:
        with h5py.File(continue_from, "r") as hdf:
            if "block" not in hdf.keys() and "step" in hdf.keys() :
                logging.warning("Warning: found deprecated key `step` in the restart file. In future versions, `block` key will be expected. All data from this run will be indexed by the key `block`.")
                blockoffset = hdf["step"][-1] // nsteps_per_block + 1
            else:
                blockoffset = hdf["block"][-1] + 1
                
            configs.load_hdf(hdf)
            weights = np.array(hdf["weights"])
            if "e_trial" not in hdf.keys():
                raise ValueError(
                    "Did not find e_trial in the restart file. This may mean that you are trying to restart from a different version of DMC"
                )
            e_trial = hdf["e_trial"][-1]
            e_est = hdf["e_est"][-1]
            esigma = hdf["esigma"][-1]
            if verbose:
                print(f"Restarting calculation {continue_from} from block {blockoffset}")
            # Evaluate once to update the wave function
            wf.recompute(configs)

    else:
        vmc_options_default = {'nsteps_per_block': 10, 'nblocks': 100,  'tstep': 0.3, "hdf_file": "vmc.hdf5"}
        if vmc_options is not None:
            vmc_options_default.update(vmc_options) 
        

        df, configs = bosonmc.abvmc(
            wf,
            configs,
            client=client,
            npartitions=npartitions,
            verbose=verbose,
            **vmc_options_default,
        )
        en = evaluate_energies(wf, configs, accumulators[ekey[0]], client, npartitions)[
            ekey[1]
        ]
        eref = np.mean(en).real
        e_trial = eref
        e_est = eref
        esigma = np.std(en)
        if verbose:
            print("eref start", eref, "esigma", esigma)

    n_already = (
        population_snapshots_count(hdf_file)
        if population_snapshots is not None
        else 0
    )
    snap_blocks = snapshot_schedule(
        nblocks,
        population_snapshots,
        blockoffset=blockoffset,
        n_already=n_already,
    )
    n_snap_slots = (
        int(population_snapshots) if population_snapshots is not None else 0
    )
    nconfig = configs.configs.shape[0]
    if weights is None:
        weights = np.ones(nconfig)

    if _bosondmc_prof_enabled():
        _bacc_mod().bdmc_profile_note_rundmc_start(client is not None)

    df = []
    if blockoffset >= nblocks:
        logging.warning(f"blockoffset {blockoffset} >= nblocks {nblocks}; no steps will be run.")
    for block in range(blockoffset, nblocks):
        _warn_profile_parallel_client(client)
        if client is None:
            df_, configs, weights = dmc_propagate(
                wf,
                configs,
                weights,
                tstep,
                branchcut_start * esigma,
                e_trial=e_trial,
                e_est=e_est,
                nsteps=nsteps_per_block,
                accumulators=accumulators,
                ekey=ekey,
                no_branching=no_branching,
            )
        else:
            df_, configs, weights = dmc_propagate_parallel(
                wf,
                configs,
                weights,
                client,
                npartitions,
                tstep,
                branchcut_start * esigma,
                e_trial=e_trial,
                e_est=e_est,
                nsteps=nsteps_per_block,
                accumulators=accumulators,
                ekey=ekey,
                no_branching=no_branching,  
            )

        df_["e_trial"] = e_trial
        df_["e_est"] = e_est
        df_["block"] = block
        df_["esigma"] = esigma
        df_["tstep"] = tstep
        df_["weight_std"] = np.std(weights)
        df_["nsteps_per_block"] = nsteps_per_block

        configs, weights, branch_info = branch(configs, weights)
        df_.update(branch_info)
        df.append(df_)
        if _bosondmc_prof_enabled():
            t_h0 = time.perf_counter()
            dmc_file(hdf_file, df_, {}, configs, weights)
            _bacc_mod().bdmc_profile_add_hdf(time.perf_counter() - t_h0)
        else:
            dmc_file(hdf_file, df_, {}, configs, weights)

        if block in snap_blocks:
            write_population_snapshot(
                hdf_file,
                n_snap_slots,
                block,
                configs,
                weights,
                e_trial,
                e_est,
                esigma,
            )
            if verbose:
                n_filled = block + 1
                n_remaining = n_snap_slots - n_filled
                print(
                    f"Saved population snapshot at block {block} "
                    f"({n_filled} filled, {n_remaining} remaining out of {n_snap_slots})"
                )
           

        e_est = estimate_energy(hdf_file, df, ekey)
        e_trial = e_est - feedback * np.log(np.mean(weights)).real

        if verbose:
            print(
                "energy",
                df_[ekey[0] + ekey[1]],
                "e_trial",
                e_trial,
                "e_est",
                e_est,
                "sigma(w)",
                df_["weight_std"],
            )
            print(branch_info)

    if len(df) > 0:
        df_ret = {k: np.asarray([d[k] for d in df]) for k in df[0].keys()}
    else:
        df_ret = {}
    return df_ret, configs, weights


def estimate_energy(hdf_file, df, ekey):
    if hdf_file is not None:
        with h5py.File(hdf_file, "r") as f:
            en = f[ekey[0] + ekey[1]][()]
            wt = f["weight"][()]
    else:
        en = np.asarray([d[ekey[0] + ekey[1]] for d in df])
        wt = np.asarray([d["weight"] for d in df])
    warmup = int(len(en) / 4)
    return np.average(en[warmup:], weights=wt[warmup:]).real
