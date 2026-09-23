#!/usr/bin/env python3
"""Diagnose DMC population-control / walker-weight health from an ABDMC HDF5 file.

Usage:
  python diagnose_weights.py [hdf5]
  python diagnose_weights.py c_dmc_cas_8_nelecas_3_1.hdf5 --out weights_diag.png

Healthy DMC roughly means:
  - block-averaged ``weight`` near O(1) after equilibration
  - modest ``weight_std``
  - e_trial ≈ e_est  (gap ≈ -log⟨w⟩ after branching)
  - not too many walkers killed / max branches each block
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CANDIDATES = (
    "c_dmc_cas_8_nelecas_3_1.hdf5",
    "c_dmc_eq_cas_8_nelecas_3_1.hdf5",
    "f_dmc_cas_8_nelecas_4_3.hdf5",
    "f_dmc_eq_cas_8_nelecas_4_3.hdf5",
)

KEYS = (
    "block",
    "weight",
    "weight_std",
    "weights",
    "e_trial",
    "e_est",
    "energytotal",
    "Number of walkers killed",
    "max branches",
    "esigma",
    "acceptance",
)


def find_default_hdf5(cwd: str) -> str | None:
    for name in DEFAULT_CANDIDATES:
        path = os.path.join(cwd, name)
        if os.path.isfile(path):
            return path
    return None


def load(path: str) -> dict:
    with h5py.File(path, "r") as f:
        missing = [k for k in ("weight", "e_trial", "e_est") if k not in f]
        if missing:
            raise KeyError(f"{path} missing required keys: {missing}")
        data = {}
        for k in KEYS:
            if k in f:
                data[k] = np.asarray(f[k][...]).ravel()
        # Align block-length series (some energy keys can be off-by-one)
        n = min(len(data[k]) for k in ("weight", "e_trial", "e_est"))
        for k, v in list(data.items()):
            if k == "weights":
                continue
            if len(v) >= n:
                data[k] = v[:n]
        if "block" not in data:
            data["block"] = np.arange(n)
        return data


def summarize(data: dict, late: int = 100) -> str:
    w = data["weight"]
    n = len(w)
    late = min(late, max(1, n // 4))
    sl = slice(-late, None)

    et = data["e_trial"]
    ee = data["e_est"]
    en = data.get("energytotal")
    ws = data.get("weight_std")
    gap = et - ee
    # After branching, e_trial = e_est - feedback*log(mean_w) with feedback=1
    # => implied post-branch mean weight
    w_implied = np.exp(-gap)

    lines = []
    lines.append(f"blocks:              {n}  (block {data['block'][0]} … {data['block'][-1]})")
    lines.append("")
    lines.append("Block-averaged weight (pre-branch average stored in HDF5 'weight'):")
    lines.append(f"  start / mid / end:  {w[0]:.4g} / {w[n//2]:.4g} / {w[-1]:.4g}")
    lines.append(f"  last {late}: mean={w[sl].mean():.4g}  min={w[sl].min():.4g}  max={w[sl].max():.4g}")
    if ws is not None:
        lines.append(
            f"  weight_std last {late}: mean={ws[sl].mean():.4g}  "
            f"(large => few walkers dominate)"
        )

    if "weights" in data:
        ww = data["weights"]
        lines.append("")
        lines.append("Current per-walker weights (last saved snapshot after branching):")
        lines.append(
            f"  mean={ww.mean():.4g}  std={ww.std():.4g}  "
            f"min={ww.min():.4g}  max={ww.max():.4g}"
        )
        lines.append(f"  unique values≈{len(np.unique(np.round(ww, 12)))}  (branching equalizes)")

    lines.append("")
    lines.append("Energies (Ha):")
    lines.append(f"  e_est   start→end: {ee[0]:.6f} → {ee[-1]:.6f}   last{late} mean={ee[sl].mean():.6f}")
    lines.append(f"  e_trial start→end: {et[0]:.6f} → {et[-1]:.6f}   last{late} mean={et[sl].mean():.6f}")
    if en is not None:
        lines.append(
            f"  E_total start→end: {en[0]:.6f} → {en[-1]:.6f}   last{late} mean={en[sl].mean():.6f}"
        )
    lines.append(
        f"  e_trial−e_est last{late}: mean={gap[sl].mean():.6f}  "
        f"(≈ −log⟨w_post⟩ if feedback=1)"
    )
    lines.append(
        f"  implied ⟨w⟩_post last{late}: mean={w_implied[sl].mean():.4g}  "
        f"(healthy ≈ 1)"
    )

    if "Number of walkers killed" in data:
        killed = data["Number of walkers killed"]
        mb = data.get("max branches")
        lines.append("")
        lines.append("Branching:")
        lines.append(
            f"  killed last{late}: mean={killed[sl].mean():.1f}  "
            f"max={killed[sl].max()}"
        )
        if mb is not None:
            lines.append(f"  max branches last{late}: mean={mb[sl].mean():.2f}  max={mb[sl].max()}")

    # Health flags
    lines.append("")
    lines.append("Health flags:")
    flags = []
    w_late = w[sl].mean()
    wi_late = w_implied[sl].mean()
    if w_late > 10 or w_late < 0.1:
        flags.append(f"FAIL  block weight far from 1 (mean last{late}={w_late:.3g})")
    elif w_late > 3 or w_late < 1 / 3:
        flags.append(f"WARN  block weight drifting (mean last{late}={w_late:.3g})")
    else:
        flags.append(f"OK    block weight near 1 (mean last{late}={w_late:.3g})")

    if wi_late > 3 or wi_late < 1 / 3:
        flags.append(
            f"WARN  implied post-branch ⟨w⟩={wi_late:.3g} "
            f"=> e_trial offset {gap[sl].mean():+.3f} Ha from e_est"
        )
    else:
        flags.append(f"OK    implied post-branch ⟨w⟩≈{wi_late:.3g}")

    if en is not None:
        # early vs late energy shift
        early = en[: max(1, n // 10)].mean()
        late_e = en[sl].mean()
        if abs(late_e - early) > 1.0:
            flags.append(
                f"WARN  energytotal still shifting "
                f"(early≈{early:.3f} → late≈{late_e:.3f}, Δ={late_e-early:+.3f} Ha)"
            )
        else:
            flags.append(f"OK    energytotal fairly flat late (Δ early→late={late_e-early:+.3f} Ha)")

    if ws is not None and ws[sl].mean() > 0.5 * max(w_late, 1e-12):
        flags.append(
            f"WARN  weight_std large vs mean "
            f"(std/mean≈{ws[sl].mean()/max(w_late,1e-12):.2f}) — effective sample size low"
        )

    for fl in flags:
        lines.append(f"  {fl}")

    lines.append("")
    lines.append("Reminder: e_trial = e_est − feedback·log⟨w⟩ (feedback=1).")
    lines.append("Until ⟨w⟩~1 and E is flat, discard blocks for production averages.")
    return "\n".join(lines)


def plot(data: dict, out: str, title: str) -> None:
    block = data["block"]
    w = data["weight"]
    et = data["e_trial"]
    ee = data["e_est"]
    gap = et - ee
    w_implied = np.exp(-gap)

    nrows = 4
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.0 * nrows), sharex=True)

    ax = axes[0]
    ax.semilogy(block, np.maximum(w, 1e-30), "-", lw=1.2, label="weight (block avg)")
    if "weight_std" in data:
        ax.semilogy(block, np.maximum(data["weight_std"], 1e-30), "-", lw=1.0, alpha=0.8, label="weight_std")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("weight")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(block, ee, "-", lw=1.2, label="e_est")
    ax.plot(block, et, "-", lw=1.2, label="e_trial")
    if "energytotal" in data:
        ax.plot(block, data["energytotal"], "-", lw=1.0, alpha=0.8, label="energytotal")
    ax.set_ylabel("energy (Ha)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(block, gap, "-", lw=1.2, label="e_trial − e_est")
    ax.plot(block, -np.log(np.maximum(w_implied, 1e-30)), "--", lw=1.0, alpha=0.0)  # noop keep style
    ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("e_trial − e_est (Ha)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.semilogy(block, np.maximum(w_implied, 1e-30), "-", lw=1.2, label="implied ⟨w⟩_post = exp(-(e_trial-e_est))")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    if "Number of walkers killed" in data:
        ax2 = ax.twinx()
        ax2.plot(
            block,
            data["Number of walkers killed"],
            color="C3",
            alpha=0.5,
            lw=0.8,
            label="killed",
        )
        ax2.set_ylabel("walkers killed", color="C3")
    ax.set_xlabel("block")
    ax.set_ylabel("implied ⟨w⟩")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdf5", nargs="?", help="ABDMC HDF5 file (default: first known name in cwd)")
    p.add_argument("--out", default="weights_diag.png", help="output figure path")
    p.add_argument("--late", type=int, default=100, help="blocks for 'late' statistics")
    p.add_argument("--no-plot", action="store_true", help="print only, skip figure")
    args = p.parse_args(argv)

    path = args.hdf5 or find_default_hdf5(os.getcwd())
    if path is None:
        print("No HDF5 given and no default candidate found in cwd.", file=sys.stderr)
        print("Candidates:", ", ".join(DEFAULT_CANDIDATES), file=sys.stderr)
        return 1
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    data = load(path)
    report = summarize(data, late=args.late)
    print(f"File: {path}")
    print(report)

    txt_out = os.path.splitext(args.out)[0] + ".txt"
    with open(txt_out, "w") as fh:
        fh.write(f"File: {path}\n")
        fh.write(report)
        fh.write("\n")
    print(f"\nWrote {txt_out}")

    if not args.no_plot:
        plot(data, args.out, title=os.path.basename(path))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
