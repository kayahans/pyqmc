#!/usr/bin/env python3
"""
Basis-set extrapolation of MO energies from the ``out`` log (MO Energies Comparison).

Default mode uses the exponential CBS model from ``basis_set_extrapolation.ipynb``:
    ε(X) = E0 + α * exp(-β X)
with X = 2, 3, 4, … in file order (DZ, TZ, QZ, 5Z, …: ``X = 2 + index`` for each basis
column to the right). If aug-cc-pV5Z columns are present, the fit uses that fourth point
as well (X = 5 for the fourth rung). E0 is reported as CBS (α) / CBS (β).

``--pair-diff`` uses a single-point estimate: the best MO energies are taken from the last
``(α)``/``(β)`` pair in the table (highest basis / cardinal), and
``CBS-ε(DZ)`` = ε(highest) - ε(DZ) using the first pair as double-zeta.

The MO block is parsed from the header row: any number of basis columns (pairs of
``(α)``/``(β)``) is supported. Optional ``--plot`` for exponential or pair style.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

_SPIN_LABEL_RE = re.compile(r"(?:\(α\)|\(β\)|\(alpha\)|\(beta\))")


def _find_mo_energies_header(text_from_block: str) -> tuple[str, int, int] | None:
    """
    Locate the MO # header line and the number of basis sets.

    Basis names in ``out`` are often glued to the next column, e.g.
    ``ccecp-aug-cc-pvdz (β)ccecp-aug-cc-pvtz (α)``, so spin labels are counted
    with a regex rather than whitespace token matching.

    Returns (header_line, n_basis, line_index_in_sub) or None.
    """
    lines = text_from_block.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("MO #"):
            continue
        parts = s.split()
        if len(parts) < 2 or parts[0] != "MO" or parts[1] != "#":
            continue
        n_ener = len(_SPIN_LABEL_RE.findall(s))
        if n_ener < 2 or n_ener % 2:
            continue
        n_basis = n_ener // 2
        return (line.rstrip("\n"), n_basis, i)
    return None


def _parse_energy_tok(tok: str) -> float | None:
    t = tok.strip()
    if t.upper() == "N/A" or t == "nan":
        return None
    return float(t)


def parse_mo_energies_block(path: Path) -> tuple[list[dict[str, Any]], int, str]:
    """
    Parse the MO Energies Comparison table from ``out``.

    Returns ``(data_rows, n_pairs, header_line)``. Each row has ``mo``, ``E`` (ordered
    α/β, α/β, … for each basis from left to right), and ``alpha_DZ``/``beta_DZ`` for
    the first (double-zeta) pair for compatibility.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("MO Energies Comparison")
    if start < 0:
        raise ValueError(f"Could not find 'MO Energies Comparison' in {path}")

    sub = text[start:]
    found = _find_mo_energies_header(sub)
    if not found:
        raise ValueError("Could not locate MO energies header row (expected 'MO #' with (α)/(β) columns)")
    header_line, n_pairs, header_idx = found
    n_ener = 2 * n_pairs

    data_rows: list[dict[str, Any]] = []
    lines = sub.splitlines()
    for line in lines[header_idx + 2 :]:
        if not line.strip() or line.strip().startswith("="):
            break
        parts = line.split()
        if len(parts) < 1 + n_ener:
            continue
        try:
            mo = int(parts[0])
        except ValueError:
            continue
        e: list[float | None] = []
        for j in range(n_ener):
            e.append(_parse_energy_tok(parts[1 + j]))
        row: dict[str, Any] = {
            "mo": mo,
            "E": e,
        }
        row["alpha_DZ"], row["beta_DZ"] = (e[0], e[1]) if n_pairs else (None, None)
        if n_pairs >= 2:
            row["alpha_TZ"] = e[2]
            row["beta_TZ"] = e[3]
        if n_pairs >= 3:
            row["alpha_QZ"] = e[4]
            row["beta_QZ"] = e[5]
        if n_pairs >= 4:
            row["alpha_5Z"] = e[6]
            row["beta_5Z"] = e[7]
        data_rows.append(row)
    return (data_rows, n_pairs, header_line)


def read_mo_header_line(path: Path) -> str:
    """Return the original MO # header line from ``out`` (without trailing newline)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("MO Energies Comparison")
    if start < 0:
        raise ValueError(f"Could not find 'MO Energies Comparison' in {path}")
    found = _find_mo_energies_header(text[start:])
    if not found:
        raise ValueError("MO energies header row not found")
    return found[0]


def collect_xy_for_channel(
    row: dict[str, Any], spin: str, n_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect all available basis-pair columns for the exponential model: the *i*th rung
    (0-based, left to right) uses X = 2 + *i* (2 = DZ, 3 = TZ, 4 = QZ, 5 = 5Z, …). Any
    missing/NaN point is skipped; the model still needs at least three remaining points.
    """
    e = row.get("E")
    if e is None or n_pairs < 1:
        return np.array([]), np.array([])
    offset = 0 if spin == "alpha" else 1
    xs: list[int] = []
    es: list[float] = []
    for i in range(n_pairs):
        v = e[2 * i + offset]
        if v is not None:
            xs.append(2 + i)
            es.append(float(v))
    return np.array(xs, dtype=float), np.array(es, dtype=float)


def extrapolate_exponential(
    x: np.ndarray, epsilon: np.ndarray
) -> dict[str, Any]:
    if len(x) < 3:
        return {
            "ok": False,
            "reason": f"exponential model needs >=3 points, got {len(x)}",
        }

    def resid_vec(p: np.ndarray) -> np.ndarray:
        e0, alpha, beta = p
        return np.array(
            [
                e0 - float(eps) + alpha * np.exp(-beta * float(xi))
                for xi, eps in zip(x, epsilon)
            ],
            dtype=float,
        )

    e0_0 = float(np.mean(epsilon))
    p0 = np.array([e0_0, 0.05, 0.5], dtype=float)
    ls = least_squares(
        resid_vec,
        p0,
        bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
        verbose=0,
    )
    e0, alpha, beta = (float(ls.x[0]), float(ls.x[1]), float(ls.x[2]))
    cost = float(np.sum(ls.fun**2))
    return {
        "ok": True,
        "E0_CBS": e0,
        "alpha": alpha,
        "beta": beta,
        "residual_sum_sq": cost,
        "X_used": x.copy(),
        "epsilon_used": epsilon.copy(),
    }


def mo_prefix_width(mo: int) -> int:
    """Match ``out`` layout for the MO label + spaces before the first energy column."""
    if mo == 1:
        return 16
    if mo < 10:
        return 17
    if mo < 100:
        return 18
    return 19


def format_energy_cell(v: float | None) -> str:
    if v is None:
        return "N/A".ljust(20)
    return f"{v:.6f}".ljust(20)


def format_cbs_cell(fit: dict[str, Any]) -> str:
    if not fit.get("ok"):
        return "N/A".ljust(20)
    return f"{float(fit['E0_CBS']):.6f}".ljust(20)


def format_cbs_minus_dz_cell(fit: dict[str, Any], eps_dz: float | None) -> str:
    """Additive correction to the double-zeta column: CBS - ε(DZ)."""
    if not fit.get("ok") or eps_dz is None:
        return "N/A".ljust(20)
    d = float(fit["E0_CBS"]) - float(eps_dz)
    return f"{d:.6f}".ljust(20)


def format_data_line(
    mo: int,
    row: dict[str, Any],
    n_pairs: int,
    fit_a: dict,
    fit_b: dict,
) -> str:
    nw = mo_prefix_width(mo)
    prefix = str(mo) + " " * (nw - len(str(mo)))
    parts = [prefix]
    e = row.get("E") or []
    for i in range(n_pairs):
        a = e[2 * i] if 2 * i < len(e) else None
        b = e[2 * i + 1] if 2 * i + 1 < len(e) else None
        parts.append(format_energy_cell(a))
        parts.append(format_energy_cell(b))
    parts.append(format_cbs_cell(fit_a))
    parts.append(format_cbs_cell(fit_b))
    parts.append(format_cbs_minus_dz_cell(fit_a, row.get("alpha_DZ")))
    parts.append(format_cbs_minus_dz_cell(fit_b, row.get("beta_DZ")))
    return "".join(parts)


def build_table_text(
    header_base: str,
    rows: list[dict[str, Any]],
    n_mos: int,
    n_pairs: int,
    fits_by_mo: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    pair_diff_mode: bool = False,
) -> str:
    cbs_head = (
        "      CBS (α)           CBS (β)           "
        "CBS-ε(DZ) (α)       CBS-ε(DZ) (β)      "
    )
    full_header = header_base + cbs_head
    title = (
        "MO Energies Comparison (CBS-ε = highest - DZ, last pair = best basis)"
        if pair_diff_mode
        else "MO Energies Comparison (with exponential CBS extrapolation)"
    )
    block = [
        "=" * 70,
        title,
        "=" * 70,
        full_header,
    ]
    data_lines: list[str] = []
    for row in rows:
        mo = int(row["mo"])
        if mo > n_mos:
            break
        fa, fb = fits_by_mo[mo]
        data_lines.append(format_data_line(mo, row, n_pairs, fa, fb))
    w = max(len(full_header), max((len(s) for s in data_lines), default=0))
    sep = "-" * w
    block[3] = full_header.ljust(w)
    block.append(sep)
    block.extend(data_lines)
    block.append("=" * w)
    return "\n".join(block) + "\n"


def pair_channel_fit(e_hi: float | None) -> dict[str, Any]:
    """Single best-basis (α) or (β) value for the CBS column."""
    if e_hi is None:
        return {"ok": False, "reason": "highest-basis value missing", "E0_CBS": 0.0}
    return {"ok": True, "E0_CBS": float(e_hi)}


def compute_pair_fits(
    rows: list[dict[str, Any]], n_mos: int, n_pairs: int
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    """CBS = last (α)/(β) pair; ε(DZ) = first pair; ``CBS-ε(DZ)`` = ε(hi) - ε(DZ) per channel."""
    if n_pairs < 1:
        return {}
    out: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        mo = int(row["mo"])
        if mo > n_mos:
            break
        e = row.get("E") or []
        if len(e) < 2 * n_pairs:
            continue
        a_hi, b_hi = e[-2], e[-1]
        out[mo] = (pair_channel_fit(a_hi), pair_channel_fit(b_hi))
    return out


def compute_fits(
    rows: list[dict[str, Any]], n_mos: int, n_pairs: int
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    out: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        mo = int(row["mo"])
        if mo > n_mos:
            break
        xa, ea = collect_xy_for_channel(row, "alpha", n_pairs)
        xb, eb = collect_xy_for_channel(row, "beta", n_pairs)
        out[mo] = (extrapolate_exponential(xa, ea), extrapolate_exponential(xb, eb))
    return out


def plot_convergence(
    rows: list[dict[str, Any]],
    fits_by_mo: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    n_mos: int,
    n_pairs: int,
    path: Path,
    pair_diff: bool = False,
    dpi: int = 150,
) -> None:
    import matplotlib.pyplot as plt

    mos = [int(r["mo"]) for r in rows if int(r["mo"]) <= n_mos]
    if not mos:
        return

    n = len(mos)
    ncol = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False)
    axes_flat = np.ravel(axes)

    x_rung_max = 2.0 + max(0, n_pairs - 1) if n_pairs else 4.0
    xf = np.linspace(1.8, max(5.2, x_rung_max + 0.25), 200)

    for idx, mo in enumerate(mos):
        ax = axes_flat[idx]
        row = next(x for x in rows if int(x["mo"]) == mo)
        fa, fb = fits_by_mo[mo]
        e = row.get("E") or []

        for spin, color, marker, fit in (
            ("alpha", "C0", "o", fa),
            ("beta", "C1", "s", fb),
        ):
            if pair_diff and n_pairs >= 1:
                off = 0 if spin == "alpha" else 1
                xs_p: list[float] = []
                es_p: list[float] = []
                for k in range(n_pairs):
                    j = 2 * k + off
                    if j < len(e) and e[j] is not None:
                        xs_p.append(2.0 + k)
                        es_p.append(float(e[j]))
                if xs_p:
                    ax.plot(
                        xs_p,
                        es_p,
                        marker=marker,
                        color=color,
                        ls="-",
                        lw=0.8,
                        alpha=0.7,
                        label=f"{spin} (basis 1..{n_pairs})",
                    )
                if fit.get("ok"):
                    e0 = float(fit["E0_CBS"])
                    ax.axhline(e0, color=color, ls=":", lw=0.9, alpha=0.7, label=f"{spin} best")
            else:
                xs, es = collect_xy_for_channel(row, spin, n_pairs)
                if len(xs):
                    ax.plot(
                        xs,
                        es,
                        marker=marker,
                        color=color,
                        ls="none",
                        label=f"{spin} data",
                    )
                if fit.get("ok") and "residual_sum_sq" in fit:
                    a, b, e0 = fit["alpha"], fit["beta"], fit["E0_CBS"]
                    yf = e0 + a * np.exp(-b * xf)
                    ax.plot(xf, yf, "-", color=color, lw=1.2, alpha=0.85)
                    ax.axhline(e0, color=color, ls=":", lw=0.9, alpha=0.7)

        ax.set_title(f"MO {mo}")
        ax.set_xlabel(
            "X = 2+idx of basis (last pair = best)" if pair_diff else "X: DZ=2, TZ=3, QZ=4, 5Z=5, … (all rungs in fit)"
        )
        ax.set_ylabel("ε (Ha)")
        ax.margins(x=0.05)
        ax.legend(fontsize=7, loc="best")

    for j in range(len(mos), nrow * ncol):
        axes_flat[j].set_visible(False)

    supt = (
        "Basis-set path (dotted = best/largest-basis energy)"
        if pair_diff
        else "Basis-set convergence (markers + exponential fit, dotted = E₀ CBS)"
    )
    fig.suptitle(supt, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).cwd()
    default_out = here / "out"
    default_table = here / "mo_energies_with_cbs.txt"
    default_plot = here / "mo_cbs_convergence.png"

    p = argparse.ArgumentParser(description="CBS extrapolation of MO energies from out log")
    p.add_argument("--out", type=Path, default=default_out, help="Path to out log")
    p.add_argument(
        "-N",
        "--n-mos",
        type=int,
        default=9,
        metavar="N",
        help="Include MO indices 1 .. N (inclusive)",
    )
    p.add_argument(
        "--write",
        type=Path,
        default=default_table,
        help="Output text table path (same style as out + CBS columns)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save convergence figure (data + exponential + E₀ lines)",
    )
    p.add_argument(
        "--plot-file",
        type=Path,
        default=default_plot,
        help="Image path when --plot is set",
    )
    p.add_argument(
        "--pair-diff",
        action="store_true",
        help=(
            "Use ε(largest-basis) from the last (α),(β) columns as CBS, with "
            "CBS-ε(DZ) = that minus the first (double-zeta) pair (no exponential fit)"
        ),
    )
    args = p.parse_args()

    rows, n_pairs, header_base = parse_mo_energies_block(args.out)
    if not rows:
        raise SystemExit(f"No MO rows parsed from {args.out}")
    if n_pairs < 1:
        raise SystemExit("Header produced zero basis column pairs; check the MO # line in out")

    if args.pair_diff:
        fits = compute_pair_fits(rows, args.n_mos, n_pairs)
    else:
        fits = compute_fits(rows, args.n_mos, n_pairs)
    text = build_table_text(
        header_base,
        rows,
        args.n_mos,
        n_pairs,
        fits,
        pair_diff_mode=args.pair_diff,
    )
    args.write.write_text(text, encoding="utf-8")
    print(f"Wrote {args.write}")

    if args.plot:
        try:
            plot_convergence(
                rows,
                fits,
                args.n_mos,
                n_pairs,
                args.plot_file,
                pair_diff=args.pair_diff,
            )
        except ImportError as e:
            raise SystemExit(
                "--plot requires matplotlib (e.g. pip install matplotlib)"
            ) from e
        print(f"Wrote {args.plot_file}")


if __name__ == "__main__":
    main()

