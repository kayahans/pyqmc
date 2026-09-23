#!/usr/bin/env python3
import glob
import os
import shutil
import tempfile
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_hdf5_snapshot(path, max_retries=8, retry_delay_s=0.25):
    """Open a *copy* of ``path`` so we do not read the live file while QMC appends to it.

    Copying can still occasionally fail if it runs mid-flush; we retry with a short backoff.
    For true concurrent read-while-write without copying, the writer must use HDF5 SWMR
    (single writer multiple reader); then you could open the original with ``swmr=True``.
    """
    last_err = None
    for attempt in range(max_retries):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".hdf5", prefix="plot_warmup_")
            os.close(fd)
            shutil.copy2(path, tmp)
            with h5py.File(tmp, "r") as f:
                data = {k: f[k][:] for k in ("energytotal", "acceptance")}
            os.remove(tmp)
            return data
        except (OSError, RuntimeError, KeyError) as e:
            last_err = e
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(retry_delay_s * (1.5**attempt))
    raise last_err


def reblocked_stats_after_discard(data, discard_fraction=0.2, block_size=None):
    n_steps = len(data)
    discard_steps = int(discard_fraction * n_steps)
    trimmed = data[discard_steps:]

    if len(trimmed) == 0:
        return np.mean(data), np.std(data)

    if block_size is None:
        block_size = max(1, int(np.sqrt(len(trimmed))))
    block_size = min(block_size, len(trimmed))
    n_blocks = len(trimmed) // block_size

    if n_blocks < 1:
        return np.mean(trimmed), np.std(trimmed)

    block_data = trimmed[: n_blocks * block_size].reshape(n_blocks, block_size)
    block_means = np.mean(block_data, axis=1)
    block_std = np.std(block_means, ddof=1) if len(block_means) > 1 else 0.0
    return np.mean(block_means), block_std


try:
    dmc_file = glob.glob('*dmc_eq_cas*.hdf5')[0]
except:
    dmc_file = None
filenames = {'dmc': dmc_file, 'vmc': 'warmup_vmc.hdf5'}

for key, filename in filenames.items():
    if filename is None:
        continue
    # Read a snapshot copy so we are not fighting the running job appending to the same file
    try:
        snap = read_hdf5_snapshot(filename)
        total_energy = snap["energytotal"]
        acceptance = snap["acceptance"]
        steps = np.arange(len(total_energy))
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        continue

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot energy on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Energy (Ha)', color=color1)
    line1 = ax1.plot(steps, total_energy, color=color1, label='Total Energy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # Add dashed line for final energy
    # Compute mean energy, discarding first 20 percent of steps ("warmup")
    mean_energy, std_energy = reblocked_stats_after_discard(total_energy, discard_fraction=0.2)
    print(f"{key.upper()} Mean energy: {mean_energy:.6f} ± {std_energy:.6f} Ha")
    ax1.axhline(y=mean_energy, color=color1, linestyle='--', alpha=0.5)
    ax1.annotate(f'{key.upper()} Mean: {mean_energy:.6f} ± {std_energy:.6f} Ha',
                xy=(steps[-1], mean_energy),
                xytext=(steps[-1]*0.7, mean_energy+0.1),
                arrowprops=dict(facecolor=color1, shrink=0.05, width=1, headwidth=5, headlength=5, alpha=0.5))

    # Create second y-axis for acceptance
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Acceptance Rate', color=color2)
    line2 = ax2.plot(steps, acceptance, color=color2, label='Acceptance Rate')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add dashed line for final acceptance
    final_acceptance = acceptance[-1]
    ax2.axhline(y=final_acceptance, color=color2, linestyle='--', alpha=0.5)
    ax2.annotate(f'Final Acceptance: {final_acceptance:.2f}',
                xy=(steps[-1], final_acceptance),
                xytext=(steps[-1]*0.7, final_acceptance+0.1),
                arrowprops=dict(facecolor=color2, shrink=0.05, width=1, headwidth=5, headlength=5, alpha=0.5))
    ax2.set_ylim(0, 1)
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Energy and Acceptance Rate During Warmup')

    # Save the plot
    plt.savefig(f'warmup_energy_trace_{key}.png', dpi=300, bbox_inches='tight')
    plt.close()
