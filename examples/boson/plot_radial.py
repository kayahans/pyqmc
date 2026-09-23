#!/usr/bin/env python
import os
import shutil
import tempfile
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_hdf5_snapshot(path, keys, max_retries=8, retry_delay_s=0.25):
    """Copy ``path`` to a temp file, then read datasets (avoids reading a live appended HDF5)."""
    last_err = None
    for attempt in range(max_retries):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".hdf5", prefix="plot_radial_")
            os.close(fd)
            shutil.copy2(path, tmp)
            with h5py.File(tmp, "r") as f:
                print(list(f.keys()))
                data = {k: f[k][:] for k in keys}
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


_SNAPSHOT_KEYS = (
    "radial_densityradial_density",
    "radial_densityint_density",
    "energytotal",
    "radial_densityr",
    "acceptance",
)

# import glob
# files = sorted(glob.glob('warmup_vmc.hdf5'))
files = ['warmup_vmc.hdf5']

if len(files) == 0:
    print('No files found')
    exit()

# squeeze=False: single column stays shape (4, 1), not a 1D length-4 array (so axs[row, col] always works)
fig, axs = plt.subplots(4, len(files), figsize=(20, 10), squeeze=False)
plt.suptitle('Radial Density and Energy at Equilibration')
plt_idx = 0
emin = None
emax = None

for file in files:
    try:
        snap = read_hdf5_snapshot(file, _SNAPSHOT_KEYS)
        y = snap["radial_densityradial_density"].T
        y2 = snap["radial_densityint_density"].T
        e = snap["energytotal"]
        r = snap["radial_densityr"]
        acc = snap["acceptance"]
        print(file + " is read successfully")
        # Combined two figures with subplots
        # Plot the radial density
        interval = int(y.shape[1]/5)
        for i in range(y.shape[1]):
            if i % interval == 0:
                axs[0, plt_idx].plot(r[0], y[:, i], label=f'{i}')
        # axs[0].set_xscale('log')
        axs[0, plt_idx].legend(title='VMC Step Number')
        axs[0, plt_idx].set_title('Radial Density')


        for i in range(y2.shape[1]):
            if i % interval == 0:
                running_average_interval = 5
                running_average = np.convolve(y2[:, i], np.ones(running_average_interval)/running_average_interval, mode='valid')
                axs[1, plt_idx].plot(r[0][running_average_interval-1:], running_average, label=f'{i}')
                # axs[1, plt_idx].plot(r[0], y2[:, i], label=f'{i}')
        # axs[1].set_xscale('log')
        axs[1, plt_idx].legend(title='VMC Step Number')
        axs[1, plt_idx].set_title('Integrated Density')
        axs[1, plt_idx].grid()
        axs[1, plt_idx].set_ylim(0, 0.2)


        axs[2, plt_idx].plot(e)
        axs[2, plt_idx].set_title('Energy')
        axs[2, plt_idx].grid()

        if emin is None:
            emin = np.min(e)
        if emax is None:
            emax = np.max(e)
        axs[2, plt_idx].set_ylim(emin-0.1, emax+0.1)
        axs[2, plt_idx].axhline(emin, color='red', linestyle='--')

        axs[3, plt_idx].plot(acc)
        axs[3, plt_idx].set_title('Acceptance')
        axs[3, plt_idx].grid()
        plt_idx += 1
    except Exception as ex:
        print(f"{file} could not be plotted: {ex}")
plt.savefig('vmc_radial_density_equilibration.png')
# plt.show()
