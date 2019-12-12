from collections import Iterator
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab

# FIG_SIZE = [20, 15]
FIG_SIZE = [10, 8]


def noise(mean, sigma, save: bool = True, prefix: str = None, decompose: bool = False, skip: int = None,
          duration: int = None):
    """ Plot External Noise. """
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.set_title("External Input Signal to population")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Voltage in mV")

    if duration or skip:
        mean = mean[skip:duration + skip]
        sigma = sigma[skip:duration + skip]

    if decompose:
        ax.plot(mean)
        ax.plot(sigma)

    ax.plot(mean + sigma)

    _save_to_file("noise", save=save, folder=prefix)


def summed_voltage(title: str, data: dict, duration, dt, prefix: str = None, save: bool = True,
                   excitatory: bool = False):
    skip = 0

    if excitatory:
        key = 'excitatory'
    else:
        key = 'inhibitory'

    lfp1, lfp2 = _calculate_local_field_potentials(data, duration, excitatory, skip)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")
    ax.plot(t, lfp1, '0.75')
    ax.plot(t, lfp2, '0.25')

    _save_to_file("summed_voltage", save, key, prefix)


def psd(title: str, model: dict, duration: int = None, dt: float = 1.0, folder: str = None, save: bool = True,
        excitatory: bool = False):
    """
    Plots the Power Spectral Density.
    """
    print("Generate PSD plot ...")
    skip = 0

    duration = duration if duration else model["params"]["runtime"]

    if excitatory:
        key = 'excitatory'
    else:
        key = 'inhibitory'

    lfp1, lfp2 = _calculate_local_field_potentials(model, duration, excitatory, skip)

    timepoints = int((duration / dt) / 2)
    fs = 1. / dt

    psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=FIG_SIZE)

    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Density")
    ax.plot(freqs * 1000, psd1, '0.25', linewidth=3.0, c='darkgray')
    ax.plot(freqs * 1000, psd2, '0.75', linewidth=3.0, c='dimgray')
    ax.set_xlim([0, 80])

    _save_to_file("psd", save, key, folder)


def raster(data: dict, x_left: int = None, x_right: int = None, save: bool = True, key: str = "", folder: str = None,
           population: int = 1, fig_size: Tuple = None):
    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)
    ax = fig.add_subplot(111)

    if population == 1:
        s_e = data['model_results']['net']['net_spikes_e']
        s_i = data['model_results']['net']['net_spikes_i1']

    else:
        s_e = data['model_results']['net']['net_spikes_e2']
        s_i = data['model_results']['net']['net_spikes_i2']

    ax.set_title("Raster Plot")

    ax.set_xlabel('Time in ms')
    ax.set_ylabel('Neuron index')

    ax.plot(s_e[1] * 1000, s_e[0], 'k.', c='darkgray', markersize="4")
    ax.plot(s_i[1] * 1000, s_i[0] + (s_e[0].max() + 1), 'k.', c='dimgray', markersize="4")

    ax.set_xlim(left=x_left, right=x_right)

    _save_to_file("raster", save=save, key=key, folder=folder)


def population_rates(model: dict):
    """
    Plots the smoothed population rates.

    @param model: model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    r_e = model['model_results']['net']['r_e']
    r_i1 = model['model_results']['net']['r_i1']
    r_e2 = model['model_results']['net']['r_e2']
    r_i2 = model['model_results']['net']['r_i2']

    axs[0, 0].plot(r_e, c="black")
    axs[0, 0].set_title("Population 1 - Excitatory")

    axs[0, 1].plot(r_i1, c="grey")
    axs[0, 1].set_title("Population 1 - Inhibitory")

    axs[1, 0].plot(r_e2, c="black")
    axs[1, 0].set_title("Population 2 - Excitatory")

    axs[1, 1].plot(r_i2, c="grey")
    axs[1, 1].set_title("Population 2 - Inhibitory")


def _calculate_local_field_potentials(data, duration, excitatory, skip):
    if excitatory:
        N_e = data['params']['N_e']

        v_e1 = data['model_results']['net']['v_all_neurons_e'][:, skip:duration + skip]
        v_e2 = data['model_results']['net']['v_all_neurons_e2'][:, skip:duration + skip]

        lfp1 = np.sum(v_e1, axis=0) / N_e
        lfp2 = np.sum(v_e2, axis=0) / N_e
    else:
        N_i = data['params']['N_i']

        v_i1 = data['model_results']['net']['v_all_neurons_i1'][:, skip:duration + skip]
        v_i2 = data['model_results']['net']['v_all_neurons_i2'][:, skip:duration + skip]

        lfp1 = np.sum(v_i1, axis=0) / N_i
        lfp2 = np.sum(v_i2, axis=0) / N_i
    return lfp1, lfp2


def all_psd(models: Iterator, n_cols, n_rows):
    models = list(models)

    # TODO: use ration and define columns and rows depending on length of models
    fig, axs = plt.subplots(n_cols, n_rows, figsize=(20, 5), sharex=True)

    for i, ax in enumerate(axs.reshape(-1)):
        try:
            title, data = models[i]
        except (KeyError, IndexError):
            # ignore
            pass
        else:
            ax.set_title(f"mean = {title}", fontsize=10)

            duration = 300
            excitatory = True
            skip = 0
            dt = 1.0

            lfp1, lfp2 = _calculate_local_field_potentials(data, duration, excitatory, skip)

            timepoints = int((duration / dt) / 2)
            fs = 1. / dt

            psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
            psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

            psd1[0] = 0.0
            psd2[0] = 0.0

            ax.set_xlabel("Frequency")
            ax.set_ylabel("Density")
            ax.plot(freqs * 1000, psd1, '0.25', linewidth=3.0, c='darkgray')
            ax.plot(freqs * 1000, psd2, '0.75', linewidth=3.0, c='dimgray')
            ax.set_xlim([0, 80])

    plt.tight_layout()
    plt.show()


def _save_to_file(name: str, save: bool, key: str = None, folder: str = None):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base
        plt.savefig(f"plots/{fname}")
