import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import constants

from typing import Tuple, List, Dict
from matplotlib import mlab
from utils import generate_ou_input

# FIG_SIZE = [20, 15]
FIG_SIZE = [8, 5]
FIG_SIZE_QUADRATIC = [8, 6]


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

    ax.plot(mean + sigma, c="black", linewidth=0.5)

    plt.tight_layout()

    _save_to_file("noise", save=save, folder=prefix)

    return fig, ax


def summed_voltage(model: dict, title: str = "Summed Voltage", dt: float = 1.0, duration: int = None,
                   prefix: str = None, save: bool = False, skip: int = None, excitatory: bool = False,
                   population: int = 1):
    duration = duration if duration else model["params"]["runtime"]

    if excitatory:
        key = 'excitatory'
    else:
        key = 'inhibitory'

    lfp1, lfp2 = calculate_local_field_potentials(model, duration, excitatory, skip, population=population)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")
    ax.plot(t, lfp1, '0.75')
    ax.plot(t, lfp2, '0.25')

    plt.tight_layout()

    _save_to_file("summed_voltage", save, key, prefix)

    return fig, ax


def psd(title: str, model: dict, duration: int = None, dt: float = 1.0, folder: str = None, save: bool = False,
        population: int = 1, excitatory: bool = False, skip: int = None, fig_size: tuple = None):
    """
    Plots the Power Spectral Density.
    """
    print("Generate PSD plot ...")

    duration = duration if duration else model["params"]["runtime"]
    if excitatory:
        key = 'excitatory'
    else:
        key = 'inhibitory'

    lfp1, lfp2 = calculate_local_field_potentials(model, duration, excitatory, skip, population=population)

    timepoints = int((duration / dt) / 2)
    fs = 1. / dt

    psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)

    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Density")
    ax.plot(freqs * 1000, psd1, '0.25', linewidth=3.0, c='darkgray')
    ax.plot(freqs * 1000, psd2, '0.75', linewidth=3.0, c='dimgray')
    ax.set_xlim([0, 80])

    _save_to_file("psd", save, key, folder)

    return fig, ax


def raster(model: dict, title: str = None, x_left: int = None, x_right: int = None, save: bool = False, key: str = "",
           folder: str = None, population: int = 1, fig_size: Tuple = None, ax=None):
    fig = None
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)
        ax = fig.add_subplot(111)

    if population == 1:
        s_e = model['model_results']['net']['net_spikes_e']
        s_i = model['model_results']['net']['net_spikes_i1']

    else:
        s_e = model['model_results']['net']['net_spikes_e2']
        s_i = model['model_results']['net']['net_spikes_i2']

    ax.set_title(title if title else "Raster")
    ax.set_xlabel('Time in ms')
    ax.set_ylabel('Neuron index')
    ax.plot(s_e[1] * 1000, s_e[0], 'k.', c='darkgray', markersize="2.0")
    ax.plot(s_i[1] * 1000, s_i[0] + (s_e[0].max() + 1), 'k.', c='black', markersize="2.0")

    # TODO: plot complete time axis, currently only x-ticks for available data is plotted
    ax.set_xlim(left=x_left, right=x_right)

    plt.tight_layout()
    _save_to_file("raster", save=save, key=key, folder=folder)
    return fig, ax


def population_rates(model: dict):
    """
    Plots the smoothed population rates for excitatory and inhibitory groups of both populations.

    :param model: model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    N_pop = model["params"]["N_pop"]

    r_e = model['model_results']['net']['r_e']
    r_i1 = model['model_results']['net']['r_i1']

    axs[0, 0].plot(r_e, c="black")
    axs[0, 0].set_title("Population 1 - Excitatory")

    axs[1, 0].plot(r_i1, c="grey")
    axs[1, 0].set_title("Population 1 - Inhibitory")

    if N_pop > 1:
        r_e2 = model['model_results']['net']['r_e2']
        r_i2 = model['model_results']['net']['r_i2']

        axs[0, 1].plot(r_e2, c="black")
        axs[0, 1].set_title("Population 2 - Excitatory")

        axs[1, 1].plot(r_i2, c="grey")
        axs[1, 1].set_title("Population 2 - Inhibitory")


def calculate_local_field_potentials(data: dict, duration: int = None, excitatory: bool = False, skip: int = None,
                                     population: int = 1) -> Tuple:
    if duration:
        duration = int(duration)

    N_e = data['params']['N_e']
    N_i = data['params']['N_i']

    if population == 1:
        v_e = data['model_results']['net']['v_all_neurons_e'][:skip][:duration]
        v_i = data['model_results']['net']['v_all_neurons_i1'][:skip][:duration]
        lfp1 = np.sum(v_e, axis=0) / N_e
        lfp2 = np.sum(v_i, axis=0) / N_i
        return lfp1, lfp2

    elif population == 2:
        v_e = data['model_results']['net']['v_all_neurons_e2'][:skip][:duration]
        v_i = data['model_results']['net']['v_all_neurons_i2'][:skip][:duration]
        lfp1 = np.sum(v_e, axis=0) / N_e
        lfp2 = np.sum(v_i, axis=0) / N_i
        return lfp1, lfp2

    if excitatory:
        v_e1 = data['model_results']['net']['v_all_neurons_e'][:skip][:duration]
        v_e2 = data['model_results']['net']['v_all_neurons_e2'][:skip][:duration]

        lfp1 = np.sum(v_e1, axis=0) / N_e
        lfp2 = np.sum(v_e2, axis=0) / N_e
    else:
        v_i1 = data['model_results']['net']['v_all_neurons_i1'][:skip][:duration]
        v_i2 = data['model_results']['net']['v_all_neurons_i2'][:skip][:duration]

        lfp1 = np.sum(v_i1, axis=0) / N_i
        lfp2 = np.sum(v_i2, axis=0) / N_i

    return lfp1, lfp2


def all_psd(models: List[Dict], n_cols, n_rows):
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

            duration = data["params"]["runtime"]
            dt = 1.0

            lfp1, lfp2 = calculate_local_field_potentials(data=data, duration=duration)

            timepoints = int((duration / dt) / 2)
            fs = 1. / dt

            psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
            psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

            psd1[0] = 0.0
            psd2[0] = 0.0

            ax.set_xlabel("Frequency")
            ax.set_ylabel("Density")
            ax.plot(freqs * 1000, psd1, '0.25', linewidth=1.0, c='black')
            ax.plot(freqs * 1000, psd2, '0.75', linewidth=1.0, c='dimgray')
            ax.set_xlim([0, 80])

    plt.tight_layout()
    return fig, axs


def _save_to_file(name: str, save: bool, key: str = None, folder: str = None):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base
        plt.savefig(f"{constants.PLOTS_PATH}/{fname}")


def ou_noise_by_params(params: dict):
    mean = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_mu'])
    sigma = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_sigma'])
    return noise(mean, sigma, save=False)


def heat_map(models: List[Dict], x: str = "mean", y: str = "sigma", metric: str = "bandpower", **kwargs):
    """
    Plots heat map for noise experiment.

    Setup:
        x: sigma
        y: mean
        z: amplitude
    """
    data = _prepare_data(metric, models, x, y)
    fig = plt.figure(figsize=FIG_SIZE_QUADRATIC)

    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by=[x, y])

    heatmap_data = pd.pivot_table(df,
                                  values=metric,
                                  index=[x],
                                  columns=y)

    ax = sns.heatmap(heatmap_data, **kwargs)

    return fig, ax, df


def _prepare_data(metric: str, models: [dict], x: str, y: str):
    data = {
        x: [],
        y: [],
        metric: []
    }

    for model in models:
        # x: mean
        # y: sigma
        # z: max_amplitude
        mean_ = model["params"]["ou_mu"]["ou_mean"]
        sigma_ = model["params"]["ou_mu"]["ou_sigma"]
        tau_ = model["params"]["ou_mu"]["ou_tau"]

        max_amplitude, peak_freq = band_power(model)

        if x == 'tau':
            data[x].append(tau_)
        elif x == 'mean':
            data[x].append(mean_)
        else:
            data[x].append(tau_)

        data[y].append(sigma_)

        if metric == "bandpower":
            value = max_amplitude
        elif metric == "freq":
            value = peak_freq
        else:
            value = max_amplitude

        data[metric].append(value)

    return data


def band_power(model):
    lfp1, lfp2 = calculate_local_field_potentials(model)

    runtime_ = model["params"]["runtime"]
    dt = 1.0
    timepoints = int((runtime_ / dt) / 2)
    fs = 1. / dt

    psd, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
    psd[0] = 0.0
    freqs = freqs * 1000
    freqs = [int(freq) for freq in freqs]

    max_amplitude = psd.max()
    peak_freq = freqs[psd.argmax()]

    return max_amplitude, peak_freq
