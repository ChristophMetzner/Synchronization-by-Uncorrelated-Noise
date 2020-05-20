import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

import constants
import processing

from typing import Tuple, List, Dict
from matplotlib import mlab

from utils import generate_ou_input

# FIG_SIZE = [20, 15]
FIG_SIZE = [10, 6]
FIG_SIZE_QUADRATIC = [8, 6]
FIG_SIZE_PSD = [8, 3]


def noise(
    mean,
    sigma,
    save: bool = True,
    prefix: str = None,
    decompose: bool = False,
    skip: int = None,
    duration: int = None,
    fig_size: bool = None,
):
    """ Plot External Noise. """
    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.set_title("External Input Signal to Population")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Voltage in mV")

    if duration or skip:
        mean = mean[skip : duration + skip]
        sigma = sigma[skip : duration + skip]

    if decompose:
        ax.plot(mean)
        ax.plot(sigma)

    ax.plot(mean + sigma, c="black", linewidth=0.5)

    plt.tight_layout()

    _save_to_file("noise", save=save, folder=prefix)

    return fig, ax


def poisson_input(model: dict):
    if (
        "poisson_input_t_e" in model["model_results"]["net"]
        and "poisson_input_spikes_e" in model["model_results"]["net"]
    ):
        net = model["model_results"]["net"]
        plt.title("Poisson Spike Train Input to E population")
        plt.xlabel("Time in ms")
        plt.ylabel("Neuron Index of Poisson Group")
        plt.plot(
            net["poisson_input_t_e"], net["poisson_input_spikes_e"], ".", c="black"
        )
    else:
        return None


def lfp(
    model: dict,
    title: str = "Summed Voltage",
    dt: float = 1.0,
    duration: int = None,
    prefix: str = None,
    save: bool = False,
    skip: int = None,
    population: int = 1,
):
    duration = duration if duration else model["params"]["runtime"]

    lfp1, lfp2 = processing.lfp(model, duration, skip, population=population)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")

    ax.plot(t, lfp1, "0.75", c="black")
    ax.plot(t, lfp2, "0.25", c="darkgrey")

    plt.legend(
        handles=[
            mpatches.Patch(color="black", label="Excitatory Group"),
            mpatches.Patch(color="darkgrey", label="Inhibitory Group"),
        ]
    )

    plt.tight_layout()

    _save_to_file("summed_voltage", save, prefix)

    return fig, ax


def psd(
    model: dict,
    title: str = None,
    duration: int = None,
    dt: float = 1.0,
    folder: str = None,
    save: bool = False,
    population: int = 1,
    excitatory: bool = False,
    fig_size: tuple = None,
):
    """
    Plots the Power Spectral Density.
    """
    print("Generate PSD plot ...")

    duration = duration if duration else model["params"]["runtime"]
    if excitatory:
        key = "excitatory"
    else:
        key = "inhibitory"

    lfp1, lfp2 = processing.lfp(model, population=population)

    timepoints = int((duration / dt) / 2)
    fs = 1.0 / dt

    psd1, freqs = mlab.psd(
        lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none
    )
    psd2, _ = mlab.psd(
        lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none
    )

    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE_PSD)

    ax = fig.add_subplot(111)
    ax.set_title(title if title else "PSD")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Density")
    ax.plot(freqs * 1000, psd1, "0.25", linewidth=3.0, c="darkgray")
    ax.plot(freqs * 1000, psd2, "0.75", linewidth=3.0, c="dimgray")

    plt.legend(
        handles=[
            mpatches.Patch(color="darkgray", label="Excitatory Group"),
            mpatches.Patch(color="dimgray", label="Inhibitory Group"),
        ]
    )

    ax.set_xlim([0, 80])

    _save_to_file("psd", save, key, folder)

    return fig, ax


def raster(
    model: dict,
    title: str = None,
    x_left: int = None,
    x_right: int = None,
    save: bool = False,
    key: str = "",
    folder: str = None,
    population: int = 1,
    fig_size: Tuple = None,
    ax=None,
):
    fig = None
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)
        ax = fig.add_subplot(111)

    if population == 1:
        s_e = model["model_results"]["net"]["net_spikes_e"]
        s_i = model["model_results"]["net"]["net_spikes_i1"]

    else:
        s_e = model["model_results"]["net"]["net_spikes_e2"]
        s_i = model["model_results"]["net"]["net_spikes_i2"]

    ax.set_title(title if title else "Raster")
    ax.set_xlabel("Time in ms")
    ax.set_ylabel("Neuron index")
    ax.plot(s_e[1] * 1000, s_e[0], "k.", c="dimgrey", markersize="4.0")
    ax.plot(
        s_i[1] * 1000, s_i[0] + (s_e[0].max() + 1), "k.", c="black", markersize="4.0"
    )

    plt.legend(
        handles=[
            mpatches.Patch(color="dimgrey", label="Excitatory Group"),
            mpatches.Patch(color="black", label="Inhibitory Group"),
        ]
    )

    # TODO: plot complete time axis, currently only x-ticks for available data is plotted
    ax.set_xlim(left=x_left, right=x_right)

    plt.tight_layout()
    _save_to_file("raster", save=save, key=key, folder=folder)
    return fig, ax


def lfp_nets(model: dict, single_net: bool = False):
    dt = 1.0
    duration = model["params"]["runtime"]

    lfp1 = processing.lfp_single_net(model)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.set_title("LFP of network" if single_net else "LFP of both networks")
    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")
    ax.plot(t, lfp1, "0.75", color="black")

    handles = [mpatches.Patch(color="black", label="Network 1")]

    if not single_net:
        lfp2 = processing.lfp_single_net(model, population=2)
        ax.plot(t, lfp2, "0.75", color="darkgrey")
        handles.append(mpatches.Patch(color="darkgrey", label="Network 2"))

    plt.legend(handles=handles)
    plt.tight_layout()

    return ax


def population_rates(model: dict):
    """
    Plots the smoothed population rates for excitatory and inhibitory groups of both populations.

    :param model: model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey="col")

    N_pop = model["params"]["N_pop"]

    r_e = model["model_results"]["net"]["r_e"]
    r_i1 = model["model_results"]["net"]["r_i1"]

    axs[0, 0].plot(r_e, c="black")
    axs[0, 0].set_title("Population 1 - Excitatory")

    axs[1, 0].plot(r_i1, c="grey")
    axs[1, 0].set_title("Population 1 - Inhibitory")

    if N_pop > 1:
        r_e2 = model["model_results"]["net"]["r_e2"]
        r_i2 = model["model_results"]["net"]["r_i2"]

        axs[0, 1].plot(r_e2, c="black")
        axs[0, 1].set_title("Population 2 - Excitatory")

        axs[1, 1].plot(r_i2, c="grey")
        axs[1, 1].set_title("Population 2 - Inhibitory")


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

            lfp1, lfp2 = processing.lfp(model=data, duration=duration)

            timepoints = int((duration / dt) / 2)
            fs = 1.0 / dt

            psd1, freqs = mlab.psd(
                lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none
            )
            psd2, _ = mlab.psd(
                lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none
            )

            psd1[0] = 0.0
            psd2[0] = 0.0

            ax.set_xlabel("Frequency")
            ax.set_ylabel("Density")
            ax.plot(freqs * 1000, psd1, "0.25", linewidth=1.0, c="black")
            ax.plot(freqs * 1000, psd2, "0.75", linewidth=1.0, c="dimgray")
            ax.set_xlim([0, 80])

    plt.tight_layout()
    return fig, axs


def ou_noise_by_params(params: dict, fig_size: Tuple = None):
    mean = generate_ou_input(
        params["runtime"], params["min_dt"], params["ou_stationary"], params["ou_mu"]
    )
    sigma = generate_ou_input(
        params["runtime"], params["min_dt"], params["ou_stationary"], params["ou_sigma"]
    )
    return noise(mean, sigma, save=False, fig_size=fig_size)


def heat_map(
    models: List[Dict],
    x: str = "mean",
    y: str = "sigma",
    metric: str = "bandpower",
    **kwargs,
):
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

    heatmap_data = pd.pivot_table(df, values=metric, index=[x], columns=y)

    ax = sns.heatmap(heatmap_data, **kwargs)

    return fig, ax, df


def band_power(model):
    lfp = processing.lfp_single_net(model)

    runtime_ = model["params"]["runtime"]
    dt = 1.0
    timepoints = int((runtime_ / dt) / 2)
    fs = 1.0 / dt

    psd, freqs = mlab.psd(
        lfp, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none
    )
    psd[0] = 0.0
    freqs = freqs * 1000
    freqs = [int(freq) for freq in freqs]

    max_amplitude = psd.max()
    peak_freq = freqs[psd.argmax()]

    return max_amplitude, peak_freq


def _prepare_data(metric: str, models: [dict], x: str, y: str):
    data = {x: [], y: [], metric: []}

    for model in models:
        # x: mean
        # y: sigma
        # z: max_amplitude
        mean_ = model["params"]["ou_mu"]["ou_mean"]
        sigma_ = model["params"]["ou_mu"]["ou_sigma"]
        tau_ = model["params"]["ou_mu"]["ou_tau"]

        max_amplitude, peak_freq = band_power(model)

        if x == "tau":
            data[x].append(tau_)
        elif x == "mean":
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


def _save_to_file(name: str, save: bool, key: str = None, folder: str = None):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base
        plt.savefig(f"{constants.PLOTS_PATH}/{fname}")
