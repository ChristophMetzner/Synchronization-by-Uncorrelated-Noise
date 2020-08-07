import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import os

from typing import Tuple, List, Dict
from matplotlib import mlab
from scipy.signal import find_peaks

from synchronization import constants
from synchronization import processing
from synchronization.utils import generate_ou_input
from mopet import mopet

FIG_SIZE = [10, 6]
FIG_SIZE_QUADRATIC = [8, 6]
FIG_SIZE_PSD = [8, 3]

# Title and Axes Fontsize
FONTSIZE = 14

# Colors
c_exc = "r"
# c_inh = "cornflowerblue"
c_inh = "midnightblue"
c_net_1 = "midnightblue"
c_net_2 = "crimson"


def plot_ING_exp_figure(
    ex: mopet.Exploration,
    param_x: str = None,
    param_y: str = None,
    vmax_phase: float = 1.0,
    vmin_phase: float = 0.0,
    vmin_ratio: int = 0,
    filename: str = "ING_exp",
):
    if not param_x or not param_y:
        axis_names = list(ex.explore_params.keys())
        param_x = axis_names[0]
        param_y = axis_names[1]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_i",
        param_X=param_x,
        param_Y=param_y,
        title="Within Phase Synchronization - Network 1 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[0],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_2_i",
        param_X=param_x,
        param_Y=param_y,
        title="Within Phase Synchronization - Net 2 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[1],
    )

    heat_map_vis(
        df=ex.df,
        value="phase_synchronization",
        param_X=param_x,
        param_Y=param_y,
        title="Phase Synchronization",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[2],
    )

    heat_map_vis(
        df=ex.df,
        value="freq_ratio",
        param_X=param_x,
        param_Y=param_y,
        title="Dominant Frequency Ratio",
        colorbar="Ratio",
        vmin=vmin_ratio,
        vmax=1.0,
        ax=axs.flat[3],
    )

    # heat_map_vis(
    #     df=ex.df,
    #     value="mean_phase_coherence",
    #     param_X=param_x,
    #     param_Y=param_y,
    #     title="Mean Phase Coherence between Networks",
    #     colorbar="Mean Phase Coherence",
    #     vmin=vmin_phase,
    #     vmax=vmax_phase,
    #     ax=axs.flat[4],
    # )

    save_to_file(filename, folder="ING")

    return fig, axs


def plot_exp_figure(
    ex: mopet.Exploration,
    param_X: str = None,
    param_Y: str = None,
    vmax_phase: float = 1.0,
    vmin_phase: float = 0.0,
    vmin_ratio: int = 0,
    filename: str = "PING_exp",
):
    if not param_X or not param_Y:
        axis_names = list(ex.explore_params.keys())
        param_X = axis_names[0]
        param_Y = axis_names[1]

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(17, 20))

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_e",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Network 1 - Excitatory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[0],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_i",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Network 1 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[1],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_2_e",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Net 2 - Excitatory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[2],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_2_i",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Net 2 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[3],
    )

    # heat_map_vis(
    #     df=ex.df,
    #     value="phase_synchronization",
    #     param_X=param_X,
    #     param_Y=param_Y,
    #     title="Phase Synchronization",
    #     colorbar="Kuramoto Order Parameter",
    #     vmin=vmin_phase,
    #     vmax=vmax_phase,
    #     ax=axs.flat[4],
    # )

    heat_map_vis(
        df=ex.df,
        value="freq_ratio",
        param_X=param_X,
        param_Y=param_Y,
        title="Dominant Frequency Ratio",
        colorbar="Ratio",
        vmin=vmin_ratio,
        vmax=1.0,
        ax=axs.flat[4],
    )

    heat_map_vis(
        df=ex.df,
        value="mean_phase_coherence",
        param_X=param_X,
        param_Y=param_Y,
        title="Mean Phase Coherence between Networks",
        colorbar="Mean Phase Coherence",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[5],
    )

    save_to_file(filename, folder="PING")

    return fig, axs


def plot_exploration(
    ex: mopet.Exploration,
    param_X: str = None,
    param_Y: str = None,
    vmax_phase: float = 1.0,
    vmin_phase: float = 0.0,
    vmax_freq: int = 120,
    vmin_ratio: int = 0,
    vmax_bandpower: int = 1000,
):
    """Plots 2 dimensional maps to visualize parameter exploration.

    :param vmin_phase:
    :param vmax_phase:
    :param ex: mopet exploration
    :type ex: mopet.Exploration
    :param param_X: param for x axis, defaults to None
    :type param_X: str, optional
    :param param_Y: param for y axis, defaults to None
    :type param_Y: str, optional
    """

    if len(ex.explore_params.keys()) == 1:
        _one_dim_exploration(ex)
        return

    if not param_X or not param_Y:
        axis_names = list(ex.explore_params.keys())
        param_X = axis_names[0]
        param_Y = axis_names[1]

    # for final figure: 3 rows and 2 columns
    fig, axs = plt.subplots(3, 4, figsize=(40, 20))

    heat_map_vis(
        df=ex.df,
        value="peak_freq",
        param_X=param_X,
        param_Y=param_Y,
        title="Dominant Frequency of Network 1",
        colorbar="Peak Frequency",
        vmin=0.0,
        vmax=vmax_freq,
        ax=axs[0, 0],
    )

    heat_map_vis(
        df=ex.df,
        value="max_amplitude",
        param_X=param_X,
        param_Y=param_Y,
        title="Bandpower of Dominant Frequency of Network 1",
        colorbar="Bandpower",
        vmin=0.0,
        vmax=vmax_bandpower,
        ax=axs[0, 1],
    )

    heat_map_vis(
        df=ex.df,
        value="peak_freq_2",
        param_X=param_X,
        param_Y=param_Y,
        title="Dominant Frequency of Network 2",
        colorbar="Peak Frequency",
        vmin=0.0,
        vmax=vmax_freq,
        ax=axs[1, 0],
    )

    heat_map_vis(
        df=ex.df,
        value="max_amplitude_2",
        param_X=param_X,
        param_Y=param_Y,
        title="Bandpower of Dominant Frequency of Network 2",
        colorbar="Bandpower",
        vmin=0.0,
        vmax=vmax_bandpower,
        ax=axs[1, 1],
    )

    # heat_map_vis(
    #     df=ex.df,
    #     value="plv_net_1",
    #     param_X=param_X,
    #     param_Y=param_Y,
    #     title="Within Phase Synchronization - Network 1",
    #     colorbar="Kuramoto Order Parameter",
    #     vmin=vmin_phase,
    #     vmax=vmax_phase,
    #     ax=axs[0, 2],
    # )

    if "plv_net_1_e" in ex.df:
        heat_map_vis(
            df=ex.df,
            value="plv_net_1_e",
            param_X=param_X,
            param_Y=param_Y,
            title="Within Phase Synchronization - Network 1 - Excitatory",
            colorbar="Kuramoto Order Parameter",
            vmin=vmin_phase,
            vmax=vmax_phase,
            ax=axs[0, 2],
        )
    else:
        axs[0, 2].set_axis_off()

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_i",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Network 1 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[0, 3],
    )

    # heat_map_vis(
    #     df=ex.df,
    #     value="plv_net_2",
    #     param_X=param_X,
    #     param_Y=param_Y,
    #     title="Within Phase Synchronization - Network 2",
    #     colorbar="Kuramoto Order Parameter",
    #     vmin=vmin_phase,
    #     vmax=vmax_phase,
    #     ax=axs[1, 2],
    # )

    if "plv_net_2_e" in ex.df:
        heat_map_vis(
            df=ex.df,
            value="plv_net_2_e",
            param_X=param_X,
            param_Y=param_Y,
            title="Within Phase Synchronization - Net 2 - Excitatory",
            colorbar="Kuramoto Order Parameter",
            vmin=vmin_phase,
            vmax=vmax_phase,
            ax=axs[1, 2],
        )
    else:
        axs[1, 2].set_axis_off()

    heat_map_vis(
        df=ex.df,
        value="plv_net_2_i",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Net 2 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[1, 3],
    )

    heat_map_vis(
        df=ex.df,
        value="phase_synchronization",
        param_X=param_X,
        param_Y=param_Y,
        title="Phase Synchronization",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[2, 0],
    )

    if "freq_ratio" in ex.df.columns:
        heat_map_vis(
            df=ex.df,
            value="freq_ratio",
            param_X=param_X,
            param_Y=param_Y,
            title="Dominant Frequency Ratio",
            colorbar="Ratio",
            vmin=vmin_ratio,
            vmax=1.0,
            ax=axs[2, 1],
        )
    else:
        axs[2, 1].set_axis_off()

    heat_map_vis(
        df=ex.df,
        value="mean_phase_coherence",
        param_X=param_X,
        param_Y=param_Y,
        title="Mean Phase Coherence between Networks",
        colorbar="Mean Phase Coherence",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[2, 2],
    )

    # No content
    axs[2, 3].set_axis_off()


def _one_dim_exploration(ex):
    param = list(ex.explore_params.keys())[0]
    metric = "freq_ratio"

    fig, ax = plt.subplots(figsize=(15, 3))
    df = ex.df.sort_values(by=param)

    ax.plot(df[param], df[metric], c=c_inh, marker=".")
    ax.set_title("Frequency Ratio")
    ax.set_xlabel(param)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Dom Freq Ratio")

    metric = "mean_phase_coherence"
    legend = []

    fig, ax = plt.subplots(figsize=(15, 3))
    df = ex.df.sort_values(by=param)
    ax.plot(df[param], df[metric], linewidth=2.0, marker=".", color=c_inh)
    legend.append("mean_phase_coherence")

    if "plv_net_1_e" in df:
        ax.plot(df[param], df["plv_net_1_e"], linewidth=2.0, marker=".")
        legend.append("plv_net_1_e")

    if "plv_net_2_e" in df:
        ax.plot(df[param], df["plv_net_2_e"], linewidth=2.0, marker=".")
        legend.append("plv_net_2_e")

    if "plv_net_1_i" in df:
        ax.plot(df[param], df["plv_net_1_i"], linewidth=2.0, marker=".")
        legend.append("plv_net_1_i")

    if "plv_net_2_i" in df:
        ax.plot(df[param], df["plv_net_2_i"], linewidth=2.0, marker=".")
        legend.append("plv_net_2_i")

    if "phase_synchronization" in df:
        ax.plot(df[param], df["phase_synchronization"], linewidth=2.0, marker=".")
        legend.append("phase_synchronization")

    plt.legend(legend)
    ax.set_title("Phase Locking")
    ax.set_xlabel(param)
    ax.set_ylabel("Mean Phase Coherence")
    ax.set_ylim(0, 1)


def plot_results(
    model,
    full_raster: bool = False,
    pop_rates: bool = True,
    phase_analysis: bool = False,
    raster_right: int = None,
    x_max_psd: int = 120,
    x_min_psd: int = 10,
    excerpt_x_left: int = 500,
    excerpt_x_right: int = 900,
    psd_group: str = None,
    skip: int = 200,
    networks: int = 2,
    folder: str = None,
    save: bool = False,
):
    """
    Plots all relevant figures needed to understand network behavior.

    * Power Spectral Density (PSD)
    * Local Field Potential (LFP)
    * Spike Raster
    * Population Rates
    * Phase Analysis

    :param model: model dict holding recorded data.
    :param full_raster: if True, raster plot will be shown.
    :param pop_rates: if True displays population rates.
    :param phase_analysis: if True plots phase analysis.
    :param raster_right: right limit for raster time axis.
    :param x_max_psd: maximum frequency x to show in PSD plot.
    :param x_min_psd: minimum frequency x to show in PSD plot.
    :param excerpt_x_left: raster left side start.
    :param excerpt_x_right: raster right side end.
    :param psd_group: INH or EXC.
    :param skip: amount of ms to skip in plots and processing.
    :param networks: number of networks in model.
    :param folder: folder to save plots in.
    :param save: if True save plots in filesystem.
    """
    psd(
        model,
        title=None,
        population=1,
        fig_size=(5, 3),
        x_max=x_max_psd,
        x_min=x_min_psd,
        groups=psd_group,
        skip=skip,
        key="1",
        folder=folder,
        save=save,
    )

    if networks > 1:
        psd(
            model,
            title=None,
            population=2,
            fig_size=(5, 3),
            x_max=x_max_psd,
            x_min=x_min_psd,
            groups=psd_group,
            skip=skip,
            key="2",
            folder=folder,
            save=save,
        )

    lfp_nets(model, skip=100, single_net=networks == 1)

    if full_raster:
        fig, axs = plt.subplots(1, 2, figsize=(40, 15))
        raster(
            title="Raster of 1st network",
            model=model,
            key="stoch_weak_PING",
            ax=axs[0],
            x_right=raster_right,
            folder=folder,
            save=save,
        )

        if networks > 1:
            raster(
                title="Raster of 2nd network",
                model=model,
                population=2,
                ax=axs[1],
                x_right=raster_right,
                folder=folder,
                save=save,
            )

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    raster(
        title=f"{excerpt_x_left}-{excerpt_x_right} ms of network 1",
        model=model,
        key="exc_net1",
        x_left=excerpt_x_left,
        x_right=excerpt_x_right,
        ax=axs[0],
        folder=folder,
        save=save,
    )

    if networks > 1:
        raster(
            title=f"{excerpt_x_left}-{excerpt_x_right} ms of network 2",
            model=model,
            key="exc_net2",
            x_left=excerpt_x_left,
            x_right=excerpt_x_right,
            population=2,
            ax=axs[1],
            folder=folder,
            save=save,
        )

    if pop_rates:
        population_rates(model, skip=2000)

    if networks > 1 and phase_analysis:
        phases_inter_nets(model, folder=folder)
        phases_intra_nets(model)
        phases_intra_nets(model)


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

    save_to_file("noise", save=save, folder=prefix)

    return fig, ax


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
    duration = duration if duration else model["runtime"]

    lfp1, lfp2 = processing.lfp(model, duration, skip, population=population)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")

    ax.plot(t, lfp1, "0.75", c=c_inh)
    ax.plot(t, lfp2, "0.25", c=c_exc)

    plt.legend(
        ["Excitatory Group", "Inhibitory Group",]
    )
    plt.tight_layout()

    save_to_file("summed_voltage", save, prefix)

    return fig, ax


def psd(
    model: dict,
    title: str = None,
    duration: int = None,
    dt: float = None,
    granularity: float = 1.0,
    skip: int = 100,
    x_max: int = 120,
    x_min: int = 0,
    population: int = 1,
    key: str = None,
    folder: str = None,
    save: bool = False,
    groups: str = None,
    fig_size: tuple = None,
):
    """
    Plots the Power Spectral Density.
    """
    duration = duration if duration else model["runtime"]
    dt = dt if dt else model["net_record_all_neurons_dt"]
    if skip:
        duration -= skip

    lfp1, lfp2 = processing.lfp(model, population=population, skip=skip)

    # number of data points used in each block for the FTT.
    # Set to number of data points in the input signal.
    nfft = int((duration / dt / granularity))

    # Calculating the Sampling frequency.
    # As our time unit is here ms and step size of 1 ms, a fs of 1.0 is the best we can do.
    fs = 1.0 / dt

    # NFFT: length of each segment, set here to 1.0.
    # Thus each segment is exactly one data point
    psd1, freqs = mlab.psd(lfp1, NFFT=nfft, Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=nfft, Fs=fs, noverlap=0, window=mlab.window_none)

    # We multiply by 1000 to get from ms to s.
    freqs = freqs * 1000

    # Remove unwanted power at 0 Hz
    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE_PSD)
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE)
    ax.set_ylabel("Power", fontsize=FONTSIZE)

    if groups == "EXC":
        ax.plot(freqs, psd1, "0.75", linewidth=4.0, c=c_exc)
        plt.legend(["Excitatory"])

    elif groups == "INH":
        ax.plot(freqs, psd2, "0.75", linewidth=4.0, c=c_inh)
        plt.legend(["Inhibitory"])

    else:
        ax.plot(freqs, psd1, "0.75", linewidth=4.0, c=c_exc)
        ax.plot(freqs, psd2, "0.75", linewidth=4.0, c=c_inh)
        plt.legend(["Excitatory", "Inhibitory"])

    ax.set_xlim([x_min, x_max])
    ax.set_xticks(range(x_min, x_max + 10, 10))

    save_to_file("psd", save, key, folder)

    return fig, ax


def psd_multiple_models(
    models: list,
    fig_size: tuple,
    duration: int = None,
    dt: float = 1.0,
    skip: int = 200,
    xlim: int = 120,
):
    duration = duration if duration else models[0]["runtime"]
    if skip:
        duration -= skip

    net_1_1, net_1_2 = processing.lfp_nets(models[0], skip=skip)
    net_2_1, net_2_2 = processing.lfp_nets(models[1], skip=skip)

    # number of data points used in each block for the FTT.
    # Set to number of data points in the input signal.
    NFFT = int((duration / dt / 1.0))

    # Calculating the Sampling frequency.
    # As our time unit is here ms and step size of 1 ms, a fs of 1.0 is the best we can do.
    fs = 1.0 / dt

    # NFFT: length of each segment, set here to 1.0.
    # Thus each segment is exactly one data point
    psd1_1, freqs = mlab.psd(
        net_1_1, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none
    )
    psd1_2, _ = mlab.psd(net_1_2, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)

    psd2_1, _ = mlab.psd(net_2_1, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)
    psd2_2, _ = mlab.psd(net_2_2, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)

    # We multiply by 1000 to get from ms to s.
    freqs = freqs * 1000

    # Remove unwanted power at 0 Hz
    psd1_1[0] = 0.0
    psd1_2[0] = 0.0
    psd2_1[0] = 0.0
    psd2_2[0] = 0.0

    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE_PSD)
    ax = fig.add_subplot(111)
    # ax.set_title("Power Spectral Density", fontsize=16)
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel("Power", fontsize=16)

    # first model
    ax.plot(freqs, psd1_1, alpha=0.3, linewidth=5.5, c=c_exc)
    ax.plot(freqs, psd1_2, alpha=0.3, linewidth=5.5, c="royalblue")

    # second model
    ax.plot(freqs, psd2_1, alpha=1.0, linewidth=5.5, c=c_exc)
    ax.plot(freqs, psd2_2, alpha=1.0, linewidth=5.5, c="royalblue")

    plt.legend(
        [
            "strength = 1.0 - Net 1",
            "strength = 1.0 - Net 2",
            "strength = 12.5 - Net 1",
            "strength = 12.5 - Net 2",
        ]
    )

    ax.set_xlim([0, xlim])
    ax.set_xticks(range(0, xlim + 10, 10))

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
        s_e = model["net_spikes_e"]
        s_i = model["net_spikes_i1"]

    else:
        s_e = model["net_spikes_e2"]
        s_i = model["net_spikes_i2"]

    if s_e[0].size == 0 and s_i[0].size == 0:
        print("0 size array of spikes, cannot create raster plot.")
        return

    ax.set_title(title if title else "Raster", fontsize=FONTSIZE)
    ax.set_xlabel("Time [ms]", fontsize=FONTSIZE)
    ax.set_ylabel("Neuron index", fontsize=FONTSIZE)

    ax.plot(s_e[1] * 1000, s_e[0], "k.", c=c_exc, markersize="2.0")
    ax.plot(
        s_i[1] * 1000,
        s_i[0] + (s_e[0].max() + 1 if s_e[0].size != 0 else 0),
        "k.",
        c=c_inh,
        markersize="4.0",
    )

    plt.legend(["Excitatory", "Inhibitory"])

    ax.set_xlim(left=x_left if x_left else 0, right=x_right)

    plt.tight_layout()
    save_to_file("raster", save=save, key=key, folder=folder)
    return fig, ax


def lfp_nets(model: dict, single_net: bool = False, skip: int = None):
    dt = model["net_record_dt"]
    duration = model["runtime"]

    if skip:
        duration -= skip

    lfp1 = processing.lfp_single_net(model, skip=skip)

    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    ax.set_title("LFP of network" if single_net else "LFP of both networks")
    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")
    ax.plot(t, lfp1, "0.75", color=c_net_1)

    handles = [mpatches.Patch(color=c_net_1, label="Network 1")]

    if not single_net:
        lfp2 = processing.lfp_single_net(model, population=2, skip=skip)
        ax.plot(t, lfp2, "0.75", color=c_net_2)
        handles.append(mpatches.Patch(color=c_net_2, label="Network 2"))

    plt.legend(handles=handles)
    plt.tight_layout()

    return ax


def membrane_potentials_sample(model: dict, detail_window=(900, 1000)):
    """
    Plots several membrane potential traces of I and E cells.

    :param model: model with recorded traces.
    :param detail_window: time window for detail look.
    """
    v_i1 = model["v_all_neurons_i1"]
    v_i2 = model["v_all_neurons_i2"]

    t = model["t_all_neurons_i1"]

    if model["model_EI"]:
        v_e1 = model["v_all_neurons_e"]

        plt.figure(figsize=(20, 3))
        plt.title("Membrane Voltages of excitatory neurons in Net 1", fontsize=FONTSIZE)
        plt.xlabel("Time in [ms]", fontsize=FONTSIZE)
        plt.ylabel("Voltage in [mV]", fontsize=FONTSIZE)
        plt.plot(t, v_e1[0], c=c_exc, linewidth=2.5)
        plt.plot(t, v_e1[1], linewidth=0.5)
        plt.plot(t, v_e1[2], linewidth=0.5)

    plt.figure(figsize=(20, 3))
    plt.title("Membrane Voltages of inhibitory neurons in Net 2", fontsize=FONTSIZE)
    plt.xlabel("Time in [ms]", fontsize=FONTSIZE)
    plt.ylabel("Voltage in [mV]", fontsize=FONTSIZE)
    plt.plot(t, v_i1[0], c=c_inh, linewidth=2.5)
    plt.plot(t, v_i1[1], linewidth=0.5)
    plt.plot(t, v_i1[2], linewidth=0.5)

    plt.figure(figsize=(20, 3))
    plt.title("Detailed look at single excitatory and inhibitory trace in Net 1")
    plt.plot(
        t[detail_window[0] : detail_window[1]],
        v_i1[0][detail_window[0] : detail_window[1]],
        c=c_inh,
        linewidth=2.5,
    )

    if model["model_EI"]:
        plt.plot(
            t[detail_window[0] : detail_window[1]],
            v_e1[0][detail_window[0] : detail_window[1]],
            c=c_exc,
            linewidth=2.5,
        )

    if detail_window[1] - detail_window[0] > 300:
        plt.xticks(np.arange(detail_window[0], detail_window[1], 50.0))
    else:
        plt.xticks(np.arange(detail_window[0], detail_window[1], 5.0))

    plt.figure(figsize=(20, 3))
    plt.title("Network 2", fontsize=14)
    plt.xlabel("Time in [ms]", fontsize=14)
    plt.ylabel("Voltage in [mV]", fontsize=14)
    plt.plot(t, v_i2[0], c="green", linewidth=0.5)
    plt.plot(t, v_i2[1], linewidth=2.5)
    plt.plot(t, v_i2[2], linewidth=0.5)


def population_rates(model: dict, skip: int = None):
    """
    Plots the smoothed population rates for excitatory and inhibitory groups of both populations.

    :param skip: amount of ms to skip.
    :param model: model.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey="col")

    N_pop = model["N_pop"]

    r_e = model["r_e"][skip:]
    r_i1 = model["r_i1"][skip:]

    if "r_e_t" in model:
        axs[0, 0].plot(model["r_e_t"][skip:], r_e, c=c_exc)
    else:
        axs[0, 0].plot(r_e, c=c_exc)
    axs[0, 0].set_title("Population 1 - Excitatory")
    axs[0, 0].set_xlabel("Time in ms")
    axs[0, 0].set_ylabel("Firing Rate")

    if "r_i1_t" in model:
        axs[1, 0].plot(model["r_i1_t"][skip:], r_i1, c=c_inh)
    else:
        axs[1, 0].plot(r_i1, c=c_inh)
    axs[1, 0].set_title("Population 1 - Inhibitory")
    axs[1, 0].set_xlabel("Time in ms")
    axs[1, 0].set_ylabel("Firing Rate")

    if N_pop > 1:
        r_e2 = model["r_e2"][skip:]
        r_i2 = model["r_i2"][skip:]

        if "r_e2_t" in model:
            axs[0, 1].plot(model["r_e2_t"][skip:], r_e2, c=c_exc)
        else:
            axs[0, 1].plot(r_e2, c=c_exc)
        axs[0, 1].set_title("Population 2 - Excitatory")
        axs[0, 1].set_xlabel("Time in ms")
        axs[0, 1].set_ylabel("Firing Rate")

        if "r_i2_t" in model:
            axs[1, 1].plot(model["r_i2_t"][skip:], r_i2, c=c_inh)
        else:
            axs[1, 1].plot(r_i2, c=c_inh)
        axs[1, 1].set_title("Population 2 - Inhibitory")
        axs[1, 1].set_xlabel("Time in ms")
        axs[1, 1].set_ylabel("Firing Rate")


def synaptic_conductance(model: dict, start: int = 100, end: int = None):
    """
    Plots g_AMPA and g_GABA trace of E and I neurons respectively.

    :param end: end time index.
    :param start: start time index.
    :param model: current model, must contain recorded AMPA and GABA conductances.
    """
    plt.figure(figsize=(20, 5))
    plt.xlabel("Time in [ms]", fontsize=14)
    plt.ylabel("Conductance in [nS]", fontsize=14)
    legend = ["GABA Conductance - Net 1"]

    print("Mean GABA Conductance - Net 1: ", model["gaba"].mean())

    # for trace in model["gaba"][:10]:
    #     plt.plot(model["gaba_t"][start:end], trace[start:end], c=c_inh)

    plt.plot(
        model["gaba_t"][start:end],
        model["gaba"].mean(axis=0)[start:end],
        c="black",
        alpha=0.7,
    )

    if "gaba_2" in model:
        plt.plot(
            model["gaba_t"][start:end],
            model["gaba_2"].mean(axis=0)[start:end],
            c="orange",
            alpha=0.7,
        )
        legend.append("GABA Conductance - Net 2")

        print("Mean GABA Conductance - Net 2: ", model["gaba_2"].mean())

    if model["model_EI"]:
        for trace in model["ampa"][:10]:
            plt.plot(model["ampa_t"][start:end], trace[start:end], c=c_exc, alpha=0.7)

            print("Mean AMAP Conductance: ", model["gaba"].mean())

        legend.append("AMPA Conductance")

    plt.legend(legend)
    plt.tight_layout()


def phases_inter_nets(
    model: dict, skip: int = 200, show_lfp: bool = False, folder: str = None
):
    """ Plots figures to analyze phases of networks and their synchronization.

    :param show_lfp: if True LFP over time will be plotted.
    :param skip: amount of ms to skip.
    :param model: the given model.
    :type model: dict
    """
    lfp1, lfp2 = processing.lfp_nets(model, skip=skip)
    f_lfp1, f_lfp2 = (
        processing.filter(lfp1, lowcut=30, highcut=120),
        processing.filter(lfp2, lowcut=30, highcut=120),
    )

    fig_size = (20, 3)

    global_order_parameter = processing.order_parameter_over_time((f_lfp1, f_lfp2))
    total_value = np.mean(global_order_parameter)
    mean_phase_coherence = processing.mean_phase_coherence(f_lfp1, f_lfp2)

    print(f"Global Order Parameter value of: {total_value}")
    print(f"Mean Phase Coherence {mean_phase_coherence}")

    if show_lfp:
        plt.figure(figsize=fig_size)
        plt.title("LFP of Net 1", fontsize=FONTSIZE)
        plt.plot(lfp1, c=c_inh)
        plt.xlabel("t in ms")
        plt.ylabel("Voltage in mV")

        plt.figure(figsize=fig_size)
        plt.title("30-80 Hz Filtered LFP of Net 1", fontsize=FONTSIZE)
        plt.xlabel("t in ms")
        plt.plot(f_lfp1, c=c_inh)

    plt.figure(figsize=fig_size)
    plt.title("Phases of Network 1 and 2 - First 800 ms", fontsize=FONTSIZE)
    plt.xlabel("Time [ms]", fontsize=FONTSIZE)
    plt.ylabel("Angle", fontsize=FONTSIZE)
    plt.plot(processing.phase(f_lfp1[:800]), linewidth=2.0, c=c_inh)
    plt.plot(processing.phase(f_lfp2[:800]), linewidth=2.0, c=c_exc)
    plt.legend(["Net 1", "Net 2"])

    if folder:
        save_to_file(name="phases", folder=folder)

    phase_difference(f_lfp1, f_lfp2)

    plt.figure(figsize=fig_size)
    plt.title(f"Phase Synchronization between Networks", fontsize=18)
    plt.xlabel("Time in ms")
    plt.ylim(0, 1.1)
    plt.ylabel("Kuramoto Order Parameter")
    plt.plot(global_order_parameter, linewidth=2.0, c=c_inh)


def phase_difference(signal_1, signal_2):
    """
    Plots the phase difference of the two signals over time.

    Signals must be filtered already and have the same dimension.

    :param signal_1:
    :param signal_2:
    :return:
    """
    # Do not unwrap because we want to have values in range of [-2*pi, 2*pi].
    phase_differences = processing.phase_difference(signal_1, signal_2, unwrap=False)
    plt.figure(figsize=(20, 3))
    plt.title(f"Phase Difference of Networks", fontsize=18)
    plt.xlabel("Time in ms")
    plt.ylabel("Phase Difference")
    plt.plot(phase_differences, linewidth=1.0, c=c_inh)


def phases_intra_nets(model: dict, skip: int = 200, duration: int = 600):
    """ Plots figures to analyze phases of neurons within a network.

    :param duration: duration that should be displayed.
    :param skip: amount of ms to skip.
    :param model: input model.
    :type model: dict
    """
    print("Computing within synchronization for network 1 and 2")

    v_i = processing.filter_signals(model["v_all_neurons_i1"])
    v_i2 = processing.filter_signals(model["v_all_neurons_i2"])

    if "model_EI" not in model or model["model_EI"]:
        v_e = processing.filter_signals(model["v_all_neurons_e"])
        v_e2 = processing.filter_signals(model["v_all_neurons_e2"])

        neurons_net_1 = np.vstack((v_i, v_e))[:, skip:]
        neurons_net_2 = np.vstack((v_i2, v_e2))[:, skip:]
    else:
        neurons_net_1 = v_i
        neurons_net_2 = v_i2

        v_e = None

    f_plv_net_1 = processing.order_parameter_over_time(neurons_net_1)
    f_plv_net_1_i = processing.order_parameter_over_time(v_i)
    f_plv_net_2 = processing.order_parameter_over_time(neurons_net_2)

    print("Within Synchronization of Network 1", np.mean(f_plv_net_1))
    print("Within Synchronization of Network 2", np.mean(f_plv_net_2))

    plt.figure(figsize=(20, 3))
    if "model_EI" not in model or model["model_EI"] and v_e is not None:
        plt.title("Phases of One Excitatory and One Inhibitory Neuron", fontsize=18)
        plt.plot(processing.phase(v_i[0][skip:duration]), linewidth=3.0, c=c_inh)
        plt.plot(processing.phase(v_e[1][skip:duration]), linewidth=3.0, c=c_exc)
    else:
        plt.title("Phases of two Inhibitory neurons", fontsize=18)
        plt.plot(processing.phase(v_i[0][skip:duration]), linewidth=3.0, c=c_inh)
        plt.plot(processing.phase(v_i[1][skip:duration]), linewidth=3.0, c=c_exc)

    plt.xlabel("t in ms")
    plt.ylabel("Angle", fontsize=18)

    plt.legend(["Inhibitory", "Excitatory"])

    plt.figure(figsize=(20, 3))
    plt.title("Phases of 5 Inhibitory Neurons", fontsize=18)
    plt.xlabel("t in ms")
    plt.ylabel("Angle", fontsize=18)
    plt.plot(processing.phase(v_i[1][skip:duration]), linewidth=1.5, c=c_inh)
    plt.plot(processing.phase(v_i[2][skip:duration]), linewidth=1.5, c=c_exc)
    plt.plot(processing.phase(v_i[3][skip:duration]), linewidth=1.5)
    plt.plot(processing.phase(v_i[4][skip:duration]), linewidth=1.5)
    plt.plot(processing.phase(v_i[5][skip:duration]), linewidth=1.5)

    if "model_EI" not in model or model["model_EI"]:
        plt.figure(figsize=(20, 3))
        plt.title("Phases of 5 Excitatory Neurons", fontsize=18)
        plt.xlabel("t in [ms]")
        plt.ylabel("Angle", fontsize=18)
        plt.plot(processing.phase(v_e[1][skip:duration]), linewidth=1.5, c=c_inh)
        plt.plot(processing.phase(v_e[2][skip:duration]), linewidth=1.5, c=c_exc)
        plt.plot(processing.phase(v_e[3][skip:duration]), linewidth=1.5)
        plt.plot(processing.phase(v_e[4][skip:duration]), linewidth=1.5)

    plt.figure(figsize=(20, 3))
    plt.title(f"Within Phase Synchronization over Time of Network 1", fontsize=18)
    plt.xlabel("Time in ms", fontsize=18)
    plt.ylim(0, 1.1)
    plt.ylabel("Phase Synchronization", fontsize=18)
    plt.plot(f_plv_net_1, linewidth=3.0, c="black")
    plt.plot(f_plv_net_1_i, linewidth=3.0, c=c_inh)

    if "model_EI" not in model or model["model_EI"]:
        f_plv_net_1_e = processing.order_parameter_over_time(v_e)
        plt.plot(f_plv_net_1_e, linewidth=3.0, c=c_exc)

    plt.legend(["All", "Inhibitory", "Excitatory"])


def ou_noise_by_params(model: dict, fig_size: Tuple = None):
    mean = generate_ou_input(
        model["runtime"],
        model["min_dt"],
        model["ou_stationary"],
        model["ou_mu_X0"][0],
        model["ou_mu_tau"][0],
        model["ou_mu_sigma"][0],
        model["ou_mu_mean"][0],
    )
    sigma = generate_ou_input(
        model["runtime"],
        model["min_dt"],
        model["ou_stationary"],
        model["ou_sigma_X0"][0],
        model["ou_sigma_tau"][0],
        model["ou_sigma_sigma"][0],
        model["ou_sigma_mean"][0],
    )
    return noise(mean, sigma, save=False, fig_size=fig_size)


def heat_map_vis(
    df: pd.DataFrame,
    param_X: str,
    param_Y: str,
    value: str,
    title: str = "",
    colorbar: str = "",
    ax=None,
    **kwargs,
):
    """
    Minimal interace to plot heat map based on DataFrame input.
    """
    heat_map_pivoted(
        pivot_table=df.pivot_table(
            values=value, index=param_Y, columns=param_X, aggfunc="first"
        ),
        extent=[
            min(df[param_X], default=0),
            max(df[param_X], default=0),
            min(df[param_Y], default=0),
            max(df[param_Y], default=0),
        ],
        title=title,
        colorbar=colorbar,
        ax=ax,
        **kwargs,
    )


def heat_map_pivoted(
    pivot_table,
    extent=None,
    title: str = "",
    colorbar: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ax=None,
    **kwargs,
):
    if not ax:
        fig, ax = plt.subplots()

    ax.set_title(title, fontsize=FONTSIZE)

    im = ax.imshow(
        pivot_table,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=plt.get_cmap("PuBu"),
        # interpolation="bilinear",
        **kwargs,
    )

    if colorbar:
        plt.colorbar(im, label=colorbar, ax=ax)

    ax.set_xlabel(xlabel if xlabel else pivot_table.columns.name)
    ax.set_ylabel(ylabel if ylabel else pivot_table.index.name)

    if not ax:
        plt.show()


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


def isi_histograms(model: dict, bins: int = 60, filter_outlier: bool = False):
    """
    Plots the inter spike interval histograms of each population.

    :param model: input model.
    :param bins: number of bins.
    :param filter_outlier: if True removes outlier from dataset.
    """
    if model["model_EI"]:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 10))
    else:
        fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

    ax = _isi_histogram(axs.flat[0], model["isi_I"], bins, filter_outlier, color=c_inh)
    ax.set_title("ISI Histogram of I Population of 1st Network")

    ax = _isi_histogram(axs.flat[1], model["isi_I2"], bins, filter_outlier, color=c_inh)
    ax.set_title("ISI Histogram of I Population of 2nd Network")

    if "model_EI" not in model or model["model_EI"]:
        ax = _isi_histogram(axs.flat[2], model["isi_E"], bins, filter_outlier)
        ax.set_title("ISI Histogram of E Population of 1st Network")

        ax = _isi_histogram(axs.flat[3], model["isi_E2"], bins, filter_outlier)
        ax.set_title("ISI Histogram of E Population of 2nd Network")

    plt.tight_layout()


def _isi_histogram(ax, isi, bins: int, filter_outlier: bool, color: str = c_exc):
    avg_E = np.average(isi)
    if filter_outlier:
        isi = processing.filter_inter_spike_intervals(isi)

    ax.set_title("ISI Histogram")
    ax.set_xlabel("Time in [ms]")
    ax.set_ylabel("Density")
    ax.hist(isi, bins=bins, color=color)
    ax.axvline(avg_E, color="orange", linestyle="dashed", linewidth=3)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(avg_E * 1.1, max_ylim * 0.9, r"$\mu$: {:.2f} ms".format(avg_E), fontsize=14)

    return ax


def spike_variability_analysis(v, v2, window, t_s, t_width=(5, 5)):
    figsize = (15, 2)
    first, second = processing.group_neurons(
        v2, window=window, spike_t=t_s - window[0], t_start=t_width[0], t_end=t_width[1]
    )

    plt.figure(figsize=figsize)
    plt.title(
        f"Membrane voltage of 1 neuron in Net 1 - with focus on spike at {t_s} ms",
        fontsize=14,
    )
    plt.xlabel("Time in [ms]", fontsize=14)
    plt.ylabel("Voltage in [mv]")
    plt.ylim(-70, -40)
    for i in range(0, 100):
        plt.plot(v[i][window[0] : window[1]], linewidth=0.75, c="grey", alpha=0.35)

    plt.figure(figsize=figsize)
    plt.title(
        "Comparison of mean membrane potential of spiking and non-spiking group",
        fontsize=14,
    )
    plt.xlabel("Time in [ms]", fontsize=14)
    plt.ylabel("Voltage in [mv]")
    plt.ylim(-70, -40)
    if first:
        plt.plot(np.mean(first, axis=0)[window[0] : window[1]], c=c_exc, linewidth=3.0)

    if second:
        plt.plot(np.mean(second, axis=0)[window[0] : window[1]], c=c_inh, linewidth=3.0)

    plt.legend(["Spiking of I in Net 2", "Suppressed of I in Net 2"])

    plt.figure(figsize=figsize)
    plt.title(
        "Membrane voltage traces showing grouping of I cells in net 2", fontsize=14
    )
    plt.xlabel("Time in [ms]", fontsize=14)
    plt.ylabel("Voltage in [mv]")
    plt.ylim(-70, -40)

    for f in first[:50]:
        plt.plot(f[window[0] : window[1]], c=c_exc, linewidth=0.5, alpha=0.75)
    for s in second[:50]:
        plt.plot(s[window[0] : window[1]], c=c_inh, linewidth=0.5, alpha=0.75)


def spike_participation_histograms(model):
    """
    The LFP signal for EI population should be split up into LFP of E and LFP of I.
    Or then simply the average membrane potentials of both cell types.
    
    It does not make sense ot use the current LFP and find spike matches for E and I when they fire not at the same time.
    This must be considered in further analysis...
    
    Also phase analysis methods should only use then LFP of E cells.
    """

    t = model["t_all_neurons_i1"]

    lfp = processing.lfp_single_net(model)
    lfp = lfp[400:2000]

    peaks = find_peaks(lfp)
    y = [lfp[p] for p in peaks[0]]

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.title("Peak detection for LFP of Network 1")
    plt.plot(lfp, c_inh, linewidth=2.0)
    plt.plot(peaks[0], y, "r+", linewidth=3, c=c_exc)
    plt.show()

    if model["model_EI"]:
        v = model["v_all_neurons_e"]
        v2 = model["v_all_neurons_e2"]
    else:
        v = model["v_all_neurons_i1"]
        v2 = model["v_all_neurons_i2"]

    p_1, p_1_peaks = processing.spike_participation(v, peaks)
    p_2, p_2_peaks = processing.spike_participation(v2, peaks)

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5))
    axs[0].set_title(
        "Histogram of spike participation per neuron in peaks of network 1"
    )
    axs[0].hist(list(p_1.values()), bins=20, color=c_exc)
    axs[0].hist(list(p_2.values()), bins=20, color=c_inh)
    axs[0].legend(
        ["Net 1 - participation per neuron", "Net 2 - participation per I neuron"]
    )

    axs[1].set_title("Histogram of general participation in peaks of network 1")
    axs[1].hist(list(p_1_peaks.values()), bins=20, color=c_exc)
    axs[1].hist(list(p_2_peaks.values()), bins=20, color=c_inh)
    axs[1].legend(
        [
            "Net 1 - participation across all neurons",
            "Net 2 - participation across all neurons",
        ]
    )


def _prepare_data(metric: str, models: [dict], x: str, y: str):
    data = {x: [], y: [], metric: []}

    for model in models:
        # x: mean
        # y: sigma
        # z: max_amplitude
        mean_ = model["ou_mu_mean"]
        sigma_ = model["ou_mu_sigma"]
        tau_ = model["ou_mu_tau"]

        max_amplitude, peak_freq = processing.band_power(model)

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


def save_to_file(
    name: str, save: bool = True, key: str = None, folder: str = None, dpi: int = 300
):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base

        if folder:
            os.makedirs(constants.PLOTS_PATH + "/" + folder, exist_ok=True)

        plt.tight_layout()
        plt.savefig(f"{constants.PLOTS_PATH}/{fname}", dpi=dpi, bbox_inches="tight")
