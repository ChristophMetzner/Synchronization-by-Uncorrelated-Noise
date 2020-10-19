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
FIG_SIZE_PSD = [10, 3]

# Title and Axes Fontsize
FONTSIZE = 14

# Colors
c_exc = "r"
c_inh = "midnightblue"
c_net_1 = "darkorange"
c_net_2 = "teal"

ING_FOLDER = "ING"


def plot_ING_exp_figure(
    ex: mopet.Exploration,
    param_x: str = None,
    param_y: str = None,
    vmax_phase: float = 1.0,
    vmin_phase: float = 0.0,
    vmin_ratio: int = 0,
    filename: str = "ING_exp",
):
    """
    Plots two dimensional heat maps of ING model exploration.

    :param ex:
    :param param_x:
    :param param_y:
    :param vmax_phase:
    :param vmin_phase:
    :param vmin_ratio:
    :param filename:
    :return:
    """
    if not param_x or not param_y:
        axis_names = list(ex.explore_params.keys())
        param_x = axis_names[0]
        param_y = axis_names[1]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    x_label = "Frequency Ratio p"
    y_label = "Noise Strength $\sigma^2$"

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_i",
        param_X=param_x,
        param_Y=param_y,
        title="Within Phase Synchronization - Net 1",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        xlabel=x_label,
        ylabel=y_label,
        ax=axs.flat[0],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_2_i",
        param_X=param_x,
        param_Y=param_y,
        title="Within Phase Synchronization - Net 2",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        xlabel=x_label,
        ylabel=y_label,
        ax=axs.flat[1],
    )

    # heat_map_vis(
    #     df=ex.df,
    #     value="phase_synchronization",
    #     param_X=param_x,
    #     param_Y=param_y,
    #     title="Phase Synchronization",
    #     colorbar="Kuramoto Order Parameter",
    #     vmin=vmin_phase,
    #     vmax=vmax_phase,
    #     xlabel=x_label,
    #     ylabel=y_label,
    #     ax=axs.flat[2],
    # )

    heat_map_vis(
        df=ex.df,
        value="freq_ratio",
        param_X=param_x,
        param_Y=param_y,
        title="Dominant Frequency Ratio",
        colorbar="Ratio",
        vmin=vmin_ratio,
        vmax=1.0,
        xlabel=x_label,
        ylabel=y_label,
        ax=axs.flat[3],
    )

    heat_map_vis(
        df=ex.df,
        value="mean_phase_coherence",
        param_X=param_x,
        param_Y=param_y,
        title="Mean Phase Coherence between Networks",
        colorbar="Mean Phase Coherence",
        vmin=vmin_phase,
        vmax=vmax_phase,
        xlabel=x_label,
        ylabel=y_label,
        ax=axs.flat[2],
    )

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
    """
    This method is used to plot the two dimensional exploration Figure in the thesis.

    Only a limited set of metrics is included.
    Use `plot_exploration` method to plot heat maps for all available metrics.

    :param ex: loaded Mopet exploration.
    :param param_X: param name for x axis.
    :param param_Y: param name for y axis.
    :param vmax_phase: max value for phase measures.
    :param vmin_phase: min value for phase measures.
    :param vmin_ratio: min value for frequency ratio.
    :param filename: name of plot on disk.
    :return: (fig, axs)
    """
    if not param_X or not param_Y:
        axis_names = list(ex.explore_params.keys())
        param_X = axis_names[0]
        param_Y = axis_names[1]

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 11))

    x_label = "Frequency Ratio p"
    y_label = "Noise Strength $\sigma^2$"

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_e",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Net 1 - Excitatory",
        colorbar="Kuramoto Order Parameter",
        xlabel=x_label,
        ylabel=y_label,
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[0],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_1_i",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Net 1 - Inhibitory",
        colorbar="Kuramoto Order Parameter",
        xlabel=x_label,
        ylabel=y_label,
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
        xlabel=x_label,
        ylabel=y_label,
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
        xlabel=x_label,
        ylabel=y_label,
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
        value="mean_phase_coherence",
        param_X=param_X,
        param_Y=param_Y,
        title="Mean Phase Coherence between Networks",
        colorbar="Mean Phase Coherence",
        xlabel=x_label,
        ylabel=y_label,
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs.flat[4],
    )

    heat_map_vis(
        df=ex.df,
        value="freq_ratio",
        param_X=param_X,
        param_Y=param_Y,
        title="Dominant Frequency Ratio",
        colorbar="Ratio",
        xlabel=x_label,
        ylabel=y_label,
        vmin=vmin_ratio,
        vmax=1.0,
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
    folder: str = None,
):
    """ Plots 2 dimensional maps to visualize parameter exploration.

    :param ex: mopet exploration
    :type ex: mopet.Exploration
    :param param_X: param for x axis, defaults to None
    :type param_X: str, optional
    :param param_Y: param for y axis, defaults to None
    :type param_Y: str, optional
    :param folder: folder ot save plot to.
    :param vmax_bandpower: max value of bandpower.
    :param vmin_ratio: min value of dom frequency ratio.
    :param vmax_freq: max value of dom frequency ratio.
    :param vmin_phase: min value of phase measures.
    :param vmax_phase: max value of phase measures.
    """
    if len(ex.explore_params.keys()) == 1:
        _one_dim_exploration(ex, folder)
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


def _one_dim_exploration(ex, folder: str = None):
    """
    Plots one dimensional exploration results.

    :param ex: given Mopet exploration.
    :param folder: optionally save figures in folder.
    """
    param = list(ex.explore_params.keys())[0]
    legend = []

    linewidth_within = 1.0
    marker_within = "x"

    linewidth_across = 2.0
    marker_across = "o"

    alpha = 0.7

    fig, ax = plt.subplots(figsize=(10, 5))
    df = ex.df.sort_values(by=param)

    ax.plot(
        df[param],
        df["freq_ratio"],
        linewidth=linewidth_across,
        marker=marker_across,
        color=c_inh,
    )
    legend.append("Dominant Frequency Ratio")

    ax.plot(
        df[param],
        df["mean_phase_coherence"],
        linewidth=linewidth_across,
        marker=marker_across,
        color="mediumblue",
    )
    legend.append("Mean Phase Coherence - Networks")

    # We excluded Kuramoto order parameter between networks
    #   as mean phase coherence captures also delays between networks.
    # if "phase_synchronization" in df:
    #     ax.plot(
    #         df[param],
    #         df["phase_synchronization"],
    #         linewidth=linewidth_across,
    #         marker=marker_across,
    #     )
    #     legend.append("Phase Synchronization - Networks")

    if "plv_net_1_e" in df:
        ax.plot(
            df[param],
            df["plv_net_1_e"],
            linewidth=linewidth_within,
            marker=marker_within,
            alpha=alpha,
        )
        legend.append("Kuramoto Order Parameter - Net 1 - E Pop")

    if "plv_net_1_i" in df:
        ax.plot(
            df[param],
            df["plv_net_1_i"],
            linewidth=linewidth_within,
            marker=marker_within,
            alpha=alpha,
        )
        legend.append("Kuramoto Order Parameter - Net 1 - I Pop")

    if "plv_net_2_e" in df:
        ax.plot(
            df[param],
            df["plv_net_2_e"],
            linewidth=linewidth_within,
            marker=marker_within,
            alpha=alpha,
        )
        legend.append("Kuramoto Order Parameter - Net 2 - E Pop")

    if "plv_net_2_i" in df:
        ax.plot(
            df[param],
            df["plv_net_2_i"],
            linewidth=linewidth_within,
            marker=marker_within,
            alpha=alpha,
        )
        legend.append("Kuramoto Order Parameter - Net 2 - I Pop")

    plt.legend(legend, loc=[0.0, 1.0])
    ax.set_xlabel("Noise Strength $\sigma^2$")
    ax.set_ylim(0, 1.1)

    save_to_file("one_dim_exp", folder=folder)


def plot_results(
    model,
    full_raster: bool = False,
    pop_rates: bool = True,
    phase_analysis: bool = False,
    show_lfp: bool = True,
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
    :param show_lfp: if True plots the LFP surrogate.
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

    if show_lfp:
        lfp_nets(model, skip=100, single_net=networks == 1)

    if full_raster:
        fig, axs = plt.subplots(1, networks, figsize=(40, 18))
        raster(
            title="Raster of 1st network",
            model=model,
            key="stoch_weak_PING",
            ax=axs[0] if networks > 1 else axs,
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

    fig, axs = plt.subplots(figsize=(10, 4))
    raster(
        title=None,
        model=model,
        key="exc_net1",
        x_left=excerpt_x_left,
        x_right=excerpt_x_right,
        ax=axs,
        folder=folder,
        save=save,
    )

    if networks > 1:
        fig, axs = plt.subplots(figsize=(10, 4))
        raster(
            title=None,
            model=model,
            key="exc_net2",
            x_left=excerpt_x_left,
            x_right=excerpt_x_right,
            population=2,
            ax=axs,
            folder=folder,
            save=save,
        )

    if pop_rates:
        population_rates(model, skip=2000)

    if networks > 1 and phase_analysis:
        phases_inter_nets(model, folder=folder)
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
    """ Plot the local field potential.

    :param model: given model
    :param title: title of plot.
    :param dt: step size.
    :param duration: duration to plot.
    :param prefix: file prefix
    :param save: if True, saves plot to disk.
    :param skip: ms to skip of recording.
    :param population: selects the network.
    :return: (fig, ax)
    """

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
        ax.plot(freqs, psd1, "0.75", linewidth=3.0, c=c_exc)
        plt.legend(["Excitatory"])

    elif groups == "INH":
        ax.plot(freqs, psd2, "0.75", linewidth=3.0, c=c_inh)
        plt.legend(["Inhibitory"])

    else:
        ax.plot(freqs, psd1, "0.75", linewidth=3.0, c=c_exc)
        ax.plot(freqs, psd2, "0.75", linewidth=3.0, c=c_inh)
        plt.legend(["Excitatory", "Inhibitory"])

    ax.set_xlim([x_min, x_max])
    ax.set_xticks(range(x_min, x_max + 10, 10))

    save_to_file("psd", save, key, folder)

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
    """
    Plots spike rasters.

    :param model: given model.
    :param title: optional title of plot.
    :param x_left: optional start of x axis.
    :param x_right: optional end of x axis.
    :param save: if True, saves plot to disk.
    :param key: key used in filename of plot.
    :param folder: folder to save plot in.
    :param population: sets the network.
    :param fig_size: figure size.
    :param ax: axis.
    :return: (fig, ax)
    """
    fig = None
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else (10, 7))
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

    if title:
        ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel("Time [ms]", fontsize=FONTSIZE)
    ax.set_ylabel("Neuron Index", fontsize=FONTSIZE)

    if model["model_EI"]:
        ax.plot(s_e[1] * 1000, s_e[0], "k.", c=c_exc, markersize="2.0")

    ax.plot(
        s_i[1] * 1000,
        s_i[0] + (s_e[0].max() + 1 if s_e[0].size != 0 and model["model_EI"] else 0),
        "k.",
        c=c_inh,
        markersize="4.0",
    )

    ax.set_xlim(left=x_left if x_left else 0, right=x_right)

    plt.tight_layout()
    save_to_file("raster", save=save, key=key, folder=folder)
    return fig, ax


def lfp_nets(model: dict, single_net: bool = False, skip: int = None):
    """
    Plots the local field potential of both networks.

    :param model: loaded model.
    :param single_net: if True, assumes only presence of network 1.
    :param skip: ms to skip in recording.
    :return: ax
    """
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


def membrane_potentials_sample(model: dict, detail_window=(900, 1000), window=(1000, 2000)):
    """
    Plots several membrane potential traces of I and E cells.

    :param model: model with recorded traces.
    :param detail_window: time window for detail look.
    """
    v_i1 = model["v_all_neurons_i1"]
    v_i2 = model["v_all_neurons_i2"]

    t = model["t_all_neurons_i1"]
    t_detailed = model["t_all_neurons_i1"]
    if window:
        t = t[window[0]: window[1]]

    if model["model_EI"]:
        v_e1 = model["v_all_neurons_e"]

        plt.figure(figsize=(20, 3))
        plt.title("Membrane Voltages of excitatory neurons in Net 1")
        plt.xlabel("Time in [ms]")
        plt.ylabel("Voltage in [mV]")
        plt.plot(t, v_e1[0][window[0]: window[1]] if window else v_e1[0], c=c_exc, linewidth=2.5)
        plt.plot(t, v_e1[1][window[0]: window[1]] if window else v_e1[1], linewidth=0.5)
        plt.plot(t, v_e1[2][window[0]: window[1]] if window else v_e1[2], linewidth=0.5)

    plt.figure(figsize=(20, 3))
    plt.title("Membrane Voltages of inhibitory neurons in Net 2")
    plt.xlabel("Time in [ms]")
    plt.ylabel("Voltage in [mV]")
    plt.plot(t, v_i1[0][window[0]: window[1]] if window else v_i1[0], c=c_inh, linewidth=2.5)
    plt.plot(t, v_i1[1][window[0]: window[1]] if window else v_i1[1], linewidth=0.5)
    plt.plot(t, v_i1[2][window[0]: window[1]] if window else v_i1[2], linewidth=0.5)

    plt.figure(figsize=(20, 3))
    plt.title("Detailed look at single excitatory and inhibitory trace in Net 1")
    plt.plot(
        t_detailed[detail_window[0] : detail_window[1]],
        v_i1[0][detail_window[0] : detail_window[1]],
        c=c_inh,
        linewidth=2.5,
    )

    if model["model_EI"]:
        plt.plot(
            t_detailed[detail_window[0] : detail_window[1]],
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
    plt.plot(t, v_i2[0][window[0]: window[1]] if window else v_i2[0], c="green", linewidth=0.5)
    plt.plot(t, v_i2[1][window[0]: window[1]] if window else v_i2[1], linewidth=2.5)
    plt.plot(t, v_i2[2][window[0]: window[1]] if window else v_i2[2], linewidth=0.5)


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

    print(f"Average E firing frequency: {np.average(r_e)}")
    print(f"Average I firing frequency: {np.average(r_i1)}")

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
    :param folder: folder path to save plots in.
    """
    lfp1, lfp2 = processing.lfp_nets(model, skip=skip)
    f_lfp1, f_lfp2 = (
        processing.filter(lfp1, lowcut=30, highcut=120),
        processing.filter(lfp2, lowcut=30, highcut=120),
    )

    duration = 500
    fig_size = (10, 2.7)

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
    plt.xlabel("Time [ms]", fontsize=FONTSIZE)
    plt.ylabel("Angle", fontsize=FONTSIZE)
    plt.plot(processing.phase(f_lfp2[skip:duration + skip]), linewidth=1.5, c=c_inh)
    plt.plot(processing.phase(f_lfp1[skip:duration + skip]), linewidth=1.5, c=c_exc)
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
    xlabel: str = None,
    ylabel: str = None,
    ax=None,
    **kwargs,
):
    """
    Minimal interface to plot heat map based on DataFrame input.
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
        xlabel=xlabel,
        ylabel=ylabel,
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

    ax.set_title(title)

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


def isi_histograms(
    model: dict,
    bins: int = 60,
    filter_outlier: bool = False,
    folder: str = None,
    key: str = None,
):
    """
    Plots the inter spike interval histograms of each population.

    :param model: input model.
    :param bins: number of bins.
    :param filter_outlier: if True removes outlier from dataset.
    :param key: key for filename of figure.
    :param folder: relative output folder.
    """
    if model["model_EI"]:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 6), sharex='all', sharey='all')
    else:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 4), sharex='all', sharey='all')

    ax = _isi_histogram(axs.flat[0], model["isi_I"], bins, filter_outlier, color=c_inh)
    ax.set_title("Inhibitory Population of Network 1")

    ax = _isi_histogram(axs.flat[1], model["isi_I2"], bins, filter_outlier, color=c_inh)
    ax.set_title("Inhibitory Population of Network 2")

    if "model_EI" not in model or model["model_EI"]:
        ax = _isi_histogram(axs.flat[2], model["isi_E"], bins, filter_outlier)
        ax.set_title("Excitatory Population of Network 1")

        ax = _isi_histogram(axs.flat[3], model["isi_E2"], bins, filter_outlier)
        ax.set_title("Excitatory Population of Network 2")

    plt.tight_layout()
    save_to_file("ISI", key=key, folder=folder)


def _isi_histogram(ax, isi, bins: int, filter_outlier: bool, color: str = c_exc):
    """
    Plots a ISI histogram based on the given `isi`s.

    :param ax: axis used for plotting.
    :param isi: list of ISIs.
    :param bins: number of bins.
    :param filter_outlier: applies simple heuristic to remove outliers.
    :param color: color for plotting.
    :return:
    """
    avg_E = np.average(isi)
    if filter_outlier:
        isi = processing.filter_inter_spike_intervals(isi)

    ax.set_title("ISI Histogram")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Count")
    ax.hist(isi, bins=bins, color=color)
    ax.axvline(avg_E, color="orange", linestyle="dashed", linewidth=2.0)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(avg_E * 1.1, max_ylim * 0.7, r"$\mu$: {:.2f} ms".format(avg_E), fontsize=12)
    return ax


def spike_variability_analysis(
    v,
    v2,
    window,
    t_s,
    t_width=(5, 5),
    folder: str = None,
    key: str = None,
    mean_voltage: bool = False,
):
    """
    Visualizes the membrane voltage variability of the second network depending on a cycle of the first network.

    Groups neurons `v2` into two groups.
    Those that spiked in a time window `t_width` around `t_s` and those that didn't spike in this cycle.

    :param v: voltage traces of first network.
    :param v2: voltage traces of second network.
    :param window: time window for plots.
    :param t_s: cycle time.
    :param t_width: tuple of (start, end). Defines the window around `t_s`.
    :param folder: folder to save plot.
    :param key: file key.
    :param mean_voltage: if True, plots the mean voltage of both groups.
    """
    first, second = processing.group_neurons(
        v2, window=window, spike_t=t_s - window[0], t_start=t_width[0], t_end=t_width[1]
    )

    fontsize = 12
    n_plots = 2
    if mean_voltage:
        n_plots += 1

    figsize = (10, n_plots * 2.5)
    fig, axs = plt.subplots(figsize=figsize, nrows=n_plots)

    # Voltage traces of first network
    axs[0].set_xlabel("Time [ms]", fontsize=fontsize)
    axs[0].set_ylabel("Voltage [mV]", fontsize=fontsize)
    axs[0].set_ylim(-70, -40)

    for i in range(0, len(v)):
        axs[0].plot(v[i][window[0] : window[1]], linewidth=0.75, c="grey", alpha=0.15)

    # Mark the cycle and the window around the cycle.
    axs[0].axvline(t_s - window[0], color="orange", linestyle="solid", linewidth=3)
    axs[0].axvline(
        t_s - window[0] - t_width[0], color="orange", linestyle="dotted", linewidth=3
    )
    axs[0].axvline(
        t_s - window[0] + t_width[1], color="orange", linestyle="dotted", linewidth=3
    )

    # Voltage traces of second network
    axs[1].set_xlabel("Time [ms]", fontsize=fontsize)
    axs[1].set_ylabel("Voltage [mV]")
    axs[1].set_ylim(-70, -40)

    for f in first[:100]:
        axs[1].plot(f[window[0] : window[1]], c=c_exc, linewidth=0.5, alpha=0.75)
    for s in second[:100]:
        axs[1].plot(s[window[0] : window[1]], c=c_inh, linewidth=0.5, alpha=0.75)

    # Legend
    exc_patch = mpatches.Patch(color=c_exc, label="Group of spiking neurons")
    inh_patch = mpatches.Patch(color=c_inh, label="Group of suppressed neurons")
    axs[1].legend(handles=[exc_patch, inh_patch])

    # Mean Voltage Plot
    if mean_voltage:
        axs[2].set_xlabel("Time [ms]", fontsize=fontsize)
        axs[2].set_ylabel("Voltage [mV]")
        axs[2].set_ylim(-70, -40)
        if first:
            axs[2].plot(
                np.mean(first, axis=0)[window[0] : window[1]], c=c_exc, linewidth=3.0
            )

        if second:
            axs[2].plot(
                np.mean(second, axis=0)[window[0] : window[1]], c=c_inh, linewidth=3.0
            )

        axs[2].legend(["Group of spiking neurons", "Group of suppressed neurons"])

    save_to_file("spike_variability", folder=folder, key=key, dpi=150)


def spike_participation_histograms_per_network(model, network: int = 1, show_detection: bool = False):
    """
    Plots the spike participation distribution of neurons in their population rhythm.

    We first detect the peaks in the population activity voltage trace.
    And then use a window with predefined width to count the spikes per detected peak.

    :param model: given model.
    :param network: either 1 or 2.
    :param show_detection: if True plots the voltage trace and marks the detected peaks.
    :return:
    """
    width = 6

    lfp_e, lfp_i = processing.lfp(model, population=network)
    lfp_e, lfp_i = lfp_e[400:], lfp_i[400:]

    peaks_e, peaks_i = find_peaks(lfp_e, height=-57), find_peaks(lfp_i, height=-53)
    print(len(peaks_e[0]), len(peaks_i[0]))

    y_e, y_i = [lfp_e[p] for p in peaks_e[0]], [lfp_i[p] for p in peaks_i[0]]

    if show_detection:
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.title("Peak detection of E Population Signal")
        plt.plot(lfp_e, c_inh, linewidth=2.0)
        plt.plot(peaks_e[0], y_e, "r+", linewidth=3, c=c_exc)
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 5))
        plt.title("Peak detection of I Population Signal")
        plt.plot(lfp_i, c_inh, linewidth=2.0)
        plt.plot(peaks_i[0], y_i, "r+", linewidth=3, c=c_exc)
        plt.show()

    if network == 1:
        v = model["v_all_neurons_e"]
        v2 = model["v_all_neurons_i1"]
    else:
        v = model["v_all_neurons_e2"]
        v2 = model["v_all_neurons_i2"]

    p_e, p_e_peaks = processing.spike_participation(v, peaks_e, width)
    p_i, p_i_peaks = processing.spike_participation(v2, peaks_i, width)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].hist(list(p_e.values()), bins=20, color=c_exc, alpha=1.0)
    axs[0, 0].set_title("Participation per Neuron")

    axs[0, 1].hist(list(p_i.values()), bins=20, color=c_inh, alpha=1.0)
    axs[0, 1].set_xlim(0, 1)

    axs[1, 0].hist(list(p_e_peaks.values()), bins=20, color=c_exc, alpha=1.0)
    axs[1, 0].set_title("Participation per Peak")
    axs[1, 0].set_xlim(0, 1)

    axs[1, 1].hist(list(p_i_peaks.values()), bins=20, color=c_inh, alpha=1.0)
    axs[1, 1].set_title("Participation per Peak")
    axs[1, 1].set_xlim(0, 1)


def spike_participation_histograms(model):
    """
    The LFP signal for EI population should be split up into LFP of E and LFP of I.
    Or then simply the average membrane potentials of both cell types.
    
    It does not make sense ot use the current LFP and find spike matches for E and I when they fire not at the same time.
    This must be considered in further analysis...
    
    Also phase analysis methods should only use then LFP of E cells.
    """
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
    axs[0].hist(list(p_1.values()), bins=20, color=c_exc, alpha=0.65)
    axs[0].hist(list(p_2.values()), bins=20, color=c_inh, alpha=0.65)
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


def save_to_file(
    name: str, save: bool = True, key: str = None, folder: str = None, dpi: int = 300
):
    """
    General purpose method to save a figure as file to disk.

    :param name: name of figure.
    :param save: if True, saves to disk.
    :param key: unique key.
    :param folder: optional folder relative to PLOTS_PATH.
    :param dpi: defines quality of figure.
    """
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base

        if folder:
            os.makedirs(constants.PLOTS_PATH + "/" + folder, exist_ok=True)

        plt.tight_layout()
        plt.savefig(f"{constants.PLOTS_PATH}/{fname}", dpi=dpi, bbox_inches="tight")
