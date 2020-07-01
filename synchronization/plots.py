import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

from typing import Tuple, List, Dict
from matplotlib import mlab

from synchronization import constants
from synchronization import processing
from synchronization.utils import generate_ou_input
from mopet import mopet

# FIG_SIZE = [20, 15]
FIG_SIZE = [10, 6]
FIG_SIZE_QUADRATIC = [8, 6]
FIG_SIZE_PSD = [8, 3]

# Colors
c_exc = "r"
# c_inh = "cornflowerblue"
c_inh = "midnightblue"
c_net_1 = "midnightblue"
c_net_2 = "crimson"


def plot_exploration(
    ex: mopet.Exploration,
    param_X: str = None,
    param_Y: str = None,
    vmax_phase: float = 1.0,
    vmin_phase: float = 0.0,
    vmax_freq: int = 120,
    vmin_ratio: int = 0,
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
    if not param_X or not param_Y:
        axis_names = list(ex.explore_params.keys())
        param_X = axis_names[0]
        param_Y = axis_names[1]

    fig, axs = plt.subplots(2, 4, figsize=(30, 10))

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
        vmax=1000,
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
        vmax=1000,
        ax=axs[1, 1],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_1",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Network 1",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[0, 2],
    )

    heat_map_vis(
        df=ex.df,
        value="plv_net_2",
        param_X=param_X,
        param_Y=param_Y,
        title="Within Phase Synchronization - Network 2",
        colorbar="Kuramoto Order Parameter",
        vmin=vmin_phase,
        vmax=vmax_phase,
        ax=axs[1, 2],
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
        ax=axs[0, 3],
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
            ax=axs[1, 3],
        )
    else:
        axs[1, 3].set_axis_off()


def plot_results(
    model,
    full_raster: bool = False,
    pop_rates: bool = False,
    raster_right: int = None,
    xlim_psd: int = 120,
    excerpt_x_left: int = 200,
    excerpt_x_right: int = 300,
):
    """
    Plots all relevant figures needed to understand network behavior.

    * Power Spectral Density (PSD)
    * Local Field Potential (LFP)
    * Spike Raster
    * Population Rates

    """
    psd(model, title="PSD of 1st network", population=1, fig_size=(8, 3), xlim=xlim_psd)
    psd(model, title="PSD of 2nd network", population=2, fig_size=(8, 3), xlim=xlim_psd)

    lfp_nets(model, skip=100)

    if full_raster:
        fig, axs = plt.subplots(1, 2, figsize=(40, 15))
        raster(
            title="Raster of 1st network",
            model=model,
            save=True,
            key="stoch_weak_PING",
            ax=axs[0],
            x_right=raster_right,
            N_e=200,
        )
        raster(
            title="Raster of 2nd network",
            model=model,
            population=2,
            ax=axs[1],
            x_right=raster_right,
            N_e=200,
        )

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    raster(
        title=f"{excerpt_x_left}-{excerpt_x_right} ms of network 1",
        model=model,
        x_left=excerpt_x_left,
        x_right=excerpt_x_right,
        ax=axs[0],
    )
    raster(
        title=f"{excerpt_x_left}-{excerpt_x_right} ms of network 2",
        model=model,
        x_left=excerpt_x_left,
        x_right=excerpt_x_right,
        population=2,
        ax=axs[1],
    )

    if pop_rates:
        population_rates(model, skip=2000)


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
    # TODO: can be removed?
    if "poisson_input_t_e" in model and "poisson_input_spikes_e" in model:
        plt.figure(figsize=(10, 8))
        plt.title("Poisson Spike Train Input to E population")
        plt.xlabel("Time in ms")
        plt.ylabel("Neuron Index of Poisson Group")
        plt.plot(
            model["poisson_input_t_e"],
            model["poisson_input_spikes_e"],
            ".",
            c="grey",
            linewidth=0.5,
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
    groups: str = "both",
    fig_size: tuple = None,
    granularity: float = 1.0,
    skip: int = 100,
    xlim: int = 120,
):
    """
    Plots the Power Spectral Density.
    """
    duration = duration if duration else model["runtime"]
    if skip:
        duration -= skip

    key = groups

    lfp1, lfp2 = processing.lfp(model, population=population, skip=skip)

    # number of data points used in each block for the FTT.
    # Set to number of data points in the input signal.
    NFFT = int((duration / dt / granularity))

    # Calculating the Sampling frequency.
    # As our time unit is here ms and step size of 1 ms, a fs of 1.0 is the best we can do.
    fs = 1.0 / dt

    # NFFT: length of each segment, set here to 1.0.
    # Thus each segment is exactly one data point
    psd1, freqs = mlab.psd(lfp1, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)

    # We multiply by 1000 to get from ms to s.
    freqs = freqs * 1000

    # Remove unwanted power at 0 Hz
    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE_PSD)
    ax = fig.add_subplot(111)
    ax.set_title(title if title else "PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")

    if groups == "excitatory":
        ax.plot(freqs, psd1, "0.75", linewidth=3.0, c=c_exc)

    elif groups == "inhibitory":
        ax.plot(freqs, psd2, "0.75", linewidth=3.0, c=c_inh)

    else:
        ax.plot(freqs, psd1, "0.75", linewidth=3.0, c=c_exc)
        ax.plot(freqs, psd2, "0.75", linewidth=3.0, c=c_inh)

    plt.legend(
        handles=[
            mpatches.Patch(color=c_exc, label="Excitatory Group"),
            mpatches.Patch(color=c_inh, label="Inhibitory Group"),
        ]
    )

    ax.set_xlim([0, xlim])
    ax.set_xticks(range(0, xlim + 10, 10))

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
    N_e: int = None,
    N_i: int = None,
    ax=None,
):
    fig = None
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else FIG_SIZE)
        ax = fig.add_subplot(111)

    if population == 1:
        s_e = model["net_spikes_e"][:N_e]
        s_i = model["net_spikes_i1"][:N_i]

    else:
        s_e = model["net_spikes_e2"][:N_e]
        s_i = model["net_spikes_i2"][:N_i]

    if s_e[0].size == 0 and s_i[0].size == 0:
        print("0 size array of spikes, cannot create raster plot.")
        return

    ax.set_title(title if title else "Raster")
    ax.set_xlabel("Time in ms")
    ax.set_ylabel("Neuron index")

    ax.plot(s_e[1] * 1000, s_e[0], "k.", c=c_exc, markersize="2.0")
    ax.plot(
        s_i[1] * 1000,
        s_i[0] + (s_e[0].max() + 1 if s_e[0].size != 0 else 0),
        "k.",
        c=c_inh,
        markersize="4.0",
    )

    plt.legend(["Excitatory", "Inhibitory"])

    # TODO: plot complete time axis, currently only x-ticks for available data is plotted
    ax.set_xlim(left=x_left if x_left else 0, right=x_right)

    plt.tight_layout()
    _save_to_file("raster", save=save, key=key, folder=folder)
    return fig, ax


def lfp_nets(model: dict, single_net: bool = False, skip: int = None):
    dt = 1.0
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


def population_rates(model: dict, skip: int = None):
    """
    Plots the smoothed population rates for excitatory and inhibitory groups of both populations.

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


def all_psd(
    models: List[Dict],
    n_cols: int,
    n_rows: int,
    single_network: bool = False,
    figsize=(20, 5),
):
    models = list(models)

    # TODO: use ratio and define columns and rows depending on length of models
    fig, axs = plt.subplots(n_cols, n_rows, figsize=figsize, sharex=False)

    for i, ax in enumerate(axs.reshape(-1)):
        try:
            title, data = models[i]
        except (KeyError, IndexError):
            # ignore
            pass
        else:
            ax.set_title(title, fontsize=10)

            duration = data["runtime"]
            dt = 1.0

            if single_network:
                lfp1, lfp2 = processing.lfp(model=data, skip=100)
            else:
                lfp1, lfp2 = processing.lfp_nets(model=data, skip=100)

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
            ax.plot(freqs * 1000, psd1, "0.25", linewidth=2.0, c="blue")
            ax.plot(freqs * 1000, psd2, "0.75", linewidth=2.0, c="green")
            ax.set_xlim([0, 80])

    plt.tight_layout()
    return fig, axs


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
            min(df[param_X]),
            max(df[param_X]),
            min(df[param_Y]),
            max(df[param_Y]),
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


def _save_to_file(name: str, save: bool, key: str = None, folder: str = None):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base
        plt.savefig(f"{constants.PLOTS_PATH}/{fname}")
