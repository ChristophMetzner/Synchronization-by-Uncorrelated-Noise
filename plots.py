import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab


def plot_noise(data, save: bool = True, prefix: str = None):
    """ Plot External Noise of 1st population. """
    ext_signal_1_mean = data['input_mean_1']
    ext_signal_1_sigma = data['input_sigma_1']

    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(111)
    ax.set_title("External Input Signal to first population")
    ax.set_xlabel("Time in ms")
    ax.set_ylabel("Voltage in mV")
    ax.plot(ext_signal_1_mean)
    ax.plot(ext_signal_1_sigma)

    _save_to_file("noise", save=save, folder=prefix)


def plot_summed_voltage(title: str, key: str, lfp1, lfp2, duration, dt, prefix: str = None, save: bool = True):
    t = np.linspace(0, duration, int(duration / dt))

    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Elapsed Time in ms")
    ax.set_ylabel("Voltage")
    ax.plot(t, lfp1, '0.75')
    ax.plot(t, lfp2, '0.25')

    _save_to_file("summed_voltage", save, key, prefix)


def psd(title: str, key: str, lfp1, lfp2, duration: int = 300, dt: float = 1.0, folder: str = None, save: bool = True):
    """
    Plots the Power Spectral Density.
    """
    print("Generate PSD plot ...")

    timepoints = int((duration / dt) / 2)
    fs = 1. / dt
    psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Density")
    ax.plot(freqs * 1000, psd1, '0.25', linewidth=3.0, c='b')
    ax.plot(freqs * 1000, psd2, '0.75', linewidth=3.0, c='y')
    ax.set_xlim([0, 80])

    _save_to_file("psd", save, key, folder)


def plot_raster(s_e, s_i, x_left: int, x_right: int, save: bool = True, key: str = "", folder: str = None):
    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(111)

    ax.set_title("Raster Plot")
    ax.set_xlabel('Time in ms')
    ax.set_ylabel('Neuron index')
    ax.plot(s_e[1] * 1000, s_e[0], 'k.', c='darkgray')
    ax.plot(s_i[1] * 1000, s_i[0] + 1000, 'k.', c='dimgray')
    ax.set_xlim(left=x_left, right=x_right)

    _save_to_file("raster", save=save, key=key, folder=folder)


def _save_to_file(name: str, save: bool, key: str = None, folder: str = None):
    if save:
        base = f"{name}-{key}.png" if key else f"{name}.png"
        fname = f"{folder}/{base}" if folder else base
        plt.savefig(f"plots/{fname}")
