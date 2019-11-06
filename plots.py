import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab


def plot(data, duration, dt):
    psd(data, duration, dt)


def psd(data, duration: int = 300, dt: float = 1.0, prefix: str = None):
    """
    Plots the Power Spectral Density.

    :param data:
    :param duration:
    :param prefix:
    :param dt:
    """
    print("Generate PSD plot ...")

    v_e1 = data['model_results']['net']['v_all_neurons_e']
    v_e2 = data['model_results']['net']['v_all_neurons_e2']

    # sum up voltages of excitatory Neuron Groups
    lfp1 = np.sum(v_e1, axis=0) / 1000
    lfp2 = np.sum(v_e2, axis=0) / 1000

    timepoints = int((duration / dt) / 2)
    fs = 1. / dt
    psd1, freqs = mlab.psd(lfp1, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)
    psd2, _ = mlab.psd(lfp2, NFFT=int(timepoints), Fs=fs, noverlap=0, window=mlab.window_none)

    psd1[0] = 0.0
    psd2[0] = 0.0

    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(111)
    ax.set_title("Power Spectral Density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Density")
    ax.plot(freqs * 1000, psd1, '0.25', linewidth=3.0, c='b')
    ax.plot(freqs * 1000, psd2, '0.75', linewidth=3.0, c='y')
    ax.set_xlim([0, 80])

    base = "psd.png"
    fname = f"{prefix}-{base}" if prefix else base
    plt.savefig(f"plots/{fname}")

    print("Saved PSD plot to file")

# TODO: add here all plots from notebooks
