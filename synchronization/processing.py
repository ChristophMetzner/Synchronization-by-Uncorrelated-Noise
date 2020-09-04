import numpy as np

from typing import Tuple

from brian2 import ms, mV
from scipy.signal import hilbert
from matplotlib import mlab
from scipy.signal.filter_design import butter
from scipy.signal.signaltools import filtfilt

"""
This module contains a diverse set of functions that process the generated neuronal data of the models.

The following areas are included:
* LFP
* PSD
* filter
* phase locking
* phase synchronization

"""


def lfp(
    model: dict, duration: int = None, skip: int = None, population: int = 1
) -> Tuple:
    if duration:
        duration = int(duration)

    N_e = model["N_e"]
    N_i = model["N_i"]

    if population == 1:
        v_e = model["v_all_neurons_e"][:, skip:][:duration]
        v_i = model["v_all_neurons_i1"][:, skip:][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2

    elif population == 2:
        v_e = model["v_all_neurons_e2"][:, skip:][:duration]
        v_i = model["v_all_neurons_i2"][:, skip:][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2


def lfp_single_net(
    model: dict, population: int = 1, skip: int = None, gamma_filter: bool = False
):
    """ Calculates local field potential (LFP) of a single network.

    LFP is approximated by taking the average over the membrane voltages of all neurons in the network.
    LFP := 1/N sum(neurons_v)

    :param model: model
    :type model: dict
    :param population: specifies network, defaults to 1
    :type population: int, optional
    :param skip: skips the first x ms, defaults to None
    :type skip: int, optional
    :param gamma_filter: if True applies band pass filter in gamma range.
    :return: lfp over time.
    :rtype: ndarray
    """
    model_ei = "model_EI" not in model or model["model_EI"]
    if population == 1:
        i_identifier = "v_all_neurons_i1"
        e_identifier = "v_all_neurons_e"
    else:
        i_identifier = "v_all_neurons_i2"
        e_identifier = "v_all_neurons_e2"

    v_i = model[i_identifier][:, skip:]
    if model_ei:
        v_e = model[e_identifier][:, skip:]

        count = model["N_i"] + model["N_e"]
        v = np.vstack((v_e, v_i))
    else:
        count = model["N_i"]
        v = v_i

    lfp = np.sum(v, axis=0) / count
    if gamma_filter:
        lfp = filter(lfp, lowcut=30, highcut=120)

    return lfp


def lfp_nets(model, skip: int = None, gamma_filter: bool = False):
    return (
        lfp_single_net(model, population=1, skip=skip, gamma_filter=gamma_filter),
        lfp_single_net(model, population=2, skip=skip, gamma_filter=gamma_filter),
    )


def _lfp(v, N: int) -> np.ndarray:
    """Calculates local field potential of `N` neurons `v`.

    :param v: array of membrane voltage over time of `N` neurons.
    :type v: ndarray
    :param N: Number of neurons.
    :type N: int
    :return: local field potential.
    :rtype: np.ndarray
    """
    return np.sum(v, axis=0) / N


def band_power(model, network: int = 1, granularity: int = 1, skip: int = None):
    lfp = lfp_single_net(model, population=network, skip=skip)

    runtime_ = model["runtime"] - skip if skip else model["runtime"]

    dt = 1.0
    timepoints = int((runtime_ / dt) / granularity)
    fs = 1.0 / dt

    psd, freqs = mlab.psd(
        lfp, NFFT=timepoints, Fs=fs, noverlap=0, window=mlab.window_none
    )
    psd[0] = 0.0
    freqs = freqs * 1000
    freqs = [int(freq) for freq in freqs]

    max_amplitude = psd.max()
    peak_freq = freqs[psd.argmax()]

    return max_amplitude, peak_freq


def band_power_raw(signal):
    dt = 1.0
    NFFT = len(signal)
    fs = 1.0 / dt

    psd, freqs = mlab.psd(signal, NFFT=NFFT, Fs=fs, noverlap=0, window=mlab.window_none)

    psd[0] = 0.0
    freqs = freqs * 1000
    freqs = [int(freq) for freq in freqs]

    max_amplitude = psd.max()
    peak_freq = freqs[psd.argmax()]

    return psd, freqs, max_amplitude, peak_freq


def hilphase(y1, y2):
    sig1_hill = hilbert(y1)
    sig2_hill = hilbert(y2)
    pdt = np.inner(sig1_hill, np.conj(sig2_hill)) / (
        np.sqrt(
            np.inner(sig1_hill, np.conj(sig1_hill))
            * np.inner(sig2_hill, np.conj(sig2_hill))
        )
    )
    phase = np.angle(pdt)
    return phase


def phase(signal):
    hil = hilbert(signal)
    return np.angle(hil)


def phase_difference(y1, y2, unwrap: bool = True) -> np.ndarray:
    """
    Calculates the mean phase coherence.

    R = | 1/N sum e^i(phi(t_j) - phi(t_k)) |

    Implements equation (21) from http://www.scholarpedia.org/article/Measures_of_neuronal_signal_synchrony.
    """
    sig1_hill = hilbert(y1)
    sig2_hill = hilbert(y2)

    # Get angle and unwrap to remove discontinuities
    angl_sig1 = np.angle(sig1_hill)
    angl_sig2 = np.angle(sig2_hill)

    if unwrap:
        angl_sig1 = np.unwrap(angl_sig1)
        angl_sig2 = np.unwrap(angl_sig2)

    # Calculate phase difference
    inst_phase_diff = angl_sig1 - angl_sig2
    return inst_phase_diff


def mean_phase_coherence(y1, y2, unwrap: bool = True) -> float:
    """
    Calculates the mean phase coherence.

    R = | 1/N sum e^i(phi(t_j) - phi(t_k)) |

    Implements equation (21) from http://www.scholarpedia.org/article/Measures_of_neuronal_signal_synchrony.
    """
    # phase differences at each time step.
    diffs = phase_difference(y1, y2, unwrap=unwrap)

    # complex form by projecting onto unit circle.
    complex_phase_diff = [np.exp(1j * phase_diff) for phase_diff in diffs]

    # absolute value of average of complex phase differences.
    phase_coherence_index = np.abs(sum(complex_phase_diff) / len(complex_phase_diff))

    return phase_coherence_index


def mean_phase_coherence_net(model: dict) -> float:
    """
    Calculates Mean Phase Coherence between networks of  given `model`.

    :param model: input model.
    :return: MPC index.
    """
    signals = lfp_nets(model, skip=200, gamma_filter=True)
    mpc = mean_phase_coherence(signals[0], signals[1])
    return mpc


def order_parameter_over_time(signals):
    """
    Computes the local order parameter / phase synchronization over time according to Meng et al. 2018.

    :param signals: array of signals of same length.
    :return: array of local order parameter value for each time step.
    """
    signals = [s - np.mean(s) for s in signals]

    phases = [np.unwrap(np.angle(hilbert(s))) for s in signals]
    complex_phases = [np.exp(1j * phase) for phase in phases]

    avg = sum(complex_phases) / len(complex_phases)
    phi = np.abs(avg)
    return phi


def phase_synchronization(signals):
    """
    Computes the average phase synchronization.

    :param signals:
    :return:
    """
    # zero mean
    signals = [s - np.mean(s) for s in signals]

    # compute analytical signal by using Hilbert transformation
    # get angle to get phase
    # then transform to complex number so that we can average it
    phases = [np.unwrap(np.angle(hilbert(s))) for s in signals]
    complex_phases = [np.exp(1j * phase) for phase in phases]

    # take the average (sum up all complex phases and divide by number of phases)
    avg = sum(complex_phases) / len(complex_phases)

    # take the length of the vector
    # it tells us about the lag of the phases
    # length -> 0 => high lag, length -> 1 => no lag, perfect synchronization
    phi = np.abs(avg)

    # mean of phi as it is currently phi over time
    return np.mean(phi)


def filter(
    signal, fs: int = 1000, lowcut: int = 30, highcut: int = 120, order: int = 2
):
    """ Applies Band Pass Filter to `signal`.

    :param signal: input signal
    :param fs: sampling frequency, defaults to 1000
    :type fs: int, optional
    :param lowcut: lowcut frequency, defaults to 10
    :type lowcut: int, optional
    :param highcut: lowcut frequency, defaults to 80
    :type highcut: int, optional
    :param order: butter filter order, defaults to 2
    :type order: int, optional
    :return: filtered signal.
    :rtype: ndarray
    """
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    filtered = filtfilt(b, a, signal)
    return filtered


def filter_signals(signals) -> list:
    return [filter(s) for s in signals]


def group_neurons(v, spike_t: int, window, t_start=5, t_end=5, thr=-70):
    """
    Simple spike detection algorithm for given time window.

    Ideas:
    * Could also use recorded SpikeMonitor.
    """

    window = window if window else (120, 170)

    first = []
    second = []

    for n_idx, n in enumerate(v):
        spiked = False
        for mv_idx, mv in enumerate(n[window[0] : window[1]]):
            if mv <= thr and spike_t + t_end >= mv_idx >= spike_t - t_start:
                spiked = True

        if spiked:
            first.append(n)
        else:
            second.append(n)

    print(len(first))
    print(len(second))
    return first, second


def inter_spike_intervals(spike_trains) -> list:
    """
    Computes the inter spike intervals for a list of `spike_trains`.

    Each spike train contains spiking times of a neuron.

    :param spike_trains: list of spike times.
    :return: list, spike intervals.
    """
    isi = []
    for n in spike_trains:
        end = len(n) - 1
        for idx, t in enumerate(n):
            if idx != end:
                t_next = n[idx + 1]
                isi.append(abs(t_next - t) / ms)

    return isi


def get_first_spike(v, t, window):
    """
    Returns first spike of first neuron in `v` in time `window`.

    First spike hit once membrane voltage is below or equal to -70 mV.

    :param v: voltage trace of neurons.
    :param t: recorded time steps.
    :param window: (start, end)
    :return:
    """
    spikes = [
        (x, y)
        for (x, y) in zip(v[0][window[0] : window[1]], t[window[0] : window[1]])
        if x <= -70
    ]

    if spikes:
        v_s, t_s = spikes[0]
        return v_s, t_s
    else:
        return None


def spike_participation(v, peaks, width: int = 3, threshold=-47):
    """
    Compute spike participation on neuron and on network level.

    :param v: membrane voltage traces.
    :param peaks: peak data structure.
    :param threshold:
    :param width:
    :return: (participation per neuron, participation per peak)
    """
    peak_count = len(peaks[0])
    participation_n = {}

    for idx, v_n in enumerate(v):
        participation_n[idx] = 0

    participation_p = {}

    for peak in peaks[0]:
        participation_p[peak] = 0

        for idx, v_n in enumerate(v):
            v_p = v_n[int(peak) - width : int(peak) + width]
            if any([s for s in v_p if s >= threshold]):
                participation_p[peak] += 1
                participation_n[idx] += 1

    for k in participation_n.keys():
        participation_n[k] = participation_n[k] / peak_count

    for p in participation_p.keys():
        participation_p[p] = participation_p[p] / len(v)

    return participation_n, participation_p


def filter_inter_spike_intervals(isi):
    """
    Removes outlier from array of inter spike intervals.

    Currently we use just a crude heuristic as it suffices to remove extremely high intervals.

    :param isi: array.
    :return: filtered array.
    """
    isi_avg = np.average(isi)
    return [e for e in isi if e < isi_avg + 40]


def set_positive_spikes(dt_sim, rates_dt, v_trace, net_spikes):
    """
    Sets for each spike in the `v_trace` voltage to positive value.

    :param dt_sim: step size of simulation.
    :param rates_dt: step size of recording monitor.
    :param v_trace: voltage trace of monitor.
    :param net_spikes: spikes from SpikeMonitor.
    :return: v_trace with positive voltage values at spike time.
    """
    # This can only be done properly if recording step size equals simulation step size.
    if dt_sim == rates_dt:
        # Brian's groups are read-only, therefore we assign it to a new variable.
        v_trace = v_trace.v

        for idx, t in zip(net_spikes[0], net_spikes[1]):
            # Amount of simulation steps to get to time t.
            # Gives us the correct index for neuron trace array.
            i = int(t / dt_sim)

            # Set to ~ 0-40 mV.
            v_trace[idx][i] = 40 * mV

        return v_trace
