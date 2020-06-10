import numpy as np

from typing import Tuple
from scipy.signal import hilbert
from matplotlib import mlab


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


def lfp_single_net(model, population: int = 1, skip: int = None):
    N_e = model["N_e"]
    N_i = model["N_i"]

    if population == 1:
        v_e = model["v_all_neurons_e"][:, skip:]
        v_i = model["v_all_neurons_i1"][:, skip:]
    else:
        v_e = model["v_all_neurons_e2"][:, skip:]
        v_i = model["v_all_neurons_i2"][:, skip:]

    # TODO: verify correctness of averaging the average!
    return (np.sum(v_e, axis=0) / N_e + np.sum(v_i, axis=0) / N_i) / 2


def lfp_nets(model, skip: int = None):
    return (
        lfp_single_net(model, population=1, skip=skip),
        lfp_single_net(model, population=2, skip=skip),
    )


def _lfp(v, N):
    return np.sum(v, axis=0) / N


def band_power(model, network: int = 1):
    lfp = lfp_single_net(model, population=network)

    runtime_ = model["runtime"]
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
    return np.unwrap(np.angle(hil))


def mean_phase_coherence(y1, y2) -> float:
    """
    Calculates the mean phase coherence.

    R = | 1/N sum e^i(phi(t_j) - phi(t_k)) |

    Implements equation (21) from http://www.scholarpedia.org/article/Measures_of_neuronal_signal_synchrony.
    """
    sig1_hill = hilbert(y1)
    sig2_hill = hilbert(y2)

    # get angle and unwrap to remove discontinuities
    phase_y1 = np.unwrap(np.angle(sig1_hill))
    phase_y2 = np.unwrap(np.angle(sig2_hill))

    # calculate phase difference
    inst_phase_diff = phase_y1 - phase_y2

    # complex form by projecting onto unit circle
    complex_phase_diff = [np.exp(1j * phase) for phase in inst_phase_diff]

    # absolute value of average of complex phase differences.
    phase_coherence_index = np.abs(sum(complex_phase_diff) / len(complex_phase_diff))

    return phase_coherence_index


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
    # it tells us about the consistency of the phases
    # length -> 0 => low consistency, length -> 1 => high consistency
    phi = np.abs(avg)

    # mean of phi as it is currently phi over time
    return np.mean(phi)
