import numpy as np

from typing import Tuple
from scipy.signal import hilbert
from matplotlib import mlab
from scipy.signal.filter_design import butter
from scipy.signal.signaltools import filtfilt


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


def lfp_single_net(model: dict, population: int = 1, skip: int = None):
    """ Calculates local field potential (LFP) of a single network.

    LFP is approximated by taking the average over the membrane voltages of all neurons in the network.
    LFP := 1/N sum(neurons_v)

    :param model: model
    :type model: dict
    :param population: specifies network, defaults to 1
    :type population: int, optional
    :param skip: skips the first x ms, defaults to None
    :type skip: int, optional
    :return: lfp over time.
    :rtype: ndarray
    """
    model_EI = model["model_EI"]
    if population == 1:
        i_identifier = "v_all_neurons_i1"
        e_identifier = "v_all_neurons_e"
    else:
        i_identifier = "v_all_neurons_i2"
        e_identifier = "v_all_neurons_e2"

    count = model["N_i"]
    v_i = model[i_identifier][:, skip:]
    if model_EI:
        v_e = model[e_identifier][:, skip:]
    else:
        v_e = None
        count += model["N_e"]

    v = v_i if v_e is None else np.vstack((v_e, v_i))
    return np.sum(v, axis=0) / count


def lfp_nets(model, skip: int = None):
    return (
        lfp_single_net(model, population=1, skip=skip),
        lfp_single_net(model, population=2, skip=skip),
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
    complex_phase_diff = [np.exp(1j * phase) for phase in diffs]

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


def filter(signal, fs: int = 1000, lowcut: int = 10, highcut: int = 80, order: int = 2):
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


def phase_locking(signals):
    """ 
    # TODO: Compute phase locking!
    """
    # std deviation of phase differences
    raise NotImplementedError
