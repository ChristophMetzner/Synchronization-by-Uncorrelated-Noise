import numpy as np

from typing import Tuple

from scipy.signal import hilbert


def lfp(
    model: dict, duration: int = None, skip: int = None, population: int = 1
) -> Tuple:
    if duration:
        duration = int(duration)

    N_e = model["params"]["N_e"]
    N_i = model["params"]["N_i"]

    if population == 1:
        v_e = model["model_results"]["net"]["v_all_neurons_e"][:skip][:duration]
        v_i = model["model_results"]["net"]["v_all_neurons_i1"][:skip][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2

    elif population == 2:
        v_e = model["model_results"]["net"]["v_all_neurons_e2"][:skip][:duration]
        v_i = model["model_results"]["net"]["v_all_neurons_i2"][:skip][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2


def lfp_single_net(model, population: int = 1):
    N_e = model["params"]["N_e"]
    N_i = model["params"]["N_i"]

    if population == 1:
        v_e = model["model_results"]["net"]["v_all_neurons_e"]
        v_i = model["model_results"]["net"]["v_all_neurons_i1"]
    else:
        v_e = model["model_results"]["net"]["v_all_neurons_e2"]
        v_i = model["model_results"]["net"]["v_all_neurons_i2"]

    # TODO: verify correctness of averaging the average!
    return (np.sum(v_e, axis=0) / N_e + np.sum(v_i, axis=0) / N_i) / 2


def lfp_nets(model):
    return lfp_single_net(model, population=1), lfp_single_net(model, population=2)


def _lfp(v, N):
    return np.sum(v, axis=0) / N


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


def hilphase_2(y1, y2):
    sig1_hill = hilbert(y1)
    sig2_hill = hilbert(y2)

    phase_y1 = np.unwrap(np.angle(sig1_hill))
    phase_y2 = np.unwrap(np.angle(sig2_hill))

    Inst_phase_diff = phase_y1 - phase_y2
    avg_phase = np.average(Inst_phase_diff)

    return Inst_phase_diff, avg_phase


def phase_locking_value(signals):
    # zero mean
    signals = [s - np.mean(s) for s in signals]

    # compute analytical signal and take angle to get phase
    phases = [np.angle(hilbert(s)) for s in signals]

    # complex phase
    complex_phase = [np.exp(1j * phase) for phase in phases]

    # take the average (sum up all complex phases and divide by number of phases)
    avg = np.average(complex_phase)

    # take the length of the vector
    # it tells us about the consistency of the phases
    # length -> 0 => low consistency, length -> 1 => high consistency
    phi = np.abs(avg)

    # mean of phi as it is currently phi over time
    return np.mean(phi)
