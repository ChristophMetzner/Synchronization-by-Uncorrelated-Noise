import numpy as np

from typing import Tuple


def lfp(model: dict, duration: int = None, skip: int = None, population: int = 1) -> Tuple:
    if duration:
        duration = int(duration)

    N_e = model['params']['N_e']
    N_i = model['params']['N_i']

    if population == 1:
        v_e = model['model_results']['net']['v_all_neurons_e'][:skip][:duration]
        v_i = model['model_results']['net']['v_all_neurons_i1'][:skip][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2

    elif population == 2:
        v_e = model['model_results']['net']['v_all_neurons_e2'][:skip][:duration]
        v_i = model['model_results']['net']['v_all_neurons_i2'][:skip][:duration]
        lfp1 = _lfp(v_e, N_e)
        lfp2 = _lfp(v_i, N_i)
        return lfp1, lfp2


def lfp_single_net(model, population: int = 1):
    N_e = model['params']['N_e']
    N_i = model['params']['N_i']

    if population == 1:
        v_e = model['model_results']['net']['v_all_neurons_e']
        v_i = model['model_results']['net']['v_all_neurons_i1']
    else:
        v_e = model['model_results']['net']['v_all_neurons_e2']
        v_i = model['model_results']['net']['v_all_neurons_i2']

    # TODO: verify correctness of averaging the average!
    return (np.sum(v_e, axis=0) / N_e + np.sum(v_i, axis=0) / N_i) / 2


def lfp_nets(model):
    return lfp_single_net(model, population=1), lfp_single_net(model, population=2)


def _lfp(v, N):
    return np.sum(v, axis=0) / N
