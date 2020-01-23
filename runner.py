import numpy as np
import sys
import pickle
import os

import params
import network as net
import constants

from utils import generate_ou_input, update

sys.path.insert(1, '..')  # allow parent modules to be imported
sys.path.insert(1, '../..')  # allow parent modules to be imported
sys.path.insert(1, '../../..')  # allow parent modules to be imported


def run(file_name: str = None, experiment_name: str = None, modified_params: dict = None):
    """
    Runs the simulation and persists the model with `basename`.

    :param modified_params: dict that updates the default params.
    :param file_name: file name of model without extension.
    :param experiment_name: used as parent folder to group by experiment.
    """
    # use as default the parameters from file params.py
    # if not specified else below
    p = params.get_params()

    p['runtime'] = 500.
    p['net_dt'] = 0.05
    p['min_dt'] = p['net_dt']
    p['t_ref'] = 0.0

    # if False --> ou process starts at X0
    p['ou_stationary'] = True

    # noise params for mu
    p['ou_mu'] = {
        'ou_X0': 0.,
        'ou_mean': 3.0,
        'ou_sigma': .5,
        'ou_tau': 1.
    }

    # noise params for sigma
    p['ou_sigma'] = {
        'ou_X0': 0.,
        'ou_mean': 1.0,
        'ou_sigma': 0.2,
        'ou_tau': 1.
    }

    if modified_params:
        # Update params with modified_params
        p = update(p, modified_params)

    # TODO: check if model already exist, if it should not be overwritten, skip this run!

    # external time trace used for generating input and plotting
    # if time step is unequal to model_dt input gets interpolated
    steps = int(p['runtime'] / p['min_dt'])
    t_ext = np.linspace(0., p['runtime'], steps + 1)

    # time trace computed with min_dt
    p['t_ext'] = t_ext

    # mu = const, sigma = OU process
    mu_ext1 = generate_ou_input(p['runtime'], p['min_dt'], p['ou_stationary'], p['ou_mu'])
    mu_ext2 = generate_ou_input(p['runtime'], p['min_dt'], p['ou_stationary'], p['ou_mu'])

    sigma_ext1 = generate_ou_input(p['runtime'], p['min_dt'], p['ou_stationary'], p['ou_sigma'])
    sigma_ext2 = generate_ou_input(p['runtime'], p['min_dt'], p['ou_stationary'], p['ou_sigma'])

    # collect ext input for model wrappers
    ext_input0 = [mu_ext1, sigma_ext1, mu_ext2, sigma_ext2]

    # saving results in global results dict
    results = dict()
    results['input_mean_1'] = mu_ext1
    results['input_sigma_1'] = sigma_ext1

    results['input_mean_2'] = mu_ext2
    results['input_sigma_2'] = sigma_ext2

    results['model_results'] = dict()

    # save parameter set for later analysis
    results['params'] = p

    # brian network sim
    # ext_input = interpolate_input(ext_input0, params, 'net')
    results['model_results']['net'] = net.network_sim(ext_input0, p)

    if experiment_name:
        try:
            os.mkdir(f"{constants.MODELS_PATH}/" + experiment_name)
        except FileExistsError:
            # ignore
            pass

    base_path = f"{constants.MODELS_PATH}/{experiment_name}" if experiment_name else constants.MODELS_PATH

    if file_name:
        with open(f"{base_path}/{file_name}.pkl", 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results
