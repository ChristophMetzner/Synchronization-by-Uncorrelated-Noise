import numpy as np
import sys

import analysis
import params_net
import pickle
import network as net

from utils import generate_ou_input

sys.path.insert(1, '..')  # allow parent modules to be imported
sys.path.insert(1, '../..')  # allow parent modules to be imported
sys.path.insert(1, '../../..')  # allow parent modules to be imported


# use the following in IPython for qt plots: %matplotlib qt

def run(base_name: str, modified_params: dict = None):
    """
    Runs the simulation and persists the model with `basename`.

    :param modified_params: dict that updates the default params.
    :param base_name: file name of model without extension.
    :return:
    """
    # use as default the parameters from file params_net.py
    # if not specified else below
    params = params_net.get_params()

    params['runtime'] = 3200.
    params['net_dt'] = 0.05
    params['min_dt'] = params['net_dt']
    params['t_ref'] = 0.0

    # if False --> ou process starts at X0
    params['ou_stationary'] = True

    # noise params for mu
    params['ou_mu'] = {
        'ou_X0': 0.,
        'ou_mean': .5,
        'ou_sigma': .5,
        'ou_tau': 50.
    }

    # noise params for sigma
    params['ou_sigma'] = {
        'ou_X0': 0.,
        'ou_mean': 2.0,
        'ou_sigma': 0.2,
        'ou_tau': 1.
    }

    if modified_params:
        # Update params with modified_params
        params.update(modified_params)

    # external time trace used for generating input and plotting
    # if time step is unequal to model_dt input gets interpolated
    steps = int(params['runtime'] / params['min_dt'])
    t_ext = np.linspace(0., params['runtime'], steps + 1)

    # time trace computed with min_dt
    params['t_ext'] = t_ext

    # mu = const, sigma = OU process
    mu_ext1 = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_mu'])
    mu_ext2 = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_mu'])

    sigma_ext1 = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_sigma'])
    sigma_ext2 = generate_ou_input(params['runtime'], params['min_dt'], params['ou_stationary'], params['ou_sigma'])

    # collect ext input for model wrappers
    ext_input0 = [mu_ext1, sigma_ext1, mu_ext2, sigma_ext2]

    # saving results in global results dict
    results = dict()

    results['input_mean_1'] = mu_ext1
    results['input_sigma_1'] = sigma_ext1

    results['input_mean_2'] = mu_ext2
    results['input_sigma_2'] = sigma_ext2

    results['model_results'] = dict()

    # brian network sim
    # ext_input = interpolate_input(ext_input0, params, 'net')
    results['model_results']['net'] = net.network_sim(ext_input0, params)

    with open(f"models/{base_name}.pkl", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


if __name__ == '__main__':
    results = run("base_low_noise")
    analysis.analyze_model(results, 'base_low_noise')

    # Sets the connection strength of synapses connecting the networks to 0.
    # Thus the networks are isolated and do not affect each other.
    # run("decoupled", {
    #     "J_ppee": 0.0
    # })

    factor = 10.

    # # low synaptic strength i -> e
    # run("low_synaptic_strength", {
    #     # e -> e
    #     "J_etoe": 0.01 * factor,
    #     # e -> i
    #     "J_etoi": .05 * factor,
    #     # low synaptic strength between i -> e
    #     "J_itoe": 0.3 * factor,
    # })
    #
    # # mid synaptic strength i -> e
    # run("mid_synaptic_strength", {
    #     # e -> e
    #     "J_etoe": 0.01 * factor,
    #     # e -> i
    #     "J_etoi": .05 * factor,
    #     # mid synaptic strength between i -> e
    #     "J_itoe": 0.7 * factor,
    # })
    #
    # # explore effects of synaptic strength
    # for i_to_e_strength in np.arange(0, 2, 0.1):
    #     i_to_e_strength = np.around(i_to_e_strength, 1)
    #     run(f"synaptic_strength_{i_to_e_strength}", {
    #         "J_etoe": 0.01 * factor,
    #         "J_etoi": .05 * factor,
    #         "J_itoe": i_to_e_strength * factor,
    #     })

    import sys

    sys.exit()
