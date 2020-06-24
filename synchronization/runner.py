import numpy as np
import sys
import pickle
import os

from synchronization import params
from synchronization import network as net
from synchronization import constants
from synchronization.utils import generate_ou_input, update
from synchronization import processing


def run(
    file_name: str = None, experiment_name: str = None, modified_params: dict = None
):
    """
    Runs the simulation and persists the model with `basename`.

    :param modified_params: dict that updates the default params.
    :param file_name: file name of model without extension.
    :param experiment_name: used as parent folder to group by experiment.
    """
    # use as default the parameters from file params.py
    # if not specified else below
    p = params.get_params()

    p["runtime"] = 500.0
    p["net_dt"] = 0.05  # ms
    p["min_dt"] = p["net_dt"]
    p["t_ref"] = 0.0

    # if False --> ou process starts at X0
    p["ou_stationary"] = True

    # noise params for OU mu
    p["ou_mu_X0"] = [0.0, 0.0]
    p["ou_mu_mean"] = [3.0, 3.0]
    p["ou_mu_sigma"] = [0.5, 0.5]
    p["ou_mu_tau"] = [1.0, 1.0]

    # noise params for OU sigma
    p["ou_sigma_X0"] = [0.0, 0.0]
    p["ou_sigma_mean"] = [1.0, 1.0]
    p["ou_sigma_sigma"] = [0.2, 0.2]
    p["ou_sigma_tau"] = [1.0, 1.0]

    if modified_params:
        # Update params with modified_params
        p = update(p, modified_params)

    # TODO: check if model already exist, if it should not be overwritten, skip this run!

    # external time trace used for generating input and plotting
    # if time step is unequal to model_dt input gets interpolated
    steps = int(p["runtime"] / p["min_dt"])
    t_ext = np.linspace(0.0, p["runtime"], steps + 1)

    # time trace computed with min_dt
    p["t_ext"] = t_ext

    # mu = const, sigma = OU process
    mu_ext1 = generate_ou_input(
        p["runtime"],
        p["min_dt"],
        p["ou_stationary"],
        p["ou_mu_X0"][0],
        p["ou_mu_tau"][0],
        p["ou_mu_sigma"][0],
        p["ou_mu_mean"][0],
    )
    mu_ext2 = generate_ou_input(
        p["runtime"],
        p["min_dt"],
        p["ou_stationary"],
        p["ou_mu_X0"][1],
        p["ou_mu_tau"][1],
        p["ou_mu_sigma"][1],
        p["ou_mu_mean"][1],
    )

    sigma_ext1 = generate_ou_input(
        p["runtime"],
        p["min_dt"],
        p["ou_stationary"],
        p["ou_sigma_X0"][0],
        p["ou_sigma_tau"][0],
        p["ou_sigma_sigma"][0],
        p["ou_sigma_mean"][0],
    )
    sigma_ext2 = generate_ou_input(
        p["runtime"],
        p["min_dt"],
        p["ou_stationary"],
        p["ou_sigma_X0"][1],
        p["ou_sigma_tau"][1],
        p["ou_sigma_sigma"][1],
        p["ou_sigma_mean"][1],
    )

    # collect ext input for model wrappers
    ext_input0 = [mu_ext1, sigma_ext1, mu_ext2, sigma_ext2]

    # saving results in global results dict
    results = {
        "input_mean_1": mu_ext1,
        "input_sigma_1": sigma_ext1,
        "input_mean_2": mu_ext2,
        "input_sigma_2": sigma_ext2,
    }

    results.update(p)

    # ext_input = interpolate_input(ext_input0, params, 'net')
    network_results = net.network_sim(ext_input0, p)
    results.update(network_results)

    if file_name:
        if experiment_name:
            try:
                os.mkdir(f"{constants.MODELS_PATH}/" + experiment_name)
            except FileExistsError:
                # ignore
                pass

        base_path = (
            f"{constants.MODELS_PATH}/{experiment_name}"
            if experiment_name
            else constants.MODELS_PATH
        )

        with open(f"{base_path}/{file_name}.pkl", "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def run_in_mopet(params) -> dict:
    """
    Run method wrapper that runs network in Mopet.

    :param params:
    :return: dict
    """
    results = run(modified_params=params)
    results = post_processing(results)

    # Remove types that are not supported yet by Mopet
    remove = [k for k in results if results[k] is None or isinstance(results[k], str)]
    # print(f"Removing keys {remove} containing NoneType from dictionary to avoid conflicts with Mopet")
    for k in remove:
        del results[k]

    return results


def post_processing(results):
    # skip first 200 ms
    skip = 200

    print("Starting Aggregation ...")
    
    max_amplitude, peak_freq = processing.band_power(results, skip=skip)
    results["max_amplitude"] = max_amplitude
    results["peak_freq"] = peak_freq

    max_amplitude, peak_freq = processing.band_power(results, network=2, skip=skip)
    results["max_amplitude_2"] = max_amplitude
    results["peak_freq_2"] = peak_freq

    results["freq_diff"] = abs(results["peak_freq"] - results["peak_freq_2"])

    ratio = 0
    if results["peak_freq"] > 0 and results["peak_freq_2"] > 0:
        if results["peak_freq"] >= results["peak_freq_2"]:
            ratio = results["peak_freq_2"] / results["peak_freq"]
        else:
            ratio = results["peak_freq"] / results["peak_freq_2"]

    results["freq_ratio"] = ratio

    lfps = processing.lfp_nets(results, skip=skip)
    f_lfps = processing.filter(lfps, lowcut=30, highcut=80)
    global_order_parameter = processing.order_parameter_over_time(f_lfps)
    total_value = np.mean(global_order_parameter)

    neurons_net_1 = np.vstack(
        (results["v_all_neurons_e"][:, 200:], results["v_all_neurons_i1"][:, 200:])
    )
    neurons_net_2 = np.vstack(
        (results["v_all_neurons_e2"][:, 200:], results["v_all_neurons_i2"][:, 200:])
    )

    f_neurons_net_1 = [processing.filter(n, lowcut=30, highcut=80) for n in neurons_net_1]
    f_neurons_net_2 = [processing.filter(n, lowcut=30, highcut=80) for n in neurons_net_2]

    plv_net_1 = np.mean(processing.order_parameter_over_time(f_neurons_net_1))
    plv_net_2 = np.mean(processing.order_parameter_over_time(f_neurons_net_2))

    results["phase_synchronization_over_time"] = global_order_parameter
    results["phase_synchronization"] = total_value
    results["plv_net_1"] = plv_net_1
    results["plv_net_2"] = plv_net_2

    print("Finished aggregation.")

    return results