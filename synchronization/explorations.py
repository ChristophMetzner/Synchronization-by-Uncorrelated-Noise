import numpy as np

from mopet import mopet
from synchronization import runner, constants


def synaptic_weight_exploration(remote: bool = True):
    default_params = {
        "runtime": 5000.0,
        "J_itoi": 3.0,
        "J_etoe": 0.4,
        "J_etoi": 0.9,
        "J_itoe": 7.0,
        "N_e": 1000,
        "N_i": 250,
        "ou_enabled": [False, False],
        "poisson_enabled": [True, True],
        "poisson_variance": 1.0,
        "poisson_p": 0.875,
        "poisson_size": 800,
        "poisson_mean_input": 200,
        "J_ppee": 0.0,
        "J_ppei": 0.0,
        "const_delay": 0.2,
        "N_pop": 2,
        "p_etoe": 0.1,
        "p_etoi": 0.4,
        "p_itoe": 0.1,
        "p_itoi": 0.4,
    }

    params = {"J_itoe": np.arange(1, 11, 0.5), "J_etoi": np.arange(1, 11, 0.5)}

    ex = mopet.Exploration(
        runner.run_in_mopet,
        explore_params=params,
        default_params=default_params,
        exploration_name="synaptic_weights",
        hdf_filename=f"{constants.get_base_path(remote)}/synaptic_weights.h5",
    )

    ex.run()


def mean_noise_exploration(remote: bool = True):
    default_params = {
        "runtime": 5000.0,
        "J_itoi": 3.0,
        "J_etoe": 0.4,
        "J_etoi": 0.9,
        "J_itoe": 7.0,
        "ou_enabled": [False, False],
        "poisson_enabled": [True, True],
        "poisson_variance": 1.0,
        "poisson_p": 0.875,
        "poisson_size": 800,
        "J_ppee": 2.0,
        "J_ppei": 2.0,
        "const_delay": 0.2,
        "N_pop": 2,
    }

    params = {
        "poisson_variance": [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "poisson_mean_input": np.arange(50, 210, 10),
    }

    ex = mopet.Exploration(
        runner.run_in_mopet,
        explore_params=params,
        default_params=default_params,
        exploration_name="mean_noise_input_4",
        hdf_filename=f"{constants.get_base_path(remote)}/mean_noise_input.h5",
    )

    ex.run()


def poisson_strength_and_ratio(remote: bool = True):
    default_params = {
        "runtime": 5000,
        "J_itoi": 5.0,
        "J_etoe": 0.6,
        "J_etoi": 1.2,
        "J_itoe": 7.0,
        "N_e": 1000,
        "N_i": 250,
        "ou_enabled": [False, False],
        "poisson_enabled": [True, True],
        "poisson_variance": 1.0,
        "poisson_p": 0.83,
        "poisson_mean_input": 200,
        "poisson_size": 800,
        "J_ppee": 2.0,
        "J_ppei": 2.0,
        # if set to default of 0.1 this leads to strange split in frequency band
        "const_delay": 0.2,
        "N_pop": 2,
        "p_etoe": 0.1,
        "p_etoi": 0.4,
        "p_itoe": 0.1,
        "p_itoi": 0.4,
    }

    params = {
        "poisson_variance": np.arange(0.1, 2, 0.1),
        "poisson_p": np.arange(0.75, 1, 0.025),
    }

    ex = mopet.Exploration(
        runner.run_in_mopet,
        explore_params=params,
        default_params=default_params,
        hdf_filename=constants.get_base_path(remote) + "/uncorrelated_noise.h5",
        exploration_name="uncorrelated_noise_3",
    )

    ex.run()


# List of all relevant explorations.
explorations = [synaptic_weight_exploration, mean_noise_exploration, poisson_strength_and_ratio]