from mopet import mopet
import numpy as np


def run(params):
    import time

    x = 2
    t_end = time.time() + 5
    while time.time() < t_end:
        y = x * x

    return {"my_list": [1, 2, 3]}


def test():
    params = {"a": np.arange(0, 100, 1)}

    ex = mopet.Exploration(
        run, params, exploration_name="testC", hdf_filename="test.h5"
    )
    ex.run(num_cpus=1)
    ex.load_results()


def mean_noise_exploration():
    from synchronization import runner
    from synchronization import constants

    default_params = {
        "runtime": 3000.0,
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
        hdf_filename=f"{constants.MODELS_PATH}/test.h5",
    )

    ex.run(num_cpus=2)


# mean_noise_exploration()

test()
