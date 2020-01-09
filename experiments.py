import numpy as np

from itertools import product
from typing import List

import runner


class Experiment:
    """ Base Experiment."""

    def run(self) -> List:
        raise NotImplementedError()


class NoiseExperiment(Experiment):
    """
    Simulates models for the "effect of noise in a single population" experiment.

    All combinations over a range of different noise parameters.

    Input Variables
        1. Mean
        2. Sigma
        3. Tau
    """

    name = 'noise'

    def __init__(self, mean_range: np.ndarray = None, sigma_range: np.array = None, tau_range: np.array = None):
        mean = mean_range if mean_range else np.arange(0, 10, 0.5)
        sigma = sigma_range if sigma_range else np.arange(0, 6, 0.5)
        tau = tau_range if tau_range else np.arange(1, 60, 5)

        self._param_space = list(product(mean, sigma, tau))

    def run(self):
        total = len(self._param_space)
        print(f"Starting simulation of {total} parameter configurations ...")

        for idx, vals in enumerate(self._param_space):
            (m, s, t) = vals
            print(f"{idx + 1} of {total} Running parameter configuration: {m} - {s} - {t}")

            config = dict()
            config['runtime'] = 1000

            config['ou_mu'] = {
                'ou_mean': m,
                'ou_sigma': s,
                'ou_tau': t
            }

            config['ou_sigma'] = {
                'ou_X0': 0.,
                'ou_mean': 0.0,
                'ou_sigma': 0.2,
                'ou_tau': 1
            }

            runner.run(f"{m}-{s}-{t}", experiment_name=self.name, modified_params=config)
