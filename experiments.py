import numpy as np

from itertools import product

import runner


class Experiment:
    """ Base Experiment."""
    pass


class NoiseExperiment(Experiment):
    """
    Simulates models for the "effect of noise in a single population" experiment.

    All combinations over a range of different noise parameters.

    Input Variables
        1. Mean
        2. Sigma
        3. Tau
    """

    def __init__(self):
        mean = np.arange(0, 5, 0.1)
        sigma = np.arange(0, 5, 0.1)
        tau = np.arange(1, 50, 1)

        self._param_space = product(mean, sigma, tau)

    def run(self):
        results = []

        for vals in self._param_space:
            (m, s, t) = vals
            print("Running parameter configuration", (m, s, t))
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

            # TODO: add base directory for the experiment
            result = runner.run(f"{m}-{s}-{t}", modified_params=config)
            results.append((m, result))
