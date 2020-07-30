import glob
import pickle
import numpy as np
import os

from synchronization import constants
from synchronization import runner

from itertools import product
from mopet import mopet


class Experiment:
    """ Base Experiment."""

    name = "name"

    def run(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, condition=lambda x: True) -> [dict]:
        models = []
        base_path = f"{constants.MODELS_PATH}/{cls.name}"
        for file in glob.glob(f"{base_path}/*.pkl"):
            with open(file, "rb") as f:
                model = pickle.load(f)
                if condition(model):
                    models.append(model)
        return models

    @classmethod
    def clean(cls):
        print(f"Start cleaning models of experiment {cls.name}")

        base_path = f"{constants.MODELS_PATH}/{cls.name}"
        for file in glob.glob(f"{base_path}/*.pkl"):
            print(f"Remove file {file}")
            os.remove(file)

        print("Finished cleaning.")


class NoiseExperiment(Experiment):
    """
    Simulates models for the "effect of noise in a single population" experiment.

    All combinations over a range of different noise parameters.

    Input Variables
        1. Mean
        2. Sigma
        3. Tau
    """

    name = "noise"

    def __init__(
        self,
        mean_range: np.ndarray = None,
        sigma_range: np.array = None,
        tau_range: np.array = None,
    ):
        mean = mean_range if mean_range else np.arange(0, 10, 0.5)
        sigma = sigma_range if sigma_range else np.arange(0, 6, 0.5)
        tau = tau_range if tau_range else np.arange(1, 60, 5)

        self._param_space = list(product(mean, sigma, tau))

    def run(self):
        total = len(self._param_space)
        print(f"Starting simulation of {total} parameter configurations ...")

        for idx, vals in enumerate(self._param_space):
            (m, s, t) = vals
            print(
                f"{idx + 1} of {total} Running parameter configuration: {m} - {s} - {t}"
            )

            config = dict()
            config["runtime"] = 1000

            config["ou_mu"] = {"ou_mean": m, "ou_sigma": s, "ou_tau": t}

            config["ou_sigma"] = {
                "ou_X0": 0.0,
                "ou_mean": 0.0,
                "ou_sigma": 0.2,
                "ou_tau": 1,
            }

            runner.run(
                f"{m}-{s}-{t}", experiment_name=self.name, modified_params=config
            )


class MopetExampleExperiment:
    def run(self):
        params = {"J_etoe": np.arange(1, 3, 1)}

        ex = mopet.Exploration(runner.run_in_mopet, params)
        ex.run()
        ex.load_results(all=True)

        return ex.df, ex.results
