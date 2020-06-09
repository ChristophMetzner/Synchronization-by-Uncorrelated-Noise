from mopet import mopet
import numpy as np

def run(params):
    return {}


params = {"a": np.arange(0, 1, 0.5)}

ex = mopet.Exploration(run, params, exploration_name="testA")
ex.run()
ex.load_results()