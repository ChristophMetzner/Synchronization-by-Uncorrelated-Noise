from mopet import mopet
import numpy as np

def run(params):
    return { "my_list": [1, 2, 3]}


params = {"a": np.arange(0, 1, 0.5)}

ex = mopet.Exploration(run, params, exploration_name="testB")
# ex.run()
ex.load_results()