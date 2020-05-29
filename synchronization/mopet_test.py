from mopet import mopet
import numpy as np

def run(params):
    print(params)
    return {
        "test": [True, False],
        "test2": False
    }


params = {"a": np.arange(0, 1, 0.5)}

ex = mopet.Exploration(run, params)
ex.run()
ex.load_results()

print(ex.results)
print(ex.df)
