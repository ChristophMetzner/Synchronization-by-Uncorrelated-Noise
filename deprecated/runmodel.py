from __future__ import print_function
import numpy as np
#  import matplotlib.pyplot as plt
import sys
#  import time
import params_net
from utils import generate_OUinput, interpolate_input
import pickle
from deprecated import network_sim_simple as net

sys.path.insert(1, '..')  # allow parent modules to be imported
sys.path.insert(1, '../..')  # allow parent modules to be imported
sys.path.insert(1, '../../..')  # allow parent modules to be imported

# use the following in IPython for qt plots: %matplotlib qt

# use as default the parameters from file params_net.py
# if not specified else below
params = params_net.get_params()

params['runtime'] = 3000.
params['net_dt'] = 0.05

params['min_dt'] = params['net_dt']

params['t_ref'] = 0.0

# plotting section
plot_rates = True
plot_input = True


# external time trace used for generating input and plotting
# if time step is unequal to model_dt input gets interpolated
steps = int(params['runtime']/params['min_dt'])
t_ext = np.linspace(0., params['runtime'], steps+1)

# time trace computed with min_dt
params['t_ext'] = t_ext


params['ou_X0'] = 0.
params['ou_mean'] = 0.5
params['ou_sigma'] = .5
params['ou_tau'] = 50.
mu_ext = generate_OUinput(params)


# mu = const, sigma = OU process
params['ou_X0'] = 0.  #
params['ou_mean'] = 2.0
params['ou_sigma'] = 0.2
params['ou_tau'] = 1.
sigma_ext = generate_OUinput(params)


# collect ext input for model wrappers
ext_input0 = [mu_ext, sigma_ext]

# saving results in global results dict
results = dict()
results['input_mean'] = mu_ext
results['input_sigma'] = sigma_ext
results['model_results'] = dict()

# brian network sim
ext_input = interpolate_input(ext_input0, params, 'net')
results['model_results']['net'] = \
    net.network_sim(ext_input, params)

# save results

with open('test.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)