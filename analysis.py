import pickle
from pathlib import Path

import plots
import os
import numpy as np

directory = "models"

for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        with open(f"{directory}/{filename}", 'rb') as handle:
            data = pickle.load(handle)
            base_name = Path(filename).stem

            duration = 300
            dt = 1.0
            skip = 0

            # rates
            v_e1 = data['model_results']['net']['v_all_neurons_e']
            v_e2 = data['model_results']['net']['v_all_neurons_e2']
            v_i1 = data['model_results']['net']['v_all_neurons_i1']
            v_i2 = data['model_results']['net']['v_all_neurons_i2']

            v_e1 = v_e1[:, skip:duration + skip]
            v_e2 = v_e2[:, skip:duration + skip]
            v_i1 = v_i1[:, skip:duration + skip]
            v_i2 = v_i2[:, skip:duration + skip]

            # External Signal to first population
            ext_signal_1_mean = data['input_mean1']
            ext_signal_1_sigma = data['input_sigma1']

            # signal
            ext_signal_1_mean = ext_signal_1_mean[skip:duration + skip]

            # sum up voltages of excitatory Neuron Groups
            lfp1 = np.sum(v_e1, axis=0) / 1000
            lfp2 = np.sum(v_e2, axis=0) / 1000

            # sum up voltages of inhibitory Neuron Groups
            inh1 = np.sum(v_i1, axis=0) / 1000
            inh2 = np.sum(v_i2, axis=0) / 1000

            # time
            t = np.linspace(0, duration, int(duration / dt))

            # spikes
            s_e1 = data['model_results']['net']['net_spikes_e']
            s_e2 = data['model_results']['net']['net_spikes_e2']
            s_i1 = data['model_results']['net']['net_spikes_i1']
            s_i2 = data['model_results']['net']['net_spikes_i2']

            plots.psd(title="Power Spectral Density of Excitatory Neurons",
                      key="excitatory",
                      lfp1=lfp1, lfp2=lfp2, duration=300, dt=1.0, prefix=base_name)

            plots.psd(title="Power Spectral Density of Inhibitory Neurons",
                      key="inhibitory",
                      lfp1=inh1, lfp2=inh2, duration=300, dt=1.0, prefix=base_name)

            plots.plot_summed_voltage(title='Summed Voltage Change of Excitatory Neurons Over Time',
                                      key="excitatory",
                                      lfp1=lfp1,
                                      lfp2=lfp2,
                                      duration=duration,
                                      dt=dt,
                                      prefix=base_name)

            plots.plot_summed_voltage(title='Summed Voltage Change of Inhibitory Neurons Over Time',
                                      key="inhibitory",
                                      lfp1=inh1,
                                      lfp2=inh2,
                                      duration=duration,
                                      dt=dt,
                                      prefix=base_name)

            plots.plot_noise(data, prefix=base_name)

            plots.plot_raster(s_e1=s_e1, s_i1=s_i1, x_left=1000, x_right=1080, key="narrow", prefix=base_name)
            plots.plot_raster(s_e1=s_e1, s_i1=s_i1, x_left=1000, x_right=2000, key="wide", prefix=base_name)
