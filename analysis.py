import pickle
from pathlib import Path

import plots
import os

DIRECTORY = "models"
MODEL_SUFFIX = ".pkl"


def analyze_model(data, base_name: str):
    params_ = data['params']

    duration = params_['runtime']
    dt = params_['net_dt']

    try:
        os.mkdir(f"plots/{base_name}")
    except FileExistsError as e:
        # can be ignored ..
        print(e)

    plots.psd(title="Power Spectral Density of Excitatory Neurons",
              model=data,
              duration=duration,
              dt=1.0,
              folder=base_name)

    plots.psd(title="Power Spectral Density of Inhibitory Neurons",
              model=data,
              duration=duration,
              dt=1.0,
              folder=base_name)

    plots.summed_voltage(title='Summed Voltage Change of Excitatory Neurons Over Time',
                         model=data,
                         duration=duration,
                         dt=dt,
                         prefix=base_name)

    plots.summed_voltage(title='Summed Voltage Change of Inhibitory Neurons Over Time',
                         model=data,
                         duration=duration,
                         dt=dt,
                         prefix=base_name)

    plots.noise(data['input_mean_1'], data['input_sigma_1'], prefix=base_name)

    x_left = duration - 100
    x_left = 0 if x_left < 0 else x_left

    plots.raster(data=data, x_left=x_left, x_right=duration, key="group1_narrow",
                 folder=base_name)

    plots.raster(data=data, x_left=0, x_right=duration, key="group1_wide", folder=base_name)

    plots.raster(data=data, x_left=x_left, x_right=duration, key="group2_narrow",
                 folder=base_name)

    plots.raster(data=data, x_left=0, x_right=duration, key="group2_wide", folder=base_name)


def analyze():
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(MODEL_SUFFIX):
            with open(f"{DIRECTORY}/{filename}", 'rb') as handle:
                data = pickle.load(handle)
                base_name = Path(filename).stem
                try:
                    analyze_model(data, base_name)
                except Exception as e:
                    print(f"Skipped analyzing model {base_name} due to unexpected failure.", e)


def load(names: [str]):
    for name in names:
        with open(f"{DIRECTORY}/{name}.pkl", 'rb') as handle:
            data = pickle.load(handle)
            yield name, data


if __name__ == '__main__':
    analyze()
