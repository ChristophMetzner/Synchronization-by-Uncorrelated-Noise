import pickle
import os
import plots

from pathlib import Path
from constants import PLOTS_PATH
from constants import MODELS_PATH

MODEL_SUFFIX = ".pkl"


def analyze_model(model, base_name: str):
    params_ = model['params']

    duration = params_['runtime']
    dt = params_['net_dt']

    try:
        os.mkdir(f"{PLOTS_PATH}/{base_name}")
    except FileExistsError as e:
        # can be ignored ..
        print(e)

    plots.psd(title="Power Spectral Density of Excitatory Neurons",
              model=model,
              dt=1.0,
              folder=base_name)

    plots.psd(title="Power Spectral Density of Inhibitory Neurons",
              model=model,
              dt=1.0,
              folder=base_name)

    plots.summed_voltage(title='Summed Voltage Change of first Network',
                         model=model,
                         dt=1.0,
                         prefix=base_name)

    plots.noise(model['input_mean_1'], model['input_sigma_1'], prefix=base_name)

    x_left = duration - 100
    x_left = 0 if x_left < 0 else x_left

    plots.raster(model=model, x_left=x_left, x_right=duration, key="group1_narrow",
                 folder=base_name)

    plots.raster(model=model, x_left=0, x_right=duration, key="group1_wide", folder=base_name)

    plots.raster(model=model, x_left=x_left, x_right=duration, key="group2_narrow",
                 folder=base_name)

    plots.raster(model=model, x_left=0, x_right=duration, key="group2_wide", folder=base_name)


def analyze():
    for filename in os.listdir(MODELS_PATH):
        if filename.endswith(MODEL_SUFFIX):
            with open(f"{MODELS_PATH}/{filename}", 'rb') as handle:
                data = pickle.load(handle)
                base_name = Path(filename).stem
                try:
                    analyze_model(data, base_name)
                except Exception as e:
                    print(f"Skipped analyzing model {base_name} due to unexpected failure.", e)


def load(names: [str]):
    for name in names:
        with open(f"{MODELS_PATH}/{name}.pkl", 'rb') as handle:
            data = pickle.load(handle)
            yield name, data


def load_model(name: str):
    with open(f"{MODELS_PATH}/{name}.pkl", 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    analyze()
