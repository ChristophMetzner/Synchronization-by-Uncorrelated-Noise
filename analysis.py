import pickle
from pathlib import Path

import plots
import os

directory = "models"

for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        with open(f"{directory}/{filename}", 'rb') as handle:
            base_name = Path(filename).stem

            data = pickle.load(handle)

            plots.psd(data, 300, 1.0, prefix=base_name)
