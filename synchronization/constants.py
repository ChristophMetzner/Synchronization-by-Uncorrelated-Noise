import models
import plots_out
import os

MODELS_PATH = os.path.dirname(models.__file__)
PLOTS_PATH = os.path.dirname(plots_out.__file__)

# Remote path to data files on cursa server.
REMOTE_PATH = "/mnt/raid/data/lrebscher/data"


def get_base_path(remote: bool = False) -> str:
    return REMOTE_PATH if remote else MODELS_PATH
