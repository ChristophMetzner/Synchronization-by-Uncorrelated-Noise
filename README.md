# Synchronization by Uncorrelated Noise

This repository contains the source code of the master thesis *Effect of Noise on the Synchronization of Interacting Excitatory-Inhibitory Networks* by Lucas Rebscher.

## Relevant Folders

* **synchronization** (python package which contains the complete source code)
* **notebooks** (contains all jupyter notebooks that utilize the `synchronization` package)
* **meng_and_riecke** (matlab code snippets provided on request by Meng and Riecke.)


## Development Guide

* Python 3.6+ is required
* We recommend to use a virtual environment

Install all required packages with

```shell script
pip install -r requirements.txt
```


Install `synchronization` package locally so that the Jupyter notebooks can import it

```shell script
pip install -e . 
```

Any change to the code in `synchronization/` is immediately reflected as `-e` installs the package in editable mode. 