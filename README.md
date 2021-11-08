# Synchronization by Uncorrelated Noise

This repository contains the source code of the paper [*Synchronization through uncorrelated noise in excitatory-inhibitory networks*](https://doi.org/10.1101/2021.10.29.466430).

## Relevant Folders

* **synchronization** (python package which contains the complete source code)
* **notebooks** (contains jupyter notebooks that utilize the `synchronization` package)
* **models** (target destination for model files created at the end of a run)

## Relevant Notebooks

* `notebooks/2_nets_ING.ipynb` reproduces **scenario 1**: two all-to-all coupled interacting inhibitory networks.
* `notebooks/2_nets_PING_all_to_all.ipynb` reproduces **scenario 2**: two all-to-all coupled excitatory-inhibitory networks.
* `notebooks/2_nets_PING_sparse.ipynb` reproduces **scenario 3**: two sparse random excitatory-inhibitory networks.

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


## Jupyter Extensions

We recommend installing the [jupyterlab-toc](https://github.com/jupyterlab/jupyterlab-toc) extension as some notebooks are grouped into sections and subsections.
By using a TOC extension, reading and editing the notebooks becomes considerably easier.
   
   


