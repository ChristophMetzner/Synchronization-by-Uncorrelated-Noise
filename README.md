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


## Jupyter Extensions

We recommend installing the [jupyterlab-toc](https://github.com/jupyterlab/jupyterlab-toc) extension as some notebooks are grouped into sections and subsections.
By using a TOC extension, reading and editing the notebooks becomes considerably easier.
   
   
## Notebooks

* `notebooks/2_nets_ING.ipynb` reproduces **scenario 1** of two all-to-all coupled interacting inhibitory networks.
* `notebooks/2_nets_PING_all_to_all.ipynb` reproduces **scenario 2** of two all-to-all coupled excitatory-inhibitory networks.
* `notebooks/2_nets_PING_sparse.ipynb` reproduces **scenario 3** of two sparse random excitatory-inhibitory networks.

