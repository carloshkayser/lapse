<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/carloshkayser/lapse/HEAD?labpath=analysis.ipynb) -->

# Lapse

This repository contains the source code and the dataset used in the paper **Lapse: Latency & Power-Aware Placement of Data Stream Applications on Edge Computing**. **Lapse** is a cost-based heuristic algorithm for the placement of stream processing applications on edge computing environments.

## Installation and usage guide

This section provides instructions on setting up your environment for running simulations and replicating the paper's results.

### Prerequisites

Before you can execute the simulations, ensure that you have the following packages installed:

- Python 3.9
- Poetry 1.7.1

To install the required dependencies, follow these steps:

```sh
poetry install
poetry shell
```

### Reproducing the results

To replicate the paper's results, you can either run the following commands or execute the Jupyter notebook [`analysis.ipynb`](analysis.ipynb).

First, optionally create the dataset:

```sh
python dataset.py --name dataset > datasets/dataset.log
```

Next, run the simulations for the algorithms:

```sh
python -B -m simulator --dataset datasets/dataset.json --algorithm storm
python -B -m simulator --dataset datasets/dataset.json --algorithm storm_la
python -B -m simulator --dataset datasets/dataset.json --algorithm aels
python -B -m simulator --dataset datasets/dataset.json --algorithm aels_pa
python -B -m simulator --dataset datasets/dataset.json --algorithm lapse
```

<!-- Alternatively, you can run the simulations for the algorithms directly on [MyBinder](https://mybinder.org/v2/gh/carloshkayser/lapse/master?filepath=analysis.ipynb). -->

## Manuscript

> Available soon
