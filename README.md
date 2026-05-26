# NILMTK-Contrib

(Note - This package currently supports Python >=3.11,<3.12. Python 3.12+ is unsupported until TensorFlow and NILMTK compatibility is verified.)

This repository contains NILMTK-compatible implementations of energy disaggregation algorithms and research baselines. You can find the NILMTK-contrib paper [here](https://doi.org/10.1145/3360322.3360844). All the notebooks that were used to can be found [here](https://github.com/nilmtk/buildsys2019-paper-notebooks).

Using the NILMTK-contrib you can use the following algorithms:
 - Additive Factorial Hidden Markov Model
 - Additive Factorial Hidden Markov Model with Signal Aggregate Constraints
 - Discriminative Sparse Coding
 - RNN
 - Denoising Auto Encoder
 - Seq2Point
 - Seq2Seq
 - WindowGRU

The historical NILMTK-contrib algorithms and newer experimental backends are tracked in the model audit matrix. See [docs/models.md](docs/models.md) for each model's backend, citation type, implementation status, required checks, known deviations, and server validation status. See [docs/references.md](docs/references.md) for citation classification.

You can do the following using the new NILMTK's Rapid Experimentation API:
 - Training and Testing across multiple appliances
 - Training and Testing across multiple datasets (Transfer learning)
 - Training and Testing across multiple buildings
 - Training and Testing with Artificial aggregate
 - Training and Testing with different sampling frequencies
 
Refer to this [notebook](https://github.com/nilmtk/nilmtk-contrib/blob/master/sample_notebooks/NILMTK%20API%20Tutorial.ipynb) to know more about the usage of the API.

## Citation


If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3360322.3360844,
author = {Batra, Nipun and Kukunuri, Rithwik and Pandey, Ayush and Malakar, Raktim and Kumar, Rajat and Krystalakos, Odysseas and Zhong, Mingjun and Meira, Paulo and Parson, Oliver},
title = {Towards Reproducible State-of-the-Art Energy Disaggregation},
year = {2019},
isbn = {9781450370059},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3360322.3360844},
doi = {10.1145/3360322.3360844},
booktitle = {Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
pages = {193–202},
numpages = {10},
keywords = {smart meters, energy disaggregation, non-intrusive load monitoring},
location = {New York, NY, USA},
series = {BuildSys '19}
}
}

```
For any enquiries, please contact the main authors.

## Installation Details

## UV Support
This Python package uses uv for installation. uv is a fast and modern Python package manager that replaces tools like pip and virtualenv, with support for pyproject.toml and ultra-fast dependency resolution.

Install the minimal package when you only need package metadata or lightweight imports:

```
uv pip install git+https://github.com/nilmtk/nilmtk-contrib.git
```

Install a backend-specific extra for model use:

```
uv pip install "nilmtk-contrib[tensorflow] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
uv pip install "nilmtk-contrib[torch] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
uv pip install "nilmtk-contrib[classical] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

Install all model backends:

```
uv pip install "nilmtk-contrib[all] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

For development:

```
uv sync --extra dev
```

For backend development, include the relevant backend extra, for example:

```
uv sync --extra dev --extra torch
```

## Docker Support
Docker is an open-source platform for developing, shipping, and running applications in lightweight, portable containers that bundle code, runtime, libraries, and system tools into a single package. It ensures everyone runs the same environment, regardless of host OS, and keeps nilmtk-contrib’s dependencies contained without polluting the system Python.


Build and run locally
```
docker build -t nilmtk-contrib .
docker run --rm -it nilmtk-contrib bash
```
The default Dockerfile installs `.[all]`. Edit the Dockerfile to use `.[torch]`, `.[tensorflow]`, or `.[classical]` for a narrower backend image.

Pull the pre-built image
```
docker pull ghcr.io/enfuego27826/nilmtk-contrib:latest
docker run --rm -it ghcr.io/enfuego27826/nilmtk-contrib:latest bash
```

Refer to this [notebook](https://github.com/nilmtk/nilmtk-contrib/tree/master/sample_notebooks) for using the nilmtk-contrib algorithms, using the new NILMTK-API.

## Dependencies

- Minimal install: no required runtime dependencies for top-level import.
- `tensorflow` extra: NILMTK, NumPy, pandas, scikit-learn, matplotlib, TensorFlow, and `tensorflow-io-gcs-filesystem`.
- `torch` extra: NILMTK, NumPy, pandas, scikit-learn, matplotlib, PyTorch, and tqdm.
- `classical` extra: NILMTK, NumPy, pandas, matplotlib, scikit-learn, SciPy, cvxpy, and hmmlearn.
- `all` extra: union of TensorFlow, PyTorch, classical, and NILMTK dependencies.
- `dev` extra: pytest, pytest-cov, black, ruff, and build.

**Note: For faster computation of neural networks, it is suggested that you install keras-gpu, since it can take advantage of GPUs. The algorithms AFHMM, AFHMM_SAC and DSC are CPU intensive, use a system with good CPU for these algorithms.**

