# NILMTK-Contrib

(Note - This package only works on Python versions <= 3.11)

This repository contains all the state-of-the-art algorithms for the task of energy disaggregation implemented using NILMTK's Rapid Experimentation API. You can find the paper [here](https://doi.org/10.1145/3360322.3360844). All the notebooks that were used to can be found [here](https://github.com/nilmtk/buildsys2019-paper-notebooks).

Using the NILMTK-contrib you can use the following algorithms:
 - Additive Factorial Hidden Markov Model
 - Additive Factorial Hidden Markov Model with Signal Aggregate Constraints
 - Discriminative Sparse Coding
 - RNN
 - Denoising Auto Encoder
 - Seq2Point
 - Seq2Seq
 - WindowGRU

The above state-of-the-art algorithms have been added to this repository. 

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

To install nilmtk_contrib, first install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:<br>
```
uv pip install git+https://github.com/nilmtk/nilmtk-contrib.git
```

## Docker Support
Docker is an open-source platform for developing, shipping, and running applications in lightweight, portable containers that bundle code, runtime, libraries, and system tools into a single package. It ensures everyone runs the same environment, regardless of host OS, and keeps nilmtk-contrib’s dependencies contained without polluting the system Python.


Build and run locally
```
docker build -t nilmtk-contrib .
docker run --rm -it nilmtk-contrib bash
```
Pull the pre-built image
```
docker pull ghcr.io/enfuego27826/nilmtk-contrib:latest
docker run --rm -it ghcr.io/enfuego27826/nilmtk-contrib:latest bash
```

Refer to this [notebook](https://github.com/nilmtk/nilmtk-contrib/tree/master/sample_notebooks) for using the nilmtk-contrib algorithms, using the new NILMTK-API.

## Dependencies

- NILMTK>=0.4
- scikit-learn>=0.21 (already required by NILMTK)
- Tensorflow >= 2.12.0 < 2.16.0 
- cvxpy>=1.0.0

**Note: For faster computation of neural networks, it is suggested that you install keras-gpu, since it can take advantage of GPUs. The algorithms AFHMM, AFHMM_SAC and DSC are CPU intensive, use a system with good CPU for these algorithms.**

