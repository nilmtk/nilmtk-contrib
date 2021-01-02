[![conda package version](https://anaconda.org/nilmtk/nilmtk-contrib/badges/version.svg)](https://anaconda.org/nilmtk/nilmtk-contrib)

# NILMTK-Contrib

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

We're currently testing a conda package. You can install in your current environment with:

```
conda install -c conda-forge -c nilmtk nilmtk-contrib
```

or create a dedicated environment (recommended) with:

```
conda create -n nilm -c conda-forge -c nilmtk nilmtk-contrib
```

Refer to this [notebook](https://github.com/nilmtk/nilmtk-contrib/tree/master/sample_notebooks) for using the nilmtk-contrib algorithms, using the new NILMTK-API.

Unless you are an advanced user, prefer using the Conda package instead of the Git repostory as the latter can contain work-in-progress changes.

## Dependencies

- NILMTK>=0.4
- scikit-learn>=0.21 (already required by NILMTK)
- Keras>=2.2.4 
- cvxpy>=1.0.0

**Note: For faster computation of neural networks, it is suggested that you install keras-gpu, since it can take advantage of GPUs. The algorithms AFHMM, AFHMM_SAC and DSC are CPU intensive, use a system with good CPU for these algorithms.**

