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

## Dependencies

- NILMTK>=0.4
- scikit-learn>=0.21 (already required by NILMTK)
- Keras>=2.2.4 
- cvxpy>=1.0.0

**Note: For faster computation of neural networks, it is suggested that you install keras-gpu, since it can take advantage of GPUs. The algorithms AFHMM, AFHMM_SAC and DSC are CPU intensive, use a system with good CPU for these algorithms.**

