
# nilmtk-contrib

This repository contains all the state-of-the-art algorithms for the task of energy disaggregation implemented using NILMTK's Rapid Experimentation API

# Installation Details

Currently, we are still working on developing a conda package, which might take some time to develop. In the meanwhile, you can install this by cloning the repository in the Lib/Site-packages  in your environment.  Rename the directory to **nilmtk_contrib**. Refer to this [notebook](https://github.com/nilmtk/nilmtk-contrib/tree/master/sample_notebooks) for using the nilmtk-contrib algorithms, using the NILMTK-API.

# Dependencies

Scikit-learn>=0.21
Keras>=2.2.4 
Cvxpy>=1.0.0


**Note: For faster computation of neural-networks, it is suggested that you install keras-gpu, since it can take advantage of GPUs. The algorithms AFHMM, AFHMM_SAC and DSC are CPU intensive, use a system with good CPU for these algorithms.**

