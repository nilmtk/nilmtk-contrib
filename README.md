# NILMTK-Contrib

NILMTK-Contrib provides NILMTK-compatible implementations of non-intrusive load monitoring (NILM) and energy disaggregation algorithms. The package is designed for use with NILMTK's rapid experimentation API and includes classical, TensorFlow, and PyTorch model backends.

The repository paper is:

Batra et al., "Towards Reproducible State-of-the-Art Energy Disaggregation", BuildSys 2019, DOI: https://doi.org/10.1145/3360322.3360844.

## Runtime Requirements

- Python `>=3.11,<3.12`.
- Install a backend extra before importing or training backend-specific models.
- NILMTK-compatible datasets are required for real experiments, notebook runs, and benchmark reproduction.
- Model training and benchmark comparisons should be run in controlled server environments with the relevant backend, dataset, and hardware available.

Python 3.12 and newer are not supported by the current package metadata because TensorFlow and NILMTK compatibility must be verified first.

## Installation

Minimal install for package metadata and lightweight imports:

```bash
uv pip install git+https://github.com/nilmtk/nilmtk-contrib.git
```

TensorFlow backend:

```bash
uv pip install "nilmtk-contrib[tensorflow] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

PyTorch backend:

```bash
uv pip install "nilmtk-contrib[torch] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

Classical backend:

```bash
uv pip install "nilmtk-contrib[classical] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

All model backends:

```bash
uv pip install "nilmtk-contrib[all] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

Development environment:

```bash
uv sync --extra dev
```

Backend development examples:

```bash
uv sync --extra dev --extra torch
uv sync --extra dev --extra tensorflow
uv sync --extra dev --extra classical
```

## Dependency Extras

| Extra | Intended use | Main dependencies |
|---|---|---|
| Minimal | Import package metadata and lightweight modules | No required runtime dependencies |
| `tensorflow` | TensorFlow/Keras disaggregators | NILMTK, NumPy, pandas, scikit-learn, matplotlib, TensorFlow, `tensorflow-io-gcs-filesystem` |
| `torch` | PyTorch disaggregators | NILMTK, NumPy, pandas, scikit-learn, matplotlib, PyTorch, tqdm |
| `classical` | AFHMM, AFHMM_SAC, DSC | NILMTK, NumPy, pandas, matplotlib, scikit-learn, SciPy, cvxpy, hmmlearn |
| `all` | All backends | Union of TensorFlow, PyTorch, classical, and NILMTK dependencies |
| `dev` | Tests, formatting, and build checks | pytest, pytest-cov, black, ruff, build |

## Models

The table below lists the public model surface. "Verification" describes how the implementation should be cited and interpreted in research use.

| Algorithm | Backend | Import path | Verification | Paper/source | Notes |
|---|---|---|---|---|---|
| AFHMM | Classical | `nilmtk_contrib.disaggregate.AFHMM` | NILM paper implementation, not independently benchmark-certified in this package state | Kolter and Jaakkola, AFHMM for energy disaggregation | Requires `classical` extra |
| AFHMM_SAC | Classical | `nilmtk_contrib.disaggregate.AFHMM_SAC` | NILM paper implementation, not independently benchmark-certified in this package state | Zhong, Goddard, and Sutton, signal aggregate constraints in AFHMMs | Requires `classical` extra |
| DSC | Classical | `nilmtk_contrib.disaggregate.DSC` | NILM paper implementation, not independently benchmark-certified in this package state | Kolter, Batra, and Ng, discriminative sparse coding | Requires `classical` extra |
| DAE | TensorFlow | `nilmtk_contrib.disaggregate.DAE` | Neural NILM implementation requiring experiment validation for new claims | Kelly and Knottenbelt, Neural NILM | TensorFlow/Keras backend |
| DAE | PyTorch | `nilmtk_contrib.torch.DAE` | PyTorch implementation requiring parity validation for new claims | Kelly and Knottenbelt, Neural NILM | PyTorch backend |
| RNN | TensorFlow | `nilmtk_contrib.disaggregate.RNN` | Neural NILM implementation requiring experiment validation for new claims | Kelly and Knottenbelt, Neural NILM | TensorFlow/Keras backend |
| RNN | PyTorch | `nilmtk_contrib.torch.RNN` | PyTorch implementation requiring parity validation for new claims | Kelly and Knottenbelt, Neural NILM | PyTorch backend |
| Seq2Point | TensorFlow | `nilmtk_contrib.disaggregate.Seq2Point` | NILM paper implementation requiring dataset-specific validation | Zhang et al., Sequence-to-Point Learning | TensorFlow/Keras backend |
| Seq2PointTorch | PyTorch | `nilmtk_contrib.torch.Seq2PointTorch` | PyTorch implementation requiring parity validation for new claims | Zhang et al., Sequence-to-Point Learning | PyTorch backend |
| Seq2Seq | TensorFlow | `nilmtk_contrib.disaggregate.Seq2Seq` | Legacy NILM baseline adapted from a generic sequence model | Sutskever, Vinyals, and Le, sequence-to-sequence learning | Generic architecture citation |
| Seq2Seq | PyTorch | `nilmtk_contrib.torch.Seq2Seq` | Legacy NILM baseline adapted from a generic sequence model | Sutskever, Vinyals, and Le, sequence-to-sequence learning | Generic architecture citation |
| WindowGRU | TensorFlow | `nilmtk_contrib.disaggregate.WindowGRU` | NILM paper implementation requiring experiment validation for new claims | Krystalakos, Nalmpantis, and Vrakas, sliding-window GRU | TensorFlow/Keras backend |
| WindowGRU | PyTorch | `nilmtk_contrib.torch.WindowGRU` | PyTorch implementation requiring parity validation for new claims | Krystalakos, Nalmpantis, and Vrakas, sliding-window GRU | PyTorch backend |
| RNN_attention | TensorFlow | `nilmtk_contrib.disaggregate.RNN_attention` | Attention-based NILM implementation | Sudoso and Piccialli, attention-based NILM | TensorFlow/Keras backend |
| RNN_attention | PyTorch | `nilmtk_contrib.torch.RNN_attention` | PyTorch attention-based NILM implementation | Attention-based NILM literature | PyTorch backend |
| RNN_attention_classification | TensorFlow | `nilmtk_contrib.disaggregate.RNN_attention_classification` | Attention-based NILM implementation with classification branch | Sudoso and Piccialli, attention-based NILM | Explicit on/off threshold parameters are supported |
| RNN_attention_classification | PyTorch | `nilmtk_contrib.torch.RNN_attention_classification` | PyTorch attention-based NILM implementation with classification branch | Attention-based NILM literature | Explicit on/off threshold parameters are supported |
| ResNet | TensorFlow | `nilmtk_contrib.disaggregate.ResNet` | 1D residual NILM adaptation of a generic architecture | He et al., Deep Residual Learning | Generic computer-vision architecture adapted to NILM |
| ResNet | PyTorch | `nilmtk_contrib.torch.ResNet` | 1D residual NILM adaptation of a generic architecture | He et al., Deep Residual Learning | Generic computer-vision architecture adapted to NILM |
| ResNet_classification | TensorFlow | `nilmtk_contrib.disaggregate.ResNet_classification` | Residual NILM model with classification branch | Residual and NILM classification literature | Explicit threshold and loss-weight parameters are supported |
| ResNet_classification | PyTorch | `nilmtk_contrib.torch.ResNet_classification` | Residual NILM model with classification branch | Residual and NILM classification literature | Explicit threshold and loss-weight parameters are supported |
| BERT | TensorFlow | `nilmtk_contrib.disaggregate.BERT` | Transformer/BERT-inspired NILM adaptation | Devlin et al., BERT | Does not claim NLP-style pretraining |
| BERT | PyTorch | `nilmtk_contrib.torch.BERT` | Transformer/BERT-inspired NILM adaptation | Devlin et al., BERT | Does not claim NLP-style pretraining |
| ConvLSTM | PyTorch | `nilmtk_contrib.torch.ConvLSTM` | ConvLSTM-inspired NILM adaptation | Shi et al., ConvLSTM | Generic spatiotemporal architecture adapted to NILM |
| TCN | PyTorch | `nilmtk_contrib.torch.TCN` | Generic TCN sequence-modeling baseline adapted to NILM | Bai, Kolter, and Koltun, TCN | PyTorch backend |
| Reformer | PyTorch | `nilmtk_contrib.torch.Reformer` | Reformer-inspired NILM adaptation | Kitaev, Kaiser, and Levskaya, Reformer | Efficient Transformer architecture adapted to NILM |
| MSDC | PyTorch | `nilmtk_contrib.torch.MSDC` | NILM paper implementation requiring experiment validation for new claims | MSDC dual-CNN NILM paper | Canonical CRF-enabled implementation path |
| MSDC without CRF | PyTorch | `nilmtk_contrib.torch.msdc_without_crf.MSDC` | MSDC ablation | MSDC paper/source implementation | No-CRF ablation, not the canonical MSDC path |
| NILMFormer | PyTorch | `nilmtk_contrib.torch.NILMFormer` | NILMFormer implementation requiring experiment validation for new claims | Petralia et al., NILMFormer | PyTorch backend |

## Research Use And Reproducibility

Use the model table to choose the correct backend and citation. Generic architecture papers support architecture inspiration only; they should not be cited as NILM-specific evidence by themselves.

For reproducible experiments:

- Record the Python version, package extras, dataset, building, appliance list, sampling period, random seed, and hardware.
- Run backend-specific smoke tests before running full experiments.
- Verify TensorFlow/PyTorch parity before comparing paired implementations.
- Verify model output lengths and indices before computing NILMTK metrics.
- Treat notebook outputs as historical examples unless rerun in the current environment.

Recommended fast checks for source validation:

```bash
python -m compileall -q nilmtk_contrib tests
python -m pytest -q tests/test_imports.py tests/test_params.py tests/test_preprocessing_windows.py tests/test_preprocessing_alignment.py tests/test_preprocessing_classification.py tests/test_validation.py tests/test_checkpoints.py tests/test_random_logging.py tests/test_model_runtime.py
python -m build
```

Backend smoke checks should be run in environments with the corresponding extras by importing the target model classes and running small dataset-specific training or prediction jobs before launching full experiments. For example:

```bash
uv sync --extra dev --extra torch
python -m pytest -q
```

## Reference Papers And Codebases

NILM-specific references:

- Kolter and Jaakkola, "Approximate Inference in Additive Factorial HMMs with Application to Energy Disaggregation", AISTATS 2012, https://proceedings.mlr.press/v22/zico12.html.
- Zhong, Goddard, and Sutton, "Signal Aggregate Constraints in Additive Factorial HMMs, with Application to Energy Disaggregation", NeurIPS 2014, https://papers.nips.cc/paper/5526-signal-aggregate-constraints-in-additive-factorial-hmms-with-application-to-energy-disaggregation.
- Kolter, Batra, and Ng, "Energy Disaggregation via Discriminative Sparse Coding", NeurIPS 2010, https://papers.nips.cc/paper/4054-energy-disaggregation-via-discriminative-sparse-coding.
- Kelly and Knottenbelt, "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation", arXiv:1507.06594, https://arxiv.org/abs/1507.06594.
- Zhang et al., "Sequence-to-Point Learning With Neural Networks for Non-Intrusive Load Monitoring", AAAI 2018, DOI: https://doi.org/10.1609/aaai.v32i1.11873.
- Krystalakos, Nalmpantis, and Vrakas, "Sliding Window Approach for Online Energy Disaggregation Using Artificial Neural Networks", DOI: https://doi.org/10.1145/3200947.3201011.
- Sudoso and Piccialli, "Non-Intrusive Load Monitoring with an Attention-based Deep Neural Network", arXiv:1912.00759, https://arxiv.org/abs/1912.00759.
- MSDC, "Exploiting Multi-State Power Consumption in Non-intrusive Load Monitoring based on A Dual-CNN Model", arXiv:2302.05565, https://arxiv.org/abs/2302.05565.
- Petralia et al., "NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity", arXiv:2506.05880, https://arxiv.org/abs/2506.05880.

Generic architecture references:

- Sutskever, Vinyals, and Le, "Sequence to Sequence Learning with Neural Networks", arXiv:1409.3215, https://arxiv.org/abs/1409.3215.
- He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385, https://arxiv.org/abs/1512.03385.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv:1810.04805, https://arxiv.org/abs/1810.04805.
- Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", arXiv:1506.04214, https://arxiv.org/abs/1506.04214.
- Bai, Kolter, and Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271, https://arxiv.org/abs/1803.01271.
- Kitaev, Kaiser, and Levskaya, "Reformer: The Efficient Transformer", arXiv:2001.04451, https://arxiv.org/abs/2001.04451.

Reference repositories:

- Attention-NILM: https://github.com/antoniosudoso/attention-nilm.
- NILMFormer: https://github.com/adrienpetralia/NILMFormer.
- TCN: https://github.com/locuslab/TCN.

## Usage

The sample notebooks under [sample_notebooks](sample_notebooks) demonstrate the NILMTK rapid experimentation API. Install the relevant backend extra and ensure datasets are available before running them.

Supported experiment workflows include:

- Training and testing across multiple appliances.
- Training and testing across multiple datasets for transfer learning.
- Training and testing across multiple buildings.
- Training and testing with artificial aggregate.
- Training and testing with different sampling frequencies.

## Docker

Build and run locally:

```bash
docker build -t nilmtk-contrib .
docker run --rm -it nilmtk-contrib bash
```

The default Dockerfile installs `.[all]`. Edit the Dockerfile to use `.[torch]`, `.[tensorflow]`, or `.[classical]` for a narrower backend image.

Pull the pre-built image:

```bash
docker pull ghcr.io/enfuego27826/nilmtk-contrib:latest
docker run --rm -it ghcr.io/enfuego27826/nilmtk-contrib:latest bash
```

## Citation

If you find this repository useful for your research, please cite:

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
pages = {193-202},
numpages = {10},
keywords = {smart meters, energy disaggregation, non-intrusive load monitoring},
location = {New York, NY, USA},
series = {BuildSys '19}
}
```
