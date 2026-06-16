# NILMBench2026

NILMBench2026 is a large-scale, reproducible benchmark for non-intrusive load monitoring (NILM) and energy disaggregation. It evaluates sixteen models across regression accuracy, event detection, computational efficiency, and generalization on public NILM datasets at both 1-minute and 15-minute resolutions.

The benchmark is implemented through the modernized `nilmtk-contrib` package, which provides NILMTK-compatible disaggregation models and experiment workflows. The package is designed for use with NILMTK's rapid experimentation API and includes classical, TensorFlow, and PyTorch model backends.

This repository is based on the original [`nilmtk-contrib`](https://github.com/nilmtk/nilmtk-contrib) repository and extends it for the NILMBench2026 benchmark.

The repository paper is:

Kuloor, Singh, Dhru, and Batra, "NILMBench2026: A Benchmark for Energy Disaggregation", BuildSys 2026, DOI: https://doi.org/10.1145/3744256.3812587.

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

Local development (from a clone of this repository):

```bash
uv sync --extra dev
```

Backend development examples:

```bash
uv sync --extra dev --extra torch
uv sync --extra dev --extra tensorflow
uv sync --extra dev --extra classical
uv sync --extra dev --extra all
```

### Verify your install

After `uv sync --extra dev`, run a quick sanity check to confirm the package and core tests are in good shape:

```bash
uv run python -m compileall -q nilmtk_contrib tests
uv run python -m pytest -q tests/test_imports.py tests/test_params.py tests/test_preprocessing_windows.py tests/test_preprocessing_alignment.py tests/test_preprocessing_classification.py tests/test_validation.py tests/test_checkpoints.py tests/test_random_logging.py tests/test_model_runtime.py
```

Before launching full experiments, smoke-test the backend you plan to use. Sync the matching extra and run the full test suite—for example, with PyTorch:

```bash
uv sync --extra dev --extra torch
uv run python -m pytest -q
```

## Docker

The repository ships a reproducible container image based on Python 3.11 (Debian Bookworm). The image installs `nilmtk-contrib` with `uv`, pins the Python runtime, and bundles the system libraries needed for NumPy, SciPy, scikit-learn, TensorFlow, and PyTorch.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+ (Docker Desktop on Windows/macOS is fine).
- Optional GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you plan to pass `--gpus all`.

### Pull the pre-built image

```bash
docker pull ghcr.io/nilmtk/nilmtk-contrib:latest
docker run --rm -it ghcr.io/nilmtk/nilmtk-contrib:latest bash
```

Backend-specific tags:

```bash
docker pull ghcr.io/nilmtk/nilmtk-contrib:torch
docker pull ghcr.io/nilmtk/nilmtk-contrib:tensorflow
docker pull ghcr.io/nilmtk/nilmtk-contrib:classical
```

GPU-enabled pre-built image:

```bash
docker run --rm -it --gpus all ghcr.io/nilmtk/nilmtk-contrib:latest bash
```

### Build locally

Default all-backend image:

```bash
docker build -t nilmtk-contrib:all .
```

Backend-specific images (smaller, faster builds):

```bash
docker build -t nilmtk-contrib:torch --build-arg INSTALL_EXTRA=torch .
docker build -t nilmtk-contrib:tensorflow --build-arg INSTALL_EXTRA=tensorflow .
docker build -t nilmtk-contrib:classical --build-arg INSTALL_EXTRA=classical .
```

Image with dev/test dependencies:

```bash
docker build -t nilmtk-contrib:dev --build-arg INSTALL_DEV=true .
```

### Run interactively

```bash
docker run --rm -it nilmtk-contrib:all bash
```

Mount a local dataset directory (read-only) at `/data`:

```bash
docker run --rm -it -v /path/to/datasets:/data:ro nilmtk-contrib:all bash
```

On Windows PowerShell, use a drive path such as `-v C:/Users/you/datasets:/data:ro`.

GPU-enabled shell (requires NVIDIA Container Toolkit):

```bash
docker run --rm -it --gpus all nilmtk-contrib:all bash
```

Inside the container, verify CUDA visibility for PyTorch:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

### Verify the image

Quick package and backend checks:

```bash
docker run --rm nilmtk-contrib:all python -c "import nilmtk_contrib; print(nilmtk_contrib.__version__)"
docker run --rm nilmtk-contrib:all python -c "import nilmtk_contrib.disaggregate, nilmtk_contrib.torch; print('imports ok')"
docker run --rm nilmtk-contrib:all python -c "import nilmtk, torch, tensorflow as tf; print('nilmtk ok'); print('torch', torch.__version__); print('tensorflow', tf.__version__)"
```

Compile and run the lightweight test subset (requires the dev image):

```bash
docker build -t nilmtk-contrib:dev --build-arg INSTALL_DEV=true .
docker run --rm nilmtk-contrib:dev python -m compileall -q nilmtk_contrib tests
docker run --rm nilmtk-contrib:dev python -m pytest -q \
  tests/test_imports.py \
  tests/test_params.py \
  tests/test_preprocessing_windows.py \
  tests/test_preprocessing_alignment.py \
  tests/test_preprocessing_classification.py \
  tests/test_validation.py \
  tests/test_checkpoints.py \
  tests/test_random_logging.py \
  tests/test_model_runtime.py
```

One-shot smoke test after building the default image:

```bash
docker run --rm nilmtk-contrib:all bash -lc "python -c \"import nilmtk_contrib; import torch; import tensorflow as tf; print('nilmtk-contrib', nilmtk_contrib.__version__); print('torch', torch.__version__); print('tensorflow', tf.__version__)\""
```

### Docker build arguments

| Argument | Default | Allowed values | Purpose |
|---|---|---|---|
| `INSTALL_EXTRA` | `all` | `all`, `torch`, `tensorflow`, `classical` | Optional dependency extra to install |
| `INSTALL_DEV` | `false` | `true`, `false` | Also install `.[dev]` for pytest and tooling |

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-backend image definition with build args |
| `.dockerignore` | Keeps build context small and excludes local artifacts |

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

The sample notebooks under [sample_notebooks](sample_notebooks) demonstrate the NILMTK rapid experimentation API used by NILMBench2026. Install the relevant backend extra and ensure datasets are available before running them.

Supported experiment workflows include:

- NILMBench2026 benchmark runs across accuracy, event-detection, efficiency, and generalization metrics.
- Training and testing across multiple appliances.
- Training and testing across multiple datasets for transfer learning.
- Training and testing across multiple buildings.
- Training and testing with artificial aggregate.
- Training and testing with different sampling frequencies.

## Citation

If you use NILMBench2026 or its benchmark results in your research, please cite:

```bibtex
@inproceedings{kuloor2026nilmbench,
  title     = {NILMBench2026: A Benchmark for Energy Disaggregation},
  author    = {Kuloor, Aayush and Singh, Anurag and Dhru, Harsh and Batra, Nipun},
  booktitle = {Proceedings of the 13th ACM International Conference on Systems for
               Energy-Efficient Buildings, Cities, and Transportation (BuildSys '26)},
  year      = {2026},
  doi       = {10.1145/3744256.3812587},
  publisher = {ACM},
  address   = {Banff, AB, Canada}
}
```

If you use `nilmtk-contrib`, please also cite the NILMTK-Contrib paper:

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
  pages = {193--202},
  numpages = {10},
  keywords = {smart meters, energy disaggregation, non-intrusive load monitoring},
  location = {New York, NY, USA},
  series = {BuildSys '19}
}
```
