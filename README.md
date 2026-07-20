# nilmtk-contrib

This repository contains maintained NILMTK-compatible disaggregation models.
Use it to run, test, or contribute an algorithm.

## Ecosystem repositories

| Research task | Repository |
| --- | --- |
| Dataset conversion, meter access, preprocessing, and metrics | [NILMTK core](https://github.com/nilmtk/nilmtk) |
| Appliance taxonomy, synonyms, meter relationships, and dataset schema | [NILM Metadata](https://github.com/nilmtk/nilm_metadata) |
| Disaggregation model implementation and testing | **nilmtk-contrib — this repository** |
| Fixed T1/T2/T3 evaluation and published result bundles | [NILMbench](https://github.com/nilmtk/nilmbench) |

The [NILMTK start page](https://nilmtk.github.io/) gives the supported install,
Docker, citation, and contribution routes for the whole ecosystem.

## Citation

If you use this model suite or its rapid experimentation interface, cite the
nilmtk-contrib paper:

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

Also cite the original paper for every model and dataset you use. Cite the
[NILMBench2026 paper](https://doi.org/10.1145/3744256.3812587) only when using
its protocols, runner, or leaderboard results.

## Install and verify

The supported environment is Python `>=3.11,<3.12`. Use Python 3.11.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv venv --python 3.11
source .venv/bin/activate
UV_TORCH_BACKEND=cpu uv pip install \
  "nilmtk-contrib[torch] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
python -c "from nilmtk_contrib.torch import Seq2PointTorch; print('ready')"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Choose a different extra only when you need that backend:

| Extra | Use it for |
| --- | --- |
| `torch` | PyTorch models, including current time-series and MoE models |
| `classical` | AFHMM and AFHMM-SAC |
| `all` | PyTorch and classical models; largest install |
| `nilm` | NILMTK integration without a model backend |

Replace `torch` in the install command with the required extra. A bare install
contains package metadata and dependency-light utilities; it cannot train a
backend model.

### Backend policy

PyTorch is the maintained backend for neural and state-space models. Historical
package-level imports such as `nilmtk_contrib.disaggregate.DAE` now resolve to
the corresponding PyTorch class and emit a `FutureWarning`; new code should
import from `nilmtk_contrib.torch`. The duplicated TensorFlow implementations
and their direct module paths have been removed. The remaining classical AFHMM
implementations stay available until equivalent Torch implementations pass
numerical and real-data comparison tests.

Historical compatibility imports map as follows:

| Compatibility import | Maintained import |
| --- | --- |
| `nilmtk_contrib.disaggregate.BERT` | `nilmtk_contrib.torch.BERT` |
| `nilmtk_contrib.disaggregate.DAE` | `nilmtk_contrib.torch.DAE` |
| `nilmtk_contrib.disaggregate.DSC` | `nilmtk_contrib.torch.DSC` |
| `nilmtk_contrib.disaggregate.RNN` | `nilmtk_contrib.torch.RNN` |
| `nilmtk_contrib.disaggregate.RNN_attention` | `nilmtk_contrib.torch.RNN_attention` |
| `nilmtk_contrib.disaggregate.RNN_attention_classification` | `nilmtk_contrib.torch.RNN_attention_classification` |
| `nilmtk_contrib.disaggregate.ResNet` | `nilmtk_contrib.torch.ResNet` |
| `nilmtk_contrib.disaggregate.ResNet_classification` | `nilmtk_contrib.torch.ResNet_classification` |
| `nilmtk_contrib.disaggregate.Seq2Point` | `nilmtk_contrib.torch.Seq2PointTorch` |
| `nilmtk_contrib.disaggregate.Seq2Seq` | `nilmtk_contrib.torch.Seq2Seq` |
| `nilmtk_contrib.disaggregate.WindowGRU` | `nilmtk_contrib.torch.WindowGRU` |

## Run a model

Public model imports are listed in the [model table](#models). For example:

```python
from nilmtk_contrib.torch import Seq2PointTorch

model = Seq2PointTorch(
    {
        "sequence_length": 99,
        "n_epochs": 1,
        "batch_size": 32,
        "device": "cpu",
        "seed": 0,
    }
)
```

Training requires NILMTK-compatible mains and appliance data. The
[`sample_notebooks`](sample_notebooks) directory shows the rapid experimentation
API. Use [NILMbench](https://github.com/nilmtk/nilmbench) when the goal is a
comparable published result rather than an exploratory run.

## Development

```bash
git clone https://github.com/nilmtk/nilmtk-contrib.git
cd nilmtk-contrib
uv sync --frozen --group dev --extra torch
uv run ruff check nilmtk_contrib tests scripts
uv run pytest -q
uv build
```

CI also smoke-tests every exported PyTorch model for one epoch. Run that gate
locally when model discovery, shared preprocessing, or the base class changes:

```bash
uv run pytest tests/test_model_smoke_synthetic.py \
  --run-model-smoke --model-smoke-backend torch \
  --model-smoke-epochs 1 -q
```

## Add a model

Keep a model PR focused on the implementation. It should:

1. use the shared validation, preprocessing, checkpoint, device, seed, and
   logging utilities instead of copying them;
2. expose the model lazily from the backend package;
3. test defaults, parameter validation, short/partial chunks, serialization,
   determinism, CPU inference, and the intended CUDA path;
4. enter the all-model smoke test and public model table;
5. cite the original model paper and state clearly when an architecture is a
   NILM adaptation rather than a paper-faithful reproduction.

After the model PR merges, open a separate NILMbench PR for its adapter and
search space. A model reaches the leaderboard only through a provenance-complete
real-data result bundle.

## Docker

This repository owns the one general NILMTK development Dockerfile. It contains
core, metadata, and the selected contrib backend. Do not add an image per model.

Build the image locally with Docker 24 or newer. Anonymous pulls from the GHCR
package are not yet part of the supported path, so this README does not publish
a `docker pull` command that may require organization access.

NILMbench owns separate pinned CPU-smoke and CUDA benchmark images. Those images
certify results; they are not general development images.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  only when using `--gpus all` on Linux.

### Build locally

The default build includes the PyTorch and classical backends:

```bash
docker build -t nilmtk-contrib:all .
```

Prefer a narrower backend for faster development builds:

```bash
docker build -t nilmtk-contrib:torch --build-arg INSTALL_EXTRA=torch .
docker build -t nilmtk-contrib:classical --build-arg INSTALL_EXTRA=classical .
```

Add tests and development tools only when needed:

```bash
docker build -t nilmtk-contrib:dev --build-arg INSTALL_DEV=true .
```

### Run interactively

```bash
docker run --rm -it nilmtk-contrib:all bash
```

Datasets are not included. Mount a licensed local dataset directory read-only:

```bash
docker run --rm -it -v /path/to/datasets:/data:ro nilmtk-contrib:all bash
```

On Windows PowerShell, use a drive path such as
`-v C:/Users/you/datasets:/data:ro`.

GPU-enabled shell (requires NVIDIA Container Toolkit):

```bash
docker run --rm -it --gpus all nilmtk-contrib:all bash
```

### Verify the image

Check the package and selected backend after building:

```bash
docker run --rm nilmtk-contrib:all python -c "import nilmtk_contrib; print(nilmtk_contrib.__version__)"
docker run --rm nilmtk-contrib:all python -c \
  "from nilmtk_contrib.torch import Seq2PointTorch; print('ready')"
```

The development image runs the same suite as the local environment:

```bash
docker build -t nilmtk-contrib:dev --build-arg INSTALL_DEV=true .
docker run --rm nilmtk-contrib:dev python -m compileall -q nilmtk_contrib tests
docker run --rm nilmtk-contrib:dev python -m pytest -q
```

### Docker build arguments

| Argument | Default | Allowed values | Purpose |
|---|---|---|---|
| `INSTALL_EXTRA` | `all` | `all`, `torch`, `classical` | Optional dependency extra to install |
| `INSTALL_DEV` | `false` | `true`, `false` | Also install `.[dev]` for pytest and tooling |

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-backend image definition with build args |
| `.dockerignore` | Keeps build context small and excludes local artifacts |

## Models

The table below lists the public model surface. "Verification" describes how the implementation should be cited and interpreted in research use.

| Algorithm | Backend | Import path | Verification | Paper/source | Notes |
|---|---|---|---|---|---|
| AFHMM | Classical | `nilmtk_contrib.disaggregate.AFHMM` | NILM paper implementation, not independently benchmark-certified in this package state | Kolter and Jaakkola, AFHMM for energy disaggregation | Requires `classical` extra |
| AFHMM_SAC | Classical | `nilmtk_contrib.disaggregate.AFHMM_SAC` | NILM paper implementation, not independently benchmark-certified in this package state | Zhong, Goddard, and Sutton, signal aggregate constraints in AFHMMs | Requires `classical` extra |
| DAE | PyTorch | `nilmtk_contrib.torch.DAE` | PyTorch implementation requiring parity validation for new claims | Kelly and Knottenbelt, Neural NILM | PyTorch backend |
| DLinear | PyTorch | `nilmtk_contrib.torch.DLinear` | DLinear-inspired sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Zeng et al., DLinear | PyTorch backend |
| DSC | PyTorch | `nilmtk_contrib.torch.DSC` | Solver-free non-negative DSC port; numerical parity passed on REDD, UK-DALE, and REFIT | Kolter, Batra, and Ng, discriminative sparse coding | Proximal sparse-code objective is checked against scikit-learn gold solutions; the historical import is a compatibility wrapper |
| HSMM | PyTorch | `nilmtk_contrib.torch.HSMM` | Supervised single-appliance explicit-duration baseline; benchmark claims require NILMbench result bundles | Chiappa, explicit-duration Markov switching models | Exact PyTorch dynamic program; no external solver |
| RNN | PyTorch | `nilmtk_contrib.torch.RNN` | PyTorch implementation requiring parity validation for new claims | Kelly and Knottenbelt, Neural NILM | PyTorch backend |
| Seq2PointTorch | PyTorch | `nilmtk_contrib.torch.Seq2PointTorch` | PyTorch implementation requiring parity validation for new claims | Zhang et al., Sequence-to-Point Learning | PyTorch backend |
| Seq2Seq | PyTorch | `nilmtk_contrib.torch.Seq2Seq` | Legacy NILM baseline adapted from a generic sequence model | Sutskever, Vinyals, and Le, sequence-to-sequence learning | Generic architecture citation |
| WindowGRU | PyTorch | `nilmtk_contrib.torch.WindowGRU` | PyTorch implementation requiring parity validation for new claims | Krystalakos, Nalmpantis, and Vrakas, sliding-window GRU | PyTorch backend |
| RNN_attention | PyTorch | `nilmtk_contrib.torch.RNN_attention` | PyTorch attention-based NILM implementation | Attention-based NILM literature | PyTorch backend |
| RNN_attention_classification | PyTorch | `nilmtk_contrib.torch.RNN_attention_classification` | PyTorch attention-based NILM implementation with classification branch | Attention-based NILM literature | Explicit on/off threshold parameters are supported |
| ResNet | PyTorch | `nilmtk_contrib.torch.ResNet` | 1D residual NILM adaptation of a generic architecture | He et al., Deep Residual Learning | Generic computer-vision architecture adapted to NILM |
| ResNet_classification | PyTorch | `nilmtk_contrib.torch.ResNet_classification` | Residual NILM model with classification branch | Residual and NILM classification literature | Explicit threshold and loss-weight parameters are supported |
| BERT | PyTorch | `nilmtk_contrib.torch.BERT` | Transformer/BERT-inspired NILM adaptation | Devlin et al., BERT | Does not claim NLP-style pretraining |
| ConvLSTM | PyTorch | `nilmtk_contrib.torch.ConvLSTM` | ConvLSTM-inspired NILM adaptation | Shi et al., ConvLSTM | Generic spatiotemporal architecture adapted to NILM |
| TCN | PyTorch | `nilmtk_contrib.torch.TCN` | Generic TCN sequence-modeling baseline adapted to NILM | Bai, Kolter, and Koltun, TCN | PyTorch backend |
| SGN | PyTorch | `nilmtk_contrib.torch.SGN` | Subtask-gated sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Shin et al., SGN | Uses a raw-power-correct soft on/off gate and auxiliary classification loss |
| TSMixer | PyTorch | `nilmtk_contrib.torch.TSMixer` | All-MLP sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Chen et al., TSMixer | Mixes along time and feature dimensions without attention or recurrence |
| TimesNet | PyTorch | `nilmtk_contrib.torch.TimesNet` | TimesNet-inspired sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Wu et al., TimesNet | PyTorch backend |
| Reformer | PyTorch | `nilmtk_contrib.torch.Reformer` | Reformer-inspired NILM adaptation | Kitaev, Kaiser, and Levskaya, Reformer | Efficient Transformer architecture adapted to NILM |
| MSDC | PyTorch | `nilmtk_contrib.torch.MSDC` | NILM paper implementation requiring experiment validation for new claims | MSDC dual-CNN NILM paper | Canonical CRF-enabled implementation path |
| MSDC without CRF | PyTorch | `nilmtk_contrib.torch.msdc_without_crf.MSDC` | MSDC ablation | MSDC paper/source implementation | No-CRF ablation, not the canonical MSDC path |
| ModernTCN | PyTorch | `nilmtk_contrib.torch.ModernTCN` | ModernTCN-inspired sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Luo and Wang, ModernTCN | PyTorch backend |
| NILMFormer | PyTorch | `nilmtk_contrib.torch.NILMFormer` | NILMFormer implementation requiring experiment validation for new claims | Petralia et al., NILMFormer | PyTorch backend |
| NILMMoE | PyTorch | `nilmtk_contrib.torch.NILMMoE` | Experimental input-conditioned mixture; benchmark claims require NILMbench result bundles | This repository | Blends DLinear, ModernTCN, and TimesNet with a load-balanced softmax gate |
| PatchTST | PyTorch | `nilmtk_contrib.torch.PatchTST` | PatchTST-inspired sequence-to-point adaptation; benchmark claims require NILMbench result bundles | Nie et al., PatchTST | PyTorch backend |
| ResidualMoE | PyTorch | `nilmtk_contrib.torch.ResidualMoE` | Experimental conservative residual mixture; benchmark claims require NILMbench result bundles | This repository | Starts exactly at TimesNet and learns a bounded signed correction from PatchTST and ModernTCN |

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
- Chiappa, "Explicit-Duration Markov Switching Models", arXiv:1909.05800, https://arxiv.org/abs/1909.05800.
- Wu et al., "A time-efficient factorial hidden Semi-Markov model for non-intrusive load monitoring", Electric Power Systems Research 2021, https://doi.org/10.1016/j.epsr.2021.107372.
- Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", arXiv:2211.14730, https://arxiv.org/abs/2211.14730.
- Luo and Wang, "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis", ICLR 2024, https://openreview.net/forum?id=vpJMJerXHU.
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023, https://arxiv.org/abs/2205.13504.
- Shin et al., "Subtask Gated Networks for Non-Intrusive Load Monitoring", AAAI 2019, https://doi.org/10.1609/aaai.v33i01.33011150.
- Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting", TMLR 2023, https://arxiv.org/abs/2303.06053.
- Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023, https://openreview.net/forum?id=ju_Uqw384Oq.

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

## Data, notebooks, and benchmark results

This package does not redistribute licensed datasets or publish leaderboard
rows. Download each dataset from its official custodian and convert it with
[NILMTK core](https://github.com/nilmtk/nilmtk).

The notebooks under [`sample_notebooks`](sample_notebooks) demonstrate the
NILMTK rapid experimentation API for exploratory work across appliances,
buildings, datasets, and sample rates. They are examples, not frozen benchmark
protocols.

For comparable real-data runs, pinned environments, provenance-complete result
bundles, and the living leaderboard, use
[NILMbench](https://github.com/nilmtk/nilmbench).
