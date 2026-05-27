# NILMTK-Contrib

NILMTK-Contrib provides NILMTK-compatible implementations of energy disaggregation algorithms and research baselines. The repository paper is Batra et al., "Towards Reproducible State-of-the-Art Energy Disaggregation", BuildSys 2019, DOI: https://doi.org/10.1145/3360322.3360844.

This package currently supports Python `>=3.11,<3.12`. Python 3.12 and newer are unsupported until TensorFlow and NILMTK compatibility is verified.

The historical NILMTK-contrib algorithms and newer experimental backends are tracked in [docs/models.md](docs/models.md). Citation classification is tracked in [docs/references.md](docs/references.md). Server validation protocol and result recording live in [docs/server-validation.md](docs/server-validation.md).

## Runtime Requirements

- Python `>=3.11,<3.12`.
- Backend-specific extras for model use; minimal installs are intentionally lightweight.
- NILMTK-compatible datasets are required for real experiments and notebook runs.
- Backend smoke tests, paper-faithfulness checks, and benchmark claims must be validated on a server environment with the right Python version, dependencies, datasets, and hardware.

## Model Matrix

Status values are intentionally conservative:

- `paper_faithful_unverified`: the cited source is NILM-specific, but implementation correspondence has not been fully validated.
- `paper_inspired`: the model adapts a generic architecture paper to NILM.
- `legacy_baseline`: retained for compatibility; not claimed as a direct NILM paper reproduction.
- `experimental`: citation, implementation, or backend behavior still needs focused audit.

| Algorithm | Backend | Import path | Status | Paper/source | Notes |
|---|---|---|---|---|---|
| AFHMM | Classical | `nilmtk_contrib.disaggregate.AFHMM` | `paper_faithful_unverified` | Kolter and Jaakkola AFHMM | Requires `classical` extra; convex objective and HMM setup still need server audit |
| AFHMM_SAC | Classical | `nilmtk_contrib.disaggregate.AFHMM_SAC` | `paper_faithful_unverified` | Zhong et al. SAC-AFHMM | Requires `classical` extra; SAC aggregate constraint behavior needs focused checks |
| DSC | Classical | `nilmtk_contrib.disaggregate.DSC` | `paper_faithful_unverified` | Kolter, Batra, and Ng DSC | Requires `classical` extra; dictionary persistence is not implemented |
| DAE | TensorFlow | `nilmtk_contrib.disaggregate.DAE` | `paper_faithful_unverified` | Kelly and Knottenbelt Neural NILM | Metadata save/load exists; architecture still needs paper check |
| DAE | PyTorch | `nilmtk_contrib.torch.DAE` | `paper_faithful_unverified` | Kelly and Knottenbelt Neural NILM | Metadata save/load exists; backend parity not server-validated |
| RNN | TensorFlow | `nilmtk_contrib.disaggregate.RNN` | `paper_faithful_unverified` | Kelly and Knottenbelt Neural NILM | Legacy implementation; output form requires backend smoke tests |
| RNN | PyTorch | `nilmtk_contrib.torch.RNN` | `paper_faithful_unverified` | Kelly and Knottenbelt Neural NILM | PyTorch adaptation; parity not server-validated |
| Seq2Point | TensorFlow | `nilmtk_contrib.disaggregate.Seq2Point` | `paper_faithful_unverified` | Zhang et al. Seq2Point | Center target and output length need backend shape tests |
| Seq2PointTorch | PyTorch | `nilmtk_contrib.torch.Seq2PointTorch` | `paper_faithful_unverified` | Zhang et al. Seq2Point | PyTorch adaptation; parity not server-validated |
| Seq2Seq | TensorFlow | `nilmtk_contrib.disaggregate.Seq2Seq` | `legacy_baseline` | Sutskever et al. generic Seq2Seq | Generic architecture citation; NILM-specific source not identified |
| Seq2Seq | PyTorch | `nilmtk_contrib.torch.Seq2Seq` | `legacy_baseline` | Sutskever et al. generic Seq2Seq | Generic architecture citation; overlap behavior needs tests |
| WindowGRU | TensorFlow | `nilmtk_contrib.disaggregate.WindowGRU` | `paper_faithful_unverified` | Krystalakos et al. sliding-window GRU | Online/right-padding semantics need backend checks |
| WindowGRU | PyTorch | `nilmtk_contrib.torch.WindowGRU` | `paper_faithful_unverified` | Krystalakos et al. sliding-window GRU | PyTorch approximation; parity not server-validated |
| RNN_attention | TensorFlow | `nilmtk_contrib.disaggregate.RNN_attention` | `paper_faithful_unverified` | Attention-NILM line | Exact paper mapping needs audit |
| RNN_attention | PyTorch | `nilmtk_contrib.torch.RNN_attention` | `paper_faithful_unverified` | Attention/classification citation unclear | Citation-to-architecture mapping is unverified |
| RNN_attention_classification | TensorFlow | `nilmtk_contrib.disaggregate.RNN_attention_classification` | `paper_faithful_unverified` | Attention-NILM classification line | Threshold metadata now explicit; branch behavior needs checks |
| RNN_attention_classification | PyTorch | `nilmtk_contrib.torch.RNN_attention_classification` | `paper_faithful_unverified` | Attention/classification citation unclear | Threshold metadata now explicit; citation mapping is unverified |
| ResNet | TensorFlow | `nilmtk_contrib.disaggregate.ResNet` | `paper_inspired` | He et al. generic ResNet | 1D residual NILM adaptation |
| ResNet | PyTorch | `nilmtk_contrib.torch.ResNet` | `paper_inspired` | He et al. generic ResNet | 1D residual NILM adaptation; parity not server-validated |
| ResNet_classification | TensorFlow | `nilmtk_contrib.disaggregate.ResNet_classification` | `experimental` | NILM classification citation unclear | Threshold and loss metadata now explicit |
| ResNet_classification | PyTorch | `nilmtk_contrib.torch.ResNet_classification` | `experimental` | NILM classification citation unclear | Threshold and loss metadata now explicit |
| BERT | TensorFlow | `nilmtk_contrib.disaggregate.BERT` | `paper_inspired` | Devlin et al. generic BERT | Transformer/BERT-inspired NILM model; no NLP-style pretraining claim |
| BERT | PyTorch | `nilmtk_contrib.torch.BERT` | `paper_inspired` | Devlin et al. generic BERT | Transformer/BERT-inspired NILM model; tokenization needs audit |
| ConvLSTM | PyTorch | `nilmtk_contrib.torch.ConvLSTM` | `paper_inspired` | Shi et al. generic ConvLSTM | Needs audit to distinguish true ConvLSTM from CNN plus LSTM |
| TCN | PyTorch | `nilmtk_contrib.torch.TCN` | `paper_inspired` | Bai, Kolter, and Koltun generic TCN | Generic sequence-modeling baseline adapted to NILM |
| Reformer | PyTorch | `nilmtk_contrib.torch.Reformer` | `paper_inspired` | Kitaev et al. generic Reformer | LSH attention and reversible residual completeness unverified |
| MSDC | PyTorch | `nilmtk_contrib.torch.MSDC` | `paper_faithful_unverified` | MSDC dual-CNN NILM paper | Canonical CRF-enabled path; state/CRF tests required |
| MSDC without CRF | PyTorch | `nilmtk_contrib.torch.msdc_without_crf.MSDC` | `experimental` | MSDC ablation/source implementation | No-CRF ablation; still exports class name `MSDC` |
| NILMFormer | PyTorch | `nilmtk_contrib.torch.NILMFormer` | `paper_faithful_unverified` | Petralia et al. NILMFormer | Server audit needed before reproduction claims |

## Installation

Install the minimal package when you only need package metadata or lightweight imports:

```bash
uv pip install git+https://github.com/nilmtk/nilmtk-contrib.git
```

Install a backend-specific extra for model use:

```bash
uv pip install "nilmtk-contrib[tensorflow] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
uv pip install "nilmtk-contrib[torch] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
uv pip install "nilmtk-contrib[classical] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

Install all model backends:

```bash
uv pip install "nilmtk-contrib[all] @ git+https://github.com/nilmtk/nilmtk-contrib.git"
```

For development:

```bash
uv sync --extra dev
```

For backend development, include the relevant backend extra:

```bash
uv sync --extra dev --extra torch
uv sync --extra dev --extra tensorflow
uv sync --extra dev --extra classical
```

## Dependencies

- Minimal install: no required runtime dependencies for top-level import.
- `tensorflow` extra: NILMTK, NumPy, pandas, scikit-learn, matplotlib, TensorFlow, and `tensorflow-io-gcs-filesystem`.
- `torch` extra: NILMTK, NumPy, pandas, scikit-learn, matplotlib, PyTorch, and tqdm.
- `classical` extra: NILMTK, NumPy, pandas, matplotlib, scikit-learn, SciPy, cvxpy, and hmmlearn.
- `all` extra: union of TensorFlow, PyTorch, classical, and NILMTK dependencies.
- `dev` extra: pytest, pytest-cov, black, ruff, and build.

## Usage

The sample notebooks under [sample_notebooks](sample_notebooks) show the NILMTK rapid experimentation API. Notebook results are historical examples; rerun them only in a server environment with the relevant datasets and backend extras installed.

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

## Known Limitations

- Paper-faithfulness is tracked per model and remains unverified unless the model row says otherwise.
- TensorFlow and PyTorch paired models may diverge in architecture, preprocessing, training defaults, and output alignment until server parity checks pass.
- No local benchmark results are claimed by this repository state.
- Dataset conversion, notebook execution, backend smoke tests, and training benchmarks must be run on server environments.

## Citation

If you find this repo useful for your research, please cite:

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
