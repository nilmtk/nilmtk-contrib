# Explicit-duration models for NILM

## Why add them

An ordinary HMM implies a geometric dwell-time distribution. That is a poor
default for appliances whose off, heating, cooling, or cycle states have
characteristic durations. A hidden semi-Markov model (HSMM) separates the
probability of the next state from the probability of the current state's
duration.

This is established model structure, not a new NILM claim. Useful primary
references are:

- [Explicit-Duration Markov Switching Models](https://arxiv.org/abs/1909.05800)
- [A time-efficient factorial hidden Semi-Markov model for non-intrusive load monitoring](https://doi.org/10.1016/j.epsr.2021.107372)
- [Neural Semi-Markov Conditional Random Fields](https://aclanthology.org/N19-1280/)

## Shared exact core

`nilmtk_contrib.torch._semi_markov` is a private PyTorch implementation of the
semi-Markov forward and Viterbi recurrences. It accepts per-sample emission,
initial-state, between-segment transition, and segment-duration scores. The
transition diagonal must be negative infinity: state persistence is represented
only by duration, so every state sequence has one maximal-segment encoding.

The forward recurrence gives an exact differentiable log-partition. Viterbi
gives the exact best segmentation and uses the same score contract. This one
core will support both planned models:

1. a classical explicit-duration HSMM with fitted Gaussian emissions and
   duration histograms;
2. a neural semi-Markov model whose compact temporal encoder learns emission
   scores while exact decoding still enforces duration and transition structure.

## Classical fitting layer

The private `nilmtk_contrib.torch._hsmm` layer now fits the classical model
from aligned mains and appliance chunks. It shares deterministic ordered state
fitting with LBM, learns initial and between-segment transition probabilities,
an explicit duration histogram for each state, and Gaussian aggregate
emissions conditioned on the labeled appliance state. Inference splits long
chunks only at the configured duration cap and sends every block through the
exact shared Viterbi recurrence.

The fitter is intentionally supervised and one-target-at-a-time. It is a
small, benchmarkable baseline, not a claim to reproduce the factorial
multi-appliance inference or timing features of TE-FHSMM. Model persistence is
fail-closed until the next artifact PR, and the class remains private until a
real-data T0 run succeeds.

## Neural scoring layer

The private `nilmtk_contrib.torch._neural_semi_markov` layer supplies the
matched modern variant. A compact length-preserving dilated TCN produces
contextual per-sample state scores. Initial-state, off-diagonal transition, and
explicit-duration scores are trainable and normalized in log space. Training
uses the exact semi-Markov log-partition rather than token-wise cross entropy;
decoding calls the same Viterbi core as the classical model.

Classical fitted probabilities can initialize the neural structure. This makes
the comparison controlled: the neural method can begin from the classical
state/duration prior and differ primarily in its learned contextual emission
model. The full NILMTK training wrapper remains a separate PR so optimizer,
windowing, checkpointing, and real-data validation do not get hidden inside
the numerical-kernel review.

Persistence, public export, the neural model, and NILMbench entries remain
separate PRs. Neither model enters the public catalog until its complete
training/inference contract and real-data T0 have passed.
