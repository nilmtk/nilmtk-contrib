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

The classical model, neural model, and NILMbench entries remain separate PRs.
Neither model enters the public catalog until its complete training/inference
wrapper and real-data T0 have passed.
