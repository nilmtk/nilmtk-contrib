# Latent Bayesian Melding: solver-free implementation plan

## Decision

Latent Bayesian Melding (LBM) is feasible in `nilmtk-contrib` without MOSEK,
CVXPY, SciPy, or another optimizer in the PyTorch runtime. It should not be
presented as an ordinary neural network: the method is structured MAP inference
over appliance HMMs and population-level summary priors.

The implementation must be clean-room. The authors' reference repository does
not contain a license file, and its implementation imports MOSEK 7 for Python
2.7. We can cite the repository and paper, but must not copy or translate its
source code into this Apache-2.0 package.

Primary references:

- [Latent Bayesian Melding for integrating individual and population models](https://proceedings.neurips.cc/paper/2015/hash/312351bff07989769097660a56395065-Abstract.html)
- [Authors' reference repository](https://github.com/MingjunZhong/LatentBayesianMelding)

## Why a PyTorch-only solver is possible

For fixed noise variances, the paper relaxes appliance states and transition
indicators to an HMM path polytope. Its objective is quadratic or linear and its
constraints are linear. A generic conic solver is one way to solve that
problem, but it is not the only way.

Conditional gradient (Frank--Wolfe) needs only a linear minimization oracle.
For an HMM path polytope that oracle is Viterbi. When the latent variable is the
number of off-to-on cycles, Viterbi gains a cycle-count dimension and returns
the cheapest path for each reachable count. The paper constrains only equality
between expected path cycles and the relaxed category expectation; it does not
require their full distributions to match. The remaining two-simplex linear
program has a closed-form enumeration whose vertices use at most three active
supports. This solves the paper's larger expected-count domain without a
generic LP solver. Every conditional-gradient iterate therefore satisfies the
state, transition-flow, and expected-cycle constraints.

This approach gives two properties that unconstrained Adam/LBFGS over logits
would not give:

- a Frank--Wolfe duality gap that upper-bounds relaxed suboptimality;
- exact simplex, HMM-flow, and expected-cycle consistency at every iterate,
  up to floating-point arithmetic.

The private `nilmtk_contrib.torch._lbm` kernel implements the state/transition
and cycle-prior part of this fixed-variance problem in float64. Appliance-level
Gaussian signal noise is collapsed into the aggregate observation variance;
the public wrapper must fit and record that effective variance. Gaussian
population/induced-prior ratios are accepted only when the population variance
is smaller than the induced variance, leaving a positive quadratic precision.
The convex extension agrees with the original Gaussian ratio whenever the
cycle category is integral. Setting the melding weight to zero removes both
the cycle constraint and all population terms, recovering the ordinary
relaxed AFHMM state problem.

## Training and provenance contract

The private `nilmtk_contrib.torch._lbm_training` fitter consumes dense,
equal-length population windows. Every window records the dataset and version,
a stable data URI, SHA-256 fingerprint, building, timezone-aware half-open
interval, and sample period. It rejects overlapping training windows so the
same evidence cannot be counted twice.

Evaluation buildings are disjoint by default: using any window from an
evaluation building is treated as leakage even when the timestamps differ.
A same-building temporal study must opt in explicitly, and overlapping times
remain forbidden. This is stricter than merely retaining a path to the source
file and makes the cross-building NILMbench boundary enforceable in code.

Training is deterministic and PyTorch-only:

- one-dimensional state means use deterministic Lloyd iterations;
- initial and transition probabilities use explicit pseudocounts, and
  transitions never cross source-window boundaries;
- cycle probabilities and cycle-conditioned daily energy/duration summaries
  use population windows only;
- induced summary moments are computed exactly from the fitted Markov chain,
  without Monte Carlo sampling;
- sparse cycle categories, non-convex population/HMM ratios, and numerical
  overflow are rejected rather than silently repaired.

The resulting metadata is JSON-safe and includes every fitted probability,
summary statistic, source window, and the appliance emission variance. Model
artifact persistence remains part of the wrapper PR.

## What this first kernel does not claim

The numerical kernel is intentionally private: it is neither exported as a
disaggregator nor included in the model catalog. It does not yet:

- adapt real NILMTK chunks into provenance-complete population windows or
  orchestrate multi-appliance fitting;
- alternate the paper's closed-form variance updates;
- infer the non-negative piecewise-smooth unknown-load signal;
- return the paper's separate latent appliance signals rather than the
  collapsed state-mean prediction;
- save and validate a model artifact with dataset/protocol provenance;
- reproduce the paper's HES-to-UK-DALE experiment;
- establish accuracy, speed, or parity on a real NILMbench task.

Calling this a benchmarkable `LBM` model before those gates pass would confuse
a tested numerical primitive with a reproduced scientific method.

## Small follow-up PRs

1. **Structured kernel.** Merge the private PyTorch conditional-gradient
   solver with synthetic feasibility, optimality-gap, determinism, adversarial,
   and optional CUDA tests.
2. **NILMTK training contract.** Fit state means, initial/transition
   probabilities, cycle categories, and daily energy/duration priors from
   training data only. Use explicit pseudocounts and reject underspecified
   population data rather than silently leaking test-house statistics.
3. **LBM inference wrapper.** Add variance alternation and the non-negative
   total-variation unknown-load block, then expose `partial_fit` and
   `disaggregate_chunk`.
4. **Scientific validation.** Compare the relaxation against tiny enumerated
   cases; run T0 on real REDD; then run the full cross-building protocol before
   adding a leaderboard claim.

## Acceptance gates for the public model

- State simplex and HMM flow residuals at most `1e-7` in float64.
- Expected off-to-on transitions equal the relaxed cycle category expectation
  within `1e-7`.
- Reported Frank--Wolfe gap at or below the configured absolute/relative
  tolerance; non-convergence is surfaced, never relabeled as success.
- Objective history is non-increasing under exact quadratic line search.
- Identical CPU results for repeated runs with identical inputs.
- CUDA parity within a documented tolerance when CUDA testing is enabled.
- No `cvxpy`, `mosek`, or `scipy` import in the PyTorch inference path.
- Population summaries are fitted only from declared training buildings and
  carry source-window provenance.
- T0 results remain labeled smoke; full leaderboard status still requires the
  NILMbench seed/provenance protocol.
