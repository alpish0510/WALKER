# WALKER — Weighted Affine Likelihood Kernel for Ensemble Randomization

*A minimal, affine-invariant ensemble MCMC sampler with explicit state and transform-aware parameter estimation.*

WALKER is a lightweight affine-invariant ensemble MCMC sampler using stretch moves, with a small Bayesian parameter-estimation wrapper. It is designed for users who want explicit control over walker state and log-posterior evaluation, without adaptive heuristics, hidden tuning, or black-box automation.

Sampling, posterior definition, and diagnostics are kept deliberately separate.

---

## Features

- Affine-invariant stretch-move ensemble sampler  
- Explicit walker state and log-posterior bookkeeping  
- Clear separation between posterior definition, sampling, and diagnostics  
- Transform-aware constrained parameter sampling with exact Jacobian corrections  
- `ParamEstimator` wrapper for model + likelihood + prior + transform  
- Lightweight `MCMCfit` interface for common summaries  
- Optional progress bar via `tqdm`  
- Designed for clarity and extensibility over automation  

---

## Installation

### Install from GitHub
```bash
pip install git+https://github.com/USERNAME/WALKER.git
cd WALKER
pip install .
```

### Editable install for development:

```bash
pip install -e .
```

## Quick start
```python
import numpy as np
import walker as wk

def model(x, theta):
    m, b = theta
    return m * x + b

def log_prior(theta):
    m, b = theta
    return 0.0 if (-10 < m < 10 and -10 < b < 10) else -np.inf

# Identity transform for unconstrained params
transform = wk.CompositeTransform([
    wk.Identity(),
    wk.Identity()
])

x = np.linspace(0, 10, 50)
yerr = 0.2 * np.ones_like(x)
y = model(x, [3.0, 1.2]) + np.random.normal(0, yerr)

estimator = wk.ParamEstimator(
    x, y, yerr,
    model=model,
    log_prior=log_prior,
    transform=transform
)

fit = wk.MCMCfit(estimator)
fit.sample(n_walkers=30, n_steps=3000, progress=True)

print("Mean:", fit.mean(burnin=1000))
print("Median:", fit.median(burnin=1000))
print("MAP:", fit.map(burnin=1000))
```

## Transforms and constrained parameters
WALKER samples in an unconstrained internal parameter space.
Transforms map internal parameters to physical space, and exact Jacobian corrections are applied automatically to ensure valid posteriors.

Available transforms:
- `Identity` → unconstrained parameters
- `Log` → strictly positive parameters
- `Logit(a, b)` → parameters bounded to $[a,b]$
- `CompositeTransform([...])` → mixed constraints

Example:
```python
transform = wk.CompositeTransform([
    wk.Log(),      # positive
    wk.Log(),      # positive
    wk.Identity()  # unconstrained
])
```
## API overview
`EnsembleSampler` — affine-invariant ensemble sampler core
`ParamEstimator` — combines model, likelihood, prior, and transform
`MCMCfit` — convenience interface for sampling and posterior summaries

## Module
- [walker.py](walker.py)

## License
MIT (see repository license file).