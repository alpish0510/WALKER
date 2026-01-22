# WALKER: Weighted Affine Likelihood Kernel for Ensemble Randomization

WALKER is a minimal affine‑invariant ensemble MCMC sampler using stretch moves, plus a lightweight Bayesian parameter‑estimation wrapper. It keeps state explicit (walkers + logp), avoids adaptive heuristics, and separates posterior definition from sampling.

## Features
- Affine‑invariant stretch move sampler
- Explicit `Transform` framework for constrained parameters
- Simple `ParamEstimator` + `MCMCfit` interface
- Optional progress bar via `tqdm`

## Installation
From the repo root:

```bash
pip install .
```

Editable install for development:

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
transform = wk.Identity()

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

## Transforms
Use transforms to enforce constraints:
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

## Module
- [walker.py](walker.py)

## License
MIT (see repository license file).