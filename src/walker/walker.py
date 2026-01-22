"""
WALKER: Weighted Affine Likelihood Kernel for Ensemble Randomization

This module implements a minimal affine-invariant ensemble MCMC sampler
using stretch moves, along with a lightweight Bayesian parameter
estimation framework.

The design philosophy is:
- explicit state (walkers and log-probabilities)
- no hidden adaptation or heuristics
- clear separation between probability definition, sampling dynamics,
  and posterior analysis

The sampler operates on an ensemble of walkers and requires only a
callable log-posterior function.
"""



import numpy as np



def z_sample(a=2.0):
    """
    Draw a stretch-move scale factor.

    Samples the stretch variable `z` from the distribution

        g(z) ∝ 1 / sqrt(z),   z ∈ [1/a, a]

    as required by the affine-invariant ensemble sampler.

    Parameters
    ----------
    a : float, optional
        Stretch scale parameter controlling proposal size.
        Must satisfy a > 1.

    Returns
    -------
    z : float
        Random stretch factor.
    """
    u=np.random.rand()
    z=((a-1)*u+1)**2/a
    return z

def propose_stretch(theta_i, theta_j, z):
    """
    Generate a stretch-move proposal for a single walker.

    The proposal is constructed as

        x' = x_j + z * (x_i - x_j)

    where `x_j` is a randomly chosen complementary walker.

    Parameters
    ----------
    x_i : ndarray, shape (n_dim,)
        Current position of the walker being updated.
    x_j : ndarray, shape (n_dim,)
        Position of the complementary walker.
    z : float
        Stretch factor drawn from the stretch distribution.

    Returns
    -------
    proposal : ndarray, shape (n_dim,)
        Proposed new position for the walker.
    """
    return theta_j + z * (theta_i - theta_j)


def ensemble_step(walkers, logp, log_prob, a):
    """
    Perform one affine-invariant ensemble update step.

    The ensemble is randomly split into two complementary subsets.
    Each subset is updated conditionally on the other using stretch
    moves. Updates are performed in place.

    Parameters
    ----------
    walkers : ndarray, shape (n_walkers, n_dim)
        Current positions of all walkers.
    logp : ndarray, shape (n_walkers,)
        Log-posterior values corresponding to `walkers`.
        Updated in place.
    log_prob : callable
        Function computing log-posterior given a parameter vector.
    a : float
        Stretch scale parameter.

    Notes
    -----
    This function mutates `walkers` and `logp` in place.
    """
    n_walkers, n_dim = walkers.shape

    # shuffle and split
    idx = np.random.permutation(n_walkers)
    half = n_walkers // 2
    A = idx[:half]
    B = idx[half:]

    z = z_sample(a)
    log_z = np.log(z)

    # update A using B
    # freeze complementary ensemble
    walkers_B = walkers[B].copy()

    # pair without replacement
    B_perm = np.random.permutation(len(B))

    for ii, jj in zip(A, B_perm):
        z = z_sample(a)
        log_z = np.log(z)

        proposal = propose_stretch(walkers[ii], walkers_B[jj], z)
        logp_new = log_prob(proposal)

        log_alpha = (n_dim - 1) * log_z + logp_new - logp[ii]
        if np.log(np.random.rand()) < log_alpha:
            walkers[ii] = proposal
            logp[ii] = logp_new

    # freeze complementary ensemble
    walkers_B = walkers[B].copy()

    # pair without replacement
    B_perm = np.random.permutation(len(B))

    for ii, jj in zip(A, B_perm):
        z = z_sample(a)
        log_z = np.log(z)

        proposal = propose_stretch(walkers[ii], walkers_B[jj], z)
        logp_new = log_prob(proposal)

        log_alpha = (n_dim - 1) * log_z + logp_new - logp[ii]
        if np.log(np.random.rand()) < log_alpha:
            walkers[ii] = proposal
            logp[ii] = logp_new


def run_sampler(walkers, logp, log_prob, n_steps, a=2.0):
    """
    Run an affine-invariant ensemble MCMC simulation.

    Repeatedly applies ensemble stretch-move updates and records the
    full chain of walker positions and log-posterior values.

    Parameters
    ----------
    walkers : ndarray, shape (n_walkers, n_dim)
        Initial walker positions. Updated in place.
    logp : ndarray, shape (n_walkers,)
        Initial log-posterior values. Updated in place.
    log_prob : callable
        Function computing log-posterior given a parameter vector.
    n_steps : int
        Number of MCMC steps to run.
    a : float, optional
        Stretch scale parameter.

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, n_dim)
        Recorded walker positions at each step.
    logp_chain : ndarray, shape (n_steps, n_walkers)
        Recorded log-posterior values at each step.

    Notes
    -----
    This function does not perform burn-in removal, thinning, or
    convergence diagnostics.
    """
    n_walkers, n_dim = walkers.shape

    chain = np.zeros((n_steps, n_walkers, n_dim))
    logp_chain = np.zeros((n_steps, n_walkers))

    for t in range(n_steps):
        ensemble_step(walkers, logp, log_prob, a)
        chain[t] = walkers
        logp_chain[t] = logp

    return chain, logp_chain



class ParamEstimator:
    """
    Bundle data, model, prior, and parameter transform into a log-posterior callable.

    This class evaluates the posterior in a transformed (internal) parameter space
    using a user-supplied `Transform`. The sampler operates on the internal
    parameters `u`, which are mapped to physical parameters `theta`.

    Parameters
    ----------
    x, y : array_like
        Observed data.
    yerr : array_like
        1σ observational uncertainties for `y` (same shape as `y`).
    model : callable
        Model function `f(x, theta)` returning predicted y.
    log_prior : callable
        Log prior `log p(theta)` in physical parameter space.
    transform : Transform
        Mapping between internal parameters `u` and physical parameters `theta`.

    Methods
    -------
    log_likelihood(theta)
        Gaussian log-likelihood in physical parameter space.
    log_posterior(u)
        Log posterior in internal space, including Jacobian correction.
    """

    def __init__(self, x, y, yerr, model, log_prior, transform):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.yerr = np.asarray(yerr)

        self.model = model              # f(x, theta)
        self.log_prior = log_prior      # log p(theta)
        self.transform = transform      # Transform instance

    def log_likelihood(self, theta):
        y_model = self.model(self.x, theta)
        resid = self.y - y_model

        return -0.5 * np.sum(
            (resid / self.yerr) ** 2
            + np.log(2 * np.pi * self.yerr ** 2)
        )

    def log_posterior(self, u):
        # 1. Transform internal → physical
        theta = self.transform.forward(u)

        # 2. Prior in physical space
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        # 3. Likelihood in physical space
        ll = self.log_likelihood(theta)

        # 4. Jacobian correction
        lj = self.transform.log_jacobian(u)

        return lp + ll + lj

    

def run_sampler(walkers, logp, log_prob, n_steps, a=2.0, progress=False):
    """
    Run affine-invariant ensemble sampler.

    Parameters
    ----------
    walkers : ndarray, shape (n_walkers, n_dim)
    logp : ndarray, shape (n_walkers,)
    log_prob : callable
    n_steps : int
    a : float
    progress : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, n_dim)
    logp_chain : ndarray, shape (n_steps, n_walkers)
    """
    n_walkers, n_dim = walkers.shape

    chain = np.zeros((n_steps, n_walkers, n_dim))
    logp_chain = np.zeros((n_steps, n_walkers))
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator=range(n_steps)
    if progress==True and tqdm is not None:
        iterator=tqdm(iterator, desc="Sampling", unit="step")
    for t in iterator:
        ensemble_step(walkers, logp, log_prob, a)
        chain[t] = walkers
        logp_chain[t] = logp

    return chain, logp_chain


class MCMCfit:
    """
    High-level interface for ensemble MCMC parameter estimation.

    This class orchestrates:
    - posterior evaluation via a ParamEstimator
    - sampling using an affine-invariant ensemble sampler
    - basic posterior analysis from MCMC samples

    It does not modify sampler dynamics or adapt proposals.
    """
    def __init__(self, estimator):
        """
        estimator : ParamEstimator
        """
        self.estimator = estimator

        self.chain = None
        self.logp_chain = None
        self.samples = None

    def sample(self, n_walkers, n_steps, a=2.0, init_scale=1e-2, progress=False):
        """
        Run ensemble MCMC sampling for the defined posterior.

        Parameters
        ----------
        n_walkers : int
            Number of walkers in the ensemble.
        n_steps : int
            Number of MCMC steps to run.
        a : float, optional
            Stretch scale parameter.
        init_scale : float, optional
            Scale of random Gaussian initialization of walkers
            in internal (unconstrained) parameter space.
        progress : bool, optional
            Whether to display a progress bar.
        """

        # infer dimensionality from transform
        n_dim = len(self.estimator.transform.transforms)

        # initialize walkers in internal space
        walkers = init_scale * np.random.randn(n_walkers, n_dim)

        # evaluate initial log posterior
        logp = np.array([
            self.estimator.log_posterior(w) for w in walkers
        ])

        # sanity check
        if not np.any(np.isfinite(logp)):
            raise RuntimeError(
                "All initial walkers have non-finite log posterior. "
                "Increase init_scale or check priors."
            )

        # run sampler
        self.chain, self.logp_chain = run_sampler(
            walkers,
            logp,
            self.estimator.log_posterior,
            n_steps,
            a=a,
            progress=progress
        )


    def _extract_samples(self, burnin=0, thin=1, physical=False):
        """
        Extract flattened posterior samples.

        Parameters
        ----------
        burnin : int
            Number of initial steps to discard.
        thin : int
            Thinning factor.
        physical : bool
            If True, return samples in physical parameter space.
            If False, return internal (unconstrained) samples.

        Returns
        -------
        samples : ndarray, shape (n_samples, n_dim)
            Flattened posterior samples.
        """
        if self.chain is None:
            raise RuntimeError("You must run sample() first")

        chain = self.chain[burnin::thin]
        u_samples = chain.reshape(-1, chain.shape[-1])

        if not physical:
            return u_samples

        return np.array([
            self.estimator.transform.forward(u)
            for u in u_samples
        ])


    def mean(self, burnin=0, thin=1):
        """
        Posterior mean of parameters in physical space.

        Parameters
        ----------
        burnin : int
            Burn-in steps to discard.
        thin : int
            Thinning factor.
        physical : bool
            If True, return samples in physical parameter space.
            If False, return internal (log) samples.
        
        Returns
        -------
        mean : ndarray, shape (n_dim,)
            Posterior mean of each parameter.
        """
        theta = self._extract_samples(burnin, thin, physical=True)
        return theta.mean(axis=0)


    def median(self, burnin=0, thin=1):
        """
        Posterior median of parameters in physical space.

        Parameters
        ----------
        burnin : int
            Burn-in steps to discard.
        thin : int
            Thinning factor.
        physical : bool
            If True, return samples in physical parameter space.
            If False, return internal (log) samples.
        
        Returns
        -------
        median : ndarray, shape (n_dim,)
            Posterior median of each parameter.
        """
        theta = self._extract_samples(burnin, thin, physical=True)
        return np.median(theta, axis=0)


    def credible_interval(self, level=0.68, burnin=0, thin=1):
        """
        Credible interval for each parameter in physical space.

        Parameters
        ----------
        level : float
            Credible level (e.g. 0.68 or 0.95).
        burnin : int
            Burn-in steps to discard.
        thin : int
            Thinning factor.

        Returns
        -------
        intervals : list of tuples
            [(low, high), ...] for each parameter.
        """
        theta = self._extract_samples(burnin, thin, physical=True)

        alpha = (1.0 - level) / 2.0
        lo = np.percentile(theta, 100 * alpha, axis=0)
        hi = np.percentile(theta, 100 * (1 - alpha), axis=0)

        return list(zip(lo, hi))


    def map(self, burnin=0):
        """
        Maximum a posteriori estimate in physical space.

        Parameters
        ----------
        burnin : int
            Burn-in steps to discard.
        
        Returns
        -------
        map_estimate : ndarray, shape (n_dim,)
            MAP estimate of each parameter. 
        """
        logp = self.logp_chain[burnin:].reshape(-1)
        idx = np.argmax(logp)

        u_map = self.chain[burnin:].reshape(-1, self.chain.shape[-1])[idx]
        return self.estimator.transform.forward(u_map)


class Transform:
    """
    Base class for parameter transforms between internal and physical space.

    Subclasses must implement:
    - `forward(u)`: map internal parameters to physical parameters.
    - `log_jacobian(u)`: log absolute Jacobian determinant of the transform.
    """
    def forward(self, u):
        """Map internal parameters `u` → physical parameters `θ`."""
        raise NotImplementedError

    def log_jacobian(self, u):
        """Return log |dθ/du| for the transform."""
        raise NotImplementedError


class Identity(Transform):
    """
    Identity transform: θ = u.

    Useful when parameters are already unconstrained.
    """
    def forward(self, u):
        return u

    def log_jacobian(self, u):
        return 0.0


class Log(Transform):
    """
    Log transform: θ = exp(u), for strictly positive parameters.

    The Jacobian is |dθ/du| = exp(u), so log|dθ/du| = u.
    """
    def forward(self, u):
        return np.exp(u)

    def log_jacobian(self, u):
        return u


class Logit(Transform):
    """
    Logit transform mapping u ∈ (-∞, ∞) to θ ∈ [a, b].

    Parameters
    ----------
    a, b : float
        Lower and upper bounds of the target interval.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, u):
        s = 1.0 / (1.0 + np.exp(-u))
        return self.a + (self.b - self.a) * s

    def log_jacobian(self, u):
        return (
            np.log(self.b - self.a)
            - u
            - 2.0 * np.log1p(np.exp(-u))
        )


class CompositeTransform(Transform):
    """
    Apply a list of transforms component-wise to a parameter vector.

    Parameters
    ----------
    transforms : sequence of Transform
        One transform per parameter dimension.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, u):
        return np.array([
            t.forward(ui) for t, ui in zip(self.transforms, u)
        ])

    def log_jacobian(self, u):
        return sum(
            t.log_jacobian(ui) for t, ui in zip(self.transforms, u)
        )