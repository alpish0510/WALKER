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
    for i in A:
        j = np.random.choice(B)
        
        proposal = propose_stretch(walkers[i], walkers[j], z)
        logp_new = log_prob(proposal)

        log_alpha = (n_dim - 1) * log_z + logp_new - logp[i]
        if np.log(np.random.rand()) < log_alpha:
            walkers[i] = proposal
            logp[i] = logp_new

    # update B using A
    for i in B:
        j = np.random.choice(A)

        z = z_sample(a)
        proposal = propose_stretch(walkers[i], walkers[j], z)
        logp_new = log_prob(proposal)

        log_alpha = (n_dim - 1) * log_z + logp_new - logp[i]
        if np.log(np.random.rand()) < log_alpha:
            walkers[i] = proposal
            logp[i] = logp_new


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
    Bayesian parameter estimation problem definition.

    This class encapsulates:
    - observed data
    - a forward model
    - a prior distribution
    - likelihood and posterior evaluation

    It is agnostic to the sampling algorithm used to explore the
    posterior.
    """
    def __init__(self, x, y, yerr, model, log_prior):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.yerr = np.asarray(yerr)

        self.model = model          # f(x, theta)
        self.log_prior = log_prior  # log p(theta)

    def log_likelihood(self, theta):
        """
        Compute the log-likelihood of the data given model parameters.

        Assumes independent Gaussian observational uncertainties.

        Parameters
        ----------
        theta : ndarray, shape (n_dim,)
            Model parameters.

        Returns
        -------
        log_like : float
            Log-likelihood log p(y | theta).
        """
        y_model = self.model(self.x, theta)
        resid = self.y - y_model

        return -0.5 * np.sum(
            (resid / self.yerr) ** 2
            + np.log(2 * np.pi * self.yerr ** 2)
        )

    def log_posterior(self, theta):
        """
        Compute the log-posterior probability of model parameters.

        Combines the user-defined prior with the Gaussian likelihood.

        Parameters
        ----------
        theta : ndarray, shape (n_dim,)
            Model parameters.

        Returns
        -------
        log_post : float
            Log-posterior log p(theta | y).
            Returns -inf for invalid prior values.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta)
    

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

    def sample(self, n_walkers, n_dim, n_steps, a=2.0, init_scale=1e-2, progress=False):
        """
        Run ensemble MCMC sampling for the defined posterior.

        Parameters
        ----------
        n_walkers : int
            Number of walkers in the ensemble.
        n_dim : int
            Dimensionality of parameter space.
        n_steps : int
            Number of MCMC steps to run.
        a : float, optional
            Stretch scale parameter.
        init_scale : float, optional
            Scale of random Gaussian initialization of walkers.
        progress : bool, optional
            Whether to display a progress bar.
        """
        walkers = init_scale * np.random.randn(n_walkers, n_dim)
        logp = np.array([
            self.estimator.log_posterior(w) for w in walkers
        ])

        # run sampler
        if progress==True:
            self.chain, self.logp_chain = run_sampler(
                walkers,
                logp,
                self.estimator.log_posterior,
                n_steps,
                a=a,
                progress=True
            )
        else:
            self.chain, self.logp_chain = run_sampler(
                walkers,
                logp,
                self.estimator.log_posterior,
                n_steps,
                a=a,
                progress=False
            )

    def extract_samples(self, burnin=0):
        """
        Flatten the ensemble chain into a sample set.

        Parameters
        ----------
        burnin : int, optional
            Number of initial steps to discard.

        Returns
        -------
        samples : ndarray, shape (n_samples, n_dim)
            Flattened posterior samples.
        """
        if self.chain is None:
            raise RuntimeError("You must run sample() first")

        samples = self.chain[burnin:]
        n_steps, n_walkers, n_dim = samples.shape

        self.samples = samples.reshape(n_steps * n_walkers, n_dim)
        return self.samples

    def mean(self):
        """
        Compute the posterior mean of the parameters.

        Returns
        -------
        mean : ndarray, shape (n_dim,)
            Posterior mean estimate.
        """
        return self.samples.mean(axis=0)

    def median(self):
        """
        Compute the posterior median of the parameters.

        Returns
        -------
        median : ndarray, shape (n_dim,)
            Posterior median estimate.
        """
        return np.median(self.samples, axis=0)

    def credible_interval(self, level=0.68):
        """
        Compute marginal Bayesian credible intervals for each parameter.

        The interval is defined by the central quantiles of the posterior
        samples and is computed independently for each parameter dimension.

        Parameters
        ----------
        level : float, optional
            Credible interval probability mass. For example, level=0.68
            returns the 16th and 84th percentiles.

        Returns
        -------
        interval : ndarray, shape (2, n_dim)
            Lower and upper bounds of the credible interval for each
            parameter.

        Notes
        -----
        This method computes marginal (not joint) credible intervals and
        does not account for parameter correlations.
        """
        lo = (1 - level) / 2
        hi = 1 - lo
        return np.quantile(self.samples, [lo, hi], axis=0)

    def map(self):
        """
        Compute the maximum a posteriori (MAP) estimate.

        The MAP estimate is defined as the sampled parameter vector with
        the highest posterior probability among the extracted samples.

        Returns
        -------
        theta_map : ndarray, shape (n_dim,)
            Parameter vector maximizing the posterior within the sampled
            set.

        Notes
        -----
        This method performs a discrete maximization over the sampled
        posterior and does not guarantee the global maximum of the
        posterior distribution.
        """
        logp = np.array([
            self.estimator.log_posterior(s) for s in self.samples
        ])
        return self.samples[np.argmax(logp)]
