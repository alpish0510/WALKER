import numpy as np


def z_sample(a=2.0):
    u=np.random.rand()
    z=((a-1)*u+1)**2/a
    return z

def propose_stretch(theta_i, theta_j, z):
    """
    Propose a new position for theta_i using theta_j as reference.
    """
    return theta_j + z * (theta_i - theta_j)

def accept_move(logp_old, logp_new, z, n_dim):    
    log_alpha = (n_dim - 1) * np.log(z) + logp_new - logp_old
    return np.log(np.random.rand()) < log_alpha


def ensemble_step(walkers, logp, log_prob, a):
    n_walkers, n_dim = walkers.shape

    # shuffle and split
    idx = np.random.permutation(n_walkers)
    half = n_walkers // 2
    A = idx[:half]
    B = idx[half:]

    # update A using B
    for i in A:
        j = np.random.choice(B)

        z = z_sample(a)
        proposal = propose_stretch(walkers[i], walkers[j], z)
        logp_new = log_prob(proposal)

        if accept_move(logp[i], logp_new, z, n_dim):
            walkers[i] = proposal
            logp[i] = logp_new

    # update B using A
    for i in B:
        j = np.random.choice(A)

        z = z_sample(a)
        proposal = propose_stretch(walkers[i], walkers[j], z)
        logp_new = log_prob(proposal)

        if accept_move(logp[i], logp_new, z, n_dim):
            walkers[i] = proposal
            logp[i] = logp_new


def run_sampler(walkers, logp, log_prob, n_steps, a=2.0):
    """
    Run affine-invariant ensemble sampler.

    Parameters
    ----------
    walkers : ndarray, shape (n_walkers, n_dim)
    logp : ndarray, shape (n_walkers,)
    log_prob : callable
    n_steps : int
    a : float

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, n_dim)
    logp_chain : ndarray, shape (n_steps, n_walkers)
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
    def __init__(self, x, y, yerr, model, log_prior):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.yerr = np.asarray(yerr)

        self.model = model          # f(x, theta)
        self.log_prior = log_prior  # log p(theta)

    def log_likelihood(self, theta):
        "This function computes the log likelihood (p(y|theta)) given some data and model"
        y_model = self.model(self.x, theta)
        resid = self.y - y_model

        return -0.5 * np.sum(
            (resid / self.yerr) ** 2
            + np.log(2 * np.pi * self.yerr ** 2)
        )

    def log_posterior(self, theta):
        "This function computes the log posterior (p(theta|y)) given the prior and the likelihood"
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta)
    

def run_sampler(walkers, logp, log_prob, n_steps, a=2.0):
    """
    Run affine-invariant ensemble sampler.

    Parameters
    ----------
    walkers : ndarray, shape (n_walkers, n_dim)
    logp : ndarray, shape (n_walkers,)
    log_prob : callable
    n_steps : int
    a : float

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, n_dim)
    logp_chain : ndarray, shape (n_steps, n_walkers)
    """
    n_walkers, n_dim = walkers.shape

    chain = np.zeros((n_steps, n_walkers, n_dim))
    logp_chain = np.zeros((n_steps, n_walkers))

    for t in range(n_steps):
        ensemble_step(walkers, logp, log_prob, a)
        chain[t] = walkers
        logp_chain[t] = logp

    return chain, logp_chain


class MCMCfit:
    def __init__(self, estimator):
        """
        estimator : ParamEstimator
        """
        self.estimator = estimator

        self.chain = None
        self.logp_chain = None
        self.samples = None

    def sample(self, n_walkers, n_dim, n_steps, a=2.0, init_scale=1e-2):
        # initialize walkers
        walkers = init_scale * np.random.randn(n_walkers, n_dim)
        logp = np.array([
            self.estimator.log_posterior(w) for w in walkers
        ])

        # run sampler
        self.chain, self.logp_chain = run_sampler(
            walkers,
            logp,
            self.estimator.log_posterior,
            n_steps,
            a=a
        )

    def extract_samples(self, burnin=0):
        if self.chain is None:
            raise RuntimeError("You must run sample() first")

        samples = self.chain[burnin:]
        n_steps, n_walkers, n_dim = samples.shape

        self.samples = samples.reshape(n_steps * n_walkers, n_dim)
        return self.samples

    def mean(self):
        return self.samples.mean(axis=0)

    def median(self):
        return np.median(self.samples, axis=0)

    def credible_interval(self, level=0.68):
        lo = (1 - level) / 2
        hi = 1 - lo
        return np.quantile(self.samples, [lo, hi], axis=0)

    def map(self):
        logp = np.array([
            self.estimator.log_posterior(s) for s in self.samples
        ])
        return self.samples[np.argmax(logp)]
