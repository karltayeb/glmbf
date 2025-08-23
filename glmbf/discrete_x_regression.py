# This module has functions for fitting GLMs assuming finite number of unique settings of the design matrix
# For GLMs this data may be summarized by summing the sufficient statistics for each unique row of X

import jax
from functools import partial
import jax.numpy as jnp
import optax
from glmbf.optimization import run_opt
import numpy as np
from scipy.stats import binom
from glmbf.binary_x_regression import px1x2
from scipy.stats import poisson_binom


def Xdist_binom(n, p):
    prob = binom.pmf(np.arange(n + 1), n, p)
    atoms = np.stack([np.ones(n + 1), np.arange(n + 1)]).T
    return dict(atoms=atoms, prob=prob)


def summarize_data(X, y, size, glm):
    X_unique, indices = jnp.unique(X, axis=0, return_inverse=True, size=size)
    Ty = jnp.zeros(X_unique.shape[0])
    n = Ty.at[indices].add(jnp.ones_like(y))
    Ty = Ty.at[indices].add(glm.suffstat(y))
    return dict(n=n, Ty=Ty, X_unique=X_unique, indices=indices)


@partial(jax.jit, static_argnames=["glm", "fixed"])
def mle(b_init, data, glm, fixed=(-1)):
    """
    When X is discrete you can summarize the data for each unique value of X and then fit the glm
    this avoids requiring O(n) cost to each update of IRLS.
    Need to specify the number of unique X ahead of time with the size arugment
    Using fixed indices you can fix parameters to their initial values in b_init,
        that is useful for dealing with the offset. Just add a column to X and give it fixed effect 1.
    """
    n, Ty, X_unique = data["n"], data["Ty"], data["X_unique"]
    fixed = jnp.isin(jnp.arange(b_init.size), jnp.atleast_1d(fixed))

    def objective(b):
        b = jax.lax.select(fixed, b_init, b)
        psi = X_unique @ b
        return -jnp.sum(Ty * psi - n * glm.log_partition(psi)) + jnp.sum(b**2) * 1e-8

    opt = optax.lbfgs()
    bhat, optstate = run_opt(b_init, objective, opt, max_iter=1000, tol=1e-5)
    return bhat, optstate


def log_likelihood(b, data, glm):
    n, Ty, X_unique = data["n"], data["Ty"], data["X_unique"]
    psi = X_unique @ b
    ll = Ty * psi - n * glm.log_partition(psi)
    return jnp.sum(ll)


def hessian(bhat, data, glm):
    n, Ty, X_unique = data["n"], data["Ty"], data["X_unique"]
    psihat = X_unique @ bhat
    H = -X_unique.T @ (
        (jax.vmap(jax.hessian(glm.log_partition))(psihat) * n)[:, None] * X_unique
    )
    return H


def row_normalize(A):
    return A / A.sum(1)[:, None]


def risk(c, b, X1dist, X2dist, P, glm):
    eta1 = X1dist["atoms"] @ b
    eta2 = X2dist["atoms"] @ c
    mean = jax.vmap(jax.grad(glm.log_partition))(eta1)
    pi2 = P.sum(0)
    P21 = row_normalize(P.T)
    return -jnp.sum(pi2 * (P21 @ mean * eta2 - jax.vmap(glm.log_partition)(eta2)))


def compute_asymptotic_mle(c_init, b, X1dist, X2dist, P, glm):
    opt = optax.lbfgs()
    objective = lambda c: risk(c, b, X1dist, X2dist, P, glm)
    c, _ = run_opt(c_init, objective, opt, max_iter=100, tol=1e-5)
    return c


def px1x2_binom_row(k, n, rho, p1, p2):
    """
    n = int, total number of trials
    x1 = int, number of successes
    P table of conditional probabilities P[0, 1] = P(X2 = 1 | X1=0), etc.
    """
    p = px1x2(rho, p1, p2)[:, 1]
    probs = np.concat([np.ones(k) * p[1], np.ones(n - k) * p[0]])
    return poisson_binom.pmf(np.arange(n + 1), probs)


def px1x2_binom(n, rho, p1, p2):
    """
    Compute P(X2 | X1) where X1 and X2 are marginally binomial (n, p1), (n, p2) respectively
    Dependence induced by correlated Bernoullis Z1, Z2 \sim Bernoulli(rho, p1, p2)
    """
    return np.array([px1x2_binom_row(k, n, rho, p1, p2) for k in np.arange(n + 1)])
