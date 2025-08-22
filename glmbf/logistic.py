# Logistic functions
import jax.numpy as jnp
import jax
import numpy as np


def log_partition(x):
    return jnp.log(1 + jnp.exp(x))


def link(x):
    return jnp.log(x) - jnp.log(1 - x)


def inv_link(x):
    return 1 / (1 + jnp.exp(-x))


def sample(x):
    return np.random.binomial(1, inv_link(x))


def log_base_measure(y):
    return 0


def suffstat(y):
    return y


def log_likelihood(b, X, y):
    psi = X @ b
    ll = y * psi - jnp.log(1 + jnp.exp(psi))
    return jnp.sum(ll)


def conditional_expected_log_likelihood(b1, b2, x):
    """
    compute E_b2[log p(Y | X=x, b1)]
    note that the parameters under which the data are generated (b2) might differ from the parameter value they are being evaluated at (b1)
    """
    p = inv_link(jnp.sum(x * b2))
    return p * log_likelihood(b1, x[None], 1.0) + (1 - p) * log_likelihood(
        b1, x[None], 0.0
    )


def conditional_information(b1, b2, x):
    """
    compute E_{b2}[nabla^2_b log p(Y | X, b) | b=b1, X=x]
    by averaging over values of x we can get the full information
    note that we set b1=b2=true parameter value to get the information
    """
    p = inv_link(jnp.sum(x * b2))  # probability Y=1
    hessian = jax.hessian(log_likelihood)
    return -(p * hessian(b1, x[None], 1.0) + (1 - p) * hessian(b1, x[None], 0.0))


def compute_marginal_py(b, Xdist):
    """
    compute the marginal probability P(Y=1) for data simulated according to (Xdist, b)
    """
    return np.array(
        [inv_link(np.sum(b * x)) * p for x, p in zip(Xdist["atoms"], Xdist["prob"])]
    ).sum()


def intercept_only(y):
    return np.array([link(y.mean()), 0.0])
