import numpy as numpy
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


# Define the link functions for exponential regression
def log_partition(x):
    return -jnp.log(-x)


inv_link = jnp.exp
link = jnp.log


def sample(x):
    return np.random.exponential(1 / inv_link(x))


def log_base_measure(y):
    return 0  # For exponential, the base measure is 0


def suffstat(y):
    return y


def kl(rate_p, rate_q):
    """
    compute KL(p || q) = E_p log (p/q)
                       = \\lambda_q / \\lambda_p - 1 - log(\\lambda_q / \\lambda_p)
    """
    return rate_q / rate_p - 1 - jnp.log(rate_q / rate_p)


def entropy(rate):
    return 1 - jnp.log(rate)


def cross_entropy(rate_p, rate_q):
    return kl(rate_p, rate_q) + entropy(rate_p)


def conditional_expected_log_likelihood(b1, b2, x):
    """
    Compute E_b2[log p(Y | X=x, b1)]

    Parameters:
    - b1: Parameter value at which the log-likelihood is evaluated (log rate for Exponential ~ exp(b1)).
    - b2: Parameter value under which the data are generated (log rate for Exponential ~ exp(b2)).
    - x: Observed data or covariate value.

    Returns:
    - Expected log-likelihood of the Exponential distribution given the parameters.
    """
    # rate parameters
    rate1 = jnp.exp(jnp.sum(b1 * x))
    rate2 = jnp.exp(jnp.sum(b2 * x))
    return -cross_entropy(rate2, rate1)


def log_likelihood(b, X, y):
    psi = X @ b
    return jnp.sum(-psi - y * jnp.exp(-psi))


def conditional_information(b1, b2, x):
    """
    compute E_{b2}[nabla^2_b log p(Y | X, b) | b=b1, X=x]
    by averaging over values of x we can get the full information
    note that we set b1=b2=true parameter value to get the information
    """
    hessian = jax.hessian(log_likelihood)
    hessian_vmap = jax.vmap(lambda y: hessian(b1, x[None], y))
    y = jnp.linspace(0, 10, 1000)  # Adjust range as needed
    rate2 = inv_link(jnp.sum(x * b2))
    p = jnp.exp(-rate2 * y) * rate2  # Exponential PDF
    return -jnp.sum(p[:, None, None] * hessian_vmap(y), axis=0)
