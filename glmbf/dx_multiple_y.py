import jax.numpy as jnp
import jax
from functools import partial
from glmbf.discrete_x_regression import mle, hessian, log_likelihood
from glmbf import glmbf
from glmbf import logistic
from glmbf.optimization import newton_with_backtracking_line_search
import optax

mle_vmap = jax.jit(
    jax.vmap(mle, (0, dict(n=None, X_unique=None, Ty=0), None, None, None, None)),
    static_argnums=[3, 4, 5],
)
hessian_vmap = jax.jit(
    jax.vmap(hessian, (0, dict(n=None, X_unique=None, Ty=0), None)), static_argnums=[2]
)
ll_vmap = jax.jit(
    jax.vmap(log_likelihood, (0, dict(n=None, X_unique=None, Ty=0), None)),
    static_argnums=[2],
)
compute_stderr2_vmap = jax.jit(jax.vmap(glmbf.stderr2, (0, None)), static_argnums=[1])


def summarize_data_multiply_y(x, Y, size, glm):
    """
    x (n,) is a scalar covariate taking finite values
    Y (k, n) is a matrix of response variables we want to regress onto x
    this function computes the mle by first aggregating the summary statistics per unique value of x
    we sort x and use the sorted indices to efficiently aggregate the summary statistics for all k.
    note: the complexity could be improved by passing through all the data once, but this is good enough for our purposes.
    """
    unique, indices = jnp.unique(x, return_inverse=True, size=size)
    sorted_idx = jnp.argsort(indices)

    X_unique = jnp.stack([jnp.ones(size), unique]).T

    n = jnp.bincount(indices[sorted_idx], minlength=size, length=size)
    cumsize = jnp.cumsum(n)
    indices = jnp.concat([jnp.zeros(1), cumsize])[:-1].astype(int)
    indices = jnp.minimum(indices, jnp.ones(size) * x.size - 1).astype(int)
    Ysorted = Y[:, sorted_idx]
    Ty = jnp.add.reduceat(glm.suffstat(Ysorted), indices, axis=1)
    summarized_data = dict(n=n, X_unique=X_unique, Ty=Ty)
    return summarized_data


@partial(jax.jit, static_argnames=["size", "glm", "opt_fun"])
def compute_summary_stats_multiple_y(
    x, Y, size, penalty=0.01, glm=logistic, opt_fun=newton_with_backtracking_line_search
):
    """
    Assuming x is scalar valued and takes a finite number of values fits y_k ~ 1 + x for multiple y
    penalty is an l2 penalty on the coefficients. typically set to a small value to closely approximate the mle while stabilizing optimzation
    """
    summarized_data = summarize_data_multiply_y(x, Y, size, glm)

    b_init = jnp.array([glm.link(Y.mean(1) + 1e-5), jnp.zeros(Y.shape[0])]).T

    Bhat, optstate = mle_vmap(b_init, summarized_data, penalty, glm, -1, opt_fun)
    gradnorm = jnp.sqrt(jnp.sum(optax.tree.get(optstate, "grad") ** 2, 1))
    H = hessian_vmap(Bhat, summarized_data, glm)
    ll0 = ll_vmap(b_init, summarized_data, glm)
    ll = ll_vmap(Bhat, summarized_data, glm)
    llr = ll - ll0
    s2 = compute_stderr2_vmap(H, -1)[:, 1]

    res = dict(
        intercept=Bhat[:, 0],
        bhat=Bhat[:, 1],
        s2=s2,
        llr=llr,
        null_intercept=b_init[0, 0],
        ll0=ll0,
        gradnorm=gradnorm,
    )
    return res
