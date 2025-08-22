from jax.scipy.stats import norm as jnorm
from functools import partial
import jax
import jax.numpy as jnp
from glmbf.optimization import run_opt, negate
import optax
import numpy as np
from jax.scipy.stats import poisson as jpoisson
from jax.scipy.special import gammaln
from glmbf import logistic, poisson


def log_likelihood(b, X, y, glm):
    """
    compute the log likelihood logp(y | X.T @ b)

    b (p, )jnp.ndarray of coefficient
    X (n, p)
    y (n)
    """
    psi = X @ b
    ll = glm.suffstat(y) * psi - glm.log_partition(psi) + glm.log_base_measure(y)
    return jnp.sum(ll)


def expected_log_likelihood(b1, b2, Xdist, glm):
    """
    Compute the expected log likelihood at a parameter value b1 when (X, Y) distributed according to (Xdist, b2)
    that is E_Xdist [E_b2[log p(Y | X, b1)] | X = x]
    note that the parameters under which the data are generated (b2) might differ from the parameter value they are being evaluated at (b)
    Xdist is a dictionary containing 'atoms' and 'prob'
    """
    X_atoms = Xdist["atoms"]
    prob = Xdist["prob"]
    conditional = jnp.array(
        [glm.conditional_expected_log_likelihood(b1, b2, x) for x in X_atoms]
    )
    return jnp.sum(conditional * prob)


def fisher_information(b1, b2, Xdist, glm):
    """
    Compute the expected log likelihood at a parameter value b1 when (X, Y) distributed according to (Xdist, b2)
    that is E_X [E_{Y ~ (b2, X)}[\nabla^2_b log p(Y | X, b1)] | X = x]
    note that the parameters under which the data are generated (b2) might differ from the parameter value they are being evaluated at (b).
    When b1 = b2 we get the information
    Xdist is a dictionary containing 'atoms' and 'prob'
    """
    X_atoms = Xdist["atoms"]
    prob = Xdist["prob"]
    conditional = jnp.array([glm.conditional_information(b1, b2, x) for x in X_atoms])
    return (conditional * prob[:, None, None]).sum(0)


def compute_score(b, X, y, glm):
    return jax.grad(log_likelihood)(b, X, y, glm)


def compute_hessian(b, X, y, glm):
    return jax.hessian(log_likelihood)(b, X, y, glm)


@partial(jax.jit, static_argnames=["glm"])
def compute_mle(b_init, X, y, glm):
    objective = negate(partial(log_likelihood, X=X, y=y, glm=glm))

    opt = optax.lbfgs()
    final_b, opt_state = run_opt(b_init, objective, opt, max_iter=1000, tol=1e-3)
    return final_b, opt_state


@partial(jax.jit, static_argnames=["glm"])
def compute_null_mle(b_init, X, y, glm):
    objective = negate(partial(log_likelihood, X=X, y=y, glm=glm))

    def null_objective(b):
        return objective(jnp.concat([b, jnp.zeros(1)]))

    opt = optax.lbfgs()
    final_b, opt_state = run_opt(b_init, null_objective, opt, max_iter=1000, tol=1e-3)
    return final_b, opt_state


def standard_errors(H, idx=-1):
    a = jnp.trace(H)
    return -jnp.diag(jnp.linalg.inv(H / a)) / a


@jax.jit
def compute_log_abf(bhat, H, prior_variance):
    s2 = standard_errors(H)[-1]
    return jnorm.logpdf(bhat[1], 0, jnp.sqrt(s2 + prior_variance)) - jnorm.logpdf(
        bhat[1], 0, jnp.sqrt(s2)
    )


@jax.jit
def compute_log_laplacebf(bhat, H, llr, prior_variance):
    s2 = standard_errors(H)[-1]
    return (
        0.5 * jnp.log(2 * jnp.pi * s2)
        + llr
        + jnorm.logpdf(bhat[1], 0, jnp.sqrt(s2 + prior_variance))
    )


@partial(jax.jit, static_argnames=["glm"])
def expected_log_abf(n, b, Xdist, prior_variance, glm):
    I = fisher_information(b, b, Xdist, glm)
    return compute_log_abf(b, -n * I, prior_variance) + 0.5 * jnp.trace(
        jnp.linalg.solve(n * I, jax.hessian(compute_log_abf)(b, -n * I, prior_variance))
    )


@partial(jax.jit, static_argnames=["glm"])
def expected_log_abf_rate(b, Xdist, glm):
    I = fisher_information(b, b, Xdist, glm)
    s2 = jnp.linalg.inv(I)[1, 1]
    return 0.5 * b[1] ** 2 / s2


def compute_b0(b, Xdist, glm):
    """
    Compute the limiting value of the MLE under and intercept only model
    The computation is link(E_X[E[Y | b, X] | X])
    """
    b0 = glm.link(
        jnp.sum(
            jnp.array(
                [
                    glm.inv_link(jnp.sum(atom * b)) * p
                    for atom, p in zip(Xdist["atoms"], Xdist["prob"])
                ]
            )
        )
    )
    return jnp.array([b0, 0.0])


@partial(jax.jit, static_argnames=["glm"])
def expected_log_laplacebf(n, b, Xdist, prior_variance, glm):
    """
    n: int sample size
    b: nd.array true effect
    b0: nd.array limiting value of effects under null model
    I: information matrix
    prior_variance: prior variance of effect
    """
    I = fisher_information(b, b, Xdist, glm)
    s2 = jnp.linalg.inv(n * I)[1, 1]

    def f(b):
        return 0.5 * jnp.log(2 * jnp.pi * s2) + jnorm.logpdf(
            b[1], 0, jnp.sqrt(s2 + prior_variance)
        )

    Ell = expected_log_likelihood(b, b, Xdist, glm)
    b0 = compute_b0(b, Xdist, glm)
    Ell0 = expected_log_likelihood(b0, b, Xdist, glm)
    Ellr = n * (Ell - Ell0)
    res = (
        f(b) + 0.5 * jnp.trace(jnp.linalg.solve(n * I, jax.hessian(f)(b))) + Ellr + 0.5
    )
    return res


@partial(jax.jit, static_argnames=["glm"])
def expected_log_laplacebf_rate(b, Xdist, glm):
    """
    b: nd.array true effect
    b0: nd.array limiting value of effects under null model
    I: information matrix
    prior_variance: prior variance of effect
    """
    b0 = compute_b0(b, Xdist, glm)
    Ell = expected_log_likelihood(b, b, Xdist, glm)
    Ell0 = expected_log_likelihood(b0, b, Xdist, glm)
    Ellr = Ell - Ell0
    return Ellr


def sample_Xdist(Xdist, n):
    return Xdist["atoms"][np.random.choice(Xdist["prob"].size, size=n, replace=True)]


def sample_rep(b, Xdist, n, k, glm, fun, **kwargs):
    """
    Sample the log
    fun is a function with signature (X, b, glm, **kwargs)
    it will typically simulate y from X@b and then compute some statistic of the data (X, y), e.g. the logABF

    b: vector of parameter value
    n: int number of samples of x to draw
    k: number of times to repeat the simulation
    """
    # simulate a fixed X
    X = sample_Xdist(Xdist, n)
    samples = np.array([fun(X, b, glm, **kwargs) for _ in range(k)])
    return samples


def sample_log_abf(X, b, glm, prior_variance=1.0):
    psi = X @ b
    y = glm.sample(psi)
    b_init = jnp.zeros(X.shape[1] - 1)
    bhat0, _ = compute_null_mle(b_init, X, y, glm)
    bhat0 = np.concat([bhat0, np.zeros(1)])
    bhat, _ = compute_mle(bhat0, X, y, glm)
    H = compute_hessian(bhat, X, y, glm)
    logabf = compute_log_abf(bhat, H, prior_variance)
    return logabf


def sample_log_laplacebf(X, b, glm, prior_variance=1.0):
    psi = X @ b
    y = glm.sample(psi)
    b_init = jnp.zeros(X.shape[1] - 1)
    bhat0, _ = compute_null_mle(b_init, X, y, glm)
    bhat0 = np.concat([bhat0, np.zeros(1)])
    bhat, _ = compute_mle(bhat0, X, y, glm)

    # compute log likelihood ratio
    ll = partial(log_likelihood, X=X, y=y, glm=glm)
    llr = ll(bhat) - ll(bhat0)
    H = compute_hessian(bhat, X, y, glm)
    loglaplace = compute_log_laplacebf(bhat, H, llr, prior_variance)
    return loglaplace


def get_available_glms():
    return {"logistic": logistic, "poisson": poisson}
