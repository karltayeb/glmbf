import jax.numpy as jnp
import numpy as np


def std(p):
    return jnp.sqrt(p * (1 - p))


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def logit(p):
    return jnp.log(p) - jnp.log(1 - p)


def maximum_correlation(p1, p2):
    """
    compute the maximum correlation between two bernoulli random varianbles with marginal probabilites p1, p2
    """
    if p1 > p2:
        return maximum_correlation(p2, p1)
    else:
        return p1 * (1 - p2) / (std(p1) * std(p2))


def minimum_correlation(p1, p2):
    """ """
    return -maximum_correlation(p1, 1 - p2)


def px1x2(rho, p1, p2):
    """
    make conditional probability table p(x1 | x2)
    given marginal probabilties p1 = P(X1=1), p2 = P(X2=1),
    and correlation rho = cor(X1, X2)
    """
    p22 = rho * std(p1) * std(p2) + p1 * p2
    a = (p2 - p22) / (1 - p1)  # P(X2=1 |X1=0)
    b = p22 / p1  # P(X2=1 | X1=1)
    conditional_probs = jnp.array([[1 - a, a], [1 - b, b]])
    return conditional_probs


def binary_x_mle(x, y, glm):
    """
    Fit GLM in special case of model y ~ 1 + x where x takes values 0 or 1
    x: (n,) binary vector
    y: (n,) response variable
    """
    ybar1 = jnp.sum(x * y) / jnp.sum(x)
    ybar0 = jnp.sum((1 - x) * y) / jnp.sum(1 - x)
    b0 = glm.link(ybar0)
    b = glm.link(ybar1) - b0
    return jnp.array([b0, b])


def binary_x_asymptotic_mle(rho, p1, p2, b, glm):
    """
    Compute the limiting value of the MLE y ~ 1 + x2 when true model is y ~ 1 + x1
    where P(X1, X2) is determined by parameters rho, p1, p2
    X1 and X2 are binary

    rho correlation between two sites
    p1 causal allele frequency
    p2 linked allele frequency
    b (2,) vector of coefficients( )intercept, effect)
    glm module of glm functions e.g glmbf.{logistic, poisson}
    """
    x = jnp.array([glm.inv_link(b[0]), glm.inv_link(b[0] + b[1])])
    P = px1x2(rho, p2, p1)  # P(X_1 | X_2)
    y = P @ x
    c0 = glm.link(y[0])
    c = glm.link(y[1]) - c0
    return jnp.array([c0, c])


def p2Xdist(p):
    return dict(atoms=jnp.array([[1.0, 0.0], [1.0, 1.0]]), prob=jnp.array([1 - p, p]))


def simulate_correlated_bernoulli(rho, p1, p2, n):
    rhomax = maximum_correlation(p1, p2)
    assert rho < rhomax
    x1 = np.random.binomial(1, p1, size=n)
    conditional_probabilities = px1x2(rho, p1, p2)[:, 1]
    x2 = np.random.binomial(1, conditional_probabilities[x1])
    return x1, x2
