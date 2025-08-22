import optax
import optax.tree
import jax
import jax.numpy as jnp


def run_opt(init_params, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def check_tol(state, tol=1e-3):
    grad = optax.tree.get(state, "grad")
    err = optax.tree.norm(grad)
    return err < tol


def negate(f):
    def f2(*args, **kwargs):
        return -f(*args, **kwargs)

    return f2


def get_bisection_range(guess, fun):
    """
    What (upper, lower) range should we use during bisection?
    This routine starts at the asymptotic rate and grows/decays exponentially until it finds bounds
    TODO: handle the situation where f is decreasing (or rate < 0) rather than increasing
    """
    lower = jax.lax.while_loop(lambda n: fun(n) > 0, lambda n: 0.5 * n, guess)
    upper = jax.lax.while_loop(lambda n: fun(n) < 0, lambda n: 2 * n, guess)
    return lower, upper


def bisection(init, fun, max_iter, tol, increasing=True):
    """
    bisection root finding of a scalar valued function f
    assumes fun(lower) < 0 < fun(upper)
    geometrically expand from init to get initial range.
    assumes function is increasing
    """
    if not increasing:
        fun = negate(fun)

    def step(carry):
        mid, (interval, it) = carry
        lower, upper = interval
        val = fun(mid)
        new_interval = jax.lax.cond(val > 0, lambda: (lower, mid), lambda: (mid, upper))
        new_midpoint = (new_interval[1] - new_interval[0]) / 2 + new_interval[0]
        return new_midpoint, (new_interval, it + 1)

    def continuing_criterion(carry):
        mid, (interval, iter_num) = carry
        err = jnp.abs(mid)
        return (err > tol) & (iter_num < max_iter)

    lower, upper = get_bisection_range(init, fun)
    init_carry = ((upper - lower) / 2 + lower, ((lower, upper), 0))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


# Examples for increasing and descreasing functions
# f = lambda x: x - 10
# bisection(100., f, 100, 1e-3)
#
# g = lambda x: -x + 10
# bisection(100., f, 100, 1e-3)


# def fun(w):
#     return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)
#
# # Example usage
# opt = optax.lbfgs()
# init_params = jnp.zeros((8,))
# final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
