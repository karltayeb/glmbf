import optax
import optax.tree
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Union, Any
from collections.abc import Callable
from optax._src import base


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


def lbfgs(init_params, fun, max_iter, tol):
    opt = optax.lbfgs()
    return run_opt(init_params, fun, opt, max_iter, tol)


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


class ScaleByInverseHessianState(NamedTuple):
    count: int


def scale_by_inverse_hessian() -> base.GradientTransformationExtraArgs:
    def init_fn(params: base.Params) -> ScaleByInverseHessianState:
        return ScaleByInverseHessianState(0)

    def update_fn(
        updates: base.Updates,
        state: ScaleByInverseHessianState,
        params: base.Params,
        *,
        value: Union[float, jax.Array],
        grad: base.Updates,
        value_fn: Callable[..., Union[jax.Array, float]],
        hess_fn: Callable[..., Union[jax.Array, float]],
        **extra_args: dict[str, Any],
    ) -> tuple[base.Updates, optax.EmptyState]:
        H = hess_fn(params)
        new_updates = -jnp.linalg.solve(H, grad)
        return new_updates, ScaleByInverseHessianState(state.count + 1)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def newton_with_backtracking_line_search(
    init_params,
    fun,
    max_iter,
    tol,
    linesearch_kwargs=dict(max_backtracking_steps=20, store_grad=True),
):
    """
    Newton's method with backtraackign linesearch
    see optax.scale_by_backtracking_linesearch for linesearch options

    # EXAMPLE:
    X = np.random.normal(size = (10000, 2))
    b = np.array([-1, 2]).astype(float)
    y = np.random.binomial(1, 1/(1 + np.exp(- X @ b)))
    obj = lambda b: -logistic.log_likelihood(b, X, y)
    b_init = np.zeros(2)
    params, state = newton_with_backtracking_line_search(b_init, obj, max_iter=100, tol=1e-2)
    """
    opt = optax.chain(
        scale_by_inverse_hessian(),
        optax.scale_by_backtracking_linesearch(**linesearch_kwargs),
    )
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    hess_fn = jax.hessian(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        # compute newton dir, and pass as grad
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun, hess_fn=hess_fn
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


# fun = lambda x: jnp.sum(x**2)
# init_params = np.arange(10).astype(float)
# params, state = newton_with_backtracking_line_search(init_params, fun, max_iter=10, tol=1e-5)
