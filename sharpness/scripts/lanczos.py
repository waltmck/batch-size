"""Lanczos with full reorthogonalisation on flat vectors, with the two stopping rules of the write-up
(sharpness/sharpness.ipynb) and the probabilistic upper bound of van Dorsselaer, Hochstenbach and
van der Vorst, "Computing probabilistic bounds for extreme eigenvalues of symmetric matrices with the Lanczos method",
SIAM J. Matrix Anal. Appl. 22(3), 2000.

After k steps from the unit start q_1 the basis Q_k = (q_1..q_k) is orthonormal, T_k = Q_k^T A Q_k is tridiagonal with
diagonal alpha_1..alpha_k and off-diagonal beta_1..beta_{k-1}, and beta_k = ||A q_k - (its projection on K_k)|| is the norm
that produces q_{k+1}. Ritz values theta_j (descending) and vectors y_j = Q_k s_j come from T_k; the residual of the j-th
Ritz pair is r_j = ||A y_j - theta_j y_j|| = beta_k |s_j[k]|.

direction(): search for a vector close to the top eigenvector from the given unit start. Stops at the first k >= min_steps for which
  (r_1 / (SpecGap(T_k) - r_2))^2 <= eps_star  at step k and at step k - 1,  SpecGap(T_k) = theta_1 - theta_2,
or at max_steps. Returns the top Ritz vector v_k.

upper_bound(): from a uniformly random unit start, L_k = theta_1 = lambda_max(T_k) <= lambda_max(A) always, and with
probability >= 1 - delta over the start, p_k(lambda_max(A)) <= C_{k,delta} for the characteristic polynomial
p_k(x) = det(xI - T_k) and C_{k,delta} = beta_1 ... beta_k / gamma_delta, where gamma_delta is such that the start's component
along any fixed unit vector is below gamma_delta with probability delta (so this holds simultaneously for all k). U_k = the
root x > L_k of p_k(x) = C_{k,delta} is then an upper bound for lambda_max(A). Stops at the first k >= min_steps with
  (U_k - L_k) / |L_k| <= eps_star
or at max_steps; returns U_k (and L_k).
"""
import math
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from scipy.linalg import eigh_tridiagonal
from scipy.special import betaincinv


def gamma_threshold(d, delta):
    """gamma such that a uniformly random unit vector in R^d has |component along a fixed unit vector| <= gamma with
    probability delta (the squared component is Beta(1/2, (d-1)/2))."""
    return math.sqrt(float(betaincinv(0.5, (d - 1) / 2.0, delta)))


HIGHEST = jax.lax.Precision.HIGHEST                       # the small Lanczos linear algebra in true fp32 whatever the run's matmul default


@partial(jax.jit, static_argnames=('rows',))
def _init(q1, rows):                                       # the basis buffer with q_1 in row 0 (built under jit: one allocation)
    return jnp.zeros((rows, q1.shape[0]), dtype=jnp.float32).at[0].set(q1 / jnp.linalg.norm(q1))


@partial(jax.jit, donate_argnames=('Q',))
def _step(Q, w, k):
    """Orthogonalise w = A q_k against the basis rows of Q (rows >= k are zero) with two Gram-Schmidt passes; returns
    (alpha_k = <q_k, A q_k>, beta_k = ||remainder||, Q with q_{k+1} = remainder / beta_k written in row k)."""
    c1 = jnp.dot(Q, w, precision=HIGHEST); w = w - jnp.dot(Q.T, c1, precision=HIGHEST)
    c2 = jnp.dot(Q, w, precision=HIGHEST); w = w - jnp.dot(Q.T, c2, precision=HIGHEST)
    alpha = c1[k - 1]
    beta = jnp.linalg.norm(w)
    Q = Q.at[k].set(jnp.where(beta > 0, w / beta, 0.0))
    return alpha, beta, Q


@jax.jit
def _combine(Q, coeffs):                                   # sum_j coeffs[j] Q[j] (coeffs zero-padded to Q's rows)
    return jnp.dot(coeffs, Q, precision=HIGHEST)


def _ritz(alphas, betas):
    """Eigen-decomposition of T_k in float64: theta descending, S columns matching."""
    theta, S = eigh_tridiagonal(np.asarray(alphas, dtype=np.float64), np.asarray(betas[:-1], dtype=np.float64))
    order = np.argsort(theta)[::-1]
    return theta[order], S[:, order]


def _iterate(hv, q1, max_steps):
    """Generator of (k, alphas, betas, Q) after each step; stops early when beta_k vanishes (invariant subspace)."""
    Q = _init(q1, rows=max_steps + 1)
    alphas, betas = [], []
    for k in range(1, max_steps + 1):
        w = hv(Q[k - 1])
        alpha, beta, Q = _step(Q, w, k)
        alphas.append(float(alpha)); betas.append(float(beta))
        yield k, alphas, betas, Q
        if betas[-1] <= 0: return


def direction(hv, q1, eps_star, max_steps, min_steps):
    """See module docstring. Returns (v, info) with v the unit top Ritz vector and info a dict (L = theta_1, theta_2, r1, r2,
    gap, ratio, steps, stop in {'eps', 'cap', 'invariant'})."""
    ok_prev = False; info = None
    for k, alphas, betas, Q in _iterate(hv, q1, max_steps):
        theta, S = _ritz(alphas, betas)
        beta_k = betas[-1]
        r1 = beta_k * abs(S[-1, 0]); r2 = beta_k * abs(S[-1, 1]) if k > 1 else math.inf
        gap = (theta[0] - theta[1] - r2) if k > 1 else -math.inf
        ratio = (r1 / gap) ** 2 if gap > 0 else math.inf
        ok = ratio <= eps_star
        info = dict(L=float(theta[0]), theta2=float(theta[1]) if k > 1 else None, r1=float(r1), r2=None if k == 1 else float(r2),
                    gap=None if k == 1 else float(gap), ratio=None if math.isinf(ratio) else float(ratio), steps=k, stop='cap')
        if beta_k <= 0: info['stop'] = 'invariant'; break
        if k >= min_steps and ok and ok_prev: info['stop'] = 'eps'; break
        ok_prev = ok
    coeffs = jnp.zeros(Q.shape[0], dtype=jnp.float32).at[:k].set(jnp.asarray(S[:, 0], dtype=jnp.float32))
    v = _combine(Q, coeffs); v = v / jnp.linalg.norm(v)
    return v, info


def upper_root(theta, betas, gamma):
    """The root x > max(theta) of prod_i (x - theta_i) = prod_i beta_i / gamma, solved in logarithms by bisection."""
    L = float(theta[0]); log_c = float(np.sum(np.log(betas))) - math.log(gamma)
    if not np.isfinite(log_c): return L                    # beta_k = 0: p_k(A) q_1 = 0, the bound collapses onto L_k
    f = lambda x: float(np.sum(np.log(x - theta))) - log_c
    step = max(abs(L), 1e-30); hi = L + step
    while f(hi) < 0: step *= 2; hi = L + step
    lo = L
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi: break
        if f(mid) < 0: lo = mid
        else: hi = mid
    return hi


def upper_bound(hv, q1, eps_star, max_steps, min_steps, gamma):
    """See module docstring. q1 must be a uniformly random unit vector of the operator's space; gamma = gamma_threshold(d, delta).
    Returns a dict (U, L, steps, stop in {'eps', 'cap', 'invariant'}, width = (U - L)/|L|)."""
    info = None
    for k, alphas, betas, Q in _iterate(hv, q1, max_steps):
        theta, S = _ritz(alphas, betas)
        U = upper_root(theta, betas, gamma); L = float(theta[0])
        width = (U - L) / abs(L) if L != 0 else math.inf
        info = dict(U=float(U), L=L, steps=k, stop='cap', width=None if math.isinf(width) else float(width))
        if betas[-1] <= 0: info['stop'] = 'invariant'; break
        if k >= min_steps and width <= eps_star: info['stop'] = 'eps'; break
    return info
