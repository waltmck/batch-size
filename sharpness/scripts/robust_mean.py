"""The 1-d mean estimator of Lee and Valiant as stated in [LMSV25] (Estimator 1), with its confidence interval.

[LMSV25] Lee, McKelvie, Song, Valiant, "All-Purpose Mean Estimation over R: Optimal Sub-Gaussianity with Outlier
Robustness and Low Moments Performance", ICML 2025. Inputs: n i.i.d. samples x_i and a confidence parameter delta.
  1. kappa: median-of-means, the data evenly partitioned (in the given order) into ceil(log(1/delta)) groups;
  2. alpha: the solution of the monotone, piecewise-linear equation  sum_i min(alpha (x_i - kappa)^2, 1) = log(1/delta) / 3;
  3. mu_hat = kappa + (1/n) sum_i (x_i - kappa) (1 - min(alpha (x_i - kappa)^2, 1)).
Guarantee (Fact 1.1): with probability at least 1 - delta, |mu_hat - mu| <= sigma (sqrt(2) + o(1)) sqrt(log(1/delta) / n).
The interval reported here is mu_hat +- sigma_hat sqrt(2 log(1/delta) / n) with sigma_hat the sample standard deviation
standing in for sigma.
"""
import math
import numpy as np


def solve_alpha(d2, target):
    """alpha with sum_i min(alpha d2_i, 1) = target (d2_i >= 0). The left side is continuous, nondecreasing and piecewise
    linear in alpha with breakpoints at 1/d2_i: with the m largest d2 clipped it equals alpha * (sum of the n-m smallest) + m,
    so the solution is found exactly by scanning m. Returns inf when even clipping every nonzero term cannot reach target."""
    s = np.sort(np.asarray(d2, dtype=np.float64)); n = s.size; cs = np.cumsum(s)
    if target >= np.count_nonzero(s): return math.inf
    for m in range(n):
        j = n - m                                   # number of unclipped terms (the j smallest)
        S = cs[j - 1]
        if S <= 0: continue
        alpha = (target - m) / S
        if alpha < 0: break
        if alpha * s[j - 1] <= 1 + 1e-12 and (m == 0 or alpha * s[j] >= 1 - 1e-12): return float(alpha)
    raise ArithmeticError('no solution found for alpha')  # unreachable for finite target < count_nonzero(d2)


def estimate(x, delta):
    """Estimator 1 of [LMSV25] on the samples x at confidence delta. Returns a dict with mean, half_width, lo, hi, sigma
    (sample standard deviation), n, delta, groups, kappa, alpha and n_clipped (samples whose weight was zeroed)."""
    x = np.asarray(x, dtype=np.float64).ravel(); n = x.size
    L = math.log(1.0 / delta); G = max(1, math.ceil(L))
    assert n >= G, f'need at least ceil(log(1/delta)) = {G} samples for the median-of-means groups, got {n}'
    kappa = float(np.median([g.mean() for g in np.array_split(x, G)]))
    d2 = (x - kappa) ** 2
    alpha = solve_alpha(d2, L / 3.0)
    clipped = (d2 > 0) if math.isinf(alpha) else (alpha * d2 >= 1.0)
    w = np.where(clipped, 0.0, 1.0 - (0.0 if math.isinf(alpha) else alpha) * d2)
    mu = kappa + float(np.mean((x - kappa) * w))
    sigma = float(x.std(ddof=1)) if n > 1 else 0.0
    half = sigma * math.sqrt(2.0 * L / n)
    return dict(mean=mu, half_width=half, lo=mu - half, hi=mu + half, sigma=sigma, n=int(n), delta=delta, groups=G,
                kappa=kappa, alpha=alpha, n_clipped=int(clipped.sum()))
