"""Diebold-Mariano test for comparing forecast accuracy.

Tests whether one model's per-time_id loss series is significantly lower
than another's (e.g. best model vs. naive persistence baseline) - not just
whether its point-estimate RMSPE happens to be smaller. statsmodels doesn't
ship this test directly, so it's implemented here from the standard
loss-differential formula (Diebold & Mariano, 1995).
"""

import math


def diebold_mariano(loss_a, loss_b, h=1):
    """Two-sided DM test on paired loss series loss_a, loss_b (equal length).

    loss_a/loss_b are per-observation loss values (e.g. squared percentage
    error) for two competing forecasts, aligned on the same observations.
    Negative dm_stat means loss_a is lower (model a is more accurate).

    h is the forecast horizon in steps, used for the Newey-West-style
    variance correction (lag = h - 1 autocovariance terms). h=1 (the
    default) reduces to the plain sample-variance case, appropriate here
    since each time_id is an independent one-step-ahead forecast.

    Returns (dm_stat, p_value).
    """
    if len(loss_a) != len(loss_b):
        raise ValueError("loss_a and loss_b must be the same length")

    n = len(loss_a)
    if n < 2:
        raise ValueError("need at least 2 paired observations")

    d = [a - b for a, b in zip(loss_a, loss_b)]
    d_mean = sum(d) / n

    # long-run variance of the loss differential: sample variance plus
    # 2 * sum of the first (h-1) autocovariances (Newey-West style).
    gamma_0 = sum((x - d_mean) ** 2 for x in d) / n
    var_d = gamma_0
    for lag in range(1, h):
        gamma_lag = sum(
            (d[t] - d_mean) * (d[t - lag] - d_mean) for t in range(lag, n)
        ) / n
        var_d += 2 * gamma_lag

    if var_d <= 0:
        raise ValueError("non-positive variance estimate; series may be degenerate")

    dm_stat = d_mean / math.sqrt(var_d / n)

    # two-sided p-value from the standard normal distribution
    p_value = 2 * (1 - _normal_cdf(abs(dm_stat)))

    return dm_stat, p_value


def _normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
