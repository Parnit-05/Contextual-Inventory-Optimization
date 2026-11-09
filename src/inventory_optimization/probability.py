"""Discrete Poisson/NB2 probability, interval, and proper scoring utilities."""
import math
import numpy as np

def _log_pmf(y, mu, alpha):
    y = int(y); mu = max(float(mu), 1e-12)
    if alpha <= 1e-12:
        return -mu + y * math.log(mu) - math.lgamma(y + 1)
    size = 1.0 / alpha
    return (math.lgamma(y + size) - math.lgamma(size) - math.lgamma(y + 1)
            + size * math.log(size / (size + mu)) + y * math.log(mu / (size + mu)))

def _cdf_to(mu, alpha, max_y):
    mu = max(float(mu), 1e-12)
    if alpha <= 1e-12:
        p = math.exp(-mu); ratio_base = mu
        probs = [p]
        for k in range(1, max_y + 1):
            p *= ratio_base / k; probs.append(p)
    else:
        size = 1.0 / alpha; p = math.exp(size * math.log(size / (size + mu)))
        probs = [p]
        for k in range(1, max_y + 1):
            p *= ((k - 1 + size) / k) * (mu / (size + mu)); probs.append(p)
    return np.minimum(np.cumsum(probs), 1.0)

def interval(mu, alpha, level=0.90):
    tail = (1.0 - level) / 2
    cdf = _cdf_to(mu, alpha, max(100, int(mu + 12 * math.sqrt(mu + alpha * mu * mu + 1) + 50)))
    low = int(np.searchsorted(cdf, tail)); high = int(np.searchsorted(cdf, 1-tail))
    return low, high

def evaluate_distribution(y, means, alpha, level=0.90):
    nll, cover, widths, crps = [], [], [], []
    for obs, mu in zip(np.asarray(y, dtype=int), np.asarray(means, dtype=float)):
        nll.append(-_log_pmf(obs, mu, alpha))
        lo, hi = interval(mu, alpha, level); cover.append(lo <= obs <= hi); widths.append(hi-lo)
        cap = max(obs, hi) + int(8 * math.sqrt(mu + alpha * mu * mu + 1) + 20)
        cdf = _cdf_to(mu, alpha, cap)
        target = (np.arange(cap + 1) >= obs).astype(float)
        crps.append(float(np.sum((cdf - target) ** 2)))
    return {"negative_log_likelihood": float(np.mean(nll)), "interval_coverage": float(np.mean(cover)),
            "interval_width": float(np.mean(widths)), "crps": float(np.mean(crps)), "interval_level": level}

def sample_demand(rng, means, alpha):
    means = np.asarray(means, dtype=float)
    rates = means if alpha <= 1e-12 else rng.gamma(shape=1.0/alpha, scale=alpha*means)
    return rng.poisson(np.maximum(rates, 0)).astype(np.int64)
