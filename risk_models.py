import numpy as np

def shrink_covariance(observations):
    """Oracle Approximating Shrinkage toward a scaled identity (daily units).

    Fit risky assets only: deterministic cash must retain exactly zero risk.
    OAS assumes approximately Gaussian, independent observations; it does not
    remove regime risk or uncertainty in historical return estimates.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Covariance requires finite observations and at least two rows")
    x = x - x.mean(axis=0)
    n, p = x.shape
    if p == 0:
        raise ValueError("No risky assets supplied")
    empirical = x.T @ x / n
    mean_variance = np.trace(empirical) / p
    alpha = np.mean(empirical ** 2)
    denominator = (n + 1) * (alpha - mean_variance ** 2 / p)
    shrinkage = 1.0 if denominator <= 0 else min(
        (alpha + mean_variance ** 2) / denominator, 1.0
    )
    estimate = (1 - shrinkage) * empirical
    estimate += shrinkage * mean_variance * np.eye(p)
    return estimate, shrinkage

