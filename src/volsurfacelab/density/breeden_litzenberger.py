"""Breeden-Litzenberger risk-neutral density extraction and moment analytics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm

from volsurfacelab.config import DensityConfig
from volsurfacelab.domain.exceptions import InvalidInputError
from volsurfacelab.domain.models import DensityResult
from volsurfacelab.surface.ssvi import SSVISurface


def risk_neutral_density_analytic(
    surface: SSVISurface,
    strike: Union[float, np.ndarray],
    forward: float,
    maturity: float,
) -> Union[float, np.ndarray]:
    """Evaluate analytic Breeden-Litzenberger risk-neutral density q_T(K).

    q_T(K) = (n(d2) / (K * sqrt(w))) * g(k, T)
    where k = ln(K / F), d2 = (-k - 0.5 * w) / sqrt(w), and g is Durrleman function.
    """
    K = np.asarray(strike, dtype=np.float64)
    is_scalar = (K.ndim == 0)

    if np.any(K <= 0.0) or forward <= 0.0 or maturity <= 0.0:
        raise InvalidInputError("Strike, forward, and maturity must be strictly positive.")

    k = np.log(K / forward)
    theta = surface.theta_curve.evaluate(maturity)
    w = surface.total_variance_from_theta(k, theta)
    sqrt_w = np.sqrt(w)

    d2 = (-k - 0.5 * w) / sqrt_w
    g = surface.durrleman_g(k, maturity)

    # Analytic density formula
    density = (norm.pdf(d2) / (K * sqrt_w)) * g

    return float(density) if is_scalar else density


def risk_neutral_cdf_analytic(
    surface: SSVISurface,
    strike: Union[float, np.ndarray],
    forward: float,
    maturity: float,
) -> Union[float, np.ndarray]:
    """Evaluate exact closed-form risk-neutral CDF Q_T(K) = P(S_T <= K).

    Q_T(K) = 1 + d(C_F)/dK = N(-d2) + (n(d2) / (2 * sqrt(w))) * w_k
    where k = ln(K / F), d2 = (-k - 0.5 * w) / sqrt(w), and w_k is analytic strike derivative.
    """
    K = np.asarray(strike, dtype=np.float64)
    is_scalar = (K.ndim == 0)

    if np.any(K <= 0.0) or forward <= 0.0 or maturity <= 0.0:
        raise InvalidInputError("Strike, forward, and maturity must be strictly positive.")

    k = np.log(K / forward)
    theta = surface.theta_curve.evaluate(maturity)
    w = surface.total_variance_from_theta(k, theta)
    w_k, _ = surface.derivatives_k(k, theta)
    sqrt_w = np.sqrt(w)

    d2 = (-k - 0.5 * w) / sqrt_w

    cdf = norm.cdf(-d2) + (norm.pdf(d2) / (2.0 * sqrt_w)) * w_k
    cdf = np.clip(cdf, 0.0, 1.0)

    return float(cdf) if is_scalar else cdf


def extract_risk_neutral_density(
    surface: SSVISurface,
    maturity: float,
    forward: float,
    discount_factor: float,
    config: Optional[DensityConfig] = None,
) -> DensityResult:
    """Extract, integrate, and validate risk-neutral density for a given expiry.

    Parameters
    ----------
    surface : SSVISurface
        SSVI volatility surface model.
    maturity : float
        Maturity T in years.
    forward : float
        Forward price F_T > 0.
    discount_factor : float
        Discount factor D_T > 0.
    config : Optional[DensityConfig]
        Strikes and quadrature settings.

    Returns
    -------
    result : DensityResult
    """
    cfg = config or DensityConfig()

    theta = surface.theta_curve.evaluate(maturity)
    atm_vol = np.sqrt(theta / maturity)
    # Domain support based on standard deviation range: F * exp(+- std_dev_range * sigma * sqrt(T))
    vol_sqrt_t = max(atm_vol * np.sqrt(maturity), 0.05)
    k_min = -cfg.std_dev_range * vol_sqrt_t
    k_max = cfg.std_dev_range * vol_sqrt_t

    strikes = forward * np.exp(np.linspace(k_min, k_max, cfg.n_strikes))
    log_moneyness = np.log(strikes / forward)

    q_density = risk_neutral_density_analytic(surface, strikes, forward, maturity)
    state_price_density = discount_factor * q_density

    # Numerical integration via Simpson's rule over sampled support
    total_mass = float(simpson(y=q_density, x=strikes))
    first_moment = float(simpson(y=strikes * q_density, x=strikes))
    rel_moment_err = float(abs(first_moment - forward) / forward)

    # Numerical higher moments of terminal price S_T
    var_rn = float(simpson(y=((strikes - first_moment)**2) * q_density, x=strikes))
    std_rn = float(np.sqrt(max(var_rn, 0.0)))
    if std_rn > 1e-12:
        skew_rn = float(simpson(y=((strikes - first_moment)**3) * q_density, x=strikes) / (std_rn**3))
        kurt_rn = float(simpson(y=((strikes - first_moment)**4) * q_density, x=strikes) / (std_rn**4) - 3.0)
    else:
        skew_rn = 0.0
        kurt_rn = 0.0

    # Higher moments of log-moneyness k = ln(S_T / F)
    # The density of k is p(k) = q(K) * K
    p_k = q_density * strikes
    mass_k = float(simpson(y=p_k, x=log_moneyness))
    p_k_norm = p_k / mass_k if mass_k > 1e-8 else p_k
    log_mean_val = float(simpson(y=log_moneyness * p_k_norm, x=log_moneyness))
    log_var_val = float(simpson(y=((log_moneyness - log_mean_val)**2) * p_k_norm, x=log_moneyness))
    log_std_val = float(np.sqrt(max(log_var_val, 0.0)))
    if log_std_val > 1e-12:
        log_skew_val = float(simpson(y=((log_moneyness - log_mean_val)**3) * p_k_norm, x=log_moneyness) / (log_std_val**3))
        log_kurt_val = float(simpson(y=((log_moneyness - log_mean_val)**4) * p_k_norm, x=log_moneyness) / (log_std_val**4) - 3.0)
    else:
        log_skew_val = 0.0
        log_kurt_val = 0.0

    # Analytic CDF on strike support
    analytic_cdf_vals = risk_neutral_cdf_analytic(surface, strikes, forward, maturity)
    num_cdf = np.zeros_like(strikes)
    dx = np.diff(strikes)
    # Align numerical trapezoidal accumulation with exact analytic left-tail probability Q(K_0)
    c0 = float(analytic_cdf_vals[0]) if isinstance(analytic_cdf_vals, np.ndarray) else float(analytic_cdf_vals)
    num_cdf[0] = c0
    num_cdf[1:] = c0 + np.cumsum(0.5 * (q_density[:-1] + q_density[1:]) * dx)
    max_cdf_err = float(np.max(np.abs(analytic_cdf_vals - num_cdf)))

    min_q = float(np.min(q_density))
    neg_count = int(np.sum(q_density < -1e-8))
    is_non_neg = bool(min_q >= -1e-6)

    warnings: List[str] = []
    if total_mass < 0.95 or total_mass > 1.05:
        warnings.append(
            f"Integrated mass {total_mass:.4f} departs from 1.0 due to tail truncation on support [{strikes[0]:.2f}, {strikes[-1]:.2f}]."
        )
    if rel_moment_err > 0.05:
        warnings.append(
            f"Relative first-moment error {rel_moment_err:.4f} indicates non-negligible tail mass outside finite grid."
        )
    if not is_non_neg:
        warnings.append(f"Risk-neutral density has {neg_count} negative values (min: {min_q:.6f}).")

    return DensityResult(
        expiry=maturity,
        forward=forward,
        discount_factor=discount_factor,
        strikes=strikes.tolist(),
        log_moneyness=log_moneyness.tolist(),
        density_q=q_density.tolist(),
        state_price_density=state_price_density.tolist(),
        total_mass=total_mass,
        first_moment=first_moment,
        relative_moment_error=rel_moment_err,
        is_non_negative=is_non_neg,
        min_density=min_q,
        negative_count=neg_count,
        support_min=float(strikes[0]),
        support_max=float(strikes[-1]),
        variance=var_rn,
        std_dev=std_rn,
        skewness=skew_rn,
        kurtosis=kurt_rn,
        log_mean=log_mean_val,
        log_variance=log_var_val,
        log_skewness=log_skew_val,
        log_kurtosis=log_kurt_val,
        analytic_cdf=analytic_cdf_vals.tolist() if isinstance(analytic_cdf_vals, np.ndarray) else [analytic_cdf_vals],
        max_cdf_error=max_cdf_err,
        warnings=warnings,
    )
