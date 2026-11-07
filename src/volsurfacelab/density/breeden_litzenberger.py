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
        warnings=warnings,
    )
