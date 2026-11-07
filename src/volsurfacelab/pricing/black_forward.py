"""Black forward pricing model, bounds validation, and sensitivity analytics for European vanilla options."""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np
from scipy.stats import norm

from volsurfacelab.domain.exceptions import StatusCode, PricingBoundError, InvalidInputError


def black_forward_price_from_total_variance(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    discount_factor: Union[float, np.ndarray],
    total_variance: Union[float, np.ndarray],
    option_type: str = "call",
) -> Union[float, np.ndarray]:
    """Compute European vanilla option price from total implied variance w = sigma^2 * T.

    Parameters
    ----------
    forward : float or np.ndarray
        Forward price F > 0.
    strike : float or np.ndarray
        Strike price K > 0.
    discount_factor : float or np.ndarray
        Discount factor D > 0 (typically in (0, 1]).
    total_variance : float or np.ndarray
        Total variance w >= 0.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    price : float or np.ndarray
        Undiscounted or discounted Black forward price.
    """
    F = np.asarray(forward, dtype=np.float64)
    K = np.asarray(strike, dtype=np.float64)
    D = np.asarray(discount_factor, dtype=np.float64)
    w = np.asarray(total_variance, dtype=np.float64)

    if np.any(F <= 0.0) or np.any(K <= 0.0) or np.any(D <= 0.0):
        raise InvalidInputError("Forward, strike, and discount factor must be strictly positive.")
    if np.any(w < 0.0):
        raise InvalidInputError("Total variance cannot be negative.")

    opt_type = option_type.lower()
    if opt_type not in ("call", "put"):
        raise InvalidInputError(f"Unsupported option type '{option_type}', must be 'call' or 'put'.")

    # Handle w near zero or exactly zero
    # When w -> 0, price is intrinsic discounted value
    is_scalar = (F.ndim == 0 and K.ndim == 0 and D.ndim == 0 and w.ndim == 0)
    
    # Broadcast arrays
    F, K, D, w = np.broadcast_arrays(F, K, D, w)
    price = np.zeros_like(F, dtype=np.float64)

    zero_w_mask = (w <= 1e-14)
    pos_w_mask = ~zero_w_mask

    if np.any(zero_w_mask):
        if opt_type == "call":
            price[zero_w_mask] = D[zero_w_mask] * np.maximum(F[zero_w_mask] - K[zero_w_mask], 0.0)
        else:
            price[zero_w_mask] = D[zero_w_mask] * np.maximum(K[zero_w_mask] - F[zero_w_mask], 0.0)

    if np.any(pos_w_mask):
        F_p = F[pos_w_mask]
        K_p = K[pos_w_mask]
        D_p = D[pos_w_mask]
        w_p = w[pos_w_mask]
        sqrt_w = np.sqrt(w_p)

        log_moneyness = np.log(F_p / K_p)
        d1 = (log_moneyness + 0.5 * w_p) / sqrt_w
        d2 = d1 - sqrt_w

        if opt_type == "call":
            price[pos_w_mask] = D_p * (F_p * norm.cdf(d1) - K_p * norm.cdf(d2))
        else:
            price[pos_w_mask] = D_p * (K_p * norm.cdf(-d2) - F_p * norm.cdf(-d1))

    # Ensure non-negative due to float rounding
    price = np.maximum(price, 0.0)

    return float(price) if is_scalar else price


def black_forward_price(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    discount_factor: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: str = "call",
) -> Union[float, np.ndarray]:
    """Compute European vanilla option price from annualized volatility sigma and maturity T."""
    T = np.asarray(time_to_maturity, dtype=np.float64)
    sigma = np.asarray(volatility, dtype=np.float64)
    if np.any(T <= 0.0):
        raise InvalidInputError("Time to maturity must be strictly positive.")
    if np.any(sigma < 0.0):
        raise InvalidInputError("Volatility cannot be negative.")
    total_variance = sigma**2 * T
    return black_forward_price_from_total_variance(
        forward=forward,
        strike=strike,
        discount_factor=discount_factor,
        total_variance=total_variance,
        option_type=option_type,
    )


def black_forward_vega(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    discount_factor: Union[float, np.ndarray],
    time_to_maturity: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute Black forward vega dPrice/dSigma."""
    F = np.asarray(forward, dtype=np.float64)
    K = np.asarray(strike, dtype=np.float64)
    D = np.asarray(discount_factor, dtype=np.float64)
    T = np.asarray(time_to_maturity, dtype=np.float64)
    sigma = np.asarray(volatility, dtype=np.float64)

    is_scalar = (F.ndim == 0 and K.ndim == 0 and D.ndim == 0 and T.ndim == 0 and sigma.ndim == 0)
    F, K, D, T, sigma = np.broadcast_arrays(F, K, D, T, sigma)

    vega = np.zeros_like(F, dtype=np.float64)
    w = sigma**2 * T
    pos_mask = (w > 1e-14) & (T > 0.0)

    if np.any(pos_mask):
        sqrt_w = np.sqrt(w[pos_mask])
        d1 = (np.log(F[pos_mask] / K[pos_mask]) + 0.5 * w[pos_mask]) / sqrt_w
        vega[pos_mask] = D[pos_mask] * F[pos_mask] * np.sqrt(T[pos_mask]) * norm.pdf(d1)

    return float(vega) if is_scalar else vega


def check_european_bounds(
    price: float,
    forward: float,
    strike: float,
    discount_factor: float,
    option_type: str = "call",
    tolerance: float = 1e-7,
) -> Tuple[bool, StatusCode, float, float]:
    """Validate whether an observed option price satisfies European no-arbitrage bounds.

    Returns
    -------
    is_valid : bool
    status_code : StatusCode
    lower_bound : float
    upper_bound : float
    """
    if forward <= 0.0 or strike <= 0.0 or discount_factor <= 0.0:
        return False, StatusCode.REJECTED_NON_POSITIVE_FORWARD, 0.0, 0.0

    opt_type = option_type.lower()
    if opt_type == "call":
        lower_bound = float(discount_factor * max(forward - strike, 0.0))
        upper_bound = float(discount_factor * forward)
    elif opt_type == "put":
        lower_bound = float(discount_factor * max(strike - forward, 0.0))
        upper_bound = float(discount_factor * strike)
    else:
        return False, StatusCode.REJECTED_INVALID_OPTION_TYPE, 0.0, 0.0

    if price < lower_bound - tolerance:
        return False, StatusCode.INVERSION_PRICE_BELOW_INTRINSIC, lower_bound, upper_bound
    if price > upper_bound + tolerance:
        return False, StatusCode.INVERSION_PRICE_ABOVE_UPPER_BOUND, lower_bound, upper_bound

    return True, StatusCode.ACCEPTED, lower_bound, upper_bound
