"""Robust bracketed implied volatility inversion for European vanilla options."""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Union
import numpy as np
from scipy.optimize import brentq

from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.pricing.black_forward import (
    black_forward_price,
    black_forward_vega,
    check_european_bounds,
)


class InversionResult(NamedTuple):
    """Structured result of an implied volatility inversion."""
    success: bool
    iv: Optional[float]
    total_variance: Optional[float]
    iterations: int
    residual: float
    status_code: StatusCode
    reason: str


def implied_volatility_brent(
    price: float,
    forward: float,
    strike: float,
    discount_factor: float,
    time_to_maturity: float,
    option_type: str = "call",
    vol_min: float = 1e-4,
    vol_max: float = 5.0,
    vol_max_expanded: float = 25.0,
    tolerance: float = 1e-8,
    max_iter: int = 100,
) -> InversionResult:
    """Invert European option implied volatility using bracketed Brent root solving.

    Parameters
    ----------
    price : float
        Observed European option price in currency units.
    forward : float
        Forward price F > 0.
    strike : float
        Strike price K > 0.
    discount_factor : float
        Discount factor D > 0.
    time_to_maturity : float
        Time to maturity T > 0 in years.
    option_type : str
        'call' or 'put'.
    vol_min : float
        Initial lower bound of volatility bracket.
    vol_max : float
        Initial upper bound of volatility bracket.
    vol_max_expanded : float
        Maximum allowable expanded volatility bracket.
    tolerance : float
        Target absolute price tolerance.
    max_iter : int
        Maximum iterations for Brent root finding.

    Returns
    -------
    result : InversionResult
    """
    # 1. Finite and positive checks
    if not (np.isfinite(price) and np.isfinite(forward) and np.isfinite(strike)
            and np.isfinite(discount_factor) and np.isfinite(time_to_maturity)):
        return InversionResult(
            success=False,
            iv=None,
            total_variance=None,
            iterations=0,
            residual=float("nan"),
            status_code=StatusCode.INVERSION_NON_FINITE_INPUT,
            reason="Input values contain NaN or Inf.",
        )

    if forward <= 0.0 or strike <= 0.0 or discount_factor <= 0.0 or time_to_maturity <= 0.0:
        return InversionResult(
            success=False,
            iv=None,
            total_variance=None,
            iterations=0,
            residual=float("nan"),
            status_code=StatusCode.REJECTED_NON_POSITIVE_FORWARD,
            reason="Forward, strike, discount factor, and maturity must be strictly positive.",
        )

    # 2. Check European no-arbitrage price bounds
    valid_bounds, status, lower_bound, upper_bound = check_european_bounds(
        price=price,
        forward=forward,
        strike=strike,
        discount_factor=discount_factor,
        option_type=option_type,
        tolerance=tolerance,
    )
    if not valid_bounds:
        return InversionResult(
            success=False,
            iv=None,
            total_variance=None,
            iterations=0,
            residual=float(price - lower_bound if status == StatusCode.INVERSION_PRICE_BELOW_INTRINSIC else price - upper_bound),
            status_code=status,
            reason=f"Price {price:.6f} outside bounds [{lower_bound:.6f}, {upper_bound:.6f}].",
        )

    # 3. Check near-intrinsic / near-zero vega boundary
    # If price is at lower bound within tolerance, sigma is effectively 0
    if abs(price - lower_bound) <= 1e-10:
        return InversionResult(
            success=True,
            iv=0.0,
            total_variance=0.0,
            iterations=0,
            residual=0.0,
            status_code=StatusCode.INVERSION_SUCCESS,
            reason="Price matches intrinsic value within numerical tolerance.",
        )

    # 4. Objective function
    def objective(sigma: float) -> float:
        model_p = black_forward_price(
            forward=forward,
            strike=strike,
            discount_factor=discount_factor,
            time_to_maturity=time_to_maturity,
            volatility=sigma,
            option_type=option_type,
        )
        return float(model_p - price)

    f_low = objective(vol_min)
    # If price is smaller than price at vol_min, but above lower bound:
    if f_low > 0.0:
        # Since price > lower_bound, the root lies strictly in [0, vol_min]
        f_zero = float(lower_bound - price)
        if abs(f_zero) <= tolerance:
            return InversionResult(
                success=True,
                iv=0.0,
                total_variance=0.0,
                iterations=0,
                residual=float(f_zero),
                status_code=StatusCode.INVERSION_SUCCESS,
                reason="Price matches lower bound within numerical tolerance.",
            )
        vol_min = 0.0
        f_low = f_zero

    f_high = objective(vol_max)
    # Expand bracket if market price exceeds price at vol_max
    if f_high < 0.0:
        current_max = vol_max
        while f_high < 0.0 and current_max < vol_max_expanded:
            current_max = min(current_max * 2.0, vol_max_expanded)
            f_high = objective(current_max)
        vol_max = current_max

    # 5. Check if bracketed
    if f_low * f_high > 0.0:
        return InversionResult(
            success=False,
            iv=None,
            total_variance=None,
            iterations=0,
            residual=float(f_low if abs(f_low) < abs(f_high) else f_high),
            status_code=StatusCode.INVERSION_BRACKET_FAILED,
            reason=f"Root cannot be bracketed on [{vol_min}, {vol_max}]: f_low={f_low:.4e}, f_high={f_high:.4e}.",
        )

    # 6. Brent root solve
    try:
        sol, root_results = brentq(
            objective,
            vol_min,
            vol_max,
            xtol=tolerance,
            rtol=tolerance,
            maxiter=max_iter,
            full_output=True,
        )
        final_residual = objective(sol)
        total_var = sol**2 * time_to_maturity
        return InversionResult(
            success=True,
            iv=float(sol),
            total_variance=float(total_var),
            iterations=int(root_results.iterations),
            residual=float(final_residual),
            status_code=StatusCode.INVERSION_SUCCESS,
            reason="Convergence achieved.",
        )
    except Exception as exc:
        return InversionResult(
            success=False,
            iv=None,
            total_variance=None,
            iterations=max_iter,
            residual=float("nan"),
            status_code=StatusCode.INVERSION_MAX_ITERATIONS,
            reason=f"Brent solver failed: {str(exc)}",
        )


def invert_implied_volatilities(
    prices: Sequence[float],
    forwards: Sequence[float],
    strikes: Sequence[float],
    discount_factors: Sequence[float],
    maturities: Sequence[float],
    option_types: Sequence[str],
) -> List[InversionResult]:
    """Invert multiple option quotes sequentially, returning structured results per contract."""
    results = []
    n = len(prices)
    for i in range(n):
        res = implied_volatility_brent(
            price=prices[i],
            forward=forwards[i],
            strike=strikes[i],
            discount_factor=discount_factors[i],
            time_to_maturity=maturities[i],
            option_type=option_types[i],
        )
        results.append(res)
    return results
