"""Independent Raw SVI slice baseline (for ablation and comparative research)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import least_squares

from volsurfacelab.domain.models import OptionQuote
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance


@dataclass
class RawSVISliceParams:
    """Raw SVI 5-parameter slice: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))."""
    expiry: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        k_arr = np.asarray(k, dtype=np.float64)
        km = k_arr - self.m
        disc = np.sqrt(km**2 + self.sigma**2)
        w = self.a + self.b * (self.rho * km + disc)
        return float(w) if k_arr.ndim == 0 else w

    def derivatives_k(self, k: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        k_arr = np.asarray(k, dtype=np.float64)
        km = k_arr - self.m
        disc = np.sqrt(km**2 + self.sigma**2)
        # w_k = b * (rho + (k - m) / disc)
        w_k = self.b * (self.rho + km / disc)
        # w_kk = b * sigma^2 / disc^3
        w_kk = self.b * (self.sigma**2) / (disc**3)
        if k_arr.ndim == 0:
            return float(w_k), float(w_kk)
        return w_k, w_kk

    def durrleman_g(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        w = self.total_variance(k)
        w_k, w_kk = self.derivatives_k(k)
        term1 = (1.0 - (k * w_k) / (2.0 * w))**2
        term2 = (w_k**2 / 4.0) * (1.0 / w + 0.25)
        term3 = 0.5 * w_kk
        return term1 - term2 + term3


class IndependentRawSVIBaseline:
    """Baseline surface composed of independently fitted Raw SVI slices.

    WARNING: Independent slices do not guarantee calendar no-arbitrage across expiries.
    Used exclusively as a baseline to demonstrate the advantages of SSVI joint calibration.
    """

    def __init__(self, slices: Dict[float, RawSVISliceParams]):
        self.slices = slices
        self.expiries = sorted(list(slices.keys()))

    def total_variance(self, k: float, T: float) -> float:
        # Nearest slice or simple linear interpolation between slices
        if T in self.slices:
            return float(self.slices[T].total_variance(k))
        # Find nearest slice
        diffs = [abs(t - T) for t in self.expiries]
        idx = int(np.argmin(diffs))
        return float(self.slices[self.expiries[idx]].total_variance(k))

    def evaluate_slice(self, k_arr: np.ndarray, T: float) -> np.ndarray:
        if T in self.slices:
            return np.asarray(self.slices[T].total_variance(k_arr))
        diffs = [abs(t - T) for t in self.expiries]
        idx = int(np.argmin(diffs))
        return np.asarray(self.slices[self.expiries[idx]].total_variance(k_arr))


def fit_raw_svi_slice(
    quotes_slice: List[OptionQuote],
    expiry: float,
    price_scale_floor: float = 0.05,
) -> RawSVISliceParams:
    """Fit a single Raw SVI slice independently using bounded least squares."""
    n = len(quotes_slice)
    F = quotes_slice[0].forward if quotes_slice[0].forward is not None else 100.0
    D = quotes_slice[0].discount_factor if quotes_slice[0].discount_factor is not None else 1.0

    k_arr = np.array([np.log(q.strike / F) for q in quotes_slice], dtype=np.float64)
    mid_arr = np.array([q.mid for q in quotes_slice], dtype=np.float64)
    spread_scale = np.array([max(0.5 * q.spread, price_scale_floor) for q in quotes_slice], dtype=np.float64)
    call_mask = np.array([q.option_type == "call" for q in quotes_slice])
    put_mask = ~call_mask
    strikes = np.array([q.strike for q in quotes_slice], dtype=np.float64)

    # Param vector p = [a, b, rho, m, sigma]
    # Bounds: a in (-1.0, 1.0), b in (0.001, 2.0), rho in (-0.99, 0.99), m in (-1.0, 1.0), sigma in (0.01, 2.0)
    lower_bounds = [-1.0, 0.001, -0.99, -1.0, 0.01]
    upper_bounds = [1.0, 2.0, 0.99, 1.0, 2.0]
    p0 = [0.04 * expiry, 0.1, -0.3, 0.0, 0.1]

    def residual_fn(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = p
        km = k_arr - m
        disc = np.sqrt(km**2 + sigma**2)
        w = a + b * (rho * km + disc)
        w = np.maximum(w, 1e-12)

        model_prices = np.zeros(n, dtype=np.float64)
        if np.any(call_mask):
            model_prices[call_mask] = black_forward_price_from_total_variance(
                F, strikes[call_mask], D, w[call_mask], "call"
            )
        if np.any(put_mask):
            model_prices[put_mask] = black_forward_price_from_total_variance(
                F, strikes[put_mask], D, w[put_mask], "put"
            )

        return (model_prices - mid_arr) / spread_scale

    opt_res = least_squares(
        residual_fn,
        x0=p0,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        max_nfev=1000,
    )

    a_opt, b_opt, rho_opt, m_opt, sigma_opt = opt_res.x
    return RawSVISliceParams(
        expiry=expiry,
        a=float(a_opt),
        b=float(b_opt),
        rho=float(rho_opt),
        m=float(m_opt),
        sigma=float(sigma_opt),
    )


def fit_independent_raw_svi(
    quotes: List[OptionQuote],
    price_scale_floor: float = 0.05,
) -> IndependentRawSVIBaseline:
    """Fit independent Raw SVI slices for each expiry."""
    slices: Dict[float, RawSVISliceParams] = {}
    expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))

    for T in expiries:
        q_slice = [q for q in quotes if abs(q.time_to_maturity - T) < 1e-5]
        if len(q_slice) >= 4:
            slice_params = fit_raw_svi_slice(q_slice, T, price_scale_floor)
            slices[T] = slice_params

    return IndependentRawSVIBaseline(slices)
