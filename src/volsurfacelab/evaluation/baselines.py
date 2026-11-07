"""Baseline models (Flat Black, Independent Raw SVI) and ablation comparison suite."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize_scalar

from volsurfacelab.arbitrage.report import run_grid_diagnostics
from volsurfacelab.config import GridConfig
from volsurfacelab.domain.models import CalibrationResult, OptionQuote
from volsurfacelab.pricing.black_forward import (
    black_forward_price,
    black_forward_price_from_total_variance,
)
from volsurfacelab.surface.raw_svi_baseline import fit_independent_raw_svi
from volsurfacelab.surface.ssvi import SSVISurface


class FlatBlackBaseline:
    """Baseline model: single constant implied volatility sigma_flat across all strikes and expiries."""

    def __init__(self, sigma_flat: float):
        self.sigma_flat = float(sigma_flat)

    def total_variance(self, k: float, T: float) -> float:
        return float(self.sigma_flat**2 * T)

    def evaluate_model_price(self, quote: OptionQuote) -> float:
        F = quote.forward if quote.forward is not None else 100.0
        D = quote.discount_factor if quote.discount_factor is not None else 1.0
        return float(black_forward_price(
            forward=F,
            strike=quote.strike,
            discount_factor=D,
            time_to_maturity=quote.time_to_maturity,
            volatility=self.sigma_flat,
            option_type=quote.option_type,
        ))


def fit_flat_black(
    quotes: List[OptionQuote],
    price_scale_floor: float = 0.05,
) -> FlatBlackBaseline:
    """Fit optimal single constant Black volatility to minimize spread-scaled price residuals."""
    mids = np.array([q.mid for q in quotes])
    scales = np.array([max(0.5 * q.spread, price_scale_floor) for q in quotes])

    def obj(sigma: float) -> float:
        prices = np.array([
            black_forward_price(
                forward=q.forward or 100.0,
                strike=q.strike,
                discount_factor=q.discount_factor or 1.0,
                time_to_maturity=q.time_to_maturity,
                volatility=sigma,
                option_type=q.option_type,
            )
            for q in quotes
        ])
        res = (prices - mids) / scales
        return float(np.sum(res**2))

    res = minimize_scalar(obj, bounds=(0.01, 3.0), method="bounded")
    return FlatBlackBaseline(sigma_flat=float(res.x))


def run_model_ablation(
    quotes: List[OptionQuote],
    ssvi_surface: SSVISurface,
    ssvi_result: CalibrationResult,
    grid_config: Optional[GridConfig] = None,
) -> Dict[str, Any]:
    """Execute unified ablation experiment comparing SSVI, Independent Raw SVI, and Flat Black."""
    g_cfg = grid_config or GridConfig()
    mids = np.array([q.mid for q in quotes])
    bids = np.array([q.bid for q in quotes])
    asks = np.array([q.ask for q in quotes])
    scales = np.array([max(0.5 * q.spread, 0.05) for q in quotes])

    # 1. SSVI metrics (from calibrated result)
    ssvi_grid = run_grid_diagnostics(ssvi_surface, g_cfg)
    expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))

    # 2. Fit Flat Black Baseline
    flat_black = fit_flat_black(quotes)
    flat_prices = np.array([flat_black.evaluate_model_price(q) for q in quotes])
    flat_price_diff = flat_prices - mids
    flat_price_rmse = float(np.sqrt(np.mean(flat_price_diff**2)))
    flat_price_mae = float(np.mean(np.abs(flat_price_diff)))
    flat_inside = float(100.0 * np.mean((flat_prices >= bids - 1e-4) & (flat_prices <= asks + 1e-4)))

    # 3. Fit Independent Raw SVI Baseline
    raw_svi = fit_independent_raw_svi(quotes)
    raw_svi_prices = []
    for q in quotes:
        F = q.forward or 100.0
        D = q.discount_factor or 1.0
        k = np.log(q.strike / F)
        w_slice = raw_svi.total_variance(k, q.time_to_maturity)
        p = black_forward_price_from_total_variance(F, q.strike, D, w_slice, q.option_type)
        raw_svi_prices.append(p)
    raw_svi_prices = np.array(raw_svi_prices)

    raw_price_diff = raw_svi_prices - mids
    raw_price_rmse = float(np.sqrt(np.mean(raw_price_diff**2)))
    raw_price_mae = float(np.mean(np.abs(raw_price_diff)))
    raw_inside = float(100.0 * np.mean((raw_svi_prices >= bids - 1e-4) & (raw_svi_prices <= asks + 1e-4)))

    # Calendar crossings check for Independent Raw SVI
    k_test = np.linspace(-1.0, 1.0, 41)
    calendar_crossings = 0
    min_w_diff = float("inf")
    for i in range(len(expiries) - 1):
        T1 = expiries[i]
        T2 = expiries[i + 1]
        w1 = raw_svi.evaluate_slice(k_test, T1)
        w2 = raw_svi.evaluate_slice(k_test, T2)
        diff = w2 - w1
        min_diff = float(np.min(diff))
        if min_diff < min_w_diff:
            min_w_diff = min_diff
        if min_diff < -1e-6:
            calendar_crossings += int(np.sum(diff < -1e-6))

    comparison_table = [
        {
            "model": "SSVI (Constrained)",
            "price_rmse": ssvi_result.price_rmse,
            "price_mae": ssvi_result.price_mae,
            "iv_rmse": ssvi_result.iv_rmse,
            "pct_inside_spread": ssvi_result.pct_inside_spread,
            "num_parameters": 3 + len(expiries),
            "calendar_guarantee": "Certified Analytical",
            "min_g": ssvi_grid.min_g,
            "min_w_T": ssvi_grid.min_w_T,
            "theorem_status": "CERTIFIED",
            "comment": "Guarantees no-arbitrage across all strikes & expiries.",
        },
        {
            "model": "Independent Raw SVI",
            "price_rmse": raw_price_rmse,
            "price_mae": raw_price_mae,
            "iv_rmse": None,
            "pct_inside_spread": raw_inside,
            "num_parameters": 5 * len(expiries),
            "calendar_guarantee": f"Min w2 - w1 = {min_w_diff:.4f} ({calendar_crossings} violations)",
            "min_g": "Slice dependent",
            "min_w_T": min_w_diff,
            "theorem_status": "NO_SURFACE_GUARANTEE",
            "comment": "Fits individual slices well but lacks cross-expiry calendar consistency.",
        },
        {
            "model": "Flat Black Volatility",
            "price_rmse": flat_price_rmse,
            "price_mae": flat_price_mae,
            "iv_rmse": None,
            "pct_inside_spread": flat_inside,
            "num_parameters": 1,
            "calendar_guarantee": "Trivial (w = sigma^2 * T)",
            "min_g": 1.0,
            "min_w_T": float(flat_black.sigma_flat**2),
            "theorem_status": "TRIVIAL_ZERO_SKEW",
            "comment": "Underfits smile/skew completely; useful as worst-case benchmark.",
        },
    ]

    return {
        "comparison_table": comparison_table,
        "quotes_count": len(quotes),
        "expiries_count": len(expiries),
        "flat_black_sigma": flat_black.sigma_flat,
    }
