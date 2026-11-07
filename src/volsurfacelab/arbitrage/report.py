"""Combined no-arbitrage diagnostics and multi-layer verification report."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from volsurfacelab.arbitrage.butterfly import check_butterfly_arbitrage_grid
from volsurfacelab.arbitrage.calendar import check_calendar_arbitrage_grid
from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.config import GridConfig
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import (
    AdmissibilityReport,
    GridDiagnostics,
    OptionQuote,
    QuoteCleaningReport,
    SSVIParameters,
)
from volsurfacelab.pricing.parity import check_put_call_parity_intervals
from volsurfacelab.surface.ssvi import SSVISurface


def run_grid_diagnostics(
    surface: SSVISurface,
    config: Optional[GridConfig] = None,
) -> GridDiagnostics:
    """Execute dense numerical grid diagnostics for butterfly and calendar arbitrage."""
    grid_cfg = config or GridConfig()

    # Base grid
    k_grid = np.linspace(grid_cfg.k_min, grid_cfg.k_max, grid_cfg.n_k)
    # Clip T to surface calibrated domain
    T_min = max(grid_cfg.T_min, surface.theta_curve.T_min)
    T_max = min(grid_cfg.T_max, surface.theta_curve.T_max)
    T_grid = np.linspace(T_min, T_max, grid_cfg.n_T)

    b_res = check_butterfly_arbitrage_grid(surface, k_grid, T_grid, tolerance=grid_cfg.g_tolerance)
    c_res = check_calendar_arbitrage_grid(surface, k_grid, T_grid, tolerance=grid_cfg.w_T_tolerance)

    # Refinement check (2x resolution)
    if grid_cfg.refinement_factor > 1:
        k_dense = np.linspace(grid_cfg.k_min, grid_cfg.k_max, grid_cfg.n_k * grid_cfg.refinement_factor)
        T_dense = np.linspace(T_min, T_max, grid_cfg.n_T * grid_cfg.refinement_factor)
        b_dense = check_butterfly_arbitrage_grid(surface, k_dense, T_dense, tolerance=grid_cfg.g_tolerance)
        c_dense = check_calendar_arbitrage_grid(surface, k_dense, T_dense, tolerance=grid_cfg.w_T_tolerance)
        # Update worst coordinates if dense scan found smaller
        if b_dense.min_g < b_res.min_g:
            b_res = b_dense
        if c_dense.min_w_T < c_res.min_w_T:
            c_res = c_dense

    # Determine overall grid status
    if b_res.status == StatusCode.GRID_DIAGNOSTIC_FAILED or c_res.status == StatusCode.GRID_DIAGNOSTIC_FAILED:
        overall_status = StatusCode.GRID_DIAGNOSTIC_FAILED
    elif b_res.status == StatusCode.GRID_DIAGNOSTIC_WARNING or c_res.status == StatusCode.GRID_DIAGNOSTIC_WARNING:
        overall_status = StatusCode.GRID_DIAGNOSTIC_WARNING
    else:
        overall_status = StatusCode.GRID_DIAGNOSTIC_PASSED

    return GridDiagnostics(
        grid_status=overall_status,
        min_g=b_res.min_g,
        worst_g_coordinate=b_res.worst_g_coordinate,
        negative_g_count=b_res.negative_g_count,
        min_w_T=c_res.min_w_T,
        worst_w_T_coordinate=c_res.worst_w_T_coordinate,
        negative_w_T_count=c_res.negative_w_T_count,
        min_call_convexity=b_res.min_call_convexity,
        convexity_violation_count=b_res.convexity_violation_count,
        k_min=float(k_grid[0]),
        k_max=float(k_grid[-1]),
        T_min=float(T_grid[0]),
        T_max=float(T_grid[-1]),
        n_k=len(k_grid),
        n_T=len(T_grid),
    )


def run_market_quote_diagnostics(
    quotes: List[OptionQuote],
) -> Dict[str, any]:
    """Inspect raw market quote quality, put-call parity intervals, and spread characteristics."""
    calls_by_key = {}
    puts_by_key = {}

    for q in quotes:
        key = (round(q.time_to_maturity, 4), round(q.strike, 2))
        if q.option_type == "call":
            calls_by_key[key] = q
        else:
            puts_by_key[key] = q

    matched_keys = set(calls_by_key.keys()).intersection(set(puts_by_key.keys()))
    parity_violations = 0
    parity_errors = []

    for key in matched_keys:
        cq = calls_by_key[key]
        pq = puts_by_key[key]
        if cq.forward is not None and cq.discount_factor is not None:
            p_res = check_put_call_parity_intervals(
                call_bid=cq.bid,
                call_ask=cq.ask,
                put_bid=pq.bid,
                put_ask=pq.ask,
                forward=cq.forward,
                strike=cq.strike,
                discount_factor=cq.discount_factor,
                time_to_maturity=cq.time_to_maturity,
            )
            parity_errors.append(p_res.absolute_parity_error)
            if not p_res.is_interval_consistent:
                parity_violations += 1

    mean_err = float(np.mean(parity_errors)) if parity_errors else 0.0
    max_err = float(np.max(parity_errors)) if parity_errors else 0.0

    return {
        "total_quotes": len(quotes),
        "matched_pairs": len(matched_keys),
        "parity_interval_violations": parity_violations,
        "mean_parity_error": mean_err,
        "max_parity_error": max_err,
        "raw_quotes_note": "A mathematically certified SSVI surface does not imply raw market quotes are perfectly arbitrage-free.",
    }
