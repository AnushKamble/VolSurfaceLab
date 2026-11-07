"""Layer B (Part 1): Independent numerical butterfly arbitrage and call convexity diagnostics."""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple
import numpy as np

from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance
from volsurfacelab.surface.ssvi import SSVISurface


class ButterflyDiagnosticResult(NamedTuple):
    """Result of dense numerical butterfly arbitrage diagnostics."""
    status: StatusCode
    min_g: float
    worst_g_coordinate: Tuple[float, float]
    negative_g_count: int
    min_call_convexity: float
    convexity_violation_count: int
    details: Dict[str, any]


def check_butterfly_arbitrage_grid(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    tolerance: float = -1e-6,
) -> ButterflyDiagnosticResult:
    """Evaluate analytic Durrleman g(k,T) and discrete call price strike convexity on a dense 2D grid.

    Parameters
    ----------
    surface : SSVISurface
        SSVI volatility surface model.
    k_grid : np.ndarray
        Array of log-forward-moneyness points.
    T_grid : np.ndarray
        Array of maturities in ascending order.
    tolerance : float
        Numerical floating-point tolerance (default: -1e-6). Materially negative values below tolerance trigger failure.

    Returns
    -------
    result : ButterflyDiagnosticResult
    """
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    g_mesh = surface.durrleman_g(K_mesh, T_mesh)

    min_g = float(np.min(g_mesh))
    min_idx = np.unravel_index(np.argmin(g_mesh), g_mesh.shape)
    worst_k = float(k_grid[min_idx[1]])
    worst_T = float(T_grid[min_idx[0]])

    neg_count = int(np.sum(g_mesh < tolerance))

    # Discrete call price strike convexity check across all T slices
    min_convexity = float("inf")
    convexity_violations = 0

    F_ref = 100.0
    D_ref = 1.0

    for T in T_grid:
        w_slice = surface.total_variance(k_grid, T)
        strikes = F_ref * np.exp(k_grid)
        call_prices = black_forward_price_from_total_variance(
            forward=F_ref,
            strike=strikes,
            discount_factor=D_ref,
            total_variance=w_slice,
            option_type="call",
        )

        # Check call monotonicity: dC/dK <= 0
        diff_calls = np.diff(call_prices)
        if np.any(diff_calls > 1e-8):
            convexity_violations += int(np.sum(diff_calls > 1e-8))

        # Check second difference: (C_{i+1} - 2*C_i + C_{i-1}) / dK^2 >= 0
        dK = np.diff(strikes)
        # For non-uniform strike grid, standard second divided difference:
        dK1 = strikes[1:-1] - strikes[:-2]
        dK2 = strikes[2:] - strikes[1:-1]
        c0 = call_prices[:-2]
        c1 = call_prices[1:-1]
        c2 = call_prices[2:]
        # Second divided difference
        second_diff = 2.0 * ((c2 - c1) / dK2 - (c1 - c0) / dK1) / (dK1 + dK2)

        local_min_conv = float(np.min(second_diff))
        if local_min_conv < min_convexity:
            min_convexity = local_min_conv

        conv_viol = int(np.sum(second_diff < tolerance))
        convexity_violations += conv_viol

    if neg_count == 0 and convexity_violations == 0:
        status = StatusCode.GRID_DIAGNOSTIC_PASSED
    elif min_g >= -1e-4 and min_convexity >= -1e-4:
        status = StatusCode.GRID_DIAGNOSTIC_WARNING
    else:
        status = StatusCode.GRID_DIAGNOSTIC_FAILED

    return ButterflyDiagnosticResult(
        status=status,
        min_g=min_g,
        worst_g_coordinate=(worst_k, worst_T),
        negative_g_count=neg_count,
        min_call_convexity=min_convexity,
        convexity_violation_count=convexity_violations,
        details={
            "n_k": len(k_grid),
            "n_T": len(T_grid),
            "k_range": (float(k_grid[0]), float(k_grid[-1])),
            "T_range": (float(T_grid[0]), float(T_grid[-1])),
            "tolerance": tolerance,
        },
    )
