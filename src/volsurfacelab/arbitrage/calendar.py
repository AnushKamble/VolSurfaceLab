"""Layer B (Part 2): Independent numerical calendar arbitrage diagnostics."""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple
import numpy as np

from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.surface.ssvi import SSVISurface


class CalendarDiagnosticResult(NamedTuple):
    """Result of dense numerical calendar arbitrage diagnostics."""
    status: StatusCode
    min_w_T: float
    worst_w_T_coordinate: Tuple[float, float]
    negative_w_T_count: int
    details: Dict[str, any]


def check_calendar_arbitrage_grid(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    tolerance: float = -1e-6,
) -> CalendarDiagnosticResult:
    """Evaluate analytic calendar derivative w_T = dw/dT on a dense 2D grid.

    Under no-arbitrage at fixed log-forward moneyness, w_T(k, T) >= 0.
    """
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    w_T_mesh = surface.derivative_T(K_mesh, T_mesh)

    min_w_T = float(np.min(w_T_mesh))
    min_idx = np.unravel_index(np.argmin(w_T_mesh), w_T_mesh.shape)
    worst_k = float(k_grid[min_idx[1]])
    worst_T = float(T_grid[min_idx[0]])

    neg_count = int(np.sum(w_T_mesh < tolerance))

    if neg_count == 0:
        status = StatusCode.GRID_DIAGNOSTIC_PASSED
    elif min_w_T >= -1e-4:
        status = StatusCode.GRID_DIAGNOSTIC_WARNING
    else:
        status = StatusCode.GRID_DIAGNOSTIC_FAILED

    return CalendarDiagnosticResult(
        status=status,
        min_w_T=min_w_T,
        worst_w_T_coordinate=(worst_k, worst_T),
        negative_w_T_count=neg_count,
        details={
            "n_k": len(k_grid),
            "n_T": len(T_grid),
            "k_range": (float(k_grid[0]), float(k_grid[-1])),
            "T_range": (float(T_grid[0]), float(T_grid[-1])),
            "tolerance": tolerance,
        },
    )
