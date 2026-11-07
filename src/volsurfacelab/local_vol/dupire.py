"""Dupire local volatility engine, numerical safeguards, and flat-volatility validation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from volsurfacelab.config import LocalVolConfig
from volsurfacelab.domain.exceptions import InvalidInputError, StatusCode
from volsurfacelab.domain.models import LocalVolResult
from volsurfacelab.surface.ssvi import SSVISurface


class FlatVolSurface:
    """Independent flat total-variance benchmark surface w(k, T) = sigma_0^2 * T."""

    def __init__(self, sigma_0: float, T_min: float = 0.01, T_max: float = 5.0):
        if sigma_0 <= 0.0:
            raise InvalidInputError("Constant volatility sigma_0 must be strictly positive.")
        self.sigma_0 = float(sigma_0)
        self.T_min = float(T_min)
        self.T_max = float(T_max)

    def total_variance(self, k: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_arr = np.asarray(T, dtype=np.float64)
        return float(self.sigma_0**2 * T_arr) if T_arr.ndim == 0 else self.sigma_0**2 * T_arr

    def derivatives_k(self, k: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        k_arr = np.asarray(k, dtype=np.float64)
        zeros = np.zeros_like(k_arr)
        return (0.0, 0.0) if k_arr.ndim == 0 else (zeros, zeros)

    def derivative_T(self, k: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_arr = np.asarray(T, dtype=np.float64)
        val = self.sigma_0**2
        return float(val) if T_arr.ndim == 0 else np.full_like(T_arr, val)

    def durrleman_g(self, k: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # For flat vol, w_k = 0, w_kk = 0 => g(k, T) = (1 - 0)^2 - 0 + 0 = 1.0
        k_arr = np.asarray(k, dtype=np.float64)
        return 1.0 if k_arr.ndim == 0 else np.ones_like(k_arr)


def evaluate_dupire_local_volatility(
    surface: SSVISurface | FlatVolSurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    forward: float = 100.0,
    config: Optional[LocalVolConfig] = None,
) -> LocalVolResult:
    """Compute Dupire local variance and local volatility on a 2D rectangular grid.

    Calculates:
    - local_variance = w_T(k, T) / g(k, T)
    - local_volatility = sqrt(local_variance) on valid cells.

    Strict Safeguards:
    - No silent clipping or smoothing is applied.
    - Invalid or ill-conditioned cells are assigned reason codes.
    """
    lv_cfg = config or LocalVolConfig()
    g_min = lv_cfg.g_denominator_min

    n_T = len(T_grid)
    n_k = len(k_grid)

    local_variance: List[List[Optional[float]]] = [[None] * n_k for _ in range(n_T)]
    local_volatility: List[List[Optional[float]]] = [[None] * n_k for _ in range(n_T)]
    validity_mask: List[List[bool]] = [[False] * n_k for _ in range(n_T)]
    reason_codes: List[List[str]] = [["" for _ in range(n_k)] for _ in range(n_T)]
    strikes_grid: List[List[float]] = [[0.0] * n_k for _ in range(n_T)]

    valid_cells = 0
    total_cells = n_T * n_k
    valid_vol_values: List[float] = []

    for i, T in enumerate(T_grid):
        for j, k in enumerate(k_grid):
            strike = float(forward * np.exp(k))
            strikes_grid[i][j] = strike

            # Check maturity
            if T <= 0.0:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_INVALID_MATURITY.value
                continue

            # Evaluate analytic derivatives
            try:
                w_T = float(surface.derivative_T(k, T))
                g = float(surface.durrleman_g(k, T))
            except Exception as exc:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_NON_FINITE.value
                continue

            if not (np.isfinite(w_T) and np.isfinite(g)):
                reason_codes[i][j] = StatusCode.LOCAL_VOL_NON_FINITE.value
                continue

            # Guard 1: Butterfly denominator g <= 0
            if g <= 0.0:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_DENOMINATOR_NON_POSITIVE.value
                continue

            # Guard 2: Calendar numerator w_T < 0
            if w_T < -1e-8:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_NUMERATOR_NEGATIVE.value
                continue

            # Guard 3: Near-singular denominator
            if g < g_min:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_DENOMINATOR_NEAR_ZERO.value
                continue

            # Compute variance ratio
            loc_var = w_T / g
            if loc_var < 0.0:
                reason_codes[i][j] = StatusCode.LOCAL_VOL_VARIANCE_NEGATIVE.value
                continue

            loc_vol = np.sqrt(loc_var)
            if not np.isfinite(loc_vol):
                reason_codes[i][j] = StatusCode.LOCAL_VOL_NON_FINITE.value
                continue

            # Cell is mathematically valid
            local_variance[i][j] = float(loc_var)
            local_volatility[i][j] = float(loc_vol)
            validity_mask[i][j] = True
            reason_codes[i][j] = StatusCode.LOCAL_VOL_VALID.value
            valid_cells += 1
            valid_vol_values.append(float(loc_vol))

    valid_fraction = float(valid_cells / total_cells) if total_cells > 0 else 0.0
    min_vol = float(np.min(valid_vol_values)) if valid_vol_values else None
    max_vol = float(np.max(valid_vol_values)) if valid_vol_values else None

    return LocalVolResult(
        expiries=T_grid.tolist(),
        log_moneyness_grid=k_grid.tolist(),
        strikes_grid=strikes_grid,
        local_variance=local_variance,
        local_volatility=local_volatility,
        validity_mask=validity_mask,
        reason_codes=reason_codes,
        valid_fraction=valid_fraction,
        total_cells=total_cells,
        valid_cells=valid_cells,
        min_local_vol=min_vol,
        max_local_vol=max_vol,
    )
