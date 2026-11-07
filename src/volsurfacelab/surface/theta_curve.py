"""Monotone shape-preserving ATM total-variance term structure interpolator."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np
from scipy.interpolate import PchipInterpolator

from volsurfacelab.domain.exceptions import InvalidInputError


class ThetaCurve:
    """Monotone shape-preserving ATM total-variance curve theta(T) using PCHIP interpolation.

    Guarantees:
    - theta(T) > 0 for all T >= T_1.
    - theta'(T) >= 0 for all T in [T_1, T_M] (preserves monotonicity).
    - Continuously differentiable C^1 across knots.
    """

    def __init__(
        self,
        expiries: Sequence[float],
        theta_knots: Sequence[float],
        allow_extrapolation: bool = False,
    ):
        T_arr = np.asarray(expiries, dtype=np.float64)
        th_arr = np.asarray(theta_knots, dtype=np.float64)

        if len(T_arr) != len(th_arr):
            raise InvalidInputError("Expiries and theta knots must have identical lengths.")
        if len(T_arr) < 2:
            raise InvalidInputError("ThetaCurve requires at least 2 maturity knots for interpolation.")
        if np.any(T_arr <= 0.0):
            raise InvalidInputError("All expiries must be strictly positive.")
        if np.any(th_arr <= 0.0):
            raise InvalidInputError("All theta knots must be strictly positive.")
        if np.any(np.diff(T_arr) <= 0.0):
            raise InvalidInputError("Expiries must be strictly increasing.")
        if np.any(np.diff(th_arr) <= 0.0):
            raise InvalidInputError("Theta knots must be strictly increasing.")

        self.expiries = T_arr
        self.theta_knots = th_arr
        self.allow_extrapolation = allow_extrapolation

        # PchipInterpolator preserves monotonicity and avoids overshoot
        self._pchip = PchipInterpolator(self.expiries, self.theta_knots, extrapolate=allow_extrapolation)
        self._pchip_deriv = self._pchip.derivative()

    @property
    def T_min(self) -> float:
        return float(self.expiries[0])

    @property
    def T_max(self) -> float:
        return float(self.expiries[-1])

    def evaluate(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate ATM total variance theta(T)."""
        T_in = np.asarray(T, dtype=np.float64)
        is_scalar = (T_in.ndim == 0)

        if not self.allow_extrapolation:
            if np.any(T_in < self.T_min - 1e-10) or np.any(T_in > self.T_max + 1e-10):
                raise InvalidInputError(
                    f"Maturity {T_in} outside calibrated domain [{self.T_min:.4f}, {self.T_max:.4f}] without extrapolation enabled."
                )

        vals = self._pchip(T_in)
        # Ensure positivity
        vals = np.maximum(vals, 1e-8)
        return float(vals) if is_scalar else vals

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate time derivative theta'(T) = d(theta)/dT >= 0."""
        T_in = np.asarray(T, dtype=np.float64)
        is_scalar = (T_in.ndim == 0)

        if not self.allow_extrapolation:
            if np.any(T_in < self.T_min - 1e-10) or np.any(T_in > self.T_max + 1e-10):
                raise InvalidInputError(
                    f"Maturity {T_in} outside calibrated domain [{self.T_min:.4f}, {self.T_max:.4f}]."
                )

        derivs = self._pchip_deriv(T_in)
        # Numerical guard: PCHIP is mathematically monotone, but floating-point rounding may give -1e-16
        derivs = np.maximum(derivs, 0.0)
        return float(derivs) if is_scalar else derivs

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.evaluate(T)
