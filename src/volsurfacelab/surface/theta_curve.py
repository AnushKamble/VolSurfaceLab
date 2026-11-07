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
        self._pchip = PchipInterpolator(self.expiries, self.theta_knots, extrapolate=False)
        self._pchip_deriv = self._pchip.derivative()
        self._d_min = max(float(self._pchip_deriv(self.T_min)), 1e-8)
        self._d_max = max(float(self._pchip_deriv(self.T_max)), 1e-8)
        # C^1 smooth power-law left extrapolation exponent alpha = d_min * T_min / theta_1
        # Guarantees theta(T_min) = theta_1, theta'(T_min) = d_min, theta(T) > 0, theta'(T) >= 0, and theta(0) = 0.
        self._alpha = max(self._d_min * self.T_min / self.theta_knots[0], 1e-4)

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

        if np.any(T_in <= 0.0):
            raise InvalidInputError(f"Maturity T must be strictly positive, got {T_in}.")

        if not self.allow_extrapolation:
            if np.any(T_in < self.T_min - 1e-10) or np.any(T_in > self.T_max + 1e-10):
                raise InvalidInputError(
                    f"Maturity {T_in} outside calibrated domain [{self.T_min:.4f}, {self.T_max:.4f}] without extrapolation enabled."
                )

        if is_scalar:
            t = float(T_in)
            if t < self.T_min:
                val = self.theta_knots[0] * (max(t, 1e-8) / self.T_min) ** self._alpha
            elif t > self.T_max:
                val = self.theta_knots[-1] + self._d_max * (t - self.T_max)
            else:
                val = float(self._pchip(t))
            val = max(val, 1e-8)
            return val

        vals = np.zeros_like(T_in, dtype=np.float64)
        interior_mask = (T_in >= self.T_min) & (T_in <= self.T_max)
        left_mask = T_in < self.T_min
        right_mask = T_in > self.T_max

        if np.any(interior_mask):
            vals[interior_mask] = self._pchip(T_in[interior_mask])
        if np.any(left_mask):
            t_left = np.maximum(T_in[left_mask], 1e-8)
            vals[left_mask] = self.theta_knots[0] * (t_left / self.T_min) ** self._alpha
        if np.any(right_mask):
            t_right = T_in[right_mask]
            vals[right_mask] = self.theta_knots[-1] + self._d_max * (t_right - self.T_max)

        vals = np.maximum(vals, 1e-8)
        return vals

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate time derivative theta'(T) = d(theta)/dT >= 0."""
        T_in = np.asarray(T, dtype=np.float64)
        is_scalar = (T_in.ndim == 0)

        if np.any(T_in <= 0.0):
            raise InvalidInputError(f"Maturity T must be strictly positive, got {T_in}.")

        if not self.allow_extrapolation:
            if np.any(T_in < self.T_min - 1e-10) or np.any(T_in > self.T_max + 1e-10):
                raise InvalidInputError(
                    f"Maturity {T_in} outside calibrated domain [{self.T_min:.4f}, {self.T_max:.4f}]."
                )

        if is_scalar:
            t = float(T_in)
            if t < self.T_min:
                deriv = self._d_min * (max(t, 1e-8) / self.T_min) ** (self._alpha - 1.0)
            elif t > self.T_max:
                deriv = self._d_max
            else:
                deriv = float(self._pchip_deriv(t))
            return max(deriv, 0.0)

        derivs = np.zeros_like(T_in, dtype=np.float64)
        interior_mask = (T_in >= self.T_min) & (T_in <= self.T_max)
        left_mask = T_in < self.T_min
        right_mask = T_in > self.T_max

        if np.any(interior_mask):
            derivs[interior_mask] = self._pchip_deriv(T_in[interior_mask])
        if np.any(left_mask):
            t_left = np.maximum(T_in[left_mask], 1e-8)
            derivs[left_mask] = self._d_min * (t_left / self.T_min) ** (self._alpha - 1.0)
        if np.any(right_mask):
            derivs[right_mask] = self._d_max

        derivs = np.maximum(derivs, 0.0)
        return derivs

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.evaluate(T)
