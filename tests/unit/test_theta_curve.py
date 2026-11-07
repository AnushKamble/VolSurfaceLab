"""Unit tests for monotone PCHIP ATM total-variance term structure."""

import numpy as np
import pytest

from volsurfacelab.domain.exceptions import InvalidInputError
from volsurfacelab.surface.theta_curve import ThetaCurve


def test_theta_curve_interpolation_exact():
    """Verify that theta curve reproduces exact knot values."""
    expiries = [0.1, 0.25, 0.5, 1.0, 2.0]
    knots = [0.015, 0.035, 0.065, 0.12, 0.22]
    curve = ThetaCurve(expiries, knots)

    for T, th in zip(expiries, knots):
        assert abs(curve.evaluate(T) - th) < 1e-12


def test_theta_curve_monotonicity_and_derivative():
    """Verify that theta(T) is strictly increasing and theta'(T) >= 0 everywhere."""
    expiries = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    knots = [0.02, 0.04, 0.07, 0.11, 0.15, 0.18]
    curve = ThetaCurve(expiries, knots)

    T_dense = np.linspace(0.1, 2.0, 500)
    theta_vals = curve.evaluate(T_dense)
    theta_primes = curve.derivative(T_dense)

    # Strictly increasing
    assert np.all(np.diff(theta_vals) > 0.0)
    # Derivative non-negative
    assert np.all(theta_primes >= 0.0)


def test_theta_curve_extrapolation_guard():
    """Verify that extrapolation is rejected by default."""
    curve = ThetaCurve([0.2, 0.5, 1.0], [0.04, 0.08, 0.15])

    with pytest.raises(InvalidInputError):
        curve.evaluate(0.1)  # T < T_min

    with pytest.raises(InvalidInputError):
        curve.evaluate(1.5)  # T > T_max


def test_theta_curve_invalid_inputs():
    """Verify ThetaCurve rejects non-monotone or non-positive knots."""
    # Non-monotone expiries
    with pytest.raises(InvalidInputError):
        ThetaCurve([0.5, 0.2, 1.0], [0.04, 0.08, 0.15])
    # Non-monotone theta knots
    with pytest.raises(InvalidInputError):
        ThetaCurve([0.2, 0.5, 1.0], [0.08, 0.04, 0.15])
    # Non-positive knots
    with pytest.raises(InvalidInputError):
        ThetaCurve([0.2, 0.5, 1.0], [-0.04, 0.08, 0.15])
