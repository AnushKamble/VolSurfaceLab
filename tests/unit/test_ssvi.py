"""Unit tests for SSVI surface basic identities, parameters, and symmetry."""

import numpy as np
import pytest

from volsurfacelab.domain.exceptions import InvalidInputError
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.surface.ssvi import SSVISurface, ssvi_phi


def test_ssvi_atm_exact_identity():
    """Verify that at log-forward-moneyness k=0, total variance w(0, theta) == theta exactly."""
    params = SSVIParameters(
        rho=-0.4,
        eta=1.2,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0],
        theta_knots=[0.02, 0.08, 0.16],
    )
    surface = SSVISurface(params)

    for T, expected_theta in zip(params.expiries, params.theta_knots):
        w_atm = surface.total_variance(k=0.0, T=T)
        assert abs(w_atm - expected_theta) < 1e-12


def test_ssvi_symmetric_smile_when_rho_zero():
    """Verify that when rho = 0, smile is perfectly symmetric in log-moneyness: w(k) == w(-k)."""
    params = SSVIParameters(
        rho=0.0,
        eta=1.0,
        gamma=0.4,
        expiries=[0.25, 1.0],
        theta_knots=[0.04, 0.12],
    )
    surface = SSVISurface(params)

    k_vals = [0.1, 0.3, 0.6, 1.0]
    for k in k_vals:
        w_pos = surface.total_variance(k, T=0.5)
        w_neg = surface.total_variance(-k, T=0.5)
        assert abs(w_pos - w_neg) < 1e-12


def test_ssvi_positive_total_variance():
    """Verify total variance is strictly positive across wide log-moneyness domain."""
    params = SSVIParameters(
        rho=-0.5,
        eta=1.5,
        gamma=0.3,
        expiries=[0.1, 1.0],
        theta_knots=[0.03, 0.15],
    )
    surface = SSVISurface(params)

    k_dense = np.linspace(-3.0, 3.0, 200)
    w_vals = surface.total_variance(k_dense, T=0.5)
    assert np.all(w_vals > 0.0)


def test_ssvi_invalid_parameter_guards():
    """Verify surface and parameters reject invalid rho, eta, and gamma bounds."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # gamma > 0.5
        SSVIParameters(rho=-0.2, eta=1.0, gamma=0.55, expiries=[0.5, 1.0], theta_knots=[0.05, 0.10])

    with pytest.raises(ValidationError):
        # abs(rho) >= 1.0
        SSVIParameters(rho=-1.0, eta=1.0, gamma=0.4, expiries=[0.5, 1.0], theta_knots=[0.05, 0.10])

    with pytest.raises(ValidationError):
        # eta <= 0
        SSVIParameters(rho=-0.2, eta=-0.5, gamma=0.4, expiries=[0.5, 1.0], theta_knots=[0.05, 0.10])

