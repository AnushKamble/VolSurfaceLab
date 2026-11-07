"""Unit tests for Breeden-Litzenberger risk-neutral density and moment diagnostics."""

import numpy as np
import pytest

from volsurfacelab.config import DensityConfig
from volsurfacelab.density.breeden_litzenberger import (
    extract_risk_neutral_density,
    risk_neutral_density_analytic,
)
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance
from volsurfacelab.surface.ssvi import SSVISurface


@pytest.fixture
def test_surface() -> SSVISurface:
    params = SSVIParameters(
        rho=-0.3,
        eta=1.1,
        gamma=0.35,
        expiries=[0.25, 0.5, 1.0, 2.0],
        theta_knots=[0.03, 0.06, 0.12, 0.20],
    )
    return SSVISurface(params)


def test_density_non_negativity(test_surface):
    """Verify risk-neutral density is non-negative everywhere on certified SSVI surface."""
    F = 100.0
    D = 0.98
    T = 0.5
    res = extract_risk_neutral_density(test_surface, maturity=T, forward=F, discount_factor=D)

    assert res.is_non_negative
    assert res.min_density >= -1e-8
    assert res.negative_count == 0


def test_density_analytic_vs_discrete_call_curvature(test_surface):
    """Verify analytic q_T(K) matches discrete second derivative of call prices."""
    F = 100.0
    D = 0.95
    T = 0.5
    h = 0.02

    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    for K in strikes:
        q_analytic = risk_neutral_density_analytic(test_surface, K, F, T)

        # Call prices at K-h, K, K+h
        k_minus = np.log((K - h) / F)
        k_center = np.log(K / F)
        k_plus = np.log((K + h) / F)

        w_minus = test_surface.total_variance(k_minus, T)
        w_center = test_surface.total_variance(k_center, T)
        w_plus = test_surface.total_variance(k_plus, T)

        c_minus = black_forward_price_from_total_variance(F, K - h, D, w_minus, "call")
        c_center = black_forward_price_from_total_variance(F, K, D, w_center, "call")
        c_plus = black_forward_price_from_total_variance(F, K + h, D, w_plus, "call")

        # C_KK / D
        c_kk = (c_plus - 2.0 * c_center + c_minus) / (h**2)
        q_fd = c_kk / D

        rel_err = abs(q_analytic - q_fd) / max(q_analytic, 1e-4)
        assert rel_err < 1e-3


def test_density_mass_and_first_moment(test_surface):
    """Verify total mass converges to ~1 and first moment converges to forward price on broad domain."""
    F = 100.0
    D = 0.96
    T = 1.0
    cfg = DensityConfig(n_strikes=501, std_dev_range=6.0)
    res = extract_risk_neutral_density(test_surface, maturity=T, forward=F, discount_factor=D, config=cfg)

    # Mass should be very close to 1 on a 6-sigma domain
    assert abs(res.total_mass - 1.0) < 0.01
    # First moment should match forward within 1%
    assert res.relative_moment_error < 0.01
    assert abs(res.first_moment - F) < 1.0


def test_density_no_artificial_renormalisation(test_surface):
    """Verify that density is NOT scaled to force integral to 1.0 when domain is narrow."""
    F = 100.0
    D = 0.98
    T = 0.5
    # Narrow domain (1.5 sigma) truncates substantial tail mass
    cfg_narrow = DensityConfig(n_strikes=201, std_dev_range=1.5)
    res_narrow = extract_risk_neutral_density(test_surface, maturity=T, forward=F, discount_factor=D, config=cfg_narrow)

    # Mass should reflect truncation (e.g. ~ 0.86), NOT be forced to 1.0
    assert res_narrow.total_mass < 0.95
    assert len(res_narrow.warnings) > 0
