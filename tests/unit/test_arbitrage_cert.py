"""Unit tests for Layer A: Theorem-based SSVI admissibility certificate."""

import pytest

from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import SSVIParameters


def test_ssvi_certificate_valid():
    """Verify valid parameter set receives affirmative theorem certificate with positive margins."""
    params = SSVIParameters(
        rho=-0.4,
        eta=1.1,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.02, 0.05, 0.10, 0.18],
    )
    report = generate_ssvi_certificate(params)

    assert report.theorem_status == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS
    assert report.rho_in_domain
    assert report.rho_margin > 0.0
    assert report.gamma_in_domain
    assert report.gamma_margin >= 0.0
    assert report.eta_positive
    assert report.eta_margin >= 0.0
    assert report.cond1_margin > 0.0
    assert report.cond2_margin >= 0.0
    assert report.theta_monotone
    assert report.calendar_ratio_bound_satisfied


def test_ssvi_certificate_invalid_eta_cap():
    """Verify that parameters with eta exceeding the sufficient condition cap fail certification."""
    from pydantic import ValidationError

    # When eta exceeds the sufficient cap, certificate must report failure
    # Valid Pydantic model with eta near cap:
    params = SSVIParameters(
        rho=-0.5,
        eta=2.6,  # Cap for rho=-0.5, gamma=0.4 is ~ 2.45
        gamma=0.4,
        expiries=[0.2, 1.0],
        theta_knots=[0.03, 0.12],
    )
    report = generate_ssvi_certificate(params)
    assert report.theorem_status == StatusCode.CERTIFICATE_FAILED
    assert report.eta_margin < 0.0 or report.cond1_margin <= 0.0 or report.cond2_margin < 0.0
