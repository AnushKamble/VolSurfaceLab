"""Unit tests for parameter admissibility, H_max calculation, eta bounds, and bijective parameter mappings."""

import numpy as np
import pytest

from volsurfacelab.domain.exceptions import InvalidInputError, StatusCode
from volsurfacelab.surface.admissible_parameters import (
    compute_eta_cap,
    compute_h_max,
    params_to_unconstrained,
    unconstrained_to_params,
    verify_ssvi_admissibility,
)


def test_h_max_values_and_limiting_branch():
    """Verify H_max(gamma) values across domain and limit at gamma = 0.5."""
    # At gamma = 0.5, H_max = 1.0 exactly
    assert abs(compute_h_max(0.5) - 1.0) < 1e-12

    # Near limit gamma = 0.499999
    assert abs(compute_h_max(0.499999) - 1.0) < 1e-4

    # At gamma = 0.25: H_max = 0.5^0.5 / 1.5^1.5 = sqrt(0.5) / (1.5 * sqrt(1.5)) ~ 0.384900
    expected_g25 = (0.5**0.5) / (1.5**1.5)
    assert abs(compute_h_max(0.25) - expected_g25) < 1e-6

    # Monotonicity: H_max must increase monotonically as gamma goes from 0.05 to 0.5
    gammas = np.linspace(0.05, 0.5, 50)
    h_vals = [compute_h_max(g) for g in gammas]
    assert all(h_vals[i] <= h_vals[i + 1] + 1e-12 for i in range(len(h_vals) - 1))


def test_eta_cap_bounds():
    """Verify eta_cap strictly enforces both sufficient conditions with safety margin."""
    rho = -0.4
    gamma = 0.35
    margin = 0.02
    eta_cap = compute_eta_cap(rho, gamma, safety_margin=margin)

    # Condition 1: eta * (1 + |rho|) < 4
    cond1_lhs = eta_cap * (1.0 + abs(rho))
    assert cond1_lhs < 4.0

    # Condition 2: eta^2 * H_max * (1 + |rho|) <= 4
    h_max = compute_h_max(gamma)
    cond2_lhs = (eta_cap**2) * h_max * (1.0 + abs(rho))
    assert cond2_lhs <= 4.0


def test_parameter_mapping_bijection_roundtrip():
    """Verify bijective mapping from constrained to unconstrained space and back."""
    rho_true = -0.35
    gamma_true = 0.38
    eta_cap = compute_eta_cap(rho_true, gamma_true, safety_margin=0.01)
    eta_true = 0.7 * eta_cap
    expiries = [0.1, 0.5, 1.0, 2.0]
    thetas_true = [0.03, 0.07, 0.12, 0.21]

    u = params_to_unconstrained(rho_true, gamma_true, eta_true, thetas_true, safety_margin=0.01)
    rho_rec, eta_rec, gamma_rec, thetas_rec = unconstrained_to_params(u, expiries, safety_margin=0.01)

    assert abs(rho_rec - rho_true) < 1e-5
    assert abs(gamma_rec - gamma_true) < 1e-5
    assert abs(eta_rec - eta_true) < 1e-5
    for th_orig, th_new in zip(thetas_true, thetas_rec):
        assert abs(th_orig - th_new) < 1e-5


def test_admissibility_verification():
    """Verify that verify_ssvi_admissibility properly accepts valid and rejects invalid parameters."""
    thetas = [0.02, 0.05, 0.10]

    # Valid
    res_valid = verify_ssvi_admissibility(rho=-0.3, eta=1.0, gamma=0.35, theta_knots=thetas)
    assert res_valid.is_admissible
    assert res_valid.status_code == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS
    assert res_valid.cond1_margin > 0.0
    assert res_valid.cond2_margin >= 0.0

    # Invalid: eta exceeds cap
    res_invalid_eta = verify_ssvi_admissibility(rho=-0.3, eta=5.0, gamma=0.35, theta_knots=thetas)
    assert not res_invalid_eta.is_admissible
    assert res_invalid_eta.status_code == StatusCode.CERTIFICATE_FAILED

    # Invalid: non-monotone theta knots
    res_invalid_theta = verify_ssvi_admissibility(rho=-0.3, eta=1.0, gamma=0.35, theta_knots=[0.10, 0.05, 0.20])
    assert not res_invalid_theta.is_admissible
