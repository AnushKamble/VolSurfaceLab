"""Unit tests for production enhancements: monotone extrapolation, drift-adjusted local vol, Roger Lee wings, analytic CDF, and higher moments."""

import numpy as np
import pytest

from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.density.breeden_litzenberger import (
    extract_risk_neutral_density,
    risk_neutral_cdf_analytic,
)
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.local_vol.dupire import (
    FlatVolSurface,
    evaluate_dupire_local_volatility,
)
from volsurfacelab.pricing.black_forward import black_forward_price
from volsurfacelab.pricing.implied_vol import implied_volatility_brent
from volsurfacelab.pricing.parity import check_put_call_parity_intervals
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.surface.theta_curve import ThetaCurve


def test_theta_curve_monotone_extrapolation():
    """Verify that ThetaCurve with allow_extrapolation=True extrapolates monotonically without negative variance."""
    expiries = [0.1, 0.5, 1.0, 2.0]
    knots = [0.015, 0.045, 0.085, 0.14]
    curve = ThetaCurve(expiries, knots, allow_extrapolation=True)

    # 1. Right extrapolation (T > 2.0)
    T_far_right = np.array([2.5, 5.0, 10.0])
    th_far_right = curve.evaluate(T_far_right)
    d_far_right = curve.derivative(T_far_right)

    # Must be strictly positive and strictly greater than terminal knot
    assert np.all(th_far_right > knots[-1])
    # Derivatives must be non-negative
    assert np.all(d_far_right >= 0.0)
    # Total variance must strictly increase
    assert np.all(np.diff(th_far_right) > 0.0)

    # 2. Left extrapolation (0 < T < 0.1)
    T_far_left = np.array([0.01, 0.05, 0.08])
    th_far_left = curve.evaluate(T_far_left)
    d_far_left = curve.derivative(T_far_left)

    assert np.all(th_far_left > 0.0)
    assert np.all(th_far_left < knots[0])
    assert np.all(d_far_left >= 0.0)
    assert np.all(np.diff(th_far_left) > 0.0)

    # 3. Continuity across boundary
    eps = 1e-6
    # Right boundary C^1 continuity
    assert abs(curve.evaluate(2.0 + eps) - curve.evaluate(2.0 - eps)) < 1e-4
    assert abs(curve.derivative(2.0 + eps) - curve.derivative(2.0 - eps)) < 1e-4
    # Left boundary C^1 continuity (verified by power-law extrapolation)
    assert abs(curve.evaluate(0.1 + eps) - curve.evaluate(0.1 - eps)) < 1e-4
    assert abs(curve.derivative(0.1 + eps) - curve.derivative(0.1 - eps)) < 1e-4


def test_dupire_local_volatility_with_forward_drift():
    """Verify that Dupire local volatility in forward log-moneyness coordinates is w_T / g.

    Proves that in forward log-moneyness k = ln(K / F_T), the convective derivative -mu * w_k
    cancels exactly with the PDE drift term mu * K * partial_K C, so local variance is w_T / g
    identically across any drift rate mu.
    """
    params = SSVIParameters(
        rho=-0.4,
        eta=1.0,
        gamma=0.35,
        expiries=[0.25, 0.5, 1.0, 2.0],
        theta_knots=[0.03, 0.06, 0.11, 0.19],
    )
    surface = SSVISurface(params)
    k_grid = np.linspace(-0.5, 0.5, 11)
    T_grid = np.linspace(0.3, 1.8, 7)

    # 1. Zero drift (martingale / forward measure)
    res_zero = evaluate_dupire_local_volatility(surface, k_grid, T_grid, forward=100.0, drift=0.0)
    assert res_zero.valid_cells > 0
    assert res_zero.drift == 0.0

    # 2. Positive drift (e.g. r=0.05, q=0.02 => mu = 0.03)
    res_drift = evaluate_dupire_local_volatility(surface, k_grid, T_grid, forward=100.0, drift=0.03)
    assert res_drift.valid_cells > 0

    # In forward log-moneyness coordinates, local variance is identically w_T / g
    for i in range(len(T_grid)):
        for j, k in enumerate(k_grid):
            if res_zero.validity_mask[i][j]:
                w_T = surface.derivative_T(k, T_grid[i])
                g = surface.durrleman_g(k, T_grid[i])
                expected_var = w_T / g
                assert abs(res_zero.local_variance[i][j] - expected_var) < 1e-12
                assert abs(res_drift.local_variance[i][j] - expected_var) < 1e-12

    # 3. Flat volatility surface (w_T = sigma_0^2, g = 1.0 => local_vol = sigma_0)
    flat = FlatVolSurface(sigma_0=0.20)
    res_flat_zero = evaluate_dupire_local_volatility(flat, k_grid, T_grid, forward=100.0, drift=0.0)
    res_flat_drift = evaluate_dupire_local_volatility(flat, k_grid, T_grid, forward=100.0, drift=0.05)
    for i in range(len(T_grid)):
        for j in range(len(k_grid)):
            assert abs(res_flat_zero.local_volatility[i][j] - 0.20) < 1e-10
            assert abs(res_flat_drift.local_volatility[i][j] - 0.20) < 1e-10


def test_roger_lee_moment_formula_wings():
    """Verify Roger Lee asymptotic wing slopes and critical moments calculation."""
    params = SSVIParameters(
        rho=-0.35,
        eta=0.95,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.02, 0.05, 0.09, 0.16],
    )
    cert = generate_ssvi_certificate(params)

    assert cert.theorem_status == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS
    assert cert.lee_wing_right is not None
    assert cert.lee_wing_left is not None
    assert cert.lee_critical_moment_right is not None
    assert cert.lee_critical_moment_left is not None

    # Condition 1 guarantees both wing slopes < 2
    assert 0.0 < cert.lee_wing_right < 2.0
    assert 0.0 < cert.lee_wing_left < 2.0

    # With rho < 0, left wing slope is steeper than right wing slope
    assert cert.lee_wing_left > cert.lee_wing_right

    # Critical moments must be strictly positive
    assert cert.lee_critical_moment_right > 0.0
    assert cert.lee_critical_moment_left > 0.0

    # Verify exact Lee formula: p = (2 - beta)^2 / (8 * beta)
    expected_p = (2.0 - cert.lee_wing_right)**2 / (8.0 * cert.lee_wing_right)
    assert abs(cert.lee_critical_moment_right - expected_p) < 1e-12

    # Boundary check: beta >= 2 implies critical moment p* = 0 (no higher moments exist)
    # Tested on an uncertified parameter set where eta is intentionally high
    bad_params = SSVIParameters(
        rho=0.5,
        eta=3.5,
        gamma=0.25,
        expiries=[0.1, 1.0],
        theta_knots=[0.04, 0.12],
    )
    bad_cert = generate_ssvi_certificate(bad_params)
    assert bad_cert.lee_wing_right >= 2.0
    assert bad_cert.lee_critical_moment_right == 0.0


def test_breeden_litzenberger_closed_form_cdf_and_moments():
    """Verify analytic closed-form CDF and risk-neutral distribution moments."""
    params = SSVIParameters(
        rho=-0.4,
        eta=1.0,
        gamma=0.35,
        expiries=[0.25, 0.5, 1.0, 2.0],
        theta_knots=[0.03, 0.06, 0.11, 0.19],
    )
    surface = SSVISurface(params)
    forward = 100.0
    maturity = 1.0
    discount = 0.96

    # 1. Analytic CDF values
    k_far_otm = forward * 0.20
    k_atm = forward
    k_deep_itm = forward * 2.5

    cdf_left = risk_neutral_cdf_analytic(surface, k_far_otm, forward, maturity)
    cdf_atm = risk_neutral_cdf_analytic(surface, k_atm, forward, maturity)
    cdf_right = risk_neutral_cdf_analytic(surface, k_deep_itm, forward, maturity)

    assert 0.0 <= cdf_left < 0.01
    assert 0.40 < cdf_atm < 0.60
    assert 0.99 < cdf_right <= 1.0
    assert cdf_left < cdf_atm < cdf_right

    # 2. Density result higher moments
    res = extract_risk_neutral_density(surface, maturity, forward, discount)

    assert res.total_mass == pytest.approx(1.0, abs=0.01)
    assert res.relative_moment_error < 0.01
    assert res.variance is not None and res.variance > 0.0
    assert res.std_dev is not None and res.std_dev > 0.0

    # Price distribution S_T is strictly positive, hence positively skewed
    assert res.skewness is not None
    assert res.skewness > 0.0

    # Negative rho generates negative risk-neutral log-return skewness
    assert res.log_skewness is not None
    assert res.log_skewness < 0.0

    # Smiles generate positive excess kurtosis (leptokurtic tails)
    assert res.kurtosis is not None
    assert res.kurtosis > 0.0
    assert res.log_kurtosis is not None
    assert res.log_kurtosis > 0.0

    # Analytic CDF and numerical CDF maximum discrepancy is small (tightened by left-tail offset alignment)
    assert res.analytic_cdf is not None
    assert res.max_cdf_error is not None
    assert res.max_cdf_error < 0.005


def test_implied_vol_near_intrinsic_tight_bracket():
    """Verify that implied volatility inversion smoothly handles prices near intrinsic without failing bracket."""
    F = 100.0
    K = 100.0
    D = 1.0
    T = 0.5
    # Very small vol: sigma = 0.0005
    tiny_vol = 0.0005
    p_tiny = black_forward_price(F, K, D, T, tiny_vol, "call")

    res = implied_volatility_brent(p_tiny, F, K, D, T, "call")
    assert res.success
    assert res.iv is not None
    assert abs(res.iv - tiny_vol) < 1e-4
    assert res.status_code == StatusCode.INVERSION_SUCCESS


def test_parity_preserves_time_to_maturity():
    """Verify that check_put_call_parity_intervals properly records time_to_maturity."""
    res = check_put_call_parity_intervals(
        call_bid=5.0,
        call_ask=5.2,
        put_bid=3.0,
        put_ask=3.2,
        forward=100.0,
        strike=98.0,
        discount_factor=1.0,
        time_to_maturity=0.75,
    )
    assert res.time_to_maturity == 0.75
    assert res.is_interval_consistent


def test_dupire_pde_drift_analytic_cancellation():
    """Prove multivariable cancellation of drift terms in Dupire's equation.

    Demonstrates that Dupire's cash PDE numerator:
        Num = partial_T C + (r - q) * K * partial_K C + q * C
    analytically equals:
        D(T) * partial_w C_B * partial_T w|_k
    with zero skew drift term -mu * w_k, confirming Gatheral's formula.
    """
    from scipy.stats import norm
    S0 = 100.0
    r = 0.05
    q = 0.02
    mu = r - q
    T0 = 1.0
    K0 = 105.0

    def vol_skew(k):
        return 0.22 - 0.12 * k + 0.06 * k**2

    def price_call(K, T):
        F = S0 * np.exp(mu * T)
        k = np.log(K / F)
        w = (vol_skew(k)**2) * T
        sqrt_w = np.sqrt(w)
        d1 = -k / sqrt_w + 0.5 * sqrt_w
        d2 = d1 - sqrt_w
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    eps = 1e-6
    C_val = price_call(K0, T0)
    C_T = (price_call(K0, T0 + eps) - price_call(K0, T0 - eps)) / (2 * eps)
    C_K = (price_call(K0 + eps, T0) - price_call(K0 - eps, T0)) / (2 * eps)

    pde_num = C_T + mu * K0 * C_K + q * C_val

    # Analytic expression D(T) * partial_w C_B * partial_T w|_k
    F0 = S0 * np.exp(mu * T0)
    k0 = np.log(K0 / F0)
    w0 = (vol_skew(k0)**2) * T0
    sqrt_w0 = np.sqrt(w0)
    d1_0 = -k0 / sqrt_w0 + 0.5 * sqrt_w0
    dC_B_dw = 0.5 * F0 * norm.pdf(d1_0) / sqrt_w0
    w_T = vol_skew(k0)**2  # partial w / partial T holding k fixed
    D0 = np.exp(-r * T0)
    analytic_num = D0 * dC_B_dw * w_T

    rel_diff = abs(pde_num - analytic_num) / pde_num
    assert rel_diff < 1e-7


def test_theta_curve_power_law_left_boundary_exact():
    """Verify C^1 continuity of power-law left extrapolation when d_min > theta_1 / T_1."""
    # Construct knot set where initial slope d_min is deliberately steeper than secant to origin
    expiries = [0.2, 0.5, 1.0, 2.0]
    knots = [0.02, 0.08, 0.16, 0.25]
    curve = ThetaCurve(expiries, knots, allow_extrapolation=True)

    T1 = expiries[0]
    eps = 1e-6
    val_diff = abs(curve.evaluate(T1 + eps) - curve.evaluate(T1 - eps))
    deriv_diff = abs(curve.derivative(T1 + eps) - curve.derivative(T1 - eps))

    # C^1 smooth match across knot
    assert val_diff < 1e-5
    assert deriv_diff < 1e-5

    # Strictly positive and approaches zero as T -> 0
    assert curve.evaluate(0.01) > 0.0
    assert curve.evaluate(1e-4) < curve.evaluate(0.01)
    assert curve.derivative(0.01) >= 0.0
