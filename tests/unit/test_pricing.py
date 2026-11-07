"""Unit tests for Black forward option pricing, bounds, and put-call parity."""

import numpy as np
import pytest

from volsurfacelab.domain.exceptions import InvalidInputError, StatusCode
from volsurfacelab.pricing.black_forward import (
    black_forward_price,
    black_forward_price_from_total_variance,
    black_forward_vega,
    check_european_bounds,
)
from volsurfacelab.pricing.parity import check_put_call_parity, check_put_call_parity_intervals


def test_black_price_scalar_vector_consistency():
    """Verify scalar and vector implementations yield identical results."""
    F = 100.0
    K_arr = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    D = 0.95
    T = 0.5
    sigma = 0.25

    vec_calls = black_forward_price(F, K_arr, D, T, sigma, "call")
    vec_puts = black_forward_price(F, K_arr, D, T, sigma, "put")

    for i, K in enumerate(K_arr):
        sc_call = black_forward_price(F, K, D, T, sigma, "call")
        sc_put = black_forward_price(F, K, D, T, sigma, "put")
        assert abs(vec_calls[i] - sc_call) < 1e-12
        assert abs(vec_puts[i] - sc_put) < 1e-12


def test_put_call_parity_exact():
    """Verify European put-call parity holds to machine precision: C - P = D * (F - K)."""
    for F in [50.0, 100.0, 5000.0]:
        for K in [0.7 * F, F, 1.3 * F]:
            for D in [0.90, 0.98, 1.0]:
                for T in [0.1, 1.0, 3.0]:
                    for sigma in [0.10, 0.30, 0.80]:
                        call_p = black_forward_price(F, K, D, T, sigma, "call")
                        put_p = black_forward_price(F, K, D, T, sigma, "put")
                        residual, synth_f = check_put_call_parity(call_p, put_p, F, K, D)
                        assert abs(residual) < 1e-10
                        assert abs(synth_f - F) < 1e-9


def test_put_call_parity_intervals():
    """Verify put-call parity interval diagnostics."""
    F = 100.0
    K = 100.0
    D = 0.95
    T = 0.5
    sigma = 0.2

    c_mid = black_forward_price(F, K, D, T, sigma, "call")
    p_mid = black_forward_price(F, K, D, T, sigma, "put")

    # Consistent quotes with realistic spreads
    res_valid = check_put_call_parity_intervals(
        call_bid=c_mid - 0.2,
        call_ask=c_mid + 0.2,
        put_bid=p_mid - 0.2,
        put_ask=p_mid + 0.2,
        forward=F,
        strike=K,
        discount_factor=D,
    )
    assert res_valid.is_interval_consistent
    assert res_valid.absolute_parity_error < 1e-10

    # Inconsistent quotes (arbitrageable)
    res_invalid = check_put_call_parity_intervals(
        call_bid=c_mid + 5.0,
        call_ask=c_mid + 5.5,
        put_bid=p_mid - 0.2,
        put_ask=p_mid + 0.2,
        forward=F,
        strike=K,
        discount_factor=D,
    )
    assert not res_invalid.is_interval_consistent
    assert res_invalid.absolute_parity_error > 4.0


def test_european_price_bounds():
    """Verify price bounds verification correctly accepts valid and rejects arbitrageable prices."""
    F, K, D = 100.0, 90.0, 0.95
    # Call bounds: [D*(F-K), D*F] = [0.95*10, 95] = [9.5, 95]
    valid_c, status, low, high = check_european_bounds(12.0, F, K, D, "call")
    assert valid_c and status == StatusCode.ACCEPTED

    # Call below intrinsic
    invalid_low, status_low, _, _ = check_european_bounds(8.0, F, K, D, "call")
    assert not invalid_low
    assert status_low == StatusCode.INVERSION_PRICE_BELOW_INTRINSIC

    # Call above upper bound
    invalid_high, status_high, _, _ = check_european_bounds(100.0, F, K, D, "call")
    assert not invalid_high
    assert status_high == StatusCode.INVERSION_PRICE_ABOVE_UPPER_BOUND


def test_volatility_monotonicity():
    """Verify Black forward price is strictly monotonically increasing in volatility."""
    F, K, D, T = 100.0, 105.0, 0.96, 0.75
    vols = np.linspace(0.01, 1.5, 50)
    prices = [black_forward_price(F, K, D, T, v, "call") for v in vols]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


def test_near_zero_volatility_limit():
    """Verify that as volatility -> 0, price converges to discounted intrinsic value."""
    F, D, T = 100.0, 0.95, 0.5
    # ITM Call
    p_itm = black_forward_price(F, 80.0, D, T, 1e-6, "call")
    assert abs(p_itm - D * 20.0) < 1e-4

    # OTM Call
    p_otm = black_forward_price(F, 120.0, D, T, 1e-6, "call")
    assert abs(p_otm - 0.0) < 1e-4


def test_invalid_input_exceptions():
    """Verify non-positive inputs raise InvalidInputError."""
    with pytest.raises(InvalidInputError):
        black_forward_price(forward=-100.0, strike=100.0, discount_factor=1.0, time_to_maturity=1.0, volatility=0.2)
    with pytest.raises(InvalidInputError):
        black_forward_price(forward=100.0, strike=100.0, discount_factor=1.0, time_to_maturity=-1.0, volatility=0.2)
    with pytest.raises(InvalidInputError):
        black_forward_price(forward=100.0, strike=100.0, discount_factor=1.0, time_to_maturity=1.0, volatility=-0.2)
