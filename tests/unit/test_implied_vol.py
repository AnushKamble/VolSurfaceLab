"""Unit tests for robust implied volatility root solving."""

import numpy as np
import pytest

from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.pricing.black_forward import black_forward_price
from volsurfacelab.pricing.implied_vol import (
    implied_volatility_brent,
    invert_implied_volatilities,
)


def test_iv_inversion_roundtrip_standard():
    """Verify round-trip recovery of volatility from Black prices across ATM and wing strikes."""
    F = 100.0
    D = 0.98
    T = 0.5
    known_vols = [0.10, 0.25, 0.50, 0.85]
    strikes = [75.0, 90.0, 100.0, 110.0, 130.0]

    for sigma_true in known_vols:
        for K in strikes:
            for opt_type in ["call", "put"]:
                price = black_forward_price(F, K, D, T, sigma_true, opt_type)
                res = implied_volatility_brent(price, F, K, D, T, opt_type)

                assert res.success
                assert res.iv is not None
                assert abs(res.iv - sigma_true) < 1e-6
                assert res.status_code == StatusCode.INVERSION_SUCCESS
                assert abs(res.residual) < 1e-6


def test_iv_inversion_high_vol():
    """Verify inversion works cleanly for high-volatility environments (e.g. sigma = 3.5)."""
    F, K, D, T = 100.0, 120.0, 0.95, 1.0
    sigma_true = 3.5
    price = black_forward_price(F, K, D, T, sigma_true, "call")

    res = implied_volatility_brent(price, F, K, D, T, "call")
    assert res.success
    assert res.iv is not None
    assert abs(res.iv - sigma_true) < 1e-5


def test_iv_inversion_deep_itm_otm():
    """Verify inversion on deep ITM and deep OTM options."""
    F, D, T = 100.0, 0.95, 0.5
    sigma_true = 0.30

    # Deep ITM call (K=50)
    p_itm = black_forward_price(F, 50.0, D, T, sigma_true, "call")
    res_itm = implied_volatility_brent(p_itm, F, 50.0, D, T, "call")
    assert res_itm.success
    assert abs(res_itm.iv - sigma_true) < 1e-5

    # Deep OTM put (K=50)
    p_otm = black_forward_price(F, 50.0, D, T, sigma_true, "put")
    res_otm = implied_volatility_brent(p_otm, F, 50.0, D, T, "put")
    assert res_otm.success
    assert abs(res_otm.iv - sigma_true) < 1e-5


def test_iv_inversion_price_below_intrinsic():
    """Verify inversion rejects prices below intrinsic bound without inventing an IV."""
    F, K, D, T = 100.0, 80.0, 0.95, 0.5
    intrinsic = D * (F - K)  # 19.0
    invalid_price = intrinsic - 1.0  # 18.0

    res = implied_volatility_brent(invalid_price, F, K, D, T, "call")
    assert not res.success
    assert res.iv is None
    assert res.status_code == StatusCode.INVERSION_PRICE_BELOW_INTRINSIC


def test_iv_inversion_price_above_upper_bound():
    """Verify inversion rejects prices above upper theoretical bound."""
    F, K, D, T = 100.0, 100.0, 0.95, 0.5
    upper_bound = D * F  # 95.0
    invalid_price = upper_bound + 5.0

    res = implied_volatility_brent(invalid_price, F, K, D, T, "call")
    assert not res.success
    assert res.iv is None
    assert res.status_code == StatusCode.INVERSION_PRICE_ABOVE_UPPER_BOUND


def test_iv_inversion_batch_vector_helper():
    """Verify batch inversion helper works consistently across multiple contracts."""
    F = 100.0
    D = 0.97
    T = 0.25
    strikes = [90.0, 100.0, 110.0]
    types = ["call", "call", "put"]
    vols = [0.22, 0.20, 0.24]
    prices = [black_forward_price(F, strikes[i], D, T, vols[i], types[i]) for i in range(3)]

    results = invert_implied_volatilities(
        prices=prices,
        forwards=[F] * 3,
        strikes=strikes,
        discount_factors=[D] * 3,
        maturities=[T] * 3,
        option_types=types,
    )
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res.success
        assert abs(res.iv - vols[i]) < 1e-6
