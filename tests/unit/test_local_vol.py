"""Unit tests for Dupire local volatility engine and flat-volatility identity."""

import numpy as np
import pytest

from volsurfacelab.config import LocalVolConfig
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.local_vol.dupire import (
    FlatVolSurface,
    evaluate_dupire_local_volatility,
)
from volsurfacelab.surface.ssvi import SSVISurface


def test_dupire_flat_volatility_identity():
    """BLOCKING ACCEPTANCE TEST: Verify flat volatility identity sigma_loc(k, T) == sigma_0 everywhere."""
    for sigma_0 in [0.15, 0.25, 0.40]:
        surface = FlatVolSurface(sigma_0=sigma_0)
        k_grid = np.linspace(-1.5, 1.5, 31)
        T_grid = np.linspace(0.1, 2.0, 20)

        result = evaluate_dupire_local_volatility(surface, k_grid, T_grid)

        # 1. 100% valid cells
        assert result.valid_fraction == 1.0
        assert result.valid_cells == len(k_grid) * len(T_grid)

        # 2. Local vol must match sigma_0 to tight tolerance
        for i in range(len(T_grid)):
            for j in range(len(k_grid)):
                loc_v = result.local_volatility[i][j]
                assert loc_v is not None
                assert abs(loc_v - sigma_0) < 1e-10
                assert result.reason_codes[i][j] == StatusCode.LOCAL_VOL_VALID.value


def test_dupire_certified_ssvi_surface():
    """Verify local volatility evaluation on certified SSVI surface produces smooth, valid cells."""
    params = SSVIParameters(
        rho=-0.35,
        eta=1.0,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.025, 0.06, 0.11, 0.19],
    )
    surface = SSVISurface(params)

    k_grid = np.linspace(-1.0, 1.0, 25)
    T_grid = np.linspace(0.15, 1.8, 15)

    result = evaluate_dupire_local_volatility(surface, k_grid, T_grid)

    assert result.valid_fraction > 0.95
    assert result.min_local_vol is not None
    assert result.max_local_vol is not None
    assert result.min_local_vol > 0.05
    assert result.max_local_vol < 1.0


def test_dupire_safeguards_and_no_clipping():
    """Verify that near-singular or invalid denominators are marked with reason codes and NOT silently clipped."""
    params = SSVIParameters(
        rho=-0.95,
        eta=15.0,  # Extreme parameters causing butterfly violation near ATM
        gamma=0.1,
        expiries=[0.1, 1.0],
        theta_knots=[0.01, 0.05],
    )
    surface = SSVISurface(params)

    k_grid = np.linspace(-0.2, 0.2, 21)
    T_grid = np.linspace(0.1, 0.5, 5)

    result = evaluate_dupire_local_volatility(surface, k_grid, T_grid)

    # Some cells must be invalid due to non-positive g
    assert result.valid_fraction < 1.0
    all_codes = {code for row in result.reason_codes for code in row}
    assert (
        StatusCode.LOCAL_VOL_DENOMINATOR_NON_POSITIVE.value in all_codes
        or StatusCode.LOCAL_VOL_DENOMINATOR_NEAR_ZERO.value in all_codes
    )

    # Invalid cells must remain None, NOT clipped to an arbitrary value
    for i in range(len(T_grid)):
        for j in range(len(k_grid)):
            if not result.validity_mask[i][j]:
                assert result.local_volatility[i][j] is None
                assert result.local_variance[i][j] is None
