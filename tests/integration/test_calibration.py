"""Integration tests for constrained joint multi-expiry SSVI calibration."""

import numpy as np
import pytest

from volsurfacelab.config import CalibrationConfig, SyntheticGeneratorConfig
from volsurfacelab.data.synthetic import generate_synthetic_market
from volsurfacelab.domain.exceptions import CalibrationError
from volsurfacelab.surface.calibrator import calibrate_ssvi


def test_calibration_on_synthetic_clean_data():
    """Verify SSVI calibration on clean synthetic option quotes achieves low RMSE and certifies no-arbitrage."""
    gen_cfg = SyntheticGeneratorConfig(
        seed=42,
        price_noise_std=0.01,
        relative_spread=0.02,
        expiries=[0.2, 0.5, 1.0, 1.5],
        n_strikes_per_expiry=15,
    )
    quotes, snapshot, ground_truth = generate_synthetic_market(gen_cfg)

    calib_cfg = CalibrationConfig(
        max_nfev=1500,
        multi_start_seeds=[42, 101],
        safety_margin=0.01,
    )

    surface, result = calibrate_ssvi(quotes, calib_cfg)

    # 1. Convergence
    assert result.evaluations > 0
    assert result.cost < float("inf")

    # 2. Fit accuracy
    # Price RMSE should be under 20 cents
    assert result.price_rmse < 0.20
    # IV RMSE should be under 1.5% volatility (0.015)
    assert result.iv_rmse < 0.015
    # High percentage inside spread
    assert result.pct_inside_spread > 50.0

    # 3. Parameter admissibility
    assert abs(result.parameters.rho) < 1.0
    assert 0.0 < result.parameters.gamma <= 0.5
    assert result.parameters.eta > 0.0

    # 4. Theta monotonicity
    thetas = result.parameters.theta_knots
    assert all(thetas[i] < thetas[i + 1] for i in range(len(thetas) - 1))


def test_calibration_insufficient_data():
    """Verify calibration gracefully refuses when data is sparse or expiries are insufficient."""
    gen_cfg = SyntheticGeneratorConfig(seed=42, expiries=[0.5], n_strikes_per_expiry=5)
    quotes, _, _ = generate_synthetic_market(gen_cfg)

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_ssvi(quotes[:3])  # fewer than 6 quotes
    assert "insufficient" in str(exc_info.value).lower()

