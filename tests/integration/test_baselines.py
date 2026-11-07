"""Integration tests for model baselines, ablations, and synthetic surface recovery."""

import numpy as np
import pytest

from volsurfacelab.config import CalibrationConfig, SyntheticGeneratorConfig
from volsurfacelab.data.synthetic import generate_synthetic_market
from volsurfacelab.evaluation.baselines import fit_flat_black, run_model_ablation
from volsurfacelab.evaluation.synthetic_recovery import evaluate_synthetic_recovery
from volsurfacelab.surface.calibrator import calibrate_ssvi
from volsurfacelab.surface.raw_svi_baseline import fit_independent_raw_svi


def test_baseline_and_ablation_pipeline():
    """Verify that all three models (SSVI, Raw SVI, Flat Black) run on identical data and emit comparison metrics."""
    gen_cfg = SyntheticGeneratorConfig(
        seed=42,
        price_noise_std=0.01,
        relative_spread=0.02,
        expiries=[0.25, 0.5, 1.0],
        n_strikes_per_expiry=12,
    )
    quotes, snapshot, ground_truth = generate_synthetic_market(gen_cfg)

    # 1. Fit SSVI
    calib_cfg = CalibrationConfig(max_nfev=1000, multi_start_seeds=[42])
    ssvi_surface, ssvi_result = calibrate_ssvi(quotes, calib_cfg)

    # 2. Fit Flat Black
    flat_black = fit_flat_black(quotes)
    assert 0.05 < flat_black.sigma_flat < 0.80

    # 3. Fit Raw SVI
    raw_svi = fit_independent_raw_svi(quotes)
    assert len(raw_svi.slices) == 3

    # 4. Run ablation comparison
    ablation = run_model_ablation(quotes, ssvi_surface, ssvi_result)
    table = ablation["comparison_table"]

    assert len(table) == 3
    model_names = [row["model"] for row in table]
    assert "SSVI (Constrained)" in model_names
    assert "Independent Raw SVI" in model_names
    assert "Flat Black Volatility" in model_names

    # SSVI must fit smile better than flat volatility
    ssvi_row = next(r for r in table if "SSVI" in r["model"])
    flat_row = next(r for r in table if "Flat" in r["model"])
    assert ssvi_row["price_rmse"] < flat_row["price_rmse"]

    # 5. Synthetic recovery evaluation
    recovery = evaluate_synthetic_recovery(ssvi_surface, ssvi_result, ground_truth)
    assert recovery["surface_total_variance_rmse"] < 0.02
    assert recovery["iv_recovery_rmse"] < 0.02
