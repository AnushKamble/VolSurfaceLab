"""Unit tests for deterministic synthetic option market generator."""

import numpy as np
import pytest

from volsurfacelab.config import SyntheticGeneratorConfig
from volsurfacelab.data.synthetic import generate_synthetic_market, SyntheticGroundTruth


def test_synthetic_market_deterministic_reproducibility():
    """Verify that generating synthetic market with the same seed yields identical outputs."""
    cfg1 = SyntheticGeneratorConfig(seed=42)
    cfg2 = SyntheticGeneratorConfig(seed=42)

    quotes1, snap1, truth1 = generate_synthetic_market(cfg1)
    quotes2, snap2, truth2 = generate_synthetic_market(cfg2)

    assert len(quotes1) == len(quotes2)
    for q1, q2 in zip(quotes1, quotes2):
        assert q1.strike == q2.strike
        assert q1.bid == q2.bid
        assert q1.ask == q2.ask
        assert q1.time_to_maturity == q2.time_to_maturity

    assert truth1 == truth2


def test_synthetic_market_seed_variation():
    """Verify different seeds generate different random price perturbations."""
    cfg1 = SyntheticGeneratorConfig(seed=42, price_noise_std=0.05)
    cfg2 = SyntheticGeneratorConfig(seed=999, price_noise_std=0.05)

    quotes1, _, _ = generate_synthetic_market(cfg1)
    quotes2, _, _ = generate_synthetic_market(cfg2)

    differences = [abs(q1.bid - q2.bid) for q1, q2 in zip(quotes1, quotes2)]
    assert np.max(differences) > 0.01


def test_synthetic_ground_truth_isolation():
    """Verify that market snapshot and quotes do not leak ground truth parameters."""
    cfg = SyntheticGeneratorConfig(seed=42)
    quotes, snapshot, ground_truth = generate_synthetic_market(cfg)

    assert isinstance(ground_truth, SyntheticGroundTruth)
    # Check that OptionQuote and MarketSnapshot have no field containing ground_truth
    assert not hasattr(snapshot, "true_rho")
    assert not hasattr(snapshot, "ground_truth")
    for q in quotes:
        assert not hasattr(q, "true_rho")
        assert not hasattr(q, "true_eta")


def test_synthetic_anomalies_injection():
    """Verify anomaly generation flag injects crossed, zero bid, and American style quotes."""
    cfg = SyntheticGeneratorConfig(seed=42, include_anomalies=True)
    quotes, _, _ = generate_synthetic_market(cfg)

    styles = {q.exercise_style for q in quotes}
    assert "american" in styles

    bids = [q.bid for q in quotes]
    assert 0.0 in bids

    crossed = any(q.ask < q.bid for q in quotes)
    assert crossed
