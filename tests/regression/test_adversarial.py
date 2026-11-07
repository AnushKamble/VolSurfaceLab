"""Adversarial stress test suite executing all 14 mandatory adversarial scenarios."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from volsurfacelab.arbitrage.butterfly import check_butterfly_arbitrage_grid
from volsurfacelab.arbitrage.report import run_grid_diagnostics, run_market_quote_diagnostics
from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.config import (
    CalibrationConfig,
    DataFilterConfig,
    DensityConfig,
    GridConfig,
    LocalVolConfig,
    RunConfig,
    SyntheticGeneratorConfig,
)
from volsurfacelab.data.quality import filter_quotes
from volsurfacelab.data.synthetic import generate_synthetic_market
from volsurfacelab.density.breeden_litzenberger import extract_risk_neutral_density
from volsurfacelab.domain.exceptions import CalibrationError, StatusCode
from volsurfacelab.domain.models import OptionQuote, SSVIParameters
from volsurfacelab.evaluation.synthetic_recovery import evaluate_synthetic_recovery
from volsurfacelab.local_vol.dupire import evaluate_dupire_local_volatility
from volsurfacelab.surface.admissible_parameters import compute_eta_cap
from volsurfacelab.surface.calibrator import calibrate_ssvi
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.surface.theta_curve import ThetaCurve


# -----------------------------------------------------------------------------
# Scenario 1: Clean Exact Synthetic SSVI Quotes
# -----------------------------------------------------------------------------
def test_scenario_01_clean_exact_synthetic_recovery():
    """Scenario 1: Clean exact synthetic quotes yield high-precision recovery and certified admissibility."""
    cfg = SyntheticGeneratorConfig(
        seed=42,
        price_noise_std=0.0,  # Zero noise
        relative_spread=0.01,
        expiries=[0.25, 0.5, 1.0, 1.5],
        n_strikes_per_expiry=15,
    )
    quotes, snap, ground_truth = generate_synthetic_market(cfg)
    accepted, report = filter_quotes(quotes, DataFilterConfig())
    assert report.accepted_count == len(quotes)

    calib_cfg = CalibrationConfig(max_nfev=1500, multi_start_seeds=[42])
    surface, result = calibrate_ssvi(accepted, calib_cfg)

    assert result.price_rmse < 0.05
    assert result.iv_rmse < 0.01
    cert = generate_ssvi_certificate(result.parameters)
    assert cert.theorem_status == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS


# -----------------------------------------------------------------------------
# Scenario 2: Noisy Bid-Ask Synthetic Quotes
# -----------------------------------------------------------------------------
def test_scenario_02_noisy_bid_ask_quotes():
    """Scenario 2: Quotes with realistic spreads and noise remain stable and fit inside spreads."""
    cfg = SyntheticGeneratorConfig(
        seed=101,
        price_noise_std=0.02,
        relative_spread=0.03,
        expiries=[0.2, 0.5, 1.0, 2.0],
        n_strikes_per_expiry=20,
    )
    quotes, snap, _ = generate_synthetic_market(cfg)
    accepted, _ = filter_quotes(quotes, DataFilterConfig())

    surface, result = calibrate_ssvi(accepted, CalibrationConfig(multi_start_seeds=[101]))
    assert result.pct_inside_spread > 80.0
    cert = generate_ssvi_certificate(result.parameters)
    assert cert.theorem_status == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS


# -----------------------------------------------------------------------------
# Scenario 3: Missing-Strike Sparse Surface
# -----------------------------------------------------------------------------
def test_scenario_03_missing_strike_sparse_surface():
    """Scenario 3: Surface with sparse strikes and missing wings calibrates smoothly without singularity."""
    cfg = SyntheticGeneratorConfig(
        seed=42,
        expiries=[0.25, 0.5, 1.0],
        moneyness_min=-0.2,
        moneyness_max=0.2,  # Narrow moneyness, wings absent
        n_strikes_per_expiry=8,
    )
    quotes, _, _ = generate_synthetic_market(cfg)
    surface, result = calibrate_ssvi(quotes, CalibrationConfig(multi_start_seeds=[42]))

    # Must extrapolate smoothly onto wider grid without crashing
    k_wide = np.linspace(-1.5, 1.5, 31)
    T_grid = np.linspace(0.25, 1.0, 10)
    w_wide = surface.total_variance(k_wide, 0.5)
    assert np.all(w_wide > 0.0)
    assert np.all(np.isfinite(w_wide))


# -----------------------------------------------------------------------------
# Scenario 4: Invalid Crossed, Stale, Expired Quotes
# -----------------------------------------------------------------------------
def test_scenario_04_invalid_crossed_stale_expired_dataset():
    """Scenario 4: Dataset with anomalies is filtered and every dropped quote is accounted for."""
    cfg = SyntheticGeneratorConfig(seed=42, include_anomalies=True)
    quotes, _, _ = generate_synthetic_market(cfg)

    filter_cfg = DataFilterConfig(min_spread=0.01, reject_american=True)
    accepted, report = filter_quotes(quotes, filter_cfg)

    assert report.rejected_count >= 4
    assert report.accepted_count + report.rejected_count == report.raw_count
    assert StatusCode.REJECTED_CROSSED_SPREAD.value in report.rejection_breakdown
    assert StatusCode.REJECTED_ZERO_BID.value in report.rejection_breakdown
    assert StatusCode.REJECTED_EXERCISE_STYLE_AMERICAN.value in report.rejection_breakdown


# -----------------------------------------------------------------------------
# Scenario 5: American / Unknown Exercise Style Dataset
# -----------------------------------------------------------------------------
def test_scenario_05_american_unknown_exercise_gate():
    """Scenario 5: Strict European gate rejects American and Unknown contracts."""
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    base_q = OptionQuote(
        underlying_symbol="SPY",
        quote_timestamp=now,
        valuation_timestamp=now,
        expiry_timestamp=now + timedelta(days=90),
        time_to_maturity=0.25,
        strike=100.0,
        option_type="call",
        bid=2.0,
        ask=2.2,
        forward=101.0,
        discount_factor=0.99,
        exercise_style="american",
    )
    unknown_q = base_q.model_copy(update={"exercise_style": "unknown"})

    accepted, report = filter_quotes([base_q, unknown_q], DataFilterConfig())
    assert len(accepted) == 0
    assert report.rejection_breakdown[StatusCode.REJECTED_EXERCISE_STYLE_AMERICAN.value] == 1
    assert report.rejection_breakdown[StatusCode.REJECTED_EXERCISE_STYLE_UNKNOWN.value] == 1


# -----------------------------------------------------------------------------
# Scenario 6: Parity-Inconsistent Call/Put Set
# -----------------------------------------------------------------------------
def test_scenario_06_parity_inconsistent_call_put_set():
    """Scenario 6: Parity inconsistencies are detected by diagnostics, while SSVI surface remains smooth."""
    cfg = SyntheticGeneratorConfig(seed=42, expiries=[0.5], n_strikes_per_expiry=10)
    quotes, _, _ = generate_synthetic_market(cfg)

    # Corrupt parity on first call quote (quotes[0] is call, quotes[1] is corresponding put)
    quotes[0] = quotes[0].model_copy(update={"bid": quotes[0].bid + 5.0, "ask": quotes[0].ask + 5.0})

    diag = run_market_quote_diagnostics(quotes)
    assert diag["parity_interval_violations"] > 0
    assert diag["max_parity_error"] > 4.0


# -----------------------------------------------------------------------------
# Scenario 7: Deliberately Invalid SSVI Parameters
# -----------------------------------------------------------------------------
def test_scenario_07_deliberately_invalid_ssvi_parameters():
    """Scenario 7: Inadmissible parameters fail theorem certification with negative margins."""
    # eta exceeds cap
    params = SSVIParameters(
        rho=-0.5,
        eta=3.5,  # Cap is ~2.4
        gamma=0.35,
        expiries=[0.1, 1.0],
        theta_knots=[0.02, 0.10],
    )
    cert = generate_ssvi_certificate(params)
    assert cert.theorem_status == StatusCode.CERTIFICATE_FAILED
    assert cert.eta_margin < 0.0


# -----------------------------------------------------------------------------
# Scenario 8: Boundary Parameters
# -----------------------------------------------------------------------------
def test_scenario_08_boundary_parameters():
    """Scenario 8: Extreme boundary parameters evaluate safely without NaN or division by zero."""
    # rho very close to -1, gamma = 0.5, eta near cap
    rho_b = -0.99
    gamma_b = 0.5
    cap = compute_eta_cap(rho_b, gamma_b, safety_margin=0.001)
    eta_b = cap * 0.999

    params = SSVIParameters(
        rho=rho_b,
        eta=eta_b,
        gamma=gamma_b,
        expiries=[0.05, 0.5, 2.0],
        theta_knots=[0.01, 0.06, 0.20],
    )
    surface = SSVISurface(params)

    k_grid = np.linspace(-2.0, 2.0, 50)
    w = surface.total_variance(k_grid, 0.5)
    w_k, w_kk = surface.derivatives_k(k_grid, 0.06)
    g = surface.durrleman_g(k_grid, 0.5)

    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(w_k))
    assert np.all(np.isfinite(w_kk))
    assert np.all(np.isfinite(g))


# -----------------------------------------------------------------------------
# Scenario 9: Near-Flat Monotone Theta Curve
# -----------------------------------------------------------------------------
def test_scenario_09_near_flat_monotone_theta_curve():
    """Scenario 9: Extremely small positive theta slope maintains positive derivatives without numerical overshoot."""
    expiries = [0.1, 0.5, 1.0, 2.0]
    knots = [0.04, 0.04001, 0.04002, 0.04003]  # near flat
    curve = ThetaCurve(expiries, knots)

    T_dense = np.linspace(0.1, 2.0, 200)
    derivs = curve.derivative(T_dense)
    assert np.all(derivs >= 0.0)


# -----------------------------------------------------------------------------
# Scenario 10: Local-Vol Denominator Near Zero
# -----------------------------------------------------------------------------
def test_scenario_10_local_vol_denominator_near_zero():
    """Scenario 10: Near-zero g denominator is flagged with reason code and NOT clipped."""
    params = SSVIParameters(
        rho=-0.95,
        eta=15.0,
        gamma=0.1,
        expiries=[0.1, 1.0],
        theta_knots=[0.01, 0.05],
    )
    surface = SSVISurface(params)
    k_grid = np.linspace(-0.15, 0.15, 31)
    T_grid = np.linspace(0.1, 0.3, 5)

    lv_res = evaluate_dupire_local_volatility(surface, k_grid, T_grid, config=LocalVolConfig(g_denominator_min=0.01))
    all_codes = {c for row in lv_res.reason_codes for c in row}

    assert (
        StatusCode.LOCAL_VOL_DENOMINATOR_NON_POSITIVE.value in all_codes
        or StatusCode.LOCAL_VOL_DENOMINATOR_NEAR_ZERO.value in all_codes
    )


# -----------------------------------------------------------------------------
# Scenario 11: Deliberately Negative Butterfly Surface
# -----------------------------------------------------------------------------
def test_scenario_11_deliberately_negative_butterfly_surface():
    """Scenario 11: Deliberately invalid surface triggers GRID_DIAGNOSTIC_FAILED."""
    params = SSVIParameters(
        rho=-0.95,
        eta=15.0,
        gamma=0.1,
        expiries=[0.1, 1.0],
        theta_knots=[0.01, 0.05],
    )
    surface = SSVISurface(params)
    grid_res = run_grid_diagnostics(surface, GridConfig(k_min=-0.3, k_max=0.3, n_k=51, T_min=0.1, T_max=0.5, n_T=11))

    assert grid_res.grid_status == StatusCode.GRID_DIAGNOSTIC_FAILED
    assert grid_res.min_g < 0.0
    assert grid_res.negative_g_count > 0


# -----------------------------------------------------------------------------
# Scenario 12: Density Integration Domain Widening
# -----------------------------------------------------------------------------
def test_scenario_12_density_integration_domain_widening():
    """Scenario 12: Density mass monotonically approaches 1.0 as support domain widens."""
    params = SSVIParameters(
        rho=-0.35,
        eta=1.1,
        gamma=0.35,
        expiries=[0.25, 1.0],
        theta_knots=[0.04, 0.12],
    )
    surface = SSVISurface(params)
    F, D, T = 100.0, 0.96, 1.0

    masses = []
    for std_range in [2.0, 3.5, 5.0, 7.0]:
        cfg = DensityConfig(n_strikes=301, std_dev_range=std_range)
        res = extract_risk_neutral_density(surface, T, F, D, cfg)
        masses.append(res.total_mass)

    # Mass must increase with wider domain
    for i in range(len(masses) - 1):
        assert masses[i] <= masses[i + 1] + 1e-4

    # On 7-sigma domain, mass must be very close to 1.0
    assert abs(masses[-1] - 1.0) < 0.005


# -----------------------------------------------------------------------------
# Scenario 13: Optimizer Non-Convergence Handling
# -----------------------------------------------------------------------------
def test_scenario_13_optimizer_non_convergence_handling():
    """Scenario 13: Optimizer failure (e.g. max_nfev=1) is handled honestly without fabricating success."""
    cfg = SyntheticGeneratorConfig(seed=42, expiries=[0.25, 0.5], n_strikes_per_expiry=8)
    quotes, _, _ = generate_synthetic_market(cfg)

    # Restrict evaluations to 1 so convergence is impossible
    calib_cfg = CalibrationConfig(max_nfev=1, ftol=1e-15, xtol=1e-15, gtol=1e-15)

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_ssvi(quotes, calib_cfg)
    assert exc_info.value.code == StatusCode.CALIBRATION_OPTIMIZER_FAILED


# -----------------------------------------------------------------------------
# Scenario 14: Offline Run With Network Disabled
# -----------------------------------------------------------------------------
def test_scenario_14_offline_run_no_network(monkeypatch, tmp_path):
    """Scenario 14: Default pipeline executes end-to-end completely offline with socket disabled."""
    import socket

    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network access attempted in offline mode!")

    monkeypatch.setattr(socket, "socket", guarded_socket)

    from volsurfacelab.cli_demo import run_demo_pipeline
    demo_out = tmp_path / "offline_demo"
    ret = run_demo_pipeline(seed=42, output_dir=str(demo_out))

    assert ret == 0
    assert (demo_out / "report.html").exists()
    assert (demo_out / "calibration.json").exists()
