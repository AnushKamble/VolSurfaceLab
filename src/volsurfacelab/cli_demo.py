"""CLI pipeline orchestration workflows for demo, calibrate, verify, report, and benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

from volsurfacelab.arbitrage.report import (
    run_grid_diagnostics,
    run_market_quote_diagnostics,
)
from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.config import RunConfig
from volsurfacelab.data.csv_provider import load_quotes_from_csv
from volsurfacelab.data.provenance import capture_environment_metadata
from volsurfacelab.data.quality import filter_quotes
from volsurfacelab.data.synthetic import generate_synthetic_market
from volsurfacelab.density.breeden_litzenberger import extract_risk_neutral_density
from volsurfacelab.evaluation.baselines import run_model_ablation
from volsurfacelab.evaluation.synthetic_recovery import evaluate_synthetic_recovery
from volsurfacelab.local_vol.dupire import evaluate_dupire_local_volatility
from volsurfacelab.reporting.artifacts import load_run_artifacts, save_run_artifacts
from volsurfacelab.reporting.html_report import generate_html_report
from volsurfacelab.surface.calibrator import calibrate_ssvi
from volsurfacelab.surface.ssvi import SSVISurface


def run_demo_pipeline(
    seed: int = 42,
    config_path: Optional[str] = None,
    output_dir: str = "outputs/demo",
) -> int:
    """Run deterministic offline end-to-end pipeline from synthetic quotes to HTML report."""
    out_path = Path(output_dir)
    print(f"===========================================================")
    print(f"VolSurfaceLab — Deterministic Synthetic Demo (Seed: {seed})")
    print(f"===========================================================")

    # 1. Load config
    if config_path and Path(config_path).exists():
        cfg = RunConfig.from_yaml(config_path)
    else:
        cfg = RunConfig(seed=seed, run_id=f"demo_seed_{seed}")
    cfg.seed = seed
    cfg.synthetic.seed = seed

    # 2. Generate deterministic synthetic market
    print("\n[1/7] Generating deterministic synthetic market quotes...")
    quotes_raw, snapshot, ground_truth = generate_synthetic_market(cfg.synthetic)
    print(f"      Generated {len(quotes_raw)} quotes across {len(snapshot.forward_curve)} expiries.")

    # 3. Quality control & filtering
    print("[2/7] Applying market data quality filters & European gate...")
    quotes_accepted, cleaning_report = filter_quotes(
        quotes=quotes_raw,
        config=cfg.data_filter,
        market_forwards=snapshot.forward_curve,
        market_discounts=snapshot.discount_curve,
    )
    print(f"      Accepted: {cleaning_report.accepted_count} | Rejected: {cleaning_report.rejected_count}")

    # 4. Calibration
    print("[3/7] Calibrating joint multi-expiry SSVI surface (TRF least-squares)...")
    provenance = capture_environment_metadata()
    surface, calib_result = calibrate_ssvi(
        quotes=quotes_accepted,
        config=cfg.calibration,
        provenance_metadata=provenance,
    )
    print(f"      Optimization status: {calib_result.optimizer_status}")
    print(f"      Fitted rho: {calib_result.parameters.rho:.4f}")
    print(f"      Fitted eta: {calib_result.parameters.eta:.4f}")
    print(f"      Fitted gamma: {calib_result.parameters.gamma:.4f}")
    print(f"      Price RMSE: ${calib_result.price_rmse:.4f} | IV RMSE: {calib_result.iv_rmse*100:.2f}%")
    print(f"      Inside spread: {calib_result.pct_inside_spread:.1f}%")

    # 5. No-arbitrage certification & grid diagnostics
    print("[4/7] Checking theorem admissibility and running grid diagnostics...")
    cert_report = generate_ssvi_certificate(
        calib_result.parameters,
        safety_margin=cfg.calibration.safety_margin,
    )
    grid_diag = run_grid_diagnostics(surface, cfg.grid)
    print(f"      Theorem Certificate: {cert_report.theorem_status.value}")
    print(f"      Grid Diagnostics: {grid_diag.grid_status.value} (min g: {grid_diag.min_g:.4f}, min w_T: {grid_diag.min_w_T:.4f})")

    # 6. Risk-neutral density & local volatility
    print("[5/7] Extracting risk-neutral density & computing Dupire local volatility...")
    mid_expiry = surface.parameters.expiries[len(surface.parameters.expiries) // 2]
    f_mid = snapshot.forward_curve.get(mid_expiry, 100.0)
    d_mid = snapshot.discount_curve.get(mid_expiry, 1.0)
    density_res = extract_risk_neutral_density(surface, mid_expiry, f_mid, d_mid, cfg.density)
    print(f"      Density (T={mid_expiry:.2f}Y): Total Mass = {density_res.total_mass:.4f}, Relative Moment Err = {density_res.relative_moment_error*100:.2f}%")

    k_grid = np.linspace(cfg.grid.k_min, cfg.grid.k_max, 35)
    T_grid = np.linspace(cfg.grid.T_min, cfg.grid.T_max, 21)
    local_vol_res = evaluate_dupire_local_volatility(surface, k_grid, T_grid, f_mid, cfg.local_vol)
    print(f"      Local Volatility: {local_vol_res.valid_fraction*100:.1f}% valid cells on {local_vol_res.total_cells} grid points.")

    # 7. Model ablation comparison & ground truth recovery
    print("[6/7] Running model ablation benchmark (SSVI vs Raw SVI vs Flat Black)...")
    ablation = run_model_ablation(quotes_accepted, surface, calib_result, cfg.grid)
    recovery = evaluate_synthetic_recovery(surface, calib_result, ground_truth)
    print(f"      True surface recovery IV RMSE: {recovery['iv_recovery_rmse']*100:.2f}%")

    # 8. Save all artefacts & report
    print("[7/7] Persisting artefacts and generating standalone HTML report...")
    saved_dir = save_run_artifacts(
        output_dir=out_path,
        config=cfg,
        provenance=provenance,
        quotes=quotes_accepted,
        cleaning_report=cleaning_report,
        surface=surface,
        calibration_result=calib_result,
        admissibility_report=cert_report,
        grid_diagnostics=grid_diag,
        density_result=density_res,
        local_vol_result=local_vol_res,
        ablation_summary=ablation,
        is_synthetic=True,
    )
    print(f"\nSUCCESS: Run artefacts and HTML report saved to: {saved_dir.resolve()}")
    print(f"Report: file://{saved_dir.resolve()}/report.html")

    return 0


def run_calibrate_pipeline(
    input_csv: str,
    config_path: str = "configs/example_market_csv.yaml",
    output_dir: str = "outputs/calibration",
) -> int:
    """Calibrate SSVI to a provided CSV dataset."""
    out_path = Path(output_dir)
    print(f"Calibrating SSVI from CSV: {input_csv}")

    cfg = RunConfig.from_yaml(config_path) if Path(config_path).exists() else RunConfig()
    quotes_raw, file_hash = load_quotes_from_csv(input_csv)
    print(f"Loaded {len(quotes_raw)} quotes (SHA-256: {file_hash[:12]}...)")

    quotes_accepted, cleaning_report = filter_quotes(quotes_raw, cfg.data_filter)
    print(f"Accepted: {cleaning_report.accepted_count} | Rejected: {cleaning_report.rejected_count}")

    provenance = capture_environment_metadata()
    provenance["input_file_hash"] = file_hash
    surface, calib_result = calibrate_ssvi(quotes_accepted, cfg.calibration, provenance)

    cert_report = generate_ssvi_certificate(calib_result.parameters, cfg.calibration.safety_margin)
    grid_diag = run_grid_diagnostics(surface, cfg.grid)

    mid_exp = surface.parameters.expiries[len(surface.parameters.expiries) // 2]
    density_res = extract_risk_neutral_density(surface, mid_exp, quotes_accepted[0].forward or 100.0, quotes_accepted[0].discount_factor or 1.0)
    k_grid = np.linspace(-1.0, 1.0, 31)
    T_grid = np.linspace(surface.theta_curve.T_min, surface.theta_curve.T_max, 15)
    local_vol_res = evaluate_dupire_local_volatility(surface, k_grid, T_grid, quotes_accepted[0].forward or 100.0)

    save_run_artifacts(
        output_dir=out_path,
        config=cfg,
        provenance=provenance,
        quotes=quotes_accepted,
        cleaning_report=cleaning_report,
        surface=surface,
        calibration_result=calib_result,
        admissibility_report=cert_report,
        grid_diagnostics=grid_diag,
        density_result=density_res,
        local_vol_result=local_vol_res,
        is_synthetic=False,
    )
    print(f"Calibration finished. Outputs saved to {out_path}")
    return 0


def run_verify_pipeline(
    artifact_path: str,
    output_dir: Optional[str] = None,
) -> int:
    """Verify saved calibration artefact."""
    artifact = load_run_artifacts(artifact_path)
    print(f"Verifying Run Artifact: {artifact.run_id}")
    surface = SSVISurface(artifact.parameters)

    cert = generate_ssvi_certificate(artifact.parameters)
    grid = run_grid_diagnostics(surface)

    print(f"Theorem Status: {cert.theorem_status.value}")
    print(f"  rho: {artifact.parameters.rho:.4f} (margin: {cert.rho_margin:.4f})")
    print(f"  eta: {artifact.parameters.eta:.4f} (cap: {cert.eta_cap:.4f}, margin: {cert.eta_margin:.4f})")
    print(f"  gamma: {artifact.parameters.gamma:.4f} (margin: {cert.gamma_margin:.4f})")
    print(f"Grid Status: {grid.grid_status.value}")
    print(f"  min g: {grid.min_g:.6f} at (k={grid.worst_g_coordinate[0]:.2f}, T={grid.worst_g_coordinate[1]:.2f})")
    print(f"  min w_T: {grid.min_w_T:.6f} at (k={grid.worst_w_T_coordinate[0]:.2f}, T={grid.worst_w_T_coordinate[1]:.2f})")
    return 0


def run_report_pipeline(
    artifact_path: str,
    output_dir: str = "outputs/report",
) -> int:
    """Generate standalone report from saved artefact."""
    artifact = load_run_artifacts(artifact_path)
    surface = SSVISurface(artifact.parameters)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.html"

    generate_html_report(
        output_path=report_file,
        run_id=artifact.run_id,
        provenance=artifact.environment,
        quotes=[],
        cleaning_report=artifact.quote_cleaning_report,
        surface=surface,
        calibration_result=artifact.calibration_result,
        admissibility_report=artifact.admissibility_report,
        grid_diagnostics=artifact.grid_diagnostics,
        density_result=None,
        local_vol_result=None,
    )
    print(f"Report generated at: {report_file.resolve()}")
    return 0


def run_benchmark_pipeline(
    config_path: str = "configs/synthetic_default.yaml",
    output_dir: str = "outputs/benchmark",
) -> int:
    """Run model ablation benchmark comparing SSVI, Raw SVI, and Flat Black."""
    cfg = RunConfig.from_yaml(config_path) if Path(config_path).exists() else RunConfig()
    quotes, snap, _ = generate_synthetic_market(cfg.synthetic)
    quotes_acc, _ = filter_quotes(quotes, cfg.data_filter)
    surface, calib_res = calibrate_ssvi(quotes_acc, cfg.calibration)
    ablation = run_model_ablation(quotes_acc, surface, calib_res, cfg.grid)

    print("\n=== Model Ablation Benchmark Results ===")
    for row in ablation["comparison_table"]:
        print(f"\nModel: {row['model']}")
        print(f"  Price RMSE: ${row['price_rmse']:.4f} | Price MAE: ${row['price_mae']:.4f}")
        print(f"  Inside Spread: {row['pct_inside_spread']:.1f}%")
        print(f"  Parameters: {row['num_parameters']}")
        print(f"  Calendar: {row['calendar_guarantee']}")
        print(f"  Theorem: {row['theorem_status']}")

    return 0
