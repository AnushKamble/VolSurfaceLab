"""Artefact serialization, persistent run bundles, and deserialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from volsurfacelab.config import RunConfig
from volsurfacelab.domain.models import (
    AdmissibilityReport,
    CalibrationResult,
    DensityResult,
    GridDiagnostics,
    LocalVolResult,
    OptionQuote,
    QuoteCleaningReport,
    RunArtifact,
    SSVIParameters,
)
from volsurfacelab.domain.schemas import model_from_dict, model_to_dict
from volsurfacelab.reporting.html_report import generate_html_report
from volsurfacelab.surface.ssvi import SSVISurface


def save_run_artifacts(
    output_dir: str | Path,
    config: RunConfig,
    provenance: Dict[str, Any],
    quotes: List[OptionQuote],
    cleaning_report: QuoteCleaningReport,
    surface: SSVISurface,
    calibration_result: CalibrationResult,
    admissibility_report: AdmissibilityReport,
    grid_diagnostics: GridDiagnostics,
    density_result: Optional[DensityResult] = None,
    local_vol_result: Optional[LocalVolResult] = None,
    ablation_summary: Optional[Dict[str, Any]] = None,
    is_synthetic: bool = True,
) -> Path:
    """Save all structured JSON, CSV, and HTML run artefacts into output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config
    config.to_yaml(out_dir / "run_config.yaml")

    # 2. Provenance
    provenance["config_hash"] = config.compute_hash()
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)

    # 3. Quote cleaning report
    with open(out_dir / "quote_cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(model_to_dict(cleaning_report), f, indent=2)

    # 4. Calibration parameters & status
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(model_to_dict(calibration_result), f, indent=2)

    # 5. Admissibility report
    with open(out_dir / "admissibility_report.json", "w", encoding="utf-8") as f:
        json.dump(model_to_dict(admissibility_report), f, indent=2)

    # 6. Grid diagnostics
    with open(out_dir / "grid_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(model_to_dict(grid_diagnostics), f, indent=2)

    # 7. Fit metrics
    fit_metrics = {
        "price_rmse": calibration_result.price_rmse,
        "price_mae": calibration_result.price_mae,
        "iv_rmse": calibration_result.iv_rmse,
        "iv_mae": calibration_result.iv_mae,
        "pct_inside_spread": calibration_result.pct_inside_spread,
        "residual_percentiles": calibration_result.residual_percentiles,
        "cost": calibration_result.cost,
        "evaluations": calibration_result.evaluations,
    }
    with open(out_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    # 8. Density CSV
    if density_result is not None:
        dens_df = pd.DataFrame({
            "strike": density_result.strikes,
            "log_moneyness": density_result.log_moneyness,
            "density_q": density_result.density_q,
            "state_price_density": density_result.state_price_density,
        })
        dens_df.to_csv(out_dir / f"density_{density_result.expiry:.2f}.csv", index=False)

    # 9. Local Vol Grid CSV
    if local_vol_result is not None:
        rows = []
        for i, T in enumerate(local_vol_result.expiries):
            for j, k in enumerate(local_vol_result.log_moneyness_grid):
                rows.append({
                    "expiry": T,
                    "log_moneyness": k,
                    "strike": local_vol_result.strikes_grid[i][j],
                    "local_variance": local_vol_result.local_variance[i][j],
                    "local_volatility": local_vol_result.local_volatility[i][j],
                    "is_valid": local_vol_result.validity_mask[i][j],
                    "reason_code": local_vol_result.reason_codes[i][j],
                })
        lv_df = pd.DataFrame(rows)
        lv_df.to_csv(out_dir / "local_vol_grid.csv", index=False)

    # 10. RunArtifact summary JSON
    artifact = RunArtifact(
        run_id=config.run_id,
        timestamp=provenance.get("timestamp_utc", ""),
        config_hash=config.compute_hash(),
        environment=provenance,
        parameters=calibration_result.parameters,
        calibration_result=calibration_result,
        admissibility_report=admissibility_report,
        grid_diagnostics=grid_diagnostics,
        quote_cleaning_report=cleaning_report,
        fit_metrics=fit_metrics,
    )
    with open(out_dir / "run_artifact.json", "w", encoding="utf-8") as f:
        json.dump(model_to_dict(artifact), f, indent=2)

    # 11. HTML Report
    generate_html_report(
        output_path=out_dir / "report.html",
        run_id=config.run_id,
        provenance=provenance,
        quotes=quotes,
        cleaning_report=cleaning_report,
        surface=surface,
        calibration_result=calibration_result,
        admissibility_report=admissibility_report,
        grid_diagnostics=grid_diagnostics,
        density_result=density_result,
        local_vol_result=local_vol_result,
        ablation_summary=ablation_summary,
        is_synthetic=is_synthetic,
    )

    return out_dir


def load_run_artifacts(artifact_path: str | Path) -> RunArtifact:
    """Load RunArtifact from a JSON file or directory."""
    path = Path(artifact_path)
    if path.is_dir():
        file_path = path / "run_artifact.json"
        if not file_path.exists():
            file_path = path / "calibration.json"
    else:
        file_path = path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "run_id" in data and "parameters" in data:
        return model_from_dict(RunArtifact, data)
    elif "parameters" in data:
        # Reconstruct RunArtifact from calibration.json
        params = model_from_dict(SSVIParameters, data["parameters"])
        calib_res = model_from_dict(CalibrationResult, data)
        from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
        admiss = generate_ssvi_certificate(params)
        surface = SSVISurface(params)
        from volsurfacelab.arbitrage.report import run_grid_diagnostics
        grid = run_grid_diagnostics(surface)
        clean = QuoteCleaningReport(raw_count=0, accepted_count=0, rejected_count=0)
        return RunArtifact(
            run_id="loaded_artifact",
            timestamp="",
            config_hash="",
            parameters=params,
            calibration_result=calib_res,
            admissibility_report=admiss,
            grid_diagnostics=grid,
            quote_cleaning_report=clean,
            fit_metrics={},
        )
    else:
        raise ValueError(f"Unrecognized artifact structure in {file_path}")
