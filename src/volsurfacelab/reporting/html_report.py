"""Standalone HTML report generator for VolSurfaceLab runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from volsurfacelab.domain.models import (
    AdmissibilityReport,
    CalibrationResult,
    DensityResult,
    GridDiagnostics,
    LocalVolResult,
    OptionQuote,
    QuoteCleaningReport,
)
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.visualization.figures import (
    plot_calendar_heatmap,
    plot_density_and_cdf,
    plot_durrleman_g_heatmap,
    plot_fitted_slices,
    plot_implied_vol_surface,
    plot_local_vol_surface,
)


def generate_html_report(
    output_path: str | Path,
    run_id: str,
    provenance: Dict[str, Any],
    quotes: List[OptionQuote],
    cleaning_report: QuoteCleaningReport,
    surface: SSVISurface,
    calibration_result: CalibrationResult,
    admissibility_report: AdmissibilityReport,
    grid_diagnostics: GridDiagnostics,
    density_result: Optional[DensityResult],
    local_vol_result: Optional[LocalVolResult],
    ablation_summary: Optional[Dict[str, Any]] = None,
    is_synthetic: bool = True,
) -> Path:
    """Generate and write a comprehensive standalone HTML report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    k_grid = np.linspace(grid_diagnostics.k_min, grid_diagnostics.k_max, 41)
    T_grid = np.linspace(grid_diagnostics.T_min, grid_diagnostics.T_max, 25)

    # Generate Plotly HTML divs
    fig_iv = plot_implied_vol_surface(surface, k_grid, T_grid)
    html_iv = fig_iv.to_html(full_html=False, include_plotlyjs="cdn")

    fig_slices = plot_fitted_slices(quotes, surface)
    html_slices = fig_slices.to_html(full_html=False, include_plotlyjs=False)

    fig_g = plot_durrleman_g_heatmap(surface, k_grid, T_grid)
    html_g = fig_g.to_html(full_html=False, include_plotlyjs=False)

    fig_cal = plot_calendar_heatmap(surface, k_grid, T_grid)
    html_cal = fig_cal.to_html(full_html=False, include_plotlyjs=False)

    html_density = ""
    if density_result is not None:
        fig_dens = plot_density_and_cdf(density_result)
        html_density = fig_dens.to_html(full_html=False, include_plotlyjs=False)

    html_local_vol = ""
    if local_vol_result is not None:
        fig_lv = plot_local_vol_surface(local_vol_result)
        html_local_vol = fig_lv.to_html(full_html=False, include_plotlyjs=False)

    # Status badges
    cert_badge = (
        '<span style="background-color: #2ca02c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">CERTIFIED</span>'
        if admissibility_report.details.get("is_certified")
        else '<span style="background-color: #d62728; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">FAILED</span>'
    )
    grid_badge = (
        '<span style="background-color: #2ca02c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">PASSED</span>'
        if grid_diagnostics.grid_status.value == "GRID_DIAGNOSTIC_PASSED"
        else '<span style="background-color: #ff7f0e; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">WARNING</span>'
    )
    data_label = "SYNTHETIC MARKET (Deterministic Admissible SSVI)" if is_synthetic else "EXTERNAL MARKET CSV"

    # Ablation rows
    ablation_html = ""
    if ablation_summary and "comparison_table" in ablation_summary:
        ablation_html = "<h3>Model Ablation & Baseline Comparison</h3><table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; margin-bottom: 20px;'>"
        ablation_html += "<tr style='background-color: #f2f2f2;'><th>Model</th><th>Price RMSE ($)</th><th>Price MAE ($)</th><th>Inside Spread %</th><th>Calendar Guarantee</th><th>Theorem Status</th><th>Comment</th></tr>"
        for row in ablation_summary["comparison_table"]:
            ablation_html += f"<tr><td><b>{row['model']}</b></td><td>{row['price_rmse']:.4f}</td><td>{row['price_mae']:.4f}</td><td>{row['pct_inside_spread']:.1f}%</td><td>{row['calendar_guarantee']}</td><td>{row['theorem_status']}</td><td>{row['comment']}</td></tr>"
        ablation_html += "</table>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VolSurfaceLab Run Report — {run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #24292e; margin: 0; padding: 20px 40px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px 40px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #1a202c; border-bottom: 1px solid #eaecef; padding-bottom: 8px; }}
        .summary-box {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ flex: 1; padding: 15px; border-radius: 6px; border: 1px solid #e1e4e8; background: #fafbfc; }}
        .card h4 {{ margin: 0 0 10px 0; color: #586069; font-size: 13px; text-transform: uppercase; }}
        .card .val {{ font-size: 20px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #e1e4e8; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f6f8fa; }}
        .plot-container {{ margin: 25px 0; padding: 15px; border: 1px solid #e1e4e8; border-radius: 6px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>VolSurfaceLab — Calibration & Arbitrage Audit Report</h1>
    <p><b>Run ID:</b> {run_id} | <b>Timestamp (UTC):</b> {provenance.get('timestamp_utc', 'N/A')} | <b>Data Origin:</b> {data_label}</p>

    <div class="summary-box">
        <div class="card">
            <h4>SSVI Theorem Certificate</h4>
            <div class="val">{cert_badge}</div>
            <p style="font-size: 12px; margin: 5px 0 0 0; color: #586069;">Gatheral-Jacquier Sufficient Conditions</p>
        </div>
        <div class="card">
            <h4>Grid Arbitrage Diagnostics</h4>
            <div class="val">{grid_badge}</div>
            <p style="font-size: 12px; margin: 5px 0 0 0; color: #586069;">min g = {grid_diagnostics.min_g:.4f}, min w_T = {grid_diagnostics.min_w_T:.4f}</p>
        </div>
        <div class="card">
            <h4>Price Fit Quality</h4>
            <div class="val">${calibration_result.price_rmse:.4f}</div>
            <p style="font-size: 12px; margin: 5px 0 0 0; color: #586069;">RMSE | {calibration_result.pct_inside_spread:.1f}% inside spread</p>
        </div>
        <div class="card">
            <h4>Implied Volatility Fit</h4>
            <div class="val">{calibration_result.iv_rmse*100:.2f}%</div>
            <p style="font-size: 12px; margin: 5px 0 0 0; color: #586069;">IV RMSE ({calibration_result.iv_rmse:.4f} vol pts)</p>
        </div>
    </div>

    <h2>1. Calibrated SSVI Parameters & Admissibility Margins</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Fitted Value</th>
            <th>Admissibility Condition</th>
            <th>Safety Cap / Bound</th>
            <th>Margin</th>
        </tr>
        <tr>
            <td>Skew <b>&rho;</b></td>
            <td>{calibration_result.parameters.rho:.4f}</td>
            <td>|&rho;| &lt; 1</td>
            <td>(-1, 1)</td>
            <td>{admissibility_report.rho_margin:.4f}</td>
        </tr>
        <tr>
            <td>Power Curvature <b>&gamma;</b></td>
            <td>{calibration_result.parameters.gamma:.4f}</td>
            <td>0 &lt; &gamma; &le; 0.5</td>
            <td>(0, 0.5]</td>
            <td>{admissibility_report.gamma_margin:.4f}</td>
        </tr>
        <tr>
            <td>Global Scale <b>&eta;</b></td>
            <td>{calibration_result.parameters.eta:.4f}</td>
            <td>&eta; &le; &eta;<sub>cap</sub>(&rho;, &gamma;)</td>
            <td>{admissibility_report.eta_cap:.4f}</td>
            <td>{admissibility_report.eta_margin:.4f}</td>
        </tr>
        <tr>
            <td>Condition 1 (&theta;&phi;)</td>
            <td>{admissibility_report.cond1_value:.4f}</td>
            <td>&theta;&phi; &times; (1 + |&rho;|) &lt; 4</td>
            <td>4.0000</td>
            <td>{admissibility_report.cond1_margin:.4f}</td>
        </tr>
        <tr>
            <td>Condition 2 (&theta;&phi;<sup>2</sup>)</td>
            <td>{admissibility_report.cond2_value:.4f}</td>
            <td>&theta;&phi;<sup>2</sup> &times; (1 + |&rho;|) &le; 4</td>
            <td>4.0000</td>
            <td>{admissibility_report.cond2_margin:.4f}</td>
        </tr>
    </table>

    <p><b>ATM Total Variance Knots (&theta;):</b> {', '.join([f'T={T:.2f}Y: &theta;={th:.4f}' for T, th in zip(calibration_result.parameters.expiries, calibration_result.parameters.theta_knots)])}</p>

    <h2>2. Market Data Quality & Ingestion Accountability</h2>
    <table>
        <tr><th>Metric</th><th>Count</th></tr>
        <tr><td>Raw Ingested Quotes</td><td>{cleaning_report.raw_count}</td></tr>
        <tr><td>Accepted European Quotes</td><td>{cleaning_report.accepted_count}</td></tr>
        <tr><td>Rejected Quotes</td><td>{cleaning_report.rejected_count}</td></tr>
    </table>
    <p><b>Rejections Breakdown:</b> {cleaning_report.rejection_breakdown or 'None'}</p>

    <h2>3. Calibrated Volatility Surface & Smile Slices</h2>
    <div class="plot-container">{html_iv}</div>
    <div class="plot-container">{html_slices}</div>

    <h2>4. Independent No-Arbitrage Diagnostics (Dense Grid)</h2>
    <p>Evaluated on dense grid ({grid_diagnostics.n_k} strikes &times; {grid_diagnostics.n_T} maturities).</p>
    <div class="plot-container">{html_g}</div>
    <div class="plot-container">{html_cal}</div>

    <h2>5. Breeden-Litzenberger Risk-Neutral Density</h2>
    <div class="plot-container">{html_density}</div>

    <h2>6. Dupire Local Volatility Surface</h2>
    <div class="plot-container">{html_local_vol}</div>

    <h2>7. Research Baseline & Ablation Comparison</h2>
    {ablation_html}

    <h2>8. Provenance & Software Environment</h2>
    <table>
        <tr><th>Key</th><th>Value</th></tr>
        <tr><td>Python Version</td><td>{provenance.get('python_version', 'N/A')}</td></tr>
        <tr><td>NumPy Version</td><td>{provenance.get('numpy_version', 'N/A')}</td></tr>
        <tr><td>SciPy Version</td><td>{provenance.get('scipy_version', 'N/A')}</td></tr>
        <tr><td>Git Commit</td><td>{provenance.get('git_commit', 'N/A')}</td></tr>
        <tr><td>Config Hash</td><td>{provenance.get('config_hash', 'N/A')}</td></tr>
    </table>
</div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return path
