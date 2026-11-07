"""Interactive Quantitative Web Workbench for VolSurfaceLab.

Arbitrage-Aware Volatility Surface, Risk-Neutral Density & Dupire Local-Risk Engine.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from volsurfacelab.arbitrage.report import run_grid_diagnostics, run_market_quote_diagnostics
from volsurfacelab.arbitrage.ssvi_certificate import generate_ssvi_certificate
from volsurfacelab.config import (
    CalibrationConfig,
    DataFilterConfig,
    DensityConfig,
    GridConfig,
    LocalVolConfig,
    SyntheticGeneratorConfig,
)
from volsurfacelab.dashboard.theme import (
    inject_custom_theme,
    render_header,
    render_kpi,
    render_theory_callout,
)
from volsurfacelab.data.csv_provider import load_quotes_from_csv
from volsurfacelab.data.provenance import capture_environment_metadata
from volsurfacelab.data.quality import filter_quotes
from volsurfacelab.data.synthetic import generate_synthetic_market
from volsurfacelab.density.breeden_litzenberger import extract_risk_neutral_density
from volsurfacelab.domain.exceptions import StatusCode
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
from volsurfacelab.evaluation.baselines import fit_flat_black, run_model_ablation
from volsurfacelab.local_vol.dupire import evaluate_dupire_local_volatility
from volsurfacelab.pricing.implied_vol import implied_volatility_brent
from volsurfacelab.reporting.artifacts import load_run_artifacts
from volsurfacelab.reporting.html_report import generate_html_report
from volsurfacelab.surface.calibrator import calibrate_ssvi
from volsurfacelab.surface.raw_svi_baseline import fit_independent_raw_svi, fit_raw_svi_slice
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.visualization.figures import (
    plot_ablation_comparison_smiles,
    plot_calendar_heatmap,
    plot_density_and_cdf,
    plot_durrleman_g_heatmap,
    plot_fitted_slices,
    plot_implied_vol_surface,
    plot_local_vol_surface,
    plot_raw_quotes_scatter,
    plot_smile_slice_with_residuals,
    plot_term_structure_curves,
    plot_theta_variance_curve,
    plot_total_variance_surface,
)


def init_session_state() -> None:
    """Initialize state on startup with a calibrated model if available, or generate a clean synthetic default."""
    if "initialized" in st.session_state and st.session_state.initialized:
        return

    st.session_state.quotes = []
    st.session_state.surface = None
    st.session_state.calibration_result = None
    st.session_state.admissibility_report = None
    st.session_state.grid_diagnostics = None
    st.session_state.density_result = None
    st.session_state.local_vol_result = None
    st.session_state.ablation_result = None
    st.session_state.cleaning_report = None
    st.session_state.dataset_source = "Synthetic Default"

    # Try loading pre-existing demo run if available
    default_artifact = Path("outputs/demo/run_artifact.json")
    default_csv = Path("data/sample_quotes.csv")

    loaded_success = False
    if default_artifact.exists():
        try:
            artifact: RunArtifact = load_run_artifacts(str(default_artifact))
            st.session_state.surface = SSVISurface(artifact.parameters)
            st.session_state.calibration_result = artifact.calibration_result
            st.session_state.admissibility_report = artifact.admissibility_report
            st.session_state.grid_diagnostics = artifact.grid_diagnostics
            st.session_state.dataset_source = "Pre-computed Run Artifact"

            # Load quotes if CSV exists
            if default_csv.exists():
                raw_quotes, _ = load_quotes_from_csv(str(default_csv))
                cleaned_quotes, clean_rep = filter_quotes(raw_quotes, DataFilterConfig())
                st.session_state.quotes = cleaned_quotes
                st.session_state.cleaning_report = clean_rep

            loaded_success = True
        except Exception:
            loaded_success = False

    if not loaded_success:
        # Generate synthetic default state
        syn_cfg = SyntheticGeneratorConfig(
            spot=100.0,
            rate=0.03,
            dividend=0.01,
            true_rho=-0.55,
            true_eta=0.65,
            true_gamma=0.40,
            expiries=[0.25, 0.5, 1.0, 2.0],
            n_strikes_per_expiry=15,
            price_noise_std=0.0005,
            seed=42,
        )
        quotes, _, _ = generate_synthetic_market(syn_cfg)
        calibrator_cfg = CalibrationConfig(safety_margin=1e-3, multi_start_seeds=[42, 101])
        surface, cal_res = calibrate_ssvi(quotes, calibrator_cfg)
        admiss = generate_ssvi_certificate(surface.parameters)
        grid_diag = run_grid_diagnostics(surface, GridConfig(n_k=35, n_T=20))

        st.session_state.quotes = quotes
        st.session_state.surface = surface
        st.session_state.calibration_result = cal_res
        st.session_state.admissibility_report = admiss
        st.session_state.grid_diagnostics = grid_diag
        st.session_state.dataset_source = "Live Synthetic Default"

    st.session_state.initialized = True


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="VolSurfaceLab — Volatility & Risk-Neutral Workbench",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_theme()
    init_session_state()

    # Determine status badges
    surface: Optional[SSVISurface] = st.session_state.get("surface")
    admiss: Optional[AdmissibilityReport] = st.session_state.get("admissibility_report")
    grid_diag: Optional[GridDiagnostics] = st.session_state.get("grid_diagnostics")

    status_text = "SURFACE CERTIFIED"
    status_type = "pass"
    if admiss is None or admiss.theorem_status != StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS:
        status_text = "CERTIFICATE FAILED"
        status_type = "warn"

    render_header(status_text=status_text, status_type=status_type)

    # Sidebar Navigation & Presets
    st.sidebar.markdown("### 🎛️ Control Panel")
    dataset_info = f"{len(st.session_state.quotes)} quotes loaded" if st.session_state.quotes else "No quotes loaded"
    st.sidebar.caption(f"**Data Source:** `{st.session_state.dataset_source}`\n\n**Size:** {dataset_info}")

    # Presets selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚡ Market Presets")
    preset = st.sidebar.selectbox(
        "Load Preset Market Scenario:",
        [
            "Custom / Current",
            "Equity Index (S&P 500 Style)",
            "FX Smile (EUR/USD Style)",
            "Commodity Smile (Crude Style)",
            "Stress Boundary Scenario",
        ],
    )

    if preset != "Custom / Current" and st.sidebar.button("Apply Selected Preset"):
        with st.spinner(f"Generating and calibrating {preset}..."):
            if preset == "Equity Index (S&P 500 Style)":
                cfg = SyntheticGeneratorConfig(
                    spot=100.0, rate=0.03, dividend=0.015,
                    true_rho=-0.72, true_eta=0.55, true_gamma=0.38,
                    expiries=[0.25, 0.5, 1.0, 2.0], n_strikes_per_expiry=17,
                    price_noise_std=0.0005, seed=101,
                )
            elif preset == "FX Smile (EUR/USD Style)":
                cfg = SyntheticGeneratorConfig(
                    spot=1.10, rate=0.025, dividend=0.02,
                    true_rho=-0.08, true_eta=0.45, true_gamma=0.45,
                    expiries=[0.1, 0.25, 0.5, 1.0, 1.5], n_strikes_per_expiry=17,
                    price_noise_std=0.0003, seed=202,
                )
            elif preset == "Commodity Smile (Crude Style)":
                cfg = SyntheticGeneratorConfig(
                    spot=75.0, rate=0.04, dividend=0.0,
                    true_rho=+0.32, true_eta=0.62, true_gamma=0.35,
                    expiries=[0.2, 0.4, 0.8, 1.2, 1.8], n_strikes_per_expiry=15,
                    price_noise_std=0.0008, seed=303,
                )
            else:  # Stress Boundary
                cfg = SyntheticGeneratorConfig(
                    spot=100.0, rate=0.02, dividend=0.01,
                    true_rho=-0.88, true_eta=1.45, true_gamma=0.25,
                    expiries=[0.25, 0.5, 1.0, 2.0], n_strikes_per_expiry=19,
                    price_noise_std=0.0010, seed=404,
                )

            quotes, _, _ = generate_synthetic_market(cfg)
            calibrator_cfg = CalibrationConfig(safety_margin=1e-3, multi_start_seeds=[101, 202])
            surf, cal_res = calibrate_ssvi(quotes, calibrator_cfg)
            adm = generate_ssvi_certificate(surf.parameters)
            gd = run_grid_diagnostics(surf, GridConfig(n_k=35, n_T=20))

            st.session_state.quotes = quotes
            st.session_state.surface = surf
            st.session_state.calibration_result = cal_res
            st.session_state.admissibility_report = adm
            st.session_state.grid_diagnostics = gd
            st.session_state.density_result = None
            st.session_state.local_vol_result = None
            st.session_state.ablation_result = None
            st.session_state.dataset_source = f"Preset: {preset}"
            st.rerun()

    # Load from external artifact file
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📁 Load Existing Artifact")
    art_path = st.sidebar.text_input("Artifact File Path:", value="outputs/demo/run_artifact.json")
    if st.sidebar.button("Load Artifact from Disk"):
        if Path(art_path).exists():
            try:
                loaded_art = load_run_artifacts(art_path)
                st.session_state.surface = SSVISurface(loaded_art.parameters)
                st.session_state.calibration_result = loaded_art.calibration_result
                st.session_state.admissibility_report = loaded_art.admissibility_report
                st.session_state.grid_diagnostics = loaded_art.grid_diagnostics
                st.session_state.dataset_source = f"Artifact: {Path(art_path).name}"
                st.sidebar.success("Artifact loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading artifact: {e}")
        else:
            st.sidebar.error("File does not exist.")

    # Main KPI Banner Cards
    cal_res: Optional[CalibrationResult] = st.session_state.get("calibration_result")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            render_kpi("Price RMSE", f"${cal_res.price_rmse:.4f}" if cal_res else "N/A", "Dollar error vs mid", "blue"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_kpi("IV RMSE", f"{cal_res.iv_rmse*100:.2f}%" if cal_res else "N/A", f"{cal_res.iv_rmse*10000:.0f} bps" if cal_res else "", "purple"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_kpi("Inside Spread", f"{cal_res.pct_inside_spread:.1f}%" if cal_res else "N/A", "Quotes within [Bid, Ask]", "green" if cal_res and cal_res.pct_inside_spread >= 80 else "amber"),
            unsafe_allow_html=True,
        )
    with col4:
        min_g = grid_diag.min_g if grid_diag else 0.0
        g_status = "green" if min_g >= 0 else "red"
        st.markdown(
            render_kpi("Min Durrleman g", f"{min_g:.4f}", "No-butterfly: g \u2265 0", g_status),
            unsafe_allow_html=True,
        )
    with col5:
        min_w_T = grid_diag.min_w_T if grid_diag else 0.0
        cal_status = "green" if min_w_T >= 0 else "red"
        st.markdown(
            render_kpi("Min Calendar w_T", f"{min_w_T:.4f}", "No-calendar: w_T \u2265 0", cal_status),
            unsafe_allow_html=True,
        )

    st.write("")

    # Primary Tabs
    tab_data, tab_calib, tab_3d, tab_slices, tab_arbitrage, tab_density, tab_ablation, tab_export = st.tabs(
        [
            "📊 Data Studio",
            "⚙️ Calibration Engine",
            "🌐 3D Surface Explorer",
            "📈 Slices & Residuals",
            "🛡️ No-Arbitrage Auditor",
            "🔬 Density Lab",
            "⚖️ Model Ablation",
            "💾 Export Center",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: DATA STUDIO & INGESTION
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("Market Quotes Ingestion & Data Quality Studio")
        st.markdown(
            "Upload live/historical options quotes via CSV or generate deterministic synthetic market quotes. "
            "All quotes are validated against arbitrage bounds and bid-ask consistency."
        )

        data_subtab1, data_subtab2, data_subtab3 = st.tabs(
            ["Generate Synthetic Quotes", "Upload CSV Dataset", "Quote Diagnostics & Parity"]
        )

        with data_subtab1:
            st.markdown("#### Synthetic Option Quote Generator")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                syn_spot = st.number_input("Underlying Spot S0:", value=100.0, step=5.0)
                syn_rate = st.number_input("Risk-Free Rate r:", value=0.03, step=0.005, format="%.4f")
                syn_div = st.number_input("Dividend Yield q:", value=0.01, step=0.005, format="%.4f")
            with col_s2:
                syn_rho = st.slider("True Skew \u03c1:", min_value=-0.95, max_value=0.95, value=-0.60, step=0.05)
                syn_eta = st.slider("True Scale \u03b7:", min_value=0.10, max_value=2.00, value=0.60, step=0.05)
                syn_gamma = st.slider("True Curvature \u03b3:", min_value=0.05, max_value=0.50, value=0.38, step=0.01)
            with col_s3:
                syn_n_strikes = st.slider("Strikes Per Expiry:", min_value=7, max_value=35, value=15, step=2)
                syn_noise = st.slider("Price Noise Std ($):", min_value=0.000, max_value=0.010, value=0.0005, step=0.0005, format="%.4f")
                syn_seed = st.number_input("Random Seed:", value=42, step=1)

            if st.button("Generate Synthetic Market Quotes", key="btn_gen_synthetic"):
                with st.spinner("Generating synthetic quotes..."):
                    cfg = SyntheticGeneratorConfig(
                        spot=float(syn_spot),
                        rate=float(syn_rate),
                        dividend=float(syn_div),
                        true_rho=float(syn_rho),
                        true_eta=float(syn_eta),
                        true_gamma=float(syn_gamma),
                        expiries=[0.25, 0.5, 1.0, 2.0],
                        n_strikes_per_expiry=int(syn_n_strikes),
                        price_noise_std=float(syn_noise),
                        seed=int(syn_seed),
                    )
                    new_quotes, _, _ = generate_synthetic_market(cfg)
                    st.session_state.quotes = new_quotes
                    st.session_state.dataset_source = f"Synthetic (Seed {syn_seed}, \u03c1={syn_rho}, \u03b7={syn_eta})"
                    st.success(f"Generated {len(new_quotes)} synthetic quotes across 4 expiries.")
                    st.rerun()

        with data_subtab2:
            st.markdown("#### Upload Custom Option Quotes CSV")
            st.markdown(
                "Required columns: `expiry` (or `time_to_maturity`), `strike`, `option_type` (`call`/`put`), `bid`, `ask`.\n\n"
                "Optional columns: `forward`, `spot`, `rate`, `dividend_yield`, `discount_factor`."
            )
            uploaded_file = st.file_uploader("Choose CSV File:", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Preview of uploaded CSV ({len(df)} rows):")
                    st.dataframe(df.head(5), use_container_width=True)

                    if st.button("Ingest and Clean Uploaded CSV"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            uploaded_file.seek(0)
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        raw_quotes, _ = load_quotes_from_csv(tmp_path)
                        cleaned_quotes, clean_rep = filter_quotes(raw_quotes, DataFilterConfig())
                        os.unlink(tmp_path)

                        st.session_state.quotes = cleaned_quotes
                        st.session_state.cleaning_report = clean_rep
                        st.session_state.dataset_source = f"Uploaded: {uploaded_file.name}"
                        st.success(f"Successfully cleaned and loaded {len(cleaned_quotes)} quotes.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Error parsing uploaded CSV: {exc}")

            if Path("data/sample_quotes.csv").exists():
                if st.button("Load Canonical Sample Dataset (`data/sample_quotes.csv`)"):
                    raw_quotes, _ = load_quotes_from_csv("data/sample_quotes.csv")
                    cleaned_quotes, clean_rep = filter_quotes(raw_quotes, DataFilterConfig())
                    st.session_state.quotes = cleaned_quotes
                    st.session_state.cleaning_report = clean_rep
                    st.session_state.dataset_source = "Canonical Sample Dataset"
                    st.success(f"Loaded {len(cleaned_quotes)} quotes from sample dataset.")
                    st.rerun()

        with data_subtab3:
            st.markdown("#### Quote Quality & Diagnostic Inspection")
            if st.session_state.quotes:
                quotes = st.session_state.quotes
                expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))
                calls = [q for q in quotes if q.option_type == "call"]
                puts = [q for q in quotes if q.option_type == "put"]
                crossed = [q for q in quotes if q.bid > q.ask]

                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                col_d1.metric("Total Quotes", len(quotes))
                col_d2.metric("Calls / Puts", f"{len(calls)} / {len(puts)}")
                col_d3.metric("Distinct Expiries", len(expiries))
                col_d4.metric("Crossed Quotes (Bid > Ask)", len(crossed))

                # Quote Parity Check
                mkt_diag = run_market_quote_diagnostics(quotes)
                st.markdown(f"**Put-Call Parity Diagnostics:** `{mkt_diag['parity_interval_violations']}` violations detected across `{mkt_diag['matched_pairs']}` matched strike-expiry pairs.")

                # Visual Scatter of Market Quotes
                fig_scatter = plot_raw_quotes_scatter(quotes)
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Dataframe inspection
                st.markdown("##### Quotes Data Table")
                quote_records = [
                    {
                        "Expiry (Y)": q.time_to_maturity,
                        "Strike": q.strike,
                        "Type": q.option_type,
                        "Bid": q.bid,
                        "Ask": q.ask,
                        "Mid": q.mid,
                        "Spread": q.spread,
                        "Forward": q.forward,
                        "Log-Moneyness k": q.log_moneyness,
                    }
                    for q in quotes[:100]
                ]
                st.dataframe(pd.DataFrame(quote_records), use_container_width=True)
                if len(quotes) > 100:
                    st.caption(f"Showing first 100 of {len(quotes)} quotes.")
            else:
                st.info("No quotes currently loaded.")

    # -------------------------------------------------------------
    # TAB 2: CALIBRATION ENGINE
    # -------------------------------------------------------------
    with tab_calib:
        st.subheader("SSVI Surface Calibration Engine")
        st.markdown(
            "Jointly calibrate the regularised power-law SSVI parameters $(\\rho, \\eta, \\gamma)$ and ATM total variance curve $\\theta(T)$. "
            "Uses Trust Region Reflective (TRF) optimization with theorem-enforced bounds."
        )

        render_theory_callout(
            "Mathematical Foundation: Power-Law SSVI Surface",
            "The regularised SSVI total variance surface is defined by:<br>"
            "$$w(k, \\theta) = \\frac{\\theta}{2} \\left[ 1 + \\rho \\phi(\\theta) k + \\sqrt{(\\phi(\\theta)k + \\rho)^2 + 1 - \\rho^2} \\right]$$<br>"
            "where $\\phi(\\theta) = \\frac{\\eta}{\\theta^\\gamma (1 + \\theta)^{1 - \\gamma}}$ with $\\gamma \\in (0, 0.5]$, $\\eta > 0$, $|\\rho| < 1$, "
            "and $\\theta(T) > 0, \\theta'(T) \\ge 0$ monotonically via PCHIP spline interpolation.",
        )

        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.markdown("#### Optimizer Configuration")
            loss_fn = st.selectbox("Loss Function:", ["soft_l1", "linear"], help="soft_l1 provides Huber-like robustness to price outliers.")
            safety_eps = st.slider("Safety Margin \u03b5:", min_value=0.0001, max_value=0.05, value=0.001, step=0.0005, format="%.4f")
            multi_seeds = st.slider("Multi-Start Random Seeds:", min_value=1, max_value=8, value=3, step=1)
            max_iter = st.number_input("Max Iterations:", value=500, step=100)

            run_calib = st.button("⚡ Calibrate SSVI Surface Live", type="primary")

        with col_c2:
            if run_calib:
                if not st.session_state.quotes:
                    st.error("No quotes available to calibrate. Please load data in the Data Studio tab first.")
                else:
                    with st.spinner("Calibrating SSVI surface with multi-start TRF solver..."):
                        start_t = time.perf_counter()
                        seeds = [42 + i * 17 for i in range(int(multi_seeds))]
                        c_cfg = CalibrationConfig(
                            loss_function=loss_fn,
                            safety_margin=float(safety_eps),
                            multi_start_seeds=seeds,
                            max_nfev=int(max_iter),
                        )
                        surf, cal_res = calibrate_ssvi(st.session_state.quotes, c_cfg)
                        elapsed = time.perf_counter() - start_t

                        adm = generate_ssvi_certificate(surf.parameters)
                        gd = run_grid_diagnostics(surf, GridConfig(n_k=35, n_T=20))

                        st.session_state.surface = surf
                        st.session_state.calibration_result = cal_res
                        st.session_state.admissibility_report = adm
                        st.session_state.grid_diagnostics = gd
                        st.session_state.density_result = None
                        st.session_state.local_vol_result = None
                        st.session_state.ablation_result = None

                        st.success(f"Calibration converged in {elapsed:.2f}s ({cal_res.evaluations} evaluations).")
                        st.rerun()

            if st.session_state.surface and st.session_state.calibration_result:
                surf = st.session_state.surface
                res = st.session_state.calibration_result
                params = surf.parameters

                st.markdown("#### Calibrated Surface Parameters")
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("Skew \u03c1", f"{params.rho:.4f}", help="Governs left/right asymmetry of volatility smile.")
                col_p2.metric("Scale \u03b7", f"{params.eta:.4f}", help="Governs overall wings curvature and slope.")
                col_p3.metric("Curvature \u03b3", f"{params.gamma:.4f}", help="Power-law parameter in (0, 0.5].")

                st.markdown("#### Fit Residuals Telemetry")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Price RMSE", f"${res.price_rmse:.4f}")
                col_r2.metric("Price MAE", f"${res.price_mae:.4f}")
                col_r3.metric("IV RMSE", f"{res.iv_rmse*100:.2f}%")

        # ATM Total Variance Curve Plot
        if st.session_state.surface:
            surf = st.session_state.surface
            params = surf.parameters
            expiries = np.array(params.expiries)
            knots = np.array(params.theta_knots)
            T_dense = np.linspace(min(expiries), max(expiries), 100)

            st.markdown("---")
            st.markdown("#### ATM Total Variance Term Structure \u03b8(T) & Monotonicity Rate")
            fig_theta = plot_theta_variance_curve(expiries, knots, surf, T_dense)
            st.plotly_chart(fig_theta, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: 3D SURFACE EXPLORER
    # -------------------------------------------------------------
    with tab_3d:
        st.subheader("Interactive 3D Surface Explorer")
        st.markdown(
            "Explore 3D surfaces of Implied Volatility $\\sigma_{IV}(k, T)$, Total Implied Variance $w(k, T)$, "
            "and Dupire Local Volatility $\\sigma_{loc}(K, T)$."
        )

        if st.session_state.surface is None:
            st.warning("Please calibrate or load a surface first.")
        else:
            surf = st.session_state.surface
            params = surf.parameters

            # Viewport controls
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
            with col_ctrl1:
                surface_type = st.selectbox(
                    "Select Surface:",
                    ["Implied Volatility \u03c3_IV(k, T)", "Total Variance w(k, T)", "Dupire Local Volatility \u03c3_loc(K, T)"],
                )
            with col_ctrl2:
                k_span = st.slider("Moneyness Range |k|:", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            with col_ctrl3:
                res_options = {
                    "25 x 15 (Fast)": (25, 15),
                    "41 x 25 (Balanced)": (41, 25),
                    "61 x 35 (Fine)": (61, 35),
                }
                res_label = st.select_slider(
                    "Resolution (Nk x NT):",
                    options=list(res_options.keys()),
                    value="41 x 25 (Balanced)",
                )
                res_grid = res_options[res_label]
            with col_ctrl4:
                cmap = st.selectbox("Color Palette:", ["Viridis", "Plasma", "Turbo", "Cividis", "Spectral", "Inferno"])

            k_grid_3d = np.linspace(-k_span, k_span, res_grid[0])
            T_grid_3d = np.linspace(min(params.expiries), max(params.expiries), res_grid[1])

            if surface_type == "Implied Volatility \u03c3_IV(k, T)":
                fig_3d = plot_implied_vol_surface(surf, k_grid_3d, T_grid_3d, colorscale=cmap)
                st.plotly_chart(fig_3d, use_container_width=True)
            elif surface_type == "Total Variance w(k, T)":
                fig_3d = plot_total_variance_surface(surf, k_grid_3d, T_grid_3d, colorscale=cmap)
                st.plotly_chart(fig_3d, use_container_width=True)
            else:  # Dupire Local Volatility
                with st.spinner("Evaluating exact Dupire local volatility surface..."):
                    lv_res = evaluate_dupire_local_volatility(surf, k_grid_3d, T_grid_3d)
                    st.session_state.local_vol_result = lv_res

                    fig_lv = plot_local_vol_surface(lv_res, colorscale=cmap)
                    st.plotly_chart(fig_lv, use_container_width=True)

                    st.markdown(
                        f"**Dupire Surface Quality Metrics:** `{lv_res.valid_fraction*100:.1f}%` valid cells "
                        f"(`{lv_res.valid_cells}/{lv_res.total_cells}` cells non-negative and well-conditioned)."
                    )

    # -------------------------------------------------------------
    # TAB 4: SLICES & RESIDUALS
    # -------------------------------------------------------------
    with tab_slices:
        st.subheader("Volatility Smile Slices & Residual Error Analysis")
        st.markdown(
            "Inspect per-expiry volatility smiles, market observations, and pricing residuals against the bid-ask corridor."
        )

        if st.session_state.surface is None or not st.session_state.quotes:
            st.warning("Please calibrate or load quotes first.")
        else:
            surf = st.session_state.surface
            quotes = st.session_state.quotes
            expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))

            col_sl1, col_sl2 = st.columns([1, 3])
            with col_sl1:
                selected_T = st.selectbox(
                    "Select Maturity T (Years):",
                    expiries,
                    format_func=lambda x: f"T = {x:.2f} Y ({int(x*365)} days)",
                )

            # Filter quotes for selected expiry
            q_slice = [q for q in quotes if abs(q.time_to_maturity - selected_T) < 1e-5]

            # Fit independent raw SVI slice if available
            raw_slice = None
            if len(q_slice) >= 5:
                try:
                    raw_slice = fit_raw_svi_slice(q_slice, selected_T)
                except Exception:
                    raw_slice = None

            # Render dual-panel smile & residual chart
            fig_slice = plot_smile_slice_with_residuals(q_slice, surf, selected_T, raw_svi_slice=raw_slice)
            st.plotly_chart(fig_slice, use_container_width=True)

            # Term Structure Slices
            st.markdown("---")
            st.markdown("#### Volatility Term Structure Curves \u03c3(T) across Moneyness Levels")
            T_grid_term = np.linspace(min(expiries), max(expiries), 50)
            fig_term = plot_term_structure_curves(surf, T_grid_term)
            st.plotly_chart(fig_term, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: NO-ARBITRAGE AUDITOR
    # -------------------------------------------------------------
    with tab_arbitrage:
        st.subheader("Arbitrage-Free Verification Suite (3-Layer Hierarchy)")
        st.markdown(
            "VolSurfaceLab enforces a rigorous three-layer no-arbitrage hierarchy: "
            "**Layer A: Analytical Gatheral-Jacquier Theorem Certificate**, "
            "**Layer B: Dense 2D Grid Diagnostics**, and "
            "**Layer C: Market Quote Parity Intervals**."
        )

        if st.session_state.surface is None:
            st.warning("Please calibrate or load a surface first.")
        else:
            surf = st.session_state.surface
            adm: AdmissibilityReport = st.session_state.admissibility_report
            gd: GridDiagnostics = st.session_state.grid_diagnostics

            # Layer A: Theorem Certificate
            st.markdown("#### Layer A: Gatheral-Jacquier (2014) Theorem Conditions")
            col_t1, col_t2 = st.columns([1, 1])

            with col_t1:
                badge_type = "pass" if adm.theorem_status == StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS else "warn"
                st.markdown(
                    f"**Theorem Admissibility Status:** <span class='badge badge-{badge_type}'>{adm.theorem_status.value}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"- **Condition 1:** $\\theta \\phi (1+|\\rho|) = {adm.cond1_value:.4f} < 4$ (Safety Margin: `{adm.cond1_margin:.4f}`)\n"
                    f"- **Condition 2:** $\\theta \\phi^2 (1+|\\rho|) = {adm.cond2_value:.4f} \\le 4$ (Safety Margin: `{adm.cond2_margin:.4f}`)\n"
                    f"- **Scale Bound:** $\\eta = {surf.parameters.eta:.4f} \\le \\eta_{{cap}} = {surf.parameters.eta + adm.eta_margin:.4f}$ (Safety Margin: `{adm.eta_margin:.4f}`)\n"
                    f"- **ATM Monotonicity:** $\\min \\theta'(T) = {adm.theta_min_derivative:.6f} \\ge 0$"
                )

            with col_t2:
                # Progress meters for margins
                c1_pct = min(max(adm.cond1_value / 4.0, 0.0), 1.0)
                c2_pct = min(max(adm.cond2_value / 4.0, 0.0), 1.0)
                st.write(f"Condition 1 Utilization: {c1_pct*100:.1f}% of limit")
                st.progress(c1_pct)
                st.write(f"Condition 2 Utilization: {c2_pct*100:.1f}% of limit")
                st.progress(c2_pct)

            # Layer B: 2D Grid Scans
            st.markdown("---")
            st.markdown("#### Layer B: Dense 2D Grid Heatmap Diagnostics")
            k_grid_diag = np.linspace(gd.k_min, gd.k_max, 45)
            T_grid_diag = np.linspace(gd.T_min, gd.T_max, 25)

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("##### Durrleman Butterfly Function $g(k, T) \\ge 0$")
                fig_g = plot_durrleman_g_heatmap(surf, k_grid_diag, T_grid_diag, show_min_pin=True)
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption(f"Minimum Durrleman g: **{gd.min_g:.4f}** (Negative points: {gd.negative_g_count})")

            with col_h2:
                st.markdown("##### Calendar Derivative $w_T(k, T) = \\partial w/\\partial T \\ge 0$")
                fig_cal = plot_calendar_heatmap(surf, k_grid_diag, T_grid_diag, show_min_pin=True)
                st.plotly_chart(fig_cal, use_container_width=True)
                st.caption(f"Minimum Calendar w_T: **{gd.min_w_T:.4f}** (Negative points: {gd.negative_w_T_count})")

    # -------------------------------------------------------------
    # TAB 6: DENSITY LAB
    # -------------------------------------------------------------
    with tab_density:
        st.subheader("Breeden–Litzenberger Risk-Neutral Density Lab")
        st.markdown(
            "Extract the exact continuous risk-neutral probability density $q_T(K) = e^{rT} \\frac{\\partial^2 C}{\\partial K^2}$ "
            "directly from the analytical SSVI strike derivatives ($w_k, w_{kk}$). "
            "Includes raw Simpson integration and martingale forward expectation audits."
        )

        render_theory_callout(
            "Breeden–Litzenberger (1978) Theorem",
            "The risk-neutral density is given by:<br>"
            "$$q_T(K) = \\frac{1}{D} \\frac{\\partial^2 C}{\\partial K^2} = \\frac{g(k, \\theta)}{K \\sqrt{2\\pi w}} \\exp\\left( -\\frac{d_2^2}{2} \\right)$$<br>"
            "where $g(k, \\theta)$ is the Durrleman butterfly function. Non-negativity $g \\ge 0$ is the exact necessary and sufficient condition "
            "for $q_T(K) \\ge 0$ (arbitrage freedom). **No artificial clipping or renormalisation is performed.**",
        )

        if st.session_state.surface is None:
            st.warning("Please calibrate or load a surface first.")
        else:
            surf = st.session_state.surface
            expiries = surf.parameters.expiries

            col_den1, col_den2, col_den3 = st.columns(3)
            with col_den1:
                den_expiry = st.selectbox(
                    "Density Maturity T (Years):",
                    expiries,
                    format_func=lambda x: f"T = {x:.2f} Y",
                )
            with col_den2:
                den_sigma = st.slider("Domain Width (\u03c3-range):", min_value=2.5, max_value=6.0, value=4.5, step=0.5)
            with col_den3:
                den_strikes = st.slider("Strike Grid Points:", min_value=101, max_value=501, value=251, step=50)

            # Extract density
            F_est = 100.0
            D_est = np.exp(-0.03 * den_expiry)
            if st.session_state.quotes:
                q_match = [q for q in st.session_state.quotes if abs(q.time_to_maturity - den_expiry) < 1e-4]
                if q_match and q_match[0].forward:
                    F_est = q_match[0].forward
                if q_match and q_match[0].discount_factor:
                    D_est = q_match[0].discount_factor

            den_cfg = DensityConfig(
                n_strikes=int(den_strikes),
                std_dev_range=float(den_sigma),
            )
            density_res = extract_risk_neutral_density(surf, den_expiry, F_est, D_est, den_cfg)
            st.session_state.density_result = density_res

            # KPI Cards for Density
            col_dm1, col_dm2, col_dm3, col_dm4 = st.columns(4)
            col_dm1.metric("Raw Integrated Mass", f"{density_res.total_mass:.5f}", help="Unrenormalised Simpson integral. Must be ~1.0.")
            col_dm2.metric("Forward Expectation E[K]", f"${density_res.first_moment:.2f}", help="Martingale condition: integral K*q(K) dK.")
            col_dm3.metric("Forward Target F(T)", f"${density_res.forward:.2f}")
            col_dm4.metric("Forward Dollar Error", f"${abs(density_res.first_moment - density_res.forward):.4f}", f"{abs(density_res.relative_moment_error)*10000:.1f} bps")

            # Plot Density & CDF
            fig_den = plot_density_and_cdf(density_res)
            st.plotly_chart(fig_den, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 7: MODEL ABLATION & BENCHMARKS
    # -------------------------------------------------------------
    with tab_ablation:
        st.subheader("Model Ablation & Baseline Benchmarking")
        st.markdown(
            "Compare the arbitrage-free **SSVI Surface** against traditional baselines: "
            "**Independent Raw SVI** (per-expiry slices) and **Flat Black-Scholes**."
        )

        if st.session_state.surface is None or not st.session_state.quotes:
            st.warning("Please calibrate or load quotes first.")
        else:
            surf = st.session_state.surface
            quotes = st.session_state.quotes
            cal_res = st.session_state.calibration_result

            if st.button("⚡ Run Full Ablation Experiment"):
                with st.spinner("Fitting baselines and evaluating calendar crossings..."):
                    ablation_res = run_model_ablation(quotes, surf, cal_res)
                    st.session_state.ablation_result = ablation_res
                    st.success("Ablation experiment completed successfully.")

            if st.session_state.ablation_result:
                ab_data = st.session_state.ablation_result
                comp_table = ab_data["comparison_table"]

                st.markdown("#### Comparative Benchmark Performance")
                df_comp = pd.DataFrame(comp_table)
                df_comp_display = df_comp.map(lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
                st.dataframe(df_comp_display, use_container_width=True)

                st.markdown(
                    render_theory_callout(
                        "Key Quant Takeaway: Why Independent Raw SVI Fails",
                        f"Independent Raw SVI fits each expiry slice in isolation. Although each slice may be butterfly-arbitrage free, "
                        f"the slices often cross between expiries, generating **{ab_data.get('calendar_crossings', 0)} calendar arbitrage crossings** "
                        f"(minimum total variance difference $\\min \\Delta w = {ab_data.get('min_calendar_variance_diff', 0):.5f}$). "
                        f"In contrast, **SSVI guarantees complete calendar monotonicity across all expiries** by design.",
                    ),
                    unsafe_allow_html=True,
                )

                # Ablation Smile Comparison on first expiry
                expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))
                first_T = expiries[0]
                q_slice = [q for q in quotes if abs(q.time_to_maturity - first_T) < 1e-5]

                flat_b = fit_flat_black(quotes)
                raw_svi_all = fit_independent_raw_svi(quotes)
                raw_slice = raw_svi_all.slices.get(first_T)

                fig_abl = plot_ablation_comparison_smiles(
                    q_slice, surf, raw_slice, flat_b.sigma_flat, first_T
                )
                st.plotly_chart(fig_abl, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 8: EXPORT CENTER
    # -------------------------------------------------------------
    with tab_export:
        st.subheader("Artifact & Report Export Center")
        st.markdown(
            "Download reproducible calibration artifacts, datasets, parameter configurations, "
            "and the complete standalone self-contained HTML audit report."
        )

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.markdown("#### Model & Provenance JSON Downloads")
            if st.session_state.surface and st.session_state.calibration_result:
                surf = st.session_state.surface
                params_dict = surf.parameters.model_dump()
                st.download_button(
                    "📥 Download Calibrated Parameters (`calibration.json`)",
                    data=json.dumps(params_dict, indent=2),
                    file_name="calibration.json",
                    mime="application/json",
                )

                # Full Run Artifact
                artifact = RunArtifact(
                    run_id="dashboard_run",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    config_hash="web_dashboard",
                    environment=capture_environment_metadata(),
                    parameters=surf.parameters,
                    calibration_result=st.session_state.calibration_result,
                    admissibility_report=st.session_state.admissibility_report,
                    grid_diagnostics=st.session_state.grid_diagnostics,
                    quote_cleaning_report=st.session_state.cleaning_report or QuoteCleaningReport(raw_count=len(st.session_state.quotes), accepted_count=len(st.session_state.quotes), rejected_count=0),
                )
                art_json = artifact.model_dump_json(indent=2)
                st.download_button(
                    "📥 Download Complete Run Artifact (`run_artifact.json`)",
                    data=art_json,
                    file_name="run_artifact.json",
                    mime="application/json",
                )

            # Quotes CSV
            if st.session_state.quotes:
                df_q = pd.DataFrame([q.model_dump() for q in st.session_state.quotes])
                csv_buffer = io.StringIO()
                df_q.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Market Quotes CSV",
                    data=csv_buffer.getvalue(),
                    file_name="quotes.csv",
                    mime="text/csv",
                )

        with col_ex2:
            st.markdown("#### HTML Audit Report")
            st.markdown(
                "Generate a self-contained, interactive HTML audit report including all figures, "
                "no-arbitrage certificates, and benchmark diagnostics."
            )
            if st.session_state.surface and st.session_state.calibration_result:
                if st.button("Generate Standalone HTML Report"):
                    with st.spinner("Compiling HTML audit report..."):
                        report_out_dir = Path("outputs/web_report")
                        report_out_dir.mkdir(parents=True, exist_ok=True)
                        report_file = report_out_dir / "report.html"
                        generate_html_report(
                            output_path=report_file,
                            run_id=artifact.run_id,
                            provenance=artifact.environment,
                            quotes=st.session_state.quotes,
                            cleaning_report=artifact.quote_cleaning_report,
                            surface=surf,
                            calibration_result=st.session_state.calibration_result,
                            admissibility_report=st.session_state.admissibility_report,
                            grid_diagnostics=st.session_state.grid_diagnostics,
                            density_result=st.session_state.density_result,
                            local_vol_result=st.session_state.local_vol_result,
                            ablation_summary=st.session_state.ablation_result,
                            is_synthetic=True,
                        )
                        st.session_state.html_report_path = str(report_file)
                        st.success(f"Report compiled at `{report_file}`.")

                if "html_report_path" in st.session_state and Path(st.session_state.html_report_path).exists():
                    with open(st.session_state.html_report_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.download_button(
                        "📥 Download Standalone HTML Report (`volsurfacelab_report.html`)",
                        data=html_content,
                        file_name="volsurfacelab_report.html",
                        mime="text/html",
                    )


if __name__ == "__main__":
    main()
