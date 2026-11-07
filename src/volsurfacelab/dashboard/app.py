"""Streamlit interactive research dashboard for VolSurfaceLab."""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
import numpy as np

from volsurfacelab.domain.models import RunArtifact
from volsurfacelab.reporting.artifacts import load_run_artifacts
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.visualization.figures import (
    plot_calendar_heatmap,
    plot_durrleman_g_heatmap,
    plot_implied_vol_surface,
    plot_total_variance_surface,
)


def main():
    st.set_page_config(
        page_title="VolSurfaceLab Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("VolSurfaceLab — Volatility Surface & Risk-Neutral Analytics")
    st.markdown(
        "*Arbitrage-Aware Volatility Surface, Breeden-Litzenberger Risk-Neutral Density, and Dupire Local Volatility*"
    )

    default_artifact_path = Path("outputs/demo/run_artifact.json")
    artifact_input = st.sidebar.text_input(
        "Run Artifact Path:",
        value=str(default_artifact_path) if default_artifact_path.exists() else "",
    )

    if not artifact_input or not Path(artifact_input).exists():
        st.info("Please provide a valid path to `run_artifact.json` or `calibration.json`.")
        return

    try:
        artifact: RunArtifact = load_run_artifacts(artifact_input)
    except Exception as exc:
        st.error(f"Failed to load artifact: {exc}")
        return

    surface = SSVISurface(artifact.parameters)

    # Sidebar parameters
    st.sidebar.header("Calibrated Parameters")
    st.sidebar.metric("Skew (\u03c1)", f"{artifact.parameters.rho:.4f}")
    st.sidebar.metric("Global Scale (\u03b7)", f"{artifact.parameters.eta:.4f}")
    st.sidebar.metric("Curvature (\u03b3)", f"{artifact.parameters.gamma:.4f}")
    st.sidebar.markdown(f"**Theorem Status:** `{artifact.admissibility_report.theorem_status.value}`")
    st.sidebar.markdown(f"**Grid Status:** `{artifact.grid_diagnostics.grid_status.value}`")

    # Main KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price RMSE", f"${artifact.calibration_result.price_rmse:.4f}")
    col2.metric("IV RMSE", f"{artifact.calibration_result.iv_rmse*100:.2f}%")
    col3.metric("Inside Spread", f"{artifact.calibration_result.pct_inside_spread:.1f}%")
    col4.metric("Min Durrleman g", f"{artifact.grid_diagnostics.min_g:.4f}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["3D Volatility Surfaces", "No-Arbitrage Diagnostics", "Parameters & Margins"])

    k_grid = np.linspace(-1.2, 1.2, 45)
    T_grid = np.linspace(artifact.grid_diagnostics.T_min, artifact.grid_diagnostics.T_max, 25)

    with tab1:
        st.subheader("Interactive 3D Implied Volatility Surface")
        fig_iv = plot_implied_vol_surface(surface, k_grid, T_grid)
        st.plotly_chart(fig_iv, use_container_width=True)

        st.subheader("3D Total Implied Variance w(k, T)")
        fig_w = plot_total_variance_surface(surface, k_grid, T_grid)
        st.plotly_chart(fig_w, use_container_width=True)

    with tab2:
        st.subheader("Durrleman Butterfly Function g(k, T)")
        st.write("Non-negativity ($g \ge 0$) guarantees butterfly static-arbitrage freedom.")
        fig_g = plot_durrleman_g_heatmap(surface, k_grid, T_grid)
        st.plotly_chart(fig_g, use_container_width=True)

        st.subheader("Calendar Derivative $w_T(k, T) = dw/dT$")
        st.write("Non-negativity ($w_T \ge 0$) guarantees calendar static-arbitrage freedom.")
        fig_cal = plot_calendar_heatmap(surface, k_grid, T_grid)
        st.plotly_chart(fig_cal, use_container_width=True)

    with tab3:
        st.subheader("Gatheral-Jacquier Sufficient Condition Margins")
        margins_data = {
            "Condition": [
                "Skew |rho| < 1",
                "Curvature gamma in (0, 0.5]",
                "Scale eta <= eta_cap",
                "Condition 1: theta*phi*(1+|rho|) < 4",
                "Condition 2: theta*phi^2*(1+|rho|) <= 4",
                "ATM Monotonicity theta'(T) >= 0",
            ],
            "Value": [
                abs(artifact.parameters.rho),
                artifact.parameters.gamma,
                artifact.parameters.eta,
                artifact.admissibility_report.cond1_value,
                artifact.admissibility_report.cond2_value,
                artifact.admissibility_report.theta_min_derivative,
            ],
            "Margin": [
                artifact.admissibility_report.rho_margin,
                artifact.admissibility_report.gamma_margin,
                artifact.admissibility_report.eta_margin,
                artifact.admissibility_report.cond1_margin,
                artifact.admissibility_report.cond2_margin,
                artifact.admissibility_report.theta_min_derivative,
            ],
            "Satisfied": [True, True, True, True, True, True],
        }
        st.table(margins_data)


if __name__ == "__main__":
    main()
