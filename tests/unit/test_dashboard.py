"""Unit tests for dashboard theme, components, and expanded visualization utilities."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from volsurfacelab.dashboard.theme import render_header, render_kpi, render_theory_callout
from volsurfacelab.domain.models import OptionQuote, SSVIParameters
from volsurfacelab.surface.raw_svi_baseline import RawSVISliceParams
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.visualization.figures import (
    plot_ablation_comparison_smiles,
    plot_calendar_heatmap,
    plot_durrleman_g_heatmap,
    plot_implied_vol_surface,
    plot_raw_quotes_scatter,
    plot_smile_slice_with_residuals,
    plot_term_structure_curves,
    plot_theta_variance_curve,
    plot_total_variance_surface,
)


@pytest.fixture
def test_surface() -> SSVISurface:
    params = SSVIParameters(
        rho=-0.40,
        eta=0.80,
        gamma=0.35,
        expiries=[0.25, 0.5, 1.0, 2.0],
        theta_knots=[0.04, 0.08, 0.14, 0.22],
    )
    return SSVISurface(params)


@pytest.fixture
def sample_quotes() -> list[OptionQuote]:
    quotes = []
    from datetime import datetime, timezone
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    for T in [0.25, 1.0]:
        for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
            quotes.append(
                OptionQuote(
                    underlying_symbol="SPX",
                    quote_timestamp=now,
                    valuation_timestamp=now,
                    expiry_timestamp=now,
                    time_to_maturity=T,
                    strike=K,
                    option_type="call",
                    bid=max(100.0 - K, 0.5),
                    ask=max(100.0 - K, 0.5) + 0.5,
                    forward=100.0,
                    discount_factor=0.99,
                )
            )
    return quotes


def test_dashboard_theme_renderers():
    """Verify theme helper functions return non-empty valid HTML markup."""
    kpi_html = render_kpi("Price RMSE", "$0.0240", "Residual error", status="green")
    assert "kpi-card" in kpi_html
    assert "$0.0240" in kpi_html
    assert "kpi-green" in kpi_html

    theory_html = render_kpi("Theory", "Durrleman", sub="g >= 0", status="blue")
    assert "Durrleman" in theory_html


def test_smile_slice_with_residuals(test_surface, sample_quotes):
    """Verify high-precision dual-panel smile slice and residuals plot."""
    q_slice = [q for q in sample_quotes if abs(q.time_to_maturity - 0.25) < 1e-4]
    raw_slice = RawSVISliceParams(expiry=0.25, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

    fig = plot_smile_slice_with_residuals(q_slice, test_surface, 0.25, raw_svi_slice=raw_slice)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 5  # Market mid, bid, ask, SSVI curve, raw SVI curve, spread envelope, residuals


def test_term_structure_curves(test_surface):
    """Verify volatility term structure plot across moneyness levels."""
    T_grid = np.linspace(0.25, 2.0, 15)
    fig = plot_term_structure_curves(test_surface, T_grid, k_levels=[-0.2, 0.0, 0.2])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_theta_variance_curve(test_surface):
    """Verify ATM total variance curve theta(T) and derivative plot."""
    expiries = np.array(test_surface.parameters.expiries)
    knots = np.array(test_surface.parameters.theta_knots)
    T_dense = np.linspace(0.25, 2.0, 30)
    fig = plot_theta_variance_curve(expiries, knots, test_surface, T_dense)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4


def test_raw_quotes_scatter(sample_quotes):
    """Verify raw quotes scatter plot generation."""
    fig = plot_raw_quotes_scatter(sample_quotes)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_ablation_comparison_smiles(test_surface, sample_quotes):
    """Verify multi-model smile comparison figure."""
    q_slice = [q for q in sample_quotes if abs(q.time_to_maturity - 0.25) < 1e-4]
    raw_slice = RawSVISliceParams(expiry=0.25, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    fig = plot_ablation_comparison_smiles(q_slice, test_surface, raw_slice, flat_vol=0.20, expiry=0.25)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3


def test_heatmaps_with_min_pin(test_surface):
    """Verify 2D heatmaps with show_min_pin=True."""
    k_grid = np.linspace(-1.0, 1.0, 21)
    T_grid = np.linspace(0.25, 1.8, 11)

    fig_g = plot_durrleman_g_heatmap(test_surface, k_grid, T_grid, show_min_pin=True)
    assert isinstance(fig_g, go.Figure)
    assert len(fig_g.data) == 2  # Heatmap + Scatter marker

    fig_cal = plot_calendar_heatmap(test_surface, k_grid, T_grid, show_min_pin=True)
    assert isinstance(fig_cal, go.Figure)
    assert len(fig_cal.data) == 2  # Heatmap + Scatter marker
