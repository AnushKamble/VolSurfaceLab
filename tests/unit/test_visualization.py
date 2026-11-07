"""Unit tests for Plotly visualizations."""

import numpy as np
import plotly.graph_objects as go
import pytest

from volsurfacelab.density.breeden_litzenberger import extract_risk_neutral_density
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.local_vol.dupire import evaluate_dupire_local_volatility
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.visualization.figures import (
    plot_calendar_heatmap,
    plot_density_and_cdf,
    plot_durrleman_g_heatmap,
    plot_implied_vol_surface,
    plot_local_vol_surface,
    plot_total_variance_surface,
)


@pytest.fixture
def viz_surface() -> SSVISurface:
    params = SSVIParameters(
        rho=-0.35,
        eta=1.0,
        gamma=0.35,
        expiries=[0.2, 0.5, 1.0, 2.0],
        theta_knots=[0.03, 0.06, 0.11, 0.19],
    )
    return SSVISurface(params)


def test_visualization_figures_generation(viz_surface):
    """Verify all Plotly figure functions return valid figures with proper titles and non-empty data."""
    k_grid = np.linspace(-1.0, 1.0, 21)
    T_grid = np.linspace(0.2, 1.8, 11)

    # 1. 3D IV
    fig_iv = plot_implied_vol_surface(viz_surface, k_grid, T_grid)
    assert isinstance(fig_iv, go.Figure)
    assert len(fig_iv.data) == 1
    assert "Implied Volatility" in fig_iv.layout.title.text

    # 2. 3D Total Variance
    fig_w = plot_total_variance_surface(viz_surface, k_grid, T_grid)
    assert isinstance(fig_w, go.Figure)
    assert len(fig_w.data) == 1

    # 3. Durrleman g Heatmap
    fig_g = plot_durrleman_g_heatmap(viz_surface, k_grid, T_grid)
    assert isinstance(fig_g, go.Figure)
    assert len(fig_g.data) == 1

    # 4. Calendar Heatmap
    fig_cal = plot_calendar_heatmap(viz_surface, k_grid, T_grid)
    assert isinstance(fig_cal, go.Figure)
    assert len(fig_cal.data) == 1

    # 5. Density & CDF
    dens = extract_risk_neutral_density(viz_surface, maturity=1.0, forward=100.0, discount_factor=0.96)
    fig_dens = plot_density_and_cdf(dens)
    assert isinstance(fig_dens, go.Figure)
    assert len(fig_dens.data) == 2  # density + cdf

    # 6. Local Vol Surface
    lv = evaluate_dupire_local_volatility(viz_surface, k_grid, T_grid, forward=100.0)
    fig_lv = plot_local_vol_surface(lv)
    assert isinstance(fig_lv, go.Figure)
    assert len(fig_lv.data) == 1
