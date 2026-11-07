"""Interactive Plotly visualizations for implied volatility, total variance, diagnostics, density, and local volatility."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go

from volsurfacelab.domain.models import DensityResult, LocalVolResult, OptionQuote
from volsurfacelab.surface.ssvi import SSVISurface


def plot_implied_vol_surface(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> go.Figure:
    """Create interactive 3D implied volatility surface plot."""
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    w_mesh = surface.total_variance(K_mesh, T_mesh)
    iv_mesh = np.sqrt(w_mesh / T_mesh)

    fig = go.Figure(
        data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=iv_mesh * 100.0,  # Vol percentage
                colorscale="Viridis",
                colorbar_title="IV (%)",
            )
        ]
    )
    fig.update_layout(
        title="SSVI Implied Volatility Surface",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k = ln(K/F)",
            yaxis_title="Maturity T (Years)",
            zaxis_title="Implied Volatility (%)",
        ),
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig


def plot_total_variance_surface(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> go.Figure:
    """Create interactive 3D total implied variance surface plot w(k, T)."""
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    w_mesh = surface.total_variance(K_mesh, T_mesh)

    fig = go.Figure(
        data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=w_mesh,
                colorscale="Plasma",
                colorbar_title="Total Var w",
            )
        ]
    )
    fig.update_layout(
        title="SSVI Total Implied Variance Surface w(k, T)",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k",
            yaxis_title="Maturity T",
            zaxis_title="Total Variance w",
        ),
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig


def plot_fitted_slices(
    quotes: List[OptionQuote],
    surface: SSVISurface,
) -> go.Figure:
    """Create multi-expiry slices comparing market IV observations with calibrated SSVI curves."""
    fig = go.Figure()
    expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, T in enumerate(expiries):
        color = colors[idx % len(colors)]
        q_slice = [q for q in quotes if abs(q.time_to_maturity - T) < 1e-5]
        k_obs = [q.log_moneyness for q in q_slice if q.log_moneyness is not None]
        mids = [q.mid for q in q_slice]
        
        # Approximate market IV for scatter points
        from volsurfacelab.pricing.implied_vol import implied_volatility_brent
        iv_obs = []
        k_valid = []
        for q in q_slice:
            inv = implied_volatility_brent(
                q.mid, q.forward or 100.0, q.strike, q.discount_factor or 1.0, q.time_to_maturity, q.option_type
            )
            if inv.success and inv.iv is not None and q.log_moneyness is not None:
                iv_obs.append(inv.iv * 100.0)
                k_valid.append(q.log_moneyness)

        # Market scatter
        fig.add_trace(
            go.Scatter(
                x=k_valid,
                y=iv_obs,
                mode="markers",
                name=f"Market T={T:.2f}",
                marker=dict(color=color, size=6, opacity=0.7),
            )
        )

        # Continuous fitted SSVI line
        k_dense = np.linspace(-0.8, 0.8, 100)
        w_dense = surface.total_variance(k_dense, T)
        iv_dense = np.sqrt(w_dense / T) * 100.0

        fig.add_trace(
            go.Scatter(
                x=k_dense,
                y=iv_dense,
                mode="lines",
                name=f"SSVI Fit T={T:.2f}",
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title="Market Observations vs Fitted SSVI Smiles",
        xaxis_title="Log-Forward Moneyness k = ln(K/F)",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_durrleman_g_heatmap(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> go.Figure:
    """Create 2D heatmap of the Durrleman butterfly function g(k, T)."""
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    g_mesh = surface.durrleman_g(K_mesh, T_mesh)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=k_grid,
                y=T_grid,
                z=g_mesh,
                colorscale="RdYlGn",
                zmin=0.0,
                colorbar_title="g(k, T)",
            )
        ]
    )
    fig.update_layout(
        title="Durrleman Butterfly Function g(k, T) (Non-negativity implies no butterfly arbitrage)",
        xaxis_title="Log-Forward Moneyness k",
        yaxis_title="Maturity T (Years)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_calendar_heatmap(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
) -> go.Figure:
    """Create 2D heatmap of calendar derivative w_T = dw/dT."""
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    w_T_mesh = surface.derivative_T(K_mesh, T_mesh)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=k_grid,
                y=T_grid,
                z=w_T_mesh,
                colorscale="Viridis",
                zmin=0.0,
                colorbar_title="w_T",
            )
        ]
    )
    fig.update_layout(
        title="Calendar Derivative w_T(k, T) (Non-negativity implies no calendar arbitrage)",
        xaxis_title="Log-Forward Moneyness k",
        yaxis_title="Maturity T (Years)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_density_and_cdf(density: DensityResult) -> go.Figure:
    """Create interactive plot of risk-neutral density q_T(K)."""
    strikes = np.array(density.strikes)
    q = np.array(density.density_q)

    # Numerical CDF via trapezoid cumulative
    dx = np.diff(strikes)
    cdf = np.zeros_like(strikes)
    cdf[1:] = np.cumsum(0.5 * (q[:-1] + q[1:]) * dx)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=q,
            mode="lines",
            name="Risk-Neutral Density q_T(K)",
            line=dict(color="#2ca02c", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=cdf,
            mode="lines",
            name="Cumulative CDF",
            line=dict(color="#1f77b4", width=1.5, dash="dash"),
            yaxis="y2",
        )
    )

    # Vertical line at forward price
    fig.add_vline(
        x=density.forward,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Forward = {density.forward:.1f}",
    )

    fig.update_layout(
        title=f"Breeden-Litzenberger Risk-Neutral Density (T={density.expiry:.2f}Y, Total Mass={density.total_mass:.4f})",
        xaxis_title="Strike K",
        yaxis=dict(title="Density q(K)", side="left"),
        yaxis2=dict(title="CDF", side="right", overlaying="y", range=[0.0, 1.05]),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_local_vol_surface(local_vol: LocalVolResult) -> go.Figure:
    """Create 3D local volatility surface with invalid cells marked."""
    T_arr = np.array(local_vol.expiries)
    k_arr = np.array(local_vol.log_moneyness_grid)
    K_mesh, T_mesh = np.meshgrid(k_arr, T_arr)

    # Convert local_volatility array (with Nones) to floats with NaN for invalid cells
    vol_matrix = np.full((len(T_arr), len(k_arr)), np.nan, dtype=np.float64)
    for i in range(len(T_arr)):
        for j in range(len(k_arr)):
            v = local_vol.local_volatility[i][j]
            if v is not None:
                vol_matrix[i, j] = v * 100.0  # Percentage

    fig = go.Figure(
        data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=vol_matrix,
                colorscale="Cividis",
                colorbar_title="Local Vol (%)",
            )
        ]
    )
    fig.update_layout(
        title=f"Dupire Local Volatility Surface ({local_vol.valid_fraction*100:.1f}% valid cells)",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k",
            yaxis_title="Maturity T",
            zaxis_title="Local Vol (%)",
        ),
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig
