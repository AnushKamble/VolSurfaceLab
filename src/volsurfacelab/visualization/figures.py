"""Interactive Plotly visualizations for implied volatility, total variance, diagnostics, density, and local volatility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from volsurfacelab.domain.models import DensityResult, LocalVolResult, OptionQuote
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance
from volsurfacelab.pricing.implied_vol import implied_volatility_brent
from volsurfacelab.surface.ssvi import SSVISurface


def plot_implied_vol_surface(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    colorscale: str = "Viridis",
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
                colorscale=colorscale,
                colorbar_title="IV (%)",
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                ),
            )
        ]
    )
    fig.update_layout(
        title="SSVI Implied Volatility Surface \u03c3_IV(k, T)",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k = ln(K/F)",
            yaxis_title="Maturity T (Years)",
            zaxis_title="Implied Volatility (%)",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.2)),
        ),
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig


def plot_total_variance_surface(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    colorscale: str = "Plasma",
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
                colorscale=colorscale,
                colorbar_title="Total Var w",
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="cyan", project_z=True)
                ),
            )
        ]
    )
    fig.update_layout(
        title="SSVI Total Implied Variance Surface w(k, T)",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k",
            yaxis_title="Maturity T (Years)",
            zaxis_title="Total Variance w",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.2)),
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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, T in enumerate(expiries):
        color = colors[idx % len(colors)]
        q_slice = [q for q in quotes if abs(q.time_to_maturity - T) < 1e-5]

        # Market IV for scatter points
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
        if k_valid:
            fig.add_trace(
                go.Scatter(
                    x=k_valid,
                    y=iv_obs,
                    mode="markers",
                    name=f"Market T={T:.2f}",
                    marker=dict(color=color, size=6, opacity=0.8),
                )
            )

        # Continuous fitted SSVI line
        k_min = min(k_valid) - 0.15 if k_valid else -0.8
        k_max = max(k_valid) + 0.15 if k_valid else 0.8
        k_dense = np.linspace(k_min, k_max, 100)
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
        title="Market Observations vs Calibrated SSVI Smiles",
        xaxis_title="Log-Forward Moneyness k = ln(K/F)",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_smile_slice_with_residuals(
    quotes_slice: List[OptionQuote],
    surface: SSVISurface,
    expiry: float,
    raw_svi_slice: Any = None,
) -> go.Figure:
    """Create high-precision dual-panel smile slice and pricing residual chart for a single maturity."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"Implied Volatility Smile (T = {expiry:.2f} Y)", "Fit Residuals vs Bid-Ask Corridor ($)"],
    )

    F = quotes_slice[0].forward if quotes_slice and quotes_slice[0].forward is not None else 100.0
    D = quotes_slice[0].discount_factor if quotes_slice and quotes_slice[0].discount_factor is not None else 1.0

    k_obs = []
    iv_mid_obs = []
    iv_bid_obs = []
    iv_ask_obs = []
    price_res = []
    spread_half = []

    for q in quotes_slice:
        k = q.log_moneyness if q.log_moneyness is not None else np.log(q.strike / F)
        k_obs.append(k)

        # Invert mid IV
        inv_mid = implied_volatility_brent(q.mid, F, q.strike, D, expiry, q.option_type)
        if inv_mid.success and inv_mid.iv is not None:
            iv_mid_obs.append(inv_mid.iv * 100.0)
        else:
            iv_mid_obs.append(np.nan)

        # Invert bid IV
        if q.bid > 0:
            inv_bid = implied_volatility_brent(q.bid, F, q.strike, D, expiry, q.option_type)
            iv_bid_obs.append(inv_bid.iv * 100.0 if inv_bid.success and inv_bid.iv is not None else np.nan)
        else:
            iv_bid_obs.append(np.nan)

        # Invert ask IV
        inv_ask = implied_volatility_brent(q.ask, F, q.strike, D, expiry, q.option_type)
        iv_ask_obs.append(inv_ask.iv * 100.0 if inv_ask.success and inv_ask.iv is not None else np.nan)

        # SSVI model price residual
        w_model = float(surface.total_variance(k, expiry))
        model_price = black_forward_price_from_total_variance(F, q.strike, D, w_model, q.option_type)
        price_res.append(model_price - q.mid)
        spread_half.append(0.5 * q.spread)

    # Sort by k
    sort_idx = np.argsort(k_obs)
    k_sorted = np.array(k_obs)[sort_idx]
    iv_mid_sorted = np.array(iv_mid_obs)[sort_idx]
    iv_bid_sorted = np.array(iv_bid_obs)[sort_idx]
    iv_ask_sorted = np.array(iv_ask_obs)[sort_idx]
    price_res_sorted = np.array(price_res)[sort_idx]
    spread_half_sorted = np.array(spread_half)[sort_idx]

    # Upper panel: Market Quotes & SSVI fit
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=iv_mid_sorted,
            mode="markers",
            name="Market Mid IV",
            marker=dict(color="#1f77b4", size=7, symbol="circle"),
        ),
        row=1,
        col=1,
    )

    # Bid / Ask error markers
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=iv_bid_sorted,
            mode="markers",
            name="Market Bid IV",
            marker=dict(color="#2ca02c", size=4, symbol="triangle-down"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=iv_ask_sorted,
            mode="markers",
            name="Market Ask IV",
            marker=dict(color="#d62728", size=4, symbol="triangle-up"),
        ),
        row=1,
        col=1,
    )

    # Continuous SSVI curve
    k_dense = np.linspace(min(k_sorted) - 0.15, max(k_sorted) + 0.15, 120)
    w_dense = surface.total_variance(k_dense, expiry)
    iv_dense = np.sqrt(w_dense / expiry) * 100.0

    fig.add_trace(
        go.Scatter(
            x=k_dense,
            y=iv_dense,
            mode="lines",
            name="SSVI Fit",
            line=dict(color="#ff7f0e", width=2.5),
        ),
        row=1,
        col=1,
    )

    # Optional Raw SVI curve
    if raw_svi_slice is not None:
        w_raw = np.asarray(raw_svi_slice.total_variance(k_dense))
        iv_raw = np.sqrt(np.maximum(w_raw, 1e-12) / expiry) * 100.0
        fig.add_trace(
            go.Scatter(
                x=k_dense,
                y=iv_raw,
                mode="lines",
                name="Raw SVI Slice",
                line=dict(color="#9467bd", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Lower panel: Residuals & Bid-Ask Band
    # Upper spread band
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=spread_half_sorted,
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 0.4)", width=1, dash="dot"),
            name="+Half Spread",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # Lower spread band with fill
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=-spread_half_sorted,
            mode="lines",
            line=dict(color="rgba(44, 160, 44, 0.4)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.12)",
            name="Inside Spread Band",
        ),
        row=2,
        col=1,
    )
    # Zero line
    fig.add_trace(
        go.Scatter(
            x=[min(k_dense), max(k_dense)],
            y=[0, 0],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # Price residuals scatter
    fig.add_trace(
        go.Scatter(
            x=k_sorted,
            y=price_res_sorted,
            mode="markers+lines",
            name="Model Price - Mid ($)",
            marker=dict(color="#ff7f0e", size=6),
            line=dict(color="#ff7f0e", width=1),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="IV (%)", row=1, col=1)
    fig.update_yaxes(title_text="Residual ($)", row=2, col=1)
    fig.update_xaxes(title_text="Log-Forward Moneyness k = ln(K/F)", row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def plot_term_structure_curves(
    surface: SSVISurface,
    T_grid: np.ndarray,
    k_levels: Optional[List[float]] = None,
) -> go.Figure:
    """Create volatility term structure plot for specific log-moneyness levels."""
    if k_levels is None:
        k_levels = [-0.25, -0.10, 0.0, 0.10, 0.25]

    fig = go.Figure()
    colors = ["#2b5c8f", "#1f77b4", "#00b4d8", "#e07a5f", "#d62728"]

    for idx, k in enumerate(k_levels):
        w = surface.total_variance(k, T_grid)
        iv = np.sqrt(w / T_grid) * 100.0
        label = f"ATM (k=0.0)" if abs(k) < 1e-4 else (f"OTM Put (k={k:+.2f})" if k < 0 else f"OTM Call (k={k:+.2f})")
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=T_grid,
                y=iv,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title="Volatility Term Structure \u03c3_IV(T) across Moneyness Levels",
        xaxis_title="Maturity T (Years)",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_theta_variance_curve(
    expiries: np.ndarray,
    theta_knots: np.ndarray,
    surface: SSVISurface,
    T_dense: np.ndarray,
) -> go.Figure:
    """Create plot of calibrated ATM total variance \u03b8(T) and its time derivative \u03b8'(T)."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["ATM Total Variance \u03b8(T) (PCHIP Monotone Spline)", "ATM Monotonicity Rate \u03b8'(T) \u2265 0"],
    )

    # Evaluate theta and derivative
    th_dense = surface.theta_curve.evaluate(T_dense)
    th_deriv = surface.theta_curve.derivative(T_dense)

    # Upper: theta(T)
    fig.add_trace(
        go.Scatter(
            x=expiries,
            y=theta_knots,
            mode="markers",
            name="Calibrated Knots",
            marker=dict(color="#d62728", size=9, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=T_dense,
            y=th_dense,
            mode="lines",
            name="PCHIP \u03b8(T)",
            line=dict(color="#1f77b4", width=2.5),
        ),
        row=1,
        col=1,
    )

    # Lower: theta'(T)
    fig.add_trace(
        go.Scatter(
            x=T_dense,
            y=th_deriv,
            mode="lines",
            name="\u03b8'(T)",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.15)",
        ),
        row=2,
        col=1,
    )
    # Zero line
    fig.add_trace(
        go.Scatter(
            x=[min(T_dense), max(T_dense)],
            y=[0, 0],
            mode="lines",
            line=dict(color="red", width=1.5, dash="dash"),
            name="Zero Bound (\u03b8' \u2265 0)",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="\u03b8(T)", row=1, col=1)
    fig.update_yaxes(title_text="d\u03b8/dT", row=2, col=1)
    fig.update_xaxes(title_text="Maturity T (Years)", row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def plot_raw_quotes_scatter(quotes: List[OptionQuote]) -> go.Figure:
    """Create scatter plot of market quotes across strikes and maturities."""
    fig = go.Figure()
    expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))

    for T in expiries:
        q_slice = [q for q in quotes if abs(q.time_to_maturity - T) < 1e-5]
        strikes = [q.strike for q in q_slice]
        mids = [q.mid for q in q_slice]
        calls = [q for q in q_slice if q.option_type == "call"]
        puts = [q for q in q_slice if q.option_type == "put"]

        if calls:
            fig.add_trace(
                go.Scatter(
                    x=[q.strike for q in calls],
                    y=[q.mid for q in calls],
                    mode="markers",
                    name=f"Calls T={T:.2f}",
                    marker=dict(symbol="circle", size=6),
                )
            )
        if puts:
            fig.add_trace(
                go.Scatter(
                    x=[q.strike for q in puts],
                    y=[q.mid for q in puts],
                    mode="markers",
                    name=f"Puts T={T:.2f}",
                    marker=dict(symbol="x", size=6),
                )
            )

    fig.update_layout(
        title="Raw Market Option Prices Across Strikes and Expiries",
        xaxis_title="Strike K",
        yaxis_title="Option Mid Price ($)",
        hovermode="closest",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_durrleman_g_heatmap(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    show_min_pin: bool = False,
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

    if show_min_pin:
        min_idx = np.unravel_index(np.argmin(g_mesh), g_mesh.shape)
        min_T = T_grid[min_idx[0]]
        min_k = k_grid[min_idx[1]]
        min_g = g_mesh[min_idx]

        fig.add_trace(
            go.Scatter(
                x=[min_k],
                y=[min_T],
                mode="markers+text",
                name=f"Min g = {min_g:.4f}",
                text=[f" Min g={min_g:.4f}"],
                textposition="top right",
                marker=dict(color="cyan", size=11, symbol="star", line=dict(color="black", width=1.5)),
            )
        )

    fig.update_layout(
        title="Durrleman Butterfly Function g(k, T) (Non-negativity g \u2265 0 guarantees no butterfly arbitrage)",
        xaxis_title="Log-Forward Moneyness k = ln(K/F)",
        yaxis_title="Maturity T (Years)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_calendar_heatmap(
    surface: SSVISurface,
    k_grid: np.ndarray,
    T_grid: np.ndarray,
    show_min_pin: bool = False,
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
                colorbar_title="w_T = \u2202w/\u2202T",
            )
        ]
    )

    if show_min_pin:
        min_idx = np.unravel_index(np.argmin(w_T_mesh), w_T_mesh.shape)
        min_T = T_grid[min_idx[0]]
        min_k = k_grid[min_idx[1]]
        min_w_T = w_T_mesh[min_idx]

        fig.add_trace(
            go.Scatter(
                x=[min_k],
                y=[min_T],
                mode="markers+text",
                name=f"Min w_T = {min_w_T:.4f}",
                text=[f" Min w_T={min_w_T:.4f}"],
                textposition="top right",
                marker=dict(color="orange", size=11, symbol="diamond", line=dict(color="black", width=1.5)),
            )
        )

    fig.update_layout(
        title="Calendar Derivative w_T(k, T) = \u2202w/\u2202T (Non-negativity w_T \u2265 0 guarantees no calendar arbitrage)",
        xaxis_title="Log-Forward Moneyness k = ln(K/F)",
        yaxis_title="Maturity T (Years)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_density_and_cdf(density: DensityResult) -> go.Figure:
    """Create interactive plot of risk-neutral density q_T(K) and CDF."""
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
            line=dict(color="#2ca02c", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.15)",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=cdf,
            mode="lines",
            name="Cumulative CDF Q(K)",
            line=dict(color="#1f77b4", width=2, dash="dash"),
            yaxis="y2",
        )
    )

    # Vertical line at forward price
    fig.add_vline(
        x=density.forward,
        line_width=1.5,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Forward = {density.forward:.2f}",
    )

    fig.update_layout(
        title=f"Breeden-Litzenberger Risk-Neutral Density (T={density.expiry:.2f}Y, Total Mass={density.total_mass:.4f})",
        xaxis_title="Strike K",
        yaxis=dict(title="Density q(K)", side="left"),
        yaxis2=dict(title="Cumulative CDF", side="right", overlaying="y", range=[0.0, 1.05]),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_local_vol_surface(
    local_vol: LocalVolResult,
    colorscale: str = "Cividis",
) -> go.Figure:
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
                colorscale=colorscale,
                colorbar_title="Local Vol (%)",
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="magenta", project_z=True)
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Dupire Local Volatility Surface \u03c3_loc(K, T) ({local_vol.valid_fraction*100:.1f}% valid cells)",
        scene=dict(
            xaxis_title="Log-Forward Moneyness k",
            yaxis_title="Maturity T (Years)",
            zaxis_title="Local Vol (%)",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.2)),
        ),
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig


def plot_ablation_comparison_smiles(
    quotes_slice: List[OptionQuote],
    ssvi_surface: SSVISurface,
    raw_svi_slice: Any,
    flat_vol: float,
    expiry: float,
) -> go.Figure:
    """Multi-model smile comparison on a single expiry slice: SSVI vs Raw SVI vs Flat Black vs Market."""
    fig = go.Figure()
    F = quotes_slice[0].forward if quotes_slice and quotes_slice[0].forward is not None else 100.0
    D = quotes_slice[0].discount_factor if quotes_slice and quotes_slice[0].discount_factor is not None else 1.0

    k_obs = []
    iv_obs = []
    for q in quotes_slice:
        k = q.log_moneyness if q.log_moneyness is not None else np.log(q.strike / F)
        inv = implied_volatility_brent(q.mid, F, q.strike, D, expiry, q.option_type)
        if inv.success and inv.iv is not None:
            k_obs.append(k)
            iv_obs.append(inv.iv * 100.0)

    # Market scatter
    fig.add_trace(
        go.Scatter(
            x=k_obs,
            y=iv_obs,
            mode="markers",
            name="Market Mid Quotes",
            marker=dict(color="#333333", size=7, symbol="circle"),
        )
    )

    k_dense = np.linspace(min(k_obs) - 0.15, max(k_obs) + 0.15, 100)

    # 1. SSVI curve
    w_ssvi = ssvi_surface.total_variance(k_dense, expiry)
    iv_ssvi = np.sqrt(w_ssvi / expiry) * 100.0
    fig.add_trace(
        go.Scatter(
            x=k_dense,
            y=iv_ssvi,
            mode="lines",
            name="SSVI (Arbitrage-Free)",
            line=dict(color="#1f77b4", width=3),
        )
    )

    # 2. Raw SVI curve
    if raw_svi_slice is not None:
        w_raw = np.asarray(raw_svi_slice.total_variance(k_dense))
        iv_raw = np.sqrt(np.maximum(w_raw, 1e-12) / expiry) * 100.0
        fig.add_trace(
            go.Scatter(
                x=k_dense,
                y=iv_raw,
                mode="lines",
                name="Independent Raw SVI",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

    # 3. Flat Black line
    fig.add_trace(
        go.Scatter(
            x=[min(k_dense), max(k_dense)],
            y=[flat_vol * 100.0, flat_vol * 100.0],
            mode="lines",
            name=f"Flat Black (\u03c3={flat_vol*100:.1f}%)",
            line=dict(color="#7f7f7f", width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title=f"Model Ablation Smile Comparison (T = {expiry:.2f} Y)",
        xaxis_title="Log-Forward Moneyness k = ln(K/F)",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
