"""Theme and institutional CSS styling for the VolSurfaceLab web workbench."""

from __future__ import annotations

import streamlit as st


def inject_custom_theme() -> None:
    """Inject institutional quant-grade CSS into the Streamlit application."""
    custom_css = """
    <style>
    /* Global layout & typography */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top banner brand header */
    .quant-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 60%, #415a77 100%);
        border: 1px solid #415a77;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .quant-title {
        color: #e0e1dd;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .quant-subtitle {
        color: #778da9;
        font-size: 14px;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 0;
        letter-spacing: 0.2px;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }

    .kpi-card {
        background: #111927;
        border: 1px solid #1f2e46;
        border-radius: 8px;
        padding: 16px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #3b82f6;
    }

    .kpi-card.kpi-green::before { background: #10b981; }
    .kpi-card.kpi-red::before { background: #ef4444; }
    .kpi-card.kpi-amber::before { background: #f59e0b; }
    .kpi-card.kpi-purple::before { background: #8b5cf6; }

    .kpi-label {
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .kpi-sub {
        font-size: 12px;
        color: #64748b;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    .badge-pass {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-fail {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    /* Theory Callout Box */
    .theory-box {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13.5px;
        color: #cbd5e1;
    }

    .theory-box h5 {
        color: #93c5fd;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 15px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
        font-size: 14px;
    }

    /* Button styling */
    .stButton>button {
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    /* Dataframe styling */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12.5px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(
    status_text: str = "Active",
    status_type: str = "pass",
) -> None:
    """Render the top product banner."""
    badge_html = f'<span class="badge badge-{status_type}">{status_text}</span>'
    html = f"""
    <div class="quant-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="quant-title">
                    <span>VolSurfaceLab</span>
                    <span style="font-size: 14px; font-weight: 500; color: #60a5fa; background: rgba(59, 130, 246, 0.15); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.3);">v1.0 Institutional</span>
                </h1>
                <p class="quant-subtitle">
                    Arbitrage-Free SSVI Volatility Surface, Risk-Neutral Density & Dupire Local-Risk Engine
                </p>
            </div>
            <div>
                {badge_html}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(
    label: str,
    value: str,
    sub: str = "",
    status: str = "blue",
) -> str:
    """Return HTML for an institutional KPI card."""
    color_class = f"kpi-{status}" if status in ["green", "red", "amber", "purple"] else ""
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_theory_callout(title: str, text: str) -> None:
    """Render an educational mathematical theory callout block."""
    html = f"""
    <div class="theory-box">
        <h5>{title}</h5>
        <div>{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
