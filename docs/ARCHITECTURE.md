# VolSurfaceLab — Software Architecture & Component Design

This document describes the package architecture, modular design principles, dependency layers, and data flow of **VolSurfaceLab**.

---

## 1. High-Level Architecture & Pipeline Flow

VolSurfaceLab is architected as a strict unidirectional scientific pipeline:

```
[Raw European Quotes]
        │
        ▼
[Data Ingestion & Quality Filters] ──► (QuoteCleaningReport)
        │ (Accepted Quotes)
        ▼
[Black Forward Pricing & Brent IV Inversion] ──► (ImpliedVolObservations)
        │
        ▼
[Constrained Multi-Expiry SSVI Calibration] ──► (SSVIParameters & Fit Metrics)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
[Theorem Certificate (Layer A)] [Grid Diagnostics (Layer B)] [Quote Diagnostics (Layer C)]
        │                             │                             │
        └─────────────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
             [Breeden-Litzenberger Risk-Neutral Density Engine]
                                      │
                                      ▼
                      [Dupire Local Volatility Engine]
                                      │
                                      ▼
                [Run Artifacts Persistence & HTML Reporting]
```

---

## 2. Directory Layout & Module Responsibilities

```
volsurfacelab/
├── domain/                      # Core domain abstractions & exceptions
│   ├── models.py                # Typed domain models (OptionQuote, SSVIParameters, etc.)
│   ├── schemas.py               # Serialization & JSON mapping
│   └── exceptions.py            # Typed exceptions & StatusCode enums
├── data/                        # Ingestion, quality filtering, and synthetic generator
│   ├── base.py
│   ├── csv_provider.py          # Canonical CSV quote parser & exporter
│   ├── quality.py               # Quantitative quote filters & strict European gate
│   ├── provenance.py            # Environment metadata & SHA-256 content hashing
│   ├── synthetic.py             # Deterministic synthetic quote generator
│   └── external_adapter.py      # Optional external market data adapter
├── pricing/                     # European forward Black analytics
│   ├── black_forward.py         # Forward Black calls/puts, bounds, and vega
│   ├── implied_vol.py           # Robust bracketed Brent root inversion
│   └── parity.py                # European put-call parity interval diagnostics
├── surface/                     # Volatility surface models & optimization
│   ├── ssvi.py                  # Regularised power-law SSVI & exact analytic derivatives
│   ├── theta_curve.py           # Shape-preserving monotone PCHIP ATM variance curve
│   ├── admissible_parameters.py # H_max, eta_cap, and bijective parameter transforms
│   ├── calibrator.py            # Joint multi-expiry TRF least-squares solver
│   └── raw_svi_baseline.py      # Independent Raw SVI slice ablation baseline
├── arbitrage/                   # Three-layer no-arbitrage verification suite
│   ├── ssvi_certificate.py      # Layer A: Analytical Gatheral-Jacquier theorem certificate
│   ├── butterfly.py             # Layer B1: Durrleman g(k,T) & discrete call convexity
│   ├── calendar.py              # Layer B2: Calendar derivative w_T(k,T) dense grid
│   └── report.py                # Layer C: Market quote diagnostics & composite report
├── density/                     # Risk-neutral probability extraction
│   └── breeden_litzenberger.py  # Analytic density, quadrature, mass & moment analytics
├── local_vol/                   # Dupire local volatility
│   └── dupire.py                # Analytic local variance w_T / g, guards & flat-vol identity
├── evaluation/                  # Comparative research & ablation suite
│   ├── baselines.py             # Flat Black baseline & multi-model ablation comparator
│   └── synthetic_recovery.py    # Known ground truth recovery benchmarking
├── reporting/                   # Persistence & presentation
│   ├── artifacts.py             # JSON/CSV artifact saving & loading
│   └── html_report.py           # Standalone responsive HTML report generator
├── visualization/               # Interactive figures
│   └── figures.py               # Plotly 3D surfaces, fitted slices, and heatmaps
├── dashboard/                   # Optional UI client
│   └── app.py                   # Streamlit interactive research dashboard
├── config.py                    # Pydantic configuration schemas & hash provenance
├── cli.py                       # Command-line interface dispatcher
├── cli_demo.py                  # CLI pipeline orchestration
├── __init__.py                  # Version & package metadata
└── __main__.py                  # python -m volsurfacelab entry point
```

---

## 3. Key Design Principles

1. **Separation of Mathematics from Presentation:**
   All pricing, surface models, derivatives, densities, and calibration routines are independent of Plotly, Streamlit, and CLI code. They operate on standard NumPy arrays and float primitives.
2. **Strict Parameter Admissibility by Construction:**
   Optimization variables are smoothly transformed via bijective mappings such that iterates can never violate sufficient static-arbitrage bounds during TRF least squares.
3. **Analytic Production Derivatives:**
   Production calculations of Durrleman $g(k, T)$, Breeden-Litzenberger density, and Dupire local variance evaluate exact analytic closed-form derivatives ($w_k, w_{kk}, w_\theta, w_T$). Finite differences are reserved for independent unit testing.
4. **Honest Numerical Safeguards:**
   No silent artificial clipping, arbitrary volatility bounds, or undisclosed Gaussian smoothing filters. Invalid regions are preserved as structured masks with explicit reason codes.
