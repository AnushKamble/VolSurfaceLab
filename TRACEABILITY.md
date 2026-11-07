# TRACEABILITY.md — Requirements to Code & Test Matrix

| Req ID | Specification Section | Description | Implementation Module / Symbol | Test File / Test Name | Evidence Path | Status |
|---|---|---|---|---|---|---|
| REQ-01 | Spec 5.2 | Black Forward European Calls/Puts & Bounds | `pricing.black_forward` | `tests/unit/test_pricing.py` | Unit test execution | PASS |
| REQ-02 | Spec 5.2 | European Put-Call Parity Diagnostics | `pricing.parity` | `tests/unit/test_pricing.py` | Unit test execution | PASS |
| REQ-03 | Spec 5.3 | Robust Bracketed Brent IV Inversion | `pricing.implied_vol` | `tests/unit/test_implied_vol.py` | Unit test execution | PASS |
| REQ-04 | Spec 6.1, 6.2 | CSV Ingestion & Quote Quality Filters | `data.csv_provider`, `data.quality` | `tests/unit/test_data_pipeline.py` | Unit test execution | PASS |
| REQ-05 | Spec 6.4 | Strict European Exercise Style Gate | `data.quality` | `tests/unit/test_data_pipeline.py` | Unit test execution | PASS |
| REQ-06 | Spec 6.5 | Deterministic Synthetic Market Generator | `data.synthetic` | `tests/unit/test_synthetic.py` | Unit test execution | PASS |
| REQ-07 | Spec 5.4 | SSVI Surface & Regularised Power-Law $\phi(\theta)$ | `surface.ssvi` | `tests/unit/test_ssvi.py` | Unit test execution | PASS |
| REQ-08 | Spec 5.4 | Analytic Derivatives $w_k, w_{kk}, w_\theta, w_T$ | `surface.ssvi` | `tests/unit/test_ssvi_derivatives.py` | Unit test execution | PASS |
| REQ-09 | Spec 5.5 | Monotone PCHIP ATM Total Variance Term Structure | `surface.theta_curve` | `tests/unit/test_theta_curve.py` | Unit test execution | PASS |
| REQ-10 | Spec 2.4, 5.6 | Admissible Bounds, $H_{max}(\gamma)$, $\eta_{cap}$ | `surface.admissible_parameters` | `tests/unit/test_admissibility.py` | Unit test execution | PASS |
| REQ-11 | Spec 7.1-7.4 | Constrained Multi-Expiry SSVI Calibration (TRF) | `surface.calibrator` | `tests/integration/test_calibration.py` | Calibration test run | PASS |
| REQ-12 | Spec 8.1 | Layer A: Theorem-Based SSVI Certificate | `arbitrage.ssvi_certificate` | `tests/unit/test_arbitrage_cert.py` | Certificate test run | PASS |
| REQ-13 | Spec 8.2 | Layer B: Independent Numerical Grid Diagnostics ($g \ge 0, w_T \ge 0$) | `arbitrage.butterfly`, `arbitrage.calendar` | `tests/unit/test_grid_diagnostics.py` | Grid diagnostics run | PASS |
| REQ-14 | Spec 8.3 | Layer C: Market Quote Diagnostics & Parity Intervals | `arbitrage.report` | `tests/unit/test_market_diagnostics.py` | Diagnostics run | PASS |
| REQ-15 | Spec 5.9, 9 | Breeden–Litzenberger Risk-Neutral Density Engine | `density.breeden_litzenberger` | `tests/unit/test_density.py` | Density test run | PASS |
| REQ-16 | Spec 5.10, 10 | Dupire Local Volatility Engine & Guards | `local_vol.dupire` | `tests/unit/test_local_vol.py` | Local vol test run | PASS |
| REQ-17 | Spec 10.3 | Flat-Volatility Dupire Identity Validation | `local_vol.dupire` | `tests/unit/test_local_vol.py` | Identity test run | PASS |
| REQ-18 | Spec 11.1-11.3 | Baselines (Flat Black, Raw SVI) & Ablation Study | `evaluation.baselines`, `surface.raw_svi_baseline` | `tests/integration/test_baselines.py` | Baseline ablation run | PASS |
| REQ-19 | Spec 13, 14 | CLI Workflows (`demo`, `calibrate`, `verify`, `report`) | `cli`, `cli_demo` | `tests/integration/test_cli.py` | CLI test execution | PASS |
| REQ-20 | Spec 13.2 | Artefacts Persistence & Standalone HTML Report | `reporting.artifacts`, `reporting.html_report` | `tests/integration/test_cli.py` | Report output check | PASS |
| REQ-21 | Spec 13.3, 15 | Interactive Visualisations (Plotly) & Dashboard | `visualization.figures`, `dashboard.app` | `tests/unit/test_visualization.py` | Figure test execution | PASS |
| REQ-22 | Spec 17 | Adversarial Scenarios & Stress Tests (14 scenarios) | `tests/regression/test_adversarial.py` | `tests/regression/test_adversarial.py` | Full adversarial suite | PASS |
| REQ-23 | Master Prompt 0, 11 | Interactive Quantitative Web Workbench Product | `dashboard.app`, `dashboard.theme` | `tests/unit/test_dashboard.py` | Headless AppTest & unit tests | PASS |
