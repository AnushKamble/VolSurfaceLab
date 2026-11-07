# CHANGELOG — VolSurfaceLab Implementation History

All notable technical milestones, mathematical derivations, bug fixes, and test results are documented in this changelog.

---

## [0.1.0] — Full Implementation & Adversarial Delivery

### Added
- **Domain Modeling & Core Schemas:**
  - `OptionQuote`, `MarketSnapshot`, `ImpliedVolObservation`, `SSVIParameters`, `CalibrationResult`, `AdmissibilityReport`, `GridDiagnostics`, `DensityResult`, `LocalVolResult`, and `RunArtifact` domain models in `volsurfacelab.domain.models`.
  - Typed exceptions and granular `StatusCode` reason codes in `volsurfacelab.domain.exceptions`.
  - JSON serialization and deserialization helpers in `volsurfacelab.domain.schemas`.
  - Pydantic configuration schemas with SHA-256 provenance hashing in `volsurfacelab.config`.
- **Pricing & Inversion Foundation:**
  - European forward Black call and put pricing, bounds validation, and analytic Black vega in `volsurfacelab.pricing.black_forward`.
  - Put-call parity residual and bid-ask interval overlap diagnostics in `volsurfacelab.pricing.parity`.
  - Robust bracketed Brent root inversion with automatic bracket expansion, near-intrinsic boundary detection, and near-zero vega guards in `volsurfacelab.pricing.implied_vol`.
- **Market Data Pipeline:**
  - Canonical CSV quote importer and exporter with timezone handling in `volsurfacelab.data.csv_provider`.
  - Quantitative quote filtering with strict European exercise style gate and full rejection accounting in `volsurfacelab.data.quality`.
  - Deterministic synthetic option market generator from certified SSVI surfaces with non-leaking ground-truth container in `volsurfacelab.data.synthetic`.
  - Committed synthetic quote sample at `data/samples/synthetic_quotes.csv`.
- **SSVI Surface & Term Structure:**
  - Regularised power-law SSVI model $\phi(\theta) = \frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}}$ in `volsurfacelab.surface.ssvi`.
  - Closed-form production derivatives $w_k, w_{kk}, w_\theta$, and $w_T = w_\theta \theta'(T)$.
  - Monotone shape-preserving PCHIP ATM variance curve $\theta(T)$ and derivative $\theta'(T)$ in `volsurfacelab.surface.theta_curve`.
- **Parameter Admissibility & Multi-Expiry Calibration:**
  - Closed-form $H_{max}(\gamma)$ and conservative sufficient $\eta_{cap}$ bound with safety margin in `volsurfacelab.surface.admissible_parameters`.
  - Bijective unconstrained-to-constrained parameter transformations ensuring admissibility iterates by construction.
  - Joint multi-expiry TRF least-squares optimizer with soft-L1 robust loss and deterministic multi-start initialisation in `volsurfacelab.surface.calibrator`.
- **No-Arbitrage Verification Suite:**
  - Layer A: Analytical Gatheral-Jacquier theorem-condition certificate in `volsurfacelab.arbitrage.ssvi_certificate`.
  - Layer B: Independent numerical grid diagnostics on Durrleman $g(k, T)$, calendar $w_T(k, T)$, discrete call convexity, and refinement factor in `volsurfacelab.arbitrage.butterfly` and `calendar`.
  - Layer C: Market quote parity interval diagnostics and summary reporting in `volsurfacelab.arbitrage.report`.
- **Breeden–Litzenberger Risk-Neutral Density:**
  - Analytic density evaluation $q_T(K) = \frac{n(d_2)}{K \sqrt{w}} g(k, T)$ in `volsurfacelab.density.breeden_litzenberger`.
  - Adaptive Simpson quadrature for total mass and first moment (forward price check). No artificial renormalisation.
- **Dupire Local Volatility Engine:**
  - Analytic local variance $\sigma_{loc}^2 = w_T / g$ and local volatility $\sigma_{loc} = \sqrt{\sigma_{loc}^2}$ in `volsurfacelab.local_vol.dupire`.
  - Safeguards and structured reason codes for singular or invalid cells; zero clipping or smoothing.
  - Flat-volatility identity validation passing to $< 10^{-10}$ precision.
- **Baselines & Ablation Research:**
  - Flat Black volatility baseline in `volsurfacelab.evaluation.baselines`.
  - Independent Raw SVI slice baseline in `volsurfacelab.surface.raw_svi_baseline`.
  - Multi-model ablation comparison and synthetic recovery evaluator in `volsurfacelab.evaluation.synthetic_recovery`.
- **Reporting & Visualization:**
  - Multi-format run persistence (YAML, JSON, CSV) in `volsurfacelab.reporting.artifacts`.
  - Standalone responsive HTML audit report with embedded Plotly figures in `volsurfacelab.reporting.html_report`.
  - Plotly interactive 3D surfaces and heatmaps in `volsurfacelab.visualization.figures`.
  - Optional Streamlit dashboard in `volsurfacelab.dashboard.app`.
- **CLI Workflows:**
  - `demo`, `calibrate`, `verify`, `report`, `benchmark`, `dashboard` commands in `volsurfacelab.cli` and `volsurfacelab.cli_demo`.
- **Adversarial Regression Test Suite:**
  - 14 mandatory adversarial stress test scenarios in `tests/regression/test_adversarial.py`.
  - Complete 80-test automated suite across unit, integration, and regression modules.
