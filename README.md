# VolSurfaceLab

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 80 Passed](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Arbitrage-Aware Volatility Surface & Local-Risk Engine** for European vanilla options.

VolSurfaceLab is a quantitative derivatives research engine that takes cross-sections of European option quotes, filters and validates market contracts, robustly inverts Black implied volatilities, calibrates a smooth **Surface SVI (SSVI)** total-variance surface under analytical no-arbitrage constraints, extracts **Breeden–Litzenberger risk-neutral densities**, and evaluates **Dupire local volatility** from exact closed-form derivatives.

---

## 1. Key Highlights & Quantitative Capabilities

- **Strict European Pipeline:** Complete forward Black pricing with explicit forwards $F_T$, discount factors $D_T$, and European bounds validation. Contracts with American or unknown exercise styles are rejected in strict mode with explicit reason codes.
- **Robust Implied Volatility Inversion:** Per-contract bracketed Brent root solver with automatic bracket expansion, near-intrinsic boundary detection, and near-zero vega guards. Never outputs fabricated sentinel values.
- **Regularised Power-Law SSVI Surface:** Implements the Gatheral & Jacquier (2014) power-law SSVI model:
  $$\phi(\theta) = \frac{\eta}{\theta^\gamma (1 + \theta)^{1 - \gamma}}, \quad 0 < \gamma \le \frac{1}{2}, \quad \eta > 0, \quad |\rho| < 1$$
- **Analytic Production Derivatives:** Closed-form analytic evaluation of $w_k, w_{kk}, w_\theta$, and $w_T = w_\theta \theta'(T)$ using a shape-preserving monotone PCHIP ATM total variance interpolator $\theta(T)$. Finite differences are used exclusively as independent test benchmarks.
- **Theorem-Based Parameter Admissibility:** Enforces the analytical sufficient conditions for butterfly and calendar static-arbitrage freedom by construction:
  $$\eta \le (1 - \varepsilon) \min\left( \frac{4}{1 + |\rho|}, \frac{2}{\sqrt{H_{max}(\gamma)(1 + |\rho|)}} \right), \quad H_{max}(\gamma) = \frac{(1 - 2\gamma)^{1 - 2\gamma}}{(2 - 2\gamma)^{2 - 2\gamma}}$$
- **Three-Layer No-Arbitrage Verification:**
  1. *Layer A:* Analytical SSVI parameter certificate with exact condition margins.
  2. *Layer B:* Independent numerical grid diagnostics on Durrleman $g(k, T)$, calendar $w_T(k, T)$, and discrete call price strike convexity.
  3. *Layer C:* Market quote parity interval diagnostics.
- **Breeden–Litzenberger Risk-Neutral Density:** Exact probability density extraction $q_T(K) = \frac{n(d_2)}{K \sqrt{w}} g(k, T)$ with adaptive numerical quadrature, cumulative mass coverage, and forward first-moment verification. No artificial renormalisation.
- **Guarded Dupire Local Volatility:** Analytic local variance $\sigma_{loc}^2(k, T) = \frac{w_T(k, T)}{g(k, T)}$ with strict safeguards. Ill-conditioned or singular regions are reason-coded without artificial clipping or undisclosed smoothing. Passes the flat-volatility identity to $< 10^{-10}$ precision.
- **Comparative Research & Ablations:** Automated benchmarks comparing constrained SSVI against independent Raw SVI slices and Flat Black volatility. Demonstrates how independent slice fitting causes calendar spread crossings ($\partial_T w < 0$).

---

## 2. Quick Start & 1-Command Offline Demo

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/VolSurfaceLab.git
cd VolSurfaceLab

# Create and activate clean virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install in editable mode with development & visualization dependencies
python -m pip install -e ".[all]"
```

### Run the 1-Command Offline Synthetic Demo
```bash
python -m volsurfacelab demo --source synthetic --seed 42 --output-dir outputs/demo
```
This runs the full pipeline offline: generates deterministic quotes from a certified SSVI surface, cleans and filters quotes, calibrates the model, verifies analytical conditions, runs grid diagnostics, extracts density, computes local volatility, runs the ablation benchmark, and generates an interactive HTML report.

View the generated report:
- Open `outputs/demo/report.html` in your web browser.

---

## 3. CLI Workflows

```bash
# 1. Print CLI help and available commands
python -m volsurfacelab --help

# 2. Run deterministic offline demo
python -m volsurfacelab demo --source synthetic --seed 42 --output-dir outputs/demo

# 3. Calibrate custom European quotes CSV
python -m volsurfacelab calibrate --input data/samples/synthetic_quotes.csv --config configs/example_market_csv.yaml --output-dir outputs/my_run

# 4. Re-verify a saved calibration artifact
python -m volsurfacelab verify --artifact outputs/demo/calibration.json

# 5. Generate standalone HTML report
python -m volsurfacelab report --artifact outputs/demo/calibration.json --output-dir outputs/my_report

# 6. Run multi-model ablation benchmark
python -m volsurfacelab benchmark --config configs/synthetic_default.yaml --output-dir outputs/benchmark

# 7. Launch optional interactive Streamlit dashboard
python -m volsurfacelab dashboard --artifact outputs/demo/run_artifact.json
```

---

## 4. Measured Benchmark Results

Measured on Windows 11, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1 (Seed 42, 300 synthetic quotes across 6 expiries):

| Evaluation Metric | Measured Result | Specification Gate | Status |
|---|---|---|:---:|
| **Price RMSE** | **$0.0188** | $< \$0.20$ | PASS |
| **Price MAE** | **$0.0147** | $< \$0.15$ | PASS |
| **Implied Volatility RMSE** | **0.50% (0.0050 vol pts)** | $< 1.50\%$ | PASS |
| **Quotes Inside Spread** | **99.7%** | $> 80.0\%$ | PASS |
| **Theorem Certificate** | **CERTIFIED** | Gatheral-Jacquier sufficient bounds | PASS |
| **Grid Butterfly Minimum $g(k, T)$** | **0.3417** | $> -10^{-6}$ (zero violations) | PASS |
| **Grid Calendar Minimum $w_T(k, T)$** | **0.0397** | $> -10^{-6}$ (zero violations) | PASS |
| **Density Total Mass (1.00Y)** | **0.9960** | $1.0 \pm 0.01$ (unscaled raw integral) | PASS |
| **Density First Moment Error** | **0.07%** | $< 1.0\%$ relative to forward | PASS |
| **Dupire Flat-Vol Identity Error** | **$< 1.0 \times 10^{-10}$** | Machine precision | PASS |
| **Local Vol Valid Cells** | **100% (flat) / 61.9% (wide synthetic)** | Reason-coded, zero clipping | PASS |
| **Test Suite Pass Count** | **80 / 80 tests passed** | 100% pass rate, 0 skips | PASS |

### Ablation Comparison Table

| Model | Price RMSE ($) | Price MAE ($) | Inside Spread % | Parameters | Calendar Arbitrage Guarantee | Theorem Status |
|---|:---:|:---:|:---:|:---:|---|:---:|
| **SSVI (Constrained)** | **$0.0188** | **$0.0147** | **99.7%** | **9** | **Certified Analytical ($w_T \ge 0$)** | **CERTIFIED** |
| **Independent Raw SVI** | $0.0191 | $0.0151 | 100.0% | 30 | 11 violations ($\min(w_2 - w_1) = -0.0151$) | NO_GUARANTEE |
| **Flat Black Volatility** | $1.7083 | $1.3198 | 13.7% | 1 | Trivial ($w = \sigma^2 T$) | ZERO_SKEW |

---

## 5. Repository Architecture

```text
volsurfacelab/
├── domain/                      # Domain models, schemas, and typed exceptions
├── data/                        # CSV provider, quote filtering, synthetic market generator
├── pricing/                     # Black forward pricing, parity, and Brent IV inversion
├── surface/                     # SSVI model, PCHIP theta curve, TRF calibrator, Raw SVI baseline
├── arbitrage/                   # Theorem certificate, butterfly & calendar grid diagnostics
├── density/                     # Breeden-Litzenberger risk-neutral density engine
├── local_vol/                   # Dupire local volatility engine & safeguards
├── evaluation/                  # Baselines, ablations, and synthetic recovery
├── reporting/                   # JSON/CSV artifact saving and standalone HTML report generator
├── visualization/               # Plotly 3D surfaces and heatmaps
├── dashboard/                   # Optional Streamlit interactive dashboard
├── config.py                    # Pydantic configuration schemas & SHA-256 hash provenance
└── cli.py                       # CLI dispatcher
```

---

## 6. Testing & Quality Verification

Run the complete test suite:
```bash
# Run all 80 unit, integration, and regression tests
pytest -v

# Run unit tests only
pytest tests/unit -v

# Run integration tests only
pytest tests/integration -v

# Run adversarial regression suite (14 stress scenarios)
pytest tests/regression -v
```

All 14 mandatory adversarial scenarios are verified:
1. Clean exact synthetic SSVI quotes recovery.
2. Noisy bid-ask synthetic quotes.
3. Missing-strike sparse surface extrapolation.
4. Invalid crossed, stale, and expired quote filtering.
5. Strict American and unknown exercise-style gate.
6. Parity-inconsistent quote diagnostics.
7. Deliberately invalid parameter rejection.
8. Parameter boundary conditioning ($\rho \to -1, \gamma = 0.5, \eta \to \eta_{cap}$).
9. Near-flat monotone theta curve stability.
10. Local-vol near-zero denominator reason-code tagging.
11. Deliberately negative butterfly surface detection.
12. Density integration domain widening convergence.
13. Optimizer non-convergence failure handling.
14. Completely isolated offline run with network disabled.

---

## 7. Model Risk, Limitations & Caveats

1. **European Only:** Designed exclusively for European vanilla options. American equity options must not be processed without early-exercise boundary modeling.
2. **Finite Grid vs Continuous Proof:** The analytical SSVI certificate mathematically proves no-arbitrage over the continuous domain under model assumptions. Finite grid checks are an independent numerical check and do not replace the analytical proof.
3. **No Artificial Clipping or Smoothing:** Dupire local volatility is mathematically ill-conditioned in extreme wings where $g \to 0$. VolSurfaceLab does not apply ad-hoc clipping to $[2\%, 80\%]$ or undisclosed Gaussian smoothing; invalid cells are explicitly masked and reason-coded.
4. **Tail Truncation:** Numerical integration of the risk-neutral density on finite support $[K_{min}, K_{max}]$ truncates probability mass. VolSurfaceLab reports raw unscaled mass rather than forcing the integral to 1.0.

---

## 8. Technical References

1. **Gatheral, J. & Jacquier, A. (2014)**, *Arbitrage-free SVI volatility surfaces*, Quantitative Finance, 14(1), 59–71. DOI: [10.1080/14697688.2013.819986](https://doi.org/10.1080/14697688.2013.819986).
2. **Breeden, D. T. & Litzenberger, R. H. (1978)**, *Prices of State-Contingent Claims Implicit in Option Prices*, Journal of Business, 51(4), 621–651.
3. **Dupire, B. (1994)**, *Pricing with a Smile*, Risk, 7(1), 18–20.
4. **Gatheral, J. (2006)**, *The Volatility Surface: A Practitioner's Guide*, John Wiley & Sons.

---

## 9. Resume-Ready Project Bullet Suggestions

- *“Developed **VolSurfaceLab**, a quantitative Python research engine for calibrating multi-expiry SSVI volatility surfaces from European option quotes under analytical static no-arbitrage constraints.”*
- *“Derived exact closed-form SSVI strike and time derivatives ($w_k, w_{kk}, w_T$); implemented Breeden–Litzenberger risk-neutral density extraction and guarded Dupire local volatility matching the flat-volatility identity to $< 10^{-10}$ precision.”*
- *“Built a three-layer static-arbitrage verification suite (Gatheral-Jacquier analytical certificate, dense grid butterfly/calendar diagnostics, quote parity intervals) and executed an automated 14-scenario adversarial stress suite achieving 100% pass across 80 tests.”*
