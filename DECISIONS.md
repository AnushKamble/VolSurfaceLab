# DECISIONS.md — Key Architectural & Mathematical Decisions

## ADR-01: SSVI vs Raw SVI
- **Context:** The original specification briefly conflated quasi-explicit Raw SVI calibration with multi-expiry SSVI.
- **Decision:** Regularised power-law SSVI is the sole primary model. Independent Raw SVI slice fitting is strictly an ablation/baseline to demonstrate how independent slice fits can lead to calendar crossings or unconstrained wing curvature.
- **Consequences:** Multi-expiry calibration jointly fits all maturities using shared parameters $(\rho, \eta, \gamma)$ and a monotonically constrained knot vector $\mathbf{\theta}$.

## ADR-02: Constrained Optimizer Choice
- **Context:** SciPy's Levenberg-Marquardt implementation (`method="lm"`) does not support box bounds.
- **Decision:** Use Trust Region Reflective (`method="trf"`) in `scipy.optimize.least_squares` paired with smooth unconstrained parameter transforms (e.g. sigmoid, softplus, tanh) to ensure that optimizer iterates never leave the admissible parameter space.
- **Consequences:** Guaranteed satisfaction of parameter bounds during and after optimisation; no post-hoc parameter clipping or fabricated solutions.

## ADR-03: SVI Sufficient No-Arbitrage Conditions & Eta Cap
- **Context:** Butterfly arbitrage freedom requires $\theta \phi(\theta)(1+|\rho|) < 4$ and $\theta \phi(\theta)^2 (1+|\rho|) \le 4$.
- **Decision:** For $\phi(\theta) = \frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}}$ with $0 < \gamma \le 1/2$, the maximum of $\theta \phi(\theta)^2$ is analytically derived as $\eta^2 H_{max}(\gamma)$ where $H_{max}(\gamma) = \frac{(1-2\gamma)^{1-2\gamma}}{(2-2\gamma)^{2-2\gamma}}$ for $0 < \gamma < 1/2$ and $H_{max}(1/2) = 1$. The admissible upper bound $\eta_{cap}$ is enforced as:
  $$\eta_{cap} = (1 - \varepsilon_{margin}) \min\left( \frac{4}{1+|\rho|}, \frac{2}{\sqrt{H_{max}(\gamma)(1+|\rho|)}} \right)$$
  with default safety margin $\varepsilon_{margin} = 0.01$.
- **Consequences:** The calibrated SSVI surface is guaranteed to be certified by construction, with affirmative theorem-condition verification.

## ADR-04: Analytic vs Numerical Derivatives
- **Context:** Numerical finite differences for strike derivatives ($w_k, w_{kk}$) can suffer from cancellation error in the wings and near ATM.
- **Decision:** All production derivatives ($w_k, w_{kk}, w_\theta, w_T$) are strictly evaluated via exact closed-form analytic formulas. Finite differences are used exclusively in unit tests as independent verification benchmarks.
- **Consequences:** Stable, high-precision Durrleman $g(k, T)$ butterfly function, density extraction, and Dupire local volatility.

## ADR-05: Strict European Exercise-Style Gate
- **Context:** Market data frequently mixes American-style options (e.g. SPY, equity options) with European index options (e.g. SPX). Applying Black forward inversion to American options introduces significant early-exercise bias.
- **Decision:** Enforce a strict exercise-style validation gate. Contracts marked `american` or `unknown` are rejected in strict mode with explicit reason codes (`REJECTED_EXERCISE_STYLE_AMERICAN`, `REJECTED_EXERCISE_STYLE_UNKNOWN`).
- **Consequences:** Ingestion and calibration integrity are preserved without silent model violations.

## ADR-06: Local Volatility Failure Policy
- **Context:** Dupire's formula $\sigma_{loc}^2 = w_T / g$ can encounter zero or negative denominators ($g \le 0$) or negative numerators ($w_T < 0$).
- **Decision:** No silent artificial clipping (e.g. clipping to [0.02, 0.80]) or ad-hoc Gaussian smoothing. Invalid or ill-conditioned cells are explicitly masked and tagged with granular reason codes (`ILL_CONDITIONED_DENOMINATOR`, `CALENDAR_VIOLATION_NEGATIVE_NUMERATOR`, `INVALID_LOCAL_VARIANCE_NEGATIVE`).
- **Consequences:** Uncompromised mathematical honesty and explicit model-risk communication.
