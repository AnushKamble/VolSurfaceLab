# VolSurfaceLab — Validation, Limitations & Model Risk Caveats

This document describes the validation hierarchy, the critical distinction between analytical theorem certificates and numerical grid diagnostics, model assumptions, and quantitative limitations of **VolSurfaceLab**.

---

## 1. Theorem-Based Certificate vs. Finite Numerical Grid

A core engineering principle of VolSurfaceLab is that **a finite grid scan is a diagnostic, not a continuous mathematical proof**.

### 1.1 Layer A: Analytical SSVI Certificate
- Recomputes Gatheral & Jacquier (2014) sufficient no-arbitrage conditions directly from calibrated parameters $(\rho, \eta, \gamma)$ and the monotone ATM curve $\theta(T)$.
- Evaluates analytical global maxima ($H_{max}(\gamma)$ and $\sup \theta\phi(\theta)$) over the continuum $\theta \in (0, \infty)$.
- Proves mathematically that the fitted surface $w(k, T)$ cannot generate negative call strike curvature ($g(k, T) \ge 0$) or negative calendar variance ($w_T(k, T) \ge 0$) under the model family assumptions.

### 1.2 Layer B: Independent Numerical Grid Diagnostics
- Independently evaluates $g(k, T)$ and $w_T(k, T)$ on a dense discrete grid extending beyond observed market strikes (into the far wings).
- Verifies discrete call price convexity $\frac{\Delta^2 C}{\Delta K^2} \ge 0$.
- Checks that floating-point truncation, catastrophic cancellation in extreme wings, or numerical derivative discretization do not introduce unexpected numerical instabilities.
- Evaluates a 2x grid refinement to test convergence stability.

### 1.3 Layer C: Market Data Arbitrage vs. Model Surface
- An admissible fitted model does **not** imply that raw market quotes were arbitrage-free.
- Market quotes may exhibit synchronous timestamp lag, bid-ask bouncing, or small parity violations.
- VolSurfaceLab reports market quote parity diagnostics independently from fitted surface admissibility.

---

## 2. Model Assumptions

1. **European Exercise:** The forward Black pricing and inversion assume European exercise. Inverting American-style equity or ETF options without early-exercise boundary modeling introduces material volatility distortion, particularly for high dividend yields and deep ITM puts.
2. **Deterministic Rates & Forwards:** Forward prices and discount factors are assumed deterministic per maturity. Stochastic interest rates or basis curve dynamics are not modeled.
3. **PCHIP ATM Term Structure:** The ATM total variance term structure $\theta(T)$ is interpolated using piecewise cubic Hermite interpolating polynomials (PCHIP). While strictly monotone and $C^1$, higher-order derivatives ($d^2\theta / dT^2$) may possess jump discontinuities at knot boundaries.

---

## 3. Quantitative Limitations & Boundary Conditioning

### 3.1 Dupire Local Volatility Singularities
Dupire's local variance $\sigma_{loc}^2(k, T) = \frac{w_T(k, T)}{g(k, T)}$ is a ratio of two derivatives:
- As strikes extend into extreme far wings, both $w_T$ and $g$ approach zero.
- Small floating-point differences can cause the ratio to become ill-conditioned.
- **VolSurfaceLab Policy:** Rather than silently clipping extreme local volatilities to arbitrary bounds (e.g. $[2\%, 80\%]$) or applying undisclosed smoothing, ill-conditioned cells are explicitly masked and tagged with reason codes (`LOCAL_VOL_DENOMINATOR_NEAR_ZERO`, `LOCAL_VOL_DENOMINATOR_NON_POSITIVE`).

### 3.2 Tail Truncation in Risk-Neutral Density
- The Breeden-Litzenberger risk-neutral density $q_T(K)$ has infinite theoretical support $K \in (0, \infty)$.
- Any numerical quadrature evaluates over a finite support $[K_{min}, K_{max}]$.
- **VolSurfaceLab Policy:** The numerical density integral is **never** artificially rescaled to force mass to 1.0. Both the raw integrated mass and the first-moment error are reported honestly with the integration domain bounds.

### 3.3 Maturity Extrapolation
- Extrapolation outside the calibrated domain $[T_1, T_M]$ is disabled by default.
- Extrapolating beyond observed expiries requires explicit user configuration and is treated as a model extension rather than a calibrated fact.
