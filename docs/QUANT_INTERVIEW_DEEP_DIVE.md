# VolSurfaceLab: Quantitative Deep-Dive & Interview Master Guide

> **Target Audience**: Quantitative Researchers, Derivatives Strats, Quantitative Developers, and Top-Tier Quant Interviewers (Citadel, Jump Trading, Millennium, Jane Street, Balyasny, Point72, D.E. Shaw, Two Sigma).

---

## Table of Contents
1. [Core Foundations: SVI vs. SSVI](#1-core-foundations-svi-vs-ssvi)
2. [No-Arbitrage Physics: Butterfly, Calendar, and Durrleman's Condition](#2-no-arbitrage-physics-butterfly-calendar-and-durrlemans-condition)
3. [Gatheral & Jacquier (2014) Theorem 4.1: Analytic Admissibility & $H_{\max}(\gamma)$](#3-gatheral--jacquier-2014-theorem-41-analytic-admissibility--h_maxgamma)
4. [Roger Lee's (2004) Moment Formula & Wing Asymptotics](#4-roger-lees-2004-moment-formula--wing-asymptotics)
5. [Breeden–Litzenberger (1978): Risk-Neutral Density, Moments & Closed-Form CDF](#5-breedenlitzenberger-1978-risk-neutral-density-moments--closed-form-cdf)
6. [Dupire (1994) Local Volatility Under Forward Drift $\mu = r - q$](#6-dupire-1994-local-volatility-under-forward-drift-mu--r---q)
7. [Numerical Architecture, Extrapolation & Production Robustness](#7-numerical-architecture-extrapolation--production-robustness)
8. [Tough Quant Interview Whiteboard Questions & Model Solutions](#8-tough-quant-interview-whiteboard-questions--model-solutions)

---

## 1. Core Foundations: SVI vs. SSVI

### 1.1 The Original Raw SVI Model
Introduced by Jim Gatheral in 2004, the Raw Stochastic Volatility Inspired (SVI) slice parameterizes the total implied variance $w(k) = \sigma_{\text{BS}}^2(k) T$ as a hyperbola in log-forward moneyness $k = \ln(K / F)$:
$$w(k; a, b, \rho, m, \sigma) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$$
where $a \in \mathbb{R}$, $b \ge 0$, $|\rho| < 1$, $m \in \mathbb{R}$, $\sigma > 0$.

### 1.2 The Slice-by-Slice Arbitrage Flaw
In practical trading, options trade across multiple expiries $T_1 < T_2 < \dots < T_M$. If an options desk fits Raw SVI independently slice-by-slice:
- Each slice optimizes its own 5 parameters $(a_i, b_i, \rho_i, m_i, \sigma_i)$.
- There is **no mechanism** linking $w(k, T_i)$ and $w(k, T_{i+1})$.
- Inevitably, the total variance curves cross in the wings:
  $$\exists k \in \mathbb{R} : w(k, T_{i+1}) < w(k, T_i) \iff \frac{\partial w}{\partial T} < 0$$
This constitutes a **calendar spread arbitrage** violation: buying the calendar spread generates a negative time value, violating monotonicity of European call prices with respect to maturity for zero-drift underlyings.

### 1.3 Surface SVI (SSVI) Formulation
Gatheral & Jacquier (2014) solved this by reformulating SVI as a **joint 2D continuous surface**:
$$w(k, \theta) = \frac{\theta}{2} \left( 1 + \rho \phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + (1 - \rho^2)} \right)$$
where:
- $\theta = \theta(T) = w(0, T)$ is the ATM total variance at maturity $T$.
- $\rho \in (-1, 1)$ governs the global smile skew.
- $\phi(\theta): \mathbb{R}^+ \to \mathbb{R}^+$ is a continuous function dictating smile curvature across maturities.

Crucially:
$$\partial_T w(k, T) = \frac{\partial w}{\partial \theta} \cdot \theta'(T)$$
If $\theta'(T) \ge 0$ (ATM total variance increases monotonically with tenor) and $\partial_\theta w(k, \theta) \ge 0$ for all $k$, then **calendar arbitrage is ruled out across the entire continuous surface $(k, T) \in \mathbb{R} \times (0, \infty)$**.

---

## 2. No-Arbitrage Physics: Butterfly, Calendar, and Durrleman's Condition

### 2.1 Static Arbitrage Conditions
A European implied volatility surface $w(k, T)$ is free of static arbitrage if and only if:
1. **Calendar Spread Arbitrage Freedom**:
   $$\left(\frac{\partial C}{\partial T}\right)_K \ge 0 \iff \partial_T w(k, T) \ge 0 \quad (\text{when } r = q = 0)$$
2. **Butterfly Spread Arbitrage Freedom**:
   $$\frac{\partial^2 C}{\partial K^2} \ge 0 \quad \forall K > 0$$

### 2.2 Durrleman's Auxiliary Function $g(k, T)$
By differentiating the Black-Scholes call price $C_{\text{BS}}(F, K, w(k, T))$ twice with respect to strike $K$ (where $k = \ln(K/F)$), Durrleman (2004) proved:
$$\frac{\partial^2 C}{\partial K^2} = \frac{n(d_2)}{K \sqrt{w}} g(k, T)$$
where:
$$d_2 = \frac{-k - \frac{1}{2}w}{\sqrt{w}}, \quad n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$$
and the auxiliary function $g(k, T)$ is:
$$g(k, T) = \left(1 - \frac{k w_k}{2 w}\right)^2 - \frac{w_k^2}{4} \left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w_{kk}}{2}$$
with $w_k = \frac{\partial w}{\partial k}$ and $w_{kk} = \frac{\partial^2 w}{\partial k^2}$.

Since $\frac{n(d_2)}{K \sqrt{w}} > 0$ strictly for all $K > 0$ and $w > 0$:
$$\frac{\partial^2 C}{\partial K^2} \ge 0 \iff g(k, T) \ge 0 \quad \forall k \in \mathbb{R}$$
Hence, butterfly arbitrage freedom is **identically equivalent** to the non-negativity of $g(k, T)$.

---

## 3. Gatheral & Jacquier (2014) Theorem 4.1: Analytic Admissibility & $H_{\max}(\gamma)$

### 3.1 Regularised Power-Law SSVI
Gatheral & Jacquier propose the power-law parameterization:
$$\phi(\theta) = \frac{\eta}{\theta^\gamma (1 + \theta)^{1 - \gamma}}, \quad 0 < \gamma \le \frac{1}{2}, \quad \eta > 0$$
This family captures:
- Fast decay of curvature for long expiries: $\phi(\theta) \sim \eta \theta^{-1}$ as $\theta \to \infty$.
- Sub-diffusive term structure of ATM implied volatility skew $\left.\partial_k \sigma_{\text{BS}}\right|_{k=0} \propto \theta^{-\gamma}$.

### 3.2 Gatheral-Jacquier Sufficient Conditions
Theorem 4.1 of Gatheral & Jacquier (2014) establishes that an SSVI slice $w(k, \theta)$ is free of butterfly arbitrage if:
1. **Condition 1**: $\theta \phi(\theta) (1 + |\rho|) < 4$
2. **Condition 2**: $\theta \phi(\theta)^2 (1 + |\rho|) \le 4$

### 3.3 Derivation of the Global Scale Cap $\eta_{\text{cap}}$
To enforce these conditions across **all** $\theta > 0$, we evaluate the suprema over $\theta \in (0, \infty)$:

#### For Condition 1:
$$\theta \phi(\theta) = \eta \left( \frac{\theta}{1 + \theta} \right)^{1 - \gamma}$$
Since $\frac{\theta}{1 + \theta} \in (0, 1)$ for $\theta > 0$, and $1 - \gamma > 0$, the function is strictly increasing, with:
$$\sup_{\theta > 0} \theta \phi(\theta) = \lim_{\theta \to \infty} \eta \left( \frac{\theta}{1 + \theta} \right)^{1 - \gamma} = \eta$$
Hence, Condition 1 requires:
$$\eta (1 + |\rho|) < 4 \implies \eta < \frac{4}{1 + |\rho|}$$

#### For Condition 2:
$$\theta \phi(\theta)^2 = \eta^2 \frac{\theta^{1 - 2\gamma}}{(1 + \theta)^{2 - 2\gamma}} =: \eta^2 f(\theta)$$
To maximize $f(\theta)$ for $\theta > 0$:
$$\ln f(\theta) = (1 - 2\gamma) \ln \theta - (2 - 2\gamma) \ln(1 + \theta)$$
Differentiating with respect to $\theta$:
$$\frac{f'(\theta)}{f(\theta)} = \frac{1 - 2\gamma}{\theta} - \frac{2 - 2\gamma}{1 + \theta} = 0$$
$$(1 - 2\gamma)(1 + \theta) = (2 - 2\gamma)\theta \implies 1 - 2\gamma + (1 - 2\gamma)\theta = (2 - 2\gamma)\theta$$
$$\theta^* = 1 - 2\gamma$$
Notice that $1 + \theta^* = 2 - 2\gamma$.
Evaluating $f(\theta^*)$:
$$H_{\max}(\gamma) := f(\theta^*) = \frac{(1 - 2\gamma)^{1 - 2\gamma}}{(2 - 2\gamma)^{2 - 2\gamma}}$$
- For $\gamma \in (0, 0.5)$, $H_{\max}(\gamma) \in (0.25, 1.0)$.
- In the limit $\gamma \to 0.5$, $\theta^* \to 0$, and $\lim_{\gamma \to 0.5} H_{\max}(\gamma) = 1.0$.

Hence, Condition 2 requires:
$$\eta^2 H_{\max}(\gamma) (1 + |\rho|) \le 4 \implies \eta \le \frac{2}{\sqrt{H_{\max}(\gamma)(1 + |\rho|)}}$$

Combining both conditions yields the **exact analytic scale cap**:
$$\eta_{\text{cap}} = (1 - \varepsilon) \min\left( \frac{4}{1 + |\rho|}, \frac{2}{\sqrt{H_{\max}(\gamma)(1 + |\rho|)}} \right)$$
where $\varepsilon \ge 0$ is a safety margin (default: $0.01$).

---

## 4. Roger Lee's (2004) Moment Formula & Wing Asymptotics

### 4.1 The Moment Formula
Roger Lee (2004) proved that the asymptotic slope of total implied variance in the wings governs the maximal finite moments of the underlying asset:
$$\beta_R = \limsup_{k \to \infty} \frac{w(k, T)}{k} \in [0, 2], \quad \beta_L = \limsup_{k \to -\infty} \frac{w(k, T)}{|k|} \in [0, 2]$$
If $\beta_R < 2$ and $\beta_L < 2$, the critical right moment $p^* = \sup\{p : \mathbb{E}[S_T^{1+p}] < \infty\}$ and left moment $q^* = \sup\{q : \mathbb{E}[S_T^{-q}] < \infty\}$ satisfy:
$$p^* = \frac{(2 - \beta_R)^2}{8 \beta_R}, \quad q^* = \frac{(2 - \beta_L)^2}{8 \beta_L}$$

### 4.2 Equivalence with Gatheral-Jacquier Condition 1
In the SSVI formulation, for a fixed slice $\theta$:
As $k \to +\infty$:
$$z = \sqrt{(\phi k + \rho)^2 + 1 - \rho^2} = \phi k + \rho + O\left(\frac{1}{k}\right)$$
$$w(k, \theta) = \frac{\theta}{2} (1 + \rho \phi k + \phi k + \rho + \dots) \approx \frac{\theta \phi(\theta)(1 + \rho)}{2} k$$
Thus:
$$\beta_R = \frac{\theta \phi(\theta)(1 + \rho)}{2}$$
Similarly, as $k \to -\infty$:
$$\beta_L = \frac{\theta \phi(\theta)(1 - \rho)}{2}$$
For the underlying asset to have finite first forward moment ($\mathbb{E}[S_T] < \infty$, i.e. $p^* > 0$), Roger Lee requires:
$$\beta_R < 2 \iff \frac{\theta \phi(\theta)(1 + \rho)}{2} < 2 \iff \theta \phi(\theta)(1 + \rho) < 4$$
$$\beta_L < 2 \iff \frac{\theta \phi(\theta)(1 - \rho)}{2} < 2 \iff \theta \phi(\theta)(1 - \rho) < 4$$
Since $|\rho| = \max(\rho, -\rho)$, both wing inequalities hold simultaneously if and only if:
$$\theta \phi(\theta)(1 + |\rho|) < 4$$
**This is identically Gatheral-Jacquier Condition 1.**
Gatheral & Jacquier's Condition 1 is therefore **Roger Lee's moment formula condition in disguise**.

---

## 5. Breeden–Litzenberger (1978): Risk-Neutral Density, Moments & Closed-Form CDF

### 5.1 Risk-Neutral Density
By Breeden & Litzenberger (1978), the risk-neutral probability density function $q_T(K)$ of the terminal underlying price $S_T$ under the forward measure $\mathbb{Q}^T$ is:
$$q_T(K) = \left. \frac{\partial^2 C_F(K, T)}{\partial K^2} \right|_{K}$$
where $C_F(K, T)$ is the undiscounted forward call price.

Using Durrleman's auxiliary formula:
$$q_T(K) = \frac{n(d_2)}{K \sqrt{w(k, T)}} g(k, T)$$
where $k = \ln(K / F_T)$ and $d_2 = \frac{-k - \frac{1}{2}w}{\sqrt{w}}$.

### 5.2 Exact Analytic Closed-Form CDF
The cumulative distribution function $Q_T(K) = \mathbb{P}^{\mathbb{Q}^T}(S_T \le K)$ can be evaluated analytically without numerical quadrature:
$$C_F(K, T) = \int_K^\infty (u - K) q_T(u) du$$
$$\frac{\partial C_F}{\partial K} = - \int_K^\infty q_T(u) du = - (1 - Q_T(K)) = Q_T(K) - 1$$
Therefore:
$$Q_T(K) = 1 + \frac{\partial C_F}{\partial K}$$

Evaluating $\frac{\partial C_F}{\partial K}$ in the Black forward model with strike-dependent total variance $w(k, T)$:
$$\frac{\partial C_F}{\partial K} = -N(d_2) + \frac{n(d_2)}{2 \sqrt{w}} w_k$$
Substituting gives the **exact closed-form CDF**:
$$Q_T(K) = 1 - N(d_2) + \frac{n(d_2)}{2 \sqrt{w}} w_k = N(-d_2) + \frac{n(d_2)}{2 \sqrt{w}} w_k$$
Properties:
- $\lim_{K \to 0^+} Q_T(K) = 0$
- $\lim_{K \to \infty} Q_T(K) = 1$
- $\frac{d Q_T}{d K} = q_T(K) = \frac{n(d_2)}{K \sqrt{w}} g(k, T) \ge 0 \iff g(k, T) \ge 0$

### 5.3 Risk-Neutral Distribution Moments
VolSurfaceLab extracts the complete statistical profile of the risk-neutral distribution:
1. **Total Mass**: $M_0 = \int_0^\infty q_T(K) dK \equiv 1.0$
2. **First Moment (Martingale Expectation)**: $\mu_1 = \int_0^\infty K q_T(K) dK \equiv F_T$
3. **Variance**: $\sigma_{\text{RN}}^2 = \int_0^\infty (K - \mu_1)^2 q_T(K) dK$
4. **Skewness**: $\mathcal{S}_{\text{RN}} = \frac{1}{\sigma_{\text{RN}}^3} \int_0^\infty (K - \mu_1)^3 q_T(K) dK$
5. **Excess Kurtosis**: $\mathcal{K}_{\text{RN}} = \frac{1}{\sigma_{\text{RN}}^4} \int_0^\infty (K - \mu_1)^4 q_T(K) dK - 3.0$

---

## 6. Dupire (1994) Local Volatility Under Forward Drift $\mu = r - q$

### 6.1 Dupire's PDE in Cash vs. Forward Coordinates
Bruno Dupire (1994) derived the local volatility equation for discounted call prices $C(K, T) = e^{-\int_0^T r(s) ds} \mathbb{E}[(S_T - K)^+]$:
$$\frac{\partial C}{\partial T} = -q(T) C - (r(T) - q(T)) K \frac{\partial C}{\partial K} + \frac{1}{2} \sigma_{\text{loc}}^2(K, T) K^2 \frac{\partial^2 C}{\partial K^2}$$

In terms of undiscounted forward call prices $c(F_T, K, T) = e^{\int_0^T r(s) ds} C(K, T)$ where $F_T = S_0 e^{\int_0^T (r(s) - q(s)) ds}$:
$$\frac{\partial c}{\partial T} = (r(T) - q(T)) \left[ c - K \frac{\partial c}{\partial K} \right] + \frac{1}{2} \sigma_{\text{loc}}^2(K, T) K^2 \frac{\partial^2 c}{\partial K^2}$$

### 6.2 The Beautiful Drift Invariance of Dupire Local Volatility in Forward Log-Moneyness
Expressing $c(K, T) = C_{\text{BS}}(F_T, K, w(k, T))$ where $k = \ln(K / F_T)$ is log-forward moneyness and $w(k, T)$ is total implied variance.
Let $\mu(T) = r(T) - q(T)$. Differentiating holding strike $K$ fixed:
$$\left( \frac{\partial c}{\partial T} \right)_K = \frac{\partial C_{\text{BS}}}{\partial F} \frac{d F_T}{d T} + \frac{\partial C_{\text{BS}}}{\partial w} \left( \left(\frac{\partial w}{\partial T}\right)_k + \frac{\partial w}{\partial k} \left(\frac{\partial k}{\partial T}\right)_K \right)$$
Since $k = \ln K - \ln F_T$, we have $\left(\frac{\partial k}{\partial T}\right)_K = -\frac{1}{F_T} \frac{d F_T}{d T} = -\mu(T)$.
And $\frac{\partial C_{\text{BS}}}{\partial F} = N(d_1)$ with $\frac{d F_T}{d T} = \mu(T) F_T$.
Therefore:
$$\left( \frac{\partial c}{\partial T} \right)_K = \mu(T) F_T N(d_1) + \frac{\partial C_{\text{BS}}}{\partial w} \left( \left(\frac{\partial w}{\partial T}\right)_k - \mu(T) \frac{\partial w}{\partial k} \right)$$

Now consider the drift term on the right-hand side of Dupire's PDE:
$$\mu(T) \left[ c - K \frac{\partial c}{\partial K} \right]$$

**Crucial Multivariable Step (Chain Rule on $w$)**:
Recall $c(K, T) = C_{\text{BS}}(F_T, K, w(k, T))$. Since $k = \ln(K / F_T)$, differentiating holding $T$ fixed yields:
$$\frac{\partial c}{\partial K} = \left. \frac{\partial C_{\text{BS}}}{\partial K} \right|_w + \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial K} = -N(d_2) + \frac{\partial C_{\text{BS}}}{\partial w} \frac{1}{K} \frac{\partial w}{\partial k}$$
Multiplying by $K$:
$$K \frac{\partial c}{\partial K} = -K N(d_2) + \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k}$$
Substituting this back into $\left[ c - K \frac{\partial c}{\partial K} \right]$:
$$c - K \frac{\partial c}{\partial K} = \left( F_T N(d_1) - K N(d_2) \right) - \left( -K N(d_2) + \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k} \right) = F_T N(d_1) - \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k}$$
Multiplying by $\mu(T)$:
$$\mu(T) \left[ c - K \frac{\partial c}{\partial K} \right] = \mu(T) F_T N(d_1) - \mu(T) \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k}$$

**Now Subtract the PDE Drift Term from $\left( \frac{\partial c}{\partial T} \right)_K$**:
$$\left( \frac{\partial c}{\partial T} \right)_K - \mu(T) \left[ c - K \frac{\partial c}{\partial K} \right] = \left[ \mu(T) F_T N(d_1) + \frac{\partial C_{\text{BS}}}{\partial w} \left( \left(\frac{\partial w}{\partial T}\right)_k - \mu(T) \frac{\partial w}{\partial k} \right) \right] - \left[ \mu(T) F_T N(d_1) - \mu(T) \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k} \right]$$
Notice the exact cancellations:
1. $\mu(T) F_T N(d_1) - \mu(T) F_T N(d_1) = 0$
2. $-\mu(T) \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k} - \left( -\mu(T) \frac{\partial C_{\text{BS}}}{\partial w} \frac{\partial w}{\partial k} \right) = 0$

**EVERY SINGLE DRIFT TERM CANCELS IDENTICALLY.**
What remains on the left side is simply:
$$\frac{\partial C_{\text{BS}}}{\partial w} \left(\frac{\partial w}{\partial T}\right)_k$$
And on the right side:
$$\frac{1}{2} \sigma_{\text{loc}}^2(K, T) K^2 \frac{\partial^2 c}{\partial K^2} = \sigma_{\text{loc}}^2(K, T) \frac{\partial C_{\text{BS}}}{\partial w} g(k, T)$$
Dividing both sides by $\frac{\partial C_{\text{BS}}}{\partial w} g(k, T)$ yields Jim Gatheral's exact formula:
$$\sigma_{\text{loc}}^2(k, T) = \frac{\left(\frac{\partial w}{\partial T}\right)_k}{g(k, T)}$$

### 6.3 Quant Whiteboard Takeaway: Spot vs. Forward Moneyness Pitfall
A classic interview pitfall is confusing **spot log-moneyness** $y = \ln(K / S_0)$ with **forward log-moneyness** $k = \ln(K / F_T)$:
- In spot coordinates $y$, the coordinate frame is stationary, so Dupire's PDE includes convective drift $(r-q) K \partial_K C$.
- In forward coordinates $k$, the coordinate frame moves at the forward drift rate $\mu(T) = r(T) - q(T)$. The convective derivative $-\mu \partial_k w$ from this moving frame *exactly cancels* the $(r-q) K \partial_K C$ term in Dupire's PDE!
Therefore, in forward log-moneyness coordinates, Gatheral's formula $\sigma_{\text{loc}}^2 = \frac{\partial_T w|_k}{g(k, T)}$ is **exact, unconditional, and universally valid** for arbitrary interest rates and dividend yields. Subtracting $\mu \partial_k w$ in forward coordinates is a fatal double-counting error!

---

## 7. Numerical Architecture, Extrapolation & Production Robustness

### 7.1 Monotone Shape-Preserving PCHIP Term Structure & $C^1$ Extrapolation
ATM total variance $\theta(T)$ is modeled via Piecewise Cubic Hermite Interpolating Polynomials (PCHIP, Fritsch & Carlson 1980):
- Preserves monotonicity: if $\theta_1 < \dots < \theta_M$, then $\theta'(T) \ge 0$ for all $T \in [T_1, T_M]$.
- Avoids the overshoot and spurious oscillation typical of natural cubic splines.

#### Linear and Power-Law Extrapolation Safeguards
Standard SciPy `PchipInterpolator(..., extrapolate=True)` evaluates unconstrained cubic polynomials beyond knots, causing $\theta(T)$ to become negative or non-monotone for $T > T_{\max}$.
VolSurfaceLab implements **exact $C^1$ smooth extrapolation**:
- **Right Extrapolation ($T > T_{\max}$)**:
  $$\theta(T) = \theta_M + \theta'(T_{\max}) \cdot (T - T_{\max}), \quad \theta'(T) = \theta'(T_{\max}) \ge 0$$
- **Left Extrapolation ($T < T_{\min}$)**:
  To eliminate derivative kinks and preserve $C^1$ smoothness while ensuring $\theta(T) > 0, \theta'(T) \ge 0$, and the physical boundary condition $\lim_{T \to 0^+} \theta(T) = 0$, VolSurfaceLab implements a **$C^1$ smooth power-law left extrapolation**:
  $$\theta(T) = \theta_1 \left( \frac{T}{T_{\min}} \right)^\alpha, \quad \alpha = \max\left( \frac{\theta'(T_{\min}) T_{\min}}{\theta_1}, 10^{-4} \right)$$
  $$\theta'(T) = \theta'(T_{\min}) \left( \frac{T}{T_{\min}} \right)^{\alpha - 1}$$
This matches both knot value $\theta(T_{\min}) = \theta_1$ and first derivative $\theta'(T_{\min})$ with zero derivative jump, guaranteeing $C^1$ continuity and calendar-arbitrage-free variance across all $T \in (0, \infty)$.
This guarantees $\theta(T) > 0$ and $\theta'(T) \ge 0$ across the entire positive half-line $T \in (0, \infty)$, preventing negative variance and calendar spread violations.

### 7.2 Bi-Continuous Parameter Mappings
The bounded optimization problem:
$$\min_{\rho, \eta, \gamma, \theta_1, \dots, \theta_M} \sum_{i} \left(\frac{C_{\text{model}} - C_{\text{market}}}{\max(0.5 \text{Spread}_i, \text{Floor})}\right)^2$$
subject to:
$$|\rho| < 1, \quad 0 < \gamma \le 0.5, \quad 0 < \eta < \eta_{\text{cap}}(\rho, \gamma), \quad 0 < \theta_1 < \theta_2 < \dots < \theta_M$$
is mapped bijectively to an unconstrained vector $u \in \mathbb{R}^{3 + M}$:
- $\rho = \tanh(u_\rho)$
- $\gamma = 0.5 \cdot \text{sigmoid}(u_\gamma)$
- $\eta = \eta_{\text{cap}}(\rho, \gamma) \cdot \text{sigmoid}(u_\eta)$
- $\theta_1 = \exp(u_{\theta, 1}), \quad \theta_{m+1} = \theta_m + \exp(u_{\theta, m+1})$

Unconstrained inputs are clipped to stable numerical domains ($u \in [-30, 30]$ for sigmoids, $[-20, 5]$ for increments) to guarantee optimizer stability and prevent floating-point overflow.

---

## 8. Tough Quant Interview Whiteboard Questions & Model Solutions

### Q1: "Why would you choose SSVI over independent Raw SVI slices when fitting an FX or equity options book?"
**Model Answer:**
> "Fitting independent Raw SVI slices fits individual cross-sections well but almost always introduces calendar spread arbitrage crossings ($\partial_T w < 0$) in the wings, especially across short and intermediate expiries or where quote liquidity is uneven. Calendar spread crossings allow counterparties to construct riskless calendar spreads with negative carrying costs.
> SSVI parameterizes the entire 2D volatility surface jointly through an ATM total variance curve $\theta(T)$ and curvature function $\phi(\theta)$. By enforcing $\theta'(T) \ge 0$ via a monotone PCHIP interpolator and bounding parameter $\eta \le \eta_{\text{cap}}$, SSVI guarantees both calendar and butterfly arbitrage freedom across the continuous $(k, T)$ domain by construction."

---

### Q2: "Can you prove that Gatheral & Jacquier's Condition 1 is identical to Roger Lee's Moment Formula?"
**Model Answer:**
> "Yes. In Roger Lee's (2004) Moment Formula, the asymptotic slope of total variance $\beta_R = \lim_{k \to \infty} w(k) / k$ must be strictly less than 2 for the underlying asset to possess a finite first forward moment $\mathbb{E}[S_T] < \infty$.
> In SSVI, evaluating the limit as $k \to \infty$ gives $w(k, \theta) \sim \frac{1}{2} \theta \phi(\theta)(1 + \rho) k$. Therefore, $\beta_R = \frac{1}{2} \theta \phi(\theta)(1 + \rho) < 2 \iff \theta \phi(\theta)(1 + \rho) < 4$.
> Symmetrically, as $k \to -\infty$, $w(k, \theta) \sim \frac{1}{2} \theta \phi(\theta)(1 - \rho) |k|$, so $\beta_L < 2 \iff \theta \phi(\theta)(1 - \rho) < 4$.
> Since $|\rho| = \max(\rho, -\rho)$, requiring both wing slopes to remain strictly below 2 is identical to $\theta \phi(\theta)(1 + |\rho|) < 4$, which is exactly Gatheral & Jacquier's Condition 1."

---

### Q3: "What is Durrleman's condition and how does it relate to Breeden-Litzenberger?"
**Model Answer:**
> "By Breeden & Litzenberger (1978), the risk-neutral density of the underlying price under the forward measure is the second strike derivative of undiscounted call prices: $q_T(K) = \partial^2 C_F / \partial K^2$.
> Differentiating the Black-Scholes formula with respect to strike where $w = w(\ln(K/F), T)$, Durrleman showed that $\partial^2 C_F / \partial K^2 = \frac{n(d_2)}{K \sqrt{w}} g(k, T)$, where $g(k, T)$ is an auxiliary function involving $w$, $w_k$, and $w_{kk}$.
> Because the prefactor $\frac{n(d_2)}{K \sqrt{w}}$ is strictly positive for all $K > 0$ and $w > 0$, the risk-neutral density $q_T(K)$ is non-negative if and only if $g(k, T) \ge 0$. Hence, Durrleman's condition $g(k, T) \ge 0$ is the exact analytical equivalent of butterfly arbitrage freedom."

---

### Q4: "How does Dupire local volatility change when interest rates and dividend yields are non-zero?"
**Model Answer:**
> "A frequent candidate misconception is assuming that non-zero drift $\mu = r - q$ requires an explicit drift subtraction in Gatheral's formula: $\sigma_{\text{loc}}^2 = \frac{\partial_T w - \mu \partial_k w}{g(k, T)}$. This is incorrect for forward log-moneyness $k = \ln(K / F_T)$.
> In cash coordinates, Dupire's equation is: $\partial_T C = -q C - \mu K \partial_K C + \frac{1}{2} \sigma_{\text{loc}}^2 K^2 \partial_{KK} C$.
> When undiscounted option prices are written in terms of total variance $w(k, T)$, holding strike $K$ fixed introduces two drift terms:
> 1. Differentiating $F_T = S_0 e^{\mu T}$ produces $\mu F_T N(d_1)$ from vega and delta.
> 2. The moving moneyness coordinate $k(T) = \ln(K / F_T)$ produces a convective derivative $-\mu \frac{\partial C_{\text{BS}}}{\partial w} \partial_k w$.
> Crucially, when evaluating the PDE drift term $\mu [c - K \partial_K c]$, applying the chain rule to $w(k, T)$ reveals that $K \partial_K c = -K N(d_2) + \frac{\partial C_{\text{BS}}}{\partial w} \partial_k w$.
> Therefore, $\mu [c - K \partial_K c] = \mu F_T N(d_1) - \mu \frac{\partial C_{\text{BS}}}{\partial w} \partial_k w$.
> Subtracting this from $\partial_T c$ causes the convective drift term $-\mu \frac{\partial C_{\text{BS}}}{\partial w} \partial_k w$ to cancel identically with the PDE drift term!
> What remains is purely $\frac{\partial C_{\text{BS}}}{\partial w} \left(\frac{\partial w}{\partial T}\right)_k$.
> Dividing by $\frac{1}{2} K^2 \partial_{KK} c = \frac{\partial C_{\text{BS}}}{\partial w} g(k, T)$ gives:
> $$\sigma_{\text{loc}}^2(k, T) = \frac{\left(\frac{\partial w}{\partial T}\right)_k}{g(k, T)}$$
> Gatheral's formula is therefore invariant to drift in forward log-moneyness coordinates. The $-\mu \partial_k w$ correction is only relevant if one starts with a surface parameterized in stationary spot log-moneyness $y = \ln(K / S_0)$."

---

### Q5: "How do you evaluate the risk-neutral CDF $Q(K)$ without numerical integration?"
**Model Answer:**
> "Since $C_F(K) = \int_K^\infty (u - K) q(u) du$, differentiating once with respect to strike yields $\partial_K C_F = - \int_K^\infty q(u) du = Q(K) - 1$.
> Therefore, $Q(K) = 1 + \partial_K C_F$.
> In the Black forward model with smile $w(k)$:
> $\partial_K C_F = -N(d_2) + \frac{n(d_2)}{2 \sqrt{w}} w_k$.
> Adding 1 yields the exact closed-form risk-neutral CDF:
> $$Q(K) = N(-d_2) + \frac{n(d_2)}{2 \sqrt{w}} w_k$$
> This formula is exact, avoids any discretization or Simpson's quadrature error, and satisfies $Q'(K) = q(K)$ everywhere."
