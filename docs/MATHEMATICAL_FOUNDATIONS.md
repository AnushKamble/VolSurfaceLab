# VolSurfaceLab — Mathematical Foundations & Technical Derivations

This document details the mathematical models, parameterizations, sufficient static-arbitrage conditions, analytic derivatives, probability densities, and local volatility relations implemented in **VolSurfaceLab**.

---

## 1. Time, Forward, and Total Variance Conventions

### 1.1 Year Fractions & ACT/365F Day Count
For an observation timestamp $t$ and expiry timestamp $T_{exp}$, the year fraction $T > 0$ is defined under the ACT/365 Fixed convention:
$$T = \frac{\text{seconds}(T_{exp} - t)}{365 \times 86400}$$

### 1.2 Coordinate System: Log-Forward-Moneyness
Rather than spot moneyness $\ln(K / S_0)$, the pricing and surface representations are expressed in **log-forward-moneyness**:
$$k = \ln\left(\frac{K}{F_T}\right)$$
where $F_T > 0$ is the forward price for maturity $T$, and $K > 0$ is the strike price.
At the money (ATM), $K = F_T \implies k = 0$.

### 1.3 Implied Total Variance
The total implied variance $w(k, T)$ is linked to the Black annualized implied volatility $\sigma_{\mathrm{imp}}(k, T)$ by:
$$w(k, T) = \sigma_{\mathrm{imp}}(k, T)^2 T$$
with $\sigma_{\mathrm{imp}}(k, T) = \sqrt{\frac{w(k, T)}{T}}$ for $T > 0$.

---

## 2. European Black Forward Pricing & Put-Call Parity

Let $w = \sigma^2 T > 0$. The dimensionless Black d-statistics are:
$$d_1 = \frac{\ln(F / K) + \frac{1}{2}w}{\sqrt{w}} = \frac{-k + \frac{1}{2}w}{\sqrt{w}}$$
$$d_2 = d_1 - \sqrt{w} = \frac{-k - \frac{1}{2}w}{\sqrt{w}}$$

The discounted forward European option prices are:
$$C(F, K, D, T, w) = D \left( F N(d_1) - K N(d_2) \right)$$
$$P(F, K, D, T, w) = D \left( K N(-d_2) - F N(-d_1) \right)$$
where $N(x)$ is the cumulative standard normal distribution function, and $D = D_T \in (0, 1]$ is the discount factor.

### Put-Call Parity
$$C - P = D (F - K)$$

### No-Arbitrage Price Bounds
For calls:
$$D \max(F - K, 0) \le C \le D F$$
For puts:
$$D \max(K - F, 0) \le P \le D K$$

---

## 3. Surface SVI (SSVI) Formulation

### 3.1 Model Definition
The SSVI surface (Gatheral & Jacquier, 2014) specifies total implied variance parameterized by log-forward-moneyness $k$ and ATM total variance $\theta = \theta_T = w(0, T)$:
$$w(k, \theta) = \frac{\theta}{2} \left[ 1 + \rho \phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + 1 - \rho^2} \right]$$

with:
- $\rho \in (-1, 1)$: skew parameter.
- $\theta = \theta(T) > 0$: ATM total variance curve, strictly increasing in $T$ ($\theta'(T) \ge 0$).
- $\phi(\theta) > 0$: skew scaling function.

At the money ($k = 0$):
$$w(0, \theta) = \frac{\theta}{2} \left[ 1 + 0 + \sqrt{\rho^2 + 1 - \rho^2} \right] = \frac{\theta}{2} [ 1 + 1 ] = \theta$$
The ATM identity $w(0, \theta) = \theta$ is exact.

### 3.2 Regularised Power-Law Skew Function
VolSurfaceLab implements the regularised power-law family:
$$\phi(\theta) = \frac{\eta}{\theta^\gamma (1 + \theta)^{1 - \gamma}}$$
with parameters:
$$\eta > 0, \qquad 0 < \gamma \le \frac{1}{2}$$

---

## 4. Analytic Derivatives of the SSVI Surface

Let $x = \phi(\theta) k$ and $z = \sqrt{(x + \rho)^2 + 1 - \rho^2}$.

### 4.1 First Strike Derivative ($w_k$)
$$\frac{\partial x}{\partial k} = \phi(\theta)$$
$$\frac{\partial z}{\partial x} = \frac{x + \rho}{z}$$
$$w_k = \frac{\partial w}{\partial k} = \frac{\theta \phi(\theta)}{2} \left( \rho + \frac{x + \rho}{z} \right)$$

### 4.2 Second Strike Derivative ($w_{kk}$)
Since $z^2 - (x + \rho)^2 = 1 - \rho^2$:
$$w_{kk} = \frac{\partial^2 w}{\partial k^2} = \frac{\theta \phi(\theta)^2}{2} \frac{1 - \rho^2}{z^3}$$
Because $|\rho| < 1$ and $z > 0$, $w_{kk} > 0$ strictly everywhere.

### 4.3 Derivative with respect to ATM Variance ($w_\theta$)
$$\phi'(\theta) = \phi(\theta) \left( -\frac{\gamma}{\theta} - \frac{1 - \gamma}{1 + \theta} \right)$$
$$w_\theta = \frac{\partial w}{\partial \theta} = \frac{1}{2}(1 + \rho x + z) + \frac{\theta}{2} \phi'(\theta) k \left( \rho + \frac{x + \rho}{z} \right)$$

### 4.4 Total Time Derivative ($w_T$)
By chain rule, for a differentiable ATM curve $\theta(T)$:
$$w_T = \frac{\partial w}{\partial T} = w_\theta \cdot \theta'(T)$$

---

## 5. Sufficient No-Arbitrage Conditions & Derivations

### 5.1 Butterfly Spread Conditions
The Gatheral-Jacquier sufficient conditions for butterfly static-arbitrage freedom are:
1. $\theta \phi(\theta) (1 + |\rho|) < 4$ for all $\theta > 0$.
2. $\theta \phi(\theta)^2 (1 + |\rho|) \le 4$ for all $\theta > 0$.

For the regularised power-law function:
$$\theta \phi(\theta) = \eta \left( \frac{\theta}{1 + \theta} \right)^{1 - \gamma} < \eta \quad (\text{since } \theta / (1 + \theta) < 1)$$
Thus, Condition 1 is satisfied if:
$$\eta (1 + |\rho|) < 4 \implies \eta < \frac{4}{1 + |\rho|}$$

For Condition 2:
$$\theta \phi(\theta)^2 = \eta^2 \frac{\theta^{1 - 2\gamma}}{(1 + \theta)^{2(1 - \gamma)}}$$
Let $f(\theta) = \frac{\theta^{1 - 2\gamma}}{(1 + \theta)^{2(1 - \gamma)}}$.
For $0 < \gamma < 1/2$, the first derivative of $\ln f(\theta)$ is:
$$\frac{f'(\theta)}{f(\theta)} = \frac{1 - 2\gamma}{\theta} - \frac{2 - 2\gamma}{1 + \theta} = 0 \implies \theta^* = 1 - 2\gamma$$
Evaluating at the stationary point $\theta^*$:
$$1 + \theta^* = 2 - 2\gamma$$
$$f(\theta^*) = \frac{(1 - 2\gamma)^{1 - 2\gamma}}{(2 - 2\gamma)^{2 - 2\gamma}} \equiv H_{max}(\gamma)$$
For $\gamma = 1/2$, $1 - 2\gamma = 0$, so $f(\theta) = \frac{1}{1 + \theta} \le 1 = H_{max}(1/2)$.

Therefore, Condition 2 holds for all $\theta > 0$ if:
$$\eta^2 H_{max}(\gamma) (1 + |\rho|) \le 4 \implies \eta \le \frac{2}{\sqrt{H_{max}(\gamma) (1 + |\rho|)}}$$

### Admissible $\eta_{cap}$ with Safety Margin $\varepsilon$:
$$\eta_{cap}(\rho, \gamma, \varepsilon) = (1 - \varepsilon) \min\left( \frac{4}{1 + |\rho|}, \frac{2}{\sqrt{H_{max}(\gamma)(1 + |\rho|)}} \right)$$

### 5.2 Calendar Spread Conditions
Calendar arbitrage freedom requires $w_T(k, T) \ge 0$.
The sufficient conditions are:
$$\theta'(T) \ge 0$$
$$0 \le \frac{\partial_\theta(\theta \phi(\theta))}{\phi(\theta)} \le \frac{1 + \sqrt{1 - \rho^2}}{\rho^2}$$
For the regularised power-law function:
$$\frac{\partial_\theta(\theta \phi(\theta))}{\phi(\theta)} = \frac{1 - \gamma}{1 + \theta} \in (0, 1 - \gamma) \subset (0, 1)$$
Since $\frac{1 + \sqrt{1 - \rho^2}}{\rho^2} \ge 1$ for all $|\rho| < 1$, this condition is analytically guaranteed for all admissible $\gamma \in (0, 0.5]$!

---

## 6. Durrleman Butterfly Function
For total implied variance $w(k, T)$:
$$g(k, T) = \left( 1 - \frac{k w_k}{2w} \right)^2 - \frac{w_k^2}{4}\left( \frac{1}{w} + \frac{1}{4} \right) + \frac{w_{kk}}{2}$$
Static butterfly no-arbitrage is satisfied if $g(k, T) \ge 0$ everywhere.

---

## 7. Breeden–Litzenberger Risk-Neutral Density
From the Breeden-Litzenberger (1978) theorem, the risk-neutral density $q_T(K)$ is:
$$q_T(K) = \frac{1}{D_T} \frac{\partial^2 C(K, T)}{\partial K^2}$$
Expressed through the smooth total-variance surface:
$$q_T(K) = \frac{n(d_2)}{K \sqrt{w(k, T)}} g(k, T)$$
where $d_2 = \frac{-k - \frac{1}{2}w}{\sqrt{w}}$, and $n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$.

---

## 8. Dupire Local Volatility
From Dupire's equation under forward-normalised total variance:
$$\sigma_{loc}^2(k, T) = \frac{w_T(k, T)}{g(k, T)}$$
$$\sigma_{loc}(k, T) = \sqrt{\sigma_{loc}^2(k, T)}$$

### Flat-Volatility Identity Check
For a flat total-variance surface $w(k, T) = \sigma_0^2 T$:
- $w_k = 0$, $w_{kk} = 0 \implies g(k, T) = (1 - 0)^2 - 0 + 0 = 1.0$.
- $w_T = \sigma_0^2$.
- $\sigma_{loc}^2 = \frac{\sigma_0^2}{1.0} = \sigma_0^2 \implies \sigma_{loc} \equiv \sigma_0$.
Matches exact flat implied volatility to machine precision.
