"""Layer A: Theorem-based SSVI admissibility certificate."""

from __future__ import annotations

import numpy as np

from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import AdmissibilityReport, SSVIParameters
from volsurfacelab.surface.admissible_parameters import compute_eta_cap, compute_h_max
from volsurfacelab.surface.theta_curve import ThetaCurve


def generate_ssvi_certificate(
    params: SSVIParameters,
    safety_margin: float = 0.0,
    dense_check_points: int = 200,
) -> AdmissibilityReport:
    """Generate an analytical theorem-based admissibility certificate for SSVI parameters.

    Checks:
    1. Parameter domain: rho in (-1, 1), gamma in (0, 0.5], eta > 0.
    2. ATM variance monotonicity: theta_m > 0 and theta_{m+1} > theta_m.
    3. Monotone interpolator: min theta'(T) >= 0 over calibrated domain.
    4. Sufficient Condition 1: max theta*phi(theta)*(1+|rho|) < 4.
    5. Sufficient Condition 2: max theta*phi(theta)^2*(1+|rho|) <= 4.
    6. Calendar derivative ratio: d_theta(theta*phi) / phi <= (1 + sqrt(1 - rho^2)) / rho^2.
    """
    rho = float(params.rho)
    eta = float(params.eta)
    gamma = float(params.gamma)
    abs_rho = abs(rho)

    # 1. Parameter domains & margins
    rho_in_domain = bool(abs_rho < 1.0)
    rho_margin = float(1.0 - abs_rho)

    gamma_in_domain = bool(0.0 < gamma <= 0.5)
    gamma_margin = float(min(gamma, 0.5 - gamma))

    eta_positive = bool(eta > 0.0)
    h_max = float(compute_h_max(gamma))
    eta_cap = float(compute_eta_cap(rho, gamma, safety_margin=safety_margin))
    eta_margin = float(eta_cap - eta)

    # 2. Term structure checks
    thetas = np.asarray(params.theta_knots, dtype=np.float64)
    theta_positive = bool(np.all(thetas > 0.0))
    theta_monotone = bool(np.all(np.diff(thetas) > 0.0))

    # Dense derivative check of PCHIP interpolator
    th_curve = ThetaCurve(params.expiries, params.theta_knots)
    T_dense = np.linspace(params.expiries[0], params.expiries[-1], dense_check_points)
    th_derivs = th_curve.derivative(T_dense)
    min_deriv = float(np.min(th_derivs))
    derivative_monotone = bool(min_deriv >= -1e-12)

    # 3. Sufficient conditions
    # For theta * phi(theta), supremum over theta > 0 is eta
    cond1_val = float(eta * (1.0 + abs_rho))
    cond1_margin = float(4.0 - cond1_val)

    # For theta * phi(theta)^2, supremum over theta > 0 is eta^2 * H_max(gamma)
    cond2_val = float((eta**2) * h_max * (1.0 + abs_rho))
    cond2_margin = float(4.0 - cond2_val)

    # Calendar ratio bound
    cal_bound = (1.0 + np.sqrt(1.0 - rho**2)) / (rho**2) if abs_rho > 1e-10 else float("inf")
    # For regularised power-law, d_theta(theta*phi)/phi = (1-gamma)/(1+theta) <= 1-gamma <= 1
    cal_satisfied = bool((1.0 - gamma) <= cal_bound)

    # Final certificate status
    is_certified = (
        rho_in_domain
        and gamma_in_domain
        and eta_positive
        and eta_margin >= 0.0
        and theta_positive
        and theta_monotone
        and derivative_monotone
        and cond1_margin > 0.0
        and cond2_margin >= 0.0
        and cal_satisfied
    )

    status = (
        StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS
        if is_certified
        else StatusCode.CERTIFICATE_FAILED
    )

    return AdmissibilityReport(
        theorem_status=status,
        rho_in_domain=rho_in_domain,
        rho_margin=rho_margin,
        gamma_in_domain=gamma_in_domain,
        gamma_margin=gamma_margin,
        eta_positive=eta_positive,
        eta_cap=eta_cap,
        eta_margin=eta_margin,
        theta_positive=theta_positive,
        theta_monotone=theta_monotone and derivative_monotone,
        theta_min_derivative=min_deriv,
        h_max=h_max,
        cond1_value=cond1_val,
        cond1_margin=cond1_margin,
        cond2_value=cond2_val,
        cond2_margin=cond2_margin,
        calendar_ratio_bound_satisfied=cal_satisfied,
        details={
            "formula_family": "regularised_power_law_ssvi",
            "spec_section": "5.6",
            "is_certified": is_certified,
        },
    )
