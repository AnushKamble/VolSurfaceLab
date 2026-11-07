"""Admissible parameter domain calculation, H_max computation, and parameter mapping."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple, Union
import numpy as np

from volsurfacelab.domain.exceptions import InvalidInputError, StatusCode


class AdmissibilityCheck(NamedTuple):
    """Result of independent algebraic admissibility evaluation."""
    is_admissible: bool
    status_code: StatusCode
    rho: float
    eta: float
    gamma: float
    h_max: float
    eta_cap: float
    eta_margin: float
    cond1_lhs: float
    cond1_margin: float
    cond2_lhs: float
    cond2_margin: float
    calendar_ratio: float
    details: Dict[str, float]


def compute_h_max(gamma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute H_max(gamma) for regularised power-law SSVI.

    H_max(gamma) = ((1 - 2*gamma)^(1 - 2*gamma)) / ((2 - 2*gamma)^(2 - 2*gamma)) for 0 < gamma < 0.5.
    H_max(0.5) = 1.0.
    """
    g = np.asarray(gamma, dtype=np.float64)
    is_scalar = (g.ndim == 0)

    if np.any(g <= 0.0) or np.any(g > 0.5):
        raise InvalidInputError(f"Gamma must be in (0, 0.5], got {g}.")

    h_max = np.ones_like(g, dtype=np.float64)

    # For gamma < 0.5 - 1e-12:
    sub_half = (g < 0.5 - 1e-12)
    if np.any(sub_half):
        g_sub = g[sub_half]
        term1 = 1.0 - 2.0 * g_sub
        term2 = 2.0 - 2.0 * g_sub
        # Stable log-space calculation: exp((1-2g)*ln(1-2g) - (2-2g)*ln(2-2g))
        val = np.exp(term1 * np.log(term1) - term2 * np.log(term2))
        h_max[sub_half] = val

    return float(h_max) if is_scalar else h_max


def compute_eta_cap(
    rho: float,
    gamma: float,
    safety_margin: float = 0.01,
) -> float:
    """Compute conservative sufficient upper bound for eta.

    eta_cap = (1 - safety_margin) * min(4 / (1 + |rho|), 2 / sqrt(H_max * (1 + |rho|))).
    """
    if abs(rho) >= 1.0:
        raise InvalidInputError(f"Rho must be strictly in (-1, 1), got {rho}.")
    if gamma <= 0.0 or gamma > 0.5:
        raise InvalidInputError(f"Gamma must be strictly in (0, 0.5], got {gamma}.")
    if safety_margin < 0.0 or safety_margin >= 0.5:
        raise InvalidInputError(f"Safety margin must be in [0, 0.5), got {safety_margin}.")

    h_max = compute_h_max(gamma)
    abs_rho = abs(rho)

    cap1 = 4.0 / (1.0 + abs_rho)
    cap2 = 2.0 / np.sqrt(h_max * (1.0 + abs_rho))

    eta_cap = (1.0 - safety_margin) * min(cap1, cap2)
    return float(eta_cap)


def verify_ssvi_admissibility(
    rho: float,
    eta: float,
    gamma: float,
    theta_knots: Sequence[float],
    safety_margin: float = 0.0,
) -> AdmissibilityCheck:
    """Independently verify all Gatheral-Jacquier sufficient conditions on parameters."""
    abs_rho = abs(rho)
    h_max = compute_h_max(gamma)
    eta_cap = compute_eta_cap(rho, gamma, safety_margin)

    # Condition 1: theta * phi(theta) * (1 + |rho|) < 4
    # Since theta * phi(theta) <= eta, lhs bound is eta * (1 + |rho|)
    cond1_lhs = float(eta * (1.0 + abs_rho))
    cond1_margin = float(4.0 - cond1_lhs)

    # Condition 2: theta * phi(theta)^2 * (1 + |rho|) <= 4
    # Since theta * phi(theta)^2 <= eta^2 * H_max(gamma), lhs bound is eta^2 * H_max * (1 + |rho|)
    cond2_lhs = float((eta**2) * h_max * (1.0 + abs_rho))
    cond2_margin = float(4.0 - cond2_lhs)

    # Calendar derivative ratio: d_theta(theta * phi) / phi = (1 - gamma) / (1 + theta) in (0, 1)
    # Compatible with (1 + sqrt(1 - rho^2)) / rho^2 >= 1
    cal_bound = (1.0 + np.sqrt(1.0 - rho**2)) / (rho**2) if abs_rho > 1e-12 else float("inf")
    calendar_ratio = 1.0 - gamma

    eta_margin = float(eta_cap - eta)

    # Monotonicity check
    th_arr = np.asarray(theta_knots)
    monotone = bool(np.all(th_arr > 0.0) and np.all(np.diff(th_arr) > 0.0))

    is_admissible = (
        abs_rho < 1.0
        and 0.0 < gamma <= 0.5
        and eta > 0.0
        and cond1_lhs < 4.0
        and cond2_lhs <= 4.0
        and monotone
    )

    status = (
        StatusCode.CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS
        if is_admissible
        else StatusCode.CERTIFICATE_FAILED
    )

    return AdmissibilityCheck(
        is_admissible=is_admissible,
        status_code=status,
        rho=rho,
        eta=eta,
        gamma=gamma,
        h_max=h_max,
        eta_cap=eta_cap,
        eta_margin=eta_margin,
        cond1_lhs=cond1_lhs,
        cond1_margin=cond1_margin,
        cond2_lhs=cond2_lhs,
        cond2_margin=cond2_margin,
        calendar_ratio=calendar_ratio,
        details={
            "abs_rho": abs_rho,
            "calendar_upper_bound": cal_bound,
            "min_theta": float(np.min(th_arr)) if len(th_arr) > 0 else 0.0,
            "max_theta": float(np.max(th_arr)) if len(th_arr) > 0 else 0.0,
        },
    )


# --- Bijective Transformations for Unconstrained/Bounded Optimization ---

def params_to_unconstrained(
    rho: float,
    gamma: float,
    eta: float,
    theta_knots: Sequence[float],
    safety_margin: float = 0.01,
) -> np.ndarray:
    """Map constrained SSVI parameters to unconstrained real vector u."""
    # rho in (-1, 1) via arctanh
    rho_clipped = np.clip(rho, -0.999, 0.999)
    u_rho = np.arctanh(rho_clipped)

    # gamma in (0, 0.5] via logit on 2*gamma
    g_ratio = np.clip(2.0 * gamma, 1e-4, 0.9999)
    u_gamma = np.log(g_ratio / (1.0 - g_ratio))

    # eta in (0, eta_cap) via logit on eta / eta_cap
    eta_cap = compute_eta_cap(rho, gamma, safety_margin)
    eta_ratio = np.clip(eta / eta_cap, 1e-4, 0.9999)
    u_eta = np.log(eta_ratio / (1.0 - eta_ratio))

    # Theta knots via log-increments
    th_arr = np.asarray(theta_knots, dtype=np.float64)
    u_thetas = [np.log(max(th_arr[0], 1e-4))]
    for i in range(1, len(th_arr)):
        inc = max(th_arr[i] - th_arr[i - 1], 1e-4)
        u_thetas.append(np.log(inc))

    return np.array([u_rho, u_gamma, u_eta] + u_thetas, dtype=np.float64)


def unconstrained_to_params(
    u: np.ndarray,
    expiries: Sequence[float],
    safety_margin: float = 0.01,
) -> Tuple[float, float, float, List[float]]:
    """Map unconstrained real vector u back to strictly admissible parameters (rho, eta, gamma, theta_knots)."""
    u_rho, u_gamma, u_eta = u[0], u[1], u[2]
    u_thetas = u[3:]

    # rho in (-1, 1)
    rho = float(np.tanh(u_rho))

    # gamma in (0, 0.5]
    sig_gamma = 1.0 / (1.0 + np.exp(-u_gamma))
    gamma = float(0.5 * sig_gamma)

    # eta in (0, eta_cap)
    eta_cap = compute_eta_cap(rho, gamma, safety_margin)
    sig_eta = 1.0 / (1.0 + np.exp(-u_eta))
    eta = float(eta_cap * sig_eta)

    # theta knots
    th1 = float(np.exp(u_thetas[0]))
    thetas = [th1]
    for inc_u in u_thetas[1:]:
        thetas.append(thetas[-1] + float(np.exp(inc_u)))

    return rho, eta, gamma, thetas
