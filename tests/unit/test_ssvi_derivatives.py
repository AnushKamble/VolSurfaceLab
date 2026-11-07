"""Unit tests comparing analytic SSVI derivatives against independent numerical finite differences."""

import numpy as np
import pytest

from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.surface.ssvi import (
    SSVISurface,
    ssvi_phi,
    ssvi_phi_prime,
)


def test_phi_derivative_vs_central_difference():
    """Verify phi'(theta) matches central finite differences."""
    h = 1e-6
    eta = 1.3
    gamma = 0.35

    theta_vals = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    for th in theta_vals:
        analytic = ssvi_phi_prime(th, eta, gamma)
        fd = (ssvi_phi(th + h, eta, gamma) - ssvi_phi(th - h, eta, gamma)) / (2.0 * h)
        rel_err = abs(analytic - fd) / max(abs(analytic), 1e-4)
        assert rel_err < 1e-6


def test_analytic_w_k_vs_central_difference():
    """Verify analytic w_k matches central finite differences dw/dk."""
    h = 1e-6
    params = SSVIParameters(
        rho=-0.4,
        eta=1.2,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.02, 0.06, 0.12, 0.20],
    )
    surface = SSVISurface(params)

    theta_vals = [0.03, 0.08, 0.15]
    k_vals = [-1.5, -0.8, -0.2, 0.0, 0.2, 0.8, 1.5]

    for th in theta_vals:
        for k in k_vals:
            w_k_analytic, _ = surface.derivatives_k(k, th)
            w_plus = surface.total_variance_from_theta(k + h, th)
            w_minus = surface.total_variance_from_theta(k - h, th)
            w_k_fd = (w_plus - w_minus) / (2.0 * h)

            assert abs(w_k_analytic - w_k_fd) < 1e-6


def test_analytic_w_kk_vs_central_difference():
    """Verify analytic w_kk matches second-order central difference d^2w/dk^2."""
    # For second-order central differences in float64, optimal step size is h ~ eps^(1/4) ~ 1e-4
    h = 1e-4
    params = SSVIParameters(
        rho=-0.35,
        eta=1.1,
        gamma=0.4,
        expiries=[0.25, 0.5, 1.0, 2.0],
        theta_knots=[0.04, 0.07, 0.13, 0.22],
    )
    surface = SSVISurface(params)

    theta_vals = [0.04, 0.10, 0.18]
    k_vals = [-1.2, -0.5, 0.0, 0.5, 1.2]

    for th in theta_vals:
        for k in k_vals:
            _, w_kk_analytic = surface.derivatives_k(k, th)
            w_plus = surface.total_variance_from_theta(k + h, th)
            w_center = surface.total_variance_from_theta(k, th)
            w_minus = surface.total_variance_from_theta(k - h, th)
            w_kk_fd = (w_plus - 2.0 * w_center + w_minus) / (h**2)

            rel_err = abs(w_kk_analytic - w_kk_fd) / max(abs(w_kk_analytic), 1e-4)
            assert rel_err < 1e-5


def test_analytic_w_theta_vs_central_difference():
    """Verify analytic dw/d(theta) matches central finite differences."""
    h = 1e-6
    params = SSVIParameters(
        rho=-0.45,
        eta=1.25,
        gamma=0.3,
        expiries=[0.1, 0.5, 1.0],
        theta_knots=[0.03, 0.07, 0.14],
    )
    surface = SSVISurface(params)

    theta_vals = [0.05, 0.10, 0.20]
    k_vals = [-1.0, -0.3, 0.0, 0.3, 1.0]

    for th in theta_vals:
        for k in k_vals:
            w_th_analytic = surface.derivative_theta(k, th)
            w_plus = surface.total_variance_from_theta(k, th + h)
            w_minus = surface.total_variance_from_theta(k, th - h)
            w_th_fd = (w_plus - w_minus) / (2.0 * h)

            rel_err = abs(w_th_analytic - w_th_fd) / max(abs(w_th_analytic), 1e-4)
            assert rel_err < 1e-5


def test_analytic_w_T_vs_central_difference():
    """Verify analytic time derivative dw/dT matches central finite differences."""
    h = 1e-6
    params = SSVIParameters(
        rho=-0.3,
        eta=1.0,
        gamma=0.45,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.02, 0.05, 0.11, 0.20],
    )
    surface = SSVISurface(params)

    T_vals = [0.3, 0.7, 1.4]
    k_vals = [-0.8, 0.0, 0.8]

    for T in T_vals:
        for k in k_vals:
            w_T_analytic = surface.derivative_T(k, T)
            w_plus = surface.total_variance(k, T + h)
            w_minus = surface.total_variance(k, T - h)
            w_T_fd = (w_plus - w_minus) / (2.0 * h)

            rel_err = abs(w_T_analytic - w_T_fd) / max(abs(w_T_analytic), 1e-4)
            assert rel_err < 1e-5
