"""Surface SVI (SSVI) model with regularised power-law phi and exact analytic derivatives."""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np

from volsurfacelab.domain.exceptions import InvalidInputError
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.surface.theta_curve import ThetaCurve


def ssvi_phi(
    theta: Union[float, np.ndarray],
    eta: float,
    gamma: float,
) -> Union[float, np.ndarray]:
    """Evaluate regularised power-law phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))."""
    th = np.asarray(theta, dtype=np.float64)
    if np.any(th <= 0.0):
        raise InvalidInputError("Theta must be strictly positive.")
    if eta <= 0.0 or gamma <= 0.0 or gamma > 0.5:
        raise InvalidInputError(f"Invalid parameters: eta={eta} must be >0, gamma={gamma} must be in (0, 0.5].")

    val = eta / ((th**gamma) * ((1.0 + th)**(1.0 - gamma)))
    return float(val) if th.ndim == 0 else val


def ssvi_phi_prime(
    theta: Union[float, np.ndarray],
    eta: float,
    gamma: float,
) -> Union[float, np.ndarray]:
    """Evaluate derivative phi'(theta) = phi(theta) * (-gamma / theta - (1 - gamma) / (1 + theta))."""
    th = np.asarray(theta, dtype=np.float64)
    phi_val = ssvi_phi(th, eta, gamma)
    factor = -(gamma / th) - ((1.0 - gamma) / (1.0 + th))
    val = phi_val * factor
    return float(val) if th.ndim == 0 else val


class SSVISurface:
    """Continuous arbitrage-aware SSVI total-variance surface w(k, T).

    Parameters
    ----------
    parameters : SSVIParameters
        Calibrated parameters (rho, eta, gamma, expiries, theta_knots).
    theta_curve : Optional[ThetaCurve]
        ATM total-variance term structure. If omitted, built from parameters.
    """

    def __init__(
        self,
        parameters: SSVIParameters,
        theta_curve: ThetaCurve | None = None,
    ):
        self.parameters = parameters
        self.rho = float(parameters.rho)
        self.eta = float(parameters.eta)
        self.gamma = float(parameters.gamma)

        if abs(self.rho) >= 1.0:
            raise InvalidInputError(f"SSVI rho must be in (-1, 1), got {self.rho}.")
        if self.eta <= 0.0:
            raise InvalidInputError(f"SSVI eta must be > 0, got {self.eta}.")
        if self.gamma <= 0.0 or self.gamma > 0.5:
            raise InvalidInputError(f"SSVI gamma must be in (0, 0.5], got {self.gamma}.")

        self.theta_curve = theta_curve or ThetaCurve(
            expiries=parameters.expiries,
            theta_knots=parameters.theta_knots,
        )

    def total_variance_from_theta(
        self,
        k: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute total implied variance w(k, theta)."""
        k_arr = np.asarray(k, dtype=np.float64)
        th_arr = np.asarray(theta, dtype=np.float64)
        is_scalar = (k_arr.ndim == 0 and th_arr.ndim == 0)

        phi_val = ssvi_phi(th_arr, self.eta, self.gamma)
        x = phi_val * k_arr
        z = np.sqrt((x + self.rho)**2 + 1.0 - self.rho**2)

        w = 0.5 * th_arr * (1.0 + self.rho * x + z)
        w = np.maximum(w, 1e-12)
        return float(w) if is_scalar else w

    def total_variance(
        self,
        k: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute total implied variance w(k, T) = w(k, theta(T))."""
        theta = self.theta_curve.evaluate(T)
        return self.total_variance_from_theta(k, theta)

    def implied_volatility(
        self,
        k: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute annualized implied volatility sigma(k, T) = sqrt(w(k, T) / T)."""
        T_arr = np.asarray(T, dtype=np.float64)
        is_scalar = (np.asarray(k).ndim == 0 and T_arr.ndim == 0)
        if np.any(T_arr <= 0.0):
            raise InvalidInputError("Maturity T must be strictly positive.")

        w = self.total_variance(k, T)
        iv = np.sqrt(w / T_arr)
        return float(iv) if is_scalar else iv

    def derivatives_k(
        self,
        k: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute exact analytic strike derivatives w_k and w_kk.

        Returns
        -------
        w_k : dw / dk
        w_kk : d^2w / dk^2
        """
        k_arr = np.asarray(k, dtype=np.float64)
        th_arr = np.asarray(theta, dtype=np.float64)
        is_scalar = (k_arr.ndim == 0 and th_arr.ndim == 0)

        phi_val = ssvi_phi(th_arr, self.eta, self.gamma)
        x = phi_val * k_arr
        z = np.sqrt((x + self.rho)**2 + 1.0 - self.rho**2)

        # w_k = 0.5 * theta * phi * (rho + (x + rho) / z)
        w_k = 0.5 * th_arr * phi_val * (self.rho + (x + self.rho) / z)

        # w_kk = 0.5 * theta * phi^2 * (1 - rho^2) / z^3
        w_kk = 0.5 * th_arr * (phi_val**2) * (1.0 - self.rho**2) / (z**3)

        if is_scalar:
            return float(w_k), float(w_kk)
        return w_k, w_kk

    def derivative_theta(
        self,
        k: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute exact analytic derivative w_theta = dw / d(theta)."""
        k_arr = np.asarray(k, dtype=np.float64)
        th_arr = np.asarray(theta, dtype=np.float64)
        is_scalar = (k_arr.ndim == 0 and th_arr.ndim == 0)

        phi_val = ssvi_phi(th_arr, self.eta, self.gamma)
        phi_p = ssvi_phi_prime(th_arr, self.eta, self.gamma)

        x = phi_val * k_arr
        z = np.sqrt((x + self.rho)**2 + 1.0 - self.rho**2)

        # w_theta = 0.5 * (1 + rho * x + z) + 0.5 * theta * phi'(theta) * k * (rho + (x + rho) / z)
        term1 = 0.5 * (1.0 + self.rho * x + z)
        term2 = 0.5 * th_arr * phi_p * k_arr * (self.rho + (x + self.rho) / z)
        w_th = term1 + term2

        return float(w_th) if is_scalar else w_th

    def derivative_T(
        self,
        k: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute exact time derivative w_T = dw / dT = w_theta * theta'(T)."""
        theta = self.theta_curve.evaluate(T)
        theta_p = self.theta_curve.derivative(T)
        w_th = self.derivative_theta(k, theta)
        w_time = w_th * theta_p
        return float(w_time) if np.asarray(w_time).ndim == 0 else w_time

    def durrleman_g(
        self,
        k: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute analytic Durrleman butterfly function g(k, T).

        g(k, T) = (1 - k * w_k / (2 * w))^2 - (w_k^2 / 4) * (1 / w + 1 / 4) + w_kk / 2
        """
        k_arr = np.asarray(k, dtype=np.float64)
        T_arr = np.asarray(T, dtype=np.float64)
        is_scalar = (k_arr.ndim == 0 and T_arr.ndim == 0)

        theta = self.theta_curve.evaluate(T_arr)
        w = self.total_variance_from_theta(k_arr, theta)
        w_k, w_kk = self.derivatives_k(k_arr, theta)

        # g(k, T) terms
        term1 = (1.0 - (k_arr * w_k) / (2.0 * w))**2
        term2 = (w_k**2 / 4.0) * (1.0 / w + 0.25)
        term3 = 0.5 * w_kk

        g = term1 - term2 + term3
        return float(g) if is_scalar else g

    def evaluate_grid(
        self,
        k_grid: np.ndarray,
        T_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate w, w_k, w_kk, w_T, and g on a 2D rectangular grid (n_T, n_k).

        Returns
        -------
        w_mesh, w_k_mesh, w_kk_mesh, w_T_mesh, g_mesh
        """
        K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
        theta_mesh = self.theta_curve.evaluate(T_mesh)
        theta_p_mesh = self.theta_curve.derivative(T_mesh)

        w_mesh = self.total_variance_from_theta(K_mesh, theta_mesh)
        w_k_mesh, w_kk_mesh = self.derivatives_k(K_mesh, theta_mesh)
        w_th_mesh = self.derivative_theta(K_mesh, theta_mesh)
        w_T_mesh = w_th_mesh * theta_p_mesh

        # Durrleman g
        term1 = (1.0 - (K_mesh * w_k_mesh) / (2.0 * w_mesh))**2
        term2 = (w_k_mesh**2 / 4.0) * (1.0 / w_mesh + 0.25)
        term3 = 0.5 * w_kk_mesh
        g_mesh = term1 - term2 + term3

        return w_mesh, w_k_mesh, w_kk_mesh, w_T_mesh, g_mesh
