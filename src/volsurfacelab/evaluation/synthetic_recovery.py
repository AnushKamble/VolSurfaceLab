"""Synthetic ground truth recovery evaluation and parameter identifiability benchmarking."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from volsurfacelab.data.synthetic import SyntheticGroundTruth
from volsurfacelab.domain.models import CalibrationResult
from volsurfacelab.surface.ssvi import SSVISurface


def evaluate_synthetic_recovery(
    calibrated_surface: SSVISurface,
    calibration_result: CalibrationResult,
    ground_truth: SyntheticGroundTruth,
    k_eval_grid: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Measure how accurately the calibration recovered the true synthetic surface.

    Parameters
    ----------
    calibrated_surface : SSVISurface
        Fitted model.
    calibration_result : CalibrationResult
        Metrics and parameters.
    ground_truth : SyntheticGroundTruth
        True parameters used to generate the synthetic data.
    k_eval_grid : Optional[np.ndarray]
        Log-moneyness evaluation points.

    Returns
    -------
    recovery_report : Dict[str, Any]
    """
    k_grid = k_eval_grid if k_eval_grid is not None else np.linspace(-0.8, 0.8, 41)
    expiries = list(ground_truth.expiries)

    # 1. Parameter discrepancies
    rho_err = abs(calibrated_surface.rho - ground_truth.rho)
    eta_err = abs(calibrated_surface.eta - ground_truth.eta)
    gamma_err = abs(calibrated_surface.gamma - ground_truth.gamma)

    # 2. Surface total-variance error on grid (surface equivalence)
    surface_errors = []
    iv_errors = []

    for T, true_th in zip(expiries, ground_truth.theta_knots):
        # True total variance
        from volsurfacelab.data.synthetic import ssvi_total_variance_exact
        w_true = np.array([
            ssvi_total_variance_exact(
                k=k,
                theta=true_th,
                rho=ground_truth.rho,
                eta=ground_truth.eta,
                gamma=ground_truth.gamma,
            )
            for k in k_grid
        ])
        w_calib = calibrated_surface.total_variance(k_grid, T)
        diff = w_calib - w_true
        surface_errors.extend(diff.tolist())

        iv_true = np.sqrt(w_true / T)
        iv_calib = np.sqrt(w_calib / T)
        iv_errors.extend((iv_calib - iv_true).tolist())

    surf_rmse = float(np.sqrt(np.mean(np.array(surface_errors)**2)))
    surf_mae = float(np.mean(np.abs(np.array(surface_errors))))
    iv_recovery_rmse = float(np.sqrt(np.mean(np.array(iv_errors)**2)))
    iv_recovery_mae = float(np.mean(np.abs(np.array(iv_errors))))

    return {
        "true_rho": ground_truth.rho,
        "calib_rho": calibrated_surface.rho,
        "rho_error": rho_err,
        "true_eta": ground_truth.eta,
        "calib_eta": calibrated_surface.eta,
        "eta_error": eta_err,
        "true_gamma": ground_truth.gamma,
        "calib_gamma": calibrated_surface.gamma,
        "gamma_error": gamma_err,
        "surface_total_variance_rmse": surf_rmse,
        "surface_total_variance_mae": surf_mae,
        "iv_recovery_rmse": iv_recovery_rmse,
        "iv_recovery_mae": iv_recovery_mae,
        "comment": "Total-variance RMSE measures true surface recovery independent of parameter trade-offs.",
    }
