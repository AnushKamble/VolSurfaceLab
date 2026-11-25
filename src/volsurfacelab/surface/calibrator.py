"""Joint multi-expiry constrained SSVI calibration using Trust-Region-Reflective least squares."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares

from volsurfacelab.config import CalibrationConfig
from volsurfacelab.domain.exceptions import CalibrationError, StatusCode
from volsurfacelab.domain.models import (
    CalibrationResult,
    OptionQuote,
    SSVIParameters,
)
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance
from volsurfacelab.pricing.implied_vol import implied_volatility_brent
from volsurfacelab.surface.admissible_parameters import (
    params_to_unconstrained,
    unconstrained_to_params,
    verify_ssvi_admissibility,
)
from volsurfacelab.surface.ssvi import SSVISurface
from volsurfacelab.surface.theta_curve import ThetaCurve


def calibrate_ssvi(
    quotes: List[OptionQuote],
    config: Optional[CalibrationConfig] = None,
    provenance_metadata: Optional[Dict[str, any]] = None,
) -> Tuple[SSVISurface, CalibrationResult]:
    """Calibrate joint multi-expiry SSVI surface to a set of European option quotes.

    Parameters
    ----------
    quotes : List[OptionQuote]
        Filtered European option quotes.
    config : Optional[CalibrationConfig]
        Optimizer and loss configuration.
    provenance_metadata : Optional[Dict[str, any]]
        Optional run and configuration hashes for audit trail.

    Returns
    -------
    surface : SSVISurface
    result : CalibrationResult
    """
    calib_cfg = config or CalibrationConfig()

    if len(quotes) < 6:
        raise CalibrationError(
            f"Insufficient data for joint SSVI calibration: {len(quotes)} quotes provided, minimum 6 required.",
            code=StatusCode.CALIBRATION_INSUFFICIENT_DATA,
        )

    # 1. Identify distinct sorted expiries
    expiries = sorted(list({round(q.time_to_maturity, 6) for q in quotes}))
    if len(expiries) < 2:
        raise CalibrationError(
            f"Joint calibration requires at least 2 distinct expiries, found {len(expiries)}.",
            code=StatusCode.CALIBRATION_INSUFFICIENT_DATA,
        )

    # 2. Extract arrays for fast vectorized residuals
    n = len(quotes)
    F_arr = np.zeros(n, dtype=np.float64)
    K_arr = np.zeros(n, dtype=np.float64)
    D_arr = np.zeros(n, dtype=np.float64)
    T_arr = np.zeros(n, dtype=np.float64)
    mid_arr = np.zeros(n, dtype=np.float64)
    spread_scale_arr = np.zeros(n, dtype=np.float64)
    opt_types = []
    k_arr = np.zeros(n, dtype=np.float64)
    market_ivs = np.zeros(n, dtype=np.float64)
    market_iv_valid = np.zeros(n, dtype=bool)

    for i, q in enumerate(quotes):
        F_arr[i] = q.forward if q.forward is not None else 100.0
        K_arr[i] = q.strike
        D_arr[i] = q.discount_factor if q.discount_factor is not None else 1.0
        T_arr[i] = q.time_to_maturity
        mid_arr[i] = q.mid
        opt_types.append(q.option_type)
        k_arr[i] = np.log(K_arr[i] / F_arr[i])

        # Scale denominator: max(half_spread, price_scale_floor)
        half_spread = 0.5 * q.spread
        spread_scale_arr[i] = max(half_spread, calib_cfg.price_scale_floor)

        # Invert market IV for fit reporting
        inv = implied_volatility_brent(
            price=mid_arr[i],
            forward=F_arr[i],
            strike=K_arr[i],
            discount_factor=D_arr[i],
            time_to_maturity=T_arr[i],
            option_type=opt_types[i],
        )
        if inv.success and inv.iv is not None and inv.iv > 0.0:
            market_ivs[i] = inv.iv
            market_iv_valid[i] = True
        else:
            market_ivs[i] = 0.20

    # 3. Initialise ATM total-variance knots theta_m
    init_thetas: List[float] = []
    for T in expiries:
        # Find near-ATM quotes (|k| < 0.20) for expiry T
        mask = (np.abs(T_arr - T) < 1e-5) & (np.abs(k_arr) < 0.20) & market_iv_valid
        if np.any(mask):
            iv_subset = market_ivs[mask]
            med_iv = float(np.median(iv_subset[iv_subset > 0])) if np.any(iv_subset > 0) else 0.20
        else:
            med_iv = 0.20
        th = max(med_iv**2 * T, 0.005)
        init_thetas.append(th)

    # Enforce strict positive monotonicity on initial guess
    for i in range(1, len(init_thetas)):
        if init_thetas[i] <= init_thetas[i - 1]:
            init_thetas[i] = init_thetas[i - 1] + 0.01

    init_rho = calib_cfg.initial_rho
    init_gamma = calib_cfg.initial_gamma
    # Initial eta chosen as 0.6 * eta_cap
    from volsurfacelab.surface.admissible_parameters import compute_eta_cap
    init_eta = 0.6 * compute_eta_cap(init_rho, init_gamma, calib_cfg.safety_margin)

    u_base = params_to_unconstrained(
        rho=init_rho,
        gamma=init_gamma,
        eta=init_eta,
        theta_knots=init_thetas,
        safety_margin=calib_cfg.safety_margin,
    )

    # 4. Residual function
    call_mask = np.array([ot == "call" for ot in opt_types])
    put_mask = ~call_mask

    def residual_fn(u_vec: np.ndarray) -> np.ndarray:
        rho, eta, gamma, thetas = unconstrained_to_params(
            u_vec,
            expiries,
            safety_margin=calib_cfg.safety_margin,
        )
        th_curve = ThetaCurve(expiries, thetas, allow_extrapolation=True)
        # Vectorized evaluation of SSVI total variance
        theta_for_quotes = th_curve.evaluate(T_arr)

        # SSVI total variance
        phi_vals = eta / ((theta_for_quotes**gamma) * ((1.0 + theta_for_quotes)**(1.0 - gamma)))
        x = phi_vals * k_arr
        z = np.sqrt((x + rho)**2 + 1.0 - rho**2)
        w_quotes = 0.5 * theta_for_quotes * (1.0 + rho * x + z)
        w_quotes = np.maximum(w_quotes, 1e-12)

        # Model prices
        model_prices = np.zeros(n, dtype=np.float64)
        if np.any(call_mask):
            model_prices[call_mask] = black_forward_price_from_total_variance(
                forward=F_arr[call_mask],
                strike=K_arr[call_mask],
                discount_factor=D_arr[call_mask],
                total_variance=w_quotes[call_mask],
                option_type="call",
            )
        if np.any(put_mask):
            model_prices[put_mask] = black_forward_price_from_total_variance(
                forward=F_arr[put_mask],
                strike=K_arr[put_mask],
                discount_factor=D_arr[put_mask],
                total_variance=w_quotes[put_mask],
                option_type="put",
            )

        # Scaled residuals
        residuals = (model_prices - mid_arr) / spread_scale_arr
        return residuals

    # 5. Multi-start bounded TRF optimisation
    best_opt_result = None
    best_cost = float("inf")
    all_runs = []

    for seed in calib_cfg.multi_start_seeds:
        rng = np.random.default_rng(seed)
        u_start = u_base.copy()
        if seed != calib_cfg.multi_start_seeds[0]:
            # Small deterministic perturbation for secondary multi-starts
            perturbation = rng.normal(0.0, 0.15, size=len(u_base))
            u_start += perturbation

        try:
            opt_res = least_squares(
                fun=residual_fn,
                x0=u_start,
                method="trf",
                loss=calib_cfg.loss_function,
                f_scale=calib_cfg.f_scale,
                max_nfev=calib_cfg.max_nfev,
                ftol=calib_cfg.ftol,
                xtol=calib_cfg.xtol,
                gtol=calib_cfg.gtol,
            )
            all_runs.append({
                "seed": seed,
                "cost": float(opt_res.cost),
                "status": int(opt_res.status),
                "message": str(opt_res.message),
                "nfev": int(opt_res.nfev),
            })
            # Prioritize converged results (status > 0)
            if opt_res.status > 0 and opt_res.cost < best_cost:
                best_cost = float(opt_res.cost)
                best_opt_result = opt_res
        except Exception as exc:
            all_runs.append({
                "seed": seed,
                "cost": float("inf"),
                "status": -1,
                "message": f"Optimization threw exception: {str(exc)}",
                "nfev": 0,
            })

    if best_opt_result is None:
        # Fallback to run with lowest cost if none strictly converged
        valid_runs = [r for r in all_runs if r["cost"] < float("inf")]
        if not valid_runs:
            raise CalibrationError(
                "SSVI optimization failed to complete on all multi-start seeds.",
                code=StatusCode.CALIBRATION_OPTIMIZER_FAILED,
            )
        raise CalibrationError(
            f"SSVI optimizer failed to converge across all starts: {all_runs}",
            code=StatusCode.CALIBRATION_OPTIMIZER_FAILED,
        )

    # 6. Extract calibrated parameters
    opt_u = best_opt_result.x
    calib_rho, calib_eta, calib_gamma, calib_thetas = unconstrained_to_params(
        opt_u,
        expiries,
        safety_margin=calib_cfg.safety_margin,
    )

    ssvi_params = SSVIParameters(
        rho=calib_rho,
        eta=calib_eta,
        gamma=calib_gamma,
        expiries=expiries,
        theta_knots=calib_thetas,
        interpolator_type="pchip",
    )

    surface = SSVISurface(ssvi_params)

    # 7. Compute detailed fit metrics
    w_model = surface.total_variance(k_arr, T_arr)
    model_prices = np.zeros(n, dtype=np.float64)
    model_prices[call_mask] = black_forward_price_from_total_variance(
        F_arr[call_mask], K_arr[call_mask], D_arr[call_mask], w_model[call_mask], "call"
    )
    model_prices[put_mask] = black_forward_price_from_total_variance(
        F_arr[put_mask], K_arr[put_mask], D_arr[put_mask], w_model[put_mask], "put"
    )

    price_diff = model_prices - mid_arr
    price_rmse = float(np.sqrt(np.mean(price_diff**2)))
    price_mae = float(np.mean(np.abs(price_diff)))

    # Model IVs evaluated against valid inverted market IVs
    model_ivs = np.sqrt(w_model / T_arr)
    if np.any(market_iv_valid):
        iv_diff = model_ivs[market_iv_valid] - market_ivs[market_iv_valid]
        iv_rmse = float(np.sqrt(np.mean(iv_diff**2)))
        iv_mae = float(np.mean(np.abs(iv_diff)))
    else:
        iv_rmse = 0.0
        iv_mae = 0.0

    # Inside bid-ask spread
    bids = np.array([q.bid for q in quotes])
    asks = np.array([q.ask for q in quotes])
    inside_mask = (model_prices >= bids - 1e-4) & (model_prices <= asks + 1e-4)
    pct_inside = float(100.0 * np.mean(inside_mask))

    # Spread-normalized residuals percentiles
    raw_res = price_diff / spread_scale_arr
    pctiles = {
        "p10": float(np.percentile(raw_res, 10)),
        "p25": float(np.percentile(raw_res, 25)),
        "p50": float(np.percentile(raw_res, 50)),
        "p75": float(np.percentile(raw_res, 75)),
        "p90": float(np.percentile(raw_res, 90)),
    }

    # 8. Check admissibility of calibrated parameters
    admiss_check = verify_ssvi_admissibility(
        rho=calib_rho,
        eta=calib_eta,
        gamma=calib_gamma,
        theta_knots=calib_thetas,
        safety_margin=calib_cfg.safety_margin,
    )

    provenance = provenance_metadata or {}
    provenance.update({
        "all_starts": all_runs,
        "selected_status": best_opt_result.status,
        "selected_message": best_opt_result.message,
        "admissibility_status": admiss_check.status_code.value,
    })

    result = CalibrationResult(
        parameters=ssvi_params,
        objective_loss=float(best_opt_result.cost),
        optimizer_status=str(best_opt_result.message),
        iterations=int(best_opt_result.nfev),
        evaluations=int(best_opt_result.nfev),
        cost=float(best_opt_result.cost),
        price_rmse=price_rmse,
        price_mae=price_mae,
        iv_rmse=iv_rmse,
        iv_mae=iv_mae,
        pct_inside_spread=pct_inside,
        residual_percentiles=pctiles,
        provenance=provenance,
    )

    return surface, result
