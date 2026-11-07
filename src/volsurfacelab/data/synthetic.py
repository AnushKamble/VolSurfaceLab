"""Deterministic synthetic option market generator based on certified SSVI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import numpy as np

from volsurfacelab.config import SyntheticGeneratorConfig
from volsurfacelab.domain.models import MarketSnapshot, OptionQuote
from volsurfacelab.pricing.black_forward import black_forward_price_from_total_variance


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """Privileged container holding exact data-generating parameters (not exposed to calibrator)."""
    rho: float
    eta: float
    gamma: float
    expiries: Tuple[float, ...]
    theta_knots: Tuple[float, ...]
    spot: float
    rate: float
    dividend: float
    seed: int


def ssvi_phi_exact(theta: float, eta: float, gamma: float) -> float:
    """Evaluate regularised power-law phi(theta)."""
    return float(eta / ((theta**gamma) * ((1.0 + theta)**(1.0 - gamma))))


def ssvi_total_variance_exact(
    k: float,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> float:
    """Compute exact SSVI total implied variance w(k, theta)."""
    phi_val = ssvi_phi_exact(theta, eta, gamma)
    x = phi_val * k
    z = np.sqrt((x + rho)**2 + 1.0 - rho**2)
    return float(0.5 * theta * (1.0 + rho * x + z))


def generate_synthetic_market(
    config: SyntheticGeneratorConfig,
    valuation_time: datetime | None = None,
) -> Tuple[List[OptionQuote], MarketSnapshot, SyntheticGroundTruth]:
    """Generate deterministic European option quotes from a known admissible SSVI surface.

    Parameters
    ----------
    config : SyntheticGeneratorConfig
        Configuration settings for strikes, maturities, surface parameters, and noise.
    valuation_time : Optional[datetime]
        Snapshot reference valuation time.

    Returns
    -------
    quotes : List[OptionQuote]
        Simulated option quotes with realistic bid/ask spreads.
    snapshot : MarketSnapshot
        Forward and discount curves aligned with the quote set.
    ground_truth : SyntheticGroundTruth
        True parameters held separately for post-calibration benchmark evaluation.
    """
    rng = np.random.default_rng(config.seed)
    val_time = valuation_time or datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    # Term structures
    expiries = sorted(config.expiries)
    forward_curve: Dict[float, float] = {}
    discount_curve: Dict[float, float] = {}
    theta_knots: List[float] = []

    for T in expiries:
        F_T = config.spot * np.exp((config.rate - config.dividend) * T)
        D_T = np.exp(-config.rate * T)
        forward_curve[T] = float(F_T)
        discount_curve[T] = float(D_T)
        # Linear increasing theta curve
        theta_T = config.true_theta_base + config.true_theta_slope * T
        theta_knots.append(float(theta_T))

    quotes: List[OptionQuote] = []

    for idx, T in enumerate(expiries):
        F = forward_curve[T]
        D = discount_curve[T]
        theta = theta_knots[idx]
        expiry_time = val_time + timedelta(days=float(T * 365.0))

        # Non-uniform log-moneyness grid (clustering near ATM)
        u = np.linspace(-1.0, 1.0, config.n_strikes_per_expiry)
        # Cubic warping for density near ATM (k=0)
        k_grid = 0.5 * (config.moneyness_max - config.moneyness_min) * (0.6 * u + 0.4 * u**3) + 0.5 * (config.moneyness_max + config.moneyness_min)
        k_grid = np.unique(np.sort(k_grid))

        for k in k_grid:
            strike = float(F * np.exp(k))
            total_var = ssvi_total_variance_exact(
                k=k,
                theta=theta,
                rho=config.true_rho,
                eta=config.true_eta,
                gamma=config.true_gamma,
            )

            # Both calls and puts
            for opt_type in ["call", "put"]:
                clean_price = black_forward_price_from_total_variance(
                    forward=F,
                    strike=strike,
                    discount_factor=D,
                    total_variance=total_var,
                    option_type=opt_type,
                )

                # Add noise and spreads
                noise = rng.normal(0.0, config.price_noise_std)
                noisy_mid = max(clean_price + noise, 0.01)

                half_spread = max(0.5 * config.relative_spread * noisy_mid, 0.05)
                bid = max(noisy_mid - half_spread, 0.01)
                ask = noisy_mid + half_spread

                quote = OptionQuote(
                    underlying_symbol="SYNTH",
                    option_symbol=f"SYNTH_{T:.2f}_{strike:.1f}_{opt_type[0].upper()}",
                    quote_timestamp=val_time,
                    valuation_timestamp=val_time,
                    expiry_timestamp=expiry_time,
                    time_to_maturity=T,
                    strike=strike,
                    option_type=opt_type,
                    bid=round(bid, 4),
                    ask=round(ask, 4),
                    bid_size=10.0,
                    ask_size=10.0,
                    volume=50.0,
                    open_interest=100.0,
                    exercise_style="european",
                    currency="USD",
                    multiplier=100.0,
                    forward=round(F, 4),
                    discount_factor=round(D, 6),
                    source="synthetic",
                    source_delay="none",
                    ingestion_timestamp=val_time,
                )
                quotes.append(quote)

    # Inject testing anomalies if requested
    if config.include_anomalies:
        # 1. Crossed quote
        crossed_q = quotes[0].model_copy(update={"bid": 10.0, "ask": 8.0, "option_symbol": "ANOMALY_CROSSED"})
        quotes.append(crossed_q)
        # 2. Zero bid
        zero_bid_q = quotes[1].model_copy(update={"bid": 0.0, "ask": 2.0, "option_symbol": "ANOMALY_ZERO_BID"})
        quotes.append(zero_bid_q)
        # 3. American style
        american_q = quotes[2].model_copy(update={"exercise_style": "american", "option_symbol": "ANOMALY_AMERICAN"})
        quotes.append(american_q)
        # 4. Expired
        expired_q = quotes[3].model_copy(update={
            "expiry_timestamp": val_time - timedelta(days=1),
            "time_to_maturity": 0.001,
            "option_symbol": "ANOMALY_EXPIRED",
        })
        quotes.append(expired_q)

    ground_truth = SyntheticGroundTruth(
        rho=config.true_rho,
        eta=config.true_eta,
        gamma=config.true_gamma,
        expiries=tuple(expiries),
        theta_knots=tuple(theta_knots),
        spot=config.spot,
        rate=config.rate,
        dividend=config.dividend,
        seed=config.seed,
    )

    snapshot = MarketSnapshot(
        valuation_timestamp=val_time,
        underlying_symbol="SYNTH",
        forward_curve=forward_curve,
        discount_curve=discount_curve,
        quotes=quotes,
    )

    return quotes, snapshot, ground_truth
