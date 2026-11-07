"""European put-call parity diagnostics and forward inference."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


class ParityCheckResult(NamedTuple):
    """Result of European put-call parity inspection for a contract pair."""
    strike: float
    time_to_maturity: float
    forward: float
    discount_factor: float
    call_mid: float
    put_mid: float
    synthetic_forward: float
    absolute_parity_error: float
    relative_parity_error: float
    is_interval_consistent: bool
    details: Dict[str, float]


def check_put_call_parity(
    call_price: float,
    put_price: float,
    forward: float,
    strike: float,
    discount_factor: float,
) -> Tuple[float, float]:
    """Check standard European put-call parity: C - P = D * (F - K).

    Returns
    -------
    residual : float
        (C - P) - D * (F - K). Positive means call is overpriced relative to put.
    synthetic_forward : float
        Implied forward price: K + (C - P) / D.
    """
    diff = call_price - put_price
    expected_diff = discount_factor * (forward - strike)
    residual = diff - expected_diff
    synthetic_forward = strike + diff / discount_factor
    return float(residual), float(synthetic_forward)


def check_put_call_parity_intervals(
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    forward: float,
    strike: float,
    discount_factor: float,
) -> ParityCheckResult:
    """Diagnose put-call parity using bid-ask quote intervals.

    Under no-arbitrage, the synthetic call [put_bid + D(F-K), put_ask + D(F-K)]
    must overlap with the observed call interval [call_bid, call_ask].
    """
    call_mid = 0.5 * (call_bid + call_ask)
    put_mid = 0.5 * (put_bid + put_ask)

    residual, synthetic_forward = check_put_call_parity(
        call_price=call_mid,
        put_price=put_mid,
        forward=forward,
        strike=strike,
        discount_factor=discount_factor,
    )

    d_fk = discount_factor * (forward - strike)
    synth_call_bid = put_bid + d_fk
    synth_call_ask = put_ask + d_fk

    # Check interval overlap: max(A_low, B_low) <= min(A_high, B_high)
    overlap_low = max(call_bid, synth_call_bid)
    overlap_high = min(call_ask, synth_call_ask)
    is_consistent = (overlap_low <= overlap_high + 1e-4)

    rel_error = abs(residual) / max(forward, 1.0)

    return ParityCheckResult(
        strike=strike,
        time_to_maturity=0.0,  # attached by caller
        forward=forward,
        discount_factor=discount_factor,
        call_mid=call_mid,
        put_mid=put_mid,
        synthetic_forward=synthetic_forward,
        absolute_parity_error=abs(residual),
        relative_parity_error=rel_error,
        is_interval_consistent=is_consistent,
        details={
            "call_bid": call_bid,
            "call_ask": call_ask,
            "synth_call_bid": synth_call_bid,
            "synth_call_ask": synth_call_ask,
            "parity_residual": residual,
        },
    )
