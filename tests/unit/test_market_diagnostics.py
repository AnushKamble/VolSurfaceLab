"""Unit tests for Layer C: Market quote arbitrage and parity interval diagnostics."""

from datetime import datetime, timezone
import pytest

from volsurfacelab.arbitrage.report import run_market_quote_diagnostics
from volsurfacelab.domain.models import OptionQuote


def test_market_quote_diagnostics_parity():
    """Verify market quote diagnostics identify matched pairs and calculate parity statistics."""
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    expiry = datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc)

    # Consistent pair
    call_q = OptionQuote(
        underlying_symbol="SPX",
        quote_timestamp=now,
        valuation_timestamp=now,
        expiry_timestamp=expiry,
        time_to_maturity=0.5,
        strike=100.0,
        option_type="call",
        bid=5.0,
        ask=5.4,
        forward=100.0,
        discount_factor=0.98,
    )
    put_q = OptionQuote(
        underlying_symbol="SPX",
        quote_timestamp=now,
        valuation_timestamp=now,
        expiry_timestamp=expiry,
        time_to_maturity=0.5,
        strike=100.0,
        option_type="put",
        bid=5.0,
        ask=5.4,
        forward=100.0,
        discount_factor=0.98,
    )

    # Inconsistent pair (call underpriced relative to put)
    call_q2 = call_q.model_copy(update={"strike": 110.0, "bid": 1.0, "ask": 1.2})
    put_q2 = put_q.model_copy(update={"strike": 110.0, "bid": 18.0, "ask": 18.5})

    quotes = [call_q, put_q, call_q2, put_q2]
    stats = run_market_quote_diagnostics(quotes)

    assert stats["total_quotes"] == 4
    assert stats["matched_pairs"] == 2
    assert stats["parity_interval_violations"] >= 1
