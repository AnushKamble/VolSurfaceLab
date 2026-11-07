"""Unit tests for canonical data ingestion, CSV parsing, and quote quality filtering."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from volsurfacelab.config import DataFilterConfig
from volsurfacelab.data.csv_provider import load_quotes_from_csv, save_quotes_to_csv
from volsurfacelab.data.quality import filter_quotes
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import OptionQuote


@pytest.fixture
def sample_quote() -> OptionQuote:
    """Fixture returning a standard valid European option quote."""
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    return OptionQuote(
        underlying_symbol="SPX",
        quote_timestamp=now,
        valuation_timestamp=now,
        expiry_timestamp=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
        time_to_maturity=0.416,
        strike=100.0,
        option_type="call",
        bid=4.8,
        ask=5.2,
        forward=101.5,
        discount_factor=0.98,
        exercise_style="european",
    )


def test_quote_filtering_clean(sample_quote):
    """Verify clean quotes pass without any rejections."""
    config = DataFilterConfig()
    accepted, report = filter_quotes([sample_quote], config)
    assert len(accepted) == 1
    assert report.raw_count == 1
    assert report.accepted_count == 1
    assert report.rejected_count == 0


def test_strict_exercise_style_gate(sample_quote):
    """Verify American and unknown exercise styles are rejected in strict mode with explicit codes."""
    config = DataFilterConfig(reject_american=True, reject_unknown=True)

    american_q = sample_quote.model_copy(update={"exercise_style": "american"})
    unknown_q = sample_quote.model_copy(update={"exercise_style": "unknown"})

    accepted, report = filter_quotes([american_q, unknown_q], config)
    assert len(accepted) == 0
    assert report.rejected_count == 2
    assert report.rejection_breakdown[StatusCode.REJECTED_EXERCISE_STYLE_AMERICAN.value] == 1
    assert report.rejection_breakdown[StatusCode.REJECTED_EXERCISE_STYLE_UNKNOWN.value] == 1


def test_crossed_and_locked_spread_filters(sample_quote):
    """Verify crossed and locked quotes are accounted for with proper reason codes."""
    config = DataFilterConfig(min_spread=0.01)

    crossed_q = sample_quote.model_copy(update={"bid": 5.5, "ask": 5.0})
    locked_q = sample_quote.model_copy(update={"bid": 5.0, "ask": 5.0})

    accepted, report = filter_quotes([crossed_q, locked_q], config)
    assert len(accepted) == 0
    assert report.rejection_breakdown[StatusCode.REJECTED_CROSSED_SPREAD.value] == 1
    assert report.rejection_breakdown[StatusCode.REJECTED_LOCKED_SPREAD.value] == 1


def test_price_bounds_rejection(sample_quote):
    """Verify quotes violating European price bounds are rejected."""
    config = DataFilterConfig()
    # Call lower bound = 0.98 * (101.5 - 100.0) = 1.47. Bid=0.2, Ask=0.4 violates bound
    below_intrinsic_q = sample_quote.model_copy(update={"bid": 0.2, "ask": 0.4})

    accepted, report = filter_quotes([below_intrinsic_q], config)
    assert len(accepted) == 0
    assert report.rejection_breakdown[StatusCode.INVERSION_PRICE_BELOW_INTRINSIC.value] == 1


def test_csv_save_and_load_roundtrip(tmp_path, sample_quote):
    """Verify CSV export and import preserves values and metadata."""
    file_path = tmp_path / "test_quotes.csv"
    h1 = save_quotes_to_csv([sample_quote], file_path)
    assert Path(file_path).exists()

    loaded_quotes, h2 = load_quotes_from_csv(file_path)
    assert h1 == h2
    assert len(loaded_quotes) == 1
    q = loaded_quotes[0]
    assert q.underlying_symbol == sample_quote.underlying_symbol
    assert q.strike == sample_quote.strike
    assert q.bid == sample_quote.bid
    assert q.ask == sample_quote.ask
    assert q.exercise_style == "european"
