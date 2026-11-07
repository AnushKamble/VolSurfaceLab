"""Unit tests for package foundation, domain objects, and configuration."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

import volsurfacelab
from volsurfacelab.cli import build_parser
from volsurfacelab.config import RunConfig
from volsurfacelab.domain.exceptions import (
    StatusCode,
    InvalidInputError,
    UnsupportedExerciseStyleError,
)
from volsurfacelab.domain.models import (
    OptionQuote,
    SSVIParameters,
    MarketSnapshot,
)
from volsurfacelab.domain.schemas import model_to_dict, model_to_json, model_from_dict, model_from_json


def test_package_import():
    """Verify package imports with valid version."""
    assert volsurfacelab.__version__ == "0.1.0"


def test_option_quote_valid():
    """Verify valid construction of OptionQuote and derived properties."""
    quote = OptionQuote(
        underlying_symbol="SPX",
        quote_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        valuation_timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        expiry_timestamp=datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc),
        time_to_maturity=0.5,
        strike=100.0,
        option_type="call",
        bid=5.0,
        ask=5.5,
        forward=102.0,
        discount_factor=0.98,
    )
    assert quote.mid == 5.25
    assert quote.spread == 0.5
    assert quote.log_moneyness is not None
    assert abs(quote.log_moneyness - (-0.0198026)) < 1e-5


def test_option_quote_invalid():
    """Verify OptionQuote rejects invalid attributes."""
    # Negative strike
    with pytest.raises(ValidationError):
        OptionQuote(
            underlying_symbol="SPX",
            quote_timestamp=datetime.now(timezone.utc),
            valuation_timestamp=datetime.now(timezone.utc),
            expiry_timestamp=datetime.now(timezone.utc),
            time_to_maturity=0.5,
            strike=-10.0,
            option_type="call",
            bid=1.0,
            ask=2.0,
        )
    # Non-positive maturity
    with pytest.raises(ValidationError):
        OptionQuote(
            underlying_symbol="SPX",
            quote_timestamp=datetime.now(timezone.utc),
            valuation_timestamp=datetime.now(timezone.utc),
            expiry_timestamp=datetime.now(timezone.utc),
            time_to_maturity=0.0,
            strike=100.0,
            option_type="call",
            bid=1.0,
            ask=2.0,
        )
    # Invalid option type
    with pytest.raises(ValidationError):
        OptionQuote(
            underlying_symbol="SPX",
            quote_timestamp=datetime.now(timezone.utc),
            valuation_timestamp=datetime.now(timezone.utc),
            expiry_timestamp=datetime.now(timezone.utc),
            time_to_maturity=0.5,
            strike=100.0,
            option_type="straddle",
            bid=1.0,
            ask=2.0,
        )


def test_ssvi_parameters_monotonicity():
    """Verify SSVIParameters enforces strictly increasing expiries and theta knots."""
    # Valid
    params = SSVIParameters(
        rho=-0.4,
        eta=1.0,
        gamma=0.4,
        expiries=[0.1, 0.5, 1.0],
        theta_knots=[0.02, 0.05, 0.10],
    )
    assert params.rho == -0.4

    # Non-monotone expiries
    with pytest.raises(ValidationError):
        SSVIParameters(
            rho=-0.4,
            eta=1.0,
            gamma=0.4,
            expiries=[0.5, 0.1, 1.0],
            theta_knots=[0.02, 0.05, 0.10],
        )

    # Non-monotone theta knots
    with pytest.raises(ValidationError):
        SSVIParameters(
            rho=-0.4,
            eta=1.0,
            gamma=0.4,
            expiries=[0.1, 0.5, 1.0],
            theta_knots=[0.05, 0.02, 0.10],
        )


def test_serialization_roundtrip():
    """Verify Pydantic models serialize and deserialize without loss."""
    params = SSVIParameters(
        rho=-0.3,
        eta=1.2,
        gamma=0.35,
        expiries=[0.25, 0.5, 1.0],
        theta_knots=[0.04, 0.08, 0.15],
    )
    json_str = model_to_json(params)
    deserialized = model_from_json(SSVIParameters, json_str)
    assert deserialized == params


def test_config_hash_provenance():
    """Verify configuration computation of SHA-256 hash is deterministic."""
    cfg1 = RunConfig(seed=42)
    cfg2 = RunConfig(seed=42)
    cfg3 = RunConfig(seed=99)
    assert cfg1.compute_hash() == cfg2.compute_hash()
    assert cfg1.compute_hash() != cfg3.compute_hash()


def test_cli_parser_help():
    """Verify CLI parser builds and contains required subcommands."""
    parser = build_parser()
    help_text = parser.format_help()
    for cmd in ["demo", "calibrate", "verify", "report", "benchmark", "dashboard"]:
        assert cmd in help_text

