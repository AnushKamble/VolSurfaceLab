"""Canonical CSV option quotes parser and exporter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from volsurfacelab.data.provenance import compute_file_hash
from volsurfacelab.domain.exceptions import InvalidInputError
from volsurfacelab.domain.models import OptionQuote


def parse_timestamp(val: any, default_tz=timezone.utc) -> datetime:
    """Parse a date/time string or timestamp into a timezone-aware datetime."""
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=default_tz)
    ts = pd.to_datetime(val)
    if ts.tz is None:
        ts = ts.tz_localize(default_tz)
    return ts.to_pydatetime()


def load_quotes_from_csv(
    file_path: str | Path,
    default_valuation_time: Optional[datetime] = None,
) -> Tuple[List[OptionQuote], str]:
    """Load canonical option quotes from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to CSV file.
    default_valuation_time : Optional[datetime]
        Valuation timestamp if not present in CSV rows.

    Returns
    -------
    quotes : List[OptionQuote]
    file_hash : str
    """
    path = Path(file_path)
    if not path.is_file():
        raise InvalidInputError(f"CSV file not found: {path}")

    file_hash = compute_file_hash(path)
    df = pd.read_csv(path)

    # Standardize column names (lowercase and stripped)
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    required_cols = {"strike", "option_type", "bid", "ask"}
    missing = required_cols - set(df.columns)
    if missing:
        raise InvalidInputError(f"Missing mandatory columns in CSV: {missing}")

    now_utc = datetime.now(timezone.utc)
    val_time_default = default_valuation_time or now_utc

    quotes: List[OptionQuote] = []
    for _, row in df.iterrows():
        # Underlying & option symbol
        underlying = str(row.get("underlying_symbol", row.get("underlying", "UNDERLYING"))).upper()
        option_sym = row.get("option_symbol", None)
        if pd.isna(option_sym):
            option_sym = None
        else:
            option_sym = str(option_sym)

        # Timestamps
        val_time = parse_timestamp(row.get("valuation_timestamp", val_time_default))
        quote_time = parse_timestamp(row.get("quote_timestamp", val_time))
        
        # Expiry timestamp & Time to maturity
        if "expiry_timestamp" in row and not pd.isna(row["expiry_timestamp"]):
            expiry_time = parse_timestamp(row["expiry_timestamp"])
        elif "expiry" in row and not pd.isna(row["expiry"]):
            expiry_time = parse_timestamp(row["expiry"])
        else:
            # If no expiry date, check if time_to_maturity is given
            if "time_to_maturity" in row and not pd.isna(row["time_to_maturity"]):
                ttm_days = float(row["time_to_maturity"]) * 365.0
                expiry_time = val_time + pd.Timedelta(days=ttm_days)
            else:
                raise InvalidInputError("CSV row missing both expiry_timestamp and time_to_maturity.")

        if "time_to_maturity" in row and not pd.isna(row["time_to_maturity"]):
            ttm = float(row["time_to_maturity"])
        else:
            # ACT/365F day count convention
            seconds = (expiry_time - val_time).total_seconds()
            ttm = seconds / (365.0 * 86400.0)

        # Type, Strike, Bid, Ask
        opt_type = str(row["option_type"]).lower().strip()
        if opt_type in ("c", "call"):
            opt_type = "call"
        elif opt_type in ("p", "put"):
            opt_type = "put"

        strike = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])

        # Optional liquidity & Greeks
        bid_size = float(row["bid_size"]) if "bid_size" in row and not pd.isna(row["bid_size"]) else None
        ask_size = float(row["ask_size"]) if "ask_size" in row and not pd.isna(row["ask_size"]) else None
        vol = float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else None
        oi = float(row["open_interest"]) if "open_interest" in row and not pd.isna(row["open_interest"]) else None

        style = str(row.get("exercise_style", "european")).lower().strip()
        currency = str(row.get("currency", "USD"))
        mult = float(row.get("multiplier", 100.0))

        forward = float(row["forward"]) if "forward" in row and not pd.isna(row["forward"]) else None
        discount = float(row["discount_factor"]) if "discount_factor" in row and not pd.isna(row["discount_factor"]) else None

        source = str(row.get("source", "csv"))
        source_delay = str(row.get("source_delay", "none"))

        quote = OptionQuote(
            underlying_symbol=underlying,
            option_symbol=option_sym,
            quote_timestamp=quote_time,
            valuation_timestamp=val_time,
            expiry_timestamp=expiry_time,
            time_to_maturity=max(ttm, 1e-6),
            strike=strike,
            option_type=opt_type,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            volume=vol,
            open_interest=oi,
            exercise_style=style,
            currency=currency,
            multiplier=mult,
            forward=forward,
            discount_factor=discount,
            source=source,
            source_delay=source_delay,
            ingestion_timestamp=now_utc,
        )
        quotes.append(quote)

    return quotes, file_hash


def save_quotes_to_csv(quotes: List[OptionQuote], file_path: str | Path) -> str:
    """Save a list of OptionQuote models to canonical CSV format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for q in quotes:
        records.append({
            "underlying_symbol": q.underlying_symbol,
            "option_symbol": q.option_symbol,
            "quote_timestamp": q.quote_timestamp.isoformat(),
            "valuation_timestamp": q.valuation_timestamp.isoformat(),
            "expiry_timestamp": q.expiry_timestamp.isoformat(),
            "time_to_maturity": q.time_to_maturity,
            "strike": q.strike,
            "option_type": q.option_type,
            "bid": q.bid,
            "ask": q.ask,
            "bid_size": q.bid_size,
            "ask_size": q.ask_size,
            "volume": q.volume,
            "open_interest": q.open_interest,
            "exercise_style": q.exercise_style,
            "currency": q.currency,
            "multiplier": q.multiplier,
            "forward": q.forward,
            "discount_factor": q.discount_factor,
            "source": q.source,
            "source_delay": q.source_delay,
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return compute_file_hash(path)
