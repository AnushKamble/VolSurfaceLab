"""Optional external market data adapter with strict exercise style and delay metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from volsurfacelab.data.csv_provider import parse_timestamp
from volsurfacelab.domain.exceptions import InvalidInputError, UnsupportedExerciseStyleError
from volsurfacelab.domain.models import OptionQuote


class ExternalMarketAdapter:
    """Adapter for importing authorized external option quotes with provenance and exercise style checks."""

    def __init__(self, default_source_delay: str = "15m_delayed"):
        self.default_source_delay = default_source_delay

    def import_cboe_style_csv(
        self,
        file_path: str | Path,
        underlying_symbol: str = "SPX",
        strict_european_only: bool = True,
    ) -> List[OptionQuote]:
        """Import Cboe DataShop style interval quote records into OptionQuote objects."""
        path = Path(file_path)
        if not path.is_file():
            raise InvalidInputError(f"File not found: {path}")

        df = pd.read_csv(path)
        col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)

        quotes: List[OptionQuote] = []
        now_utc = datetime.now(timezone.utc)

        for _, row in df.iterrows():
            style = str(row.get("exercise_style", "european")).lower().strip()
            if strict_european_only and style != "european":
                raise UnsupportedExerciseStyleError(
                    f"Contract {row.get('option_symbol', '')} has exercise style '{style}'. Strict European pipeline requires European options.",
                    exercise_style=style,
                )

            val_time = parse_timestamp(row.get("valuation_timestamp", now_utc))
            quote_time = parse_timestamp(row.get("quote_timestamp", val_time))

            if "expiry_timestamp" in row:
                expiry_time = parse_timestamp(row["expiry_timestamp"])
                ttm = (expiry_time - val_time).total_seconds() / (365.0 * 86400.0)
            elif "time_to_maturity" in row:
                ttm = float(row["time_to_maturity"])
                expiry_time = val_time + pd.Timedelta(days=ttm * 365.0)
            else:
                raise InvalidInputError("CSV missing both expiry_timestamp and time_to_maturity.")

            q = OptionQuote(
                underlying_symbol=underlying_symbol,
                option_symbol=str(row.get("option_symbol", f"{underlying_symbol}_{ttm:.2f}")),
                quote_timestamp=quote_time,
                valuation_timestamp=val_time,
                expiry_timestamp=expiry_time,
                time_to_maturity=max(ttm, 1e-6),
                strike=float(row["strike"]),
                option_type=str(row["option_type"]).lower(),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                bid_size=float(row["bid_size"]) if "bid_size" in row and not pd.isna(row["bid_size"]) else None,
                ask_size=float(row["ask_size"]) if "ask_size" in row and not pd.isna(row["ask_size"]) else None,
                volume=float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else None,
                open_interest=float(row["open_interest"]) if "open_interest" in row and not pd.isna(row["open_interest"]) else None,
                exercise_style=style,
                forward=float(row["forward"]) if "forward" in row and not pd.isna(row["forward"]) else None,
                discount_factor=float(row["discount_factor"]) if "discount_factor" in row and not pd.isna(row["discount_factor"]) else None,
                source="cboe_external_adapter",
                source_delay=str(row.get("source_delay", self.default_source_delay)),
                ingestion_timestamp=now_utc,
            )
            quotes.append(q)

        return quotes
