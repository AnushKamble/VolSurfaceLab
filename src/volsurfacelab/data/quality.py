"""Data quality controls, contract validation, and strict exercise-style gating."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from volsurfacelab.config import DataFilterConfig
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import OptionQuote, QuoteCleaningReport
from volsurfacelab.pricing.black_forward import check_european_bounds


def filter_quotes(
    quotes: List[OptionQuote],
    config: DataFilterConfig,
    market_forwards: Optional[Dict[float, float]] = None,
    market_discounts: Optional[Dict[float, float]] = None,
) -> Tuple[List[OptionQuote], QuoteCleaningReport]:
    """Filter raw option quotes against quantitative quality and European validity gates.

    Parameters
    ----------
    quotes : List[OptionQuote]
        Raw ingested quotes.
    config : DataFilterConfig
        Thresholds and filter flags.
    market_forwards : Optional[Dict[float, float]]
        Optional mapping of maturity T -> forward price F.
    market_discounts : Optional[Dict[float, float]]
        Optional mapping of maturity T -> discount factor D.

    Returns
    -------
    accepted_quotes : List[OptionQuote]
    report : QuoteCleaningReport
    """
    accepted: List[OptionQuote] = []
    rejection_counts: Dict[str, int] = defaultdict(int)
    warnings: List[str] = []

    for q in quotes:
        # 1. Non-positive strike or maturity
        if q.strike <= 0.0:
            rejection_counts[StatusCode.REJECTED_NON_POSITIVE_STRIKE.value] += 1
            continue
        if q.time_to_maturity <= 0.0:
            rejection_counts[StatusCode.REJECTED_NON_POSITIVE_MATURITY.value] += 1
            continue

        # 2. Expiry timestamp sanity
        if q.expiry_timestamp <= q.valuation_timestamp:
            rejection_counts[StatusCode.REJECTED_EXPIRED.value] += 1
            continue

        # 3. Exercise style gate
        style = q.exercise_style.lower().strip()
        if style == "american":
            if config.reject_american:
                rejection_counts[StatusCode.REJECTED_EXERCISE_STYLE_AMERICAN.value] += 1
                continue
        elif style != "european":
            if config.reject_unknown:
                rejection_counts[StatusCode.REJECTED_EXERCISE_STYLE_UNKNOWN.value] += 1
                continue

        # 4. Bid/Ask price validity
        if q.bid < 0.0:
            rejection_counts[StatusCode.REJECTED_NEGATIVE_BID.value] += 1
            continue
        if q.ask < 0.0:
            rejection_counts[StatusCode.REJECTED_NEGATIVE_ASK.value] += 1
            continue
        if q.ask < q.bid:
            rejection_counts[StatusCode.REJECTED_CROSSED_SPREAD.value] += 1
            continue
        if q.ask == q.bid and config.min_spread > 0.0:
            rejection_counts[StatusCode.REJECTED_LOCKED_SPREAD.value] += 1
            continue
        if q.bid == 0.0:
            rejection_counts[StatusCode.REJECTED_ZERO_BID.value] += 1
            continue

        # 5. Spread width filter
        spread = q.spread
        mid = q.mid
        if spread < config.min_spread:
            rejection_counts[StatusCode.REJECTED_LOCKED_SPREAD.value] += 1
            continue
        if mid > 0.0 and (spread / mid) > config.max_relative_spread:
            rejection_counts[StatusCode.REJECTED_WIDE_SPREAD.value] += 1
            continue

        # 6. Volume & Open Interest filters
        if q.volume is not None and q.volume < config.min_volume:
            rejection_counts[StatusCode.REJECTED_INSUFFICIENT_LIQUIDITY.value] += 1
            continue
        if q.open_interest is not None and q.open_interest < config.min_open_interest:
            rejection_counts[StatusCode.REJECTED_INSUFFICIENT_LIQUIDITY.value] += 1
            continue

        # 7. Stale quote check
        quote_latency = (q.valuation_timestamp - q.quote_timestamp).total_seconds()
        if quote_latency > config.max_stale_seconds:
            rejection_counts[StatusCode.REJECTED_STALE_QUOTE.value] += 1
            continue

        # 8. Forward and discount factors
        F = q.forward
        if F is None and market_forwards is not None:
            F = market_forwards.get(q.time_to_maturity)
        D = q.discount_factor
        if D is None and market_discounts is not None:
            D = market_discounts.get(q.time_to_maturity)

        if F is None or F <= 0.0:
            rejection_counts[StatusCode.REJECTED_NON_POSITIVE_FORWARD.value] += 1
            continue
        if D is None or D <= 0.0:
            rejection_counts[StatusCode.REJECTED_NON_POSITIVE_DISCOUNT.value] += 1
            continue

        # Update quote with matched forward/discount if missing
        if q.forward is None or q.discount_factor is None:
            q = q.model_copy(update={"forward": F, "discount_factor": D})

        # 9. European no-arbitrage price bounds check on mid price
        is_bound_valid, bound_code, _, _ = check_european_bounds(
            price=mid,
            forward=F,
            strike=q.strike,
            discount_factor=D,
            option_type=q.option_type,
        )
        if not is_bound_valid:
            rejection_counts[bound_code.value] += 1
            continue

        accepted.append(q)

    raw_total = len(quotes)
    accepted_total = len(accepted)
    rejected_total = raw_total - accepted_total

    if accepted_total == 0 and raw_total > 0:
        warnings.append("All ingested quotes were rejected by quality filters.")

    report = QuoteCleaningReport(
        raw_count=raw_total,
        accepted_count=accepted_total,
        rejected_count=rejected_total,
        rejection_breakdown=dict(rejection_counts),
        warnings=warnings,
    )
    return accepted, report
