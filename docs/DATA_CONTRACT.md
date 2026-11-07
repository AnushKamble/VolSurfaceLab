# VolSurfaceLab — Data Contract & CSV Schema Specification

This document specifies the canonical input and output data contracts, column mappings, time conventions, and exercise-style validation requirements for **VolSurfaceLab**.

---

## 1. Canonical Option Quote Schema

VolSurfaceLab ingests European option quotes in canonical tabular format. External vendor CSV files are parsed and mapped to internal `OptionQuote` models.

| Field Name | Type | Format / Constraints | Required | Description |
|---|---|---|:---:|---|
| `underlying_symbol` | String | e.g. `SPX`, `SYNTH` | Yes | Canonical ticker of the underlying index or asset. |
| `option_symbol` | String | e.g. `SPX_20260619_C5000` | Optional | Unique contract identifier from market vendor. |
| `quote_timestamp` | String / DateTime | ISO-8601 (UTC preferred) | Yes | Source timestamp when the quote was emitted. |
| `valuation_timestamp` | String / DateTime | ISO-8601 (UTC preferred) | Yes | Snapshot snapshot timestamp for the evaluation. |
| `expiry_timestamp` | String / DateTime | ISO-8601 (UTC preferred) | Yes | Contract expiration timestamp. |
| `time_to_maturity` | Float | $T > 0$ in years | Derived/Yes | Year fraction computed under ACT/365F. |
| `strike` | Float | $K > 0$ | Yes | Strike price in currency units. |
| `option_type` | String | `'call'` or `'put'` (`'c'`, `'p'`) | Yes | Exercise payoff style. |
| `bid` | Float | $\ge 0.0$ | Yes | Current best bid price. |
| `ask` | Float | $\ge \text{bid}$ | Yes | Current best ask price. |
| `bid_size` | Float | $\ge 0.0$ | Optional | Bid depth / number of contracts. |
| `ask_size` | Float | $\ge 0.0$ | Optional | Ask depth / number of contracts. |
| `volume` | Float | $\ge 0.0$ | Optional | Total traded contract volume for the session. |
| `open_interest` | Float | $\ge 0.0$ | Optional | Total outstanding open contracts. |
| `exercise_style` | String | `'european'` (strictly required) | Yes | Exercise style. `'american'` and `'unknown'` are rejected in strict mode. |
| `currency` | String | e.g. `USD`, `EUR` | Yes | Contract quote currency (default `USD`). |
| `multiplier` | Float | $> 0$ (default `100.0`) | Yes | Contract lot multiplier. |
| `forward` | Float | $F > 0$ | Yes* | Forward price $F_T$ for expiry $T$ (from row or snapshot curve). |
| `discount_factor` | Float | $D \in (0, 1.5]$ | Yes* | Discount factor $D_T$ for expiry $T$ (from row or snapshot curve). |
| `source` | String | e.g. `synthetic`, `cboe` | Yes | Data provenance origin label. |
| `source_delay` | String | e.g. `none`, `15m_delayed` | Yes | Real-time vs delay metadata tag. |

\* If `forward` or `discount_factor` are omitted on individual rows, they must be supplied in the accompanying market snapshot curve.

---

## 2. Ingestion Quality Control Gates

Every ingested quote passes through sequential validation filters:
1. **Positivity:** Strike $K > 0$, Forward $F > 0$, Discount $D > 0$, Time to Maturity $T > 0$.
2. **Chronology:** Expiry timestamp must strictly follow valuation timestamp.
3. **Strict Exercise-Style Gate:**
   - `exercise_style == 'european'`: Eligible for pricing and calibration.
   - `exercise_style == 'american'`: Rejected with `REJECTED_EXERCISE_STYLE_AMERICAN`.
   - `exercise_style == 'unknown'`: Rejected with `REJECTED_EXERCISE_STYLE_UNKNOWN`.
4. **Spread Quality:**
   - Non-negative bids and asks: $\text{bid} \ge 0, \text{ask} \ge 0$.
   - Uncrossed spread: $\text{ask} \ge \text{bid}$.
   - Unlocked spread: $\text{ask} > \text{bid}$ if `min_spread > 0`.
   - Zero-bid rejection: Options with $\text{bid} = 0$ are discarded as illiquid.
   - Relative spread width: $(\text{ask} - \text{bid}) / \text{mid} \le \text{max\_relative\_spread}$.
5. **No-Arbitrage Bounds:**
   - Mid price must satisfy European call/put bounds under the forward price and discount factor:
     - Call: $D \max(F - K, 0) \le \text{mid} \le D F$.
     - Put: $D \max(K - F, 0) \le \text{mid} \le D K$.
6. **Staleness:** Latency between quote timestamp and valuation snapshot must not exceed `max_stale_seconds`.

Every rejected quote is logged in `QuoteCleaningReport` with an explicit reason code. No row is ever silently discarded.

---

## 3. CSV File Example

```csv
underlying_symbol,option_symbol,quote_timestamp,valuation_timestamp,expiry_timestamp,time_to_maturity,strike,option_type,bid,ask,exercise_style,currency,multiplier,forward,discount_factor,source,source_delay
SPX,SPX_0.50_4900_C,2026-01-01T10:00:00Z,2026-01-01T10:00:00Z,2026-07-02T16:00:00Z,0.5,4900.0,call,285.50,287.50,european,USD,100.0,5020.0,0.9802,synthetic,none
SPX,SPX_0.50_4900_P,2026-01-01T10:00:00Z,2026-01-01T10:00:00Z,2026-07-02T16:00:00Z,0.5,4900.0,put,168.00,170.00,european,USD,100.0,5020.0,0.9802,synthetic,none
```
