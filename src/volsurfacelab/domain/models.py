"""Core domain models for VolSurfaceLab."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

from volsurfacelab.domain.exceptions import StatusCode


class OptionQuote(BaseModel):
    """Normalized European or external option quote representation."""
    underlying_symbol: str
    option_symbol: Optional[str] = None
    quote_timestamp: datetime
    valuation_timestamp: datetime
    expiry_timestamp: datetime
    time_to_maturity: float = Field(..., gt=0.0, description="Time to maturity T in years (ACT/365F)")
    strike: float = Field(..., gt=0.0, description="Positive strike K")
    option_type: str = Field(..., pattern="^(call|put)$", description="'call' or 'put'")
    bid: float = Field(..., ge=0.0, description="Bid price in quote currency")
    ask: float = Field(..., ge=0.0, description="Ask price in quote currency")
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    exercise_style: str = Field("european", pattern="^(european|american|unknown)$")
    currency: str = "USD"
    multiplier: float = 100.0
    underlying_bid: Optional[float] = None
    underlying_ask: Optional[float] = None
    forward: Optional[float] = Field(None, gt=0.0, description="Forward price F for expiry T")
    discount_factor: Optional[float] = Field(None, gt=0.0, le=1.5, description="Discount factor D for expiry T")
    source: str = "synthetic"
    source_delay: str = "none"
    ingestion_timestamp: Optional[datetime] = None

    @property
    def mid(self) -> float:
        """Mid-quote price."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def log_moneyness(self) -> Optional[float]:
        """Log-forward moneyness k = ln(K / F)."""
        import numpy as np
        if self.forward is not None and self.forward > 0 and self.strike > 0:
            return float(np.log(self.strike / self.forward))
        return None


class MarketSnapshot(BaseModel):
    """Contemporaneous snapshot containing discount curves, forwards, and quote sets."""
    valuation_timestamp: datetime
    underlying_symbol: str
    forward_curve: Dict[float, float] = Field(..., description="Map of maturity T -> forward price F")
    discount_curve: Dict[float, float] = Field(..., description="Map of maturity T -> discount factor D")
    quotes: List[OptionQuote] = Field(default_factory=list)


class QuoteCleaningReport(BaseModel):
    """Accountability log for market quote filtering and data quality checks."""
    raw_count: int
    accepted_count: int
    rejected_count: int
    rejection_breakdown: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class ImpliedVolObservation(BaseModel):
    """Observation mapping an accepted quote to its implied volatility and total variance."""
    quote: OptionQuote
    mid_price: float
    iv: float = Field(..., ge=0.0, description="Annualised implied volatility sigma")
    total_variance: float = Field(..., ge=0.0, description="Total implied variance w = sigma^2 * T")
    log_moneyness: float = Field(..., description="Log-forward moneyness k = ln(K / F)")
    inversion_status: StatusCode
    iterations: int = 0
    residual: float = 0.0
    weight: float = 1.0


class SSVIParameters(BaseModel):
    """Calibrated parameters for the regularised power-law SSVI surface."""
    rho: float = Field(..., gt=-1.0, lt=1.0, description="Skew parameter rho in (-1, 1)")
    eta: float = Field(..., gt=0.0, description="Global scale parameter eta > 0")
    gamma: float = Field(..., gt=0.0, le=0.5, description="Power-law parameter gamma in (0, 0.5]")
    expiries: List[float] = Field(..., description="Maturity knots T_1 < ... < T_M in ascending order")
    theta_knots: List[float] = Field(..., description="ATM total variance knots theta_1 < ... < theta_M")
    interpolator_type: str = "pchip"

    @model_validator(mode="after")
    def validate_monotonicity(self) -> SSVIParameters:
        if len(self.expiries) != len(self.theta_knots):
            raise ValueError(f"Expiries length ({len(self.expiries)}) must match theta knots ({len(self.theta_knots)})")
        for i in range(len(self.expiries) - 1):
            if self.expiries[i] >= self.expiries[i + 1]:
                raise ValueError("Expiries must be strictly increasing")
            if self.theta_knots[i] >= self.theta_knots[i + 1]:
                raise ValueError("Theta knots must be strictly increasing")
        return self


class CalibrationResult(BaseModel):
    """Full results, convergence diagnostics, and fit metrics for an SSVI calibration."""
    parameters: SSVIParameters
    objective_loss: float
    optimizer_status: str
    iterations: int
    evaluations: int
    cost: float
    price_rmse: float
    price_mae: float
    iv_rmse: float
    iv_mae: float
    pct_inside_spread: float
    residual_percentiles: Dict[str, float] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AdmissibilityReport(BaseModel):
    """Analytical verification of the Gatheral-Jacquier SSVI sufficient no-arbitrage conditions."""
    theorem_status: StatusCode
    rho_in_domain: bool
    rho_margin: float
    gamma_in_domain: bool
    gamma_margin: float
    eta_positive: bool
    eta_cap: float
    eta_margin: float
    theta_positive: bool
    theta_monotone: bool
    theta_min_derivative: float
    h_max: float
    cond1_value: float = Field(..., description="max theta*phi*(1+|rho|)")
    cond1_margin: float = Field(..., description="4 - cond1_value")
    cond2_value: float = Field(..., description="max theta*phi^2*(1+|rho|)")
    cond2_margin: float = Field(..., description="4 - cond2_value")
    calendar_ratio_bound_satisfied: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class GridDiagnostics(BaseModel):
    """Numerical finite-grid inspection of Durrleman g(k,T), calendar derivative, and call convexity."""
    grid_status: StatusCode
    min_g: float
    worst_g_coordinate: Tuple[float, float] = Field(..., description="(k, T) of minimum g")
    negative_g_count: int
    min_w_T: float
    worst_w_T_coordinate: Tuple[float, float] = Field(..., description="(k, T) of minimum w_T")
    negative_w_T_count: int
    min_call_convexity: float
    convexity_violation_count: int
    k_min: float
    k_max: float
    T_min: float
    T_max: float
    n_k: int
    n_T: int


class DensityResult(BaseModel):
    """Breeden-Litzenberger risk-neutral density and diagnostic metrics."""
    expiry: float
    forward: float
    discount_factor: float
    strikes: List[float]
    log_moneyness: List[float]
    density_q: List[float]
    state_price_density: List[float]
    total_mass: float
    first_moment: float
    relative_moment_error: float
    is_non_negative: bool
    min_density: float
    negative_count: int
    support_min: float
    support_max: float
    warnings: List[str] = Field(default_factory=list)


class LocalVolResult(BaseModel):
    """Dupire local variance and volatility grid with full conditioning mask and reason codes."""
    expiries: List[float]
    log_moneyness_grid: List[float]
    strikes_grid: List[List[float]]
    local_variance: List[List[Optional[float]]]
    local_volatility: List[List[Optional[float]]]
    validity_mask: List[List[bool]]
    reason_codes: List[List[str]]
    valid_fraction: float
    total_cells: int
    valid_cells: int
    min_local_vol: Optional[float] = None
    max_local_vol: Optional[float] = None


class RunArtifact(BaseModel):
    """Immutable bundle containing all calibration, verification, density, and local vol artefacts."""
    run_id: str
    timestamp: str
    config_hash: str
    environment: Dict[str, Any] = Field(default_factory=dict)
    parameters: SSVIParameters
    calibration_result: CalibrationResult
    admissibility_report: AdmissibilityReport
    grid_diagnostics: GridDiagnostics
    quote_cleaning_report: QuoteCleaningReport
    fit_metrics: Dict[str, Any] = Field(default_factory=dict)
