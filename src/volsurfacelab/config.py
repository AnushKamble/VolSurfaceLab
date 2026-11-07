"""Configuration parsing, provenance hashing, and default experiment settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml


class DataFilterConfig(BaseModel):
    """Configuration for raw option quote filtering."""
    min_spread: float = Field(0.01, ge=0.0, description="Minimum half-spread or spread floor in dollars")
    max_relative_spread: float = Field(2.0, ge=0.0, description="Max allowed spread relative to mid")
    min_volume: float = Field(0.0, ge=0.0, description="Minimum contract volume filter")
    min_open_interest: float = Field(0.0, ge=0.0, description="Minimum open interest filter")
    reject_american: bool = Field(True, description="Enforce strict European-only gate")
    reject_unknown: bool = Field(True, description="Reject quotes with unknown exercise style")
    max_stale_seconds: float = Field(86400.0, ge=0.0, description="Max allowed latency between quote and snapshot")
    parity_interval_check: bool = Field(True, description="Enable put-call parity interval diagnostics")


class CalibrationConfig(BaseModel):
    """Configuration for constrained SSVI calibration."""
    safety_margin: float = Field(0.01, ge=0.0, lt=0.5, description="Safety margin epsilon for eta cap")
    loss_function: str = Field("soft_l1", pattern="^(soft_l1|linear|huber)$")
    price_scale_floor: float = Field(0.05, gt=0.0, description="Minimum spread divisor to prevent exploding weights")
    max_nfev: int = Field(2000, gt=0, description="Maximum function evaluations in TRF least-squares")
    ftol: float = Field(1e-8, gt=0.0)
    xtol: float = Field(1e-8, gt=0.0)
    gtol: float = Field(1e-8, gt=0.0)
    multi_start_seeds: List[int] = Field(default_factory=lambda: [42, 101, 777])
    initial_rho: float = Field(-0.3, gt=-1.0, lt=1.0)
    initial_gamma: float = Field(0.35, gt=0.0, le=0.5)
    f_scale: float = Field(1.0, gt=0.0, description="Soft-L1 inlier threshold")


class GridConfig(BaseModel):
    """Configuration for dense numerical grid diagnostics."""
    k_min: float = Field(-1.5, description="Minimum log-forward moneyness")
    k_max: float = Field(1.5, description="Maximum log-forward moneyness")
    n_k: int = Field(121, ge=11, description="Number of strike points")
    T_min: float = Field(0.02, gt=0.0, description="Minimum maturity fraction")
    T_max: float = Field(3.0, gt=0.0, description="Maximum maturity fraction")
    n_T: int = Field(61, ge=5, description="Number of maturity points")
    g_tolerance: float = Field(-1e-6, description="Tolerance for butterfly function g >= 0")
    w_T_tolerance: float = Field(-1e-6, description="Tolerance for calendar derivative w_T >= 0")
    refinement_factor: int = Field(2, ge=1, description="Refinement factor for numerical stability checks")


class DensityConfig(BaseModel):
    """Configuration for Breeden-Litzenberger risk-neutral density integration."""
    n_strikes: int = Field(301, ge=51)
    std_dev_range: float = Field(5.0, ge=1.0, description="Integration domain width in standard deviations")
    integration_limit: int = Field(200, ge=50)
    epsabs: float = Field(1e-7, gt=0.0)
    epsrel: float = Field(1e-6, gt=0.0)


class LocalVolConfig(BaseModel):
    """Configuration for Dupire local volatility evaluation and safeguards."""
    g_denominator_min: float = Field(1e-4, gt=0.0, description="Threshold below which g is flagged near-singular")
    require_positive_w_T: bool = Field(True, description="Flag negative w_T as invalid calendar")


class SyntheticGeneratorConfig(BaseModel):
    """Configuration for deterministic synthetic quote generator."""
    seed: int = Field(42, description="Random seed for reproducibility")
    spot: float = Field(100.0, gt=0.0)
    rate: float = Field(0.04, ge=0.0)
    dividend: float = Field(0.015, ge=0.0)
    expiries: List[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    moneyness_min: float = Field(-0.6)
    moneyness_max: float = Field(0.6)
    n_strikes_per_expiry: int = Field(25, ge=5)
    true_rho: float = Field(-0.4)
    true_eta: float = Field(1.1)
    true_gamma: float = Field(0.35)
    true_theta_base: float = Field(0.04)
    true_theta_slope: float = Field(0.05)
    relative_spread: float = Field(0.03, ge=0.0)
    price_noise_std: float = Field(0.02, ge=0.0)
    include_anomalies: bool = Field(False, description="Whether to inject crossed or stale quotes for testing")


class RunConfig(BaseModel):
    """Root configuration object controlling full pipeline runs."""
    run_id: str = "default_run"
    seed: int = 42
    data_filter: DataFilterConfig = Field(default_factory=DataFilterConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    local_vol: LocalVolConfig = Field(default_factory=LocalVolConfig)
    synthetic: SyntheticGeneratorConfig = Field(default_factory=SyntheticGeneratorConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration for provenance."""
        dumped = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
