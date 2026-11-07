"""Domain exceptions and structured reason codes for VolSurfaceLab."""

from enum import Enum


class StatusCode(str, Enum):
    """Reason codes for quote filtering, inversion, calibration, and local vol."""
    
    # Filter / Ingestion codes
    ACCEPTED = "ACCEPTED"
    REJECTED_NON_POSITIVE_STRIKE = "REJECTED_NON_POSITIVE_STRIKE"
    REJECTED_NON_POSITIVE_FORWARD = "REJECTED_NON_POSITIVE_FORWARD"
    REJECTED_NON_POSITIVE_DISCOUNT = "REJECTED_NON_POSITIVE_DISCOUNT"
    REJECTED_NON_POSITIVE_MATURITY = "REJECTED_NON_POSITIVE_MATURITY"
    REJECTED_EXPIRED = "REJECTED_EXPIRED"
    REJECTED_NEGATIVE_BID = "REJECTED_NEGATIVE_BID"
    REJECTED_NEGATIVE_ASK = "REJECTED_NEGATIVE_ASK"
    REJECTED_CROSSED_SPREAD = "REJECTED_CROSSED_SPREAD"
    REJECTED_LOCKED_SPREAD = "REJECTED_LOCKED_SPREAD"
    REJECTED_ZERO_BID = "REJECTED_ZERO_BID"
    REJECTED_WIDE_SPREAD = "REJECTED_WIDE_SPREAD"
    REJECTED_EXERCISE_STYLE_AMERICAN = "REJECTED_EXERCISE_STYLE_AMERICAN"
    REJECTED_EXERCISE_STYLE_UNKNOWN = "REJECTED_EXERCISE_STYLE_UNKNOWN"
    REJECTED_PRICE_BOUND_VIOLATION = "REJECTED_PRICE_BOUND_VIOLATION"
    REJECTED_STALE_QUOTE = "REJECTED_STALE_QUOTE"
    REJECTED_INSUFFICIENT_LIQUIDITY = "REJECTED_INSUFFICIENT_LIQUIDITY"
    REJECTED_INVALID_OPTION_TYPE = "REJECTED_INVALID_OPTION_TYPE"

    # Implied Volatility Inversion codes
    INVERSION_SUCCESS = "INVERSION_SUCCESS"
    INVERSION_PRICE_BELOW_INTRINSIC = "INVERSION_PRICE_BELOW_INTRINSIC"
    INVERSION_PRICE_ABOVE_UPPER_BOUND = "INVERSION_PRICE_ABOVE_UPPER_BOUND"
    INVERSION_BRACKET_FAILED = "INVERSION_BRACKET_FAILED"
    INVERSION_MAX_ITERATIONS = "INVERSION_MAX_ITERATIONS"
    INVERSION_NEAR_ZERO_VEGA = "INVERSION_NEAR_ZERO_VEGA"
    INVERSION_NON_FINITE_INPUT = "INVERSION_NON_FINITE_INPUT"

    # Calibration codes
    CALIBRATION_CONVERGED = "CALIBRATION_CONVERGED"
    CALIBRATION_MAX_ITERATIONS = "CALIBRATION_MAX_ITERATIONS"
    CALIBRATION_INSUFFICIENT_DATA = "CALIBRATION_INSUFFICIENT_DATA"
    CALIBRATION_OPTIMIZER_FAILED = "CALIBRATION_OPTIMIZER_FAILED"
    CALIBRATION_INADMISSIBLE_PARAMETERS = "CALIBRATION_INADMISSIBLE_PARAMETERS"

    # Arbitrage / Certification codes
    CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS = "CERTIFIED_BY_SELECTED_SSVI_SUFFICIENT_CONDITIONS"
    CERTIFICATE_FAILED = "CERTIFICATE_FAILED"
    GRID_DIAGNOSTIC_PASSED = "GRID_DIAGNOSTIC_PASSED"
    GRID_DIAGNOSTIC_WARNING = "GRID_DIAGNOSTIC_WARNING"
    GRID_DIAGNOSTIC_FAILED = "GRID_DIAGNOSTIC_FAILED"
    ADMISSIBLE_BUT_POOR_FIT = "ADMISSIBLE_BUT_POOR_FIT"
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"

    # Local Volatility codes
    LOCAL_VOL_VALID = "LOCAL_VOL_VALID"
    LOCAL_VOL_INVALID_MATURITY = "LOCAL_VOL_INVALID_MATURITY"
    LOCAL_VOL_DENOMINATOR_NON_POSITIVE = "LOCAL_VOL_DENOMINATOR_NON_POSITIVE"
    LOCAL_VOL_DENOMINATOR_NEAR_ZERO = "LOCAL_VOL_DENOMINATOR_NEAR_ZERO"
    LOCAL_VOL_NUMERATOR_NEGATIVE = "LOCAL_VOL_NUMERATOR_NEGATIVE"
    LOCAL_VOL_VARIANCE_NEGATIVE = "LOCAL_VOL_VARIANCE_NEGATIVE"
    LOCAL_VOL_NON_FINITE = "LOCAL_VOL_NON_FINITE"


class VolSurfaceLabError(Exception):
    """Base exception for all VolSurfaceLab errors."""
    pass


class InvalidInputError(VolSurfaceLabError):
    """Raised when input parameters, market quotes, or matrices violate preconditions."""
    def __init__(self, message: str, code: StatusCode = StatusCode.REJECTED_NON_POSITIVE_STRIKE):
        super().__init__(message)
        self.code = code


class UnsupportedExerciseStyleError(VolSurfaceLabError):
    """Raised when non-European options (American, Bermuda, Unknown) are provided to the strict pipeline."""
    def __init__(self, message: str, exercise_style: str):
        super().__init__(f"Unsupported exercise style '{exercise_style}': {message}")
        self.exercise_style = exercise_style


class PricingBoundError(VolSurfaceLabError):
    """Raised when option prices violate European no-arbitrage bounds."""
    def __init__(self, message: str, code: StatusCode = StatusCode.REJECTED_PRICE_BOUND_VIOLATION):
        super().__init__(message)
        self.code = code


class InversionError(VolSurfaceLabError):
    """Raised when implied volatility cannot be bracketed or inverted."""
    def __init__(self, message: str, code: StatusCode = StatusCode.INVERSION_BRACKET_FAILED):
        super().__init__(message)
        self.code = code


class CalibrationError(VolSurfaceLabError):
    """Raised when calibration fails to converge, has insufficient data, or produces invalid parameters."""
    def __init__(self, message: str, code: StatusCode = StatusCode.CALIBRATION_OPTIMIZER_FAILED):
        super().__init__(message)
        self.code = code


class InvalidSurfaceError(VolSurfaceLabError):
    """Raised when an operation is requested on an inadmissible or structurally invalid surface."""
    def __init__(self, message: str, code: StatusCode = StatusCode.CERTIFICATE_FAILED):
        super().__init__(message)
        self.code = code
