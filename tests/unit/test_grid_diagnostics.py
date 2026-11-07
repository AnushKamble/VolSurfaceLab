"""Unit tests for Layer B: Independent numerical butterfly and calendar grid diagnostics."""

import numpy as np
import pytest

from volsurfacelab.arbitrage.butterfly import check_butterfly_arbitrage_grid
from volsurfacelab.arbitrage.calendar import check_calendar_arbitrage_grid
from volsurfacelab.arbitrage.report import run_grid_diagnostics
from volsurfacelab.config import GridConfig
from volsurfacelab.domain.exceptions import StatusCode
from volsurfacelab.domain.models import SSVIParameters
from volsurfacelab.surface.ssvi import SSVISurface


@pytest.fixture
def certified_surface() -> SSVISurface:
    """Fixture returning a certified admissible SSVI surface."""
    params = SSVIParameters(
        rho=-0.35,
        eta=1.0,
        gamma=0.35,
        expiries=[0.1, 0.5, 1.0, 2.0],
        theta_knots=[0.025, 0.06, 0.11, 0.19],
    )
    return SSVISurface(params)


def test_grid_diagnostics_certified_surface(certified_surface):
    """Verify that dense grid diagnostics pass on certified admissible SSVI surface."""
    config = GridConfig(
        k_min=-1.2,
        k_max=1.2,
        n_k=61,
        T_min=0.15,
        T_max=1.8,
        n_T=31,
        refinement_factor=2,
    )
    diag = run_grid_diagnostics(certified_surface, config)

    assert diag.grid_status == StatusCode.GRID_DIAGNOSTIC_PASSED
    assert diag.min_g >= -1e-6
    assert diag.negative_g_count == 0
    assert diag.min_w_T >= -1e-6
    assert diag.negative_w_T_count == 0
    assert diag.convexity_violation_count == 0


def test_butterfly_grid_catches_negative_g():
    """Verify butterfly grid diagnostic catches negative g on deliberately invalid surface."""
    # Construct an extreme surface with eta far beyond sufficient bounds causing butterfly arbitrage
    params = SSVIParameters(
        rho=-0.95,
        eta=15.0,  # Excessive eta causing negative Durrleman g near ATM
        gamma=0.1,
        expiries=[0.1, 1.0],
        theta_knots=[0.01, 0.05],
    )
    surface = SSVISurface(params)
    k_grid = np.linspace(-0.5, 0.5, 101)
    T_grid = np.linspace(0.1, 1.0, 11)

    res = check_butterfly_arbitrage_grid(surface, k_grid, T_grid)
    assert res.min_g < 0.0
    assert res.negative_g_count > 0
    assert res.status == StatusCode.GRID_DIAGNOSTIC_FAILED

