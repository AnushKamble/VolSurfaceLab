# AUDIT_LOG.md — Independent Audit Findings & Resolutions

This log tracks all findings identified during independent review, static analysis, adversarial stress testing, and regression runs.

## Severity Classification
- **P0:** Critical mathematical defect, silent arbitrage violation, silent data corruption, unbracketed root finder failure, or broken pipeline.
- **P1:** Incorrect numerical derivative, missing boundary condition, improper parameter clipping, failure in clean synthetic recovery, or missing required specification requirement.
- **P2:** Suboptimal initialisation, edge-case conditioning warning not surfaced, minor documentation discrepancy.
- **P3:** Code style, naming consistency, minor visual polish.

---

| Finding ID | Severity | Phase | Description | Reproduction / Evidence | Root Cause & Fix | Regression Test | Disposition |
|---|---|---|---|---|---|---|---|
| AUD-INIT | Info | Phase 0 | Audit log established. Initial state clean. | Inspection | None required | N/A | CLOSED |
| AUD-01 | P2 | Phase 4 | Finite-difference second derivative $w_{kk}$ comparison test failed at step size $h=10^{-5}$ due to floating-point subtractive cancellation ($\epsilon_{mach} / h^2 \approx 10^{-6}$). | `tests/unit/test_ssvi_derivatives.py` assertion failure `7.5e-5 < 1e-5`. | In float64, optimal step size for second-order central difference is $h \sim \epsilon_{mach}^{1/4} \approx 10^{-4}$. Updated test step size to $h=10^{-4}$ where relative error dropped to $8.0 \times 10^{-8}$. | `test_analytic_w_kk_vs_central_difference` | CLOSED |
| AUD-02 | P2 | Phase 7 | `DensityConfig` enforced overly restrictive lower bound `std_dev_range >= 3.0`, preventing deliberate diagnostic checks on narrow truncated domains. | `tests/unit/test_density.py` validation error when instantiating `std_dev_range=1.5`. | Relaxed `DensityConfig.std_dev_range` constraint to `ge=1.0` to support truncation testing. | `test_density_no_artificial_renormalisation` | CLOSED |
| AUD-03 | P2 | Phase 10 | `RunArtifact.environment` was typed as `Dict[str, str]`, causing Pydantic validation error when storing multi-start metadata dictionaries. | `cli_demo.py` raised ValidationError on `all_starts`. | Updated `RunArtifact.environment` type to `Dict[str, Any]`. | `test_cli_demo_execution` | CLOSED |
| AUD-04 | P2 | Phase 13 | Adversarial Scenario 6 failed because shifting both call and put bids/asks by +5.0 preserved $C - P = D(F - K)$, unintentionally keeping put-call parity exact. | `test_scenario_06_parity_inconsistent_call_put_set` failed assertion `violations > 0`. | Modified corruption to perturb only the call contract, creating genuine parity arbitrage. | `test_scenario_06_parity_inconsistent_call_put_set` | CLOSED |

### Summary
- Total Findings: 4
- Open P0 / P1 Findings: **0**
- Open P2 / P3 Findings: **0**
- Final Quality Status: **CLEAN / PRODUCTION-READY**
