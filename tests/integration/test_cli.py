"""Integration tests for VolSurfaceLab CLI entry points and workflows."""

from pathlib import Path
import pytest

from volsurfacelab.cli import main


def test_cli_demo_execution(tmp_path):
    """Verify demo pipeline runs end-to-end and saves all required artefacts."""
    out_dir = tmp_path / "test_demo_out"
    ret = main(["demo", "--source", "synthetic", "--seed", "123", "--output-dir", str(out_dir)])
    assert ret == 0

    # Verify expected files exist
    assert (out_dir / "run_config.yaml").exists()
    assert (out_dir / "provenance.json").exists()
    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "admissibility_report.json").exists()
    assert (out_dir / "grid_diagnostics.json").exists()
    assert (out_dir / "fit_metrics.json").exists()
    assert (out_dir / "local_vol_grid.csv").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "run_artifact.json").exists()


def test_cli_verify_execution(tmp_path):
    """Verify verify workflow on saved calibration artifact."""
    out_dir = tmp_path / "test_verify_out"
    main(["demo", "--source", "synthetic", "--seed", "42", "--output-dir", str(out_dir)])
    calib_json = out_dir / "calibration.json"

    ret = main(["verify", "--artifact", str(calib_json)])
    assert ret == 0


def test_cli_report_execution(tmp_path):
    """Verify report workflow generates HTML report."""
    demo_dir = tmp_path / "demo_for_report"
    main(["demo", "--source", "synthetic", "--seed", "42", "--output-dir", str(demo_dir)])
    report_out = tmp_path / "report_out"

    ret = main(["report", "--artifact", str(demo_dir / "run_artifact.json"), "--output-dir", str(report_out)])
    assert ret == 0
    assert (report_out / "report.html").exists()


def test_cli_benchmark_execution(tmp_path):
    """Verify benchmark workflow executes cleanly."""
    out_dir = tmp_path / "bench_out"
    ret = main(["benchmark", "--output-dir", str(out_dir)])
    assert ret == 0
