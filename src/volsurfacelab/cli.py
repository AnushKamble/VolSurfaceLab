"""Command-Line Interface (CLI) entry point for VolSurfaceLab."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for VolSurfaceLab."""
    parser = argparse.ArgumentParser(
        prog="volsurfacelab",
        description="VolSurfaceLab — Arbitrage-Aware Volatility Surface & Local-Risk Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: demo
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run deterministic offline end-to-end pipeline using synthetic data",
    )
    demo_parser.add_argument(
        "--source",
        choices=["synthetic"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic market (default: 42)",
    )
    demo_parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_default.yaml",
        help="Path to YAML run configuration",
    )
    demo_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo",
        help="Directory to persist run artefacts and HTML report",
    )

    # Command: calibrate
    calib_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate SSVI surface to a CSV quotes dataset",
    )
    calib_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input option quotes CSV",
    )
    calib_parser.add_argument(
        "--config",
        type=str,
        default="configs/example_market_csv.yaml",
        help="Path to configuration YAML file",
    )
    calib_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/calibration",
        help="Directory to save calibration results",
    )

    # Command: verify
    verify_parser = subparsers.add_parser(
        "verify",
        help="Re-verify no-arbitrage theorem conditions and grid diagnostics from saved artefact",
    )
    verify_parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Path to calibration.json or run_artifact.json",
    )
    verify_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save verification outputs",
    )

    # Command: report
    report_parser = subparsers.add_parser(
        "report",
        help="Generate standalone HTML report from a saved calibration artefact",
    )
    report_parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Path to calibration.json or run_artifact.json",
    )
    report_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/report",
        help="Directory to save HTML report",
    )

    # Command: benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run ablation comparisons between SSVI, Raw SVI, and Flat Black baseline",
    )
    bench_parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_default.yaml",
        help="Path to configuration YAML file",
    )
    bench_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Directory to persist benchmark results",
    )

    # Command: dashboard
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the optional Streamlit interactive visual dashboard",
    )
    dash_parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Path to saved run_artifact.json (optional)",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI execution dispatcher."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == "demo":
        from volsurfacelab.cli_demo import run_demo_pipeline
        return run_demo_pipeline(parsed.seed, parsed.config, parsed.output_dir)

    if parsed.command == "calibrate":
        from volsurfacelab.cli_demo import run_calibrate_pipeline
        return run_calibrate_pipeline(parsed.input, parsed.config, parsed.output_dir)

    if parsed.command == "verify":
        from volsurfacelab.cli_demo import run_verify_pipeline
        return run_verify_pipeline(parsed.artifact, parsed.output_dir)

    if parsed.command == "report":
        from volsurfacelab.cli_demo import run_report_pipeline
        return run_report_pipeline(parsed.artifact, parsed.output_dir)

    if parsed.command == "benchmark":
        from volsurfacelab.cli_demo import run_benchmark_pipeline
        return run_benchmark_pipeline(parsed.config, parsed.output_dir)

    if parsed.command == "dashboard":
        import subprocess
        from pathlib import Path
        app_path = Path(__file__).parent / "dashboard" / "app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(parsed.port)]
        return subprocess.run(cmd).returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
