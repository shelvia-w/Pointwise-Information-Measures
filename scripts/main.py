"""Command-line entry point for paper experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.benchmark_configs import BENCHMARKS, DEFAULT_BENCHMARK


COMMANDS = {
    "train_models": "experiments.train_models",
    "pmi_analysis": "experiments.run_pmi_analysis",
    "pvi_analysis": "experiments.run_pvi_analysis",
    "psi_analysis": "experiments.run_psi_analysis",
    "filtering_error": "experiments.run_filtering_error",
    "reliability_diagram": "experiments.run_reliability_diagram",
    "calibration_error": "experiments.run_calibration_error",
    "selective_prediction": "experiments.run_selective_prediction",
    "umap": "experiments.run_umap",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run pointwise-information paper experiments.")
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, choices=BENCHMARKS.keys())
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument(
        "--psi-estimator",
        choices=["histogram", "gaussian", "random_forest", "neural"],
        default=None,
        help="Override the PSI estimator for psi_analysis.",
    )
    return parser


def main(argv: list[str] | None = None):
    """Parse arguments and dispatch the selected experiment."""
    args = build_parser().parse_args(argv)
    module = __import__(COMMANDS[args.command], fromlist=["run"])
    result = module.run(args)
    output_dir = result.get("output_dir") if isinstance(result, dict) else None
    if output_dir:
        print(f"Finished {args.command}. Outputs written to {output_dir}")
    else:
        print(f"Finished {args.command}.")


if __name__ == "__main__":
    main()
