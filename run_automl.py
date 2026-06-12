"""Simple command-line entry point for the AutoML workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.pipeline import run_comparison_workflow, run_minimal_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TEP anomaly-detection AutoML workflow.")
    parser.add_argument(
        "--data-dir",
        default="automl/data",
        help="Path to the folder containing the TEP parquet files.",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        help="Compare detectors. If no names are given, all registered detectors are used.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.compare is not None:
        detector_names = args.compare or None
        result = run_comparison_workflow(data_dir, detector_names)
    else:
        result = run_minimal_workflow(data_dir)

    print(result)


if __name__ == "__main__":
    main()