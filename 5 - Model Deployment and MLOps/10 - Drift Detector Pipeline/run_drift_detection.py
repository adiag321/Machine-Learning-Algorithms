# -*- coding: utf-8 -*-
"""
run_drift_detection.py — CLI entry point for the Drift Detector Pipeline.

Usage:
    # Default — uses paths from config.py (Drift_Detection/reference_data.csv + current_data.csv):
        python3 run_drift_detection.py

    # Custom datasets:
        python3 run_drift_detection.py --reference path/to/ref.csv --current path/to/cur.csv --target Churn

    # With batch trend analysis:
        python3 run_drift_detection.py --timestamp date_column --output ./reports
"""

import argparse
import sys
import os

# Make the package importable regardless of the working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detector.pipeline import DriftPipeline
from drift_detector.config import get_logger

log = get_logger("run_drift_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect data and concept drift between a reference and current dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reference", "-r", type=str, default="",
                        help="Path to reference (baseline) CSV. Defaults to config.REFERENCE_DATA_PATH.")
    parser.add_argument("--current", "-c", type=str, default="",
                        help="Path to current (production) CSV. Defaults to config.CURRENT_DATA_PATH.")
    parser.add_argument("--target", "-t", type=str, default=None,
                        help="Binary label column name (e.g. 'Churn'). Required for concept drift.")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Date column for time-series batch trend analysis.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for all reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = DriftPipeline(
        reference_path=args.reference,
        current_path=args.current,
        target_col=args.target,
        timestamp_col=args.timestamp,
        output_dir=args.output,
    )

    try:
        summary = pipeline.run()
        overall = summary.get("overall_severity", "unknown").upper()
        print(f"\n✅  Pipeline complete — Overall status: {overall}")
        print(f"    Reports saved to: {pipeline.output_dir}\n")

    except FileNotFoundError as e:
        log.error(str(e))
        print(f"\n❌  {e}")
        print("    Run create_drift_datasets.py first, or pass --reference / --current paths.\n")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        print(f"\n❌  Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
