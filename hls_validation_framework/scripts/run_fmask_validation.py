#!/usr/bin/env python3
"""
run_fmask_validation.py
-----------------------
Programmatic runner for HLS_Fmask_acceptance_test.ipynb using Papermill.

Usage:
    # From repo root:
    python hls_validation_framework/scripts/run_fmask_validation.py

    # With custom config or output path:
    python hls_validation_framework/scripts/run_fmask_validation.py \
        --config hls_validation_framework/config/fmask_acceptance_config.yaml \
        --output hls_validation_framework/reports/my_run.ipynb

Requirements:
    pip install papermill

Environment variables required (never pass credentials as arguments):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN  (if using temporary credentials)
"""

import argparse
import datetime
import os
import sys

# Resolve framework root (one level up from this script)
FRAMEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Run the HLS Fmask acceptance test notebook via Papermill."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(FRAMEWORK_ROOT, "config", "fmask_acceptance_config.yaml"),
        help="Path to the YAML config file (default: hls_validation_framework/config/fmask_acceptance_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the executed output notebook "
            "(default: hls_validation_framework/reports/fmask_validation_<timestamp>.ipynb)"
        ),
    )
    args = parser.parse_args()

    notebook_in  = os.path.join(FRAMEWORK_ROOT, "HLS_Fmask_acceptance_test.ipynb")
    config_path  = os.path.abspath(args.config)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_out = args.output or os.path.join(
        FRAMEWORK_ROOT, "reports", f"fmask_validation_{ts}.ipynb"
    )
    os.makedirs(os.path.dirname(notebook_out), exist_ok=True)

    # ── Check AWS credentials ──────────────────────────────────────────────────
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("❌ AWS_ACCESS_KEY_ID not set. Export your credentials before running.")
        sys.exit(1)

    # ── Check papermill ────────────────────────────────────────────────────────
    try:
        import papermill as pm
    except ImportError:
        print("❌ papermill not installed. Run: pip install papermill")
        sys.exit(1)

    print(f"📓 Input notebook  : {notebook_in}")
    print(f"⚙️  Config          : {config_path}")
    print(f"📄 Output notebook : {notebook_out}")
    print(f"{'─'*60}")

    pm.execute_notebook(
        input_path=notebook_in,
        output_path=notebook_out,
        parameters={"config_path": config_path},
        kernel_name="python3",
        progress_bar=True,
    )

    print(f"\n✅ Validation complete. Report saved to: {notebook_out}")
    print(f"   View with: jupyter nbconvert --to html {notebook_out}")


if __name__ == "__main__":
    main()
