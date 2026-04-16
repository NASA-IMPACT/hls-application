#!/usr/bin/env python3
"""
run_fmask_validation.py
-----------------------
Programmatic runner for the HLS validation notebooks using Papermill.

Supports both notebooks:
  - notebooks/HLS_Fmask_acceptance_test.ipynb  (Fmask acceptance test)
  - notebooks/HLS_validation_general.ipynb      (SR band regression test)

Usage:
    # From repo root — run Fmask acceptance test (default):
    python hls_validation_framework/scripts/run_fmask_validation.py

    # Run SR regression test:
    python hls_validation_framework/scripts/run_fmask_validation.py --test sr

    # Custom config and output:
    python hls_validation_framework/scripts/run_fmask_validation.py \
        --test fmask \
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

# Framework root: one level above this script (hls_validation_framework/)
FRAMEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NOTEBOOKS = {
    "fmask": os.path.join(FRAMEWORK_ROOT, "notebooks", "HLS_Fmask_acceptance_test.ipynb"),
    "sr":    os.path.join(FRAMEWORK_ROOT, "notebooks", "HLS_validation_general.ipynb"),
}

DEFAULT_CONFIGS = {
    "fmask": os.path.join(FRAMEWORK_ROOT, "config", "fmask_acceptance_config.yaml"),
    "sr":    os.path.join(FRAMEWORK_ROOT, "config", "sr_regression_config.yaml"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run HLS validation notebooks via Papermill."
    )
    parser.add_argument(
        "--test",
        choices=["fmask", "sr"],
        default="fmask",
        help="Which test to run: 'fmask' (Fmask acceptance) or 'sr' (SR regression). Default: fmask",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file (default: config/<test>_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the executed output notebook (default: reports/<test>_<timestamp>.ipynb)",
    )
    args = parser.parse_args()

    notebook_in = NOTEBOOKS[args.test]
    config_path = os.path.abspath(args.config or DEFAULT_CONFIGS[args.test])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_out = args.output or os.path.join(
        FRAMEWORK_ROOT, "reports", f"{args.test}_validation_{ts}.ipynb"
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

    print(f"🧪 Test            : {args.test.upper()}")
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

    print(f"\n✅ Complete. Report saved to: {notebook_out}")
    print(f"   View with: jupyter nbconvert --to html {notebook_out}")


if __name__ == "__main__":
    main()
