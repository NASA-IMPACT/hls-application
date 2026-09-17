#!/usr/bin/env python3
"""
run_fmask_validation.py
-----------------------
Programmatic runner for the HLS Fmask acceptance notebook using Papermill.

Usage:
    # From repo root:
    python scripts/run_fmask_validation.py

    # Custom config and output:
    python scripts/run_fmask_validation.py \
        --config config/fmask_acceptance_config.yaml \
        --output reports/my_run.ipynb

Requirements:
    pip install papermill

Credential options (never pass credentials as arguments):
    AWS_PROFILE        (for an AWS CLI or SSO profile)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
                       (for temporary or exported credentials)
"""

import argparse
import datetime
import os
import sys

# Repository root: one level above this script.
REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NOTEBOOK = os.path.join(REPOSITORY_ROOT, "notebooks", "HLS_Fmask_acceptance_test.ipynb")
DEFAULT_CONFIG = os.path.join(REPOSITORY_ROOT, "config", "fmask_acceptance_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run the HLS Fmask acceptance notebook via Papermill.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file (default: config/fmask_acceptance_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the executed output notebook (default: reports/fmask_validation_<timestamp>.ipynb)",
    )
    args = parser.parse_args()

    notebook_in = NOTEBOOK
    config_path = os.path.abspath(args.config or DEFAULT_CONFIG)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_out = args.output or os.path.join(
        REPOSITORY_ROOT, "reports", f"fmask_validation_{ts}.ipynb"
    )
    os.makedirs(os.path.dirname(notebook_out), exist_ok=True)

    # ── Check AWS credentials ──────────────────────────────────────────────────
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        print("❌ No AWS credential source found. Export AWS credentials or set AWS_PROFILE before running.")
        sys.exit(1)

    # ── Check papermill ────────────────────────────────────────────────────────
    try:
        import papermill as pm
    except ImportError:
        print("❌ papermill not installed. Run: pip install papermill")
        sys.exit(1)

    print("🧪 Test            : FMASK ACCEPTANCE")
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
