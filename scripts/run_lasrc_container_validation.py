#!/usr/bin/env python3
"""Thin wrapper for the notebook-derived LaSRC validation CLI."""

from pathlib import Path
import runpy


CLI_SCRIPT = Path(__file__).resolve().with_name("lasrc_container_validation_cli.py")


if __name__ == "__main__":
    runpy.run_path(str(CLI_SCRIPT), run_name="__main__")
