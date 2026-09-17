#!/usr/bin/env python3
"""Batch LaSRC container validation script.

This script captures the core workflow from LaSRC_container_validation.ipynb:
1. discover pre/post rasters from local directories or S3 prefixes
2. build a symmetric pre/post band catalog
3. compute pixel-level comparison metrics for each pair
4. export summary CSVs
5. render per-granule panels and summary plots
6. optionally compare runtime logs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:  # pragma: no cover
    boto3 = None
    NoCredentialsError = RuntimeError

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover
    pearsonr = None

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


_GDAL = None


def get_gdal():
    global _GDAL
    if _GDAL is None:
        try:
            from osgeo import gdal as _imported_gdal
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GDAL (osgeo.gdal) is required. Activate the validation environment first."
            ) from exc
        _imported_gdal.UseExceptions()
        _GDAL = _imported_gdal
    return _GDAL


def try_get_gdal():
    try:
        return get_gdal()
    except ImportError:
        return None


REFERENCE_FINAL_OUTPUT_THRESHOLDS = {
    "B01": 0.0051,
    "B02": 0.0048,
    "B03": 0.0055,
    "B04": 0.0052,
    "B05": 0.0075,
    "B06": 0.0093,
    "B07": 0.0064,
    "B8A": 0.0075,
    "B11": 0.0093,
    "B12": 0.0064,
}

REFERENCE_IMMEDIATE_ACIX_THRESHOLDS_S30 = {
    "B01": 0.0053,
    "B02": 0.0056,
    "B03": 0.0054,
    "B04": 0.0066,
    "B8A": 0.0099,
    "B11": 0.0131,
    "B12": 0.0087,
}

REFERENCE_IMMEDIATE_ACIX_THRESHOLDS_L30 = {
    "B01": 0.0053,
    "B02": 0.0056,
    "B03": 0.0054,
    "B04": 0.0066,
    "B05": 0.0099,
    "B06": 0.0131,
    "B07": 0.0087,
}

DEFAULT_BAND_LABELS = {
    "B01": "Ultrablue",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Red Edge 1",
    "B06": "Red Edge 2",
    "B07": "Red Edge 3",
    "B08": "NIR Broad",
    "B8A": "NIR Narrow",
    "B09": "Water Vapor",
    "B10": "Cirrus",
    "B11": "SWIR1",
    "B12": "SWIR2",
}

PRODUCT_BAND_LABELS = {
    "S30": dict(DEFAULT_BAND_LABELS),
    "L30": {
        "B01": "Coastal Aerosol",
        "B02": "Blue",
        "B03": "Green",
        "B04": "Red",
        "B05": "NIR",
        "B06": "SWIR1",
        "B07": "SWIR2",
    },
}

BAND_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

S30_IMMEDIATE_BAND_MAP = {
    "1": "B01",
    "2": "B02",
    "3": "B03",
    "4": "B04",
    "5": "B05",
    "6": "B06",
    "7": "B07",
    "8": "B08",
    "8A": "B8A",
    "9": "B09",
    "10": "B10",
    "11": "B11",
    "12": "B12",
}

L30_IMMEDIATE_BAND_MAP = {
    "1": "B01",
    "2": "B02",
    "3": "B03",
    "4": "B04",
    "5": "B05",
    "6": "B06",
    "7": "B07",
}

DEFAULT_PLOT_BANDS_BY_PRODUCT = {
    "S30": ["B01", "B02", "B03", "B04", "B11", "B12"],
    "L30": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "folder_pre": "hls-science-container-testing/lasrc-port/c/",
    "folder_post": "hls-science-container-testing/lasrc-port/rust/",
    "pre_label": "Pre container",
    "post_label": "Post container",
    "aws_profile": None,
    "aws_region": "us-west-2",
    "s3_read_mode": "cache",
    "data_source_mode": "auto",
    "local_pre_root": None,
    "local_post_root": None,
    "comparison_stage": "auto",
    "granule_filters": [],
    "granule_filter_file": None,
    "max_granules": None,
    "bands_to_include": [],
    "bands_to_exclude": ["B8A"],
    "panel_granule": None,
    "panel_stage": "auto",
    "panel_granules": [],
    "render_single_panel": True,
    "render_all_panels": False,
    "pixel_diff_eps": 0.0,
    "panel_bands": [],
    "hls_scale_factor": 0.0001,
    "reflectance_scale_factor": None,
    "reflectance_add_offset": 0.0,
    "immediate_reflectance_scale_factor": None,
    "immediate_reflectance_add_offset": None,
    "threshold_mode": "reference",
    "threshold_units": "reflectance",
    "threshold_global_default": None,
    "final_output_thresholds": {},
    "immediate_thresholds_s30": {},
    "immediate_thresholds_l30": {},
    "final_fill_value": -9999,
    "immediate_fill_value": 0,
    "pre_zero_is_nodata": False,
    "post_zero_is_nodata": False,
    "runtime_log_source": None,
    "save_summary_plots": True,
    "env_file": None,
    "output_root": "./lasrc_validation_run",
}


@dataclass
class ValidationConfig:
    folder_pre: str
    folder_post: str
    pre_label: str
    post_label: str
    aws_profile: Optional[str]
    aws_region: str
    s3_read_mode: str
    data_source_mode: str
    local_pre_root: Optional[str]
    local_post_root: Optional[str]
    comparison_stage: str
    granule_filters: List[str] = field(default_factory=list)
    granule_filter_file: Optional[str] = None
    max_granules: Optional[int] = None
    bands_to_include: List[str] = field(default_factory=list)
    bands_to_exclude: List[str] = field(default_factory=list)
    panel_granule: Optional[str] = None
    panel_stage: str = "auto"
    panel_granules: List[str] = field(default_factory=list)
    render_single_panel: bool = True
    render_all_panels: bool = False
    pixel_diff_eps: float = 0.0
    panel_bands: List[str] = field(default_factory=list)
    hls_scale_factor: float = 0.0001
    reflectance_scale_factor: Optional[float] = None
    reflectance_add_offset: float = 0.0
    immediate_reflectance_scale_factor: Optional[float] = None
    immediate_reflectance_add_offset: Optional[float] = None
    threshold_mode: str = "reference"
    threshold_units: str = "reflectance"
    threshold_global_default: Optional[float] = None
    final_output_thresholds: Dict[str, float] = field(default_factory=dict)
    immediate_thresholds_s30: Dict[str, float] = field(default_factory=dict)
    immediate_thresholds_l30: Dict[str, float] = field(default_factory=dict)
    final_fill_value: int = -9999
    immediate_fill_value: int = 0
    pre_zero_is_nodata: bool = False
    post_zero_is_nodata: bool = False
    runtime_log_source: Optional[str] = None
    save_summary_plots: bool = True
    env_file: Optional[str] = None
    output_root: str = "./lasrc_validation_run"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        granule_filters = list(merged.get("granule_filters") or [])
        granule_filter_file = merged.get("granule_filter_file")
        if granule_filter_file:
            path = Path(granule_filter_file).expanduser()
            granule_filters.extend(
                [line.strip() for line in path.read_text().splitlines() if line.strip()]
            )
        merged["granule_filters"] = sorted(dict.fromkeys(granule_filters))
        if merged.get("reflectance_scale_factor") in ("", None):
            merged["reflectance_scale_factor"] = float(merged.get("hls_scale_factor", 0.0001))
        else:
            merged["reflectance_scale_factor"] = float(merged["reflectance_scale_factor"])
        if merged.get("reflectance_add_offset") in ("", None):
            merged["reflectance_add_offset"] = 0.0
        else:
            merged["reflectance_add_offset"] = float(merged["reflectance_add_offset"])
        if merged.get("immediate_reflectance_scale_factor") in ("", None):
            merged["immediate_reflectance_scale_factor"] = None
        else:
            merged["immediate_reflectance_scale_factor"] = float(
                merged["immediate_reflectance_scale_factor"]
            )
        if merged.get("immediate_reflectance_add_offset") in ("", None):
            merged["immediate_reflectance_add_offset"] = None
        else:
            merged["immediate_reflectance_add_offset"] = float(
                merged["immediate_reflectance_add_offset"]
            )
        for field_name in (
            "final_output_thresholds",
            "immediate_thresholds_s30",
            "immediate_thresholds_l30",
        ):
            raw_values = merged.get(field_name) or {}
            merged[field_name] = {
                str(key).upper(): float(value)
                for key, value in raw_values.items()
                if value is not None and str(value) != ""
            }
        if merged.get("threshold_global_default") in ("", None):
            merged["threshold_global_default"] = None
        else:
            merged["threshold_global_default"] = float(merged["threshold_global_default"])
        return cls(**merged)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def stage_display_name(stage: str) -> str:
    return {
        "final": "Final outputs",
        "immediate": "Immediate outputs after LaSRC",
    }.get(stage, stage)


def band_sort_key(band: str) -> int:
    return BAND_ORDER.index(band) if band in BAND_ORDER else 999


def get_band_label(product: Optional[str], band: str) -> str:
    if product in PRODUCT_BAND_LABELS and band in PRODUCT_BAND_LABELS[product]:
        return PRODUCT_BAND_LABELS[product][band]
    return DEFAULT_BAND_LABELS.get(band, band)


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


ENVI_DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


class ValidationRunner:
    def __init__(self, config: ValidationConfig) -> None:
        self.cfg = config
        self.output_root = Path(config.output_root).expanduser().resolve()
        self.report_dir = self.output_root / "reports"
        self.plots_dir = self.output_root / "plots"
        self.panel_dir = self.plots_dir / "granule_panels"
        self.summary_plot_dir = self.plots_dir / "summary"
        self.cache_dir = self.output_root / "_cache"
        for path in (
            self.output_root,
            self.report_dir,
            self.plots_dir,
            self.panel_dir,
            self.summary_plot_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.session = None
        self.s3 = None
        self.paired_catalog = pd.DataFrame()
        self.df_results = pd.DataFrame()
        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, message: str) -> None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    def get_reflectance_scaling(self, stage: str) -> Tuple[float, float]:
        if stage == "immediate" and self.cfg.immediate_reflectance_scale_factor is not None:
            return (
                float(self.cfg.immediate_reflectance_scale_factor),
                float(self.cfg.immediate_reflectance_add_offset or 0.0),
            )
        return float(self.cfg.reflectance_scale_factor), float(self.cfg.reflectance_add_offset)

    def threshold_value_to_reflectance(self, value: Optional[float], stage: str) -> float:
        if value is None or not np.isfinite(value):
            return np.nan
        units = str(self.cfg.threshold_units).lower()
        if units == "dn":
            scale, _ = self.get_reflectance_scaling(stage)
            return float(value) * scale
        if units == "reflectance":
            return float(value)
        raise ValueError(
            f"Unsupported threshold_units={self.cfg.threshold_units!r}. "
            "Use 'reflectance' or 'dn'."
        )

    @staticmethod
    def reference_threshold_lookup(product: str, stage: str) -> Dict[str, float]:
        if stage == "final":
            return REFERENCE_FINAL_OUTPUT_THRESHOLDS
        if stage == "immediate" and product == "S30":
            return REFERENCE_IMMEDIATE_ACIX_THRESHOLDS_S30
        if stage == "immediate" and product == "L30":
            return REFERENCE_IMMEDIATE_ACIX_THRESHOLDS_L30
        return {}

    def config_threshold_lookup(self, product: str, stage: str) -> Dict[str, float]:
        if stage == "final":
            return self.cfg.final_output_thresholds
        if stage == "immediate" and product == "S30":
            return self.cfg.immediate_thresholds_s30
        if stage == "immediate" and product == "L30":
            return self.cfg.immediate_thresholds_l30
        return {}

    def resolve_threshold(self, product: str, stage: str, band: str) -> Tuple[float, str]:
        mode = str(self.cfg.threshold_mode).lower()
        reference_lookup = self.reference_threshold_lookup(product, stage)
        config_lookup = self.config_threshold_lookup(product, stage)

        reference_threshold = reference_lookup.get(band, np.nan)
        override_threshold = config_lookup.get(band, self.cfg.threshold_global_default)

        reference_refl = self.threshold_value_to_reflectance(reference_threshold, stage)
        override_refl = self.threshold_value_to_reflectance(override_threshold, stage)
        units_label = str(self.cfg.threshold_units).lower()

        if mode == "reference":
            if np.isfinite(reference_refl):
                return reference_refl, "Reference threshold"
            return np.nan, "No threshold available"

        if mode == "custom":
            if np.isfinite(override_refl):
                return override_refl, f"Custom threshold ({units_label})"
            return np.nan, "No custom threshold available"

        if mode == "cap_reference":
            if np.isfinite(reference_refl) and np.isfinite(override_refl):
                return min(reference_refl, override_refl), f"Reference threshold capped ({units_label})"
            if np.isfinite(reference_refl):
                return reference_refl, "Reference threshold"
            if np.isfinite(override_refl):
                return override_refl, f"Custom threshold ({units_label})"
            return np.nan, "No threshold available"

        raise ValueError(
            f"Unsupported threshold_mode={self.cfg.threshold_mode!r}. "
            "Use 'reference', 'custom', or 'cap_reference'."
        )

    def threshold_plot_note(self) -> Optional[str]:
        units = str(self.cfg.threshold_units).lower()
        default_val = self.cfg.threshold_global_default
        if units == "dn" and default_val is not None and np.isfinite(default_val):
            return f"{float(default_val):g} DN reference"
        if units == "reflectance" and default_val is not None and np.isfinite(default_val):
            hls_scale = float(self.cfg.hls_scale_factor)
            if hls_scale > 0:
                hls_dn = float(default_val) / hls_scale
                if abs(hls_dn - round(hls_dn)) < 1e-6:
                    hls_dn_text = f"{int(round(hls_dn))} HLS DN"
                else:
                    hls_dn_text = f"{hls_dn:.3f} HLS DN"
                return f"{float(default_val):.4f} reflectance; equivalent to {hls_dn_text}"
        return None

    def distribution_threshold_label(self, value: float, stage: str) -> str:
        if not np.isfinite(value):
            return "n/a"
        units = str(self.cfg.threshold_units).lower()
        if units == "dn":
            default_val = self.cfg.threshold_global_default
            if default_val is not None and np.isfinite(default_val):
                scale, _ = self.get_reflectance_scaling(stage)
                return f"{float(default_val) * scale:.4f}"
        if units == "reflectance":
            hls_scale = float(self.cfg.hls_scale_factor)
            if hls_scale > 0:
                hls_dn = float(value) / hls_scale
                if abs(hls_dn - round(hls_dn)) < 1e-6:
                    hls_dn_text = f"{int(round(hls_dn))} HLS DN"
                else:
                    hls_dn_text = f"{hls_dn:.3f} HLS DN"
                return f"{value:.4f} refl.\n({hls_dn_text})"
        return f"{value:.4f}"

    def threshold_display_with_reflectance(self, value: float, stage: str) -> str:
        if not np.isfinite(value):
            return "n/a"
        units = str(self.cfg.threshold_units).lower()
        if units == "dn":
            scale, _ = self.get_reflectance_scaling(stage)
            if scale == 0:
                return self.format_threshold(value, stage)
            dn_value = float(value) / scale
            if abs(dn_value - round(dn_value)) < 1e-6:
                dn_text = f"{int(round(dn_value))} DN"
            else:
                dn_text = f"{dn_value:.3f} DN"
            return f"{dn_text} ({value:.4f})"
        if units == "reflectance":
            hls_scale = float(self.cfg.hls_scale_factor)
            if hls_scale > 0:
                hls_dn = float(value) / hls_scale
                if abs(hls_dn - round(hls_dn)) < 1e-6:
                    hls_dn_text = f"{int(round(hls_dn))} HLS DN"
                else:
                    hls_dn_text = f"{hls_dn:.3f} HLS DN"
                return f"{value:.4f} reflectance\n({hls_dn_text})"
        return self.format_threshold(value, stage)

    def load_env_file(self) -> None:
        env_file = self.cfg.env_file
        if not env_file:
            return
        path = Path(env_file).expanduser()
        if not path.exists():
            self.log(f"Environment file not found, skipping: {path}")
            return
        loaded = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ[key] = value
            loaded.append(key)
        if loaded:
            self.log(f"Loaded environment variables from {path}")

    def needs_s3(self) -> bool:
        def looks_local(value: Optional[str]) -> bool:
            if not value:
                return False
            return Path(str(value)).expanduser().exists()

        if self.cfg.data_source_mode == "s3":
            return True
        if self.cfg.data_source_mode == "local":
            return False
        if not looks_local(self.cfg.folder_pre) or not looks_local(self.cfg.folder_post):
            return True
        if self.cfg.runtime_log_source and not looks_local(self.cfg.runtime_log_source):
            return True
        return False

    def setup_aws(self) -> None:
        if not self.needs_s3():
            return
        if boto3 is None:  # pragma: no cover
            raise ImportError("boto3 is required for S3-backed validation.")
        session_kwargs = {"region_name": self.cfg.aws_region}
        if self.cfg.aws_profile:
            session_kwargs["profile_name"] = self.cfg.aws_profile
        self.session = boto3.Session(**session_kwargs)
        creds = self.session.get_credentials()
        if creds is None:
            raise NoCredentialsError()
        frozen = creds.get_frozen_credentials()
        os.environ["AWS_DEFAULT_REGION"] = self.cfg.aws_region
        os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
        if frozen.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen.token

        gdal = try_get_gdal()
        if gdal is not None:
            gdal.SetConfigOption("AWS_REGION", self.cfg.aws_region)
            gdal.SetConfigOption("AWS_ACCESS_KEY_ID", frozen.access_key)
            gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", frozen.secret_key)
            if frozen.token:
                gdal.SetConfigOption("AWS_SESSION_TOKEN", frozen.token)
        self.s3 = self.session.client("s3", region_name=self.cfg.aws_region)
        self.log(f"AWS credentials loaded ({frozen.access_key[:8]}...)")

    @staticmethod
    def split_bucket_prefix(s3_path: str) -> Tuple[str, str]:
        cleaned = s3_path.strip().replace("s3://", "").strip("/")
        if not cleaned:
            raise ValueError("S3 path is empty")
        if "/" in cleaned:
            bucket, prefix = cleaned.split("/", 1)
            prefix = prefix.rstrip("/") + "/"
        else:
            bucket, prefix = cleaned, ""
        return bucket, prefix

    def list_s3_files(
        self,
        bucket: str,
        prefix: str = "",
        suffixes: Sequence[str] = (".tif", ".img"),
    ) -> List[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        files: List[str] = []
        total_size = 0
        suffixes_lc = tuple(s.lower() for s in suffixes)
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if suffixes_lc and not key.lower().endswith(suffixes_lc):
                    continue
                files.append(key)
                total_size += obj["Size"]
        self.log(
            f"s3://{bucket}/{prefix} -> {len(files)} matching rasters ({total_size / (1024 ** 3):.2f} GB)"
        )
        return files

    def list_local_files(
        self, root_path: str, suffixes: Sequence[str] = (".tif", ".img")
    ) -> List[str]:
        root = Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"Local root does not exist: {root}")
        suffixes_lc = tuple(s.lower() for s in suffixes)
        files = []
        total_size = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if suffixes_lc and not path.name.lower().endswith(suffixes_lc):
                continue
            files.append(path.relative_to(root).as_posix())
            total_size += path.stat().st_size
        self.log(
            f"local://{root} -> {len(files)} matching rasters ({total_size / (1024 ** 3):.2f} GB)"
        )
        return sorted(files)

    def determine_data_sources(self) -> Dict[str, Dict[str, str]]:
        mode = str(self.cfg.data_source_mode).lower()
        if mode not in {"auto", "local", "s3"}:
            raise ValueError(f"Unsupported data_source_mode={self.cfg.data_source_mode!r}")

        def local_existing(value: Optional[str]) -> Optional[Path]:
            if not value:
                return None
            candidate = Path(str(value)).expanduser()
            return candidate if candidate.exists() else None

        local_pre_root = local_existing(self.cfg.local_pre_root)
        local_post_root = local_existing(self.cfg.local_post_root)
        folder_pre_local = local_existing(self.cfg.folder_pre)
        folder_post_local = local_existing(self.cfg.folder_post)

        if mode == "local":
            if local_pre_root and local_post_root:
                return {
                    "pre": {"storage_type": "local", "storage_root": str(local_pre_root)},
                    "post": {"storage_type": "local", "storage_root": str(local_post_root)},
                }
            if folder_pre_local and folder_post_local:
                return {
                    "pre": {"storage_type": "local", "storage_root": str(folder_pre_local)},
                    "post": {"storage_type": "local", "storage_root": str(folder_post_local)},
                }
            raise FileNotFoundError(
                "Local mode requires LOCAL_PRE_ROOT / LOCAL_POST_ROOT or local folder_pre / folder_post."
            )

        if mode == "auto":
            if local_pre_root and local_post_root:
                return {
                    "pre": {"storage_type": "local", "storage_root": str(local_pre_root)},
                    "post": {"storage_type": "local", "storage_root": str(local_post_root)},
                }
            if folder_pre_local and folder_post_local:
                return {
                    "pre": {"storage_type": "local", "storage_root": str(folder_pre_local)},
                    "post": {"storage_type": "local", "storage_root": str(folder_post_local)},
                }

        bucket_pre, prefix_pre = self.split_bucket_prefix(self.cfg.folder_pre)
        bucket_post, prefix_post = self.split_bucket_prefix(self.cfg.folder_post)
        return {
            "pre": {"storage_type": "s3", "storage_root": bucket_pre, "prefix": prefix_pre},
            "post": {"storage_type": "s3", "storage_root": bucket_post, "prefix": prefix_post},
        }

    @staticmethod
    def format_storage_location(storage_type: str, storage_root: str, key: str) -> str:
        if storage_type == "local":
            return str(Path(storage_root) / key)
        if storage_type == "s3":
            return f"s3://{storage_root}/{key}"
        raise ValueError(f"Unsupported storage_type={storage_type!r}")

    @staticmethod
    def extract_granule_from_record(record_path: str) -> Optional[str]:
        hls_match = re.search(r"(HLS\.[^/]+?\.v\d+\.\d+)", record_path)
        if hls_match:
            return hls_match.group(1)
        landsat_match = re.search(
            r"((?:LC08|LC09|LO08|LO09|LE07|LT05)_[A-Z0-9]{4}_[0-9]{6}_[0-9]{8}_[0-9]{8}_[0-9]{2}_(?:T1|T2|RT))",
            record_path,
            re.IGNORECASE,
        )
        if landsat_match:
            return landsat_match.group(1).upper()
        return None

    @staticmethod
    def infer_product_from_granule(granule: str) -> Optional[str]:
        granule = str(granule).upper()
        if ".S30." in granule:
            return "S30"
        if ".L30." in granule:
            return "L30"
        if granule.startswith(("LC08_", "LC09_", "LO08_", "LO09_", "LE07_", "LT05_")):
            return "L30"
        return None

    @staticmethod
    def normalize_immediate_band(token: str, product: str) -> Optional[str]:
        token = str(token).upper()
        if product == "S30":
            return S30_IMMEDIATE_BAND_MAP.get(token)
        if product == "L30":
            return L30_IMMEDIATE_BAND_MAP.get(token)
        return None

    def get_threshold_bundle(self, product: str, stage: str, band: str) -> Dict[str, Any]:
        threshold, source = self.resolve_threshold(product, stage, band)
        return {
            "threshold_source": source,
            "threshold_refl": threshold,
            "strict_threshold_refl": threshold,
            "relaxed_threshold_refl": threshold,
            "reference_threshold_refl": threshold,
        }

    def build_records(
        self, file_keys: Sequence[str], storage_type: str, storage_root: str
    ) -> pd.DataFrame:
        final_regex = re.compile(
            r"^(HLS\.(S30|L30)\.[^.]+\.[^.]+\.v\d+\.\d+)\.(B(?:0[1-9]|1[0-2])|B8A)\.tif$",
            re.IGNORECASE,
        )
        immediate_regex = re.compile(r"_sr_band(\d{1,2}A?)\.img$", re.IGNORECASE)
        rows = []

        for key in sorted(file_keys):
            key = key.replace("\\", "/")
            basename = Path(key).name
            granule = self.extract_granule_from_record(key)
            if not granule:
                continue
            product = self.infer_product_from_granule(granule)
            if product is None:
                continue

            final_match = final_regex.match(basename)
            if final_match:
                band = final_match.group(3).upper()
                if self.cfg.bands_to_include and band not in set(self.cfg.bands_to_include):
                    continue
                if self.cfg.bands_to_exclude and band in set(self.cfg.bands_to_exclude):
                    continue
                bundle = self.get_threshold_bundle(product, "final", band)
                rows.append(
                    {
                        "pair_id": f"{granule}|final|{band}",
                        "granule": granule,
                        "product": product,
                        "stage": "final",
                        "band": band,
                        "band_label": get_band_label(product, band),
                        "key": key,
                        "storage_type": storage_type,
                        "storage_root": storage_root,
                        **bundle,
                    }
                )
                continue

            if basename.lower().endswith("_hdf.img"):
                continue

            immediate_match = immediate_regex.search(basename)
            if immediate_match:
                band = self.normalize_immediate_band(immediate_match.group(1), product)
                if not band:
                    continue
                if self.cfg.bands_to_include and band not in set(self.cfg.bands_to_include):
                    continue
                if self.cfg.bands_to_exclude and band in set(self.cfg.bands_to_exclude):
                    continue
                bundle = self.get_threshold_bundle(product, "immediate", band)
                rows.append(
                    {
                        "pair_id": f"{granule}|immediate|{band}",
                        "granule": granule,
                        "product": product,
                        "stage": "immediate",
                        "band": band,
                        "band_label": get_band_label(product, band),
                        "key": key,
                        "storage_type": storage_type,
                        "storage_root": storage_root,
                        **bundle,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if self.cfg.comparison_stage in {"final", "immediate"}:
            df = df[df["stage"] == self.cfg.comparison_stage].copy()
        df["band_order"] = df["band"].apply(band_sort_key)
        df = (
            df.sort_values(["granule", "stage", "band_order", "band"])
            .drop_duplicates(subset=["pair_id"], keep="first")
            .reset_index(drop=True)
        )
        return df

    def discover_records(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sources = self.determine_data_sources()
        self.log(f"Resolved data sources: {sources}")
        if sources["pre"]["storage_type"] == "local":
            pre_files = self.list_local_files(sources["pre"]["storage_root"])
        else:
            pre_files = self.list_s3_files(
                sources["pre"]["storage_root"], sources["pre"].get("prefix", "")
            )
        if sources["post"]["storage_type"] == "local":
            post_files = self.list_local_files(sources["post"]["storage_root"])
        else:
            post_files = self.list_s3_files(
                sources["post"]["storage_root"], sources["post"].get("prefix", "")
            )
        records_pre = self.build_records(
            pre_files, sources["pre"]["storage_type"], sources["pre"]["storage_root"]
        )
        records_post = self.build_records(
            post_files, sources["post"]["storage_type"], sources["post"]["storage_root"]
        )
        self.log(
            f"Discovered {len(records_pre)} pre records and {len(records_post)} post records"
        )
        return records_pre, records_post

    def pair_catalog(self, records_pre: pd.DataFrame, records_post: pd.DataFrame) -> pd.DataFrame:
        pre_lookup = records_pre.rename(
            columns={
                "key": "pre_key",
                "storage_type": "pre_storage_type",
                "storage_root": "pre_storage_root",
            }
        ).copy()
        post_lookup = records_post.rename(
            columns={
                "key": "post_key",
                "storage_type": "post_storage_type",
                "storage_root": "post_storage_root",
            }
        ).copy()

        common_pair_ids = sorted(set(pre_lookup["pair_id"]) & set(post_lookup["pair_id"]))
        only_pre_pair_ids = sorted(set(pre_lookup["pair_id"]) - set(post_lookup["pair_id"]))
        only_post_pair_ids = sorted(set(post_lookup["pair_id"]) - set(pre_lookup["pair_id"]))

        pre_lookup = pre_lookup[pre_lookup["pair_id"].isin(common_pair_ids)].copy()
        post_lookup = post_lookup[post_lookup["pair_id"].isin(common_pair_ids)].copy()
        paired_catalog = pre_lookup.merge(
            post_lookup[["pair_id", "post_key", "post_storage_type", "post_storage_root"]],
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )

        if self.cfg.granule_filters:
            paired_catalog = paired_catalog[
                paired_catalog["granule"].isin(self.cfg.granule_filters)
            ].copy()
        if self.cfg.max_granules is not None:
            keep = sorted(paired_catalog["granule"].unique())[: self.cfg.max_granules]
            paired_catalog = paired_catalog[paired_catalog["granule"].isin(keep)].copy()

        paired_catalog["band_order"] = paired_catalog["band"].apply(band_sort_key)
        paired_catalog = paired_catalog.sort_values(
            ["granule", "stage", "band_order", "band"]
        ).reset_index(drop=True)

        inventory_summary = (
            paired_catalog.groupby(["granule", "stage"]).size().rename("paired_bands").reset_index()
        )
        inventory_summary.to_csv(
            self.report_dir / f"inventory_summary_{self.run_stamp}.csv", index=False
        )
        paired_catalog.to_csv(
            self.report_dir / f"paired_catalog_{self.run_stamp}.csv", index=False
        )
        self.log(
            f"Common pairs retained: {len(paired_catalog)} "
            f"(pre-only={len(only_pre_pair_ids)}, post-only={len(only_post_pair_ids)})"
        )
        if paired_catalog.empty:
            raise RuntimeError("No symmetric pre/post pairs remain after filtering.")
        self.paired_catalog = paired_catalog
        return paired_catalog

    @staticmethod
    def companion_hdr_key(img_key: str) -> str:
        stem, _ = os.path.splitext(img_key)
        return stem + ".hdr"

    @staticmethod
    def s3_to_gdal_path(bucket: str, key: str) -> str:
        return f"/vsis3/{bucket}/{key.lstrip('/')}"

    @staticmethod
    def read_gdal_dataset(ds: Any, source_path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if ds is None:
            raise RuntimeError(f"GDAL could not open {source_path}")
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        meta = {
            "path": str(source_path),
            "nodata": band.GetNoDataValue(),
            "shape": array.shape,
            "transform": ds.GetGeoTransform(),
            "projection": ds.GetProjection(),
        }
        ds = None
        return array, meta

    @staticmethod
    def read_rasterio_dataset(ds: Any, source_path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if ds is None:
            raise RuntimeError(f"Rasterio could not open {source_path}")
        array = ds.read(1).astype(np.float32)
        meta = {
            "path": str(source_path),
            "nodata": ds.nodata,
            "shape": array.shape,
            "transform": tuple(ds.transform) if ds.transform is not None else None,
            "projection": ds.crs.to_string() if ds.crs is not None else None,
        }
        return array, meta

    @staticmethod
    def parse_envi_header_value(value: str) -> str:
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1].strip()
        return value

    def parse_envi_header(self, hdr_path: Path) -> Dict[str, str]:
        header: Dict[str, str] = {}
        pending_key: Optional[str] = None
        pending_value: List[str] = []

        for raw_line in hdr_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if pending_key is not None:
                pending_value.append(line)
                joined = " ".join(pending_value)
                if "}" in line:
                    header[pending_key] = self.parse_envi_header_value(joined)
                    pending_key = None
                    pending_value = []
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if value.startswith("{") and not value.endswith("}"):
                pending_key = key
                pending_value = [value]
                continue
            header[key] = self.parse_envi_header_value(value)

        return header

    def infer_raw_img_layout(self, img_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        file_size = img_path.stat().st_size
        if file_size % 2 != 0:
            raise RuntimeError(
                f"Cannot infer raw layout for {img_path}: byte size {file_size} is not divisible by 2."
            )

        n_values = file_size // 2
        side = int(round(math.sqrt(n_values)))
        if side * side != n_values:
            raise RuntimeError(
                f"Cannot infer raw layout for {img_path}: {n_values} uint16 values do not form a square raster. "
                "A matching .hdr sidecar is required."
            )

        data = np.fromfile(img_path, dtype=np.dtype(np.uint16).newbyteorder("<"))
        array = data.reshape((side, side)).astype(np.float32)
        meta = {
            "path": str(img_path),
            "nodata": 0.0,
            "shape": array.shape,
            "transform": None,
            "projection": None,
        }
        self.log(
            f"Inferred raw immediate raster layout for {img_path.name}: "
            f"{side}x{side}, uint16 little-endian, nodata=0"
        )
        return array, meta

    def read_envi_raster(self, img_path: Path, hdr_path: Optional[Path] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        hdr_path = hdr_path or img_path.with_suffix(".hdr")
        if not hdr_path.exists():
            return self.infer_raw_img_layout(img_path)

        header = self.parse_envi_header(hdr_path)
        try:
            samples = int(header["samples"])
            lines = int(header["lines"])
            bands = int(header.get("bands", "1"))
            data_type_code = int(header["data type"])
        except KeyError as exc:
            raise RuntimeError(f"Missing ENVI header field {exc.args[0]!r} in {hdr_path}") from exc

        dtype = ENVI_DTYPE_MAP.get(data_type_code)
        if dtype is None:
            raise RuntimeError(f"Unsupported ENVI data type {data_type_code} in {hdr_path}")

        byte_order = int(header.get("byte order", "0"))
        interleave = header.get("interleave", "bsq").strip().lower()
        header_offset = int(header.get("header offset", "0"))

        base_dtype = np.dtype(dtype)
        if base_dtype.itemsize > 1:
            endian = "<" if byte_order == 0 else ">"
            file_dtype = base_dtype.newbyteorder(endian)
        else:
            file_dtype = base_dtype

        data = np.fromfile(img_path, dtype=file_dtype, offset=header_offset)
        expected = lines * samples * bands
        if data.size != expected:
            raise RuntimeError(
                f"ENVI size mismatch for {img_path}: expected {expected} values, found {data.size}"
            )

        if interleave == "bsq":
            data = data.reshape((bands, lines, samples))
            array = data[0]
        elif interleave == "bil":
            data = data.reshape((lines, bands, samples))
            array = data[:, 0, :]
        elif interleave == "bip":
            data = data.reshape((lines, samples, bands))
            array = data[:, :, 0]
        else:
            raise RuntimeError(f"Unsupported ENVI interleave {interleave!r} in {hdr_path}")

        array = array.astype(np.float32)
        nodata = header.get("data ignore value")
        nodata_value = float(nodata) if nodata is not None else None
        meta = {
            "path": str(img_path),
            "nodata": nodata_value,
            "shape": array.shape,
            "transform": None,
            "projection": None,
        }
        return array, meta

    def open_with_rasterio(self, local_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rasterio is None:
            raise ImportError(
                "Neither GDAL nor rasterio is available. Install rasterio in the validation environment."
            )
        with rasterio.open(local_path) as ds:
            return self.read_rasterio_dataset(ds, local_path)

    def download_s3_object(self, bucket: str, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        self.s3.download_file(bucket, key, str(local_path))
        return local_path

    def try_open_direct_s3(self, bucket: str, key: str) -> Tuple[Any, str]:
        gdal_path = self.s3_to_gdal_path(bucket, key)
        gdal = try_get_gdal()
        if gdal is None:
            return None, gdal_path
        try:
            ds = gdal.Open(gdal_path)
        except RuntimeError:
            return None, gdal_path
        return ds, gdal_path

    def open_local_raster(self, root_path: str, key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        local_path = Path(root_path) / key
        hdr_path = local_path.with_suffix(".hdr")
        gdal = try_get_gdal()
        if gdal is not None:
            ds = gdal.Open(str(local_path))
            if ds is not None:
                return self.read_gdal_dataset(ds, local_path)
        if local_path.suffix.lower() == ".img":
            try:
                return self.read_envi_raster(local_path, hdr_path)
            except Exception:
                pass
        return self.open_with_rasterio(local_path)

    def open_s3_raster(
        self, bucket: str, key: str, stage: str, cache_tag: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        read_mode = str(self.cfg.s3_read_mode).lower()
        if read_mode not in {"auto", "direct", "cache"}:
            raise ValueError(f"Unsupported s3_read_mode={self.cfg.s3_read_mode!r}")

        cache_dir = self.cache_dir / sanitize_name(cache_tag)
        cache_dir.mkdir(parents=True, exist_ok=True)

        local_path = cache_dir / os.path.basename(key)
        hdr_path = None
        if stage == "immediate":
            hdr_key = self.companion_hdr_key(key)
            hdr_path = cache_dir / os.path.basename(hdr_key)

        cache_ready = local_path.exists() and local_path.stat().st_size > 0
        if hdr_path is not None:
            cache_ready = cache_ready and hdr_path.exists() and hdr_path.stat().st_size > 0
        if cache_ready:
            gdal = try_get_gdal()
            if gdal is not None:
                ds = gdal.Open(str(local_path))
                if ds is not None:
                    return self.read_gdal_dataset(ds, local_path)
            if stage == "immediate":
                try:
                    return self.read_envi_raster(local_path, hdr_path)
                except Exception:
                    pass
            return self.open_with_rasterio(local_path)

        if read_mode in {"auto", "direct"}:
            ds, gdal_path = self.try_open_direct_s3(bucket, key)
            if ds is not None:
                return self.read_gdal_dataset(ds, gdal_path)
            if read_mode == "direct":
                raise RuntimeError(f"GDAL could not open {gdal_path} directly from S3")

        local_path = self.download_s3_object(bucket, key, local_path)
        if stage == "immediate":
            self.download_s3_object(bucket, hdr_key, hdr_path)
        gdal = try_get_gdal()
        if gdal is not None:
            ds = gdal.Open(str(local_path))
            if ds is not None:
                return self.read_gdal_dataset(ds, local_path)
        if stage == "immediate":
            try:
                return self.read_envi_raster(local_path, hdr_path)
            except Exception:
                pass
        return self.open_with_rasterio(local_path)

    def open_raster_source(
        self, storage_type: str, storage_root: str, key: str, stage: str, cache_tag: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if storage_type == "local":
            return self.open_local_raster(storage_root, key)
        if storage_type == "s3":
            return self.open_s3_raster(storage_root, key, stage, cache_tag)
        raise ValueError(f"Unsupported storage_type={storage_type!r}")

    def to_reflectance(
        self, array: np.ndarray, stage: str, nodata: Optional[float] = None, zero_is_nodata: bool = False
    ) -> np.ndarray:
        scale, offset = self.get_reflectance_scaling(stage)
        refl = array.astype(np.float32) * scale + offset
        mask = ~np.isfinite(array)
        if nodata is not None:
            mask |= array == nodata
        if zero_is_nodata:
            mask |= array == 0
        if stage == "final":
            mask |= array == self.cfg.final_fill_value
        else:
            mask |= array == self.cfg.immediate_fill_value
        refl[mask] = np.nan
        return refl

    def format_threshold(self, value: float, stage: Optional[str] = None) -> str:
        if not np.isfinite(value):
            return "n/a"
        units = str(self.cfg.threshold_units).lower()
        if units == "dn":
            scale, _ = self.get_reflectance_scaling(stage or "final")
            if scale == 0:
                return "n/a"
            dn_value = float(value) / scale
            if np.isfinite(dn_value):
                if abs(dn_value - round(dn_value)) < 1e-6:
                    return f"{int(round(dn_value))} DN"
                return f"{dn_value:.3f} DN"
        return f"{value:.7f}"

    @staticmethod
    def decision_from_mean_diff(abs_mean_diff: float, threshold: float) -> Tuple[str, float]:
        if not np.isfinite(threshold):
            return "NO_THRESHOLD", np.nan
        if abs_mean_diff <= threshold:
            return "PASS", 1.0
        return "FAIL", 0.0

    def compute_pair_metrics(
        self, pre_refl: np.ndarray, post_refl: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        valid = np.isfinite(pre_refl) & np.isfinite(post_refl)
        n_total = int(pre_refl.size)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {
                "n_total_pixels": n_total,
                "n_valid_pixels": 0,
                "n_changed_pixels": 0,
                "pct_pixels_changed": np.nan,
                "mean_pre_refl": np.nan,
                "mean_post_refl": np.nan,
                "mean_diff_refl": np.nan,
                "abs_mean_diff_refl": np.nan,
                "mean_abs_diff_refl": np.nan,
                "rmse_refl": np.nan,
                "decision": "NO_VALID_PIXELS",
                "pass_score": np.nan,
            }

        diff = post_refl - pre_refl
        diff[~valid] = np.nan
        diff_vals = diff[valid]

        n_changed = int(np.count_nonzero(np.abs(diff_vals) > self.cfg.pixel_diff_eps))
        pct_changed = 100.0 * n_changed / n_valid
        mean_pre = float(np.nanmean(pre_refl))
        mean_post = float(np.nanmean(post_refl))
        mean_diff = float(np.nanmean(diff_vals))
        abs_mean_diff = float(abs(mean_diff))
        mean_abs_diff = float(np.nanmean(np.abs(diff_vals)))
        rmse = float(np.sqrt(np.nanmean(diff_vals ** 2)))
        decision, pass_score = self.decision_from_mean_diff(abs_mean_diff, threshold)
        return {
            "n_total_pixels": n_total,
            "n_valid_pixels": n_valid,
            "n_changed_pixels": n_changed,
            "pct_pixels_changed": pct_changed,
            "mean_pre_refl": mean_pre,
            "mean_post_refl": mean_post,
            "mean_diff_refl": mean_diff,
            "abs_mean_diff_refl": abs_mean_diff,
            "mean_abs_diff_refl": mean_abs_diff,
            "rmse_refl": rmse,
            "decision": decision,
            "pass_score": pass_score,
        }

    def resolve_pair_storage(self, row: Any) -> Any:
        required = [
            "pre_storage_type",
            "pre_storage_root",
            "pre_key",
            "post_storage_type",
            "post_storage_root",
            "post_key",
        ]
        if all(hasattr(row, attr) for attr in required):
            return row
        pair_id = getattr(row, "pair_id", None)
        if pair_id is None and isinstance(row, pd.Series):
            pair_id = row.get("pair_id")
        if pair_id is None:
            raise AttributeError("Row is missing pair_id and storage fields cannot be backfilled.")
        match = self.paired_catalog.loc[self.paired_catalog["pair_id"] == pair_id]
        if match.empty:
            raise KeyError(f"Could not backfill storage metadata for pair_id={pair_id!r}")
        return match.iloc[0]

    def load_pair_reflectance(
        self, row: Any
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
        resolved = self.resolve_pair_storage(row)
        pair_id = getattr(row, "pair_id", None) or row["pair_id"]
        stage = getattr(row, "stage", None) or row["stage"]
        pre_arr, pre_meta = self.open_raster_source(
            resolved.pre_storage_type,
            resolved.pre_storage_root,
            resolved.pre_key,
            resolved.stage,
            f"pre_{pair_id}",
        )
        post_arr, post_meta = self.open_raster_source(
            resolved.post_storage_type,
            resolved.post_storage_root,
            resolved.post_key,
            resolved.stage,
            f"post_{pair_id}",
        )
        if pre_arr.shape != post_arr.shape:
            raise ValueError(f"Shape mismatch for {pair_id}: pre={pre_arr.shape}, post={post_arr.shape}")
        pre_refl = self.to_reflectance(
            pre_arr, stage, pre_meta.get("nodata"), zero_is_nodata=self.cfg.pre_zero_is_nodata
        )
        post_refl = self.to_reflectance(
            post_arr, stage, post_meta.get("nodata"), zero_is_nodata=self.cfg.post_zero_is_nodata
        )
        return pre_refl, post_refl, pre_meta, post_meta

    def threshold_reference_table(self) -> pd.DataFrame:
        rows = []
        final_band_map = {
            "L30": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
            "S30": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        }
        for product, band_list in final_band_map.items():
            for band in band_list:
                if self.cfg.bands_to_include and band not in set(self.cfg.bands_to_include):
                    continue
                if self.cfg.bands_to_exclude and band in set(self.cfg.bands_to_exclude):
                    continue
                bundle = self.get_threshold_bundle(product, "final", band)
                rows.append(
                    {
                        "stage": "final",
                        "product": product,
                        "band": band,
                        "band_label": get_band_label(product, band),
                        "threshold_source": bundle["threshold_source"],
                        "reference_threshold_refl": bundle["reference_threshold_refl"],
                        "strict_threshold_refl": bundle["strict_threshold_refl"],
                        "relaxed_threshold_refl": bundle["relaxed_threshold_refl"],
                    }
                )
        immediate_band_map = {
            "L30": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
            "S30": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        }
        for product, band_list in immediate_band_map.items():
            for band in band_list:
                if self.cfg.bands_to_include and band not in set(self.cfg.bands_to_include):
                    continue
                if self.cfg.bands_to_exclude and band in set(self.cfg.bands_to_exclude):
                    continue
                bundle = self.get_threshold_bundle(product, "immediate", band)
                rows.append(
                    {
                        "stage": "immediate",
                        "product": product,
                        "band": band,
                        "band_label": get_band_label(product, band),
                        "threshold_source": bundle["threshold_source"],
                        "reference_threshold_refl": bundle["reference_threshold_refl"],
                        "strict_threshold_refl": bundle["strict_threshold_refl"],
                        "relaxed_threshold_refl": bundle["relaxed_threshold_refl"],
                    }
                )
        df = pd.DataFrame(rows)
        df["band_order"] = df["band"].apply(band_sort_key)
        df = df.sort_values(["stage", "product", "band_order", "band"]).drop(columns="band_order")
        df.to_csv(self.report_dir / f"threshold_reference_{self.run_stamp}.csv", index=False)
        return df

    def compare_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        comparison_rows = []
        load_errors = []
        total_pairs = len(self.paired_catalog)
        self.log(f"Starting pixel-level comparison for {total_pairs} raster pairs")
        for idx, row in enumerate(self.paired_catalog.itertuples(index=False), start=1):
            pair_start = time.perf_counter()
            try:
                pre_refl, post_refl, pre_meta, _ = self.load_pair_reflectance(row)
                metrics = self.compute_pair_metrics(pre_refl, post_refl, row.threshold_refl)
                comparison_rows.append(
                    {
                        "pair_id": row.pair_id,
                        "granule": row.granule,
                        "product": row.product,
                        "stage": row.stage,
                        "band": row.band,
                        "band_label": row.band_label,
                        "band_order": row.band_order,
                        "threshold_source": row.threshold_source,
                        "threshold_refl": row.threshold_refl,
                        "strict_threshold_refl": row.strict_threshold_refl,
                        "relaxed_threshold_refl": row.relaxed_threshold_refl,
                        "reference_threshold_refl": row.reference_threshold_refl,
                        "pre_storage_type": row.pre_storage_type,
                        "pre_storage_root": row.pre_storage_root,
                        "post_storage_type": row.post_storage_type,
                        "post_storage_root": row.post_storage_root,
                        "pre_key": row.pre_key,
                        "post_key": row.post_key,
                        "pre_url": self.format_storage_location(
                            row.pre_storage_type, row.pre_storage_root, row.pre_key
                        ),
                        "post_url": self.format_storage_location(
                            row.post_storage_type, row.post_storage_root, row.post_key
                        ),
                        "shape": pre_meta["shape"],
                        **metrics,
                    }
                )
                self.log(
                    f"[{idx:04d}/{total_pairs:04d}] {row.granule} | {row.stage:9s} | {row.band:3s} "
                    f"| abs(mean diff)={metrics['abs_mean_diff_refl']:.6f} "
                    f"| threshold={self.format_threshold(row.threshold_refl, row.stage)} "
                    f"| {metrics['decision']} | {time.perf_counter() - pair_start:.1f}s"
                )
            except Exception as exc:
                load_errors.append(
                    {
                        "pair_id": row.pair_id,
                        "granule": row.granule,
                        "stage": row.stage,
                        "band": row.band,
                        "error": str(exc),
                    }
                )
                self.log(
                    f"FAILED {row.granule} | {row.stage} | {row.band} -> {exc} "
                    f"| {time.perf_counter() - pair_start:.1f}s"
                )

        df_results = pd.DataFrame(comparison_rows)
        df_errors = pd.DataFrame(load_errors)
        if df_results.empty:
            raise RuntimeError("No comparisons were completed successfully.")
        df_results = df_results.sort_values(
            ["granule", "stage", "band_order", "band"]
        ).reset_index(drop=True)
        results_csv = self.report_dir / f"lasrc_validation_metrics_{self.run_stamp}.csv"
        errors_csv = self.report_dir / f"lasrc_validation_load_errors_{self.run_stamp}.csv"
        df_results.to_csv(results_csv, index=False)
        if not df_errors.empty:
            df_errors.to_csv(errors_csv, index=False)
        summary_by_band = (
            df_results.groupby(["stage", "band", "band_label"], dropna=False)
            .agg(
                compared_pairs=("pair_id", "size"),
                granules=("granule", "nunique"),
                threshold_source=("threshold_source", "first"),
                threshold_refl=("threshold_refl", "first"),
                strict_threshold_refl=("strict_threshold_refl", "first"),
                relaxed_threshold_refl=("relaxed_threshold_refl", "first"),
                reference_threshold_refl=("reference_threshold_refl", "first"),
                mean_abs_mean_diff_refl=("abs_mean_diff_refl", "mean"),
                max_abs_mean_diff_refl=("abs_mean_diff_refl", "max"),
                mean_mean_abs_diff_refl=("mean_abs_diff_refl", "mean"),
                mean_pct_pixels_changed=("pct_pixels_changed", "mean"),
                pass_rate_pct=(
                    "pass_score",
                    lambda s: float(np.nanmean(s) * 100.0) if np.isfinite(s).any() else np.nan,
                ),
            )
            .reset_index()
        )
        summary_by_band["band_order"] = summary_by_band["band"].apply(band_sort_key)
        summary_by_band = summary_by_band.sort_values(["stage", "band_order", "band"]).drop(
            columns="band_order"
        )
        summary_by_band.to_csv(
            self.report_dir / f"summary_by_band_{self.run_stamp}.csv", index=False
        )
        self.df_results = df_results
        self.log(f"Comparison complete: {len(df_results)} paired rasters")
        return df_results, df_errors, summary_by_band

    @staticmethod
    def style_image_axis(ax: Any) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

    def get_plot_band_order(self, products: Sequence[str]) -> List[str]:
        if self.cfg.panel_bands:
            return [band for band in self.cfg.panel_bands if band not in set(self.cfg.bands_to_exclude)]

        ordered: List[str] = []
        seen = set()
        for product in [p for p in products if p]:
            for band in DEFAULT_PLOT_BANDS_BY_PRODUCT.get(product, []):
                if band in set(self.cfg.bands_to_exclude):
                    continue
                if band not in seen:
                    ordered.append(band)
                    seen.add(band)
        return ordered

    def plot_granule_panel(
        self, granule_id: str, stage: Optional[str] = "final"
    ) -> Optional[Path]:
        subset = self.df_results[self.df_results["granule"] == granule_id].copy()
        if stage is not None:
            subset = subset[subset["stage"] == stage].copy()
        plot_band_order = self.get_plot_band_order(subset["product"].dropna().unique().tolist())
        if plot_band_order:
            subset = subset[subset["band"].isin(plot_band_order)].copy()
        subset = subset.sort_values(["band_order", "band"])
        if subset.empty:
            raise RuntimeError(f"No results found for granule={granule_id!r}, stage={stage!r}")

        nrows = len(subset)
        fig, axes = plt.subplots(
            nrows,
            5,
            figsize=(29, max(4.5 * nrows, 6)),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.08, 1.08]},
        )
        fig.suptitle(f"{granule_id} | {stage_display_name(stage)}", fontsize=16, y=1.01)

        for idx, (_, rec) in enumerate(subset.iterrows()):
            panel_text_color = "firebrick"
            pre_refl, post_refl, _, _ = self.load_pair_reflectance(rec)
            diff = post_refl - pre_refl
            valid = np.isfinite(pre_refl) & np.isfinite(post_refl)
            diff[~valid] = np.nan

            pre_vals = pre_refl[np.isfinite(pre_refl)]
            post_vals = post_refl[np.isfinite(post_refl)]
            pre_common = pre_refl[valid]
            post_common = post_refl[valid]
            diff_vals = diff[np.isfinite(diff)]

            if pre_vals.size and post_vals.size:
                combined = np.concatenate([pre_vals, post_vals])
                vmin, vmax = np.percentile(combined, [2, 98])
                if vmax <= vmin:
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0

            if diff_vals.size:
                diff_lim = max(float(np.percentile(np.abs(diff_vals), 98)), 1e-6)
            else:
                diff_lim = 1e-6

            ax_pre, ax_post, ax_diff, ax_hist, ax_scatter = axes[idx]

            im_pre = ax_pre.imshow(np.ma.masked_invalid(pre_refl), cmap="gray", vmin=vmin, vmax=vmax)
            ax_pre.set_title(f"{rec.band} {rec.band_label}\n{self.cfg.pre_label}", fontweight="bold")
            self.style_image_axis(ax_pre)
            cbar_pre = fig.colorbar(im_pre, ax=ax_pre, fraction=0.046, pad=0.03)
            cbar_pre.set_label("Reflectance")
            cbar_pre.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            if pre_vals.size:
                pre_stats = (
                    f"mean: {np.nanmean(pre_vals):.4f}\n"
                    f"std: {np.nanstd(pre_vals):.4f}\n"
                    f"min: {np.nanmin(pre_vals):.4f}\n"
                    f"max: {np.nanmax(pre_vals):.4f}"
                )
                ax_pre.text(
                    0.03,
                    0.03,
                    pre_stats,
                    transform=ax_pre.transAxes,
                    fontsize=9,
                    color=panel_text_color,
                    va="bottom",
                    ha="left",
                    bbox=dict(
                        boxstyle="square,pad=0.20",
                        facecolor="white",
                        alpha=0.78,
                        edgecolor="black",
                        linewidth=0.6,
                    ),
                )

            im_post = ax_post.imshow(np.ma.masked_invalid(post_refl), cmap="gray", vmin=vmin, vmax=vmax)
            ax_post.set_title(f"{rec.band} {rec.band_label}\n{self.cfg.post_label}", fontweight="bold")
            self.style_image_axis(ax_post)
            cbar_post = fig.colorbar(im_post, ax=ax_post, fraction=0.046, pad=0.03)
            cbar_post.set_label("Reflectance")
            cbar_post.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            if post_vals.size:
                post_stats = (
                    f"mean: {np.nanmean(post_vals):.4f}\n"
                    f"std: {np.nanstd(post_vals):.4f}\n"
                    f"min: {np.nanmin(post_vals):.4f}\n"
                    f"max: {np.nanmax(post_vals):.4f}"
                )
                ax_post.text(
                    0.03,
                    0.03,
                    post_stats,
                    transform=ax_post.transAxes,
                    fontsize=9,
                    color=panel_text_color,
                    va="bottom",
                    ha="left",
                    bbox=dict(
                        boxstyle="square,pad=0.20",
                        facecolor="white",
                        alpha=0.78,
                        edgecolor="black",
                        linewidth=0.6,
                    ),
                )

            norm = mcolors.TwoSlopeNorm(vmin=-diff_lim, vcenter=0.0, vmax=diff_lim)
            im = ax_diff.imshow(np.ma.masked_invalid(diff), cmap="RdBu_r", norm=norm)
            ax_diff.set_title(
                f"{rec.band} difference\n({self.cfg.post_label} - {self.cfg.pre_label})",
                fontweight="bold",
            )
            self.style_image_axis(ax_diff)
            stats_text = (
                f"Mean diff: {rec.mean_diff_refl:.4f}\n"
                f"|Mean diff|: {rec.abs_mean_diff_refl:.4f}\n"
                f"% pixels with diff: {rec.pct_pixels_changed:.2f}%\n"
                f"Threshold: {self.threshold_display_with_reflectance(rec.threshold_refl, stage)}\n"
                f"Decision: {rec.decision}"
            )
            ax_diff.text(
                0.03,
                0.03,
                stats_text,
                transform=ax_diff.transAxes,
                fontsize=10,
                color=panel_text_color,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="square,pad=0.22",
                    facecolor="white",
                    alpha=0.78,
                    edgecolor="black",
                    linewidth=0.6,
                ),
            )
            cbar_diff = fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.02)
            cbar_diff.set_label("Reflectance diff")
            cbar_diff.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.6f"))

            hist_bins = (
                np.linspace(vmin, vmax, 50)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin
                else 18
            )
            if post_vals.size:
                ax_hist.hist(
                    post_vals,
                    bins=hist_bins,
                    histtype="step",
                    linewidth=1.8,
                    color="darkorange",
                    label=self.cfg.post_label,
                    alpha=0.98,
                    density=True,
                )
            if pre_vals.size:
                ax_hist.hist(
                    pre_vals,
                    bins=hist_bins,
                    histtype="step",
                    linewidth=1.8,
                    linestyle="--",
                    color="steelblue",
                    label=self.cfg.pre_label,
                    alpha=0.98,
                    density=True,
                )
            ax_hist.set_title(f"{rec.band} reflectance histogram", fontweight="bold")
            ax_hist.set_xlabel("Surface reflectance")
            ax_hist.set_ylabel("Pixel count")
            ax_hist.set_xlim(vmin, vmax)
            ax_hist.grid(False)
            hist_legend = ax_hist.legend(
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.78,
                fancybox=False,
            )
            if hist_legend is not None:
                for text in hist_legend.get_texts():
                    text.set_color(panel_text_color)

            if pre_common.size and post_common.size:
                scatter_n = min(10000, pre_common.size)
                rng = np.random.default_rng(0)
                scatter_idx = rng.choice(pre_common.size, size=scatter_n, replace=False)
                x = pre_common[scatter_idx]
                y = post_common[scatter_idx]
                xy_all = np.concatenate([pre_common, post_common])
                smin = float(np.nanpercentile(xy_all, 2))
                smax = float(np.nanpercentile(xy_all, 98))
                if smax <= smin:
                    smax = smin + 1e-6
                ax_scatter.scatter(
                    x,
                    y,
                    s=18,
                    alpha=0.72,
                    color="slateblue",
                    marker="o",
                    edgecolors="black",
                    linewidth=0.3,
                )
                ax_scatter.plot([smin, smax], [smin, smax], linestyle="--", color="black", linewidth=1)
                ax_scatter.set_xlim(smin, smax)
                ax_scatter.set_ylim(smin, smax)
                ax_scatter.set_aspect("equal", adjustable="box")
                corr = float(np.corrcoef(pre_common, post_common)[0, 1]) if pre_common.size >= 2 else np.nan
                p_value = np.nan
                if scatter_n >= 3 and pearsonr is not None:
                    try:
                        corr_stat = pearsonr(x, y)
                        corr = float(corr_stat.statistic)
                        p_value = float(corr_stat.pvalue)
                    except Exception:
                        p_value = np.nan
                sig = significance_label(p_value)
                ax_scatter.text(
                    0.03,
                    0.97,
                    f"n={pre_common.size:,}\nr={corr:.4f} ({sig})",
                    transform=ax_scatter.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    color=panel_text_color,
                    bbox=dict(
                        boxstyle="square,pad=0.22",
                        facecolor="white",
                        alpha=0.78,
                        edgecolor="black",
                        linewidth=0.6,
                    ),
                )
            ax_scatter.set_title(f"{rec.band} pre vs post scatter", fontweight="bold")
            ax_scatter.set_xlabel(self.cfg.pre_label)
            ax_scatter.set_ylabel(self.cfg.post_label)
            ax_scatter.grid(False)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.44, hspace=0.28)
        panel_path = self.panel_dir / f"{sanitize_name(granule_id)}_{stage}_panel.png"
        fig.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log(f"Saved panel image: {panel_path}")
        return panel_path

    def render_panels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.df_results.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.cfg.render_all_panels:
            granule_ids = sorted(self.df_results["granule"].dropna().unique())
        elif self.cfg.panel_granules:
            valid = set(self.df_results["granule"].dropna().unique())
            granule_ids = [g for g in self.cfg.panel_granules if g in valid]
        elif self.cfg.render_single_panel:
            valid = sorted(self.df_results["granule"].dropna().unique())
            granule = self.cfg.panel_granule or valid[0]
            granule_ids = [granule]
        else:
            granule_ids = []

        saved_rows: List[Dict[str, Any]] = []
        failed_rows: List[Dict[str, Any]] = []
        for idx, granule_id in enumerate(granule_ids, start=1):
            available_stages = sorted(
                self.df_results[self.df_results["granule"] == granule_id]["stage"].dropna().unique()
            )
            if not available_stages:
                failed_rows.append({"granule": granule_id, "stage": None, "error": "No stages available"})
                continue
            if self.cfg.panel_stage == "auto":
                stages_to_render = available_stages
            else:
                stages_to_render = [self.cfg.panel_stage] if self.cfg.panel_stage in available_stages else []
            if not stages_to_render:
                failed_rows.append(
                    {"granule": granule_id, "stage": self.cfg.panel_stage, "error": "Requested stage unavailable"}
                )
                continue
            for stage_name in stages_to_render:
                self.log(
                    f"[{idx:03d}/{len(granule_ids):03d}] Plotting {granule_id} | {stage_display_name(stage_name)}"
                )
                try:
                    panel_path = self.plot_granule_panel(granule_id=granule_id, stage=stage_name)
                    saved_rows.append(
                        {"granule": granule_id, "stage": stage_name, "panel_path": str(panel_path)}
                    )
                except Exception as exc:
                    failed_rows.append({"granule": granule_id, "stage": stage_name, "error": str(exc)})
                    self.log(f"Failed to plot {granule_id} | {stage_name}: {exc}")

        df_saved = pd.DataFrame(saved_rows)
        df_failed = pd.DataFrame(failed_rows)
        if not df_saved.empty:
            df_saved.to_csv(self.report_dir / f"panel_manifest_{self.run_stamp}.csv", index=False)
        if not df_failed.empty:
            df_failed.to_csv(self.report_dir / f"panel_failures_{self.run_stamp}.csv", index=False)
        return df_saved, df_failed

    def render_summary_plots(self, summary_by_band: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "threshold_source",
            "threshold_refl",
            "strict_threshold_refl",
            "relaxed_threshold_refl",
            "reference_threshold_refl",
            "pre_storage_type",
            "pre_storage_root",
            "post_storage_type",
            "post_storage_root",
            "pre_key",
            "post_key",
        ]
        missing = [c for c in required_cols if c not in self.df_results.columns]
        if missing:
            lookup = self.paired_catalog[["pair_id"] + [c for c in required_cols if c in self.paired_catalog.columns]]
            lookup = lookup.drop_duplicates(subset=["pair_id"])
            self.df_results = self.df_results.merge(lookup, on="pair_id", how="left", suffixes=("", "_ref"))
            for col in required_cols:
                ref_col = f"{col}_ref"
                if col not in self.df_results.columns and ref_col in self.df_results.columns:
                    self.df_results[col] = self.df_results[ref_col]
                elif ref_col in self.df_results.columns:
                    self.df_results[col] = self.df_results[col].where(
                        self.df_results[col].notna(), self.df_results[ref_col]
                    )
            self.df_results = self.df_results.drop(
                columns=[c for c in self.df_results.columns if c.endswith("_ref")]
            )

        overall_summary = (
            self.df_results.groupby(["stage", "band", "band_label"], dropna=False)
            .agg(
                granules=("granule", "nunique"),
                compared_pairs=("pair_id", "size"),
                threshold_source=("threshold_source", "first"),
                threshold_refl=("threshold_refl", "first"),
                strict_threshold_refl=("strict_threshold_refl", "first"),
                relaxed_threshold_refl=("relaxed_threshold_refl", "first"),
                reference_threshold_refl=("reference_threshold_refl", "first"),
                mean_pre_refl=("mean_pre_refl", "mean"),
                mean_post_refl=("mean_post_refl", "mean"),
                mean_mean_diff_refl=("mean_diff_refl", "mean"),
                mean_abs_mean_diff_refl=("abs_mean_diff_refl", "mean"),
                max_abs_mean_diff_refl=("abs_mean_diff_refl", "max"),
                pass_rate_pct=(
                    "pass_score",
                    lambda s: float(np.nanmean(s) * 100.0) if np.isfinite(s).any() else np.nan,
                ),
            )
            .reset_index()
        )
        overall_summary["band_order"] = overall_summary["band"].apply(band_sort_key)
        overall_summary = overall_summary.sort_values(["stage", "band_order", "band"]).drop(
            columns="band_order"
        )
        overall_summary.to_csv(
            self.report_dir / f"overall_summary_{self.run_stamp}.csv", index=False
        )

        if not self.cfg.save_summary_plots:
            return overall_summary

        stages_to_plot = sorted(self.df_results["stage"].dropna().unique())
        threshold_note = self.threshold_plot_note()
        for stage_name in stages_to_plot:
            stage_df = self.df_results[self.df_results["stage"] == stage_name].copy()
            plot_band_order = self.get_plot_band_order(stage_df["product"].dropna().unique().tolist())
            if plot_band_order:
                bands = [band for band in plot_band_order if band in set(stage_df["band"].dropna())]
            else:
                bands = sorted(stage_df["band"].dropna().unique(), key=band_sort_key)
            if not bands:
                continue
            ncols = 3
            nrows = math.ceil(len(bands) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            fig.suptitle(
                f"Mean surface reflectance: {self.cfg.pre_label} vs {self.cfg.post_label} | stage={stage_name}",
                fontsize=16,
                y=1.02,
            )
            if threshold_note:
                fig.text(
                    0.5,
                    0.985,
                    f"Threshold note: {threshold_note}",
                    ha="center",
                    va="top",
                    fontsize=11,
                    color="dimgray",
                )
            for ax in axes.ravel():
                ax.axis("off")
            for ax, band in zip(axes.ravel(), bands):
                ax.axis("on")
                sub = stage_df[stage_df["band"] == band].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue
                x = sub["mean_pre_refl"].to_numpy(dtype=np.float64)
                y = sub["mean_post_refl"].to_numpy(dtype=np.float64)
                colors = np.where(sub["decision"].to_numpy() == "PASS", "forestgreen", "firebrick")
                ax.scatter(x, y, c=colors, s=90, alpha=0.88, edgecolor="black", linewidth=0.45)
                finite = np.isfinite(x) & np.isfinite(y)
                corr = np.nan
                if finite.any():
                    xy = np.concatenate([x[finite], y[finite]])
                    lo = float(np.nanmin(xy))
                    hi = float(np.nanmax(xy))
                    if hi <= lo:
                        hi = lo + 1e-6
                    pad = (hi - lo) * 0.05
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1)
                    ax.set_xlim(lo - pad, hi + pad)
                    ax.set_ylim(lo - pad, hi + pad)
                    if finite.sum() >= 2:
                        corr = float(np.corrcoef(x[finite], y[finite])[0, 1])
                pass_rate = (
                    float(np.nanmean(sub["pass_score"]) * 100.0)
                    if np.isfinite(sub["pass_score"]).any()
                    else np.nan
                )
                title = f"{band} {sub['band_label'].iloc[0]}\n" f"n={len(sub)} | pass={pass_rate:.1f}%"
                if np.isfinite(corr):
                    title += f" | r={corr:.6f}"
                ax.set_title(title)
                ax.set_xlabel(f"{self.cfg.pre_label} mean reflectance")
                ax.set_ylabel(f"{self.cfg.post_label} mean reflectance")
                ax.grid(False)
            plt.tight_layout()
            out = self.summary_plot_dir / f"scatter_summary_{stage_name}_{self.run_stamp}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        for stage_name in stages_to_plot:
            stage_df = self.df_results[self.df_results["stage"] == stage_name].copy()
            stage_summary = overall_summary[overall_summary["stage"] == stage_name].copy()
            if stage_df.empty or stage_summary.empty:
                continue
            plot_band_order = self.get_plot_band_order(stage_df["product"].dropna().unique().tolist())
            if plot_band_order:
                stage_df = stage_df[stage_df["band"].isin(plot_band_order)].copy()
                stage_summary = stage_summary[stage_summary["band"].isin(plot_band_order)].copy()
            stage_summary["band_order"] = stage_summary["band"].apply(band_sort_key)
            stage_summary = stage_summary.sort_values(["band_order", "band"])
            bands = stage_summary["band"].tolist()
            box_data = [
                stage_df.loc[stage_df["band"] == band, "abs_mean_diff_refl"]
                .dropna()
                .to_numpy(dtype=float)
                for band in bands
            ]
            fig, ax = plt.subplots(figsize=(max(10, 1.35 * len(bands)), 6))
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.55, showfliers=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue")
                patch.set_alpha(0.8)
            for median in bp["medians"]:
                median.set_color("navy")
                median.set_linewidth(1.5)
            x = np.arange(1, len(bands) + 1)
            threshold_vals = stage_summary["threshold_refl"].to_numpy(dtype=float)
            finite_box_vals = (
                np.concatenate([vals for vals in box_data if vals.size])
                if any(vals.size for vals in box_data)
                else np.array([], dtype=float)
            )
            y_span = float(np.nanmax(finite_box_vals) - np.nanmin(finite_box_vals)) if finite_box_vals.size else 0.0
            text_offset = max(y_span * 0.035, 0.00025)
            for xi, threshold_val in zip(x, threshold_vals):
                if np.isfinite(threshold_val):
                    ax.hlines(
                        threshold_val,
                        xi - 0.28,
                        xi + 0.28,
                        colors="firebrick",
                        linestyles="--",
                        linewidth=2.4,
                    )
                    ax.text(
                        xi,
                        threshold_val + text_offset,
                        self.distribution_threshold_label(threshold_val, stage_name),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="firebrick",
                        fontweight="bold",
                    )
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color="firebrick",
                    linestyle="--",
                    linewidth=2.4,
                    label=f"Threshold ({threshold_note})" if threshold_note else "Threshold reference",
                )
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{band}\n{label}" for band, label in zip(stage_summary["band"], stage_summary["band_label"])]
            )
            ax.set_ylabel("|Mean surface reflectance difference| across granules")
            ax.set_title(
                f"Distribution of per-granule mean differences vs thresholds | {stage_display_name(stage_name)}",
                fontweight="bold",
            )
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            ax.grid(axis="y", alpha=0.25)
            ax.legend(handles=legend_handles, loc="upper left", fontsize=13)
            plt.tight_layout()
            out = self.summary_plot_dir / f"distribution_summary_{stage_name}_{self.run_stamp}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return overall_summary

    def list_log_objects(self, log_source: str) -> pd.DataFrame:
        path = Path(log_source).expanduser()
        if path.exists():
            rows = []
            for item in sorted(path.glob("*.log")):
                rows.append(
                    {
                        "storage_type": "local",
                        "storage_root": str(path),
                        "key": item.name,
                        "filename": item.name,
                        "last_modified": datetime.fromtimestamp(item.stat().st_mtime),
                        "size_bytes": item.stat().st_size,
                    }
                )
            return pd.DataFrame(rows)
        bucket, prefix = self.split_bucket_prefix(log_source)
        paginator = self.s3.get_paginator("list_objects_v2")
        rows = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not key.lower().endswith(".log"):
                    continue
                rows.append(
                    {
                        "storage_type": "s3",
                        "storage_root": bucket,
                        "key": key,
                        "filename": os.path.basename(key),
                        "last_modified": obj["LastModified"],
                        "size_bytes": obj["Size"],
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def classify_runtime_mode(*parts: str) -> Optional[str]:
        upper = " ".join(str(part) for part in parts if part).upper()
        if "VIIRS" in upper:
            return "VIIRS"
        if "GFS" in upper:
            return "GFS"
        return None

    def read_text_source(self, storage_type: str, storage_root: str, key: str) -> str:
        if storage_type == "local":
            return (Path(storage_root) / key).read_text(errors="replace")
        response = self.s3.get_object(Bucket=storage_root, Key=key)
        return response["Body"].read().decode("utf-8", errors="replace")

    @staticmethod
    def extract_s2_granule_from_key(key: Optional[str]) -> Optional[str]:
        if not isinstance(key, str):
            return None
        match = re.search(r"/(S2[ABC]_MSIL1C_[^/]+?)(?:\.SAFE)?/", key)
        return match.group(1) if match else None

    def build_s2_to_hls_lookup(self) -> pd.DataFrame:
        rows = []
        for rec in self.paired_catalog.itertuples(index=False):
            for key in [getattr(rec, "pre_key", None), getattr(rec, "post_key", None)]:
                s2_granule_id = self.extract_s2_granule_from_key(key)
                if s2_granule_id is None:
                    continue
                rows.append({"s2_granule_id": s2_granule_id, "hls_granule": rec.granule})
        if not rows:
            return pd.DataFrame(columns=["s2_granule_id", "hls_granule"])
        return pd.DataFrame(rows).drop_duplicates(subset=["s2_granule_id"])

    @staticmethod
    def parse_runtime_log_text(
        text: str, mode: str, storage_root: str, key: str, filename: str, last_modified: Any
    ) -> List[Dict[str, Any]]:
        start_re = re.compile(r"(?:Running|\| START \|)\s+(S2[ABC]_MSIL1C_[A-Z0-9_]+)")
        granule_token_re = re.compile(r"granule=(S2[ABC]_MSIL1C_[A-Z0-9_]+)")
        failed_re = re.compile(r"Granule failed:\s*(S2[ABC]_MSIL1C_[A-Z0-9_]+)")
        metric_patterns = {
            "sentinel_runtime_seconds": re.compile(r"sentinel_runtime_seconds=([0-9]+(?:\.[0-9]+)?)"),
            "granule_runtime_seconds": re.compile(r"granule_runtime_seconds=([0-9]+(?:\.[0-9]+)?)"),
            "granule_cpu_percent": re.compile(r"granule_cpu_percent=([0-9]+(?:\.[0-9]+)?)"),
            "granule_max_rss_mb": re.compile(r"granule_max_rss_mb=([0-9]+(?:\.[0-9]+)?)"),
            "granule_max_rss_kb": re.compile(r"granule_max_rss_kb=([0-9]+(?:\.[0-9]+)?)"),
            "granule_user_time_seconds": re.compile(r"granule_user_time_seconds=([0-9]+(?:\.[0-9]+)?)"),
            "granule_system_time_seconds": re.compile(r"granule_system_time_seconds=([0-9]+(?:\.[0-9]+)?)"),
            "granule_output_total_gb": re.compile(r"granule_output_total_gb=([0-9]+(?:\.[0-9]+)?)"),
            "granule_status": re.compile(r"granule_status=([0-9]+)"),
        }
        current_granule: Optional[str] = None
        metrics_by_granule: Dict[str, Dict[str, Any]] = {}

        for line_number, line in enumerate(text.splitlines(), start=1):
            start_match = start_re.search(line)
            if start_match:
                current_granule = start_match.group(1)

            granule_match = granule_token_re.search(line)
            granule_id = granule_match.group(1) if granule_match else current_granule
            if granule_id is None:
                continue

            rec = metrics_by_granule.setdefault(
                granule_id,
                {
                    "mode": mode,
                    "storage_root": storage_root,
                    "log_key": key,
                    "filename": filename,
                    "last_modified": last_modified,
                    "line_number": line_number,
                    "s2_granule_id": granule_id,
                    "run_status": "unknown",
                },
            )
            rec["line_number"] = line_number

            updated = False
            for metric_name, pattern in metric_patterns.items():
                metric_match = pattern.search(line)
                if metric_match:
                    value = metric_match.group(1)
                    if metric_name == "granule_status":
                        rec[metric_name] = int(value)
                        rec["run_status"] = "ok" if int(value) == 0 else f"failed_{value}"
                    else:
                        rec[metric_name] = float(value)
                    updated = True

            failed_match = failed_re.search(line)
            if failed_match:
                failed_granule = failed_match.group(1)
                fail_rec = metrics_by_granule.setdefault(
                    failed_granule,
                    {
                        "mode": mode,
                        "storage_root": storage_root,
                        "log_key": key,
                        "filename": filename,
                        "last_modified": last_modified,
                        "line_number": line_number,
                        "s2_granule_id": failed_granule,
                    },
                )
                fail_rec["run_status"] = "failed"
                updated = True

            if updated and current_granule is None:
                current_granule = granule_id

        return list(metrics_by_granule.values())

    def compare_runtime_logs(self) -> Optional[pd.DataFrame]:
        if not self.cfg.runtime_log_source:
            return None
        runtime_log_objects = self.list_log_objects(self.cfg.runtime_log_source)
        if runtime_log_objects.empty:
            raise RuntimeError(f"No runtime logs found under {self.cfg.runtime_log_source}")
        runtime_log_objects["mode"] = runtime_log_objects.apply(
            lambda row: self.classify_runtime_mode(row["filename"], row["key"], row["storage_root"]), axis=1
        )
        runtime_log_objects = runtime_log_objects[
            runtime_log_objects["mode"].isin(["VIIRS", "GFS"])
        ].copy()
        runtime_log_objects = runtime_log_objects.sort_values(
            ["mode", "last_modified", "filename"]
        ).reset_index(drop=True)
        runtime_rows = []
        for obj in runtime_log_objects.itertuples(index=False):
            log_text = self.read_text_source(obj.storage_type, obj.storage_root, obj.key)
            runtime_rows.extend(
                self.parse_runtime_log_text(
                    log_text, obj.mode, obj.storage_root, obj.key, obj.filename, obj.last_modified
                )
            )
        df_runtime = pd.DataFrame(runtime_rows)
        if df_runtime.empty:
            self.log(
                "No sentinel_runtime_seconds entries were parsed from the runtime logs; "
                "skipping runtime comparison."
            )
            return None
        df_runtime = df_runtime.sort_values(
            ["mode", "s2_granule_id", "last_modified", "line_number"]
        ).reset_index(drop=True)
        df_runtime_latest = df_runtime.drop_duplicates(subset=["mode", "s2_granule_id"], keep="last").copy()
        s2_to_hls_lookup = self.build_s2_to_hls_lookup()
        df_runtime_latest = df_runtime_latest.merge(s2_to_hls_lookup, on="s2_granule_id", how="left")
        metrics_to_compare = {
            "runtime_seconds": ["granule_runtime_seconds", "sentinel_runtime_seconds"],
            "cpu_percent": ["granule_cpu_percent"],
            "peak_memory_mb": ["granule_max_rss_mb"],
        }
        merged = df_runtime_latest[["s2_granule_id", "hls_granule"]].drop_duplicates().copy()
        for output_metric, candidate_cols in metrics_to_compare.items():
            chosen_col = next((col for col in candidate_cols if col in df_runtime_latest.columns), None)
            if chosen_col is None:
                continue
            temp = (
                df_runtime_latest.pivot_table(
                    index=["s2_granule_id", "hls_granule"],
                    columns="mode",
                    values=chosen_col,
                    aggfunc="first",
                )
                .reset_index()
            )
            temp.columns.name = None
            rename_map = {"VIIRS": f"{output_metric}_viirs", "GFS": f"{output_metric}_gfs"}
            temp = temp.rename(columns=rename_map)
            merged = merged.merge(temp, on=["s2_granule_id", "hls_granule"], how="outer")

        status_wide = (
            df_runtime_latest.pivot_table(
                index=["s2_granule_id", "hls_granule"],
                columns="mode",
                values="run_status",
                aggfunc="first",
            )
            .reset_index()
        )
        status_wide.columns.name = None
        status_wide = status_wide.rename(columns={"VIIRS": "run_status_viirs", "GFS": "run_status_gfs"})
        df_runtime_pairs = merged.merge(status_wide, on=["s2_granule_id", "hls_granule"], how="outer")
        df_runtime_pairs.to_csv(self.report_dir / f"runtime_pairs_{self.run_stamp}.csv", index=False)

        plot_specs = [
            ("runtime_seconds", "Total Runtime (seconds)"),
            ("cpu_percent", "Average CPU (%)"),
            ("peak_memory_mb", "Peak Memory (MB)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        summary_lines = [f"Paired granules: {len(df_runtime_pairs)}"]

        for ax, (metric_key, title) in zip(axes[:3], plot_specs):
            x_col = f"{metric_key}_viirs"
            y_col = f"{metric_key}_gfs"
            if x_col not in df_runtime_pairs.columns or y_col not in df_runtime_pairs.columns:
                ax.set_visible(False)
                summary_lines.append(f"{title}: unavailable")
                continue
            plot_df = df_runtime_pairs.dropna(subset=[x_col, y_col]).copy()
            x = plot_df[x_col].to_numpy(dtype=float)
            y = plot_df[y_col].to_numpy(dtype=float)
            if x.size == 0:
                ax.set_visible(False)
                summary_lines.append(f"{title}: no paired values")
                continue
            ax.scatter(x, y, s=90, alpha=0.84, color="teal", edgecolor="black", linewidth=0.45)
            lo = float(np.nanmin(np.concatenate([x, y])))
            hi = float(np.nanmax(np.concatenate([x, y])))
            if hi <= lo:
                hi = lo + 1.0
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(f"VIIRS {title}")
            ax.set_ylabel(f"GFS {title}")
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")
            corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else np.nan
            delta = float(np.nanmean(y - x)) if x.size else np.nan
            ratio = float(np.nanmean(y / x)) if np.all(x != 0) else np.nan
            summary_lines.append(
                f"{title}: n={x.size}, mean(GFS-VIIRS)={delta:.2f}, mean ratio={ratio:.3f}, r={corr:.4f}"
            )
            ax.text(
                0.03,
                0.97,
                f"n={x.size}\nΔ={delta:.2f}\nr={corr:.4f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
            )

        axes[3].axis("off")
        axes[3].text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.95),
        )
        fig.suptitle("Runtime Resource Comparison: VIIRS vs GFS", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        out = self.summary_plot_dir / f"runtime_resource_summary_{self.run_stamp}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return df_runtime_pairs

    def run(self) -> None:
        self.log("Starting LaSRC container validation")
        self.load_env_file()
        self.setup_aws()
        records_pre, records_post = self.discover_records()
        self.pair_catalog(records_pre, records_post)
        self.threshold_reference_table()
        _, _, summary_by_band = self.compare_all()
        self.render_panels()
        self.render_summary_plots(summary_by_band)
        self.compare_runtime_logs()
        self.log(f"Validation outputs written to {self.output_root}")


def write_example_config(path: str) -> None:
    target = Path(path).expanduser()
    target.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    print(f"Wrote example config to {target}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch LaSRC container validation")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--write-example-config", help="Write an example JSON config and exit")
    parser.add_argument("--output-root", help="Override output root directory")
    parser.add_argument("--max-granules", type=int, help="Override max granules")
    parser.add_argument(
        "--granule-filter-file",
        help="Override granule_filter_file with a newline-delimited granule list",
    )
    parser.add_argument(
        "--render-all-panels",
        action="store_true",
        help="Render per-granule panels for every granule",
    )
    parser.add_argument(
        "--skip-runtime-logs",
        action="store_true",
        help="Skip runtime log comparison even if runtime_log_source is configured",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.write_example_config:
        write_example_config(args.write_example_config)
        return 0
    if not args.config:
        raise SystemExit("Please provide --config PATH.json")
    config_data = json.loads(Path(args.config).expanduser().read_text())
    if args.output_root:
        config_data["output_root"] = args.output_root
    if args.max_granules is not None:
        config_data["max_granules"] = args.max_granules
    if args.granule_filter_file:
        config_data["granule_filter_file"] = args.granule_filter_file
    if args.render_all_panels:
        config_data["render_all_panels"] = True
    if args.skip_runtime_logs:
        config_data["runtime_log_source"] = None
    runner = ValidationRunner(ValidationConfig.from_dict(config_data))
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
