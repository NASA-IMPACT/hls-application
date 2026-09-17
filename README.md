# LaSRC Container Validation

Standalone tools for comparing paired LaSRC/HLS output trees from two container builds. The shared workflow is intended for S3-backed experiments and applies the CLI's defined reference thresholds by default.

## Layout

```text
.
├── README.md
├── environment.yml
├── config/
│   └── lasrc_s3.example.json
├── notebooks/
│   └── LaSRC_container_validation.ipynb
└── scripts/
    ├── lasrc_container_validation_cli.py
    └── run_lasrc_container_validation.py
```

- **Notebook:** visual inspection of up to 10 paired granules.
- **CLI:** reproducible, large-set comparison with reports and summary plots.
- **Config:** an S3 template. Copy it for each experiment; do not commit credentials or run-specific config files.

## Setup

From this directory:

```bash
mamba env create -f environment.yml
mamba activate lasrc_validation
```

Authenticate to AWS using your organization-approved method:

```bash
aws sso login --profile YOUR_PROFILE
```

Alternatively, export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`. The identity needs `s3:ListBucket` and `s3:GetObject` permissions for both output roots.

S3 roots use `bucket/prefix/` notation, for example:

```text
YOUR-BUCKET/your-experiment/pre/
```

## Visual Review: Notebook

The notebook defaults to a final-output comparison of 10 granules.

1. Open `notebooks/LaSRC_container_validation.ipynb` in JupyterLab or Cursor.
2. Select the `lasrc_validation` kernel.
3. In the first configuration cell, set `folder_pre`, `folder_post`, labels, and `AWS_PROFILE` if required.
4. Optionally set `GRANULE_FILTERS` to inspect particular IDs.
5. Run all cells from top to bottom.

```bash
jupyter lab notebooks/LaSRC_container_validation.ipynb
```

Figures and reports are written under `outputs/lasrc_validation/` and `reports/` relative to the notebook's parent directory.

## Large Comparison: CLI

1. Create a working config:

```bash
cp config/lasrc_s3.example.json config/my_validation.json
```

2. Edit these fields in `config/my_validation.json`:

```json
"folder_pre": "YOUR-BUCKET/path/to/pre-output/",
"folder_post": "YOUR-BUCKET/path/to/post-output/",
"pre_label": "Pre container",
"post_label": "Post container",
"aws_profile": "YOUR_PROFILE",
"output_root": "./outputs/my_validation"
```

3. Smoke test three paired granules:

```bash
python scripts/run_lasrc_container_validation.py \
  --config config/my_validation.json \
  --max-granules 3 \
  --skip-runtime-logs
```

4. Run the complete catalog after reviewing the smoke-test output:

```bash
python scripts/run_lasrc_container_validation.py \
  --config config/my_validation.json \
  --skip-runtime-logs
```

Add `--render-all-panels` only when needed. It creates a panel for every paired granule and can take substantial time and storage.

## Configuration Notes

- `data_source_mode: "s3"` reads the two S3 roots directly.
- `s3_read_mode: "cache"` stores downloaded rasters under `<output_root>/_cache/`; delete that folder after a run if needed.
- `comparison_stage` accepts `final`, `immediate`, or `auto`. Start with `final` unless both output trees retain comparable immediate files.
- `threshold_mode: "reference"` applies the reference thresholds defined in the CLI. Use `custom` only for a study with explicitly approved alternative thresholds.
- Leave `runtime_log_source` as `null` unless the two datasets have comparable logs.

## Outputs

Each run writes timestamped outputs under `output_root`:

- `reports/paired_catalog_*.csv`: discovered matching records.
- `reports/lasrc_validation_metrics_*.csv`: metrics per granule, band, and stage.
- `reports/summary_by_band_*.csv` and `overall_summary_*.csv`: aggregate results.
- `plots/`: summary figures and requested granule panels.
