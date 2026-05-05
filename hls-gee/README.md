# HLS + CCDC Forest Monitoring

This repository contains a publication-ready notebook workflow for demonstrating how Harmonized Landsat Sentinel-2 (`HLS`) improves forest disturbance monitoring relative to `L30` or `S30` alone.

The main notebook is:

- `notebooks/hls_ccdc_forest_monitoring.ipynb`

The notebook now keeps most reusable logic in:

- `gee_hls/notebook_helpers.py`

That refactor keeps the notebook focused on demonstration, interpretation, and figure generation rather than long blocks of utility code.

## What the Notebook Covers

- site scouting and observation-count comparison
- pixel-level CCDC fitting for `L30`, `S30`, and merged `HLS`
- spatial disturbance mapping
- Hansen Global Forest Change comparison and validation
- optional hotspot scouting, animation, and diagnostics

## Repository Layout

```text
gee_hls/
├── gee_hls/
│   ├── __init__.py
│   └── notebook_helpers.py
├── notebooks/
│   ├── hls_ccdc_forest_monitoring.ipynb
│   └── figures/
├── environment.yml
├── requirements.txt
└── README.md
```

## Environment Setup

Choose either Conda or pip.

### Option 1 — Conda

```bash
conda env create -f environment.yml
conda activate hls-ccdc
```

### Option 2 — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Google Earth Engine Setup

The notebook requires an authenticated Earth Engine environment.

Typical first-time setup:

```python
import ee
ee.Authenticate()
```

You can also provide a project through the `EE_PROJECT` environment variable before starting Jupyter:

```bash
export EE_PROJECT=your-ee-project-id
```

## How to Use the Notebook

1. Open `notebooks/hls_ccdc_forest_monitoring.ipynb`.
2. Run the setup/import cell first.
3. Optionally run the scouting utilities near the top of the notebook if you want to choose a site from hotspot screening.
4. Edit only the **Site Configuration** cell for routine use.
5. Run the notebook top-to-bottom.

## Recommended Workflow

- Use the site-survey or hotspot cells to identify a location with both:
  - meaningful recent forest loss
  - a clear HLS observation-density advantage
- Use precomputed Earth Engine CCDC assets whenever possible for the spatial workflow.
- Use the Hansen preview cell before quantitative validation to check domain alignment.

## Main Outputs

Outputs are written under `notebooks/figures/`, usually in a site-specific subdirectory.

Core outputs include:

- annual observation-frequency figure
- pixel-level CCDC time-series figures
- representative RGB snapshot figure
- spatial disturbance maps
- Hansen validation figure
- diagnostics figure
- Markdown validation report

## Notes for Contributors

- The notebook is intended to stay presentation-friendly.
- Reusable logic should go into `gee_hls/notebook_helpers.py`.
- Keep paths repo-relative whenever possible so the workflow remains portable.
