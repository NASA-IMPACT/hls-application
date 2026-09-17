# HLS Application

This repository contains notebooks, utilities, and validation workflows for the
Harmonized Landsat and Sentinel-2 (HLS) project. It is not limited to validation:
the repository also includes example HLS applications and Google Earth Engine
utilities.

## Repository Layout

```
.
├── hls-notebooks/                 # HLS application notebooks
│   ├── hls_query_by_granule.ipynb
│   ├── hls_urbanization.ipynb
│   └── hls_vegetation_phenology.ipynb
├── hls-gee/                       # Google Earth Engine utilities
│   └── gee_ccdc.py
├── hls_validation_framework/      # Reusable validation resources
│   ├── config/                    # Validation configuration files
│   ├── notebooks/                 # Interactive validation notebooks
│   ├── scripts/                   # Command-line validation runners
│   └── README.md                  # Validation framework instructions
└── .github/workflows/             # Optional GitHub Actions workflows
```

## HLS Applications

The notebooks in [`hls-notebooks/`](hls-notebooks/) demonstrate common HLS
workflows, including granule discovery, urbanization analysis, and vegetation
phenology. Their Python dependencies are listed in
[`hls-notebooks/requirements.txt`](hls-notebooks/requirements.txt).

[`hls-gee/`](hls-gee/) contains Google Earth Engine-oriented utilities for HLS
analysis.

## Validation

The [`hls_validation_framework/`](hls_validation_framework/) directory contains
reusable HLS validation materials, including Fmask and surface-reflectance
acceptance resources.

The LaSRC container-comparison workflow is maintained on the
[`lasrc_validation`](https://github.com/NASA-IMPACT/hls-application/tree/lasrc_validation)
branch so it can evolve independently from the general application notebooks.

## Related Repositories

- [hls-science-container](https://github.com/NASA-IMPACT/hls-science-container) - HLS processing container
- [hls_development](https://github.com/NASA-IMPACT/hls_development) - HLS development tracking
