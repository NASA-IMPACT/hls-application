# HLS Validation and Application Notebooks

Welcome to the **HLS Validation and Application Notebooks** repository! This repository contains Jupyter notebooks and supporting resources for validating the Harmonized Landsat Sentinel (HLS) datasets and demonstrating their practical applications in remote sensing and environmental analysis.

---

## 📚 Repository Overview

This repo provides:

- **Validation notebooks** that assess the accuracy and quality of HLS products against reference data.
- **Application notebooks** illustrating use cases such as land cover classification, change detection, vegetation monitoring, and more.
- Supporting scripts and utilities for data processing and visualization.

├── hls-application/                # Jupyter notebooks for scientific applications
├── hls-validation/   # Jupyter notebooks for validation 
├── environmental.yml          # Python package dependencies
└── README.md                 # This README file


---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- Jupyter Notebook or JupyterLab  
- Key Python libraries: `numpy`, `pandas`, `matplotlib`, `rasterio`, `xarray`, `geopandas`, `scikit-learn`, `h5py`, etc.

You can install the dependencies using:
```bash
mamba env create -f environment.yml


├── notebooks/                # Jupyter notebooks for validation and applications
│   ├── hls_validation.ipynb
│   ├── land_cover_classification.ipynb
│   └── vegetation_monitoring.ipynb
├── data/                     # Sample data or links to datasets (if applicable)
├── scripts/                  # Python scripts for data processing and utilities
├── requirements.txt          # Python package dependencies
└── README.md                 # This README file
