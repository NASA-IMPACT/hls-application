# HLS Validation Suite

Validation notebooks and container acceptance testing for the **Harmonized Landsat and Sentinel-2 (HLS)** product suite.

---

## Repository Structure

```
hls-validation/
│
├── hls_validation/                        # Existing HLS analysis notebooks
│   ├── HLS_validation_general.ipynb       # SR band container regression test (all bands)
│   ├── HLS_validation.ipynb               # Original HLS validation notebook
│   ├── IGARSS_2026_paper_figures.ipynb    # IGARSS 2026 paper figures
│   └── module/                            # Shared utility modules
│       ├── fmask.ipynb
│       ├── data_access.ipynb
│       ├── plotting.ipynb
│       ├── statistics.ipynb
│       ├── time_series.ipynb
│       └── ultilities.ipynb
│
├── hls_validation_framework/              # ← Fmask acceptance test framework
│   ├── HLS_Fmask_acceptance_test.ipynb    # Main acceptance test notebook
│   ├── environment.yml
│   ├── README.md                          # Framework-specific documentation
│   ├── config/
│   │   └── fmask_acceptance_config.yaml  # Test parameters, granule list, thresholds
│   ├── module/                            # Shared modules (synced from hls_validation/module)
│   ├── scripts/
│   │   └── run_fmask_validation.py        # CLI runner (Papermill)
│   └── reports/                           # Generated reports (gitignored)
│
└── .github/
    └── workflows/
        └── fmask_validation.yml           # GitHub Actions CI workflow
```

---

## Two-Tier Validation Strategy

### Tier 1 — SR Band Regression Test
**Notebook:** `hls_validation/HLS_validation_general.ipynb`  
**Purpose:** Detect any unintended changes across all output bands (SR + VI + Fmask) between two container builds.  
**When to run:** Every container rebuild.

### Tier 2 — Fmask Acceptance Test
**Notebook:** `hls_validation_framework/HLS_Fmask_acceptance_test.ipynb`  
**Purpose:** Confirm the new container correctly applies the validated Fmask algorithm.  
**Comparison:** Against a golden reference — the scientifically validated Fmask5 outputs.  
**When to run:** Every container rebuild that touches Fmask; every new Fmask release.

See [`hls_validation_framework/README.md`](hls_validation_framework/README.md) for full documentation.

---

## Related Repositories

- [hls-science-container](https://github.com/NASA-IMPACT/hls-science-container) — HLS processing container
- [hls_development](https://github.com/NASA-IMPACT/hls_development) — HLS development tracking
