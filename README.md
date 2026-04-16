# HLS Validation Suite

Validation notebooks and repeatable container acceptance testing for the **Harmonized Landsat and Sentinel-2 (HLS)** product suite.

---

## Repository Structure

```
hls-validation/
│
├── hls_validation_framework/              # ← Repeatable validation framework
│   ├── config/
│   │   ├── fmask_acceptance_config.yaml  # Fmask test parameters
│   │   └── sr_regression_config.yaml     # SR regression test parameters
│   ├── notebooks/
│   │   ├── HLS_Fmask_acceptance_test.ipynb
│   │   └── HLS_validation_general.ipynb
│   ├── module/                            # Shared utility modules
│   ├── scripts/
│   │   └── run_fmask_validation.py        # CLI runner (Papermill)
│   └── README.md                          # Framework documentation
│
├── hls_validation/                        # Existing analysis notebooks (unchanged)
│   ├── HLS_validation.ipynb
│   ├── IGARSS_2026_paper_figures.ipynb
│   └── module/
│
└── .github/
    └── workflows/
        └── fmask_validation.yml           # GitHub Actions CI
```

See **[`hls_validation_framework/README.md`](hls_validation_framework/README.md)** for full documentation on how to run the validation suite.

---

## Related Repositories

- [hls-science-container](https://github.com/NASA-IMPACT/hls-science-container) — HLS processing container
- [hls_development](https://github.com/NASA-IMPACT/hls_development) — HLS development tracking
