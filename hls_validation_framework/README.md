# HLS Fmask Acceptance Test Framework

Repeatable acceptance test suite confirming that a new HLS container correctly applies the validated Fmask algorithm.

> **This is container acceptance testing, not scientific validation.**  
> The scientific validation of Fmask5 has already been completed. This framework confirms
> that any new container build reproduces the validated Fmask5 results on a curated set
> of test cases.

---

## Folder Structure

```
hls_validation_framework/
├── HLS_Fmask_acceptance_test.ipynb   # Main acceptance test notebook
├── environment.yml                    # Conda environment
├── README.md                          # This file
│
├── config/
│   └── fmask_acceptance_config.yaml  # ← Edit this to update test parameters
│
├── module/                            # Shared utility modules (from hls_validation)
│   ├── fmask.ipynb                    # Fmask bit-decoding functions
│   ├── data_access.ipynb              # S3 access helpers
│   ├── plotting.ipynb                 # Visualization helpers
│   ├── ultilities.ipynb               # General utilities
│   └── read_file.ipynb                # File reading helpers
│
├── scripts/
│   └── run_fmask_validation.py        # Run notebook from command line (Papermill)
│
└── reports/                           # Generated outputs (gitignored except .gitkeep)
    └── .gitkeep
```

---

## Quick Start

### 1. Set up environment

```bash
mamba env create -f hls_validation_framework/environment.yml
mamba activate lpdaac_vitals
pip install papermill
```

### 2. Set AWS credentials (never hardcode)

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

### 3. Run the test

**Interactive (JupyterLab):**
```bash
jupyter lab hls_validation_framework/HLS_Fmask_acceptance_test.ipynb
```

**Command line (recommended for CI):**
```bash
# From repo root
python hls_validation_framework/scripts/run_fmask_validation.py
```

---

## How It Works

### Test Structure

| Test set | Size | Pass threshold | Purpose |
|---|---|---|---|
| **Curated granules** | ~5–10 granules | 100% pixel agreement | Known Fmask5 improvement cases — must match exactly |
| **Random sample** | ~25–30 granules | ≥ 99.5% pixel agreement | Broader regression coverage |

Both test sets compare the candidate container output against a **golden reference** —
the scientifically validated Fmask5 outputs stored in S3.

### Fmask Comparison Method

Unlike surface reflectance bands, Fmask is **bit-encoded** (not a continuous value).  
Each pixel is decoded into individual class layers before comparison:

| Bit | Class |
|-----|-------|
| 1 | Cloud |
| 2 | Adjacent to cloud |
| 3 | Cloud shadow |
| 4 | Snow / Ice |
| 5 | Water |
| 6–7 | Aerosol level |

Agreement is measured **per class**, not as raw pixel subtraction.

---

## Updating for a New Fmask Release (Science Team)

1. Run the validated container on the curated test granules.
2. Upload output `Fmask.tif` files to the golden reference S3 path:
   ```
   s3://hls-validation/golden-reference/fmask<VERSION>/
   ```
3. Update `config/fmask_acceptance_config.yaml`:
   - Bump `fmask_version`
   - Update `golden_reference_s3_prefix`
   - Update `curated_granules` with new improvement examples if available
4. Commit the config change:
   ```bash
   git add hls_validation_framework/config/fmask_acceptance_config.yaml
   git commit -m "Update golden reference to Fmask5.1"
   git push origin main
   ```

---

## CI / Automation

The GitHub Actions workflow (`.github/workflows/fmask_validation.yml` at repo root)
can trigger this test automatically.

**Manual trigger:** Go to GitHub → Actions → "Fmask Acceptance Test" → Run workflow.  
Input the candidate S3 bucket and prefix for the container build to test.

**Artifacts:** The executed notebook and HTML report are uploaded as GitHub Actions artifacts
and retained for 90 days.
