# HLS Validation Framework

Repeatable test suite for HLS container validation covering both **surface reflectance band regression** and **Fmask acceptance testing**.

---

## Folder Structure

```
hls_validation_framework/
├── README.md                              # This file
├── environment.yml                        # Conda environment
│
├── config/                                # ← All parameters live here, not in notebooks
│   ├── fmask_acceptance_config.yaml       # Fmask test: golden reference, granule list, thresholds
│   └── sr_regression_config.yaml         # SR test: pre/post bucket paths, product types
│
├── notebooks/
│   ├── HLS_Fmask_acceptance_test.ipynb   # Tier 2: Fmask acceptance test
│   └── HLS_validation_general.ipynb      # Tier 1: SR band regression test
│
├── module/                                # Shared utility modules
│   ├── fmask.ipynb                        # Fmask bit-decoding functions
│   ├── data_access.ipynb                  # S3 access helpers
│   ├── plotting.ipynb                     # Visualization helpers
│   ├── ultilities.ipynb                   # General utilities
│   └── read_file.ipynb                    # File reading helpers
│
├── scripts/
│   └── run_fmask_validation.py            # CLI runner (Papermill) for both notebooks
│
└── reports/                               # Generated outputs (gitignored)
    └── .gitkeep
```

---

## Two-Tier Validation Strategy

### Tier 1 — SR Band Regression Test
**Notebook:** `notebooks/HLS_validation_general.ipynb`  
**Config:** `config/sr_regression_config.yaml`  
**Question:** *Did anything change across all output bands between two container builds?*  
**Scope:** All `.tif` files — SR bands (B01–B12), VI indices, angles, Fmask.  
**Comparison:** `prod` container vs `dev` container (pixel-level numerical diff).  
**When to run:** Every container rebuild.

### Tier 2 — Fmask Acceptance Test
**Notebook:** `notebooks/HLS_Fmask_acceptance_test.ipynb`  
**Config:** `config/fmask_acceptance_config.yaml`  
**Question:** *Did the container correctly apply the validated Fmask algorithm?*  
**Scope:** `Fmask.tif` files only — bit-decoded per class (cloud/shadow/water/snow).  
**Comparison:** New container vs **golden reference** (scientifically validated Fmask5 outputs).  
**When to run:** Every container rebuild touching Fmask; every new Fmask release.

---

## Quick Start

### 1. Set up environment

```bash
mamba env create -f hls_validation_framework/environment.yml
mamba activate lpdaac_vitals
pip install papermill
```

### 2. Set AWS credentials

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

### 3. Run tests

**Command line (recommended):**
```bash
# Run Fmask acceptance test
python hls_validation_framework/scripts/run_fmask_validation.py --test fmask

# Run SR regression test
python hls_validation_framework/scripts/run_fmask_validation.py --test sr

# Run both
python hls_validation_framework/scripts/run_fmask_validation.py --test fmask
python hls_validation_framework/scripts/run_fmask_validation.py --test sr
```

**Interactive (JupyterLab):**
```bash
jupyter lab hls_validation_framework/notebooks/
```

---

## Updating Config for a New Container Build (Engineering Team)

Edit `config/sr_regression_config.yaml`:
```yaml
pre_s3_prefix:  "outputs/<old-experiment>/prod/"
post_s3_prefix: "outputs/<new-experiment>/dev/"
```

No notebook changes needed.

---

## Updating for a New Fmask Release (Science Team)

1. Upload validated Fmask outputs to:
   ```
   s3://hls-validation/golden-reference/fmask<VERSION>/
   ```
2. Edit `config/fmask_acceptance_config.yaml` — bump `fmask_version`, update `golden_reference_s3_prefix`, add new `curated_granules` if applicable.
3. Commit and push — CI auto-triggers on config file changes.

---

## CI / GitHub Actions

Workflow: `.github/workflows/fmask_validation.yml`

**Manual trigger:** GitHub → Actions → "HLS Validation Suite" → Run workflow  
**Inputs:** `test_type` (fmask / sr / both), candidate S3 bucket and prefix.  
**Auto-trigger:** Runs when either config file is updated on `main`.  
**Artifacts:** HTML reports retained 90 days.
