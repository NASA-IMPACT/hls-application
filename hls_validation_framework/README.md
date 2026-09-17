# HLS Fmask Acceptance Validation

This workflow compares a candidate HLS `Fmask.tif` output collection with a
scientifically validated golden reference collection. It evaluates agreement for
cloud, cloud shadow, snow/ice, water, adjacency-to-cloud-shadow, and aerosol bits.

## Relevant Files

```
hls_validation_framework/
├── config/fmask_acceptance_config.yaml
├── environment.yml
├── notebooks/HLS_Fmask_acceptance_test.ipynb
├── scripts/run_fmask_validation.py
└── reports/                         # Generated reports; ignored by Git
```

## Configure

Edit [`config/fmask_acceptance_config.yaml`](config/fmask_acceptance_config.yaml):

- `golden_reference_s3_bucket` and `golden_reference_s3_prefix`: validated Fmask outputs.
- `candidate_s3_bucket` and `candidate_s3_prefix`: output collection being tested.
- `curated_granules`: optional known science cases. Leave it empty until real validated IDs are available.
- `random_sample`: reproducible paired-granule sample and agreement threshold.

The configuration contains no credentials. Use either an AWS profile or standard
AWS environment variables.

```bash
aws sso login --profile YOUR_PROFILE
export AWS_PROFILE=YOUR_PROFILE
```

Or:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

## Run Interactively

Create the environment, then open the notebook from the framework directory so
the relative configuration path resolves correctly.

```bash
mamba env create -f hls_validation_framework/environment.yml
mamba activate lpdaac_vitals
cd hls_validation_framework/notebooks
jupyter lab HLS_Fmask_acceptance_test.ipynb
```

Run the cells from top to bottom. Reports, CSV summaries, and optional comparison
figures are written under `hls_validation_framework/reports/`.

## Run From the Command Line

The runner executes the same notebook with Papermill and writes an executed
notebook report.

```bash
pip install papermill

python hls_validation_framework/scripts/run_fmask_validation.py \
  --config hls_validation_framework/config/fmask_acceptance_config.yaml \
  --output hls_validation_framework/reports/fmask_validation.ipynb
```

Convert a completed notebook to HTML if needed:

```bash
jupyter nbconvert --to html hls_validation_framework/reports/fmask_validation.ipynb
```

## Interpretation

- Curated scenes, when configured, must meet `curated_pass_pct`.
- Random paired scenes must meet `random_pass_pct`.
- The run fails if any evaluated curated or random scene fails its applicable threshold.
- An empty curated list is valid, but means the run only reports the random paired-scene result.

## GitHub Actions

`.github/workflows/fmask_validation.yml` provides an optional manual or
configuration-triggered Fmask acceptance run. Repository AWS secrets must be
configured before enabling it.
