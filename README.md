# HLS Fmask Validation

A focused validation workflow for comparing candidate HLS `Fmask.tif` outputs
with a scientifically validated golden reference collection. It measures
agreement for cloud, cloud shadow, snow/ice, water, adjacency-to-cloud-shadow,
and aerosol bits.

## Repository Layout

```
.
├── config/
│   └── fmask_acceptance_config.yaml  # S3 inputs, thresholds, and sample settings
├── notebooks/
│   └── HLS_Fmask_acceptance_test.ipynb
├── scripts/
│   └── run_fmask_validation.py       # Papermill command-line runner
├── environment.yml                   # Conda environment
└── .gitignore
```

## Configure

Edit [`config/fmask_acceptance_config.yaml`](config/fmask_acceptance_config.yaml):

- `golden_reference_s3_bucket` and `golden_reference_s3_prefix`: validated Fmask outputs.
- `candidate_s3_bucket` and `candidate_s3_prefix`: candidate output collection.
- `curated_granules`: optional known science cases. Keep it empty until real validated IDs are available.
- `random_sample`: reproducible paired-granule sample and agreement threshold.

Use an AWS profile or standard environment variables. Do not store credentials in
the configuration file.

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

```bash
mamba env create -f environment.yml
mamba activate lpdaac_vitals
cd notebooks
jupyter lab HLS_Fmask_acceptance_test.ipynb
```

Run the cells from top to bottom. Generated CSV summaries and optional figures
are written under `reports/`, which is created automatically and ignored by Git.

## Run From the Command Line

The runner executes the notebook with Papermill and saves an executed notebook
report.

```bash
pip install papermill

python scripts/run_fmask_validation.py \
  --config config/fmask_acceptance_config.yaml \
  --output reports/fmask_validation.ipynb
```

To create an HTML report:

```bash
jupyter nbconvert --to html reports/fmask_validation.ipynb
```

## Interpretation

- Curated scenes, when configured, must meet `curated_pass_pct`.
- Random paired scenes must meet `random_pass_pct`.
- The run fails if any evaluated curated or random scene fails its threshold.
- An empty curated list is valid; the result then reflects the random paired-scene check only.
