# BEARING-FDD Kaggle Package

This folder contains a reproducible Kaggle version of the project pipeline.
The notebook trains each dataset from scratch, frees TensorFlow memory between
datasets, computes HI/threshold/FFP, produces XAI correlations, and diagnoses
representative samples using Kurtogram, bandpass filtering, envelope FFT, and
bearing harmonics.

## Kaggle Inputs

Create or attach a Kaggle dataset containing these CSV files:

```text
IMS1.csv
IMS2.csv
IMS3.csv
XJTU_SY_2_1.csv
XJTU_SY_2_3.csv
XJTU_SY_3_1.csv
XJTU_SY_3_4.csv
healthyIMS1.csv
healthyIMS2.csv
healthyIMS3.csv
healthyXJTU_SY_2_1.csv
healthyXJTU_SY_2_3.csv
healthyXJTU_SY_3_1.csv
healthyXJTU_SY_3_4.csv
```

The healthy baseline files are optional. When absent, the mean of the first 20
samples is used for fault isolation.

### How to provide the input

**Recommended:** open the notebook on Kaggle, select **Add Input**, and attach
the dataset containing the CSV files. Keep:

```python
ATTACHED_INPUT_ROOT = "/kaggle/input"
KAGGLE_DATASET_URL = ""
```

Kaggle mounts the selected dataset below
`/kaggle/input/<dataset-slug>/`. The notebook searches recursively, so do not
paste the web URL into `input_root`.

**Optional URL mode:** set:

```python
KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/<owner>/<dataset-slug>"
)
```

This invokes `kagglehub.dataset_download()` and may download the full dataset
again. It can require Internet access or authentication for private datasets.

When running locally, replace `ATTACHED_INPUT_ROOT` with the directory that
contains the CSV files, for example:

```python
ATTACHED_INPUT_ROOT = "/path/to/bearing-data"
```

## Recommended Run Order

1. Upload `01_bearing_fdd_pipeline.ipynb` to Kaggle.
2. Attach the dataset containing the CSV files.
3. Enable a GPU accelerator.
4. First set `DATASETS_TO_RUN = ("IMS2",)` and run all cells.
5. After IMS-2 passes, select all seven datasets and run again.
6. Download `/kaggle/working/bearing_fdd_results.zip`.

The notebook uses zero-based CSV row indices. This is stated in every diagnosis
record to avoid silently mixing paper labels with Python indices.

## Lambda Sensitivity

Use `02_lambda_sensitivity.ipynb` to screen the project-defined
monotonicity and smoothing coefficients on IMS-2. The default matrix contains
10 cases covering no regularization, each regularizer alone, and equal-weight
combinations at `1e-6`, `1e-4`, and `1e-2`.

The sensitivity notebook:

- keeps every case in a separate output directory;
- resumes completed cases after a Kaggle session interruption;
- disables XAI and diagnosis during screening;
- records weighted regularization-to-reconstruction loss ratios;
- reports both zero-based CSV indices and one-based paper-comparable FFPs;
- removes saved model files after evaluation to control disk usage.

After screening, rerun the best two or three configurations with at least three
seeds before enabling XAI and diagnosis. A close FFP alone is not sufficient
evidence that a coefficient pair is scientifically preferable.

## Latent-Dimension Sensitivity

Use `03_latent_dimension_sensitivity.ipynb` to test full-sample
bottleneck dimensions `1, 8, 16, 32, 64` on IMS-2:

```text
IS-3500-700-200-k-200-700-3500-IS
```

The decoder reconstructs from all `k` latent values. The scalar sample HI is
full-sample reconstruction MSE, not a coordinate of the latent vector. This
separates representation capacity from fault scoring. It is a project
experiment, not an exact reproduction of the paper's one-node Table-1 model.

## Important Method Choice

The default model is `regularized_full_sample`:

```text
full raw sample
-> Dense 3500 monotonic-feature representation
-> Dense 700
-> Dense 200 smoothed-feature representation
-> sigmoid scalar HI
-> symmetric decoder
```

Training keeps samples in chronological order and minimizes reconstruction MSE
plus monotonic penalties on the 3,500-dimensional representation and HI, and
second-difference smoothing penalties on the 200-dimensional representation
and HI.

The paper does not publish these equations or lambda values. This is a
transparent mathematical implementation, not a claim of exact source
reproduction. The previous implementation remains available as
`model_variant="dense_windowed_baseline"`.

For both variants, the FFP threshold is P95 of healthy HI. FFP is the first
sample in a run of five consecutive values above that threshold.

## Package Layout

```text
kaggle/
├── notebooks/   01_bearing_fdd_pipeline, 02_lambda_sensitivity, 03_latent_dimension_sensitivity
├── src/         kaggle_bearing_fdd.py (pipeline, embedded into every notebook)
│                dense_windowed_baseline_audit.py (zero-output / RMS² baseline audit)
├── scripts/     build_<notebook>.py (regenerate a notebook from src/) and validate_package.py
├── tests/       unit tests for the audit helpers
└── notes/       method notes on regularisation, multi-latent reconstruction and the Kaggle port
```

Results of the two sensitivity studies (IMS-2, seed 42) are in
[`../results/05_lambda_sensitivity`](../results/05_lambda_sensitivity) and
[`../results/06_latent_dimension_sensitivity`](../results/06_latent_dimension_sensitivity).

After editing `src/kaggle_bearing_fdd.py`, rebuild and check the notebooks:

```bash
python scripts/build_01_bearing_fdd_pipeline.py
python scripts/build_02_lambda_sensitivity.py
python scripts/build_03_latent_dimension_sensitivity.py
python scripts/validate_package.py
python -m pytest tests -q
```
