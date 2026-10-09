# Multi-Latent Full-Sample Reconstruction Experiment

## Motivation

The scalar-latent sensitivity run on IMS-2 did not validate the regularized
full-sample variant: nine of ten lambda cases produced no FFP, and the only FFP
was sample #971 versus paper #536. A one-node bottleneck was simultaneously
used as the Health Index and as the complete representation for reconstructing
20,480 waveform values.

## Project experiment

The new variant is:

```text
IS-3500-700-200-k-200-700-3500-IS
```

with:

```text
k in {1, 8, 16, 32, 64}
training loss = full-sample reconstruction MSE
sample HI = full-sample reconstruction MSE
threshold = empirical P95 of the first 300 healthy HI values
FFP = first of five consecutive HI values above P95
```

The bottleneck activation is linear. The decoder receives all `k` latent
coordinates. There is no scalar latent HI head and no monotonicity/smoothing
regularization in this experiment.

## Interpretation

- `k=1` is a representation-capacity control, not the previous sigmoid
  scalar-HI model.
- Increasing `k` tests reconstruction capacity while keeping all other layer
  widths fixed.
- The scalar HI is derived from reconstruction error and is never mixed with a
  latent coordinate in the same run.
- This is a project modification, not the paper's Table-1 architecture.
- It is also separate from the 2,048-point windowed baseline, whose sample HI
  is P95 across window reconstruction errors.

## Reproduction

Run:

```text
03_latent_dimension_sensitivity.ipynb
```

on IMS-2 with Kaggle GPU. The notebook saves each dimension in a separate
directory, supports resume, records both zero-based and one-based FFP values,
and removes model files after evaluation to limit disk use.

Do not select `k` from a single seed or FFP distance alone. Confirm the best one
or two dimensions with seeds 7, 42, and 123 before enabling XAI and diagnosis.
