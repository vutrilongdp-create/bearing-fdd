# IMS-2 Lambda Sensitivity Checkpoint

## Experiment contract

- Dataset: IMS-2, 984 CSV rows, 20,480 points/sample, 20,480 Hz.
- Healthy/training rows: zero-based 0-299 (300 samples).
- Model: regularized full-sample dense Table-1-inspired autoencoder.
- Architecture: `IS-3500-700-200-1-200-700-3500-IS`.
- HI: scalar sigmoid bottleneck.
- Training: Adam, learning rate 0.001, 5 epochs, batch size 64, seed 42,
  chronological overlapping batches.
- Fault threshold: empirical P95 of healthy HI.
- FFP: first of five consecutive HI values above P95.
- Paper comparison uses one-based sample labels; paper IMS-2 FFP is #536.
- XAI and diagnosis were disabled for this screening run.

## Integrity

- All 10 planned cases completed.
- Each case contains `run_config.json`, training history, HI CSV/plot, result
  JSON, and summary CSV.
- Model files were intentionally removed after evaluation.

## Primary result

Under the paper-style signed upper-tail P95 detector:

- Nine cases produced no FFP.
- Only `lambda_monotonic=0.01`, `lambda_smoothing=0.01` produced an FFP:
  zero-based row 970 / one-based sample #971, 435 samples later than paper #536.
- Therefore no tested lambda pair is validated for the declared signed HI/P95
  contract.

## HI behavior

- For nine cases, the HI-time correlation over all 984 samples was negative
  (about -0.28 to -0.51), although the intended degradation direction was
  increasing.
- The unregularized case had only 16/684 evaluation values above P95 and a
  longest run of two.
- `lambda_monotonic=lambda_smoothing=0.01` had 72/684 values above P95 and its
  first qualifying run began at sample #971.
- `lambda_smoothing=0.01` compressed the all-sample HI standard deviation to
  0.00162. The joint 0.01/0.01 case had standard deviation 0.00238. These are
  collapse warnings, not evidence of a useful smooth HI.
- Decreasing-step ratios remained near 0.5 for all cases.

## Loss allocation

For the joint 0.01/0.01 case at epoch 5:

- Reconstruction loss: 0.00594848.
- Weighted monotonic contribution / reconstruction: 0.0686.
- Weighted smoothing contribution / reconstruction: 0.00243.
- Only 5.24% of raw monotonic loss came from the scalar HI; the rest came from
  3,500-dimensional features.
- Only 6.59% of raw smoothing loss came from the scalar HI; the rest came from
  200-dimensional features.

The shared lambdas therefore act mainly on hidden representations and do not
directly guarantee a useful out-of-sample HI.

## Post-hoc diagnostic checks

These checks are project diagnostics, not the paper's method:

- Reversing the latent direction (`1 - HI`) moved the best FFP to #622 for
  `(1e-4, 0)` and `(1e-4, 1e-4)`, still 86 samples later than paper.
- A sign-invariant absolute-deviation score gave #554 for `(1e-4, 1e-4)`, but
  gave an implausibly early #351 for `(1e-4, 0)`. This detector is not yet
  stable enough to adopt.

## Decision

Do not select 0.01/0.01 and do not report the regularized full-sample variant
as reproduced. The windowed project baseline previously produced approximately
#532, but it uses reconstruction-P95 HI and is a separate engineering variant.

The next regularized experiment should separate hidden-feature and scalar-HI
coefficients, prioritize HI-only penalties, and rerun `(0, 0)`, `(1e-4, 0)`,
and `(1e-4, 1e-4)`-derived candidates using multiple seeds. Signed P95 and any
project-only sign-invariant detector must be reported separately.
