# IMS-2 Multi-Latent Dimension Checkpoint

## Experiment contract

- Dataset: IMS-2, 984 rows, 20,480 points/sample, 20,480 Hz.
- Training/healthy rows: zero-based 0-299.
- Architecture: `IS-3500-700-200-k-200-700-3500-IS`.
- Tested bottlenecks: `k = 1, 8, 16, 32, 64`.
- Training: Adam, learning rate 0.001, 5 epochs, batch size 64, seed 42.
- HI: full-sample reconstruction MSE.
- Threshold: empirical P95 of the first 300 healthy HI values.
- FFP: first of five consecutive HI values above P95.
- Paper comparison: one-based IMS-2 FFP #536.
- XAI and diagnosis were disabled.

## Integrity

- All five planned cases completed.
- Every case contains config, training history, HI CSV/plot, result JSON, and
  summary CSV.
- Model files were intentionally removed after evaluation.

## Reported FFP

All five bottleneck dimensions produced:

```text
zero-based FFP row = 532
one-based FFP sample = #533
difference from paper #536 = -3 samples
```

Training reconstruction losses, P95 thresholds, and HI statistics were also
nearly identical across dimensions.

## Degeneracy audit

The apparently strong FFP result does not validate the learned latent
representation:

- Pairwise correlations between the five HI trajectories were greater than
  0.99999985.
- Correlations between each reconstruction-MSE HI and raw sample energy
  `mean(x^2)` were greater than 0.99999899.
- Raw sample energy alone produced one-based FFP #533.
- From #533 onward, the `k=1` HI stayed above P95 for 450 consecutive samples;
  450 of the remaining 452 samples exceeded the threshold. The energy change is
  therefore persistent rather than a five-sample spike.
- Mean absolute differences between reconstruction HI and raw energy were only
  about 0.21%-0.25% of mean raw energy.
- On the 300 healthy samples, the trained autoencoders reduced MSE by only
  about 0.49%-0.58% relative to a zero-output reconstruction.

Therefore the decoder is operating close to a zero predictor. Its
reconstruction MSE is effectively the raw vibration mean square, equivalent to
RMS squared. Changing `k` cannot materially affect the FFP under this regime.

## Decision

- Do not select a latent dimension from this run.
- Do not claim that multi-latent compression learned degradation features.
- Report #533, if used, as an RMS/energy baseline result rather than an
  autoencoder representation result.
- Do not spend the next run on multiple seeds for `k`; the five variants are
  functionally indistinguishable.

## Recommended next audit

1. Add the explicit raw `mean(x^2)`/RMS baseline to all comparisons.
2. Audit the existing 2,048-point windowed baseline against a zero-output
   predictor to determine whether its approximately #532 result is also driven
   mainly by signal energy.
3. For learned reconstruction, move to a local-pattern representation such as
   a 1D convolutional autoencoder on windows, and record reconstruction
   improvement over the zero baseline before accepting its HI.
4. Keep any convolutional/windowed experiment separate from the paper's dense
   one-node Table-1 architecture.
