# Kaggle Port Audit

## Scope

The Kaggle package consolidates the current project scripts for MS2AE training,
HI/FFP evaluation, XAI, Kurtogram selection, envelope analysis, and harmonic
diagnosis. It does not copy the Flask or Streamlit interface because Kaggle is
used here as a reproducible compute environment.

## Inconsistencies Resolved

1. **HI definition:** Some training code computed healthy thresholds from the
   one-neuron encoder output, while the final evaluation computed HI from
   reconstruction error. The Kaggle package exposes two explicit variants:
   regularized full-sample latent HI (default) and the windowed
   reconstruction-P95 baseline. A single run never mixes the definitions.
2. **Dataset coverage:** IMS and XJTU-SY had separate training scripts, and
   IMS-2 was omitted from one evaluation registry. The Kaggle registry includes
   all seven datasets.
3. **Memory cleanup:** Every dataset is trained, saved, evaluated, and followed
   by `clear_session()` and garbage collection.
4. **Data dimensions:** Validation requires 20,480 values for IMS and 32,768
   values for XJTU-SY. The XJTU dimension corresponds to 25,600 Hz multiplied
   by 1.28 s.
5. **DTW transparency:** If `fastdtw` is unavailable, the output explicitly
   records `direct_difference_fastdtw_unavailable`; it never labels that result
   as DTW.
6. **Stage thresholds:** The notebook records equations (5) and (6) exactly as
   printed and keeps them separate from the P95 fault-detection threshold.
7. **Indexing:** Representative diagnosis samples are explicitly recorded as
   zero-based CSV row indices.

## Deliberate Project Choices

- The regularized variant uses the full 20,480/32,768-point sample. The
  optional baseline uses 2,048-point windows as a project engineering choice.
- Monotonic/smoothing equations and lambda values are explicit project choices
  because the paper does not publish those details.
- `multilatent_full_sample` is a separate project experiment. It uses a
  `k`-dimensional bottleneck and full-sample reconstruction MSE as HI; it must
  not be reported as the paper's one-node Table-1 model or mixed with latent HI.
- No hidden normalization is applied. Any future normalization must be added to
  both training and evaluation and recorded in `run_config.json`.
- XAI defaults to all samples to reproduce current project tables. Set
  `xai_scope="faulty"` to analyze only samples from FFP onward.
- XAI uses engineering features and Pearson correlation, not SHAP or LIME.

## Expected Outputs

The run writes model files, training logs, loss plots, HI curves, correlation
matrices, per-sample Kurtogram tables, envelope spectra, a CSV summary, and a
complete JSON record under `/kaggle/working/bearing_fdd_results`.
