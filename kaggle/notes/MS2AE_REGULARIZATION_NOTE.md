# Mathematical MS2AE Implementation

## Scope

The paper specifies the structure
`IS-3500-700-200-1-200-700-3500-IS` and names the 3,500-node and
200-node layers monotonicity and smoothing layers. It does not publish their
forward equations, regularization objective, or regularization coefficients.

The project implementation therefore separates two claims:

- **Paper-specified:** layer widths, Adam, learning rate 0.001, five epochs,
  batch size 64, and one-node HI.
- **Project-defined:** differentiable monotonicity and smoothing losses and
  their coefficients.

The maintained sensitivity experiment is
`02_lambda_sensitivity.ipynb`. Its first stage separates the effects of
the two regularizers instead of changing both coefficients at once. Results
must report one-based FFP labels when comparing with the paper and retain the
zero-based CSV row index for code traceability.

The separate `multilatent_full_sample` experiment does not use these sequence
regularizers. It tests bottleneck capacity and defines HI as full-sample
reconstruction MSE. Its results must not be mixed with scalar latent-HI results.

## Objective

For chronological sample index \(t\), let \(m_t\) denote the 3,500-dimensional
monotonic representation, \(s_t\) the 200-dimensional smoothed representation,
and \(h_t\) the scalar HI.

\[
\mathcal{L}_{mon}(z)
=\operatorname{mean}\left[\max(0,z_t-z_{t+1})\right].
\]

This loss is zero when every considered component is non-decreasing.

\[
\mathcal{L}_{smooth}(z)
=\operatorname{mean}\left[
(z_{t+1}-2z_t+z_{t-1})^2
\right].
\]

The second-difference penalty suppresses local oscillation without forcing the
trajectory to be constant.

The complete training objective is:

\[
\begin{aligned}
\mathcal{L}={}&\mathcal{L}_{reconstruction}\\
&+\lambda_m\left[
\mathcal{L}_{mon}(m)+\mathcal{L}_{mon}(h)
\right]\\
&+\lambda_s\left[
\mathcal{L}_{smooth}(s)+\mathcal{L}_{smooth}(h)
\right].
\end{aligned}
\]

## Training Requirements

- Preserve chronological sample order.
- Do not use `shuffle=True`.
- Use batches of at least three samples.
- Overlap adjacent batches by two samples so first and second differences at
  batch boundaries are included.
- Save \(\lambda_m\), \(\lambda_s\), component losses, random seed, and model
  variant with each run.

## Interpretation Limit

These are soft penalties. They reduce monotonicity violations but cannot
guarantee a perfectly monotonic HI on unseen faulty samples. The pipeline
therefore reports the decreasing-step ratio and second-difference RMSE rather
than assuming the desired behavior was achieved.
