"""Audit dense windowed baseline against zero-output and raw RMS^2 baselines.

This script intentionally keeps three health-index definitions separate:

1. dense_windowed_hi: P95 of dense-AE window reconstruction MSE.
2. zero_window_p95: P95 of zero-output window reconstruction MSE.
3. raw_rms2: mean(x^2) over the usable sample length.

Indices in saved summaries include both zero-based CSV row indices and
one-based paper-style sample labels.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path(
    "BEARING_FDD_WebApp_ProjectModels/backend_flask/prog_analizador/data/IMS2.csv"
)
DEFAULT_MODEL_PATH = Path(
    "BEARING_FDD_WebApp_ProjectModels/backend_flask/prog_analizador/models/"
    "IMS2.windowed_ms2ae_autoencoder.keras"
)
DEFAULT_OUTPUT_DIR = Path("KAGGLE_BEARING_FDD/dense_windowed_audit_2026-07-03")


def find_ffp_after_healthy(
    hi: np.ndarray,
    threshold: float,
    healthy_samples: int,
    consecutive: int,
) -> dict[str, int | None]:
    """Return the first post-healthy run of `consecutive` points above threshold."""

    values = np.asarray(hi, dtype=float)
    for start in range(healthy_samples, len(values) - consecutive + 1):
        if np.all(values[start : start + consecutive] > threshold):
            run_length = 0
            for value in values[start:]:
                if value > threshold:
                    run_length += 1
                else:
                    break
            return {
                "ffp_zero_based": int(start),
                "ffp_one_based": int(start + 1),
                "run_length": int(run_length),
            }

    return {
        "ffp_zero_based": None,
        "ffp_one_based": None,
        "run_length": None,
    }


def compute_zero_output_baselines(
    samples: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Compute sample-level raw RMS^2 and window-P95 zero-output MSE."""

    sample_array = np.asarray(samples, dtype=np.float64)
    usable_dim = (sample_array.shape[1] // window_size) * window_size
    if usable_dim == 0:
        raise ValueError("window_size is larger than the sample dimension")

    trimmed = sample_array[:, :usable_dim]
    windows_per_sample = usable_dim // window_size
    windows = trimmed.reshape(sample_array.shape[0], windows_per_sample, window_size)
    zero_window_mse = np.mean(np.square(windows), axis=2)

    return {
        "raw_rms2": np.mean(np.square(trimmed), axis=1),
        "zero_window_p95": np.percentile(zero_window_mse, 95, axis=1),
        "zero_window_mean": np.mean(zero_window_mse, axis=1),
        "zero_window_mse": zero_window_mse,
    }


def reconstruction_improvement_percent(
    recon_mse: np.ndarray,
    zero_mse: np.ndarray,
) -> float:
    """Return global percentage improvement of reconstruction MSE over zero output."""

    recon_total = float(np.sum(np.asarray(recon_mse, dtype=np.float64)))
    zero_total = float(np.sum(np.asarray(zero_mse, dtype=np.float64)))
    if zero_total == 0:
        return float("nan")
    return 100.0 * (1.0 - recon_total / zero_total)


def longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same length")
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def load_samples_csv(data_path: Path) -> np.ndarray:
    frame = pd.read_csv(data_path, header=None, dtype=np.float32)
    return frame.to_numpy(dtype=np.float32, copy=False)


def predict_dense_window_mse(
    model_path: Path,
    samples: np.ndarray,
    window_size: int,
    batch_size: int,
    eval_chunk_samples: int,
) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    usable_dim = (samples.shape[1] // window_size) * window_size
    windows_per_sample = usable_dim // window_size
    window_mse_chunks: list[np.ndarray] = []

    for start in range(0, samples.shape[0], eval_chunk_samples):
        stop = min(start + eval_chunk_samples, samples.shape[0])
        chunk = samples[start:stop, :usable_dim]
        windows = chunk.reshape(-1, window_size)
        reconstructed = model.predict(windows, batch_size=batch_size, verbose=0)
        mse = np.mean(np.square(windows - reconstructed), axis=1)
        window_mse_chunks.append(mse.reshape(stop - start, windows_per_sample))

    return np.vstack(window_mse_chunks)


def summarize_hi(
    hi: np.ndarray,
    healthy_samples: int,
    consecutive: int,
) -> dict[str, Any]:
    threshold = float(np.percentile(hi[:healthy_samples], 95))
    ffp = find_ffp_after_healthy(hi, threshold, healthy_samples, consecutive)
    healthy_mask = np.asarray(hi[:healthy_samples]) > threshold
    post_mask = np.asarray(hi[healthy_samples:]) > threshold
    return {
        "threshold_p95_healthy": threshold,
        **ffp,
        "healthy_exceedance_count": int(np.sum(healthy_mask)),
        "healthy_longest_run": longest_true_run(healthy_mask),
        "post_healthy_exceedance_count": int(np.sum(post_mask)),
        "post_healthy_longest_run": longest_true_run(post_mask),
    }


def audit_dense_windowed_baseline(
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    window_size: int,
    healthy_samples: int,
    consecutive: int,
    batch_size: int,
    eval_chunk_samples: int,
    paper_ffp_one_based: int,
) -> dict[str, Any]:
    samples = load_samples_csv(data_path)
    usable_dim = (samples.shape[1] // window_size) * window_size
    windows_per_sample = usable_dim // window_size
    zero = compute_zero_output_baselines(samples, window_size)

    dense_window_mse = predict_dense_window_mse(
        model_path=model_path,
        samples=samples,
        window_size=window_size,
        batch_size=batch_size,
        eval_chunk_samples=eval_chunk_samples,
    )
    dense_hi = np.percentile(dense_window_mse, 95, axis=1)
    dense_window_mean = np.mean(dense_window_mse, axis=1)

    methods = {
        "dense_windowed_hi": dense_hi,
        "zero_window_p95": zero["zero_window_p95"],
        "raw_rms2": zero["raw_rms2"],
    }
    summaries = {
        name: summarize_hi(values, healthy_samples, consecutive)
        for name, values in methods.items()
    }

    zero_window_mse = zero["zero_window_mse"]
    improvement_all = reconstruction_improvement_percent(
        dense_window_mse.ravel(),
        zero_window_mse.ravel(),
    )
    improvement_healthy = reconstruction_improvement_percent(
        dense_window_mse[:healthy_samples].ravel(),
        zero_window_mse[:healthy_samples].ravel(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hi_table = pd.DataFrame(
        {
            "sample_zero_based": np.arange(samples.shape[0], dtype=int),
            "sample_one_based": np.arange(1, samples.shape[0] + 1, dtype=int),
            "dense_windowed_hi": dense_hi,
            "dense_window_mean_mse": dense_window_mean,
            "zero_window_p95": zero["zero_window_p95"],
            "zero_window_mean_mse": zero["zero_window_mean"],
            "raw_rms2": zero["raw_rms2"],
        }
    )
    hi_csv = output_dir / "ims2_hi_series.csv"
    hi_table.to_csv(hi_csv, index=False)

    summary: dict[str, Any] = {
        "dataset": "IMS-2",
        "data_path": str(data_path),
        "model_path": str(model_path),
        "model_type": "dense Table-1-inspired 2048-window autoencoder",
        "hi_definitions": {
            "dense_windowed_hi": "P95 across 2048-point window reconstruction MSE",
            "zero_window_p95": "P95 across 2048-point zero-output window MSE",
            "raw_rms2": "mean(x^2) over the usable sample dimension",
        },
        "indexing": {
            "zero_based": "CSV row index",
            "one_based": "paper-style sample label = zero_based + 1",
            "paper_ffp_one_based": int(paper_ffp_one_based),
        },
        "config": {
            "sample_count": int(samples.shape[0]),
            "sample_dim": int(samples.shape[1]),
            "usable_dim": int(usable_dim),
            "window_size": int(window_size),
            "windows_per_sample": int(windows_per_sample),
            "healthy_samples": int(healthy_samples),
            "threshold_rule": "empirical P95 of first healthy_samples HI values",
            "ffp_rule": f"first of {consecutive} consecutive HI values > threshold, searched after healthy region",
            "batch_size": int(batch_size),
            "eval_chunk_samples": int(eval_chunk_samples),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "method_summaries": summaries,
        "reconstruction_improvement_vs_zero_percent": {
            "all_windows": improvement_all,
            "healthy_windows": improvement_healthy,
        },
        "correlations": {
            "dense_hi_vs_zero_window_p95_all": pearson_corr(
                dense_hi, zero["zero_window_p95"]
            ),
            "dense_hi_vs_raw_rms2_all": pearson_corr(dense_hi, zero["raw_rms2"]),
            "zero_window_p95_vs_raw_rms2_all": pearson_corr(
                zero["zero_window_p95"], zero["raw_rms2"]
            ),
            "dense_mean_mse_vs_raw_rms2_all": pearson_corr(
                dense_window_mean, zero["raw_rms2"]
            ),
        },
        "outputs": {
            "hi_series_csv": str(hi_csv),
            "summary_json": str(output_dir / "ims2_dense_windowed_audit.json"),
            "readme": str(output_dir / "README.md"),
        },
    }

    summary_json = output_dir / "ims2_dense_windowed_audit.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(output_dir / "README.md", summary)
    return summary


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# IMS-2 dense_windowed_baseline audit",
        "",
        "This checkpoint compares the project dense windowed reconstruction HI "
        "against zero-output reconstruction and raw RMS² baselines.",
        "",
        "## Scope",
        "",
        "- Dataset: IMS-2.",
        "- Model: dense Table-1-inspired 2048-window autoencoder.",
        "- Threshold: empirical P95 on first healthy samples.",
        "- FFP: first of five consecutive samples above threshold after the healthy region.",
        "- Indexing: zero-based CSV row and one-based paper label are both reported.",
        "",
        "## Key results",
        "",
    ]
    for name, item in summary["method_summaries"].items():
        lines.append(
            f"- `{name}`: FFP zero-based={item['ffp_zero_based']}, "
            f"one-based={item['ffp_one_based']}, "
            f"threshold={item['threshold_p95_healthy']:.10g}, "
            f"healthy_longest_run={item['healthy_longest_run']}"
        )
    lines.extend(
        [
            "",
            "## Reconstruction improvement vs zero-output",
            "",
            f"- All windows: {summary['reconstruction_improvement_vs_zero_percent']['all_windows']:.6f}%.",
            f"- Healthy windows: {summary['reconstruction_improvement_vs_zero_percent']['healthy_windows']:.6f}%.",
            "",
            "## Correlations",
            "",
        ]
    )
    for name, value in summary["correlations"].items():
        lines.append(f"- `{name}`: {value:.12f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--healthy-samples", type=int, default=300)
    parser.add_argument("--consecutive", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-chunk-samples", type=int, default=40)
    parser.add_argument("--paper-ffp-one-based", type=int, default=536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_dense_windowed_baseline(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        healthy_samples=args.healthy_samples,
        consecutive=args.consecutive,
        batch_size=args.batch_size,
        eval_chunk_samples=args.eval_chunk_samples,
        paper_ffp_one_based=args.paper_ffp_one_based,
    )
    print(json.dumps(summary["method_summaries"], indent=2, ensure_ascii=False))
    print(
        json.dumps(
            summary["reconstruction_improvement_vs_zero_percent"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(summary["correlations"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
