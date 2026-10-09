"""Reproducible Kaggle pipeline for the BEARING-FDD research project.

Each run keeps one declared Health Index definition from training through
evaluation: scalar bottleneck HI for the regularized full-sample variant, or
P95 window-reconstruction HI for the windowed baseline.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.stats import kurtosis, skew


DATASETS = {
    "IMS1": {
        "file": "IMS1.csv",
        "healthy_file": "healthyIMS1.csv",
        "expected_dim": 20480,
        "fs": 20480.0,
        "shaft": 33.33,
        "BPFO": 236.0,
        "BPFI": 297.0,
        "BSF": 278.0,
        "FTF": 15.0,
        "filter_band": (1280.0, 2560.0),
        "paper_ffp": 1857,
        "representative_samples": [1857, 2138, 2155],
    },
    "IMS2": {
        "file": "IMS2.csv",
        "healthy_file": "healthyIMS2.csv",
        "expected_dim": 20480,
        "fs": 20480.0,
        "shaft": 33.33,
        "BPFO": 236.0,
        "BPFI": 297.0,
        "BSF": 278.0,
        "FTF": 15.0,
        "filter_band": (2560.0, 5120.0),
        "paper_ffp": 536,
        "representative_samples": [532, 871, 979],
    },
    "IMS3": {
        "file": "IMS3.csv",
        "healthy_file": "healthyIMS3.csv",
        "expected_dim": 20480,
        "fs": 20480.0,
        "shaft": 33.33,
        "BPFO": 236.0,
        "BPFI": 297.0,
        "BSF": 278.0,
        "FTF": 15.0,
        "filter_band": (2560.0, 5120.0),
        "paper_ffp": 5967,
        "representative_samples": [5967, 6178, 6319],
    },
    "XJTU2-1": {
        "file": "XJTU_SY_2_1.csv",
        "healthy_file": "healthyXJTU_SY_2_1.csv",
        "expected_dim": 32768,
        "fs": 25600.0,
        "shaft": 37.5,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "filter_band": (1600.0, 3200.0),
        "paper_ffp": 451,
        "representative_samples": [451, 460, 486],
    },
    "XJTU2-3": {
        "file": "XJTU_SY_2_3.csv",
        "healthy_file": "healthyXJTU_SY_2_3.csv",
        "expected_dim": 32768,
        "fs": 25600.0,
        "shaft": 37.5,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "filter_band": (8533.33, 10666.67),
        "paper_ffp": 301,
        "representative_samples": [301, 419, 532],
    },
    "XJTU3-1": {
        "file": "XJTU_SY_3_1.csv",
        "healthy_file": "healthyXJTU_SY_3_1.csv",
        "expected_dim": 32768,
        "fs": 25600.0,
        "shaft": 40.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "filter_band": (6400.0, 8533.33),
        "paper_ffp": 2347,
        "representative_samples": [2347, 2445, 2533],
    },
    "XJTU3-4": {
        "file": "XJTU_SY_3_4.csv",
        "healthy_file": "healthyXJTU_SY_3_4.csv",
        "expected_dim": 32768,
        "fs": 25600.0,
        "shaft": 40.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "filter_band": (3200.0, 6400.0),
        "paper_ffp": 1416,
        "representative_samples": [1416, 1453, 1505],
    },
}


@dataclass
class RunConfig:
    input_root: str = "/kaggle/input"
    output_root: str = "/kaggle/working/bearing_fdd_results"
    datasets: tuple[str, ...] = tuple(DATASETS)
    healthy_samples: int = 300
    window_size: int = 2048
    epochs: int = 5
    batch_size: int = 64
    eval_chunk: int = 40
    consecutive: int = 5
    random_seed: int = 42
    model_variant: str = "regularized_full_sample"
    latent_dim: int = 1
    lambda_monotonic: float = 0.01
    lambda_smoothing: float = 0.01
    force_retrain: bool = True
    run_xai: bool = True
    xai_scope: str = "all"
    run_diagnosis: bool = True
    harmonic_tolerance_hz: float = 8.0
    max_harmonic: int = 6


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def cleanup_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def locate_unique_file(root: Path, filename: str, required: bool = True) -> Path | None:
    """Find a CSV below the configured input root.

    On Kaggle, ``root`` is normally ``/kaggle/input``. Kaggle mounts every
    attached dataset in a child folder such as
    ``/kaggle/input/bearing-fdd-data/``; callers should not put the web URL in
    ``input_root``.
    """
    matches = sorted(root.rglob(filename))
    if not matches:
        if required:
            available = sorted(path.name for path in root.rglob("*.csv"))[:30]
            available_text = ", ".join(available) if available else "(no CSV files found)"
            raise FileNotFoundError(
                f"Cannot find {filename!r} under {root}. "
                "On Kaggle, attach the data with Add Input and keep "
                "input_root='/kaggle/input'. "
                f"Available CSV files: {available_text}"
            )
        return None
    if len(matches) > 1:
        print(f"[WARN] Multiple {filename} files found; using {matches[0]}")
    return matches[0]


def discover_data(config: RunConfig) -> dict[str, dict[str, Path | None]]:
    root = Path(config.input_root)
    found = {}
    for name in config.datasets:
        meta = DATASETS[name]
        found[name] = {
            "data": locate_unique_file(root, meta["file"]),
            "healthy": locate_unique_file(root, meta["healthy_file"], required=False),
        }
    return found


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def read_rows(path: Path, start: int, count: int, expected_dim: int | None = None) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if index >= start + count:
                break
            if not line.strip():
                continue
            values = np.fromstring(line, sep=",", dtype=np.float32)
            if expected_dim is not None and values.size != expected_dim:
                raise ValueError(
                    f"{path.name} row {index}: expected {expected_dim} values, got {values.size}"
                )
            rows.append(values)
    if not rows:
        return np.empty((0, expected_dim or 0), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def validate_dataset(path: Path, meta: dict) -> dict:
    first = read_rows(path, 0, 1)
    actual_dim = int(first.shape[1])
    if actual_dim != meta["expected_dim"]:
        raise ValueError(
            f"{path.name}: expected dimension {meta['expected_dim']}, got {actual_dim}"
        )
    return {
        "path": str(path),
        "rows": count_rows(path),
        "dimension": actual_dim,
        "fs": meta["fs"],
        "duration_seconds": actual_dim / meta["fs"],
    }


def to_windows(samples: np.ndarray, window_size: int) -> tuple[np.ndarray, int, int]:
    samples = np.asarray(samples, dtype=np.float32)
    usable_dim = (samples.shape[1] // window_size) * window_size
    if usable_dim <= 0:
        raise ValueError(f"window_size={window_size} exceeds input dimension={samples.shape[1]}")
    windows_per_sample = usable_dim // window_size
    windows = samples[:, :usable_dim].reshape(-1, window_size)
    return windows, windows_per_sample, usable_dim


def build_ms2ae(input_dim: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Dense Table-1 baseline without explicit sequence regularization."""
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="input_signal")
    x = tf.keras.layers.Dense(3500, activation="relu", name="encoder_3500")(inputs)
    x = tf.keras.layers.Dense(700, activation="relu", name="encoder_700")(x)
    x = tf.keras.layers.Dense(200, activation="relu", name="encoder_200")(x)
    hi = tf.keras.layers.Dense(1, activation="sigmoid", name="hi_output")(x)
    x = tf.keras.layers.Dense(200, activation="relu", name="decoder_200")(hi)
    x = tf.keras.layers.Dense(700, activation="relu", name="decoder_700")(x)
    x = tf.keras.layers.Dense(3500, activation="relu", name="decoder_3500")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = tf.keras.Model(inputs, outputs, name="MS2AE_Table1")
    encoder = tf.keras.Model(inputs, hi, name="MS2AE_Table1_Encoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return autoencoder, encoder


def build_regularized_ms2ae(
    input_dim: int,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build the full-sample Table-1 architecture used by the regularized trainer.

    The paper names the 3500-node and 200-node layers "monotonicity" and
    "smoothing", but does not publish their mathematical operations. This
    project therefore exposes those representations and applies explicit,
    auditable sequence losses during training instead of relying on layer names.

    Returns:
        training_backbone:
            Outputs reconstruction, 3500-D monotonic features, 200-D smoothed
            features, and scalar HI.
        autoencoder:
            Standard reconstruction model that can be saved without custom
            objects.
        encoder:
            Standard scalar-HI model used during evaluation.
    """
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="input_signal")
    monotonic_features = tf.keras.layers.Dense(
        3500,
        activation="relu",
        name="monotonic_features_3500",
    )(inputs)
    x = tf.keras.layers.Dense(700, activation="relu", name="encoder_700")(
        monotonic_features
    )
    smoothed_features = tf.keras.layers.Dense(
        200,
        activation="relu",
        name="smoothed_features_200",
    )(x)
    hi = tf.keras.layers.Dense(1, activation="sigmoid", name="hi_output")(
        smoothed_features
    )

    x = tf.keras.layers.Dense(200, activation="relu", name="decoder_200")(hi)
    x = tf.keras.layers.Dense(700, activation="relu", name="decoder_700")(x)
    x = tf.keras.layers.Dense(3500, activation="relu", name="decoder_3500")(x)
    reconstruction = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(x)

    backbone = tf.keras.Model(
        inputs,
        [reconstruction, monotonic_features, smoothed_features, hi],
        name="MS2AE_Regularized_Training_Backbone",
    )
    autoencoder = tf.keras.Model(
        inputs,
        reconstruction,
        name="MS2AE_Regularized_Autoencoder",
    )
    encoder = tf.keras.Model(inputs, hi, name="MS2AE_Regularized_Encoder")
    return backbone, autoencoder, encoder


def build_multilatent_autoencoder(
    input_dim: int,
    latent_dim: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a full-sample reconstruction AE with a multi-dimensional bottleneck.

    This is a project experiment rather than the paper's one-node Table-1
    architecture. The scalar sample HI is computed later from reconstruction
    MSE; no scalar latent head is asked to reconstruct the raw waveform.
    """
    if latent_dim < 1:
        raise ValueError("latent_dim must be at least 1")
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="input_signal")
    x = tf.keras.layers.Dense(3500, activation="relu", name="encoder_3500")(inputs)
    x = tf.keras.layers.Dense(700, activation="relu", name="encoder_700")(x)
    x = tf.keras.layers.Dense(200, activation="relu", name="encoder_200")(x)
    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="linear",
        name=f"latent_{latent_dim}",
    )(x)
    x = tf.keras.layers.Dense(200, activation="relu", name="decoder_200")(latent)
    x = tf.keras.layers.Dense(700, activation="relu", name="decoder_700")(x)
    x = tf.keras.layers.Dense(3500, activation="relu", name="decoder_3500")(x)
    reconstruction = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(x)
    autoencoder = tf.keras.Model(
        inputs,
        reconstruction,
        name=f"MultiLatent_Autoencoder_{latent_dim}",
    )
    encoder = tf.keras.Model(
        inputs,
        latent,
        name=f"MultiLatent_Encoder_{latent_dim}",
    )
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return autoencoder, encoder


def estimate_ms2ae_parameters(input_dim: int, bottleneck_dim: int = 1) -> int:
    """Return the exact trainable-parameter count of the symmetric dense model."""
    layer_sizes = [
        input_dim,
        3500,
        700,
        200,
        bottleneck_dim,
        200,
        700,
        3500,
        input_dim,
    ]
    return int(
        sum(
            layer_sizes[index] * layer_sizes[index + 1]
            + layer_sizes[index + 1]
            for index in range(len(layer_sizes) - 1)
        )
    )


def monotonic_increasing_loss(sequence: tf.Tensor) -> tf.Tensor:
    """Penalize decreases between consecutive time-ordered samples.

    L_mon = mean(ReLU(z_t - z_(t+1))). A zero value means that every represented
    feature is non-decreasing inside the current chronological batch.
    """
    sequence = tf.convert_to_tensor(sequence)
    delta = sequence[1:] - sequence[:-1]
    return tf.reduce_mean(tf.nn.relu(-delta))


def second_difference_smoothing_loss(sequence: tf.Tensor) -> tf.Tensor:
    """Penalize curvature while preserving a possible linear degradation trend.

    L_smooth = mean((z_(t+1) - 2*z_t + z_(t-1))^2).
    """
    sequence = tf.convert_to_tensor(sequence)
    second_difference = sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2]
    return tf.reduce_mean(tf.square(second_difference))


def regularized_loss_components(
    targets: tf.Tensor,
    reconstruction: tf.Tensor,
    monotonic_features: tf.Tensor,
    smoothed_features: tf.Tensor,
    hi: tf.Tensor,
    lambda_monotonic: float,
    lambda_smoothing: float,
) -> dict[str, tf.Tensor]:
    """Compute the transparent MS2AE objective used by this project.

    Total loss:
        MSE(x, x_hat)
        + lambda_monotonic * (L_mon(features_3500) + L_mon(HI))
        + lambda_smoothing * (L_smooth(features_200) + L_smooth(HI))

    Batches must contain consecutive samples in chronological order and must
    have at least three elements.
    """
    reconstruction_loss = tf.reduce_mean(tf.square(targets - reconstruction))
    monotonic_features_loss = monotonic_increasing_loss(monotonic_features)
    monotonic_hi_loss = monotonic_increasing_loss(hi)
    smoothing_features_loss = second_difference_smoothing_loss(smoothed_features)
    smoothing_hi_loss = second_difference_smoothing_loss(hi)
    monotonic_loss = monotonic_features_loss + monotonic_hi_loss
    smoothing_loss = smoothing_features_loss + smoothing_hi_loss
    total_loss = (
        reconstruction_loss
        + lambda_monotonic * monotonic_loss
        + lambda_smoothing * smoothing_loss
    )
    return {
        "loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "monotonic_loss": monotonic_loss,
        "monotonic_features_loss": monotonic_features_loss,
        "monotonic_hi_loss": monotonic_hi_loss,
        "smoothing_loss": smoothing_loss,
        "smoothing_features_loss": smoothing_features_loss,
        "smoothing_hi_loss": smoothing_hi_loss,
    }


def chronological_batch_starts(
    sample_count: int,
    batch_size: int,
    overlap: int = 2,
) -> list[int]:
    """Return starts for ordered batches with boundary coverage.

    Two overlapping samples ensure that first and second differences crossing a
    batch boundary are included in at least one optimization step.
    """
    if batch_size < 3:
        raise ValueError("Regularized MS2AE requires batch_size >= 3")
    stride = batch_size - overlap
    starts = list(range(0, sample_count, stride))
    if starts and sample_count - starts[-1] < 3:
        starts[-1] = max(0, sample_count - batch_size)
    return sorted(set(starts))


def train_regularized_ms2ae(
    healthy_samples: np.ndarray,
    input_dim: int,
    epochs: int,
    batch_size: int,
    lambda_monotonic: float,
    lambda_smoothing: float,
) -> tuple[tf.keras.Model, tf.keras.Model, dict[str, list[float]]]:
    """Train on chronological full samples using the explicit MS2AE objective."""
    backbone, autoencoder, encoder = build_regularized_ms2ae(input_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    history_names = (
        "loss",
        "reconstruction_loss",
        "monotonic_loss",
        "monotonic_features_loss",
        "monotonic_hi_loss",
        "smoothing_loss",
        "smoothing_features_loss",
        "smoothing_hi_loss",
    )
    history = {name: [] for name in history_names}
    starts = chronological_batch_starts(len(healthy_samples), batch_size)

    # Do not shuffle: differences are meaningful only for adjacent timestamps.
    for epoch in range(epochs):
        epoch_values = {name: [] for name in history_names}
        for start in starts:
            batch = tf.convert_to_tensor(
                healthy_samples[start : start + batch_size],
                dtype=tf.float32,
            )
            if int(batch.shape[0]) < 3:
                continue
            with tf.GradientTape() as tape:
                reconstruction, monotonic_features, smoothed_features, hi = backbone(
                    batch,
                    training=True,
                )
                losses = regularized_loss_components(
                    batch,
                    reconstruction,
                    monotonic_features,
                    smoothed_features,
                    hi,
                    lambda_monotonic,
                    lambda_smoothing,
                )
            gradients = tape.gradient(losses["loss"], backbone.trainable_variables)
            optimizer.apply_gradients(zip(gradients, backbone.trainable_variables))
            for name in history_names:
                epoch_values[name].append(float(losses[name].numpy()))

        for name in history_names:
            history[name].append(float(np.mean(epoch_values[name])))
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss={history['loss'][-1]:.6g} - "
            f"recon={history['reconstruction_loss'][-1]:.6g} - "
            f"mon={history['monotonic_loss'][-1]:.6g} - "
            f"smooth={history['smoothing_loss'][-1]:.6g}"
        )
    return autoencoder, encoder, history


def reconstruction_hi(
    autoencoder: tf.keras.Model,
    samples: np.ndarray,
    window_size: int,
    batch_size: int,
) -> tuple[np.ndarray, int, int]:
    windows, windows_per_sample, usable_dim = to_windows(samples, window_size)
    reconstructed = autoencoder.predict(windows, batch_size=batch_size, verbose=0)
    window_mse = np.mean((windows - reconstructed) ** 2, axis=1)
    sample_hi = np.percentile(window_mse.reshape(-1, windows_per_sample), 95, axis=1)
    return sample_hi.astype(np.float64), windows_per_sample, usable_dim


def full_sample_reconstruction_hi(
    autoencoder: tf.keras.Model,
    samples: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return one full-sample reconstruction MSE value per acquisition."""
    samples = np.asarray(samples, dtype=np.float32)
    reconstructed = autoencoder.predict(
        samples,
        batch_size=batch_size,
        verbose=0,
    )
    return np.mean((samples - reconstructed) ** 2, axis=1).astype(np.float64)


def latent_hi(
    encoder: tf.keras.Model,
    samples: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict the scalar sigmoid HI emitted by the one-node bottleneck."""
    return np.asarray(
        encoder.predict(samples, batch_size=batch_size, verbose=0)
    ).reshape(-1).astype(np.float64)


def find_ffp(hi: Iterable[float], threshold: float, consecutive: int = 5) -> int | None:
    run = 0
    for index, value in enumerate(hi):
        run = run + 1 if value > threshold else 0
        if run >= consecutive:
            return index - consecutive + 1
    return None


def paper_stage_thresholds(healthy_hi: np.ndarray) -> dict[str, float]:
    """Return the equations exactly as printed in the paper.

    These values are separate from the P95 fault-detection threshold. The
    equations should be reported explicitly because they do not add P95 back
    to the MD/LD offsets.
    """
    p95 = float(np.percentile(healthy_hi, 95))
    delta = float(np.max(healthy_hi) - p95)
    return {"P95": p95, "MD": delta * 50.0, "LD": delta * 100.0}


def time_features(samples: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-12
    absolute = np.abs(samples)
    rms = np.sqrt(np.mean(samples**2, axis=1))
    peak = np.max(absolute, axis=1)
    mean_abs = np.mean(absolute, axis=1)
    sqrt_abs_mean = np.mean(np.sqrt(absolute + eps), axis=1)
    return {
        "RMS": rms,
        "Sk": skew(samples, axis=1, nan_policy="omit"),
        "K": kurtosis(samples, axis=1, fisher=False, nan_policy="omit"),
        "CF": peak / (rms + eps),
        "SF": rms / (mean_abs + eps),
        "IF": peak / (mean_abs + eps),
        "MF": peak / (sqrt_abs_mean**2 + eps),
    }


def power_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    spectrum = np.abs(np.fft.rfft(np.hanning(signal.size) * signal, norm="forward")) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    spectrum[:5] = 0.0
    return spectrum, frequencies


def bandpass_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    low = max(band[0] / nyquist, 0.001)
    high = min(band[1] / nyquist, 0.99)
    if low >= high:
        raise ValueError(f"Invalid band {band} for fs={fs}")
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return power_spectrum(envelope, fs)


def max_near(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    target_hz: float,
    tolerance_hz: float = 10.0,
) -> float:
    mask = np.abs(frequencies - target_hz) <= tolerance_hz
    return float(np.max(spectrum[mask])) if np.any(mask) else 0.0


def generate_kurtogram_bands(fs: float) -> list[dict]:
    nyquist = fs / 2.0
    levels = [(0.0, 1), (1.0, 2), (1.6, 3), (2.0, 4), (2.6, 6), (3.0, 8)]
    bands = []
    for level, parts in levels:
        width = nyquist / parts
        for index in range(parts):
            bands.append(
                {
                    "level": level,
                    "index": index,
                    "parts": parts,
                    "low": index * width,
                    "high": (index + 1) * width,
                }
            )
    return bands


def select_kurtogram_band(signal: np.ndarray, fs: float) -> tuple[tuple[float, float], pd.DataFrame]:
    records = []
    nyquist = fs / 2.0
    for item in generate_kurtogram_bands(fs):
        low = max(item["low"] / nyquist, 0.001)
        high = min(item["high"] / nyquist, 0.99)
        score = np.nan
        try:
            b, a = butter(4, [low, high], btype="band")
            envelope = np.abs(hilbert(filtfilt(b, a, signal)))
            score = float(kurtosis(envelope, fisher=True, nan_policy="omit"))
        except Exception:
            pass
        eligible = (
            item["level"] in (2.0, 2.6, 3.0)
            and item["index"] not in (0, item["parts"] - 1)
            and np.isfinite(score)
        )
        records.append({**item, "kurtosis": score, "eligible": eligible})
    table = pd.DataFrame(records)
    candidates = table[table["eligible"]]
    if candidates.empty:
        raise RuntimeError("No valid Kurtogram band")
    best = candidates.loc[candidates["kurtosis"].idxmax()]
    return (float(best["low"]), float(best["high"])), table


def load_baseline(
    data_path: Path,
    healthy_path: Path | None,
    expected_dim: int,
) -> np.ndarray:
    if healthy_path is not None:
        baseline = read_rows(healthy_path, 0, 1)
        if baseline.size and baseline.shape[1] == expected_dim:
            return baseline[0]
    return np.mean(read_rows(data_path, 0, 20, expected_dim), axis=0)


def isolate_fault(
    sample: np.ndarray,
    baseline: np.ndarray,
    use_dtw: bool = True,
) -> tuple[np.ndarray, str]:
    if not use_dtw:
        return sample - baseline, "direct_difference"
    try:
        from fastdtw import fastdtw

        _, path = fastdtw(baseline, sample, dist=lambda x, y: abs(x - y))
        aligned = np.zeros(sample.size, dtype=np.float64)
        counts = np.zeros(sample.size, dtype=np.float64)
        for base_index, sample_index in path:
            if base_index < baseline.size and sample_index < sample.size:
                aligned[sample_index] += baseline[base_index]
                counts[sample_index] += 1
        missing = counts == 0
        counts[missing] = 1
        aligned /= counts
        aligned[missing] = baseline[missing]
        return sample - aligned, "fastdtw_difference"
    except ImportError:
        return sample - baseline, "direct_difference_fastdtw_unavailable"


def match_harmonics(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    meta: dict,
    tolerance_hz: float,
    max_harmonic: int,
) -> dict[str, list[int]]:
    peaks, _ = find_peaks(spectrum)
    if peaks.size == 0:
        return {fault: [] for fault in ("BPFO", "BPFI", "BSF", "FTF")}
    top = peaks[np.argsort(spectrum[peaks])[-max(10, max_harmonic * 4) :]]
    peak_frequencies = frequencies[top]
    matches = {}
    for fault in ("BPFO", "BPFI", "BSF", "FTF"):
        base = meta[fault]
        matches[fault] = [
            harmonic
            for harmonic in range(1, max_harmonic + 1)
            if np.any(np.abs(peak_frequencies - harmonic * base) <= tolerance_hz)
        ]
    return matches


class OnlineCorrelation:
    def __init__(self) -> None:
        self.names: list[str] | None = None
        self.n = 0
        self.total: np.ndarray | None = None
        self.cross: np.ndarray | None = None

    def update(self, hi: np.ndarray, features: dict[str, np.ndarray]) -> None:
        names = ["HI", *features]
        matrix = np.column_stack([hi, *[features[name] for name in features]]).astype(float)
        if self.names is None:
            self.names = names
            self.total = np.zeros(matrix.shape[1], dtype=float)
            self.cross = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
        if names != self.names:
            raise ValueError("Feature order changed")
        self.n += matrix.shape[0]
        self.total += np.sum(matrix, axis=0)
        self.cross += matrix.T @ matrix

    def matrix(self) -> tuple[list[str], np.ndarray]:
        if self.n == 0 or self.names is None:
            return [], np.empty((0, 0))
        mean = self.total / self.n
        covariance = self.cross / self.n - np.outer(mean, mean)
        standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = covariance / np.outer(standard, standard)
        correlation = np.nan_to_num(correlation)
        np.fill_diagonal(correlation, 1.0)
        return self.names, correlation


def frequency_features(
    samples: np.ndarray,
    meta: dict,
    filter_band: tuple[float, float],
    tolerance_hz: float,
) -> dict[str, np.ndarray]:
    names = [
        "Fund nf", "BPFO nf", "BPFI nf", "BSF nf", "FTF nf",
        "Fund f", "BPFO f", "BPFI f", "BSF f", "FTF f",
    ]
    output = {name: [] for name in names}
    target_map = {
        "Fund": meta["shaft"],
        "BPFO": meta["BPFO"],
        "BPFI": meta["BPFI"],
        "BSF": meta["BSF"],
        "FTF": meta["FTF"],
    }
    for sample in samples:
        raw_spectrum, raw_freq = power_spectrum(sample, meta["fs"])
        filtered_spectrum, filtered_freq = bandpass_envelope_spectrum(
            sample, meta["fs"], filter_band
        )
        for feature, target in target_map.items():
            output[f"{feature} nf"].append(
                max_near(raw_spectrum, raw_freq, target, tolerance_hz)
            )
            output[f"{feature} f"].append(
                max_near(filtered_spectrum, filtered_freq, target, tolerance_hz)
            )
    return {name: np.asarray(values) for name, values in output.items()}


def plot_loss(history: dict, destination: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    epochs = range(1, len(history["loss"]) + 1)
    keys = ["loss"]
    if "reconstruction_loss" in history:
        keys.extend(["reconstruction_loss", "monotonic_loss", "smoothing_loss"])
    for key in keys:
        ax.plot(epochs, history[key], marker="o", label=key)
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{dataset} MS2AE training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def plot_hi(
    hi: np.ndarray,
    threshold: float,
    ffp: int | None,
    destination: Path,
    dataset: str,
    hi_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.plot(hi, color="#2867B2", linewidth=1.0, label="Sample HI")
    ax.axhline(threshold, color="#C43C39", linestyle="--", label=f"P95={threshold:.6g}")
    if ffp is not None:
        ax.axvline(ffp, color="#D66A00", linestyle="-.", label=f"FFP=#{ffp}")
    ax.set(xlabel="Sample index", ylabel=hi_label, title=dataset)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def plot_correlation(names: list[str], matrix: np.ndarray, destination: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    sns.heatmap(
        np.abs(matrix),
        xticklabels=names,
        yticklabels=names,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        square=True,
        ax=ax,
    )
    ax.set_title(f"{dataset}: absolute Pearson correlation")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def train_dataset(
    dataset: str,
    paths: dict[str, Path | None],
    config: RunConfig,
) -> tuple[Path, dict]:
    meta = DATASETS[dataset]
    output_dir = Path(config.output_root)
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    figure_dir = output_dir / "figures"
    for path in (model_dir, log_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)
    if config.model_variant == "regularized_full_sample":
        model_stem = f"{dataset}.regularized_full_ms2ae"
    elif config.model_variant == "multilatent_full_sample":
        model_stem = f"{dataset}.multilatent_{config.latent_dim}"
    elif config.model_variant == "dense_windowed_baseline":
        model_stem = f"{dataset}.windowed_ms2ae"
    else:
        raise ValueError(
            "model_variant must be 'regularized_full_sample', "
            "'multilatent_full_sample', or 'dense_windowed_baseline'"
        )
    model_path = model_dir / f"{model_stem}_autoencoder.keras"
    encoder_path = model_dir / f"{model_stem}_encoder.keras"
    log_path = log_dir / f"{dataset}.training.json"
    if model_path.exists() and not config.force_retrain:
        return model_path, json.loads(log_path.read_text(encoding="utf-8"))

    cleanup_memory()
    set_reproducibility(config.random_seed)
    healthy = read_rows(
        paths["data"], 0, config.healthy_samples, meta["expected_dim"]
    )
    started = time.time()
    if config.model_variant == "regularized_full_sample":
        autoencoder, encoder, history_values = train_regularized_ms2ae(
            healthy_samples=healthy,
            input_dim=meta["expected_dim"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            lambda_monotonic=config.lambda_monotonic,
            lambda_smoothing=config.lambda_smoothing,
        )
        windows_per_sample = 1
        usable_dim = meta["expected_dim"]
        architecture_input = meta["expected_dim"]
        hi_definition = "scalar sigmoid bottleneck with sequence regularization"
        shuffle = False
        print(
            f"[MODEL] regularized full-sample params={autoencoder.count_params():,}"
        )
    elif config.model_variant == "multilatent_full_sample":
        autoencoder, encoder = build_multilatent_autoencoder(
            meta["expected_dim"],
            config.latent_dim,
        )
        history_object = autoencoder.fit(
            healthy,
            healthy,
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=True,
            verbose=2,
        )
        history_values = {
            key: [float(value) for value in values]
            for key, values in history_object.history.items()
        }
        windows_per_sample = 1
        usable_dim = meta["expected_dim"]
        architecture_input = meta["expected_dim"]
        architecture_bottleneck = config.latent_dim
        hi_definition = (
            "full-sample reconstruction MSE from "
            f"{config.latent_dim}-dimensional bottleneck"
        )
        shuffle = True
        print(
            "[MODEL] multi-latent full-sample "
            f"k={config.latent_dim} params={autoencoder.count_params():,}"
        )
    else:
        windows, windows_per_sample, usable_dim = to_windows(
            healthy, config.window_size
        )
        autoencoder, encoder = build_ms2ae(config.window_size)
        history_object = autoencoder.fit(
            windows,
            windows,
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=True,
            verbose=2,
        )
        history_values = {
            key: [float(value) for value in values]
            for key, values in history_object.history.items()
        }
        architecture_input = config.window_size
        architecture_bottleneck = 1
        hi_definition = "P95 of window reconstruction MSE"
        shuffle = True
        print(f"[MODEL] windowed baseline params={autoencoder.count_params():,}")

    if config.model_variant == "regularized_full_sample":
        architecture_bottleneck = 1

    autoencoder.save(model_path)
    encoder.save(encoder_path)
    log = {
        "dataset": dataset,
        "model_variant": config.model_variant,
        "architecture": (
            f"IS({architecture_input})-3500-700-200-{architecture_bottleneck}"
            f"-200-700-3500-IS({architecture_input})"
        ),
        "hi_definition": hi_definition,
        "latent_dim": architecture_bottleneck,
        "regularization_equation": (
            "MSE + lambda_monotonic*(Lmon(features3500)+Lmon(HI)) "
            "+ lambda_smoothing*(L2diff(features200)+L2diff(HI))"
            if config.model_variant == "regularized_full_sample"
            else None
        ),
        "lambda_monotonic": config.lambda_monotonic,
        "lambda_smoothing": config.lambda_smoothing,
        "chronological_batches": not shuffle,
        "shuffle": shuffle,
        "healthy_samples": config.healthy_samples,
        "raw_dimension": meta["expected_dim"],
        "usable_dimension": usable_dim,
        "windows_per_sample": windows_per_sample,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "autoencoder_params": int(autoencoder.count_params()),
        "encoder_params": int(encoder.count_params()),
        "history": history_values,
        "loss": history_values["loss"],
        "elapsed_seconds": round(time.time() - started, 2),
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    plot_loss(history_values, figure_dir / f"{dataset}_loss.png", dataset)
    del autoencoder, encoder, healthy
    if config.model_variant == "multilatent_full_sample":
        del history_object
    elif config.model_variant == "dense_windowed_baseline":
        del history_object, windows
    cleanup_memory()
    return model_path, log


def evaluate_dataset(
    dataset: str,
    paths: dict[str, Path | None],
    model_path: Path,
    config: RunConfig,
) -> dict:
    meta = DATASETS[dataset]
    output_dir = Path(config.output_root)
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    diagnosis_dir = output_dir / "diagnosis" / dataset
    for path in (result_dir, figure_dir, diagnosis_dir):
        path.mkdir(parents=True, exist_ok=True)

    cleanup_memory()
    total = count_rows(paths["data"])
    healthy = read_rows(paths["data"], 0, config.healthy_samples, meta["expected_dim"])
    encoder_path = Path(str(model_path).replace("_autoencoder.keras", "_encoder.keras"))
    if config.model_variant == "regularized_full_sample":
        analysis_model = tf.keras.models.load_model(encoder_path, compile=False)
        healthy_hi = latent_hi(analysis_model, healthy, config.batch_size)
        windows_per_sample = 1
        usable_dim = meta["expected_dim"]
        hi_definition = "scalar sigmoid bottleneck with sequence regularization"
        hi_axis_label = "Latent HI"
    elif config.model_variant == "multilatent_full_sample":
        analysis_model = tf.keras.models.load_model(model_path, compile=False)
        healthy_hi = full_sample_reconstruction_hi(
            analysis_model,
            healthy,
            config.batch_size,
        )
        windows_per_sample = 1
        usable_dim = meta["expected_dim"]
        hi_definition = (
            "full-sample reconstruction MSE from "
            f"{config.latent_dim}-dimensional bottleneck"
        )
        hi_axis_label = "HI (full-sample reconstruction MSE)"
    else:
        analysis_model = tf.keras.models.load_model(model_path, compile=False)
        healthy_hi, windows_per_sample, usable_dim = reconstruction_hi(
            analysis_model, healthy, config.window_size, config.batch_size
        )
        hi_definition = "P95 of window reconstruction MSE"
        hi_axis_label = "HI (reconstruction P95)"
    threshold = float(np.percentile(healthy_hi, 95))
    hi_values = np.empty(total, dtype=np.float64)

    for start in range(0, total, config.eval_chunk):
        size = min(config.eval_chunk, total - start)
        batch = read_rows(paths["data"], start, size, meta["expected_dim"])
        if config.model_variant == "regularized_full_sample":
            hi_values[start : start + size] = latent_hi(
                analysis_model, batch, config.batch_size
            )
        elif config.model_variant == "multilatent_full_sample":
            hi_values[start : start + size] = full_sample_reconstruction_hi(
                analysis_model,
                batch,
                config.batch_size,
            )
        else:
            hi_values[start : start + size], _, _ = reconstruction_hi(
                analysis_model, batch, config.window_size, config.batch_size
            )
        del batch
        gc.collect()

    relative_ffp = find_ffp(
        hi_values[config.healthy_samples :], threshold, config.consecutive
    )
    ffp = None if relative_ffp is None else relative_ffp + config.healthy_samples
    stage_thresholds = paper_stage_thresholds(healthy_hi)

    np.savetxt(
        result_dir / f"{dataset}_hi.csv",
        np.column_stack([np.arange(total), hi_values]),
        delimiter=",",
        header="sample_index,hi",
        comments="",
    )
    plot_hi(
        hi_values,
        threshold,
        ffp,
        figure_dir / f"{dataset}_hi.png",
        dataset,
        hi_axis_label,
    )

    hi_delta = np.diff(hi_values)
    hi_second_difference = np.diff(hi_values, n=2)
    healthy_slice = hi_values[: config.healthy_samples]
    unlabeled_slice = hi_values[config.healthy_samples :]
    monotonicity_audit = {
        "decreasing_steps": int(np.sum(hi_delta < 0)),
        "decreasing_step_ratio": float(np.mean(hi_delta < 0)),
        "mean_decrease_magnitude": float(
            np.mean(np.maximum(-hi_delta, 0.0))
        ),
        "second_difference_rmse": float(
            np.sqrt(np.mean(hi_second_difference**2))
        ),
        "note": (
            "Soft regularization reduces violations but does not mathematically "
            "guarantee zero violations on unseen samples."
        ),
    }
    hi_statistics = {
        "all": {
            "minimum": float(np.min(hi_values)),
            "maximum": float(np.max(hi_values)),
            "mean": float(np.mean(hi_values)),
            "standard_deviation": float(np.std(hi_values)),
        },
        "healthy": {
            "minimum": float(np.min(healthy_slice)),
            "maximum": float(np.max(healthy_slice)),
            "mean": float(np.mean(healthy_slice)),
            "standard_deviation": float(np.std(healthy_slice)),
            "p95": float(np.percentile(healthy_slice, 95)),
        },
        "unlabeled": {
            "minimum": float(np.min(unlabeled_slice)),
            "maximum": float(np.max(unlabeled_slice)),
            "mean": float(np.mean(unlabeled_slice)),
            "standard_deviation": float(np.std(unlabeled_slice)),
        },
    }

    xai = {}
    if config.run_xai:
        xai_start = ffp if config.xai_scope == "faulty" and ffp is not None else 0
        correlation = OnlineCorrelation()
        default_band = meta["filter_band"]
        for start in range(xai_start, total, config.eval_chunk):
            size = min(config.eval_chunk, total - start)
            batch = read_rows(paths["data"], start, size, meta["expected_dim"])
            features = time_features(batch)
            features.update(
                frequency_features(
                    batch,
                    meta,
                    default_band,
                    config.harmonic_tolerance_hz,
                )
            )
            correlation.update(hi_values[start : start + size], features)
            del batch, features
            gc.collect()
        names, matrix = correlation.matrix()
        top3 = sorted(
            (
                {"feature": names[index], "correlation": float(matrix[0, index])}
                for index in range(1, len(names))
            ),
            key=lambda item: abs(item["correlation"]),
            reverse=True,
        )[:3]
        xai = {
            "scope": config.xai_scope,
            "start_index": xai_start,
            "features": names,
            "matrix": matrix.tolist(),
            "top3": top3,
            "default_filter_band": list(default_band),
        }
        plot_correlation(
            names,
            matrix,
            figure_dir / f"{dataset}_correlation.png",
            dataset,
        )

    diagnoses = []
    if config.run_diagnosis:
        baseline = load_baseline(paths["data"], paths["healthy"], meta["expected_dim"])
        for sample_index in meta["representative_samples"]:
            sample = read_rows(
                paths["data"], sample_index, 1, meta["expected_dim"]
            )[0]
            isolated, isolation_method = isolate_fault(sample, baseline, use_dtw=True)
            selected_band, kurtogram = select_kurtogram_band(isolated, meta["fs"])
            spectrum, frequencies = bandpass_envelope_spectrum(
                isolated, meta["fs"], selected_band
            )
            harmonics = match_harmonics(
                spectrum,
                frequencies,
                meta,
                config.harmonic_tolerance_hz,
                config.max_harmonic,
            )
            kurtogram.to_csv(
                diagnosis_dir / f"sample_{sample_index}_kurtogram.csv", index=False
            )
            np.savetxt(
                diagnosis_dir / f"sample_{sample_index}_envelope_fft.csv",
                np.column_stack([frequencies, spectrum]),
                delimiter=",",
                header="frequency_hz,amplitude",
                comments="",
            )
            diagnoses.append(
                {
                    "sample_index": sample_index,
                    "indexing": "zero_based_csv_row",
                    "isolation_method": isolation_method,
                    "selected_band_hz": list(selected_band),
                    "harmonics": harmonics,
                }
            )

    ffp_one_based = None if ffp is None else ffp + 1
    paper_ffp_one_based = meta["paper_ffp"]
    difference_one_based = (
        None if ffp_one_based is None else ffp_one_based - paper_ffp_one_based
    )
    result = {
        "dataset": dataset,
        "model_variant": config.model_variant,
        "data_validation": validate_dataset(paths["data"], meta),
        "hi_definition": hi_definition,
        "healthy_samples": config.healthy_samples,
        "window_size": (
            None
            if config.model_variant in {
                "regularized_full_sample",
                "multilatent_full_sample",
            }
            else config.window_size
        ),
        "latent_dim": (
            config.latent_dim
            if config.model_variant == "multilatent_full_sample"
            else 1
        ),
        "windows_per_sample": windows_per_sample,
        "usable_dimension": usable_dim,
        "threshold_p95": threshold,
        "ffp": ffp,
        "ffp_zero_based": ffp,
        "ffp_one_based": ffp_one_based,
        "ffp_rule": f"first of {config.consecutive} consecutive HI values above P95",
        "indexing_note": (
            "Project ffp/ffp_zero_based uses CSV row indices; ffp_one_based is "
            "the comparable sample label used by the paper."
        ),
        "paper_ffp": paper_ffp_one_based,
        "paper_ffp_one_based": paper_ffp_one_based,
        "difference_from_paper": difference_one_based,
        "difference_from_paper_one_based": difference_one_based,
        "paper_stage_thresholds_as_printed": stage_thresholds,
        "hi_statistics": hi_statistics,
        "monotonicity_audit": monotonicity_audit,
        "xai": xai,
        "diagnoses": diagnoses,
    }
    (result_dir / f"{dataset}_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del analysis_model, healthy, healthy_hi, hi_values
    cleanup_memory()
    return result


def run_pipeline(config: RunConfig) -> list[dict]:
    if config.model_variant not in {
        "regularized_full_sample",
        "multilatent_full_sample",
        "dense_windowed_baseline",
    }:
        raise ValueError(f"Unsupported model_variant: {config.model_variant}")
    if config.lambda_monotonic < 0 or config.lambda_smoothing < 0:
        raise ValueError("Regularization coefficients must be non-negative")
    if config.model_variant == "regularized_full_sample" and config.batch_size < 3:
        raise ValueError("Regularized full-sample training requires batch_size >= 3")
    if config.model_variant == "multilatent_full_sample" and config.latent_dim < 1:
        raise ValueError("multilatent_full_sample requires latent_dim >= 1")

    set_reproducibility(config.random_seed)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths = discover_data(config)
    results = []
    for dataset in config.datasets:
        print("\n" + "=" * 80)
        print(f"[RUN] {dataset}")
        model_path, _ = train_dataset(dataset, paths[dataset], config)
        result = evaluate_dataset(dataset, paths[dataset], model_path, config)
        results.append(result)
        print(
            f"[DONE] {dataset}: FFP={result['ffp']}, "
            f"P95={result['threshold_p95']:.8g}"
        )
        cleanup_memory()

    summary = pd.DataFrame(
        [
            {
                "dataset": item["dataset"],
                "threshold_p95": item["threshold_p95"],
                "project_ffp_zero_based": item["ffp_zero_based"],
                "project_ffp_one_based": item["ffp_one_based"],
                "paper_ffp_one_based": item["paper_ffp_one_based"],
                "difference_one_based": item["difference_from_paper_one_based"],
                "top3_xai": "; ".join(
                    f"{entry['feature']} ({abs(entry['correlation']):.2f})"
                    for entry in item.get("xai", {}).get("top3", [])
                ),
            }
            for item in results
        ]
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "all_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAll outputs: {output_root}")
    return results


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def _lambda_tag(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def run_lambda_sensitivity(
    base_config: RunConfig,
    lambda_cases: Iterable[tuple[float, float]],
    *,
    seeds: Iterable[int] = (42,),
    sensitivity_root: str = "/kaggle/working/bearing_fdd_lambda_sensitivity",
    resume: bool = True,
    keep_models: bool = False,
) -> pd.DataFrame:
    """Run auditable lambda ablations without overwriting case outputs.

    Screening should normally disable XAI and diagnosis in ``base_config``.
    Those expensive downstream stages can be rerun for the selected cases.
    """
    if base_config.model_variant != "regularized_full_sample":
        raise ValueError("Lambda sensitivity requires regularized_full_sample")
    if tuple(base_config.datasets) != ("IMS2",):
        raise ValueError("Screen lambda sensitivity on IMS2 only")

    root = Path(sensitivity_root)
    root.mkdir(parents=True, exist_ok=True)
    cases = [(float(lm), float(ls)) for lm, ls in lambda_cases]
    seed_values = [int(seed) for seed in seeds]
    plan = {
        "base_config": asdict(base_config),
        "lambda_cases": [
            {"lambda_monotonic": lm, "lambda_smoothing": ls} for lm, ls in cases
        ],
        "seeds": seed_values,
        "resume": resume,
        "keep_models": keep_models,
        "indexing": {
            "project_ffp_zero_based": "CSV row index",
            "project_ffp_one_based": "paper-comparable sample label",
        },
    }
    (root / "sensitivity_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    records: list[dict] = []
    case_number = 0
    for seed in seed_values:
        for lambda_monotonic, lambda_smoothing in cases:
            case_id = (
                f"case_{case_number:02d}"
                f"_lm_{_lambda_tag(lambda_monotonic)}"
                f"_ls_{_lambda_tag(lambda_smoothing)}"
                f"_seed_{seed}"
            )
            case_number += 1
            case_root = root / case_id
            record_path = case_root / "sensitivity_case.json"
            if resume and record_path.exists():
                prior = json.loads(record_path.read_text(encoding="utf-8"))
                if prior.get("status") == "completed":
                    print(f"[SKIP] {case_id} already completed")
                    records.append(prior)
                    continue

            print(
                f"\n[SENSITIVITY] {case_id}: "
                f"lambda_monotonic={lambda_monotonic:g}, "
                f"lambda_smoothing={lambda_smoothing:g}"
            )
            case_root.mkdir(parents=True, exist_ok=True)
            config = replace(
                base_config,
                output_root=str(case_root),
                lambda_monotonic=lambda_monotonic,
                lambda_smoothing=lambda_smoothing,
                random_seed=seed,
                force_retrain=True,
            )
            started = time.time()
            try:
                result = run_pipeline(config)[0]
                training = json.loads(
                    (case_root / "logs" / "IMS2.training.json").read_text(
                        encoding="utf-8"
                    )
                )
                history = training["history"]
                reconstruction = float(history["reconstruction_loss"][-1])
                monotonic = float(history["monotonic_loss"][-1])
                smoothing = float(history["smoothing_loss"][-1])
                weighted_monotonic = lambda_monotonic * monotonic
                weighted_smoothing = lambda_smoothing * smoothing
                audit = result["monotonicity_audit"]
                stats = result["hi_statistics"]
                record = {
                    "status": "completed",
                    "case_id": case_id,
                    "lambda_monotonic": lambda_monotonic,
                    "lambda_smoothing": lambda_smoothing,
                    "seed": seed,
                    "threshold_p95": result["threshold_p95"],
                    "project_ffp_zero_based": result["ffp_zero_based"],
                    "project_ffp_one_based": result["ffp_one_based"],
                    "paper_ffp_one_based": result["paper_ffp_one_based"],
                    "difference_one_based": result["difference_from_paper_one_based"],
                    "absolute_difference_one_based": (
                        None
                        if result["difference_from_paper_one_based"] is None
                        else abs(result["difference_from_paper_one_based"])
                    ),
                    "final_reconstruction_loss": reconstruction,
                    "final_monotonic_loss": monotonic,
                    "final_smoothing_loss": smoothing,
                    "weighted_monotonic_loss": weighted_monotonic,
                    "weighted_smoothing_loss": weighted_smoothing,
                    "weighted_monotonic_to_reconstruction": _finite_ratio(
                        weighted_monotonic, reconstruction
                    ),
                    "weighted_smoothing_to_reconstruction": _finite_ratio(
                        weighted_smoothing, reconstruction
                    ),
                    "hi_minimum": stats["all"]["minimum"],
                    "hi_maximum": stats["all"]["maximum"],
                    "hi_standard_deviation": stats["all"]["standard_deviation"],
                    "healthy_hi_standard_deviation": stats["healthy"][
                        "standard_deviation"
                    ],
                    "unlabeled_hi_standard_deviation": stats["unlabeled"][
                        "standard_deviation"
                    ],
                    "decreasing_step_ratio": audit["decreasing_step_ratio"],
                    "second_difference_rmse": audit["second_difference_rmse"],
                    "elapsed_seconds": round(time.time() - started, 2),
                    "output_root": str(case_root),
                    "models_kept": keep_models,
                }
                if not keep_models:
                    for model_file in (case_root / "models").glob("*"):
                        if model_file.is_file():
                            model_file.unlink()
            except Exception as exc:
                record = {
                    "status": "failed",
                    "case_id": case_id,
                    "lambda_monotonic": lambda_monotonic,
                    "lambda_smoothing": lambda_smoothing,
                    "seed": seed,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "output_root": str(case_root),
                }
                print(f"[FAILED] {case_id}: {type(exc).__name__}: {exc}")
            record_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            records.append(record)
            cleanup_memory()

            table = pd.DataFrame(records)
            table.to_csv(root / "sensitivity_summary.csv", index=False)
            (root / "sensitivity_results.json").write_text(
                json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    summary = pd.DataFrame(records)
    if not summary.empty and "absolute_difference_one_based" in summary:
        completed = summary["status"].eq("completed")
        summary.loc[completed].sort_values(
            [
                "absolute_difference_one_based",
                "decreasing_step_ratio",
                "second_difference_rmse",
            ],
            na_position="last",
        ).to_csv(root / "sensitivity_ranking.csv", index=False)
    print(f"\nSensitivity outputs: {root}")
    return summary


def run_latent_dimension_sensitivity(
    base_config: RunConfig,
    latent_dimensions: Iterable[int],
    *,
    seeds: Iterable[int] = (42,),
    sensitivity_root: str = "/kaggle/working/bearing_fdd_latent_sensitivity",
    resume: bool = True,
    keep_models: bool = False,
) -> pd.DataFrame:
    """Screen full-sample bottleneck dimensions using reconstruction-MSE HI."""
    if base_config.model_variant != "multilatent_full_sample":
        raise ValueError(
            "Latent-dimension sensitivity requires multilatent_full_sample"
        )
    if tuple(base_config.datasets) != ("IMS2",):
        raise ValueError("Screen latent dimensions on IMS2 only")

    dimensions = [int(value) for value in latent_dimensions]
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("latent_dimensions must contain positive integers")
    seed_values = [int(seed) for seed in seeds]
    root = Path(sensitivity_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment": "full-sample multi-latent reconstruction-MSE HI",
        "paper_fidelity_note": (
            "Project experiment; not the paper's one-node Table-1 architecture."
        ),
        "base_config": asdict(base_config),
        "latent_dimensions": dimensions,
        "seeds": seed_values,
        "resume": resume,
        "keep_models": keep_models,
        "indexing": {
            "project_ffp_zero_based": "CSV row index",
            "project_ffp_one_based": "paper-comparable sample label",
        },
    }
    (root / "latent_sensitivity_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    records: list[dict] = []
    case_number = 0
    for seed in seed_values:
        for latent_dim in dimensions:
            case_id = f"case_{case_number:02d}_latent_{latent_dim}_seed_{seed}"
            case_number += 1
            case_root = root / case_id
            record_path = case_root / "latent_sensitivity_case.json"
            if resume and record_path.exists():
                prior = json.loads(record_path.read_text(encoding="utf-8"))
                if prior.get("status") == "completed":
                    print(f"[SKIP] {case_id} already completed")
                    records.append(prior)
                    continue

            print(
                f"\n[LATENT SENSITIVITY] {case_id}: latent_dim={latent_dim}"
            )
            case_root.mkdir(parents=True, exist_ok=True)
            config = replace(
                base_config,
                output_root=str(case_root),
                latent_dim=latent_dim,
                random_seed=seed,
                force_retrain=True,
            )
            started = time.time()
            try:
                result = run_pipeline(config)[0]
                training = json.loads(
                    (case_root / "logs" / "IMS2.training.json").read_text(
                        encoding="utf-8"
                    )
                )
                final_reconstruction = float(training["history"]["loss"][-1])
                audit = result["monotonicity_audit"]
                stats = result["hi_statistics"]
                record = {
                    "status": "completed",
                    "case_id": case_id,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    "hi_definition": result["hi_definition"],
                    "threshold_p95": result["threshold_p95"],
                    "project_ffp_zero_based": result["ffp_zero_based"],
                    "project_ffp_one_based": result["ffp_one_based"],
                    "paper_ffp_one_based": result["paper_ffp_one_based"],
                    "difference_one_based": result[
                        "difference_from_paper_one_based"
                    ],
                    "absolute_difference_one_based": (
                        None
                        if result["difference_from_paper_one_based"] is None
                        else abs(result["difference_from_paper_one_based"])
                    ),
                    "final_training_reconstruction_loss": final_reconstruction,
                    "autoencoder_params": training["autoencoder_params"],
                    "encoder_params": training["encoder_params"],
                    "hi_minimum": stats["all"]["minimum"],
                    "hi_maximum": stats["all"]["maximum"],
                    "hi_standard_deviation": stats["all"]["standard_deviation"],
                    "healthy_hi_standard_deviation": stats["healthy"][
                        "standard_deviation"
                    ],
                    "unlabeled_hi_standard_deviation": stats["unlabeled"][
                        "standard_deviation"
                    ],
                    "decreasing_step_ratio": audit["decreasing_step_ratio"],
                    "second_difference_rmse": audit["second_difference_rmse"],
                    "elapsed_seconds": round(time.time() - started, 2),
                    "output_root": str(case_root),
                    "models_kept": keep_models,
                }
                if not keep_models:
                    for model_file in (case_root / "models").glob("*"):
                        if model_file.is_file():
                            model_file.unlink()
            except Exception as exc:
                record = {
                    "status": "failed",
                    "case_id": case_id,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "output_root": str(case_root),
                }
                print(f"[FAILED] {case_id}: {type(exc).__name__}: {exc}")

            record_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            records.append(record)
            cleanup_memory()
            table = pd.DataFrame(records)
            table.to_csv(root / "latent_sensitivity_summary.csv", index=False)
            (root / "latent_sensitivity_results.json").write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    summary = pd.DataFrame(records)
    if not summary.empty and "absolute_difference_one_based" in summary:
        completed_with_ffp = (
            summary["status"].eq("completed")
            & summary["project_ffp_one_based"].notna()
        )
        summary.loc[completed_with_ffp].sort_values(
            [
                "absolute_difference_one_based",
                "decreasing_step_ratio",
                "second_difference_rmse",
            ]
        ).to_csv(root / "latent_sensitivity_ranking.csv", index=False)
    print(f"\nLatent-dimension sensitivity outputs: {root}")
    return summary


if __name__ == "__main__":
    run_pipeline(RunConfig())
