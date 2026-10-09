import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew


DATA_DIR = Path("prog_analizador/data")
MODEL_DIR = Path("prog_analizador/models")
RESULT_DIR = Path("test_results_windowed_ms2ae")
WINDOW_SIZE = int(os.environ.get("MS2AE_WINDOW_SIZE", "2048"))
HEALTHY_SAMPLES = int(os.environ.get("MS2AE_HEALTHY_SAMPLES", "300"))
CHUNK_SAMPLES = int(os.environ.get("MS2AE_EVAL_CHUNK", "40"))
CONSECUTIVE = int(os.environ.get("MS2AE_CONSECUTIVE", "5"))

DATASETS = {
    "IMS1": {
        "csv": "IMS1",
        "fs": 20480.0,
        "shaft": 33.33,
        "BPFO": 236.0,
        "BPFI": 297.0,
        "BSF": 278.0,
        "FTF": 15.0,
        "filter_band": (1280.0, 2560.0),
        "paper_ffp": 1857,
        "do_xai": False,
    },
    "IMS3": {
        "csv": "IMS3",
        "fs": 20480.0,
        "shaft": 33.33,
        "BPFO": 236.0,
        "BPFI": 297.0,
        "BSF": 278.0,
        "FTF": 15.0,
        "filter_band": (2560.0, 5120.0),
        "paper_ffp": 5967,
        "do_xai": True,
    },
    "XJTU2-1": {
        "csv": "XJTU_SY_2_1",
        "fs": 25600.0,
        "shaft": 37.5,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "filter_band": (1600.0, 3200.0),
        "paper_ffp": 451,
        "do_xai": True,
    },
    "XJTU2-3": {
        "csv": "XJTU_SY_2_3",
        "fs": 25600.0,
        "shaft": 37.5,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "filter_band": (8533.33, 10666.67),
        "paper_ffp": 301,
        "do_xai": True,
    },
    "XJTU3-1": {
        "csv": "XJTU_SY_3_1",
        "fs": 25600.0,
        "shaft": 40.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "filter_band": (6400.0, 8533.33),
        "paper_ffp": 2347,
        "do_xai": True,
    },
    "XJTU3-4": {
        "csv": "XJTU_SY_3_4",
        "fs": 25600.0,
        "shaft": 40.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "filter_band": (3200.0, 6400.0),
        "paper_ffp": 1416,
        "do_xai": True,
    },
}


def configure_runtime():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def count_rows(path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def read_rows(path, start, count, input_dim=None):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx < start:
                continue
            if idx >= start + count:
                break
            line = line.strip()
            if not line:
                continue
            values = np.fromstring(line, sep=",", dtype=np.float32)
            if input_dim is not None:
                if values.size < input_dim:
                    values = np.pad(values, (0, input_dim - values.size), mode="constant")
                elif values.size > input_dim:
                    values = values[:input_dim]
            rows.append(values)
    if not rows:
        return np.empty((0, input_dim or 0), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def to_windows(samples):
    samples = np.asarray(samples, dtype=np.float32)
    usable_dim = (samples.shape[1] // WINDOW_SIZE) * WINDOW_SIZE
    samples = samples[:, :usable_dim]
    windows_per_sample = usable_dim // WINDOW_SIZE
    windows = samples.reshape(samples.shape[0] * windows_per_sample, WINDOW_SIZE)
    return windows, windows_per_sample, usable_dim


def reconstruction_hi(autoencoder, samples):
    windows, windows_per_sample, usable_dim = to_windows(samples)
    reconstructed = autoencoder.predict(windows, verbose=0, batch_size=64)
    window_hi = np.mean((windows - reconstructed) ** 2, axis=1)
    sample_hi = np.percentile(window_hi.reshape(-1, windows_per_sample), 95, axis=1)
    return sample_hi.astype(float), usable_dim, windows_per_sample


def find_ffp(values, threshold):
    consecutive = 0
    for idx, value in enumerate(values):
        if value > threshold:
            consecutive += 1
            if consecutive >= CONSECUTIVE:
                return idx - CONSECUTIVE + 1
        else:
            consecutive = 0
    return None


def time_features(samples):
    eps = 1e-12
    abs_x = np.abs(samples)
    rms = np.sqrt(np.mean(samples ** 2, axis=1))
    peak = np.max(abs_x, axis=1)
    mean_abs = np.mean(abs_x, axis=1)
    sqrt_abs_mean = np.mean(np.sqrt(abs_x + eps), axis=1)
    return {
        "RMS": rms,
        "Sk": skew(samples, axis=1, nan_policy="omit"),
        "K": scipy_kurtosis(samples, axis=1, fisher=False, nan_policy="omit"),
        "CF": peak / (rms + eps),
        "SF": rms / (mean_abs + eps),
        "IF": peak / (mean_abs + eps),
        "MF": peak / ((sqrt_abs_mean ** 2) + eps),
    }


def power_spectrum(signal, fs):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    spectrum = np.abs(np.fft.rfft(np.hanning(signal.size) * signal, norm="forward")) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    spectrum[:5] = 0.0
    return spectrum, freqs


def filtered_envelope_spectrum(signal, fs, band):
    nyquist = fs / 2.0
    low = max(band[0] / nyquist, 0.001)
    high = min(band[1] / nyquist, 0.99)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return power_spectrum(envelope, fs)


def max_near(spectrum, freqs, target_hz, width_hz=10.0):
    mask = (freqs >= target_hz - width_hz) & (freqs <= target_hz + width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(spectrum[mask]))


def frequency_features(samples, meta):
    targets = {
        "Fund nf": meta["shaft"],
        "BPFO nf": meta["BPFO"],
        "BPFI nf": meta["BPFI"],
        "BSF nf": meta["BSF"],
        "FTF nf": meta["FTF"],
        "Fund f": meta["shaft"],
        "BPFO f": meta["BPFO"],
        "BPFI f": meta["BPFI"],
        "BSF f": meta["BSF"],
        "FTF f": meta["FTF"],
    }
    values = {name: [] for name in targets}
    for sample in samples:
        nf_spectrum, nf_freqs = power_spectrum(sample, meta["fs"])
        f_spectrum, f_freqs = filtered_envelope_spectrum(sample, meta["fs"], meta["filter_band"])
        for name, target in targets.items():
            if name.endswith("nf"):
                values[name].append(max_near(nf_spectrum, nf_freqs, target))
            else:
                values[name].append(max_near(f_spectrum, f_freqs, target))
    return {name: np.asarray(vals, dtype=float) for name, vals in values.items()}


def corr_dict(hi_values, feature_values):
    result = {}
    hi = np.asarray(hi_values, dtype=float)
    for name, values in feature_values.items():
        values = np.asarray(values, dtype=float)
        if np.std(hi) == 0 or np.std(values) == 0:
            result[name] = 0.0
        else:
            result[name] = float(np.corrcoef(hi, values)[0, 1])
    return result


class OnlineCorrelation:
    def __init__(self):
        self.names = None
        self.n = 0
        self.sum = None
        self.cross = None

    def update(self, hi_values, feature_values):
        feature_names = list(feature_values.keys())
        names = ["HI", *feature_names]
        matrix = np.column_stack([hi_values, *[feature_values[name] for name in feature_names]]).astype(float)
        if self.names is None:
            self.names = names
            width = len(names)
            self.sum = np.zeros(width, dtype=float)
            self.cross = np.zeros((width, width), dtype=float)
        elif names != self.names:
            raise ValueError("Feature order changed during online correlation update")
        self.n += matrix.shape[0]
        self.sum += np.sum(matrix, axis=0)
        self.cross += matrix.T @ matrix

    def correlation_matrix(self):
        if self.n == 0:
            return [], np.empty((0, 0), dtype=float)
        mean = self.sum / self.n
        covariance = self.cross / self.n - np.outer(mean, mean)
        variance = np.diag(covariance)
        denom = np.sqrt(np.maximum(variance, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = covariance / np.outer(denom, denom)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        return self.names, corr


def evaluate_dataset(dataset):
    configure_runtime()
    meta = DATASETS[dataset]
    csv_path = DATA_DIR / f"{meta['csv']}.csv"
    autoencoder_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_autoencoder.keras"
    if not autoencoder_path.exists():
        raise FileNotFoundError(f"Missing autoencoder: {autoencoder_path}")

    print("=" * 72, flush=True)
    print(f"[DATASET] {dataset}", flush=True)
    print(f"[LOAD] {autoencoder_path}", flush=True)
    autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)

    total = count_rows(csv_path)
    input_dim = read_rows(csv_path, 0, 1).shape[1]
    print(f"[DATA] total={total} input_dim={input_dim}", flush=True)

    healthy = read_rows(csv_path, 0, HEALTHY_SAMPLES, input_dim)
    healthy_hi, usable_dim, windows_per_sample = reconstruction_hi(autoencoder, healthy)
    threshold = float(np.percentile(healthy_hi, 95))

    online_corr = OnlineCorrelation() if meta.get("do_xai", False) else None
    ffp = None
    consecutive = 0
    scan_hi_min = None
    scan_hi_max = None
    start = 0
    while start < total:
        count = min(CHUNK_SAMPLES, total - start)
        batch = read_rows(csv_path, start, count, input_dim)
        hi_batch, _, _ = reconstruction_hi(autoencoder, batch)
        for idx, value in enumerate(hi_batch):
            global_idx = start + idx
            if global_idx < HEALTHY_SAMPLES:
                continue
            value = float(value)
            scan_hi_min = value if scan_hi_min is None else min(scan_hi_min, value)
            scan_hi_max = value if scan_hi_max is None else max(scan_hi_max, value)
            if ffp is None:
                if value > threshold:
                    consecutive += 1
                    if consecutive >= CONSECUTIVE:
                        ffp = global_idx - CONSECUTIVE + 1
                else:
                    consecutive = 0
        if meta.get("do_xai", False):
            batch_features = time_features(batch)
            batch_features.update(frequency_features(batch, meta))
            online_corr.update(hi_batch, batch_features)
        start += count
        del batch, hi_batch
        gc.collect()
        if start % 400 < CHUNK_SAMPLES or start >= total:
            print(f"[PROGRESS] {dataset}: {min(start, total)}/{total}", flush=True)

    corr_names, corr_matrix = online_corr.correlation_matrix() if online_corr is not None else ([], np.empty((0, 0)))
    correlations = {}
    if corr_names:
        hi_row = corr_matrix[0]
        correlations = {name: float(hi_row[idx]) for idx, name in enumerate(corr_names) if idx != 0}
    top3 = sorted(
        [(name, abs(value), value) for name, value in correlations.items()],
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    result = {
        "dataset": dataset,
        "total_samples": total,
        "healthy_samples": HEALTHY_SAMPLES,
        "window_size": WINDOW_SIZE,
        "usable_dim": int(usable_dim),
        "windows_per_sample": int(windows_per_sample),
        "aggregation": "recon_p95",
        "threshold": threshold,
        "project_ffp": ffp,
        "paper_ffp": meta.get("paper_ffp"),
        "diff_vs_paper": None if ffp is None or meta.get("paper_ffp") is None else ffp - meta["paper_ffp"],
        "healthy_hi_min": float(np.min(healthy_hi)),
        "healthy_hi_max": float(np.max(healthy_hi)),
        "scan_hi_min": scan_hi_min,
        "scan_hi_max": scan_hi_max,
        "filter_band": [float(meta["filter_band"][0]), float(meta["filter_band"][1])],
        "correlations": correlations,
        "correlation_matrix_features": corr_names,
        "correlation_matrix": corr_matrix.tolist() if corr_names else [],
        "top3_correlated_features": [
            {"feature": name, "abs_corr": abs_corr, "corr": corr}
            for name, abs_corr, corr in top3
        ],
    }

    del autoencoder, healthy
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def write_outputs(results):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "project_ffp_xai_refresh.json"
    md_path = RESULT_DIR / "project_ffp_xai_refresh.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Project FFP/Threshold/XAI Refresh",
        "",
        f"- Aggregation: `recon_p95`",
        f"- Healthy samples: `{HEALTHY_SAMPLES}`",
        f"- FFP rule: first `{CONSECUTIVE}` consecutive samples over P95 threshold",
        "",
        "| Dataset | Project FFP | Paper FFP | Diff | Threshold | Top 3 correlated features |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        top3 = "; ".join(
            f"{item['feature']} ({item['abs_corr']:.2f})"
            for item in result.get("top3_correlated_features", [])
        ) or "--"
        lines.append(
            f"| {result['dataset']} | {result['project_ffp'] if result['project_ffp'] is not None else 'NOT FOUND'} | "
            f"{result.get('paper_ffp', '--')} | "
            f"{result['diff_vs_paper'] if result['diff_vs_paper'] is not None else 'NA'} | "
            f"{result['threshold']:.6f} | {top3} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {json_path}", flush=True)
    print(f"[SAVED] {md_path}", flush=True)


def main():
    datasets = sys.argv[1:] or ["IMS1", "IMS3", "XJTU2-1", "XJTU2-3", "XJTU3-1", "XJTU3-4"]
    results = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"Unknown dataset: {dataset}")
        results.append(evaluate_dataset(dataset))
    write_outputs(results)


if __name__ == "__main__":
    main()
