import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

import enter_utils


DATASETS = {
    "XJTU2-1": {"paper_ffp": 451},
    "XJTU2-3": {"paper_ffp": 301},
    "XJTU3-1": {"paper_ffp": 2347},
    "XJTU3-4": {"paper_ffp": 1416},
}
WINDOW_SIZE = int(os.environ.get("MS2AE_WINDOW_SIZE", "2048"))
HEALTHY_SAMPLES = int(os.environ.get("MS2AE_HEALTHY_SAMPLES", "300"))
CHUNK_SAMPLES = int(os.environ.get("MS2AE_EVAL_CHUNK", "50"))
CONSECUTIVE = int(os.environ.get("MS2AE_CONSECUTIVE", "5"))
AGGREGATIONS = ["p95", "max", "mean"]
MODEL_DIR = Path("prog_analizador/models")
RESULT_DIR = Path("test_results_windowed_ms2ae")
RESULT_JSON = RESULT_DIR / "xjtu_windowed_ffp_results.json"
RESULT_MD = RESULT_DIR / "xjtu_windowed_ffp_results.md"


def dataset_file_name(name):
    return {
        "XJTU2-1": "XJTU_SY_2_1",
        "XJTU2-3": "XJTU_SY_2_3",
        "XJTU3-1": "XJTU_SY_3_1",
        "XJTU3-4": "XJTU_SY_3_4",
    }.get(name, name)


def count_rows(dataset):
    path = Path("prog_analizador/data") / f"{dataset_file_name(dataset)}.csv"
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def to_windows(samples):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, input_dim = samples.shape
    usable_dim = (input_dim // WINDOW_SIZE) * WINDOW_SIZE
    if usable_dim <= 0:
        raise ValueError(f"WINDOW_SIZE={WINDOW_SIZE} is larger than input_dim={input_dim}")
    samples = samples[:, :usable_dim]
    windows_per_sample = usable_dim // WINDOW_SIZE
    return samples.reshape(n_samples * windows_per_sample, WINDOW_SIZE), windows_per_sample, usable_dim


def aggregate(window_hi, windows_per_sample, method):
    hi = np.asarray(window_hi, dtype=np.float32).reshape(-1, windows_per_sample)
    if method == "p95":
        return np.percentile(hi, 95, axis=1)
    if method == "max":
        return np.max(hi, axis=1)
    if method == "mean":
        return np.mean(hi, axis=1)
    raise ValueError(f"Unknown aggregation: {method}")


def predict_sample_hi(model, samples):
    windows, windows_per_sample, usable_dim = to_windows(samples)
    window_hi = model.predict(windows, verbose=0, batch_size=256).flatten()
    return {
        method: aggregate(window_hi, windows_per_sample, method)
        for method in AGGREGATIONS
    }, windows_per_sample, usable_dim


def predict_sample_reconstruction_hi(model, samples):
    windows, windows_per_sample, usable_dim = to_windows(samples)
    reconstructed = model.predict(windows, verbose=0, batch_size=64)
    window_hi = np.mean((windows - reconstructed) ** 2, axis=1)
    return {
        method: aggregate(window_hi, windows_per_sample, method)
        for method in AGGREGATIONS
    }, windows_per_sample, usable_dim


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


def evaluate_dataset(dataset, paper_ffp):
    model_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_encoder.keras"
    autoencoder_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_autoencoder.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    print(f"[LOAD] {dataset}: {model_path}", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    autoencoder = None
    if autoencoder_path.exists():
        print(f"[LOAD] {dataset}: {autoencoder_path}", flush=True)
        autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
    total = count_rows(dataset)
    print(f"[DATA] {dataset}: total={total}", flush=True)

    healthy, _ = enter_utils.getDataset(dataset, HEALTHY_SAMPLES, 0)
    healthy_hi_by_method, windows_per_sample, usable_dim = predict_sample_hi(model, healthy)
    healthy_recon_by_method = None
    if autoencoder is not None:
        healthy_recon_by_method, _, _ = predict_sample_reconstruction_hi(autoencoder, healthy)

    thresholds = {
        f"latent_{method}": float(np.percentile(values, 95))
        for method, values in healthy_hi_by_method.items()
    }
    if healthy_recon_by_method is not None:
        thresholds.update({
            f"recon_{method}": float(np.percentile(values, 95))
            for method, values in healthy_recon_by_method.items()
        })
    scanned = {name: [] for name in thresholds}

    start = HEALTHY_SAMPLES
    while start < total:
        n = min(CHUNK_SAMPLES, total - start)
        samples, _ = enter_utils.getDataset(dataset, n, start)
        hi_by_method, _, _ = predict_sample_hi(model, samples)
        for method, values in hi_by_method.items():
            scanned[f"latent_{method}"].extend([float(v) for v in values])
        if autoencoder is not None:
            recon_by_method, _, _ = predict_sample_reconstruction_hi(autoencoder, samples)
            for method, values in recon_by_method.items():
                scanned[f"recon_{method}"].extend([float(v) for v in values])
        start += n

    result = {
        "dataset": dataset,
        "paper_ffp": paper_ffp,
        "total_samples": total,
        "healthy_samples": HEALTHY_SAMPLES,
        "window_size": WINDOW_SIZE,
        "usable_dim": int(usable_dim),
        "windows_per_sample": int(windows_per_sample),
        "consecutive_rule": CONSECUTIVE,
        "model_path": str(model_path),
        "autoencoder_path": str(autoencoder_path) if autoencoder_path.exists() else None,
        "aggregations": {},
    }

    healthy_sources = {
        **{f"latent_{method}": values for method, values in healthy_hi_by_method.items()},
    }
    if healthy_recon_by_method is not None:
        healthy_sources.update({
            f"recon_{method}": values for method, values in healthy_recon_by_method.items()
        })

    for method in thresholds:
        relative_ffp = find_ffp(scanned[method], thresholds[method])
        detected_ffp = None if relative_ffp is None else HEALTHY_SAMPLES + relative_ffp
        diff = None if detected_ffp is None else detected_ffp - paper_ffp
        result["aggregations"][method] = {
            "threshold": thresholds[method],
            "detected_ffp": detected_ffp,
            "diff_vs_paper": diff,
            "healthy_hi_min": float(np.min(healthy_sources[method])),
            "healthy_hi_max": float(np.max(healthy_sources[method])),
            "scan_hi_min": float(np.min(scanned[method])) if scanned[method] else None,
            "scan_hi_max": float(np.max(scanned[method])) if scanned[method] else None,
        }

    del model, autoencoder
    tf.keras.backend.clear_session()
    return result


def write_markdown(results):
    lines = [
        "# XJTU Windowed MS2AE FFP Evaluation",
        "",
        f"- Window size: `{WINDOW_SIZE}`",
        f"- Healthy samples: `{HEALTHY_SAMPLES}`",
        f"- FFP rule: first `{CONSECUTIVE}` consecutive samples over P95 threshold",
        "",
        "| Dataset | Aggregation | Paper FFP | Detected FFP | Diff | Threshold | HI scan max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for method, data in result["aggregations"].items():
            detected = data["detected_ffp"]
            diff = data["diff_vs_paper"]
            lines.append(
                f"| {result['dataset']} | {method} | {result['paper_ffp']} | "
                f"{detected if detected is not None else 'NOT FOUND'} | "
                f"{diff if diff is not None else 'NA'} | "
                f"{data['threshold']:.8f} | "
                f"{data['scan_hi_max']:.8f} |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, meta in DATASETS.items():
        results.append(evaluate_dataset(dataset, meta["paper_ffp"]))

    RESULT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(results)
    print(f"[SAVED] {RESULT_JSON}", flush=True)
    print(f"[SAVED] {RESULT_MD}", flush=True)


if __name__ == "__main__":
    main()
