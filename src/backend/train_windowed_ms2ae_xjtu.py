import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

import enter_utils


DATASETS = ["XJTU2-1", "XJTU2-3", "XJTU3-1", "XJTU3-4"]
HEALTHY_SAMPLES = 300
WINDOW_SIZE = int(os.environ.get("MS2AE_WINDOW_SIZE", "2048"))
EPOCHS = int(os.environ.get("MS2AE_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("MS2AE_BATCH_SIZE", "64"))
AGGREGATION = os.environ.get("MS2AE_AGGREGATION", "p95")

MODEL_DIR = Path("prog_analizador/models")
LOG_DIR = MODEL_DIR / "ms2ae_windowed_training_logs"
BACKUP_DIR = MODEL_DIR / "backup_before_ms2ae_windowed_retrain"
SUMMARY_PATH = MODEL_DIR / "xjtu_ms2ae_windowed_retrain_summary.json"


def remove_path(path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def reset_memory(stage):
    print(f"[MEMORY] cleanup: {stage}", flush=True)
    tf.keras.backend.clear_session()
    gc.collect()


def configure_runtime():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    np.random.seed(42)
    tf.random.set_seed(42)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"[INFO] GPU count: {len(gpus)}", flush=True)


def to_windows(samples, window_size):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, input_dim = samples.shape
    if input_dim % window_size != 0:
        usable = (input_dim // window_size) * window_size
        if usable <= 0:
            raise ValueError(f"window_size={window_size} is larger than input_dim={input_dim}")
        samples = samples[:, :usable]
        input_dim = usable
    windows_per_sample = input_dim // window_size
    return samples.reshape(n_samples * windows_per_sample, window_size), windows_per_sample, input_dim


def aggregate_hi(window_hi, windows_per_sample, method):
    hi = np.asarray(window_hi, dtype=np.float32).reshape(-1, windows_per_sample)
    if method == "max":
        return np.max(hi, axis=1)
    if method == "mean":
        return np.mean(hi, axis=1)
    if method == "p95":
        return np.percentile(hi, 95, axis=1)
    raise ValueError(f"Unknown aggregation method: {method}")


def backup_existing(dataset):
    existing = MODEL_DIR / f"{dataset}.windowed_ms2ae_encoder.keras"
    if not existing.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamped = BACKUP_DIR / f"{dataset}.windowed_ms2ae_encoder.keras"
    if stamped.exists():
        if stamped.is_dir():
            shutil.rmtree(stamped)
        else:
            stamped.unlink()
    if existing.is_dir():
        shutil.copytree(existing, stamped)
    else:
        shutil.copy2(existing, stamped)
    return str(stamped)


def train_one(dataset):
    configure_runtime()
    reset_memory("before model build")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print("=" * 72, flush=True)
    print(f"[DATASET] {dataset}", flush=True)
    print("[MODEL] Window-level dense MS2AE Table 1", flush=True)
    print(
        f"[STRUCTURE] IS({WINDOW_SIZE})-3500-700-200-1-200-700-3500-IS({WINDOW_SIZE})",
        flush=True,
    )
    print(
        f"[HYPERPARAMS] optimizer=adam lr=0.001 epochs={EPOCHS} "
        f"batch_size={BATCH_SIZE} aggregation={AGGREGATION}",
        flush=True,
    )

    backup_path = backup_existing(dataset)
    if backup_path:
        print(f"[BACKUP] {backup_path}", flush=True)

    healthy_samples, _ = enter_utils.getDataset(dataset, HEALTHY_SAMPLES, 0)
    healthy_samples = np.asarray(healthy_samples, dtype=np.float32)
    if healthy_samples.size == 0:
        raise RuntimeError(f"No healthy samples loaded for {dataset}")

    windows, windows_per_sample, usable_input_dim = to_windows(healthy_samples, WINDOW_SIZE)
    print(
        f"[DATA] raw={healthy_samples.shape} usable_dim={usable_input_dim} "
        f"windows={windows.shape} windows_per_sample={windows_per_sample}",
        flush=True,
    )

    autoencoder, encoder = enter_utils.build_ms2ae_table1(WINDOW_SIZE)
    print(f"[PARAMS] autoencoder={autoencoder.count_params():,} encoder={encoder.count_params():,}", flush=True)

    history = autoencoder.fit(
        windows,
        windows,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    output_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_encoder.keras"
    autoencoder_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_autoencoder.keras"
    remove_path(output_path)
    remove_path(autoencoder_path)
    encoder.save(output_path)
    autoencoder.save(autoencoder_path)

    window_hi = encoder.predict(windows, verbose=0, batch_size=BATCH_SIZE).flatten()
    sample_hi = aggregate_hi(window_hi, windows_per_sample, AGGREGATION)
    threshold = float(enter_utils.getThreshold(sample_hi))
    elapsed = round(time.time() - start_time, 2)

    result = {
        "dataset": dataset,
        "status": "OK",
        "model": "Windowed_MS2AE_Table1_Encoder",
        "structure": f"IS({WINDOW_SIZE})-3500-700-200-1-200-700-3500-IS({WINDOW_SIZE})",
        "note": "Raw sample is split into fixed windows; sample-level HI is aggregated from window-level HI.",
        "paper_exact_structure": False,
        "paper_consistent_core": True,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "healthy_samples": HEALTHY_SAMPLES,
        "raw_input_dim": int(healthy_samples.shape[1]),
        "usable_input_dim": int(usable_input_dim),
        "window_size": WINDOW_SIZE,
        "windows_per_sample": int(windows_per_sample),
        "aggregation": AGGREGATION,
        "autoencoder_params": int(autoencoder.count_params()),
        "encoder_params": int(encoder.count_params()),
        "final_loss": float(history.history["loss"][-1]),
        "healthy_hi_min": float(np.min(sample_hi)),
        "healthy_hi_max": float(np.max(sample_hi)),
        "healthy_hi_threshold_p95": threshold,
        "encoder_path": str(output_path),
        "autoencoder_path": str(autoencoder_path),
        "backup_path": backup_path,
        "elapsed_seconds": elapsed,
    }

    log_path = LOG_DIR / f"{dataset}.json"
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {output_path}", flush=True)
    print(f"[LOSS] {result['final_loss']:.8f}", flush=True)
    print(f"[HI] min={result['healthy_hi_min']:.6f} max={result['healthy_hi_max']:.6f} p95={threshold:.6f}", flush=True)
    print(f"[LOG] {log_path}", flush=True)

    del autoencoder, encoder, history, healthy_samples, windows, window_hi, sample_hi
    reset_memory("after model save")
    return result


def collect_summary():
    results = []
    for dataset in DATASETS:
        log_path = LOG_DIR / f"{dataset}.json"
        if log_path.exists():
            results.append(json.loads(log_path.read_text(encoding="utf-8")))
        else:
            results.append({"dataset": dataset, "status": "MISSING_LOG"})
    SUMMARY_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SUMMARY] {SUMMARY_PATH}", flush=True)


def main():
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
        if dataset not in DATASETS:
            raise SystemExit(f"Unknown dataset: {dataset}")
        try:
            train_one(dataset)
        except Exception as exc:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            (LOG_DIR / f"{dataset}.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "window_size": WINDOW_SIZE,
                        "batch_size": BATCH_SIZE,
                        "aggregation": AGGREGATION,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            reset_memory("after failed train")
            raise
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for old_log in LOG_DIR.glob("XJTU*.json"):
        old_log.unlink()

    for dataset in DATASETS:
        print("\n" + "#" * 72, flush=True)
        print(f"[SPAWN] Training {dataset}", flush=True)
        proc = subprocess.run([sys.executable, __file__, dataset])
        if proc.returncode != 0:
            print(f"[ERROR] {dataset} failed with code {proc.returncode}", flush=True)
        reset_memory(f"after child process {dataset}")
        time.sleep(5)

    collect_summary()


if __name__ == "__main__":
    main()
