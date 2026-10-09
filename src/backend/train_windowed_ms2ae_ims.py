import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

import enter_utils


DATASETS = ["IMS1", "IMS3"]
HEALTHY_SAMPLES = 300
WINDOW_SIZE = int(os.environ.get("MS2AE_WINDOW_SIZE", "2048"))
EPOCHS = int(os.environ.get("MS2AE_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("MS2AE_BATCH_SIZE", "64"))
MODEL_DIR = Path("prog_analizador/models")
LOG_DIR = MODEL_DIR / "ms2ae_windowed_training_logs"


def configure_runtime():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    np.random.seed(42)
    tf.random.set_seed(42)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def cleanup(stage):
    print(f"[MEMORY] cleanup: {stage}", flush=True)
    tf.keras.backend.clear_session()
    gc.collect()


def to_windows(samples):
    samples = np.asarray(samples, dtype=np.float32)
    usable_dim = (samples.shape[1] // WINDOW_SIZE) * WINDOW_SIZE
    samples = samples[:, :usable_dim]
    windows_per_sample = usable_dim // WINDOW_SIZE
    return samples.reshape(samples.shape[0] * windows_per_sample, WINDOW_SIZE), windows_per_sample, usable_dim


def train_one(dataset):
    configure_runtime()
    cleanup("before train")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    encoder_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_encoder.keras"
    autoencoder_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_autoencoder.keras"
    if encoder_path.exists() and autoencoder_path.exists() and os.environ.get("FORCE_RETRAIN", "0") != "1":
        print(f"[SKIP] Existing model found for {dataset}", flush=True)
        return

    start = time.time()
    print("=" * 72, flush=True)
    print(f"[DATASET] {dataset}", flush=True)
    healthy, _ = enter_utils.getDataset(dataset, HEALTHY_SAMPLES, 0)
    healthy = np.asarray(healthy, dtype=np.float32)
    windows, windows_per_sample, usable_dim = to_windows(healthy)
    print(
        f"[DATA] raw={healthy.shape} usable_dim={usable_dim} "
        f"windows={windows.shape} windows_per_sample={windows_per_sample}",
        flush=True,
    )

    autoencoder, encoder = enter_utils.build_ms2ae_table1(WINDOW_SIZE)
    print(f"[PARAMS] autoencoder={autoencoder.count_params():,}", flush=True)
    history = autoencoder.fit(
        windows,
        windows,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    if encoder_path.exists():
        import shutil

        shutil.rmtree(encoder_path) if encoder_path.is_dir() else encoder_path.unlink()
    if autoencoder_path.exists():
        import shutil

        shutil.rmtree(autoencoder_path) if autoencoder_path.is_dir() else autoencoder_path.unlink()
    encoder.save(encoder_path)
    autoencoder.save(autoencoder_path)

    log = {
        "dataset": dataset,
        "status": "OK",
        "model": "Windowed_MS2AE_Table1",
        "structure": f"IS({WINDOW_SIZE})-3500-700-200-1-200-700-3500-IS({WINDOW_SIZE})",
        "healthy_samples": HEALTHY_SAMPLES,
        "raw_input_dim": int(healthy.shape[1]),
        "usable_input_dim": int(usable_dim),
        "window_size": WINDOW_SIZE,
        "windows_per_sample": int(windows_per_sample),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "final_loss": float(history.history["loss"][-1]),
        "encoder_path": str(encoder_path),
        "autoencoder_path": str(autoencoder_path),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    log_path = LOG_DIR / f"{dataset}.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {encoder_path}", flush=True)
    print(f"[SAVED] {autoencoder_path}", flush=True)
    print(f"[LOG] {log_path}", flush=True)

    del autoencoder, encoder, history, healthy, windows
    cleanup("after train")


def main():
    datasets = sys.argv[1:] or DATASETS
    for dataset in datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"Unsupported dataset: {dataset}")
        train_one(dataset)


if __name__ == "__main__":
    main()
