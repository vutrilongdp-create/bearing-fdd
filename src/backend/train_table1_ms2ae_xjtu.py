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
EPOCHS = 5
BATCH_SIZE = int(os.environ.get("MS2AE_BATCH_SIZE", "64"))
DTYPE_POLICY = os.environ.get("MS2AE_DTYPE_POLICY", "float32")
MODEL_DIR = Path("prog_analizador/models")
BACKUP_DIR = MODEL_DIR / "backup_before_ms2ae_table1_retrain"
LOG_DIR = MODEL_DIR / "ms2ae_table1_training_logs"
SUMMARY_PATH = MODEL_DIR / "xjtu_ms2ae_table1_retrain_summary.json"


def reset_memory(stage):
    print(f"[MEMORY] cleanup: {stage}", flush=True)
    try:
        tf.keras.backend.clear_session()
    except TypeError:
        tf.keras.backend.clear_session()
    gc.collect()


def configure_runtime():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.mixed_precision.set_global_policy(DTYPE_POLICY)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"[INFO] GPU count: {len(gpus)}", flush=True)


def backup_existing(dataset):
    existing = MODEL_DIR / f"{dataset}.ms2ae_encoder.keras"
    if not existing.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamped = BACKUP_DIR / f"{dataset}.ms2ae_encoder.keras"
    if stamped.exists():
        shutil.rmtree(stamped)
    shutil.copytree(existing, stamped)
    return str(stamped)


def train_one(dataset):
    configure_runtime()
    reset_memory("before model build")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print("=" * 72, flush=True)
    print(f"[DATASET] {dataset}", flush=True)
    print("[MODEL] MS2AE Table 1: IS-3500-700-200-1-200-700-3500-IS", flush=True)
    print(
        f"[HYPERPARAMS] optimizer=adam lr=0.001 epochs={EPOCHS} "
        f"batch_size={BATCH_SIZE} dtype_policy={DTYPE_POLICY}",
        flush=True,
    )

    backup_path = backup_existing(dataset)
    if backup_path:
        print(f"[BACKUP] {backup_path}", flush=True)

    healthy_samples, _ = enter_utils.getDataset(dataset, HEALTHY_SAMPLES, 0)
    healthy_samples = np.asarray(healthy_samples, dtype=np.float32)
    if healthy_samples.size == 0:
        raise RuntimeError(f"No healthy samples loaded for {dataset}")
    print(f"[DATA] healthy_samples shape={healthy_samples.shape}", flush=True)

    input_dim = int(healthy_samples.shape[1])
    autoencoder, encoder = enter_utils.build_ms2ae_table1(input_dim)
    print(f"[PARAMS] autoencoder={autoencoder.count_params():,} encoder={encoder.count_params():,}", flush=True)

    history = autoencoder.fit(
        healthy_samples,
        healthy_samples,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    output_path = MODEL_DIR / f"{dataset}.ms2ae_encoder.keras"
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    encoder.save(output_path)

    hi_healthy = encoder.predict(healthy_samples, verbose=0, batch_size=BATCH_SIZE).flatten()
    threshold = float(enter_utils.getThreshold(hi_healthy))
    elapsed = round(time.time() - start_time, 2)
    result = {
        "dataset": dataset,
        "status": "OK",
        "model": "MS2AE_Table1_Encoder",
        "structure": "IS-3500-700-200-1-200-700-3500-IS",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dtype_policy": DTYPE_POLICY,
        "paper_exact_hyperparams": BATCH_SIZE == 64 and DTYPE_POLICY == "float32",
        "healthy_samples": HEALTHY_SAMPLES,
        "input_dim": input_dim,
        "autoencoder_params": int(autoencoder.count_params()),
        "encoder_params": int(encoder.count_params()),
        "final_loss": float(history.history["loss"][-1]),
        "healthy_hi_min": float(np.min(hi_healthy)),
        "healthy_hi_max": float(np.max(hi_healthy)),
        "healthy_hi_threshold_p95": threshold,
        "encoder_path": str(output_path),
        "backup_path": backup_path,
        "elapsed_seconds": elapsed,
    }

    log_path = LOG_DIR / f"{dataset}.json"
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {output_path}", flush=True)
    print(f"[LOSS] {result['final_loss']:.8f}", flush=True)
    print(f"[HI] min={result['healthy_hi_min']:.6f} max={result['healthy_hi_max']:.6f} p95={threshold:.6f}", flush=True)
    print(f"[LOG] {log_path}", flush=True)

    del autoencoder, encoder, history, healthy_samples, hi_healthy
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
            result = train_one(dataset)
            if result["status"] != "OK":
                raise SystemExit(1)
        except Exception as exc:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            error_log = {
                "dataset": dataset,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "structure": "IS-3500-700-200-1-200-700-3500-IS",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "dtype_policy": DTYPE_POLICY,
                "paper_exact_hyperparams": BATCH_SIZE == 64 and DTYPE_POLICY == "float32",
            }
            (LOG_DIR / f"{dataset}.json").write_text(
                json.dumps(error_log, indent=2, ensure_ascii=False),
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
