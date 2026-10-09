import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

import enter_utils
from evaluate_ffp_windowed_xjtu import aggregate, to_windows


WINDOW_SIZE = 2048
MODEL_DIR = Path("prog_analizador/models")
RESULT_DIR = Path("test_results_windowed_ms2ae")
FFP_JSON = RESULT_DIR / "xjtu_windowed_ffp_results.json"
OUT_MD = RESULT_DIR / "xjtu_table2_stage_comparison.md"
OUT_JSON = RESULT_DIR / "xjtu_table2_stage_comparison.json"

TABLE2_XJTU = {
    "XJTU2-1": {
        "paper_ffp": 451,
        "paper_threshold": 0.50449,
        "samples": [(451, "Medium"), (460, "Medium"), (486, "Last")],
    },
    "XJTU2-3": {
        "paper_ffp": 301,
        "paper_threshold": 0.50158,
        "samples": [(301, "Early"), (419, "Medium"), (532, "Medium")],
    },
    "XJTU3-1": {
        "paper_ffp": 2347,
        "paper_threshold": 0.51112,
        "samples": [(2347, "Early"), (2445, "Medium"), (2533, "Last")],
    },
    "XJTU3-4": {
        "paper_ffp": 1416,
        "paper_threshold": 0.50718,
        "samples": [(1416, "Early"), (1453, "Medium"), (1505, "Last")],
    },
}


def classify_stage(hi, threshold, healthy_max):
    delta = healthy_max - threshold
    medium_threshold = delta * 50.0
    last_threshold = delta * 100.0
    if hi <= threshold:
        return "Healthy", medium_threshold, last_threshold
    if hi < medium_threshold:
        return "Early", medium_threshold, last_threshold
    if hi < last_threshold:
        return "Medium", medium_threshold, last_threshold
    return "Last", medium_threshold, last_threshold


def load_ffp_stats():
    results = json.loads(FFP_JSON.read_text(encoding="utf-8"))
    stats = {}
    for result in results:
        recon = result["aggregations"]["recon_p95"]
        stats[result["dataset"]] = {
            "detected_ffp": recon["detected_ffp"],
            "threshold": recon["threshold"],
            "healthy_max": recon["healthy_hi_max"],
            "hi_max": recon["scan_hi_max"],
        }
    return stats


def sample_recon_p95(model, dataset, sample_idx):
    samples, _ = enter_utils.getDataset(dataset, 1, sample_idx)
    used_sample_idx = sample_idx
    if samples.size == 0:
        used_sample_idx = sample_idx - 1
        samples, _ = enter_utils.getDataset(dataset, 1, used_sample_idx)
    if samples.size == 0:
        raise ValueError(f"Cannot load sample {sample_idx} or {used_sample_idx} for {dataset}")
    windows, windows_per_sample, _ = to_windows(samples)
    reconstructed = model.predict(windows, verbose=0, batch_size=64)
    window_hi = np.mean((windows - reconstructed) ** 2, axis=1)
    return float(aggregate(window_hi, windows_per_sample, "p95")[0]), used_sample_idx


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stats = load_ffp_stats()
    rows = []

    for dataset, meta in TABLE2_XJTU.items():
        model_path = MODEL_DIR / f"{dataset}.windowed_ms2ae_autoencoder.keras"
        model = tf.keras.models.load_model(model_path, compile=False)
        threshold = stats[dataset]["threshold"]
        healthy_max = stats[dataset]["healthy_max"]
        hi_max = stats[dataset]["hi_max"]

        for sample_idx, paper_stage in meta["samples"]:
            hi, used_sample_idx = sample_recon_p95(model, dataset, sample_idx)
            detected_stage, medium_thr, last_thr = classify_stage(hi, threshold, healthy_max)
            rows.append({
                "dataset": dataset,
                "paper_ffp": meta["paper_ffp"],
                "detected_ffp_recon_p95": stats[dataset]["detected_ffp"],
                "sample": sample_idx,
                "used_loader_index": used_sample_idx,
                "paper_stage": paper_stage,
                "project_stage_recon_p95": detected_stage,
                "recon_p95_hi": hi,
                "project_threshold": threshold,
                "project_healthy_max": healthy_max,
                "project_medium_threshold": medium_thr,
                "project_last_threshold": last_thr,
                "project_hi_max": hi_max,
            })
        del model
        tf.keras.backend.clear_session()

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# XJTU Table 2 Stage Comparison",
        "",
        "Stage thresholds use Eqs. (5)-(6) on reconstruction-HI:",
        "",
        "`T_MD = (max(HI_healthy) - P95(HI_healthy)) * 50`",
        "",
        "`T_LD = (max(HI_healthy) - P95(HI_healthy)) * 100`",
        "",
        "| Dataset | Sample | Paper Stage | Project Stage | Recon HI | Project FFP | Paper FFP |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['sample']} | {row['paper_stage']} | "
            f"{row['project_stage_recon_p95']} | {row['recon_p95_hi']:.6f} | "
            f"{row['detected_ffp_recon_p95']} | {row['paper_ffp']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {OUT_JSON}")
    print(f"[SAVED] {OUT_MD}")


if __name__ == "__main__":
    main()
