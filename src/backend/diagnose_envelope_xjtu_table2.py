import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from scipy.signal import argrelextrema

import enter_utils


RESULT_DIR = Path("test_results_windowed_ms2ae")
OUT_JSON = RESULT_DIR / "xjtu_table2_envelope_diagnosis.json"
OUT_MD = RESULT_DIR / "xjtu_table2_envelope_diagnosis.md"

TABLE2_XJTU = {
    "XJTU2-1": {
        "fs": 25600.0,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "samples": [451, 460, 486],
    },
    "XJTU2-3": {
        "fs": 25600.0,
        "BPFO": 112.19,
        "BPFI": 178.94,
        "BSF": 75.21,
        "FTF": 14.20,
        "samples": [301, 419, 532],
    },
    "XJTU3-1": {
        "fs": 25600.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "samples": [2347, 2445, 2533],
    },
    "XJTU3-4": {
        "fs": 25600.0,
        "BPFO": 123.20,
        "BPFI": 196.49,
        "BSF": 82.58,
        "FTF": 15.40,
        "samples": [1416, 1453, 1505],
    },
}


def load_sample(dataset, sample_index):
    samples, denoised_healthy = enter_utils.getDataset(dataset, 1, sample_index)
    used_index = sample_index
    if samples.size == 0:
        used_index = sample_index - 1
        samples, denoised_healthy = enter_utils.getDataset(dataset, 1, used_index)
    if samples.size == 0:
        raise ValueError(f"Cannot load sample {sample_index} for {dataset}")
    return samples[0], np.asarray(denoised_healthy).flatten(), used_index


def top_peak_frequencies(spectrum, freqs, n_peaks=20):
    peak_indices = argrelextrema(spectrum, np.greater)[0]
    if peak_indices.size == 0:
        return []
    amplitudes = spectrum[peak_indices]
    order = np.argsort(amplitudes)[::-1][:n_peaks]
    selected = peak_indices[order]
    return [
        {"freq": float(freqs[idx]), "amplitude": float(spectrum[idx])}
        for idx in selected
    ]


def match_harmonics(peaks, base_freq, tolerance_hz=5.0, max_harmonic=6):
    matches = []
    for harmonic in range(1, max_harmonic + 1):
        expected = base_freq * harmonic
        candidates = [
            peak for peak in peaks
            if abs(peak["freq"] - expected) <= tolerance_hz
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda peak: peak["amplitude"])
        matches.append({
            "harmonic": harmonic,
            "expected_freq": float(expected),
            "detected_freq": best["freq"],
            "amplitude": best["amplitude"],
        })
    return matches


def diagnose_sample(dataset, sample_index, meta):
    signal, healthy_baseline, used_index = load_sample(dataset, sample_index)
    isolated = enter_utils.differenceSignals(healthy_baseline, signal)

    kurtogram = enter_utils.computeKurtogram(isolated, meta["fs"], 3.0)
    fstart, fend = enter_utils.getFilterBands(kurtogram, meta["fs"], 3)
    envelope_fft, freqs = enter_utils.filteredFFT(4, meta["fs"], fstart, fend, isolated)
    peaks = top_peak_frequencies(envelope_fft, freqs, n_peaks=20)

    fault_bases = {
        "BPFO": meta["BPFO"],
        "BPFI": meta["BPFI"],
        "BSF": meta["BSF"],
        "FTF": meta["FTF"],
    }
    matches = {
        name: match_harmonics(peaks, freq)
        for name, freq in fault_bases.items()
    }
    detected = [name for name, hits in matches.items() if hits]

    return {
        "dataset": dataset,
        "sample": sample_index,
        "used_loader_index": used_index,
        "bandpass_range": [float(fstart), float(fend)],
        "detected_fault_frequencies": detected,
        "harmonics": matches,
        "top_peaks": peaks[:10],
    }


def format_harmonics(matches):
    parts = []
    for name, hits in matches.items():
        if hits:
            labels = ", ".join(f"{hit['harmonic']}X" for hit in hits)
            parts.append(f"{labels} {name}")
    return " - ".join(parts) if parts else "No BPFO/BPFI/BSF/FTF harmonic"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, meta in TABLE2_XJTU.items():
        for sample in meta["samples"]:
            print(f"[DIAG] {dataset} sample={sample}", flush=True)
            rows.append(diagnose_sample(dataset, sample, meta))

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# XJTU Table 2 Envelope Diagnosis",
        "",
        "Pipeline: isolated faulty sample -> Kurtogram band -> bandpass -> Hilbert envelope -> FFT -> BPFO/BPFI/BSF/FTF harmonic matching.",
        "",
        "| Dataset | Sample | Bandpass range | Detected harmonics |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        low, high = row["bandpass_range"]
        lines.append(
            f"| {row['dataset']} | {row['sample']} | "
            f"{low:.2f}-{high:.2f} Hz | "
            f"{format_harmonics(row['harmonics'])} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {OUT_JSON}", flush=True)
    print(f"[SAVED] {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
