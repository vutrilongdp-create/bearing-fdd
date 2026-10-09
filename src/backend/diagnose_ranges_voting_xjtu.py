import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema

import enter_utils


RESULT_DIR = Path("test_results_windowed_ms2ae")
OUT_JSON = RESULT_DIR / "xjtu_user_table_range_voting_comparison.json"
OUT_MD = RESULT_DIR / "xjtu_user_table_range_voting_comparison.md"

TOP_N_PEAKS = 10
AMPLITUDE_RATIO = 0.05
TOLERANCE_HZ = 5.0
VOTE_RATIO = 0.30

ROWS = [
    ("XJTU2-1", 450, 457, "Medium", {"BPFI", "BSF", "FTF"}, "1X, 2X BPFI - 1X BSF - 5X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU2-1", 455, 460, "Medium", {"BPFI", "BSF", "FTF"}, "1X, 2X BPFI - 1X BSF - 5X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU2-1", 460, 470, "Last", {"BPFI", "BSF", "FTF"}, "1X BPFI - 1X BSF - 2X, 5X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU2-3", 301, 311, "Early", {"BSF", "FTF"}, "1X BSF - 1X, 2X, 4X, 5X, 6X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU2-3", 410, 420, "Medium", {"BPFO", "BPFI", "FTF"}, "1X BPFO - 1X BPFI - 1X, 2X, 4X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU2-3", 525, 532, "Medium", {"BPFO", "BPFI", "BSF", "FTF"}, "1X, 5X, 6X BPFO - 2X BPFI - 1X, 3X BSF - 5X FTF", 37.5, 112.19, 178.94, 75.21, 14.20),
    ("XJTU3-1", 2340, 2360, "Early", {"BPFO", "BSF", "FTF"}, "1X, 2X BPFO - 1X, 3X BSF - 3X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
    ("XJTU3-1", 2440, 2460, "Medium", {"BPFO", "BPFI", "BSF", "FTF"}, "2X, 3X BPFO - 1X BPFI - 1X, 3X BSF - 1X, 3X, 5X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
    ("XJTU3-1", 2520, 2540, "Last", {"BPFO", "BPFI", "BSF", "FTF"}, "1X, 2X, 3X, 5X BPFO - 5X BPFI - 2X, 3X BSF - 6X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
    ("XJTU3-4", 1410, 1425, "Early", {"BPFI", "BSF", "FTF"}, "1X BPFI - 1X BSF - 5X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
    ("XJTU3-4", 1445, 1460, "Medium", {"BPFO", "BPFI", "BSF", "FTF"}, "1X BPFO - 1X BPFI - 1X, 3X BSF - 2X, 5X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
    ("XJTU3-4", 1495, 1510, "Last", {"BPFO", "BPFI", "BSF", "FTF"}, "1X BPFO - 1X, 2X BPFI - 1X BSF - 2X, 5X FTF", 40.0, 123.20, 196.49, 82.58, 15.40),
]


def load_sample(dataset, sample_index):
    samples, denoised_healthy = enter_utils.getDataset(dataset, 1, sample_index)
    if samples.size == 0:
        return None, None
    return samples[0], np.asarray(denoised_healthy).flatten()


def strict_top_peaks(spectrum, freqs):
    peak_indices = argrelextrema(spectrum, np.greater)[0]
    if peak_indices.size == 0:
        return []
    max_amp = float(np.max(spectrum)) if spectrum.size else 0.0
    min_amp = AMPLITUDE_RATIO * max_amp
    peak_indices = np.array([idx for idx in peak_indices if spectrum[idx] >= min_amp])
    if peak_indices.size == 0:
        return []
    amplitudes = spectrum[peak_indices]
    order = np.argsort(amplitudes)[::-1][:TOP_N_PEAKS]
    selected = peak_indices[order]
    return [{"freq": float(freqs[idx]), "amplitude": float(spectrum[idx])} for idx in selected]


def match_harmonics(peaks, base_freq):
    hits = []
    for harmonic in range(1, 7):
        expected = base_freq * harmonic
        candidates = [p for p in peaks if abs(p["freq"] - expected) <= TOLERANCE_HZ]
        if candidates:
            best = max(candidates, key=lambda p: p["amplitude"])
            hits.append((harmonic, best["freq"]))
    return hits


def diagnose_one_sample(dataset, sample_index, fs, bpfo, bpfi, bsf, ftf):
    signal, healthy = load_sample(dataset, sample_index)
    if signal is None:
        return None
    isolated = enter_utils.differenceSignals(healthy, signal)
    band = enter_utils.computeKurtogram(isolated, fs, 3.0)
    fstart, fend = enter_utils.getFilterBands(band, fs, 3)
    envelope_fft, freqs = enter_utils.filteredFFT(4, fs, fstart, fend, isolated)
    peaks = strict_top_peaks(envelope_fft, freqs)
    bases = {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}
    hits = {name: match_harmonics(peaks, base) for name, base in bases.items()}
    detected = {name for name, found in hits.items() if found}
    return {
        "sample": sample_index,
        "bandpass_range": [float(fstart), float(fend)],
        "detected": sorted(detected),
        "harmonics": {name: [h for h, _ in found] for name, found in hits.items()},
    }


def summarize_harmonics(harmonic_votes):
    parts = []
    for fault in ("BPFO", "BPFI", "BSF", "FTF"):
        if fault not in harmonic_votes or not harmonic_votes[fault]:
            continue
        labels = ", ".join(f"{h}X" for h, _ in harmonic_votes[fault].most_common())
        parts.append(f"{labels} {fault}")
    return " - ".join(parts) if parts else "No stable harmonic"


def compare_status(expected, actual):
    if expected == actual:
        return "MATCH"
    if expected and expected.issubset(actual):
        return "OVER-DETECTED"
    if actual and actual.issubset(expected):
        return "PARTIAL"
    if expected & actual:
        return "MIXED"
    return "MISS"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, start, end, stage, expected_faults, expected_diag, shaft, bpfo, bpfi, bsf, ftf in ROWS:
        print(f"[RANGE] {dataset} #{start}-#{end}", flush=True)
        sample_results = []
        fault_votes = Counter()
        harmonic_votes = defaultdict(Counter)
        for sample in range(start, end + 1):
            result = diagnose_one_sample(dataset, sample, 25600.0, bpfo, bpfi, bsf, ftf)
            if result is None:
                continue
            sample_results.append(result)
            for fault in result["detected"]:
                fault_votes[fault] += 1
                for harmonic in result["harmonics"].get(fault, []):
                    harmonic_votes[fault][harmonic] += 1

        n_valid = len(sample_results)
        min_votes = max(1, int(np.ceil(VOTE_RATIO * n_valid)))
        actual_faults = {fault for fault, count in fault_votes.items() if count >= min_votes}
        stable_harmonics = {
            fault: Counter({h: c for h, c in harmonic_votes[fault].items() if c >= min_votes})
            for fault in actual_faults
        }
        actual_diag = summarize_harmonics(stable_harmonics)
        results.append({
            "dataset": dataset,
            "range": f"#{start}-#{end}",
            "stage": stage,
            "valid_samples": n_valid,
            "vote_threshold": min_votes,
            "expected_faults": sorted(expected_faults),
            "project_faults": sorted(actual_faults),
            "status": compare_status(expected_faults, actual_faults),
            "expected_diagnosis": expected_diag,
            "project_diagnosis": actual_diag,
            "fault_vote_counts": dict(fault_votes),
            "sample_results": sample_results,
        })

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# XJTU Range Voting Comparison",
        "",
        f"Rules: top {TOP_N_PEAKS} peaks, amplitude >= {AMPLITUDE_RATIO:.0%} of max, harmonic tolerance ±{TOLERANCE_HZ} Hz, fault vote >= {VOTE_RATIO:.0%} samples.",
        "",
        "| Dataset | Range | Stage | Expected faults | Project faults | Status | Votes | Project diagnosis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['range']} | {row['stage']} | "
            f"{', '.join(row['expected_faults'])} | {', '.join(row['project_faults']) or 'None'} | "
            f"{row['status']} | {row['fault_vote_counts']} | {row['project_diagnosis']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {OUT_JSON}", flush=True)
    print(f"[SAVED] {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
