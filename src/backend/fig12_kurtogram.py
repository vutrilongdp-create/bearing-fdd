import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis

try:
    from fastdtw import fastdtw
except Exception:
    fastdtw = None


FS = 20480.0
DATA_DIR = Path("prog_analizador/data")
OUT_DIR = Path("report_results/ims2_paper_form_visuals")
SAMPLES = [532, 871, 979]
LABELS = ["Early stage", "Medium stage", "Last stage"]


sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    },
)
plt.rcParams.update({"font.family": "Times New Roman"})


def read_csv_rows(path, start=0, count=None):
    end = None if count is None else start + int(count)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = list(itertools.islice(handle, int(start), end))
    data = np.loadtxt(lines, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def difference_signals(baseline, sample):
    baseline = np.asarray(baseline, dtype=np.float32).flatten()
    sample = np.asarray(sample, dtype=np.float32).flatten()
    if baseline.size == 0:
        return sample
    if baseline.size != sample.size:
        fixed = np.zeros(sample.size, dtype=np.float32)
        n = min(baseline.size, sample.size)
        fixed[:n] = baseline[:n]
        fixed[n:] = baseline[n - 1]
        baseline = fixed
    if fastdtw is None:
        return sample - baseline
    try:
        _, path = fastdtw(baseline, sample, dist=lambda x, y: abs(x - y))
        aligned = np.zeros(sample.size, dtype=np.float32)
        counts = np.zeros(sample.size, dtype=np.float32)
        for x, y in path:
            if 0 <= x < baseline.size and 0 <= y < sample.size:
                aligned[y] += baseline[x]
                counts[y] += 1
        missing = counts == 0
        counts[missing] = 1
        aligned = aligned / counts
        aligned[missing] = baseline[missing]
        return sample - aligned
    except Exception:
        return sample - baseline


def isolated_signal(sample_index):
    sample = read_csv_rows(DATA_DIR / "IMS2.csv", sample_index, 1)[0]
    baseline = read_csv_rows(DATA_DIR / "healthyIMS2.csv", 0, None).flatten()
    return difference_signals(baseline, sample)


def generate_1_3_binary_tree_bands(fs):
    nyq = fs / 2.0
    level_parts = [(0.0, 1), (1.0, 2), (1.6, 3), (2.0, 4), (2.6, 6), (3.0, 8)]
    bands = []
    for level, parts in level_parts:
        width = nyq / parts
        for index in range(parts):
            bands.append({
                "level": level,
                "index": index,
                "parts": parts,
                "low": index * width,
                "high": (index + 1) * width,
            })
    return bands


def kurtosis_for_band(signal, low, high):
    nyq = FS / 2.0
    try:
        if low <= 0 and high >= nyq:
            filtered = signal
        elif low <= 0:
            b, a = butter(4, min(high / nyq, 0.99), btype="lowpass")
            filtered = filtfilt(b, a, signal)
        elif high >= nyq:
            b, a = butter(4, max(low / nyq, 0.001), btype="highpass")
            filtered = filtfilt(b, a, signal)
        else:
            b, a = butter(4, [max(low / nyq, 0.001), min(high / nyq, 0.99)], btype="band")
            filtered = filtfilt(b, a, signal)
        env = np.abs(hilbert(filtered))
        return float(kurtosis(env, fisher=True, nan_policy="omit"))
    except Exception:
        return np.nan


def compute_selected_band(signal):
    best_kurt = -np.inf
    best_band = (FS / 8.0, FS / 4.0)
    for band in generate_1_3_binary_tree_bands(FS):
        if band["level"] not in (2.0, 2.6, 3.0):
            continue
        if band["index"] == 0 or band["index"] == band["parts"] - 1:
            continue
        k_val = kurtosis_for_band(signal, band["low"], band["high"])
        if np.isfinite(k_val) and k_val > best_kurt:
            best_kurt = k_val
            best_band = (band["low"], band["high"])
    return best_band


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {svg}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55), sharey=True, constrained_layout=True)
    display_levels = [0.0, 1.0, 1.6, 2.0, 2.6, 3.0]
    level_to_row = {level: idx for idx, level in enumerate(display_levels)}
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, sample, label, panel in zip(axes, SAMPLES, LABELS, panel_labels):
        sig = isolated_signal(sample)
        bands = generate_1_3_binary_tree_bands(FS)
        rows = []
        for band in bands:
            rows.append({**band, "kurtosis": kurtosis_for_band(sig, band["low"], band["high"])})

        values = np.array([row["kurtosis"] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        vmin, vmax = (np.min(finite), np.max(finite)) if finite.size else (0.0, 1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("jet")

        for row in rows:
            y0 = level_to_row[row["level"]]
            color = cmap(norm(row["kurtosis"])) if np.isfinite(row["kurtosis"]) else "lightgray"
            ax.add_patch(Rectangle((row["low"], y0), row["high"] - row["low"], 1.0,
                                   facecolor=color, edgecolor=color, linewidth=0.0))

        selected = compute_selected_band(sig)
        selected_row = min(rows, key=lambda r: abs(r["low"] - selected[0]) + abs(r["high"] - selected[1]))
        y0 = level_to_row[selected_row["level"]]
        ax.add_patch(Rectangle((selected_row["low"], y0), selected_row["high"] - selected_row["low"], 1.0,
                               facecolor="none", edgecolor="#d62728", linewidth=1.3))

        ax.set_title(f"{panel} Faulty sample #{sample}", fontsize=10, y=-0.34)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_xlim(0, FS / 2)
        ax.set_ylim(0, len(display_levels))
        ax.set_yticks([level_to_row[level] + 0.5 for level in display_levels])
        ax.set_yticklabels(["0", "1", "1.6", "2", "2.6", "3"])
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(False)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Kurtogram level", fontsize=10)
    save_fig(fig, "fig12_kurtograms_ims2_project")


if __name__ == "__main__":
    main()
