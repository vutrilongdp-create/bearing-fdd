import itertools
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert

try:
    from fastdtw import fastdtw
except Exception:
    fastdtw = None


FS = 20480.0
BPFO = 236.0
DATA_DIR = Path("prog_analizador/data")
OUT_DIR = Path("report_results/ims2_paper_form_visuals")

SAMPLES = [532, 871, 979]
STAGES = ["Early stage", "Medium stage", "Last stage"]
BANDS = {
    532: (2560.0, 5120.0),
    871: (5120.0, 7680.0),
    979: (5120.0, 7680.0),
}
ANNOTATED_HARMONICS = {
    532: [1, 2],
    871: [1, 2, 3, 4, 5],
    979: [1, 2, 3, 4, 6],
}
LABEL_OFFSETS = {
    532: {
        1: (18, 0.12),
        2: (18, 0.16),
    },
    871: {
        1: (18, 0.10),
        2: (18, 0.16),
        3: (18, 0.12),
        4: (18, 0.16),
        5: (18, 0.12),
    },
    979: {
        1: (18, 0.10),
        2: (18, 0.16),
        3: (18, 0.12),
        4: (18, 0.16),
        6: (-92, 0.12),
    },
}


sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 120,
        "savefig.dpi": 220,
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


def amplitude_spectrum(signal):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    windowed = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed)) / signal.size
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / FS)
    spectrum[0:5] = 0
    return spectrum, freqs


def filtered_envelope_spectrum(signal, low_freq, high_freq):
    nyq = FS / 2.0
    low = max(low_freq / nyq, 0.001)
    high = min(high_freq / nyq, 0.99)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return amplitude_spectrum(envelope)


def harmonic_peak(freqs, spectrum, harmonic, tolerance=8.0):
    target = harmonic * BPFO
    mask = (freqs >= target - tolerance) & (freqs <= target + tolerance)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs - target)))
        return target, freqs[idx], spectrum[idx]
    local_indices = np.where(mask)[0]
    idx = local_indices[np.argmax(spectrum[local_indices])]
    return target, freqs[idx], spectrum[idx]


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    gc.collect()
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {svg}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), sharex=True)
    panel_labels = ["(a)", "(b)", "(c)"]
    detected = {}

    for ax, sample, stage, panel in zip(axes, SAMPLES, STAGES, panel_labels):
        sig = isolated_signal(sample)
        low, high = BANDS[sample]
        nf_spectrum, nf_freqs = amplitude_spectrum(sig)
        f_spectrum, f_freqs = filtered_envelope_spectrum(sig, low, high)

        mask_nf = nf_freqs <= 1600
        mask_f = f_freqs <= 1600
        nf_plot = nf_spectrum[mask_nf]
        f_plot = f_spectrum[mask_f]
        nf_freq_plot = nf_freqs[mask_nf]
        f_freq_plot = f_freqs[mask_f]

        ax.plot(nf_freq_plot, nf_plot, color="#00cc22", linewidth=0.55,
                label="FFT of non-filtered sample")
        ax.plot(f_freq_plot, f_plot, color="#1f4eeb", linewidth=0.75,
                label="FFT of filtered sample")

        max_filtered = float(np.max(f_plot)) if f_plot.size else 0.0
        max_combined = max(float(np.max(nf_plot)), max_filtered) if nf_plot.size else max_filtered
        sample_detected = []

        for harmonic in ANNOTATED_HARMONICS[sample]:
            target, peak_freq, peak_amp = harmonic_peak(f_freqs, f_spectrum, harmonic)
            if target > 1600:
                continue
            sample_detected.append(f"{harmonic}X BPFO")
            ax.plot(peak_freq, peak_amp, marker="o", markersize=4.0,
                    markerfacecolor="none", markeredgecolor="red", markeredgewidth=0.9)
            x_offset, y_fraction = LABEL_OFFSETS[sample][harmonic]
            y_offset = max_combined * y_fraction
            y_text = min(peak_amp + y_offset, max_combined * 0.92)
            ax.text(peak_freq + x_offset, y_text, f"{harmonic}XBPFO",
                    color="red", fontsize=6.6, ha="left", va="bottom")

        detected[str(sample)] = sample_detected
        ax.set_xlim(0, 1600)
        ax.set_ylim(0, max_combined * 1.12 if max_combined > 0 else 1.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7.5)
        ax.grid(True, linestyle=(0, (5, 5)), linewidth=0.65, color="#d4d4d4")
        ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=1.0, borderpad=0.4)

    fig.subplots_adjust(wspace=0.28, bottom=0.36, left=0.065, right=0.985, top=0.94)
    subcaptions = [
        "(a) Faulty sample #532",
        "(b) Faulty sample #871",
        "(c) Faulty sample #979",
    ]
    for ax, caption in zip(axes, subcaptions):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2.0, 0.185, caption,
                 ha="center", va="center", fontsize=8)
    fig.text(
        0.5,
        0.065,
        "Fig. 13. FFT of the filtered isolated faulty samples of IMS-2 dataset.",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    save_fig(fig, "fig13_fft_paper_style_ims2_project")

    for sample, labels in detected.items():
        print(f"[DETECTED] #{sample}: {', '.join(labels)}")


if __name__ == "__main__":
    main()
