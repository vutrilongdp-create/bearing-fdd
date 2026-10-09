import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as scipy_kurtosis, skew

import enter_utils


DATASET = "IMS2"
FS = 20480.0
SHAFT = 33.33
BPFO = 236.0
BPFI = 297.0
BSF = 278.0
FTF = 15.0
HEALTHY_SAMPLES = 300
WINDOW_SIZE = 2048
REPRESENTATIVE_SAMPLES = [536, 871, 979]
REPRESENTATIVE_LABELS = ["Early stage", "Medium stage", "Last stage"]

MODEL_DIR = Path("prog_analizador/models")
OUT_DIR = Path("report_results/ims2_case_study_figures")
SUMMARY_PATH = OUT_DIR / "ims2_case_study_summary.json"


sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    },
)
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def read_rows(dataset, start, count):
    data, _ = enter_utils.getDataset(dataset, count, start)
    return np.asarray(data, dtype=np.float32)


def count_csv_rows(path):
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def to_windows(samples):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples, input_dim = samples.shape
    usable_dim = (input_dim // WINDOW_SIZE) * WINDOW_SIZE
    samples = samples[:, :usable_dim]
    windows_per_sample = usable_dim // WINDOW_SIZE
    return samples.reshape(n_samples * windows_per_sample, WINDOW_SIZE), windows_per_sample


def train_or_load_ms2ae():
    encoder_path = MODEL_DIR / "IMS2.windowed_ms2ae_encoder.keras"
    autoencoder_path = MODEL_DIR / "IMS2.windowed_ms2ae_autoencoder.keras"
    if encoder_path.exists() and autoencoder_path.exists():
        print("[MODEL] Loading cached IMS2 windowed MS2AE", flush=True)
        encoder = tf.keras.models.load_model(encoder_path, compile=False)
        autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
        return autoencoder, encoder

    print("[MODEL] Training IMS2 windowed MS2AE", flush=True)
    healthy = read_rows(DATASET, 0, HEALTHY_SAMPLES)
    windows, _ = to_windows(healthy)
    autoencoder, encoder = enter_utils.build_ms2ae_table1(WINDOW_SIZE)
    autoencoder.fit(windows, windows, epochs=5, batch_size=64, shuffle=True, verbose=2)
    encoder.save(encoder_path)
    autoencoder.save(autoencoder_path)
    return autoencoder, encoder


def compute_reconstruction_hi(autoencoder, samples):
    windows, windows_per_sample = to_windows(samples)
    reconstructed = autoencoder.predict(windows, verbose=0, batch_size=128)
    window_hi = np.mean((windows - reconstructed) ** 2, axis=1)
    return np.percentile(window_hi.reshape(-1, windows_per_sample), 95, axis=1)


def normalize_hi(hi):
    hi = np.asarray(hi, dtype=np.float64)
    lo = np.percentile(hi[:HEALTHY_SAMPLES], 5)
    hi_shifted = hi - lo
    hi_shifted[hi_shifted < 0] = 0
    scale = np.percentile(hi_shifted, 99)
    if scale <= 0:
        scale = np.max(hi_shifted) or 1.0
    return np.clip(hi_shifted / scale, 0, 1.2)


def find_ffp(hi, threshold, start=HEALTHY_SAMPLES, consecutive=5):
    count = 0
    for idx in range(start, len(hi)):
        if hi[idx] >= threshold:
            count += 1
            if count >= consecutive:
                return idx - consecutive + 1
        else:
            count = 0
    return None


def stage_thresholds_paper(healthy_hi):
    p95 = float(np.percentile(healthy_hi, 95))
    mx = float(np.max(healthy_hi))
    delta = mx - p95
    return p95, delta * 50.0, delta * 100.0


def save_fig(fig, name):
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png}", flush=True)
    return str(png), str(svg)


def plot_hi_curve(hi_norm, threshold, ffp):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    x = np.arange(len(hi_norm))
    ax.plot(x, hi_norm, color="#1f4e79", linewidth=1.4, label="Health Index")
    ax.axhline(threshold, color="#b22222", linestyle="--", linewidth=1.2, label=f"P95 threshold = {threshold:.4f}")
    if ffp is not None:
        ax.axvline(ffp, color="#d95f02", linestyle="-.", linewidth=1.2, label=f"Detected FFP = #{ffp}")
    ax.axvline(536, color="#555555", linestyle=":", linewidth=1.2, label="Paper FFP = #536")
    ax.set_title("IMS-2 Health Index")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized HI")
    ax.set_xlim(0, len(hi_norm) - 1)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92)
    return save_fig(fig, "01_ims2_health_index")


def plot_health_stage_division(hi_norm, t_early, t_medium, t_last):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    x = np.arange(len(hi_norm))
    ax.plot(x, hi_norm, color="#1f4e79", linewidth=1.2)
    ax.axhline(t_early, color="#b22222", linestyle="--", linewidth=1.0, label="Early threshold")
    ax.axhline(t_medium, color="#8a6d3b", linestyle="--", linewidth=1.0, label="Medium threshold")
    ax.axhline(t_last, color="#4b0082", linestyle="--", linewidth=1.0, label="Last threshold")
    ax.axvspan(536, 870, color="#fee8c8", alpha=0.45)
    ax.axvspan(871, 978, color="#fdbb84", alpha=0.35)
    ax.axvspan(979, len(hi_norm) - 1, color="#e34a33", alpha=0.18)
    y_top = min(1.15, float(np.nanmax(hi_norm)) * 0.94)
    ax.text(690, y_top, "Early", ha="center", va="top", fontsize=8, color="#6b4e16")
    ax.text(925, y_top, "Medium", ha="center", va="top", fontsize=8, color="#8a3b12")
    ax.text(970, y_top, "Last", ha="center", va="top", fontsize=8, color="#8f1d13")
    for sample, label in zip(REPRESENTATIVE_SAMPLES, REPRESENTATIVE_LABELS):
        ax.axvline(sample, color="#333333", linestyle=":", linewidth=0.9)
        ax.text(sample + 3, 0.03, f"#{sample}", fontsize=7.5, rotation=90, va="bottom")
    ax.set_title("IMS-2 Health Stage Division")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized HI")
    ax.set_xlim(480, 990)
    ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.92)
    return save_fig(fig, "02_ims2_health_stage_division")


def load_baseline_and_sample(sample_index):
    data, denoised = enter_utils.getDataset(DATASET, 1, sample_index)
    return np.asarray(data[0], dtype=np.float32), np.asarray(denoised).flatten().astype(np.float32)


def isolated_signal(sample_index):
    sample, baseline = load_baseline_and_sample(sample_index)
    return enter_utils.differenceSignals(baseline, sample)


def plot_isolated_samples():
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    t = np.arange(20480) / FS
    for ax, sample, label in zip(axes, REPRESENTATIVE_SAMPLES, REPRESENTATIVE_LABELS):
        sig = isolated_signal(sample)
        ax.plot(t, sig, color="#2b6c7f", linewidth=0.55)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Sample #{sample} ({label})", loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Isolated Faulty Samples of IMS-2 Dataset", y=1.01, fontsize=11)
    return save_fig(fig, "03_ims2_isolated_faulty_samples")


def kurtogram_candidate_values(signal):
    bands = enter_utils.generate_1_3_binary_tree_bands(FS)
    rows = []
    nyq = FS / 2.0
    for band in bands:
        if band["level"] not in (2.0, 2.6, 3.0):
            continue
        if band["index"] == 0 or band["index"] == band["parts"] - 1:
            continue
        low, high = band["low"], band["high"]
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.99)
        try:
            b, a = butter(4, [low_n, high_n], btype="band")
            filtered = filtfilt(b, a, signal)
            env = np.abs(hilbert(filtered))
            k_val = scipy_kurtosis(env, fisher=True, nan_policy="omit")
        except Exception:
            k_val = np.nan
        rows.append({**band, "kurtosis": float(k_val)})
    return rows


def plot_kurtograms():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True, constrained_layout=True)
    selected_bands = {}
    for ax, sample, label in zip(axes, REPRESENTATIVE_SAMPLES, REPRESENTATIVE_LABELS):
        sig = isolated_signal(sample)
        rows = kurtogram_candidate_values(sig)
        values = np.array([r["kurtosis"] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        vmin, vmax = (np.min(finite), np.max(finite)) if finite.size else (0, 1)
        cmap = plt.get_cmap("viridis")
        for r in rows:
            color = cmap((r["kurtosis"] - vmin) / (vmax - vmin + 1e-12)) if np.isfinite(r["kurtosis"]) else "lightgray"
            ax.barh(r["level"], r["high"] - r["low"], left=r["low"], height=0.22, color=color, edgecolor="white", linewidth=0.4)
        band = enter_utils.computeKurtogram(sig, FS, 3.0)
        selected_bands[sample] = band
        ax.axvspan(band[0], band[1], facecolor="none", edgecolor="red", linewidth=1.8)
        ax.set_title(f"#{sample} ({label})\n{band[0]:.0f}-{band[1]:.0f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, FS / 2)
        ax.set_yticks([2.0, 2.6, 3.0])
    axes[0].set_ylabel("Kurtogram level")
    fig.suptitle("Kurtograms of Isolated Faulty Samples of IMS-2 Dataset", y=1.04, fontsize=11)
    paths = save_fig(fig, "04_ims2_kurtograms")
    return selected_bands, paths


def plot_filtered_envelope_fft(selected_bands):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    for ax, sample, label in zip(axes, REPRESENTATIVE_SAMPLES, REPRESENTATIVE_LABELS):
        sig = isolated_signal(sample)
        fstart, fend = selected_bands[sample]
        spectrum, freqs = enter_utils.filteredFFT(4, FS, fstart, fend, sig)
        mask = freqs <= 1600
        ax.plot(freqs[mask], spectrum[mask], color="#1f4e79", linewidth=0.8)
        for h in range(1, 7):
            f = BPFO * h
            if f <= 1600:
                ax.axvline(f, color="#b22222", linestyle="--", linewidth=0.8)
                ax.text(f + 8, ax.get_ylim()[1] * 0.75, f"{h}X", rotation=90, fontsize=7, color="#b22222")
        ax.set_title(f"Sample #{sample} ({label}), band {fstart:.0f}-{fend:.0f} Hz", loc="left", fontsize=9)
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("FFT of Filtered Isolated Faulty Samples of IMS-2 Dataset", y=0.985, fontsize=11)
    fig.subplots_adjust(top=0.92, hspace=0.52)
    return save_fig(fig, "05_ims2_filtered_envelope_fft")


def time_domain_features(samples):
    eps = 1e-12
    abs_x = np.abs(samples)
    rms = np.sqrt(np.mean(samples ** 2, axis=1))
    peak = np.max(abs_x, axis=1)
    mean_abs = np.mean(abs_x, axis=1)
    sqrt_abs_mean = np.mean(np.sqrt(abs_x + eps), axis=1)
    features = {
        "RMS": rms,
        "Skewness": skew(samples, axis=1, nan_policy="omit"),
        "Kurtosis": scipy_kurtosis(samples, axis=1, fisher=False, nan_policy="omit"),
        "Crest factor": peak / (rms + eps),
        "Shape factor": rms / (mean_abs + eps),
        "Impulse factor": peak / (mean_abs + eps),
        "Margin factor": peak / ((sqrt_abs_mean ** 2) + eps),
    }
    return features


def amplitude_at(freqs, spectrum, target):
    idx = int(np.argmin(np.abs(freqs - target)))
    return float(spectrum[idx])


def frequency_domain_features(samples, filtered_band):
    labels = {
        "Fund nf": SHAFT,
        "BPFO nf": BPFO,
        "BPFI nf": BPFI,
        "BSF nf": BSF,
        "FTF nf": FTF,
        "Fund f": SHAFT,
        "BPFO f": BPFO,
        "BPFI f": BPFI,
        "BSF f": BSF,
        "FTF f": FTF,
    }
    values = {name: [] for name in labels}
    fstart, fend = filtered_band
    for sample in samples:
        nf_spectrum, nf_freqs = np.abs(enter_utils.get_power_spectrum(sample, FS))
        f_spectrum, f_freqs = enter_utils.filteredFFT(4, FS, fstart, fend, sample)
        for name, target in labels.items():
            if name.endswith("nf"):
                values[name].append(amplitude_at(nf_freqs, nf_spectrum, target))
            else:
                values[name].append(amplitude_at(f_freqs, f_spectrum, target))
    return {name: np.asarray(vals) for name, vals in values.items()}


def correlation_row(hi, features):
    data = {"HI": hi}
    data.update(features)
    names = list(data.keys())
    matrix = np.corrcoef(np.vstack([data[name] for name in names]))
    return names, matrix


def plot_corr_matrix(names, matrix, title, output_name):
    fig, ax = plt.subplots(
        figsize=(max(6.2, len(names) * 0.55), max(4.8, len(names) * 0.45)),
        constrained_layout=True,
    )
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap="vlag", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7.5}, linewidths=0.4,
                cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return save_fig(fig, output_name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    autoencoder, _ = train_or_load_ms2ae()

    total = count_csv_rows("prog_analizador/data/IMS2.csv")
    all_samples = read_rows(DATASET, 0, total)
    print(f"[DATA] IMS2 samples={all_samples.shape}", flush=True)

    hi_raw = compute_reconstruction_hi(autoencoder, all_samples)
    hi_norm = normalize_hi(hi_raw)
    t_early, t_medium, t_last = stage_thresholds_paper(hi_norm[:HEALTHY_SAMPLES])
    ffp = find_ffp(hi_norm, t_early)

    outputs = {}
    outputs["health_index"] = plot_hi_curve(hi_norm, t_early, ffp)
    outputs["health_stage_division"] = plot_health_stage_division(hi_norm, t_early, t_medium, t_last)
    outputs["isolated_faulty_samples"] = plot_isolated_samples()
    selected_bands, kurtogram_paths = plot_kurtograms()
    outputs["kurtograms"] = kurtogram_paths
    outputs["filtered_envelope_fft"] = plot_filtered_envelope_fft(selected_bands)

    features_time = time_domain_features(all_samples)
    time_names, time_matrix = correlation_row(hi_norm, features_time)
    outputs["time_correlation"] = plot_corr_matrix(
        time_names,
        time_matrix,
        "Correlation Matrix between HI and Time-Domain Features (IMS-2)",
        "06_ims2_time_domain_correlation",
    )

    # Use the FFP band for frequency-domain XAI so all samples are compared consistently.
    ffp_band = selected_bands[REPRESENTATIVE_SAMPLES[0]]
    features_freq = frequency_domain_features(all_samples, ffp_band)
    freq_names, freq_matrix = correlation_row(hi_norm, features_freq)
    outputs["frequency_correlation"] = plot_corr_matrix(
        freq_names,
        freq_matrix,
        "Correlation Matrix between HI and Frequency-Domain Features (IMS-2)",
        "07_ims2_frequency_domain_correlation",
    )

    summary = {
        "dataset": DATASET,
        "samples": int(total),
        "window_size": WINDOW_SIZE,
        "representative_samples": REPRESENTATIVE_SAMPLES,
        "project_ffp": ffp,
        "paper_ffp": 536,
        "thresholds": {
            "early": t_early,
            "medium": t_medium,
            "last": t_last,
        },
        "selected_bands": {str(k): [float(v[0]), float(v[1])] for k, v in selected_bands.items()},
        "outputs": outputs,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
