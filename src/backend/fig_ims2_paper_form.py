import json
import gc
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis as scipy_kurtosis, probplot, skew

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
MEDIUM_SAMPLE = 871
LAST_SAMPLE = 979

MODEL_DIR = Path("prog_analizador/models")
OUT_DIR = Path("report_results/ims2_paper_form_visuals")
SUMMARY_PATH = OUT_DIR / "ims2_paper_form_summary.json"


sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    rc={
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    },
)
plt.rcParams.update({
    "font.family": "Times New Roman",
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
    window_hi = np.empty(windows.shape[0], dtype=np.float32)
    chunk_size = 256
    for start in range(0, windows.shape[0], chunk_size):
        end = min(start + chunk_size, windows.shape[0])
        batch = windows[start:end]
        reconstructed = autoencoder.predict(batch, verbose=0, batch_size=64)
        window_hi[start:end] = np.mean((batch - reconstructed) ** 2, axis=1)
        del reconstructed
        gc.collect()
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
    gc.collect()
    print(f"[SAVED] {png}", flush=True)
    return str(png), str(svg)


def load_baseline_and_sample(sample_index):
    data, denoised = enter_utils.getDataset(DATASET, 1, sample_index)
    return np.asarray(data[0], dtype=np.float32), np.asarray(denoised).flatten().astype(np.float32)


def isolated_signal(sample_index):
    sample, baseline = load_baseline_and_sample(sample_index)
    return enter_utils.differenceSignals(baseline, sample)


def plot_fig08_hi_values(hi_norm, threshold, ffp):
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    x_healthy = np.arange(HEALTHY_SAMPLES)
    x_unlabeled = np.arange(HEALTHY_SAMPLES, len(hi_norm))
    ax.plot(x_healthy, hi_norm[:HEALTHY_SAMPLES], color="#2ca25f", linewidth=1.2, label="Healthy samples")
    ax.plot(x_unlabeled, hi_norm[HEALTHY_SAMPLES:], color="#2b6cb0", linewidth=1.2, label="Unlabeled samples")
    ax.axhline(threshold, color="#b22222", linestyle="--", linewidth=0.9, label=f"P95 = {threshold:.4f}")
    ax.axvline(ffp, color="#d95f02", linestyle="-.", linewidth=1.0, label=f"FFP = #{ffp}")
    ax.set_xlim(0, len(hi_norm) - 1)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized HI")
    ax.set_title("Fig. 8. Health index values of IMS-2 dataset")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    return save_fig(fig, "fig08_health_index_values_ims2_project")


def plot_fig09_qq(healthy_hi):
    fig, ax = plt.subplots(figsize=(4.3, 3.6), constrained_layout=True)
    osm, osr = probplot(healthy_hi, dist="norm", fit=False)
    slope, intercept = np.polyfit(osm, osr, 1)
    ax.scatter(osm, osr, s=13, facecolor="#2b6cb0", edgecolor="white", linewidth=0.25)
    line_x = np.asarray([np.min(osm), np.max(osm)])
    ax.plot(line_x, intercept + slope * line_x, color="#b22222", linewidth=1.0)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered HI values")
    ax.set_title("Fig. 9. Normal Q-Q plot of healthy IMS-2 samples")
    return save_fig(fig, "fig09_normal_qq_healthy_ims2_project")


def plot_fig10_health_stage(hi_norm, thresholds, ffp, representative_samples):
    t_early, t_medium, t_last = thresholds
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(len(hi_norm))
    ax.plot(x, hi_norm, color="#1f4e79", linewidth=1.15)
    ax.axhline(t_early, color="#b22222", linestyle="--", linewidth=0.95, label="Healthy/fault threshold")
    ax.axhline(t_medium, color="#8a6d3b", linestyle="--", linewidth=0.95, label="Medium threshold")
    ax.axhline(t_last, color="#4b0082", linestyle="--", linewidth=0.95, label="Last threshold")
    ax.axvspan(ffp, MEDIUM_SAMPLE - 1, color="#fee8c8", alpha=0.45)
    ax.axvspan(MEDIUM_SAMPLE, LAST_SAMPLE - 1, color="#fdbb84", alpha=0.33)
    ax.axvspan(LAST_SAMPLE, len(hi_norm) - 1, color="#e34a33", alpha=0.18)
    ax.text((ffp + MEDIUM_SAMPLE) / 2, 1.08, "Early", ha="center", va="center", color="#6b4e16")
    ax.text((MEDIUM_SAMPLE + LAST_SAMPLE) / 2, 1.08, "Medium", ha="center", va="center", color="#8a3b12")
    ax.text(978, 1.08, "Last", ha="right", va="center", color="#8f1d13")
    for sample in representative_samples:
        ax.axvline(sample, color="#333333", linestyle=":", linewidth=0.9)
        ax.text(sample + 2, 0.03, f"#{sample}", rotation=90, va="bottom", fontsize=7.5)
    ax.set_xlim(480, 990)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized HI")
    ax.set_title("Fig. 10. Health stage division of IMS-2 dataset")
    ax.legend(loc="upper left", ncol=3, frameon=True)
    return save_fig(fig, "fig10_health_stage_division_ims2_project")


def plot_fig11_isolated_samples(representative_samples, representative_labels):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.65), sharex=True)
    t = np.arange(20480) / FS
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, sample, label, panel in zip(axes, representative_samples, representative_labels, panel_labels):
        sig = isolated_signal(sample)
        ax.plot(t, sig, color="#2b6c7f", linewidth=0.50)
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{panel} Sample #{sample} ({label})", loc="left", fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(top=0.96, hspace=0.38)
    return save_fig(fig, "fig11_isolated_faulty_samples_ims2_project")


def kurtosis_for_display_band(signal, low, high):
    nyq = FS / 2.0
    try:
        if low <= 0 and high >= nyq:
            filtered = signal
        elif low <= 0:
            high_n = min(high / nyq, 0.99)
            b, a = butter(4, high_n, btype="lowpass")
            filtered = filtfilt(b, a, signal)
        elif high >= nyq:
            low_n = max(low / nyq, 0.001)
            b, a = butter(4, low_n, btype="highpass")
            filtered = filtfilt(b, a, signal)
        else:
            low_n = max(low / nyq, 0.001)
            high_n = min(high / nyq, 0.99)
            b, a = butter(4, [low_n, high_n], btype="band")
            filtered = filtfilt(b, a, signal)
        env = np.abs(hilbert(filtered))
        return float(scipy_kurtosis(env, fisher=True, nan_policy="omit"))
    except Exception:
        return np.nan


def kurtogram_display_values(signal):
    bands = enter_utils.generate_1_3_binary_tree_bands(FS)
    rows = []
    for band in bands:
        low, high = band["low"], band["high"]
        k_val = kurtosis_for_display_band(signal, low, high)
        rows.append({**band, "kurtosis": float(k_val)})
    return rows


def plot_fig12_kurtograms(representative_samples, representative_labels):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55), sharey=True, constrained_layout=True)
    selected_bands = {}
    panel_labels = ["(a)", "(b)", "(c)"]
    display_levels = [0.0, 1.0, 1.6, 2.0, 2.6, 3.0]
    level_to_row = {level: idx for idx, level in enumerate(display_levels)}
    for ax, sample, label, panel in zip(axes, representative_samples, representative_labels, panel_labels):
        sig = isolated_signal(sample)
        rows = kurtogram_display_values(sig)
        values = np.array([r["kurtosis"] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        vmin, vmax = (np.min(finite), np.max(finite)) if finite.size else (0.0, 1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("jet")
        for r in rows:
            color = cmap(norm(r["kurtosis"])) if np.isfinite(r["kurtosis"]) else "lightgray"
            y0 = level_to_row[r["level"]]
            ax.add_patch(
                Rectangle(
                    (r["low"], y0),
                    r["high"] - r["low"],
                    1.0,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.0,
                )
            )
        band = enter_utils.computeKurtogram(sig, FS, 3.0)
        selected_bands[sample] = band
        selected_row = min(
            rows,
            key=lambda r: abs(r["low"] - band[0]) + abs(r["high"] - band[1]),
        )
        selected_y0 = level_to_row[selected_row["level"]]
        ax.add_patch(
            Rectangle(
                (selected_row["low"], selected_y0),
                selected_row["high"] - selected_row["low"],
                1.0,
                facecolor="none",
                edgecolor="#d62728",
                linewidth=1.25,
            )
        )
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
    paths = save_fig(fig, "fig12_kurtograms_ims2_project")
    return selected_bands, paths


def plot_fig13_filtered_fft(representative_samples, representative_labels, selected_bands):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.45), sharex=True)
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, sample, label, panel in zip(axes, representative_samples, representative_labels, panel_labels):
        sig = isolated_signal(sample)
        fstart, fend = selected_bands[sample]
        spectrum, freqs = enter_utils.filteredFFT(4, FS, fstart, fend, sig)
        mask = freqs <= 1600
        ax.plot(freqs[mask], spectrum[mask], color="#1f4e79", linewidth=0.78)
        for h in range(1, 7):
            freq = BPFO * h
            if freq <= 1600:
                ax.axvline(freq, color="#b22222", linestyle="--", linewidth=0.82)
                ax.text(freq + 8, ax.get_ylim()[1] * 0.72, f"{h}X", rotation=90,
                        fontsize=7, color="#b22222")
        ax.set_title(f"{panel} Sample #{sample} ({label}), band {fstart:.0f}-{fend:.0f} Hz",
                     loc="left", fontsize=9)
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Fig. 13. FFT of filtered isolated faulty samples of IMS-2 dataset",
                 y=0.99, fontsize=11)
    fig.subplots_adjust(top=0.90, hspace=0.56)
    return save_fig(fig, "fig13_filtered_fft_ims2_project")


def time_domain_features(samples):
    eps = 1e-12
    abs_x = np.abs(samples)
    rms = np.sqrt(np.mean(samples ** 2, axis=1))
    peak = np.max(abs_x, axis=1)
    mean_abs = np.mean(abs_x, axis=1)
    sqrt_abs_mean = np.mean(np.sqrt(abs_x + eps), axis=1)
    return {
        "RMS": rms,
        "Sk": skew(samples, axis=1, nan_policy="omit"),
        "K": scipy_kurtosis(samples, axis=1, fisher=False, nan_policy="omit"),
        "CF": peak / (rms + eps),
        "SF": rms / (mean_abs + eps),
        "IF": peak / (mean_abs + eps),
        "MF": peak / ((sqrt_abs_mean ** 2) + eps),
    }


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


def corr_vector(hi, features):
    vals = []
    names = []
    for name, data in features.items():
        if np.std(data) == 0 or np.std(hi) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(hi, data)[0, 1]
        names.append(name)
        vals.append(corr)
    return names, np.asarray(vals, dtype=float)


def cluster_corr_matrix(hi, features, start_index, n_clusters=16):
    indices = np.arange(start_index, len(hi))
    clusters = np.array_split(indices, n_clusters)
    names = list(features.keys())
    matrix = np.zeros((len(names), n_clusters), dtype=float)
    for c_idx, cluster in enumerate(clusters):
        hi_c = hi[cluster]
        for f_idx, name in enumerate(names):
            data_c = features[name][cluster]
            if len(cluster) < 3 or np.std(data_c) == 0 or np.std(hi_c) == 0:
                matrix[f_idx, c_idx] = np.nan
            else:
                matrix[f_idx, c_idx] = np.corrcoef(hi_c, data_c)[0, 1]
    return names, matrix


def plot_correlation_paper_form(feature_names, overall_corr, cluster_matrix, title, output_name):
    fig = plt.figure(figsize=(10.4, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        overall_corr.reshape(1, -1),
        ax=ax1,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar=False,
        xticklabels=feature_names,
        yticklabels=["HI"],
        linewidths=0.4,
        annot_kws={"fontsize": 7.5},
    )
    ax1.set_title("(a) Overall correlation", fontsize=9)
    ax1.tick_params(axis="x", rotation=45)

    cluster_labels = [str(i + 1) for i in range(cluster_matrix.shape[1])]
    sns.heatmap(
        cluster_matrix,
        ax=ax2,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        annot=False,
        cbar_kws={"label": "Pearson r"},
        xticklabels=cluster_labels,
        yticklabels=feature_names,
        linewidths=0.35,
    )
    ax2.set_title("(b) Faulty-sample cluster correlation", fontsize=9)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Feature")
    fig.suptitle(title, y=1.02, fontsize=11)
    return save_fig(fig, output_name)


def write_caption_note(summary):
    lines = [
        "# IMS-2 Paper-Form Visuals (Project Results)",
        "",
        "Bộ hình này bám theo chuỗi Fig. 8-Fig. 15 trong bài báo, nhưng sử dụng kết quả tính toán từ project hiện tại.",
        "",
        f"- Dataset: IMS-2, {summary['samples']} samples.",
        f"- Healthy samples for MS2AE training: first {HEALTHY_SAMPLES} samples.",
        f"- Project FFP: sample #{summary['project_ffp']} (paper reported #536).",
        f"- Early threshold P95: {summary['thresholds']['early']:.6f}.",
        f"- Medium threshold: {summary['thresholds']['medium']:.6f}.",
        f"- Last threshold: {summary['thresholds']['last']:.6f}.",
        "",
        "Selected samples and project-selected bandpass ranges:",
    ]
    for sample, band in summary["selected_bands"].items():
        lines.append(f"- Sample #{sample}: {band[0]:.2f}-{band[1]:.2f} Hz.")
    lines.extend([
        "",
        "Suggested captions:",
        "- Fig. 8. Health index values of IMS-2 dataset obtained by the proposed implementation.",
        "- Fig. 9. Normal Q-Q plot of the HI values computed from healthy IMS-2 samples.",
        "- Fig. 10. Health stage division of IMS-2 dataset using the project FFP and thresholds.",
        "- Fig. 11. Isolated faulty samples selected from early, medium and last degradation stages.",
        "- Fig. 12. Kurtograms of isolated faulty samples and selected bandpass ranges.",
        "- Fig. 13. Envelope spectra of filtered isolated faulty samples with BPFO harmonics.",
        "- Fig. 14. Correlation between HI and time-domain features in IMS-2 dataset.",
        "- Fig. 15. Correlation between HI and frequency-domain features in IMS-2 dataset.",
    ])
    note_path = OUT_DIR / "README_captions.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {note_path}", flush=True)
    return str(note_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    autoencoder, _ = train_or_load_ms2ae()

    total = count_csv_rows("prog_analizador/data/IMS2.csv")
    all_samples = read_rows(DATASET, 0, total)
    print(f"[DATA] IMS2 samples={all_samples.shape}", flush=True)

    hi_raw = compute_reconstruction_hi(autoencoder, all_samples)
    hi_norm = normalize_hi(hi_raw)
    del autoencoder
    tf.keras.backend.clear_session()
    gc.collect()
    thresholds = stage_thresholds_paper(hi_norm[:HEALTHY_SAMPLES])
    ffp = find_ffp(hi_norm, thresholds[0])
    representative_samples = [ffp, MEDIUM_SAMPLE, LAST_SAMPLE]
    representative_labels = ["Early stage", "Medium stage", "Last stage"]

    outputs = {}
    outputs["fig08"] = plot_fig08_hi_values(hi_norm, thresholds[0], ffp)
    outputs["fig09"] = plot_fig09_qq(hi_norm[:HEALTHY_SAMPLES])
    outputs["fig10"] = plot_fig10_health_stage(hi_norm, thresholds, ffp, representative_samples)
    outputs["fig11"] = plot_fig11_isolated_samples(representative_samples, representative_labels)
    selected_bands, fig12_paths = plot_fig12_kurtograms(representative_samples, representative_labels)
    outputs["fig12"] = fig12_paths
    outputs["fig13"] = plot_fig13_filtered_fft(representative_samples, representative_labels, selected_bands)

    features_time = time_domain_features(all_samples)
    time_feature_names, time_overall = corr_vector(hi_norm, features_time)
    _, time_cluster = cluster_corr_matrix(hi_norm, features_time, ffp, n_clusters=16)
    outputs["fig14"] = plot_correlation_paper_form(
        time_feature_names,
        time_overall,
        time_cluster,
        "Fig. 14. Correlation matrices between HI and time-domain features in IMS-2 dataset",
        "fig14_time_domain_correlation_ims2_project",
    )

    ffp_band = selected_bands[ffp]
    features_freq = frequency_domain_features(all_samples, ffp_band)
    freq_feature_names, freq_overall = corr_vector(hi_norm, features_freq)
    _, freq_cluster = cluster_corr_matrix(hi_norm, features_freq, ffp, n_clusters=16)
    outputs["fig15"] = plot_correlation_paper_form(
        freq_feature_names,
        freq_overall,
        freq_cluster,
        "Fig. 15. Correlation matrices between HI and frequency-domain features in IMS-2 dataset",
        "fig15_frequency_domain_correlation_ims2_project",
    )

    summary = {
        "dataset": DATASET,
        "samples": int(total),
        "window_size": WINDOW_SIZE,
        "healthy_samples": HEALTHY_SAMPLES,
        "project_ffp": int(ffp),
        "paper_ffp": 536,
        "representative_samples": [int(x) for x in representative_samples],
        "thresholds": {
            "early": float(thresholds[0]),
            "medium": float(thresholds[1]),
            "last": float(thresholds[2]),
        },
        "selected_bands": {str(k): [float(v[0]), float(v[1])] for k, v in selected_bands.items()},
        "top_time_correlations": sorted(
            [(name, float(val)) for name, val in zip(time_feature_names, time_overall)],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5],
        "top_frequency_correlations": sorted(
            [(name, float(val)) for name, val in zip(freq_feature_names, freq_overall)],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5],
        "outputs": outputs,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["caption_note"] = write_caption_note(summary)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
