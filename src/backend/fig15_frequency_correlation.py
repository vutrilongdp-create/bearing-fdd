import itertools
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert


DATA_DIR = Path("prog_analizador/data")
OUT_DIR = Path("report_results/ims2_paper_form_visuals")
SOURCE_SVG = OUT_DIR / "fig08_square_rescaled_paper_style_ims2_project.svg"

FS = 20480.0
SHAFT = 33.33
BPFO = 236.0
BPFI = 297.0
BSF = 278.0
FTF = 15.0
FILTER_BAND = (2560.0, 5120.0)

N_SAMPLES = 984
FFP = 532
X_MAX = 983
Y_MIN = 0.4978
Y_MAX = 0.5084


sns.set_theme(
    style="white",
    font="Times New Roman",
    rc={
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.8,
        "figure.dpi": 130,
        "savefig.dpi": 260,
    },
)
plt.rcParams.update({"font.family": "Times New Roman"})


def read_rows(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        rows = [line for line in handle if line.strip()]
    data = np.loadtxt(rows, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def extract_path_for_stroke(svg_text, stroke_hex):
    group_pattern = re.compile(r'<g id="line2d_\d+">(?P<block>.*?)</g>', re.S)
    path_pattern = re.compile(
        r'<path d="(?P<path>.*?)"\s+clip-path="[^"]+"\s+style="[^"]*stroke: '
        + re.escape(stroke_hex)
        + r'[^"]*"/>',
        re.S,
    )
    candidates = []
    for group in group_pattern.finditer(svg_text):
        match = path_pattern.search(group.group("block"))
        if not match:
            continue
        pairs = re.findall(r"[ML]\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)", match.group("path"))
        if not pairs:
            continue
        points = np.asarray([(float(x), float(y)) for x, y in pairs], dtype=float)
        in_axes = (
            (points[:, 0] >= 46.874531)
            & (points[:, 0] <= 346.79952)
            & (points[:, 1] >= 20.14375)
            & (points[:, 1] <= 317.858739)
        )
        candidates.append((int(np.sum(in_axes)), len(points), match.group("path")))
    if not candidates:
        raise RuntimeError(f"Cannot find plotted path with stroke {stroke_hex}")
    candidates.sort(reverse=True)
    return candidates[0][2]


def path_to_data(path_d):
    pairs = re.findall(r"[ML]\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)", path_d)
    points = np.asarray([(float(x), float(y)) for x, y in pairs], dtype=float)

    svg_x0, svg_x1 = 46.874531, 346.79952
    svg_y_bottom, svg_y_top = 317.858739, 20.14375
    x_data = (points[:, 0] - svg_x0) / (svg_x1 - svg_x0) * X_MAX
    y_data = Y_MIN + (svg_y_bottom - points[:, 1]) / (svg_y_bottom - svg_y_top) * (Y_MAX - Y_MIN)
    order = np.argsort(x_data)
    return x_data[order], y_data[order]


def load_hi_from_fig08_svg():
    svg_text = SOURCE_SVG.read_text(encoding="utf-8")
    hx, hy = path_to_data(extract_path_for_stroke(svg_text, "#2ca25f"))
    ux, uy = path_to_data(extract_path_for_stroke(svg_text, "#1f4eeb"))
    x = np.concatenate([hx, ux])
    y = np.concatenate([hy, uy])
    order = np.argsort(x)
    return np.interp(np.arange(N_SAMPLES, dtype=float), x[order], y[order])


def power_spectrum(signal):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    windowed = np.hanning(signal.size) * signal
    spectrum = np.abs(np.fft.rfft(windowed, norm="forward")) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / FS)
    spectrum[0:5] = 0.0
    return spectrum, freqs


def filtered_envelope_spectrum(signal):
    low_freq, high_freq = FILTER_BAND
    nyquist = FS / 2.0
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.99)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return power_spectrum(envelope)


def max_near(spectrum, freqs, target_hz, width_hz=10.0):
    freqs = np.asarray(freqs)
    spectrum = np.asarray(spectrum)
    mask = (freqs >= target_hz - width_hz) & (freqs <= target_hz + width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(spectrum[mask]))


def frequency_domain_features(samples):
    targets = [
        ("Fund f", SHAFT),
        ("BPFO f", BPFO),
        ("BPFI f", BPFI),
        ("BSF f", BSF),
        ("FTF f", FTF),
        ("Fund nf", SHAFT),
        ("BPFO nf", BPFO),
        ("BPFI nf", BPFI),
        ("BSF nf", BSF),
        ("FTF nf", FTF),
    ]
    values = {name: [] for name, _ in targets}
    for idx, sample in enumerate(samples):
        nf_spectrum, nf_freqs = power_spectrum(sample)
        f_spectrum, f_freqs = filtered_envelope_spectrum(sample)
        for name, target in targets[:5]:
            values[name].append(max_near(f_spectrum, f_freqs, target))
        for name, target in targets[5:]:
            values[name].append(max_near(nf_spectrum, nf_freqs, target))
        if (idx + 1) % 100 == 0:
            print(f"[FEATURES] {idx + 1}/{len(samples)} samples", flush=True)
    return {name: np.asarray(vals, dtype=float) for name, vals in values.items()}


def absolute_corr_matrix(names, series):
    stacked = np.vstack([series[name] for name in names])
    matrix = np.corrcoef(stacked)
    return np.abs(np.nan_to_num(matrix, nan=0.0))


def cluster_corr_matrix(hi, features, n_clusters=16):
    indices = np.arange(FFP, len(hi))
    clusters = np.array_split(indices, n_clusters)
    names = ["HI", *features.keys()]
    matrix = np.zeros((len(names), n_clusters), dtype=float)
    matrix[0, :] = 1.0
    for col, cluster in enumerate(clusters):
        hi_c = hi[cluster]
        for row, name in enumerate(names[1:], start=1):
            data_c = features[name][cluster]
            if len(cluster) < 3 or np.std(data_c) == 0 or np.std(hi_c) == 0:
                matrix[row, col] = 0.0
            else:
                matrix[row, col] = abs(np.corrcoef(hi_c, data_c)[0, 1])
    return names, matrix


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, dpi=260)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {svg}")


def main():
    samples = read_rows(DATA_DIR / "IMS2.csv")
    hi = load_hi_from_fig08_svg()
    features = frequency_domain_features(samples)

    names = ["HI", *features.keys()]
    series = {"HI": hi, **features}
    overall_matrix = absolute_corr_matrix(names, series)
    cluster_names, cluster_matrix = cluster_corr_matrix(hi, features, n_clusters=16)

    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    fig = plt.figure(figsize=(10.8, 5.05))
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.16, 0.055, 0.15, 1.18, 0.055],
        left=0.065,
        right=0.965,
        top=0.94,
        bottom=0.25,
        wspace=0.10,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 3])
    cax2 = fig.add_subplot(gs[0, 4])

    sns.heatmap(
        overall_matrix,
        ax=ax1,
        cbar_ax=cax1,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2g",
        xticklabels=names,
        yticklabels=names,
        square=True,
        linewidths=0.35,
        linecolor="#7a7a7a",
        annot_kws={"fontsize": 7.3, "color": "#b7a300"},
        cbar_kws={"ticks": np.linspace(0, 1, 6)},
    )
    ax1.tick_params(axis="x", rotation=45)
    ax1.tick_params(axis="y", rotation=0)
    cax1.tick_params(labelsize=8)
    cax1.set_ylabel("")

    cluster_labels = [f"c{i}" for i in range(1, 17)]
    sns.heatmap(
        cluster_matrix,
        ax=ax2,
        cbar_ax=cax2,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=False,
        xticklabels=cluster_labels,
        yticklabels=cluster_names,
        linewidths=0.35,
        linecolor="#7a7a7a",
        cbar_kws={"ticks": np.linspace(0, 1, 6)},
    )
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)
    cax2.tick_params(labelsize=8)
    cax2.set_ylabel("")

    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.text(
        0.255,
        0.120,
        "(a) Correlation matrix between HI value and frequency-domain features",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.text(
        0.715,
        0.120,
        "(b) Correlation matrix over time between HI value and frequency-domain features",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.040,
        "Fig. 15. Correlation matrices between HI value and frequency-domain features in IMS-2 dataset.",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    save_fig(fig, "fig15_frequency_domain_correlation_ims2_project")

    top = sorted(
        [(name, overall_matrix[0, idx]) for idx, name in enumerate(names[1:], start=1)],
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    print("[TOP]", ", ".join(f"{name}={value:.2f}" for name, value in top))


if __name__ == "__main__":
    main()
