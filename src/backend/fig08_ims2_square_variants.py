from pathlib import Path

import numpy as np

from fig_ims2_paper_form import (
    HEALTHY_SAMPLES,
    compute_reconstruction_hi,
    find_ffp,
    normalize_hi,
    read_rows,
    stage_thresholds_paper,
    train_or_load_ms2ae,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


OUT_DIR = Path("report_results/ims2_paper_form_visuals")


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
plt.rcParams.update({"font.family": "Times New Roman"})


def save_fig(fig, name):
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png}")
    return str(png), str(svg)


def plot_square_hi(hi_values, threshold, ffp, ylabel, title, name, ylim=None, show_threshold=True):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
    x_healthy = np.arange(HEALTHY_SAMPLES)
    x_unlabeled = np.arange(HEALTHY_SAMPLES, len(hi_values))

    ax.plot(
        x_healthy,
        hi_values[:HEALTHY_SAMPLES],
        color="#2ca25f",
        linewidth=0.9,
        label="HI of healthy samples",
    )
    ax.plot(
        x_unlabeled,
        hi_values[HEALTHY_SAMPLES:],
        color="#1f4eeb",
        linewidth=0.9,
        label="HI of unlabeled samples",
    )

    if show_threshold:
        ax.axhline(threshold, color="#b22222", linestyle="--", linewidth=0.85, label="P95 threshold")
        ax.axvline(ffp, color="#d95f02", linestyle="-.", linewidth=0.9, label=f"FFP #{ffp}")

    ax.set_xlim(0, len(hi_values) - 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=0.95)
    return save_fig(fig, name)


def plot_square_hi_paper_regions(hi_values, ffp, name):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
    x_healthy = np.arange(HEALTHY_SAMPLES)
    x_pred_healthy = np.arange(HEALTHY_SAMPLES, ffp)
    x_faulty = np.arange(ffp, len(hi_values))

    ax.axvspan(0, ffp, color="#d8f5d8", alpha=0.85, zorder=0)
    ax.axvspan(ffp, len(hi_values) - 1, color="#f8dada", alpha=0.85, zorder=0)

    ax.plot(
        x_healthy,
        hi_values[:HEALTHY_SAMPLES],
        color="#1a9c35",
        linewidth=0.75,
        label="HI of healthy samples",
    )
    ax.plot(
        x_pred_healthy,
        hi_values[HEALTHY_SAMPLES:ffp],
        color="#1f4eeb",
        linewidth=0.75,
        label="HI of predicted healthy samples",
    )
    ax.plot(
        x_faulty,
        hi_values[ffp:],
        color="#e41a1c",
        linewidth=0.75,
        label="HI of predicted faulty samples",
    )

    ax.set_xlim(0, len(hi_values) - 1)
    ax.set_ylim(0.4977, 0.5084)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Health Index")
    ax.set_title("Health index values of IMS-2 dataset")
    ax.grid(True, linestyle=(0, (5, 5)), linewidth=0.65, color="#cfcfcf")
    ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=1.0)
    return save_fig(fig, name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    autoencoder, _ = train_or_load_ms2ae()
    samples = read_rows("IMS2", 0, 984)
    hi_raw = compute_reconstruction_hi(autoencoder, samples)
    hi_norm = normalize_hi(hi_raw)
    threshold = stage_thresholds_paper(hi_norm[:HEALTHY_SAMPLES])[0]
    ffp = find_ffp(hi_norm, threshold)

    plot_square_hi(
        hi_norm,
        threshold,
        ffp,
        "Normalized Health Index",
        "Health index values of IMS-2 dataset",
        "fig08_square_normalized_ims2_project",
        ylim=(-0.04, 1.25),
        show_threshold=True,
    )

    # Display-only affine rescaling so the y-axis visually matches the paper range.
    # This does not change FFP logic; it only maps the project HI to a paper-like axis.
    hi_rescaled = 0.498 + hi_norm * (0.010 / 1.2)
    threshold_rescaled = 0.498 + threshold * (0.010 / 1.2)
    plot_square_hi(
        hi_rescaled,
        threshold_rescaled,
        ffp,
        "Health Index",
        "Health index values of IMS-2 dataset",
        "fig08_square_rescaled_paper_style_ims2_project",
        ylim=(0.4978, 0.5084),
        show_threshold=False,
    )

    plot_square_hi_paper_regions(
        hi_rescaled,
        ffp,
        "fig08_square_rescaled_regions_ims2_project",
    )

    print(f"[INFO] Project FFP = #{ffp}")
    print(f"[INFO] Normalized P95 threshold = {threshold:.6f}")
    print(f"[INFO] Rescaled P95 threshold = {threshold_rescaled:.6f}")


if __name__ == "__main__":
    main()
