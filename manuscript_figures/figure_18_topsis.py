"""Generate Figure 18 Pareto projections and TOPSIS radar chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_figures.common import DATA, FIGURES, configure_style, save_figure


def _normalize(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(float)
    low, high = values.min(0), values.max(0)
    return (values - low) / np.where(high == low, 1, high - low)


def generate(output_dir: Path = FIGURES / "figure_18", dpi: int = 600):
    configure_style(18)
    ranked = pd.read_csv(DATA / "optimization" / "PE" / "topsis_ranking.csv")
    best = ranked.loc[ranked["TOPSIS Rank"].idxmin()]
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.25), hspace=.35, wspace=.38)
    pairs = [
        ("UTX(MPa)", "G(KJm3)", r"$\sigma_u$ (MPa)", r"$G_t$ (kJ/m$^3$)"),
        ("UTX(MPa)", "MiniSF(cm)", r"$\sigma_u$ (MPa)", r"$D_{spread}$ (cm)"),
        ("G(KJm3)", "MiniSF(cm)", r"$G_t$ (kJ/m$^3$)", r"$D_{spread}$ (cm)"),
    ]
    for i, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(ranked[xcol], ranked[ycol], c=ranked["TOPSIS Score"],
                        cmap="viridis", s=48, alpha=.78)
        ax.scatter(best[xcol], best[ycol], marker="*", s=220, c="red",
                   edgecolor="black", label="Best TOPSIS solution", zorder=5)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(alpha=.25)
        ax.legend(loc="best")
        cb = fig.colorbar(sc, ax=ax, pad=.04)
        cb.set_label("TOPSIS score", fontsize=22)
        cb.ax.tick_params(labelsize=18)

    columns = ["UTX(MPa)", "G(KJm3)", "MiniSF(cm)", "UTX Uncertainty", "G Uncertainty", "MiniSF Uncertainty"]
    labels = [r"$\sigma_u$", r"$G_t$", r"$D_{spread}$", r"$\Delta\sigma_u$", r"$\Delta G_t$", r"$\Delta D_{spread}$"]
    normed = _normalize(ranked, columns)
    # Lower uncertainty is preferable; invert those three axes.
    normed[:, 3:] = 1 - normed[:, 3:]
    theta = np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
    closed_theta = np.r_[theta, theta[0]]
    ax = fig.add_subplot(gs[1, :], projection="polar")
    cmap = plt.get_cmap("viridis")
    score_norm = plt.Normalize(ranked["TOPSIS Score"].min(), ranked["TOPSIS Score"].max())
    for row, score in zip(normed, ranked["TOPSIS Score"]):
        ax.plot(closed_theta, np.r_[row, row[0]], color=cmap(score_norm(score)), alpha=.35, lw=.8)
    best_idx = ranked["TOPSIS Rank"].idxmin()
    best_values = normed[best_idx]
    ax.plot(closed_theta, np.r_[best_values, best_values[0]], color="red", lw=3, label="Best TOPSIS solution")
    ax.fill(closed_theta, np.r_[best_values, best_values[0]], color="red", alpha=.16)
    ax.set_xticks(theta, labels, fontsize=22)
    ax.set_yticks([.2, .4, .6, .8, 1.0])
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    ax.set_title("Pareto solutions performance (PE fiber)", pad=30, fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=score_norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=.12, shrink=.8)
    cb.set_label("TOPSIS score", fontsize=22)
    cb.ax.tick_params(labelsize=18)
    return save_figure(fig, output_dir / "Figure_18_PE_pareto_TOPSIS", dpi)


if __name__ == "__main__":
    generate()
