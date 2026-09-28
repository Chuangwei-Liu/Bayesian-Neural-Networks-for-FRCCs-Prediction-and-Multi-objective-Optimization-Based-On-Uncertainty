"""Generate Figure 19 TOPSIS preference-weight sensitivity radar panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_figures.common import DATA, FIGURES, configure_style, save_figure


def generate(output_dir: Path = FIGURES / "figure_19", dpi: int = 600):
    configure_style(18)
    directory = DATA / "topsis_sensitivity"
    rankings = pd.read_csv(directory / "corrected_topsis_rankings_PE_minmax.csv")
    tops = pd.read_csv(directory / "corrected_topsis_top_solutions_minmax.csv")
    metrics = ["UTX(MPa)", "G(KJm3)", "MiniSF(cm)", "UTX Uncertainty", "G Uncertainty", "MiniSF Uncertainty"]
    labels = [r"$\sigma_u$", r"$G_t$", r"$D_{spread}$", r"$\Delta\sigma_u$", r"$\Delta G_t$", r"$\Delta D_{spread}$"]
    pe = rankings[rankings["fiber"].astype(str).str.upper() == "PE"].copy()
    tested = pe[pe["pareto_index"] == 28].iloc[0]
    maxima = np.array([pe[c].max() for c in metrics[:3]]
                      + [1.0 / pe[c].min() for c in metrics[3:]])
    def radar_values(row):
        raw = np.array([row[c] for c in metrics[:3]]
                       + [1.0 / row[c] for c in metrics[3:]], dtype=float)
        return raw / maxima
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed = np.r_[theta, theta[0]]
    tested_values = radar_values(tested)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), subplot_kw={"projection": "polar"})
    for ax, scheme in zip(axes, ("50/50", "70/30")):
        top = tops[(tops["weight_scheme"].astype(str) == scheme)
                   & (tops["fiber"].astype(str).str.upper() == "PE")].iloc[0]
        best = pe[pe["pareto_index"] == int(top["pareto_index"])].iloc[0]
        best_values = radar_values(best)
        ax.plot(closed, np.r_[best_values, best_values[0]], "o-", lw=3,
                color="#C00000", label=f"Rank-1 at {scheme}")
        ax.fill(closed, np.r_[best_values, best_values[0]], color="#C00000", alpha=.12)
        ax.plot(closed, np.r_[tested_values, tested_values[0]], "o--", lw=3,
                color="#2F5597", label="Tested formulation (60/40)")
        ax.fill(closed, np.r_[tested_values, tested_values[0]], color="#2F5597", alpha=.08)
        ax.set_xticks(theta, labels, fontsize=22)
        ax.set_yticks([.2, .4, .6, .8, 1.0])
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, 1)
        ax.set_title(f"Performance/uncertainty = {scheme}", pad=28, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir / "Figure_19_TOPSIS_weight_sensitivity", dpi)


if __name__ == "__main__":
    generate()
