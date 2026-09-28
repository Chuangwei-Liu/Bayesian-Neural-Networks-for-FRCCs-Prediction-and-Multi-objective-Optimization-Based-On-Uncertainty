"""Generate Figure 16: effects of fiber type and volume fraction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_figures.common import DATA, FIGURES, configure_style, save_figure

SPECS = [
    ("Predicted_CX_mean", "Predicted_CX_total_std", r"$\sigma_{cs}$ (MPa)"),
    ("Predicted_UTX_mean", "Predicted_UTX_total_std", r"$\sigma_u$ (MPa)"),
    ("Predicted_G_t_mean", "Predicted_G_t_total_std", r"$G_t$ (kJ/m$^3$)"),
    ("Predicted_D_spread_mean", "Predicted_D_spread_total_std", r"$D_{spread}$ (cm)"),
]


def generate(output_dir: Path = FIGURES / "figure_16", dpi: int = 600):
    configure_style(18)
    data = pd.read_excel(DATA / "index_analysis_predictions.xlsx", sheet_name="Fig3")
    fiber_map = {float(v): name for v, name in zip(sorted(data["If"].unique()), ("PVA", "PE"))}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    colors = {"PVA": "#2878B5", "PE": "#C82423"}
    for ax, (mean_col, std_col, ylabel) in zip(axes.flat, SPECS):
        for value in sorted(data["If"].unique()):
            subset = data[np.isclose(data["If"], value)].sort_values("Vf(%)")
            name = fiber_map[float(value)]
            ax.errorbar(subset["Vf(%)"], subset[mean_col], yerr=subset[std_col],
                        marker="o", ms=8, lw=2.2, capsize=5, label=name, color=colors[name])
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(alpha=.25, linestyle="--")
        ax.legend(frameon=True)
    for ax in axes[-1]:
        ax.set_xlabel(r"$V_f$ (%)", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir / "Figure_16_fiber_type_and_volume", dpi)


if __name__ == "__main__":
    generate()
