"""Generate Figure 2 correlation matrices and Figure 3 distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from manuscript_figures.common import DATA, FIGURES, configure_style, save_figure

BEFORE = ["FA/C", "W/B", "S/B", "SP/B", "HPMC/B", "Vf(%)", "Df(um)", "Lf(mm)", "Ef(GPa)", "Tf(MPa)"]
AFTER = ["FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If"]
DISPLAY = {"HPMC/B": "VMA/B", "Vf(%)": r"$V_f$", "Df(um)": r"$D_f$", "Lf(mm)": r"$L_f$", "Ef(GPa)": r"$E_f$", "Tf(MPa)": r"$T_f$", "If": r"$I_f$"}
DIST = ["FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If", "CX(MPa)", "FCX(MPa)", "UTX(Mpa)", "UTS(%)", "G(KJm3)", "MiniSF(cm)"]
DIST_LABELS = ["FA/C", "W/B", "S/B", "SP/B", r"$V_f$", r"$I_f$", r"$\sigma_{cs}$ (MPa)", r"$\sigma_{fc}$ (MPa)", r"$\sigma_u$ (MPa)", r"$\varepsilon_u$ (%)", r"$G_t$ (kJ/m$^3$)", r"$D_{spread}$ (cm)"]


def generate(output_dir: Path = FIGURES / "figure_02_03", dpi: int = 600):
    configure_style(18)
    df = pd.read_excel(DATA / "frcc_database.xlsx", sheet_name="Sheet1")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, columns, title in (("a", BEFORE, "Before dimensionality reduction"), ("b", AFTER, "After dimensionality reduction")):
        labels = [DISPLAY.get(c, c) for c in columns]
        corr = df[columns].astype(float).corr(method="spearman")
        corr.index = corr.columns = labels
        fig, ax = plt.subplots(figsize=(11, 9))
        heat = sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1,
                           annot=True, fmt=".2f", square=True, linewidths=.5,
                           annot_kws={"size": 14}, cbar_kws={"label": "Spearman correlation"}, ax=ax)
        ax.set_title(title, fontweight="bold", pad=15)
        ax.tick_params(labelsize=15)
        heat.collections[0].colorbar.ax.tick_params(labelsize=15)
        heat.collections[0].colorbar.set_label("Spearman correlation", size=18)
        outputs += save_figure(fig, output_dir / f"Figure_02{suffix}_spearman", dpi)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for ax, column, label in zip(axes.flat, DIST, DIST_LABELS):
        values = df[column].dropna().astype(float).to_numpy()
        ax.hist(values, bins=30, density=True, color="#8ecae6", edgecolor="black", alpha=.72)
        if values.std() > 0:
            x = np.linspace(values.min(), values.max(), 300)
            ax.plot(x, norm.pdf(x, values.mean(), values.std()), "--", color="#023047", lw=2)
        ax.set_title(label + "\n" + rf"$\mu={values.mean():.2f},\ \sigma={values.std():.2f}$", fontweight="bold")
        ax.set_ylabel("Density")
        ax.grid(alpha=.2)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    outputs += save_figure(fig, output_dir / "Figure_03_parameter_distributions", dpi)
    return outputs


if __name__ == "__main__":
    generate()
