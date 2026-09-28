"""Assemble Figure 20 experimental assets and redraw panel (e)."""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from manuscript_figures.common import DATA, FIGURES, configure_style, save_figure
from manuscript_figures.digitize_figure_20e import OUTPUT as DIGITIZED, digitize


def generate(output_dir: Path = FIGURES / "figure_20", dpi: int = 600):
    configure_style(20)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    photos = DATA / "experimental_validation" / "source_photos"
    for letter, source in zip("abcd", sorted(photos.glob("*.png"))):
        target = output_dir / f"Figure_20{letter}_experimental_photo.png"
        shutil.copy2(source, target)
        outputs.append(target)
    if not DIGITIZED.exists():
        digitize()
    data = pd.read_csv(DIGITIZED)
    colors = {"PE-FRCCs-1": "#555555", "PE-FRCCs-2": "#EF4040", "PE-FRCCs-3": "#086AD8"}
    fig, ax = plt.subplots(figsize=(10, 9))
    for specimen, subset in data.groupby("specimen", sort=False):
        ax.scatter(subset["strain_percent"], subset["stress_mpa"], s=1.2,
                   color=colors[specimen], label=specimen, rasterized=True)
    ax.set_xlim(0, 6.35)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Strain (%)", fontweight="bold")
    ax.set_ylabel("Stress (MPa)", fontweight="bold")
    ax.legend(loc="lower right", markerscale=5)
    ax.spines[["top", "right"]].set_visible(False)
    outputs += save_figure(fig, output_dir / "Figure_20e_stress_strain", dpi)
    return outputs


if __name__ == "__main__":
    generate()
