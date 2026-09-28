#!/usr/bin/env python
"""Reproduce the W/B--S/B parametric-analysis bar charts (Figure 15a--c).

The source workbook contains the BNN predictive means and total standard
deviations used by the former Origin figures.  This renderer keeps the plot
data-driven so that the publication figures can be regenerated without the
lost Origin project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "index_analysis_predictions.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "figure_15"

# Reuse the lightweight spreadsheet dependencies installed for the revision
# experiments when they are not available in the active Python environment.
VENDOR_DIR = SCRIPT_DIR / "_vendor"
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))

import pandas as pd

# Axis labels retain their current size.  All other visible text receives the
# requested additional 4-point enlargement relative to the previous version.
TICK_SIZE = 23
AXIS_LABEL_SIZE = 24
ANNOTATION_SIZE = 24
LEGEND_SIZE = 24


@dataclass(frozen=True)
class PlotSpec:
    filename: str
    mean_column: str
    std_column: str
    ylabel: str
    ylim: tuple[float, float]
    yticks: np.ndarray
    legend_location: str
    use_header_band: bool = False


PLOT_SPECS = (
    PlotSpec(
        filename="Figure_15a",
        mean_column="Predicted_CX_mean",
        std_column="Predicted_CX_total_std",
        ylabel=r"$\sigma_{cs}$ (MPa)",
        ylim=(0.0, 94.0),
        yticks=np.arange(0.0, 81.0, 20.0),
        legend_location="upper right",
        use_header_band=True,
    ),
    PlotSpec(
        filename="Figure_15b",
        mean_column="Predicted_UTX_mean",
        std_column="Predicted_UTX_total_std",
        ylabel=r"$\sigma_u$ (MPa)",
        ylim=(0.0, 9.6),
        yticks=np.arange(0.0, 10.0, 1.0),
        legend_location="upper right",
        use_header_band=True,
    ),
    PlotSpec(
        filename="Figure_15c",
        mean_column="Predicted_D_spread_mean",
        std_column="Predicted_D_spread_total_std",
        ylabel=r"$D_{spread}$ (cm)",
        ylim=(0.0, 39.0),
        yticks=np.arange(0.0, 36.0, 5.0),
        legend_location="upper left",
    ),
)

SERIES_STYLES = {
    0.3: {"label": "S/B=0.3", "color": "#B7E61E", "hatch": "//"},
    0.6: {"label": "S/B=0.6", "color": "#25A896", "hatch": "\\\\"},
    1.2: {"label": "S/B=1.2", "color": "#2E6F91", "hatch": "...."},
}

# Figure-specific vertical placement keeps the enlarged header text close to
# the data while preserving a small gap above the tallest error bar underneath.
HEADER_POSITIONS = {
    "Figure_15a": {"annotation_y": 0.925, "legend_y": 0.875},
    "Figure_15b": {"annotation_y": 0.830, "legend_y": 0.780},
}


def configure_style() -> None:
    """Apply publication typography consistently to text and math."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_figure_data(workbook_path: Path, sheet_name: str = "Fig1") -> pd.DataFrame:
    """Load and validate the 5-by-3 W/B--S/B design used in Figure 15."""
    if not workbook_path.exists():
        raise FileNotFoundError(f"Source workbook not found: {workbook_path}")

    required = {
        "SF/C",
        "FA/C",
        "W/B",
        "S/B",
        "SP/B",
        "Vf(%)",
        "If",
        *(spec.mean_column for spec in PLOT_SPECS),
        *(spec.std_column for spec in PLOT_SPECS),
    }
    data = pd.read_excel(workbook_path, sheet_name=sheet_name)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing}")

    expected_wb = np.array([0.22, 0.24, 0.26, 0.28, 0.30])
    expected_sb = np.array([0.3, 0.6, 1.2])
    actual_wb = np.sort(data["W/B"].unique().astype(float))
    actual_sb = np.sort(data["S/B"].unique().astype(float))
    if len(data) != 15 or not np.allclose(actual_wb, expected_wb) or not np.allclose(actual_sb, expected_sb):
        raise ValueError(
            "Fig1 must contain exactly 15 rows covering W/B="
            f"{expected_wb.tolist()} and S/B={expected_sb.tolist()}."
        )
    if data.duplicated(["W/B", "S/B"]).any():
        raise ValueError("Fig1 contains duplicate W/B--S/B combinations.")

    fixed_values = {"SF/C": 0.0, "FA/C": 1.2, "SP/B": 0.003, "Vf(%)": 2.0}
    for column, expected in fixed_values.items():
        if not np.allclose(data[column].astype(float), expected):
            raise ValueError(f"Unexpected {column}; Figure 15 requires {expected}.")

    return data.sort_values(["W/B", "S/B"]).reset_index(drop=True)


def add_axis_arrow(ax: plt.Axes) -> None:
    """Reproduce the upward arrow used by the original Origin y axis."""
    ax.annotate(
        "",
        xy=(0.0, 1.025),
        xytext=(0.0, 0.985),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.4, "mutation_scale": 14},
        annotation_clip=False,
    )


def draw_chart(data: pd.DataFrame, spec: PlotSpec, output_dir: Path, dpi: int) -> list[Path]:
    """Draw one grouped bar chart and save raster and vector versions."""
    configure_style()
    # Matches the physical size of the archived 3216x2462 px, 300-dpi figures.
    fig, ax = plt.subplots(figsize=(10.72, 2462 / 300))
    # Figures 13(a) and 13(b) need a dedicated header band after the legend and
    # four-line condition block were enlarged.  Keeping both outside the data
    # axes prevents them from covering error bars without changing axis limits.
    plot_top = 0.86 if spec.use_header_band else 0.94
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.15, top=plot_top)

    wb_values = np.sort(data["W/B"].unique().astype(float))
    x = np.arange(len(wb_values), dtype=float)
    width = 0.22

    for series_index, sb_value in enumerate(SERIES_STYLES):
        style = SERIES_STYLES[sb_value]
        subset = data[np.isclose(data["S/B"].astype(float), sb_value)].sort_values("W/B")
        ax.bar(
            x + (series_index - 1) * width,
            subset[spec.mean_column].to_numpy(dtype=float),
            width=width,
            yerr=subset[spec.std_column].to_numpy(dtype=float),
            capsize=4.0,
            color=style["color"],
            edgecolor="black",
            linewidth=1.0,
            hatch=style["hatch"],
            error_kw={"ecolor": "#606060", "elinewidth": 1.0, "capthick": 1.0},
            label=style["label"],
            zorder=3,
        )

    ax.set_xlim(-0.55, len(x) - 0.20)
    ax.set_ylim(*spec.ylim)
    ax.set_yticks(spec.yticks)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2f}" if value < 0.3 else "0.3" for value in wb_values])
    ax.set_ylabel(spec.ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=18)
    ax.set_xlabel(
        "W/B",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        labelpad=14,
    )

    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.2, labelsize=TICK_SIZE)
    ax.tick_params(axis="y", which="minor", direction="out", length=4, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.grid(False)
    add_axis_arrow(ax)

    conditions = "FA/C = 1.2\nSP/B = 0.003\n" + r"$V_f$ = 2%" + "\nFiber Type = PVA"
    if spec.use_header_band:
        annotation_x = 0.455
        annotation_y = HEADER_POSITIONS[spec.filename]["annotation_y"]
        annotation_transform = fig.transFigure
    else:
        annotation_x, annotation_y = 0.37, 0.99
        annotation_transform = ax.transAxes
    ax.text(
        annotation_x,
        annotation_y,
        conditions,
        transform=annotation_transform,
        ha="left",
        va="top",
        fontsize=ANNOTATION_SIZE,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.5},
        zorder=10,
    )

    if spec.use_header_band:
        legend_anchor = (0.91, HEADER_POSITIONS[spec.filename]["legend_y"])
        legend_transform = fig.transFigure
    else:
        legend_anchor = (0.015, 1.0)
        legend_transform = ax.transAxes
    legend = ax.legend(
        loc=spec.legend_location,
        bbox_to_anchor=legend_anchor,
        bbox_transform=legend_transform,
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        facecolor="white",
        framealpha=1.0,
        fontsize=LEGEND_SIZE,
        handlelength=2.0,
        handleheight=1.2,
        borderpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        output_path = output_dir / f"{spec.filename}.{extension}"
        save_options = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(output_path, **save_options)
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prediction workbook path.")
    parser.add_argument("--sheet", default="Fig1", help="Workbook sheet containing Figure 15 data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Figure output directory.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_figure_data(args.data.resolve(), args.sheet)
    for spec in PLOT_SPECS:
        outputs = draw_chart(data, spec, args.output_dir.resolve(), args.dpi)
        print("Generated:", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
