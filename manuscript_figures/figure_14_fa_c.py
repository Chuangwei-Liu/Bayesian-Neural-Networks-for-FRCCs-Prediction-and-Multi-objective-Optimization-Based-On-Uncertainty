"""Reproduce the FA/C sensitivity plots in manuscript Figure 14(a--c).

The former Origin figures are redrawn directly from the saved BNN predictive
means and total standard deviations so they remain reproducible after release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "index_analysis_predictions.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "figure_14"

VENDOR_DIR = SCRIPT_DIR / "_vendor"
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))

import pandas as pd


# Match the final enlarged typography used throughout the manuscript.
TICK_SIZE = 27
AXIS_LABEL_SIZE = 28
ANNOTATION_SIZE = 30
LEGEND_SIZE = 30


@dataclass(frozen=True)
class PlotSpec:
    filename: str
    mean_column: str
    std_column: str
    ylabel: str
    legend_label: str
    ylim: tuple[float, float]
    yticks: np.ndarray
    annotation_position: tuple[float, float]


PLOT_SPECS = (
    PlotSpec(
        filename="Figure_14a",
        mean_column="Predicted_CX_mean",
        std_column="Predicted_CX_total_std",
        ylabel=r"$\sigma_{cs}$ (MPa)",
        legend_label=r"$\sigma_{cs}$",
        ylim=(30.0, 94.0),
        yticks=np.arange(30.0, 91.0, 10.0),
        annotation_position=(0.675, 0.865),
    ),
    PlotSpec(
        filename="Figure_14b",
        mean_column="Predicted_UTX_mean",
        std_column="Predicted_UTX_total_std",
        ylabel=r"$\sigma_u$ (MPa)",
        legend_label=r"$\sigma_u$",
        ylim=(0.0, 9.4),
        yticks=np.arange(0.0, 9.0, 2.0),
        annotation_position=(0.075, 0.44),
    ),
    PlotSpec(
        filename="Figure_14c",
        mean_column="Predicted_D_spread_mean",
        std_column="Predicted_D_spread_total_std",
        ylabel=r"$D_{spread}$ (cm)",
        legend_label=r"$D_{spread}$",
        ylim=(0.0, 25.0),
        yticks=np.arange(0.0, 25.0, 2.0),
        annotation_position=(0.675, 0.40),
    ),
)


def configure_style() -> None:
    """Apply consistent Times New Roman typography to text and mathematics."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_figure_data(workbook_path: Path, sheet_name: str = "Fig2") -> pd.DataFrame:
    """Load and validate the five-point FA/C sweep used by Figure 14."""
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

    expected_fa_c = np.array([0.0, 0.6, 1.2, 2.2, 4.0])
    actual_fa_c = np.sort(data["FA/C"].astype(float).unique())
    if len(data) != 5 or not np.allclose(actual_fa_c, expected_fa_c):
        raise ValueError(
            f"{sheet_name} must contain exactly FA/C={expected_fa_c.tolist()}."
        )
    if data["FA/C"].duplicated().any():
        raise ValueError(f"{sheet_name} contains duplicate FA/C values.")

    fixed_values = {
        "SF/C": 0.0,
        "W/B": 0.24,
        "S/B": 0.6,
        "SP/B": 0.003,
        "Vf(%)": 2.0,
        "If": 0.247431979,
    }
    for column, expected in fixed_values.items():
        if not np.allclose(data[column].astype(float), expected):
            raise ValueError(f"Unexpected {column}; Figure 14 requires {expected}.")

    for spec in PLOT_SPECS:
        values = data[[spec.mean_column, spec.std_column]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, 1] < 0).any():
            raise ValueError(f"Invalid mean or standard deviation for {spec.filename}.")

    return data.sort_values("FA/C").reset_index(drop=True)


def natural_cubic_spline(x: np.ndarray, y: np.ndarray, samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a dependency-free natural cubic spline through the data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    if len(x) < 3 or np.any(h <= 0):
        raise ValueError("Spline x values must be strictly increasing and contain at least three points.")

    matrix = np.zeros((len(x) - 2, len(x) - 2), dtype=float)
    rhs = np.zeros(len(x) - 2, dtype=float)
    for row in range(len(x) - 2):
        left_h = h[row]
        right_h = h[row + 1]
        matrix[row, row] = 2.0 * (left_h + right_h)
        if row > 0:
            matrix[row, row - 1] = left_h
        if row < len(x) - 3:
            matrix[row, row + 1] = right_h
        rhs[row] = 6.0 * (
            (y[row + 2] - y[row + 1]) / right_h
            - (y[row + 1] - y[row]) / left_h
        )

    second = np.zeros(len(x), dtype=float)
    second[1:-1] = np.linalg.solve(matrix, rhs)
    smooth_x = np.linspace(x[0], x[-1], samples)
    interval = np.searchsorted(x, smooth_x, side="right") - 1
    interval = np.clip(interval, 0, len(x) - 2)
    x0 = x[interval]
    x1 = x[interval + 1]
    width = x1 - x0
    a = (x1 - smooth_x) / width
    b = (smooth_x - x0) / width
    smooth_y = (
        a * y[interval]
        + b * y[interval + 1]
        + ((a**3 - a) * second[interval] + (b**3 - b) * second[interval + 1])
        * width**2
        / 6.0
    )
    return smooth_x, smooth_y


def add_axis_arrows(ax: plt.Axes) -> None:
    """Add arrowheads to the positive ends of the left and bottom axes."""
    arrow_style = {
        "arrowstyle": "-|>",
        "color": "black",
        "lw": 1.4,
        "mutation_scale": 14,
    }
    ax.annotate(
        "",
        xy=(0.0, 1.025),
        xytext=(0.0, 0.985),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(1.025, 0.0),
        xytext=(0.985, 0.0),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        annotation_clip=False,
    )


def draw_chart(data: pd.DataFrame, spec: PlotSpec, output_dir: Path, dpi: int) -> list[Path]:
    """Draw one FA/C sensitivity chart and save raster and vector versions."""
    configure_style()
    fig, ax = plt.subplots(figsize=(10.72, 2461 / 300))
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.15, top=0.94)

    x = data["FA/C"].to_numpy(dtype=float)
    mean = data[spec.mean_column].to_numpy(dtype=float)
    std = data[spec.std_column].to_numpy(dtype=float)
    smooth_x, smooth_y = natural_cubic_spline(x, mean)

    ax.plot(
        smooth_x,
        smooth_y,
        color="#ff1a1a",
        linewidth=1.5,
        linestyle=(0, (5.0, 4.0)),
        zorder=2,
    )
    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="^",
        markersize=10,
        markerfacecolor="#ff1a1a",
        markeredgecolor="#ff1a1a",
        ecolor="#ff3030",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        linestyle="none",
        zorder=3,
    )

    ax.set_xlim(-0.5, 4.7)
    ax.set_ylim(*spec.ylim)
    ax.set_xticks(np.arange(0.0, 5.0, 1.0))
    ax.set_yticks(spec.yticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("FA/C", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=14)
    ax.set_ylabel(spec.ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=18)
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.2, labelsize=TICK_SIZE)
    ax.tick_params(axis="both", which="minor", direction="out", length=4, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.grid(False)
    add_axis_arrows(ax)

    conditions = (
        "W/B = 0.24\n"
        "S/B = 0.6\n"
        "SP/B = 0.3%\n"
        + r"$V_f$ = 2%"
        + "\nFiber Type: PVA"
    )
    ax.text(
        *spec.annotation_position,
        conditions,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_SIZE,
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.0},
        zorder=5,
    )

    marker = Line2D(
        [],
        [],
        linestyle="none",
        marker="^",
        markersize=9,
        markerfacecolor="#ff1a1a",
        markeredgecolor="#ff1a1a",
        label=spec.legend_label,
    )
    legend = ax.legend(
        handles=[marker],
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        facecolor="white",
        framealpha=1.0,
        fontsize=LEGEND_SIZE,
        handlelength=2.0,
        handleheight=1.2,
        handletextpad=0.8,
        borderpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
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
    parser.add_argument("--sheet", default="Fig2", help="Workbook sheet containing Figure 14 data.")
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
