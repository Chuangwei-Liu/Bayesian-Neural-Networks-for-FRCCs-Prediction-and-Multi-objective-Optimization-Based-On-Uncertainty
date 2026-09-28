r"""Regenerate manuscript Figures 6--9 (BNN/RF/GP comparison).

The original comparison figures were embedded directly in the R2 manuscript and
their standalone plotting source was not retained.  This script reconstructs the
comparison from the repository's database and trained MC-Dropout models using the
same common validation split (random_state=42) and feature scaling.

Run from the repository root with ``python reproduce_all.py`` or execute this
module directly.
"""

from __future__ import annotations

import argparse
import __main__
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor"
if VENDOR.exists():
    # Use the active environment's compiled scientific stack first.  The
    # bundled directory is a fallback for pure-Python I/O dependencies; putting
    # it first can select binaries built for a different Python/OS ABI.
    sys.path.append(str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn


ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from utils.Drawers import (
    MODEL_COMPARISON_COLORS,
    configure_model_comparison_style,
    plot_model_comparison_figure,
)


DATABASE = ROOT / "data" / "frcc_database.xlsx"
BEST_MODELS = ROOT / "models" / "mc_dropout"
DEFAULT_OUTPUT = ROOT / "figures" / "model_comparison"

FEATURES = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If"]
SPLIT_SEED = 42
RF_SEED = 42
Z_95 = 1.96

TARGETS = {
    "CX(MPa)": {
        "figure": 6,
        "model_seed": 618,
        "title": r"$\sigma_{cs}$ (MPa)",
        "slug": "compressive_strength",
    },
    "UTX(Mpa)": {
        "figure": 7,
        "model_seed": 424,
        "title": r"$\sigma_u$ (MPa)",
        "slug": "ultimate_tensile_strength",
    },
    "G(KJm3)": {
        "figure": 8,
        "model_seed": 857,
        "title": r"$G_t$ (kJ/m$^3$)",
        "slug": "tensile_strain_energy_density",
    },
    "MiniSF(cm)": {
        "figure": 9,
        "model_seed": 114,
        "title": r"$D_{spread}$ (cm)",
        "slug": "slump_flow_diameter",
    },
}

class BayesianRegressor(nn.Module):
    """Architecture used by the saved MC-Dropout models."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.output = nn.Linear(2 * hidden_size, output_size * 2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.output(x)
        return output[:, :1], output[:, 1:]


class BayesianNeuralNetwork(nn.Module):
    """Wrapper retained for compatibility with the serialized model objects."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.model = BayesianRegressor(
            input_size, hidden_size, output_size, dropout_p
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)


torch.serialization.add_safe_globals([BayesianRegressor, BayesianNeuralNetwork])


def minmax_scale(values: np.ndarray, minimum: np.ndarray,
                 data_range: np.ndarray) -> np.ndarray:
    safe_range = np.where(data_range == 0.0, 1.0, data_range)
    return (values - minimum) / safe_range


def bnn_prediction(model_path: Path, x_scaled: np.ndarray, y_min: float,
                   y_range: float, mc_samples: int,
                   mc_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Legacy checkpoints were serialized from a script and therefore refer to
    # ``__main__``; expose the compatible classes before unpickling.
    __main__.BayesianRegressor = BayesianRegressor
    __main__.BayesianNeuralNetwork = BayesianNeuralNetwork
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mean_scaled, log_variance = model(x_tensor)
        aleatoric_scaled = torch.exp(log_variance).sqrt().squeeze(-1)

        torch.manual_seed(mc_seed)
        model.train()
        mc_means = torch.stack(
            [model(x_tensor)[0].squeeze(-1) for _ in range(mc_samples)]
        )
        epistemic_scaled = mc_means.std(dim=0, unbiased=True)

    prediction = mean_scaled.squeeze(-1).cpu().numpy() * y_range + y_min
    aleatoric_std = aleatoric_scaled.cpu().numpy() * y_range
    epistemic_std = epistemic_scaled.cpu().numpy() * y_range
    total_std = np.sqrt(aleatoric_std ** 2 + epistemic_std ** 2)
    return prediction, total_std, aleatoric_std


def interval_metrics(y_true: np.ndarray, prediction: np.ndarray,
                     std: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "picp": float(np.mean(np.abs(y_true - prediction) <= Z_95 * std)),
    }


def metric_text(metrics: dict[str, float], ratio: float | None = None) -> str:
    text = (
        rf"$R^2$ = {metrics['r2']:.3f}" + "\n"
        + f"MAE = {metrics['mae']:.3f}\n"
        + f"PICP = {metrics['picp']:.3f}"
    )
    if ratio is not None:
        text += "\n" + rf"$\bar{{\sigma}}_{{data}}/\bar{{\sigma}}_{{total}}$ = {ratio:.2f}"
    return text


def generate_target_figure(data: pd.DataFrame, target: str, settings: dict,
                           x_min: np.ndarray, x_range: np.ndarray,
                           output_dir: Path, dpi: int,
                           mc_samples: int) -> list[dict[str, float | str | int]]:
    subset = data.dropna(subset=[target]).reset_index(drop=True)
    x_raw = subset[FEATURES].to_numpy(dtype=float)
    y = subset[target].to_numpy(dtype=float)
    indices = np.arange(len(subset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SPLIT_SEED
    )
    x_scaled = minmax_scale(x_raw, x_min, x_range)
    y_min = float(y.min())
    y_range = float(y.max() - y_min)

    y_test = y[test_idx]
    x_train = x_scaled[train_idx]
    x_test = x_scaled[test_idx]
    y_train = y[train_idx]

    bnn_pred, bnn_total_std, bnn_aleatoric_std = bnn_prediction(
        BEST_MODELS / f"{target}_{settings['model_seed']}_best_model.pth",
        x_test, y_min, y_range, mc_samples,
        mc_seed=20260907 + int(settings["model_seed"]),
    )
    bnn_metrics = interval_metrics(y_test, bnn_pred, bnn_total_std)
    bnn_ratio = float(np.mean(bnn_aleatoric_std / bnn_total_std))

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=RF_SEED,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_tree_predictions = np.vstack([
        tree.predict(x_test) for tree in rf.estimators_
    ])
    rf_pred = rf_tree_predictions.mean(axis=0)
    rf_std = rf_tree_predictions.std(axis=0, ddof=1)
    rf_metrics = interval_metrics(y_test, rf_pred, rf_std)

    gp = GaussianProcessRegressor(
        kernel=RBF() + WhiteKernel(),
        normalize_y=True,
        random_state=SPLIT_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_train, y_train)
    gp_pred, gp_std = gp.predict(x_test, return_std=True)
    gp_metrics = interval_metrics(y_test, gp_pred, gp_std)

    stem = f"Figure_{settings['figure']}_{settings['slug']}_BNN_RF_GP"
    # The layout is centralized in utils.Drawers so every entry point uses
    # the same publication styling and uncertainty-band semantics.
    plot_model_comparison_figure(
        y_true=y_test,
        model_specs=[
            {
                "title": "BNN (MC-Dropout)",
                "prediction": bnn_pred,
                "std": bnn_total_std,
                "point_color": MODEL_COMPARISON_COLORS["bnn"],
                "band_color": MODEL_COMPARISON_COLORS["bnn_total"],
                "band_label": "Total (Epistemic + Aleatoric)",
                "secondary_std": bnn_aleatoric_std,
                "secondary_color": MODEL_COMPARISON_COLORS["bnn_aleatoric"],
                "secondary_label": "Aleatoric",
                "metrics_text": metric_text(bnn_metrics, bnn_ratio),
            },
            {
                "title": "Random Forest",
                "prediction": rf_pred,
                "std": rf_std,
                "point_color": MODEL_COMPARISON_COLORS["rf"],
                "band_color": MODEL_COMPARISON_COLORS["rf_band"],
                "band_label": "Prediction std (single)",
                "metrics_text": metric_text(rf_metrics),
            },
            {
                "title": "Gaussian Process",
                "prediction": gp_pred,
                "std": gp_std,
                "point_color": MODEL_COMPARISON_COLORS["gp"],
                "band_color": MODEL_COMPARISON_COLORS["gp_band"],
                "band_label": "Posterior std (single)",
                "metrics_text": metric_text(gp_metrics),
            },
        ],
        target_title=settings["title"],
        output_dir=output_dir,
        stem=stem,
        dpi=dpi,
        z95=Z_95,
    )

    rows: list[dict[str, float | str | int]] = []
    for model_name, metrics in (
        ("BNN (MC-Dropout)", bnn_metrics),
        ("Random Forest", rf_metrics),
        ("Gaussian Process", gp_metrics),
    ):
        rows.append({
            "figure": int(settings["figure"]),
            "target": target,
            "model": model_name,
            "n_validation": len(test_idx),
            **metrics,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Destination directory for PNG, PDF, and metrics files.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--mc-samples", type=int, default=1000)
    return parser.parse_args()


def generate_model_comparison_figures(
    output_dir: Path = DEFAULT_OUTPUT,
    dpi: int = 600,
    mc_samples: int = 1000,
) -> pd.DataFrame:
    """Generate manuscript Figures 6--9 and return their metric table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The drawing function also applies this style, but configuring it once at
    # the batch boundary makes direct reuse from another pipeline explicit.
    configure_model_comparison_style()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    data = pd.read_excel(DATABASE, sheet_name="Sheet1")
    x_all = data[FEATURES].to_numpy(dtype=float)
    x_min = np.nanmin(x_all, axis=0)
    x_range = np.nanmax(x_all, axis=0) - x_min

    metric_rows = []
    for target, settings in TARGETS.items():
        metric_rows.extend(generate_target_figure(
            data, target, settings, x_min, x_range,
            output_dir, dpi, mc_samples,
        ))

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "Figure_6-9_model_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = generate_model_comparison_figures(
        output_dir=args.output_dir,
        dpi=args.dpi,
        mc_samples=args.mc_samples,
    )
    print(f"Generated Figures 6--9 in: {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
