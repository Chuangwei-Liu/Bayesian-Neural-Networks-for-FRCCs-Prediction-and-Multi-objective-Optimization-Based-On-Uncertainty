from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor"
if VENDOR.exists():
    # Keep compiled packages from the active environment ahead of the bundled
    # fallback dependencies, which may target a different Python ABI.
    sys.path.append(str(VENDOR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn


ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
DATABASE = ROOT / "data" / "frcc_database.xlsx"
BEST_MODELS = ROOT / "models" / "mc_dropout"
SCALER_ROOT = ROOT / "models" / "scalers"
PARETO_ROOT = ROOT / "data" / "optimization"

RESULTS = ROOT / "revision_outputs" / "results"
FIGURES = ROOT / "revision_outputs" / "figures"
FOLD_MODELS = ROOT / "revision_outputs" / "grouped_cv_models"
for directory in (RESULTS, FIGURES, FOLD_MODELS):
    directory.mkdir(parents=True, exist_ok=True)

FEATURES = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If"]
LEVELS = [0.50, 0.80, 0.90, 0.95]
TARGETS = {
    "CX(MPa)": {
        "label": "Compressive strength",
        "short": "CX",
        "seed": 618,
        "hidden_size": 128,
        "lr": 0.005697,
        "dropout": 0.25,
        "unit": "MPa",
    },
    "UTX(Mpa)": {
        "label": "Ultimate tensile strength",
        "short": "UTX",
        "seed": 424,
        "hidden_size": 120,
        "lr": 0.009954,
        "dropout": 0.15,
        "unit": "MPa",
    },
    "G(KJm3)": {
        "label": "Fracture energy",
        "short": "G",
        "seed": 857,
        "hidden_size": 88,
        "lr": 0.008770,
        "dropout": 0.25,
        "unit": "kJ/m^3",
    },
    "MiniSF(cm)": {
        "label": "Mini-slump flow spread",
        "short": "MiniSF",
        "seed": 114,
        "hidden_size": 120,
        "lr": 0.008849,
        "dropout": 0.20,
        "unit": "cm",
    },
}


class BayesianRegressor(nn.Module):
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
    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.model = BayesianRegressor(
            input_size, hidden_size, output_size, dropout_p
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, logvar = self(x)
        inv_var = torch.exp(-logvar)
        return torch.sum(inv_var * (y - mean) ** 2 + logvar) / y.shape[0]


torch.serialization.add_safe_globals([BayesianRegressor, BayesianNeuralNetwork])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_database() -> pd.DataFrame:
    data = pd.read_excel(DATABASE, sheet_name="Sheet1")
    data["Title_filled"] = data["Title"].ffill().fillna("Unknown title")
    data["Author_filled"] = data["Author"].ffill().fillna("Unknown author")
    data["Year_filled"] = data["Year"].ffill().fillna("Unknown year")
    data["Study_ID"] = (
        data["Author_filled"].astype(str).str.strip()
        + " | " + data["Year_filled"].astype(str).str.strip()
        + " | " + data["Title_filled"].astype(str).str.strip()
    )
    return data


def predictive_distribution(model: nn.Module, x_scaled: np.ndarray,
                            target_range: float, mc_samples: int,
                            mc_seed: int) -> tuple[np.ndarray, ...]:
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mean, logvar = model(x_tensor)
        data_var = torch.exp(logvar).squeeze(-1)

        torch.manual_seed(mc_seed)
        model.train()
        mc_means = torch.stack([model(x_tensor)[0] for _ in range(mc_samples)])
        model_var = mc_means.var(dim=0, unbiased=True).squeeze(-1)

    mean_scaled = mean.squeeze(-1).cpu().numpy()
    data_std = data_var.sqrt().cpu().numpy() * target_range
    model_std = model_var.sqrt().cpu().numpy() * target_range
    total_std = np.sqrt(data_std ** 2 + model_std ** 2)
    return mean_scaled, total_std, data_std, model_std


def add_interval_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for level in LEVELS:
        z = norm.ppf((1.0 + level) / 2.0)
        key = int(level * 100)
        result[f"lower_{key}"] = result["y_pred"] - z * result["total_std"]
        result[f"upper_{key}"] = result["y_pred"] + z * result["total_std"]
        result[f"covered_{key}"] = (
            (result["y_true"] >= result[f"lower_{key}"])
            & (result["y_true"] <= result[f"upper_{key}"])
        )
    return result


def summarize_predictions(frame: pd.DataFrame, target: str,
                          evaluation: str, fold: int | None = None) -> dict:
    row = {
        "evaluation": evaluation,
        "target": target,
        "short": TARGETS[target]["short"],
        "fold": fold,
        "n": len(frame),
        "r2": r2_score(frame["y_true"], frame["y_pred"]),
        "mae": mean_absolute_error(frame["y_true"], frame["y_pred"]),
        "rmse": math.sqrt(mean_squared_error(frame["y_true"], frame["y_pred"])),
    }
    target_range = frame["target_range"].iloc[0]
    for level in LEVELS:
        key = int(level * 100)
        width = frame[f"upper_{key}"] - frame[f"lower_{key}"]
        row[f"picp_{key}"] = frame[f"covered_{key}"].mean()
        row[f"mpiw_{key}"] = width.mean()
        row[f"nmpiw_{key}"] = width.mean() / target_range
    return row


def leakage_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, params in TARGETS.items():
        subset = data.dropna(subset=[target]).reset_index(drop=False)
        indices = np.arange(len(subset))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=params["seed"]
        )
        train_studies = set(subset.loc[train_idx, "Study_ID"])
        test_studies = subset.loc[test_idx, "Study_ID"]
        overlap = test_studies.isin(train_studies)
        rows.append({
            "target": target,
            "short": params["short"],
            "n_samples": len(subset),
            "n_studies": subset["Study_ID"].nunique(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "leaked_test_samples": int(overlap.sum()),
            "leakage_fraction": float(overlap.mean()),
            "test_studies": int(test_studies.nunique()),
            "overlapping_test_studies": len(set(test_studies) & train_studies),
        })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "random_split_study_leakage.csv", index=False)
    return result


def random_holdout_calibration(data: pd.DataFrame,
                               mc_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions = []
    summaries = []
    feature_scaler_path = SCALER_ROOT / "feature_scaler" / "input_feature_scaler.pkl"
    import pickle
    with feature_scaler_path.open("rb") as stream:
        feature_scaler = pickle.load(stream)

    for target, params in TARGETS.items():
        subset = data.dropna(subset=[target]).reset_index(drop=False)
        indices = np.arange(len(subset))
        _, test_idx = train_test_split(
            indices, test_size=0.2, random_state=params["seed"]
        )
        with (SCALER_ROOT / "target_scaler" / f"{target}_target_scaler.pkl").open("rb") as stream:
            target_scaler = pickle.load(stream)

        x_scaled = feature_scaler.transform(subset.loc[test_idx, FEATURES].to_numpy())
        model = torch.load(
            BEST_MODELS / f"{target}_{params['seed']}_best_model.pth",
            map_location="cpu", weights_only=False,
        )
        mean_scaled, total_std, data_std, model_std = predictive_distribution(
            model, x_scaled, float(target_scaler.data_range_[0]),
            mc_samples, 20260901 + params["seed"],
        )
        y_pred = target_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        frame = pd.DataFrame({
            "evaluation": "random_holdout",
            "target": target,
            "short": params["short"],
            "seed": params["seed"],
            "database_row": subset.loc[test_idx, "index"].to_numpy() + 2,
            "study_id": subset.loc[test_idx, "Study_ID"].to_numpy(),
            "y_true": subset.loc[test_idx, target].to_numpy(dtype=float),
            "y_pred": y_pred,
            "aleatoric_std": data_std,
            "epistemic_std": model_std,
            "total_std": total_std,
            "target_range": float(target_scaler.data_range_[0]),
        })
        frame = add_interval_columns(frame)
        all_predictions.append(frame)
        summaries.append(summarize_predictions(frame, target, "random_holdout"))

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.DataFrame(summaries)
    predictions.to_csv(RESULTS / "random_holdout_predictions.csv", index=False)
    summary.to_csv(RESULTS / "random_holdout_calibration_summary.csv", index=False)
    return predictions, summary


def train_fold_model(x_train: np.ndarray, y_train: np.ndarray,
                     params: dict, seed: int, epochs: int) -> tuple[nn.Module, list[float]]:
    set_seed(seed)
    model = BayesianNeuralNetwork(
        input_size=len(FEATURES),
        hidden_size=params["hidden_size"],
        dropout_p=params["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.nll(x_tensor, y_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            history.append(float(loss.detach().cpu()))
    return model, history


def grouped_cross_validation(data: pd.DataFrame, epochs: int,
                             mc_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    fold_summaries = []
    histories = []

    for target, params in TARGETS.items():
        subset = data.dropna(subset=[target]).reset_index(drop=False)
        x_raw = subset[FEATURES].to_numpy(dtype=float)
        y_raw = subset[[target]].to_numpy(dtype=float)
        groups = subset["Study_ID"].to_numpy()
        # The non-shuffled GroupKFold greedily balances sample counts while still
        # keeping each study wholly inside one fold. With only 13--32 studies per
        # target, shuffled GroupKFold balances group counts rather than sample
        # counts and can create test folds with only 5--7 observations.
        splitter = GroupKFold(n_splits=5)

        for fold, (train_idx, test_idx) in enumerate(
                splitter.split(x_raw, y_raw, groups), start=1):
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
            x_scaler = MinMaxScaler().fit(x_raw[train_idx])
            y_scaler = MinMaxScaler().fit(y_raw[train_idx])
            x_train = x_scaler.transform(x_raw[train_idx])
            y_train = y_scaler.transform(y_raw[train_idx])
            x_test = x_scaler.transform(x_raw[test_idx])

            fold_seed = params["seed"] * 100 + fold
            model, history = train_fold_model(
                x_train, y_train, params, fold_seed, epochs
            )
            model_path = FOLD_MODELS / f"{params['short']}_fold{fold}_state_dict.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "target": target,
                "fold": fold,
                "params": params,
                "x_min": x_scaler.data_min_,
                "x_max": x_scaler.data_max_,
                "y_min": y_scaler.data_min_,
                "y_max": y_scaler.data_max_,
                "train_studies": sorted(set(groups[train_idx])),
                "test_studies": sorted(set(groups[test_idx])),
            }, model_path)

            target_range = float(y_scaler.data_range_[0])
            mean_scaled, total_std, data_std, model_std = predictive_distribution(
                model, x_test, target_range, mc_samples,
                20260901 + fold_seed,
            )
            y_pred = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            frame = pd.DataFrame({
                "evaluation": "grouped_5fold",
                "target": target,
                "short": params["short"],
                "fold": fold,
                "training_seed": fold_seed,
                "database_row": subset.loc[test_idx, "index"].to_numpy() + 2,
                "study_id": groups[test_idx],
                "y_true": y_raw[test_idx].ravel(),
                "y_pred": y_pred,
                "aleatoric_std": data_std,
                "epistemic_std": model_std,
                "total_std": total_std,
                "target_range": target_range,
            })
            frame = add_interval_columns(frame)
            predictions.append(frame)
            fold_summaries.append(
                summarize_predictions(frame, target, "grouped_5fold", fold)
            )
            for checkpoint, loss in enumerate(history):
                histories.append({
                    "target": target,
                    "fold": fold,
                    "checkpoint": checkpoint,
                    "loss": loss,
                })
            print(
                f"Completed {params['short']} fold {fold}/5: "
                f"n_train={len(train_idx)}, n_test={len(test_idx)}"
            )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.DataFrame(fold_summaries)
    pooled_rows = []
    for target in TARGETS:
        pooled = prediction_frame[prediction_frame["target"] == target]
        pooled_rows.append(summarize_predictions(pooled, target, "grouped_5fold_pooled"))
    pooled_frame = pd.DataFrame(pooled_rows)

    prediction_frame.to_csv(RESULTS / "grouped_5fold_predictions.csv", index=False)
    fold_frame.to_csv(RESULTS / "grouped_5fold_fold_metrics.csv", index=False)
    pooled_frame.to_csv(RESULTS / "grouped_5fold_pooled_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(RESULTS / "grouped_5fold_training_history.csv", index=False)
    return prediction_frame, fold_frame, pooled_frame


def topsis_scores(matrix: np.ndarray, weights: np.ndarray,
                  benefit: np.ndarray,
                  normalization: str = "minmax") -> np.ndarray:
    """Calculate TOPSIS scores with an explicit normalization convention.

    ``minmax`` implements Eq. (23) in the manuscript.  ``vector`` is retained
    only so the original implementation can still be reproduced and compared.
    """
    if normalization == "minmax":
        column_min = matrix.min(axis=0)
        column_range = matrix.max(axis=0) - column_min
        column_range[column_range == 0] = 1.0
        normalized = (matrix - column_min) / column_range
    elif normalization == "vector":
        denominator = np.sqrt(np.sum(matrix ** 2, axis=0))
        denominator[denominator == 0] = 1.0
        normalized = matrix / denominator
    else:
        raise ValueError(f"Unsupported TOPSIS normalization: {normalization}")

    weighted = normalized * (weights / weights.sum())
    ideal_best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    distance_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    distance_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))
    return distance_worst / (distance_best + distance_worst)


def topsis_sensitivity(
        normalization: str = "minmax",
        output_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    objective_cols = [
        "UTX(MPa)", "G(KJm3)", "MiniSF(cm)",
        "UTX Uncertainty", "G Uncertainty", "MiniSF Uncertainty",
    ]
    design_cols = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", "V_f"]
    benefit = np.array([True, True, True, False, False, False])
    schemes = {"50/50": 0.50, "60/40": 0.60, "70/30": 0.70}
    ranking_outputs = []
    top_rows = []
    correlations = []
    experimental_target = np.array([0.0, 0.25, 0.25, 0.32, 0.001, 2.0])

    for fiber in ("PE", "PVA"):
        frame = pd.read_csv(PARETO_ROOT / fiber / "pareto_front.csv")
        matrix = frame[objective_cols].to_numpy(dtype=float)
        rank_vectors = {}
        design_ranges = frame[design_cols].max().to_numpy() - frame[design_cols].min().to_numpy()
        design_ranges[design_ranges == 0] = 1.0
        experimental_idx = int(np.argmin(np.sum(
            ((frame[design_cols].to_numpy() - experimental_target) / design_ranges) ** 2,
            axis=1,
        ))) if fiber == "PE" else None

        output = frame.copy()
        output.insert(0, "pareto_index", np.arange(len(frame)))
        for scheme, performance_weight in schemes.items():
            weights = np.array(
                [performance_weight / 3] * 3
                + [(1 - performance_weight) / 3] * 3
            )
            scores = topsis_scores(matrix, weights, benefit, normalization)
            order = np.argsort(-scores)
            ranks = np.empty(len(scores), dtype=int)
            ranks[order] = np.arange(1, len(scores) + 1)
            rank_vectors[scheme] = ranks
            output[f"score_{scheme}"] = scores
            output[f"rank_{scheme}"] = ranks

            best = output.iloc[order[0]].to_dict()
            best.update({
                "fiber": fiber,
                "normalization": normalization,
                "weight_scheme": scheme,
                "performance_weight": performance_weight,
                "uncertainty_weight": 1 - performance_weight,
                "experimental_design_rank": (
                    int(ranks[experimental_idx]) if experimental_idx is not None else np.nan
                ),
            })
            top_rows.append(best)

        for left, right in (("50/50", "60/40"), ("60/40", "70/30"), ("50/50", "70/30")):
            rho, p_value = spearmanr(rank_vectors[left], rank_vectors[right])
            correlations.append({
                "fiber": fiber,
                "comparison": f"{left} vs {right}",
                "spearman_rho": rho,
                "p_value": p_value,
                "top10_overlap": len(
                    set(np.where(rank_vectors[left] <= 10)[0])
                    & set(np.where(rank_vectors[right] <= 10)[0])
                ),
            })
        output["fiber"] = fiber
        output["normalization"] = normalization
        output.to_csv(
            RESULTS / f"corrected_topsis_rankings_{fiber}{output_suffix}.csv",
            index=False,
        )
        ranking_outputs.append(output)

    top_frame = pd.DataFrame(top_rows)
    correlation_frame = pd.DataFrame(correlations)
    all_rankings = pd.concat(ranking_outputs, ignore_index=True)
    correlation_frame["normalization"] = normalization
    top_frame.to_csv(
        RESULTS / f"corrected_topsis_top_solutions{output_suffix}.csv", index=False
    )
    correlation_frame.to_csv(
        RESULTS / f"corrected_topsis_rank_correlations{output_suffix}.csv", index=False
    )
    return all_rankings, top_frame, correlation_frame


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_topsis_rank_figure(
        topsis_top: pd.DataFrame,
        output_suffix: str = "",
        normalization: str = "minmax",
) -> None:
    pe = topsis_top[topsis_top["fiber"] == "PE"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(pe["weight_scheme"], pe["experimental_design_rank"],
                  color=["#A5A5A5", "#5B9BD5", "#ED7D31"])
    ax.invert_yaxis()
    ax.set_ylabel("Rank of experimentally tested PE design (1 = best)")
    ax.set_xlabel("Performance / uncertainty weight")
    ax.set_title(f"Corrected TOPSIS sensitivity ({normalization} normalization)")
    for bar, value in zip(bars, pe["experimental_design_rank"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 2,
                f"Rank {int(value)}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"figure_5_topsis_experimental_design_rank{output_suffix}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_topsis_comparison_radars(
        all_rankings: pd.DataFrame,
        topsis_top: pd.DataFrame,
        output_suffix: str = "",
        normalization: str = "minmax",
) -> None:
    """Compare each alternative-weight Rank-1 with the tested PE formulation.

    The blue reference is deliberately labelled as the experimentally tested
    formulation/original 60/40 selection.  It is not relabelled as the Rank-1
    solution under the recalculated TOPSIS convention.
    """
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    })
    pe = all_rankings[all_rankings["fiber"] == "PE"].copy()
    tested = pe[pe["pareto_index"] == 28].iloc[0]
    categories = [
        r"$\sigma_u$", r"$G_t$", r"$D_{spread}$",
        r"$\Delta\sigma_u$", r"$\Delta G_t$", r"$\Delta D_{spread}$",
    ]
    max_values = np.array([
        pe["UTX(MPa)"].max(),
        pe["G(KJm3)"].max(),
        pe["MiniSF(cm)"].max(),
        1.0 / pe["UTX Uncertainty"].min(),
        1.0 / pe["G Uncertainty"].min(),
        1.0 / pe["MiniSF Uncertainty"].min(),
    ])

    def radar_values(row: pd.Series) -> np.ndarray:
        raw = np.array([
            row["UTX(MPa)"],
            row["G(KJm3)"],
            row["MiniSF(cm)"],
            1.0 / row["UTX Uncertainty"],
            1.0 / row["G Uncertainty"],
            1.0 / row["MiniSF Uncertainty"],
        ], dtype=float)
        return raw / max_values

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.append(angles, angles[0])
    tested_values = np.append(radar_values(tested), radar_values(tested)[0])

    for figure_number, scheme in ((6, "50/50"), (7, "70/30")):
        top_record = topsis_top[
            (topsis_top["fiber"] == "PE")
            & (topsis_top["weight_scheme"] == scheme)
        ].iloc[0]
        best_index = int(top_record["pareto_index"])
        best = pe[pe["pareto_index"] == best_index].iloc[0]
        best_values = np.append(radar_values(best), radar_values(best)[0])
        fig, ax = plt.subplots(figsize=(9.0, 7.6), subplot_kw={"polar": True})
        ax.plot(
            angles, best_values, "o-", color="#C00000", linewidth=3.2,
            markersize=8,
            label=f"Rank-1 at {scheme} (Pareto point {best_index})",
        )
        ax.fill(angles, best_values, color="#C00000", alpha=0.12)
        ax.plot(
            angles, tested_values, "o--", color="#2F5597", linewidth=3.2,
            markersize=8,
            label="Experimentally tested formulation\n(original 60/40 selection)",
        )
        ax.fill(angles, tested_values, color="#2F5597", alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            categories, fontsize=20, fontname="Times New Roman",
        )
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["0.2", "0.4", "0.6", "0.8", "1.0"],
            fontsize=14, fontname="Times New Roman",
        )
        ax.set_rlabel_position(18)
        ax.grid(alpha=0.35)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.10),
            frameon=False,
            prop={"family": "Times New Roman", "size": 20},
            labelspacing=0.9,
            handlelength=3.0,
        )
        fig.tight_layout(rect=(0.02, 0.16, 0.98, 0.99))
        scheme_token = scheme.replace("/", "_")
        fig.savefig(
            FIGURES / (
                f"figure_{figure_number}_topsis_radar_{scheme_token}"
                f"{output_suffix}.png"
            ),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)


def make_figures(leakage: pd.DataFrame, random_summary: pd.DataFrame,
                 grouped_predictions: pd.DataFrame, grouped_pooled: pd.DataFrame,
                 topsis_top: pd.DataFrame,
                 topsis_output_suffix: str = "",
                 topsis_normalization: str = "minmax") -> None:
    set_plot_style()
    colors = ["#2F5597", "#ED7D31", "#70AD47", "#8064A2"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = leakage["leakage_fraction"] * 100
    bars = ax.bar(leakage["short"], vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Test samples sharing a study with training set (%)")
    ax.set_title("Study-level leakage in the original random holdout")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_random_split_study_leakage.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for idx, target in enumerate(TARGETS):
        row = random_summary[random_summary["target"] == target].iloc[0]
        empirical = [row[f"picp_{int(level * 100)}"] for level in LEVELS]
        ax.plot(np.array(LEVELS) * 100, np.array(empirical) * 100,
                marker="o", linewidth=2, color=colors[idx],
                label=TARGETS[target]["short"])
    ax.plot([45, 100], [45, 100], "--", color="#555555", label="Ideal calibration")
    ax.set_xlim(45, 100)
    ax.set_ylim(30, 105)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage, PICP (%)")
    ax.set_title("Calibration of the selected random-holdout BNNs")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_random_holdout_calibration.png", bbox_inches="tight")
    plt.close(fig)

    merged = random_summary[["target", "short", "r2"]].merge(
        grouped_pooled[["target", "r2"]], on="target", suffixes=("_random", "_grouped")
    )
    x = np.arange(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - width / 2, merged["r2_random"], width,
           label="Random holdout", color="#5B9BD5")
    ax.bar(x + width / 2, merged["r2_grouped"], width,
           label="Study-grouped 5-fold OOF", color="#ED7D31")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, merged["short"])
    ax.set_ylabel("R-squared")
    ax.set_title("Generalization drops when entire studies are held out")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_3_random_vs_grouped_r2.png", bbox_inches="tight")
    plt.close(fig)

    grouped_rows = []
    for target in TARGETS:
        subset = grouped_predictions[grouped_predictions["target"] == target]
        for level in LEVELS:
            grouped_rows.append({
                "target": target,
                "short": TARGETS[target]["short"],
                "level": level,
                "picp": subset[f"covered_{int(level * 100)}"].mean(),
            })
    grouped_cal = pd.DataFrame(grouped_rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for idx, target in enumerate(TARGETS):
        subset = grouped_cal[grouped_cal["target"] == target]
        ax.plot(subset["level"] * 100, subset["picp"] * 100,
                marker="o", linewidth=2, color=colors[idx],
                label=TARGETS[target]["short"])
    ax.plot([45, 100], [45, 100], "--", color="#555555", label="Ideal calibration")
    ax.set_xlim(45, 100)
    ax.set_ylim(20, 105)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Pooled OOF coverage, PICP (%)")
    ax.set_title("Calibration under study-grouped five-fold validation")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_4_grouped_cv_calibration.png", bbox_inches="tight")
    plt.close(fig)

    make_topsis_rank_figure(
        topsis_top,
        output_suffix=topsis_output_suffix,
        normalization=topsis_normalization,
    )


def write_run_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "database": str(DATABASE),
        "features": FEATURES,
        "targets": TARGETS,
        "nominal_coverage_levels": LEVELS,
        "epochs_per_grouped_fold": args.epochs,
        "mc_samples": args.mc_samples,
        "grouped_split": "GroupKFold(n_splits=5); deterministic sample-balanced study folds",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (RESULTS / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--mc-samples", type=int, default=1000)
    parser.add_argument("--skip-grouped", action="store_true")
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Skip manuscript Figures 6--9 (BNN/RF/GP comparison).",
    )
    parser.add_argument(
        "--skip-index-analysis-bars",
        action="store_true",
        help="Skip manuscript Figure 15(a--c) grouped bar charts.",
    )
    parser.add_argument(
        "--skip-fa-c-sensitivity",
        action="store_true",
        help="Skip manuscript Figure 14(a--c) FA/C sensitivity charts.",
    )
    parser.add_argument("--only-topsis", action="store_true")
    parser.add_argument(
        "--topsis-normalization",
        choices=("minmax", "vector"),
        default="minmax",
    )
    parser.add_argument(
        "--topsis-output-suffix",
        default="",
        help="Suffix added before .csv/.png, for example _minmax.",
    )
    args = parser.parse_args()

    if args.only_topsis:
        all_rankings, topsis_top, _ = topsis_sensitivity(
            normalization=args.topsis_normalization,
            output_suffix=args.topsis_output_suffix,
        )
        set_plot_style()
        make_topsis_rank_figure(
            topsis_top,
            output_suffix=args.topsis_output_suffix,
            normalization=args.topsis_normalization,
        )
        make_topsis_comparison_radars(
            all_rankings,
            topsis_top,
            output_suffix=args.topsis_output_suffix,
            normalization=args.topsis_normalization,
        )
        print(f"TOPSIS sensitivity completed. Results: {RESULTS}")
        return

    write_run_manifest(args)
    data = load_database()
    leakage = leakage_audit(data)
    random_predictions, random_summary = random_holdout_calibration(
        data, args.mc_samples
    )

    if args.skip_grouped:
        grouped_predictions = pd.read_csv(RESULTS / "grouped_5fold_predictions.csv")
        grouped_pooled = pd.read_csv(RESULTS / "grouped_5fold_pooled_summary.csv")
    else:
        grouped_predictions, _, grouped_pooled = grouped_cross_validation(
            data, args.epochs, args.mc_samples
        )

    all_rankings, topsis_top, _ = topsis_sensitivity(
        normalization=args.topsis_normalization,
        output_suffix=args.topsis_output_suffix,
    )
    make_figures(
        leakage, random_summary, grouped_predictions, grouped_pooled, topsis_top,
        topsis_output_suffix=args.topsis_output_suffix,
        topsis_normalization=args.topsis_normalization,
    )
    make_topsis_comparison_radars(
        all_rankings,
        topsis_top,
        output_suffix=args.topsis_output_suffix,
        normalization=args.topsis_normalization,
    )
    if not args.skip_model_comparison:
        # Keep Figures 6--9 in the same reproducible entry point as the other
        # revision figures.  The actual panel layout lives in utils.Drawers.
        from manuscript_figures.figure_06_09_model_comparison import generate_model_comparison_figures

        comparison_metrics = generate_model_comparison_figures(
            output_dir=FIGURES / "model_comparison",
            dpi=600,
            mc_samples=args.mc_samples,
        )
        print("Model-comparison metrics:")
        print(comparison_metrics.to_string(index=False))
    if not args.skip_index_analysis_bars:
        from manuscript_figures.figure_15_wb_sb import (
            DEFAULT_DATA,
            DEFAULT_OUTPUT_DIR,
            PLOT_SPECS,
            draw_chart,
            load_figure_data,
        )

        index_data = load_figure_data(DEFAULT_DATA)
        for plot_spec in PLOT_SPECS:
            draw_chart(index_data, plot_spec, DEFAULT_OUTPUT_DIR, dpi=300)
        print(f"Figure 15(a--c): {DEFAULT_OUTPUT_DIR}")
    if not args.skip_fa_c_sensitivity:
        from manuscript_figures.figure_14_fa_c import (
            DEFAULT_DATA as FA_C_DATA,
            DEFAULT_OUTPUT_DIR as FA_C_OUTPUT_DIR,
            PLOT_SPECS as FA_C_PLOT_SPECS,
            draw_chart as draw_fa_c_chart,
            load_figure_data as load_fa_c_data,
        )

        fa_c_data = load_fa_c_data(FA_C_DATA)
        for plot_spec in FA_C_PLOT_SPECS:
            draw_fa_c_chart(fa_c_data, plot_spec, FA_C_OUTPUT_DIR, dpi=300)
        print(f"Figure 14(a--c): {FA_C_OUTPUT_DIR}")
    print(f"Revision experiments completed. Results: {RESULTS}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
