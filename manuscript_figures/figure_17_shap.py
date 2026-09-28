"""Generate Figure 17 mean-absolute SHAP feature-importance panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from manuscript_figures.common import (DATA, FEATURES, FIGURES, LABELS,
    configure_style, load_pickled_model, minmax, save_figure)

TARGETS = {"CX(MPa)": 618, "UTX(Mpa)": 424, "G(KJm3)": 857, "MiniSF(cm)": 114}
FEATURE_LABELS = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", r"$V_f$", r"$I_f$"]


def generate(output_dir: Path = FIGURES / "figure_17", dpi: int = 600,
             explain_samples: int = 40, background_samples: int = 30):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Figure 17 requires the optional 'shap' package.") from exc
    configure_style(18)
    raw = pd.read_excel(DATA / "frcc_database.xlsx", sheet_name="Sheet1")
    reference = raw[FEATURES].to_numpy(float)
    rng = np.random.default_rng(33)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    table = []
    for ax, (target, seed) in zip(axes.flat, TARGETS.items()):
        frame = raw.dropna(subset=[target]).reset_index(drop=True)
        x, _, _ = minmax(frame[FEATURES].to_numpy(float), reference)
        y = frame[target].to_numpy(float)
        y_min, y_span = y.min(), y.max() - y.min()
        model = load_pickled_model(Path(__file__).resolve().parents[1] / "models" / "mc_dropout" / f"{target}_{seed}_best_model.pth", "mc")
        model.eval()
        def predict(values):
            with torch.no_grad():
                mean, _ = model(torch.as_tensor(values, dtype=torch.float32))
            return mean.numpy().ravel() * y_span + y_min
        bg_idx = rng.choice(len(x), min(background_samples, len(x)), replace=False)
        ex_idx = rng.choice(len(x), min(explain_samples, len(x)), replace=False)
        explainer = shap.PermutationExplainer(predict, x[bg_idx], feature_names=FEATURE_LABELS)
        values = explainer(x[ex_idx], max_evals=2 * len(FEATURES) + 1).values
        importance = np.abs(values).mean(0)
        # SF/C is constant in the final design domain and was omitted from the
        # published importance panels; retain it in the model input only.
        shown = np.arange(1, len(FEATURES))
        order = shown[np.argsort(importance[shown])]
        ax.barh(np.asarray(FEATURE_LABELS)[order], importance[order], color="#4682B4")
        ax.set_xlabel("Mean absolute SHAP value", fontweight="bold")
        ax.set_title(LABELS[target], fontweight="bold")
        ax.grid(axis="x", alpha=.25, linestyle="--")
        for feature, value in zip(FEATURES, importance):
            table.append({"target": target, "feature": feature, "mean_abs_shap": value})
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(output_dir / "Figure_17_SHAP_values.csv", index=False)
    return save_figure(fig, output_dir / "Figure_17_parameter_importance", dpi)


if __name__ == "__main__":
    generate()
