"""Generate Figures 10--13: BNN performance and uncertainty decomposition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from manuscript_figures.common import (DATA, FEATURES, FIGURES, LABELS,
    configure_style, load_pickled_model, minmax, predict_uncertainty, save_figure)

TARGETS = {
    "CX(MPa)": (10, 618),
    "UTX(Mpa)": (11, 424),
    "G(KJm3)": (12, 857),
    "MiniSF(cm)": (13, 114),
}


def _row(axs, y_true, pred, aleatoric, epistemic, total, title, method):
    parts = ((aleatoric, "Aleatoric"), (epistemic, "Epistemic"), (total, "Total"))
    order = np.argsort(y_true)
    for ax, (std, name) in zip(axs, parts):
        ax.fill_between(y_true[order], pred[order] - 1.96 * std[order],
                        pred[order] + 1.96 * std[order], alpha=.22, color="#ED7D31")
        ax.scatter(y_true, pred, s=42, color="#0072B2", edgecolor="white", linewidth=.4)
        lo = min(y_true.min(), (pred - 1.96 * std).min())
        hi = max(y_true.max(), (pred + 1.96 * std).max())
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.3)
        ax.set_title(f"{method}: {name} uncertainty", fontweight="bold")
        ax.set_xlabel("Experimental value", fontweight="bold")
        ax.set_ylabel("Predicted value", fontweight="bold")
        ax.grid(alpha=.2)
        picp = np.mean(np.abs(y_true - pred) <= 1.96 * std)
        ax.text(.04, .96, rf"$R^2$ = {r2_score(y_true, pred):.3f}" + "\n"
                + f"MAE = {mean_absolute_error(y_true, pred):.3f}\nPICP = {picp:.3f}",
                transform=ax.transAxes, va="top", fontsize=20,
                bbox={"facecolor": "white", "alpha": .86, "edgecolor": "#bbbbbb"})


def generate(output_dir: Path = FIGURES / "figure_10_13", dpi: int = 600,
             samples: int = 300):
    configure_style(18)
    raw = pd.read_excel(DATA / "frcc_database.xlsx", sheet_name="Sheet1")
    all_x = raw[FEATURES].to_numpy(float)
    outputs = []
    for target, (figure_no, seed) in TARGETS.items():
        frame = raw.dropna(subset=[target]).reset_index(drop=True)
        x_scaled, _, _ = minmax(frame[FEATURES].to_numpy(float), all_x)
        y = frame[target].to_numpy(float)
        _, test_idx = train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed)
        methods = [("MC-Dropout", "mc", Path("models/mc_dropout") / f"{target}_{seed}_best_model.pth")]
        if target == "CX(MPa)":
            methods.insert(0, ("BBB", "bbb", Path("models/bbb/CX(MPa)_857_best_model.pth")))
        fig, axes = plt.subplots(len(methods), 3, figsize=(20, 6.5 * len(methods)), squeeze=False)
        for row, (name, method, relpath) in enumerate(methods):
            model = load_pickled_model(Path(__file__).resolve().parents[1] / relpath, method)
            pred, ale, epi, total = predict_uncertainty(model, x_scaled[test_idx], y,
                                                        method, samples, seed)
            _row(axes[row], y[test_idx], pred, ale, epi, total, LABELS[target], name)
        fig.suptitle(f"{LABELS[target]} prediction and uncertainty", fontweight="bold", y=1.01)
        fig.tight_layout()
        outputs += save_figure(fig, output_dir / f"Figure_{figure_no:02d}_uncertainty", dpi)
    return outputs


if __name__ == "__main__":
    generate()
