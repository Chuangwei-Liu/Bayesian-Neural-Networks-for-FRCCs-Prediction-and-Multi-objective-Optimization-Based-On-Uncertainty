"""Shared paths, typography, model definitions, and prediction helpers."""

from __future__ import annotations

import __main__
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FEATURES = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If"]

LABELS = {
    "CX(MPa)": r"$\sigma_{cs}$ (MPa)",
    "UTX(Mpa)": r"$\sigma_u$ (MPa)",
    "G(KJm3)": r"$G_t$ (kJ/m$^3$)",
    "MiniSF(cm)": r"$D_{spread}$ (cm)",
}


def configure_style(base_size: int = 18) -> None:
    """Apply the final manuscript's Times New Roman publication style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": base_size,
        "axes.titlesize": base_size + 4,
        "axes.labelsize": base_size + 4,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size + 2,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, output: Path, dpi: int = 600) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=dpi if suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def minmax(values: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.nanmin(reference, axis=0)
    span = np.nanmax(reference, axis=0) - low
    span = np.where(span == 0, 1.0, span)
    return (values - low) / span, low, span


class BayesianRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.output = nn.Linear(2 * hidden_size, output_size * 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        out = self.output(x)
        return out[:, :1], out[:, 1:]


class MCDropoutBNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.model = BayesianRegressor(input_size, hidden_size, output_size, dropout_p)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Linear_BBB(nn.Module):
    def __init__(self, input_features: int, output_features: int, prior_var: float = 1.0):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.w_mu = nn.Parameter(torch.empty(output_features, input_features).normal_(0, .1))
        self.w_rho = nn.Parameter(torch.empty(output_features, input_features).uniform_(-3, -2))
        self.b_mu = nn.Parameter(torch.empty(output_features).normal_(0, .1))
        self.b_rho = nn.Parameter(torch.empty(output_features).uniform_(-3, -2))
        self.prior = torch.distributions.Normal(0, math.sqrt(prior_var))
        self.w_epsilon = None
        self.b_epsilon = None

    def forward(self, x: torch.Tensor, sample: bool = True):
        if sample or self.w_epsilon is None:
            self.w_epsilon = torch.randn_like(self.w_mu)
            self.b_epsilon = torch.randn_like(self.b_mu)
        weight = self.w_mu + F.softplus(self.w_rho) * self.w_epsilon
        bias = self.b_mu + F.softplus(self.b_rho) * self.b_epsilon
        return F.linear(x, weight, bias)


class BBBNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 output_size: int = 1, prior_var: float = .5):
        super().__init__()
        self.fc1 = Linear_BBB(input_size, hidden_size, prior_var)
        self.fc2 = Linear_BBB(hidden_size, hidden_size, prior_var)
        self.out = Linear_BBB(hidden_size, output_size * 2, prior_var)

    def forward(self, x: torch.Tensor):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x))))).chunk(2, dim=1)


def load_pickled_model(path: Path, method: str) -> nn.Module:
    """Load legacy full-object PyTorch files without their original script module."""
    if method == "mc":
        __main__.BayesianRegressor = BayesianRegressor
        __main__.BayesianNeuralNetwork = MCDropoutBNN
    elif method == "bbb":
        __main__.Linear_BBB = Linear_BBB
        __main__.BayesianNeuralNetwork = BBBNeuralNetwork
    else:
        raise ValueError("method must be 'mc' or 'bbb'")
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_uncertainty(model: nn.Module, x_scaled: np.ndarray, y: np.ndarray,
                        method: str = "mc", samples: int = 300,
                        seed: int = 33) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(x_scaled, dtype=torch.float32)
    torch.manual_seed(seed)
    means, variances = [], []
    model.train()
    with torch.no_grad():
        for _ in range(samples):
            mean, logvar = model(x)
            means.append(mean.squeeze(1))
            variances.append(torch.exp(logvar.squeeze(1)))
    mean_stack = torch.stack(means)
    var_stack = torch.stack(variances)
    epistemic = mean_stack.std(0, unbiased=True).numpy()
    aleatoric = torch.sqrt(var_stack.mean(0)).numpy()
    prediction = mean_stack.mean(0).numpy()
    y_min, y_span = float(np.nanmin(y)), float(np.nanmax(y) - np.nanmin(y))
    prediction = prediction * y_span + y_min
    epistemic *= y_span
    aleatoric *= y_span
    total = np.sqrt(epistemic ** 2 + aleatoric ** 2)
    return prediction, aleatoric, epistemic, total
