# Uncertainty-aware BNN prediction and multi-objective design of FRCCs

![Graphical abstract](assets/graphical_abstract.png)

This repository is the reproducibility package for a study that connects an experimental database of fiber-reinforced cementitious composites (FRCCs) to uncertainty-aware material design. The workflow first learns compressive strength, ultimate tensile strength, tensile strain-energy density, and slump-flow diameter with Bayesian neural networks (BNNs). It then separates data and model uncertainty, interprets the learned relationships, and uses NSGA-II plus TOPSIS to identify practical performance–reliability compromises.

> The manuscript has been accepted. This public repository contains the complete reproducibility package, including the curated data, selected model checkpoints, training and validation scripts, optimization code, revision analyses, and manuscript figure pipeline.

## What is reproducible

The repository was audited against the final R2 manuscript, not the older draft figure numbers. Conceptual diagrams (Figures 1, 4, and 5) are deliberately excluded from the plotting pipeline. All quantitative figures are covered:

| Final figure | Content | Reproduction source |
|---|---|---|
| 2 | Spearman matrices before/after dimensionality reduction | `figure_02_03_database.py` |
| 3 | Input/output parameter distributions | `figure_02_03_database.py` |
| 6–9 | BNN, random forest, and Gaussian-process comparisons | `figure_06_09_model_comparison.py` |
| 10–13 | Prediction performance and aleatoric/epistemic/total uncertainty | `figure_10_13_uncertainty.py` |
| 14 | FA/C sensitivity | `figure_14_fa_c.py` |
| 15 | W/B–S/B sensitivity | `figure_15_wb_sb.py` |
| 16 | Fiber type and volume-fraction effects | `figure_16_fiber_effects.py` |
| 17 | SHAP parameter importance | `figure_17_shap.py` |
| 18 | PE-FRCC Pareto projections and TOPSIS radar chart | `figure_18_topsis.py` |
| 19 | TOPSIS preference-weight sensitivity | `figure_19_topsis_sensitivity.py` |
| 20 | Experimental photographs and tensile stress–strain response | `figure_20_experimental.py` |

The detailed final-caption audit and data provenance are recorded in [`docs/figure_audit.md`](docs/figure_audit.md).

Figures 14 and 15 were formerly Figures 12 and 13 in the plotting filenames; Figures 18–20 likewise inherited several older draft filenames. They are renamed here to match the submitted R2 manuscript.

## Reproduce the figures

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce_all.py
```

For a fast installation check:

```bash
python reproduce_all.py --quick --dpi 150
python -m unittest discover -s tests -v
```

Outputs are written to `figures/` as publication-resolution PNG/PDF files (and SVG for the Origin-replacement parametric plots). The generated folder is ignored by Git so running the pipeline does not dirty the repository.

## Repository map

```text
data/                   Curated database, parametric predictions, Pareto sets
models/                 Selected BBB and MC-Dropout model checkpoints
training/               Original model-training and seed-search programs
optimization/           NSGA-II and TOPSIS optimization program
manuscript_figures/     Final-numbered, publication-style figure renderers
revision_analysis/      R2 robustness and weighting-sensitivity analysis
notebooks/              Historical exploratory notebooks
assets/                 Graphical abstract used above
tests/                  Data/model integrity checks
```

All plotting modules use Times New Roman-compatible math text, enlarged publication typography, and the corrected tensile strain-energy-density unit `$G_t$ (kJ/m³)`.

## Data and provenance notes

- `data/frcc_database.xlsx` is the modeling database used for feature analysis and training/validation reconstruction.
- `data/index_analysis_predictions.xlsx` contains saved predictive means and total standard deviations used to redraw Figures 14–16 after the original Origin projects were lost.
- `data/optimization/` contains the PE and PVA Pareto fronts and TOPSIS rankings; `data/topsis_sensitivity/` contains the R2 decision-preference robustness analysis.
- Figure 20(a–d) consists of empirical photographs and is therefore copied as source media, not synthesized by plotting code.
- The raw acquisition/Origin project for the final Figure 20(e) stress–strain chart was unavailable. `figure20e_digitized.csv` is transparently reconstructed from the archived raster by `digitize_figure_20e.py`; it is derived display data, not raw instrument data. The reference raster is retained for traceability.

## Reproducibility details

Saved checkpoints are the selected manuscript models (MC-Dropout seeds 618, 424, 857, and 114 for the four targets; BBB seed 857 for the compressive-strength comparison). The plotting pipeline uses deterministic train/test splits and explicit random seeds. `figures/reproduction_manifest.json` records each run's outputs, resolution, and mode.

## Citation

Please cite the associated article once its bibliographic record is available. Publication metadata and the final archival DOI will be added when the paper is accepted.

## Contact

For questions about the database, model checkpoints, or optimization workflow, open an issue in this repository.
