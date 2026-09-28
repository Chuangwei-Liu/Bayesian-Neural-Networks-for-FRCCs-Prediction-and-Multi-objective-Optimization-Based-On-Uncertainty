# Final-manuscript figure audit

Audit basis: `ASE-26-0362_R2_Revised_Manuscript_Marked.docx`. The final R2 numbering differs from filenames retained during earlier revisions, so the repository uses the final numbering below.

| Figure | Final caption/topic | Status | Code/data path |
|---:|---|---|---|
| 1 | Methodological framework | Excluded: conceptual artwork | Graphical abstract retained separately in `assets/` |
| 2 | Spearman correlation matrices | Reproducible | `manuscript_figures/figure_02_03_database.py`; `data/frcc_database.xlsx` |
| 3 | Parameter distributions | Reproducible | `manuscript_figures/figure_02_03_database.py`; `data/frcc_database.xlsx` |
| 4 | BBB and MC-Dropout schematic | Excluded: conceptual artwork | — |
| 5 | Training process | Excluded: conceptual artwork | — |
| 6–9 | BNN/RF/GP model comparisons for four targets | Reproducible | `manuscript_figures/figure_06_09_model_comparison.py` |
| 10–13 | BNN performance and uncertainty decomposition | Reproducible | `manuscript_figures/figure_10_13_uncertainty.py` |
| 14 | FA/C effects | Reproducible from saved predictions; replaces lost Origin project | `manuscript_figures/figure_14_fa_c.py`; workbook sheet `Fig2` |
| 15 | W/B and S/B effects | Reproducible from saved predictions; replaces lost Origin project | `manuscript_figures/figure_15_wb_sb.py`; workbook sheet `Fig1` |
| 16 | Fiber type and volume-fraction effects | Reproducible from saved predictions | `manuscript_figures/figure_16_fiber_effects.py`; workbook sheet `Fig3` |
| 17 | BNN parameter importance | Reproducible | `manuscript_figures/figure_17_shap.py` |
| 18 | PE-FRCC Pareto fronts and TOPSIS ranking | Reproducible from archived Pareto/ranking tables | `manuscript_figures/figure_18_topsis.py`; `data/optimization/PE/` |
| 19 | TOPSIS weight sensitivity | Reproducible | `manuscript_figures/figure_19_topsis_sensitivity.py`; `data/topsis_sensitivity/` |
| 20(a–d) | Experimental setup photographs | Source media retained; not code-generated | `data/experimental_validation/source_photos/` |
| 20(e) | Tensile stress–strain curves | Reproducible from transparently digitized archived raster | `digitize_figure_20e.py`; `figure_20_experimental.py` |

The full optimization can be rerun with `optimization/run_nsga2_topsis.py`. Archived Pareto tables are retained so manuscript plots can be regenerated without repeating a long stochastic optimization. The R2 grouped-validation and TOPSIS preference analyses are retained in `revision_analysis/run_revision_experiments.py`.
