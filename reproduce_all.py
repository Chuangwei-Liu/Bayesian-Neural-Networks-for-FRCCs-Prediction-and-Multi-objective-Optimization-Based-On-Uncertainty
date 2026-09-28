"""One-command reproduction of every quantitative manuscript figure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from manuscript_figures import figure_02_03_database as f0203
from manuscript_figures import figure_06_09_model_comparison as f0609
from manuscript_figures import figure_10_13_uncertainty as f1013
from manuscript_figures import figure_14_fa_c as f14
from manuscript_figures import figure_15_wb_sb as f15
from manuscript_figures import figure_16_fiber_effects as f16
from manuscript_figures import figure_17_shap as f17
from manuscript_figures import figure_18_topsis as f18
from manuscript_figures import figure_19_topsis_sensitivity as f19
from manuscript_figures import figure_20_experimental as f20
from manuscript_figures.common import FIGURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--quick", action="store_true", help="Use fewer Monte Carlo/SHAP samples for a smoke run.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip the optional SHAP calculation.")
    args = parser.parse_args()
    mc_samples = 40 if args.quick else 1000
    shap_samples = 8 if args.quick else 40
    generated: list[str] = []

    generated += map(str, f0203.generate(dpi=args.dpi))
    f0609.generate_model_comparison_figures(dpi=args.dpi, mc_samples=mc_samples)
    generated += [str(p) for p in (FIGURES / "model_comparison").glob("Figure_*")]
    generated += map(str, f1013.generate(dpi=args.dpi, samples=mc_samples))

    data14 = f14.load_figure_data(f14.DEFAULT_DATA, "Fig2")
    for spec in f14.PLOT_SPECS:
        generated += map(str, f14.draw_chart(data14, spec, f14.DEFAULT_OUTPUT_DIR, args.dpi))
    data15 = f15.load_figure_data(f15.DEFAULT_DATA, "Fig1")
    for spec in f15.PLOT_SPECS:
        generated += map(str, f15.draw_chart(data15, spec, f15.DEFAULT_OUTPUT_DIR, args.dpi))

    generated += map(str, f16.generate(dpi=args.dpi))
    if not args.skip_shap:
        generated += map(str, f17.generate(dpi=args.dpi, explain_samples=shap_samples,
                                             background_samples=max(8, shap_samples // 2)))
    generated += map(str, f18.generate(dpi=args.dpi))
    generated += map(str, f19.generate(dpi=args.dpi))
    generated += map(str, f20.generate(dpi=args.dpi))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": args.quick,
        "dpi": args.dpi,
        "outputs": sorted(set(generated)),
    }
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(manifest['outputs'])} files under {FIGURES}")


if __name__ == "__main__":
    main()
