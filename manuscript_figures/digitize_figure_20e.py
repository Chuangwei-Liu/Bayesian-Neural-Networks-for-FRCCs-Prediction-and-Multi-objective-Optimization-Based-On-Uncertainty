"""Recover stress-strain pixels from the archived Figure 20(e) raster.

The original Origin project and raw acquisition files for this final PE-FRCC
plot were unavailable.  This script provides a transparent, deterministic
fallback: it extracts the three colored curves in calibrated plot coordinates.
The resulting CSV is derived data and must not be confused with raw instrument
output.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"

REFERENCE = DATA / "experimental_validation" / "figure20e_reference.png"
OUTPUT = DATA / "experimental_validation" / "figure20e_digitized.csv"


def digitize(reference: Path = REFERENCE, output: Path = OUTPUT) -> Path:
    image = np.asarray(Image.open(reference).convert("RGB"))
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    masks = {
        "PE-FRCCs-1": (np.max(image, 2) - np.min(image, 2) < 18) & (image.mean(2) > 45) & (image.mean(2) < 135),
        "PE-FRCCs-2": (red > 155) & (red > 1.45 * green) & (red > 1.45 * blue),
        "PE-FRCCs-3": (blue > 115) & (blue > 1.45 * red) & (blue > 1.18 * green),
    }
    # Pixel calibration read from the figure's integer tick marks.
    x0, x6, y0, y12 = 210.0, 2048.0, 2001.5, 199.0
    records = []
    for specimen, mask in masks.items():
        yy, xx = np.where(mask)
        keep = (xx >= x0) & (xx <= 2095) & (yy >= y12) & (yy <= y0)
        # Remove the lower-right legend swatches.
        keep &= ~((xx > 1190) & (yy > 1490))
        xx, yy = xx[keep], yy[keep]
        strain = (xx - x0) * 6.0 / (x6 - x0)
        stress = (y0 - yy) * 12.0 / (y0 - y12)
        valid = (strain >= 0) & (strain <= 6.2) & (stress >= 0) & (stress <= 11)
        # Pixel clouds are thinned deterministically while retaining vertical drops.
        for x, y in zip(strain[valid][::3], stress[valid][::3]):
            records.append({"specimen": specimen, "strain_percent": x, "stress_mpa": y})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(digitize())
