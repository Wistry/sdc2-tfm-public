#!/usr/bin/env python3
"""Filter the 40GB truth catalogue outside the central 10GB development region."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import LARGE_CUBE, LARGE_TRUTH, ensure_dir, outside_10gb_region


def main() -> None:
    if not LARGE_TRUTH.exists():
        raise FileNotFoundError(f"Missing 40GB truth catalogue: {LARGE_TRUTH}")
    if not LARGE_CUBE.exists():
        raise FileNotFoundError(f"Missing 40GB FITS cube for WCS conversion: {LARGE_CUBE}")
    out_dir = ensure_dir(VALIDATION_ROOT / "outputs" / "external_truth")
    truth = pd.read_csv(LARGE_TRUTH, sep=r"\s+", comment="#", engine="python")
    required = {"ra", "dec", "central_freq"}
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"Truth catalogue missing required WCS columns: {missing}")
    header = fits.getheader(LARGE_CUBE)
    wcs = WCS(header)
    pix = wcs.all_world2pix(truth["ra"].to_numpy(), truth["dec"].to_numpy(), truth["central_freq"].to_numpy(), 0)
    annotated = truth.copy()
    annotated["x"] = pix[0]
    annotated["y"] = pix[1]
    annotated["z"] = pix[2]
    annotated["outside_10gb_region"] = outside_10gb_region(annotated)
    external = annotated[annotated["outside_10gb_region"]].copy()
    annotated_path = out_dir / "sky_ldev_truthcat_v2_annotated.csv"
    external_txt_path = out_dir / "sky_ldev_truthcat_v2_external_only.txt"
    summary_path = out_dir / "external_truth_summary.csv"
    annotated.to_csv(annotated_path, index=False)
    external[truth.columns].to_csv(external_txt_path, sep=" ", index=False)
    summary = pd.DataFrame(
        [
            {
                "truth_path": str(LARGE_TRUTH),
                "n_truth_total": len(annotated),
                "n_truth_inside_10gb_region": int((~annotated["outside_10gb_region"]).sum()),
                "n_truth_external": int(annotated["outside_10gb_region"].sum()),
                "x_min": float(annotated["x"].min()),
                "x_max": float(annotated["x"].max()),
                "y_min": float(annotated["y"].min()),
                "y_max": float(annotated["y"].max()),
                "annotated_path": str(annotated_path),
                "external_truth_path": str(external_txt_path),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
