from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = DEMO_DIR / "outputs"

FITS_PATH = PROJECT_ROOT / "data" / "sky_dev_v2.fits"
TRUTH_PATH = PROJECT_ROOT / "data" / "sky_dev_truthcat_v2.txt"
TRUTH_PIXELS_CSV = OUTPUTS_DIR / "truth_pixels.csv"

# Region used by the current SoFiA configs: x_min, x_max, y_min, y_max, z_min, z_max.
REGION = (0, 643, 0, 643, 5000, 5800)


def require_inputs() -> None:
    if not FITS_PATH.exists():
        raise SystemExit(f"Falta FITS para leer WCS/header: {FITS_PATH}")
    if not TRUTH_PATH.exists():
        raise SystemExit(f"Falta truth catalogue: {TRUTH_PATH}")


def read_truth_catalog(path: Path) -> pd.DataFrame:
    truth_df = pd.read_csv(path, sep=r"\s+", comment="#")
    required = {"ra", "dec", "central_freq"}
    missing = required.difference(truth_df.columns)
    if missing:
        raise SystemExit(f"Truth catalogue sin columnas esperadas: {sorted(missing)}")
    return truth_df


def add_truth_pixels(truth_df: pd.DataFrame, wcs: WCS) -> pd.DataFrame:
    truth_df = truth_df.copy()
    truth_x, truth_y, truth_z = wcs.world_to_pixel_values(
        truth_df["ra"].to_numpy(),
        truth_df["dec"].to_numpy(),
        truth_df["central_freq"].to_numpy(),
    )
    truth_df["truth_x"] = truth_x
    truth_df["truth_y"] = truth_y
    truth_df["truth_z"] = truth_z
    return truth_df


def filter_region(truth_df: pd.DataFrame) -> pd.DataFrame:
    x_min, x_max, y_min, y_max, z_min, z_max = REGION
    in_region = (
        truth_df["truth_x"].between(x_min, x_max)
        & truth_df["truth_y"].between(y_min, y_max)
        & truth_df["truth_z"].between(z_min, z_max)
    )
    return truth_df[in_region].copy()


def main() -> None:
    require_inputs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    truth_df = read_truth_catalog(TRUTH_PATH)
    header = fits.getheader(FITS_PATH)
    wcs = WCS(header)
    truth_pixels = filter_region(add_truth_pixels(truth_df, wcs))

    keep_columns = [
        column for column in ["id", "ra", "dec", "central_freq", "truth_x", "truth_y", "truth_z"]
        if column in truth_pixels.columns
    ]
    truth_pixels[keep_columns].to_csv(TRUTH_PIXELS_CSV, index=False)

    print(f"Truth pixels guardado en: {TRUTH_PIXELS_CSV}")
    print(f"Fuentes truth en region: {len(truth_pixels)}")


if __name__ == "__main__":
    main()
