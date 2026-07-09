#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


VALIDATION_ROOT = Path(__file__).resolve().parents[1]


def read_info(path):
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        hdr = hdul[0].header
        data_shape = hdul[0].data.shape

        info = {
            "path": str(path),
            "shape_numpy": tuple(int(v) for v in data_shape),
            "NAXIS1_x": int(hdr["NAXIS1"]),
            "NAXIS2_y": int(hdr["NAXIS2"]),
            "NAXIS3_z": int(hdr["NAXIS3"]),
            "CTYPE1": hdr.get("CTYPE1"),
            "CTYPE2": hdr.get("CTYPE2"),
            "CTYPE3": hdr.get("CTYPE3"),
            "CRPIX1": float(hdr.get("CRPIX1")),
            "CRPIX2": float(hdr.get("CRPIX2")),
            "CRPIX3": float(hdr.get("CRPIX3")),
            "CRVAL1": float(hdr.get("CRVAL1")),
            "CRVAL2": float(hdr.get("CRVAL2")),
            "CRVAL3": float(hdr.get("CRVAL3")),
            "CDELT1": float(hdr.get("CDELT1")),
            "CDELT2": float(hdr.get("CDELT2")),
            "CDELT3": float(hdr.get("CDELT3")),
        }

    return info


def load_wcs_header(path):
    with fits.open(path, memmap=True) as hdul:
        hdr = hdul[0].header.copy()
    return WCS(hdr), hdr


def corners_from_header(hdr):
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    nz = int(hdr["NAXIS3"])

    corners = []
    for x in [0, nx - 1]:
        for y in [0, ny - 1]:
            for z in [0, nz - 1]:
                corners.append([x, y, z])

    return np.array(corners, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-fits", required=True)
    parser.add_argument("--large-fits", required=True)
    parser.add_argument("--out-json", type=Path, default=VALIDATION_ROOT / "overlap_10gb_40gb_report.json")
    args = parser.parse_args()

    small = Path(args.small_fits)
    large = Path(args.large_fits)

    print("[INFO] Reading FITS headers with memmap=True")
    small_info = read_info(small)
    large_info = read_info(large)

    print("\n=== SMALL CUBE ===")
    print(json.dumps(small_info, indent=2))

    print("\n=== LARGE CUBE ===")
    print(json.dumps(large_info, indent=2))

    same_cdelt = {
        "CDELT1": bool(np.isclose(small_info["CDELT1"], large_info["CDELT1"])),
        "CDELT2": bool(np.isclose(small_info["CDELT2"], large_info["CDELT2"])),
        "CDELT3": bool(np.isclose(small_info["CDELT3"], large_info["CDELT3"])),
    }

    same_ctype = {
        "CTYPE1": small_info["CTYPE1"] == large_info["CTYPE1"],
        "CTYPE2": small_info["CTYPE2"] == large_info["CTYPE2"],
        "CTYPE3": small_info["CTYPE3"] == large_info["CTYPE3"],
    }

    print("\n=== WCS BASIC COMPARISON ===")
    print("same CDELT:", same_cdelt)
    print("same CTYPE:", same_ctype)

    small_wcs, small_hdr = load_wcs_header(small)
    large_wcs, large_hdr = load_wcs_header(large)

    print("\n[INFO] Mapping small cube corners into large cube pixel coordinates")

    small_pix_corners = corners_from_header(small_hdr)

    # Small pixel -> world
    world_corners = small_wcs.all_pix2world(small_pix_corners, 0)

    # World -> large pixel
    large_pix_corners = large_wcs.all_world2pix(world_corners, 0)

    mins = np.nanmin(large_pix_corners, axis=0)
    maxs = np.nanmax(large_pix_corners, axis=0)

    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs

    nx_l = large_info["NAXIS1_x"]
    ny_l = large_info["NAXIS2_y"]
    nz_l = large_info["NAXIS3_z"]

    inside = (
        x_min >= -0.5 and x_max <= nx_l - 0.5 and
        y_min >= -0.5 and y_max <= ny_l - 0.5 and
        z_min >= -0.5 and z_max <= nz_l - 0.5
    )

    bbox = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "inside_large_cube": bool(inside),
    }

    print("\n=== SMALL FOOTPRINT IN LARGE PIXELS ===")
    print(json.dumps(bbox, indent=2))

    if inside:
        print("\n[OK] El cubo pequeño parece estar contenido dentro del cubo grande según WCS.")
        print("[INFO] Para validación limpia, luego conviene excluir este bounding box.")
    else:
        print("\n[WARN] El cubo pequeño no parece estar completamente contenido según WCS.")
        print("[WARN] Revisa rutas, headers, WCS y orientación de ejes.")

    report = {
        "small_info": small_info,
        "large_info": large_info,
        "same_cdelt": same_cdelt,
        "same_ctype": same_ctype,
        "small_footprint_in_large_pixels": bbox,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[INFO] Report saved to: {out}")


if __name__ == "__main__":
    main()
    
