#!/usr/bin/env python3
"""
Warp raw oriented images onto the ground plane and georeference them, for QGIS.

ROUGH VISUALISATION ONLY. A grid of pixels is projected through the camera onto the
plane Z = ground (same as footprint.py) and used as GCPs for a thin-plate spline warp.
The ground lines up; anything with height (roofs, trees) is shifted by relief displacement.

Each image <name>.tif is matched with Orientation-<name>.tif.xml in the external
orientation directory and written to <output-dir>/<name>_warped.tif.
"""
import argparse
import glob
import os

import numpy as np
from osgeo import gdal

from footprint import (CITYJSON, EXTERNAL_XML, IMAGE, INTERNAL_XML, T1_EXTERNAL, T1_INPUT_DIR,
                       T1_INTERNAL, T1_OUTPUT_DIR, TEST_T1_DIR,
                       load_external_param, load_internal_param, median_ground, pixels_to_ground)

gdal.UseExceptions()

TEST_WARPED = f"{TEST_T1_DIR}/oriented_warped.tif"
WARPED_DIR = f"{T1_OUTPUT_DIR}/warped"
EPSG = 2154                                   # Lambert-93
RES = 0.5                                     # output pixel size (m)
GRID = 11                                     # GCPs per image side


def warp(image, internal_xml, external_xml, out, ground, res=RES):
    """Warp one image onto the plane Z = ground using a GRID x GRID set of GCPs."""
    f, pp, (w, h) = load_internal_param(os.path.expanduser(internal_xml))
    C, R = load_external_param(external_xml)
    A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])

    # GCPs: a regular pixel grid and where each pixel hits the ground plane
    u, v = np.meshgrid(np.linspace(0, w, GRID), np.linspace(0, h, GRID))
    px = np.column_stack([u.ravel(), v.ravel()])
    xyz = pixels_to_ground(A, R, C, px, ground)
    gcps = [gdal.GCP(*map(float, X), *map(float, p)) for p, X in zip(px, xyz)]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vrt = gdal.Translate("", image, format="VRT", GCPs=gcps, outputSRS=f"EPSG:{EPSG}")
    gdal.Warp(out, vrt, tps=True, dstAlpha=True, xRes=res, yRes=res, resampleAlg="bilinear",
              creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    gdal.Open(out, gdal.GA_Update).BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    print(f"  wrote {out}")


def list_images(paths):
    """Files are kept as is; a directory expands to the *.tif files directly inside it."""
    out = []
    for p in map(os.path.expanduser, paths):
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.tif")))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f"no .tif images found in {paths}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "images", nargs="*",
        help=f"raw images, or directories holding *.tif (default: {T1_INPUT_DIR})",
        default=[T1_INPUT_DIR]
    )
    parser.add_argument(
        "-i", "--internal",
        help=f"internal calibration XML shared by all images (default: {T1_INTERNAL})",
        default=T1_INTERNAL
    )
    parser.add_argument(
        "-e", "--external-dir",
        help=f"directory holding Orientation-<image>.xml (default: {T1_EXTERNAL})",
        default=T1_EXTERNAL
    )
    parser.add_argument(
        "-o", "--output-dir",
        help=f"where <image>_warped.tif go (default: {WARPED_DIR})",
        default=WARPED_DIR
    )
    parser.add_argument(
        "-r", "--res", type=float,
        help=f"output pixel size in metres (default: {RES})",
        default=RES
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"run the single-image test case in {TEST_T1_DIR}/ (ignores other args)"
    )
    args = parser.parse_args()

    ground = median_ground(CITYJSON)
    print(f"ground z = {ground:.2f}")

    if args.test:
        warp(IMAGE, INTERNAL_XML, EXTERNAL_XML, TEST_WARPED, ground)
        return
    else:
        images = list_images(args.images)
        ext_dir = os.path.expanduser(args.external_dir)
        print(f"images = {len(images)}")
        for i, image in enumerate(images, 1):
            name = os.path.basename(image)
            external_xml = os.path.join(ext_dir, f"Orientation-{name}.xml")
            print(f"[{i}/{len(images)}] {name}")
            if not os.path.exists(external_xml):
                print(f"  skipped: no {external_xml}")
                continue
            out = os.path.join(args.output_dir, f"{os.path.splitext(name)[0]}_warped.tif")
            warp(image, args.internal, external_xml, out, ground, args.res)


if __name__ == "__main__":
    main()
