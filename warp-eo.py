#!/usr/bin/env python3
"""
Warp the raw oriented image onto the ground plane and georeference it, for QGIS.

ROUGH VISUALISATION ONLY. A grid of pixels is projected through the camera onto the
plane Z = ground (same as footprint.py) and used as GCPs for a thin-plate spline warp.
The ground lines up; anything with height (roofs, trees) is shifted by relief displacement.
"""
import numpy as np
from osgeo import gdal

from footprint import (BASE_DIR, CITYJSON, EXTERNAL_XML, IMAGE_NAME, INTERNAL_XML,
                       load_external_param, load_internal_param, median_ground, pixels_to_ground)

gdal.UseExceptions()

IMAGE = f"{BASE_DIR}/{IMAGE_NAME}"
WARPED = f"{BASE_DIR}/oriented_warped.tif"
EPSG = 2154                                   # Lambert-93
RES = 0.1                                     # output pixel size (m)
GRID = 11                                     # GCPs per image side


if __name__ == "__main__":
    f, pp, (w, h) = load_internal_param(INTERNAL_XML)
    C, R = load_external_param(EXTERNAL_XML)
    A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    ground = median_ground(CITYJSON)

    # GCPs: a regular pixel grid and where each pixel hits the ground plane
    u, v = np.meshgrid(np.linspace(0, w, GRID), np.linspace(0, h, GRID))
    px = np.column_stack([u.ravel(), v.ravel()])
    xyz = pixels_to_ground(A, R, C, px, ground)
    gcps = [gdal.GCP(*map(float, X), *map(float, p)) for p, X in zip(px, xyz)]

    vrt = gdal.Translate("", IMAGE, format="VRT", GCPs=gcps, outputSRS=f"EPSG:{EPSG}")
    gdal.Warp(WARPED, vrt, tps=True, dstAlpha=True, xRes=RES, yRes=RES, resampleAlg="bilinear",
              creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    gdal.Open(WARPED, gdal.GA_Update).BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    print(f"wrote {WARPED}  (ground z = {ground:.2f})")
