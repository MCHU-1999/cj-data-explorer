#!/usr/bin/env python3
"""
Footprint of one oriented aerial image (MicMac) on a flat ground plane, for QGIS:
  footprint.csv  image footprint (POLYGON, WKT), corners in image order TL, TR, BR, BL
  centre.csv     camera projection centre (POINT Z, WKT)

MicMac conventions (eConvApero_DistM2C):
  R maps camera -> world:  X_world = R @ X_cam + C
  camera frame: x right, y down (image rows), z towards the scene
  pixel = PP + F * (Xc[0]/Xc[2], Xc[1]/Xc[2])   (lens distortion ignored)
"""
import os
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pyproj

BASE_DIR = "data_subset"
INTERNAL_XML = f"{BASE_DIR}/internal.xml"
EXTERNAL_XML = f"{BASE_DIR}/external.xml"
IMAGE_NAME = "oriented.tif"
GROUND_Z = 132.65933
FOOTPRINT_CSV = f"{BASE_DIR}/footprint.csv"
CENTRE_CSV = f"{BASE_DIR}/centre.csv"
GEOID_GRID = f"{BASE_DIR}/fr_ign_RAF20.tif"                  # IGN geoid model, downloaded once
GEOID_URL = "https://cdn.proj.org/fr_ign_RAF20.tif"


def numbers(root, tag):
    """Whitespace-separated numbers of the first <tag> in the XML, as an array."""
    return np.array(root.findtext(f".//{tag}").split(), float)


def load_internal_param(path):
    """Focal length F (px), principal point PP (px), image size (w, h)."""
    root = ET.parse(path).getroot()
    w, h = numbers(root, "SzIm").astype(int)
    return numbers(root, "F")[0], numbers(root, "PP"), (w, h)


def load_external_param(path):
    """Projection centre C and rotation R (camera -> world)."""
    root = ET.parse(path).getroot()
    R = np.array([numbers(root, "L1"), numbers(root, "L2"), numbers(root, "L3")])
    C = numbers(root, "Centre")
    C[2] -= geoid_height(C[0], C[1])                  # ellipsoidal height -> altitude
    return C, R


def geoid_height(x, y):
    """Geoid height N (m) above the GRS80 ellipsoid at Lambert-93 x, y (RAF20 grid)."""
    if not os.path.exists(GEOID_GRID):                # first run only, then read locally
        urllib.request.urlretrieve(GEOID_URL, GEOID_GRID)
    geoid = pyproj.Transformer.from_pipeline(
        "+proj=pipeline "
        "+step +inv +proj=lcc +lat_0=46.5 +lon_0=3 +lat_1=49 +lat_2=44 "
        "+x_0=700000 +y_0=6600000 +ellps=GRS80 "      # Lambert-93 -> geographic
        f"+step +proj=vgridshift +grids={os.path.abspath(GEOID_GRID)} +multiplier=1")
    return geoid.transform(x, y, 0.0)[2]


def pixels_to_ground(A, R, C, uv, ground):
    """Intersect the rays through pixels uv (N, 2) with the plane Z = ground -> (N, 3)."""
    rays = R @ np.linalg.solve(A, np.column_stack([uv, np.ones(len(uv))]).T)   # 3 x N, world
    if C[2] <= ground or np.any(rays[2] >= 0):
        raise ValueError("ray does not hit the ground plane; check GROUND_Z / conventions")
    s = (ground - C[2]) / rays[2]                     # ray length to reach the plane
    return C + (rays * s).T


def world_to_pixels(X, C, R, f, pp):
    """Project world points X (N, 3) into the image -> (N, 2) pixels."""
    Xc = (X - C) @ R                                  # == (R.T @ (X - C).T).T, camera frame
    return pp + f * Xc[:, :2] / Xc[:, 2:3]


if __name__ == "__main__":
    f, pp, (w, h) = load_internal_param(INTERNAL_XML)
    C, R = load_external_param(EXTERNAL_XML)
    A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])

    # image corners TL, TR, BR, BL, and TL again to close the ring
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], float)
    ring = pixels_to_ground(A, R, C, corners, GROUND_Z)

    # sanity check: the ground corners project back onto the image corners
    err = np.abs(world_to_pixels(ring, C, R, f, pp) - corners).max()
    print(f"C = {np.round(C, 2)}  F = {f:.0f}px  size = {w}x{h}  round-trip = {err:.1e}px")

    coords = ", ".join(f"{x:.3f} {y:.3f}" for x, y in ring[:, :2])
    with open(FOOTPRINT_CSV, "w") as fh:
        fh.write(f'image,wkt\n{IMAGE_NAME},"POLYGON (({coords}))"\n')
    with open(CENTRE_CSV, "w") as fh:
        fh.write(f'image,wkt\n{IMAGE_NAME},"POINT Z ({C[0]:.3f} {C[1]:.3f} {C[2]:.3f})"\n')
    print(f"wrote {FOOTPRINT_CSV} and {CENTRE_CSV}")
