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
import argparse
import glob
import os
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pyproj

from cj_measure import load_cityjson

# Base data at t = T0
T0_DIR = "data_t0"
GEOID_URL = "https://cdn.proj.org/fr_ign_RAF20.tif"
GEOID_GRID = f"{T0_DIR}/fr_ign_RAF20.tif"
CITYJSON = f"{T0_DIR}/toulouse.city.json"

# This is just default values
T1_INPUT_DIR = "/home/MCHu/Documents/projects/Toulouse_EO"
T1_INTERNAL = f"{T1_INPUT_DIR}/Ori-Compense-Figee5/AutoCal_Foc-123900_Cam-UCE.xml"
T1_EXTERNAL = f"{T1_INPUT_DIR}/Ori-Compense-Figee5"
T1_OUTPUT_DIR = "data_t1"

# These are test case paths
TEST_T1_DIR = "data_t1_test"
INTERNAL_XML = f"{TEST_T1_DIR}/internal.xml"
EXTERNAL_XML = f"{TEST_T1_DIR}/external.xml"
IMAGE = f"{TEST_T1_DIR}/oriented.tif"
FOOTPRINT_CSV = f"{TEST_T1_DIR}/footprint.csv"
CENTRE_CSV = f"{TEST_T1_DIR}/centre.csv"


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


def median_ground(path, lod="2.2"):
    """Median z of all LoD2.2 building ground surfaces: a guess of the terrain height."""
    tile = load_cityjson(path)
    return np.median([r[:, 2].min() for o in tile.objects.values()
                      for s in o.surfaces(lod, "GroundSurface") for r in s.rings])


def pixels_to_ground(A, R, C, uv, ground):
    """Intersect the rays through pixels uv (N, 2) with the plane Z = ground -> (N, 3)."""
    rays = R @ np.linalg.solve(A, np.column_stack([uv, np.ones(len(uv))]).T)   # 3 x N, world
    if C[2] <= ground or np.any(rays[2] >= 0):
        raise ValueError("ray does not hit the ground plane; check ground z / conventions")
    s = (ground - C[2]) / rays[2]                     # ray length to reach the plane
    return C + (rays * s).T


def world_to_pixels(X, C, R, f, pp):
    """Project world points X (N, 3) into the image -> (N, 2) pixels."""
    Xc = (X - C) @ R                                  # == (R.T @ (X - C).T).T, camera frame
    return pp + f * Xc[:, :2] / Xc[:, 2:3]


def image_name(external_xml):
    """MicMac 'Orientation-<image>.xml' -> '<image>'."""
    name = os.path.basename(external_xml)
    name = name.removeprefix("Orientation-")
    return name.removesuffix(".xml")


def expand_externals(paths):
    """Files are kept as is; a directory expands to its Orientation-*.xml files."""
    out = []
    for p in map(os.path.expanduser, paths):
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "Orientation-*.xml")))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f"no external orientation files found in {paths}")
    return out


def footprints(internal_xml, external_xmls, footprint_csv, centre_csv, names=None):
    """One footprint polygon and one centre point per external file, all sharing one camera."""
    f, pp, (w, h) = load_internal_param(os.path.expanduser(internal_xml))
    A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    # image corners TL, TR, BR, BL, and TL again to close the ring
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], float)
    ground = median_ground(CITYJSON)
    print(f"F = {f:.0f}px  size = {w}x{h}  ground z = {ground:.2f}  images = {len(external_xmls)}")

    names = names or [image_name(p) for p in external_xmls]
    poly_rows, centre_rows = [], []
    for name, external_xml in zip(names, external_xmls):
        C, R = load_external_param(external_xml)
        ring = pixels_to_ground(A, R, C, corners, ground)

        # sanity check: the ground corners project back onto the image corners
        err = np.abs(world_to_pixels(ring, C, R, f, pp) - corners).max()
        print(f"  {name}  C = {np.round(C, 2)}  round-trip = {err:.1e}px")

        coords = ", ".join(f"{x:.3f} {y:.3f}" for x, y in ring[:, :2])
        poly_rows.append(f'{name},"POLYGON (({coords}))"\n')
        centre_rows.append(f'{name},"POINT Z ({C[0]:.3f} {C[1]:.3f} {C[2]:.3f})"\n')

    for path, rows in ((footprint_csv, poly_rows), (centre_csv, centre_rows)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write("image,wkt\n" + "".join(rows))
    print(f"wrote {footprint_csv} and {centre_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "externals", nargs="*",
        help=f"external orientation XMLs, or directories holding Orientation-*.xml (default: {T1_EXTERNAL})",
        default=[T1_EXTERNAL]
    )
    parser.add_argument(
        "-i", "--internal", 
        help=f"internal calibration XML shared by all images (default: {T1_INTERNAL})",
        default=T1_INTERNAL
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help=f"where footprint.csv and centre.csv go (default: {T1_OUTPUT_DIR})",
        default=T1_OUTPUT_DIR
    )
    parser.add_argument(
        "--test", action="store_true", 
        help=f"run the single-image test case in {TEST_T1_DIR}/ (ignores other args)"
    )
    args = parser.parse_args()

    if args.test:
        footprints(INTERNAL_XML, [EXTERNAL_XML], FOOTPRINT_CSV, CENTRE_CSV, names=[os.path.basename(IMAGE)])
        return
    else: 
        footprints(args.internal, expand_externals(args.externals),
                os.path.join(args.output_dir, "footprint.csv"),
                os.path.join(args.output_dir, "centre.csv"))


if __name__ == "__main__":
    main()
