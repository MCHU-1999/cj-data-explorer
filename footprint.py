#!/usr/bin/env python3
"""
Footprint of one oriented aerial image (MicMac) on a flat ground plane, for QGIS:
  footprint.csv  image footprint (POLYGON, WKT), corners in image order TL, TR, BR, BL
  centre.csv     camera projection centre (POINT Z, WKT)

Camera conventions: see camera.py.
"""
import argparse
import os

import numpy as np

from camera import (CENTRE_CSV, CITYJSON, EXTERNAL_XML, FOOTPRINT_CSV, IMAGE, INTERNAL_XML,
                    T1_EXTERNAL, T1_INTERNAL, T1_OUTPUT_DIR, TEST_T1_DIR, expand_externals,
                    image_name, load_external_param, load_internal_param, pixels_to_ground,
                    world_to_pixels)
from cj_measure import load_cityjson


def median_ground(path, lod="2.2"):
    """Median z of all LoD2.2 building ground surfaces: a guess of the terrain height."""
    tile = load_cityjson(path)
    return np.median([r[:, 2].min() for o in tile.objects.values()
                      for s in o.surfaces(lod, "GroundSurface") for r in s.rings])


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
