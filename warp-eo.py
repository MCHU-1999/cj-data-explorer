#!/usr/bin/env python3
"""
Warp raw aerial images onto their ground footprints so they can be viewed in QGIS.

ROUGH VISUALISATION ONLY. The four image corners are mapped onto the four footprint
corners with a projective transform (homography). The ground plane lines up; anything
with height (roofs, trees) is shifted by relief displacement, more so on oblique images.

Input is footprints.csv from footprint.py, whose rings are already in image-corner
order (TL, TR, BR, BL), so no corner guessing or rotation is needed.

Needs only GDAL/OGR + numpy, so it also runs from the OSGeo4W shell or QGIS's Python.

Example:
    python warp-eo.py footprints.csv images/ warped/ --res 0.5 --limit 3
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()


def homography(src, dst):
    """Solve the 3x3 projective transform mapping 4 src points onto 4 dst points."""
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def read_footprints(csv_path):
    """Read footprints.csv from footprint.py: yields (image name, 4x2 corners).

    The ring is written in image-corner order, so no guessing is needed:
    pixel (0,0) -> corner 0, (w,0) -> 1, (w,h) -> 2, (0,h) -> 3 (last point closes it).
    """
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            ring = ogr.CreateGeometryFromWkt(row["wkt"]).GetGeometryRef(0)
            pts = np.array(ring.GetPoints(), float)[:, :2]
            if len(pts) != 5:
                raise ValueError(f"{row['image']}: footprint has {len(pts)} points, expected 5")
            yield row["image"], pts[:4]


def find_image(image_dir, name):
    p = image_dir / name
    if p.exists():
        return p
    hits = sorted(image_dir.rglob(Path(name).stem + ".*"))
    hits = [h for h in hits if h.suffix.lower() in
            (".tif", ".tiff", ".jpg", ".jpeg", ".jp2", ".png", ".ecw")]
    return hits[0] if hits else None


def warp_one(img_path, corners, srs_wkt, out_path, res=None, grid=11):
    ds = gdal.Open(str(img_path))
    w, h = ds.RasterXSize, ds.RasterYSize
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)   # TL, TR, BR, BL
    H = homography(src, corners)

    # Dense GCP grid from the exact homography, then thin-plate spline warp:
    # gives a true projective mapping instead of a least-squares affine.
    u, v = np.meshgrid(np.linspace(0, w, grid), np.linspace(0, h, grid))
    px = np.column_stack([u.ravel(), v.ravel()])
    xy = apply_h(H, px)
    gcps = [gdal.GCP(float(X), float(Y), 0.0, float(c), float(r))
            for (c, r), (X, Y) in zip(px, xy)]

    vrt = gdal.Translate("", ds, format="VRT", GCPs=gcps, outputSRS=srs_wkt)
    gdal.Warp(str(out_path), vrt, format="GTiff", tps=True, dstAlpha=True,
              xRes=res, yRes=res, resampleAlg="bilinear",
              creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    out = gdal.Open(str(out_path), gdal.GA_Update)
    out.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    out = None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("footprints", help="footprints.csv written by footprint.py")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--epsg", type=int, default=2154,
                    help="CRS of the orientation / footprints (default 2154, Lambert-93)")
    parser.add_argument("--res", type=float, help="output pixel size in map units (e.g. 0.5)")
    parser.add_argument("--limit", type=int, help="only process the first N footprints (for testing)")
    parser.add_argument("--overwrite", action="store_true")
    a = parser.parse_args()

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(a.epsg)
    if srs.IsGeographic():
        raise SystemExit("Footprints need a projected CRS (e.g. EPSG:2154).")
    srs_wkt = srs.ExportToWkt()

    # print(f"footprints: {a.footprints}\nimages: {a.image_dir}\nout: {a.out_dir}\nCRS: {srs_wkt}\n")
    # exit()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, corners) in enumerate(read_footprints(a.footprints)):
        if a.limit is not None and i >= a.limit:
            break
        img = find_image(a.image_dir, name)
        if img is None:
            print(f"[skip] image not found: {name}"); continue
        out = a.out_dir / f"{img.stem}_warped.tif"
        if out.exists() and not a.overwrite:
            print(f"[skip] exists: {out.name}"); continue
        try:
            warp_one(img, corners, srs_wkt, out, a.res)
            print(f"[ok]   {out.name}")
        except Exception as e:
            print(f"[fail] {name}: {e}")


if __name__ == "__main__":
    main()
