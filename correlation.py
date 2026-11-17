#!/usr/bin/env python3
"""
Check a LoD2.2 CityJSON model against oriented aerial images (MicMac) with a multi-image
correlation score, after Taillandier & Deriche (ISPRS 2004, sec. 4.3) and the score of
Paparoditis et al. (2000).

1. Sample     one 3D point per cell of an ortho grid (aligned with a DSM) on every roof
              facet, z from the facet plane. Walls are left out.
2. Visibility per image, a point counts if it projects inside the image, its facet faces
              the camera, and it is not hidden: its depth is compared with a depth map
              rendered from all buildings plus the DSM outside the building roofs.
3. Score      a k x k window on the facet plane around each point is projected into all
              views that see it. Each window becomes a vector v_i, normalised to zero mean
              and unit std (views with too little texture are dropped), and
                  score = Var(sum v_i) / n^2           in [0, 1]   (Paparoditis)
                  zncc  = (n * score - 1) / (n - 1)    mean pairwise ZNCC, independent of n
              Points seen in fewer than 2 textured views are undecidable (NaN).
4. Map back   score.tif, zncc.tif, nviews.tif, nvisible.tif on the ortho grid, and
              facets.csv / buildings.csv (with WKT geometry for QGIS) aggregating them.

Camera conventions: see camera.py.
"""
import argparse
import csv
import os
import time
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from camera import CITYJSON, T1_EXTERNAL, T1_INPUT_DIR, T1_INTERNAL, T1_OUTPUT_DIR, \
    list_images, load_cameras
from cj_measure import (Raster, cityjson_to_dsm, grid_like, load_cityjson, load_dsm,
                        roof_facets)
from render import building_mesh, depth_map, heightfield_mesh

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)   # raw images: no georef

DSM_T1 = f"{T1_OUTPUT_DIR}/DSM-cropped.tif"
OUT_DIR = f"{T1_OUTPUT_DIR}/correlation"
LOD = "2.2"
OCCLUDER_RES = 1.0                  # m, grid of the DSM heightfield used for occlusion
CHUNK = 100_000                     # points per sampling batch (memory ~ CHUNK * k^2 * 40 B)
ZNCC_BAD = 0.5                      # below this a point counts as "bad" in the CSVs


# --------------------------------------------------------------------------
# 1. Sampling
# --------------------------------------------------------------------------

def sample_points(tile, grid, bbox=None):
    """Roof cell centres -> xyz (P, 3), facet id (P,), flat grid index (P,), facets."""
    z, ids, facets = roof_facets(tile, grid, LOD, "RoofSurface")
    rows, cols = np.nonzero(ids >= 0)
    t = grid.transform
    xs, ys = t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e
    keep = np.ones(len(rows), bool)
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        keep = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    rows, cols, xs, ys = rows[keep], cols[keep], xs[keep], ys[keep]
    xyz = np.column_stack([xs, ys, z.data[rows, cols]]).astype(np.float64)
    return xyz, ids[rows, cols], rows * grid.shape[1] + cols, facets


def window_offsets(k, step):
    """k x k planimetric offsets (k^2, 2) centred on 0, spacing `step`."""
    d = (np.arange(k) - (k - 1) / 2) * step
    dx, dy = np.meshgrid(d, d)
    return np.column_stack([dx.ravel(), dy.ravel()])


def window_points(xyz, normals, offsets):
    """Lift the planimetric offsets onto each point's facet plane -> (P, k^2, 3)."""
    dz = -(normals[:, None, 0] * offsets[None, :, 0]
           + normals[:, None, 1] * offsets[None, :, 1]) / normals[:, None, 2]
    out = np.empty((len(xyz), len(offsets), 3))
    out[..., :2] = xyz[:, None, :2] + offsets[None]
    out[..., 2] = xyz[:, None, 2] + dz
    return out


# --------------------------------------------------------------------------
# 2. Visibility
# --------------------------------------------------------------------------

def occluders(tile, origin, grid, occluder_dsm):
    """Meshes for the depth maps: all LoD2.2 surfaces, plus (optionally) a DSM heightfield in
    which every cell covered by a roof holds the model's own z, so the DSM never hides a roof."""
    meshes = [building_mesh(tile, origin, LOD, colours={}, default=(128, 128, 128))]
    if occluder_dsm:
        coarse = grid_like_raster(grid, OCCLUDER_RES)
        dsm = load_dsm(occluder_dsm, like=coarse, resampling=Resampling.max)
        dsm = cityjson_to_dsm(tile, like=dsm, lod=LOD, semantic="RoofSurface", keep_like=True)
        meshes.append(heightfield_mesh(dsm, origin))
    return meshes


def grid_like_raster(grid: Raster, res):
    """Same origin and extent as `grid`, pixel size `res`."""
    t = grid.transform
    h, w = grid.shape
    shape = (int(np.ceil(h * abs(t.e) / res)), int(np.ceil(w * t.a / res)))
    return Raster(np.full(shape, np.nan, np.float32), Affine(res, 0, t.c, 0, -res, t.f), grid.crs)


def visible(cam, xyz, normals, depth, render_scale, depth_tol, margin_px):
    """Boolean (P,) visibility of the points in `cam`."""
    uv, d = cam.project(xyz)
    w, h = cam.size
    ok = (d > 0) & (uv[:, 0] >= margin_px) & (uv[:, 0] < w - margin_px) \
        & (uv[:, 1] >= margin_px) & (uv[:, 1] < h - margin_px)
    ok &= np.einsum("ij,ij->i", normals, cam.C - xyz) > 0           # facet faces the camera
    idx = np.nonzero(ok)[0]
    r = np.clip((uv[idx, 1] * render_scale).astype(int), 0, depth.shape[0] - 1)
    c = np.clip((uv[idx, 0] * render_scale).astype(int), 0, depth.shape[1] - 1)
    ok[idx] = d[idx] <= depth[r, c] + depth_tol
    return ok


# --------------------------------------------------------------------------
# 3. Scoring
# --------------------------------------------------------------------------

def read_grey(path, cols, rows, scale):
    """Grey image (mean of bands 1-3) of the full-resolution window [c0, c1) x [r0, r1),
    read at `scale`. Returns the array and the actual (sx, sy) scale factors."""
    c0, c1 = cols
    r0, r1 = rows
    out_h, out_w = max(round((r1 - r0) * scale), 1), max(round((c1 - c0) * scale), 1)
    with rasterio.open(path) as src:
        bands = min(src.count, 3)
        grey = np.zeros((out_h, out_w), np.float32)
        for b in range(1, bands + 1):
            grey += src.read(b, window=Window(c0, r0, c1 - c0, r1 - r0),
                             out_shape=(out_h, out_w), resampling=Resampling.average)
    return grey / bands, out_w / (c1 - c0), out_h / (r1 - r0)


def accumulate(cam, xyz, normals, sel, offsets, S, n, args):
    """Add the normalised windows of the selected points in `cam` to S and n.
    Returns the number of textured views added."""
    if not sel.size:
        return 0
    uv, d = cam.project(xyz[sel])
    # read window: bbox of the centres, padded by the window radius in pixels (generous)
    radius_m = np.abs(offsets).max() * np.sqrt(2) * 2
    pad = int(np.ceil(radius_m * cam.f / d.min())) + 2
    w, h = cam.size
    c0, r0 = np.maximum(np.floor(uv.min(axis=0)).astype(int) - pad, 0)
    c1, r1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + pad + 1, (w, h))
    grey, sx, sy = read_grey(cam.image, (c0, c1), (r0, r1), args.image_scale)
    gh, gw = grey.shape

    added = 0
    for start in range(0, sel.size, CHUNK):
        idx = sel[start:start + CHUNK]
        pts = window_points(xyz[idx], normals[idx], offsets)             # (c, k^2, 3)
        suv, _ = cam.project(pts)                                        # (c, k^2, 2)
        # full-res pixel (centre at integer coords) -> index in the downsampled read window
        col = (suv[..., 0] - c0 + 0.5) * sx - 0.5
        row = (suv[..., 1] - r0 + 0.5) * sy - 0.5
        inside = ((col >= 0) & (col <= gw - 1) & (row >= 0) & (row <= gh - 1)).all(axis=1)
        v = map_coordinates(grey, [row.ravel(), col.ravel()], order=1, mode="nearest",
                            prefilter=False).reshape(row.shape)
        mean, std = v.mean(axis=1, keepdims=True), v.std(axis=1, keepdims=True)
        good = inside & (std[:, 0] >= args.min_std)
        S[idx[good]] += ((v[good] - mean[good]) / std[good]).astype(np.float32)
        n[idx[good]] += 1
        added += int(good.sum())
    return added


def scores(S, n):
    """Paparoditis score and mean pairwise ZNCC per point; NaN where n < 2."""
    var = S.var(axis=1, dtype=np.float64)
    nf = n.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = var / nf ** 2
        zncc = (nf * score - 1) / (nf - 1)
    score[n < 2] = np.nan
    zncc[n < 2] = np.nan
    return score, zncc


# --------------------------------------------------------------------------
# 4. Output
# --------------------------------------------------------------------------

def to_raster(grid, flat, values, dtype=np.float32, nodata=np.nan):
    data = np.full(grid.shape[0] * grid.shape[1], nodata, dtype)
    data[flat] = values
    return data.reshape(grid.shape)


def save(path, grid, data):
    """Single band GeoTIFF; NaN (float) or 0 (int) = nodata."""
    nodata = np.nan if np.issubdtype(data.dtype, np.floating) else 0
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs=grid.crs, transform=grid.transform,
                       nodata=nodata, compress="DEFLATE", tiled=True) as dst:
        dst.write(data, 1)


def wkt_polygon(rings):
    parts = []
    for r in rings:
        r = np.vstack([r[:, :2], r[:1, :2]])
        parts.append("(" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in r) + ")")
    return f"POLYGON ({', '.join(parts)})"


def aggregate(keys, score, zncc):
    """Group per-point values by integer key -> {key: stats dict}."""
    order = np.argsort(keys, kind="stable")
    uniq, starts = np.unique(keys[order], return_index=True)
    out = {}
    for key, idx in zip(uniq, np.split(order, starts[1:])):
        s, z = score[idx], zncc[idx]
        dec = np.isfinite(z)
        out[int(key)] = {
            "n_cells": int(idx.size),
            "frac_decidable": round(float(dec.mean()), 3),
            "mean_score": float(np.mean(s[dec])) if dec.any() else "",
            "median_score": float(np.median(s[dec])) if dec.any() else "",
            "mean_zncc": float(np.mean(z[dec])) if dec.any() else "",
            "median_zncc": float(np.median(z[dec])) if dec.any() else "",
            f"frac_zncc_below_{ZNCC_BAD}": float(np.mean(z[dec] < ZNCC_BAD)) if dec.any() else "",
        }
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", default=[T1_INPUT_DIR],
                        help=f"raw images, or directories holding *.tif (default: {T1_INPUT_DIR})")
    parser.add_argument("-i", "--internal", default=T1_INTERNAL,
                        help=f"internal calibration XML shared by all images (default: {T1_INTERNAL})")
    parser.add_argument("-e", "--external-dir", default=T1_EXTERNAL,
                        help=f"directory holding Orientation-<image>.xml (default: {T1_EXTERNAL})")
    parser.add_argument("-c", "--cityjson", default=CITYJSON, help=f"(default: {CITYJSON})")
    parser.add_argument("--grid-like", default=DSM_T1,
                        help=f"GeoTIFF whose origin/extent/crs define the ortho grid (default: {DSM_T1})")
    parser.add_argument("--res", type=float, default=0.5, help="ortho grid cell size, m (default: 0.5)")
    parser.add_argument("--occluder-dsm", default=DSM_T1,
                        help=f"DSM adding occluders outside the roofs, or 'none' (default: {DSM_T1})")
    parser.add_argument("-k", type=int, default=7, help="window size in samples (default: 7)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="window sample spacing on the facet, m (default: 0.1)")
    parser.add_argument("--image-scale", type=float, default=0.5,
                        help="images are read at this fraction of full resolution (default: 0.5)")
    parser.add_argument("--min-std", type=float, default=3.0,
                        help="views whose window std is below this (grey levels) are dropped (default: 3)")
    parser.add_argument("--depth-tol", type=float, default=0.5,
                        help="visibility tolerance against the depth map, m (default: 0.5)")
    parser.add_argument("--render-scale", type=float, default=0.25,
                        help="depth maps are rendered at this fraction of the image size (default: 0.25)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="only score points inside this box (for quick tests)")
    parser.add_argument("-o", "--output-dir", default=OUT_DIR, help=f"(default: {OUT_DIR})")
    args = parser.parse_args()
    t_start = time.time()

    # 1. sample
    tile = load_cityjson(args.cityjson)
    origin = tile.local_origin()
    grid = grid_like(args.grid_like, args.res)
    xyz, fid, flat, facets = sample_points(tile, grid, args.bbox)
    normals = np.array([f.normal for f in facets])[fid]
    offsets = window_offsets(args.k, args.step)
    print(f"grid {grid.shape[1]}x{grid.shape[0]} @ {args.res} m, {len(facets)} roof facets, "
          f"{len(xyz)} points, window {args.k}x{args.k} @ {args.step} m")
    if not len(xyz):
        raise SystemExit("no roof points to score")

    # 2 + 3. per image: depth map, visibility, windows
    cams = load_cameras(args.internal, args.external_dir, list_images(args.images))
    print(f"building occluders ({'buildings + DSM' if args.occluder_dsm != 'none' else 'buildings'})")
    meshes = occluders(tile, origin, grid, None if args.occluder_dsm == "none" else args.occluder_dsm)

    S = np.zeros((len(xyz), args.k ** 2), np.float32)
    n = np.zeros(len(xyz), np.int16)
    nvis = np.zeros(len(xyz), np.int16)
    margin_px = 2.0
    for i, cam in enumerate(cams, 1):
        t0 = time.time()
        uv, d = cam.project(xyz)
        w, h = cam.size
        inside = (d > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if not inside.any():
            print(f"[{i}/{len(cams)}] {cam.name}: no points in view")
            continue
        depth = depth_map(meshes, cam, origin, args.render_scale)
        vis = visible(cam, xyz, normals, depth, args.render_scale, args.depth_tol, margin_px)
        sel = np.nonzero(vis)[0]
        nvis[sel] += 1
        added = accumulate(cam, xyz, normals, sel, offsets, S, n, args)
        print(f"[{i}/{len(cams)}] {cam.name}: in view {inside.sum()}, visible {sel.size}, "
              f"textured {added}  ({time.time() - t0:.0f}s)")

    score, zncc = scores(S, n)

    # 4. map back
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)               # noqa: E731
    save(out("score.tif"), grid, to_raster(grid, flat, score))
    save(out("zncc.tif"), grid, to_raster(grid, flat, zncc))
    save(out("nviews.tif"), grid, to_raster(grid, flat, n, np.int16, 0))
    save(out("nvisible.tif"), grid, to_raster(grid, flat, nvis, np.int16, 0))

    per_facet = aggregate(fid, score, zncc)
    write_csv(out("facets.csv"), [
        {"facet_id": k, "building_id": facets[k].building_id, "object_id": facets[k].object_id,
         "surface_idx": facets[k].index, **stats, "wkt": wkt_polygon(facets[k].surface.rings)}
        for k, stats in per_facet.items()])

    building_ids = sorted({f.building_id for f in facets})
    bindex = {b: j for j, b in enumerate(building_ids)}
    per_building = aggregate(np.array([bindex[f.building_id] for f in facets])[fid], score, zncc)
    rows = []
    for j, stats in per_building.items():
        bid = building_ids[j]
        fp = tile.objects[bid].footprint() if bid in tile.objects else None
        rows.append({"building_id": bid, **stats, "wkt": wkt_polygon(fp.rings) if fp else ""})
    write_csv(out("buildings.csv"), rows)

    dec = np.isfinite(zncc)
    print(f"decidable {dec.mean():.0%} of {len(xyz)} points; median score "
          f"{np.median(score[dec]) if dec.any() else float('nan'):.3f}, median zncc "
          f"{np.median(zncc[dec]) if dec.any() else float('nan'):.3f}, zncc < {ZNCC_BAD}: "
          f"{np.mean(zncc[dec] < ZNCC_BAD) if dec.any() else float('nan'):.0%}")
    print(f"wrote {args.output_dir}/ (score, zncc, nviews, nvisible .tif; facets, buildings .csv) "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
