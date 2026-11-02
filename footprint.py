#!/usr/bin/env python3
"""
Read MicMac orientations (Ori-XXX/Orientation-*.xml) and export, for QGIS:
  - footprints.csv  approximate image footprints on a flat ground plane (POLYGON)
  - trajectory.csv  camera centres in flight order, i.e. the plane trajectory (POINT Z)

Usage (run from the folder that contains the images and the Ori- folder):
  python footprint.py ../Toulouse_overlap/Ori-Compense-Figee5 --ground-z 146.865955

Conventions (MicMac default, eConvApero_DistM2C):
  R in <CodageMatr> maps camera -> world:   X_world = R @ X_cam + C
  camera frame: x right, y down (image rows), z towards the scene
  pixel = PP + F * (Xc[0]/Xc[2], Xc[1]/Xc[2])   (lens distortion ignored here)
"""
import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np


def _nums(text):
    return [float(v) for v in text.split()]


def _text(root, tag):
    el = root.find(f".//{tag}")
    if el is None or el.text is None:
        raise ValueError(f"<{tag}> not found")
    return el.text.strip()


def load_internal_param(path):
    """Focal length (px), principal point (px), image size (w, h)."""
    root = ET.parse(path).getroot()
    f = float(_text(root, "F"))
    pp = np.array(_nums(_text(root, "PP")))
    w, h = (int(v) for v in _nums(_text(root, "SzIm")))
    return f, pp, (w, h)


def load_external_param(path):
    """Projection centre C, rotation R (camera->world), calibration file name."""
    root = ET.parse(path).getroot()
    ext = root.find(".//Externe")
    if ext is None:
        raise ValueError("<Externe> not found")
    C = np.array(_nums(_text(ext, "Centre")))
    cm = ext.find(".//CodageMatr")
    if cm is None:
        raise ValueError("rotation is not stored as <CodageMatr>")
    R = np.array([_nums(cm.find(t).text) for t in ("L1", "L2", "L3")])
    return C, R, _text(root, "FileInterne")


def world_to_pixel(X, C, R, f, pp):
    """Project a 3D world point into the image (no distortion)."""
    Xc = R.T @ (np.asarray(X, float) - C)
    if Xc[2] <= 0:
        return None  # point is behind the camera
    return pp + f * Xc[:2] / Xc[2]


def stereo_footprint_new(A: np.ndarray, R: np.ndarray, pixels, xyz: np.ndarray, ground: float):
    '''
    Calculates the full coverage of a stereo image on the ground (the footprint)
    based on the calibration matrix A, the camera->world rotation R (MicMac
    <CodageMatr>), the image size in pixels and the projection centre xyz.

    Returns a closed ring of 5 points (x, y, ground), as a 5x3 array.
    '''

    # Image corners (0-based, MicMac convention), last one closes the ring
    uv = np.array([[0, 0], [pixels[0], 0], [pixels[0], pixels[1]], [0, pixels[1]], [0, 0]], float)
    return pixels_to_ground(A, R, xyz, uv, ground)


def pixels_to_ground(A: np.ndarray, R: np.ndarray, xyz: np.ndarray, uv: np.ndarray, ground: float):
    '''
    Intersect the rays through pixels uv (N x 2) with the plane Z = ground.
    A: calibration matrix, R: camera->world rotation, xyz: projection centre.
    Returns an (N, 3) array.
    '''
    uv = np.asarray(uv, float)
    xy1 = np.vstack((uv.T, np.ones(len(uv))))

    # Height from camera center to ground
    height = xyz[2] - ground

    # Pixels to camera-space rays (3xN), then camera -> world
    rays = R @ np.linalg.solve(A, xy1)

    # Rays must point down towards the ground
    if height <= 0 or np.any(rays[2, :] >= 0):
        raise ValueError("ray does not hit the ground plane; check ground-z / conventions")

    # Extend rays to ground level (scale so that dz = -height), then translate
    rays = rays / rays[2, :] * -height
    out = rays[0:2, :] + np.asarray(xyz[:2], float).reshape(2, 1)
    return np.vstack((out, np.full((1, len(uv)), ground))).T


def world_to_pixels(X, C, R, f, pp):
    '''Project (N, 3) world points into the image (no distortion).
    Returns (N, 2) pixels and an (N,) mask of points in front of the camera.'''
    Xc = (np.asarray(X, float) - C) @ R    # == (R.T @ (X - C).T).T
    front = Xc[:, 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pp + f * Xc[:, :2] / Xc[:, 2:3]
    return uv, front


def resolve_calib(file_interne, ori_dir, cache):
    candidates = [
        file_interne,
        os.path.join(os.path.dirname(os.path.abspath(ori_dir)), file_interne),
        os.path.join(ori_dir, os.path.basename(file_interne)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            if p not in cache:
                cache[p] = load_internal_param(p)
            return cache[p]
    raise FileNotFoundError(f"calibration file not found: {file_interne}")


def micmac_footprints(ori_dir, ground_z=0.0, out_dir=None, verbose=True):
    '''
    Compute camera centres and ground footprints for every
    Orientation-*.xml in ori_dir, projected on the plane Z = ground_z.

    Returns (centres, footprints), two dicts keyed by image name:
      centres[name]    -> (3,) array, projection centre
      footprints[name] -> (5, 3) array, closed ring on the ground
    If out_dir is given, also writes footprints.csv and trajectory.csv (WKT) there.
    '''
    files = sorted(glob.glob(os.path.join(ori_dir, "Orientation-*.xml")))
    if not files:
        raise FileNotFoundError(f"no Orientation-*.xml in {ori_dir}")

    cache, centres, footprints = {}, {}, {}
    for path in files:
        name = os.path.basename(path)[len("Orientation-"):-len(".xml")]
        C, R, fi = load_external_param(path)
        f, pp, (w, h) = resolve_calib(fi, ori_dir, cache)

        A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
        ring = stereo_footprint_new(A, R, (w, h), C, ground_z)
        centres[name], footprints[name] = C, ring

        if verbose:
            # sanity checks: R orthonormal, corner round-trip
            ortho_err = np.abs(R.T @ R - np.eye(3)).max()
            corners = [(0, 0), (w, 0), (w, h), (0, h)]
            back = [world_to_pixel(g, C, R, f, pp) for g in ring[:4]]
            rt_err = max(np.abs(np.array(b) - np.array(c)).max() for b, c in zip(back, corners))
            print(f"{name}: C = ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})  "
                  f"F = {f:.1f}px  size = {w}x{h}  |RtR-I| = {ortho_err:.1e}  round-trip = {rt_err:.1e}px")

    if out_dir is not None:
        write_csv(footprints, out_dir)
        write_trajectory(centres, out_dir)
        if verbose:
            print(f"\nwrote {len(files)} images to footprints.csv and trajectory.csv in {out_dir}")

    return centres, footprints


def write_csv(footprints, out_dir):
    '''Write footprints.csv (POLYGON as WKT) for QGIS.'''
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "footprints.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "wkt"])
        for name, ring in footprints.items():
            coords = ", ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in ring)
            wr.writerow([name, f"POLYGON (({coords}))"])


def write_trajectory(centres, out_dir):
    '''Write trajectory.csv: camera centres (POINT Z as WKT) in image-name order,
    which is the flight order for sequentially numbered images.'''
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["order", "image", "x", "y", "z", "wkt"])
        for i, (name, C) in enumerate(sorted(centres.items())):
            wr.writerow([i, name, f"{C[0]:.3f}", f"{C[1]:.3f}", f"{C[2]:.3f}",
                         f"POINT Z ({C[0]:.3f} {C[1]:.3f} {C[2]:.3f})"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_dir", help="e.g. Ori-Compense-Figee5")
    parser.add_argument("--ground-z", type=float, default=0.0,
                    help="mean terrain height, same vertical reference as the orientation")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    micmac_footprints(args.ori_dir, args.ground_z, args.out_dir)


if __name__ == "__main__":
    main()