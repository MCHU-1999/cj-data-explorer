"""
Off-screen rendering of CityJSON surfaces and DSMs from an oriented camera (pyrender, EGL).

All meshes are built in local coordinates (world - origin) so float32 keeps cm precision.
Camera conventions: see camera.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # headless rendering

import mapbox_earcut as earcut
import numpy as np
import pyrender

from camera import Camera
from cj_measure import CityJSON, Raster

COLOURS = {"RoofSurface": (230, 60, 40), "WallSurface": (40, 200, 230)}
GREY = (128, 128, 128)


def triangulate(rings):
    """Ear-clip one planar surface (exterior + holes) -> triangle indices (m, 3)."""
    verts = np.vstack(rings)
    normal = np.linalg.svd(verts - verts.mean(axis=0))[2][-1]
    xy = np.delete(verts, np.argmax(np.abs(normal)), axis=1)    # flatten on the best axis plane
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    return earcut.triangulate_float64(xy, ends).reshape(-1, 3)


def _mesh(positions, faces, colours):
    """pyrender mesh; every triangle is drawn in both windings (orientation isn't reliable)."""
    faces = np.vstack([faces, faces[:, ::-1]])
    return pyrender.Mesh([pyrender.Primitive(positions=positions.astype(np.float32),
                                             color_0=(np.asarray(colours) / 255).astype(np.float32),
                                             indices=faces.astype(np.uint32))])


def building_mesh(tile: CityJSON, origin, lod="2.2", colours=COLOURS, default=None):
    """One mesh of all surfaces at `lod`, each vertex coloured by its surface class.
    Surfaces whose class is not in `colours` are dropped, unless `default` gives a colour."""
    surfs = [s for o in tile.objects.values() for s in o.surfaces(lod)
             if s.semantic in colours or default is not None]
    verts = [np.vstack(s.rings) - origin for s in surfs]
    starts = np.cumsum([0] + [len(v) for v in verts[:-1]])
    faces = np.vstack([triangulate(s.rings) + i for s, i in zip(surfs, starts)])
    cols = np.vstack([np.tile([*colours.get(s.semantic, default), 255], (len(v), 1))
                      for s, v in zip(surfs, verts)])
    return _mesh(np.vstack(verts), faces, cols)


def heightfield_mesh(dsm: Raster, origin, colour=GREY):
    """DSM -> triangle grid through the pixel centres; cells touching NaN are dropped."""
    rows, cols = dsm.shape
    t = dsm.transform
    xs = t.c + (np.arange(cols) + 0.5) * t.a
    ys = t.f + (np.arange(rows) + 0.5) * t.e
    X, Y = np.meshgrid(xs, ys)
    pos = np.column_stack([X.ravel(), Y.ravel(), np.nan_to_num(dsm.data.ravel())]) - origin
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok = np.isfinite(dsm.data)
    keep = (ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]).ravel()
    faces = np.vstack([np.column_stack([a.ravel(), c.ravel(), b.ravel()])[keep],
                       np.column_stack([b.ravel(), c.ravel(), d.ravel()])[keep]])
    return _mesh(pos, faces, np.tile([*colour, 255], (len(pos), 1)))


def _scene_and_camera(meshes, cam: Camera, origin, scale, znear, zfar):
    camera = pyrender.IntrinsicsCamera(cam.f * scale, cam.f * scale,
                                       cam.pp[0] * scale, cam.pp[1] * scale,
                                       znear=znear, zfar=zfar)
    # MicMac camera: x right, y down, z forward. OpenGL camera: x right, y up, z backward.
    pose = np.eye(4)
    pose[:3, :3] = cam.R @ np.diag([1, -1, -1])
    pose[:3, 3] = cam.C - origin
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    for m in meshes:
        scene.add(m)
    scene.add(camera, pose=pose)
    return scene


def render(meshes, cam: Camera, origin, scale=0.25, znear=10, zfar=1e5):
    """Render at `scale` of the full image -> RGB (h, w, 3) and depth (h, w), 0 = empty.
    Depth is linear, in metres along the optical axis (== Camera.project depth)."""
    w, h = round(cam.size[0] * scale), round(cam.size[1] * scale)
    scene = _scene_and_camera(meshes, cam, origin, scale, znear, zfar)
    renderer = pyrender.OffscreenRenderer(w, h)
    try:
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        renderer.delete()
    return rgb, depth


def depth_map(meshes, cam: Camera, origin, scale=0.25, znear=100, zfar=1e5):
    """Depth only, float32 (h, w), +inf where nothing is drawn.
    Pixel (r, c) of the map covers full-image pixels [c/scale, (c+1)/scale) x [r/scale, ...)."""
    w, h = round(cam.size[0] * scale), round(cam.size[1] * scale)
    scene = _scene_and_camera(meshes, cam, origin, scale, znear, zfar)
    renderer = pyrender.OffscreenRenderer(w, h)
    try:
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        renderer.delete()
    depth = depth.astype(np.float32)
    depth[depth <= 0] = np.inf
    return depth
