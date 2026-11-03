#!/usr/bin/env python3
"""
Render the LoD2.2 CityJSON buildings from the camera of one oriented image (MicMac) and
blend the rendering over that image, to check how well model and image line up.
Roofs red, walls cyan; only what the camera sees (hidden surfaces are occluded).
White lines mark where the rendered colour changes (surface / building outlines).

Camera conventions: see footprint.py.
"""
import os
import warnings

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # headless rendering

import mapbox_earcut as earcut
import numpy as np
import pyrender
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from cj_measure import CityJSON, load_cityjson
from footprint import BASE_DIR, EXTERNAL_XML, INTERNAL_XML, load_external_param, load_internal_param

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)   # raw image: no georef

CITYJSON = f"{BASE_DIR}/toulouse.city.json"
IMAGE = f"{BASE_DIR}/oriented.tif"
OVERLAY = f"{BASE_DIR}/overlay.jpg"
LOD = "2.2"
SCALE = 0.25                                    # render size relative to the full image
ALPHA = 0.45                                    # opacity of the rendered colours
COLOURS = {"RoofSurface": (230, 60, 40), "WallSurface": (40, 200, 230)}


def triangulate(rings):
    """Ear-clip one planar surface (exterior + holes) -> triangle indices (m, 3)."""
    verts = np.vstack(rings)
    normal = np.linalg.svd(verts - verts.mean(axis=0))[2][-1]
    xy = np.delete(verts, np.argmax(np.abs(normal)), axis=1)    # flatten on the best axis plane
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    return earcut.triangulate_float64(xy, ends).reshape(-1, 3)


def building_mesh(tile: CityJSON, origin):
    """One mesh of all roof and wall surfaces, each vertex coloured by its surface class."""
    surfs = [s for o in tile.objects.values() for s in o.surfaces(LOD) if s.semantic in COLOURS]
    verts = [np.vstack(s.rings) - origin for s in surfs]          # local coords: float32-safe
    starts = np.cumsum([0] + [len(v) for v in verts[:-1]])
    faces = np.vstack([triangulate(s.rings) + i for s, i in zip(surfs, starts)])
    colours = np.vstack([np.tile([*COLOURS[s.semantic], 255], (len(v), 1)) for s, v in zip(surfs, verts)])
    # ring orientation isn't reliable in CityJSON: draw every triangle in both windings
    faces = np.vstack([faces, faces[:, ::-1]])
    return pyrender.Mesh([pyrender.Primitive(positions=np.vstack(verts).astype(np.float32),
                                             color_0=(colours / 255).astype(np.float32),
                                             indices=faces.astype(np.uint32))])


def render(mesh, C, R, f, pp, size, origin):
    """Render the mesh from the camera at SCALE -> RGB (h, w, 3) and mask of rendered pixels."""
    w, h = round(size[0] * SCALE), round(size[1] * SCALE)
    camera = pyrender.IntrinsicsCamera(f * SCALE, f * SCALE, pp[0] * SCALE, pp[1] * SCALE,
                                       znear=10, zfar=1e5)
    # MicMac camera: x right, y down, z forward. OpenGL camera: x right, y up, z backward.
    pose = np.eye(4)
    pose[:3, :3] = R @ np.diag([1, -1, -1])
    pose[:3, 3] = C - origin
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh)
    scene.add(camera, pose=pose)
    renderer = pyrender.OffscreenRenderer(w, h)
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    renderer.delete()
    return rgb, depth > 0


if __name__ == "__main__":
    f, pp, size = load_internal_param(INTERNAL_XML)
    C, R = load_external_param(EXTERNAL_XML)
    tile = load_cityjson(CITYJSON)
    origin = tile.local_origin()

    rgb, mask = render(building_mesh(tile, origin), C, R, f, pp, size, origin)

    # the photo, downsampled to the render size
    with rasterio.open(IMAGE) as src:
        photo = src.read([1, 2, 3], out_shape=(3, *mask.shape), resampling=Resampling.average)
    out = np.moveaxis(photo, 0, -1).astype(np.float32)

    # blend the rendering over the photo, then draw outlines where the rendered colour changes
    out[mask] = (1 - ALPHA) * out[mask] + ALPHA * rgb[mask]
    rgb = rgb.astype(int)
    edge = np.zeros(mask.shape, bool)
    edge[1:] |= np.any(rgb[1:] != rgb[:-1], axis=-1)
    edge[:, 1:] |= np.any(rgb[:, 1:] != rgb[:, :-1], axis=-1)
    out[edge] = 255

    Image.fromarray(out.astype(np.uint8)).save(OVERLAY, quality=90)
    print(f"wrote {OVERLAY} ({mask.shape[1]}x{mask.shape[0]}, {mask.mean():.0%} of pixels rendered)")
