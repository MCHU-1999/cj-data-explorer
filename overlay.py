#!/usr/bin/env python3
"""
Render the LoD2.2 CityJSON buildings from the camera of one oriented image (MicMac) and
blend the rendering over that image, to check how well model and image line up.
Roofs red, walls cyan; only what the camera sees (hidden surfaces are occluded).
White lines mark where the rendered colour changes (surface / building outlines).

Camera conventions: see camera.py.
"""
import warnings

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from camera import CITYJSON, EXTERNAL_XML, IMAGE, INTERNAL_XML, TEST_T1_DIR, load_camera
from cj_measure import load_cityjson
from render import building_mesh, render

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)   # raw image: no georef

OVERLAY = f"{TEST_T1_DIR}/overlay.jpg"
LOD = "2.2"
SCALE = 0.25                                    # render size relative to the full image
ALPHA = 0.45                                    # opacity of the rendered colours


if __name__ == "__main__":
    cam = load_camera(INTERNAL_XML, EXTERNAL_XML, image=IMAGE)
    tile = load_cityjson(CITYJSON)
    origin = tile.local_origin()

    rgb, depth = render([building_mesh(tile, origin, LOD)], cam, origin, SCALE)
    mask = depth > 0

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
