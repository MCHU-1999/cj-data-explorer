"""
Oriented aerial cameras (MicMac) and the project's shared data paths.

MicMac conventions (eConvApero_DistM2C):
  R maps camera -> world:  X_world = R @ X_cam + C
  camera frame: x right, y down (image rows), z towards the scene
  pixel = PP + F * (Xc[0]/Xc[2], Xc[1]/Xc[2])   (lens distortion ignored)
"""
from __future__ import annotations

import glob
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pyproj

# Base data at t = T0
T0_DIR = "data_t0"
GEOID_URL = "https://cdn.proj.org/fr_ign_RAF20.tif"
GEOID_GRID = f"{T0_DIR}/fr_ign_RAF20.tif"
CITYJSON = f"{T0_DIR}/toulouse.city.json"

# Default t = T1 inputs and outputs
T1_INPUT_DIR = "/home/MCHu/Documents/projects/Toulouse_EO"
T1_INTERNAL = f"{T1_INPUT_DIR}/Ori-Compense-Figee5/AutoCal_Foc-123900_Cam-UCE.xml"
T1_EXTERNAL = f"{T1_INPUT_DIR}/Ori-Compense-Figee5"
T1_OUTPUT_DIR = "data_t1"

# Single-image test case
TEST_T1_DIR = "data_t1_test"
INTERNAL_XML = f"{TEST_T1_DIR}/internal.xml"
EXTERNAL_XML = f"{TEST_T1_DIR}/external.xml"
IMAGE = f"{TEST_T1_DIR}/oriented.tif"
FOOTPRINT_CSV = f"{TEST_T1_DIR}/footprint.csv"
CENTRE_CSV = f"{TEST_T1_DIR}/centre.csv"

MICMAC_SIDE_FILES = ("MTD-", "Masq-", "Ortho-", "Densite_")


# --------------------------------------------------------------------------
# Orientation files
# --------------------------------------------------------------------------

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


def list_images(paths):
    """Files are kept as is; a directory expands to the *.tif files directly inside it."""
    out = []
    for p in map(os.path.expanduser, paths):
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.tif")))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f"no .tif images found in {paths}")
    return out


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

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


@dataclass
class Camera:
    name: str
    image: str | None               # path of the raw image, if known
    f: float                        # focal length (px)
    pp: np.ndarray                  # principal point (px)
    size: tuple[int, int]           # image (w, h)
    C: np.ndarray                   # projection centre, world (altitude)
    R: np.ndarray                   # rotation camera -> world

    @property
    def A(self) -> np.ndarray:
        """Intrinsic matrix."""
        return np.array([[self.f, 0, self.pp[0]], [0, self.f, self.pp[1]], [0, 0, 1]])

    def project(self, X):
        """World points X (..., 3) -> pixels (..., 2) and depth (...) along the optical axis."""
        Xc = (X - self.C) @ self.R
        depth = Xc[..., 2]
        return self.pp + self.f * Xc[..., :2] / depth[..., None], depth


def load_camera(internal_xml, external_xml, image=None, name=None) -> Camera:
    f, pp, size = load_internal_param(os.path.expanduser(internal_xml))
    C, R = load_external_param(os.path.expanduser(external_xml))
    return Camera(name or image_name(external_xml), image, f, pp, size, C, R)


def load_cameras(internal_xml, external_dir, images) -> list[Camera]:
    """Pair each image <name>.tif with <external_dir>/Orientation-<name>.tif.xml.
    Images without an orientation file are skipped (with a message)."""
    ext_dir = os.path.expanduser(external_dir)
    cams = []
    for image in images:
        name = os.path.basename(image)
        if name.startswith(MICMAC_SIDE_FILES):          # MicMac masks / metadata, not images
            continue
        external_xml = os.path.join(ext_dir, f"Orientation-{name}.xml")
        if not os.path.exists(external_xml):
            print(f"  skipped {name}: no {external_xml}")
            continue
        cams.append(load_camera(internal_xml, external_xml, image=image, name=name))
    return cams
