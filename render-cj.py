#!/usr/bin/env python3
"""
Render a LoD2.2 CityJSON from each camera and export what the camera sees, per building,
as three vector layers in map coordinates:
  roof    visible parts of RoofSurfaces   (occlusion handled by rendering)
  wall    visible parts of WallSurfaces   (occlusion handled by rendering)
  ground  GroundSurfaces seen through everything (no occlusion)

Each building is projected through the camera onto its own ground plane: Z = mean height
of the building's LoD2.2 GroundSurfaces. So the ground outline lands at the building's true
map position, and roofs / walls are displaced from it as the camera sees them (relief
displacement). Note: warp-eo.py uses one global ground plane, so where a building's ground
height differs from that plane the warped image and these layers drift apart slightly.
One feature per building id (MultiPolygon). Output: OUT_DIR/<image>_cj.gpkg

Example:
  python render-cj.py data/Toulouse_cj/573960_6275966.city.json \
      data/Toulouse_eo/Ori-Compense-Figee5 data/Toulouse_eo/cj --limit 1
"""
import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # headless rendering

import mapbox_earcut as earcut
import numpy as np
import pyrender
from osgeo import ogr, osr
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from footprint import (load_external_param, pixels_to_ground, resolve_calib,  # noqa: E402
                       world_to_pixels)

# cj-measure.py has a hyphen in its name, so load it by path
_spec = importlib.util.spec_from_file_location("cj_measure", HERE / "cj-measure.py")
cj_measure = importlib.util.module_from_spec(_spec)
sys.modules["cj_measure"] = cj_measure   # dataclasses look the module up while it loads
_spec.loader.exec_module(cj_measure)

ogr.UseExceptions()

LOD = "2.2"
CLASSES = {"RoofSurface": "roof", "WallSurface": "wall"}   # rendered (with occlusion)
CODE_STRIDE = 997   # odd -> label <-> colour is a bijection mod 2^24; spreads codes so that
                    # antialiased edge colours rarely decode to a valid label


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------

def building_id(obj):
    """Geometry lives in BuildingParts; group by their parent Building."""
    return obj.parents[0] if obj.parents else obj.id


def building_ground_z(tile):
    """Ground height per building: mean z of the vertices of its LoD2.2 GroundSurfaces
    (all parts together). Buildings without a GroundSurface use their lowest vertex."""
    ground, lowest = {}, {}
    for obj in tile.objects.values():
        bid = building_id(obj)
        for surf in obj.surfaces(LOD):
            z = np.concatenate([r[:, 2] for r in surf.rings])
            lowest[bid] = min(lowest.get(bid, np.inf), z.min())
            if surf.semantic == "GroundSurface":
                ground.setdefault(bid, []).append(z)
    return {bid: float(np.concatenate(ground[bid]).mean()) if bid in ground else float(z0)
            for bid, z0 in lowest.items()}


def triangulate(rings):
    """Ear-clip one planar surface (exterior + holes). Returns (vertices (n,3), faces (m,3))."""
    verts = np.vstack(rings)
    normal = np.linalg.svd(verts - verts.mean(axis=0))[2][-1]
    drop = int(np.argmax(np.abs(normal)))            # project onto the best-fitting axis plane
    xy = np.delete(verts, drop, axis=1)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    faces = earcut.triangulate_float64(xy, ends).reshape(-1, 3)
    return verts, faces


def build_scene(tile, origin):
    """One mesh for all roof/wall surfaces. Each (building, class) gets its own flat colour.
    Returns (pyrender.Scene, labels) with labels[k] = (building_id, layer) for code k."""
    positions, colours, faces, labels, index = [], [], [], {}, {}
    n = 0
    for obj in tile.objects.values():
        bid = building_id(obj)
        for surf in obj.surfaces(LOD):
            layer = CLASSES.get(surf.semantic)
            if layer is None:
                continue
            key = (bid, layer)
            if key not in index:
                k = len(index) + 1
                code = (k * CODE_STRIDE) % (1 << 24)
                index[key], labels[code] = code, key
            code = index[key]
            try:
                v, f = triangulate(surf.rings)
            except Exception:
                continue
            if len(f) == 0:
                continue
            rgb = np.array([code & 255, (code >> 8) & 255, (code >> 16) & 255], np.float32) / 255
            positions.append(v - origin)
            colours.append(np.tile(np.append(rgb, 1.0), (len(v), 1)))
            faces.append(f + n)
            n += len(v)

    # No custom material: with one, pyrender no longer outputs the vertex colours exactly.
    # Ring orientation isn't reliable in CityJSON, so draw every triangle in both windings
    # instead of relying on double-sided materials.
    faces = np.vstack(faces)
    faces = np.vstack([faces, faces[:, ::-1]])
    prim = pyrender.Primitive(positions=np.vstack(positions).astype(np.float32),
                              color_0=np.vstack(colours).astype(np.float32),
                              indices=faces.astype(np.uint32),
                              mode=pyrender.constants.GLTF.TRIANGLES)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(pyrender.Mesh([prim]))
    return scene, labels


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_labels(scene, labels, C, R, f, pp, size, scale, origin, scene_verts):
    """Render the label image (rows, cols) int32 at `scale` of the full image; 0 = nothing."""
    w, h = round(size[0] * scale), round(size[1] * scale)
    dist = np.linalg.norm(scene_verts - (C - origin), axis=1)
    cam = pyrender.IntrinsicsCamera(fx=f * scale, fy=f * scale, cx=pp[0] * scale, cy=pp[1] * scale,
                                    znear=max(dist.min() * 0.9, 0.1), zfar=dist.max() * 1.1)
    # MicMac camera: x right, y down, z forward. OpenGL camera: x right, y up, z backward.
    pose = np.eye(4)
    pose[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = C - origin
    node = scene.add(cam, pose=pose)
    try:
        r = pyrender.OffscreenRenderer(w, h)
        rgb, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
        r.delete()
    finally:
        scene.remove_node(node)

    rgb = rgb.astype(np.int64)
    codes = rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)
    # Antialiased edges blend colours into codes that don't exist: take the nearest valid pixel
    valid = (codes == 0) | np.isin(codes, np.fromiter(labels, np.int64))
    if not valid.all():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        codes = codes[tuple(idx)]
    return codes.astype(np.int32)


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------

def ring_to_ogr(points):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in points:
        ring.AddPoint_2D(float(x), float(y))
    ring.CloseRings()
    return ring


def polygon_to_ground(rings_px, A, R, C, ground_z):
    """List of pixel rings -> OGR polygon on the ground plane."""
    poly = ogr.Geometry(ogr.wkbPolygon)
    for ring in rings_px:
        g = pixels_to_ground(A, R, C, np.asarray(ring, float), ground_z)
        poly.AddGeometry(ring_to_ogr(g[:, :2]))
    return poly


def merge(polys, clip=None, simplify=0.0):
    """Union polygons into one MultiPolygon (optionally clipped and simplified)."""
    mp = ogr.Geometry(ogr.wkbMultiPolygon)
    for p in polys:
        mp.AddGeometry(p)
    g = mp.UnionCascaded()
    if simplify > 0:
        g = g.SimplifyPreserveTopology(simplify)
    if clip is not None:
        g = g.Intersection(clip)
    if g is None or g.IsEmpty():
        return None
    return ogr.ForceToMultiPolygon(g)


def visible_layers(codes, labels, scale, A, R, C, f, ground_z):
    """Label image -> {layer: {building_id: MultiPolygon}}, each building projected onto
    its own ground plane Z = ground_z[building_id]."""
    polys = {}
    for geom, code in shapes(codes, mask=codes > 0, connectivity=4,
                             transform=Affine.scale(1 / scale)):   # -> full-res pixel coords
        key = labels.get(int(code))
        if key is not None:
            polys.setdefault(key, []).append(
                polygon_to_ground(geom["coordinates"], A, R, C, ground_z[key[0]]))
    out = {layer: {} for layer in CLASSES.values()}
    for (bid, layer), ps in polys.items():
        # one render pixel on this building's ground plane: smooths the pixel staircase
        simplify = (C[2] - ground_z[bid]) / (f * scale)
        g = merge(ps, simplify=simplify)
        if g is not None:
            out[layer][bid] = g
    return out


def ground_layer(tile, C, R, f, pp, A, size, ground_z):
    """GroundSurfaces as seen through everything (no occlusion), clipped to the image
    frame in pixel space, then projected onto the building's ground plane."""
    frame = ogr.Geometry(ogr.wkbPolygon)
    frame.AddGeometry(ring_to_ogr([(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]))
    polys = {}
    for obj in tile.objects.values():
        bid = building_id(obj)
        for surf in obj.surfaces(LOD, "GroundSurface"):
            poly_px = ogr.Geometry(ogr.wkbPolygon)
            for ring in surf.rings:
                uv, front = world_to_pixels(ring, C, R, f, pp)
                if not front.all():
                    break
                poly_px.AddGeometry(ring_to_ogr(uv))
            else:
                inside = poly_px.Intersection(frame)
                if inside is None or inside.IsEmpty():
                    continue
                for part in ogr.ForceToMultiPolygon(inside):
                    rings = [np.array(part.GetGeometryRef(i).GetPoints())[:, :2]
                             for i in range(part.GetGeometryCount())]
                    polys.setdefault(bid, []).append(
                        polygon_to_ground(rings, A, R, C, ground_z[bid]))
    out = {}
    for bid, ps in polys.items():
        g = merge(ps)
        if g is not None:
            out[bid] = g
    return out


def write_gpkg(path, layers, image, srs):
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    for name in ("ground", "wall", "roof"):
        lyr = ds.CreateLayer(name, srs, ogr.wkbMultiPolygon)
        lyr.CreateField(ogr.FieldDefn("building_id", ogr.OFTString))
        lyr.CreateField(ogr.FieldDefn("image", ogr.OFTString))
        lyr.StartTransaction()
        for bid, g in layers[name].items():
            feat = ogr.Feature(lyr.GetLayerDefn())
            feat.SetField("building_id", bid)
            feat.SetField("image", image)
            feat.SetGeometry(g)
            lyr.CreateFeature(feat)
        lyr.CommitTransaction()
    ds = None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cityjson")
    ap.add_argument("ori_dir", help="e.g. Ori-Compense-Figee5")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scale", type=float, default=0.25, help="render scale of the full image (default 0.25)")
    ap.add_argument("--images", nargs="+", help="only these image names (e.g. 20FE31100x00028_02285.tif)")
    ap.add_argument("--limit", type=int, help="only process the first N images")
    a = ap.parse_args()

    tile = cj_measure.load_cityjson(a.cityjson)
    srs = osr.SpatialReference()
    srs.SetFromUserInput(cj_measure._epsg(tile.crs) or "EPSG:2154")
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    origin = tile.local_origin()
    scene, labels = build_scene(tile, origin)
    scene_verts = next(iter(scene.meshes)).primitives[0].positions
    ground_z = building_ground_z(tile)
    print(f"scene: {len(scene_verts)} vertices, {len(labels)} (building, class) labels, "
          f"ground z {min(ground_z.values()):.2f} .. {max(ground_z.values()):.2f}")

    files = sorted(glob.glob(os.path.join(a.ori_dir, "Orientation-*.xml")))
    names = [os.path.basename(p)[len("Orientation-"):-len(".xml")] for p in files]
    jobs = [(n, p) for n, p in zip(names, files) if not a.images or n in a.images]
    if a.limit is not None:
        jobs = jobs[:a.limit]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for name, path in jobs:
        C, R, fi = load_external_param(path)
        f, pp, size = resolve_calib(fi, a.ori_dir, cache)
        A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])

        codes = render_labels(scene, labels, C, R, f, pp, size, a.scale, origin, scene_verts)
        layers = visible_layers(codes, labels, a.scale, A, R, C, f, ground_z)
        layers["ground"] = ground_layer(tile, C, R, f, pp, A, size, ground_z)

        out = a.out_dir / f"{Path(name).stem}_cj.gpkg"
        write_gpkg(str(out), layers, name, srs)
        print(f"[ok] {out.name}: roof {len(layers['roof'])}, wall {len(layers['wall'])}, "
              f"ground {len(layers['ground'])} buildings")


if __name__ == "__main__":
    main()
