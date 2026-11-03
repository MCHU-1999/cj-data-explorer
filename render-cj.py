#!/usr/bin/env python3
"""
Project the LoD2.2 CityJSON roof and ground surfaces through the camera of one oriented
image (MicMac) into image space, and from there onto the plane Z = ground. No visibility:
every surface is output, hidden or not. Same mapping as warp-eo.py, so the result overlays
oriented_warped.tif in QGIS.
  projected.gpkg   layers roof and ground, one MultiPolygon per building

Camera conventions: see footprint.py.
"""
import os

import numpy as np
from osgeo import ogr, osr

from cj_measure import CityJSON, load_cityjson
from footprint import (BASE_DIR, CITYJSON, EXTERNAL_XML, INTERNAL_XML, load_external_param,
                       load_internal_param, median_ground, pixels_to_ground, world_to_pixels)

ogr.UseExceptions()

PROJECTED_GPKG = f"{BASE_DIR}/projected.gpkg"
EPSG = 2154                                     # Lambert-93
LOD = "2.2"
LAYERS = {"RoofSurface": "roof", "GroundSurface": "ground"}


def surface_to_ground(surface, A, R, C, f, pp, ground):
    """One CityJSON surface -> image pixels -> OGR polygon on the plane Z = ground."""
    polygon = ogr.Geometry(ogr.wkbPolygon)
    for ring_world in surface.rings:
        ring_pixels = world_to_pixels(ring_world, C, R, f, pp)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y, _ in pixels_to_ground(A, R, C, ring_pixels, ground):
            ring.AddPoint_2D(float(x), float(y))
        ring.CloseRings()
        polygon.AddGeometry(ring)
    return polygon


def project_layer(tile: CityJSON, semantic, A, R, C, f, pp, ground):
    """All surfaces of one class -> {building id: MultiPolygon on the ground plane}.
    Geometry lives in BuildingParts, so the building id is the part's parent."""
    polygons = {}
    for o in tile.objects.values():
        building_id = o.parents[0] if o.parents else o.id
        for surface in o.surfaces(LOD, semantic):
            # some CityJSON rings are slightly invalid (self-touching, degenerate):
            # Buffer(0) repairs them, and may split one polygon into several
            polygon = surface_to_ground(surface, A, R, C, f, pp, ground).Buffer(0)
            multi = polygons.setdefault(building_id, ogr.Geometry(ogr.wkbMultiPolygon))
            for part in ogr.ForceToMultiPolygon(polygon):
                multi.AddGeometry(part)
    # merge the surfaces of each building into one feature
    return {building_id: ogr.ForceToMultiPolygon(multi.UnionCascaded())
            for building_id, multi in polygons.items()}


def write_gpkg(path, layers):
    """layers: {layer name: {building id: MultiPolygon}} -> GeoPackage, one feature per building."""
    if os.path.exists(path):
        os.remove(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    for layer_name, polygons in layers.items():
        layer = dataset.CreateLayer(layer_name, srs, ogr.wkbMultiPolygon)
        layer.CreateField(ogr.FieldDefn("building_id", ogr.OFTString))
        layer.StartTransaction()
        for building_id, geometry in polygons.items():
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("building_id", building_id)
            feature.SetGeometry(geometry)
            layer.CreateFeature(feature)
        layer.CommitTransaction()
    dataset = None


if __name__ == "__main__":
    f, pp, size = load_internal_param(INTERNAL_XML)
    C, R = load_external_param(EXTERNAL_XML)
    A = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]])
    tile = load_cityjson(CITYJSON)
    ground = median_ground(CITYJSON)

    layers = {name: project_layer(tile, semantic, A, R, C, f, pp, ground)
              for semantic, name in LAYERS.items()}
    write_gpkg(PROJECTED_GPKG, layers)
    counts = ", ".join(f"{name} {len(polygons)}" for name, polygons in layers.items())
    print(f"wrote {PROJECTED_GPKG} ({counts} buildings, ground z = {ground:.2f})")
