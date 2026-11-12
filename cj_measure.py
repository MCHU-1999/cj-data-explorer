"""
Comments
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, reproject


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Surface:
    ring_indices: list[list[int]]   # vertex indices; first ring = exterior, rest = holes
    rings: list[np.ndarray]         # same rings as (n, 3) arrays in real-world coords
    semantic: str | None            # "RoofSurface", "WallSurface", ... or None
    attributes: dict = field(default_factory=dict)  # other semantic-surface attributes

    def __str__(self) -> str:
        return "Surface: " + str(self.rings)


@dataclass
class CityObject:
    id: str
    type: str                       # "Building", "BuildingPart", ...
    attributes: dict
    parents: list[str]
    children: list[str]
    geometries: dict[str, list[Surface]]  # LoD (as string, e.g. "2.2") -> surfaces

    def surfaces(self, lod: str, semantic: str | None = None) -> list[Surface]:
        """All surfaces at one LoD, optionally filtered by semantic type."""
        return [s for s in self.geometries.get(lod, [])
                if semantic is None or s.semantic == semantic]

    def footprint(self) -> Surface | None:
        """The LoD0 surface == footprint"""
        if not self.parents and 'Part' not in self.type:
            surfs = self.geometries.get("0", [])
            return surfs[0] if surfs else None
        else:
            return None


@dataclass
class CityJSON:
    path: Path
    objects: dict[str, CityObject]
    vertices: np.ndarray            # (n, 3) float64, real-world coords (transform applied)
    crs: str | None                 # as given in metadata.referenceSystem

    def buildings(self) -> list[CityObject]:
        return [o for o in self.objects.values() if o.type == "Building"]

    def parts(self, building_id: str) -> list[CityObject]:
        """Children of a building. In the 3D BAG the geometry lives in BuildingParts."""
        return [self.objects[c] for c in self.objects[building_id].children]

    def local_origin(self) -> np.ndarray:
        """Offset to subtract before rendering (avoids float32 precision loss)."""
        return self.vertices.min(axis=0)


# --------------------------------------------------------------------------
# CityJSON reading
# --------------------------------------------------------------------------

def _get(values, *idx):
    """Index into nested semantics 'values', returning None where entries are null."""
    for i in idx:
        if values is None:
            return None
        values = values[i]
    return values


def _flatten(geom: dict) -> list[tuple[list[list[int]], int | None]]:
    """Return (rings, semantic_index) for every surface of a geometry, at any depth."""
    t, b = geom["type"], geom["boundaries"]
    v = geom.get("semantics", {}).get("values")
    out = []
    if t in ("MultiSurface", "CompositeSurface"):
        for i, srf in enumerate(b):
            out.append((srf, _get(v, i)))
    elif t == "Solid":
        for s, shell in enumerate(b):
            for i, srf in enumerate(shell):
                out.append((srf, _get(v, s, i)))
    elif t in ("MultiSolid", "CompositeSolid"):
        for d, solid in enumerate(b):
            for s, shell in enumerate(solid):
                for i, srf in enumerate(shell):
                    out.append((srf, _get(v, d, s, i)))
    # Points, lines and GeometryInstance (templates) are skipped for now.
    return out


def load_cityjson(path: str | Path) -> CityJSON:
    path = Path(path)
    cj = json.loads(path.read_text())

    tr = cj.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
    verts = (np.asarray(cj["vertices"], dtype=np.float64)
             * np.asarray(tr["scale"]) + np.asarray(tr["translate"]))

    objects = {}
    for oid, co in cj["CityObjects"].items():
        geoms: dict[str, list[Surface]] = {}
        for g in co.get("geometry", []):
            sem_surfaces = g.get("semantics", {}).get("surfaces", [])
            surfs = geoms.setdefault(str(g.get("lod")), [])
            for rings, si in _flatten(g):
                sem = sem_surfaces[si] if si is not None else {}
                surfs.append(Surface(
                    ring_indices=rings,
                    rings=[verts[r] for r in rings],
                    semantic=sem.get("type"),
                    attributes={k: val for k, val in sem.items()
                                if k not in ("type", "parent", "children")},
                ))
        objects[oid] = CityObject(
            id=oid,
            type=co["type"],
            attributes=co.get("attributes", {}),
            parents=co.get("parents", []),
            children=co.get("children", []),
            geometries=geoms,
        )

    crs = cj.get("metadata", {}).get("referenceSystem")
    return CityJSON(path=path, objects=objects, vertices=verts, crs=crs)


# --------------------------------------------------------------------------
# Rasters (DSM)
# --------------------------------------------------------------------------

@dataclass
class Raster:
    data: np.ndarray                # (rows, cols) float32, NaN = nodata
    transform: Affine               # north-up pixel grid
    crs: CRS | str | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def save(self, path: str | Path) -> None:
        with rasterio.open(path, "w", driver="GTiff", height=self.shape[0], width=self.shape[1],
                           count=1, dtype="float32", crs=self.crs, transform=self.transform,
                           nodata=np.nan, compress="DEFLATE", predictor=3) as dst:
            dst.write(self.data.astype(np.float32), 1)


def load_dsm(path: str | Path, like: Raster | None = None,
             resampling: Resampling = Resampling.average) -> Raster:
    """Read band 1 of a GeoTIFF; nodata becomes NaN.

    If `like` is given, the raster is warped on the fly onto `like`'s grid (extent,
    resolution, crs), so only the target window is ever held in memory. For going
    from fine to coarse use `average` (or `max` to keep roof edges, `med` against noise);
    nodata pixels are ignored by the aggregation.
    """
    with rasterio.open(path) as src:
        if like is None:
            data = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            return Raster(data=data, transform=src.transform, crs=src.crs)
        # If `like` has no crs, or one equivalent to the source's, keep the source crs.
        # This avoids a needless (and sometimes failing) datum lookup between two
        # WKT spellings of the same CRS.
        dst_crs = src.crs if like.crs is None or CRS.from_user_input(like.crs) == src.crs \
            else CRS.from_user_input(like.crs)
        with WarpedVRT(src, crs=dst_crs, transform=like.transform,
                       width=like.shape[1], height=like.shape[0],
                       resampling=resampling, src_nodata=src.nodata, nodata=np.nan) as vrt:
            data = vrt.read(1, masked=True).astype(np.float32).filled(np.nan)
        return Raster(data=data, transform=like.transform, crs=dst_crs)


def geoid_undulation(geoid_path: str | Path, like: Raster) -> Raster:
    """Geoid grid (e.g. IGN RAF20, N in metres) bilinearly sampled on `like`'s grid."""
    n = np.full(like.shape, np.nan, np.float32)
    with rasterio.open(geoid_path) as src:
        reproject(rasterio.band(src, 1), n, dst_transform=like.transform,
                  dst_crs=like.crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
    return Raster(data=n, transform=like.transform, crs=like.crs)


def ellipsoidal_to_altitude(dsm: Raster, geoid_path: str | Path) -> Raster:
    """Ellipsoidal heights h -> altitudes H = h - N (RAF20: NGF-IGN69, EPSG:5720)."""
    n = geoid_undulation(geoid_path, dsm)
    if np.isnan(n.data).any():
        raise ValueError("Geoid grid does not cover the whole DSM")
    return Raster(data=dsm.data - n.data, transform=dsm.transform, crs=dsm.crs)


def _epsg(crs: str | None) -> str | None:
    """'https://www.opengis.net/def/crs/EPSG/0/7415' -> 'EPSG:7415'."""
    m = re.search(r"EPSG/\d+/(\d+)", crs or "")
    return f"EPSG:{m.group(1)}" if m else crs


def _empty_grid(tile: CityJSON, res: float) -> Raster:
    """Grid covering all vertices, snapped outward to multiples of res."""
    xmin, ymin = np.floor(tile.vertices[:, :2].min(axis=0) / res) * res
    xmax, ymax = np.ceil(tile.vertices[:, :2].max(axis=0) / res) * res
    shape = (round((ymax - ymin) / res), round((xmax - xmin) / res))
    return Raster(data=np.full(shape, np.nan, np.float32),
                  transform=Affine(res, 0, xmin, 0, -res, ymax), crs=_epsg(tile.crs))


def cityjson_to_dsm(tile: CityJSON, like: Raster | None = None, res: float = 0.5,
                    lod: str = "2.2", semantic: str = "RoofSurface",
                    keep_like: bool = False) -> Raster:
    """Burn the roof surfaces of a CityJSON into a DSM.

    Each roof surface is planar, so its height at a pixel centre is taken from the
    plane fitted through its exterior ring. Where surfaces overlap the highest wins.
    If `like` is given, its grid (transform, shape, crs) is used so the result can be
    diffed pixel by pixel; otherwise a grid of size `res` covering the tile is made.
    If `keep_like` is True (requires `like`), pixels not covered by any surface keep
    `like`'s values; covered pixels are always overwritten by the CityJSON, even
    where `like` is higher. Otherwise they are NaN.
    """
    if keep_like and like is None:
        raise ValueError("keep_like=True requires `like`")
    if like is not None:
        out = Raster(np.full(like.shape, np.nan, np.float32), like.transform, like.crs)
    else:
        out = _empty_grid(tile, res)
    t, (nrows, ncols) = out.transform, out.shape
    inv = ~t

    for obj in tile.objects.values():
        for surface in obj.surfaces(lod, semantic):
            exterior = surface.rings[0]
            centroid = exterior.mean(axis=0)    # mean vertex: a point on the fitted plane
            normal = np.linalg.svd(exterior - centroid)[2][-1]
            if abs(normal[2]) < 1e-6:           # vertical: invisible from above
                continue

            # Pixel window covering the surface's bbox, clipped to the grid.
            # North-up grid, so (xmin, ymax) is the top-left pixel and (xmax, ymin) the bottom-right.
            (xmin, ymin), (xmax, ymax) = exterior[:, :2].min(axis=0), exterior[:, :2].max(axis=0)
            col0, row0 = inv * (xmin, ymax)
            col1, row1 = inv * (xmax, ymin)
            col0, row0 = max(int(np.floor(col0)), 0), max(int(np.floor(row0)), 0)
            col1, row1 = min(int(np.ceil(col1)), ncols), min(int(np.ceil(row1)), nrows)
            if col0 >= col1 or row0 >= row1:
                continue

            # 2D polygon (holes included); GeoJSON rings must be closed, CityJSON ones aren't
            polygon = {"type": "Polygon",
                       "coordinates": [np.vstack([ring[:, :2], ring[:1, :2]]).tolist()
                                       for ring in surface.rings]}
            # Transform of the window: its pixel (0, 0) is pixel (col0, row0) of the full grid
            window_transform = t * Affine.translation(col0, row0)
            mask = np.asarray(rasterize([(polygon, 1)], out_shape=(row1 - row0, col1 - col0),
                                        transform=window_transform, dtype="uint8"),
                              dtype=bool)
            if not mask.any():
                continue

            # Plane height at pixel centres.
            # Affine transform t maps (col, row) -> (x, y) as
            #   x = t.a * col + t.b * row + t.c
            #   y = t.d * col + t.e * row + t.f
            # For a north-up grid (no rotation, t.b = t.d = 0):
            #   t.c, t.f = world x, y of the grid's top-left corner
            #   t.a      = pixel width  (> 0, x grows with col)
            #   t.e      = pixel height (< 0, y shrinks as row grows)
            # +0.5 moves from the pixel's corner to its centre.
            xs = t.c + (np.arange(col0, col1) + 0.5) * t.a
            ys = t.f + (np.arange(row0, row1) + 0.5) * t.e
            # Plane through `centroid` with `normal`: n · (p - centroid) = 0, solved for z
            dx, dy = xs[None, :] - centroid[0], ys[:, None] - centroid[1]
            z = centroid[2] - (normal[0] * dx + normal[1] * dy) / normal[2]

            # `window` is a view into out.data; fmax skips NaN, so the highest surface wins
            window = out.data[row0:row1, col0:col1]
            window[mask] = np.fmax(window[mask], z[mask])

    if keep_like and like is not None:
        # Filled after burning (not used as the starting grid) so that fmax above
        # compares roofs only with each other, never with `like`'s values.
        uncovered = np.isnan(out.data)
        out.data[uncovered] = like.data[uncovered]

    return out


def diff_dsm(a: Raster, b: Raster, fill: float | None = 0.0) -> Raster:
    """a - b on a shared grid.

    Nodata (NaN) in either input is replaced by `fill` for the subtraction only; the
    inputs are left untouched. Pixels with no data on both sides stay NaN. Pass
    `fill=None` to keep NaN wherever either side has no data.
    """
    if a.shape != b.shape or not a.transform.almost_equals(b.transform):
        raise ValueError("Rasters are not on the same grid; build one with cityjson_to_dsm(like=...)")
    if fill is None:
        data = a.data - b.data
    else:
        data = np.nan_to_num(a.data, nan=fill) - np.nan_to_num(b.data, nan=fill)
        data[np.isnan(a.data) & np.isnan(b.data)] = np.nan
    return Raster(data=data, transform=a.transform, crs=a.crs)


def clip_to_valid(r: Raster, mask: Raster, crop: bool = False) -> Raster:
    """Keep `r` only where `mask` has valid (finite) data; everything else becomes NaN.

    If `mask` is on a different grid it is warped onto `r`'s grid first (nearest
    neighbour, so the valid footprint isn't smeared). With `crop=True` the result is
    also trimmed to the bounding box of the pixels that remain valid.
    """
    if mask.shape == r.shape and mask.transform.almost_equals(r.transform):
        valid = np.isfinite(mask.data)
    else:
        m = np.full(r.shape, np.nan, np.float32)
        reproject(mask.data.astype(np.float32), m,
                  src_transform=mask.transform, src_crs=mask.crs, src_nodata=np.nan,
                  dst_transform=r.transform, dst_crs=r.crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        valid = np.isfinite(m)

    data = np.where(valid, r.data, np.nan).astype(np.float32)
    transform = r.transform

    if crop:
        valid &= np.isfinite(data)
        if not valid.any():
            raise ValueError("No valid pixels left after clipping")
        rows, cols = np.where(valid.any(axis=1))[0], np.where(valid.any(axis=0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        data = data[r0:r1, c0:c1]
        transform = r.transform * Affine.translation(c0, r0)

    return Raster(data=data, transform=transform, crs=r.crs)


def diff_stats(d: Raster) -> dict[str, float]:
    v = d.data[np.isfinite(d.data)]
    return {"pixels": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "rmse": float(np.sqrt((v ** 2).mean())),
            "abs_p95": float(np.percentile(np.abs(v), 95))}



# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
if __name__ == "__main__":

    CITYJSON_PATH = Path("data_t0/toulouse.city.json")
    DSM_T0 = Path("data_t0/DSM.tif")
    DSM_T1 = Path("data_t1/DSM-cropped.tif")
    DSM_DIFF = Path("data_diff/DSM-CJ-diff.tif")
    DSM_CJ = Path("data_t0/DSM-cityjson.tif")
    GEOID = Path("data_t0/fr_ign_RAF20.tif")


    # tile = load_cityjson(CITYJSON_PATH)
    # print(f"Loaded file {CITYJSON_PATH}")
    # print(f"  - {len(tile.objects)} city objects, {len(tile.vertices)} vertices")

    dsm_1 = load_dsm(DSM_T1)
    print(f"Loaded file {DSM_T1}")
    print(f"  - CRS: {dsm_1.crs}")

    dsm_0 = load_dsm(DSM_T0, like=dsm_1, resampling=Resampling.average)
    print(f"Loaded file {DSM_T0}")
    print(f"  - CRS: {dsm_0.crs}")

    # cj_dsm = cityjson_to_dsm(tile, like=dsm_0, res=0.1, lod="2.2", semantic="RoofSurface", keep_like=True)
    # cj_dsm.save(DSM_CJ)

    # diff = diff_dsm(dsm_1, dsm_0, fill=None)
    # diff.save(DSM_DIFF)