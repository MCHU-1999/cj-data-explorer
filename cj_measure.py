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
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


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
    extent: tuple[float, ...] | None = None  # metadata.geographicalExtent (minx, miny, minz, maxx, maxy, maxz)

    def bounds(self) -> tuple[float, float, float, float]:
        """2D (xmin, ymin, xmax, ymax): metadata extent if present, else from the vertices."""
        if self.extent is not None:
            xmin, ymin, _, xmax, ymax, _ = self.extent
            return xmin, ymin, xmax, ymax
        (xmin, ymin), (xmax, ymax) = self.vertices[:, :2].min(axis=0), self.vertices[:, :2].max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

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

    meta = cj.get("metadata", {})
    extent = meta.get("geographicalExtent")
    return CityJSON(path=path, objects=objects, vertices=verts, crs=meta.get("referenceSystem"),
                    extent=tuple(extent) if extent else None)


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


def grid_like(path: str | Path, res: float) -> Raster:
    """Empty (NaN) grid with the extent, origin and crs of a GeoTIFF, but pixel size `res`.

    Only the header is read. The origin (top-left corner) is kept so the grid stays
    aligned with the source; the extent is rounded outward to whole `res` pixels.
    """
    with rasterio.open(path) as src:
        t, (h, w), crs = src.transform, src.shape, src.crs
    shape = (int(np.ceil(h * abs(t.e) / res - 1e-9)), int(np.ceil(w * t.a / res - 1e-9)))
    return Raster(data=np.full(shape, np.nan, np.float32),
                  transform=Affine(res, 0, t.c, 0, -res, t.f), crs=crs)


@dataclass
class _Burn:
    """One surface rasterised on a grid (see _burn_surfaces)."""
    obj: CityObject
    index: int                      # index of the surface in obj.surfaces(lod, semantic)
    surface: Surface
    normal: np.ndarray              # unit normal of the fitted plane, n_z > 0
    centroid: np.ndarray            # a point on the plane
    rows: slice                     # window into the full grid
    cols: slice
    mask: np.ndarray                # (rows, cols) bool, pixel centre inside the surface
    z: np.ndarray                   # (rows, cols) plane height at the pixel centres


def _burn_surfaces(tile: CityJSON, grid: Raster, lod: str, semantic: str | None):
    """Yield every non-vertical surface rasterised on `grid` (north-up), with its plane
    height at the pixel centres. Each surface is planar, so the height is taken from the
    plane fitted through its exterior ring. Surfaces outside the grid are skipped."""
    t, (nrows, ncols) = grid.transform, grid.shape
    inv = ~t
    for obj in tile.objects.values():
        for index, surface in enumerate(obj.surfaces(lod, semantic)):
            exterior = surface.rings[0]
            centroid = exterior.mean(axis=0)    # mean vertex: a point on the fitted plane
            normal = np.linalg.svd(exterior - centroid)[2][-1]
            if abs(normal[2]) < 1e-6:           # vertical: invisible from above
                continue
            if normal[2] < 0:
                normal = -normal

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

            yield _Burn(obj, index, surface, normal, centroid,
                        slice(row0, row1), slice(col0, col1), mask, z)


@dataclass
class Facet:
    building_id: str                # the Building (parent of the BuildingPart, if any)
    object_id: str                  # the CityObject holding the surface
    index: int                      # index in object.surfaces(lod, semantic)
    surface: Surface
    normal: np.ndarray              # unit normal, n_z > 0


def roof_facets(tile: CityJSON, grid: Raster, lod: str = "2.2",
                semantic: str = "RoofSurface") -> tuple[Raster, np.ndarray, list[Facet]]:
    """Rasterise the facets of one semantic class on `grid`.

    Returns the z raster (NaN off facets), an int32 array of facet ids (-1 off facets;
    where facets overlap the highest wins, as in cityjson_to_dsm) and the facet table,
    indexed by facet id.
    """
    z = np.full(grid.shape, np.nan, np.float32)
    ids = np.full(grid.shape, -1, np.int32)
    facets: list[Facet] = []
    for b in _burn_surfaces(tile, grid, lod, semantic):
        fid = len(facets)
        facets.append(Facet(b.obj.parents[0] if b.obj.parents else b.obj.id, b.obj.id,
                            b.index, b.surface, b.normal))
        zw, iw = z[b.rows, b.cols], ids[b.rows, b.cols]
        win = b.mask & ~(zw >= b.z)             # NaN compares False: empty cells are taken
        zw[win], iw[win] = b.z[win], fid
    return Raster(z, grid.transform, grid.crs), ids, facets


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

    for b in _burn_surfaces(tile, out, lod, semantic):
        # `window` is a view into out.data; fmax skips NaN, so the highest surface wins
        window = out.data[b.rows, b.cols]
        window[b.mask] = np.fmax(window[b.mask], b.z[b.mask])

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



def clip_to_bounds(src_path: str | Path, dst_path: str | Path,
                   bounds: tuple[float, float, float, float], bounds_crs: CRS | str | None = None,
                   pad: float = 0.0) -> None:
    """Crop a GeoTIFF on disk to (xmin, ymin, xmax, ymax), e.g. `tile.bounds()`.

    Only the pixels inside the window are read; all bands, the dtype, nodata and
    colour interpretation are kept as they are (works for RGB orthophotos too). The
    window is snapped outward to whole pixels, so the source grid is not resampled.
    If `bounds_crs` differs from the raster's CRS the bounds are reprojected first.
    """
    with rasterio.open(src_path) as src:
        if bounds_crs is not None and CRS.from_user_input(_epsg(str(bounds_crs))) != src.crs:
            bounds = transform_bounds(CRS.from_user_input(_epsg(str(bounds_crs))), src.crs, *bounds)
        xmin, ymin, xmax, ymax = bounds
        win = from_bounds(xmin - pad, ymin - pad, xmax + pad, ymax + pad, src.transform)
        c0, r0 = max(int(np.floor(win.col_off)), 0), max(int(np.floor(win.row_off)), 0)
        c1 = min(int(np.ceil(win.col_off + win.width)), src.width)
        r1 = min(int(np.ceil(win.row_off + win.height)), src.height)
        if c0 >= c1 or r0 >= r1:
            raise ValueError("Bounds do not overlap the raster")
        win = Window(c0, r0, c1 - c0, r1 - r0)

        profile = src.profile | {
            "driver": "GTiff", "width": win.width, "height": win.height,
            "transform": src.window_transform(win),
            "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "DEFLATE", "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.colorinterp = src.colorinterp
            # Copy block by block so memory stays bounded however large the crop is.
            for _, block in dst.block_windows(1):
                src_block = Window(win.col_off + block.col_off, win.row_off + block.row_off,
                                   block.width, block.height)
                dst.write(src.read(window=src_block), window=block)


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
    DSM_DIFF = Path("data_diff/DSM-diff.tif")
    DSM_CJ = Path("data_t0/DSM-cityjson.tif")
    ORTHO_T1 = Path("/home/MCHu//Documents/projects/Toulouse_EO/results/Ortho.tif")
    GEOID = Path("data_t0/fr_ign_RAF20.tif")


    tile = load_cityjson(CITYJSON_PATH)
    print(f"Loaded file {CITYJSON_PATH}")
    print(f"  - {len(tile.objects)} city objects, {len(tile.vertices)} vertices")

    # dsm_1 = load_dsm(DSM_T1)
    # print(f"Loaded file {DSM_T1}")
    # print(f"  - CRS: {dsm_1.crs}")

    # dsm_0 = load_dsm(DSM_T0, like=dsm_1, resampling=Resampling.average)
    # print(f"Loaded file {DSM_T0}")
    # print(f"  - CRS: {dsm_0.crs}")

    clip_to_bounds(ORTHO_T1, "data_t1/Ortho-cropped.tif", tile.bounds(), bounds_crs=tile.crs, pad=0.5)

    # cj_dsm = cityjson_to_dsm(tile, like=dsm_0, res=0.1, lod="2.2", semantic="RoofSurface", keep_like=True)
    # cj_dsm.save(DSM_CJ)

    # diff = diff_dsm(dsm_1, dsm_0, fill=None)
    # diff.save(DSM_DIFF)