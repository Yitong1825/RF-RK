# OUT_PATH    = "roads_with_population.geojson"
# Khwaeng(subdistricts).geojson

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask

RASTER_PATH = "pop.tif"
ZONES_PATH  = "Khwaeng(subdistricts).geojson"
ROADS_PATH  = "roads_2k.geojson"
OUT_PATH    = "roads_with_population.geojson"

METRIC_EPSG = 32647
POP_COL     = "population"
ADD_ROAD_ID = True

# Retain the division of polygons in the geometry; if there are no polygons, return None
def keep_polygonal_part(geom):
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    if hasattr(geom, "geoms"):  # GeometryCollection 等
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return unary_union(polys)
    return None

# Unify Polygon/MultiPolygon into a single Polygon (explode), and fix self-intersections
def to_singlepart_polygons(gdf):
    g = gdf.copy()
    g["geometry"] = g.geometry.apply(keep_polygonal_part)
    g = g[~g.geometry.isna() & ~g.geometry.is_empty].copy()
    # 仅对面数据用 buffer(0) 修复
    g["geometry"] = g.geometry.buffer(0)
    g = g.explode(index_parts=False, ignore_index=True)
    g = g[g.geometry.type.eq("Polygon")].copy()
    return g

# Unify into LineString
def to_singlepart_lines(gdf):
    g = gdf.copy()
    g = g[g.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    g = g.explode(index_parts=False, ignore_index=True)
    g = g[g.geometry.type.eq("LineString")].copy()
    return g

# ========= read data =========
zones = gpd.read_file(ZONES_PATH)
roads = gpd.read_file(ROADS_PATH)

if ADD_ROAD_ID and "road_id" not in roads.columns:
    roads["road_id"] = np.arange(len(roads))

orig_crs_roads = roads.crs
orig_crs_zones = zones.crs

with rasterio.open(RASTER_PATH) as src:
    raster_crs = src.crs

zones_r = zones.to_crs(raster_crs)
roads_r = roads.to_crs(raster_crs)

# ========= 2) population data is given to every sub-district =========
def sample_raster_at_point(src, x, y, nodata_val):
    val = list(src.sample([(x, y)]))[0][0]
    if nodata_val is not None and val == nodata_val:
        return None
    if np.isnan(val):
        return None
    return float(val)

def mean_in_polygon(src, poly, nodata_val):
    try:
        out, _ = mask(src, [mapping(poly)], crop=True, nodata=nodata_val)
        arr = out[0].astype("float64")
        if nodata_val is not None:
            valid = arr[arr != nodata_val]
        else:
            valid = arr[~np.isnan(arr)]
        return float(np.nanmean(valid)) if valid.size else None
    except Exception:
        return None

with rasterio.open(RASTER_PATH) as src:
    nodata = src.nodata
    zone_vals = []
    for geom in zones_r.geometry:
        if geom is None or geom.is_empty:
            zone_vals.append(0.0); continue
        p = geom.representative_point()
        v = sample_raster_at_point(src, p.x, p.y, nodata)
        if v is None:
            v = mean_in_polygon(src, geom, nodata)
            if v is None:
                v = 0.0
        zone_vals.append(v)

zones_r[POP_COL] = np.array(zone_vals, dtype=float)

# ========= 4)Clean up the geometric types to avoid the "mixed geometry" error reported by the overlay =========
zones_r_slim = to_singlepart_polygons(zones_r[[POP_COL, "geometry"]])
roads_r_single = to_singlepart_lines(roads_r[["road_id", "geometry"]] if "road_id" in roads_r.columns else roads_r[["geometry"]])


segments = gpd.overlay(
    roads_r_single,
    zones_r_slim,
    how="intersection",
    keep_geom_type=True
)


if len(segments) == 0:
    roads_out = roads.copy()
    roads_out[POP_COL] = 0.0
    roads_out.to_file(OUT_PATH, driver="GeoJSON")
    print("No intersection")
else:
    # ========= 6) In metric projection, calculate the weighted average based on "segment length" =========
    segments_m = segments.to_crs(epsg=METRIC_EPSG)
    segments_m["len_m"] = segments_m.geometry.length

    segments_m = segments_m[segments_m.geometry.type.eq("LineString")].copy()

    # Addition road_id
    if "road_id" not in segments_m.columns:
        segments_m["road_id"] = np.arange(len(segments_m))

    grp = segments_m.groupby("road_id", dropna=False)
    wmean = (grp.apply(lambda df: (df[POP_COL] * df["len_m"]).sum() / df["len_m"].sum())
             .rename(POP_COL)
             .astype(float))

    roads_out = roads.copy()
    roads_out = roads_out.merge(wmean.to_frame(), on="road_id", how="left")
    roads_out[POP_COL] = roads_out[POP_COL].fillna(0.0).astype(float)

    roads_out = roads_out.to_crs(orig_crs_roads)
    roads_out.to_file(OUT_PATH, driver="GeoJSON")
    print("out put:", OUT_PATH)
