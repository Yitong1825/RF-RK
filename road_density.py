import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

IN_ROADS    = "roads_with_poi_feats.geojson"
OUT_ROADS   = "roads_with_density.geojson"
CRS_METRIC  = 32647
RADIUS_M    = 300.0
INCLUDE_SELF = True

roads = gpd.read_file(IN_ROADS)
orig_crs = roads.crs
roads = roads[roads.geometry.notna()].copy()

roads_m = roads.to_crs(epsg=CRS_METRIC)

roads_m = roads_m[roads_m.geometry.type.isin(["LineString", "MultiLineString"])].copy()
try:
    roads_m = roads_m.explode(index_parts=False).reset_index(drop=True)
except TypeError:
    roads_m = roads_m.explode(ignore_index=True)

roads_m["seg_id"] = np.arange(len(roads_m))
roads_m["seg_len_m"] = roads_m.geometry.length

# Build a circular neighborhood centered on the center of mass
centers = roads_m[["seg_id", "geometry"]].copy()
centers["geometry"] = centers.geometry.centroid
buffers = centers.copy()
buffers["geometry"] = buffers.geometry.buffer(RADIUS_M)

# Spatial connection: Road segment ∩ Circular neighborhood of each road
left  = roads_m[["seg_id", "geometry"]].copy()
right = buffers[["seg_id", "geometry"]].rename(columns={"seg_id": "buf_id"})  # 保持活动几何列名为 'geometry'

pairs = gpd.sjoin(left, right, how="inner", predicate="intersects")

if "seg_id_left" in pairs.columns:
    pairs = pairs.rename(columns={"seg_id_left": "seg_id"})
if "buf_id_right" in pairs.columns:
    pairs = pairs.rename(columns={"buf_id_right": "buf_id"})
if "buf_id" not in pairs.columns and "index_right" in pairs.columns:
    pairs = pairs.merge(
        right.reset_index()[["index", "buf_id"]].rename(columns={"index": "index_right"}),
        on="index_right", how="left"
    )

# Use the neighborhood geometry union (named buf_geom) for clipping the length
pairs = pairs.merge(
    buffers[["seg_id", "geometry"]].rename(columns={"seg_id": "buf_id", "geometry": "buf_geom"}),
    on="buf_id", how="left"
)

# Calculate the intersection length between the line segment and the circular area.
def clip_len(row):
    try:
        return row["geometry"].intersection(row["buf_geom"]).length
    except Exception:
        return 0.0

pairs["clip_len_m"] = pairs.apply(clip_len, axis=1)

if not INCLUDE_SELF:
    pairs.loc[pairs["seg_id"] == pairs["buf_id"], "clip_len_m"] = 0.0

# # Summarized by buf_id (circles representing each road): Total length (meters)
sum_len_m = pairs.groupby("buf_id", as_index=True)["clip_len_m"].sum()

area_km2 = (np.pi * (RADIUS_M**2)) / 1e6
density_km_per_km2 = (sum_len_m / 1000.0) / area_km2  # km / km²

# Write back to the road surface
roads_m["road_length_in_R_m"] = roads_m["seg_id"].map(sum_len_m).fillna(0.0)
roads_m[f"road_density_R{int(RADIUS_M)}_km_per_km2"] = (
    roads_m["seg_id"].map(density_km_per_km2).fillna(0.0).astype(float)
)

roads_m["road_density"] = roads_m[f"road_density_R{int(RADIUS_M)}_km_per_km2"]

out = roads_m.to_crs(orig_crs)
out.to_file(OUT_ROADS)
print(f"out put：{OUT_ROADS}")