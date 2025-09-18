
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
from scipy.linalg import solve
from tqdm import tqdm

def build_network_with_virtual_links(gdf, connect_threshold=500):
    G = nx.Graph()
    edge_index = {}

    # 添加道路段为边
    for idx, row in gdf.iterrows():
        geom = row.geometry
        osm_id = row["osm_id"]
        if isinstance(geom, MultiLineString):
            lines = geom.geoms
        else:
            lines = [geom]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                dist = LineString([p1, p2]).length
                G.add_edge(tuple(p1), tuple(p2), weight=dist)
                edge_index[(tuple(p1), tuple(p2))] = idx

    # 添加中心点之间的虚拟连接
    centroids = np.array([list(pt.coords)[0] for pt in gdf["centroid"]])
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=connect_threshold)
    for i, j in pairs:
        p1, p2 = tuple(centroids[i]), tuple(centroids[j])
        dist = LineString([p1, p2]).length
        G.add_edge(p1, p2, weight=dist)
    return G, edge_index

def get_centroids(gdf):
    def get_center(geom):
        if geom.is_empty: return None
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda g: g.length)
        return geom.interpolate(0.5, normalized=True)
    gdf["centroid"] = gdf.geometry.apply(get_center)
    return gdf[gdf["centroid"].notnull()].copy()

def extract_distance_matrix(graph, known_pts, target_pts):
    known_xy = [tuple(pt.coords[0]) for pt in known_pts]
    target_xy = [tuple(pt.coords[0]) for pt in target_pts]
    D = np.full((len(known_xy), len(target_xy)), np.inf)
    for i, src in enumerate(known_xy):
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, src, weight="weight")
        except:
            lengths = {}
        for j, tgt in enumerate(target_xy):
            if tgt in lengths:
                D[i, j] = lengths[tgt]
    return D

def network_kriging(D_known_known, D_known_unknown, z_known, variogram_range=8000, variogram_sill=1):
    def cov(dist):
        return variogram_sill * np.exp(-dist / variogram_range)
    C = cov(D_known_known)
    C += np.eye(len(C)) * 1e-10
    k = cov(D_known_unknown)
    z_pred = k.T @ solve(C, z_known)
    return z_pred

def run_virtual_network_kriging(geojson_path, csv_path, output_path, connect_threshold=500):
    print("加载数据...")
    gdf = gpd.read_file(geojson_path).to_crs(epsg=32647)
    gdf["osm_id"] = gdf["osm_id"].astype("Int64")
    gdf = get_centroids(gdf)

    df_obs = pd.read_csv(csv_path)
    df_obs["osm_id"] = df_obs["osm_id"].astype("Int64")

    gdf_known = gdf[gdf["osm_id"].isin(df_obs["osm_id"])].merge(df_obs, on="osm_id")
    gdf_unknown = gdf[~gdf["osm_id"].isin(df_obs["osm_id"])]

    print(f"已知点数量: {len(gdf_known)}, 未知待插值点数量: {len(gdf_unknown)}")

    print("构建增强网络...")
    G, edge_index = build_network_with_virtual_links(gdf, connect_threshold=connect_threshold)

    print("计算距离矩阵...")
    D_kk = extract_distance_matrix(G, gdf_known["centroid"], gdf_known["centroid"])
    D_ku = extract_distance_matrix(G, gdf_known["centroid"], gdf_unknown["centroid"])
    D_kk[np.isinf(D_kk)] = 1e6
    D_ku[np.isinf(D_ku)] = 1e6

    print("执行克里金插值...")
    z_known = gdf_known["aadt"].astype(float).values
    z_pred = network_kriging(D_kk, D_ku, z_known, variogram_range=8000)

    gdf.loc[gdf_unknown.index, "aadt"] = np.round(z_pred).astype(int)
    gdf.loc[gdf_known.index, "aadt"] = gdf_known["aadt"].astype(int)

    gdf.drop(columns=["centroid"], errors="ignore").to_file(output_path, driver="GeoJSON", encoding="utf-8")
    print(f"插值完成，结果保存至：{output_path}")
if __name__ == "__main__":
    run_virtual_network_kriging(
        geojson_path="roads_split.geojson",
        csv_path="osm_id_with_aadt.csv",
        output_path="network_kriging.geojson"
    )