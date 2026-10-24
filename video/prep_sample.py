"""Scene data for the 五楊 route samples: freeways, counties, stations and the ambulance path.

Inputs : output/points_100m.csv, data/fire_stations.csv, county outlines,
         data/raw/osrm/longen_to_viaduct.json (OSRM demo route, 隆恩分隊 -> 五楊北上 49.5K)
Output : video/samples/scene_data.json
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "samples"
BBOX = (121.12, 24.90, 121.50, 25.12)          # lon/lat window of the scene
LON0, LAT0 = 121.30, 25.00                      # origin of the local km grid
JOIN_KM = 49.5                                  # where the OSRM leg ends on the 五楊 deck (北上)
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 110.57


def xy(lon, lat):
    return [round((lon - LON0) * KX, 3), round((lat - LAT0) * KY, 3)]


def main():
    p = pd.read_csv(ROOT / "output" / "points_100m.csv", dtype={"ref": str}, low_memory=False)
    p = p[p.lon.between(BBOX[0], BBOX[2]) & p.lat.between(BBOX[1], BBOX[3])]
    p["viaduct"] = p.road_name.fillna("").str.contains("高架")
    roads = []
    for (ch, v), g in p.groupby(["chain", "viaduct"]):
        g = g.sort_values("s_m")
        # split where the chain leaves the window or switches level
        brk = np.where(np.diff(g.s_m.values) > 150)[0] + 1
        for a, b in zip([0, *brk], [*brk, len(g)]):
            part = g.iloc[a:b]
            if len(part) > 2:
                roads.append({"ref": part.ref.iloc[0], "viaduct": bool(v),
                              "pts": [xy(a, b) for a, b in zip(part.lon, part.lat)]})

    # ambulance path: 隆恩分隊 -> (surface roads and other freeways) -> 五楊北上 49.5K -> chain 4 to 30.5K
    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    stations = [{"name": n.replace("政府消防局", "").replace("市", "市 "), "p": xy(a, b)}
                for n, a, b in zip(st.name, st.lon, st.lat)
                if BBOX[0] <= a <= BBOX[2] and BBOX[1] <= b <= BBOX[3]]
    osrm = json.load(open(ROOT / "data" / "raw" / "osrm" / "longen_to_viaduct.json"))
    road = [xy(a, b) for a, b in osrm["routes"][0]["geometry"]["coordinates"]]
    # interchange labels at the route's own manoeuvre points
    steps = {}
    for st_ in osrm["routes"][0]["legs"][0]["steps"]:
        name = st_["name"].replace("交流道", "")
        if st_["name"].endswith("交流道") and name not in ("機場系統", "三鶯") and name not in steps:
            steps[name if "系統" in name else st_["name"]] = st_["maneuver"]["location"]
    via = p[p.chain == 4].sort_values("s_m")
    via = via[(via.km <= JOIN_KM) & (via.km >= 30.46)]
    viaduct = [xy(a, b) for a, b in zip(via.lon, via.lat)]
    up = p[(p.chain == 4) & (p.km > JOIN_KM)].sort_values("s_m")    # deck south of the join, to find the ramp
    viaduct_up = [xy(a, b) for a, b in zip(up.lon, up.lat)]
    start = xy(121.35733978850249, 24.940700736686043)

    cty = gpd.read_file(ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json",
                        layer="counties").set_crs(4326)
    cty = cty[cty.COUNTYNAME.isin(["臺北市", "新北市", "桃園市"])]
    counties = []
    for name, geom in zip(cty.COUNTYNAME, cty.geometry.intersection(box(*BBOX))):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            counties.append({"name": name, "ring": [xy(a, b) for a, b in poly.exterior.coords]})

    data = {"roads": roads, "counties": counties, "stations": stations,
            "path": {"surface_real": road, "surface_start": start, "viaduct": viaduct, "viaduct_up": viaduct_up},
            # model travel times (07_wuyang): reaching the 機場系統 merge and 五股 30.5K
            "meta": {"station": "隆恩分隊", "min_join": 19.4, "min_end": 31.6},
            "labels": [["隆恩分隊（新北市三峽）", start, "st"], ["機場系統", xy(121.2825, 25.0262), "ic"],
                       *[[n, xy(*loc), "ic"] for n, loc in steps.items()]]}
    (OUT / "scene_data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(len(roads), "road pieces,", len(counties), "county rings,", len(stations), "stations,",
          len(road), "surface pts,", len(viaduct), "viaduct pts")


if __name__ == "__main__":
    main()
