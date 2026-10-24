"""Scene data for the 五楊 route samples: freeways, counties, stations and the ambulance path.

Inputs : output/points_100m.csv, data/fire_stations.csv, county outlines,
         video/data/model_paths.json (the model's route, 隆恩分隊 -> 五楊北上 30.5K)
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
STATION = "新北市政府消防局隆恩分隊"
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

    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    stations = [{"name": n.replace("政府消防局", "").replace("市", "市 "), "p": xy(a, b)}
                for n, a, b in zip(st.name, st.lon, st.lat)
                if BBOX[0] <= a <= BBOX[2] and BBOX[1] <= b <= BBOX[3]]
    # the model's own route (video/model_paths.py): vertices with elapsed seconds and deck flags
    mp = json.load(open(ROOT / "video" / "data" / "model_paths.json"))
    rt = mp["routes"][STATION]
    route = {"pts": [xy(a, b) for a, b in zip(rt["lon"], rt["lat"])], "sec": rt["sec"], "viaduct": rt["viaduct"]}
    # interchange labels where the route first enters each named interchange
    steps = {}
    for name, a, b in zip(rt["road"], rt["lon"], rt["lat"]):
        if name.endswith("交流道") and name not in steps and not name.startswith("三鶯"):
            steps[name.replace("系統交流道", "系統")] = xy(a, b)
    start = route["pts"][0]

    cty = gpd.read_file(ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json",
                        layer="counties").set_crs(4326)
    cty = cty[cty.COUNTYNAME.isin(["臺北市", "新北市", "桃園市"])]
    counties = []
    for name, geom in zip(cty.COUNTYNAME, cty.geometry.intersection(box(*BBOX))):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            counties.append({"name": name, "ring": [xy(a, b) for a, b in poly.exterior.coords]})

    data = {"roads": roads, "counties": counties, "stations": stations,
            "route": route, "meta": {"station": "隆恩分隊"},
            "labels": [["隆恩分隊（新北市三峽）", start, "st"], *[[n, p_, "ic"] for n, p_ in steps.items()]]}
    (OUT / "scene_data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(len(roads), "road pieces,", len(counties), "county rings,", len(stations), "stations,",
          len(route["pts"]), "route pts, labels", list(steps))


if __name__ == "__main__":
    main()
