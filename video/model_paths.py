"""Ambulance routes for the 五楊 animation, taken from the project's own road model (03_route.py).

For each station: single-source Dijkstra (travel time) on the same graph 07_wuyang.py uses, to the
五楊北上 point nearest TARGET_KM, with the path rebuilt from the predecessor tree. Every vertex
carries the model's elapsed seconds, so the on-screen clock and the drawn route come from the
same calculation.

Inputs : data/raw/osm_roads.json (01), data/processed/stations_snapped.csv (03),
         output/points_100m.csv, data/processed/wuyang_points.csv (07, for the cross-check)
Output : video/data/model_paths.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("route", ROOT / "scripts" / "03_route.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

VIADUCT = ("五股楊梅高架道路", "汐止五股高架道路")
TARGET = (4, 30.5)        # chain 4 = 五楊北上 deck; km of the incident point (五股前)
STATIONS = ["桃園市政府消防局興國分隊", "桃園市政府消防局青埔分隊", "新北市政府消防局隆恩分隊",
            "新北市政府消防局文化分隊", "桃園市政府消防局大園分隊"]


def main():
    ways = r.load_ways()
    nodes, edges = r.build_graph(ways)
    keep = r.giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    chains, _ = r.build_chains(ways, nodes, edges, keep)
    g = r.sparse(edges, "sec", len(nodes))
    way_of = edges.groupby(["u", "v"]).way.first()

    # the target point, re-sampled exactly as 03 did
    pts = pd.read_csv(ROOT / "output" / "points_100m.csv", dtype={"ref": str}, low_memory=False)
    ci, km = TARGET
    smp = r.sample_chain(chains[ci], nodes)
    sel = pts[pts.chain == ci].sort_values("s_m").reset_index(drop=True)
    i = int(((sel.km - km).abs() + (~sel.road_name.isin(VIADUCT)) * 99).idxmin())
    tail = int(smp["p"][smp["ei"][i]])
    kmh = r.DEFAULT_KMH.get("motorway", 100)
    kmh = edges[(edges.u == tail) & (edges.v == smp["p"][smp["ei"][i] + 1])].kmh.max() or kmh
    tgt = sel.iloc[i]
    print(f"target: chain {ci} km {tgt.km} ({tgt.road_name}) {tgt.lon:.5f},{tgt.lat:.5f}")

    st = pd.read_csv(ROOT / "data" / "processed" / "stations_snapped.csv")
    check = pd.read_csv(ROOT / "data" / "processed" / "wuyang_points.csv") if \
        (ROOT / "data" / "processed" / "wuyang_points.csv").exists() else None
    out = {"target": {"km": float(tgt.km), "lon": float(tgt.lon), "lat": float(tgt.lat)}, "routes": {}}
    for name in STATIONS:
        s = st[st.name == name].iloc[0]
        dist, pred = dijkstra(g, directed=True, indices=int(s.snap_node), return_predecessors=True)
        path = [tail]
        while path[-1] != s.snap_node:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        snap_s = s.snap_m / (r.SNAP_SPEED_KMH / 3.6)
        sec = [0.0] + [snap_s + dist[n] for n in path]
        lon = [float(s.lon)] + nodes.lon.values[path].tolist()
        lat = [float(s.lat)] + nodes.lat.values[path].tolist()
        via = [False]
        for a, b in zip([path[0], *path[:-1]], path):
            w = ways.get(int(way_of.get((a, b), -1)), {}).get("tags", {}) if a != b else {}
            via.append(w.get("name") in VIADUCT)
        end = sec[-1] + smp["off"][i] / (kmh / 3.6)
        sec.append(end); lon.append(float(tgt.lon)); lat.append(float(tgt.lat)); via.append(True)
        names = [ways[int(way_of[(a, b)])]["tags"].get("name", "") for a, b in zip(path[:-1], path[1:])]
        vname = ["", ""] + names + [names[-1]]          # name of the road leading to each vertex
        print(f"{name}: {end / 60:.2f} min, {len(path)} nodes, via {list(dict.fromkeys(n for n in names if n))[:12]}")
        out["routes"][name] = {"lon": [round(v, 6) for v in lon], "lat": [round(v, 6) for v in lat],
                               "sec": [round(v, 1) for v in sec], "viaduct": via, "road": vname}
    (ROOT / "video" / "data").mkdir(exist_ok=True)
    (ROOT / "video" / "data" / "model_paths.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
