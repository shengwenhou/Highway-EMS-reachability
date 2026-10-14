"""Print the actual route behind one segment's result, to check direction and ramp use by hand.

Usage:
  python scripts/05_trace.py 3 南下 301 hospital   # point -> nearest heavy-level hospital
  python scripts/05_trace.py 10 東行 32 station    # nearest fire station -> point

Uses the distance-optimal result of the main carriageway segment (worst point of that km).
Needs data/raw/osm_roads.json and data/processed/ from 01 and 03.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("route", ROOT / "scripts" / "03_route.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def main(ref, direction, km, leg):
    seg = pd.read_csv(ROOT / "data" / "processed" / "segments.csv", dtype={"ref": str})
    row = seg[(seg.ref == ref) & (seg.direction == direction) & (seg.km_from == km)
              & (seg.kind == "main")].iloc[0]
    ways = r.load_ways()
    nodes, edges = r.build_graph(ways)
    keep = r.giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    n = len(nodes)
    chains, _ = r.build_chains(ways, nodes, edges, keep)
    edge_way = edges.groupby(["u", "v"]).way.first()

    prefix = "worst_hosp" if leg == "hospital" else "worst_dist"
    p = np.array(chains[row.chain]["nodes"])
    x, y = r.to_m.transform(row[prefix + "_lon"], row[prefix + "_lat"])
    point = p[np.argmin(np.hypot(nodes.x.values[p] - x, nodes.y.values[p] - y))]

    if leg == "hospital":
        fac = pd.read_csv(ROOT / "data" / "processed" / "hospitals_snapped.csv")
        name = row.hospital_dist
        g, src = r.sparse(edges, "len", n), point             # point -> hospital
    else:
        fac = pd.read_csv(ROOT / "data" / "processed" / "stations_snapped.csv")
        name = row.station_dist
        g, src = r.sparse(edges.rename(columns={"u": "v", "v": "u"}), "len", n), point  # reversed
    tgt = fac[fac.name == name].snap_node.iloc[0]
    dist, pred = dijkstra(g, indices=src, return_predecessors=True, limit=150_000)
    path = [tgt]
    while path[-1] != src:
        path.append(pred[path[-1]])
    if leg == "hospital":
        path = path[::-1]  # station leg is already station -> point

    legs = []
    for a, b in zip(path[:-1], path[1:]):
        t = ways[edge_way[(a, b)]]["tags"]
        label = f"{t.get('name') or t.get('ref') or '(無名)'} [{t['highway']}]"
        d = float(np.hypot(nodes.x[a] - nodes.x[b], nodes.y[a] - nodes.y[b]))
        if legs and legs[-1][0] == label:
            legs[-1][1] += d
        else:
            legs.append([label, d])
    print(f"國道{ref} {direction} {km}K  {'送醫' if leg == 'hospital' else '反應'}：{name}  "
          f"{dist[tgt] / 1000:.1f} km（不含吸附距離）")
    for label, d in legs:
        if d >= 100:
            print(f"  {label}  {d / 1000:.1f} km")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
