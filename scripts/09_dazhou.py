"""國1 x 北外環 大洲交流道 (317.6K, opened 2026-10-28): what changes for ambulances?

The interchange is a half one: 北外環東行 -> 國1北上 (entrance) and 國1南下 -> 北外環西行 (exit).
The same OSM extract is routed twice, without and with the interchange ramps (OSM names them
大洲交流道; before the opening they are tagged highway=construction, so they are switched to
motorway_link here). Everything else is as in 03_route.py.

Outputs:
  output/dazhou_points.csv   國1 main line 305-325K, every 100 m, both variants side by side
  stdout                     per-station time to the new 北上 merge and per-hospital time from the
                             new 南下 diverge: without / with the ramps, and forced through them
"""
import importlib.util
from pathlib import Path

import numpy as np
import osmium
import pandas as pd
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"

spec = importlib.util.spec_from_file_location("route", ROOT / "scripts" / "03_route.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

RAMP_NAME = "大洲交流道"
KM_RANGE = (305, 325)      # project mileage; the interchange sits at about 316.1-316.3K here
STATION_ROWS = 12


class RampCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways = {}

    def way(self, w):
        if w.tags.get("name") == RAMP_NAME and "highway" in w.tags:
            self.ways[w.id] = {"type": "way", "id": w.id, "nodes": [nd.ref for nd in w.nodes],
                               "geometry": [{"lat": nd.location.lat, "lon": nd.location.lon} for nd in w.nodes],
                               "tags": {**dict(w.tags), "highway": "motorway_link"}}


def network(ways):
    nodes, edges = r.build_graph(ways)
    keep = r.giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    hw = {w: ways[w]["tags"].get("highway") for w in edges.way.unique()}
    fw = set(edges.u[edges.way.map(hw).isin(["motorway", "motorway_link"])])
    cand = np.array([i for i in np.unique(edges.u) if i not in fw])
    tree = r.cKDTree(np.c_[nodes.x.values[cand], nodes.y.values[cand]])
    places = []
    for f in ("fire_stations.csv", "heavy_er_hospitals.csv"):
        df = pd.read_csv(ROOT / "data" / f)
        x, y = r.to_m.transform(df.lon.values, df.lat.values)
        d, k = tree.query(np.c_[x, y])
        df["snap_node"], df["snap_m"] = cand[k], d
        places.append(df)
    return nodes, edges, keep, places


def points(ways, nodes, edges, keep, st, hp):
    """Same per-point response / transport as 03_route.py, 國1 main line within KM_RANGE."""
    n = len(nodes)
    snap_s = lambda df: df.snap_m.values / (r.SNAP_SPEED_KMH / 3.6)
    dist_m, lab_d = r.multi_source(edges, n, st.snap_node.values, st.snap_m.values, "len")
    time_s, lab_t = r.multi_source(edges, n, st.snap_node.values, snap_s(st), "sec")
    rev = edges.rename(columns={"u": "v", "v": "u"})
    hdist_m, hlab_d = r.multi_source(rev, n, hp.snap_node.values, hp.snap_m.values, "len")
    htime_s, hlab_t = r.multi_source(rev, n, hp.snap_node.values, snap_s(hp), "sec")
    chains, _ = r.build_chains(ways, nodes, edges, keep)
    jkm = r.junction_km(ways, nodes)
    centres = r.interchange_anchors(chains, nodes, jkm)
    ekmh = edges.groupby(["u", "v"]).kmh.max()
    out = []
    for ch in chains:
        if ch["ref"] != "1" or ch["kind"] != "main":
            continue
        smp = r.sample_chain(ch, nodes)
        dirn = r.direction_name("1", smp["x"], smp["y"])
        anchors = r.anchors_on_chain("1", smp["x"], smp["y"], smp["s"], centres)
        anchors = r.field_anchors("1", dirn, smp["x"], smp["y"], smp["s"], anchors)
        km = r.km_mapping(smp["p"], smp["s"], jkm, anchors)(smp["ss"])
        sel = (km >= KM_RANGE[0]) & (km <= KM_RANGE[1])
        if not sel.any():
            continue
        p, ei, off = smp["p"], smp["ei"], smp["off"]
        tail, head = p[ei], p[ei + 1]
        kmh = np.array([ekmh.get((a, b), 100) for a, b in zip(tail, head)])
        rem = smp["s"][ei + 1] - smp["ss"]
        lon, lat = r.to_ll.transform(smp["px"], smp["py"])
        out.append(pd.DataFrame({
            "direction": dirn, "km": np.round(km, 2), "lon": np.round(lon, 6), "lat": np.round(lat, 6),
            "dist_km": np.round((dist_m[tail] + off) / 1000, 3), "station_dist": st.name.values[lab_d[tail]],
            "time_min": np.round((time_s[tail] + off / (kmh / 3.6)) / 60, 3),
            "station_time": st.name.values[lab_t[tail]],
            "hosp_dist_km": np.round((hdist_m[head] + rem) / 1000, 3), "hospital_dist": hp.name.values[hlab_d[head]],
            "hosp_time_min": np.round((htime_s[head] + rem / (kmh / 3.6)) / 60, 3),
            "hospital_time": hp.name.values[hlab_t[head]]})[sel])
    return pd.concat(out)


def ramp_ends(ramps, ways):
    """(北外環 node, 國1 北上 merge node) and (國1 南下 diverge node, 北外環 node), by OSM id."""
    starts = {w["nodes"][0] for w in ramps.values()}
    ends = {w["nodes"][-1] for w in ramps.values()}
    roads = {}
    for w in ways.values():
        if w["id"] in ramps:
            continue
        for nd in w["nodes"]:
            roads.setdefault(nd, set()).add(w["tags"].get("highway"))
    is_fw = lambda nd: "motorway" in roads.get(nd, ())
    src = [nd for nd in starts - ends if nd in roads]
    dst = [nd for nd in ends - starts if nd in roads]
    entrance = (next(nd for nd in src if not is_fw(nd)), next(nd for nd in dst if is_fw(nd)))
    exit_ = (next(nd for nd in src if is_fw(nd)), next(nd for nd in dst if not is_fw(nd)))
    return entrance, exit_


def via_ramp(nodes, edges, st, hp, entrance, exit_):
    idx = pd.Series(np.arange(len(nodes)), index=nodes.osm_id.values)
    n = len(nodes)
    G = r.sparse(edges, "sec", n)
    GR = r.sparse(edges.rename(columns={"u": "v", "v": "u"}), "sec", n)
    snap_min = lambda df: df.snap_m.values / (r.SNAP_SPEED_KMH / 3.6) / 60
    ring_in, merge = idx[entrance[0]], idx[entrance[1]]
    to_merge, to_ring = dijkstra(GR, indices=merge), dijkstra(GR, indices=ring_in)
    sv = pd.DataFrame({"station": st.name,
                       "best_min": to_merge[st.snap_node] / 60 + snap_min(st),
                       "via_new_min": (to_ring[st.snap_node] + to_merge[ring_in]) / 60 + snap_min(st)})
    diverge, ring_out = idx[exit_[0]], idx[exit_[1]]
    from_div, from_ring = dijkstra(G, indices=diverge), dijkstra(G, indices=ring_out)
    hv = pd.DataFrame({"hospital": hp.name,
                       "best_min": from_div[hp.snap_node] / 60 + snap_min(hp),
                       "via_new_min": (from_div[ring_out] + from_ring[hp.snap_node]) / 60 + snap_min(hp)})
    return sv.set_index("station"), hv.set_index("hospital")


def main():
    rc = RampCollector()
    rc.apply_file(str(RAW / "taiwan.osm.pbf"), locations=True)
    if not rc.ways:
        raise SystemExit(f"no ways named {RAMP_NAME} in the OSM extract")
    base = r.load_ways()
    without = {k: v for k, v in base.items() if k not in rc.ways}
    with_ = {**without, **rc.ways}
    entrance, exit_ = ramp_ends(rc.ways, with_)

    res, sv, hv = {}, {}, {}
    for tag, ways in (("without", without), ("with", with_)):
        nodes, edges, keep, (st, hp) = network(ways)
        res[tag] = points(ways, nodes, edges, keep, st, hp)
        sv[tag], hv[tag] = via_ramp(nodes, edges, st, hp, entrance, exit_)
    key = ["direction", "km", "lon", "lat"]
    m = res["without"].merge(res["with"], on=key, suffixes=("_without", "_with"))
    m.sort_values(key).to_csv(OUT / "dazhou_points.csv", index=False)

    pd.set_option("display.width", 200)
    for col in ("dist_km", "time_min", "hosp_dist_km", "hosp_time_min"):
        d = m[f"{col}_with"] - m[f"{col}_without"]
        ch = m[d.abs() > 0.005]
        print(f"{col}: {len(ch)} points change, by {d.min():.2f} .. {d.max():.2f}",
              "" if ch.empty else f"({', '.join(f'{k} {g.km.min()}-{g.km.max()}K' for k, g in ch.groupby('direction'))})")
    for title, t, rows in (("station -> new 北上 merge (min)", sv, STATION_ROWS),
                           ("new 南下 diverge -> hospital (min)", hv, 3)):
        tab = pd.DataFrame({"without": t["without"].best_min, "with": t["with"].best_min,
                            "via_new_ramp": t["with"].via_new_min}).sort_values("with").head(rows)
        print(f"\n{title}: best route without / with the ramps, and forced through the new ramp")
        print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
