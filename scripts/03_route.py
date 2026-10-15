"""Build a directed road graph and, for every point of every national freeway carriageway,
compute (a) the ambulance response from the nearest fire station and (b) the transport onward
to the nearest heavy-level emergency hospital (重度級急救責任醫院). Summarise per 1-km segment.

Model (see README for the full list of assumptions):
- Ambulances follow one-way rules on every road class except residential / living_street.
- A point on a freeway is reachable only along its own carriageway, i.e. from an upstream
  on-ramp. No U-turns across the median, no emergency gates.
- Speeds = OSM maxspeed where tagged, otherwise a per-class default (free flow).
- Distance-optimal and time-optimal routes are computed separately (two Dijkstra runs).
- Transport leaves the point downstream along the same carriageway, to the next exit.

Outputs (data/processed/):
  segments.geojson   one feature per carriageway x 1-km segment
  segments.csv       same, without geometry
"""
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_KMH = {
    "motorway": 100, "motorway_link": 50, "trunk": 80, "trunk_link": 50,
    "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30, "unclassified": 40, "residential": 30,
    "living_street": 20,
}
IGNORE_ONEWAY = {"residential", "living_street"}
FREEWAY_REFS = {"1", "2", "2甲", "3", "3甲", "4", "5", "6", "8", "10"}
# Directions: N-S freeways are 北上/南下, the rest 東行/西行.
NS_REFS = {"1", "3", "5"}
NOT_FREEWAY_NAMES = ("高雄港聯外高架道路",)
# OSM ways whose oneway direction is drawn backwards, verified against 高公局 interchange diagrams.
# Value "-1" reverses the way's direction. See VERIFY.md section 一.
ONEWAY_FIX = {
    317134259: "-1",   # 國3 白河交流道南下出口 (OSM drew it as an entrance; 高公局 diagram 03-311)
    317134263: "-1",
}
SNAP_SPEED_KMH = 20      # station -> nearest graph node, straight-line
SAMPLE_STEP_M = 100      # resolution used to find the worst point inside each 1-km segment

to_m = Transformer.from_crs(4326, 3826, always_xy=True)
to_ll = Transformer.from_crs(3826, 4326, always_xy=True)


def parse_speed(v):
    if not v:
        return None
    m = re.match(r"\s*(\d+)", v)
    if not m:
        return None
    s = float(m.group(1))
    if "mph" in v:
        s *= 1.609
    return s if 5 <= s <= 130 else None


def load_ways():
    ways = {}
    for f in [RAW / "osm_roads.json"]:
        for e in json.load(open(f))["elements"]:
            if e["type"] == "way" and "geometry" in e:
                ways[e["id"]] = e
    print(f"{len(ways)} ways")
    return ways


def build_graph(ways):
    node_ll = {}
    us, vs, lens, secs, edge_way = [], [], [], [], []
    for wid, w in ways.items():
        t = w.get("tags", {})
        hw = t.get("highway")
        if t.get("access") in ("no", "private") or t.get("motor_vehicle") in ("no", "private"):
            continue
        nodes = w["nodes"]
        geom = w["geometry"]
        if len(nodes) != len(geom):
            continue
        for n, g in zip(nodes, geom):
            if g:
                node_ll[n] = (g["lon"], g["lat"])
        ow = ONEWAY_FIX.get(wid, t.get("oneway", ""))
        if hw in IGNORE_ONEWAY:
            fwd, bwd = True, True
        elif ow in ("yes", "1", "true"):
            fwd, bwd = True, False
        elif ow == "-1":
            fwd, bwd = False, True
        elif ow == "no":
            fwd, bwd = True, True
        elif hw == "motorway" or t.get("junction") in ("roundabout", "circular"):
            fwd, bwd = True, False
        else:
            fwd, bwd = True, True
        kmh = parse_speed(t.get("maxspeed")) or DEFAULT_KMH.get(hw, 30)
        for a, b in zip(nodes[:-1], nodes[1:]):
            if fwd:
                us.append(a); vs.append(b); secs.append(kmh); edge_way.append(wid)
            if bwd:
                us.append(b); vs.append(a); secs.append(kmh); edge_way.append(wid)

    ids = np.array(sorted(node_ll))
    idx = {n: i for i, n in enumerate(ids)}
    lon = np.array([node_ll[n][0] for n in ids])
    lat = np.array([node_ll[n][1] for n in ids])
    x, y = to_m.transform(lon, lat)
    u = np.array([idx[a] for a in us])
    v = np.array([idx[b] for b in vs])
    length = np.hypot(x[u] - x[v], y[u] - y[v])
    kmh = np.array(secs, dtype=float)
    sec = length / (kmh / 3.6)
    edges = pd.DataFrame({"u": u, "v": v, "len": length, "sec": sec, "kmh": kmh,
                          "way": np.array(edge_way)})
    nodes = pd.DataFrame({"osm_id": ids, "lon": lon, "lat": lat, "x": x, "y": y})
    print(f"graph: {len(nodes)} nodes, {len(edges)} directed edges")
    return nodes, edges


def giant_scc(nodes, edges):
    n = len(nodes)
    g = csr_matrix((np.ones(len(edges)), (edges.u, edges.v)), shape=(n, n))
    _, lab = connected_components(g, directed=True, connection="strong")
    big = np.bincount(lab).argmax()
    keep = lab == big
    print(f"giant SCC: {keep.sum()} / {n} nodes")
    return keep


def sparse(edges, col, n):
    # csgraph sums duplicate entries, so collapse parallel edges to their minimum first.
    e = edges.groupby(["u", "v"], as_index=False)[col].min()
    w = np.maximum(e[col].values, 1e-3)
    return csr_matrix((w, (e.u.values, e.v.values)), shape=(n, n))


def multi_source(edges, n, src_nodes, src_cost, col):
    """Dijkstra from a virtual super-source; returns cost and index of the winning source."""
    S = n
    extra = pd.DataFrame({"u": S, "v": src_nodes, col: src_cost})
    allE = pd.concat([edges[["u", "v", col]], extra], ignore_index=True)
    g = sparse(allE, col, n + 1)
    dist, pred = dijkstra(g, directed=True, indices=S, return_predecessors=True)
    # Propagate the source label along the shortest-path tree, in order of distance.
    best_src = {}
    for i, (nd, c) in enumerate(zip(src_nodes, src_cost)):
        if nd not in best_src or c < best_src[nd][0]:
            best_src[nd] = (c, i)
    label = np.full(n + 1, -1)
    order = np.argsort(dist)
    for nd in order:
        if not np.isfinite(dist[nd]) or nd == S:
            continue
        p = pred[nd]
        label[nd] = best_src[nd][1] if p == S else label[p]
    return dist[:n], label[:n]


def freeway_ref(t):
    ref = (t.get("ref") or "").strip()
    if ref in FREEWAY_REFS:
        return ref
    return None


def build_chains(ways, nodes, edges, keep):
    """Return a list of carriageway chains: dict(ref, nodes=[graph idx...], kind)."""
    osm2i = pd.Series(np.arange(len(nodes)), index=nodes.osm_id.values)
    mw = edges[edges.way.map(lambda w: ways[w]["tags"].get("highway") == "motorway")].copy()
    mw = mw[keep[mw.u.values] & keep[mw.v.values]]
    mw["ref"] = mw.way.map(lambda w: freeway_ref(ways[w]["tags"]))
    mw["name"] = mw.way.map(lambda w: ways[w]["tags"].get("name", ""))
    mw = mw[~mw.name.isin(NOT_FREEWAY_NAMES)]
    # Short unlabeled pieces (bridges etc.) inherit ref from the motorway edges they touch.
    for _ in range(50):
        lab = mw[mw.ref.notna()]
        n2ref = pd.concat([lab.set_index("u").ref, lab.set_index("v").ref])
        n2ref = n2ref[~n2ref.index.duplicated()]
        miss = mw.ref.isna()
        if not miss.any():
            break
        cand = mw.loc[miss, "u"].map(n2ref).fillna(mw.loc[miss, "v"].map(n2ref))
        if cand.notna().sum() == 0:
            break
        mw.loc[cand.index, "ref"] = cand
    mw = mw.dropna(subset=["ref"])

    chains = []
    for ref, grp in mw.groupby("ref"):
        G = nx.DiGraph()
        for r in grp.itertuples():
            G.add_edge(r.u, r.v, len=r.len)
        for comp in nx.weakly_connected_components(G):
            H = G.subgraph(comp).copy()
            if sum(d["len"] for *_, d in H.edges(data=True)) < 1000:
                continue
            kind = "main"
            while H.number_of_edges():
                while not nx.is_directed_acyclic_graph(H):
                    cyc = nx.find_cycle(H)
                    a, b = min(cyc, key=lambda e: H.edges[e]["len"])[:2]
                    H.remove_edge(a, b)
                path = nx.dag_longest_path(H, weight="len")
                plen = sum(H.edges[a, b]["len"] for a, b in zip(path[:-1], path[1:]))
                if plen < 2000:
                    break
                chains.append({"ref": ref, "nodes": path, "kind": kind})
                H.remove_edges_from(list(zip(path[:-1], path[1:])))
                H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
                kind = "branch"
    return chains, mw


RAMP_SKIP = ("服務區", "休息站", "收費站", "緊急")   # not usable entrances/exits for this model


def ramp_list(p, s, km, edges_out, edges_in, ways, jkm, nodes, branch):
    """Entrance and exit ramps along a chain: [[s (m), km, name, "in"/"out"], ...].

    Taken from the road graph itself (a motorway_link leaving or joining the chain), so it also
    covers viaducts where OSM has few motorway_junction nodes. On a viaduct (branch chain), a
    motorway merging in or splitting off part-way (e.g. 五股 traffic joining 汐五) counts too.
    A ramp without a name takes the name of the nearest interchange node within 2.5 km."""
    on = set(zip(p[:-1], p[1:]))
    jn = [n for n in jkm if jkm[n][1]]
    jx, jy = nodes.x.values[jn], nodes.y.values[jn]

    def nearest_name(n):
        d = np.hypot(jx - nodes.x.values[n], jy - nodes.y.values[n])
        k = int(np.argmin(d)) if len(d) else -1
        return jkm[jn[k]][1] if k >= 0 and d[k] < 2500 else ""

    out = []
    for i, n in enumerate(p):
        for kind, table, other in (("out", edges_out, "v"), ("in", edges_in, "u")):
            if n not in table:
                continue
            for e in table[n]:
                pair = (n, e[other]) if kind == "out" else (e[other], n)
                if pair in on:
                    continue
                t = ways[e["way"]]["tags"]
                hw = t.get("highway")
                transfer = hw == "motorway" and branch and 0 < i < len(p) - 1
                if hw != "motorway_link" and not transfer:
                    continue
                name = "" if transfer else t.get("name", "")
                name = name or nearest_name(n) or ("轉接道" if transfer else "未命名匝道")
                if any(k in name for k in RAMP_SKIP):
                    continue
                out.append([round(float(s[i]), 1), round(float(km[i]), 1), name, kind])
    out.sort()
    dedup = []
    for r in out:   # the same ramp can touch the chain at several consecutive nodes
        if not any(d[2] == r[2] and d[3] == r[3] and abs(d[0] - r[0]) < 800 for d in dedup):
            dedup.append(r)
    return dedup


def junction_km(ways, nodes):
    j = json.load(open(RAW / "osm_junctions.json"))["elements"]
    osm2i = dict(zip(nodes.osm_id.values, range(len(nodes))))
    out = {}
    for e in j:
        ref = e.get("tags", {}).get("ref", "")
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", ref)
        if m and e["id"] in osm2i:
            out[osm2i[e["id"]]] = (float(m.group(1)), e["tags"].get("name", ""))
    return out


def chain_geometry(chain, nodes):
    p = np.array(chain["nodes"])
    x, y = nodes.x.values[p], nodes.y.values[p]
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    return p, x, y, s


def interchange_anchors(chains, nodes, jkm):
    """Interchange centres per freeway. A motorway_junction node marks where an exit leaves its
    own carriageway, which is before the interchange in that direction of travel, so the two
    directions' nodes sit on opposite sides of it. Their mean position is the interchange."""
    pos = defaultdict(list)
    for ch in chains:
        for n in ch["nodes"]:
            if n in jkm:
                pos[(ch["ref"], jkm[n][1], jkm[n][0])].append((nodes.x.values[n], nodes.y.values[n]))
    return {k: np.mean(v, axis=0) for k, v in pos.items()}


def anchors_on_chain(ref, x, y, s, centres, max_off=400.0):
    """Project each interchange centre of this freeway onto the chain -> (s, km) anchors."""
    out = []
    for (r, name, kmv), (cx, cy) in centres.items():
        if r != ref:
            continue
        d = np.hypot(x - cx, y - cy)
        k = int(np.argmin(d))
        if d[k] <= max_off:
            out.append((s[k], kmv))
    return out


def km_mapping(p, s, jkm, anchors=None):
    """Chain distance (m) -> official mileage (km), interpolated between interchange numbers.

    Taiwanese interchange numbers are their mileage, so junction nodes on the chain give
    (distance, km) anchors. Outliers are dropped with a robust linear fit; outside the first
    and last anchor the mileage runs at 1 km per km. Returns None when there are too few anchors.
    """
    pts = sorted(anchors) if anchors is not None else sorted(
        (s[i], jkm[n][0]) for i, n in enumerate(p) if n in jkm)
    if len(pts) < 2:
        return None
    a = np.array(pts)
    slope, icpt = np.polyfit(a[:, 0] / 1000, a[:, 1], 1)
    good = np.abs(a[:, 1] - (slope * a[:, 0] / 1000 + icpt)) < 3
    if good.sum() < 2 or not 0.9 < abs(slope) < 1.1:
        return None
    a = a[good]
    sign = math.copysign(1.0, slope)
    mono = [a[0]]
    for row in a[1:]:
        if sign * (row[1] - mono[-1][1]) > 0:
            mono.append(row)
    a = np.array(mono)

    def f(ss):
        ss = np.asarray(ss, dtype=float)
        km = np.interp(ss, a[:, 0], a[:, 1])
        lo, hi = ss < a[0, 0], ss > a[-1, 0]
        km[lo] = a[0, 1] + sign * (ss[lo] - a[0, 0]) / 1000
        km[hi] = a[-1, 1] + sign * (ss[hi] - a[-1, 0]) / 1000
        return km
    return f


def direction_name(ref, x, y):
    if ref in NS_REFS:
        return "北上" if y[-1] > y[0] else "南下"
    return "東行" if x[-1] > x[0] else "西行"


def sample_chain(ch, nodes):
    """Sample a chain every SAMPLE_STEP_M metres: positions, coordinates and unit tangents."""
    p, x, y, s = chain_geometry(ch, nodes)
    L = s[-1]
    ss = np.append(np.arange(0, L, SAMPLE_STEP_M), L - 1e-6)
    ei = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(p) - 2)
    f = (ss - s[ei]) / np.maximum(s[ei + 1] - s[ei], 1e-9)
    px = x[ei] + f * (x[ei + 1] - x[ei])
    py = y[ei] + f * (y[ei + 1] - y[ei])
    tx, ty = x[ei + 1] - x[ei], y[ei + 1] - y[ei]
    norm = np.maximum(np.hypot(tx, ty), 1e-9)
    return dict(p=p, x=x, y=y, s=s, L=L, ss=ss, ei=ei, off=ss - s[ei], px=px, py=py,
                tx=tx / norm, ty=ty / norm)


def km_from_main(sb, mains):
    """Mileage for a branch (e.g. an elevated parallel carriageway): borrow it from the
    nearest same-direction point of a main carriageway of the same freeway."""
    best = None
    for m in mains:
        tree = cKDTree(np.c_[m["smp"]["px"], m["smp"]["py"]])
        d, k = tree.query(np.c_[sb["px"], sb["py"]])
        dot = np.mean(sb["tx"] * m["smp"]["tx"][k] + sb["ty"] * m["smp"]["ty"][k])
        if dot > 0.5 and (best is None or d.mean() < best[0]):
            best = (d.mean(), m, k)
    if best is None:
        return None, None
    _, m, k = best
    return m["km"][k], m["direction"]


def main():
    ways = load_ways()
    nodes, edges = build_graph(ways)
    keep = giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    n = len(nodes)

    # --- stations / hospitals: snap to nearest non-freeway node of the giant SCC
    hw_of_way = {w: ways[w]["tags"].get("highway") for w in edges.way.unique()}
    fw_nodes = set(edges.u[edges.way.map(hw_of_way).isin(["motorway", "motorway_link"])])
    cand = np.array([i for i in np.unique(edges.u) if i not in fw_nodes])
    tree = cKDTree(np.c_[nodes.x.values[cand], nodes.y.values[cand]])

    def snap(df, what):
        x, y = to_m.transform(df.lon.values, df.lat.values)
        d, k = tree.query(np.c_[x, y])
        df["snap_node"], df["snap_m"] = cand[k], d
        print(f"{what} snap distance (m):", df.snap_m.describe().round(1).to_dict())

    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    snap(st, "station")
    dist_m, lab_d = multi_source(edges, n, st.snap_node.values, st.snap_m.values, "len")
    time_s, lab_t = multi_source(edges, n, st.snap_node.values,
                                 st.snap_m.values / (SNAP_SPEED_KMH / 3.6), "sec")

    # Transport: cost from every node TO the nearest hospital = Dijkstra on the reversed graph.
    hp = pd.read_csv(ROOT / "data" / "heavy_er_hospitals.csv")
    snap(hp, "hospital")
    rev = edges.rename(columns={"u": "v", "v": "u"})
    hdist_m, hlab_d = multi_source(rev, n, hp.snap_node.values, hp.snap_m.values, "len")
    htime_s, hlab_t = multi_source(rev, n, hp.snap_node.values,
                                   hp.snap_m.values / (SNAP_SPEED_KMH / 3.6), "sec")

    chains, mw = build_chains(ways, nodes, edges, keep)
    jkm = junction_km(ways, nodes)
    # speed of each chain edge (for interpolating time inside an edge)
    ekmh = edges.groupby(["u", "v"]).kmh.max()

    chain_way = mw.groupby(["u", "v"]).way.first()
    centres = interchange_anchors(chains, nodes, jkm)
    mains = []
    order = sorted(range(len(chains)), key=lambda i: chains[i]["kind"] != "main")
    for ci in order:
        ch = chains[ci]
        smp = sample_chain(ch, nodes)
        ch["smp"] = smp
        names = pd.Series([ways[chain_way.get((a, b))]["tags"].get("name", "")
                           for a, b in zip(smp["p"][:-1], smp["p"][1:])
                           if (a, b) in chain_way.index])
        ch["name"] = names.value_counts().index[0] if len(names) else ""
        f = km_mapping(smp["p"], smp["s"], jkm,
                       anchors_on_chain(ch["ref"], smp["x"], smp["y"], smp["s"], centres))
        ch["direction"] = direction_name(ch["ref"], smp["x"], smp["y"])
        ch["km_source"] = "interchange"
        if ch["kind"] == "main":
            if f is None:
                ch["km"], ch["km_source"] = smp["ss"] / 1000, "geometric"
            else:
                ch["km"] = f(smp["ss"])
            mains.append(ch)
        else:
            km, dirn = km_from_main(smp, [m for m in mains if m["ref"] == ch["ref"]])
            if km is not None:
                ch["km"], ch["direction"], ch["km_source"] = km, dirn, "main_line"
            elif f is not None:
                ch["km"] = f(smp["ss"])
            else:
                ch["km"], ch["km_source"] = smp["ss"] / 1000, "geometric"

    # ramp lookup, limited to edges that touch a chain (indexing all 3.6 M edges is slow)
    on_chain = np.zeros(len(nodes), bool)
    for ch in chains:
        on_chain[ch["smp"]["p"]] = True
    near = edges[on_chain[edges.u.values] | on_chain[edges.v.values]]
    eo, ei_ = {}, {}
    for u, v, w in zip(near.u.values, near.v.values, near.way.values):
        eo.setdefault(u, []).append({"v": v, "way": w})
        ei_.setdefault(v, []).append({"u": u, "way": w})
    rows, feats, samples, chain_geo = [], [], [], []
    r_speed = lambda t: parse_speed(t.get("maxspeed")) or 0
    for ci in order:
        ch = chains[ci]
        smp, km = ch["smp"], ch["km"]
        p, ei, off = smp["p"], smp["ei"], smp["off"]
        tail = p[ei]
        kmh = np.array([ekmh.get((a, b), 100) for a, b in zip(p[ei], p[ei + 1])])
        Dm = dist_m[tail] + off
        Ts = time_s[tail] + off / (kmh / 3.6)
        sd, stt = lab_d[tail], lab_t[tail]
        head = p[ei + 1]
        rem = smp["s"][ei + 1] - smp["ss"]
        HDm = hdist_m[head] + rem
        HTs = htime_s[head] + rem / (kmh / 3.6)
        hd, ht = hlab_d[head], hlab_t[head]
        seg = np.clip(np.floor(km).astype(int), 0, None)
        lon_all, lat_all = to_ll.transform(smp["px"], smp["py"])
        samples.append(pd.DataFrame({
            "chain": ci, "ref": ch["ref"], "direction": ch["direction"], "kind": ch["kind"],
            "s_m": np.round(smp["ss"], 1), "km_from": seg, "km": np.round(km, 2),
            "lon": np.round(lon_all, 6), "lat": np.round(lat_all, 6),
            "dist_km": np.round(Dm / 1000, 3), "station_dist": st.name.values[sd],
            "time_min": np.round(Ts / 60, 3), "station_time": st.name.values[stt],
            "hosp_dist_km": np.round(HDm / 1000, 3), "hospital_dist": hp.name.values[hd],
            "hosp_time_min": np.round(HTs / 60, 3), "hospital_time": hp.name.values[ht],
            "road_name": [ways[chain_way[(a, b)]]["tags"].get("name", "") if (a, b) in chain_way.index
                          else "" for a, b in zip(p[ei], p[ei + 1])]}))
        # carriageway geometry for later analyses (sight distance etc.)
        p_lon, p_lat = to_ll.transform(smp["x"], smp["y"])
        flags = [ways[chain_way[(a, b)]]["tags"] if (a, b) in chain_way.index else {}
                 for a, b in zip(p[:-1], p[1:])]
        chain_geo.append({
            "chain": ci, "ref": ch["ref"], "direction": ch["direction"], "kind": ch["kind"],
            "name": ch["name"], "x": np.round(smp["x"], 2).tolist(), "y": np.round(smp["y"], 2).tolist(),
            "lon": np.round(p_lon, 7).tolist(), "lat": np.round(p_lat, 7).tolist(),
            "s": np.round(smp["s"], 2).tolist(),
            "km": np.round(np.interp(smp["s"], smp["ss"], km), 3).tolist(),
            "tunnel": [1 if t.get("tunnel") not in (None, "no") else 0 for t in flags],
            "bridge": [1 if t.get("bridge") not in (None, "no") else 0 for t in flags],
            "maxspeed": [r_speed(t) for t in flags],
            # interchanges on this carriageway: [distance along chain (m), mileage, name]
            "junctions": [[round(float(smp["s"][i]), 1), jkm[n][0], jkm[n][1]]
                          for i, n in enumerate(p) if n in jkm],
            # entrance / exit ramps: [distance along chain (m), km on this chain, name, "in"/"out"]
            "ramps": ramp_list(p, smp["s"], np.interp(smp["s"], smp["ss"], km), eo, ei_, ways, jkm, nodes,
                               ch["kind"] == "branch")})
        for k in np.unique(seg):
            idx = np.flatnonzero(seg == k)
            if len(idx) < 2:
                continue
            iD = idx[np.argmax(Dm[idx])]
            iT = idx[np.argmax(Ts[idx])]
            jD = idx[np.argmax(HDm[idx])]
            jT = idx[np.argmax(HTs[idx])]
            # extend by one sample so consecutive segments join up on the map
            gidx = np.append(idx, idx[-1] + 1) if idx[-1] + 1 < len(km) else idx
            row = {
                "chain": ci, "ref": ch["ref"], "direction": ch["direction"], "kind": ch["kind"],
                "road_name": ch["name"], "km_source": ch["km_source"],
                "km_from": int(k), "km_to": int(k) + 1,
                "worst_dist_km_post": round(float(km[iD]), 1),
                "worst_dist_lon": round(float(lon_all[iD]), 6),
                "worst_dist_lat": round(float(lat_all[iD]), 6),
                "max_dist_km": round(float(Dm[iD]) / 1000, 2),
                "station_dist": st.name[sd[iD]] if sd[iD] >= 0 else None,
                "worst_time_km_post": round(float(km[iT]), 1),
                "max_time_min": round(float(Ts[iT]) / 60, 2),
                "station_time": st.name[stt[iT]] if stt[iT] >= 0 else None,
                "worst_hosp_km_post": round(float(km[jD]), 1),
                "worst_hosp_lon": round(float(lon_all[jD]), 6),
                "worst_hosp_lat": round(float(lat_all[jD]), 6),
                "max_hosp_dist_km": round(float(HDm[jD]) / 1000, 2),
                "hospital_dist": hp.name[hd[jD]] if hd[jD] >= 0 else None,
                "max_hosp_time_min": round(float(HTs[jT]) / 60, 2),
                "hospital_time": hp.name[ht[jT]] if ht[jT] >= 0 else None,
            }
            rows.append(row)
            feats.append({"type": "Feature", "properties": row, "geometry": {
                "type": "LineString",
                "coordinates": [[round(float(lon_all[i]), 6), round(float(lat_all[i]), 6)]
                                for i in gidx]}})
        print(f"chain {ci}: 國道{ch['ref']} {ch['direction']} {ch['kind']} {ch['name']} "
              f"{smp['L']/1000:.1f} km (km {km[0]:.1f}->{km[-1]:.1f}, {ch['km_source']})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "segments.csv", index=False)
    json.dump({"type": "FeatureCollection", "features": feats},
              open(OUT / "segments.geojson", "w"), ensure_ascii=False)
    pd.concat(samples).to_csv(OUT / "samples.csv", index=False)
    json.dump(chain_geo, open(OUT / "chains.json", "w"), ensure_ascii=False)
    st.to_csv(OUT / "stations_snapped.csv", index=False)
    hp.to_csv(OUT / "hospitals_snapped.csv", index=False)
    print(df.describe())


if __name__ == "__main__":
    main()
