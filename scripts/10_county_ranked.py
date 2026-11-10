"""臺北市、新北市、桃園市境內國道：每個點位到場最快的前三個消防分隊，以及由外縣市分隊較快的路段。

The main analysis (03_route.py) keeps only the winning station per point. Here every station
near these counties gets its own single-source Dijkstra (travel time), so each 100-m point
inside them has a ranked list. Same model as 07_wuyang.py: free-flow speeds, no turnout time.

Run after 03_route.py and 04_maps.py (needs data/processed/stations_snapped.csv and
output/points_100m.csv built from the same OSM extract).

Outputs:
  output/<county>_ranked_points.csv   every 100-m point in the county: three fastest stations and times
  output/<county>_cross_county.csv    contiguous stretches where a station of another county is fastest
with <county> = taipei, newtaipei, taoyuan.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r = load("route", "03_route.py")
w = load("wuyang", "07_wuyang.py")

COUNTIES = {"臺北市": "taipei", "新北市": "newtaipei", "桃園市": "taoyuan"}
CANDIDATE_RADIUS_KM = 25.0  # stations considered (straight line to the nearest point)
TIME_LIMIT_S = 45 * 60
TOP = 3
ELEVATED = ("汐止五股高架道路", "五股楊梅高架道路")


def compute():
    pts = pd.read_csv(OUT / "points_100m.csv", dtype={"ref": str})
    ways = r.load_ways()
    nodes, edges = r.build_graph(ways)
    keep = r.giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    n = len(nodes)
    chains, _ = r.build_chains(ways, nodes, edges, keep)
    ekmh = edges.groupby(["u", "v"]).kmh.max()

    # tail node / offset / speed for every point, by re-sampling its chain exactly as 03 did
    rows = []
    for ci in sorted(pts[pts.county.isin(COUNTIES)].chain.unique()):
        smp = r.sample_chain(chains[ci], nodes)
        p, ei = smp["p"], smp["ei"]
        sel = pts[pts.chain == ci].sort_values("s_m")
        assert len(sel) == len(smp["ss"]), "chain sampling differs from points_100m"
        kmh = np.array([ekmh.get((a, b), 100) for a, b in zip(p[ei], p[ei + 1])])
        df = sel.assign(tailnode=p[ei], off=smp["off"], kmh=kmh)
        rows.append(df[df.county.isin(COUNTIES)])
    tp = pd.concat(rows, ignore_index=True)

    st = pd.read_csv(PROC / "stations_snapped.csv")
    lat0 = np.radians(tp.lat.mean())
    dx = (st.lon.values[:, None] - tp.lon.values[None, :]) * 111.32 * np.cos(lat0)
    dy = (st.lat.values[:, None] - tp.lat.values[None, :]) * 110.54
    cand = w.merge_colocated(st[np.hypot(dx, dy).min(axis=1) <= CANDIDATE_RADIUS_KM])
    print(f"{len(tp)} points in {'、'.join(COUNTIES)}, {len(cand)} candidate stations (co-located units merged)")

    g = r.sparse(edges, "sec", n)
    T = np.full((len(cand), len(tp)), np.inf)
    for k, s in cand.iterrows():
        d = dijkstra(g, directed=True, indices=int(s.snap_node), limit=TIME_LIMIT_S)
        T[k] = d[tp.tailnode.values] + tp.off.values / (tp.kmh.values / 3.6) + s.snap_m / (r.SNAP_SPEED_KMH / 3.6)
    order = np.argsort(T, axis=0)
    cols = np.arange(len(tp))
    for i in range(TOP):
        k = order[i]
        tp[f"s{i+1}"] = cand.name.values[k]
        tp[f"s{i+1}_label"] = cand.label.values[k]
        tp[f"s{i+1}_county"] = cand.name.str[:3].values[k]
        tp[f"t{i+1}_min"] = np.round(T[k, cols] / 60, 2)
    # fastest station of the incident's own county, for the time saved by crossing the boundary
    k = np.zeros(len(tp), int)
    for county in COUNTIES:
        own = np.flatnonzero(cand.name.str.startswith(county).values)
        m = (tp.county == county).values
        k[m] = own[np.argmin(T[own][:, m], axis=0)]
    tp["own_label"] = cand.label.values[k]
    tp["own_min"] = np.round(T[k, cols] / 60, 2)
    tp["own_rank"] = (T < T[k, cols]).sum(axis=0) + 1
    agree = ((tp.t1_min - tp.time_min).abs() <= 0.01).mean()
    print(f"fastest time agrees with the main analysis at {agree:.1%} of points")
    tp["elevated"] = tp.road_name.fillna("").isin(ELEVATED)
    keepcols = (["chain", "ref", "direction", "kind", "elevated", "road_name", "s_m", "km", "lon", "lat", "county"]
                + [c for i in range(1, TOP + 1) for c in (f"s{i}", f"s{i}_label", f"s{i}_county", f"t{i}_min")]
                + ["own_label", "own_min", "own_rank"])
    return tp[keepcols]


def rng(a):
    return f"{a.min():.1f}–{a.max():.1f}"


def segments(tp, county):
    """Contiguous stretches (in travel order) where the same out-of-county station is fastest and the
    runners-up do not change."""
    out = []
    for (ci,), g in tp.groupby(["chain"]):
        g = g.sort_values("s_m").reset_index(drop=True)
        gap = g.s_m.diff().fillna(0) > 150         # the carriageway leaves the county and comes back
        key = g.s1 + "|" + g.s2 + "|" + g.s3 + "|" + g.elevated.astype(str)
        blk = ((key != key.shift()) | gap).cumsum()
        for _, s in g.groupby(blk):
            if s.s1_county.iloc[0] == county:
                continue
            row = {"ref": s.ref.iloc[0], "direction": s.direction.iloc[0],
                   "type": "高架" if s.elevated.iloc[0] else "平面",
                   "km_start": round(float(s.km.iloc[0]), 1), "km_end": round(float(s.km.iloc[-1]), 1),
                   "length_km": round(len(s) / 10, 1)}
            for i in range(1, TOP + 1):
                row[f"rank{i}"] = s[f"s{i}_county"].iloc[0] + s[f"s{i}_label"].iloc[0]
                row[f"rank{i}_min"] = rng(s[f"t{i}_min"])
            own = s.own_label.value_counts().index[0]
            row["own_best"] = own
            row["own_best_min"] = rng(s.own_min)
            row["own_best_rank"] = rng(s.own_rank).replace(".0", "")
            row["saved_min"] = rng(s.own_min - s.t1_min)
            out.append(row)
    return pd.DataFrame(out)


def main():
    allp = compute()
    pd.set_option("display.width", 300, "display.max_rows", 300, "display.max_columns", 30)
    for county, slug in COUNTIES.items():
        tp = allp[allp.county == county]
        tp.to_csv(OUT / f"{slug}_ranked_points.csv", index=False, encoding="utf-8-sig")
        seg = segments(tp, county)
        seg.to_csv(OUT / f"{slug}_cross_county.csv", index=False, encoding="utf-8-sig")
        cross = tp.s1_county != county
        print(f"\n{county}: {cross.sum()} of {len(tp)} points ({cross.mean():.0%}) are reached fastest from another county")
        print(seg.to_string())


if __name__ == "__main__":
    main()
