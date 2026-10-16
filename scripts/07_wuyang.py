"""五楊・汐五高架：which fire station reaches each point of the elevated carriageways fastest,
and which is second fastest.

The main analysis (03_route.py) keeps only the winning station per point. Here every station
within reach of the viaducts gets its own single-source Dijkstra (travel time), so each 100-m
point has a ranked list. Cross-county response is allowed.

Outputs:
  data/processed/wuyang_points.csv    every 100-m point: best / second station and times
  output/wuyang_segments.csv          contiguous stretches with the same fastest station
  output/wuyang.html                  stand-alone page (two maps + tables)
  output/map_wuyang_north.png, output/map_wuyang_south.png
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import box
from scipy.sparse.csgraph import dijkstra

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
COUNTY_TOPO = ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json"
LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"

spec = importlib.util.spec_from_file_location("route", ROOT / "scripts" / "03_route.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

ROADS = ("五股楊梅高架道路", "汐止五股高架道路")
MIN_CHAIN_KM = 3.0          # skip short connector pieces
CANDIDATE_RADIUS_KM = 15.0  # stations considered (straight line to the nearest viaduct point)
# Reported practice for 五楊北上 (before 五股): 青埔 first, 隆恩 as 新北市's unit. Compared on the page.
RULE_KM_FROM = 50.7         # just past the 機場系統 entrance merge (50.8K), the first entrance north of 中壢
RULE_MARKS = [[70.9, "楊梅端"], [59.0, "中壢轉接道"], [50.8, "機場系統"], [36.0, "泰山轉接道"], [30.5, "五股"]]
RULE_STATIONS = ("桃園市政府消防局青埔分隊", "新北市政府消防局隆恩分隊")
TIME_LIMIT_S = 45 * 60
SNAP_SPEED_KMH = r.SNAP_SPEED_KMH
# categorical palette (dataviz reference, light / dark); assigned in order of appearance
PALETTE = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
           ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
           ("#6250d6", "#9085e9"), ("#e34948", "#e66767")]


def short(name):
    for pre in ("政府消防局", "消防局"):
        if pre in name:
            name = name.split(pre, 1)[1]
    for pre in ("第一救災救護大隊", "第二救災救護大隊", "第三救災救護大隊", "第四救災救護大隊",
                "第五救災救護大隊", "第六救災救護大隊", "第七救災救護大隊"):
        name = name.replace(pre, "")
    return name


def merge_colocated(st):
    """Units sharing a building (e.g. a 中隊 HQ and the 分隊 it houses) are one origin: keep the
    分隊 name and note the others, so the runner-up is never the same building."""
    out = []
    for _, g in st.groupby("snap_node"):
        g = g.sort_values("name", key=lambda c: c.str.contains("中隊"))
        row = g.iloc[0].copy()
        others = [short(n) for n in g.name.iloc[1:]]
        row["label"] = short(row["name"]) + (f"（與{'、'.join(others)}同址）" if others else "")
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def compute():
    pts = pd.read_csv(OUT / "points_100m.csv", dtype={"ref": str})
    viad = pts[pts.road_name.fillna("").isin(ROADS)]
    ways = r.load_ways()
    nodes, edges = r.build_graph(ways)
    keep = r.giant_scc(nodes, edges)
    edges = edges[keep[edges.u.values] & keep[edges.v.values]].reset_index(drop=True)
    n = len(nodes)
    chains, _ = r.build_chains(ways, nodes, edges, keep)
    ekmh = edges.groupby(["u", "v"]).kmh.max()

    # tail node / offset / speed for every viaduct point, by re-sampling its chain exactly as 03 did
    rows = []
    for ci in sorted(viad.chain.unique()):
        smp = r.sample_chain(chains[ci], nodes)
        if smp["L"] / 1000 < MIN_CHAIN_KM:
            continue
        p, ei = smp["p"], smp["ei"]
        sel = pts[(pts.chain == ci)].sort_values("s_m")
        assert len(sel) == len(smp["ss"]), "chain sampling differs from points_100m"
        kmh = np.array([ekmh.get((a, b), 100) for a, b in zip(p[ei], p[ei + 1])])
        df = sel.assign(tailnode=p[ei], off=smp["off"], kmh=kmh)
        rows.append(df[df.road_name.fillna("").isin(ROADS)])
    vp = pd.concat(rows, ignore_index=True)

    st = pd.read_csv(PROC / "stations_snapped.csv")
    lat0 = np.radians(vp.lat.mean())
    dx = (st.lon.values[:, None] - vp.lon.values[None, :]) * 111.32 * np.cos(lat0)
    dy = (st.lat.values[:, None] - vp.lat.values[None, :]) * 110.54
    near = (np.hypot(dx, dy).min(axis=1) <= CANDIDATE_RADIUS_KM) | st.name.isin(RULE_STATIONS)
    cand = merge_colocated(st[near])
    print(f"{len(vp)} viaduct points, {len(cand)} candidate stations (co-located units merged)")

    g = r.sparse(edges, "sec", n)
    T = np.full((len(cand), len(vp)), np.inf)
    for k, s in cand.iterrows():
        d = dijkstra(g, directed=True, indices=int(s.snap_node), limit=TIME_LIMIT_S)
        T[k] = d[vp.tailnode.values] + vp.off.values / (vp.kmh.values / 3.6) + s.snap_m / (SNAP_SPEED_KMH / 3.6)
    order = np.argsort(T, axis=0)
    b1, b2 = order[0], order[1]
    cols = np.arange(len(vp))
    label = dict(zip(cand.name, cand.label))
    vp["best"] = cand.name.values[b1]
    vp["best_min"] = np.round(T[b1, cols] / 60, 2)
    vp["second"] = cand.name.values[b2]
    vp["second_min"] = np.round(T[b2, cols] / 60, 2)
    vp["margin_min"] = np.round(vp.second_min - vp.best_min, 2)
    vp["best_label"] = vp.best.map(label)
    vp["second_label"] = vp.second.map(label)
    rank = np.argsort(order, axis=0) + 1          # 1 = fastest, per point
    for nm in RULE_STATIONS:
        k = int(np.flatnonzero(cand.name.values == nm)[0])
        tag = short(nm).replace("分隊", "")
        vp[f"t_{tag}"] = np.round(T[k] / 60, 2)
        vp[f"rank_{tag}"] = rank[k]
    ntp = np.flatnonzero(cand.name.str.startswith("新北市").values)
    b_ntp = ntp[np.argmin(T[ntp], axis=0)]
    vp["ntp_best_label"] = cand.label.values[b_ntp]
    vp["ntp_best_min"] = np.round(T[b_ntp, cols] / 60, 2)
    vp["n_cand"] = len(cand)
    agree = ((vp.best_min - vp.time_min).abs() <= 0.01).mean()
    print(f"fastest time agrees with the main analysis at {agree:.1%} of points")
    keepcols = ["chain", "direction", "road_name", "s_m", "km", "lon", "lat", "county",
                "best", "best_label", "best_min", "second", "second_label", "second_min", "margin_min",
                "station_dist", "dist_km", "t_青埔", "rank_青埔", "t_隆恩", "rank_隆恩",
                "ntp_best_label", "ntp_best_min", "n_cand"]
    vp[keepcols].to_csv(PROC / "wuyang_points.csv", index=False)
    return vp[keepcols]


def segments(vp, st):
    """Contiguous stretches (in travel order) with the same fastest station."""
    loc = dict(zip(st.name, zip(st.lon.round(5), st.lat.round(5))))
    vp = vp.assign(dist_differs=[loc[a] != loc[b] for a, b in zip(vp.station_dist, vp.best)])
    out = []
    for d, g in vp.groupby("direction"):
        g = g.sort_values(["chain", "s_m"]).reset_index(drop=True)
        key = g.best + "|" + g.chain.astype(str)
        blk = (key != key.shift()).cumsum()
        for _, s in g.groupby(blk):
            sec = s.second.value_counts()
            out.append({
                "direction": d, "road": "、".join(dict.fromkeys(s.road_name)),
                "km_start": round(float(s.km.iloc[0]), 1), "km_end": round(float(s.km.iloc[-1]), 1),
                "length_km": round(len(s) / 10, 1), "counties": "、".join(dict.fromkeys(s.county)),
                "best": s.best.iloc[0], "best_label": s.best_label.iloc[0], "best_min_range": f"{s.best_min.min():.1f}–{s.best_min.max():.1f}",
                "best_max_min": round(float(s.best_min.max()), 1),
                "second": sec.index[0],
                "second_label": s[s.second == sec.index[0]].second_label.iloc[0], "margin_min_min": round(float(s.margin_min.min()), 1),
                "margin_min_median": round(float(s.margin_min.median()), 1),
                "dist_best_differs": bool((s.dist_differs).mean() > 0.5),
                "dist_station": s.station_dist.mode().iloc[0],
            })
    return pd.DataFrame(out)


def colours(seg, direction):
    """Station -> palette slot, in order of first appearance along this direction."""
    order = list(dict.fromkeys(seg[seg.direction == direction].best))
    if len(order) > len(PALETTE):
        raise SystemExit(f"{len(order)} stations > {len(PALETTE)} palette slots")
    return {name: i for i, name in enumerate(order)}


def static_map(vp, seg, cmap, counties, st, direction, fname):
    g = vp[vp.direction == direction].sort_values(["chain", "s_m"])
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    x0, x1 = g.lon.min() - 0.08, g.lon.max() + 0.06
    y0, y1 = g.lat.min() - 0.05, g.lat.max() + 0.05
    counties.plot(ax=ax, color="#eef0ec", edgecolor="#7d857b", linewidth=1.2, linestyle="--")
    for (ci, best), s in g.groupby([g.chain, (g.best != g.best.shift()).cumsum()]):
        nm = s.best.iloc[0]
        ax.add_collection(LineCollection([np.c_[s.lon, s.lat]], colors=PALETTE[cmap[nm]][0],
                                         linewidths=6, capstyle="butt"))
    used = [n for n in cmap if n in set(g.best)]
    sts = st[st.name.isin(used)]
    for _, s in sts.iterrows():
        col = PALETTE[cmap[s["name"]]][0]
        ax.plot(s.lon, s.lat, "o", ms=11, mfc=col, mec="white", mew=2, zorder=5)
        ax.annotate(short(s["name"]), (s.lon, s.lat), xytext=(8, 6), textcoords="offset points",
                    fontsize=11, weight="bold", color="#1e2420",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2))
    x0 = min(x0, sts.lon.min() - 0.03); x1 = max(x1, sts.lon.max() + 0.08)
    y0 = min(y0, sts.lat.min() - 0.03); y1 = max(y1, sts.lat.max() + 0.03)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    frame = box(x0, y0, x1, y1)
    for _, row in counties.cx[x0:x1, y0:y1].iterrows():
        part = row.geometry.intersection(frame)
        if part.area > 0.02 * frame.area:   # label every county with a visible share of the frame
            c = part.representative_point()
            ax.text(c.x, c.y, row.COUNTYNAME.replace("台", "臺"), fontsize=14, color="#6b7369",
                    ha="center", weight="bold", alpha=0.85)
    ax.set_aspect(1 / np.cos(np.radians(25)))
    ax.set_axis_off()
    s_dir = seg[seg.direction == direction]
    ax.set_title(f"汐五・五楊高架 {direction}：反應時間最短的消防分隊", fontsize=17, pad=12)
    ax.text(0.5, -0.04, "可跨縣市救護；虛線為縣市界。依 OpenStreetMap 匝道資料計算（北上高架在機場系統到五股之間沒有入口，已查核），不塞車。",
             ha="center", fontsize=9.5, color="#555", transform=ax.transAxes)
    fig.savefig(OUT / fname, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def rule_data(vp):
    """Reported practice on 五楊北上 (青埔 first, 隆恩 for 新北市) against the model's fastest station."""
    g = vp[(vp.direction == "北上") & (vp.road_name == "五股楊梅高架道路")].sort_values("km", ascending=False)
    pts = [[round(r_.km, 1), r_.best_label, r_.best_min, r_.t_青埔, int(r_.rank_青埔), r_.t_隆恩, int(r_.rank_隆恩),
            r_.ntp_best_label, r_.ntp_best_min] for r_ in g.itertuples()]
    a = g[g.km <= RULE_KM_FROM]           # 機場系統 → 五股: the stretch the reported rule is about
    summary = {
        "km_from": RULE_KM_FROM, "km_to": round(float(a.km.min()), 1), "n_cand": int(vp.n_cand.iloc[0]),
        "qp_gap": [round(float((a.t_青埔 - a.best_min).min()), 1), round(float((a.t_青埔 - a.best_min).max()), 1)],
        "qp_rank": [int(a.rank_青埔.min()), int(a.rank_青埔.max())],
        "qp_min": [round(float(a.t_青埔.min()), 1), round(float(a.t_青埔.max()), 1)],
        "ln_gap": [round(float((a.t_隆恩 - a.t_青埔).min()), 1), round(float((a.t_隆恩 - a.t_青埔).max()), 1)],
        "ln_rank": [int(a.rank_隆恩.min()), int(a.rank_隆恩.max())],
        "ln_min": [round(float(a.t_隆恩.min()), 1), round(float(a.t_隆恩.max()), 1)],
        "ntp_best": a.ntp_best_label.mode().iloc[0],
        "ntp_gap": [round(float((a.t_隆恩 - a.ntp_best_min).min()), 1), round(float((a.t_隆恩 - a.ntp_best_min).max()), 1)],
        "best": a.best_label.mode().iloc[0],
    }
    return {"rule": pts, "rule_summary": summary, "rule_marks": RULE_MARKS}


def page(vp, seg, cmaps, counties, st):
    css = subprocess.run(["curl", "-sSfL", LEAFLET_CSS], capture_output=True, text=True, check=True).stdout
    bbox = (vp.lon.min() - 0.12, vp.lat.min() - 0.08, vp.lon.max() + 0.08, vp.lat.max() + 0.08)
    cty = counties.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
    cty["geometry"] = cty.geometry.simplify(0.0008)
    def stations_for(d):
        cm = cmaps[d]
        out = []
        for n in cm:
            row = st[st.name == n].iloc[0]
            lab = vp[vp.best == n].best_label.iloc[0]
            out.append({"name": n, "short": short(n), "label": lab, "slot": cm[n],
                        "lon": float(row.lon), "lat": float(row.lat)})
        return out
    data = {
        "palette": PALETTE,
        "stations": {d: stations_for(d) for d in ("北上", "南下")},
        "counties": json.loads(cty[["COUNTYNAME", "geometry"]].to_json()),
        "points": {d: [[round(r_.lon, 5), round(r_.lat, 5), cmaps[d][r_.best], round(r_.km, 1), r_.best_min,
                        r_.second_label, r_.second_min, r_.road_name[:2]]
                       for r_ in vp[vp.direction == d].sort_values(["chain", "s_m"]).itertuples()]
                   for d in ("北上", "南下")},
        "segments": json.loads(seg.to_json(orient="records", force_ascii=False)),
        **rule_data(vp),
    }
    tpl = (ROOT / "scripts" / "wuyang_template.html").read_text()
    html = (tpl.replace("/*__LEAFLET_CSS__*/", css)
               .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    (OUT / "wuyang.html").write_text(html)


def main():
    vp = compute()
    seg = segments(vp, pd.read_csv(ROOT / "data" / "fire_stations.csv"))
    seg.to_csv(OUT / "wuyang_segments.csv", index=False, encoding="utf-8-sig")
    cmaps = {d: colours(seg, d) for d in ("北上", "南下")}
    counties = gpd.read_file(COUNTY_TOPO, layer="counties").set_crs(4326)
    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    static_map(vp, seg, cmaps["北上"], counties, st, "北上", "map_wuyang_north.png")
    static_map(vp, seg, cmaps["南下"], counties, st, "南下", "map_wuyang_south.png")
    page(vp, seg, cmaps, counties, st)
    pd.set_option("display.width", 250)
    print(seg.drop(columns=["road"]).to_string())


if __name__ == "__main__":
    main()
