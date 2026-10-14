"""Make the deliverables from data/processed/segments.geojson:

  output/map_distance.png           response: fire station -> point, distance
  output/map_time.png               response: fire station -> point, time
  output/map_hospital_distance.png  transport: point -> heavy-level emergency hospital, distance
  output/map_hospital_time.png      transport: point -> heavy-level emergency hospital, time
  output/map_cross_county.png       segments whose nearest fire station is in another county
  output/cross_county_segments.csv  those segments, with incident county and station county
  output/segments.csv               every carriageway x 1-km segment
  output/farthest_points.csv        local maxima per metric (worst point of each stretch), ranked
  output/index.html            interactive map (Leaflet), self-contained apart from the Leaflet script
"""
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
COUNTY_TOPO = ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json"
LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"

# Fixed class breaks; green = close, purple = far. Transport runs longer, so its breaks are wider.
COLORS = ["#1f9e5a", "#d9b310", "#ec7a1c", "#d33a2c", "#7d3c98"]
# key: (value column, unit, name, nearest-facility column, worst-point lon/lat prefix, bins)
METRICS = {
    "distance": ("max_dist_km", "公里", "反應距離", "station_dist", "worst_dist", [5, 10, 15, 20]),
    "time": ("max_time_min", "分鐘", "反應時間", "station_time", "worst_dist", [5, 10, 15, 20]),
    "hospital_distance": ("max_hosp_dist_km", "公里", "送醫距離", "hospital_dist", "worst_hosp",
                          [10, 20, 30, 40]),
    "hospital_time": ("max_hosp_time_min", "分鐘", "送醫時間", "hospital_time", "worst_hosp",
                      [10, 15, 20, 30]),
}
TITLES = {
    "distance": "國道每公里路段：最近消防分隊救護車反應距離（取該公里段內最差點）",
    "time": "國道每公里路段：最近消防分隊救護車反應時間（取該公里段內最差點）",
    "hospital_distance": "國道每公里路段：送往最近重度級急救責任醫院的距離（取該公里段內最差點）",
    "hospital_time": "國道每公里路段：送往最近重度級急救責任醫院的時間（取該公里段內最差點）",
}
NOTES = {
    "response": "依車道方向計算，救護車只能由上游交流道進入；自由車流速度、不含出勤準備時間。"
                "資料：OpenStreetMap、消防署救援單位點位",
    "transport": "依車道方向計算，傷患須沿同方向車道開到下一個出口；自由車流速度、不含現場處置時間。"
                 "資料：OpenStreetMap、衛福部急救責任醫院分區名單（115/4/29）",
}
DIR_GROUP = {"北上": "北上／東行", "東行": "北上／東行", "南下": "南下／西行", "西行": "南下／西行"}


def bin_index(v, bins):
    return int(np.searchsorted(bins, v, side="right"))


def bin_labels(unit, bins):
    edges = [0] + bins
    labels = [f"{a}–{b} {unit}" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f"≥ {bins[-1]} {unit}")
    labels[0] = f"< {bins[0]} {unit}"
    return labels


def load():
    gj = json.load(open(PROC / "segments.geojson"))
    rows = [f["properties"] | {"coords": f["geometry"]["coordinates"]} for f in gj["features"]]
    seg = pd.DataFrame(rows)
    counties = gpd.read_file(COUNTY_TOPO, layer="counties").set_crs(4326)
    counties = counties.cx[119.9:122.2, 21.8:25.4]
    stations = pd.read_csv(PROC / "stations_snapped.csv")
    hospitals = pd.read_csv(PROC / "hospitals_snapped.csv")
    return seg, counties, stations, hospitals


CROSS_COLOR = "#c2410c"


def norm_county(name):
    return name.replace("台", "臺")


def cross_county(seg, counties):
    """Flag segments whose nearest fire station (by road distance) belongs to a different
    county than the one the incident is in. Decided per 100-m sample; a 1-km segment is
    flagged when more than half of its samples are served from another county."""
    smp = pd.read_csv(PROC / "samples.csv")
    pts = gpd.GeoDataFrame(smp, geometry=gpd.points_from_xy(smp.lon, smp.lat), crs=4326)
    poly = counties[["COUNTYNAME", "geometry"]]
    j = gpd.sjoin(pts, poly, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j.COUNTYNAME.isna()
    if miss.any():  # points on coastal bridges etc.: nearest county polygon
        near = gpd.sjoin_nearest(pts[miss].to_crs(3826), poly.to_crs(3826), how="left")
        j.loc[miss, "COUNTYNAME"] = near[~near.index.duplicated()].COUNTYNAME
    smp["county"] = j.COUNTYNAME.map(norm_county).values
    # Fire stations are named after the county fire bureau that runs them (臺中市政府消防局…).
    smp["station_county"] = smp.station_dist.str[:3]
    smp["station_county_time"] = smp.station_time.str[:3]
    smp["cross"] = smp.county != smp.station_county
    smp["cross_time"] = smp.county != smp.station_county_time

    def mode(x):
        return x.value_counts().index[0]
    agg = smp.groupby(["chain", "km_from"]).agg(
        county=("county", mode), station_county=("station_county", mode),
        cross_share=("cross", "mean"), cross_time_share=("cross_time", "mean"),
        cross_station=("station_dist", lambda x: mode(x)))
    agg["cross_county"] = agg.cross_share > 0.5
    agg["cross_county_time"] = agg.cross_time_share > 0.5
    # name the station that serves the cross-county samples
    cs = smp[smp.cross].groupby(["chain", "km_from"]).agg(
        cross_station=("station_dist", mode), cross_station_county=("station_county", mode))
    agg = agg.drop(columns="cross_station").join(cs)
    return seg.merge(agg.reset_index(), on=["chain", "km_from"], how="left")


def cross_runs(seg):
    """Contiguous flagged stretches per carriageway, for the ranking list."""
    runs = []
    for _, g in seg[seg.kind == "main"].groupby("chain"):
        sign = 1 if g.km_from.iloc[-1] >= g.km_from.iloc[0] else -1
        g = g.assign(_o=g.km_from * sign).sort_values("_o")
        key = g.cross_county.astype(str) + g.county + g.cross_station_county.fillna("")
        grp = (key != key.shift()).cumsum()
        for _, r in g[g.cross_county].groupby(grp[g.cross_county]):
            runs.append({"ref": r.ref.iloc[0], "direction": r.direction.iloc[0],
                         "km_start": int(r.km_from.iloc[0]), "km_end": int(r.km_from.iloc[-1]),
                         "length_km": len(r), "county": r.county.mode().iloc[0],
                         "station": r.cross_station.mode().iloc[0],
                         "station_county": r.cross_station_county.mode().iloc[0],
                         "lon": r.worst_dist_lon.iloc[len(r) // 2],
                         "lat": r.worst_dist_lat.iloc[len(r) // 2]})
    return pd.DataFrame(runs).sort_values("length_km", ascending=False)


def static_cross_map(seg, counties, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), dpi=150)
    for ax, grp in zip(axes, ["北上／東行", "南下／西行"]):
        counties.plot(ax=ax, color="#eef0ec", edgecolor="#b9beb6", linewidth=0.5)
        sub = seg[seg.direction.map(DIR_GROUP) == grp]
        for flag, color, lw in [(False, "#c9cdc6", 2.0), (True, CROSS_COLOR, 3.0)]:
            part = sub[sub.cross_county == flag]
            ax.add_collection(LineCollection([np.array(c) for c in part.coords], colors=color,
                                             linewidths=lw, capstyle="round"))
        main = sub[sub.kind == "main"]
        ax.set_xlim(119.95, 122.05)
        ax.set_ylim(21.85, 25.35)
        ax.set_aspect(1 / np.cos(np.radians(23.7)))
        ax.set_axis_off()
        ax.set_title(f"{grp}　{int(main.cross_county.sum())} / {len(main)} 個主線公里段", fontsize=15)
    handles = [Patch(color=CROSS_COLOR, label="跨區救護較近"),
               Patch(color="#c9cdc6", label="同縣市救護較近")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("國道事故點最近的消防分隊（以行車距離計）來自鄰近縣市的路段", fontsize=17)
    fig.text(0.5, 0.045, "依車道方向計算；一公里段內過半數的點由外縣市分隊最近即標示。"
             "縣市界：taiwan-atlas；分隊所屬縣市依消防局名稱", ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.02)
    fig.savefig(OUT / "map_cross_county.png")
    plt.close(fig)


def find_peaks(seg, col):
    """A segment is a local peak if it is worse than both neighbours along the carriageway:
    for response that is the end of a stretch with no upstream on-ramp, for transport the
    start of a stretch with no downstream exit."""
    out = []
    for _, g in seg.groupby("chain"):
        sign = 1 if g.km_from.iloc[-1] >= g.km_from.iloc[0] else -1
        g = g.assign(_o=g.km_from * sign).sort_values("_o")
        v = g[col].values
        nxt = np.append(v[1:], -np.inf)
        prv = np.insert(v[:-1], 0, -np.inf)
        out.append(g[(v > nxt) & (v >= prv)])
    return pd.concat(out).drop(columns="_o")


def static_map(seg, counties, metric):
    col, unit, title, _, wp, bins = METRICS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), dpi=150)
    for ax, grp in zip(axes, ["北上／東行", "南下／西行"]):
        counties.plot(ax=ax, color="#eef0ec", edgecolor="#b9beb6", linewidth=0.5)
        sub = seg[seg.direction.map(DIR_GROUP) == grp].sort_values(col)
        lines = [np.array(c) for c in sub.coords]
        cols = [COLORS[bin_index(v, bins)] for v in sub[col]]
        ax.add_collection(LineCollection(lines, colors=cols, linewidths=2.6, capstyle="round"))
        ax.set_xlim(119.95, 122.05)
        ax.set_ylim(21.85, 25.35)
        ax.set_aspect(1 / np.cos(np.radians(23.7)))
        ax.set_axis_off()
        ax.set_title(grp, fontsize=15)
        top = find_peaks(sub, col).sort_values(col, ascending=False).head(3)
        for rank, w in enumerate(top.itertuples(), 1):
            xy = (getattr(w, wp + "_lon"), getattr(w, wp + "_lat"))
            right = xy[0] < 121.0
            ax.annotate(f"{rank}. 國道{w.ref} {w.direction} {w.km_from}K　{getattr(w, col):.1f} {unit}",
                        xy=xy, xytext=(-14 if not right else 14, 10 - 14 * rank), textcoords="offset points",
                        ha="right" if not right else "left", va="center", fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.5, alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="#444", lw=0.6))
            ax.plot(*xy, "o", ms=5, mfc="none", mec="#222", mew=1)
    handles = [Patch(color=c, label=l) for c, l in zip(COLORS, bin_labels(unit, bins))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle(TITLES[metric], fontsize=17)
    fig.text(0.5, 0.045, NOTES["transport" if metric.startswith("hospital") else "response"],
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.02)
    fig.savefig(OUT / f"map_{metric}.png")
    plt.close(fig)


def web_data(seg, counties, stations, hospitals):
    st_index = {n: i for i, n in enumerate(stations.name)}
    hp_index = {n: i for i, n in enumerate(hospitals.name)}
    segs = []
    for r in seg.itertuples():
        segs.append([r.ref, r.direction, 1 if r.kind == "main" else 0, r.road_name, r.km_from,
                     r.max_dist_km, r.max_time_min,
                     st_index.get(r.station_dist, -1), st_index.get(r.station_time, -1),
                     r.max_hosp_dist_km, r.max_hosp_time_min,
                     hp_index.get(r.hospital_dist, -1), hp_index.get(r.hospital_time, -1),
                     [round(v, 5) for xy in r.coords for v in xy]])
    simp = counties.copy()
    simp["geometry"] = simp.geometry.simplify(0.003)
    county_gj = json.loads(simp[["COUNTYNAME", "geometry"]].to_json())
    for f in county_gj["features"]:
        f["geometry"]["coordinates"] = _round(f["geometry"]["coordinates"])
    pts = lambda df: [[n, round(lo, 5), round(la, 5)] for n, lo, la in zip(df.name, df.lon, df.lat)]
    return {"segments": segs, "counties": county_gj, "stations": pts(stations),
            "hospitals": pts(hospitals)}


def _round(c):
    if isinstance(c, (int, float)):
        return round(c, 4)
    return [_round(x) for x in c]


def main():
    seg, counties, stations, hospitals = load()
    for m in METRICS:
        static_map(seg, counties, m)
    seg = cross_county(seg, counties)
    runs = cross_runs(seg)
    static_cross_map(seg, counties, runs)
    seg.drop(columns="coords").to_csv(OUT / "segments.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(OUT / "cross_county_segments.csv", index=False, encoding="utf-8-sig")

    peaks = []
    for m, (col, unit, title, fcol, wp, _) in METRICS.items():
        p = find_peaks(seg, col).sort_values(col, ascending=False)
        p = p.assign(metric=title, value=p[col], unit=unit, nearest=p[fcol],
                     lon=p[wp + "_lon"], lat=p[wp + "_lat"])
        peaks.append(p[["metric", "ref", "direction", "kind", "road_name", "km_from", "km_to",
                        "value", "unit", "nearest", "lon", "lat"]])
    pk = pd.concat(peaks)
    pk.to_csv(OUT / "farthest_points.csv", index=False, encoding="utf-8-sig")

    css = subprocess.run(["curl", "-sSfL", LEAFLET_CSS], capture_output=True, text=True, check=True).stdout
    data = web_data(seg, counties, stations, hospitals)
    data["metrics"] = {m: {"name": v[2], "unit": v[1], "bins": v[5]} for m, v in METRICS.items()}
    data["peaks"] = {
        m: [[r.ref, r.direction, r.road_name, r.km_from, r.value, r.nearest, r.lon, r.lat]
            for r in pk[pk.metric == METRICS[m][2]].head(20).itertuples()]
        for m in METRICS}
    data["cross"] = [1 if c else 0 for c in seg.cross_county]
    data["cross_runs"] = [[r.ref, r.direction, r.km_start, r.km_end, r.length_km, r.county,
                           r.station, r.station_county, r.lon, r.lat] for r in runs.itertuples()]
    main_seg = seg[seg.kind == "main"]
    data["summary"] = {
        "km_main": int(len(main_seg)),
        "stations": int(len(stations)),
        "hospitals": int(len(hospitals)),
        "median": {m: float(main_seg[v[0]].median()) for m, v in METRICS.items()},
        "cross_main": int(main_seg.cross_county.sum()),
        "cross_main_time": int(main_seg.cross_county_time.sum()),
    }
    tpl = (ROOT / "scripts" / "map_template.html").read_text()
    html = (tpl.replace("/*__LEAFLET_CSS__*/", css)
               .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    (OUT / "index.html").write_text(html)
    print(pk.groupby("metric").head(10).to_string())
    print(data["summary"])
    print(runs.head(25).to_string())


if __name__ == "__main__":
    main()
