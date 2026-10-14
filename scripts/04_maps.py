"""Make the deliverables from data/processed/segments.geojson:

  output/map_distance.png      static map, coloured by ambulance travel distance
  output/map_time.png          static map, coloured by ambulance travel time
  output/segments.csv          every carriageway x 1-km segment
  output/farthest_points.csv   local maxima (end of each stretch with no upstream on-ramp), ranked
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

# Fixed class breaks agreed for the project; green = close, purple = far.
BINS = [5, 10, 15, 20]
COLORS = ["#1f9e5a", "#d9b310", "#ec7a1c", "#d33a2c", "#7d3c98"]
METRICS = {
    "distance": ("max_dist_km", "公里", "反應距離", "station_dist"),
    "time": ("max_time_min", "分鐘", "反應時間", "station_time"),
}
DIR_GROUP = {"北上": "北上／東行", "東行": "北上／東行", "南下": "南下／西行", "西行": "南下／西行"}


def bin_index(v):
    return int(np.searchsorted(BINS, v, side="right"))


def bin_labels(unit):
    edges = [0] + BINS
    labels = [f"{a}–{b} {unit}" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f"≥ {BINS[-1]} {unit}")
    labels[0] = f"< {BINS[0]} {unit}"
    return labels


def load():
    gj = json.load(open(PROC / "segments.geojson"))
    rows = [f["properties"] | {"coords": f["geometry"]["coordinates"]} for f in gj["features"]]
    seg = pd.DataFrame(rows)
    counties = gpd.read_file(COUNTY_TOPO, layer="counties").set_crs(4326)
    counties = counties.cx[119.9:122.2, 21.8:25.4]
    stations = pd.read_csv(PROC / "stations_snapped.csv")
    return seg, counties, stations


def find_peaks(seg, col):
    """A segment is a local peak if it is worse than the next one in the direction of travel:
    that is where a stretch with no upstream on-ramp ends."""
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
    col, unit, title, _ = METRICS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), dpi=150)
    for ax, grp in zip(axes, ["北上／東行", "南下／西行"]):
        counties.plot(ax=ax, color="#eef0ec", edgecolor="#b9beb6", linewidth=0.5)
        sub = seg[seg.direction.map(DIR_GROUP) == grp].sort_values(col)
        lines = [np.array(c) for c in sub.coords]
        cols = [COLORS[bin_index(v)] for v in sub[col]]
        ax.add_collection(LineCollection(lines, colors=cols, linewidths=2.6, capstyle="round"))
        ax.set_xlim(119.95, 122.05)
        ax.set_ylim(21.85, 25.35)
        ax.set_aspect(1 / np.cos(np.radians(23.7)))
        ax.set_axis_off()
        ax.set_title(grp, fontsize=15)
        top = find_peaks(sub, col).sort_values(col, ascending=False).head(3)
        for rank, w in enumerate(top.itertuples(), 1):
            xy = w.coords[-1]
            right = xy[0] < 121.0
            ax.annotate(f"{rank}. 國道{w.ref} {w.direction} {w.km_from}K　{getattr(w, col):.1f} {unit}",
                        xy=xy, xytext=(-14 if not right else 14, 10 - 14 * rank), textcoords="offset points",
                        ha="right" if not right else "left", va="center", fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.5, alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="#444", lw=0.6))
            ax.plot(*xy, "o", ms=5, mfc="none", mec="#222", mew=1)
    handles = [Patch(color=c, label=l) for c, l in zip(COLORS, bin_labels(unit))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle(f"國道每公里路段：最近消防分隊救護車{title}（取該公里段內最差點）", fontsize=17)
    fig.text(0.5, 0.045, "依車道方向計算，救護車只能由上游交流道進入；自由車流速度、不含出勤準備時間。"
             "資料：OpenStreetMap、消防署救援單位點位", ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.02)
    fig.savefig(OUT / f"map_{metric}.png")
    plt.close(fig)


def web_data(seg, counties, stations):
    st_index = {n: i for i, n in enumerate(stations.name)}
    segs = []
    for r in seg.itertuples():
        segs.append([r.ref, r.direction, 1 if r.kind == "main" else 0, r.road_name, r.km_from,
                     r.max_dist_km, r.max_time_min,
                     st_index.get(r.station_dist, -1), st_index.get(r.station_time, -1),
                     [round(v, 5) for xy in r.coords for v in xy]])
    simp = counties.copy()
    simp["geometry"] = simp.geometry.simplify(0.003)
    county_gj = json.loads(simp[["COUNTYNAME", "geometry"]].to_json())
    for f in county_gj["features"]:
        f["geometry"]["coordinates"] = _round(f["geometry"]["coordinates"])
    sts = [[n, round(lo, 5), round(la, 5)] for n, lo, la in zip(stations.name, stations.lon, stations.lat)]
    return {"segments": segs, "counties": county_gj, "stations": sts}


def _round(c):
    if isinstance(c, (int, float)):
        return round(c, 4)
    return [_round(x) for x in c]


def main():
    seg, counties, stations = load()
    for m in METRICS:
        static_map(seg, counties, m)
    seg.drop(columns="coords").to_csv(OUT / "segments.csv", index=False, encoding="utf-8-sig")

    peaks = []
    for m, (col, unit, title, stcol) in METRICS.items():
        p = find_peaks(seg, col).sort_values(col, ascending=False)
        p = p.assign(metric=title, value=p[col], unit=unit, station=p[stcol],
                     lon=p.worst_dist_lon, lat=p.worst_dist_lat)
        peaks.append(p[["metric", "ref", "direction", "kind", "road_name", "km_from", "km_to",
                        "value", "unit", "station", "lon", "lat"]])
    pk = pd.concat(peaks)
    pk.to_csv(OUT / "farthest_points.csv", index=False, encoding="utf-8-sig")

    css = subprocess.run(["curl", "-sSfL", LEAFLET_CSS], capture_output=True, text=True, check=True).stdout
    data = web_data(seg, counties, stations)
    data["peaks"] = {
        m: [[r.ref, r.direction, r.road_name, r.km_from, r.value, r.station, r.lon, r.lat]
            for r in pk[pk.metric == METRICS[m][2]].head(20).itertuples()]
        for m in METRICS}
    data["summary"] = {
        "segments": int(len(seg)),
        "km_main": int((seg.kind == "main").sum()),
        "stations": int(len(stations)),
        "median_dist": float(seg.max_dist_km.median()),
        "median_time": float(seg.max_time_min.median()),
        "share_over_10km": float((seg.max_dist_km >= 10).mean()),
        "share_over_10min": float((seg.max_time_min >= 10).mean()),
    }
    tpl = (ROOT / "scripts" / "map_template.html").read_text()
    html = (tpl.replace("/*__LEAFLET_CSS__*/", css)
               .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    (OUT / "index.html").write_text(html)
    print(pk.groupby("metric").head(10).to_string())
    print(data["summary"])


if __name__ == "__main__":
    main()
