"""National-freeway A1/A2 crashes (警政署「傷亡道路交通事故資料」) mapped onto the carriageway chains.

Inputs : data/raw/crash/{112,113,114}.zip   (downloaded here if missing; one row per party)
         output/points_100m.csv              (carriageway samples from 03/04)
Outputs: output/crashes_freeway.csv          one row per crash, with parsed freeway / direction / km
         output/crash_bins.csv               crashes per BIN_KM stretch of each carriageway, with rate
         output/crash.html                   interactive map page
         output/map_crash.png                static map

Only A1 (death within 24 h) and A2 (injury) crashes are published per case; A3 (property damage
only) is not, so the map understates all crashes. The data says which police unit handled the
crash, not which fire station sent the ambulance, so nothing about EMS is inferred here.
"""
import json
import re
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "WenQuanYi Zen Hei", "sans-serif"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "crash"
OUT = ROOT / "output"
COUNTY_TOPO = ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json"
LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"

# data.gov.tw datasets 167905 / 172969 / 177136 (政府資料開放授權條款第 1 版)
SOURCES = {
    112: "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/E68DFD97-92B3-4A78-A447-87F1390B54B0/resource/AD4D5B48-4954-403A-827F-B363DBD98C91/download",
    113: "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/CCAD7AA8-5139-4066-8BE3-D6CC3154C137/resource/0A260239-9958-4046-8834-E68E5EC38406/download",
    114: "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/FCD9C2D4-CB71-4EAA-AA4C-B088E5FE3157/resource/EF9B9192-7E4D-4E48-BF8A-AA3D4B298788/download",
}
YEARS = len(SOURCES)
# 優先改善 quadrants: default thresholds (≈ top quarter of each); the page lets readers move them
PRIORITY_RATE = 2.0        # A1+A2 crashes per km per year
PRIORITY_RESP_MIN = 8.0    # median ambulance response time over the stretch, minutes
BIN_KM = 5                 # stretch length the rate is computed over (see README for the choice)
MIN_TAIL_KM = 2.5          # a shorter last stretch is merged into the one before it
COLS = ["發生日期", "發生時間", "事故類別名稱", "處理單位名稱警局層", "發生地點",
        "死亡受傷人數", "經度", "緯度"]
DIRS = {"北": "北上", "南": "南下", "東": "東行", "西": "西行"}
NS = {"1", "3", "5"}       # north-south freeways; the rest run east-west
CN = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "八": "8", "十": "10"}
COLORS = ["#1f9e5a", "#d9b310", "#ec7a1c", "#d33a2c", "#7d3c98"]


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for y, url in SOURCES.items():
        f = RAW / f"{y}.zip"
        if not f.exists():
            subprocess.run(["curl", "-sSfL", "-m", "1200", "-o", str(f), url], check=True)


def load():
    rows = []
    for y in SOURCES:
        z = zipfile.ZipFile(RAW / f"{y}.zip")
        for n in [x for x in z.namelist() if "交通事故資料" in x]:
            for ch in pd.read_csv(z.open(n), usecols=COLS, dtype=str, chunksize=200_000, encoding="utf-8-sig"):
                rows.append(ch[ch["處理單位名稱警局層"].str.contains("國道公路警察局", na=False)].assign(year=y))
    df = pd.concat(rows, ignore_index=True)
    # one row per party -> one row per crash
    crash = df.drop_duplicates(["發生日期", "發生時間", "發生地點", "經度", "緯度"]).copy()
    print(f"{len(df)} party rows -> {len(crash)} crashes handled by 國道公路警察局")
    return crash


def parse_place(s):
    """'新北市林口區國道1號 38公里900.0公尺處南側向路肩' -> ('1', 38.9, '南下')."""
    t = s.replace("國道一號", "國道1號").replace("國道三號", "國道3號")
    t = re.sub(r"國([一二三四五六八十])(?=\d|\s|號|K|k)", lambda m: "國道" + CN[m.group(1)] + "號", t)
    m = re.search(r"國道\s*(\d+)\s*(甲)?\s*號?", t)
    if not m:
        return None, None, None
    ref = m.group(1) + ("甲" if m.group(2) else "")
    rest = t[m.end():]
    km = None
    k = re.search(r"([\d.]+)\s*(?:公里|[Kk])", rest)
    if k:
        km = float(k.group(1))
        mm = re.match(r"\s*([\d.]+)\s*公尺", rest[k.end():])
        if mm:
            km += float(mm.group(1)) / 1000
    else:
        mm = re.match(r"\s*([\d.]+)\s*公尺", rest)
        if mm:
            km = float(mm.group(1)) / 1000
    d = re.search(r"([北南東西])(?:側)?向", rest) or re.search(r"([北南東西])", rest)
    direction = DIRS[d.group(1)] if d else None
    if direction and (ref in NS) != (direction in ("北上", "南下")):
        direction = None          # e.g. 國道1號 ... 西側: no usable carriageway
    return ref, km, direction


def parse(crash):
    p = crash["發生地點"].fillna("").map(parse_place)
    crash["ref"] = [a for a, _, _ in p]
    crash["km_post"] = [b for _, b, _ in p]
    crash["direction"] = [c for _, _, c in p]
    s = crash["發生地點"].fillna("")
    crash["viaduct"] = s.str.contains("高架")
    crash["ramp"] = s.str.contains("交流道|匝道|出口|入口|轉接道")
    dt = crash["死亡受傷人數"].fillna("").str.extract(r"死亡(\d+);受傷(\d+)").astype(float)
    crash["deaths"], crash["injuries"] = dt[0], dt[1]
    crash["lon"] = pd.to_numeric(crash["經度"], errors="coerce")
    crash["lat"] = pd.to_numeric(crash["緯度"], errors="coerce")
    crash["date"] = crash["發生日期"]
    crash["time"] = crash["發生時間"].str.zfill(6)
    crash["class"] = crash["事故類別名稱"]
    crash["place"] = crash["發生地點"]
    return crash


VIADUCT_ROADS = ("高架",)    # road_name fragments that mark an elevated carriageway


def chains_of(points):
    """Per (ref, direction): each chain's km samples and whether each sample is on a viaduct."""
    out = {}
    for (ref, d, kind, ch), g in points.groupby(["ref", "direction", "kind", "chain"]):
        g = g.sort_values("km")
        out.setdefault((ref, d), []).append({
            "chain": ch, "kind": kind, "km": g.km.values,
            "viaduct": g.road_name.fillna("").str.contains("|".join(VIADUCT_ROADS)).values})
    return out


def assign(crash, points):
    """Pick the carriageway at the recorded km: an elevated one when the record says 高架, otherwise
    the ground-level one (preferring the main chain). Some chains switch between the two."""
    chains = chains_of(points)
    chain = []
    for r in crash.itertuples():
        best = None
        if r.km_post == r.km_post and r.km_post is not None:
            for x in chains.get((r.ref, r.direction), []):
                if not (x["km"][0] - 0.5 <= r.km_post <= x["km"][-1] + 0.5):
                    continue
                i = min(np.searchsorted(x["km"], r.km_post), len(x["km"]) - 1)
                score = (bool(x["viaduct"][i]) == bool(r.viaduct), x["kind"] == "main")
                if best is None or score > best[0]:
                    best = (score, x["chain"])
        chain.append(best[1] if best else None)
    crash["chain"] = pd.array(chain, dtype="Int64")
    return crash


def bins(points, crash):
    """Cut every chain into BIN_KM stretches of official km and count crashes in each."""
    rows = []
    for ch, g in points.groupby("chain"):
        g = g.sort_values("km")
        ref, d, kind = g.ref.iloc[0], g.direction.iloc[0], g.kind.iloc[0]
        kmin, kmax = g.km.min(), g.km.max()
        lo = np.floor(kmin / BIN_KM) * BIN_KM
        edges = list(np.arange(lo, kmax + BIN_KM, BIN_KM))
        edges[0] = kmin
        edges[-1] = kmax + 1e-6
        if len(edges) > 2 and edges[-1] - edges[-2] < MIN_TAIL_KM:
            del edges[-2]
        if len(edges) > 2 and edges[1] - edges[0] < MIN_TAIL_KM:
            del edges[1]
        c = crash[crash.chain == ch]
        for a, b in zip(edges[:-1], edges[1:]):
            pts = g[(g.km >= a) & (g.km < b)]
            if len(pts) < 2:
                continue
            cc = c[(c.km_post >= a) & (c.km_post < b)]
            length = len(pts) * 0.1
            road = pts.road_name.mode().iloc[0] if pts.road_name.notna().any() else ""
            rows.append({
                "chain": ch, "ref": ref, "direction": d, "kind": kind, "road_name": road,
                "km_from": round(a, 1), "km_to": round(min(b, kmax), 1), "length_km": round(length, 1),
                "crashes": len(cc), "a1": int((cc["class"] == "A1").sum()),
                "deaths": int(cc.deaths.sum()), "injuries": int(cc.injuries.sum()),
                "ramp_share": round(float(cc.ramp.mean()), 2) if len(cc) else 0.0,
                "rate": round(len(cc) / length / YEARS, 2),
                "county": pts.county.mode().iloc[0] if pts.county.notna().any() else "",
                # ambulance response over the stretch (minutes, from the routing model)
                "resp_med_min": round(float(pts.time_min.median()), 1),
                "resp_max_min": round(float(pts.time_min.max()), 1),
                "resp_station": pts.station_time.mode().iloc[0] if pts.station_time.notna().any() else "",
                "coords": pts[["lon", "lat"]].round(5).values.tolist(),
            })
    return pd.DataFrame(rows)


def colour_bins(rate):
    """Five classes at fixed, readable breaks (crashes per km per year)."""
    return [0.5, 1, 2, 4]


def static_map(b, counties, breaks):
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=150)
    for ax, grp in zip(axes, (("北上", "東行"), ("南下", "西行"))):
        counties.plot(ax=ax, color="#eef0ec", edgecolor="#9aa198", linewidth=0.5)
        sub = b[b.direction.isin(grp)].sort_values("rate")
        segs = [np.array(c) for c in sub.coords]
        cols = [COLORS[int(np.searchsorted(breaks, v, side="right"))] for v in sub.rate]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.6, capstyle="round"))
        ax.set_xlim(119.95, 122.05); ax.set_ylim(21.85, 25.35)
        ax.set_aspect(1 / np.cos(np.radians(23.7))); ax.set_axis_off()
        ax.set_title("／".join(grp), fontsize=15)
    labels = [f"< {breaks[0]}"] + [f"{a}–{b_}" for a, b_ in zip(breaks[:-1], breaks[1:])] + [f"≥ {breaks[-1]}"]
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, labels)], loc="lower center", ncol=5,
               title=f"每公里每年 A1＋A2 事故件數（{BIN_KM} 公里一段）", fontsize=11, title_fontsize=11, frameon=False)
    fig.suptitle(f"國道 A1＋A2 交通事故發生率（民國 {min(SOURCES)}–{max(SOURCES)} 年）", fontsize=18, y=0.97)
    fig.text(0.5, 0.025, "資料：內政部警政署傷亡道路交通事故資料（僅 A1、A2，不含 A3 財損事故）；未依車流量標準化。",
             ha="center", fontsize=9.5, color="#555")
    fig.savefig(OUT / "map_crash.png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def page(b, crash, counties, breaks, stats):
    css = subprocess.run(["curl", "-sSfL", LEAFLET_CSS], capture_output=True, text=True, check=True).stdout
    cty = counties.copy()
    cty["geometry"] = cty.geometry.simplify(0.003)
    data = {
        "breaks": breaks, "years": [min(SOURCES), max(SOURCES)], "bin_km": BIN_KM, "stats": stats,
        "priority": {"rate": PRIORITY_RATE, "resp": PRIORITY_RESP_MIN},
        "counties": json.loads(cty[["COUNTYNAME", "geometry"]].to_json()),
        "bins": [[r.ref, r.direction, r.kind == "main", r.road_name, r.km_from, r.km_to, r.length_km, r.crashes,
                  r.a1, r.deaths, r.injuries, r.rate, r.county, [v for xy in r.coords[::2] + [r.coords[-1]] for v in xy],
                  r.resp_med_min, r.resp_max_min, r.resp_station.split("消防局")[-1]]
                 for r in b.itertuples()],
    }
    tpl = (ROOT / "scripts" / "crash_template.html").read_text()
    html = (tpl.replace("/*__LEAFLET_CSS__*/", css)
               .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    (OUT / "crash.html").write_text(html)


def main():
    download()
    crash = parse(load())
    points = pd.read_csv(OUT / "points_100m.csv", dtype={"ref": str}, low_memory=False)
    crash = assign(crash, points)
    keep = ["year", "date", "time", "class", "deaths", "injuries", "ref", "direction", "km_post",
            "viaduct", "ramp", "chain", "lon", "lat", "place"]
    crash[keep].sort_values(["date", "time"]).to_csv(OUT / "crashes_freeway.csv", index=False, encoding="utf-8-sig")

    n = len(crash)
    placed = crash.chain.notna()
    print(f"placed on a carriageway: {placed.sum()} / {n} ({placed.mean():.1%})")
    print("not placed, by reason:",
          {"no 國道 ref": int(crash.ref.isna().sum()),
           "no km": int((crash.ref.notna() & crash.km_post.isna()).sum()),
           "no direction": int((crash.ref.notna() & crash.km_post.notna() & crash.direction.isna()).sum()),
           "other (e.g. 2甲, km out of range)": int((crash.ref.notna() & crash.km_post.notna()
                                                     & crash.direction.notna() & ~placed).sum())})
    b = bins(points, crash[placed])
    breaks = colour_bins(b.rate)
    b.drop(columns="coords").to_csv(OUT / "crash_bins.csv", index=False, encoding="utf-8-sig")
    counties = gpd.read_file(COUNTY_TOPO, layer="counties").set_crs(4326)
    static_map(b, counties, breaks)
    stats = {"crashes": int(n), "placed": int(placed.sum()), "a1": int((crash["class"] == "A1").sum()),
             "deaths": int(crash.deaths.sum()), "injuries": int(crash.injuries.sum()),
             "ramp_share": round(float(crash.ramp.mean()), 3), "median_rate": float(b.rate.median()),
             "per_year": {str(y): int((crash.year == y).sum()) for y in SOURCES}}
    print(stats)
    page(b, crash, counties, breaks, stats)
    pd.set_option("display.width", 250)
    print(b.drop(columns="coords").sort_values("rate", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
