"""National-freeway A1/A2 crashes (警政署「傷亡道路交通事故資料」) mapped onto the carriageway chains.

Inputs : data/raw/crash/{112,113,114}.zip   (downloaded here if missing; one row per party)
         data/raw/freeway/                   高公局 daily traffic per interchange section and the
                                             interchange mileage tables (downloaded here if missing)
         output/points_100m.csv              (carriageway samples from 03/04)
Outputs: output/crashes_freeway.csv          one row per crash, with parsed freeway / direction / km
         output/crash_bins.csv               crashes per BIN_KM stretch of each carriageway, with rate
                                             per km and per 100 million vehicle-km
         output/traffic_sections.csv         daily traffic per interchange section, with mileage
         output/crash.html                   interactive map page
         output/map_crash.png                static map

Only A1 (death within 24 h) and A2 (injury) crashes are published per case; A3 (property damage
only) is not, so the map understates all crashes. The data says which police unit handled the
crash, not which fire station sent the ambulance, so nothing about EMS is inferred here.
"""
import io
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
DAYS = {112: 365, 113: 366, 114: 365}     # 2023, 2024 (leap), 2025
VKM_OFFICIAL_113 = 35302   # million vehicle-km on all national freeways in 2024 (高公局 2024 traffic statistics)
# data.gov.tw dataset 21115「交通量參考值」(高速公路局): weekday / Saturday / Sunday daily traffic of
# every section between interchanges, per direction, from the ETC gantries in May of each year
TRAFFIC = ROOT / "data" / "raw" / "freeway"
TRAFFIC_SOURCES = {y: f"https://www.freeway.gov.tw/Download_File.ashx?id={i}&FileConditionsID=1"
                   for y, i in {112: 19145, 113: 20597, 114: 23650}.items()}
# 高公局「交流道、服務區里程一覽表」, one page per freeway: page id -> refs of its tables in order
IC_PAGES = {4617: ["1", "1高架"], 4622: ["3", "3甲"], 4621: ["5"], 4618: ["2", "2甲"], 4620: ["4"],
            4619: ["6"], 4623: ["8"], 4624: ["10"]}
IC_URL = "https://www.freeway.gov.tw/Publish.aspx?cnid=1906&p={}"
# section end points missing from the mileage tables (km on our mileage scale)
IC_EXTRA = {
    ("1", "高公局"): 34.2,          # 高公局聯絡道, 北上 34.1K / 南下 34.3K (see 03_route.py)
    ("1", "中正路"): 368.0,         # 高雄交流道's 中正路 ramps, between 高雄 (367) and 瑞隆路 (369)
    ("1", "漁港路"): 373.0,         # last ramps before the end of 國1
    ("1", "高雄端"): 373.8,         # end of our 國1 chains (the table rounds it to 373)
    ("5", "坪林交控專用道"): 14.0,   # 坪林交控中心專用道 in the mileage table
}
# 優先改善 quadrants: default thresholds (≈ top quarter of each); the page lets readers move them
PRIORITY_RATE_VKM = 12.0   # A1+A2 crashes per 100 million vehicle-km (the page's default rate)
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
    TRAFFIC.mkdir(parents=True, exist_ok=True)
    # www.freeway.gov.tw sometimes resets the connection, hence the retries
    get = ["curl", "-sSfL", "-m", "120", "--retry", "6", "--retry-all-errors", "--retry-delay", "3", "-o"]
    for y, url in TRAFFIC_SOURCES.items():
        f = TRAFFIC / f"traffic_{y}_main.csv"
        if not f.exists():
            subprocess.run(get + [str(f), url], check=True)
    for p in IC_PAGES:
        f = TRAFFIC / f"ic_{p}.html"
        if not f.exists():
            subprocess.run(get + [str(f), IC_URL.format(p)], check=True)


def ic_name(s):
    """'霧峰交流道  (100.12.31起 開放通往台74 )' -> '霧峰', '汐止系統交流道' -> '汐止系統'.
    轉接道 keeps its suffix: 五股轉接道 (32K) and 五股交流道 (33K) are different places."""
    s = re.sub(r"\s+|\(.*?\)|（.*?）", "", str(s))
    return re.sub(r"(交流道|出口匝道)$", "", s).removeprefix("高架道路")


def ic_mileage():
    """{(ref, name): km} from the 高公局 mileage tables; ref '1高架' is the 汐止–楊梅 viaduct."""
    km = {}
    for p, refs in IC_PAGES.items():
        h = (TRAFFIC / f"ic_{p}.html").read_text(encoding="utf-8")
        h = h[h.find("FCKdetail"):]
        tables = pd.read_html(io.StringIO(h[:h.find("date_font_01")]))
        for ref, t in zip(refs, tables):
            if "設施名稱" not in t.columns:            # header in the first row
                t = t.iloc[1:].set_axis(t.iloc[0], axis=1)
            for name, k in zip(t["設施名稱"], t["里程"].astype(str)):
                m = re.search(r"[\d.]+", k)
                if m and "合計" not in str(name):
                    km.setdefault((ref, ic_name(name)), float(m.group()))
    km.update(IC_EXTRA)
    return km


def section_end(ref, name, km):
    """Mileage of a traffic-section end point such as '汐止&汐止系統', '林口(文化一路)', '高架楊梅端'."""
    ks = []
    for part in name.split("&"):
        part = re.sub(r"\(.*?\)", "", part)
        for cand in (part, part.removeprefix("高架"), ic_name(part), ic_name(part).removesuffix("北上")):
            if (ref, cand) in km:
                ks.append(km[(ref, cand)])
                break
        else:
            raise KeyError(f"no mileage for 國{ref} {name}")
    return float(np.mean(ks))


def traffic_sections():
    """Daily traffic of every interchange section and direction, averaged over the crash years.

    Each year's file gives the traffic of a Tuesday–Thursday, a Saturday and a Sunday in May; a
    week is 5 weekdays + Saturday + Sunday. The three years are weighted by their number of days."""
    km = ic_mileage()
    rows = []
    for y in TRAFFIC_SOURCES:
        t = pd.read_csv(TRAFFIC / f"traffic_{y}_main.csv", encoding="cp950")
        for r in t.itertuples(index=False):
            m = re.match(r"國(\d+甲?)(高架)?([東西南北])向路段", r[0])
            ref, viaduct, d = m.group(1), bool(m.group(2)), DIRS[m.group(3)]
            if ref == "2甲":                           # not one of our carriageways
                continue
            a, b = r[1].split("-")
            tab = ref + ("高架" if viaduct else "")
            ka, kb = section_end(tab, a, km), section_end(tab, b, km)
            rows.append({"year": y, "ref": ref, "direction": d, "viaduct": viaduct, "section": r[1],
                         "km_from": min(ka, kb), "km_to": max(ka, kb),
                         "weekday": r[6], "sat": r[4], "sun": r[5]})
    t = pd.DataFrame(rows)
    t["adt"] = (5 * t.weekday + t.sat + t.sun) / 7
    t["days"] = t.year.map(DAYS)
    keys = ["ref", "direction", "viaduct", "section", "km_from", "km_to"]
    s = (t.assign(w=t.adt * t.days).groupby(keys, sort=False)
          .agg(w=("w", "sum"), days=("days", "sum"), years=("year", "size")).reset_index())
    s["adt"] = (s.w / s.days).round(0)
    for y in TRAFFIC_SOURCES:
        s[f"adt_{y}"] = s.set_index(keys).index.map(
            t[t.year == y].set_index(keys).adt.round(0).to_dict()).astype(float)
    return s.drop(columns=["w", "days"])


def point_traffic(points, sec):
    """Daily traffic at every 100 m sample: the interchange section it lies in, on the viaduct or
    the ground-level road as its road_name says. Samples outside every section (a few hundred m
    beyond the end of a freeway) take the nearest section of the same carriageway."""
    viaduct = points.road_name.fillna("").str.contains("|".join(VIADUCT_ROADS))
    adt = np.full(len(points), np.nan)
    inside = np.zeros(len(points), bool)
    for (ref, d, v), s in sec[sec.km_to > sec.km_from].groupby(["ref", "direction", "viaduct"]):
        idx = np.where((points.ref == ref).values & (points.direction == d).values & (viaduct.values == v))[0]
        k = points.km.values[idx]
        lo, hi, val = s.km_from.values, s.km_to.values, s.adt.values
        hit = (k[:, None] >= lo) & (k[:, None] < hi)
        dist = np.where(hit, 0, np.minimum(np.abs(k[:, None] - lo), np.abs(k[:, None] - hi)))
        j = dist.argmin(axis=1)
        adt[idx], inside[idx] = val[j], hit.any(axis=1)
    return adt, inside


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
            vkm = pts.adt.sum() * 0.1 * sum(DAYS.values())     # vehicle-km over the crash years
            road = pts.road_name.mode().iloc[0] if pts.road_name.notna().any() else ""
            rows.append({
                "chain": ch, "ref": ref, "direction": d, "kind": kind, "road_name": road,
                "km_from": round(a, 1), "km_to": round(min(b, kmax), 1), "length_km": round(length, 1),
                "crashes": len(cc), "a1": int((cc["class"] == "A1").sum()),
                "deaths": int(cc.deaths.sum()), "injuries": int(cc.injuries.sum()),
                "ramp_share": round(float(cc.ramp.mean()), 2) if len(cc) else 0.0,
                "rate": round(len(cc) / length / YEARS, 2),
                # exposure: 高公局 daily traffic of the interchange sections the stretch runs through
                "adt": int(round(pts.adt.mean())),
                "vkm_million": round(vkm / 1e6, 1),
                "rate_vkm": round(len(cc) / vkm * 1e8, 1),          # per 100 million vehicle-km
                "adt_matched": round(float(pts.adt_inside.mean()), 2),
                "county": pts.county.mode().iloc[0] if pts.county.notna().any() else "",
                # ambulance response over the stretch (minutes, from the routing model)
                "resp_med_min": round(float(pts.time_min.median()), 1),
                "resp_max_min": round(float(pts.time_min.max()), 1),
                "resp_station": pts.station_time.mode().iloc[0] if pts.station_time.notna().any() else "",
                "coords": pts[["lon", "lat"]].round(5).values.tolist(),
            })
    return pd.DataFrame(rows)


VKM_BREAKS = [4, 8, 12, 20]   # crashes per 100 million vehicle-km


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
        "priority": {"resp": PRIORITY_RESP_MIN},
        # the two crash rates the page switches between: colour breaks, threshold, slider, scatter axis
        "metrics": {
            "vkm": {"unit": "件／億車公里", "legend": "每億車公里 A1＋A2 件數", "breaks": VKM_BREAKS,
                    "thr": PRIORITY_RATE_VKM, "slider": [3, 40, 0.5],
                    "axis": {"lo": 1, "hi": 60, "ticks": [1, 2, 5, 10, 20, 50]}},
            "km": {"unit": "件／公里／年", "legend": "每公里每年 A1＋A2 件數", "breaks": breaks,
                   "thr": PRIORITY_RATE, "slider": [0.5, 6, 0.1],
                   "axis": {"lo": 0.05, "hi": 16, "ticks": [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]}},
        },
        "counties": json.loads(cty[["COUNTYNAME", "geometry"]].to_json()),
        "bins": [[r.ref, r.direction, r.kind == "main", r.road_name, r.km_from, r.km_to, r.length_km, r.crashes,
                  r.a1, r.deaths, r.injuries, r.rate, r.county, [v for xy in r.coords[::2] + [r.coords[-1]] for v in xy],
                  r.resp_med_min, r.resp_max_min, r.resp_station.split("消防局")[-1], r.adt, r.rate_vkm]
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
    sec = traffic_sections()
    sec.to_csv(OUT / "traffic_sections.csv", index=False, encoding="utf-8-sig")
    points["adt"], points["adt_inside"] = point_traffic(points, sec)
    points["adt_113"], _ = point_traffic(points, sec.assign(adt=sec.adt_113))
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
             "median_rate_vkm": float(b.rate_vkm.median()), "vkm_million": round(float(b.vkm_million.sum())),
             "rate_vkm": round(float(b.crashes.sum() / b.vkm_million.sum() * 100), 1),
             # check: 民國 113 (2024) vehicle-km of the whole network vs 高公局's published total
             "vkm_check": round(float((points.adt_113 * 0.1).sum() * DAYS[113] / 1e6)),
             "vkm_official": VKM_OFFICIAL_113,
             "per_year": {str(y): int((crash.year == y).sum()) for y in SOURCES}}
    print(stats)
    page(b, crash, counties, breaks, stats)
    pd.set_option("display.width", 250)
    print(b.drop(columns="coords").sort_values("rate", ascending=False).head(20).to_string())
    print(b.drop(columns="coords").sort_values("rate_vkm", ascending=False).head(20).to_string())
    print("rate_vkm quantiles:", b.rate_vkm.quantile([.25, .5, .75, .9]).round(1).to_dict(),
          "resp quantiles:", b.resp_med_min.quantile([.5, .75]).to_dict())
    print("stretches with traffic matched on < 90% of their samples:",
          b[b.adt_matched < 0.9][["ref", "direction", "km_from", "km_to", "adt_matched"]].to_string())


if __name__ == "__main__":
    main()
