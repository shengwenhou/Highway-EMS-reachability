"""Build output/trivia.html: 國道救護冷知識 — a card page of facts drawn from the point data.

Every number on the page is recomputed from output/points_100m.csv (and the station / hospital
lists), so the page stays consistent when the analysis is re-run. Lengths of carriageway are
counted per direction; where both directions are summed the text says so.

Each card links to the map page's point query with #pt-<index>, the index into the query table
(points ordered by carriageway and distance along it, as in 04_maps.query_data).
"""
import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COUNTY_TOPO = ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json"

# map frame (main island)
LON0, LON1, LAT0, LAT1 = 120.0, 122.05, 21.88, 25.32
KX = np.cos(np.radians(23.6))
W = 300
H = W * (LAT1 - LAT0) / ((LON1 - LON0) * KX)


def xy(lon, lat):
    return ((np.asarray(lon) - LON0) * KX / ((LON1 - LON0) * KX) * W,
            (LAT1 - np.asarray(lat)) / (LAT1 - LAT0) * H)


def path_of(lon, lat):
    x, y = xy(lon, lat)
    return "M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))


def load():
    p = pd.read_csv(OUT / "points_100m.csv", dtype={"ref": str})
    p = p.sort_values(["chain", "s_m"]).reset_index(drop=True)   # same order as the query table
    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    hp = pd.read_csv(ROOT / "data" / "heavy_er_hospitals.csv")
    counties = gpd.read_file(COUNTY_TOPO, layer="counties").set_crs(4326).cx[119.9:122.2, 21.8:25.4]
    return p, st, hp, counties


def base_symbol(p, counties):
    """Taiwan outline + all freeways, defined once and reused by every card."""
    simp = counties.geometry.simplify(0.008)
    land = []
    for geom in simp:
        polys = getattr(geom, "geoms", [geom])
        for poly in polys:
            c = np.asarray(poly.exterior.coords)
            if c[:, 0].max() < LON0 or c[:, 1].min() > LAT1:
                continue
            land.append(path_of(c[:, 0], c[:, 1]) + "Z")
    roads = []
    for _, g in p[p.kind == "main"].groupby("chain"):
        g = g.iloc[::4]
        roads.append(path_of(g.lon.values, g.lat.values))
    return (f'<symbol id="tw" viewBox="0 0 {W} {H:.0f}">'
            f'<path d="{" ".join(land)}" style="fill:var(--land);stroke:var(--land-edge);stroke-width:.6"/>'
            f'<path d="{" ".join(roads)}" style="fill:none;stroke:var(--road);stroke-width:1.4;stroke-linecap:round"/>'
            f'</symbol>')


def runs(df, flag):
    """Contiguous True stretches per carriageway: list of (index array)."""
    out = []
    for _, g in df.groupby("chain"):
        f = flag.loc[g.index].values
        idx = g.index.values
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j + 1 < len(f) and f[j + 1]:
                    j += 1
                out.append(idx[i:j + 1])
                i = j + 1
            else:
                i += 1
    return sorted(out, key=len, reverse=True)


def stretch(p, ix, cls="hl"):
    sub = p.loc[ix]
    return f'<path class="{cls}" d="{path_of(sub.lon.values, sub.lat.values)}"/>'


def dot(lon, lat, cls="pt", r=13):
    x, y = xy(lon, lat)
    return f'<circle class="{cls}" cx="{float(x):.1f}" cy="{float(y):.1f}" r="{r}"/>'


def dots(df, cls, r=3.5):
    x, y = xy(df.lon.values, df.lat.values)
    return "".join(f'<circle class="{cls}" cx="{a:.1f}" cy="{b:.1f}" r="{r}"/>' for a, b in zip(x, y))


def short(name):
    for pre in ("政府消防局", "消防局"):
        if pre in name:
            name = name.split(pre, 1)[1]
    return name


def km_fmt(v):
    return f"{v:.1f}K"


def build_cards(p, st, hp):
    m = p[p.kind == "main"]
    cards = []

    # 1 response farthest (main line and viaducts)
    i = p.dist_km.idxmax()
    r = p.loc[i]
    rm = m.loc[m.dist_km.idxmax()]       # farthest on a ground-level main line, for comparison
    viaduct = "高架" in str(r.road_name)
    cards.append(dict(
        tag="極端值", title="救護極遠點",
        place=f"國道 {r.ref} 號{r.direction} {r.km:.1f}K（{r.road_name if viaduct else r.county}）",
        big=f"{r.dist_km:.1f}", unit="公里", sub=f"{r.time_min:.1f} 分鐘",
        body=(f"全國國道（含高架）離消防分隊最遠的點。北上五楊高架在機場系統到五股之間沒有任何入口，"
              f"泰山轉接道北上也只出不進，所以最近的{short(r.station_dist)}得先到機場系統上高架，再一路往北開。"
              f"平面主線上最遠的則是國道 {rm.ref} 號{rm.direction} {rm.km:.1f}K（{rm.dist_km:.1f} 公里），"
              "因為嶺口交流道東行只有出口。") if viaduct else
             ("離消防分隊最遠的點。最近的旗山分隊其實只在 2.5 公里外，但嶺口交流道東行只有出口，"
              f"救護車只能從{short(r.station_dist)}上國 3、再轉國 10 一路開過來。"),
        svg=dot(r.lon, r.lat), link=i))

    # 2 transport farthest
    i = m.hosp_dist_km.idxmax()
    r = p.loc[i]
    cards.append(dict(
        tag="極端值", title="送醫極遠點", place=f"國道 {r.ref} 號{r.direction} {r.km:.1f}K（{r.county}）",
        big=f"{r.hosp_dist_km:.1f}", unit="公里", sub=f"{r.hosp_time_min:.1f} 分鐘",
        body=f"送到最近的重度級醫院（{short(r.hospital_dist)}）最遠的點。埔里一帶沒有重度級醫院，"
             "國 6 全線都要送到南投市；這個點又剛錯過出口，得先往東開到下一個交流道才能掉頭。",
        svg=dot(r.lon, r.lat), link=i))

    # 3 luckiest point: smallest response + transport time
    tot = m.time_min + m.hosp_time_min
    i = tot.idxmin()
    r = p.loc[i]
    hs = m.hosp_dist_km.idxmin()
    cards.append(dict(
        tag="極端值", title="全台最幸運的車禍地點", place=f"國道 {r.ref} 號{r.direction} {r.km:.1f}K（{r.county}）",
        big=f"{tot.loc[i]:.1f}", unit="分鐘", sub="救護車到場＋送進急診",
        body=f"救護車 {r.time_min:.1f} 分鐘到場，再 {r.hosp_time_min:.1f} 分鐘就進{short(r.hospital_time).replace('東元醫療社團法人', '')}急診。"
             f"南下 {p.loc[hs].km:.1f}K 離重度級醫院只有 {p.loc[hs].hosp_dist_km:.1f} 公里，是全國最近。",
        svg=dot(r.lon, r.lat), link=i))

    # 4 elevated vs ground
    br = p[(p.ref == "1") & (p.kind == "branch") & (p.direction == "北上")]
    gr = m[(m.ref == "1") & (m.direction == "北上")]
    t = cKDTree(gr[["km"]].values)
    _, k = t.query(br[["km"]].values)
    diff = br.dist_km.values - gr.dist_km.values[k]
    j = int(np.argmax(diff))
    ie, ig = br.index[j], gr.index[k[j]]
    e, g = p.loc[ie], p.loc[ig]
    cards.append(dict(
        tag="反差", title=f"上下兩層，差了 {int(e.dist_km - g.dist_km)} 公里", place=f"國道 1 號北上 {g.km:.1f}K（{g.county}）",
        big=f"{e.dist_km - g.dist_km:.1f}", unit="公里差距", sub=f"平面 {g.dist_km:.1f} km／高架 {e.dist_km:.1f} km",
        body=f"在平面主線出事，救護車開 {g.dist_km:.1f} 公里就到；在正上方的五楊高架出事，"
             f"要從機場系統上高架一路開過來，共 {e.dist_km:.1f} 公里。",
        svg=dot(e.lon, e.lat), link=ie))

    # 5 opposite carriageways
    a5 = m[(m.ref == "5") & (m.direction == "北上")]
    b5 = m[(m.ref == "5") & (m.direction == "南下")]
    t = cKDTree(np.c_[b5.lon * 101000, b5.lat * 111000])
    dd, k = t.query(np.c_[a5.lon * 101000, a5.lat * 111000])
    diff = np.where(dd < 80, a5.dist_km.values - b5.dist_km.values[k], -1)
    j = int(np.argmax(diff))
    ia, ib = a5.index[j], b5.index[k[j]]
    a, b = p.loc[ia], p.loc[ib]
    cards.append(dict(
        tag="反差", title=f"只隔一道護欄，差了 {int(a.dist_km - b.dist_km)} 公里", place=f"國道 5 號 {a.km:.1f}K（坪林附近）",
        big=f"{a.dist_km - b.dist_km:.1f}", unit="公里差距", sub=f"北上 {a.dist_km:.1f} km／南下 {b.dist_km:.1f} km",
        body=f"同一個位置，南下車道的救護車 {b.dist_km:.1f} 公里就到；北上車道要從宜蘭端出發、"
             f"穿過雪山隧道，共 {a.dist_km:.1f} 公里。國道 10 號 33K 也一樣：東行 24.5 公里，西行 2.8 公里。",
        svg=dot(a.lon, a.lat), link=ia))

    # 6 station with the longest freeway responsibility
    counts = m.station_dist.value_counts()
    top = counts.index[0]
    own = m[m.station_dist == top]
    other = own[own.cross_county_dist == True]  # noqa: E712
    s0 = st[st.name == top].iloc[0]
    cards.append(dict(
        tag="分隊與醫院", title="國道責任範圍最長的分隊", place=short(top) + "（" + top[:3] + "）",
        big=f"{len(own) / 20:.0f}", unit="公里國道", sub=f"雙向合計 {len(own) / 10:.0f} 公里",
        body=f"國道上約 {len(own) / 20:.0f} 公里的最近分隊都是它，全國第一。其中約 {len(other) / 20:.0f} 公里"
             f"（雙向合計 {len(other) / 10:.0f} 公里）在{other.county.mode().iloc[0]}境內，也是跨區救護範圍最長的分隊。",
        svg=dots(own, "hl-dot", 3.5) + dot(s0.lon, s0.lat, "pt", 10), link=int(own.index[len(own) // 2])))

    # 7 how many stations / hospitals ever nearest
    used = set(m.station_dist)
    st_used = st[st.name.isin(used)]
    st_not = st[~st.name.isin(used)]
    n_h = m.hospital_dist.nunique()
    cards.append(dict(
        tag="分隊與醫院", title="只有四分之一的分隊會跑國道", place="全國消防分隊",
        big=f"{len(used)}", unit=f"／{len(st)} 個分隊", sub=f"{len(used) / len(st) * 100:.0f}%",
        body=f"{len(st)} 個分隊中，只有 {len(used)} 個曾是國道上某個點的最近分隊。"
             f"{len(hp)} 家重度級醫院中，則有 {n_h} 家是某段國道的最近醫院。",
        svg=dots(st_not, "mute-dot", 2.6) + dots(st_used, "hl-dot", 3.6), link=None))

    # 8 hospital with the longest freeway responsibility
    hc = m.hospital_dist.value_counts()
    htop = hc.index[0]
    own = m[m.hospital_dist == htop]
    h0 = hp[hp.name == htop].iloc[0]
    second = [short(n) for n in hc.index[1:3]]
    cards.append(dict(
        tag="分隊與醫院", title="責任範圍最長的重度級醫院", place="衛福部桃園醫院",
        big=f"{len(own) / 20:.0f}", unit="公里國道", sub=f"雙向合計 {len(own) / 10:.0f} 公里",
        body=f"國道 1、2、3 號在桃園的路段，約 {len(own) / 20:.0f} 公里的最近重度級醫院都是它。"
             "其次是奇美醫院和衛福部南投醫院。（以路段長度計，未以事故件數加權。）",
        svg=dots(own, "hl-dot", 3.5) + dot(h0.lon, h0.lat, "pt", 10), link=int(own.index[len(own) // 2])))

    # 9 longest cross-county run
    rr = runs(m, m.cross_county_dist == True)  # noqa: E712
    ix = rr[0]
    a, b = p.loc[ix[0]], p.loc[ix[-1]]
    k0, k1 = sorted([a.km, b.km])
    cards.append(dict(
        tag="跨區救護", title="明明在高雄，最近的卻是屏東", place=f"國道 {a.ref} 號{a.direction} {k0:.0f}–{k1:.0f}K",
        big=f"{len(ix) / 10:.0f}", unit="公里", sub="單一方向連續",
        body=f"這一段在{a.county}境內，但只能從燕巢系統的上游進入，最近的反而是{short(a.station_dist)}（{a.station_dist[:3]}）。"
             "是全國最長的連續跨區救護路段。",
        svg=stretch(p, ix), link=int(ix[len(ix) // 2])))

    # 10 Freeway 6 transport
    med = m.groupby("ref").hosp_dist_km.median()
    others = med.drop("6")
    cards.append(dict(
        tag="路線", title="送醫最辛苦的國道", place="國道 6 號（水沙連高速公路）",
        big=f"{med['6']:.1f}", unit="公里", sub="送醫距離中位數",
        body=f"其他國道大多在 {others.drop('3甲').min():.0f}–{others.max():.0f} 公里。埔里一帶沒有重度級急救責任醫院，"
             "國道 6 號全線的傷患都要送到南投市的衛福部南投醫院。",
        svg=stretch(p, m[(m.ref == "6") & (m.direction == "東行")].index), link=int(m[(m.ref == "6")].index[100])))

    # 11 longest green stretch
    rr = runs(m, m.risk == 0)
    ix = rr[0]
    a, b = p.loc[ix[0]], p.loc[ix[-1]]
    nr = runs(m, m.risk.isin([0, 1]))[0]
    na, nb = p.loc[nr[0]], p.loc[nr[-1]]
    cards.append(dict(
        tag="救護風險", title="最長的安心停車路段", place=f"國道 {a.ref} 號{a.direction} {a.km:.1f}–{b.km:.1f}K（{a.county}）",
        big=f"{len(ix) / 10:.1f}", unit="公里全綠", sub="單一方向連續",
        body="救護車停在這一段任何地方，後方 110 km/h 的來車都能及早看見並煞停。"
             f"如果只要求「沒有紅色」，國道 {na.ref} 號{na.direction} {na.km:.1f}–{nb.km:.1f}K（{na.county}）"
             f"連續 {len(nr) / 10:.1f} 公里，是全國最長。",
        svg=stretch(p, ix, "hl-green"), link=int(ix[len(ix) // 2])))

    # 12 longest red stretch
    rr = runs(m, m.risk == 2)
    ix = rr[0]
    a, b = p.loc[ix[0]], p.loc[ix[-1]]
    k0, k1 = sorted([a.km, b.km])
    cards.append(dict(
        tag="救護風險", title="最長的連續紅色路段", place=f"國道 {a.ref} 號{a.direction} {k0:.1f}–{k1:.1f}K（{a.county}）",
        big=f"{len(ix) / 10:.1f}", unit="公里", sub=f"最短視距約 {p.loc[ix].asd_m.min():.0f} m",
        body="一段連續的彎道，救護車停在這裡時，後方來車看到它的距離短於時速 110 公里所需的停車視距，"
             "可能來不及煞停。",
        svg=stretch(p, ix, "hl-red"), link=int(ix[len(ix) // 2])))

    # 13 red share by freeway
    rated = m[m.risk >= 0]   # points with a risk class (excludes "data start" points)
    share = rated.groupby("ref").apply(lambda g: (g.risk == 2).mean())
    red_km = (m.risk == 2).sum() / 10
    cards.append(dict(
        tag="救護風險", title="一半是紅色的國道", place="國道 3 甲（台北聯絡線）",
        big=f"{share['3甲'] * 100:.0f}", unit="% 紅色", sub=f"國道 8 號只有 {share['8'] * 100:.0f}%",
        body=f"國道 3 甲短而彎，是紅色比例最高的國道。全國國道主線約有 {(rated.risk == 2).mean() * 100:.0f}% 是紅色，"
             f"雙向合計約 {red_km:.0f} 公里。",
        svg=stretch(p, m[(m.ref == "3甲") & (m.direction == "西行")].index, "hl-red"),
        link=int(m[m.ref == "3甲"].index[10])))

    # 14 overall
    within = (m.time_min <= 10).mean() * 100
    differ = (m.station_dist != m.station_time).mean() * 100
    slow = m[m.time_min > 10]
    cards.append(dict(
        tag="整體", title="十分鐘內，救護車可以到", place="全國國道主線",
        big=f"{within:.1f}", unit="%", sub="不塞車的情況下",
        body=f"全國 {within:.1f}% 的國道點位，救護車 10 分鐘內可以到（地圖上標出的是超過 10 分鐘的點）。"
             f"另外有 {differ:.0f}% 的點，「距離最近」和「時間最快」的分隊不是同一個。",
        svg=dots(slow, "hl-dot", 3), link=None))
    return cards


PAGE = """<!doctype html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>國道救護冷知識</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@700;900&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: same atlas-sheet identity as the map page; a header, then a grid of fact cards,
   each card = number + headline + big figure + note + a small locator map. */
:root {
  --bg: #f5f6f2; --surface: #ffffff; --land: #e7eae3; --land-edge: #bfc5ba; --road: #c3c9bf;
  --fg: #1e2420; --muted: #5d665f; --rule: #d6dbd3; --accent: #0f5c4a;
  --hl: #c2410c; --red: #d33a2c; --green: #1f9e5a;
  --font-display: "Noto Serif TC", "Songti TC", serif;
  --font-body: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #131714; --surface: #1b201c; --land: #252b26; --land-edge: #3c453e; --road: #485049;
    --fg: #e8ece6; --muted: #9ba59d; --rule: #323a34; --accent: #5cc4a4;
    --hl: #f97316; --red: #ec5a4a; --green: #34b874; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #131714; --surface: #1b201c; --land: #252b26; --land-edge: #3c453e; --road: #485049;
  --fg: #e8ece6; --muted: #9ba59d; --rule: #323a34; --accent: #5cc4a4;
  --hl: #f97316; --red: #ec5a4a; --green: #34b874; color-scheme: dark;
}
* { box-sizing: border-box; }
html { padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.65; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.wrap { max-width: 1200px; margin: 0 auto; padding-inline: 16px; padding-block: 24px 48px; display: grid; gap: 24px; }
header { display: grid; gap: 8px; }
.back { font-size: 14px; font-weight: 700; text-decoration: none; }
.eyebrow { font-size: 12px; letter-spacing: .12em; color: var(--accent); font-weight: 700; }
h1 { font-family: var(--font-display); font-weight: 900; font-size: clamp(28px, 5vw, 44px); line-height: 1.2; margin: 0; text-wrap: balance; }
.lede { color: var(--muted); max-width: 68ch; margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 16px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 18px; display: grid; grid-template-columns: minmax(0, 1fr) 120px; gap: 6px 14px; align-content: start; }
.head { grid-column: 1 / -1; display: flex; gap: 10px; align-items: baseline; }
.no { font-family: var(--font-data); font-size: 13px; color: var(--muted); }
.tag { font-size: 12px; letter-spacing: .06em; color: var(--accent); font-weight: 700; }
.card h2 { grid-column: 1; font-family: var(--font-display); font-size: 21px; line-height: 1.3; margin: 0; text-wrap: balance; }
.place { grid-column: 1; font-size: 13px; color: var(--muted); }
.figure { grid-column: 1; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px; margin-top: 6px; }
.big { font-family: var(--font-data); font-weight: 500; font-size: 44px; line-height: 1; font-variant-numeric: tabular-nums; }
.unit { font-size: 15px; font-weight: 700; }
.subfig { font-size: 13px; color: var(--muted); width: 100%; }
.mini { grid-column: 2; grid-row: 2 / span 3; width: 120px; height: auto; align-self: start; }
.body { grid-column: 1 / -1; margin: 6px 0 0; font-size: 14px; color: var(--fg); }
.more { grid-column: 1 / -1; font-size: 13px; font-weight: 700; text-decoration: none; margin-top: 4px; }
.more:hover { text-decoration: underline; }
.hl { fill: none; stroke: var(--hl); stroke-width: 12; stroke-linecap: round; }
.hl-red { fill: none; stroke: var(--red); stroke-width: 14; stroke-linecap: round; }
.hl-green { fill: none; stroke: var(--green); stroke-width: 14; stroke-linecap: round; }
.hl-dot { fill: var(--hl); }
.mute-dot { fill: var(--road); }
.pt { fill: var(--hl); stroke: var(--surface); stroke-width: 5; }
footer { color: var(--muted); font-size: 13px; max-width: 80ch; display: grid; gap: 6px; border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 420px) { .big { font-size: 38px; } .card { grid-template-columns: minmax(0, 1fr) 96px; } .mini { width: 96px; } }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">__SYMBOL__</svg>
<div class="wrap">
  <header>
    <a class="back" href="index.html">← 回到國道救護可及性地圖</a>
    <a class="back" href="wuyang.html">看汐五・五楊高架的救護分工 →</a>
    <a class="back" href="crash.html">看國道事故地圖 →</a>
    <div class="eyebrow">從 2 萬個國道點位算出來的 14 件事</div>
    <h1>國道救護冷知識</h1>
    <p class="lede">我們把台灣每一條國道依行車方向每 100 公尺取一個點，算出救護車多久到、送醫要多遠，以及救護車停在路上時被追撞的風險。以下是資料裡最極端、也最出人意料的幾個地方。</p>
  </header>
  <main class="grid">
__CARDS__
  </main>
  <footer>
    <div>所有數字都是模型推估：不塞車、依 OpenStreetMap 的匝道資料計算，視距用 30 公尺地形資料估算。部分結果取決於個別交流道的匝道配置，詳見 <a href="https://github.com/shengwenhou/Highway-EMS-reachability/blob/main/VERIFY.md">VERIFY.md</a>。</div>
    <div>「責任範圍」只計算路段長度，沒有用事故件數加權，不代表實際出勤或收治人數。資料：© OpenStreetMap contributors、內政部消防署、衛生福利部、Copernicus DEM。</div>
  </footer>
</div>
</body>
</html>
"""


def card_html(n, c):
    mini = (f'<svg class="mini" viewBox="0 0 {W} {H:.0f}" role="img" aria-label="位置示意圖">'
            f'<use href="#tw"/>{c["svg"]}</svg>')
    link = (f'<a class="more" href="index.html#pt-{c["link"]}">在地圖上查詢這個點 →</a>'
            if c["link"] is not None else '<a class="more" href="index.html">打開地圖 →</a>')
    e = html.escape
    return f"""    <article class="card">
      <div class="head"><span class="no">{n:02d}</span><span class="tag">{e(c["tag"])}</span></div>
      <h2>{e(c["title"])}</h2>
      <div class="place">{e(c["place"])}</div>
      <div class="figure"><span class="big">{e(c["big"])}</span><span class="unit">{e(c["unit"])}</span><span class="subfig">{e(c["sub"])}</span></div>
      {mini}
      <p class="body">{e(c["body"])}</p>
      {link}
    </article>"""


def main():
    p, st, hp, counties = load()
    cards = build_cards(p, st, hp)
    page = (PAGE.replace("__SYMBOL__", base_symbol(p, counties))
                .replace("__CARDS__", "\n".join(card_html(i + 1, c) for i, c in enumerate(cards))))
    (OUT / "trivia.html").write_text(page)
    for i, c in enumerate(cards, 1):
        print(f"{i:02d} {c['title']} | {c['place']} | {c['big']} {c['unit']} | {c['sub']}")
        print("    ", c["body"])


if __name__ == "__main__":
    main()
