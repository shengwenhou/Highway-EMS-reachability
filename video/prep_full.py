"""Scene data and timeline for the full 五楊 animation (video/film.html).

Inputs : output/points_100m.csv, data/processed/wuyang_points.csv and chains.json (07 / 03),
         video/data/model_paths.json (model_paths.py), data/fire_stations.csv, county outlines,
         video/narration.md and video/audio/scene{N}.wav (tts.py)
Output : video/data/film.json   geometry, routes, colour strips, timeline and subtitle cues
"""
import json
import re
import subprocess
import wave
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "video"
BBOX = (121.05, 24.85, 121.70, 25.13)
LON0, LAT0 = 121.38, 25.00
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 110.57
FPS = 30
TITLE_S, END_S = 4.0, 6.0          # title card before scene 1, closing card after scene 7
LEAD_S, TAIL_S = 1.0, 1.2          # silence before / after each scene's narration
EXTRA_S = {4: 6.0}                 # scene 4 lingers on the finished race
ROUTES = {"興國分隊": "桃園市政府消防局興國分隊", "青埔分隊": "桃園市政府消防局青埔分隊",
          "隆恩分隊": "新北市政府消防局隆恩分隊", "大園分隊": "桃園市政府消防局大園分隊",
          "文化分隊": "新北市政府消防局文化分隊"}
KEY_STATIONS = ("興國", "青埔", "隆恩", "大園", "文化", "德音", "社子", "民權", "社后", "內壢", "楊梅分隊")


def xy(lon, lat):
    return [round((lon - LON0) * KX, 3), round((lat - LAT0) * KY, 3)]


def short(name):
    for pre in ("政府消防局", "消防局"):
        if pre in name:
            name = name.split(pre, 1)[1]
    return re.sub(r"第.救災救護大隊", "", name)


def roads(p):
    out = []
    p = p.assign(viaduct=p.road_name.fillna("").str.contains("高架"))
    for (ch, v), g in p.groupby(["chain", "viaduct"]):
        g = g.sort_values("s_m")
        brk = np.where(np.diff(g.s_m.values) > 150)[0] + 1
        for a, b in zip([0, *brk], [*brk, len(g)]):
            part = g.iloc[a:b:2] if b - a > 4 else g.iloc[a:b]
            if len(part) > 2:
                out.append({"ref": part.ref.iloc[0], "dir": part.direction.iloc[0], "viaduct": bool(v),
                            "road": part.road_name.mode().iloc[0], "pts": [xy(x, y) for x, y in zip(part.lon, part.lat)]})
    return out


def strips():
    """Viaduct carriageways coloured by the fastest station, per direction, in travel order."""
    w = pd.read_csv(ROOT / "data" / "processed" / "wuyang_points.csv")
    names = list(dict.fromkeys(w.sort_values(["direction", "chain", "s_m"]).best_label))
    out = []
    for (d, ch), g in w.groupby(["direction", "chain"]):
        g = g.sort_values("s_m")
        out.append({"dir": d, "pts": [xy(x, y) for x, y in zip(g.lon, g.lat)],
                    "best": [names.index(n) for n in g.best_label], "km": g.km.round(1).tolist(),
                    "county": g.county.tolist()})
    return names, out


def entrances():
    """Entrance ramps of the northbound viaduct (chains 3 and 4), from 03's ramp list."""
    chains = {c["chain"]: c for c in json.load(open(ROOT / "data" / "processed" / "chains.json"))}
    out = []
    for ci in (3, 4):
        c = chains[ci]
        for s, km, name, kind in c["ramps"]:
            # chain 3 (北上 main) is on the viaduct only from 楊梅端 (70.9K) to 59.7K
            if kind == "in" and 50 <= km <= 71 and (ci == 4 or km >= 59.7):
                lon = float(np.interp(s, c["s"], c["lon"])); lat = float(np.interp(s, c["s"], c["lat"]))
                out.append({"name": name.replace("交流道", ""), "km": km, "p": xy(lon, lat)})
    return sorted(out, key=lambda e: -e["km"])


def routes():
    mp = json.load(open(VID / "data" / "model_paths.json"))
    out = {}
    for label, full in ROUTES.items():
        r = mp["routes"][full]
        out[label] = {"pts": [xy(a, b) for a, b in zip(r["lon"], r["lat"])], "sec": r["sec"], "viaduct": r["viaduct"]}
    return out, xy(mp["target"]["lon"], mp["target"]["lat"])


def duration(f):
    with wave.open(str(f)) as w:
        return w.getnframes() / w.getframerate()


def pauses(f):
    """Mid-points of pauses (>= 0.22 s under -35 dB) in a narration file, seconds."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(f), "-af", "silencedetect=n=-35dB:d=0.22",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    return [(a + b) / 2 for a, b in zip(st, en)]


def cues(text, dur, gaps):
    """Split a scene's narration into subtitle lines and time them: proportional to characters,
    each boundary snapped to the nearest real pause in the audio."""
    parts = [p for p in re.split(r"(?<=[。？！；：])", text.replace("\n", "")) if p]
    lines = []
    for p in parts:                     # long sentences break at commas into lines of <= ~24 chars
        cur = ""
        for seg in re.split(r"(?<=，)", p):
            if cur and len(cur) + len(seg) > 24:
                lines.append(cur); cur = ""
            cur += seg
        lines.append(cur)
    n = np.array([len(re.sub(r"[，。：；、？！「」]", "", ln)) for ln in lines], float)
    # walk the lines in order: estimate each end from the remaining text and time, then snap it to
    # the nearest real pause, so an early misestimate does not carry over to the rest
    edges = [0.0]
    for i in range(len(lines) - 1):
        rate = n[i:].sum() / (dur - edges[-1])
        est = edges[-1] + n[i] / rate
        near = [g for g in gaps if abs(g - est) < 1.2 and g > edges[-1] + 0.4 * n[i] / rate]
        edges.append(min(near, key=lambda g: abs(g - est)) if near else est)
    edges.append(dur)
    return [[round(float(a), 2), round(float(b), 2), re.sub(r"[，。；：、]$", "", ln)]
            for a, b, ln in zip(edges[:-1], edges[1:], lines)]


def timeline():
    text = (VID / "narration.md").read_text()
    paras = [p.strip() for p in re.findall(r"\*\*旁白\*\*：\n(.*?)(?=\n## |\Z)", text, re.S)]
    t = TITLE_S
    scenes = []
    for i, para in enumerate(paras, 1):
        f = VID / "audio" / f"scene{i}.wav"
        d = duration(f)
        start = t + LEAD_S
        subs = [[a + start, b + start, s] for a, b, s in cues(para, d, pauses(f))]
        length = LEAD_S + d + TAIL_S + EXTRA_S.get(i, 0)
        scenes.append({"n": i, "t0": round(t, 2), "t1": round(t + length, 2), "audio_at": round(start, 2),
                       "audio_dur": round(d, 2), "subs": subs})
        t += length
    return scenes, round(t + END_S, 2)


def main():
    p = pd.read_csv(ROOT / "output" / "points_100m.csv", dtype={"ref": str}, low_memory=False)
    p = p[p.lon.between(BBOX[0], BBOX[2]) & p.lat.between(BBOX[1], BBOX[3])]
    st = pd.read_csv(ROOT / "data" / "fire_stations.csv")
    st = st[st.lon.between(BBOX[0], BBOX[2]) & st.lat.between(BBOX[1], BBOX[3])]
    stations = [{"name": short(n), "p": xy(a, b), "key": any(k in n for k in KEY_STATIONS)}
                for n, a, b in zip(st.name, st.lon, st.lat)]
    cty = gpd.read_file(ROOT / "data" / "raw" / "county" / "package" / "counties-10t.json",
                        layer="counties").set_crs(4326)
    cty = cty[cty.COUNTYNAME.isin(["臺北市", "新北市", "桃園市", "基隆市"])]
    # county borders to draw: real boundaries only, not the edge of the clipping window
    frame = box(*BBOX)
    borders = []
    for geom in cty.geometry:
        line = geom.boundary.intersection(frame).difference(frame.exterior.buffer(1e-4))
        for ln in (getattr(line, "geoms", None) or [line]):
            if ln.geom_type == "LineString" and ln.length > 0.002:
                borders.append([xy(a, b) for a, b in ln.simplify(0.0005).coords])
    counties = []
    for name, geom in zip(cty.COUNTYNAME, cty.geometry.intersection(frame)):
        for poly in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
            if poly.area > 1e-5:
                counties.append({"name": name, "ring": [xy(a, b) for a, b in poly.simplify(0.0005).exterior.coords]})
    names, strip = strips()
    rts, target = routes()
    scenes, total = timeline()
    data = {"roads": roads(p), "counties": counties, "borders": borders, "stations": stations, "strips": strip, "strip_names": names,
            "entrances": entrances(), "routes": rts, "target": target, "scenes": scenes, "total": total,
            "places": {"汐止": xy(121.6330, 25.0677), "五股": xy(121.4610, 25.0750), "楊梅": xy(121.1450, 24.9070),
                       "機場系統": xy(121.2825, 25.0262), "中壢": xy(121.2200, 24.9600), "三峽": xy(121.3700, 24.9300),
                       "林口": xy(121.3800, 25.0750), "臺北市": xy(121.5500, 25.0650),
                       "新北市": xy(121.4300, 24.9800), "桃園市": xy(121.2300, 24.9300)}}
    (VID / "data" / "film.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(data['roads'])} roads, {len(counties)} county rings, {len(stations)} stations, "
          f"{len(strip)} strips, entrances {[e['name'] + ' ' + str(e['km']) for e in data['entrances']]}")
    for s in scenes:
        print(s["n"], s["t0"], s["t1"], len(s["subs"]), "cues")
    print("total", total, "s")


if __name__ == "__main__":
    main()
