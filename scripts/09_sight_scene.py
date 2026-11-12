"""救護風險 3D 原型：把紅色路段的 20 m 視距剖面嵌進 sight_template.html，產生 output/sight.html。

The page shows, for a stopped ambulance on a red stretch, what a driver approaching at 110 km/h
can see: the road surface from the sight profile (03b_sight.py), the ±6 m sight corridor the
model assumes, the sight line from the driver's eye (1.08 m) to the ambulance (1.0 m), and the
stopping / decision sight distances. The numbers shown (ASD, SSD, class) are the published ones
from sight_20m.csv; the browser only draws them.

Inputs (output/): sight_20m.csv, risk_red_stretches.csv
Output: output/sight.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

UPSTREAM_M = 800      # context before the red stretch: the model looks back up to 600 m
DOWNSTREAM_M = 300
PER_LIMIT = 16        # longest stretches of each kind (平面彎道, 縱坡坡頂)
LIM = {"": 0, "平面彎道": 1, "縱坡坡頂": 2, "資料起點": 3}


def pick(red):
    parts = [red[red.limit == lim].sort_values(["length_m", "min_asd_m"], ascending=[False, True]).head(PER_LIMIT)
             for lim in ("平面彎道", "縱坡坡頂")]
    sel = pd.concat(parts)
    return sel.sort_values(["length_m", "min_asd_m"], ascending=[False, True]).reset_index(drop=True)


def stretch(row, pts):
    ch = pts[pts.chain == row.chain].reset_index(drop=True)
    inside = (ch.km >= row.km_start - 1e-6) & (ch.km <= row.km_end + 1e-6)
    idx = np.flatnonzero(inside)
    i0, i1 = idx[0], idx[-1]
    step = float(np.median(np.diff(ch.s_m)))
    a = max(0, i0 - int(round(UPSTREAM_M / step)))
    b = min(len(ch) - 1, i1 + int(round(DOWNSTREAM_M / step)))
    seg = ch.iloc[a:b + 1].reset_index(drop=True)
    lon0, lat0 = seg.lon.mean(), seg.lat.mean()
    x = (seg.lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    y = (seg.lat - lat0) * 110574.0
    red_rows = seg.iloc[i0 - a:i1 - a + 1]
    # default ambulance position: the shortest sight distance among the points blocked for the
    # stretch's main reason (a crest stretch can contain a few curve-limited points)
    same = red_rows[red_rows.limit == row.limit]
    worst = int((same if len(same) else red_rows).asd_m.idxmin())
    r = lambda v, n=1: [round(float(t), n) for t in v]
    return {
        "title": f"國道{row.ref} {row.direction} {row.km_start:g}K–{row.km_end:g}K",
        "county": row.county, "ref": str(row.ref), "direction": row.direction,
        "kind": "高架" if row.kind == "branch" else "主線",
        "limit": row.limit, "length_m": int(row.length_m), "min_asd": float(row.min_asd_m), "ssd_worst": float(row.ssd_m),
        "range": [int(i0 - a), int(i1 - a)], "worst": worst,
        "s": r(seg.s_m - seg.s_m.iloc[0], 0), "km": r(seg.km, 3),
        "x": r(x, 1), "y": r(y, 1), "z": r(seg.elev_m, 1),
        "asd": r(seg.asd_m, 0), "ssd": r(seg.ssd_m, 0),
        "risk": [int(v) for v in seg.risk], "lim": [LIM[v] for v in seg.limit.fillna("")],
        "grade": r(seg.grade_pct, 1), "radius": [int(min(v, 99999)) for v in seg.curve_radius_m],
    }


def main():
    pts = pd.read_csv(OUT / "sight_20m.csv", dtype={"ref": str}, low_memory=False)
    red = pd.read_csv(OUT / "risk_red_stretches.csv", dtype={"ref": str})
    sel = pick(red)
    data = {"n_red": int(len(red)), "stretches": [stretch(row, pts) for row in sel.itertuples()]}
    tpl = (ROOT / "scripts" / "sight_template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "sight.html").write_text(html)
    n_pts = sum(len(s["s"]) for s in data["stretches"])
    print(f"{len(sel)} stretches, {n_pts} points, {len(html) / 1e6:.2f} MB -> output/sight.html")
    for s in data["stretches"]:
        print(f"  {s['title']}（{s['county']}）{s['limit']} {s['length_m']} m, ASD {s['min_asd']:g} / SSD {s["ssd_worst"]:g}")


if __name__ == "__main__":
    main()
