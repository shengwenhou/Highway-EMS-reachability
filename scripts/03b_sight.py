"""救護風險：available sight distance to an ambulance stopped on the carriageway.

For a point every 20 m on every carriageway chain, look back UPSTREAM (the direction traffic
comes from) and find how far away an approaching driver can first see a vehicle stopped there.
The sight line is blocked when

  - horizontally, it leaves the paved carriageway on the inside of a curve: the road centreline
    strays more than LATERAL_CLEARANCE_M from the straight line between driver and ambulance;
  - vertically, the road surface rises above the straight line from the driver's eye
    (EYE_HEIGHT_M) to the visible part of the ambulance (OBJECT_HEIGHT_M), i.e. a crest.

The available sight distance (ASD) is compared with what a driver at 110 km/h needs:
  stopping sight distance  SSD = 0.278·V·t + V² / (254·(a/9.81 + G))   (AASHTO, grade G)
  decision sight distance  DSD = 335 m   (AASHTO 1994, 110 km/h, manoeuvre C:
                                          speed/path/direction change on a rural road)
  紅 high   ASD < SSD
  黃 medium SSD ≤ ASD < DSD
  綠 low    ASD ≥ DSD

Road elevation: Copernicus GLO-30 DEM (a surface model, so viaduct decks are included).
Inside OSM tunnels the DEM shows the mountain, so the profile is interpolated between portals.

Outputs (data/processed/):
  sight_20m.csv    every 20-m point: position, elevation, grade, curve radius, ASD, SSD, class
  risk_100m.csv    the same aggregated to the 100-m sample points of samples.csv (worst point)
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numba import njit
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DEM_DIR = ROOT / "data" / "raw" / "dtm"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"

STEP_M = 10.0               # profile resolution
TARGET_EVERY = 2            # evaluate a stopped ambulance every 20 m
MAX_LOOK_M = 600.0          # beyond this the sight distance is "long enough"
LATERAL_CLEARANCE_M = 6.0   # carriageway half-width + shoulder an unobstructed sight line may use
EYE_HEIGHT_M = 1.08         # AASHTO driver eye height
OBJECT_HEIGHT_M = 1.0       # visible part of a stopped ambulance (compromise: tail-light 0.6, roof 2.5)
SPEED_KMH = 110.0
REACTION_S = 2.5
DECEL_MS2 = 3.4
DSD_M = 335.0
MAX_GRADE = 0.06            # steepest plausible freeway grade (design maximum ≈ 5 %)
LONG_VIADUCT_M = 1500.0     # shorter bridges are interpolated between abutments
CLASS_NAMES = {0: "綠", 1: "黃", 2: "紅"}


def fetch_dem():
    for lat in range(21, 26):
        for lon in (120, 121):
            n = f"Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM"
            f = DEM_DIR / f"{n}.tif"
            if not f.exists():
                DEM_DIR.mkdir(parents=True, exist_ok=True)
                import subprocess
                r = subprocess.run(["curl", "-sSfL", "-o", str(f), DEM_URL.format(n=n)])
                if r.returncode != 0:
                    f.unlink(missing_ok=True)  # all-sea tiles do not exist


def load_dem():
    fetch_dem()
    srcs = [rasterio.open(f) for f in sorted(glob.glob(str(DEM_DIR / "*.tif")))]
    mosaic, transform = merge(srcs, bounds=(119.95, 21.85, 122.05, 25.35))
    for s in srcs:
        s.close()
    return mosaic[0].astype(np.float32), transform


def sample_dem(dem, transform, lon, lat):
    inv = ~transform
    col, row = inv * (np.asarray(lon), np.asarray(lat))
    # pixel centres sit at +0.5
    return map_coordinates(dem, [row - 0.5, col - 0.5], order=1, mode="nearest")


@njit(cache=True)
def sight_distances(x, y, z, targets, max_k, clear, eye, obj):
    """For each target index i, look back k = 1..max_k steps; return the number of steps that
    stay visible and why the sight line ended (0 none/limit, 1 horizontal, 2 vertical,
    3 start of data)."""
    n_t = targets.shape[0]
    vis = np.zeros(n_t, np.int64)
    why = np.zeros(n_t, np.int64)
    for t in range(n_t):
        i = targets[t]
        k_ok = 0
        reason = 0
        for k in range(1, max_k + 1):
            j = i - k
            if j < 0:
                reason = 3
                break
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            L = np.sqrt(dx * dx + dy * dy)
            if L < 1e-6:
                k_ok = k
                continue
            ze = z[j] + eye
            zo = z[i] + obj
            blocked = 0
            for m in range(j + 1, i):
                # horizontal: distance of centreline point from the chord
                dev = abs((x[m] - x[j]) * dy - (y[m] - y[j]) * dx) / L
                if dev > clear:
                    blocked = 1
                    break
                # vertical: sight-line height above the road surface
                f = (m - j) / (i - j)
                if ze + f * (zo - ze) < z[m]:
                    blocked = 2
                    break
            if blocked:
                reason = blocked
                break
            k_ok = k
        vis[t] = k_ok
        why[t] = reason
    return vis, why


def run_length(mask):
    """For each True element, the length (in samples) of the run of Trues it belongs to."""
    out = np.zeros(len(mask), int)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def limit_grade(z, max_step):
    """Keep the profile within a physically possible freeway grade: clamp forwards and backwards
    and average the two passes."""
    f = z.copy()
    for i in range(1, len(f)):
        f[i] = min(max(f[i], f[i - 1] - max_step), f[i - 1] + max_step)
    b = z.copy()
    for i in range(len(b) - 2, -1, -1):
        b[i] = min(max(b[i], b[i + 1] - max_step), b[i + 1] + max_step)
    return (f + b) / 2


def curve_radius(x, y, half):
    """Radius of the circle through points i-half, i, i+half (metres; inf on straights)."""
    n = len(x)
    a = np.clip(np.arange(n) - half, 0, n - 1)
    c = np.clip(np.arange(n) + half, 0, n - 1)
    ax, ay, bx, by, cx, cy = x[a], y[a], x, y, x[c], y[c]
    ab = np.hypot(bx - ax, by - ay)
    bc = np.hypot(cx - bx, cy - by)
    ca = np.hypot(ax - cx, ay - cy)
    cross = np.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2 * cross)
    r[~np.isfinite(r)] = np.inf
    return r


def profile(ch, dem, transform):
    s = np.asarray(ch["s"])
    ss = np.arange(0, s[-1], STEP_M)
    x = np.interp(ss, s, ch["x"])
    y = np.interp(ss, s, ch["y"])
    lon = np.interp(ss, s, ch["lon"])
    lat = np.interp(ss, s, ch["lat"])
    km = np.interp(ss, s, ch["km"])
    seg = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(s) - 2)
    tunnel = np.asarray(ch["tunnel"], bool)[seg]
    bridge = np.asarray(ch["bridge"], bool)[seg]

    z = sample_dem(dem, transform, lon, lat).astype(float)
    # The DEM is a surface model: overpasses crossing the freeway, vehicles and roadside trees
    # show up as bumps of several metres a few tens of metres long. A 150-m running median removes
    # them; real freeway vertical curves are several hundred metres long and survive it.
    z = median_filter(z, size=15, mode="nearest")
    # Tunnels: the DEM shows the mountain above. Valley viaducts: a 30-m pixel mixes the deck
    # with the valley floor and the wooded slopes. In both cases take the road straight between
    # the portals / abutments. Long urban viaducts (五楊高架 etc.) dominate their pixels and keep
    # the DEM.
    interp = tunnel | (bridge & (run_length(bridge) * STEP_M < LONG_VIADUCT_M))
    if interp.any() and not interp.all():
        z[interp] = np.interp(np.flatnonzero(interp), np.flatnonzero(~interp), z[~interp])
    z = limit_grade(z, MAX_GRADE * STEP_M)
    z = gaussian_filter1d(z, sigma=6, mode="nearest")  # σ = 60 m smoothing of residual noise
    return dict(ss=ss, x=x, y=y, lon=lon, lat=lat, km=km, z=z, tunnel=tunnel, bridge=bridge)


def main():
    chains = json.load(open(PROC / "chains.json"))
    dem, transform = load_dem()
    max_k = int(MAX_LOOK_M / STEP_M)
    v_ms = SPEED_KMH / 3.6
    rows = []
    for ch in chains:
        pr = profile(ch, dem, transform)
        n = len(pr["ss"])
        targets = np.arange(0, n, TARGET_EVERY)
        vis, why = sight_distances(pr["x"], pr["y"], pr["z"], targets, max_k,
                                   LATERAL_CLEARANCE_M, EYE_HEIGHT_M, OBJECT_HEIGHT_M)
        asd = vis * STEP_M
        # grade over the 200 m a driver covers while braking (negative = downhill towards the ambulance)
        back = np.clip(targets - 20, 0, n - 1)
        run = np.maximum((targets - back) * STEP_M, STEP_M)
        grade = (pr["z"][targets] - pr["z"][back]) / run
        grade = np.clip(grade, -0.09, 0.09)
        ssd = 0.278 * SPEED_KMH * REACTION_S + SPEED_KMH ** 2 / (254 * (DECEL_MS2 / 9.81 + grade))
        cls = np.where(asd < ssd, 2, np.where(asd < DSD_M, 1, 0))
        # A sight line cut short only by the start of the data (freeway start, branch start)
        # is not evidence of risk.
        truncated = (why == 3) & (asd < DSD_M)
        cls = np.where(truncated, -1, cls)
        radius = curve_radius(pr["x"], pr["y"], 5)[targets]
        rows.append(pd.DataFrame({
            "chain": ch["chain"], "ref": ch["ref"], "direction": ch["direction"], "kind": ch["kind"],
            "s_m": pr["ss"][targets], "km": np.round(pr["km"][targets], 3),
            "lon": np.round(pr["lon"][targets], 6), "lat": np.round(pr["lat"][targets], 6),
            "elev_m": np.round(pr["z"][targets], 1), "grade_pct": np.round(grade * 100, 2),
            "curve_radius_m": np.round(np.minimum(radius, 99999), 0),
            "tunnel": pr["tunnel"][targets].astype(int), "bridge": pr["bridge"][targets].astype(int),
            "asd_m": asd, "ssd_m": np.round(ssd, 0),
            "limit": pd.Series(why).map({0: "", 1: "平面彎道", 2: "縱坡坡頂", 3: "資料起點"}).values,
            "risk": cls}))
        c = rows[-1].risk.value_counts().to_dict()
        print(f"chain {ch['chain']}: 國道{ch['ref']} {ch['direction']} {ch['kind']} "
              f"red {c.get(2, 0)} yellow {c.get(1, 0)} green {c.get(0, 0)} n/a {c.get(-1, 0)}")
    df = pd.concat(rows, ignore_index=True)
    df["risk_label"] = df.risk.map({2: "紅", 1: "黃", 0: "綠", -1: "資料不足"})
    df.to_csv(PROC / "sight_20m.csv", index=False)

    # Aggregate to the 100-m points of samples.csv: each sample covers [s, s+100).
    smp = pd.read_csv(PROC / "samples.csv")
    df["s_bin"] = (df.s_m // 100) * 100
    g = df[df.risk >= 0].sort_values(["risk", "asd_m"], ascending=[False, True])
    worst = g.groupby(["chain", "s_bin"]).first().reset_index()
    worst = worst[["chain", "s_bin", "risk", "risk_label", "asd_m", "ssd_m", "limit",
                   "grade_pct", "curve_radius_m", "elev_m", "tunnel", "bridge"]]
    smp["s_bin"] = (smp.s_m // 100) * 100
    out = smp[["chain", "s_m", "s_bin"]].merge(worst, on=["chain", "s_bin"], how="left")
    out.drop(columns="s_bin").to_csv(PROC / "risk_100m.csv", index=False)
    print(df.risk_label.value_counts())
    print(out.risk_label.value_counts(dropna=False))


if __name__ == "__main__":
    main()
