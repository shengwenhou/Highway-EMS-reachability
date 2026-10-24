"""Ground imagery for the 五楊 animation, re-projected onto the scene's local km grid.

wide  : Sentinel-2 cloudless 2023 (EOX, CC BY-NC-SA 4.0), whole scene, ~10 m
close : 國土測繪中心 114 年正射影像 (PHOTO2025), around 五股 (scene 1 close-up), ~2 m

Tiles are cached under data/raw/tiles/ and fetched politely (one at a time). Both images are
darkened and desaturated so the routes, colour strips and labels stay readable.
Output: video/data/ground_wide.jpg, video/data/ground_close.jpg, video/data/ground.json (extents in km)
"""
import io
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "tiles"
OUT = ROOT / "video" / "data"
LON0, LAT0 = 121.38, 25.00                       # must match prep_full.py
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 110.57
SOURCES = {
    "eox": ("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg", 0.2),
    "nlsc": ("https://wmts.nlsc.gov.tw/wmts/PHOTO2025/default/GoogleMapsCompatible/{z}/{y}/{x}", 0.4),
}
JOBS = {
    "wide": dict(src="eox", z=14, bbox=(121.05, 24.85, 121.70, 25.13), width=6144, bright=0.62, sat=0.6),
    "close": dict(src="nlsc", z=16, bbox=(121.425, 25.050, 121.497, 25.100), width=3072, bright=0.66, sat=0.65),
}


def merc(lon, lat, z):
    """lon/lat -> global Web Mercator pixel coordinates at zoom z."""
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    r = np.radians(lat)
    y = (1 - np.log(np.tan(r) + 1 / np.cos(r)) / math.pi) / 2 * n
    return x, y


def tile(src, z, x, y):
    f = CACHE / src / str(z) / f"{x}_{y}.jpg"
    if not f.exists():
        url, pause = SOURCES[src]
        f.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            r = subprocess.run(["curl", "-sSf", "-m", "30", "-o", str(f), url.format(z=z, x=x, y=y)])
            if r.returncode == 0:
                break
            time.sleep(2 ** attempt)
        else:
            raise SystemExit(f"tile {src} {z}/{x}/{y} failed")
        time.sleep(pause)
    return Image.open(f).convert("RGB")


def build(name, src, z, bbox, width, bright, sat):
    x0, y1 = merc(bbox[0], bbox[1], z)
    x1, y0 = merc(bbox[2], bbox[3], z)
    tx0, tx1, ty0, ty1 = int(x0 // 256), int(x1 // 256), int(y0 // 256), int(y1 // 256)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    total = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"{name}: {total} tiles from {src} z{z}", flush=True)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mosaic.paste(tile(src, z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    # resample onto the local km grid (equirectangular around LON0/LAT0), north up
    w_km = (bbox[2] - bbox[0]) * KX
    h_km = (bbox[3] - bbox[1]) * KY
    height = int(round(width * h_km / w_km))
    lon = bbox[0] + (np.arange(width) + 0.5) / width * (bbox[2] - bbox[0])
    lat = bbox[3] - (np.arange(height) + 0.5) / height * (bbox[3] - bbox[1])
    mx, my = merc(lon[None, :], lat[:, None], z)
    mx = np.broadcast_to(mx - tx0 * 256, (height, width))
    my = np.broadcast_to(my - ty0 * 256, (height, width))
    a = np.asarray(mosaic, dtype=np.float32)
    out = np.stack([map_coordinates(a[..., c], [my, mx], order=1) for c in range(3)], axis=-1)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(sat)
    img = ImageEnhance.Brightness(img).enhance(bright)
    # a slight blue cast so it sits with the navy palette
    arr = np.asarray(img, dtype=np.float32) * np.array([0.92, 0.97, 1.08])
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(OUT / f"ground_{name}.jpg", quality=88)
    ext = [round((bbox[0] - LON0) * KX, 4), round((bbox[1] - LAT0) * KY, 4),
           round((bbox[2] - LON0) * KX, 4), round((bbox[3] - LAT0) * KY, 4)]
    print(f"  -> ground_{name}.jpg {width}x{height}, extent km {ext}")
    return ext


def main():
    meta = {name: {"extent": build(name, **job), "file": f"ground_{name}.jpg"} for name, job in JOBS.items()}
    (OUT / "ground.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
