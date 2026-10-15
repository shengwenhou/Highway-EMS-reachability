"""Download and clean the national list of fire units (消防署「救援與應變單位點位」, data.gov.tw #5969).

Keeps field units that dispatch ambulances (分隊 only); drops HQs, 大隊 and 中隊 offices,
special search-and-rescue units and harbour units. Output: data/fire_stations.csv (committed to the repo).
"""
import io
from pathlib import Path

import pandas as pd
import subprocess
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/"
    "57F3DD1D-A40E-49A6-8410-57303B2FF87E/resource/C38B7AC2-E7F3-4DD5-A3F3-88E623B55924/download"
)
# Rows published with 0,0 coordinates; geocoded by hand from the listed address
# (OSM Nominatim, 2026-10). Accuracy is street-level (a few hundred metres at worst).
MANUAL_COORDS = {
    "蘇澳港分隊": (121.8581413, 24.5963647),
    "屏東縣政府消防局枋山分隊": (120.6950524, 22.2071131),
    "苗栗縣政府消防局頭份分隊": (120.9118406, 24.6756758),
    "高雄市政府消防局路竹分隊": (120.2589437, 22.8549157),
    "臺中市政府消防局國光分隊": (120.6801763, 24.1071886),
    "臺南市政府消防局將軍分隊": (120.1567366, 23.1998943),
}
EXCLUDE = ("特種", "特搜", "搜救救助", "訓練", "港務", "防災教育")
# Harbour fire units (former port-authority brigades at Keelung, Taipei, Taichung, Su'ao and
# Hualien ports). They serve the port areas and are not dispatched to freeway incidents.
# Listed by exact name because many ordinary units also contain 港 (北港, 東港, 南港 ...).
HARBOUR_UNITS = {
    "蘇澳港分隊", "花蓮港分隊", "臺北港分隊",          # 蘇澳港、花蓮港、臺北港
    "西岸分隊", "東岸分隊",                            # 基隆港
    "西碼頭分隊", "防風林分隊", "南堤分隊", "第一分隊",  # 臺中港
}


def main() -> None:
    raw = ROOT / "data" / "raw" / "fire_units_raw.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        # curl rather than requests: the MOI server's cert chain fails Python's strict X.509 checks.
        subprocess.run(["curl", "-sSfL", "-m", "120", "-o", str(raw), URL], check=True)
    content = raw.read_bytes()
    for enc in ("utf-8-sig", "cp950", "big5hkscs"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df = df.iloc[:, :5]
    df.columns = ["name", "address", "phone", "x", "y"]
    df = df.dropna(subset=["name", "x", "y"])
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)

    for name, (lon, lat) in MANUAL_COORDS.items():
        df.loc[df["name"] == name, ["x", "y"]] = (lon, lat)

    # The header says TWD97TM121 but most rows are WGS84 lon/lat; convert any metric rows.
    metric = df["x"] > 1000
    if metric.any():
        tf = Transformer.from_crs(3826, 4326, always_xy=True)
        lon, lat = tf.transform(df.loc[metric, "x"].values, df.loc[metric, "y"].values)
        df.loc[metric, "x"], df.loc[metric, "y"] = lon, lat
    df = df.rename(columns={"x": "lon", "y": "lat"})

    # Only 分隊 run ambulances. 大隊 and 中隊 are command levels and do not, so a name must
    # end in 分隊; names like 「…第二大隊大美分隊」 carry 大隊 only as a hierarchy prefix.
    keep = (df["name"].str.endswith("分隊") & ~df["name"].str.contains("|".join(EXCLUDE))
            & ~df["name"].isin(HARBOUR_UNITS))
    out = df[keep].copy()
    # Main island only.
    out = out[out.lon.between(119.95, 122.05) & out.lat.between(21.85, 25.35)]
    out = out[~out["address"].fillna("").str.contains("澎湖|金門|連江|綠島|蘭嶼|琉球")]
    out = out.drop_duplicates(subset=["name"]).reset_index(drop=True)
    out.insert(0, "station_id", range(len(out)))
    out.to_csv(ROOT / "data" / "fire_stations.csv", index=False)
    print(f"{len(df)} units in source, {len(out)} kept as dispatching stations")
    print("dropped:", sorted(set(df["name"]) - set(out["name"])))


if __name__ == "__main__":
    main()
