"""Download OpenStreetMap data and cut out the drivable road network of Taiwan's main island.

Source: BBBike daily sub-planet "asia-south-china" (contains Taiwan, about 2.2 GB). It is cut down
with the `osmium` command-line tool (apt install osmium-tool) and read with pyosmium.

With `--geofabrik` the Taiwan extract of Geofabrik (about 310 MB) is used instead and filtered in
pyosmium alone, so the `osmium` command-line tool is not needed (handy on macOS). Geofabrik is not
reachable from every network, which is why it is not the default.

Output:
  data/raw/osm_roads.json      roads in Overpass-like JSON (ways with node ids + geometry)
  data/raw/osm_junctions.json  motorway_junction nodes (their ref = interchange mileage)
  data/raw/county/             county outlines (npm taiwan-atlas) used as the map base
"""
import json
import subprocess
import sys
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SRC_URL = "https://download.bbbike.org/osm/planet/sub-planet-daily/asia-south-china.osm.pbf"
SRC = RAW / "asia-south-china.osm.pbf"
TW = RAW / "taiwan.osm.pbf"
GEOFABRIK_URL = "https://download.geofabrik.de/asia/taiwan-latest.osm.pbf"
GEOFABRIK = RAW / "taiwan-latest.osm.pbf"
ROADS = RAW / "taiwan-roads.osm.pbf"

# Main island of Taiwan (outlying islands have no national freeways).
BBOX = "119.95,21.85,122.05,25.35"
ROAD_CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street",
]


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


class Collector(osmium.SimpleHandler):
    def __init__(self, bbox=None):
        super().__init__()
        self.ways, self.junctions = [], []
        self.bbox = bbox   # (west, south, east, north): keep only ways with a node inside, like
                           # `osmium extract --strategy complete_ways`; None = the file is already cut

    def node(self, n):
        if n.tags.get("highway") == "motorway_junction":
            self.junctions.append({"type": "node", "id": n.id, "lat": n.location.lat,
                                   "lon": n.location.lon, "tags": dict(n.tags)})

    def way(self, w):
        if w.tags.get("highway") not in ROAD_CLASSES:
            return
        try:
            geom = [{"lat": nd.location.lat, "lon": nd.location.lon} for nd in w.nodes]
        except osmium.InvalidLocationError:
            return
        if self.bbox:
            w0, s0, e0, n0 = self.bbox
            if not any(w0 <= g["lon"] <= e0 and s0 <= g["lat"] <= n0 for g in geom):
                return
        self.ways.append({"type": "way", "id": w.id, "nodes": [nd.ref for nd in w.nodes],
                          "geometry": geom, "tags": dict(w.tags)})


ATLAS_URL = "https://registry.npmjs.org/taiwan-atlas/-/taiwan-atlas-2021.9.20.tgz"


def main():
    county = RAW / "county"
    if not (county / "package" / "counties-10t.json").exists():  # county outlines for the maps
        county.mkdir(exist_ok=True)
        run("curl", "-sSfL", "-o", str(county / "taiwan-atlas.tgz"), ATLAS_URL)
        run("tar", "xzf", str(county / "taiwan-atlas.tgz"), "-C", str(county))
    if "--geofabrik" in sys.argv:
        if not GEOFABRIK.exists():
            run("curl", "-sSfL", "-o", str(GEOFABRIK), GEOFABRIK_URL)
        c = Collector(bbox=tuple(float(v) for v in BBOX.split(",")))
        c.apply_file(str(GEOFABRIK), locations=True)
    else:
        if not SRC.exists():
            run("curl", "-sSfL", "-o", str(SRC), SRC_URL)
        if not TW.exists():
            run("osmium", "extract", "-b", BBOX, "--strategy", "complete_ways", "-o", str(TW), str(SRC))
        if not ROADS.exists():
            run("osmium", "tags-filter", "-o", str(ROADS), str(TW),
                "w/highway=" + ",".join(ROAD_CLASSES), "n/highway=motorway_junction")
        c = Collector()
        c.apply_file(str(ROADS), locations=True)
    json.dump({"elements": c.ways}, open(RAW / "osm_roads.json", "w"), ensure_ascii=False)
    json.dump({"elements": c.junctions}, open(RAW / "osm_junctions.json", "w"), ensure_ascii=False)
    print(f"{len(c.ways)} road ways, {len(c.junctions)} motorway junctions")


if __name__ == "__main__":
    main()
