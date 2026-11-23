# tools/israel_fits_prep_geo.py: python3 -I tools/israel_fits_prep_geo.py <natural earth geojson dir> templates/opener/data/geo.json templates/data/israel_fits_geo.json
"""Builds templates/data/israel_fits_geo.json for the israel_fits template (run once, offline, by Builder X).
Inputs: Natural Earth 50m geojson (public domain) in ne/, the repo's opener/data/geo.json (Israel shapes).
Tile centres: equal-area cells (k-means / centroidal Voronoi) inside each target in a target-centred Lambert azimuthal
equal-area plane, so N tiles = N cells of about one Israel each."""
import json, math, sys, os
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, mapping, Point
from shapely.ops import unary_union
from shapely import prepared
from scipy.cluster.vq import kmeans2

R = 6371.0072
NE = sys.argv[1]; GEO = sys.argv[2]; OUT = sys.argv[3]


def fwd(lon, lat, lon0, lat0):
    l, p, l0, p0 = map(np.radians, (np.asarray(lon, float), np.asarray(lat, float), lon0, lat0))
    cc = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0)
    k = np.sqrt(2 / np.maximum(1 + cc, 1e-9))
    return R * k * np.cos(p) * np.sin(l - l0), R * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0))


def inv(x, y, lon0, lat0):
    l0, p0 = math.radians(lon0), math.radians(lat0)
    rho = np.hypot(x, y); c = 2 * np.arcsin(np.clip(rho / (2 * R), -1, 1)); rho = np.where(rho == 0, 1e-12, rho)
    lat = np.arcsin(np.cos(c) * math.sin(p0) + y * np.sin(c) * math.cos(p0) / rho)
    lon = l0 + np.arctan2(x * np.sin(c), rho * math.cos(p0) * np.cos(c) - y * math.sin(p0) * np.sin(c))
    return (np.degrees(lon) + 540) % 360 - 180, np.degrees(lat)


def proj_geom(g, lon0, lat0):
    def pr(poly):
        ex = np.array(poly.exterior.coords); x, y = fwd(ex[:, 0], ex[:, 1], lon0, lat0)
        holes = []
        for h in poly.interiors:
            hh = np.array(h.coords); hx, hy = fwd(hh[:, 0], hh[:, 1], lon0, lat0); holes.append(list(zip(hx, hy)))
        return Polygon(list(zip(x, y)), holes)
    polys = [g] if isinstance(g, Polygon) else list(g.geoms)
    return unary_union([pr(p).buffer(0) for p in polys])


def rings(g, tol=0.0):
    """MultiPolygon -> [[ [lon,lat], ... ], ...] exterior rings only, rounded"""
    if tol: g = g.simplify(tol, preserve_topology=True)
    polys = [g] if isinstance(g, Polygon) else list(g.geoms)
    out = []
    for p in polys:
        if p.is_empty: continue
        c = np.round(np.array(p.exterior.coords), 3).tolist()
        if len(c) >= 4: out.append(c)
    return out


def area_km2(g, lon0, lat0): return proj_geom(g, lon0, lat0).area


C = json.load(open(f"{NE}/ne_50m_admin_0_countries.geojson"))
cty = {}
for f in C["features"]:
    k = f["properties"]["ADM0_A3"]; cty[k] = shape(f["geometry"]).buffer(0)
S = json.load(open(f"{NE}/ne_50m_admin_1_states_provinces.geojson"))
st = {f["properties"]["name"]: shape(f["geometry"]).buffer(0) for f in S["features"] if f["properties"].get("iso_a2") == "US"}
LK = json.load(open(f"{NE}/ne_50m_lakes.geojson"))
lk = {f["properties"]["name"]: shape(f["geometry"]).buffer(0) for f in LK["features"] if f["properties"].get("name")}
SU = json.load(open(f"{NE}/ne_50m_admin_0_map_subunits.geojson"))
su = {f["properties"]["SU_A3"]: shape(f["geometry"]).buffer(0) for f in SU["features"]}
RG = json.load(open(f"{NE}/ne_50m_geography_regions_polys.geojson"))
rg = {f["properties"]["NAME"]: shape(f["geometry"]).buffer(0) for f in RG["features"]}

G = json.load(open(GEO))
def gj(gm): return shape(gm).buffer(0)
isr49, golan, wb, gaza = gj(G["isr49"]), gj(G["golan"]), gj(G["wb"]), gj(G["gaza"])
tile = unary_union([isr49, golan]).buffer(0)
c0 = tile.centroid; ILON, ILAT = c0.x, c0.y
tile_km = proj_geom(tile, ILON, ILAT)
if isinstance(tile_km, MultiPolygon): tile_km = max(tile_km.geoms, key=lambda p: p.area)   # drop tiny slivers
print("Israel tile shape area (isr49 + Golan, geo.json):", round(tile_km.area), "km2; CBS 22,072")
SCALE = math.sqrt(22072 / tile_km.area)          # scale the outline to the official CBS area (Israel incl. Golan Heights and East Jerusalem)
print("scale to CBS area:", round(SCALE, 4))
tx = (np.array(tile_km.exterior.coords) - [tile_km.centroid.x, tile_km.centroid.y]) * SCALE
tile_xy = np.round(tx, 2).tolist()

ARAB = ["DZA", "BHR", "COM", "DJI", "EGY", "IRQ", "JOR", "KWT", "LBN", "LBY", "MRT", "MAR", "OMN", "QAT", "SAU", "SOM", "SOL", "SDN", "SYR",
        "TUN", "ARE", "YEM"]
RUS = cty["RUS"]
crimea = Polygon([(32.3, 44.2), (36.8, 44.2), (36.8, 46.3), (32.3, 46.3)])
if RUS.intersects(crimea): print("RUS polygon includes Crimea area:", round(area_km2(RUS.intersection(crimea), 34, 45))); RUS = RUS.difference(crimea).buffer(0)
ata = cty["ATA"]
TG = {   # key: (geometry, lon0, lat0 for the equal-area plane, official area km2, n tiles)
    "LBN": (cty["LBN"], 35.9, 33.9, 10452), "JOR": (cty["JOR"], 36.8, 31.2, 89318), "SYR": (cty["SYR"], 38.5, 35.0, 185180),
    "EGY": (cty["EGY"], 30, 26.5, 1001450), "SAU": (cty["SAU"], 45, 24, 2149690),
    "ARAB": (unary_union([cty[k] for k in ARAB]), 20, 22, 13132000),
    "NJ": (st["New Jersey"], -74.6, 40.1, 22591), "MICH": (lk["Lake Michigan"], -87.0, 43.8, 57757),
    "CA": (st["California"], -119.5, 37.2, 423967), "TX": (st["Texas"], -99.3, 31.3, 695662), "USA": (cty["USA"], -112, 50, 9833517),
    "WLS": (su["WLS"], -3.7, 52.3, 20783), "GBR": (cty["GBR"], -2.5, 54.5, 244376), "DEU": (cty["DEU"], 10.4, 51.1, 357581),
    "RUS": (RUS, 100, 62, 17098246),
    "GRL": (cty["GRL"], -41, 72, 2166086), "AUS": (cty["AUS"], 134, -26, 7688287), "BRA": (cty["BRA"], -53, -11, 8509361),
    "SAHARA": (rg["SAHARA"], 12, 22, 8600000), "ATA": (ata, 0, -90, 14200000),
}
NT = {"LBN": 1, "JOR": 4, "SYR": 8, "EGY": 45, "SAU": 100, "ARAB": 593, "NJ": 1, "MICH": 2, "CA": 19, "TX": 31, "USA": 445, "WLS": 1, "GBR": 11, "DEU": 16, "RUS": 775,
      "GRL": 98, "AUS": 348, "BRA": 386, "SAHARA": 390, "ATA": 643}
START = {"JOR": (35.8, 31.9), "SYR": (36.3, 33.5), "EGY": (31.5, 30.2), "SAU": (40, 27), "ARAB": (34, 31), "TX": (-97, 32), "CA": (-120, 38), "USA": (-100, 38), "GBR": (-1.5, 52),
         "DEU": (10, 51), "RUS": (38, 55), "GRL": (-45, 64), "AUS": (134, -24), "BRA": (-47.9, -15.8), "SAHARA": (12, 22), "ATA": (0, -85)}

out = dict(tile_xy=tile_xy, tile_scale=round(SCALE, 5), israel_centroid=[round(ILON, 4), round(ILAT, 4)],
           isr49=rings(isr49), golan=rings(golan), wb=rings(wb), gaza=rings(gaza), targets={})
rng = np.random.default_rng(7)
for key, (g, l0, p0, off) in TG.items():
    gp = proj_geom(g, l0, p0); A = gp.area; n = NT[key]
    print(f"{key}: NE polygon {A:,.0f} km2 vs official {off:,} ({A / off - 1:+.1%}); tiles {n}")
    minx, miny, maxx, maxy = gp.bounds; P = prepared.prep(gp)
    m = max(4000, 70 * n); pts = []
    while len(pts) < m:
        q = rng.uniform([minx, miny], [maxx, maxy], (m * 2, 2))
        pts += [p for p in q if P.contains(Point(p))]
    pts = np.array(pts[:m])
    if n == 1:
        cen = np.array([[gp.representative_point().x, gp.representative_point().y]]) if key != "NJ" else np.array([[gp.centroid.x, gp.centroid.y]])
        cen = np.array([[gp.centroid.x, gp.centroid.y]]) if P.contains(gp.centroid) else cen
    else:
        cen, _ = kmeans2(pts, n, iter=40, minit="++", seed=7)
    lon, lat = inv(cen[:, 0], cen[:, 1], l0, p0)
    s = START.get(key)
    if s:
        sx, sy = fwd(s[0], s[1], l0, p0); order = np.argsort(np.hypot(cen[:, 0] - sx, cen[:, 1] - sy))
    else: order = np.arange(n)
    tol = 0.02 if A < 100000 else (0.05 if A < 1e6 else 0.1)
    out["targets"][key] = dict(rings=rings(g, tol), tiles=np.round(np.c_[lon, lat][order], 3).tolist(), plane=[l0, p0], ne_km2=round(A))

# world background (Natural Earth 50m, simplified), Israel and Palestinian territories drawn separately from geo.json
W_ = {}
for k, g in cty.items():
    if k in ("ISR", "PSX"): continue
    W_[k] = rings(g, 0.04)
out["world"] = W_
out["lakes"] = {k: rings(lk[k], 0.02) for k in ("Lake Michigan", "Lake Victoria", "Lake Superior", "Lake Huron")}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
