#!/usr/bin/env python3
"""osm_city: OpenStreetMap -> rasters for the city_flyover template (run on GitHub, the container cannot reach Overpass).

  python tools/osm_city.py <name> <south> <west> <north> <east> [m_per_px]

writes templates/data/city_<name>/
  height.png   uint16 PNG, building height in decimetres per pixel (0 = no building)
  ground.png   RGB ground colours (dark brand style: navy land, sea, parks, water, roads, rail)
  masks.png    RGB highlight masks:  R = named big park (Yarkon), G = Ayalon corridor (motorway 20 + Ayalon stream + rail
               between them), B = rivers and streams (water lines)
  masks2.png   RGB:  R = named boulevards (Geddes plan), G = small public gardens and squares, B = all parks
  meta.json    bbox, size, metres per pixel, counts, licence line (OpenStreetMap contributors, ODbL)
Data (c) OpenStreetMap contributors, ODbL. Heights: OSM height, else building:levels x 3.2 m + 1.5, else 9 to 15 m (id hash).
"""
import json, math, os, sys, time, zlib
import numpy as np, cv2, requests

name, S, Wl, N, E = sys.argv[1], *map(float, sys.argv[2:6])
MPP = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
OUT = f"templates/data/city_{name}"; os.makedirs(OUT, exist_ok=True)
LAT0 = (S + N) / 2; MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
WM, HM = (E - Wl) * MLON, (N - S) * MLAT; PW, PH = int(round(WM / MPP)), int(round(HM / MPP))
print("raster", PW, PH, "m/px", MPP)
UA = {"User-Agent": "reels-drop-osm/1.0 (github.com/avinoamroz/reels-drop)"}
EPS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
       "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def q(body):
    bb = f"({S},{Wl},{N},{E})"; ql = f"[out:json][timeout:300][maxsize:1073741824];({body.replace('BB', bb)});out geom;"
    for k in range(6):
        ep = EPS[k % len(EPS)]
        try:
            r = requests.post(ep, data={"data": ql}, headers=UA, timeout=400)
            if r.status_code == 200: return r.json()["elements"]
            print("overpass", ep, r.status_code, r.text[:200])
        except Exception as e: print("overpass", ep, e)
        time.sleep(15)
    raise SystemExit("overpass failed")


def px(lon, lat): return ((lon - Wl) * MLON / MPP, (N - lat) * MLAT / MPP)


def rings(el):
    """outer rings (and inner rings) of a way or multipolygon relation as pixel arrays"""
    outs, ins = [], []
    if el["type"] == "way" and el.get("geometry"):
        outs.append(np.array([px(g["lon"], g["lat"]) for g in el["geometry"]], np.float64))
    elif el["type"] == "relation":
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("geometry"):
                a = np.array([px(g["lon"], g["lat"]) for g in m["geometry"] if g], np.float64)
                (ins if m.get("role") == "inner" else outs).append(a)
    return outs, ins


def fill(img, el, val):
    outs, ins = rings(el)
    sh = 4
    for r in outs:
        if len(r) >= 3: cv2.fillPoly(img, [np.round(r * (1 << sh)).astype(np.int32)], val, cv2.LINE_8, sh)
    for r in ins:
        if len(r) >= 3: cv2.fillPoly(img, [np.round(r * (1 << sh)).astype(np.int32)], 0 if np.isscalar(val) else (0, 0, 0), cv2.LINE_8, sh)


def line(img, el, val, w):
    if not el.get("geometry"): return
    a = np.array([px(g["lon"], g["lat"]) for g in el["geometry"]], np.float64)
    cv2.polylines(img, [np.round(a * 16).astype(np.int32)], False, val, max(1, int(round(w))), cv2.LINE_AA, 4)


def hgt(el):
    t = el.get("tags", {})
    try:
        h = float(str(t.get("height", "")).replace("m", "").split(";")[0])
        if h > 0: return min(h, 300)
    except ValueError: pass
    try:
        lv = float(str(t.get("building:levels", "")).split(";")[0])
        if lv > 0: return min(lv * 3.2 + 1.5, 300)
    except ValueError: pass
    b = t.get("building", "yes")
    if b in ("house", "detached", "garage", "shed", "roof", "kiosk", "hut", "service"): return 5.0
    return 9.0 + (zlib.crc32(str(el["id"]).encode()) % 7)


t0 = time.time()
B = q('way["building"]BB;relation["building"]BB;'); print("buildings", len(B), round(time.time() - t0))
G = q('way["leisure"~"park|garden|golf_course|pitch|nature_reserve|playground"]BB;relation["leisure"~"park|garden|nature_reserve"]BB;'
      'way["landuse"~"grass|forest|recreation_ground|meadow|orchard|village_green|cemetery|allotments"]BB;'
      'relation["landuse"~"grass|forest|recreation_ground|meadow"]BB;way["natural"~"wood|scrub|grassland|beach|sand"]BB;'
      'way["place"="square"]BB;'); print("green", len(G))
Wt = q('way["natural"="water"]BB;relation["natural"="water"]BB;way["waterway"]BB;way["natural"="coastline"]BB;'
       'way["landuse"~"basin|reservoir"]BB;'); print("water", len(Wt))
R = q('way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|secondary|tertiary|residential|unclassified|living_street|pedestrian)$"]BB;'
      'way["railway"="rail"]BB;'); print("roads", len(R))

# ---------------------------------------------------------------- height
Hh = np.zeros((PH, PW), np.float32)
for el in sorted(B, key=hgt):
    fill(Hh, el, float(hgt(el)))
cv2.imwrite(f"{OUT}/height.png", np.clip(Hh * 10, 0, 65535).astype(np.uint16))

# ---------------------------------------------------------------- ground
LAND, SEA, PARK, PARK2, WATER, ROAD, ROAD2, RAIL, SAND = ((48, 30, 18), (84, 52, 20), (66, 150, 52), (58, 120, 46), (214, 150, 52),
                                                          (92, 72, 58), (74, 58, 46), (120, 104, 92), (90, 120, 140))
gnd = np.zeros((PH, PW, 3), np.uint8); gnd[:] = LAND
# sea: coastline as a barrier, flood fill from the west edge
bar = np.zeros((PH + 2, PW + 2), np.uint8)
coast = [el for el in Wt if el.get("tags", {}).get("natural") == "coastline"]
cm = np.zeros((PH, PW), np.uint8)
for el in coast: line(cm, el, 255, 3)
bar[1:-1, 1:-1] = cm
seeds = [(2, y) for y in range(10, PH - 10, max(1, PH // 40)) if cm[y, 2] == 0]
img = np.zeros((PH, PW), np.uint8)
best = None
for sx, sy in seeds[:1]:
    m2 = bar.copy(); cv2.floodFill(img, m2, (sx, sy), 255, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    best = (m2[1:-1, 1:-1] == 255) & (cm == 0)
frac = float(best.mean()) if best is not None else 0
print("sea fraction", round(frac, 3))
if best is None or frac > 0.45 or frac < 0.02:      # leak: fall back to the per-row first coastline crossing from the west
    best = np.zeros((PH, PW), bool)
    for y in range(PH):
        xs = np.nonzero(cm[y])[0]
        if len(xs): best[y, :xs[0]] = True
    print("sea fallback", round(float(best.mean()), 3))
gnd[best] = SEA
for el in G:
    t = el.get("tags", {}); v = t.get("leisure") or t.get("landuse") or t.get("natural") or t.get("place")
    col = SAND if v in ("beach", "sand") else (PARK2 if v in ("pitch", "playground", "cemetery", "allotments", "square") else PARK)
    fill(gnd, el, col)
for el in Wt:
    t = el.get("tags", {})
    if t.get("natural") == "water" or t.get("landuse") in ("basin", "reservoir"): fill(gnd, el, WATER)
for el in Wt:
    t = el.get("tags", {}); ww = t.get("waterway")
    if ww in ("river", "canal"):
        try: wd = float(str(t.get("width", "20")).split(";")[0].replace("m", ""))
        except ValueError: wd = 20.0
        line(gnd, el, WATER, max(5, wd / MPP))
    elif ww in ("stream", "drain", "ditch"): line(gnd, el, WATER, 3)
RW = dict(motorway=26, motorway_link=8, trunk=16, trunk_link=6, primary=12, secondary=10, tertiary=8, residential=5, unclassified=5, living_street=4, pedestrian=4)
for el in sorted(R, key=lambda e: RW.get(e.get("tags", {}).get("highway"), 0)):
    t = el.get("tags", {})
    if t.get("railway") == "rail": line(gnd, el, RAIL, 3); continue
    hw = t.get("highway"); line(gnd, el, ROAD if RW.get(hw, 4) >= 8 else ROAD2, RW.get(hw, 4) / MPP * 1.0)
cv2.imwrite(f"{OUT}/ground.png", gnd)

# ---------------------------------------------------------------- masks
def nm(t): return " ".join(str(t.get(k, "")) for k in ("name", "name:en", "name:he", "alt_name", "official_name"))

Z = lambda: np.zeros((PH, PW), np.uint8)
yark, ayal, wline, blvd, small, allpark = Z(), Z(), Z(), Z(), Z(), Z()
YARKON = ("Yarkon Park", "HaYarkon Park", "פארק הירקון", "גני יהושע", "Ganei Yehoshua")
for el in G:
    t = el.get("tags", {})
    fill(allpark, el, 255)
    if any(k in nm(t) for k in YARKON): fill(yark, el, 255)
    if t.get("leisure") in ("park", "garden") or t.get("place") == "square":
        outs, _ = rings(el)
        area = sum(abs(cv2.contourArea(r.astype(np.float32))) for r in outs if len(r) >= 3) * MPP * MPP
        if area < 40000: fill(small, el, 255)
for el in R:
    t = el.get("tags", {})
    if t.get("highway", "").startswith("motorway") and (str(t.get("ref", "")).split(";")[0] == "20" or "איילון" in nm(t) or "Ayalon" in nm(t)):
        line(ayal, el, 255, 30 / MPP)
BLVD_ONLY = ("Rothschild", "רוטשילד", "Ben Gurion", "בן גוריון", "Chen", "ח\"ן", "Ben Tsiyon", "Ben Zion", "בן ציון", "Nordau", "נורדאו")
for el in R:
    t = el.get("tags", {}); n_ = nm(t)
    if any(k in n_ for k in BLVD_ONLY) and any(k in n_ for k in ("Boulevard", "Blvd", "שדרות", "Sderot", "Avenue")):
        line(blvd, el, 255, 16 / MPP)
for el in Wt:
    t = el.get("tags", {}); ww = t.get("waterway"); n_ = nm(t)
    if ww in ("river", "stream", "canal", "drain"):
        line(wline, el, 255, max(4, 24 / MPP))
        if "איילון" in n_ or "Ayalon" in n_ or "Ayyalon" in n_: line(ayal, el, 255, 30 / MPP)
    if t.get("natural") == "water" and ("ירקון" in n_ or "Yarkon" in n_): fill(wline, el, 255)
near = cv2.dilate(ayal, np.ones((int(120 / MPP) | 1,) * 2, np.uint8))
for el in R:   # rail inside the Ayalon corridor (within 120 m of the motorway)
    if el.get("tags", {}).get("railway") == "rail":
        tmp = Z(); line(tmp, el, 255, 3); ayal = np.maximum(ayal, ((tmp > 0) & (near > 0)).astype(np.uint8) * 255)
cv2.imwrite(f"{OUT}/masks.png", np.stack([wline, ayal, yark], -1))       # BGR file: R = Yarkon park, G = Ayalon, B = water lines
cv2.imwrite(f"{OUT}/masks2.png", np.stack([allpark, small, blvd], -1))   # R = boulevards, G = small gardens, B = all parks
M1 = np.stack([wline, ayal, yark], -1)

names = sorted({el.get("tags", {}).get("name:en") or el.get("tags", {}).get("name", "") for el in G + Wt
                if el.get("tags", {}).get("leisure") == "park" or el.get("tags", {}).get("waterway") in ("river", "stream", "canal", "drain")})
tall = sorted(((hgt(el), el.get("tags", {}).get("name:en") or el.get("tags", {}).get("name", ""), el["id"]) for el in B), reverse=True)[:25]
yk = (M1[..., 2] > 0).sum() * MPP * MPP / 1e6
json.dump(dict(name=name, bbox=[S, Wl, N, E], w=PW, h=PH, mpp=MPP, lat0=LAT0, m_lat=MLAT, m_lon=MLON, buildings=len(B), green=len(G),
               water=len(Wt), roads=len(R), yarkon_mask_km2=round(yk, 3), tallest=tall, park_and_river_names=names[:200],
               licence="Map data (c) OpenStreetMap contributors, ODbL", made=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())),
          open(f"{OUT}/meta.json", "w"), indent=1, ensure_ascii=False)
print("done", round(time.time() - t0), "s")
