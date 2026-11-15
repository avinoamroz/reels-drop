"""city_flyover: a 3D bird's-eye flight over a real city built from OpenStreetMap (buildings extruded from their footprints,
parks green, rivers and sea blue, roads and rail), dark brand look, the camera moving from frame 0. Beats highlight what is
engineered in the city: a park glows, a corridor floods, boulevards light up, a footprint of another place is laid over at true
scale, pins label places, photo cards slide in over the sky, fact cards count up. Hook from frame 0 over the city, "Israel
Facts" mark, end card, burned-in subtitles, OSM credit line, circular ending (the last camera key = the first one and the last
0.5 s dissolves into frame 0).

City rasters come from tools/osm_city.py (workflow osm-city.yml): templates/data/city_<name>/{height,ground,masks,masks2}.png.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  city      "templates/data/city_tlv"
  exag      vertical exaggeration of building heights (default 1.5)
  cam0      {lon, lat, dist, yaw, pitch}    frame-0 camera: look-at point, distance (m), yaw (deg, 0 = looking north,
                                            90 = east), pitch (deg from straight down)
  cams      [{at, dur, lon, lat, dist, yaw, pitch}]   keys (missing fields keep the previous value); eased between keys
  drift     {yaw: 20, dist: -0.12}           half-sine drift over the reel (deg, fraction of dist): moving from frame 0, zero at the end
  hl        [{at, until, layer, color, a, edge, pulse}]   layer: yarkon | ayalon | channel | water | blvd | gardens | parks
            tint on the ground (and a glowing edge line); color hex
  flood     {at, rise, spill, until}          the Ayalon channel fills (flowing stripes) from `at`, spills over the lanes at `spill`
  shapes    [{at, until, kind: "rect", lon, lat, w, h, rot, color, label, dash}]   true-scale footprint (metres, rot deg from east)
            or {kind: "poly", pts: [[lon, lat], ...], color, label, fill}
  pins      [{at, until, lon, lat, label, color}]
  photos    [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1]}]   card in the upper half (y0..y1 = photo_box)
  big       [{at, until, value, from, prefix, suffix, dec, sub, text}]    fact card (yellow number or text line + sub)
  stamp     [{at, until, text, color}]        rubber stamp (e.g. CLOSED)
  brand     {text, show: [[at, until], ...]}  "Israel Facts" mark (default: during the hook and the last 1.2 s)
  photo_box [y0, y1] default [360, 930];  title_y 300;  hook_y 640;  big_y 1230;  src_y 1488;  subs_y 1640;  osm_y 1752
"""
import os, sys, math, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

LAB = "#C9D6EA"; HAZE = np.array([70, 40, 20], np.float32)       # BGR, deep navy haze
FY = (H / 2) / math.tan(math.radians(25))                            # vertical fov 50 deg
KMAT = np.array([[FY, 0, W / 2], [0, FY, H / 2], [0, 0, 1]], np.float64)
LAYER_COL = dict(yarkon="#5BE37D", ayalon="#45A4DD", channel="#45A4DD", water="#45A4DD", blvd="#FFD23F", gardens="#5BE37D", parks="#5BE37D")


def hexbgr(h): r, g, b = rgb(h); return np.array([b, g, r], np.float32) * 255


class City:
    def __init__(s, path, exag):
        p = path if os.path.isabs(path) else os.path.join(REPO, path)
        s.meta = json.load(open(f"{p}/meta.json"))
        s.mpp = s.meta["mpp"]; s.S, s.Wl, s.N, s.E = s.meta["bbox"]; s.mlat, s.mlon = s.meta["m_lat"], s.meta["m_lon"]
        hd = cv2.imread(f"{p}/height.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 10.0
        s.h = np.clip(np.round(hd), 0, 255).astype(np.uint8); s.PH, s.PW = s.h.shape
        s.ground = cv2.imread(f"{p}/ground.png", cv2.IMREAD_COLOR)
        m1 = cv2.imread(f"{p}/masks.png", cv2.IMREAD_COLOR); m2 = cv2.imread(f"{p}/masks2.png", cv2.IMREAD_COLOR)
        wl, ay, yk = m1[..., 0], m1[..., 1], m1[..., 2]; ap, sm, bl = m2[..., 0], m2[..., 1], m2[..., 2]
        ch = ((wl > 0) & (ay > 0)).astype(np.uint8) * 255
        s.masks = dict(yarkon=yk, ayalon=ay, channel=ch, water=wl, blvd=bl, gardens=sm, parks=ap)
        s.exag = exag
        s.A = np.array([[s.mpp, 0, 0], [0, -s.mpp, s.PH * s.mpp], [0, 0, 1]], np.float64)     # tex px -> world metres
        lv = np.concatenate([np.arange(2, 30, 1.5), np.arange(30, 80, 3), np.arange(80, 256, 6)])
        s.levels = []
        for k, z in enumerate(lv):
            zi = int(math.ceil(z)); m = s.h >= zi
            if not m.any(): break
            ys, xs = np.nonzero(m[::4, ::4]); y0, y1 = max(0, ys.min() * 4 - 4), min(s.PH, ys.max() * 4 + 8); x0, x1 = max(0, xs.min() * 4 - 4), min(s.PW, xs.max() * 4 + 8)
            s.levels.append((zi, (x0, y0, x1, y1)))
        s.levels = [(z, bb, s.levels[i + 1][0] if i + 1 < len(s.levels) else 999) for i, (z, bb) in enumerate(s.levels)]
        y = np.arange(s.PH, dtype=np.float32)[:, None]
        s.phase = np.repeat(((y * s.mpp / 30.0) % 1.0 * 255).astype(np.uint8), s.PW, 1)   # flow stripes along north

    def world(s, lon, lat): return ((lon - s.Wl) * s.mlon, (lat - s.S) * s.mlat)


class Cam:
    def __init__(s, C, tx, ty, dist, yaw, pitch):
        ps, ys = math.radians(pitch), math.radians(yaw)
        f = np.array([math.sin(ps) * math.sin(ys), math.sin(ps) * math.cos(ys), -math.cos(ps)])
        r = np.array([math.cos(ys), -math.sin(ys), 0.0]); u = np.cross(r, f)
        s.R = np.stack([r, -u, f]); s.Cw = np.array([tx, ty, 0.0]) - dist * f; s.t = -s.R @ s.Cw; s.C = C
    def Hz(s, z):
        """homography: texture pixel (u, v, 1) on the plane at world height z -> screen"""
        M = np.column_stack([s.R[:, 0], s.R[:, 1], s.R[:, 2] * z + s.t])
        return KMAT @ M @ s.C.A
    def proj(s, X, Y, Z=0.0):
        pc = s.R @ np.array([X, Y, Z]) + s.t
        if pc[2] <= 1: return None
        return (KMAT[0, 0] * pc[0] / pc[2] + W / 2, KMAT[1, 1] * pc[1] / pc[2] + H / 2, pc[2])


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    C = City(D.get("city", "templates/data/city_tlv"), float(D.get("exag", 1.5)))
    c0 = dict(D["cam0"]); keys = [(0.0, 0.01, c0)]
    for c in D.get("cams", []):
        k = dict(keys[-1][2]); k.update({x: c[x] for x in ("lon", "lat", "dist", "yaw", "pitch") if x in c}); keys.append((A(c["at"]), float(c.get("dur", 1.6)), k))
    DR = D.get("drift") or {}; DY, DD = float(DR.get("yaw", 20)), float(DR.get("dist", -0.12))

    def cam_at(t):
        cur = keys[0][2]
        for i, (t0, d, k) in enumerate(keys):
            if t < t0: break
            prev = keys[i - 1][2] if i else k; u = ease((t - t0) / d)
            cur = {x: prev[x] + (k[x] - prev[x]) * u if x != "dist" else math.exp(math.log(prev[x]) + (math.log(k[x]) - math.log(prev[x])) * u) for x in k}
            if u < 1: break
        # drift: a half sine over the whole reel, moving from frame 0 and back to zero at the end (circular)
        sn = math.sin(math.pi * min(1.0, t / max(1.0, T.END - 0.6)))
        tx, ty = C.world(cur["lon"], cur["lat"])
        return Cam(C, tx, ty, cur["dist"] * (1 + DD * sn), cur["yaw"] + DY * sn, cur["pitch"])

    HL = [dict(h, t0=A(h["at"]), t1=A(h.get("until"), T.END - 0.6)) for h in D.get("hl", [])]
    FL = D.get("flood"); FL = dict(FL, t0=A(FL["at"]), ts=A(FL.get("spill"), A(FL["at"]) + 3), t1=A(FL.get("until"), T.END - 0.6)) if FL else None
    SH = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("shapes", [])]
    PN = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("pins", [])]
    BG_ = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("big", [])]
    ST = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("stamp", [])]
    from photo_story import _load
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6), im=_load(p["file"])) for p in D.get("photos", [])]
    PB = D.get("photo_box", [360, 930]); PY0, PY1 = int(PB[0]), int(PB[1]); PX0, PX1 = 60, 1020; PW_, PH_ = PX1 - PX0, PY1 - PY0
    BR = D.get("brand") or {}; BRS = [(A(a), A(b)) for a, b in BR.get("show", [[0, "L0e+0.4"], ["END-1.2", "END"]])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    E = S.get("end") or {}; END0 = A(E.get("at"), T.END - 3) if E else T.END + 9; END1 = A(E.get("until"), T.END - 0.5) if E else T.END + 9
    TITLE_Y, HOOK_Y, BIG_Y = D.get("title_y", 300), D.get("hook_y", 640), D.get("big_y", 1230)
    WALL0, WALL1, GLASS, ROOF, ROOFG = hexbgr("#7F93B3"), hexbgr("#C3CFE2"), hexbgr("#6FA9DA"), hexbgr("#F2F5FA"), hexbgr("#BFE0F7")
    yy = np.arange(H, dtype=np.float32)[:, None]
    import map_counter as MC

    # rounded mask for photo cards
    msk = np.zeros((PH_, PW_), np.uint8); r = 34
    cv2.rectangle(msk, (r, 0), (PW_ - r, PH_), 255, -1); cv2.rectangle(msk, (0, r), (PW_, PH_ - r), 255, -1)
    for cx_, cy_ in ((r, r), (PW_ - r, r), (r, PH_ - r), (PW_ - r, PH_ - r)): cv2.circle(msk, (cx_, cy_), r, 255, -1)
    PM = (msk.astype(np.float32) / 255)[..., None]

    def kb(im, cx, cy, z):
        h_, w_ = im.shape[:2]; s_ = max(PW_ / w_, PH_ / h_) * max(1.0, z); cw, ch = PW_ / s_, PH_ / s_
        x0_ = min(max(cx * w_ - cw / 2, 0), w_ - cw); y0_ = min(max(cy * h_ - ch / 2, 0), h_ - ch)
        M_ = np.array([[s_, 0, -x0_ * s_], [0, s_, -y0_ * s_]], np.float32)
        return cv2.warpAffine(im, M_, (PW_, PH_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    def warp(src, Hm, interp=cv2.INTER_LINEAR, border=0):
        return cv2.warpPerspective(src, Hm, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def city(t):
        cam = cam_at(t); Hg = cam.Hz(0.0)
        img = warp(C.ground, Hg, cv2.INTER_LINEAR, (48, 30, 18)).astype(np.float32)
        # ground highlights
        for h in HL:
            a = win(t, h["t0"], h["t1"], 0.4, 0.4) * h.get("a", 0.75)
            if a <= 0.01: continue
            if h.get("pulse"): a *= 0.75 + 0.25 * math.sin((t - h["t0"]) * 5)
            m = warp(C.masks[h["layer"]], Hg).astype(np.float32)[..., None] / 255
            col = hexbgr(h.get("color", LAYER_COL.get(h["layer"], SKY)))
            img = img * (1 - m * a) + (img * 0.25 + col * 0.95) * (m * a)
            if h.get("edge", True):
                e = cv2.morphologyEx((m[..., 0] > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(np.float32)[..., None]
                e = cv2.GaussianBlur(e, (0, 0), 3)[..., None] * 1.6
                img = img + col * np.clip(e, 0, 1) * a * 0.9
        if FL:
            a = win(t, FL["t0"], FL["t1"], 0.4, 0.5)
            if a > 0.01:
                ph = warp(C.phase, Hg).astype(np.float32) / 255
                st = 0.8 + 0.2 * np.sin(2 * math.pi * (ph - t * 0.6))
                u = seg(t, FL["ts"], FL["ts"] + 1.4)
                m = warp(C.masks["channel"], Hg).astype(np.float32) / 255
                if u > 0.01:
                    ay = warp(C.masks["ayalon"], Hg).astype(np.float32) / 255
                    m = np.maximum(m, ay * u)
                m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]
                wcol = np.array([235, 170, 60], np.float32)
                img = img * (1 - m * a) + (wcol * st[..., None] + 25) * (m * a)
        # buildings: stacked slices, bottom to top
        for z, (x0, y0, x1, y1), znext in C.levels:
            Hz = cam.Hz(z * C.exag)
            Tr = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64); Hc = Hz @ Tr
            cr = np.array([[0, 0, 1], [x1 - x0, 0, 1], [0, y1 - y0, 1], [x1 - x0, y1 - y0, 1]], np.float64).T
            pc = Hc @ cr
            if (pc[2] <= 1e-6).any(): rx0, ry0, rx1, ry1 = 0, 0, W, H          # crop reaches behind the camera: whole screen
            else:
                sx, sy = pc[0] / pc[2], pc[1] / pc[2]
                rx0, ry0 = int(max(0, math.floor(sx.min()))), int(max(0, math.floor(sy.min())))
                rx1, ry1 = int(min(W, math.ceil(sx.max()) + 1)), int(min(H, math.ceil(sy.max()) + 1))
            if rx1 <= rx0 or ry1 <= ry0: continue
            Ho = np.array([[1, 0, -rx0], [0, 1, -ry0], [0, 0, 1]], np.float64) @ Hc
            hs = cv2.warpPerspective(C.h[y0:y1, x0:x1], Ho, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = hs >= z
            if not m.any(): continue
            roof = m & (hs < znext)
            g = min(1.0, max(0.0, (z - 40) / 50))
            wall = (WALL0 + (WALL1 - WALL0) * min(1.0, z / 28)) * (1 - g) + GLASS * g
            rf = ROOF * (1 - g) + ROOFG * g
            reg = img[ry0:ry1, rx0:rx1]
            reg[m] = wall; reg[roof] = rf
        # haze toward the top (distance) and a soft vignette
        f = np.clip((820 - yy) / 760, 0, 0.92)[..., None]
        img = img * (1 - f) + HAZE * f
        f2 = np.clip((yy - 1400) / 260, 0, 0.62)[..., None]                  # darker floor for sources, subtitles, credit
        img = img * (1 - f2) + HAZE * 0.6 * f2
        return img, cam

    def draw_shapes(cv, cam, t):
        for x in SH:
            a = win(t, x["t0"], x["t1"], 0.35, 0.35)
            if a <= 0.01: continue
            col = x.get("color", YEL)
            if x.get("kind", "rect") == "rect":
                cx, cy = C.world(x["lon"], x["lat"]); w_, h_ = x["w"] / 2, x["h"] / 2; ro = math.radians(x.get("rot", 0))
                cs = [(-w_, -h_), (w_, -h_), (w_, h_), (-w_, h_)]
                P = [(cx + px * math.cos(ro) - py * math.sin(ro), cy + px * math.sin(ro) + py * math.cos(ro)) for px, py in cs]
            else:
                P = [C.world(lo, la) for lo, la in x["pts"]]
            u = seg(t, x["t0"], x["t0"] + 0.9, eout)
            pts = [cam.proj(X, Y, 1.0) for X, Y in P]
            if any(p is None for p in pts): continue
            pts = [(p[0], p[1]) for p in pts]
            # grow the outline from a point
            mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
            pts = [(mx + (p[0] - mx) * (0.3 + 0.7 * u), my + (p[1] - my) * (0.3 + 0.7 * u)) for p in pts]
            if x.get("fill", 0.18) > 0:
                c = cv.c; c.new_path(); c.move_to(*pts[0])
                for p in pts[1:]: c.line_to(*p)
                c.close_path(); cv.col(col, x.get("fill", 0.18) * a); c.fill()
            cv.line(pts + [pts[0]], col, 7, a, dash=[22, 14] if x.get("dash", True) else None)
            if x.get("label"):
                lx = x.get("label_xy"); lx = cam.proj(*C.world(*lx), 1.0)[:2] if lx else (mx, min(p[1] for p in pts) - 50)
                WG.tag(cv, x["label"], lx[0], lx[1], 46, "#0B1730", a, bg=col)

    def draw_pins(cv, cam, t):
        for p in PN:
            a = win(t, p["t0"], p["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            q = cam.proj(*C.world(p["lon"], p["lat"]), 0.0)
            if q is None or not (-50 < q[0] < W + 50 and 100 < q[1] < H - 200): continue
            s = min(1.0, WG.pop_s(t, p["t0"])); col = p.get("color", YEL)
            cv.line([(q[0], q[1]), (q[0], q[1] - 110 * s)], WHITE, 4, a * 0.9)
            cv.circ(q[0], q[1], 12 * s, col, a, stroke=WHITE, lw=3)
            lab = p["label"]; sz = 48; tw = min(cv.tw(lab, sz) + 50, 900); x = min(max(q[0], tw / 2 + 20), W - 20 - tw / 2)
            cv.rrect(x - tw / 2 * s, q[1] - 110 * s - 76, tw * s, 72, 36, "#0B1730", 0.86 * a, stroke=col, lw=3, sa=a)
            cv.text(lab, x, q[1] - 110 * s - 40, sz * s, WHITE, a, maxw=tw - 30, minsz=36, shadow=False)

    def draw_photos(base, t):
        img, tot = None, 0.0; cur = None
        for p in PH:
            a_ = win(t, p["t0"], p["t1"], 0.35, 0.35)
            if a_ <= 0.001: continue
            k = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            ph = kb(p["im"], k[0] + (k[3] - k[0]) * u, k[1] + (k[4] - k[1]) * u, k[2] + (k[5] - k[2]) * u)
            img = ph * a_ if img is None else img * (1 - a_) + ph * a_; tot = 1 - (1 - tot) * (1 - a_); cur = (p, a_)
        if img is None: return base, None
        sl = (1 - eout(tot)) * 60
        y0 = int(PY0 + sl); y1 = y0 + PH_
        reg = base[y0:y1, PX0:PX1]; m = PM * tot
        base[y0:y1, PX0:PX1] = reg * (1 - m) + (img / max(tot, 1e-3)) * m
        return base, (cur, y0, y1)

    def draw_big(cv, t):
        for f in BG_:
            a = win(t, f["t0"], f["t1"], 0.25, 0.25)
            if a <= 0.01: continue
            big = f.get("text")
            if f.get("value") is not None:
                v = WG.roll(t, f["t0"] + 0.1, f.get("from", 0), f["value"], f.get("dur", 0.9))
                big = WG.fmtn(v, f.get("dec", 0), f.get("prefix", ""), f.get("suffix", ""))
            p = min(1.0, WG.pop_s(t, f["t0"])); h = 250 if f.get("sub") else 150; y0 = BIG_Y - h / 2
            WG.panel(cv, 50, y0, 980, h, a)
            if f.get("sub"):
                cv.text(big, 540, y0 + 86, 104 * (0.85 + 0.15 * p), YEL, a, maxw=920, minsz=60)
                WG.title(cv, f["sub"], y0 + 188, a, 60, maxw=920)
            else: cv.text(big, 540, BIG_Y, 92 * (0.85 + 0.15 * p), YEL, a, maxw=920, minsz=60)

    def draw_stamp(cv, t):
        for s_ in ST:
            a = win(t, s_["t0"], s_["t1"], 0.08, 0.3)
            if a <= 0.01: continue
            u = seg(t, s_["t0"], s_["t0"] + 0.18, eout); sc = 1.6 - 0.6 * u
            col = {"red": RED, "green": GREEN, "yellow": YEL}.get(s_.get("color", "red"), RED)
            c = cv.c; c.save(); c.translate(540, s_.get("y", 1000)); c.rotate(math.radians(s_.get("angle", -8))); c.scale(sc, sc)
            tw = cv.tw(s_["text"], 120) + 80
            cv.rrect(-tw / 2, -90, tw, 180, 26, None, 1, stroke=col, lw=12, sa=a * u)
            cv.text(s_["text"], 0, 0, 120, col, a * u, shadow=False, maxw=1000)
            c.restore()

    def end_card(cv, t):
        if not E: return
        a = win(t, END0 - 0.1, END1, 0.35, 0.3)
        if a <= 0.01: return
        y = E.get("y", 1060)
        WG.panel(cv, 50, y - 250, 980, 520, a)
        WG.title(cv, E["balance"], y - 140, a, 62, maxw=920)
        if E.get("balance2"): WG.title(cv, E["balance2"], y + 10, a * seg(t, END0 + 0.4, END0 + 0.8), 54, col=LAB, maxw=920)
        if E.get("cta"):
            ca = a * seg(t, END0 + 0.6, END0 + 1.0); txt = E["cta"].replace("*", ""); sz = 58; tw = min(cv.tw(txt, sz) + 90, 940)
            cv.rrect(540 - tw / 2, y + 120, tw, 112, 56, SKY, ca); cv.text(txt, 540, y + 176, sz, WHITE, ca, maxw=900, minsz=40)

    def frame(t):
        img, cam = city(t)
        img, phinfo = draw_photos(img, t)
        cv = Canvas(img)
        draw_shapes(cv, cam, t); draw_pins(cv, cam, t)
        if phinfo:
            (p, a_), y0, y1 = phinfo
            if p.get("credit"):
                txt = "Photo: " + p["credit"]; tw = min(PW_ - 30, cv.tw(txt, 36, bold=False) + 30)
                cv.rrect(540 - tw / 2, y1 - 62, tw, 50, 14, "#0B1322", 0.75 * a_)
                cv.text(txt, 540, y1 - 37, 36, "#E6EEF8", a_, bold=False, maxw=PW_ - 40, minsz=36, shadow=False)
        dime = win(t, END0 - 0.1, END1, 0.35, 0.3) if E else 0
        for txt, t0, t1, yy_, ss in cards:
            a = win(t, t0 - 0.1, t1, 0.3, 0.3) * (1 - dime)
            if a <= 0.01: continue
            cv.rrect(40, (yy_ or TITLE_Y) - 78, 1000, 156, 34, "#0B1730", 0.62 * a)
            WG.title(cv, txt.replace("|", " "), yy_ or TITLE_Y, a, ss or 72, maxw=940)
        draw_big(cv, t); draw_stamp(cv, t)
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); end_card(cv, t)
        bra = max(((1 - seg(t, b_ - 0.3, b_)) if a_ <= 0.01 and t <= b_ else win(t, a_, b_, 0.2, 0.3) for a_, b_ in BRS), default=0)
        if bra > 0.01: MC.draw_brand(cv, BR.get("text", "Israel Facts"), bra)
        osm = D.get("osm_credit", "Map data: OpenStreetMap contributors")
        cv.text(osm, 540, D.get("osm_y", 1752), 36, "#AFC0D8", 0.85, bold=False, maxw=1000, shadow=True)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HOOK_Y, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(k[0], "whoosh", -30) for k in keys[1:]] + [(p["t0"], "pop", -28) for p in PN] + [(x["t0"], "whoosh", -32) for x in SH]
    ev += [(f["t0"], "ding", -28) for f in BG_] + [(s_["t0"], "thud", -22) for s_ in ST] + [(p["t0"], "whoosh", -34) for p in PH if p["t0"] > 0.5]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, city=C.meta.get("name"), osm_made=C.meta.get("made"), buildings=C.meta.get("buildings"),
                photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
