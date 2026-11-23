"""israel_fits: "How many times does Israel fit into X?" True-scale copies of Israel dropped onto real places, a big yellow
counter ticking up ("x 1, x 2 ... x 97"), camera flights between places. Builder X, 23 Nov 2026.

Projection: Lambert azimuthal equal-area centred on the camera (true area everywhere, true shape near the centre), so no
Mercator inflation. Every copy of Israel is placed in its own local equal-area plane at its centre and only then projected,
so each copy is the true ground size for its latitude (Greenland, Russia, Antarctica included).
Israel = 22,072 km2 (CBS: incl. Golan Heights and East Jerusalem; outline scaled to that area). Home map: whole land of Israel,
Israel in sky blue with a white outline, West Bank and Gaza in a slightly different shade with the Green Line in green.
Tile centres: templates/data/israel_fits_geo.json (equal-area cells per target, built offline from Natural Earth 50m).

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  cam0      [lon, lat, width_km, screen_y]        frame 0 view (screen_y: where the view centre sits, default 1000)
  drift     [d_width_frac, secs]                   slow push in from frame 0 (default [0.12, 1.6]): the map moves at once
  cams      [{at, view: [lon, lat, width_km, screen_y], dur, bump}]  camera flights (bump: extra zoom out mid flight, log units)
  beats     [{target, at, until, n, value, dec, name, sub, hero, hero_gap, fill, fly, clip}]
             target: key in israel_fits_geo.json targets; at: first drop; n: tiles (default: all); value: final counter (e.g. 31.5)
             hero: slow first drops (default 3), hero_gap s between them; fill: s for the rest (eased, accelerating)
             fly: [lift_at, land_at] the first copy lifts off Israel and flies to the target (true size all the way)
  home      [[at, until], ...]  windows with "Israel" / "West Bank" / "Gaza" tags on the home map
  tags      [{at, until, lon, lat, text, size, col}]   map tags
  counter_y y of the counter block (default 330)
  title_y   y of the line screen cards (default 1300)
  overlays / brand / hook_y / subs_y as in map_counter
"""
import os, sys, math, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
import common as CM
from map_counter import draw_overlay, draw_brand

R = 6371.0072
GEOF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "israel_fits_geo.json")
LAND, LAND_B, OCEAN0, OCEAN1 = "#1D4579", "#3C70B0", "#07142E", "#0C1D40"
TGT_FILL, TGT_LINE = "#2A5E9C", "#8FD0F5"
ISR_FILL, TERR_FILL = SKY, "#2C7DB3"


def fwd(lon, lat, lon0, lat0):
    """Lambert azimuthal equal-area (sphere). Returns x, y (km, y north) and cos(c) (angular distance) for culling."""
    l = np.radians(lon); p = np.radians(lat); l0 = math.radians(lon0); p0 = math.radians(lat0)
    cc = math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0)
    k = np.sqrt(2 / np.maximum(1 + np.maximum(cc, -0.6), 1e-9))
    return R * k * np.cos(p) * np.sin(l - l0), R * k * (math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(l - l0)), cc


def inv(x, y, lon0, lat0):
    l0, p0 = math.radians(lon0), math.radians(lat0)
    rho = np.hypot(x, y); c = 2 * np.arcsin(np.clip(rho / (2 * R), -1, 1)); rho = np.where(rho == 0, 1e-12, rho)
    lat = np.arcsin(np.clip(np.cos(c) * math.sin(p0) + y * np.sin(c) * math.cos(p0) / rho, -1, 1))
    lon = l0 + np.arctan2(x * np.sin(c), rho * math.cos(p0) * np.cos(c) - y * math.sin(p0) * np.sin(c))
    return (np.degrees(lon) + 540) % 360 - 180, np.degrees(lat)


def slerp(a, b, u):
    """great circle point between (lon, lat) a and b"""
    def v(p): l, q = math.radians(p[0]), math.radians(p[1]); return np.array([math.cos(q) * math.cos(l), math.cos(q) * math.sin(l), math.sin(q)])
    A, B = v(a), v(b); w = math.acos(max(-1, min(1, float(A @ B))))
    if w < 1e-6: return a
    P = (math.sin((1 - u) * w) * A + math.sin(u * w) * B) / math.sin(w)
    return (math.degrees(math.atan2(P[1], P[0])), math.degrees(math.asin(max(-1, min(1, P[2])))))


def ang(a, b):
    l1, p1, l2, p2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return math.degrees(math.acos(max(-1, min(1, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(l1 - l2)))))


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    G = json.load(open(GEOF))
    TXY = np.array(G["tile_xy"], float)                                   # Israel outline, km, centred
    ICEN = tuple(G["israel_centroid"])

    def rings_ll(rs): return [np.array(r, float) for r in rs]
    WORLD = [r for rs in G["world"].values() for r in rings_ll(rs)]
    for i, r in enumerate(WORLD):                                         # Antarctica seam: drop pole and +-180 seam points
        if r[:, 1].min() < -60:
            keep = ~((r[:, 1] < -89.5) | ((np.abs(r[:, 0]) > 179.9) & (r[:, 1] < -60)))
            WORLD[i] = r[keep]
    LAKES = [r for rs in G["lakes"].values() for r in rings_ll(rs)]
    HOME_I = rings_ll(G["isr49"]) + rings_ll(G["golan"]); HOME_T = rings_ll(G["wb"]) + rings_ll(G["gaza"])

    # ---------------------------------------------------------------- camera
    def cam(v): return np.array([float(v[0]), float(v[1]), math.log(float(v[2])), float(v[3]) if len(v) > 3 else 1000.0])
    K0 = cam(D["cam0"]); dr = D.get("drift", [0.12, 1.6])
    keys = []
    prev = K0.copy(); prev[2] += math.log(1 - dr[0])
    for c in D.get("cams", []):
        k = cam(c["view"]); t0 = A(c["at"]); d = float(c.get("dur", 1.3))
        bump = c.get("bump"); bump = (0.9 * min(1.0, ang(prev[:2], k[:2]) / 50.0)) if bump is None else float(bump)
        keys.append((t0, d, k, bump)); prev = k

    def cam_at(t):
        cur = K0.copy(); cur[2] = K0[2] + math.log(1 - dr[0] * ease(min(1.0, t / dr[1])))
        for t0, d, k, bump in keys:
            if t < t0: break
            u = ease((t - t0) / d); a0 = cur.copy()
            lon0, lat0 = a0[0], a0[1]; dl = ((k[0] - lon0 + 540) % 360) - 180
            if abs(lat0) > 89 or abs(k[1]) > 89: dl = k[0] - lon0          # pole views: rotate plainly
            cur = np.array([lon0 + dl * u, lat0 + (k[1] - lat0) * u, a0[2] + (k[2] - a0[2]) * u + bump * math.sin(math.pi * u), a0[3] + (k[3] - a0[3]) * u])
            if u < 1: return cur
            cur = k.copy()
        return cur

    # ---------------------------------------------------------------- beats and tiles
    def tile_ll(lon, lat): return inv(TXY[:, 0], TXY[:, 1], lon, lat)
    BEATS = []
    for b in D.get("beats", []):
        tg = G["targets"][b["target"]]; cen = tg["tiles"][:int(b.get("n", len(tg["tiles"])))]
        n = len(cen); t0 = A(b["at"]); hero = min(n, int(b.get("hero", 3))); hg = float(b.get("hero_gap", 0.42)); fill = float(b.get("fill", 2.2))
        times = [A(x) for x in b["hero_at"]][:hero] if b.get("hero_at") else [t0 + i * hg for i in range(hero)]
        rest = n - hero; ts = (times[-1] + 0.3) if b.get("hero_at") else t0 + hero * hg
        for j in range(rest):
            u = (j + 1) / rest; times.append(ts + fill * (u ** 0.55))      # accelerating: many tiles land per frame at the end
        tiles = [np.c_[tile_ll(lo, la)] for lo, la in cen]
        fly = [A(x) for x in b["fly"]] if b.get("fly") else None
        if fly and not b.get("hero_at"): times[0] = fly[1]; times[1:] = [max(x, fly[1] + 0.35 + i * 0.01) for i, x in enumerate(times[1:])]
        if fly: fly[1] = times[0]
        t_end = A(b.get("until"), T.END - 0.6)
        BEATS.append(dict(b, cen=cen, n=n, times=times, tiles=tiles, t0=t0, t1=t_end, fly=fly, rings=rings_ll(tg["rings"]),
                          tfull=max(times), stack=np.concatenate(tiles), lens=[len(x) for x in tiles]))
    HLT = [dict(h, t0=A(h["at"]), t1=A(h.get("until"), T.END), rings=rings_ll(G["targets"][h["target"]]["rings"])) for h in D.get("highlight", [])]
    TAGS = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("tags", [])]
    HOMEW = [(A(w_[0]), A(w_[1])) for w_ in D.get("home", [])]
    OVL = [dict(o, t0=A(o["at"]), t1=A(o.get("until"), T.END - 0.6)) for o in D.get("overlays", [])]
    BR = D.get("brand") or {}; BRW = [(A(w_[0]), A(w_[1])) for w_ in BR.get("show", [])]
    DIM = [(A(d_["at"]), A(d_.get("until"), T.END)) for d_ in D.get("dim", [])]
    if S.get("end"): DIM.append((A(S["end"].get("at"), T.S(len(T.L) - 2)), A(S["end"].get("until"), T.END)))
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 700)
    CY = D.get("counter_y", 330); TY = D.get("title_y", 1300)

    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 620, 0, 1) ** 1.6 * 0.55, np.clip((y_ - 1240) / 680, 0, 1) ** 1.2 * 0.62))[..., None]
    OCEAN = (np.array(rgb(OCEAN0)[::-1]) * 255 * (1 - y_[..., None] / H) + np.array(rgb(OCEAN1)[::-1]) * 255 * (y_[..., None] / H)).astype(np.float32)
    OCEAN = np.broadcast_to(OCEAN, (H, W, 3)).copy()

    def project(st):
        lon0, lat0, lw, sy = st; s = W / math.exp(lw)
        def P(ll):
            x, y, cc = fwd(ll[:, 0], ll[:, 1], lon0, lat0)
            return np.c_[540 + x * s, sy - y * s], cc
        return P, s

    def path(cv, pts):
        c = cv.c; c.move_to(*pts[0])
        for p in pts[1:]: c.line_to(p[0], p[1])
        c.close_path()

    def visible(pts, cc):
        if cc.max() < -0.2: return False
        return not (pts[:, 0].max() < -50 or pts[:, 0].min() > W + 50 or pts[:, 1].max() < -50 or pts[:, 1].min() > H + 50)

    def draw_rings(cv, rs, P, fill, fa, stroke=None, sa=0.0, lw=1.2):
        c = cv.c; c.new_path(); any_ = False
        for r in rs:
            if len(r) < 3: continue
            pts, cc = P(r)
            if visible(pts, cc): path(cv, pts); any_ = True
        if not any_: return
        c.set_fill_rule(cairo.FILL_RULE_WINDING)
        if fill: cv.col(fill, fa); c.fill_preserve() if stroke else c.fill()
        if stroke: cv.col(stroke, sa); c.set_line_width(lw); c.set_line_join(cairo.LINE_JOIN_ROUND); c.stroke()
        c.new_path()

    def graticule(cv, P, s):
        c = cv.c; c.new_path()
        for lon in range(-180, 180, 10):
            ll = np.c_[np.full(61, lon), np.linspace(-85, 85, 61)]; pts, cc = P(ll); ok = cc > 0.05
            if ok.sum() < 2: continue
            first = True
            for p, o in zip(pts, ok):
                if not o: first = True; continue
                (c.move_to if first else c.line_to)(p[0], p[1]); first = False
        for lat in range(-80, 81, 10):
            ll = np.c_[np.linspace(-180, 180, 121), np.full(121, lat)]; pts, cc = P(ll); ok = cc > 0.05
            first = True
            for p, o in zip(pts, ok):
                if not o: first = True; continue
                (c.move_to if first else c.line_to)(p[0], p[1]); first = False
        cv.col(SKY, 0.09); c.set_line_width(1.2); c.stroke(); c.new_path()

    def tile_path(cv, pts, cx, cy, sc):
        c = cv.c; q = (pts - [cx, cy]) * sc + [cx, cy]; c.move_to(*q[0])
        for p in q[1:]: c.line_to(p[0], p[1])
        c.close_path()

    def draw_beat(cv, b, t, P, s):
        ba = win(t, min(b["t0"], b["fly"][0] if b["fly"] else 1e9) - 0.6, b["t1"], 0.4, 0.35)
        if ba <= 0.01: return
        draw_rings(cv, b["rings"], P, TGT_FILL, 0.95 * ba, TGT_LINE, 0.95 * ba, max(2.0, min(4.0, s * 8)))
        pts_all, cc = P(b["stack"]); lw = max(1.0, min(2.6, s * 5.0)); c = cv.c
        if b.get("clip", b["n"] > 2):
            c.save(); c.new_path()
            for r in b["rings"]:
                pr, ccr = P(r)
                if visible(pr, ccr): path(cv, pr)
            c.clip()
        i0 = 0
        for k, (L, tk) in enumerate(zip(b["lens"], b["times"])):
            pts = pts_all[i0:i0 + L]; i0 += L
            if t < tk - 0.25 or (b["fly"] and k == 0 and t < b["fly"][0]): continue
            if b["fly"] and k == 0 and t < tk: continue                     # drawn by the flight
            hero = k < int(b.get("hero", 3))
            u = (t - (tk - 0.22)) / 0.22
            if u < 0: continue
            sc = 1 + (0.45 if hero else 0.3) * (1 - eout(min(1, u))); al = min(1, u * 1.6) * ba
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean() - (70 * (1 - eout(min(1, u))) if hero else 0)
            c.new_path(); tile_path(cv, pts + [0, cy - pts[:, 1].mean()], cx, cy, sc)
            cv.col(ISR_FILL, 0.86 * al); c.fill_preserve(); cv.col("#FFFFFF", 0.95 * al); c.set_line_width(lw); c.set_line_join(cairo.LINE_JOIN_ROUND); c.stroke()
        if b.get("clip", b["n"] > 2): c.restore()
        if b["fly"] and b["fly"][0] <= t < b["times"][0]:                  # the first copy lifts off Israel and flies there, true size
            u = ease((t - b["fly"][0]) / max(0.1, b["times"][0] - b["fly"][0]))
            lo, la = slerp(ICEN, tuple(b["cen"][0]), u); ll = np.c_[tile_ll(lo, la)]; pts, _ = P(ll)
            lift = math.sin(math.pi * u) * 26
            c.new_path(); tile_path(cv, pts, pts[:, 0].mean(), pts[:, 1].mean(), 1.0 + 0.06 * math.sin(math.pi * u))
            cv.col("#000000", 0.35 * ba); c.save(); c.translate(lift * 0.5, lift); c.fill(); c.restore()
            c.new_path(); tile_path(cv, pts - [0, lift], pts[:, 0].mean(), pts[:, 1].mean() - lift, 1.0 + 0.06 * math.sin(math.pi * u))
            cv.col(ISR_FILL, 0.95 * ba); c.fill_preserve(); cv.col("#FFFFFF", ba); c.set_line_width(max(1.4, lw)); c.stroke()

    def draw_counter(cv, b, t):
        a = win(t, b["t0"] - 0.35, b["t1"], 0.3, 0.3)
        if a <= 0.01: return
        landed = sum(1 for x in b["times"] if x <= t)
        final = b.get("value", b["n"]); dec = int(b.get("dec", 0))
        if t >= b["tfull"] + 0.25:
            v = WG.roll(t, b["tfull"] + 0.25, landed, final, 0.5) if abs(final - landed) > 1e-6 else final
            txt = "x " + WG.fmtn(v, dec)
        else:
            txt = "x " + (f"{landed:,}" if landed else "0")
        pulse = 1 + 0.06 * (1 - eout(min(1, (t - max([x for x in b["times"] if x <= t] or [-9])) / 0.18)))
        cv.rrect(90, CY - 140, 900, 360, 44, "#0B1A3C", 0.72 * a, stroke=SKY, lw=2.5, sa=0.5 * a)
        cv.text(b.get("name", ""), 540, CY - 70, 66, WHITE, a, maxw=840)
        c = cv.c; c.save(); c.translate(540, CY + 60); c.scale(pulse, pulse); c.translate(-540, -(CY + 60))
        cv.text(txt, 540, CY + 60, 150, YEL, a, maxw=840); c.restore()
        cv.text(b.get("sub", "true-size Israels"), 540, CY + 170, 44, "#C9D6EA", a, bold=False, maxw=840)

    def frame(t):
        st = cam_at(t); P, s = project(st)
        cv = Canvas(OCEAN); graticule(cv, P, s)
        draw_rings(cv, WORLD, P, LAND, 1.0, LAND_B, 0.55, max(0.8, min(1.6, s * 3)))
        for h in HLT:
            ha = 1.0 if h["t0"] <= 0.01 and t < h["t1"] - 0.3 else win(t, h["t0"], h["t1"], 0.3, 0.4)
            if ha > 0.01: draw_rings(cv, h["rings"], P, TGT_FILL, 0.95 * ha, TGT_LINE, 0.9 * ha, max(2.0, min(4.0, s * 8)))
        draw_rings(cv, LAKES, P, OCEAN1, 1.0, LAND_B, 0.4, 1.0)
        for b in BEATS: draw_beat(cv, b, t, P, s)
        draw_rings(cv, HOME_T, P, TERR_FILL, 1.0, GREEN, 0.95, max(1.5, min(4.0, s * 6)))
        draw_rings(cv, HOME_I, P, ISR_FILL, 1.0, "#FFFFFF", 0.95, max(1.2, min(3.5, s * 5)))
        for w0, w1 in HOMEW:
            ha = win(t, w0, w1, 0.3, 0.3)
            if ha > 0.01:
                for lab, lo, la, dx in (("Israel", 34.9, 30.7, 200), ("Judea and Samaria|(West Bank)", 35.3, 32.0, 250), ("Gaza", 34.4, 31.45, -170)):
                    pts, _ = P(np.array([[lo, la]])); x, y = pts[0]
                    cv.line([(x, y), (x + dx * 0.55, y)], "#FFFFFF", 3, 0.8 * ha)
                    parts = lab.split("|")             # Noam 23 Nov: "Judea and Samaria (West Bank)", stacked on two tags
                    for j, part in enumerate(parts):
                        WG.tag(cv, part, x + dx, y + (j - (len(parts) - 1) / 2) * 62, 44 if len(parts) > 1 else 46, YEL if lab == "Israel" else "#C9D6EA", ha)
        for g in TAGS:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.3, 0.3)
            if ga > 0.01:
                pts, _ = P(np.array([[g["lon"], g["lat"]]])); x, y = pts[0]
                WG.tag(cv, g["text"], x, y, g.get("size", 46), COLS.get(g.get("col", "white"), WHITE), ga)
        base = cv.out().astype(np.float32) * VIG
        dm = max([win(t, a - 0.3, b, 0.4, 0.4) for a, b in DIM] or [0])
        if dm > 0: base *= (1 - 0.35 * dm)
        cv = Canvas(base)
        for b in BEATS: draw_counter(cv, b, t)
        draw_screen(cv, cards, t, TY, 72)
        for o in OVL:
            oa = win(t, o["t0"] - 0.1, o["t1"], 0.3, 0.3)
            if oa > 0.01: draw_overlay(cv, o, oa, max(0.05, back(seg(t, o["t0"], o["t0"] + 0.35, lambda u: u))) if t < o["t0"] + 0.35 else 1.0)
        for b0, b1 in BRW:
            bra = 1.0 if b0 <= 0.01 and t < b1 - 0.3 else win(t, b0, b1, 0.3, 0.3)
            if bra > 0.01: draw_brand(cv, BR.get("text", "Israel Facts"), bra)
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        ha = 1 - seg(t, T.E(0) + 0.15, T.E(0) + 0.55)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 0.94 + 0.06 * eout(t / 0.2) if t < 0.2 else 1.0, ha)
        cv2_ = Canvas(out); draw_subs(cv2_, CH, t, D.get("subs_y", 1612))
        return np.clip(cv2_.out(), 0, 255).astype(np.uint8)

    ev = []
    for b in BEATS:
        for k, tk in enumerate(b["times"][:int(b.get("hero", 3))]): ev.append((tk, "thud", -20)); ev.append((tk, "pop", -26))
        if b["n"] > 3: ev.append((b["times"][3], "roll", b["tfull"]))
        ev.append((b["tfull"] + 0.3, "ding", -26))
        if b["fly"]: ev.append((b["fly"][0], "whoosh", -24))
    for t0, d, k, bump in keys:
        if bump > 0.3: ev.append((t0, "whoosh", -30))
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], body_s=T.END, mix=aud)


COLS = dict(white=WHITE, yellow=YEL, sky=SKY, grey="#C9D6EA")
