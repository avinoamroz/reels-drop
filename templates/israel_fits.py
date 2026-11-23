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
Opt-in additions (Builder BB, 23 Nov; specs without these keys render exactly as before):
  globe     true: whole-globe views (far side opened, space outside the Earth's disc)
  surround  [{code, box, center, area, ratio, at, grow, until, panel_until, follow, land_total, mode, ...}] ring around a
            country with area x ratio of neighbours; land counter vs need (see SURROUND in run)
  counters  [{at, until, name, value, dec, prefix, suffix, sub, from, y}] rolling number panels
  hl_world  [{codes, box, at, until, fill, line, fa}] tint any world country
  photos    [{file, at, until, label, credit, y, w, h, focus_y}] real photo band with Ken Burns and credit
"""
import os, sys, math, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
import common as CM
from map_counter import draw_overlay, draw_brand

R = 6371.0072
CLAMP = [-0.6]          # far side clamp of the projection; data.globe (opt-in) opens it to the whole globe
GEOF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "israel_fits_geo.json")
LAND, LAND_B, OCEAN0, OCEAN1 = "#1D4579", "#3C70B0", "#07142E", "#0C1D40"
TGT_FILL, TGT_LINE = "#2A5E9C", "#8FD0F5"
ISR_FILL, TERR_FILL = SKY, "#2C7DB3"


def fwd(lon, lat, lon0, lat0):
    """Lambert azimuthal equal-area (sphere). Returns x, y (km, y north) and cos(c) (angular distance) for culling."""
    l = np.radians(lon); p = np.radians(lat); l0 = math.radians(lon0); p0 = math.radians(lat0)
    cc = math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0)
    k = np.sqrt(2 / np.maximum(1 + np.maximum(cc, CLAMP[0]), 1e-9))
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
    GLOBE = bool(D.get("globe"))                                          # opt-in (surround reels): whole-globe views
    CLAMP[0] = -0.995 if GLOBE else -0.6
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

    SUR = []                                                              # opt-in "surround" beats (see SURROUND below)
    for sb in D.get("surround", []):
        g0, g1 = A(sb["grow"][0]), A(sb["grow"][1])
        SUR.append(dict(sb, t0=A(sb["at"]), t1=A(sb.get("until"), T.END - 0.6), g0=g0, g1=g1, need=float(sb["area"]) * float(sb["ratio"])))

    def theta_at(b, t):
        """cap radius in degrees: from the country's own size to the whole globe, eased (slow start, fast middle)"""
        th0 = math.degrees(math.sqrt(float(b["area"]) / math.pi) / R) * 1.15; th1 = float(b.get("theta_max", 180.0))
        if t <= b["g0"]: return th0
        u = min(1.0, (t - b["g0"]) / max(0.1, b["g1"] - b["g0"]))
        u = 0.5 * u + 0.5 * ease(u) if b.get("ease", "soft") == "soft" else ease(u)      # soft: near linear, the ring keeps moving
        return th0 + (th1 - th0) * u ** float(b.get("shape", 1.0))

    _cam_base = cam_at
    def cam_at(t):
        cur = _cam_base(t)
        for b in SUR:
            f = b.get("follow")
            if not f or not (b["g0"] <= t <= b["g1"] + 0.05): continue
            th = math.radians(theta_at(b, t)); diam = 4 * R * math.sin(th / 2)
            wk = min(float(f[1]), max(float(f[0]), diam / float(b.get("follow_fill", 0.62))))
            cur = np.array([b["center"][0], b["center"][1], math.log(wk), float(f[2]) if len(f) > 2 else cur[3]])
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
        if GLOBE:
            if cc.min() < -0.997: return False                            # rings touching the antipode smear along the rim: skip
        elif cc.max() < -0.2: return False
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
            ll = np.c_[np.full(61, lon), np.linspace(-85, 85, 61)]; pts, cc = P(ll); ok = cc > (-0.9 if GLOBE else 0.05)
            if ok.sum() < 2: continue
            first = True
            for p, o in zip(pts, ok):
                if not o: first = True; continue
                (c.move_to if first else c.line_to)(p[0], p[1]); first = False
        for lat in range(-80, 81, 10):
            ll = np.c_[np.linspace(-180, 180, 121), np.full(121, lat)]; pts, cc = P(ll); ok = cc > (-0.9 if GLOBE else 0.05)
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


    # ---------------------------------------------------------------- SURROUND (opt-in, Builder BB 23 Nov): "surround X like Israel"
    # data.surround [{center: [lon, lat], code: "FRA", box: [lon0, lat0, lon1, lat1] (which rings of the country to outline),
    #   area: km2, ratio: 593, at, grow: [t0, t1], until, follow: [w_min_km, w_max_km, screen_y] (camera rides the ring),
    #   land_total: 148.94e6 (land counter scaled to this), mode: "land" | "planets", name, name_grow, sub_full, planets}]
    # A spherical cap grows around the country (true area on the equal-area map); oceans get a gold wash, land under the
    # ring turns gold, and the counter shows the land swallowed against the land needed (area x ratio).
    def land_curve(lon0, lat0, res=0.25):
        nx, ny = int(360 / res), int(180 / res)
        sf = cairo.ImageSurface(cairo.FORMAT_A8, nx, ny); c = cairo.Context(sf); c.set_fill_rule(cairo.FILL_RULE_WINDING)
        def ring(r):
            c.move_to((r[0, 0] + 180) / res, (90 - r[0, 1]) / res)
            for x, y in r[1:]: c.line_to((x + 180) / res, (90 - y) / res)
            c.close_path()
        for rs in G["world"].values():
            for r in rings_ll(rs):
                if len(r) >= 3: ring(r)
        c.set_source_rgba(0, 0, 0, 1); c.fill(); c.set_operator(cairo.OPERATOR_CLEAR)
        for r in LAKES:
            if len(r) >= 3: ring(r)
        c.fill(); sf.flush()
        m = np.ndarray((ny, sf.get_stride()), np.uint8, sf.get_data())[:, :nx].astype(np.float32) / 255
        la = np.radians(90 - (np.arange(ny) + 0.5) * res)[:, None]; lo = np.radians(-180 + (np.arange(nx) + 0.5) * res)[None, :]
        area = (math.radians(res) ** 2) * np.cos(la) * R ** 2 * np.ones_like(lo)
        p0, l0 = math.radians(lat0), math.radians(lon0)
        th = np.degrees(np.arccos(np.clip(math.sin(p0) * np.sin(la) + math.cos(p0) * np.cos(la) * np.cos(lo - l0), -1, 1))).ravel()
        e = np.linspace(0, 180, 721); h, _ = np.histogram(th, e, weights=(m * area).ravel()); cum = np.r_[0, np.cumsum(h)]
        return e, cum / cum[-1]
    for b in SUR:
        b["curve"] = land_curve(*b["center"])
        rs = rings_ll(G["world"][b["code"]]) if b.get("code") else []
        if b.get("box"):
            x0, y0, x1, y1 = b["box"]; rs = [r for r in rs if x0 <= r[:, 0].mean() <= x1 and y0 <= r[:, 1].mean() <= y1]
        b["rings"] = rs
        b["own"] = float(b["area"])

    def land_at(b, t):
        e, cf = b["curve"]; th = theta_at(b, t)
        return max(b["own"], float(np.interp(th, e, cf)) * float(b.get("land_total", 148.94e6)))    # land inside the ring, incl. the country

    def small_circle(lon0, lat0, th, n=240):
        p0, l0, d = math.radians(lat0), math.radians(lon0), math.radians(th); az = np.linspace(0, 2 * np.pi, n, endpoint=False)
        la = np.arcsin(np.clip(math.sin(p0) * math.cos(d) + math.cos(p0) * math.sin(d) * np.cos(az), -1, 1))
        lo = l0 + np.arctan2(np.sin(az) * math.sin(d) * math.cos(p0), math.cos(d) - math.sin(p0) * np.sin(la))
        return np.c_[np.degrees(lo), np.degrees(la)]

    def cap_path(cv, b, th, st, P, s):
        c = cv.c; c.new_path()
        if ang(st[:2], b["center"]) < 0.3:                                  # camera on the centre: the cap is an exact circle
            pc, _ = P(np.array([b["center"]], float)); r = 2 * R * math.sin(math.radians(min(th, 179.99)) / 2) * s
            c.arc(pc[0][0], pc[0][1], r, 0, 2 * math.pi); return
        pts, cc = P(small_circle(b["center"][0], b["center"][1], th)); path(cv, pts)

    RING_W, RING_L, RING_E = "#F2B53A", "#E9A93A", "#FFD55A"
    def draw_surround(cv, b, t, st, P, s):
        ba = win(t, b["t0"] - 0.3, b["t1"], 0.4, 0.4)
        if ba <= 0.01: return
        c = cv.c
        if t >= b["g0"] - 0.05:
            th = theta_at(b, t)
            cap_path(cv, b, th, st, P, s); cv.col(RING_W, 0.30 * ba); c.fill()                         # the ring eats the oceans
            c.save(); cap_path(cv, b, th, st, P, s); c.clip()
            draw_rings(cv, WORLD, P, RING_L, 0.88 * ba, "#FFE6A0", 0.55 * ba, max(0.8, min(1.6, s * 3)))   # and the land
            c.restore()
            if th < 179.5:
                for lw_, al_ in ((16, 0.18), (6, 0.95)):
                    cap_path(cv, b, th, st, P, s); cv.col(RING_E, al_ * ba); c.set_line_width(lw_); c.stroke()
        draw_rings(cv, b["rings"], P, ISR_FILL, ba, "#FFFFFF", ba, max(1.6, min(3.5, s * 5)))           # the country itself

    def fmt_big(v, unit):
        if v >= 1e9: return f"{v / 1e9:,.1f}B {unit}".strip()
        if v >= 1e6: return f"{v / 1e6:,.0f}M {unit}".strip()
        return f"{v:,.0f} {unit}".strip()

    def draw_sur_panel(cv, b, t):
        a = win(t, b["t0"] - 0.2, A(b.get("panel_until"), b["t1"]), 0.3, 0.3)
        if a <= 0.01: return
        top = CY - 140; cv.rrect(90, top, 900, 420, 44, "#0B1A3C", 0.78 * a, stroke=SKY, lw=2.5, sa=0.5 * a)
        need = b["need"]; mode = b.get("mode", "land")
        if t < b["g0"]:                                                      # phase A: area x ratio = need
            v = WG.roll(t, b["t0"] + 0.25, 1, need, 1.1)
            cv.text(b.get("name", ""), 540, CY - 75, 64, WHITE, a, maxw=840)
            cv.text(fmt_big(v, "km²"), 540, CY + 55, 140, YEL, a, maxw=840)
            cv.text(b.get("sub", "of neighbours needed"), 540, CY + 170, 48, "#C9D6EA", a, bold=False, maxw=840)
            return
        land = land_at(b, t); full = t >= b["g1"]
        if mode == "planets" and t >= b["g1"] + 0.15:                        # phase C (planets): how many Earths of land
            k = float(b.get("planets", need / float(b.get("land_total", 148.94e6))))
            v = WG.roll(t, b["g1"] + 0.15, 1, k, float(b.get("planets_dur", 1.6)))
            cv.text(b.get("name_planets", "Planets of land needed"), 540, CY - 75, 60, WHITE, a, maxw=840)
            cv.text("x " + f"{v:,.0f}", 540, CY + 45, 150, YEL, a, maxw=840)
            nd = int(round(v)); cols_ = 20
            for i in range(nd):
                x = 540 - (cols_ - 1) * 21 + (i % cols_) * 42; y = CY + 160 + (i // cols_) * 42
                pa = min(1.0, (t - (b["g1"] + 0.15)) * 6) * a
                cv.circ(x, y, 15, "#2A6FB5", pa, stroke="#8FD0F5", lw=2)
                cv.circ(x - 4, y - 3, 6, "#3FA34D", pa)
            return
        cv.text(b.get("name_grow", "Land swallowed"), 540, CY - 75, 60, WHITE, a, maxw=840)
        pulse = 1 + 0.05 * math.sin(t * 9) * (1 - seg(t, b["g1"], b["g1"] + 0.6)) * (t < b["g1"] + 0.6)
        c = cv.c; c.save(); c.translate(540, CY + 40); c.scale(pulse, pulse); c.translate(-540, -(CY + 40))
        cv.text(fmt_big(land, "km²"), 540, CY + 40, 128, YEL, a, maxw=840); c.restore()
        bx, by, bw, bh = 150, CY + 130, 780, 30; frac = min(1.0, land / need)
        cv.rrect(bx, by, bw, bh, 15, "#1A2F57", a, stroke="#8FD0F5", lw=2, sa=0.6 * a)
        if frac > 0.004: cv.rrect(bx, by, max(bh, bw * frac), bh, 15, YEL, a)
        cv.text(b.get("bar_end", "Needed: " + fmt_big(need, "")), 930, by + bh + 34, 40, "#C9D6EA", a, bold=False, maxw=420, align="r")
        if full:
            fa = a * seg(t, b["g1"], b["g1"] + 0.4)
            cv.text(b.get("sub_full", "All land on Earth. Not enough.").replace("*", ""), 150, by + bh + 34, 40, YEL, fa, maxw=430, align="l")

    CNT = [dict(k, t0=A(k["at"]), t1=A(k.get("until"), T.END - 0.6)) for k in D.get("counters", [])]
    def draw_counters(cv, t):
        for k in CNT:
            a = win(t, k["t0"] - 0.2, k["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            y = k.get("y", CY); v = WG.roll(t, k["t0"] + 0.15, float(k.get("from", 1)), float(k["value"]), float(k.get("dur", 0.9)))
            cv.rrect(90, y - 140, 900, 360, 44, "#0B1A3C", 0.78 * a, stroke=SKY, lw=2.5, sa=0.5 * a)
            cv.text(k.get("name", ""), 540, y - 70, 62, WHITE, a, maxw=840)
            cv.text(WG.fmtn(v, int(k.get("dec", 0)), k.get("prefix", ""), k.get("suffix", "")), 540, y + 55, 140, YEL, a, maxw=840)
            cv.text(k.get("sub", ""), 540, y + 165, 46, "#C9D6EA", a, bold=False, maxw=860)

    HLW = []
    for h in D.get("hl_world", []):
        rs = [r for cd in h["codes"] for r in rings_ll(G["world"][cd])]
        if h.get("box"):
            x0, y0, x1, y1 = h["box"]; rs = [r for r in rs if x0 <= r[:, 0].mean() <= x1 and y0 <= r[:, 1].mean() <= y1]
        HLW.append(dict(h, rings=rs, t0=A(h["at"]), t1=A(h.get("until"), T.END)))

    PH = []
    for ph in D.get("photos", []):
        from PIL import Image
        if not os.path.exists(os.path.join(REPO, ph["file"])):        # the CI checkout is sparse: pull the photo in
            subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + ph["file"]], check=False, capture_output=True)
        im = Image.open(os.path.join(REPO, ph["file"])).convert("RGB"); pw, phh = int(ph.get("w", 940)), int(ph.get("h", 470))
        z = 1.08; tw_, th_ = int(pw * z), int(phh * z); r_ = max(tw_ / im.width, th_ / im.height)
        im = im.resize((max(tw_, int(im.width * r_)), max(th_, int(im.height * r_))), Image.LANCZOS)
        fy = float(ph.get("focus_y", 0.5)); x0 = (im.width - tw_) // 2; y0 = int((im.height - th_) * fy)
        arr = np.asarray(im.crop((x0, y0, x0 + tw_, y0 + th_)).convert("RGBA"))
        PH.append(dict(ph, arr=arr, pw=pw, ph_=phh, t0=A(ph["at"]), t1=A(ph.get("until"), T.END - 0.6)))
    def draw_photos(cv, t):
        c = cv.c
        for ph in PH:
            a = win(t, ph["t0"] - 0.15, ph["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            y = ph.get("y", 430); pw, phh = ph["pw"], ph["ph_"]; x0 = 540 - pw / 2; y0 = y - phh / 2
            u = min(1.0, max(0.0, (t - ph["t0"]) / max(0.5, ph["t1"] - ph["t0"])))
            sc = (1 / 1.08) * (1 + 0.07 * u)
            cv.rrect(x0 - 4, y0 - 4, pw + 8, phh + 8, 36, "#0B1730", 0.9 * a, stroke=SKY, lw=3, sa=0.8 * a)
            c.save(); c.new_path(); r = 32
            c.arc(x0 + pw - r, y0 + r, r, -math.pi / 2, 0); c.arc(x0 + pw - r, y0 + phh - r, r, 0, math.pi / 2)
            c.arc(x0 + r, y0 + phh - r, r, math.pi / 2, math.pi); c.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2); c.close_path(); c.clip()
            cv.image(ph["arr"], 540 + (u - 0.5) * 14, y, sc, a); c.restore()
            if ph.get("label"): WG.tag(cv, ph["label"], 540, y0 + phh - 44, 46, YEL, a)
            if ph.get("credit"): cv.text(ph["credit"], 540, y0 + phh + 34, 36, "#C9D6EA", a, bold=False, maxw=960, minsz=36)

    def draw_space(cv, st, P, s):
        """globe mode: deep space outside the Earth's disc, thin sky rim"""
        rr = 2 * R * s * 0.9999
        if rr > 2400: return
        pc = (540.0, float(st[3])); c = cv.c; c.new_path()
        c.rectangle(0, 0, W, H); c.arc_negative(pc[0], pc[1], rr, 2 * math.pi, 0); cv.col("#03091A", 1.0); c.fill()
        c.new_path(); c.arc(pc[0], pc[1], rr, 0, 2 * math.pi); cv.col(SKY, 0.55); c.set_line_width(3); c.stroke()
        c.new_path(); c.arc(pc[0], pc[1], rr + 9, 0, 2 * math.pi); cv.col(SKY, 0.12); c.set_line_width(14); c.stroke()

    def frame(t):
        st = cam_at(t); P, s = project(st)
        cv = Canvas(OCEAN); graticule(cv, P, s)
        draw_rings(cv, WORLD, P, LAND, 1.0, LAND_B, 0.55, max(0.8, min(1.6, s * 3)))
        for h in HLT:
            ha = 1.0 if h["t0"] <= 0.01 and t < h["t1"] - 0.3 else win(t, h["t0"], h["t1"], 0.3, 0.4)
            if ha > 0.01: draw_rings(cv, h["rings"], P, TGT_FILL, 0.95 * ha, TGT_LINE, 0.9 * ha, max(2.0, min(4.0, s * 8)))
        for h in HLW:
            ha = 1.0 if h["t0"] <= 0.01 and t < h["t1"] - 0.3 else win(t, h["t0"], h["t1"], 0.3, 0.4)
            if ha > 0.01: draw_rings(cv, h["rings"], P, h.get("fill", TGT_FILL), float(h.get("fa", 0.95)) * ha, h.get("line", TGT_LINE), 0.9 * ha, max(1.6, min(3.5, s * 6)))
        draw_rings(cv, LAKES, P, OCEAN1, 1.0, LAND_B, 0.4, 1.0)
        for b in SUR: draw_surround(cv, b, t, st, P, s)
        for b in BEATS: draw_beat(cv, b, t, P, s)
        draw_rings(cv, HOME_T, P, TERR_FILL, 1.0, GREEN, 0.95, max(1.5, min(4.0, s * 6)))
        draw_rings(cv, HOME_I, P, ISR_FILL, 1.0, "#FFFFFF", 0.95, max(1.2, min(3.5, s * 5)))
        if GLOBE: draw_space(cv, st, P, s)
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
        for b in SUR: draw_sur_panel(cv, b, t)
        draw_counters(cv, t); draw_photos(cv, t)
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
    for b in SUR:
        ev.append((b["g0"], "whoosh", -24)); ev.append((b["g0"] + 0.2, "roll", b["g1"])); ev.append((b["g1"], "thud", -20))
        if b.get("mode") == "planets": ev.append((b["g1"] + 0.15, "roll", b["g1"] + 1.6)); ev.append((b["g1"] + 1.8, "ding", -26))
    for k in CNT: ev.append((k["t0"] + 1.0, "ding", -28))
    for ph in PH: ev.append((ph["t0"], "whoosh", -32))
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], body_s=T.END, mix=aud)


COLS = dict(white=WHITE, yellow=YEL, sky=SKY, grey="#C9D6EA")
