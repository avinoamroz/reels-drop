"""map_counter: motion opener (globe spin / map tear / fold ...) lands on a map, region zooms, a big live counter, pins
with labels, bar cards and a big-number card, source chips, end card (balance line + CTA), loop back into the opener.
Map rules built in: whole land of Israel gold, West Bank and Gaza a darker gold, Green Line green (core/base.py MapR).

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  cam0        [lon, lat, width_deg] where the opener lands (frame 0 of the body)
  cams        [{at, view: [lon, lat, width_deg, screen_y], dur}]       camera moves
  highlight   [{at, until, code}]                                        countries tinted red (ISO3, e.g. "RUS")
  pins        [{at, until, lon, lat, label, color: red|yellow|grey}]
  counter     {label, of, at, until, y, steps: [{at, value}], prefix, suffix, dec}
  bars        [{at, until, title, y, rows: [{label, flag, value, txt, at, hl}], vmax}]
  big         [{at, until, value, from, prefix, suffix, sub, y, dec}]
  dim         [{at, until}]                                              darken the map behind cards
  hook_y      y of the hook text in the body's frame 0 (default 930)
  counter     may also be a list of counters (each shows in its own at/until window)
  routes      [{at, until, path: [[lon, lat], ...], kind: flow|pipe|power|wall|line, color, width, draw, speed,
                emo, label, label_at: [lon, lat], label_dy, cut, cut_label}]  geo lines drawn on over `draw` s;
                flow/pipe/power move arrows or dots along the path; `cut` (anchor) turns it into a red dashed line with an X
  zones       [{at, until, poly: [[lon, lat], ...], color, alpha, label, label_at, emo}]   filled areas (buffer zone ...)
  icons       [{at, until, lon, lat, emo, size, label, label_dy, col}]    emoji on the map with an optional tag
  lines[i].screen_icon   emoji drawn above that line's screen card (title_y - 125)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as CM

COL = dict(red=RED, yellow=YEL, grey=GREY, sky=SKY, green=GREEN, white=WHITE, blue="#4FC3F7")
ROUTE_DEF = dict(flow=(SKY, 10), pipe=("#4FC3F7", 14), power=(YEL, 9), wall=("#E8ECF4", 14), line=(WHITE, 6))


def _poly_cut(pts, frac):
    """first `frac` of a polyline (by length) + total length"""
    seglen = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    tot = sum(seglen) or 1.0; want = tot * max(0.0, min(1.0, frac)); out = [pts[0]]; acc = 0.0
    for i, L in enumerate(seglen):
        if acc + L >= want:
            u = (want - acc) / (L or 1); out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * u, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u)); break
        out.append(pts[i + 1]); acc += L
    return out, tot


def _along(pts, d):
    """point and unit direction at distance d along a polyline"""
    acc = 0.0
    for i in range(len(pts) - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if acc + L >= d and L > 0:
            u = (d - acc) / L; dx, dy = (pts[i + 1][0] - pts[i][0]) / L, (pts[i + 1][1] - pts[i][1]) / L
            return (pts[i][0] + dx * u * L, pts[i][1] + dy * u * L), (dx, dy)
        acc += L
    return pts[-1], (1.0, 0.0)


def draw_route(cv, r, t, geo):
    ra = win(t, r["t0"] - 0.1, r["t1"], 0.3, 0.35)
    if ra <= 0.01: return
    kind = r.get("kind", "flow"); col0, w0 = ROUTE_DEF.get(kind, ROUTE_DEF["line"])
    col = COL.get(r.get("color"), r.get("color") or col0); w = r.get("width", w0)
    cut = r["tc"] is not None and t >= r["tc"]
    pts = [geo(*p) for p in r["path"]]
    sub, tot = _poly_cut(pts, seg(t, r["t0"], r["t0"] + float(r.get("draw", 0.8)), eout))
    if len(sub) < 2: return
    if cut:
        ca = seg(t, r["tc"], r["tc"] + 0.3)
        cv.line(sub, RED, w, ra * (0.35 + 0.5 * ca), dash=[18, 14])
        (mx, my), _ = _along(pts, tot * 0.5); s_ = 26 * max(0.05, back(seg(t, r["tc"], r["tc"] + 0.35, lambda u: u)))
        cv.circ(mx, my, s_ * 1.25, RED, ra)
        cv.line([(mx - s_ * 0.5, my - s_ * 0.5), (mx + s_ * 0.5, my + s_ * 0.5)], "#FFFFFF", 6, ra)
        cv.line([(mx - s_ * 0.5, my + s_ * 0.5), (mx + s_ * 0.5, my - s_ * 0.5)], "#FFFFFF", 6, ra)
    elif kind == "wall":
        cv.line(sub, "#000000", w + 8, 0.45 * ra); cv.line(sub, col, w, ra)
        _, stot = _poly_cut(sub, 1.0)
        for k in range(int(stot // 26)):   # brick ticks along the wall
            (x, y), (dx, dy) = _along(sub, k * 26 + 13)
            cv.line([(x - dy * w * 0.5, y + dx * w * 0.5), (x + dy * w * 0.5, y - dx * w * 0.5)], "#8A93A6", 2, ra)
    else:
        cv.line(sub, "#0B1730", w + 8, 0.55 * ra); cv.line(sub, col, w + 14, 0.18 * ra); cv.line(sub, col, w, 0.9 * ra)
        if kind == "power": cv.line(sub, "#FFFFFF", 2, 0.7 * ra, dash=[6, 10])
        _, stot = _poly_cut(sub, 1.0); sp = float(r.get("speed", 110)); gap = float(r.get("gap", 70))
        off = (t * sp) % gap; d = off
        while d < stot - 4:
            (x, y), (dx, dy) = _along(sub, d); a_ = ra * min(1, d / 30, (stot - d) / 30)
            if kind == "flow" or kind == "power":
                s_ = w * 1.9; nx, ny = -dy, dx
                cv.c.new_path(); cv.c.move_to(x + dx * s_, y + dy * s_); cv.c.line_to(x - dx * s_ * 0.7 + nx * s_ * 0.8, y - dy * s_ * 0.7 + ny * s_ * 0.8)
                cv.c.line_to(x - dx * s_ * 0.7 - nx * s_ * 0.8, y - dy * s_ * 0.7 - ny * s_ * 0.8); cv.c.close_path()
                cv.col("#FFFFFF" if kind == "flow" else YEL, a_); cv.c.fill()
            else:
                cv.circ(x, y, w * 0.45, "#FFFFFF", a_)
            d += gap
    la = ra * seg(t, r["t0"] + 0.3, r["t0"] + 0.7)
    if r.get("emo") or r.get("label"):
        x, y = geo(*r["label_at"]) if r.get("label_at") else _along(pts, tot * 0.5)[0]
        if r.get("emo"): WG.emo(cv, r["emo"], x, y, r.get("emo_size", 70), la)
        lab = r.get("cut_label") if cut and r.get("cut_label") else r.get("label")
        if lab: WG.tag(cv, lab, x, y + r.get("label_dy", 70), r.get("size", 46), RED if cut else YEL, la)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75))
    A = T.at
    def cam(lon, lat, wd, sy=960.0):
        cosf = max(0.6, math.cos(math.radians(min(62, abs(lat))))); return np.array([lon, lat, math.log(W / (wd * cosf)), cosf, sy], float)
    K0 = cam(*D["cam0"]); keys = [(0.0, 0.01, K0)] + [(A(c["at"]), float(c.get("dur", 1.3)), cam(*c["view"])) for c in D.get("cams", [])]
    def cam_at(t):
        cur = keys[0][2]
        for i, (t0, d, k) in enumerate(keys):
            if t < t0: break
            prev = keys[i - 1][2] if i else k; u = ease((t - t0) / d); cur = prev + (k - prev) * u
            if u < 1: return cur
        return cur
    HL = [(A(h["at"]), A(h.get("until"), T.END), h["code"]) for h in D.get("highlight", [])]
    PINS = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6)) for p in D.get("pins", [])]
    CTS = D.get("counter") or []
    if isinstance(CTS, dict): CTS = [CTS]
    CTS = [dict(c, t0=A(c.get("at"), A(c["steps"][0]["at"]) - 0.4), t1=A(c.get("until"), T.END - 0.6),
                st=[(A(s["at"]), float(s["value"])) for s in c["steps"]]) for c in CTS]
    ROUTES = [dict(r, t0=A(r["at"]), t1=A(r.get("until"), T.END - 0.6), tc=A(r["cut"]) if r.get("cut") is not None else None)
              for r in D.get("routes", [])]
    ZONES = [dict(z, t0=A(z["at"]), t1=A(z.get("until"), T.END - 0.6)) for z in D.get("zones", [])]
    ICONS = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6)) for c in D.get("icons", [])]
    SICON = []
    for i, L in enumerate(S["lines"]):
        if L.get("screen_icon") and L.get("screen"):
            for c in screen_cards(S, T):
                if c[0] == L["screen"]: SICON.append((L["screen_icon"], c[1], c[2], (c[3] or D.get("title_y", 270)) - 125)); break
    BARS = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("bars", [])]
    for b in BARS:
        for k, r in enumerate(b["rows"]): r["_at"] = A(r["at"]) if r.get("at") is not None else b["t0"] + 0.35 + 0.18 * k
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    DIM = [(A(d_["at"]), A(d_.get("until"), T.END)) for d_ in D.get("dim", [])] + [(b["t0"], b["t1"]) for b in BARS] + [(g["t0"], g["t1"]) for g in BIG]
    if S.get("end"): DIM.append((A(S["end"].get("at"), T.S(len(T.L) - 2)), T.END))
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 930)
    MR = MapR(); cache = {}

    def mapimg(cm, hi):
        key = (tuple(np.round(cm, 4)), tuple(sorted((k, round(v, 2)) for k, v in hi.items())))
        if key not in cache:
            if len(cache) > 12: cache.clear()
            lon, lat, lk, cosf, sy = cm
            cache[key] = MR.render(MR.view(lon, lat, math.exp(lk), cosf, sy), H, hi=[k for k, v in hi.items() if v > 0.02], hi_a=hi)
        return cache[key]
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.6 * 0.45, np.clip((y_ - 1180) / 740, 0, 1) ** 1.2 * 0.62))[..., None]

    def frame(t):
        hi = {}
        for t0, t1, code in HL:
            v = 0.85 * seg(t, t0, t0 + 0.4) * (1 - seg(t, t1 - 0.3, t1 + 0.2))
            if v > 0.02: hi[code] = max(hi.get(code, 0), v)
        img, M = mapimg(cam_at(t), hi); base = img.astype(np.float32) * VIG
        dm = max([win(t, a - 0.3, b, 0.4, 0.4) for a, b in DIM] or [0])
        if dm > 0: base *= (1 - 0.5 * dm)
        cv = Canvas(base); geo = lambda lon, lat: geo2px(M, lon, lat)
        for z in ZONES:
            za = win(t, z["t0"] - 0.1, z["t1"], 0.4, 0.3)
            if za > 0.01:
                pts = [geo(*p) for p in z["poly"]]; c = cv.c; c.new_path(); c.move_to(*pts[0])
                for p in pts[1:]: c.line_to(*p)
                c.close_path(); cv.col(COL.get(z.get("color", "red"), z.get("color", RED)), z.get("alpha", 0.45) * za); c.fill_preserve()
                cv.col("#FFFFFF", 0.7 * za); c.set_line_width(2.5); c.set_dash([10, 8]); c.stroke(); c.set_dash([]); c.new_path()
                if z.get("label") or z.get("emo"):
                    x, y = geo(*z.get("label_at", z["poly"][0]))
                    if z.get("emo"): WG.emo(cv, z["emo"], x, y - 70, 72, za)
                    if z.get("label"): WG.tag(cv, z["label"], x, y, z.get("size", 44), YEL, za)
        for r in ROUTES:
            draw_route(cv, r, t, geo)
        for c_ in ICONS:
            ia = win(t, c_["t0"] - 0.1, c_["t1"], 0.3, 0.3)
            if ia > 0.01:
                x, y = geo(c_["lon"], c_["lat"]); sc = back(seg(t, c_["t0"], c_["t0"] + 0.4, lambda u: u)) if t < c_["t0"] + 0.4 else 1.0
                WG.emo(cv, c_["emo"], x, y, c_.get("size", 80) * max(0.05, sc), ia)
                if c_.get("label"): WG.tag(cv, c_["label"], x, y + c_.get("label_dy", 75), c_.get("tsize", 44), COL.get(c_.get("col", "yellow"), YEL), ia)
        for p in PINS:
            pa = win(t, p["t0"] - 0.1, p["t1"], 0.3, 0.3)
            if pa > 0.01:
                x, y = geo(p["lon"], p["lat"]); c = COL.get(p.get("color", "red"), p.get("color", RED))
                WG.pin(cv, x, y, c, pa, t)
                if p.get("label"): WG.tag(cv, p["label"], x, y - 50, p.get("size", 40), YEL if c == YEL else WHITE, pa)
        for CT in CTS:
            ca = win(t, CT["t0"], CT["t1"], 0.35, 0.35)
            if ca > 0.01:
                v = 0.0; prev = 0.0
                for t0, val in CT["st"]:
                    if t >= t0: v = WG.roll(t, t0, prev, val, 0.7)
                    prev = val
                y = CT.get("y", 1240); WG.panel(cv, 90, y - 150, 900, 300, ca)
                cv.text(CT.get("label", ""), 540, y - 85, 48, "#C9D6EA", ca, maxw=820)
                txt = WG.fmtn(v, CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", "")) + (f" / {CT['of']}" if CT.get("of") else "")
                cv.text(txt, 540, y + 40, 150, YEL, ca, maxw=820)
        for b in BARS:
            ba = win(t, b["t0"] - 0.1, b["t1"], 0.4, 0.4)
            if ba > 0.01:
                n = len(b["rows"]); rowh = min(110, 640 // max(1, n)); y0 = b.get("y", 560)
                WG.panel(cv, 50, y0 - 90, 980, rowh * n + 120, ba)
                if b.get("title"): WG.title(cv, b["title"], y0 - 190, ba, 76)
                rows = [dict(label=r["label"], flag=r.get("flag"), emo=r.get("emo"), value=float(r["value"]), at=r["_at"],
                             color=YEL if r.get("hl") else r.get("color", RED), txt=r.get("txt"), tcol=YEL if r.get("hl") else WHITE) for r in b["rows"]]
                WG.bars(cv, rows, t, ba, y0, rowh=rowh, vmax=b.get("vmax"), lab_w=b.get("lab_w", 400))
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(120, y - 170, 840, 340, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), 1.2)
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 30, 170, YEL, ga, maxw=780)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 52, WHITE, ga, maxw=780)
        draw_screen(cv, cards, t, D.get("title_y", 270))
        for e_, t0, t1, y in SICON:
            ea = win(t, t0 - 0.1, t1, 0.3, 0.3)
            if ea > 0.01: WG.emo(cv, e_, 540, y, 96 * max(0.05, back(seg(t, t0, t0 + 0.4, lambda u: u)) if t < t0 + 0.4 else 1.0), ea)
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        ha = 1 - seg(t, T.E(0) + 0.15, T.E(0) + 0.55)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 1.0, ha)
        cv2_ = Canvas(out); draw_subs(cv2_, CH, t, D.get("subs_y", 1612))
        return np.clip(cv2_.out(), 0, 255).astype(np.uint8)

    ev = [(p["t0"], "pop", -22) for p in PINS] + [(b["t0"] - 0.2, "whoosh", -24) for b in BARS] + [(g["t0"], "roll", g["t0"] + 1.2) for g in BIG]
    ev += [(g["t0"] + 1.2, "ding", -26) for g in BIG]
    for CT in CTS: ev += [(t0, "roll", t0 + 0.7) for t0, _ in CT["st"]]
    ev += [(r["t0"], "whoosh", -28) for r in ROUTES if r.get("kind") != "wall"] + [(r["t0"], "thud", -22) for r in ROUTES if r.get("kind") == "wall"]
    ev += [(r["tc"], "buzz", -24) for r in ROUTES if r["tc"] is not None] + [(c_["t0"], "pop", -24) for c_ in ICONS]
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, ev)
    body = f"{work}/body.mp4"
    encode(frame, T.END, f"{work}/mix.m4a", body, loop=not S.get("opener"))
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    if S.get("opener"):
        op = wrap_opener(body, S, final, D["cam0"], HY, work)
    else:
        shutil.copy(body, final); op = None
    return dict(voice=J["voice"], body_s=T.END, opener=op, mix=aud)

