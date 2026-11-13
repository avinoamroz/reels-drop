"""lens_podium: map_counter's map + bar cards, plus a podium. One country measured against rivals through several
"lenses" (one ranked bar card each, sorted best first), then a combined score and a 1-2-3 podium reveal.
Frame 0: the map with the rivals tinted (sky, not red), flags popping on their capitals and a small "?" podium
rising under the hook (moving by 1.5 s); the last 0.5 s dissolves back into frame 0 (no opener).

data keys: everything map_counter takes (cam0, cams, highlight, pins, icons, bars, big, dim, photos, ...), plus
  hl_color    tint of `highlight` countries (default sky "#45A4DD")
  bar_color   default bar colour (default sky); rows with hl: true are yellow
  podium      [{at, until, y (base line, default 1400), sc (scale, default 1), teaser (bool: "?" instead of flags),
                steps: [{place: 1|2|3, label, flag, txt, at}]}]       blocks rise at `at`, 2nd left, 1st centre, 3rd right
  rest        [{at, until, y, rows: [{n, label, flag, txt}]}]          places 4+ in a 2-column grid
  notes       [{at, until, text, y, size (>= 40), col}]                 small lines, *word* in yellow (formula, math)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as CM
import base as _B

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


def draw_comment(cv, C, a, sc=1.0):
    """generic anonymous comment bubble: white card, grey avatar silhouette, placeholder name, comment text"""
    if a <= 0.01: return
    y = C.get("y", 330); sz = C.get("size", 60); x0, w = 60, 960; tx = x0 + 168; maxw = x0 + w - 50 - tx
    words = C["text"].split(); lines = [[]]
    for wd in words:
        if lines[-1] and cv.tw(" ".join(lines[-1] + [wd]), sz) > maxw: lines.append([wd])
        else: lines[-1].append(wd)
    h = 112 + int(sz * 1.18) * len(lines) + 34; c = cv.c
    c.save(); c.translate(540, y + h / 2); c.scale(sc, sc); c.translate(-540, -(y + h / 2))
    cv.rrect(x0, y, w, h, 42, "#FFFFFF", 0.97 * a)
    cv.circ(x0 + 84, y + 84, 50, "#D5DBE5", a)
    cv.circ(x0 + 84, y + 70, 18, "#8C97AA", a)
    c.new_path(); c.arc(x0 + 84, y + 122, 32, math.pi, 2 * math.pi); c.close_path(); cv.col("#8C97AA", a); c.fill(); c.new_path()
    cv.text(C.get("name", "anonymous comment"), tx, y + 56, 40, "#6B7588", a, bold=False, align="l", shadow=False, maxw=maxw)
    for i, ln in enumerate(lines):
        cv.text(" ".join(ln), tx, y + 120 + i * int(sz * 1.18), sz, "#111827", a, align="l", shadow=False, maxw=maxw, minsz=44)
    c.restore()


PODX = {1: 540, 2: 245, 3: 835}; PODH = {1: 360, 2: 270, 3: 205}; PODC = {1: YEL, 2: "#C9D6EA", 3: "#D9925A"}


def draw_podium(cv, P, t, a):
    """1-2-3 podium: blocks rise from the base line, rank number on the block, flag, name and score above it"""
    base = P.get("y", 1400); sc = float(P.get("sc", 1.0)); bw = 270 * sc
    for s_ in P["st"]:
        pl = int(s_["place"]); u = eout(seg(t, s_["_at"], s_["_at"] + 0.55, lambda v: v))
        if u <= 0.001: continue
        x = 540 + (PODX[pl] - 540) * sc; h = PODH[pl] * sc * u; top = base - h
        cv.rrect(x - bw / 2, top, bw, h + 4, 18 * sc, PODC[pl], 0.95 * a)
        cv.rrect(x - bw / 2, top, bw, min(h, 16 * sc), 8, "#FFFFFF", 0.35 * a)
        if h > 110 * sc: cv.text(str(pl), x, top + 80 * sc, 120 * sc, NAVY0, a, shadow=False, maxw=bw)
        la = a * seg(t, s_["_at"] + 0.35, s_["_at"] + 0.7)
        if la <= 0.01: continue
        if P.get("teaser"):
            cv.text("?", x, top - 70 * sc, 130 * sc, YEL, la, maxw=bw)
            continue
        pop = back(seg(t, s_["_at"] + 0.35, s_["_at"] + 0.75, lambda v: v)) if t < s_["_at"] + 0.75 else 1.0
        if s_.get("flag"): WG.flag(cv, s_["flag"], x, top - 150 * sc, 110 * sc * max(0.05, pop), la)
        cv.text(s_.get("txt", ""), x, top - 50 * sc, 84 * sc, YEL, la, maxw=bw + 20)
        cv.text(s_.get("label", ""), x, top - 250 * sc, 56 * sc, WHITE, la, maxw=bw + 30, minsz=40)


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
    CMT = D.get("comment"); CMW = [(A(w_[0]), A(w_[1])) for w_ in (CMT or {}).get("show", [])]
    _B.mt.RED = _B.mt.hexc(D.get("hl_color", "#45A4DD").lstrip("#"))
    MR = MapR(); cache = {}
    BARC = D.get("bar_color", SKY)
    POD = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6), st=[dict(s_, _at=A(s_.get("at"), A(p["at"]) + 0.2 * k)) for k, s_ in enumerate(p["steps"])])
           for p in D.get("podium", [])]
    REST = [dict(r, t0=A(r["at"]), t1=A(r.get("until"), T.END - 0.6)) for r in D.get("rest", [])]
    NOTES = [dict(n, t0=A(n["at"]), t1=A(n.get("until"), T.END - 0.6)) for n in D.get("notes", [])]
    DIM += [(p["t0"], p["t1"]) for p in POD if not p.get("teaser")]
    PH = []
    if D.get("photos"):
        from photo_story import _kb, _load
        for p in D["photos"]: PH.append(dict(p, t0=A(p["at"]), im=_load(p["file"], p.get("gray", False))))
        for k, p in enumerate(PH): p["t1"] = A(p["until"]) if p.get("until") else (PH[k + 1]["t0"] if k + 1 < len(PH) else T.END)
    PXF = float(D.get("photo_cut", 0.25))

    def photo_mix(t):
        """(alpha, layer) of the photo(s) showing at t, or (0, None)"""
        a_tot, out = 0.0, None
        for k, p in enumerate(PH):
            gap = k + 1 >= len(PH) or PH[k + 1]["t0"] > p["t1"] + 0.01      # next photo follows straight away: no fade out
            a_in = 1.0 if p["t0"] <= 0.01 else seg(t, p["t0"] - PXF, p["t0"])
            a = a_in * (1 - seg(t, p["t1"] - PXF, p["t1"])) if gap else (a_in if t < p["t1"] + PXF else 0.0)
            if a <= 0.001: continue
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            lay = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u) * (1 - p.get("dim", 0.3)) * VIG
            out = lay if out is None else out * (1 - a) + lay * a; a_tot = max(a_tot, a)
        return a_tot, out

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
        if PH:
            pha, phl = photo_mix(t)
            if pha > 0.001:
                cv = Canvas(cv.out().astype(np.float32) * (1 - pha) + phl * pha)
                for p in PH:
                    ca = win(t, p["t0"], p["t1"], 0.25, 0.25)
                    if ca > 0.01 and p.get("credit"):
                        cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ca, bold=False, maxw=1000, minsz=36)
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
                             color=YEL if r.get("hl") else r.get("color", BARC), txt=r.get("txt"), tcol=YEL if r.get("hl") else WHITE) for r in b["rows"]]
                WG.bars(cv, rows, t, ba, y0, rowh=rowh, vmax=b.get("vmax"), lab_w=b.get("lab_w", 400))
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(120, y - 170, 840, 340, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), 1.2)
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 30, 170, YEL, ga, maxw=780)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 52, WHITE, ga, maxw=780)
        for P in POD:
            pa = win(t, P["t0"] - 0.1, P["t1"], 0.3, 0.35)
            if pa > 0.01: draw_podium(cv, P, t, pa)
        for R in REST:
            ra = win(t, R["t0"] - 0.1, R["t1"], 0.3, 0.35)
            if ra > 0.01:
                y0 = R.get("y", 560)
                for k, r in enumerate(R["rows"]):
                    x = 290 if k % 2 == 0 else 790; y = y0 + (k // 2) * 92; a_ = ra * seg(t, R["t0"] + 0.12 * k, R["t0"] + 0.12 * k + 0.3)
                    if a_ <= 0.01: continue
                    cv.rrect(x - 240, y - 40, 480, 80, 24, "#0B1A3C", 0.85 * a_, stroke=SKY, lw=2, sa=0.5 * a_)
                    cv.text(str(r["n"]), x - 205, y, 52, "#C9D6EA", a_, maxw=60)
                    if r.get("flag"): WG.flag(cv, r["flag"], x - 145, y, 54, a_)
                    cv.text(r["label"], x - 108, y, 50, WHITE, a_, align="l", maxw=250, minsz=40)
                    cv.text(r["txt"], x + 205, y, 54, YEL, a_, maxw=90)
        for N in NOTES:
            na = win(t, N["t0"] - 0.1, N["t1"], 0.3, 0.3)
            if na > 0.01: WG.title(cv, N["text"], N.get("y", 420), na, max(40, N.get("size", 46)), COL.get(N.get("col", "white"), WHITE))
        draw_screen(cv, cards, t, D.get("title_y", 270))
        for e_, t0, t1, y in SICON:
            ea = win(t, t0 - 0.1, t1, 0.3, 0.3)
            if ea > 0.01: WG.emo(cv, e_, 540, y, 96 * max(0.05, back(seg(t, t0, t0 + 0.4, lambda u: u)) if t < t0 + 0.4 else 1.0), ea)
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        for c0, c1 in CMW:
            cma = 1.0 if c0 <= 0.01 and t < c1 - 0.3 else win(t, c0, c1, 0.2, 0.3)
            if cma > 0.01: draw_comment(cv, CMT, cma, 1.0 if c0 <= 0.01 else max(0.05, back(seg(t, c0, c0 + 0.35, lambda u: u)) if t < c0 + 0.35 else 1.0))
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
    ev += [(c0, "pop", -24) for c0, _ in CMW if c0 > 0.01]
    for P in POD:
        ev += [(s_["_at"], "pop" if P.get("teaser") else ("ding" if s_.get("place") == 1 else "thud"), -26 if P.get("teaser") else -22) for s_ in P["st"]]
    ev += [(R["t0"], "whoosh", -28) for R in REST]
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

