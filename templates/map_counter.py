"""map_counter: motion opener (globe spin / map tear / fold ...) lands on a map, region zooms, a big live counter, pins
with labels, bar cards and a big-number card, source chips, end card (balance line + CTA), loop back into the opener.
Map rules built in: whole land of Israel gold, West Bank and Gaza a darker gold, Green Line green (core/base.py MapR).

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  cam0        [lon, lat, width_deg] where the opener lands (frame 0 of the body)
  cams        [{at, view: [lon, lat, width_deg, screen_y], dur}]       camera moves
  highlight   [{at, until, code, color, alpha, flash, flash_s}]          countries tinted red (ISO3, e.g. "RUS"); optional color
              (name or #hex), alpha (default 0.85; 1.0 is fastest), flash: colour for the first flash_s s (default 0.6)
  names       [{at, until, text, lon, lat, size, dx, dy}]   plain place names on the map (no marker), e.g. the states
              in each scene (Noam 14 Nov: "add names of states"); white with a soft shadow, size >= 36
  israel_at   anchor: before it Israel is drawn as plain land (no gold, no Green Line); it fades in gold at the anchor
  clock       {at, until, y, size, label, era, steps: [{at, year}], tick}   big running year clock, linear between steps
              (era: true writes 1000 BCE / 70 CE; tick: true adds soft clock ticks while it runs)
  pins        [{at, until, lon, lat, label, color: red|yellow|grey}]
  counter     {label, of, at, until, y, h, steps: [{at, value | text, src}], prefix, suffix, dec}   a step with "text" shows that
              text instead of a number (ranges, "No reliable count"), "src" a small source + year line under it
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
  dots        [{at, until, lon, lat, n, color, r, step, fill, dx, dy, label, size}]  a cluster of n dots (e.g. 1 dot = 10,000 people) around a
              point, sunflower packed (dot radius r px, default 2.6; spacing step, default 4.4), filling in over `fill` s (default 1.5)
  lines[i].screen_icon   emoji drawn above that line's screen card (title_y - 125)
  comment     {text, show: [[at, until], ...], y, name, size}   a show window may carry its own text: [at, until, "text"]   a generic, anonymous social comment bubble (grey avatar,
              placeholder name, never a real user or screenshot); pops in at each window start, e.g. [[0, "L0e"], ["END-2", "END"]]
              so the loop lands on it again; text wraps to 2 lines (default size 60, name 40)
  photos      [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim, gray}]   optional real photos, full bleed
  photo_box   [x, y, w, h, radius]  optional: show the photos inside this rounded panel instead (map, pins and routes stay visible)
              over the map in their window (crossfade `photo_cut`, default 0.25 s); map pins/icons/routes/zones fade
              out under a photo, counter/cards/subs stay on top; "Photo: <credit>" at credit_y (default 1712, 36 px)
  overlays    [{at, until, text, y, size, pill, cue}]   extra on-screen text not tied to a spoken line (*word* yellow):
              pill: background colour (e.g. "#45A4DD") or omitted for a dark pill; cue: true draws the sky chevron cue below
  brand       {text, show: [[at, until], ...]}   small mark top left (default "Israel Facts", shown on the hook and the loop end)
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


def draw_chevron(cv, cx, cy, r, a):
    """sky circle with a white rounded down-chevron (the house cue, no hand shapes)"""
    if a <= 0.01: return
    cv.circ(cx, cy, r, SKY, a); c = cv.c
    c.save(); c.set_line_cap(1); c.set_line_join(1)
    cv.line([(cx - r * 0.42, cy - r * 0.16), (cx, cy + r * 0.26), (cx + r * 0.42, cy - r * 0.16)], "#FFFFFF", r * 0.2, a)
    c.restore()


def draw_overlay(cv, o, a, sc=1.0):
    """pill with text (*word* yellow), optional chevron cue under it"""
    y = o.get("y", 1180); sz = o.get("size", 64); lines = o["text"].split("|"); lh = sz * 1.2
    tw = min(980, max(cv.tw(l.replace("*", ""), sz) for l in lines) + 100); h = int(sz * 0.7 + lh * len(lines))
    c = cv.c; c.save(); c.translate(540, y); c.scale(sc, sc); c.translate(-540, -y)
    r = min(h / 2, 48)
    if o.get("pill"): cv.rrect(540 - tw / 2, y - h / 2, tw, h, r, o["pill"], 0.95 * a)
    else: cv.rrect(540 - tw / 2, y - h / 2, tw, h, r, "#0B1730", 0.88 * a, stroke=SKY, lw=3, sa=0.8 * a)
    for k, l in enumerate(lines): WG.title(cv, l, y + (k - (len(lines) - 1) / 2) * lh, a, sz, maxw=940)
    c.restore()
    if o.get("cue"): draw_chevron(cv, 540, y + h / 2 + 70, 46, a)


def draw_brand(cv, txt, a):
    """small 'Israel Facts' mark, top left inside the safe zone"""
    w = cv.tw(txt, 38) + 56
    cv.rrect(70, 196, w, 64, 32, "#0B1730", 0.8 * a, stroke=SKY, lw=2, sa=0.7 * a)
    cv.circ(70 + 30, 228, 9, YEL, a)
    cv.text(txt, 70 + 50, 228, 38, WHITE, a, maxw=w, align="l", shadow=False)


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
    CLK = D.get("clock")
    if CLK: CLK = dict(CLK, t0=A(CLK.get("at"), A(CLK["steps"][0]["at"])), t1=A(CLK.get("until"), T.END - 0.6),
                       st=[(A(s_["at"]), float(s_["year"])) for s_ in CLK["steps"]])
    def TY(y):
        """time at which the clock shows year y (items may give "year" instead of "at")"""
        st = CLK["st"]
        if y <= st[0][1]: return st[0][0]
        for k_ in range(len(st) - 1):
            (a0, y0), (a1, y1) = st[k_], st[k_ + 1]
            if y0 <= y <= y1 and y1 > y0: return a0 + (a1 - a0) * (y - y0) / (y1 - y0)
        return st[-1][0]
    def AY(it, key="at"): return TY(float(it["year"])) if it.get("year") is not None and CLK else A(it[key])
    HL = [(AY(h), A(h.get("until"), T.END), h["code"], h) for h in D.get("highlight", [])]
    ISR_AT = A(D["israel_at"]) if D.get("israel_at") is not None else None
    PINS = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6)) for p in D.get("pins", [])]
    CTS = D.get("counter") or []
    if isinstance(CTS, dict): CTS = [CTS]
    CTS = [dict(c, t0=A(c.get("at"), AY(c["steps"][0]) - 0.4), t1=A(c.get("until"), T.END - 0.6),
                st=[(AY(s), float(s.get("value", 0))) for s in c["steps"]],
                tx=[(AY(s), s.get("text"), s.get("src")) for s in c["steps"]]) for c in CTS]
    ROUTES = [dict(r, t0=A(r["at"]), t1=A(r.get("until"), T.END - 0.6), tc=A(r["cut"]) if r.get("cut") is not None else None)
              for r in D.get("routes", [])]
    ZONES = [dict(z, t0=A(z["at"]), t1=A(z.get("until"), T.END - 0.6)) for z in D.get("zones", [])]
    ICONS = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6)) for c in D.get("icons", [])]
    DOTS = [dict(d, t0=A(d["at"]), t1=A(d.get("until"), T.END - 0.6)) for d in D.get("dots", [])]
    NAMES = [dict(n_, t0=A(n_["at"]), t1=A(n_.get("until"), T.END - 0.6)) for n_ in D.get("names", [])]
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
    CMT = D.get("comment"); CMW = [(A(w_[0]), A(w_[1]), (w_[2] if len(w_) > 2 else None)) for w_ in (CMT or {}).get("show", [])]   # optional 3rd item: text for that window
    OVL = [dict(o, t0=A(o["at"]), t1=A(o.get("until"), T.END - 0.6)) for o in D.get("overlays", [])]
    BR = D.get("brand") or {}; BRW = [(A(w_[0]), A(w_[1])) for w_ in BR.get("show", [])]
    MR = MapR(); cache = {}
    PH = []
    if D.get("photos"):
        from photo_story import _kb, _load
        for p in D["photos"]: PH.append(dict(p, t0=A(p["at"]), im=_load(p["file"], p.get("gray", False))))
        for k, p in enumerate(PH): p["t1"] = A(p["until"]) if p.get("until") else (PH[k + 1]["t0"] if k + 1 < len(PH) else T.END)
    PXF = float(D.get("photo_cut", 0.25))
    BOX = D.get("photo_box")                      # [x, y, w, h, radius]: photos in a rounded panel (map stays visible) instead of full bleed
    if BOX:
        bw_, bh_, br_ = int(BOX[2]), int(BOX[3]), int(BOX[4]) if len(BOX) > 4 else 28
        BOXM = np.ones((bh_, bw_), np.float32); yy, xx = np.mgrid[0:bh_, 0:bw_]
        for cx_, cy_ in ((br_, br_), (bw_ - br_ - 1, br_), (br_, bh_ - br_ - 1), (bw_ - br_ - 1, bh_ - br_ - 1)):
            q = ((np.abs(xx - cx_) <= br_) & (np.abs(yy - cy_) <= br_) & ((xx < br_) | (xx >= bw_ - br_)) & ((yy < br_) | (yy >= bh_ - br_)))
            BOXM[q] = np.clip(br_ + 0.5 - np.hypot(xx - cx_, yy - cy_), 0, 1)[q]

    def _kb_box(im, cx, cy, z, bw, bh):
        h, w = im.shape[:2]; s_ = max(bw / w, bh / h) * max(1.0, z); cw, ch = bw / s_, bh / s_
        x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
        M = np.array([[s_, 0, -x0 * s_], [0, s_, -y0 * s_]], np.float32)
        return cv2.warpAffine(im, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

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

    def mapimg(cm, hi, hc=None, isr=True):
        key = (tuple(np.round(cm, 4)), tuple(sorted((k, round(v, 2)) for k, v in hi.items())), tuple(sorted((hc or {}).items())), isr)
        if key not in cache:
            if len(cache) > 12: cache.clear()
            lon, lat, lk, cosf, sy = cm
            cache[key] = MR.render(MR.view(lon, lat, math.exp(lk), cosf, sy), H, hi=[k for k, v in hi.items() if v > 0.02], hi_a=hi,
                                   hi_c=hc or None, israel=isr)
        return cache[key]
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.6 * 0.45, np.clip((y_ - 1180) / 740, 0, 1) ** 1.2 * 0.62))[..., None]

    def frame(t):
        hi, hc = {}, {}
        for t0, t1, code, h_ in HL:
            v = float(h_.get("alpha", 0.85)) * seg(t, t0, t0 + 0.4) * (1 - seg(t, t1 - 0.3, t1 + 0.2))
            if v > 0.02:
                hi[code] = max(hi.get(code, 0), v)
                c_ = h_.get("flash") if h_.get("flash") and t < t0 + float(h_.get("flash_s", 0.6)) else h_.get("color")
                if c_: hc[code] = COL.get(c_, c_)
        cm_ = cam_at(t)
        if ISR_AT is not None and t < ISR_AT + 0.5:
            u_ = seg(t, ISR_AT, ISR_AT + 0.5)
            img, M = mapimg(cm_, {k: v for k, v in hi.items() if k != "ISR"}, {k: v for k, v in hc.items() if k != "ISR"}, False)
            if u_ > 0.001:
                img2, _ = mapimg(cm_, hi, hc, True); img = (img.astype(np.float32) * (1 - u_) + img2.astype(np.float32) * u_)
        else:
            img, M = mapimg(cm_, hi, hc)
        base = img.astype(np.float32) * VIG
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
        for n_ in NAMES:
            na = win(t, n_["t0"] - 0.1, n_["t1"], 0.35, 0.3)
            if na > 0.01:
                x, y = geo(n_["lon"], n_["lat"])
                cv.text(n_["text"], x + n_.get("dx", 0), y + n_.get("dy", 0), max(36, n_.get("size", 40)), n_.get("color", "#F5F5F7"), 0.92 * na, maxw=600, minsz=36)
        for r in ROUTES:
            draw_route(cv, r, t, geo)
        for c_ in ICONS:
            ia = win(t, c_["t0"] - 0.1, c_["t1"], 0.3, 0.3)
            if ia > 0.01:
                x, y = geo(c_["lon"], c_["lat"]); sc = back(seg(t, c_["t0"], c_["t0"] + 0.4, lambda u: u)) if t < c_["t0"] + 0.4 else 1.0
                WG.emo(cv, c_["emo"], x, y, c_.get("size", 80) * max(0.05, sc), ia)
                if c_.get("label"): WG.tag(cv, c_["label"], x, y + c_.get("label_dy", 75), c_.get("tsize", 44), COL.get(c_.get("col", "yellow"), YEL), ia)
        for d_ in DOTS:
            da = win(t, d_["t0"] - 0.05, d_["t1"], 0.2, 0.3)
            if da > 0.01:
                x, y = geo(d_["lon"], d_["lat"]); x += d_.get("dx", 0); y += d_.get("dy", 0)
                n_ = int(round(d_["n"] * min(1.0, max(0.0, (t - d_["t0"]) / float(d_.get("fill", 1.5))))))
                r_, st = float(d_.get("r", 2.6)), float(d_.get("step", 4.4)); c = cv.c; c.new_path()
                for k in range(n_):
                    rr = st * math.sqrt(k + 0.5); th = k * 2.39996323
                    c.new_sub_path(); c.arc(x + rr * math.cos(th), y + rr * math.sin(th), r_, 0, 2 * math.pi)
                cv.col(COL.get(d_.get("color", "yellow"), d_.get("color", YEL)), 0.95 * da); c.fill()
                if d_.get("label"): WG.tag(cv, d_["label"], x, y - st * math.sqrt(d_["n"]) - 34, d_.get("size", 40), WHITE, da)
        for p in PINS:
            pa = win(t, p["t0"] - 0.1, p["t1"], 0.3, 0.3)
            if pa > 0.01:
                x, y = geo(p["lon"], p["lat"]); c = COL.get(p.get("color", "red"), p.get("color", RED))
                WG.pin(cv, x, y, c, pa, t)
                if p.get("label"): WG.tag(cv, p["label"], x, y - 50, p.get("size", 40), YEL if c == YEL else WHITE, pa)
        if PH and BOX:
            for p in PH:
                ba = win(t, p["t0"] - PXF, p["t1"], PXF, PXF)
                if ba <= 0.01: continue
                bx, by, bw, bh = [int(v) for v in BOX[:4]]
                kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
                lay = _kb_box(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u, bw, bh) * (1 - p.get("dim", 0.15))
                base_ = cv.out().astype(np.float32); m = BOXM[..., None] * ba
                base_[by:by + bh, bx:bx + bw] = base_[by:by + bh, bx:bx + bw] * (1 - m) + lay * m
                cv = Canvas(base_)
                cv.rrect(bx, by, bw, bh, BOX[4] if len(BOX) > 4 else 28, None, 0, stroke="#FFFFFF", lw=3, sa=0.55 * ba)
                if p.get("credit"):
                    cv.text("Photo: " + p["credit"], 540, by + bh - 30, 36, "#FFFFFF", 0.9 * ba, bold=False, maxw=bw - 40, minsz=36)
        elif PH:
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
                y = CT.get("y", 1240); ph = CT.get("h", 300); WG.panel(cv, 90, y - 150, 900, ph, ca)
                cv.text(CT.get("label", ""), 540, y - 85, 48, "#C9D6EA", ca, maxw=820)
                cur = [s_ for s_ in CT["tx"] if t >= s_[0]]
                if cur and (cur[-1][1] or cur[-1][2]):          # text step (range, "No reliable count") and/or source line
                    t0_, txt_, src_ = cur[-1]; fa = seg(t, t0_, t0_ + 0.3); sc_ = 0.85 + 0.15 * back(seg(t, t0_, t0_ + 0.35, lambda u: u))
                    if len(cur) > 1 and fa < 1:                 # crossfade from the previous step
                        p_ = cur[-2]; ptxt = p_[1] or WG.fmtn(CT["st"][len(cur) - 2][1], CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", ""))
                        cv.text(ptxt, 540, y + 30, CT.get("size", 130), YEL, ca * (1 - fa), maxw=820, minsz=56)
                        if p_[2]: cv.text(p_[2], 540, y + 112, 38, "#C9D6EA", ca * (1 - fa), bold=False, maxw=840, minsz=36)
                    txt = txt_ or WG.fmtn(v, CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", ""))
                    c_ = cv.c; c_.save(); c_.translate(540, y + 30); c_.scale(sc_, sc_); c_.translate(-540, -(y + 30))
                    cv.text(txt, 540, y + 30, CT.get("size", 130) if txt_ is None or len(txt_) < 14 else CT.get("tsize", 92), YEL, ca * fa, maxw=820, minsz=56)
                    c_.restore()
                    if src_: cv.text(src_, 540, y + 112, 38, "#C9D6EA", ca * fa, bold=False, maxw=840, minsz=36)
                else:
                    txt = WG.fmtn(v, CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", "")) + (f" / {CT['of']}" if CT.get("of") else "")
                    cv.text(txt, 540, y + 40, 150, YEL, ca, maxw=820)
        if CLK:
            ka = (1.0 - seg(t, CLK["t1"] - 0.3, CLK["t1"])) if CLK["t0"] <= 0.01 else win(t, CLK["t0"] - 0.1, CLK["t1"], 0.3, 0.3)
            if ka > 0.01:
                st = CLK["st"]; yr = st[0][1]
                for k_ in range(len(st)):
                    if t >= st[k_][0]:
                        yr = st[k_][1] if k_ + 1 >= len(st) else st[k_][1] + (st[k_ + 1][1] - st[k_][1]) * min(1.0, (t - st[k_][0]) / max(1e-3, st[k_ + 1][0] - st[k_][0]))
                yi = int(math.floor(yr + 1e-6)) if yr >= 0 else -int(math.floor(-yr + 1e-6))
                if yi == 0: yi = 1 if yr >= 0 else -1                 # there is no year 0
                if CLK.get("era"): ytxt = f"{-yi} BCE" if yi < 0 else (f"{yi} CE" if yi < 1000 else f"{yi}")
                else: ytxt = f"{yi}"
                ky = CLK.get("y", 470); ks = CLK.get("size", 170)
                if CLK.get("label"): cv.text(CLK["label"], 540, ky - ks * 0.62, 44, "#C9D6EA", ka, maxw=900)
                cv.text(ytxt, 540, ky, ks, YEL if CLK.get("color", "yellow") == "yellow" else WHITE, ka, maxw=1000)
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
        for o in OVL:
            oa = win(t, o["t0"] - 0.1, o["t1"], 0.3, 0.3)
            if oa > 0.01: draw_overlay(cv, o, oa, max(0.05, back(seg(t, o["t0"], o["t0"] + 0.35, lambda u: u))) if t < o["t0"] + 0.35 else 1.0)
        for b0, b1 in BRW:
            bra = 1.0 if b0 <= 0.01 and t < b1 - 0.3 else win(t, b0, b1, 0.3, 0.3)
            if bra > 0.01: draw_brand(cv, BR.get("text", "Israel Facts"), bra)
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        for c0, c1, ctx in CMW:
            cma = 1.0 if c0 <= 0.01 and t < c1 - 0.3 else win(t, c0, c1, 0.2, 0.3)
            if cma > 0.01: draw_comment(cv, dict(CMT, text=ctx) if ctx else CMT, cma, 1.0 if c0 <= 0.01 else max(0.05, back(seg(t, c0, c0 + 0.35, lambda u: u)) if t < c0 + 0.35 else 1.0))
        out = cv.out().astype(np.float32)
        ha = 1 - seg(t, T.E(0) + 0.15, T.E(0) + 0.55)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 1.0, ha)
        cv2_ = Canvas(out); draw_subs(cv2_, CH, t, D.get("subs_y", 1612))
        return np.clip(cv2_.out(), 0, 255).astype(np.uint8)

    ev = [(p["t0"], "pop", -22) for p in PINS] + [(b["t0"] - 0.2, "whoosh", -24) for b in BARS] + [(g["t0"], "roll", g["t0"] + 1.2) for g in BIG]
    ev += [(g["t0"] + 1.2, "ding", -26) for g in BIG]
    for CT in CTS: ev += [(t0, "roll", t0 + 0.7) if not tx_ else (t0, "pop", -26) for (t0, _), (_, tx_, _s) in zip(CT["st"], CT["tx"])]
    if CLK and CLK.get("tick"):
        for k_ in range(len(CLK["st"]) - 1):
            a_, b_ = CLK["st"][k_][0], CLK["st"][k_ + 1][0]
            if CLK["st"][k_ + 1][1] != CLK["st"][k_][1]: ev += [(tt_, "tick", -34) for tt_ in np.arange(a_, b_, 0.25)]
    ev += [(r["t0"], "whoosh", -28) for r in ROUTES if r.get("kind") != "wall"] + [(r["t0"], "thud", -22) for r in ROUTES if r.get("kind") == "wall"]
    ev += [(r["tc"], "buzz", -24) for r in ROUTES if r["tc"] is not None] + [(c_["t0"], "pop", -24) for c_ in ICONS]
    ev += [(c0, "pop", -24) for c0, _ in CMW if c0 > 0.01]
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

