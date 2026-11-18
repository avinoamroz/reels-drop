"""faith_tree: a history told as ONE family tree (the "one family, three faiths" format) in house style.
Fixed HUD the whole reel: a photo band on top (real photos, Ken Burns, credit) or a drawn icon scene, a big yellow year
counter ("c. 2000 BC", "AD 610") that rolls between steps, a place label with a pin that travels on a mini map
(whole land of Israel gold, West Bank and Gaza darker gold, Green Line green: core/base.py MapR), and a horizontal
branching timeline: one line that grows with the year, splits into coloured branches (dashed while dormant), relabels
("ISAAC" -> "ISRAELITES" -> "JUDAISM"), with small markers. Chapter cards ("SPLIT #1 / CHRISTIANITY"), sourced quote
cards, screen cards from the lines, end card, burned-in subtitles, hook from frame 0, circular ending.
No prophet is drawn as a figure (respect for all three faiths): places, photos, icons and anonymous walkers only.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  band_h    photo band height (default 640)
  band      [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]  real photos (repo paths)
            or {icons: [[emoji, x, y, size], ...], walk: {n, y, dir, load}, at, until, caption}  drawn scene
  years     {steps: [{at, y (negative = BC), approx}], x: 300, y: 730, size: 120}
  places    {coords: {Name: [lon, lat]}, steps: [{at, name}], map: {lon, lat, k, x, y, w, h}}
  tree      {x0, x1, xmap: [[year, x], ...], top, bottom,
             branches: [{id, col, y, parent, split (year), at, dash_until (year), end (year), lab_x, lab_dy,
                          labels: [{at, text}]}],
             marks: [{branch, year, at, text, dy, until}], walkers: [{at, until, branch, n}]}
  chapters  [{at, until, kicker, title, col}]       big card over the band
  quotes    [{at, until, text, who}]                 quote card over the band (real quotes only)
  scr_y     y of line "screen" cards (default 330); hook_y default 330; subs_y 1612; src_y 1488
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

COLS = dict(sky=SKY, gold="#F2B33D", green="#3FBF7F", teal="#2EC4B6", mint="#8FE3B0", white=WHITE, grey="#9FB0C8", violet="#B892FF")
NAVY = np.array([48, 23, 11], np.float32)


def C(c): return COLS.get(c, c)


def _load(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"faith_tree: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z, bw, bh):
    h, w = im.shape[:2]; s = max(bw / w, bh / h) * max(1.0, z)
    cw, ch = bw / s, bh / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def fmt_year(v, approx=False):
    y = int(round(v))
    s = f"{abs(y)} BC" if y < 0 else ("today" if y >= 2026 else f"AD {max(1, y)}")
    return ("c. " + s) if approx and y < 2026 else s


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    BH = int(D.get("band_h", 640))
    # ---------------------------------------------------------------- band
    P = []
    for p in D.get("band", []):
        q = dict(p, t0=A(p["at"]))
        if p.get("file"): q["im"] = _load(p["file"])
        P.append(q)
    for k, p in enumerate(P):
        p["t1"] = A(p["until"]) if p.get("until") else (P[k + 1]["t0"] if k + 1 < len(P) else T.END + 1)
    XF = 0.3
    yb = np.arange(BH, dtype=np.float32)[:, None]
    BVIG = (1 - np.clip((yb - (BH - 240)) / 240, 0, 1) ** 1.3 * 0.95 - np.clip(1 - yb / 240, 0, 1) ** 1.5 * 0.3)[..., None]

    def band_layer(p, t):
        if p.get("file"):
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"] + XF))
            img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u, W, BH)
            dim = p.get("dim", 0.12)
            return img * (1 - dim) * BVIG + NAVY * (1 - BVIG)
        cv = Canvas(np.zeros((H, W, 3), np.float32) + NAVY + np.array([20, 10, 4], np.float32))
        u = (t - p["t0"]); z = 1.0 + 0.03 * math.sin(u * 1.3)
        for ic in p.get("icons", []):
            ch, x, y, sz = ic[0], ic[1], ic[2], ic[3] if len(ic) > 3 else 200
            bob = 6 * math.sin(u * 2.2 + x * 0.01)
            WG.emo(cv, ch, x, y + bob, sz * z, 1.0)
        wk = p.get("walk")
        if wk:
            n = int(wk.get("n", 5)); dirn = wk.get("dir", 1); yy = wk.get("y", BH - 120)
            for i in range(n):
                x = ((u * 70 * dirn + i * (1080 / n) + 40) % 1160) - 40
                head = figure(cv, x, yy, 1.05, WHITE, badge=False, arms="down", a=0.95, walk=u * 7 + i, flip=dirn)
                if wk.get("load") == "brick": cv.rrect(head[0] - 30, head[1] - 52, 60, 30, 4, "#C9773B", 0.95)
                elif wk.get("load") == "bundle": cv.circ(head[0] + 22 * dirn, head[1] + 60, 20, "#C9A15B", 0.95)
        return cv.out()[:BH].astype(np.float32) * BVIG + NAVY * (1 - BVIG)

    def base_at(t):
        out = BG.copy(); band = out[:BH].copy() * 0 + NAVY
        lin = lambda x: min(1.0, max(0.0, x))
        for k, p in enumerate(P):
            if t < p["t0"] - XF or t >= p["t1"] + XF: continue
            a = 1.0 if p["t0"] <= 0.01 else seg(t, p["t0"] - XF, p["t0"], lin)
            nxt = P[k + 1]["t0"] if k + 1 < len(P) else None
            if (nxt is None or nxt > p["t1"] + 0.01) and t >= p["t1"]: a *= 1 - seg(t, p["t1"], p["t1"] + XF, lin)
            if a <= 0.001: continue
            lay = band_layer(p, t); band = lay if a >= 0.999 else band * (1 - a) + lay * a
        out[:BH] = band
        return out

    # ---------------------------------------------------------------- years
    Y = D.get("years", {})
    YST = [(A(s_["at"]), float(s_["y"]), bool(s_.get("approx"))) for s_ in Y.get("steps", [])]

    def year_at(t):
        if not YST: return 0.0, False
        v, ap = YST[0][1], YST[0][2]
        for k, (ts, val, apx) in enumerate(YST):
            if t >= ts:
                prev = YST[k - 1][1] if k else val
                v = prev + (val - prev) * eout((t - ts) / 0.9); ap = apx
        return v, ap

    # ---------------------------------------------------------------- mini map
    PL = D.get("places", {}); CO = PL.get("coords", {})
    PST = [(A(s_["at"]), s_["name"]) for s_ in PL.get("steps", [])]
    MP = PL.get("map", {}); mx, my, mw, mh = MP.get("x", 620), MP.get("y", 655), MP.get("w", 420), MP.get("h", 370)
    lon0, lat0 = MP.get("lon", 39.5), MP.get("lat", 27.8); kk = MP.get("k", 58.0)
    MR = MapR(); cosf = math.cos(math.radians(lat0))
    mimg, MM = MR.render(MR.view(lon0, lat0, kk, cosf, 540), 1080)
    cy0 = 540 - mh * 1080 / mw / 2
    sc_ = mw / 1080.0
    crop = mimg[int(max(0, cy0)):int(max(0, cy0)) + int(mh / sc_)]
    MINI = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32)
    def mpx(name):
        lo, la = CO[name]; x, y = geo2px(MM, lo, la)
        return mx + x * sc_, my + (y - int(max(0, cy0))) * sc_

    def place_at(t):
        if not PST: return None, None, 1.0
        cur = PST[0]; prev = None
        for k, s_ in enumerate(PST):
            if t >= s_[0]: cur = s_; prev = PST[k - 1] if k else None
        u = eout((t - cur[0]) / 0.9) if prev else 1.0
        return cur, prev, u

    # ---------------------------------------------------------------- tree
    TR = D.get("tree", {}); XM = sorted(TR.get("xmap", [[-2000, 110], [2026, 1000]]))
    def X(yr):
        xs = [a for a, _ in XM]; ps = [b for _, b in XM]
        return float(np.interp(yr, xs, ps))
    BR = {}
    for b in TR.get("branches", []):
        q = dict(b); q["t0"] = A(b.get("at"), 0.0); q["col"] = C(b.get("col", "sky"))
        q["labs"] = [(A(l_["at"], 0.0), l_["text"]) for l_ in b.get("labels", [])]
        BR[b["id"]] = q
    MK = [dict(m, t0=A(m["at"]), t1=A(m.get("until"), T.END + 5)) for m in TR.get("marks", [])]
    WKS = [dict(w_, t0=A(w_["at"]), t1=A(w_["until"])) for w_ in TR.get("walkers", [])]

    def b_geom(b):
        par = BR.get(b.get("parent"))
        xs = X(b["split"]) if b.get("split") is not None else TR.get("x0", 90)
        yp = par["y"] if par else b["y"]
        return xs, yp

    def b_y(b, x):
        xs, yp = b_geom(b)
        if x <= xs: return yp
        u = min(1.0, (x - xs) / 70.0); u = u * u * (3 - 2 * u)
        return yp + (b["y"] - yp) * u

    def b_head(b, t, pyr):
        xs, _ = b_geom(b)
        g = seg(t, b["t0"], b["t0"] + 0.7)
        end = X(min(pyr, b.get("end", 99999)))
        if b.get("parent"): end = max(end, xs + b.get("stub", 120))
        return xs + (max(xs, end) - xs) * g

    def draw_tree(cv, t, pyr):
        top, bot = TR.get("top", 1050), TR.get("bottom", 1425)
        cv.rrect(40, top, 1000, bot - top, 30, "#0B1A3C", 0.55, stroke=SKY, lw=2, sa=0.25)
        order = sorted(BR.values(), key=lambda b: b["t0"])
        for b in order:
            if t < b["t0"] - 0.01: continue
            xs, yp = b_geom(b); hx = b_head(b, t, pyr)
            if hx <= xs + 1: continue
            xx = np.arange(xs, hx, 5.0).tolist() + [hx]
            pts = [(x, b_y(b, x)) for x in xx]
            dx = X(b["dash_until"]) if b.get("dash_until") is not None else -1
            dash = [(x, y) for x, y in pts if x <= dx]; sol = [(x, y) for x, y in pts if x >= dx - 0.01]
            if len(dash) > 1: cv.line(dash, b["col"], 9, 0.55, dash=[2, 20])
            if len(sol) > 1:
                cv.line(sol, b["col"], 26, 0.16); cv.line(sol, b["col"], 13, 1.0)
            live = hx >= min(X(min(pyr, b.get("end", 99999))), hx) - 2 and b.get("end", 99999) > pyr - 1
            hy = b_y(b, hx)
            if live and hx > dx:
                pulse = 0.5 + 0.5 * math.sin(t * 6)
                cv.circ(hx, hy, 22 + 4 * pulse, b["col"], 0.25); cv.circ(hx, hy, 13, b["col"], 1.0); cv.circ(hx, hy, 5, NAVY0, 1.0)
            # label (crossfade between labels)
            la = seg(t, b["t0"] + 0.2, b["t0"] + 0.6)
            if la > 0.01 and b["labs"]:
                lx = b.get("lab_x", xs + 80); ly = b["y"] + b.get("lab_dy", -42)
                cur = None; prv = None
                for k, (ts, txt) in enumerate(b["labs"]):
                    if t >= ts - 0.001: cur = (ts, txt); prv = b["labs"][k - 1] if k else None
                if cur:
                    u = seg(t, cur[0], cur[0] + 0.45) if prv else 1.0
                    if prv and u < 1: cv.text(prv[1], lx, ly - 30 * u, 56, b["col"], la * (1 - u), align="l", maxw=900, shadow=True)
                    cv.text(cur[1], lx, ly + 30 * (1 - u), 56, b["col"], la * u, align="l", maxw=900, shadow=True)
        for m in MK:
            b = BR[m["branch"]]; a = win(t, m["t0"], m["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            x = X(m["year"]); y = b_y(b, x); s_ = back(seg(t, m["t0"], m["t0"] + 0.35, lambda u: u))
            cv.circ(x, y, 12 * s_, WHITE, a); cv.circ(x, y, 6 * s_, b["col"], a)
            if m.get("text"):
                dy = m.get("dy", 50); tw = cv.tw(m["text"], 40) + 36; tx = min(max(x, 60 + tw / 2), 1020 - tw / 2)
                cv.rrect(tx - tw / 2, y + dy - 28, tw, 56, 28, "#0B1730", 0.88 * a, stroke=b["col"], lw=2, sa=0.8 * a)
                cv.text(m["text"], tx, y + dy, 40, WHITE, a, shadow=False, maxw=900)
        for w_ in WKS:
            a = win(t, w_["t0"], w_["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            b = BR[w_["branch"]]; hx = b_head(b, t, pyr); hy = b_y(b, hx)
            for i in range(int(w_.get("n", 3))):
                x = hx - 34 - i * 40
                if x < 50: continue
                figure(cv, x, hy - 10, 0.42, WHITE, badge=False, a=a, lw=6, walk=t * 9 + i * 1.7)

    # ---------------------------------------------------------------- cards
    CHP = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), A(c["at"]) + 1.8)) for c in D.get("chapters", [])]
    QT = [dict(q, t0=A(q["at"]), t1=A(q.get("until"), A(q["at"]) + 3.0)) for q in D.get("quotes", [])]
    SCR = screen_cards(S, T); SY = D.get("scr_y", 330)
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 430)

    def draw_hud(cv, t):
        pyr, ap = year_at(t)
        cv.rrect(40, 650, 560, 380, 30, "#0B1A3C", 0.7, stroke=SKY, lw=2, sa=0.3)
        cv.text(fmt_year(pyr, ap), Y.get("x", 320), Y.get("y", 740), Y.get("size", 116), YEL, 1.0, maxw=520, minsz=64)
        cur, prv, u = place_at(t)
        if cur:
            na = seg(t, cur[0], cur[0] + 0.4) if prv else 1.0
            if prv and na < 1: cv.text(prv[1], 352, 880 + 20 * na, 66, WHITE, 1 - na, maxw=460, minsz=48)
            cv.text(cur[1], 352, 880 - 20 * (1 - na), 66, WHITE, na, maxw=460, minsz=48)
            WG.emo(cv, "\U0001F4CD", 90, 878, 58, 1.0)
        # mini map
        cv.rrect(mx - 4, my - 4, mw + 8, mh + 8, 26, None, 1.0, stroke=SKY, lw=3, sa=0.6)
        if cur:
            x1, y1 = mpx(cur[1])
            if prv and prv[1] in CO:
                x0, y0 = mpx(prv[1]); xx, yy = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                if u < 1: cv.line([(x0, y0), (xx, yy)], YEL, 4, 0.9, dash=[3, 10])
            else: xx, yy = x1, y1
            pulse = (t * 1.2) % 1.0
            cv.circ(xx, yy, 10 + 26 * pulse, None, 1, stroke=YEL, lw=3 * (1 - pulse) + 0.5)
            cv.circ(xx, yy, 11, YEL, 1.0); cv.circ(xx, yy, 4, NAVY0, 1.0)
        return pyr

    def draw_cards(cv, t):
        for c in CHP:
            a = win(t, c["t0"] - 0.05, c["t1"], 0.2, 0.3)
            if a <= 0.01: continue
            s_ = back(seg(t, c["t0"], c["t0"] + 0.4, lambda u: u)); col = C(c.get("col", "sky"))
            WG.panel(cv, 90, 210, 900, 330, a, stroke=col)
            cv.text(c.get("kicker", ""), 540, 290, 60, col, a, maxw=820)
            cv.text(c["title"], 540, 420, int(118 * (0.85 + 0.15 * s_)), WHITE, a, maxw=840, minsz=64)
        for q in QT:
            a = win(t, q["t0"] - 0.05, q["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            WG.panel(cv, 50, 230, 980, 330, a, stroke=YEL)
            WG.title(cv, q["text"], 365, a, 64, maxw=900)
            if q.get("who"): cv.text(q["who"], 540, 505, 42, "#C9D6EA", a, bold=False, maxw=900)
        for txt, t0, t1, yy, ss in SCR:
            a = win(t, t0 - 0.1, t1, 0.3, 0.3)
            if a <= 0.01: continue
            WG.panel(cv, 50, (yy or SY) - 130, 980, 260, a)
            WG.title(cv, txt.replace("|", " "), yy or SY, a, ss or 76, maxw=900)

    def draw_end_band(cv, a):
        E = S.get("end") or {}; y = E.get("y", 400)
        WG.panel(cv, 50, y - 190, 980, 380, a, stroke=YEL)
        if E.get("balance"): WG.title(cv, E["balance"], y - 70, a, 68, maxw=900)
        if E.get("cta"):
            tw = min(cv.tw(E["cta"].replace("*", ""), 60) + 90, 940)
            cv.rrect(540 - tw / 2, y + 50, tw, 110, 55, SKY, a); WG.title(cv, E["cta"], y + 105, a, 60)

    def frame(t):
        base = base_at(t)
        END_T1 = min([c[1] for c in SCR if END_AT is not None and c[1] > END_AT] or [T.END - 0.5])
        endov = win(t, END_AT - 0.1, END_T1, 0.4, 0.3) if END_AT else 0
        cv = Canvas(base)
        cv.image(np.dstack([MINI[..., 2], MINI[..., 1], MINI[..., 0], np.full(MINI.shape[:2], 255, np.float32)]).astype(np.uint8),
                 mx + mw / 2, my + mh / 2, 1.0, 1.0)
        for p in P:
            ca = win(t, p["t0"], p["t1"], 0.25, 0.25)
            if ca > 0.01 and p.get("credit"):
                cv.text(("Photo: " if p.get("file") else "") + p["credit"], 540, BH - 34, 36, "#C9D6EA", 0.9 * ca, bold=False, maxw=1000, minsz=36)
        import map_counter as MC
        MC.draw_brand(cv, D.get("brand", "Israel Facts"), 1.0)
        pyr = draw_hud(cv, t); draw_tree(cv, t, pyr)
        draw_cards(cv, t)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if endov > 0:
            o = cv.out().astype(np.float32); o[:D.get("band_h", 640)] *= (1 - 0.5 * endov); cv = Canvas(o)
            draw_end_band(cv, endov)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -30) for p in P[1:]]
    ev += [(ts, "tick", -26) for ts, _, _ in YST[1:]]
    ev += [(c["t0"], "thud", -20) for c in CHP]
    ev += [(m["t0"], "pop", -28) for m in MK]
    ev += [(b["t0"], "whoosh", -30) for b in BR.values() if b["t0"] > 0.2]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p.get("file"), credit=p.get("credit")) for p in P])
