"""people_stock: "if peoples were stocks". ONE shared trading-screen chart where one line per people (group) draws left to
right over a long time span (BCE years are negative numbers), green where the group grows and red where it shrinks, with
a soft glow in the group's own colour under its line, a ticker label at each line head ("[dot] JEW 350k") with a small
dot in the group's colour LEFT of the ticker (peoples, not countries: no flags), a big year counter ("586 BCE"), tick
sounds while drawing. The window follows the line heads (span set per beat) and the y axis rescales to what is on screen,
like a trading app; a final view can zoom out to the whole history. Frame 0 is the map of the land (whole land of Israel
gold, West Bank and Gaza a darker gold, Green Line green) with the hook over it; the chart takes over by about 1.5 s.
Real photos between beats (full bleed, Ken Burns, credit line), end card, subtitles; the map + hook come back at the end
so the loop into frame 0 is seamless.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  file     repo path to a JSON {"series": [{"ticker": "JEW", "name": "Jews", "col": "#3B82F6", "hl": false,
           "pts": [[year, value], ...]}]}  (linear between points; 0 = not present; no line where both ends are 0)
  map      {view: [lon, lat, k, sy], zoom: 1.06, until: anchor, back: anchor}   frame 0 map; `back` = map returns
  keys     [[year, anchor], ...]   the line head is at `year` at that time (repeat a year = hold)
  views    [{at, span} | {at, x0, x1}]   window: span years trailing the head (head at about 86 %), or fixed x0..x1;
           changes ease over `dur` (default 0.9 s)
  ymin     smallest y axis max (default 50000);  ylag (s, default 0.35) how fast the y axis follows
  chart    {"x0": 120, "x1": 790, "y0": 560, "y1": 1320}
  year_y   year counter centre y (default 470);  title_y 300;  src_y 1488;  subs_y 1612;  credit_y 1712;  hook_y 330
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]
  marks    [{at, until, text, year, ticker, dy, col}]  callout pinned to a line point (col: yellow | red | green)
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

UP, DOWN, FLAT = "#2BD67B", "#F0524F", "#9FB0C8"
GRID = "#22324F"; AXIS = "#8FA3C0"
SCREEN = np.array([34, 20, 10], np.float32)        # BGR near-black navy trading screen


def fmt_v(v):
    v = max(0.0, float(v))
    if v >= 1e6: return f"{v / 1e6:.2f}M"
    if v >= 1e4: return f"{v / 1e3:.0f}k"
    if v >= 1e3: return f"{v / 1e3:.1f}k".replace(".0k", "k")
    return f"{v:.0f}"


def fmt_axis(v):
    if v == 0: return "0"
    if v >= 1e6: return f"{v / 1e6:g}M"
    return f"{v / 1e3:g}k"


def fmt_year(y):
    y = int(round(y))
    if y <= 0: return f"{1 - y if y == 0 else -y} BCE"
    if y < 1000: return f"{y} CE"
    return str(y)


def tick_label(y):
    y = int(round(y))
    if y < 0: return f"{-y} BCE"
    if y == 0: return "1 CE"
    if y < 1000: return f"{y} CE"
    return str(y)


def nice_step(span, n=4.0):
    raw = span / n; m = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for k in (1, 2, 2.5, 5, 10):
        if k * m >= raw: return k * m
    return 10 * m


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    DT = json.load(open(os.path.join(REPO, D["file"]), encoding="utf-8"))
    SER = []
    for s in DT["series"]:
        P = np.array(s["pts"], float); SER.append(dict(s, X=P[:, 0], V=P[:, 1]))
    C = dict(dict(x0=120, x1=790, y0=560, y1=1320), **D.get("chart", {}))
    YMIN = float(D.get("ymin", 50000))

    def val(s, yr):
        X, V = s["X"], s["V"]
        if yr < X[0] or yr > X[-1] + 1e-6: return 0.0 if yr < X[0] else float(V[-1])
        return float(np.interp(yr, X, V))

    # ------------------------------------------------------------- head year over time
    KY = [(float(k[0]), A(k[1])) for k in D["keys"]]
    def head(t):
        if t <= KY[0][1]: return KY[0][0]
        for (y0, t0), (y1, t1) in zip(KY, KY[1:]):
            if t <= t1:
                u = min(1, max(0, (t - t0) / max(1e-3, t1 - t0))); return y0 + (y1 - y0) * u
        return KY[-1][0]

    # ------------------------------------------------------------- window (x) and y scale, precomputed on a time grid
    VW = [dict(v, t0=A(v["at"])) for v in D["views"]]
    YR0 = min(float(s["X"][0]) for s in SER)
    def view_of(v, hy):
        if "span" in v:
            sp = float(v["span"]); x1 = max(YR0 + sp, hy + sp * 0.16); return x1 - sp, x1
        return float(v["x0"]), float(v["x1"])
    def window(t):
        hy = head(t); cur = None; prev = None
        for v in VW:
            if t >= v["t0"]: prev, cur = cur, v
        if cur is None: cur = VW[0]
        a, b = view_of(cur, hy)
        if prev is not None:
            u = ease((t - cur["t0"]) / float(cur.get("dur", 0.9)))
            if u < 1:
                pa, pb = view_of(prev, hy); a = pa + (a - pa) * u; b = pb + (b - pb) * u
        return a, b
    DTG = 1 / 30; NG = int((T.END + 1) / DTG) + 2
    TG = np.arange(NG) * DTG
    WX = np.array([window(t) for t in TG])
    def vis_max(t, x0, x1):
        hy = head(t); hi = min(hy, x1); m = 0.0
        if hi <= x0: return YMIN
        xs = np.linspace(x0, hi, 60)
        for s in SER:
            ks = [x for x in s["X"] if x0 <= x <= hi]
            m = max(m, max(val(s, x) for x in list(xs) + ks))
        return max(YMIN, m)
    YT = np.array([vis_max(t, a, b) for t, (a, b) in zip(TG, WX)])
    lag = float(D.get("ylag", 0.35)); YS = YT.copy()
    k = 1 - math.exp(-DTG / max(1e-3, lag))
    for i in range(1, NG):
        YS[i] = YS[i - 1] + (YT[i] - YS[i - 1]) * k if YT[i] < YS[i - 1] else YS[i - 1] + (YT[i] - YS[i - 1]) * min(1, 2.2 * k)
    def at_grid(t):
        i = min(NG - 1, max(0, int(round(t / DTG)))); return WX[i][0], WX[i][1], YS[i] * 1.16

    # ------------------------------------------------------------- map (frame 0 + loop)
    MP = D.get("map", {})
    MAP_T1 = A(MP.get("until"), 1.2); MAP_BACK = A(MP.get("back"), T.END + 5)
    lon, lat, kk, sy = MP.get("view", [35.05, 31.55, 330, 1090])
    MR = MapR(); cosf = math.cos(math.radians(lat))
    mimg, MM = MR.render(MR.view(lon, lat, kk, cosf, sy), H); mimg = mimg.astype(np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    mimg = mimg * (1 - np.clip(1 - yy / 620, 0, 1) ** 1.6 * 0.55)[..., None]
    ZM = float(MP.get("zoom", 1.06))
    LABS = [dict(l_, xy=geo2px(MM, l_["lon"], l_["lat"])) for l_ in MP.get("labels", [])]
    def map_at(z):
        Mz = cv2.getRotationMatrix2D((540, sy), 0, z)
        img = cv2.warpAffine(mimg, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if LABS:
            cv = Canvas(img)
            for l_ in LABS:
                x, y = l_["xy"]; x = 540 + (x - 540) * z; y = sy + (y - sy) * z
                WG.tag(cv, l_["text"], x + l_.get("dx", 0), y + l_.get("dy", 0), l_.get("size", 44), l_.get("col", WHITE), 1.0)
            img = cv.out()
        return img.astype(np.float32)

    # ------------------------------------------------------------- photos
    from photo_story import _load, _kb
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))) for p in D.get("photos", [])]
    VIG = (1 - np.maximum(np.clip(1 - yy / 560, 0, 1) ** 1.5 * 0.62, np.clip((yy - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    NAVYC = np.array([97, 50, 22], np.float32)
    def ph_alpha(p, t): return win(t, p["t0"] - 0.25, p["t1"] + 0.25, 0.3, 0.3)
    def ph_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.25)
        return img * (1 - dim) * VIG + NAVYC * dim * 0.35

    MK = [dict(m, t0=A(m["at"]), t1=A(m.get("until"), T.END - 0.6)) for m in D.get("marks", [])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 330)
    bg = SCREEN[None, None] + (NAVYC - SCREEN)[None, None] * 0.35 * np.clip(yy[..., None] / H, 0, 1)
    bg = np.broadcast_to(bg, (H, W, 3)).astype(np.float32)
    END_AT = A((S.get("end") or {}).get("at"), T.END - 3) if S.get("end") else None

    def draw_chart(cv, t, ca):
        if ca <= 0.01: return
        x0, x1, ym = at_grid(t); hy = head(t)
        X = lambda yr: C["x0"] + (yr - x0) / (x1 - x0) * (C["x1"] - C["x0"])
        Yp = lambda v: C["y1"] - v / ym * (C["y1"] - C["y0"])
        c = cv.c; c.save(); c.rectangle(0, C["y0"] - 60, W, C["y1"] - C["y0"] + 140); c.clip()
        st = nice_step(ym / 1.16, 3.2); v = 0.0
        while v <= ym * 1.001:
            y = Yp(v); cv.line([(C["x0"], y), (C["x1"] + 30, y)], GRID, 2, ca)
            cv.text(fmt_axis(v), C["x0"] - 14, y, 36, AXIS, ca, bold=False, align="r", shadow=False)
            v += st
        xs = nice_step(x1 - x0, 3.6); g = math.ceil(x0 / xs) * xs
        while g <= x1:
            x = X(g); cv.line([(x, C["y0"] - 10), (x, C["y1"])], GRID, 1.5, 0.7 * ca)
            if C["x0"] + 30 < x < C["x1"] + 20: cv.text(tick_label(g), x, C["y1"] + 40, 36, AXIS, ca, bold=False, shadow=False)
            g += xs
        c.restore()
        c.save(); c.rectangle(C["x0"] - 4, C["y0"] - 60, C["x1"] + 260 - C["x0"], C["y1"] - C["y0"] + 70); c.clip()
        heads = []
        for s in sorted(SER, key=lambda s: bool(s.get("hl"))):
            hi = min(hy, x1 + (x1 - x0) * 0.02)
            ks = [x for x in s["X"] if x0 - (x1 - x0) * 0.05 < x < hi]
            lo = max(x0 - (x1 - x0) * 0.05, float(s["X"][0]))
            if hi <= lo: continue
            xs_ = sorted(set([lo, hi] + ks)); pts = [(x, val(s, x)) for x in xs_]
            glow = []
            for (ya, va), (yb, vb) in zip(pts, pts[1:]):
                if va <= 0 and vb <= 0: continue
                col = UP if vb > va + 1e-9 else (DOWN if vb < va - 1e-9 else FLAT)
                seg_ = [(X(ya), Yp(va)), (X(yb), Yp(vb))]
                cv.line(seg_, s["col"], 18, 0.36 * ca)
                glow.append((seg_, col))
            for seg_, col in glow: cv.line(seg_, col, 7, ca)
            hv = val(s, hy)
            if hy < float(s["X"][0]): continue
            # trend at the head: compare with a little earlier
            dv = hv - val(s, hy - max(1.0, (x1 - x0) * 0.01))
            col = UP if dv > 1e-6 else (DOWN if dv < -1e-6 else FLAT)
            zero_since = None
            if hv <= 0:
                nz = [x for x, v in zip(s["X"], s["V"]) if v > 0 and x <= hy]
                if not nz: continue
                zero_since = max(x for x, v in zip(s["X"], s["V"]) if x <= hy and v <= 0)
            la = ca if zero_since is None else ca * max(0.0, 1 - (hy - zero_since) / max(1.0, (x1 - x0) * 0.08))
            if la <= 0.01: continue
            hx, hyp = X(hy), Yp(hv)
            cv.circ(hx, hyp, 9, WHITE, la)
            heads.append(dict(s=s, x=hx, y=hyp, v=hv, col=col, a=la))
        c.restore()
        heads.sort(key=lambda h: h["y"]); gap = 54; ys = [h["y"] for h in heads]
        for _ in range(120):
            for k_ in range(1, len(ys)):
                if ys[k_] - ys[k_ - 1] < gap:
                    d = (gap - (ys[k_] - ys[k_ - 1])) / 2; ys[k_ - 1] -= d; ys[k_] += d
            ys = [min(max(y, C["y0"] - 20), C["y1"] + 10) for y in ys]
        for h, ly in zip(heads, ys):
            sz = 46; tk = h["s"]["ticker"]; vt = fmt_v(h["v"])
            arrow = "▲" if h["col"] == UP else ("▼" if h["col"] == DOWN else "")
            dot = sz * 0.62; gap_ = sz * 0.3
            wt = cv.tw(tk + " ", sz); wv = cv.tw(vt + (" " + arrow if arrow else ""), sz)
            lx = min(h["x"] + 34, 1066 - dot - gap_ - wt - wv)
            if abs(ly - h["y"]) > 8 or lx > h["x"] + 30: cv.line([(h["x"] + 12, h["y"]), (lx - 6, ly)], h["s"]["col"], 2, 0.5 * h["a"])
            cv.rrect(lx - 8, ly - sz * 0.62, dot + gap_ + wt + wv + 18, sz * 1.24, 14, "#0B1322", 0.72 * h["a"])
            cv.circ(lx + dot / 2, ly, dot / 2, h["s"]["col"], h["a"], stroke=WHITE, lw=2)          # group marker LEFT of the ticker
            cv.text(tk + " ", lx + dot + gap_, ly, sz, WHITE, h["a"], align="l", shadow=False)
            cv.text(vt + (" " + arrow if arrow else ""), lx + dot + gap_ + wt, ly, sz, h["col"], h["a"], align="l", shadow=False)
        ya = ca * seg(t, T.E(0) - 0.1, T.E(0) + 0.3)
        if ya > 0.01: cv.text(fmt_year(hy), 1040, D.get("year_y", 470), 96, YEL, ya, align="r", shadow=True)

    BAND = D.get("band")
    if BAND:
        bx0, bx1, by0, by1 = BAND.get("x0", 24), BAND.get("x1", 1056), BAND["y0"], BAND["y1"]
        bw, bh = bx1 - bx0, by1 - by0; bcy = (by0 + by1) / 2
        bk = float(BAND.get("map_k", kk * 0.36))
        bmap, _ = MR.render(MR.view(lon, lat, bk, cosf, bcy + BAND.get("map_dy", 0)), H)
        BMAP = bmap.astype(np.float32)[by0:by1, bx0:bx1].copy()
        msk = np.zeros((bh, bw), np.uint8); r = int(BAND.get("r", 30))
        cv2.rectangle(msk, (r, 0), (bw - r, bh), 255, -1); cv2.rectangle(msk, (0, r), (bw, bh - r), 255, -1)
        for cx_, cy_ in ((r, r), (bw - r, r), (r, bh - r), (bw - r, bh - r)): cv2.circle(msk, (cx_, cy_), r, 255, -1)
        BM = (msk.astype(np.float32) / 255)[..., None]
        byy = np.arange(bh, dtype=np.float32)[:, None]
        BVIG = (1 - np.clip((byy - bh * 0.62) / (bh * 0.38), 0, 1) ** 1.3 * 0.55)[..., None]
        def band_kb(im, cx, cy, z):
            h_, w_ = im.shape[:2]; s_ = max(bw / w_, bh / h_) * max(1.0, z)
            cw, ch = bw / s_, bh / s_
            x0_ = min(max(cx * w_ - cw / 2, 0), w_ - cw); y0_ = min(max(cy * h_ - ch / 2, 0), h_ - ch)
            M_ = np.array([[s_, 0, -x0_ * s_], [0, s_, -y0_ * s_]], np.float32)
            return cv2.warpAffine(im, M_, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        def band_img(t):
            img = BMAP * 0.85
            for p in PH:
                a_ = win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)
                if a_ <= 0.001: continue
                kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
                u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
                ph = band_kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
                img = img * (1 - a_) + ph * (1 - 0.6 * p.get("dim", 0.25)) * a_
            return img * BVIG

    def frame_band(t):
        ma = 1 - seg(t, MAP_T1 - 0.2, MAP_T1 + 0.2)
        if ma >= 0.999: base = map_at(1 + (ZM - 1) * eout(t / max(0.5, MAP_T1)))
        else:
            base = bg.copy()
            reg = base[by0:by1, bx0:bx1]; base[by0:by1, bx0:bx1] = reg * (1 - BM) + band_img(t) * BM
            if ma > 0.001: base = base * (1 - ma) + map_at(ZM) * ma
        cv = Canvas(base)
        if ma < 0.999: cv.rrect(bx0, by0, bw, bh, int(BAND.get("r", 30)), None, 1 - ma, stroke="#2A3B5C", lw=3, sa=1 - ma)
        draw_chart(cv, t, 1 - ma)
        for mk in MK:
            a = win(t, mk["t0"], mk["t1"], 0.25, 0.25) * (1 - ma)
            if a <= 0.01: continue
            x0, x1, ym = at_grid(t)
            s = next(x for x in SER if x["ticker"] == mk["ticker"])
            px = C["x0"] + (mk["year"] - x0) / (x1 - x0) * (C["x1"] - C["x0"]); py = C["y1"] - val(s, mk["year"]) / ym * (C["y1"] - C["y0"])
            ly = py + mk.get("dy", -130); col = {"red": DOWN, "green": UP, "yellow": YEL}.get(mk.get("col", "yellow"), YEL)
            tw = cv.tw(mk["text"], 46) + 40; lx = min(max(px + mk.get("dx", 0), tw / 2 + 16), W - 16 - tw / 2)
            cv.line([(lx, ly + (34 if ly < py else -34)), (px, py + (-16 if ly < py else 16))], col, 4, a)
            cv.circ(px, py, 15, None, a, stroke=col, lw=4)
            cv.rrect(lx - tw / 2, ly - 34, tw, 68, 18, "#0B1730", 0.92 * a, stroke=col, lw=3, sa=a)
            cv.text(mk["text"], lx, ly, 46, WHITE, a, maxw=1040)
        for k_, p in enumerate(PH):
            a_ = win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)
            for q in PH[k_ + 1:]: a_ *= 1 - win(t, q["t0"] - 0.15, q["t1"] + 0.15, 0.3, 0.3)
            if a_ > 0.01 and p.get("credit"):
                txt = "Photo: " + p["credit"]; tw = min(bw - 30, cv.tw(txt, 36, bold=False) + 30)
                cv.rrect(540 - tw / 2, by1 - 62, tw, 50, 14, "#0B1322", 0.7 * a_)
                cv.text(txt, 540, by1 - 37, 36, "#E6EEF8", a_, bold=False, maxw=bw - 40, minsz=36, shadow=False)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if t < MAP_BACK: draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0), HY, 1.0)
        if t >= MAP_BACK: out = HK.paste(out, hook_img, 540, D.get("hook_end_y", bcy), 1.0, seg(t, MAP_BACK, MAP_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    def frame(t):
        if BAND: return frame_band(t)
        ma = max(1 - seg(t, MAP_T1 - 0.2, MAP_T1 + 0.2), seg(t, MAP_BACK, MAP_BACK + 0.4))
        if t < MAP_BACK: z = 1 + (ZM - 1) * eout(t / max(0.5, MAP_T1))
        else: z = ZM - (ZM - 1) * eout((t - MAP_BACK) / max(0.5, T.END - MAP_BACK))
        base = bg.copy(); pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        if ma > 0.001: base = base * (1 - ma) + map_at(z) * ma
        endo = (seg(t, END_AT - 0.1, END_AT + 0.3) * (1 - seg(t, MAP_BACK - 0.2, MAP_BACK + 0.2))) if END_AT else 0.0
        cv = Canvas(base)
        ca = (1 - pa) * (1 - ma) * (1 - 0.8 * endo)
        draw_chart(cv, t, ca)
        for mk in MK:
            a = win(t, mk["t0"], mk["t1"], 0.25, 0.25) * (1 - pa) * (1 - ma)
            if a <= 0.01: continue
            x0, x1, ym = at_grid(t)
            s = next(x for x in SER if x["ticker"] == mk["ticker"])
            px = C["x0"] + (mk["year"] - x0) / (x1 - x0) * (C["x1"] - C["x0"]); py = C["y1"] - val(s, mk["year"]) / ym * (C["y1"] - C["y0"])
            ly = py + mk.get("dy", -130); col = {"red": DOWN, "green": UP, "yellow": YEL}.get(mk.get("col", "yellow"), YEL)
            tw = cv.tw(mk["text"], 46) + 40; lx = min(max(px + mk.get("dx", 0), tw / 2 + 16), W - 16 - tw / 2)
            cv.line([(lx, ly + (34 if ly < py else -34)), (px, py + (-16 if ly < py else 16))], col, 4, a)
            cv.circ(px, py, 15, None, a, stroke=col, lw=4)
            cv.rrect(lx - tw / 2, ly - 34, tw, 68, 18, "#0B1730", 0.92 * a, stroke=col, lw=3, sa=a)
            cv.text(mk["text"], lx, ly, 46, WHITE, a, maxw=1040)
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1040, minsz=36)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 300), 72)
        if t < MAP_BACK - 0.1: draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if t < MAP_BACK: draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0) if t < MAP_BACK else T.END + 5, HY, 1.0)
        if t >= MAP_BACK: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, MAP_BACK, MAP_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(MAP_T1, "whoosh", -28), (MAP_BACK, "whoosh", -30)] + [(p["t0"], "whoosh", -32) for p in PH if p["t0"] > 0.01]
    for (y0, t0), (y1, t1) in zip(KY, KY[1:]):
        if y1 > y0 + 0.01 and t1 > t0 + 0.2:
            for tt in np.arange(t0, t1, 0.16): ev.append((float(tt), "tick", -32))
    ev += [(m["t0"], "buzz" if m.get("col") == "red" else "ding", -28) for m in MK]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
