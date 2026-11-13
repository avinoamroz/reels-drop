"""pop_stock: "stock chart" story of populations. Frame 0 is a map (whole land of Israel gold, West Bank and Gaza a darker
gold, Green Line green) with the hook on it; by about 1.5 s the chart screen takes over: dark trading-screen panels, one
per population, a line drawing left to right over the years, green where the number rises and red where it falls, and a
ticker header per panel ("ARABS IN ISRAEL 2.16M ▲ 1,283%"). Real photos cut in between beats (full bleed, Ken Burns,
credit line), number cards and a quote card sit on top, end card, subtitles, and the last line returns to the map
(circular: the last 0.5 s dissolves into frame 0).

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  map      {view: [lon, lat, k, sy], until, back, labels: [{lon, lat, text}], zoom: 1.08}
           frame 0 map; `until` = when the chart takes over; `back` = when the map returns for the circular ending
  years    [1945, 2028]    shared x axis;  ticks [1950, 1970, 1990, 2010]
  panels   [{label, flags: ["IL"], unit_max, y, h, at, until, points: [[year, value], ...], draw: [{at, until, to}], notes: [{year, text, at}]}]
           `at` = panel slides in; each draw step moves the line head from the previous `to` year to this `to` year
           between `at` and `until`. Segments that fall are red, rising green; the ticker shows the head value, an arrow
           and the % change since the first point (rising) or since the peak (falling). Dots mark the real data points.
  focus    [{at, panel}]   the panel the voice is on (others dim a little); panel -1 = none
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]   full-bleed photos over the chart
  big      [{at, until, value, from, prefix, suffix, dec, sub, y, size, col, stroke, nodim}]   counting number card (nodim: chart stays bright)
  quote    [{at, until, text, who, y, size}]                                    sourced quote card
  hook_y   default 330 (hook over the map);  title_y 300;  subs_y 1612;  src_y 1488;  credit_y 1712
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

UP, DN = "#22C55E", "#EF4444"
DARK = np.array([34, 19, 11], np.float32)          # BGR of #0B1322, the trading-screen background


def _load(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"pop_stock: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z):
    h, w = im.shape[:2]; s = max(W / w, H / h) * max(1.0, z)
    cw, ch = W / s, H / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def fmt_pop(v):
    if v >= 1e6: return f"{v / 1e6:.2f}M"
    return f"{v / 1e3:.0f}k"


def _wrapped(cv, txt, cy, a, sz, maxw):
    toks, hl = [], False
    for w in txt.split():
        on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
        toks.append((w.replace("*", ""), hl or on)); hl = (hl or on) and not off
    lines = [[]]
    for tk in toks:
        if lines[-1] and cv.tw(" ".join(x[0] for x in lines[-1] + [tk]), sz) > maxw: lines.append([tk])
        else: lines[-1].append(tk)
    lh = sz * 1.2; y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        parts = [(w + (" " if k < len(ln) - 1 else ""), YEL if h else WHITE) for k, (w, h) in enumerate(ln)]
        cv.rich(parts, 540, y0 + i * lh, sz, a, maxw + 60)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    MP = D.get("map", {}); MAP_T1 = A(MP.get("until"), 1.3); MAP_BACK = A(MP.get("back"), T.END + 5)
    Y0, Y1 = D.get("years", [1945, 2028]); TICKS = D.get("ticks", [1950, 1970, 1990, 2010])
    X0, X1 = 170, 1030                                       # plot x range (px)

    # ---------------------------------------------------------------- map (static render, Ken Burns zoom)
    lon, lat, k, sy = MP.get("view", [35.0, 31.45, 300, 1060])
    MR = MapR(); cosf = math.cos(math.radians(lat))
    mimg, M = MR.render(MR.view(lon, lat, k, cosf, sy), H)
    mimg = mimg.astype(np.float32)
    ZM = float(MP.get("zoom", 1.08))
    yy = np.arange(H, dtype=np.float32)[:, None]
    MVIG = (1 - np.clip(1 - yy / 620, 0, 1) ** 1.6 * 0.55)[..., None]
    mimg = mimg * MVIG
    LABS = [dict(l_, xy=geo2px(M, l_["lon"], l_["lat"])) for l_ in MP.get("labels", [])]

    def map_at(t, z):
        Mz = cv2.getRotationMatrix2D((540, sy), 0, z)
        img = cv2.warpAffine(mimg, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cv = Canvas(img)
        for l_ in LABS:
            x, y = l_["xy"]; x = 540 + (x - 540) * z; y = sy + (y - sy) * z
            WG.tag(cv, l_["text"], x + l_.get("dx", 0), y + l_.get("dy", 0), l_.get("size", 46), l_.get("col", WHITE), 1.0)
        return cv.out().astype(np.float32)

    # ---------------------------------------------------------------- panels
    P = []
    for p in D["panels"]:
        q = dict(p); q["t0"] = A(p["at"]); q["t1"] = A(p.get("until"), T.END + 5); pts = [tuple(x) for x in p["points"]]; q["pts"] = pts
        steps, prev = [], pts[0][0]
        for s_ in p["draw"]:
            a0 = A(s_["at"]); a1 = A(s_.get("until"), a0 + 1.5); steps.append((a0, max(a0 + 0.2, a1), prev, s_["to"])); prev = s_["to"]
        q["steps"] = steps
        q["notes"] = [dict(n, t0=A(n["at"])) for n in p.get("notes", [])]
        P.append(q)
    FOC = [(A(f["at"]), f["panel"]) for f in D.get("focus", [])]

    def head_year(q, t):
        y = q["pts"][0][0]
        for a0, a1, f, to in q["steps"]:
            if t <= a0: break
            u = min(1.0, (t - a0) / (a1 - a0)); u = u * u * (3 - 2 * u)
            y = f + (to - f) * u
        return y

    def val_at(q, yr):
        pts = q["pts"]
        if yr <= pts[0][0]: return pts[0][1]
        for (ya, va), (yb, vb) in zip(pts, pts[1:]):
            if yr <= yb: return va + (vb - va) * (yr - ya) / (yb - ya)
        return pts[-1][1]

    xof = lambda yr: X0 + (X1 - X0) * (yr - Y0) / (Y1 - Y0)

    def draw_panel(cv, q, t, a, foc):
        if a <= 0.01: return
        y0 = q["y"]; h = q["h"]; vmax = q["unit_max"]
        top = y0 + 92; bot = y0 + h - 58                       # plot area
        yof = lambda v: bot - (bot - top) * v / vmax
        da = a * (1.0 if foc else 0.55)
        cv.rrect(28, y0, 1024, h, 26, "#0F1B2E", 0.92 * a, stroke="#22324A", lw=2, sa=a)
        step = q.get("grid", vmax / 2)
        g = 0.0
        while g <= vmax + 1e-6:                                # grid + y labels
            yg = yof(g); cv.line([(X0, yg), (X1, yg)], "#22324A", 2, a)
            cv.text((f"{g / 1e6:g}M" if g >= 1e6 else fmt_pop(g)) if g else "0", X0 - 18, yg, 36, "#8FA3BF", da, bold=False, align="r", shadow=False)
            g += step
        for yr in TICKS:
            x = xof(yr); cv.line([(x, top), (x, bot)], "#1B2A40", 2, a)
            cv.text(str(yr), x, bot + 32, 36, "#8FA3BF", da, bold=False, shadow=False)
        hy = head_year(q, t)
        if hy > q["pts"][0][0] + 1e-3:
            pts = [(yr, v) for yr, v in q["pts"] if yr < hy] + [(hy, val_at(q, hy))]
            # area fill per segment, then line per segment
            for (ya, va), (yb, vb) in zip(pts, pts[1:]):
                col = UP if vb >= va - 1e-9 else DN; c = cv.c; c.new_path()
                c.move_to(xof(ya), bot); c.line_to(xof(ya), yof(va)); c.line_to(xof(yb), yof(vb)); c.line_to(xof(yb), bot); c.close_path()
                cv.col(col, 0.13 * da); c.fill()
            for (ya, va), (yb, vb) in zip(pts, pts[1:]):
                col = UP if vb >= va - 1e-9 else DN
                cv.line([(xof(ya), yof(va)), (xof(yb), yof(vb))], col, 9, da)
            for yr, v in q["pts"]:                             # real data points
                if yr <= hy + 1e-6: cv.circ(xof(yr), yof(v), 7, "#E6EEF8", 0.9 * da)
            hv = val_at(q, hy); last = pts[-1]; prev = pts[-2]
            falling = last[1] < prev[1] - 1e-9
            col = DN if falling else UP
            pr = (t * 1.2) % 1.0
            cv.circ(xof(hy), yof(hv), 12 + 18 * pr, None, 1, stroke=col, lw=3); cv.c.new_path()
            cv.circ(xof(hy), yof(hv), 12, col, da)
            if falling:
                peak = max(v for yr, v in pts); pct = (peak - hv) / peak * 100; arrow = "▼"
            else:
                pct = (hv - q["pts"][0][1]) / q["pts"][0][1] * 100; arrow = "▲"
            tick = f"{fmt_pop(hv)} {arrow} {pct:,.1f}%" if pct < 10 else f"{fmt_pop(hv)} {arrow} {pct:,.0f}%"
        else:
            col = "#C9D6EA"; tick = fmt_pop(q["pts"][0][1])
        fl = q.get("flags", []); lx = 56
        for f_ in fl: WG.flag(cv, f_, lx + 26, y0 + 50, 52, a); lx += 62
        if fl: lx += 6
        cv.text(q["label"], lx, y0 + 50, 52, col, a, align="l", maxw=560 - (lx - 56), shadow=False)
        cv.text(tick, 1024, y0 + 50, 56, col, a, align="r", maxw=520, shadow=False)
        for n in q["notes"]:
            na = a * seg(t, n["t0"], n["t0"] + 0.3)
            if na > 0.01: cv.text(n["text"], n.get("x", xof(n["year"]) + 14), n.get("y", top + 24), n.get("size", 36), {"red": DN, "green": UP}.get(n.get("col"), n.get("col", "#C9D6EA")), na, bold=n.get("bold", False), align=n.get("align", "l"), shadow=True)

    # ---------------------------------------------------------------- overlays
    PH = [dict(p, t0=A(p["at"]), im=_load(p["file"])) for p in D.get("photos", [])]
    for i, p in enumerate(PH): p["t1"] = A(p.get("until"), PH[i + 1]["t0"] if i + 1 < len(PH) else T.END)
    XF = float(D.get("cut", 0.25))
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    QT = [dict(q, t0=A(q["at"]), t1=A(q.get("until"), T.END - 0.6)) for q in D.get("quote", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.END - 3) if S.get("end") else None
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 330)
    VIG = (1 - np.maximum(np.clip(1 - yy / 560, 0, 1) ** 1.5 * 0.62, np.clip((yy - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    NAVY = np.array([97, 50, 22], np.float32)
    DARKBG = np.broadcast_to(DARK, (H, W, 3)).astype(np.float32)

    def photo_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"] + XF))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.25)
        return img * (1 - dim) * VIG + NAVY * dim * 0.35

    def photo_alpha(p, t):
        return min(seg(t, p["t0"] - XF, p["t0"], lambda x: min(1, max(0, x))), 1 - seg(t, p["t1"], p["t1"] + XF, lambda x: min(1, max(0, x))))

    def chart_layer(t):
        cv = Canvas(DARKBG.copy())
        foc = None
        for a0, pi in FOC:
            if t >= a0: foc = pi
        for i, q in enumerate(P):
            a = seg(t, q["t0"] - 0.1, q["t0"] + 0.35) * (1 - seg(t, q["t1"] - 0.3, q["t1"]))
            if a > 0.01:
                c = cv.c; c.save(); c.translate(0, 40 * (1 - a)); draw_panel(cv, q, t, a, foc is None or foc == i or foc < 0); c.restore()
        return cv.out().astype(np.float32)

    def frame(t):
        # map (frame 0 and the circular ending) <-> chart screen
        ma = max(1 - seg(t, MAP_T1 - 0.15, MAP_T1 + 0.25), seg(t, MAP_BACK, MAP_BACK + 0.4))
        if ma >= 0.999:
            z = 1 + (ZM - 1) * (eout(t / max(0.5, MAP_T1)) if t < MAP_BACK else 1 - eout((t - MAP_BACK) / max(0.5, T.END - MAP_BACK)))
            base = map_at(t, z)
        else:
            base = chart_layer(t)
            if ma > 0.001:
                z = ZM if t < MAP_BACK else ZM
                base = base * (1 - ma) + map_at(t, z if t < MAP_BACK else ZM) * ma
        for p in PH:
            pa = photo_alpha(p, t)
            if pa > 0.001: base = base * (1 - pa) + photo_layer(p, t) * pa
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in [b_ for b_ in BIG if not b_.get("nodim")] + QT] + ([seg(t, END_AT - 0.1, END_AT + 0.3) * (1 - seg(t, MAP_BACK - 0.2, MAP_BACK + 0.2))] if END_AT else []) + [0])
        if over > 0: base = base * (1 - 0.5 * over)
        cv = Canvas(base)
        for p in PH:
            ca = win(t, p["t0"], p["t1"], 0.25, 0.25)
            if ca > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ca, bold=False, maxw=1000, minsz=36)
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(90, y - 185, 900, 370, 40, "#0B1730", 0.9 * ga, stroke=g.get("stroke", YEL), lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.1))
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 45, g.get("size", 160), g.get("col", YEL), ga, maxw=820)
                for j, sl in enumerate(g.get("sub", "").split("|")):
                    if sl: cv.text(sl, 540, y + 85 + 64 * j, 54, WHITE, ga, maxw=840)
        for q in QT:
            qa = win(t, q["t0"] - 0.1, q["t1"], 0.35, 0.35)
            if qa > 0.01:
                y = q.get("y", 920); WG.panel(cv, 60, y - 280, 960, 560, qa, stroke=YEL)
                cv.text("“", 140, y - 205, 150, YEL, qa, align="l", shadow=False)
                _wrapped(cv, q["text"], y - 20, qa, q.get("size", 64), 860)
                if q.get("who"): cv.text(q["who"], 540, y + 200, 42, "#C9D6EA", qa, bold=False, maxw=900, minsz=36)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 300), 76)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if END_AT is None or t < MAP_BACK: draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0) if t < MAP_BACK else T.END + 5, HY, 1.0)
        if t >= MAP_BACK: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, MAP_BACK, MAP_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(MAP_T1, "whoosh", -28), (MAP_BACK, "whoosh", -30)] + [(p["t0"], "whoosh", -32) for p in PH]
    ev += [(q["t0"], "pop", -28) for q in P]
    ev += [(g["t0"], "roll", g["t0"] + 1.1) for g in BIG] + [(q["t0"], "pop", -26) for q in QT]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
