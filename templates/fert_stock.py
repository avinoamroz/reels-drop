"""fert_stock: small-number series (e.g. babies per woman) drawn like a trading screen, panel by panel. Frame 0 is a map
of the places compared (hook over it), the first chart is up by about 1.2 s. Each panel draws its lines left to right
on one shared chart (green segments rise, red fall, grey flat, year by year), with a ticker at every line head
([flag] "ITA 1.2 [arrow]": flag LEFT of the ticker, value with one decimal, arrow = change from the year before, like a
daily stock move), tick sounds while drawing. Then a "board" card (watchlist: flag, name, value, change since a base
year, sorted best first), full-bleed real photos between beats, number cards, end card, and the last 0.5 s dissolves
into the frame-0 map (circular). (Own template; gdp_stock.py / stock_chart.py are untouched.)

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  file     repo JSON {"years": [1960, ...], "series": [{"code", "ticker", "name", "flag", "v": [... | null]}]}
  map      {"cam": [lon, lat, width_deg], "y": 960, "until": anchor, "codes": [...], "labels": [{lon, lat, text, hl}], "credit"}
  panels   [{at, until, codes: [...], ymax, ystep, keys: [[year, anchor], ...], title, hl: "ISR"}]
           keys move the drawing head (repeat a year = hold); the panel shows from `at` to `until`
  board    [{at, until, title, sub, base: 1990, codes: [...], y, rowh, hl}]  watchlist rows sorted best first
  photos   [{file, at, until, credit, kb, dim}]  full-bleed photos; charts fade under them
  big      [{at, until, value, from, prefix, suffix, dec, sub, y, size}]
  notes    [{at, until, text, y, size, col}]  small fixed lines (>= 36 px)
  chart    {"x0": 130, "x1": 640, "y0": 560, "y1": 1330}
  hook_y (300), hook_until (end of line 0), title_y (300), src_y (1488), subs_y (1612), credit_y (1712)
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

UP, DOWN, FLAT = "#2BD67B", "#F0524F", "#9FB0C8"
GRID = "#22324F"; AXIS = "#8FA3C0"; SCREEN = (34, 20, 10)   # BGR near-black navy trading screen


def colof(a, b):
    return UP if b > a + 1e-9 else (DOWN if b < a - 1e-9 else FLAT)


def tri(cv, x, y, s, kind, col, a):
    c = cv.c; c.new_path()
    if kind == "up": c.move_to(x - s, y + s * 0.8); c.line_to(x + s, y + s * 0.8); c.line_to(x, y - s * 0.9)
    elif kind == "down": c.move_to(x - s, y - s * 0.8); c.line_to(x + s, y - s * 0.8); c.line_to(x, y + s * 0.9)
    else: c.rectangle(x - s, y - s * 0.22, 2 * s, s * 0.44)
    c.close_path(); cv.col(col, a); c.fill()


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    DT = json.load(open(os.path.join(REPO, D["file"]), encoding="utf-8"))
    YRS = np.array(DT["years"], float); NY = len(YRS)
    SER = {}
    for s in DT["series"]:
        SER[s["code"]] = dict(s, V=np.array([np.nan if x is None else float(x) for x in s["v"]]))
    C = dict(dict(x0=130, x1=640, y0=560, y1=1330), **D.get("chart", {}))
    X = lambda yr: C["x0"] + (yr - YRS[0]) / (YRS[-1] - YRS[0]) * (C["x1"] - C["x0"])

    PN = []
    for p in D["panels"]:
        ky = [(float(k[0]), A(k[1])) for k in p["keys"]]
        PN.append(dict(p, KY=ky, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6)))
    def head(P, t):
        KY = P["KY"]
        if t < KY[0][1]: return KY[0][0]
        for (y0, t0), (y1, t1) in zip(KY, KY[1:]):
            if t <= t1: return y0 + (y1 - y0) * min(1, max(0, (t - t0) / max(1e-3, t1 - t0)))
        return KY[-1][0]

    MP = D.get("map") or {}
    MAPU = A(MP.get("until"), 1.2) if MP else -1
    if MP:
        lon, lat, wd = MP.get("cam", [30, 40, 70]); my = MP.get("y", 960)
        cosf = max(0.6, math.cos(math.radians(min(62, abs(lat))))); MR = MapR()
        mimg, MM = MR.render(MR.view(lon, lat, W / (wd * cosf), cosf, my), H)
        mimg = mimg.astype(np.float32)
        geo = lambda lo, la: geo2px(MM, lo, la)

    PH = []
    if D.get("photos"):
        from photo_story import _load, _kb
        for p in D["photos"]:
            PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))))
        y_ = np.arange(H, dtype=np.float32)[:, None]
        VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.62, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
        NAVYC = np.array([97, 50, 22], np.float32)
    for p in PH: p["cont"] = any(abs(q["t0"] - p["t1"]) < 0.05 for q in PH if q is not p)
    def ph_alpha(p, t):
        a_in = 1.0 if p["t0"] <= 0.01 else seg(t, p["t0"] - 0.3, p["t0"])
        a_out = (1.0 if t < p["t1"] + 0.05 else 0.0) if p["cont"] else 1.0 - seg(t, p["t1"] - 0.3, p["t1"])
        return min(a_in, a_out)
    def ph_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.25)
        return img * (1 - dim) * VIG + NAVYC * dim * 0.35

    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    BD = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("board", [])]
    NT = [dict(n, t0=A(n.get("at"), 0.0), t1=A(n.get("until"), T.END - 0.6)) for n in D.get("notes", [])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    HU = A(D.get("hook_until"), T.E(0))
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    bg = np.array(SCREEN, np.float32)[None, None] + (np.array([97, 50, 22], np.float32) - np.array(SCREEN, np.float32))[None, None] * 0.35 * np.clip(yy / H, 0, 1)
    bg = np.broadcast_to(bg, (H, W, 3)).astype(np.float32)

    def tint(cv, code, a, col):
        c = cv.c; c.new_path()
        for ring, _ in MR.country(code):
            pts = [geo(*p) for p in ring]; c.move_to(*pts[0])
            for p in pts[1:]: c.line_to(*p)
            c.close_path()
        cv.col(col, 0.5 * a); c.fill_preserve(); cv.col("#FFFFFF", 0.8 * a); c.set_line_width(2.2); c.stroke(); c.new_path()

    def draw_panel(cv, P, t, ca):
        if ca <= 0.01: return
        ymax, ystep = P.get("ymax", 4), P.get("ystep", 1)
        Yp = lambda v: C["y1"] - v / ymax * (C["y1"] - C["y0"])
        hy = head(P, t)
        if P.get("title"): cv.text(P["title"], 60, C["y0"] - 100, 60, WHITE, ca, align="l", maxw=720, minsz=40)
        v = 0.0
        while v <= ymax + 1e-6:
            y = Yp(v); cv.line([(C["x0"], y), (C["x1"] + 10, y)], GRID, 2, ca)
            cv.text(f"{v:g}", C["x0"] - 16, y, 40, AXIS, ca, bold=False, align="r", shadow=False)
            v += ystep
        for yr in range(int(YRS[0]), int(YRS[-1]) + 1, 20):
            x = X(yr); cv.line([(x, C["y0"] - 10), (x, C["y1"])], GRID, 1.5, 0.7 * ca)
            cv.text(str(yr), x, C["y1"] + 44, 38, AXIS, ca, bold=False, shadow=False)
        heads = []
        codes = sorted(P["codes"], key=lambda c: c == P.get("hl"))
        for code in codes:
            s = SER[code]; val = s["V"]; hl = code == P.get("hl")
            lw = 11 if hl else 5; aa = 1.0 if hl else 0.85
            last = None
            for i in range(NY - 1):
                if YRS[i] >= hy: break
                if not (np.isfinite(val[i]) and np.isfinite(val[i + 1])): continue
                f = min(1.0, (hy - YRS[i]) / (YRS[i + 1] - YRS[i]))
                v1 = val[i] + (val[i + 1] - val[i]) * f
                col = colof(val[i], val[i + 1]); xe = X(YRS[i] + (YRS[i + 1] - YRS[i]) * f)
                if hl: cv.line([(X(YRS[i]), Yp(val[i])), (xe, Yp(v1))], YEL, lw + 12, 0.30 * aa * ca)
                cv.line([(X(YRS[i]), Yp(val[i])), (xe, Yp(v1))], col, lw, aa * ca)
                last = (xe, Yp(v1), i, f)
            if last is None:
                idx = [i for i in range(NY) if np.isfinite(val[i]) and YRS[i] <= hy + 1e-6]
                if not idx: continue
                i = idx[-1]; last = (X(YRS[i]), Yp(val[i]), i, 0.0)
            xe, ye, i, f = last
            # ticker shows the last REAL data point reached, arrow vs the year before it
            j = i + 1 if (f >= 0.999 and i + 1 < NY and np.isfinite(val[i + 1])) else i
            prev = val[j - 1] if j > 0 and np.isfinite(val[j - 1]) else val[j]
            kind = "up" if val[j] > prev + 1e-9 else ("down" if val[j] < prev - 1e-9 else "flat")
            col = {"up": UP, "down": DOWN, "flat": FLAT}[kind]
            cv.circ(xe, ye, lw * 0.8 + 3, col, aa * ca)
            heads.append(dict(s=s, hl=hl, x=xe, y=ye, v=val[j], kind=kind, col=col, a=aa * ca))
        heads.sort(key=lambda h: h["y"]); gap = 54
        ys = [h["y"] for h in heads]
        for _ in range(150):
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < gap:
                    d = (gap - (ys[k] - ys[k - 1])) / 2; ys[k - 1] -= d; ys[k] += d
            ys = [min(max(y, C["y0"] - 40), C["y1"] + 30) for y in ys]
        for h, ly in zip(heads, ys):
            sz = 46 if h["hl"] else 42; fs = sz + 4
            txt = f"{h['s']['ticker']} {h['v']:.1f}"
            tw = cv.tw(txt, sz)
            lx = min(h["x"] + 24, 1060 - fs - 10 - tw - sz * 0.9)       # lx = flag's left edge
            if abs(ly - h["y"]) > 8: cv.line([(h["x"] + 8, h["y"]), (lx - 6, ly)], h["col"], 2, 0.45 * h["a"])
            if h["s"].get("flag"): WG.flag(cv, h["s"]["flag"], lx + fs / 2, ly, fs, h["a"])
            tx = lx + fs + 10
            cv.text(txt, tx, ly, sz, YEL if h["hl"] else WHITE, h["a"], align="l", maxw=1060 - tx, minsz=36, shadow=True)
            tri(cv, tx + tw + sz * 0.55, ly, sz * 0.3, h["kind"], h["col"], h["a"])
        yr = int(math.floor(hy + 1e-6))
        cv.text(str(min(yr, int(YRS[-1]))), 1020, C["y0"] - 100, 84, YEL, 0.9 * ca, align="r", shadow=True)

    def draw_board(cv, b, t, a):
        base = b.get("base", 1990); bi = int(np.argmin(np.abs(YRS - base)))
        rows = []
        for code in b["codes"]:
            v = SER[code]["V"]; rows.append(dict(s=SER[code], now=v[-1], pct=(v[-1] / v[bi] - 1) * 100))
        rows.sort(key=lambda r: -r["pct"])
        n = len(rows); rh = b.get("rowh", min(64, 900 // n)); y0 = b.get("y", 470)
        WG.panel(cv, 30, y0 - 150, 1020, rh * n + 180, a, stroke=YEL)
        cv.text(b.get("title", f"Since {base}"), 540, y0 - 95, 58, WHITE, a, maxw=980)
        if b.get("sub"): cv.text(b["sub"], 540, y0 - 45, 38, "#C9D6EA", a, bold=False, maxw=980)
        sz = min(46, int(rh * 0.76))
        for k, r in enumerate(rows):
            t0r = b["t0"] + 0.06 * k; ra = a * seg(t, t0r, t0r + 0.25); y = y0 + 20 + k * rh + rh / 2
            if ra <= 0.01: continue
            hl = r["s"]["code"] == b.get("hl"); up = r["pct"] > 0; col = UP if up else DOWN
            if hl: cv.rrect(44, y - rh * 0.48, 992, rh * 0.96, 14, YEL, 0.16 * ra)
            if r["s"].get("flag"): WG.flag(cv, r["s"]["flag"], 88, y, sz + 4, ra)
            cv.text(r["s"].get("name", r["s"]["ticker"]), 128, y, sz, YEL if hl else WHITE, ra, align="l", maxw=440, minsz=36)
            cv.text(f"{r['now']:.1f}", 650, y, sz, YEL if hl else WHITE, ra, align="r", maxw=140, minsz=36)
            tri(cv, 710, y, sz * 0.3, "up" if up else "down", col, ra)
            cv.text(f"{'+' if up else ''}{r['pct']:.0f}%", 1010, y, sz, col, ra, align="r", maxw=270, minsz=36)

    def frame(t):
        base = bg.copy()
        ma = (1.0 - seg(t, MAPU - 0.25, MAPU + 0.15)) if MP else 0.0
        pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        if ma > 0.001: base = base * (1 - ma) + mimg * ma
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BIG + BD] + [0])
        cv = Canvas(base)
        if ma > 0.01:
            for code in MP.get("codes", []): tint(cv, code, ma, YEL if code == "ISR" else SKY)
            for lab in MP.get("labels", []):
                x, y = geo(lab["lon"], lab["lat"]); WG.tag(cv, lab["text"], x, y, 40, YEL if lab.get("hl") else WHITE, ma)
        for P in PN:
            pa_ = win(t, P["t0"] - 0.05, P["t1"], 0.3, 0.3) * (1 - pa) * (1 - ma) * (1 - 0.9 * over)
            draw_panel(cv, P, t, pa_)
        for n in NT:
            a = win(t, n["t0"], n["t1"], 0.25, 0.25) * (1 - pa) * (1 - ma) * (1 - over)
            if a > 0.01: cv.text(n["text"], 540, n.get("y", 1420), n.get("size", 40), n.get("col", "#C9D6EA"), a, bold=False, maxw=1040, minsz=36)
        for b in BD:
            a = win(t, b["t0"] - 0.1, b["t1"], 0.3, 0.3)
            if a > 0.01: draw_board(cv, b, t, a)
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 900); cv.rrect(110, y - 175, 860, 350, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.0))
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 35, g.get("size", 170), YEL, ga, maxw=800)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 52, WHITE, ga, maxw=820, minsz=36)
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1040, minsz=36)
        if ma > 0.01 and MP.get("credit"):
            cv.text(MP["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ma, bold=False, maxw=1040, minsz=36)
        draw_screen(cv, cards, t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, HU, HY, 0.92)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    for P in PN:
        for (y0, t0), (y1, t1) in zip(P["KY"], P["KY"][1:]):
            if y1 > y0 + 0.01:
                step = max(0.12, (t1 - t0) / max(1, y1 - y0))
                for tt in np.arange(t0, t1, step): ev.append((float(tt), "tick", -32))
    ev += [(p["t0"], "whoosh", -30) for p in PH if p["t0"] > 0.01] + [(g["t0"], "pop", -22) for g in BIG] + [(b["t0"], "pop", -22) for b in BD]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
