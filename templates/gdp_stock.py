"""gdp_stock: "if countries were stocks" on ONE shared chart. A dark trading screen where one line per country draws left
to right (year by year, green segments up, red down) with ticker labels at the line heads ("ISR $60,337"), tick sounds
while drawing, annotations on key years, an optional second pass that morphs every line to an adjusted series (e.g. GDP
per person minus natural resource rents; years without adjusted data fade out = honest gaps, the unadjusted lines stay
as dashed ghosts), a bar card, big number cards, full-bleed real photos between beats, a map of the compared countries at
frame 0 (hook over it, chart by about 1.5 s), circular ending (the last 0.5 s dissolves into the frame-0 map).
(Separate from stock_chart.py / pop_stock.py, which draw one panel per place.)

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  file       repo path to a JSON: {"years": [1960, ...], "series": [{"code": "ISR", "ticker": "ISR", "main": true,
             "hl": true, "v": [...|null], "adj": [...|null]}]}   null = no data (gap)
  map        {"cam": [lon, lat, width_deg], "y": 1000, "until": anchor, "codes": [...], "labels": [{lon, lat, text, hl}], "credit"}
  keys       [[year, anchor], ...]  pass 1: the drawing head is at `year` at that time (repeat a year = hold)
  pass2      {"at": anchor, "dur": 2.0, "ghost": true}  lines morph v -> adj; years without adj fade out
  y          {"max": 70000, "step": 10000}   linear axis from 0
  chart      {"x0": 120, "x1": 730, "y0": 470, "y1": 1330}
  marks      [{at, until, year, code, text, dx, dy, col: yellow|red|green}]  arrow + label at a series' value in a year
  notes      [{at, until, text, y, size, col, bold}]  small fixed labels (>= 36 px), hidden under photos
  bars       [{at, until, title, sub, rows: [{label, flag, value|null, txt, hl, color}], dec, suffix, y, rowh}]
  big        [{at, until, value, from, prefix, suffix, dec, sub, y, size}]
  photos     [{file, at, until, credit, kb, dim}]  (as in photo_story); the chart fades while a photo is up
  disclaimer {"text", "y", "at", "until"}  always-on small line under the chart
  hook_y     hook centre over the map (default 300); hook_until (default end of line 0)
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

UP, DOWN = "#2BD67B", "#F0524F"
GRID = "#22324F"; AXIS = "#8FA3C0"; SCREEN = (34, 20, 10)   # BGR near-black navy trading screen


def _fmt(v):
    if v is None or not np.isfinite(v): return "no data"
    return f"${v:,.0f}"


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    DT = json.load(open(os.path.join(REPO, D["file"]), encoding="utf-8"))
    YRS = np.array(DT["years"], float); NY = len(YRS)
    SER = []
    for s in DT["series"]:
        v = np.array([np.nan if x is None else float(x) for x in s["v"]]); a = s.get("adj")
        adj = np.array([np.nan if x is None else float(x) for x in a]) if a else v.copy()
        SER.append(dict(s, V=v, ADJ=adj))
    C = dict(dict(x0=120, x1=730, y0=470, y1=1330), **D.get("chart", {}))
    Y = dict(dict(max=70000, step=10000), **D.get("y", {}))
    X = lambda yr: C["x0"] + (yr - YRS[0]) / (YRS[-1] - YRS[0]) * (C["x1"] - C["x0"])
    Yp = lambda v: C["y1"] - v / Y["max"] * (C["y1"] - C["y0"])

    KY = [(float(k[0]), A(k[1])) for k in D["keys"]]
    def head(t):
        if t < KY[0][1]: return KY[0][0]
        for (y0, t0), (y1, t1) in zip(KY, KY[1:]):
            if t <= t1: return y0 + (y1 - y0) * min(1, max(0, (t - t0) / max(1e-3, t1 - t0)))
        return KY[-1][0]
    P2 = D.get("pass2") or {}
    P2t = A(P2.get("at")) if P2 else 1e9; P2d = float(P2.get("dur", 2.0))
    def morph(t): return ease((t - P2t) / P2d) if P2 else 0.0

    MP = D.get("map") or {}
    MAPU = A(MP.get("until"), 1.2) if MP else -1
    if MP:
        lon, lat, wd = MP.get("cam", [42, 28, 38]); my = MP.get("y", 1000)
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
    BARS = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("bars", [])]
    MK = [dict(m, t0=A(m["at"]), t1=A(m.get("until"), T.END - 0.6)) for m in D.get("marks", [])]
    NT = [dict(n, t0=A(n.get("at"), 0.0), t1=A(n.get("until"), T.END - 0.6)) for n in D.get("notes", [])]
    DS = D.get("disclaimer")
    if DS: DS = dict(DS, t0=A(DS.get("at"), MAPU), t1=A(DS.get("until"), T.END - 0.6))
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    HU = A(D.get("hook_until"), T.E(0))
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    bg = np.array(SCREEN, np.float32)[None, None] + (np.array([97, 50, 22], np.float32) - np.array(SCREEN, np.float32))[None, None] * 0.35 * np.clip(yy / H, 0, 1)
    bg = np.broadcast_to(bg, (H, W, 3)).astype(np.float32)

    def series_vals(s, m):
        v, a = s["V"], s["ADJ"]
        val = np.where(np.isfinite(a), v * (1 - m) + a * m, v)
        al = np.where(np.isfinite(a), 1.0, 1.0 - m) * np.isfinite(v)
        return val, al

    def tint(cv, code, a, col):
        c = cv.c; c.new_path()
        for ring, _ in MR.country(code):
            pts = [geo(*p) for p in ring]; c.move_to(*pts[0])
            for p in pts[1:]: c.line_to(*p)
            c.close_path()
        cv.col(col, 0.5 * a); c.fill_preserve(); cv.col("#FFFFFF", 0.8 * a); c.set_line_width(2.2); c.stroke(); c.new_path()

    def draw_chart(cv, t, ca):
        if ca <= 0.01: return
        m = morph(t); hy = head(t)
        v = 0
        while v <= Y["max"] + 1:
            y = Yp(v); cv.line([(C["x0"], y), (C["x1"] + 10, y)], GRID, 2, ca)
            cv.text("0" if v == 0 else f"${v // 1000:.0f}k", C["x0"] - 14, y, 36, AXIS, ca, bold=False, align="r", shadow=False)
            v += Y["step"]
        for yr in range(int(YRS[0]), int(YRS[-1]) + 1, 10):
            x = X(yr); cv.line([(x, C["y0"] - 10), (x, C["y1"])], GRID, 1.5, 0.7 * ca)
            cv.text(f"'{yr % 100:02d}", x, C["y1"] + 42, 36, AXIS, ca, bold=False, shadow=False)
        if m > 0.01 and P2.get("ghost", True):
            for s in SER:
                pts = [(X(YRS[i]), Yp(s["V"][i])) for i in range(NY) if np.isfinite(s["V"][i]) and YRS[i] <= hy]
                if len(pts) > 1: cv.line(pts, "#7F8CA3", 3 if s.get("main") else 2, 0.4 * m * ca, dash=[8, 8])
        heads = []
        for s in sorted(SER, key=lambda s: (bool(s.get("hl")), bool(s.get("main")))):
            val, al = series_vals(s, m)
            lw = 11 if s.get("hl") else (7 if s.get("main") else 4); aa = 1.0 if (s.get("main") or s.get("hl")) else 0.75
            last = None
            for i in range(NY - 1):
                if YRS[i] >= hy: break
                if not (np.isfinite(val[i]) and np.isfinite(val[i + 1])): continue
                f = min(1.0, (hy - YRS[i]) / (YRS[i + 1] - YRS[i]))
                v1 = val[i] + (val[i + 1] - val[i]) * f
                sa = min(al[i], al[i + 1]) * aa * ca
                if sa <= 0.01: continue
                col = UP if val[i + 1] >= val[i] else DOWN
                xe = X(YRS[i] + (YRS[i + 1] - YRS[i]) * f)
                if s.get("hl"): cv.line([(X(YRS[i]), Yp(val[i])), (xe, Yp(v1))], YEL, lw + 12, 0.32 * sa)
                cv.line([(X(YRS[i]), Yp(val[i])), (xe, Yp(v1))], col, lw, sa)
                last = (xe, Yp(v1), v1, col, min(al[i], al[i + 1]))
            if last is None:
                idx = [i for i in range(NY) if np.isfinite(val[i]) and YRS[i] <= hy + 1e-6]
                if idx and al[idx[-1]] > 0.01: i = idx[-1]; last = (X(YRS[i]), Yp(val[i]), val[i], UP, al[i])
            if last and last[4] > 0.01:
                cv.circ(last[0], last[1], lw * 0.8 + 3, last[3], aa * ca * last[4])
                heads.append(dict(s=s, x=last[0], y=last[1], v=last[2], col=last[3], a=aa * ca * last[4]))
        heads.sort(key=lambda h: h["y"]); gap = 50
        ys = [h["y"] for h in heads]
        for _ in range(80):
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < gap:
                    d = (gap - (ys[k] - ys[k - 1])) / 2; ys[k - 1] -= d; ys[k] += d
            ys = [min(max(y, C["y0"] - 30), C["y1"] + 8) for y in ys]
        for h, ly in zip(heads, ys):
            sz = 46 if (h["s"].get("main") or h["s"].get("hl")) else 40
            txt = f"{h['s']['ticker']} {_fmt(h['v'])}" if (h["s"].get("main") or h["s"].get("hl")) else h["s"]["ticker"]
            lx = min(h["x"] + 22, 1072 - cv.tw(txt, sz))
            if abs(ly - h["y"]) > 8: cv.line([(h["x"] + 8, h["y"]), (lx - 6, ly)], h["col"], 2, 0.45 * h["a"])
            col = YEL if h["s"].get("hl") else h["col"]
            cv.text(txt, lx, ly, sz, col, h["a"], align="l", maxw=1074 - lx, minsz=36, shadow=True)
        yr = int(math.floor(hy + 1e-6))
        if m < 0.99: cv.text(str(min(yr, int(YRS[-1]))), C["x0"] + 130, C["y0"] + 20, 100, YEL, 0.9 * ca * (1 - m), shadow=True)

    def frame(t):
        base = bg.copy()
        ma = (1.0 - seg(t, MAPU - 0.25, MAPU + 0.15)) if MP else 0.0
        pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        if ma > 0.001: base = base * (1 - ma) + mimg * ma
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BIG + BARS] + [0])
        cv = Canvas(base)
        if ma > 0.01:
            for code in MP.get("codes", []): tint(cv, code, ma, YEL if code == "ISR" else SKY)
            for lab in MP.get("labels", []):
                x, y = geo(lab["lon"], lab["lat"]); WG.tag(cv, lab["text"], x, y, 40, YEL if lab.get("hl") else WHITE, ma)
        ca = (1 - pa) * (1 - ma) * (1 - 0.85 * over)
        draw_chart(cv, t, ca)
        for mk in MK:
            a = win(t, mk["t0"], mk["t1"], 0.25, 0.25) * (1 - pa) * (1 - 0.85 * over)
            if a <= 0.01: continue
            s = next(x for x in SER if x["code"] == mk["code"]); i = int(np.argmin(np.abs(YRS - mk["year"])))
            val, _ = series_vals(s, morph(t)); px, py = X(YRS[i]), Yp(val[i])
            lx, ly = px + mk.get("dx", 0), py + mk.get("dy", -140)
            col = {"red": DOWN, "green": UP, "yellow": YEL}.get(mk.get("col", "yellow"), YEL)
            tw = cv.tw(mk["text"], 50) + 40; lx = min(max(lx, tw / 2 + 16), W - 16 - tw / 2)
            cv.line([(lx, ly + (38 if ly < py else -38)), (px, py + (-16 if ly < py else 16))], col, 4, a)
            cv.circ(px, py, 15, None, a, stroke=col, lw=4)
            cv.rrect(lx - tw / 2, ly - 38, tw, 76, 20, "#0B1730", 0.9 * a, stroke=col, lw=3, sa=a)
            cv.text(mk["text"], lx, ly, 50, WHITE, a, maxw=1040)
        for n in NT:
            a = win(t, n["t0"], n["t1"], 0.25, 0.25) * (1 - pa) * (1 - ma)
            if a > 0.01: cv.text(n["text"], 540, n.get("y", 420), n.get("size", 44), n.get("col", "#C9D6EA"), a, bold=n.get("bold", False), maxw=1040, minsz=36)
        if DS:
            a = win(t, DS["t0"], DS["t1"], 0.25, 0.25) * (1 - pa) * (1 - ma)
            if a > 0.01: cv.text(DS["text"], 540, DS.get("y", 1418), 38, "#9FB0C8", a, bold=False, maxw=1040, minsz=36)
        for b in BARS:
            a = win(t, b["t0"] - 0.1, b["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            y0 = b.get("y", 560); n = len(b["rows"]); rh = b.get("rowh", min(90, 760 // max(1, n)))
            WG.panel(cv, 40, y0 - 175, 1000, rh * n + 215, a, stroke=YEL)
            WG.title(cv, b["title"], y0 - 115, a, 60)
            if b.get("sub"): cv.text(b["sub"], 540, y0 - 55, 38, "#C9D6EA", a, bold=False, maxw=960)
            vmax = max([r["value"] for r in b["rows"] if r.get("value") is not None] or [1])
            for k, r in enumerate(b["rows"]):
                t0r = b["t0"] + 0.07 * k; ra = a * seg(t, t0r, t0r + 0.25); y = y0 + 25 + k * rh
                if ra <= 0.01: continue
                if r.get("flag"): WG.flag(cv, r["flag"], 92, y, 50, ra)
                cv.text(r["label"], 130, y, 46, YEL if r.get("hl") else WHITE, ra, align="l", maxw=250, minsz=36)
                bx = 395; bwm = 450
                if r.get("value") is None:
                    cv.rrect(bx, y - rh * 0.28, 230, rh * 0.56, 12, "#3A4A66", ra); cv.text(r.get("txt", "no data"), bx + 115, y, 38, "#C9D6EA", ra, bold=False, maxw=220)
                    continue
                u = eout((t - t0r) / 0.6); bw = max(8, bwm * r["value"] / vmax * u)
                cv.rrect(bx, y - rh * 0.3, bw, rh * 0.6, 12, YEL if r.get("hl") else r.get("color", SKY), ra)
                cv.text(r.get("txt") or WG.fmtn(r["value"], b.get("dec", 1), "", b.get("suffix", "%")), bx + bw + 14, y, 44,
                        YEL if r.get("hl") else WHITE, ra, align="l", maxw=1060 - bx - bw - 14, minsz=36)
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
    for (y0, t0), (y1, t1) in zip(KY, KY[1:]):
        if y1 > y0 + 0.01:
            step = max(0.09, (t1 - t0) / max(1, y1 - y0))
            for tt in np.arange(t0, t1, step): ev.append((float(tt), "tick", -30))
    if P2: ev.append((P2t, "whoosh", -26))
    ev += [(p["t0"], "whoosh", -30) for p in PH if p["t0"] > 0.01] + [(g["t0"], "pop", -22) for g in BIG] + [(b["t0"], "pop", -22) for b in BARS]
    ev += [(m["t0"], "buzz" if m.get("col") == "red" else "ding", -28) for m in MK]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
