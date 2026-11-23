"""h2h (@noamroz__ data story): "Israel vs X, N numbers". Graphs on screen the whole time. Builder AA, 23 Nov 2026.

Beats follow each other with a slide (in from the right, out to the left); every beat is one chart:
  map    true-size copies of Israel drop onto a target country (equal-area, from templates/data/israel_fits_geo.json,
         same data as israel_fits), a big yellow "x N" counter ticks up. Use it for frame 0 (hook over the map).
  pair   head-to-head: two thick horizontal bars on ONE linear scale (Israel yellow, the other country sky), values count up
         with the bars, optional lighter "extra" segment per row (e.g. reservists on top of active troops) with a legend,
         a yellow punch sticker ("423x bigger") and a small note.
  rank   one sorted bar chart (rows given best first), Israel highlighted in its true place, "no data" pills,
         optional punch sticker.
A progress chip ("3/12") sits top right, the "Israel Facts" mark top left. Circular: draw_end card, the last 0.5 s
dissolves into frame 0 (the map + hook).

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  map      {at, until, view: [lon, lat, width_km, screen_y], target: "USA", n, value, dec, name, sub, hero, hero_gap, fill,
            drop_at, counter_y}
  beats    [{kind: "pair" | "rank", at, until, num, title, title_at (title waits, e.g. under the frame 0 hook), unit,
             rows: [{label, flag, value, dec, prefix, suffix, txt, at, extra, extra_txt, hl}],
             legend: ["solid label", "light label"], punch: {txt, sub, at}, note, vmax}]
           rows of a pair: Israel first (hl true = yellow); row "at" missing = 0.3 s after the beat, 0.35 s apart
  total    progress total (the chip shows "num/total"); omit for no chip
  brand    "Israel Facts" (text of the top left mark, "" = none)
  hook_y / subs_y / src_y
"""
import os, sys, math, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
from israel_fits import fwd, inv, GEOF
from map_counter import draw_brand

LAB = "#C9D6EA"; DARK = "#0B1A3C"; GREYP = "#5B6B85"
OCEAN0, OCEAN1, LAND, LAND_B = "#07142E", "#0C1D40", "#1D4579", "#3C70B0"
TGT_FILL, TGT_LINE = "#2A5E9C", "#8FD0F5"


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    TOTAL = D.get("total"); BRAND = D.get("brand", "Israel Facts")

    # ------------------------------------------------------------ beats
    B = [dict(b) for b in D.get("beats", [])]
    for b in B: b["t0"] = A(b["at"])
    MP = dict(D["map"]) if D.get("map") else None
    if MP:
        MP["t0"] = A(MP.get("at", 0.0))
        MP["t1"] = A(MP.get("until")) if MP.get("until") is not None else (B[0]["t0"] if B else T.END - 0.6)
    for i, b in enumerate(B):
        b["t1"] = A(b["until"]) if b.get("until") is not None else (B[i + 1]["t0"] if i + 1 < len(B) else T.END - 0.6)
        for k, r in enumerate(b["rows"]):
            r["t"] = A(r["at"]) if r.get("at") is not None else b["t0"] + 0.3 + 0.35 * k
        p = b.get("punch")
        b["pt"] = A(p.get("at")) if p and p.get("at") is not None else (max(r["t"] for r in b["rows"]) + 0.9 if p else None)

    # ------------------------------------------------------------ map (equal-area, true-size Israels)
    if MP:
        G = json.load(open(GEOF)); TXY = np.array(G["tile_xy"], float)
        tg = G["targets"][MP.get("target", "USA")]
        cen = tg["tiles"][:int(MP.get("n", len(tg["tiles"])))]
        TILES = [np.c_[inv(TXY[:, 0], TXY[:, 1], lo, la)] for lo, la in cen]
        TRINGS = [np.array(r, float) for r in tg["rings"]]
        WORLD = [np.array(r, float) for rs in G["world"].values() for r in rs]
        WORLD = [r for r in WORLD if r[:, 1].min() > -60]
        lon0, lat0, wkm = MP["view"][:3]; SY = MP["view"][3] if len(MP["view"]) > 3 else 1000
        sc = W / wkm
        def P(ll):
            x, y, cc = fwd(ll[:, 0], ll[:, 1], lon0, lat0); return np.c_[540 + x * sc, SY - y * sc], cc
        # map image without tiles (static camera, so render once)
        y_ = np.arange(H, dtype=np.float32)[:, None]
        oc = (np.array(rgb(OCEAN0)[::-1]) * 255 * (1 - y_[..., None] / H) + np.array(rgb(OCEAN1)[::-1]) * 255 * (y_[..., None] / H)).astype(np.float32)
        cvm = Canvas(np.broadcast_to(oc, (H, W, 3)).copy()); c = cvm.c
        def rings(cv, rs, fill, fa, stroke, sa, lw):
            cv.c.new_path(); n_ = 0
            for r in rs:
                pts, cc = P(r)
                if cc.max() < 0 or pts[:, 0].max() < -60 or pts[:, 0].min() > W + 60 or pts[:, 1].max() < -60 or pts[:, 1].min() > H + 60: continue
                cv.c.move_to(*pts[0])
                for q in pts[1:]: cv.c.line_to(q[0], q[1])
                cv.c.close_path(); n_ += 1
            if not n_: return
            cv.c.set_fill_rule(cairo.FILL_RULE_WINDING); cv.col(fill, fa); cv.c.fill_preserve()
            cv.col(stroke, sa); cv.c.set_line_width(lw); cv.c.set_line_join(cairo.LINE_JOIN_ROUND); cv.c.stroke(); cv.c.new_path()
        rings(cvm, WORLD, LAND, 1.0, LAND_B, 0.55, 1.2)
        rings(cvm, TRINGS, TGT_FILL, 0.95, TGT_LINE, 0.95, 3.0)
        MAPIMG = cvm.out().astype(np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None]
        MAPIMG *= (1 - np.maximum(np.clip(1 - yy / 560, 0, 1) ** 1.6 * 0.5, np.clip((yy - 1300) / 620, 0, 1) ** 1.2 * 0.6))[..., None]
        TPTS = [P(tl)[0] for tl in TILES]
        n = len(TILES); hero = min(n, int(MP.get("hero", 3))); hg = float(MP.get("hero_gap", 0.4)); fill = float(MP.get("fill", 2.0))
        td = A(MP.get("drop_at"), MP["t0"] + 0.6)
        TT = [td + i * hg for i in range(hero)]
        ts = td + hero * hg
        for j in range(n - hero): TT.append(ts + fill * (((j + 1) / (n - hero)) ** 0.55))
        TFULL = max(TT)

    def draw_map(cv, t, a):
        c = cv.c
        c.save(); c.new_path()                                     # copies clipped to the target's shape (as israel_fits)
        for r in TRINGS:
            pts, cc = P(r)
            if cc.max() < 0: continue
            c.move_to(*pts[0])
            for q_ in pts[1:]: c.line_to(q_[0], q_[1])
            c.close_path()
        c.clip()
        for k, (pts, tk) in enumerate(zip(TPTS, TT)):
            u = (t - (tk - 0.22)) / 0.22
            if u < 0: continue
            hero_ = k < hero
            s_ = 1 + (0.45 if hero_ else 0.3) * (1 - eout(min(1, u))); al = min(1, u * 1.6) * a
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean(); dy = -(70 * (1 - eout(min(1, u)))) if hero_ else 0
            q = (pts - [cx, cy]) * s_ + [cx, cy + dy]
            c.new_path(); c.move_to(*q[0])
            for p in q[1:]: c.line_to(p[0], p[1])
            c.close_path(); cv.col(SKY, 0.88 * al); c.fill_preserve(); cv.col("#FFFFFF", 0.95 * al); c.set_line_width(1.6); c.stroke()
        c.restore()
        # counter
        CY = MP.get("counter_y", 1290)
        landed = sum(1 for x in TT if x <= t); final = MP.get("value", n); dec = int(MP.get("dec", 0))
        ca = a * seg(t, td - 0.2, td + 0.1)
        if ca <= 0.01: return
        if t >= TFULL + 0.2 and abs(final - landed) > 1e-6: v = WG.roll(t, TFULL + 0.2, landed, final, 0.4)
        else: v = landed if t < TFULL + 0.2 else final
        txt = "x " + WG.fmtn(v, dec if t >= TFULL + 0.2 else 0)
        pulse = 1 + 0.06 * (1 - eout(min(1, (t - max([x for x in TT if x <= t] or [-9])) / 0.18)))
        cv.rrect(110, CY - 120, 860, 300, 44, DARK, 0.78 * ca, stroke=SKY, lw=2.5, sa=0.5 * ca)
        cv.text(MP.get("name", ""), 540, CY - 58, 60, WHITE, ca, maxw=800)
        c.save(); c.translate(540, CY + 48); c.scale(pulse, pulse); c.translate(-540, -(CY + 48))
        cv.text(txt, 540, CY + 48, 132, YEL, ca, maxw=800); c.restore()
        cv.text(MP.get("sub", "true-size Israels"), 540, CY + 140, 42, LAB, ca, bold=False, maxw=800)

    # ------------------------------------------------------------ charts
    def fmt_row(r, v, done):
        if done and r.get("txt"): return r["txt"]
        return WG.fmtn(v, int(r.get("dec", 0)), r.get("prefix", ""), r.get("suffix", ""))

    def sticker(cv, p, ts, t, y, a, dx):
        if t < ts: return
        pp = WG.pop_s(t, ts); sub = p.get("sub"); bh = 150 if sub else 120
        w = max(cv.tw(p["txt"], 84) + 80, (cv.tw(sub, 40) + 60) if sub else 0, 260); w = min(w, 1000)
        sx = 540 + dx; al = a * min(1, pp)
        c = cv.c; c.save(); c.translate(sx, y); c.scale(min(1.12, pp), min(1.12, pp)); c.translate(-sx, -y)
        cv.rrect(sx - w / 2, y - bh / 2, w, bh, 34, YEL, al)
        cv.text(p["txt"], sx, y - (22 if sub else 0), 84, DARK, al, maxw=w - 30, shadow=False, minsz=48)
        if sub: cv.text(sub, sx, y + 44, 40, DARK, al, maxw=w - 30, shadow=False, bold=False, minsz=36)
        c.restore()

    def draw_pair(cv, b, t, a, dx, ta):
        WG.title(cv, b["title"], 400, ta, 76)
        if b.get("unit"): cv.text(b["unit"], 540, 500, 44, LAB, ta, bold=False, maxw=960)
        rows = b["rows"]; vmax = b.get("vmax") or max((r["value"] or 0) + (r.get("extra") or 0) for r in rows)
        x0, bwmax = 70, 940
        for k, r in enumerate(rows):
            yl = 650 + k * 300; yb = yl + 112
            ra = a * seg(t, r["t"] - 0.15, r["t"] + 0.2)
            if ra <= 0.01: continue
            col = YEL if r.get("hl") else SKY
            WG.flag(cv, r["flag"], x0 + 38 + dx, yl, 74, ra)
            cv.text(r["label"], x0 + 92 + dx, yl, 68, WHITE, ra, align="l", maxw=520)
            u = eout((t - r["t"]) / 0.9); v = (r["value"] or 0) * u
            w1 = max(14, bwmax * v / vmax)
            cv.rrect(x0 + dx, yb - 62, w1, 124, 26, col, ra)
            ex = r.get("extra")
            tx_end = x0 + w1
            if ex:
                ue = eout((t - r["t"] - 0.7) / 0.9)
                if ue > 0:
                    w2 = bwmax * ex * ue / vmax
                    cv.rrect(x0 + w1 - 26 + dx, yb - 62, w2 + 26, 124, 26, col, 0.38 * ra, stroke=col, lw=3, sa=0.9 * ra)
                    cv.rrect(x0 + w1 - 4 + dx, yb - 62, 8, 124, 0, DARK, 0.5 * ra)
                    tx_end = x0 + w1 + w2
            done = u > 0.995
            txt = fmt_row(r, v, done)
            if ex and t > r["t"] + 0.7: txt = r.get("extra_txt") or txt
            tw = cv.tw(txt, 84)
            if tx_end + 22 + tw <= 1050: cv.text(txt, tx_end + 22 + dx, yb, 84, YEL if r.get("hl") else WHITE, ra, align="l", maxw=1050 - tx_end - 22, minsz=56)
            else: cv.text(txt, tx_end - 26 + dx, yb, 84, DARK, ra, align="r", maxw=tx_end - x0 - 40, shadow=False, minsz=56)
        if b.get("legend"):
            la = a * seg(t, rows[0]["t"] + 0.7, rows[0]["t"] + 1.0)
            l0, l1 = b["legend"]; y = 1200
            cv.rrect(90 + dx, y - 22, 44, 44, 10, YEL, la); cv.text(l0, 150 + dx, y, 44, LAB, la, align="l", bold=False, maxw=360)
            cv.rrect(560 + dx, y - 22, 44, 44, 10, YEL, 0.38 * la, stroke=YEL, lw=3, sa=0.9 * la); cv.text(l1, 620 + dx, y, 44, LAB, la, align="l", bold=False, maxw=420)
        if b.get("punch"): sticker(cv, b["punch"], b["pt"], t, 1320, a, dx)
        if b.get("note"): cv.text(b["note"], 540 + dx, 1425, 42, LAB, a * seg(t, b["t0"] + 0.4, b["t0"] + 0.8), bold=False, maxw=980, minsz=36)

    def draw_rank(cv, b, t, a, dx, ta):
        WG.title(cv, b["title"], 400, ta, 74)
        if b.get("unit"): cv.text(b["unit"], 540, 496, 44, LAB, ta, bold=False, maxw=960)
        rows = b["rows"]; n = len(rows); rowh = min(104, 800 // max(1, n)); y0 = 590 + rowh / 2
        vals = [r["value"] for r in rows if r.get("value") is not None]; vmax = b.get("vmax") or max(vals)
        xl, xb, bwmax = 56, 440, 400
        for k, r in enumerate(rows):
            y = y0 + k * rowh; ra = a * seg(t, r["t"] - 0.1, r["t"] + 0.2)
            if ra <= 0.01: continue
            if r.get("hl"): cv.rrect(40 + dx, y - rowh / 2 + 6, 1000, rowh - 12, 22, YEL, 0.16 * ra, stroke=YEL, lw=2.5, sa=0.7 * ra)
            WG.flag(cv, r["flag"], xl + 34 + dx, y, 58, ra)
            cv.text(r["label"], xl + 78 + dx, y, 56, YEL if r.get("hl") else WHITE, ra, align="l", maxw=xb - xl - 92, minsz=40)
            if r.get("value") is None:
                cv.rrect(xb + dx, y - 28, 190, 56, 28, GREYP, ra); cv.text("no data", xb + 95 + dx, y, 40, WHITE, ra, maxw=180, shadow=False)
                continue
            u = eout((t - r["t"]) / 0.8); v = r["value"] * u
            w1 = max(10, bwmax * v / vmax)
            cv.rrect(xb + dx, y - rowh * 0.3, w1, rowh * 0.6, 14, YEL if r.get("hl") else SKY, ra)
            cv.text(fmt_row(r, v, u > 0.995), xb + w1 + 16 + dx, y, 56, YEL if r.get("hl") else WHITE, ra, align="l", maxw=1060 - xb - w1 - 16, minsz=40)
        if b.get("punch"): sticker(cv, b["punch"], b["pt"], t, y0 + n * rowh + 40, a, dx)
        if b.get("note"): cv.text(b["note"], 540 + dx, 1415, 42, LAB, a, bold=False, maxw=980, minsz=36)

    def chip(cv, b, a):
        if not TOTAL or not b.get("num"): return
        txt = f"{b['num']}/{TOTAL}"; w = cv.tw(txt, 46) + 60
        cv.rrect(1010 - w, 196, w, 64, 32, DARK, 0.8 * a, stroke=YEL, lw=2, sa=0.8 * a)
        cv.text(txt, 1010 - w / 2, 228, 46, YEL, a, maxw=w, shadow=False)

    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 560)
    E = S.get("end") or {}
    END0 = A(E.get("at"), T.S(len(T.L) - 2)) if E else T.END
    END1 = A(E.get("until"), T.END - 0.5) if E else T.END

    def frame(t):
        if MP and t < MP["t1"] + 0.5:
            ma = 1 - seg(t, MP["t1"] - 0.05, MP["t1"] + 0.45)
            base = bars_moving_bg(t, 10) * (1 - ma) + MAPIMG * ma
        else: base = bars_moving_bg(t, 10)
        cv = Canvas(base)
        if MP and t < MP["t1"] + 0.5:
            ma = 1 - seg(t, MP["t1"] - 0.05, MP["t1"] + 0.45)
            if ma > 0.01: draw_map(cv, t, ma)
        dime = win(t, END0 - 0.1, END1, 0.35, 0.3) if E else 0
        for b in B:
            if not (b["t0"] - 0.05 <= t <= b["t1"] + 0.45): continue
            u_in = 1.0 if b["t0"] <= 0.01 else seg(t, b["t0"] - 0.05, b["t0"] + 0.35); u_out = seg(t, b["t1"] - 0.05, b["t1"] + 0.35)
            dx = 1080 * (1 - u_in) - 1080 * u_out
            a = (1 - 0.95 * dime) * min(1.0, 0.25 + 0.75 * u_in) * (1 - 0.75 * u_out)
            if a <= 0.01: continue
            ta = (1 - 0.95 * dime) * u_in * (1 - u_out)           # titles fade (no overlap), charts slide
            if b.get("title_at") is not None: ta *= seg(t, A(b["title_at"]), A(b["title_at"]) + 0.35)
            (draw_pair if b["kind"] == "pair" else draw_rank)(cv, b, t, a, dx, ta)
            chip(cv, b, ta)
        if BRAND: draw_brand(cv, BRAND, 1.0)
        draw_screen(cv, cards, t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1490)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    if MP:
        for tk in TT[:hero]: ev += [(tk, "thud", -20), (tk, "pop", -26)]
        if len(TT) > 3: ev.append((TT[3], "roll", TFULL))
        ev.append((TFULL + 0.3, "ding", -26))
    for b in B:
        ev.append((b["t0"] - 0.05, "whoosh", -27))
        for r in b["rows"]: ev.append((r["t"], "tick", -30))
        if b.get("pt") is not None: ev.append((b["pt"], "pop", -24))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], beats=len(B))
