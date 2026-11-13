"""zoom_stock: "zoom out on the chart". A dark trading screen with a vertical 0 to 100 scale. It opens ZOOMED IN: a gold
UTOPIA line at the top (100), Israel's price line well below it with a small map of Israel, the hook over a top band.
Flaw blocks (red) drop into the gap between Israel and utopia one by one (each a sourced metric, real photo in the
top band). Then the scale ZOOMS OUT to 0 to 100 and the neighbours' lines draw in far below with right axis price tags
(de-overlapped), bar cards (distance to utopia, a second index), end card, and the scale zooms back in for the loop.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  scale     {"lo0": 58, "hi": 104, "lo1": 0, "zoom_at": anchor, "zoom_dur": 1.8, "back_at": anchor, "back_dur": 1.2}
  box0/box1 chart pixel box zoomed in / out: {"x0": 120, "x1": 820, "y0": 600, "y1": 1400}
  utopia    {"value": 100, "label": "UTOPIA", "note": "small method label (>= 36 px)"}
  israel    {"value": 73, "ticker": "ISR", "flag": "IL", "map": true}
  blocks    [{at, label, value}]   red blocks stacked from israel.value up to utopia.value (equal heights)
  rows      [{ticker, flag, value, at, label}]   neighbour lines (draw in left to right at `at`)
  bars      [{at, until, title, sub, rows: [{label, flag, value, txt, hl}], dec, suffix, y}]
  photos    [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], band: true|false, dim}]
            band (default true): photo fills the top band (0..band_h) and fades into the screen; false: full bleed
  blocks_until  anchor: the blocks fade out (for the loop back to frame 0)
  band_h    default 640;  hook_y default 300;  title_y default 330;  subs_y 1612;  src_y 1488;  credit_y 1712
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

UP, DOWN = "#2BD67B", "#F0524F"
GRID = "#1C2A44"; AXIS = "#8FA3C0"; SCREEN = (34, 20, 10)   # BGR near-black navy trading screen
GOLDL = "#FFD23F"


def _band(im, cx, cy, z, bh):
    """cover crop of im to W x bh at zoom z centred at (cx, cy) (0..1 image coords) -> float32 BGR"""
    h, w = im.shape[:2]; s = max(W / w, bh / h) * max(1.0, z)
    cw, ch = W / s, bh / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    SC = dict(dict(lo0=58, hi=104, lo1=0, zoom_dur=1.8, back_dur=1.2), **D.get("scale", {}))
    B0 = dict(dict(x0=120, x1=820, y0=600, y1=1400), **D.get("box0", {}))
    B1 = dict(dict(x0=120, x1=820, y0=480, y1=1420), **D.get("box1", {}))
    Zt, Zd = A(SC.get("zoom_at"), 1e9), float(SC["zoom_dur"])
    Bt, Bd = A(SC.get("back_at"), 1e9), float(SC["back_dur"])
    def zoom(t): return ease((t - Zt) / Zd) * (1 - ease((t - Bt) / Bd))
    def geom(t):
        z = zoom(t); lo = SC["lo0"] + (SC["lo1"] - SC["lo0"]) * z; hi = SC["hi"]
        bx = {k: B0[k] + (B1[k] - B0[k]) * z for k in B0}
        return z, lo, hi, bx
    def Yv(v, g): z, lo, hi, bx = g; return bx["y1"] - (v - lo) / (hi - lo) * (bx["y1"] - bx["y0"])

    U = dict(dict(value=100, label="UTOPIA"), **D.get("utopia", {}))
    IL = dict(dict(value=73, ticker="ISR", flag="IL", map=True), **D.get("israel", {}))
    BL = [dict(b, t0=A(b["at"])) for b in D.get("blocks", [])]
    BLU = A(D.get("blocks_until"), 1e9)
    RW = [dict(r, t0=A(r["at"])) for r in D.get("rows", [])]
    BARS = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("bars", [])]
    BH = int(D.get("band_h", 640))

    # small map of Israel (whole land gold, West Bank and Gaza darker, Green Line green) for the ISR line
    MAPI = None
    if IL.get("map"):
        MR = MapR(); lon, lat = 35.0, 31.45; cosf = math.cos(math.radians(lat)); k = 62.0
        mimg, _ = MR.render(MR.view(lon, lat, k, cosf, 960), H)
        cy, cx = 960, 540; hh, ww = 150, 95
        crop = mimg[cy - hh:cy + hh, cx - ww:cx + ww].astype(np.float32)
        yy, xx = np.mgrid[0:2 * hh, 0:2 * ww].astype(np.float32)
        r = 26; dx = np.maximum(0, np.maximum(r - xx, xx - (2 * ww - 1 - r))); dy = np.maximum(0, np.maximum(r - yy, yy - (2 * hh - 1 - r)))
        MAPI = (crop, np.clip(r - np.sqrt(dx ** 2 + dy ** 2) + 0.5, 0, 1)[..., None])

    PH = []
    if D.get("photos"):
        from photo_story import _load, _kb
        for p in D["photos"]:
            PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))))
        for k_, p in enumerate(PH):
            if not p.get("until"): p["t1"] = PH[k_ + 1]["t0"] if k_ + 1 < len(PH) else T.END
    yb = np.arange(BH, dtype=np.float32)[:, None, None]
    BANDF = np.clip(1 - (yb - BH * 0.55) / (BH * 0.45), 0, 1) ** 1.3          # band fades into the screen at its bottom
    yf = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - yf / 560, 0, 1) ** 1.5 * 0.62, np.clip((yf - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    def ph_alpha(p, t): return win(t, p["t0"] - 0.15, p["t1"] + 0.1, 0.3, 0.3) if p["t0"] > 0.01 else 1 - seg(t, p["t1"] - 0.2, p["t1"] + 0.1)
    def ph_img(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        c = [kb[i] + (kb[i + 3] - kb[i]) * u for i in range(3)]
        dim = p.get("dim", 0.3)
        if p.get("band", True): return _band(p["im"], *c, BH) * (1 - dim), True
        from photo_story import _kb
        return _kb(p["im"], *c) * (1 - dim) * VIG, False

    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    bg = np.zeros((H, W, 3), np.float32); bg[:] = SCREEN
    yy_ = np.arange(H, dtype=np.float32)[:, None, None]
    bg = bg + (np.array([97, 50, 22], np.float32) - np.array(SCREEN, np.float32)) * 0.35 * np.clip(yy_ / H, 0, 1)

    def tag(cv, txt, y, col, a, sz=44, fg="#06101F"):
        tw = cv.tw(txt, sz) + 30; x = W - 14 - tw
        cv.rrect(x, y - sz * 0.68, tw, sz * 1.36, 10, col, a)
        cv.text(txt, x + tw / 2, y, sz, fg, a, shadow=False, maxw=tw)

    def draw_chart(cv, t, ca, g):
        if ca <= 0.01: return
        z, lo, hi, bx = g
        step = 10 if z < 0.5 else 20
        v = math.ceil(lo / step) * step
        while v <= hi - 2:
            y = Yv(v, g); cv.line([(bx["x0"], y), (bx["x1"] + 10, y)], GRID, 2, ca)
            cv.text(f"{v:.0f}", bx["x0"] - 16, y, 38, AXIS, ca, bold=False, align="r", shadow=False)
            v += step
        for k in range(1, 7):
            x = bx["x0"] + (bx["x1"] - bx["x0"]) * k / 6; cv.line([(x, bx["y0"] - 10), (x, bx["y1"])], GRID, 1.5, 0.6 * ca)
        x0, x1 = bx["x0"], bx["x1"] + 10
        yu, yi = Yv(U["value"], g), Yv(IL["value"], g)
        # gap shading + bracket
        ga = ca * (1 - 0.6 * z)
        cv.rrect(x0, yu, x1 - x0, yi - yu, 0, DOWN, 0.06 * ga)
        # blocks
        n = max(1, len(BL)); hb = (IL["value"] - U["value"]) / n * -1
        for k, b in enumerate(BL):
            u = seg(t, b["t0"] - 0.05, b["t0"] + 0.35, eout) * (1 - seg(t, BLU, BLU + 0.6))
            if u <= 0.001: continue
            vb, vt = IL["value"] + hb * k, IL["value"] + hb * (k + 1)
            ya, yb_ = Yv(vt, g), Yv(vb, g); drop = (1 - u) * -90
            hpx = yb_ - ya; pad = min(5, hpx * 0.08)
            cv.rrect(x0 + 6, ya + pad + drop, x1 - x0 - 12, hpx - 2 * pad, min(14, hpx * 0.25), DOWN, 0.82 * u * ca * (1 - 0.6 * z))
            ta = u * ca * (1 - seg(z, 0.15, 0.45, lambda q: q))
            if ta > 0.01:
                sz = min(50, hpx * 0.56)
                cv.text(b["label"], x0 + 30, (ya + yb_) / 2 + drop, sz, WHITE, ta, align="l", maxw=430, minsz=36)
                if b.get("value"): cv.text(b["value"], x1 - 30, (ya + yb_) / 2 + drop, sz, YEL, ta, align="r", maxw=300, minsz=36)
        # utopia line
        cv.line([(x0, yu), (x1, yu)], GOLDL, 6, ca, dash=[22, 12])
        cv.text(U["label"], x0 + 8, yu - 46, 64 - 14 * z, GOLDL, ca, align="l", maxw=600)
        tag(cv, f"{U['value']:.0f}", yu, GOLDL, ca)
        if U.get("note"):
            nx = x0 + 8 + cv.tw(U["label"], 64 - 14 * z) + 26
            for k, ln in enumerate(U["note"].split("|")):
                cv.text(ln, nx, yu - 64 + 34 * k, 36, "#C9D6EA", ca * 0.95, bold=False, align="l", maxw=x1 - nx)
        # neighbour lines
        heads = []
        for r in RW:
            u = seg(t, r["t0"], r["t0"] + 0.45, eout)
            ra = ca * min(1.0, zoom(t) * 1.5) * u
            if ra <= 0.01: continue
            y = Yv(r["value"], g); xe = x0 + (x1 - x0) * u
            cv.line([(x0, y), (xe, y)], DOWN if r["value"] < 50 else SKY, 4, ra)
            heads.append(dict(r=r, y=y, a=ra))
        # israel line on top
        cv.line([(x0, yi), (x1, yi)], UP, 9, ca); cv.circ(x1, yi, 12, UP, ca)
        heads.append(dict(r=dict(ticker=IL["ticker"], value=IL["value"], flag=IL.get("flag"), hl=True), y=yi, a=ca))
        heads.sort(key=lambda h: h["y"]); gap = 58
        ys = [h["y"] for h in heads]
        for _ in range(80):
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < gap:
                    d = (gap - (ys[k] - ys[k - 1])) / 2; ys[k - 1] -= d; ys[k] += d
            ys = [min(max(y, yu + 60), 1450) for y in ys]
        for h, ly in zip(heads, ys):
            r = h["r"]; hl = r.get("hl")
            txt = f"{r['ticker']} {r['value']:.0f}"
            if abs(ly - h["y"]) > 4: cv.line([(x1 + 6, h["y"]), (W - 210, ly)], UP if hl else DOWN, 2, 0.7 * h["a"])
            tag(cv, txt, ly, UP if hl else (DOWN if r["value"] < 50 else SKY), h["a"], 44 if hl else 40)
            if r.get("flag"): WG.flag(cv, r["flag"], W - 30 - cv.tw(txt, 44 if hl else 40) - 60, ly, 44, h["a"])
        # "gap" bracket in the zoomed view
        ba = ca * (1 - z) * (1 - seg(t, BL[0]["t0"] - 0.2, BL[0]["t0"] + 0.2)) if BL else 0
        if ba > 0.01:
            xb = x1 - 30; cv.line([(xb, yu + 12), (xb, yi - 12)], WHITE, 4, ba, dash=[10, 10])
            cv.text(f"{U['value'] - IL['value']:.0f} points", (x0 + x1) / 2, (yu + yi) / 2, 64, WHITE, ba, maxw=600)

    def frame(t):
        g = geom(t); z = g[0]
        base = bg.copy(); pa_full = 0.0; band_a = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ <= 0.001: continue
            img, is_band = ph_img(p, t)
            if is_band:
                fa = BANDF * a_; base[:BH] = base[:BH] * (1 - fa) + img * fa; band_a = max(band_a, a_)
            else:
                base = base * (1 - a_) + img * a_; pa_full = 1 - (1 - pa_full) * (1 - a_)
        # israel mini map (under the line, left), zoomed in only
        if MAPI is not None:
            ma = (1 - seg(z, 0.0, 0.35, lambda q: q)) * (1 - pa_full)
            if ma > 0.01:
                crop, msk = MAPI; yi = Yv(IL["value"], g); hh, ww = crop.shape[0] // 2, crop.shape[1] // 2
                cx, cy = int(g[3]["x0"] + 30 + ww), int(yi + 30 + hh)
                y0_, x0_ = cy - hh, cx - ww
                if y0_ + 2 * hh <= H:
                    m = msk * ma; base[y0_:y0_ + 2 * hh, x0_:x0_ + 2 * ww] = base[y0_:y0_ + 2 * hh, x0_:x0_ + 2 * ww] * (1 - m) + crop * m
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BARS] + [0])
        cv = Canvas(base)
        ca = (1 - pa_full) * (1 - 0.82 * over)
        draw_chart(cv, t, ca, g)
        if MAPI is not None:
            ma = (1 - seg(z, 0.0, 0.35, lambda q: q)) * (1 - pa_full)
            if ma > 0.01:
                yi = Yv(IL["value"], g); crop = MAPI[0]
                cv.rrect(g[3]["x0"] + 30, yi + 30, crop.shape[1], crop.shape[0], 26, None, ma, stroke=UP, lw=3, sa=0.8 * ma)
                cv.text("Israel", g[3]["x0"] + 30 + crop.shape[1] + 24, yi + 30 + crop.shape[0] / 2, 52, WHITE, ma, align="l", maxw=400)
        for b in BARS:
            a = win(t, b["t0"] - 0.1, b["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            y0 = b.get("y", 620); n = len(b["rows"]); rh = b.get("rowh", min(92, 760 // max(1, n)))
            WG.panel(cv, 40, y0 - 170, 1000, rh * n + 220, a, stroke=YEL)
            WG.title(cv, b["title"], y0 - 110, a, 58)
            if b.get("sub"): cv.text(b["sub"], 540, y0 - 50, 38, "#C9D6EA", a, bold=False, maxw=960)
            vmax = max([r["value"] for r in b["rows"] if r.get("value") is not None] or [1])
            for k, r in enumerate(b["rows"]):
                ra = a * seg(t, b["t0"] + 0.07 * k, b["t0"] + 0.07 * k + 0.25); y = y0 + 30 + k * rh
                if ra <= 0.01: continue
                if r.get("flag"): WG.flag(cv, r["flag"], 100, y, 52, ra)
                cv.text(r["label"], 140, y, 46, YEL if r.get("hl") else WHITE, ra, align="l", maxw=250, minsz=36)
                bx = 400; bwm = 400
                if r.get("value") is None:
                    cv.rrect(bx, y - rh * 0.28, 220, rh * 0.56, 12, "#3A4A66", ra); cv.text(r.get("txt", "no data"), bx + 110, y, 38, "#C9D6EA", ra, bold=False, maxw=210)
                    continue
                u = eout((t - b["t0"] - 0.07 * k) / 0.6); bw = max(8, bwm * r["value"] / vmax * u)
                cv.rrect(bx, y - rh * 0.3, bw, rh * 0.6, 12, YEL if r.get("hl") else r.get("color", SKY), ra)
                cv.text(r.get("txt") or WG.fmtn(r["value"], b.get("dec", 0), "", b.get("suffix", "")), bx + bw + 14, y, 44, YEL if r.get("hl") else WHITE, ra, align="l", maxw=230, minsz=36)
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1040, minsz=36)
        draw_screen(cv, cards, t, D.get("title_y", 330), 70)
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0) if D.get("hook_until") is None else A(D["hook_until"]), HY, 0.92)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(b["t0"], "thud", -22) for b in BL] + [(r["t0"], "tick", -24) for r in RW]
    if Zt < 1e8: ev.append((Zt, "whoosh", -24))
    if Bt < 1e8: ev.append((Bt, "whoosh", -30))
    ev += [(p["t0"], "whoosh", -32) for p in PH if p["t0"] > 0.01] + [(b["t0"], "pop", -22) for b in BARS]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
