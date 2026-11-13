"""swot: "SWOT analysis of nations". One country at a time on a 2x2 SWOT grid (Strengths green, Weaknesses red,
Opportunities sky, Threats yellow), a header with the flag LEFT of the country name, one short item per quadrant
(max 4 words, >= 64 px) that stamps in on its word, a real photo band on top with a mini map of the country in the
corner, a small source label per country. Frame 0: a world map with the countries tinted and pinned, the hook over it,
an empty S W O T grid flying in by about 1.5 s. Then a scoreboard of all the mini grids on one screen (the red boxes
pulse on a word), the end card (balance line + CTA), and the world map + hook return for the loop into frame 0.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  map      {view: [lon, lat, k, sy], grid_at: anchor (empty grid flies in, default 0.9), back: anchor (map + hook return)}
  countries [{name, flag: "IL", code: "ISR", at: anchor (country comes in),
             mini: [lon, lat, k]  (mini map view; the country tinted sky, Israel gold as on every map),
             pin: [lon, lat]      (pin on the frame 0 world map, default = mini centre), fdx/fdy: flag offset from the pin,
             photo: {file, credit, kb: [cx0, cy0, z0, cx1, cy1, z1]},
             items: {S: {text, short, at}, W: {...}, O: {...}, T: {...}}}]   text max 4 words; short = scoreboard word(s)
  board    {at: anchor, red_at: anchor (red boxes pulse), legend: true}
  hook_y 420; band [y0, y1] default [150, 610]; grid_y0 640; grid_y1 1440; src_y 1488; subs_y 1612
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
import map_tear as mt

QC = {"S": GREEN, "W": RED, "O": SKY, "T": YEL}
QL = {"S": "STRENGTH", "W": "WEAKNESS", "O": "OPPORTUNITY", "T": "THREAT"}
ORDER = ("S", "W", "O", "T")
NAVYC = np.array([97, 50, 22], np.float32)


def _wrap(cv, txt, maxw, sz, min_sz=64, max_lines=2):
    """-> (lines, size): greedy wrap, shrinking down to min_sz before allowing a 3rd line"""
    words = txt.split()
    while True:
        lines = [[]]
        for w in words:
            if lines[-1] and cv.tw(" ".join(lines[-1] + [w]), sz) > maxw: lines.append([w])
            else: lines[-1].append(w)
        if (len(lines) <= max_lines and all(cv.tw(" ".join(l), sz) <= maxw for l in lines)) or sz <= min_sz:
            return [" ".join(l) for l in lines], sz
        sz -= 2


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    MP = D.get("map", {})
    GRID_AT = A(MP.get("grid_at"), 0.9); BACK = A(MP.get("back"), T.END + 5)
    CS = []
    for c in D["countries"]:
        c = dict(c); c["t0"] = A(c["at"])
        c["it"] = {k: dict(v, t=A(v["at"])) for k, v in c["items"].items()}
        CS.append(c)
    for i, c in enumerate(CS): c["t1"] = CS[i + 1]["t0"] if i + 1 < len(CS) else None
    BD = D.get("board", {}); BOARD_AT = A(BD.get("at"), T.END - 6); RED_AT = A(BD.get("red_at"), BOARD_AT + 1.5)
    for c in CS:
        if c["t1"] is None: c["t1"] = BOARD_AT
    END_AT = A((S.get("end") or {}).get("at"), T.END - 3) if S.get("end") else BACK

    by0, by1 = D.get("band", [150, 610]); bx0, bx1 = 24, 1056; bw, bh = bx1 - bx0, by1 - by0
    gy0, gy1 = D.get("grid_y0", 640), D.get("grid_y1", 1440); gap = 16
    qw = (1080 - 48 - gap) / 2; qh = (gy1 - gy0 - gap) / 2
    QR = {"S": (24, gy0), "W": (24 + qw + gap, gy0), "O": (24, gy0 + qh + gap), "T": (24 + qw + gap, gy0 + qh + gap)}

    # ------------------------------------------------------------- world map (frame 0 + loop)
    MR = MapR()
    lon, lat, kk, sy = MP.get("view", [12.0, 36.0, 5.0, 1040])
    cosf = math.cos(math.radians(lat)); v0 = MR.view(lon, lat, kk, cosf, sy)
    wimg, WM = MR.render(v0, H)
    vb = (-400, -400, 400, 400)
    for c in CS:
        if c["code"] in ("ISR",): continue
        mt.fill(wimg, WM, MR.country(c["code"]), mt.hexc("45A4DD"), 0.92, None, outline=mt.hexc("FFFFFF"), lw=2, oalpha=0.8)
    if any(c["code"] == "ISR" for c in CS):       # Israel is tiny at world scale: redraw it gold on top so it shows
        mt.fill(wimg, WM, MR.isr, mt.GOLD, 1.0, None); mt.fill(wimg, WM, MR.terr, mt.hexc("D9A93F") * 0.78 + mt.hexc("7A5A10") * 0.22, 1.0, None)
    yy = np.arange(H, dtype=np.float32)[:, None]
    wimg = wimg.astype(np.float32) * (1 - np.clip(1 - yy / 700, 0, 1) ** 1.5 * 0.5)[..., None]
    PINS = [(c, geo2px(WM, *(c.get("pin") or c["mini"][:2]))) for c in CS]
    def world_at(z, dx):
        Mz = cv2.getRotationMatrix2D((540, sy), 0, z); Mz[0, 2] += dx
        return cv2.warpAffine(wimg, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32), Mz

    # ------------------------------------------------------------- mini maps (one per country)
    MS = 236
    def mini_map(c):
        mlon, mlat, mk = c["mini"]; cf = math.cos(math.radians(mlat))
        v = MR.view(mlon, mlat, mk, cf, 300)
        img, M = MR.render(v, 600)
        if c["code"] != "ISR":
            mt.fill(img, M, MR.country(c["code"]), mt.hexc("45A4DD"), 0.95, None, outline=mt.hexc("FFFFFF"), lw=2, oalpha=0.9)
            mt.fill(img, M, MR.isr, mt.GOLD, 1.0, None); mt.fill(img, M, MR.terr, mt.hexc("D9A93F") * 0.78 + mt.hexc("7A5A10") * 0.22, 1.0, None)
        return img[300 - MS // 2:300 + MS // 2, 540 - MS // 2:540 + MS // 2].astype(np.float32)
    for c in CS: c["mm"] = mini_map(c)
    mmask = np.zeros((MS, MS), np.uint8); r_ = 26
    cv2.rectangle(mmask, (r_, 0), (MS - r_, MS), 255, -1); cv2.rectangle(mmask, (0, r_), (MS, MS - r_), 255, -1)
    for cx_, cy_ in ((r_, r_), (MS - r_, r_), (r_, MS - r_), (MS - r_, MS - r_)): cv2.circle(mmask, (cx_, cy_), r_, 255, -1)
    MMK = (mmask.astype(np.float32) / 255)[..., None]
    mx0, my0 = bx1 - 20 - MS, by0 + 20

    # ------------------------------------------------------------- photos in the band
    from photo_story import _load
    for c in CS:
        p = c.get("photo")
        if p and os.environ.get("TPL_SNAPS") and not os.path.exists(os.path.join(REPO, p["file"])): p = None   # local look only
        c["im"] = _load(p["file"]) if p else None
    bmask = np.zeros((bh, bw), np.uint8); r_ = 30
    cv2.rectangle(bmask, (r_, 0), (bw - r_, bh), 255, -1); cv2.rectangle(bmask, (0, r_), (bw, bh - r_), 255, -1)
    for cx_, cy_ in ((r_, r_), (bw - r_, r_), (r_, bh - r_), (bw - r_, bh - r_)): cv2.circle(bmask, (cx_, cy_), r_, 255, -1)
    BM = (bmask.astype(np.float32) / 255)[..., None]
    byy = np.arange(bh, dtype=np.float32)[:, None]
    BVIG = (1 - np.clip((byy - bh * 0.45) / (bh * 0.55), 0, 1) ** 1.2 * 0.72)[..., None]
    def band_kb(im, cx, cy, z):
        h_, w_ = im.shape[:2]; s_ = max(bw / w_, bh / h_) * max(1.0, z)
        cw, ch = bw / s_, bh / s_
        x0_ = min(max(cx * w_ - cw / 2, 0), w_ - cw); y0_ = min(max(cy * h_ - ch / 2, 0), h_ - ch)
        M_ = np.array([[s_, 0, -x0_ * s_], [0, s_, -y0_ * s_]], np.float32)
        return cv2.warpAffine(im, M_, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    def photo_img(c, t):
        if c["im"] is None: return np.zeros((bh, bw, 3), np.float32) + NAVYC
        kb = c["photo"].get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - c["t0"]) / max(0.5, c["t1"] - c["t0"] + 0.4))
        return band_kb(c["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u) * 0.86

    bg = BG.copy()
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 420)
    CH = sub_chunks(T, S.get("subs_replace")); cards = screen_cards(S, T)

    def cur_country(t):
        """-> list of (country, alpha) for the band/header/items crossfade"""
        out = []
        for c in CS:
            a = win(t, c["t0"] - 0.2, c["t1"] + 0.2, 0.3, 0.3) if c is not CS[-1] else seg(t, c["t0"] - 0.2, c["t0"] + 0.1) * (1 - seg(t, BOARD_AT - 0.2, BOARD_AT + 0.25))
            if a > 0.001: out.append((c, a))
        return out

    # ------------------------------------------------------------- drawing pieces
    def draw_quads(cv, t, ga, act):
        """the 2x2 grid; act = [(country, alpha)]"""
        for k in ORDER:
            x, y = QR[k]; col = QC[k]
            glow = 0.0
            for c, ca in act:
                it = c["it"].get(k)
                if it: glow = max(glow, ca * (1 - seg(t, it["t"], it["t"] + 0.7)) * (t >= it["t"]))
            cv.rrect(x, y, qw, qh, 30, col, (0.16 + 0.22 * glow) * ga, stroke=col, lw=4 + 4 * glow, sa=(0.75 + 0.25 * glow) * ga)
            cv.text(k, x + qw - 70, y + qh - 82, 210, col, 0.13 * ga, align="c", shadow=False)
            cv.text(QL[k], x + 30, y + 52, 42, col, ga, align="l", shadow=False)
            for c, ca in act:
                it = c["it"].get(k)
                if not it or t < it["t"] - 0.02: continue
                u = seg(t, it["t"], it["t"] + 0.28, lambda z: z)
                sc = 1.35 - 0.35 * eout(u); ia = ga * ca * min(1.0, u * 2.5)
                lines, sz = _wrap(cv, it["text"], qw - 60, 76)
                sz = int(sz * sc); lh = sz * 1.12; cy = y + 52 + (qh - 52) / 2 + 8
                for li, ln in enumerate(lines):
                    cv.text(ln, x + qw / 2, cy + (li - (len(lines) - 1) / 2) * lh, sz, WHITE, ia, maxw=2000, minsz=64)

    def draw_header(cv, t, act):
        for c, ca in act:
            u = seg(t, c["t0"] - 0.2, c["t0"] + 0.25)
            dx = (1 - eout(u)) * 120
            hy = by1 - 62
            WG.flag(cv, c["flag"], bx0 + 70 + dx, hy, 92, ca)
            cv.text(c["name"], bx0 + 152 + dx, hy, 100, WHITE, ca, align="l", maxw=740)
            n = CS.index(c) + 1
            cv.text(f"{n} / {len(CS)}", bx1 - 30, hy + 4, 44, "#C9D6EA", ca, align="r", bold=False)

    def draw_board(cv, t, ba):
        cw_, ch_ = 508, 286; xs = (24, 548); y0 = 236
        for i, c in enumerate(CS + [None]):
            col_, row = i % 2, i // 2
            x, y = xs[col_], y0 + row * (ch_ + 16)
            pa = ba * seg(t, BOARD_AT + 0.12 * i, BOARD_AT + 0.12 * i + 0.3)
            if pa <= 0.01: continue
            sc = 0.9 + 0.1 * back(seg(t, BOARD_AT + 0.12 * i, BOARD_AT + 0.12 * i + 0.4, lambda z: z))
            cx, cy = x + cw_ / 2, y + ch_ / 2
            X = lambda px: cx + (px - cx) * sc; Y = lambda py: cy + (py - cy) * sc
            cv.rrect(X(x), Y(y), cw_ * sc, ch_ * sc, 26, "#0B1A3C", 0.9 * pa, stroke="#2A4A7C", lw=2.5, sa=pa)
            if c is None:      # legend slot: one row per quadrant colour
                for j, k in enumerate(ORDER):
                    ly = y + 50 + j * 62
                    cv.rrect(X(x + 30), Y(ly - 26), 52 * sc, 52 * sc, 12, QC[k], 0.9 * pa)
                    cv.text(k, X(x + 56), Y(ly), 40, "#0B1730", pa, shadow=False)
                    cv.text(QL[k].title(), X(x + 104), Y(ly), 40, WHITE, pa, align="l", bold=False, maxw=380)
                continue
            WG.flag(cv, c["flag"], X(x + 44), Y(y + 40), 50 * sc, pa)
            cv.text(c["name"], X(x + 92), Y(y + 40), int(46 * sc), WHITE, pa, align="l", maxw=400)
            qw_, qh_ = (cw_ - 30) / 2, (ch_ - 86) / 2
            for j, k in enumerate(ORDER):
                qx = x + 10 + (j % 2) * (qw_ + 10); qy = y + 76 + (j // 2) * (qh_ + 8)
                pulse = 0.0
                if k == "W" and t >= RED_AT: pulse = 0.5 + 0.5 * math.sin((t - RED_AT) * 9) * math.exp(-(t - RED_AT) * 0.8)
                cv.rrect(X(qx), Y(qy), qw_ * sc, qh_ * sc, 14, QC[k], (0.32 + 0.5 * max(0, pulse)) * pa, stroke=QC[k], lw=2 + 4 * max(0, pulse), sa=pa)
                cv.text(c["it"][k].get("short", ""), X(qx + qw_ / 2), Y(qy + qh_ / 2), int(38 * sc), WHITE, pa, maxw=qw_ - 14, minsz=36)

    def frame(t):
        # world map layer: frame 0 until the first country, and again from BACK
        ma = max(1 - seg(t, CS[0]["t0"] - 0.35, CS[0]["t0"] + 0.1), seg(t, BACK, BACK + 0.45))
        z = 1.0 + 0.05 * eout(t / 2.5) if t < BACK else 1.0 + 0.05 * (1 - eout((t - BACK) / max(0.5, T.END - BACK)))
        dx = -14 * eout(t / 2.5) if t < BACK else -14 * (1 - eout((t - BACK) / max(0.5, T.END - BACK)))
        base = bg.copy(); Mz = None
        if ma > 0.001:
            wm, Mz = world_at(z, dx); base = base * (1 - ma) + wm * ma
        act = cur_country(t)
        boa = seg(t, BOARD_AT - 0.1, BOARD_AT + 0.3) * (1 - seg(t, BACK - 0.1, BACK + 0.3))
        sa_ = sum(a for _, a in act)
        if sa_ > 0.001 and ma < 0.999:
            reg = base[by0:by1, bx0:bx1]; ph = np.zeros((bh, bw, 3), np.float32); tot = 0.0
            for c, a in act: ph += photo_img(c, t) * a; tot += a
            ph = ph / max(tot, 1e-3) * BVIG
            al = min(1.0, sa_) * (1 - ma)
            base[by0:by1, bx0:bx1] = reg * (1 - BM * al) + ph * BM * al
            for c, a in act:
                a2 = a * (1 - ma) * seg(t, c["t0"] - 0.1, c["t0"] + 0.3)
                if a2 <= 0.001: continue
                reg = base[my0:my0 + MS, mx0:mx0 + MS]; base[my0:my0 + MS, mx0:mx0 + MS] = reg * (1 - MMK * a2) + c["mm"] * MMK * a2
        cv = Canvas(base)
        if ma > 0.01 and Mz is not None:
            for c, (px, py) in PINS:
                X_ = Mz[0, 0] * px + Mz[0, 1] * py + Mz[0, 2]; Y_ = Mz[1, 0] * px + Mz[1, 1] * py + Mz[1, 2]
                pa = ma * (1 - 0.5 * seg(t, GRID_AT, GRID_AT + 0.4))
                WG.pin(cv, X_, Y_, YEL, pa, t + 0.13 * CS.index(c))
                fdx, fdy = c.get("fdx", 0), c.get("fdy", -52)
                if fdx or fdy != -52: cv.line([(X_, Y_), (X_ + fdx * 0.8, Y_ + fdy * 0.8)], WHITE, 2.5, 0.7 * pa)
                WG.flag(cv, c["flag"], X_ + fdx, Y_ + fdy, 58, pa)
        if sa_ > 0.001 and ma < 0.999:
            cv.rrect(bx0, by0, bw, bh, 30, None, 1 - ma, stroke="#2A3B5C", lw=3, sa=1 - ma)
            for c, a in act:
                a2 = a * (1 - ma) * seg(t, c["t0"] - 0.1, c["t0"] + 0.3)
                if a2 > 0.01: cv.rrect(mx0, my0, MS, MS, 26, None, a2, stroke=WHITE, lw=3, sa=0.85 * a2)
                if a2 > 0.01 and c.get("photo", {}).get("credit"):
                    txt = "Photo: " + c["photo"]["credit"]
                    cv.text(txt, bx0 + 26, by0 + 40, 36, "#E6EEF8", a2 * 0.9, align="l", bold=False, maxw=bw - MS - 70, minsz=36)
            draw_header(cv, t, [(c, a * (1 - ma)) for c, a in act])
        # grid: empty grid flies in at GRID_AT over the world map, stays during the countries, gives way to the board
        gin = seg(t, GRID_AT, GRID_AT + 0.45)
        ga = gin * (1 - boa) * (1 - seg(t, BACK - 0.2, BACK + 0.2))
        if ga > 0.01:
            if t < CS[0]["t0"] + 0.3:     # fly-in: scale from 0.6 around the grid centre (drawn smaller via a temp canvas)
                gs = 0.6 + 0.4 * back(seg(t, GRID_AT, GRID_AT + 0.5, lambda u: u))
                sub = Canvas(np.zeros((H, W, 3), np.float32)); sub.surf.flush(); sub.buf[..., 3] = 0; sub.surf.mark_dirty()
                draw_quads(sub, t, 1.0, [])
                g = sub.buf.astype(np.float32)
                gcy = (gy0 + gy1) / 2
                Mg = cv2.getRotationMatrix2D((540, gcy), 0, gs); Mg[1, 2] += (1 - gin) * 120
                g = cv2.warpAffine(g, Mg, (W, H), flags=cv2.INTER_LINEAR)
                out = cv.out().astype(np.float32)
                al = g[..., 3:4] / 255 * ga
                # sub canvas is premultiplied BGRA over black
                out = out * (1 - al) + g[..., :3] * ga
                cv = Canvas(out)
            else:
                draw_quads(cv, t, ga, act)
        if boa > 0.01: draw_board(cv, t, boa)
        draw_screen(cv, cards, t, D.get("title_y", 300), 72)
        if t < BACK - 0.1: draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if t < BACK: draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0), HY, 1.0)
        if t >= BACK: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, BACK, BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(GRID_AT, "whoosh", -28), (BOARD_AT, "whoosh", -28), (BACK, "whoosh", -30), (RED_AT, "buzz", -30)]
    for c in CS:
        ev.append((c["t0"] - 0.1, "whoosh", -32))
        for k, it in c["it"].items(): ev.append((it["t"], "thud" if k in ("W", "T") else "pop", -26))
    for i in range(len(CS)): ev.append((BOARD_AT + 0.12 * i, "tick", -30))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=c["photo"]["file"], credit=c["photo"].get("credit")) for c in CS if c.get("photo")])
