"""chem_lab (@noamroz__ funny science): an idea explained as chemistry, clean and animated (no real faces, no violence).
A cartoon FREE RADICAL (one red atom with a face, two orbits, paired electrons and ONE lone, unpaired electron that
bounces around) roams the screen and zaps small healthy molecules. Then it sits in a lab beaker with a label, bubbling
hot, a red thermometer, a speech bubble; a yellow "missing: 1 electron" tag rings the empty slot; a cartoon bucket of
cold water splashes into the beaker (steam, the thermometer drops: "cooling the reaction"); an ANTIOXIDANT molecule
floats in and hands over one electron (the electron flies along an arc, pairs up, the radical calms down and turns sky).
Then periodic-table tiles ("Rh", "Ex", ...), one per real tool, each over a real photo band with a credit line; earlier
tiles shrink into a row (a mini periodic table). Big stat card, screen titles, source chips, end card, subtitles, hook
from frame 0 over the moving radical; the hook comes back at the end and the last 0.5 s dissolves into frame 0 (loop).

Chemistry kept right: a free radical has an UNPAIRED electron; an antioxidant DONATES an electron to stabilise it.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  label_at, label_until     "Free radical" + "1 unpaired electron" labels on the roaming radical
  zap_until                 healthy molecules get zapped until here (they fade out by beaker_at)
  photos                    [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]  top band photos
  stat                      {at, until, value, prefix, suffix, top, sub}           big number card in the top band
  beaker_at                 the beaker rises, the radical drops into it; beaker_label (default "EXTREMISM")
  bubble                    {at, until, text}                                      speech bubble from the radical
  missing                   {at, until, text}                                      yellow tag on the empty slot
  splash_at                 bucket tips, water, steam, thermometer drops
  donor_at, give_at         antioxidant floats in; the electron flies over at give_at (radical calms ~0.8 s later)
  donor_label               default "Antioxidant"
  tiles                     [{at, num, sym, name, line, year, photo: {file, credit, kb, dim}}]
  tiles_until               tiles leave, the radical comes back for the ending
  hook_back                 hook returns (default END-1.6)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *
from photo_story import _load, _kb

INK = "#0E2250"; HOT = "#E5484D"; CALM = "#45A4DD"; GREEN_A = "#3FBF7F"; WATER = "#7CC8F5"; GLASS = "#C9D6EA"
TILE_COLS = ["#3FBF7F", "#45A4DD", "#F2A93B", "#B07CF5", "#FFD23F", "#F07FA8"]


def lerp(a, b, u): return a + (b - a) * u


def mixc(c0, c1, u):
    a, b = rgb(c0), rgb(c1); return "#%02X%02X%02X" % tuple(int(round((x + (y - x) * u) * 255)) for x, y in zip(a, b))


def ring(cv, x, y, r, col, lw, a):
    if a <= 0.01: return
    c = cv.c; c.new_path(); c.arc(x, y, r, 0, 2 * math.pi); cv.col(col, a); c.set_line_width(lw); c.stroke(); c.new_path()


def electron(cv, x, y, r, a=1.0, col=YEL, glow=0.35):
    if a <= 0.01: return
    cv.circ(x, y, r * 2.1, col, glow * a); cv.circ(x, y, r, col, a); cv.circ(x - r * 0.3, y - r * 0.3, r * 0.35, WHITE, 0.7 * a)


def ellipse_pt(cx, cy, A, B, rot, th):
    x, y = A * math.cos(th), B * math.sin(th); cr, sr = math.cos(rot), math.sin(rot)
    return cx + x * cr - y * sr, cy + x * sr + y * cr


def orbit(cv, cx, cy, A, B, rot, a, lw=3, col=WHITE):
    pts = [ellipse_pt(cx, cy, A, B, rot, i * 2 * math.pi / 72) for i in range(73)]
    cv.line(pts, col, lw, a)


def face(cv, x, y, r, mood, surprise, a):
    """mood 0 angry, 1 calm/smiling; surprise 0..1 (round mouth, raised brows)"""
    ex = r * 0.36; ey = y - r * 0.12; er = r * 0.2
    for s in (-1, 1):
        cv.circ(x + s * ex, ey, er, WHITE, a)
        cv.circ(x + s * ex + r * 0.04, ey + r * 0.03, er * 0.5, INK, a)
        tilt = (1 - mood) * 0.55 * (1 - surprise) - 0.25 * surprise
        bx0, bx1 = x + s * (ex + er * 1.15), x + s * (ex - er * 1.05)
        by = ey - er * 1.55 - surprise * r * 0.12
        cv.line([(bx0, by - tilt * r * 0.25 * -1), (bx1, by + tilt * r * 0.25)], INK, r * 0.09, a)
    my = y + r * 0.45
    if surprise > 0.5:
        cv.circ(x, my, r * 0.14, INK, a)
    else:
        c = cv.c; c.new_path(); curv = lerp(-0.22, 0.26, mood) * r       # frown -> smile
        c.move_to(x - r * 0.32, my); c.curve_to(x - r * 0.12, my + curv, x + r * 0.12, my + curv, x + r * 0.32, my)
        cv.col(INK, a); c.set_line_width(r * 0.09); c.set_line_cap(1); c.stroke()


def molecule(cv, x, y, sc, t, mood=0.0, lone="wild", slot=1.0, a=1.0, surprise=0.0, given=None, label=None, wob=0.0):
    """the radical. lone: 'wild' (bounces off the orbit), 'paired' (sits in its slot next to the donated one).
    slot: alpha of the dashed empty slot ring. given: (gx, gy) position of the incoming electron or None."""
    if a <= 0.01: return None
    R = 92 * sc; col = mixc(HOT, CALM, mood)
    x += math.sin(t * 23) * 5 * wob; y += math.cos(t * 19) * 4 * wob
    A1, B1 = 205 * sc, 78 * sc
    rots = (0.42, -0.42)
    for rr in rots: orbit(cv, x, y, A1, B1, rr, 0.38 * a, max(2, 3 * sc))
    # paired electrons (2 pairs per orbit)
    sp = 0.16
    for k, rr in enumerate(rots):
        for p in range(2 if k == 0 else 1):
            th = t * (1.6 + 0.3 * k) + p * math.pi + k * 1.3
            for d in (-sp / 2, sp / 2): electron(cv, *ellipse_pt(x, y, A1, B1, rr, th + d), 11 * sc, a, col=WHITE, glow=0.18)
    # the slot (second orbit, opposite side of its pair)
    th_s = t * 1.9 + 1.3 + math.pi
    sx, sy = ellipse_pt(x, y, A1, B1, rots[1], th_s)
    ex_, ey_ = ellipse_pt(x, y, A1, B1, rots[1], th_s - sp)
    if lone == "wild":
        jx = math.sin(t * 7.3) * 70 + math.sin(t * 12.1 + 1) * 30; jy = math.cos(t * 6.1) * 50 + math.sin(t * 15.7) * 25
        lx, ly = sx + jx * sc, sy + jy * sc
        cv.circ(lx, ly, 34 * sc, HOT, 0.25 * a)
        electron(cv, lx, ly, 15 * sc, a, glow=0.45)
        ring(cv, sx, sy, 22 * sc, YEL, 3, slot * a)
    else:
        electron(cv, ex_, ey_, 13 * sc, a); electron(cv, sx, sy, 13 * sc, a); lx, ly = ex_, ey_
    # core
    cv.circ(x, y, R * 1.35, col, 0.16 * a)
    cv.circ(x, y, R, col, a, stroke=mixc(col, WHITE, 0.35), lw=4 * sc)
    cv.circ(x - R * 0.35, y - R * 0.4, R * 0.22, WHITE, 0.18 * a)
    face(cv, x, y, R, mood, surprise, a)
    if label: cv.text(label, x, y + R + 52 * sc, 52, WHITE, a)
    return dict(x=x, y=y, slot=(sx, sy), lone=(lx, ly), R=R)


def healthy(cv, x, y, sc, t, a, zap=0.0):
    """a small stable molecule (two atoms, a bond, paired electrons). zap: red flash 0..1"""
    if a <= 0.01: return
    c1 = mixc(GREEN_A, HOT, zap); dx = 38 * sc
    cv.line([(x - dx, y), (x + dx, y)], GLASS, 8 * sc, 0.8 * a)
    for s in (-1, 1): cv.circ(x + s * dx, y, 30 * sc, c1, a, stroke=WHITE, lw=2)
    for s in (-1, 1):
        th = t * 2.2 + s
        for d in (-0.2, 0.2):
            electron(cv, x + s * dx + 52 * sc * math.cos(th + d), y + 52 * sc * math.sin(th + d), 6 * sc, a, col=WHITE, glow=0.1)
    if zap > 0.05:
        for k in range(6):
            ang = k * math.pi / 3 + t * 3
            cv.line([(x + 50 * math.cos(ang), y + 50 * math.sin(ang)), (x + 82 * math.cos(ang), y + 82 * math.sin(ang))], YEL, 5, zap * a)


def beaker(cv, cx, top, hot, a, label, t, bubbles=True, k=1.0):
    """glass beaker: x cx-230k..cx+230k, y top..top+600k. hot 1 = red liquid + bubbles, 0 = calm sky liquid"""
    if a <= 0.01: return
    c = cv.c; c.save(); c.translate(cx, top); c.scale(k, k); c.translate(-cx, -top)
    _beaker(cv, cx, top, hot, a, label, t, bubbles); c.restore(); c.new_path()


def _beaker(cv, cx, top, hot, a, label, t, bubbles):
    c = cv.c; x0, x1, y0, y1 = cx - 230, cx + 230, top, top + 600; lvl = top + 170
    liq = mixc("#2A6FA8", "#C4413F", hot)
    c.new_path(); c.move_to(x0 + 12, lvl); c.line_to(x1 - 12, lvl); c.line_to(x1 - 12, y1 - 40)
    c.arc(x1 - 52, y1 - 40, 40, 0, math.pi / 2); c.arc(x0 + 52, y1 - 40, 40, math.pi / 2, math.pi); c.close_path()
    cv.col(liq, 0.55 * a); c.fill(); c.new_path()
    # wave on top of the liquid
    pts = [(xx, lvl + math.sin(xx / 40 + t * (4 + 6 * hot)) * (4 + 8 * hot)) for xx in range(int(x0 + 12), int(x1 - 11), 8)]
    cv.line(pts, mixc(WATER, "#FF8A80", hot), 5, 0.8 * a)
    if bubbles and hot > 0.05:
        rng = np.random.default_rng(3)
        for k in range(18):
            bx = rng.uniform(x0 + 40, x1 - 40); sp = rng.uniform(90, 170); ph = rng.uniform(0, 1)
            yy = y1 - 60 - ((t * sp + ph * 400) % (y1 - 60 - lvl))
            cv.circ(bx + math.sin(t * 3 + k) * 6, yy, rng.uniform(6, 13), WHITE, 0.35 * a * hot, stroke=None)
    # glass
    c.new_path(); c.move_to(x0 - 26, y0); c.line_to(x0, y0 + 26); c.line_to(x0, y1 - 40); c.arc_negative(x0 + 40, y1 - 40, 40, math.pi, math.pi / 2)
    c.line_to(x1 - 40, y1); c.arc_negative(x1 - 40, y1 - 40, 40, math.pi / 2, 0); c.line_to(x1, y0)
    cv.col(GLASS, 0.9 * a); c.set_line_width(7); c.set_line_join(1); c.stroke()
    for k in range(4):
        yy = y0 + 150 + k * 95; cv.line([(x0 + 10, yy), (x0 + 60, yy)], GLASS, 4, 0.6 * a)
    cv.rrect(x0 + 10, y0 + 22, 18, y1 - y0 - 90, 9, WHITE, 0.10 * a)
    if label:
        tw_ = cv.tw(label, 58) + 60
        cv.rrect(cx - tw_ / 2, y1 - 132, tw_, 92, 20, "#F5F5F7", 0.95 * a)
        cv.text(label, cx, y1 - 86, 58, INK, a, shadow=False)


def thermometer(cv, x, top, level, a):
    if a <= 0.01: return
    h = 500; col = mixc(CALM, HOT, level)
    cv.rrect(x - 26, top, 52, h, 26, "#0B1A3C", 0.9 * a, stroke=GLASS, lw=4, sa=0.9 * a)
    cv.circ(x, top + h + 22, 46, "#0B1A3C", 0.9 * a, stroke=GLASS, lw=4)
    fh = lerp(80, h - 40, level)
    cv.rrect(x - 13, top + h - fh, 26, fh + 20, 13, col, a)
    cv.circ(x, top + h + 22, 34, col, a)
    for k in range(5): cv.line([(x + 30, top + 60 + k * 85), (x + 50, top + 60 + k * 85)], GLASS, 3, 0.7 * a)


def bucket(cv, x, y, ang, a):
    if a <= 0.01: return
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    c.new_path(); c.move_to(-95, -80); c.line_to(95, -80); c.line_to(70, 90); c.line_to(-70, 90); c.close_path()
    cv.col("#8FA3BF", a); c.fill_preserve(); cv.col(WHITE, a); c.set_line_width(5); c.stroke()
    c.new_path(); c.save(); c.scale(1, 0.28); c.arc(0, -80 / 0.28, 95, 0, 2 * math.pi); c.restore()
    cv.col(WATER, a); c.fill_preserve(); cv.col(WHITE, a); c.set_line_width(4); c.stroke()
    c.new_path(); c.arc(0, -80, 110, math.pi * 1.05, math.pi * 1.95); cv.col(GLASS, a); c.set_line_width(6); c.stroke()
    for k in range(3): cv.line([(-60 + k * 10, -40 + k * 40), (60 - k * 10, -40 + k * 40)], WHITE, 3, 0.25 * a)
    c.restore(); c.new_path()


def tile(cv, x, y, w, num, sym, name, year, col, a, big=True):
    """periodic-table tile, top-left corner (x, y), width w (height 1.12 w)"""
    if a <= 0.01: return
    h = w * 1.12
    cv.rrect(x, y, w, h, w * 0.08, "#0B1A3C", 0.95 * a, stroke=col, lw=max(4, w * 0.025), sa=a)
    cv.rrect(x, y, w, h * 0.06, 0, col, 0.0)
    if big:
        cv.text(str(num), x + w * 0.12, y + h * 0.12, max(44, w * 0.13), col, a, align="l", shadow=False)
        if year: cv.text(str(year), x + w * 0.88, y + h * 0.12, max(40, w * 0.11), GLASS, a, align="r", shadow=False, bold=False)
        cv.text(sym, x + w / 2, y + h * 0.47, w * 0.42, WHITE, a, shadow=False)
        cv.text(name, x + w / 2, y + h * 0.82, max(44, w * 0.14), col, a, maxw=w * 0.9, shadow=False)
    else:
        cv.text(sym, x + w / 2, y + h * 0.48, max(40, w * 0.42), WHITE, a, shadow=False)


def bubble(cv, txt, x, y, tx, ty, a, sz=66):
    """speech bubble centred at (x, y) with a tail to (tx, ty)"""
    if a <= 0.01: return
    w = min(980, cv.tw(txt, sz) + 90); h = sz * 1.9; c = cv.c
    c.new_path(); c.move_to(x - 40, y + h / 2 - 4); c.line_to(tx, ty); c.line_to(x + 30, y + h / 2 - 4); c.close_path()
    cv.col(WHITE, a); c.fill(); c.new_path()
    cv.rrect(x - w / 2, y - h / 2, w, h, h / 2.2, WHITE, a)
    cv.text(txt, x, y, sz, HOT, a, shadow=False, maxw=w - 60)


# ------------------------------------------------------------------ render
def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.4)); A = T.at; END = T.END
    tLab0, tLab1 = A(D.get("label_at"), 1e9), A(D.get("label_until"), 1e9)
    tBk = A(D["beaker_at"]); tZap = A(D.get("zap_until"), tBk)
    B = D.get("bubble") or {}; tB0, tB1 = A(B.get("at"), 1e9), A(B.get("until"), 1e9)
    MS = D.get("missing") or {}; tM0, tM1 = A(MS.get("at"), 1e9), A(MS.get("until"), 1e9)
    tSp = A(D["splash_at"]); tDo = A(D["donor_at"]); tGv = A(D["give_at"]); tCalm = tGv + 0.9
    ST = D.get("stat") or {}; tS0, tS1 = A(ST.get("at"), 1e9), A(ST.get("until"), 1e9)
    TL = [dict(tl, t0=A(tl["at"])) for tl in D.get("tiles", [])]
    tT1 = A(D.get("tiles_until"), END - 3)
    for k, tl in enumerate(TL): tl["t1"] = TL[k + 1]["t0"] if k + 1 < len(TL) else tT1
    tHB = A(D.get("hook_back"), END - 1.6)
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), 1e9), im=_load(p["file"])) for p in D.get("photos", [])]
    for tl in TL:
        if tl.get("photo"): PH.append(dict(tl["photo"], t0=tl["t0"], t1=tl["t1"], im=_load(tl["photo"]["file"])))
    E = S.get("end") or {}; tEnd = A(E.get("at"), tT1)
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 330); tHk = T.E(0)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    y_ = np.arange(H, dtype=np.float32)[:, None]
    BAND = np.clip((900 - y_) / 260, 0, 1)[..., None]           # photo band: full to y 640, gone at 900
    HEAL = [(170, 1290, 0.0), (900, 1250, 1.2), (160, 960, 2.4), (910, 930, 3.6), (560, 1390, 4.8)]
    BK = D.get("beaker_k", 1.2); BTOP = 740; LVL = BTOP + 170 * BK
    tBack = tT1 + 0.2                                             # radical returns for the ending

    def rad_pos(t):
        """where the radical is (x, y, scale)"""
        wx = 540 + 150 * math.sin(t * 0.9) + 50 * math.sin(t * 2.3); wy = 1110 + 110 * math.sin(t * 1.3 + 0.5)
        u = seg(t, tBk, tBk + 1.0)
        x, y, s = lerp(wx, 540, u), lerp(wy, 1170, u), lerp(1.4, 1.1, u)
        v = seg(t, tBack, tBack + 0.9)
        wx2 = 540 + 100 * math.sin(t * 0.9); wy2 = 1060 + 50 * math.sin(t * 1.3)
        return lerp(x, wx2, v), lerp(y, wy2, v), lerp(s, 1.15, v)

    def photos(t):
        out = BG.copy(); shown = 0.0
        for p in PH:
            a = win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)
            if a <= 0.01: continue
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            img = _kb(p["im"], lerp(kb[0], kb[3], u), lerp(kb[1], kb[4], u), lerp(kb[2], kb[5], u))
            dim = p.get("dim", 0.3); img = img * (1 - dim)
            m = BAND * a; out = out * (1 - m) + img * m; shown = max(shown, a)
        return out, shown

    def credits(cv, t):
        for p in PH:
            a = win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)
            if a > 0.01 and p.get("credit"): cv.text(p["credit"], 40, 812, 36, GLASS, 0.9 * a, bold=False, align="l", maxw=1000)

    def scene(cv, t):
        x, y, s = rad_pos(t)
        hot = 1 - seg(t, tSp + 0.5, tSp + 1.6)
        mood = seg(t, tCalm - 0.5, tCalm + 0.4)
        scene_a = 1 - seg(t, TL[0]["t0"] - 0.4, TL[0]["t0"]) if TL else 1.0
        scene_a = max(scene_a, seg(t, tBack, tBack + 0.6))
        # healthy molecules (intro)
        ha = 1 - seg(t, tZap, tZap + 0.6)
        for k, (hx, hy, ph) in enumerate(HEAL):
            hx2 = hx + 14 * math.sin(t * 1.1 + ph); hy2 = hy + 12 * math.cos(t * 0.9 + ph)
            d = math.hypot(hx2 - x, hy2 - y); zap = max(0.0, 1 - (d - 250) / 140) if t < tZap else 0.0
            healthy(cv, hx2, hy2, 1.35, t + ph, ha * scene_a, min(1, zap))
        # beaker + thermometer
        bu = seg(t, tBk - 0.2, tBk + 0.9, eout); ba = bu * (1 - seg(t, tBack - 0.6, tBack)) * scene_a
        top = lerp(1950, BTOP, bu)
        if ba > 0.01:
            thermometer(cv, 975, top + 60, lerp(0.25, 0.95, hot) if t < tCalm else 0.25, ba)
        # radical (inside the beaker: drawn between liquid and glass looks fine on top)
        if ba > 0.01: beaker(cv, 540, top, hot, ba, D.get("beaker_label", "EXTREMISM"), t, k=BK)
        sur = win(t, tSp + 0.35, tSp + 1.5, 0.1, 0.3)
        lone = "paired" if t >= tGv + 0.9 else "wild"
        info = molecule(cv, x, y, s, t, mood=mood, lone=lone, slot=0.0, a=scene_a, surprise=sur, wob=hot * seg(t, tBk, tBk + 0.5) * (1 - mood))
        # labels on the roaming radical
        la = win(t, tLab0, tLab1, 0.3, 0.3) * scene_a
        if la > 0.01 and info:
            lx, ly = info["lone"]
            cv.line([(lx, ly - 30), (lx + 40, ly - 120)], YEL, 4, la)
            WG.tag(cv, "1 unpaired electron", min(900, max(180, lx + 40)), max(620, ly - 150), 56, INK, la, bg=YEL)
        # missing tag
        ma = win(t, tM0, tM1, 0.25, 0.3)
        if ma > 0.01 and info:
            sx, sy = info["slot"]; rr = 34 + 6 * math.sin(t * 8)
            ring(cv, sx, sy, rr, YEL, 5, ma)
            cv.line([(sx, sy - rr), (540, 660)], YEL, 4, ma)
            WG.tag(cv, MS.get("text", "Missing: 1 electron"), 540, 620, 66, INK, ma, bg=YEL)
        # splash
        if tSp - 0.2 < t < tSp + 2.4:
            ang = -0.15 - 1.6 * seg(t, tSp, tSp + 0.45) + 1.6 * seg(t, tSp + 1.2, tSp + 1.8)
            bxy = (lerp(1250, 790, seg(t, tSp - 0.2, tSp + 0.2)), 560)
            bucket(cv, *bxy, ang, 1 - seg(t, tSp + 1.6, tSp + 2.2))
            if tSp + 0.3 < t < tSp + 1.4:                                       # water stream + drops
                rng = np.random.default_rng(int(t * 30))
                for k in range(26):
                    u = rng.uniform(0, 1); fx = lerp(bxy[0] - 100, 560 + rng.uniform(-220, 220), u); fy = lerp(520, LVL + 40, u * u)
                    cv.circ(fx, fy, rng.uniform(10, 24), WATER, 0.85)
            if tSp + 0.55 < t < tSp + 1.6:                                      # splash crown
                u = seg(t, tSp + 0.55, tSp + 1.6, lambda z: z)
                for k in range(14):
                    ang2 = math.pi * (1.05 + 0.9 * k / 13); r_ = 80 + 260 * u
                    cv.circ(540 + r_ * math.cos(ang2), LVL + r_ * math.sin(ang2) + 300 * u * u, 18 * (1 - u) + 5, WATER, 0.9 * (1 - u))
        if tSp + 0.8 < t < tSp + 3.2:                                           # steam
            u = seg(t, tSp + 0.8, tSp + 3.2, lambda z: z)
            for k in range(7):
                ph_ = (u * 1.6 + k * 0.37) % 1.0
                sx_ = 330 + k * 70 + math.sin(t * 2 + k) * 18; sy_ = LVL - 30 - ph_ * 360
                cv.circ(sx_, sy_, 36 + 40 * ph_, WHITE, 0.30 * (1 - ph_) * (1 - seg(u, 0.75, 1.0)))
        # antioxidant donor
        if tDo - 0.1 < t < tT1:
            u = seg(t, tDo, tDo + 1.0, eout); da = u * (1 - seg(t, TL[0]["t0"] - 0.5, TL[0]["t0"]) if TL else u)
            dx, dy = lerp(-250, 200, u), 560 + math.sin(t * 1.4) * 14
            cv.circ(dx, dy, 130, GREEN_A, 0.16 * da)
            cv.circ(dx, dy, 95, GREEN_A, da); ring(cv, dx, dy, 95, WHITE, 4, da)
            face(cv, dx, dy, 95, 1.0, 0.0, da)
            for k in range(3):
                th = t * 2 + k * 2.1
                if not (k == 0 and t >= tGv): electron(cv, dx + 135 * math.cos(th), dy + 135 * math.sin(th), 14, da)
            WG.tag(cv, D.get("donor_label", "Antioxidant"), dx + 40, dy - 170, 58, INK, da, bg=GREEN_A)
            if info and tGv <= t < tGv + 1.0:                                   # the electron flies over
                v = seg(t, tGv, tGv + 0.9, ease); sx, sy = info["slot"]
                px = lerp(dx + 135, sx, v); py = lerp(dy, sy, v) - math.sin(v * math.pi) * 220
                electron(cv, px, py, 16, 1.0, glow=0.6)
            if tGv + 0.9 <= t < tGv + 1.6 and info:
                w_ = seg(t, tGv + 0.9, tGv + 1.6, lambda z: z); sx, sy = info["slot"]
                ring(cv, sx, sy, 30 + 90 * w_, YEL, 6, 1 - w_)
        # speech bubble
        bba = win(t, tB0, tB1, 0.2, 0.25)
        if bba > 0.01 and info:
            bubble(cv, B.get("text", ""), 540, 600, info["x"] + 20, info["y"] - info["R"] - 10, bba * min(1.0, WG.pop_s(t, tB0)))
        return info

    def stat_card(cv, t):
        a = win(t, tS0, tS1, 0.3, 0.3)
        if a <= 0.01: return
        WG.panel(cv, 60, 400, 960, 420, a)
        if ST.get("top"): WG.title(cv, ST["top"], 480, a, 60)
        v = WG.roll(t, tS0 + 0.2, 0, ST["value"], 1.0)
        WG.counter(cv, v, 540, 630, 210, a, ST.get("dec", 0), ST.get("prefix", ""), ST.get("suffix", ""))
        if ST.get("sub"): WG.title(cv, ST["sub"], 760, a, 52, col=GLASS)

    def tiles(cv, t):
        n = len(TL); mw = 116; gap = 20; row_w = n * mw + (n - 1) * gap; rx = 540 - row_w / 2; ry = 1322
        for k, tl in enumerate(TL):
            if t < tl["t0"] - 0.05: continue
            col = TILE_COLS[k % len(TILE_COLS)]
            out = seg(t, tT1 - 0.3, tT1 + 0.3)
            a = 1 - out * 0.0
            u = seg(t, tl["t1"] - 0.25, tl["t1"] + 0.35) if k + 1 < n or t > tT1 - 0.3 else 0.0   # big -> mini
            pop = WG.pop_s(t, tl["t0"])
            bw = 410 * min(1.12, pop); bx, by = 540 - bw / 2, 1068 - bw * 1.12 / 2
            mx, my = rx + k * (mw + gap), ry
            w = lerp(bw, mw, u); x = lerp(bx, mx, u); y = lerp(by, my, u)
            ra = 1 - seg(t, tBack + 1.2, tBack + 1.8) * 0.0
            tile(cv, x, y, w, tl.get("num", k + 1), tl["sym"], tl["name"], tl.get("year"), col, ra * min(1, pop + u), big=u < 0.5)
            la = win(t, tl["t0"] + 0.15, tl["t1"] - 0.1, 0.25, 0.25) if k + 1 < n or t < tT1 else 0.0
            if la > 0.01 and tl.get("line"): WG.title(cv, tl["line"], 640, la, 70)
        return

    def end_card(cv, t):
        a = win(t, tEnd - 0.1, tHB + 0.1, 0.35, 0.35)
        if a <= 0.01: return
        y = 610; WG.panel(cv, 60, y - 210, 960, 420, a)
        if E.get("balance"): WG.title(cv, E["balance"], y - 80, a, 66)
        if E.get("cta"):
            ca = a * seg(t, tEnd + 0.5, tEnd + 0.9); tw_ = min(900, cv.tw(E["cta"].replace("*", ""), 56) + 90)
            cv.rrect(540 - tw_ / 2, y + 30, tw_, 112, 44, SKY, ca); WG.title(cv, E["cta"], y + 86, ca, 56, col=INK)

    def layer(t, hook=True, subs=True, overlays=True):
        base, pa = photos(t)
        cv = Canvas(base if pa > 0.01 else bars_moving_bg(t, 8))
        if pa > 0.01 and pa < 0.99:
            cv = Canvas(base * 1.0)
        scene(cv, t)
        if t >= (TL[0]["t0"] - 0.1 if TL else 1e9): tiles(cv, t)
        if overlays:
            credits(cv, t); stat_card(cv, t)
            draw_screen(cv, cards, t, y=262, sz=74)
            draw_sources(cv, S, T, t)
            end_card(cv, t)
        out = cv.out().astype(np.float32)
        if hook:
            out = paste_hook(out, hook_img, t, tHk, HY, 1.0)
            if t > tHB: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, tHB, tHB + 0.4))
        c3 = Canvas(out)
        if subs: draw_subs(c3, CH, t, D.get("subs_y", 1612))
        return np.clip(c3.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    if (S.get("cover") or {}).get("bg") == "scene":
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(D.get("cover_t", 0.8), hook=False, subs=False, overlays=False)); S["_cover_bg"] = p
    ev = [(0.05, "whoosh", -28), (tBk, "whoosh", -24), (tSp, "whoosh", -20), (tSp + 0.55, "thud", -18), (tSp + 0.6, "whoosh", -22),
          (tDo, "pop", -24), (tGv, "whoosh", -26), (tGv + 0.9, "ding", -18)]
    if tB0 < 1e8: ev.append((tB0, "buzz", -26))
    if tM0 < 1e8: ev.append((tM0, "tick", -18))
    if tS0 < 1e8: ev.append((tS0 + 0.2, "roll", tS0 + 1.2))
    if tLab0 < 1e8: ev.append((tLab0, "pop", -24))
    for tl in TL: ev.append((tl["t0"], "pop", -18))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, phases=dict(beaker=round(tBk, 2), splash=round(tSp, 2), give=round(tGv, 2),
                                                      tiles=[round(tl["t0"], 2) for tl in TL], end=round(END, 2)))
