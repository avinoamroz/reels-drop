"""templates/core/base.py: canvas, colours, easing, map renderer (whole land of Israel gold, West Bank and Gaza a darker gold,
Green Line green), stick figures. Copied from /home/claude/fix_1109/rmk/rlib.py (the 9 Nov remake engine), paths made repo relative."""
import sys, math, json, re
import numpy as np, cv2, cairo
from PIL import Image, ImageFont, ImageDraw
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opener"))
import map_tear as mt
W, H, FPS = 1080, 1920, 30
DASHES = tuple(chr(x) for x in (0x2013, 0x2014, 0x2012, 0x2015))
def rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
NAVY0, NAVY1, SKY, YEL, RED, GREEN, WHITE, GREY = "#0B1730", "#163261", "#45A4DD", "#FFD23F", "#E5484D", "#3FBF7F", "#F5F5F7", "#9FB0C8"
def ease(u): u = min(1.0, max(0.0, u)); return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2
def eout(u): u = min(1.0, max(0.0, u)); return 1 - (1 - u) ** 3
def back(u, c=1.7): u = min(1.0, max(0.0, u)); return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2
def seg(t, a, b, f=ease): return f((t - a) / (b - a)) if b > a else float(t >= a)
def win(t, a, b, fi=0.25, fo=0.25): return min(seg(t, a, a + fi), 1 - seg(t, b - fo, b))
MINSZ = [999]; USED = []

# ------------------------------------------------------------------ background
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32); _g = (_yy / H)[:, :, None]
_c0 = np.array(rgb(NAVY0)[::-1]) * 255; _c1 = np.array(rgb(NAVY1)[::-1]) * 255
BG = (_c0 * (1 - _g) + _c1 * _g).astype(np.float32); del _yy, _xx, _g

# ------------------------------------------------------------------ map
class MapR:
    LAND, LAND_B, OCEAN = mt.hexc("1D4579"), mt.hexc("3C70B0"), mt.hexc("0C1D40")
    def __init__(s):
        g = json.load(open(mt.GEO)); s.g = g
        def pack(gm):
            cs = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
            out = []
            for p in cs:
                r = np.asarray(p[0], np.float64)
                if len(r) >= 3: out.append((r, (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())))
            return out
        s.pack = pack
        s.world = {k: pack(v) for k, v in g["world"].items() if k != "ATA"}
        s.local = {k: pack(v) for k, v in g["local"].items()}
        s.golan = pack(g["golan"]); s.sinai = pack(g["sinai"]); s.isr = pack(g["isr49"]); s.terr = pack(g["wb"]) + pack(g["gaza"])
    def country(s, code): return s.local.get(code) or s.world[code]
    def view(s, lon, lat, k, cosf, sy): return mt.View((lon, lat), (540.0, sy), k, 0.0, cosf)
    def render(s, v, bh, hi=(), hi_a=None, israel=True, extra=None):
        """BGR uint8 (bh, W) of the map under View v. hi: country codes tinted red (alpha per code via hi_a dict)"""
        M = v.M(); inv = np.linalg.inv(M); c = mt.apply(inv, np.array([[-40, -40], [W + 40, -40], [-40, bh + 40], [W + 40, bh + 40]], float))
        vb = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
        cv = np.empty((bh, W, 3), np.uint8); cv[:] = s.OCEAN.astype(np.uint8)
        lay = cv.copy(); step = 10
        for lon in range(-180, 181, step):
            x = int(round(mt.apply(M, np.array([[lon, 0.0]]))[0, 0] * mt.SC)); cv2.line(lay, (x, -10), (x, (bh + 10) * mt.SC), mt.SKY.tolist(), 1, cv2.LINE_AA, mt.SS)
        for lat in range(-80, 81, step):
            y = int(round(mt.apply(M, np.array([[0.0, lat]]))[0, 1] * mt.SC)); cv2.line(lay, (-10, y), ((W + 10) * mt.SC, y), mt.SKY.tolist(), 1, cv2.LINE_AA, mt.SS)
        cv2.addWeighted(lay, 0.10, cv, 0.90, 0, cv)
        lw = max(1.0, min(v.k / 40, 2.4))
        for code, rings in s.world.items():
            if code in ("ISR", "PSE") and israel: continue
            r2 = s.country(code) if code in hi else rings
            mt.fill(cv, M, r2, s.LAND, 1.0, vb, outline=s.LAND_B, lw=lw, oalpha=0.55)
        for code in hi:
            a = (hi_a or {}).get(code, 0.88)
            mt.fill(cv, M, s.country(code), mt.RED, a, vb, outline=mt.hexc("FFB3B5"), lw=lw * 1.2, oalpha=0.9)
        if israel:
            mt.fill(cv, M, s.sinai, s.LAND, 1.0, vb, outline=s.LAND_B, lw=lw, oalpha=0.55)
            if "SYR" in hi: mt.fill(cv, M, s.golan, mt.RED, (hi_a or {}).get("SYR", 0.88), vb)
            else: mt.fill(cv, M, s.golan, s.LAND, 1.0, vb, outline=s.LAND_B, lw=lw, oalpha=0.55)
            mt.fill(cv, M, s.isr, mt.GOLD, 1.0, vb)
            mt.fill(cv, M, s.terr, mt.hexc("D9A93F") * 0.78 + mt.hexc("7A5A10") * 0.22, 1.0, vb, outline=mt.GREEN if v.k > 25 else None, lw=max(2, min(v.k / 45, 5)), oalpha=0.95)
        return cv, M

def geo2px(M, lon, lat, y0=0):
    p = mt.apply(M, np.array([[lon, lat]], float))[0]; return float(p[0]), float(p[1]) + y0

# ------------------------------------------------------------------ cairo helpers
class Canvas:
    def __init__(s, base):   # base float32 BGR (H, W, 3)
        s.buf = np.zeros((H, W, 4), np.uint8); s.buf[..., :3] = np.clip(base, 0, 255).astype(np.uint8); s.buf[..., 3] = 255
        s.surf = cairo.ImageSurface.create_for_data(s.buf, cairo.FORMAT_ARGB32, W, H, W * 4); s.c = cairo.Context(s.surf)
        s.c.set_antialias(cairo.ANTIALIAS_BEST)
    def out(s): s.surf.flush(); return s.buf[..., :3].copy()
    def col(s, h, a=1.0): r, g, b = rgb(h); s.c.set_source_rgba(r, g, b, a)
    def face(s, bold=True):
        s.c.select_font_face("Inter Display" if bold else "Inter", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    def tw(s, txt, sz, bold=True): s.face(bold); s.c.set_font_size(sz); return s.c.text_extents(txt).x_advance
    def text(s, txt, cx, cy, sz, col=WHITE, a=1.0, bold=True, maxw=1000, align="c", shadow=True, minsz=36):
        if a <= 0.01 or not txt: return
        assert not any(d in txt for d in DASHES), txt
        s.face(bold)
        while s.tw(txt, sz, bold) > maxw and sz > minsz: sz -= 2
        MINSZ[0] = min(MINSZ[0], sz); USED.append((txt, sz))
        s.c.set_font_size(sz); e = s.c.text_extents(txt); fe = s.c.font_extents()
        x = cx - e.x_advance / 2 if align == "c" else (cx if align == "l" else cx - e.x_advance)
        y = cy + (fe[0] - fe[1]) / 2 - 0.0   # approx vertical centre on cap/x height
        y = cy + sz * 0.35
        if shadow:
            s.col("#000000", 0.45 * a); s.c.move_to(x + 2, y + 3); s.c.show_text(txt)
        s.col(col, a); s.c.move_to(x, y); s.c.show_text(txt)
    def rich(s, parts, cx, cy, sz, a=1.0, maxw=1000):
        tot = lambda z: sum(s.tw(p, z) for p, _ in parts)
        while tot(sz) > maxw and sz > 36: sz -= 2
        x = cx - tot(sz) / 2
        for p, c in parts: s.text(p, x, cy, sz, c, a, align="l", maxw=2000); x += s.tw(p, sz)
    def rrect(s, x, y, w, h, r, col, a=1.0, stroke=None, lw=3, sa=1.0):
        c = s.c; c.new_sub_path(); c.arc(x + w - r, y + r, r, -math.pi / 2, 0); c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        c.arc(x + r, y + h - r, r, math.pi / 2, math.pi); c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi); c.close_path()
        if col: s.col(col, a); c.fill_preserve() if stroke else c.fill()
        if stroke: s.col(stroke, sa); c.set_line_width(lw); c.stroke()
        c.new_path()
    def circ(s, x, y, r, col, a=1.0, stroke=None, lw=3):
        s.c.new_path(); s.c.arc(x, y, r, 0, 2 * math.pi)
        if col: s.col(col, a); s.c.fill_preserve() if stroke else s.c.fill()
        if stroke: s.col(stroke); s.c.set_line_width(lw); s.c.stroke()
        s.c.new_path()
    def line(s, pts, col, lw, a=1.0, cap=cairo.LINE_CAP_ROUND, dash=None):
        c = s.c; c.set_line_cap(cap); c.set_line_join(cairo.LINE_JOIN_ROUND); c.set_line_width(lw); s.col(col, a)
        c.set_dash(dash or [])
        c.move_to(*pts[0])
        for p in pts[1:]: c.line_to(*p)
        c.stroke(); c.set_dash([])
    def star(s, x, y, r, col, a=1.0):   # six-pointed star (two triangles)
        for ph in (-math.pi / 2, math.pi / 2):
            s.c.new_path()
            for i in range(3): 
                ang = ph + i * 2 * math.pi / 3; (s.c.move_to if i == 0 else s.c.line_to)(x + r * math.cos(ang), y + r * math.sin(ang))
            s.c.close_path(); s.col(col, a); s.c.fill()
    def image(s, arr_rgba, cx, cy, scale, a=1.0):
        h, w = arr_rgba.shape[:2]; p = arr_rgba.astype(np.float32); al = p[..., 3:4] / 255
        bgra = np.concatenate([p[..., 2:3] * al, p[..., 1:2] * al, p[..., 0:1] * al, p[..., 3:4]], 2).astype(np.uint8)
        sf = cairo.ImageSurface.create_for_data(np.ascontiguousarray(bgra), cairo.FORMAT_ARGB32, w, h, w * 4)
        s.c.save(); s.c.translate(cx, cy); s.c.scale(scale, scale); s.c.set_source_surface(sf, -w / 2, -h / 2); s.c.paint_with_alpha(a); s.c.restore()

_emo = {}
def emoji(ch, sz=109):
    if ch not in _emo:
        f = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)
        im = Image.new("RGBA", (136, 128), (0, 0, 0, 0)); ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
        _emo[ch] = np.asarray(im)
    return _emo[ch]

# ------------------------------------------------------------------ stick figures
def figure(cv, x, y, sc=1.0, col=WHITE, badge=True, arms="down", sit=False, spiral=False, brief=False, a=1.0, lw=7, flip=1, walk=0.0):
    """y = feet (or seat) line. plain stick figure with an optional small sky-blue star badge on the chest"""
    c = cv.c; hs = 22 * sc; L = 60 * sc; T = 62 * sc
    hip = (x, y - (L * (0.55 if sit else 1.0)))
    neck = (x, hip[1] - T); head = (x, neck[1] - hs - 4 * sc)
    cv.line([hip, neck], col, lw * sc, a)
    if sit:
        cv.line([hip, (x + flip * 38 * sc, hip[1]), (x + flip * 38 * sc, y)], col, lw * sc, a)
    else:
        sw = math.sin(walk) * 18 * sc
        cv.line([hip, (x - 20 * sc + sw, y)], col, lw * sc, a); cv.line([hip, (x + 20 * sc - sw, y)], col, lw * sc, a)
    sh = (x, neck[1] + 12 * sc)
    if arms == "down": ends = [(x - 28 * sc, sh[1] + 44 * sc), (x + 28 * sc, sh[1] + 44 * sc)]
    elif arms == "temple": ends = [(x - 22 * sc, head[1] + 8 * sc), (x + 22 * sc, head[1] + 8 * sc)]
    elif arms == "up": ends = [(x - 36 * sc, sh[1] - 40 * sc), (x + 36 * sc, sh[1] - 40 * sc)]
    elif arms == "shrug": ends = [(x - 44 * sc, sh[1] - 10 * sc), (x + 44 * sc, sh[1] - 10 * sc)]
    elif arms == "point": ends = [(x - 24 * sc, sh[1] + 44 * sc), (x + flip * 60 * sc, sh[1] + 6 * sc)]
    else: ends = [(x - 28 * sc, sh[1] + 44 * sc), (x + 28 * sc, sh[1] + 44 * sc)]
    for e in ends: cv.line([sh, (e[0], e[1] + (10 * sc if arms == "temple" else 0)) if arms == "temple" else e], col, lw * sc, a)
    cv.circ(*head, hs, "#0E2250", a, stroke=col, lw=lw * sc)
    if spiral:
        for sgn in (-1, 1):
            cx, cy = head[0] + sgn * 8 * sc, head[1] - 2 * sc
            pts = [(cx + (0.4 + 0.5 * i / 14) * 7 * sc * math.cos(i * 0.9) * 1.0, cy + (0.4 + 0.5 * i / 14) * 7 * sc * math.sin(i * 0.9)) for i in range(15)]
            cv.line(pts, YEL, 2.2 * sc, a)
    else:
        for sgn in (-1, 1): cv.circ(head[0] + sgn * 8 * sc, head[1] - 2 * sc, 2.6 * sc, col, a)
    if badge: cv.star(x, neck[1] + 34 * sc, 8.5 * sc, SKY, a)
    if brief:
        bx, by = ends[1][0], ends[1][1]; cv.rrect(bx - 14 * sc, by, 34 * sc, 26 * sc, 4 * sc, "#C9A15B", a); cv.line([(bx - 4 * sc, by), (bx - 4 * sc, by - 6 * sc), (bx + 10 * sc, by - 6 * sc), (bx + 10 * sc, by)], "#C9A15B", 3 * sc, a)
    return head

def brain(cv, x, y, sc, glow, a=1.0):
    r = 20 * sc
    for rr, aa in ((46, 0.10), (34, 0.16), (24, 0.26)): cv.circ(x, y, rr * sc * (0.9 + 0.12 * glow), "#E070FF", aa * a)
    for dx, dy in ((-9, 2), (9, 2), (-4, -7), (6, -8)): cv.circ(x + dx * sc, y + dy * sc, 11 * sc, "#F29BFF", 0.95 * a)
    cv.line([(x - 12 * sc, y), (x - 3 * sc, y - 4 * sc), (x + 3 * sc, y + 3 * sc), (x + 12 * sc, y - 2 * sc)], "#9A2BCC", 2.4 * sc, a)

def beam_path(p0, p1, bend=0.0, n=40):
    (x0, y0), (x1, y1) = p0, p1; mx, my = (x0 + x1) / 2 + bend, (y0 + y1) / 2
    return [((1 - u) ** 2 * x0 + 2 * u * (1 - u) * mx + u * u * x1, (1 - u) ** 2 * y0 + 2 * u * (1 - u) * my + u * u * y1) for u in [i / n for i in range(n + 1)]]

def pulse_beam(cv, p0, p1, t, bend=0.0, col="#E070FF", prog=1.0, a=1.0, waves=True):
    pts = beam_path(p0, p1, bend); m = int(len(pts) * prog)
    if m < 2: return
    cv.line(pts[:m], col, 14, 0.16 * a); cv.line(pts[:m], col, 6, 0.55 * a); cv.line(pts[:m], "#FFFFFF", 2, 0.8 * a)
    if waves:
        for k in range(4):
            u = ((t * 0.9 + k / 4) % 1.0); i = int(u * (m - 1)); x, y = pts[i]
            cv.circ(x, y, 7 + 10 * u, None, 1, stroke=col, lw=2.5)
            cv.c.new_path()
