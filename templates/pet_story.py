"""pet_story (story): "If Israel were a dog / a cat". One original cartoon pet (a dog or a cat, drawn here, not any known
character) with a blue collar and an Israeli flag tag sits centre stage for the whole reel. Each beat maps a pet trait to a
sourced fact: the pet's mood changes (ears, eyes, mouth, tail eased between moods), a drawn prop acts the trait out around it
(magnifying glass, light bulb, home, shield dome, barking bubbles, drip line, water bowl with a friend, spotlight, flag,
medals, devices, nine lives), and the info zone under the pet shows either a big yellow number card (counts up) or a real
photo in a rounded panel (slow Ken Burns, credit). Frame 0: the pet as a sky silhouette with a big "?" under the hook; the
loop line brings the silhouette and the hook back, and the last 0.5 s dissolves into frame 0.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  pet      "dog" (default) | "cat"
  colors   optional {body, dark, light, eye}  (dog default Malinois fawn with a black mask; cat default ginger tabby)
  quiz     {until, back_at}  silhouette + "?" from frame 0 until `until` (default end of line 0), back at `back_at`
  hook_back  anchor where the hook comes back (default quiz.back_at)
  beats    [{at, mood, prop, move, extra}]  a beat lasts until the next beat's `at` (the last until quiz.back_at)
           mood   neutral | happy | proud | alert | sad | sleep | wink   (eased change)
           prop   lens | bulb | home | dome | bark | drip | bowl | spot | flag | medals | devices | none
           move   shrink (pet shrinks to 62%, lens gag) | flip (jump, full turn, lands on its feet) | none
           extra  bark: {words: ["WOOF!", ...], blame: "HIM!"}; bowl: {friend_flag: "JO", friend_at}; medals: {n};
                  devices: {items: [[kind, anchor], ...]} kind phone | car | chip; spot: {text, at}; flip: {at}
  cards    [{at, until, kind: num | text | photo, ...}]  info zone (y 1110 to 1450)
           text:  text (*word* yellow, max 2 lines), size (default 76)
           num:   value, from, dec, prefix, suffix, sub (white, 64 px), log (bool, log count for big jumps)
           photo: file (repo path, e.g. photos/hic_decl.jpg), credit, kb [cx0, cy0, z0, cx1, cy1, z1]
  lives    {n: 9, at, until, used: [anchor, ...], y: 430}  row of red hearts under the title; each anchor greys one
  ground_y default 1090;  pet_scale default 1.05;  title_y default 300;  src_y default 1505;  subs_y default 1650
  tail     seconds after the last word (default 0.6)
Example: specs/israel_dog.json, specs/israel_cat.json.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

INK = "#0E2250"; SOFT = "#C9D6EA"; IL_BLUE = "#0038B8"
PAL = {
    "dog": dict(body="#D9A15A", dark="#3A2A20", light="#F2D3A0", eye="#4A2E1A", inner="#B07A3E"),
    "cat": dict(body="#EE9A4D", dark="#C2682A", light="#FFF1DE", eye="#7CCB4E", inner="#F4A7B0"),
}
MOODS = {   # ear (0 droop .. 1 up), eye (open size), lid (1 open .. 0 closed), mouth (0 smile, 1 open), tongue, brow, wag (tail speed), head (lift px)
    "neutral": dict(ear=0.8, eye=1.0, lid=1.0, mouth=0.0, tongue=0.0, brow=0.0, wag=2.2, head=0.0, happy=0.0),
    "happy":   dict(ear=0.85, eye=1.0, lid=1.0, mouth=0.75, tongue=1.0, brow=0.0, wag=9.0, head=4.0, happy=0.0),
    "proud":   dict(ear=1.0, eye=1.0, lid=0.0, mouth=0.25, tongue=0.0, brow=0.0, wag=4.5, head=14.0, happy=1.0),
    "alert":   dict(ear=1.0, eye=1.25, lid=1.0, mouth=0.55, tongue=0.0, brow=-1.0, wag=1.2, head=8.0, happy=0.0),
    "sad":     dict(ear=0.15, eye=1.0, lid=0.8, mouth=0.0, tongue=0.0, brow=1.0, wag=0.6, head=-10.0, happy=0.0),
    "sleep":   dict(ear=0.55, eye=1.0, lid=0.0, mouth=0.0, tongue=0.0, brow=0.0, wag=0.4, head=-14.0, happy=0.0),
    "wink":    dict(ear=0.9, eye=1.0, lid=1.0, mouth=0.35, tongue=0.0, brow=0.0, wag=5.0, head=6.0, happy=0.0, wink=1.0),
}
MOOD_T = 0.35


def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def ell(cv, x, y, rx, ry, col, a=1.0, ang=0.0, stroke=None, lw=3):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang); c.scale(max(rx, 0.01), max(ry, 0.01)); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def curve(cv, p0, p1, p2, col, lw, a=1.0):
    c = cv.c; c.new_path(); c.move_to(*p0); c.curve_to(p0[0] + (p1[0] - p0[0]) * 0.66, p0[1] + (p1[1] - p0[1]) * 0.66,
                                                       p2[0] + (p1[0] - p2[0]) * 0.66, p2[1] + (p1[1] - p2[1]) * 0.66, *p2)
    c.set_line_cap(1); c.set_line_width(lw); cv.col(col, a); c.stroke(); c.new_path()


def il_tag(cv, x, y, r, a=1.0):
    """round collar tag with the Israeli flag (white, two blue stripes, blue Star of David)"""
    cv.circ(x, y + 3, r + 3, "#000000", 0.25 * a)
    cv.circ(x, y, r, "#C9CED8", a); cv.circ(x, y, r * 0.9, "#FFFFFF", a)
    c = cv.c; c.save(); c.new_path(); c.arc(x, y, r * 0.9, 0, 2 * math.pi); c.clip()
    for sy in (-0.56, 0.44): cv.rrect(x - r, y + sy * r, 2 * r, 0.14 * r, 0, IL_BLUE, a)
    for ph in (-math.pi / 2, math.pi / 2):
        pts = [(x + 0.3 * r * math.cos(ph + i * 2 * math.pi / 3), y + 0.3 * r * math.sin(ph + i * 2 * math.pi / 3)) for i in range(3)]
        c.new_path(); c.move_to(*pts[0]); c.line_to(*pts[1]); c.line_to(*pts[2]); c.close_path()
        cv.col(IL_BLUE, a); c.set_line_width(max(1.5, r * 0.07)); c.stroke()
    c.restore()


def flag_tag(cv, x, y, r, code, a=1.0):
    cv.circ(x, y + 3, r + 3, "#000000", 0.25 * a); cv.circ(x, y, r, "#C9CED8", a); cv.circ(x, y, r * 0.9, "#FFFFFF", a)
    WG.flag(cv, code, x, y, r * 1.35, a)


def heart(cv, x, y, s, col, a=1.0):
    c = cv.c; c.save(); c.translate(x, y); c.scale(s, s); c.new_path()
    c.move_to(0, 0.35); c.curve_to(-1.1, -0.35, -0.55, -1.05, 0, -0.5); c.curve_to(0.55, -1.05, 1.1, -0.35, 0, 0.35); c.close_path()
    c.restore(); cv.col(col, a); c.fill(); c.new_path()


# ------------------------------------------------------------------ the pet
def draw_pet(cv, kind, x, gy, s, t, P, C, a=1.0, sil=None, tag="IL", rot=0.0, squash=1.0):
    """front-facing sitting pet. x, gy = centre and ground line; s scale; P mood params; C palette; sil = silhouette colour"""
    if a <= 0.01 or s <= 0.01: return
    c = cv.c; c.save(); c.translate(x, gy); c.scale(s, s * squash)
    if rot:
        c.translate(0, -300); c.rotate(rot); c.translate(0, 300)
    col = (lambda k: sil) if sil else (lambda k: C[k])
    breathe = 1 + 0.018 * math.sin(t * 2.6); hb = P["head"] - 6 * math.sin(t * 1.3)
    if not sil: ell(cv, 0, 0, 210, 26, "#000000", 0.28 * a)
    # tail (behind the body)
    sw = math.sin(t * P["wag"]) * (0.55 if kind == "dog" else 0.3)
    if kind == "dog":
        base = (120, -95); tip = (base[0] + 150 * math.cos(-1.0 + sw), base[1] - 150 * math.sin(1.0 + sw * 0.6) - 40 * P["ear"])
        curve(cv, base, ((base[0] + tip[0]) / 2 + 40, (base[1] + tip[1]) / 2 + 30), tip, col("body"), 44, a)
        curve(cv, ((base[0] + 2 * tip[0]) / 3, (base[1] + 2 * tip[1]) / 3), tip, tip, col("dark"), 34, a)
    else:
        base = (110, -60); mid = (250 + 30 * sw, -150); tip = (200 + 70 * sw, -330 + 20 * math.cos(t * 1.7))
        curve(cv, base, (240, -40), mid, col("body"), 40, a); curve(cv, mid, (290 + 40 * sw, -260), tip, col("body"), 38, a)
        if not sil:
            for k in range(3):
                u = 0.3 + 0.22 * k; px = mid[0] + (tip[0] - mid[0]) * u; py = mid[1] + (tip[1] - mid[1]) * u
                cv.line([(px - 20, py + 4), (px + 20, py - 4)], C["dark"], 9, a)
    # haunches + body
    for sx in (-1, 1): ell(cv, sx * 112, -72, 88, 72, col("body"), a)
    c.save(); c.translate(0, -60); c.scale(1, breathe); c.translate(0, 60)
    ell(cv, 0, -200, 132 if kind == "dog" else 118, 178, col("body"), a)
    if not sil:
        ell(cv, 0, -205, 78, 122, C["light"], a)
        if kind == "cat":
            for sx in (-1, 1):
                for k in range(3): curve(cv, (sx * 112, -250 + k * 52), (sx * 82, -238 + k * 52), (sx * 96, -216 + k * 52), C["dark"], 10, a)
    c.restore()
    # front legs + paws
    for sx in (-1, 1):
        cv.rrect(sx * 56 - 27, -250, 54, 236, 26, col("body"), a)
        ell(cv, sx * 60, -14, 46, 24, col("light"), a)
        if not sil:
            for k in (-1, 0, 1): cv.line([(sx * 60 + k * 13, -22), (sx * 60 + k * 13, -8)], C["inner"] if kind == "dog" else C["dark"], 3, 0.6 * a)
    # collar + tag (tag swings)
    cy_ = -338 + hb * 0.3
    cv.rrect(-102, cy_ - 16, 204, 32, 16, "#1E5BB0" if not sil else sil, a)
    if not sil:
        cv.circ(0, cy_ + 14, 7, "#C9CED8", a)
        c.save(); c.translate(0, cy_ + 14); c.rotate(math.sin(t * 3.1) * 0.12)
        if tag == "IL": il_tag(cv, 0, 40, 36, a)
        else: flag_tag(cv, 0, 40, 36, tag, a)
        c.restore()
    # head
    hy = -440 - hb
    c.save(); c.translate(0, hy); c.rotate(math.sin(t * 0.9) * 0.04)
    e = P["ear"]
    for sx in (-1, 1):
        if kind == "dog":
            droop = (1 - e) * 1.1
            c.save(); c.translate(sx * 78, -70); c.rotate(sx * (0.28 + droop))
            poly(cv, [(-44, 10), (0, -150), (44, 10)], col("body"), a)
            if not sil:
                poly(cv, [(-26, 0), (0, -118), (26, 0)], C["inner"], a)
                poly(cv, [(-19, -88), (0, -150), (19, -88)], C["dark"], a)
            c.restore()
        else:
            c.save(); c.translate(sx * 86, -60); c.rotate(sx * (0.32 + (1 - e) * 0.7))
            poly(cv, [(-50, 14), (6, -118), (46, 20)], col("body"), a)
            if not sil: poly(cv, [(-28, 6), (4, -80), (26, 10)], C["inner"], a)
            c.restore()
    if kind == "dog": ell(cv, 0, 0, 124, 112, col("body"), a)
    else: ell(cv, 0, 6, 138, 110, col("body"), a)
    if sil:
        c.restore(); c.restore(); return
    if kind == "dog":
        ell(cv, 0, 52, 74, 58, C["dark"], a)                       # black Malinois mask
        ell(cv, 0, 22, 28, 19, "#111111", a); ell(cv, -8, 16, 9, 5, "#FFFFFF", 0.5 * a)
    else:
        for k in (-1, 0, 1): cv.line([(k * 24, -96), (k * 20, -60)], C["dark"], 11, a)
        ell(cv, 0, 44, 70, 42, C["light"], a)
        poly(cv, [(-14, 22), (14, 22), (0, 38)], "#E57C8A", a)
        for sx in (-1, 1):
            for k in (-1, 0, 1): cv.line([(sx * 44, 46 + k * 12), (sx * 148, 34 + k * 26)], "#FFFFFF", 3.5, 0.85 * a)
    # mouth
    my = 58 if kind == "dog" else 48; m = P["mouth"]
    if m > 0.05:
        mh = 10 + 46 * m
        c.new_path(); c.move_to(-34, my); c.curve_to(-30, my + mh, 30, my + mh, 34, my); c.close_path(); cv.col("#5A1E22", a); c.fill()
        if P["tongue"] > 0.05:
            tl = 20 + 34 * P["tongue"] * (0.8 + 0.2 * math.sin(t * 9))
            ell(cv, 0, my + mh * 0.55 + tl * 0.45, 20, tl * 0.6, "#F07A8C", a)
    else:
        curve(cv, (-30, my - 4), (-15, my + 14), (0, my), "#111111" if kind == "dog" else C["dark"], 5, a)
        curve(cv, (0, my), (15, my + 14), (30, my - 4), "#111111" if kind == "dog" else C["dark"], 5, a)
    # eyes (blink about every 3.3 s)
    bl = 1.0 - max(0.0, 1 - abs(((t + 0.7) % 3.3) - 0.12) / 0.12)
    for sx in (-1, 1):
        ex, ey = sx * 48, -22
        lid = P["lid"] * bl
        if P.get("wink", 0) > 0.5 and sx == 1: lid = 0.0
        if P["happy"] > 0.5 and lid < 0.2:
            curve(cv, (ex - 22, ey + 6), (ex, ey - 18), (ex + 22, ey + 6), "#111111", 6, a)
        elif lid < 0.15:
            curve(cv, (ex - 22, ey), (ex, ey + 12), (ex + 22, ey), "#111111", 6, a)
        else:
            es = P["eye"]; rx = 23 * es; ry = 27 * es * lid
            ell(cv, ex, ey, rx, ry, "#FFFFFF", a)
            if kind == "dog":
                ell(cv, ex + 2, ey + 3 * lid, 15 * es, 16 * es * lid, C["eye"], a); ell(cv, ex + 2, ey + 3 * lid, 9 * es, 10 * es * lid, "#111111", a)
            else:
                ell(cv, ex, ey + 2, 19 * es, 22 * es * lid, C["eye"], a); ell(cv, ex, ey + 2, 5 * es, 18 * es * lid, "#111111", a)
            cv.circ(ex - 6, ey - 8 * lid, 5 * es, "#FFFFFF", 0.9 * a)
        if abs(P["brow"]) > 0.05:
            ang = sx * 0.35 * P["brow"]
            c.save(); c.translate(ex, ey - 42); c.rotate(ang); cv.rrect(-20, -5, 40, 10, 5, C["dark"], a); c.restore()
    c.restore(); c.restore()


# ------------------------------------------------------------------ props: (cv, u = seconds into the beat, t, B, G) G = geometry
def pr_lens(cv, u, t, B, G):
    p = seg(u, 0.4, 1.4); x = 980 - 380 * p + 30 * math.sin(u * 1.6) * p; y = G["head_y"] + 40 + 20 * math.cos(u * 1.3)
    if p <= 0: return
    a = min(1.0, p * 3)
    c = cv.c; c.save(); c.translate(x, y); c.rotate(0.5)
    cv.rrect(150, -22, 230, 44, 22, "#6B4A2E", a)
    c.restore()
    cv.circ(x, y, 150, "#BFE6FF", 0.16 * a); cv.circ(x, y, 150, None, a, stroke="#E9EEF5", lw=22); cv.circ(x, y, 150, None, a, stroke="#9AA6B8", lw=6)
    c.save(); c.new_path(); c.arc(x - 50, y - 60, 40, 3.6, 4.6); cv.col("#FFFFFF", 0.7 * a); c.set_line_width(10); c.stroke(); c.restore()


def pr_bulb(cv, u, t, B, G):
    q = WG.pop_s(u, 0.3)
    if q < 0.02: return
    x, y = 870, G["head_y"] - 60
    glow = 0.5 + 0.5 * math.sin(u * 5)
    for r, al in ((150, 0.08), (110, 0.13), (80, 0.2)): cv.circ(x, y, r * q * (0.9 + 0.1 * glow), YEL, al)
    for k in range(8):
        ang = k * math.pi / 4 + 0.2; r0, r1 = 105 * q, (140 + 14 * glow) * q
        cv.line([(x + r0 * math.cos(ang), y + r0 * math.sin(ang)), (x + r1 * math.cos(ang), y + r1 * math.sin(ang))], YEL, 9, 0.9)
    ell(cv, x, y, 62 * q, 70 * q, "#FFE58A"); ell(cv, x - 18 * q, y - 22 * q, 14 * q, 20 * q, "#FFFFFF", 0.7)
    cv.rrect(x - 32 * q, y + 56 * q, 64 * q, 46 * q, 10 * q, "#9AA6B8")
    for k in range(3): cv.line([(x - 32 * q, y + (64 + k * 13) * q), (x + 32 * q, y + (64 + k * 13) * q)], "#6E7A8C", 4)
    for k in range(4):                                                  # sparkles
        ph = (u * 0.8 + k / 4) % 1.0; ang = k * 1.7 + 0.5
        sx, sy = x + (170 + 60 * ph) * math.cos(ang), y + (170 + 60 * ph) * math.sin(ang)
        cv.star(sx, sy, 14 * (1 - ph) + 4, WHITE, 0.8 * (1 - ph))


def pr_home(cv, u, t, B, G):
    q = seg(u, 0.1, 0.7, eout)
    if q <= 0: return
    gy = G["gy"]; hx = 165 - 260 * (1 - q)
    cv.rrect(hx - 120, gy - 230, 240, 230, 10, "#F1E3C8"); poly(cv, [(hx - 150, gy - 220), (hx, gy - 350), (hx + 150, gy - 220)], "#B5523B")
    cv.rrect(hx - 44, gy - 140, 88, 140, 40, "#6B4A2E"); cv.rrect(hx + 50, gy - 200, 50, 50, 8, "#7FB6E0")
    il_tag(cv, hx - 70, gy - 175, 26)
    for k in range(5):                                                   # paw prints trail from the pet to home
        tk = 0.6 + k * 0.22
        if u < tk: continue
        px = 420 - k * 50; py = gy + 18 - (k % 2) * 16; al = min(1.0, (u - tk) * 4)
        ell(cv, px, py, 13, 10, "#F2D3A0", al)
        for d in (-12, 0, 12): cv.circ(px + d, py - 14, 5, "#F2D3A0", al)
    for k in range(4):                                                   # hearts rising
        ph = ((u - 0.8) * 0.45 + k / 4) % 1.0
        if u < 0.8: continue
        heart(cv, hx - 60 + k * 40 + 18 * math.sin(ph * 6 + k), gy - 360 - 220 * ph, 26 * (0.6 + 0.4 * (1 - ph)), RED, 0.85 * (1 - ph))


def pr_dome(cv, u, t, B, G):
    q = seg(u, 0.1, 0.8, eout)
    if q <= 0: return
    gy = G["gy"]; rx, ry = 440 * q, 640 * q
    c = cv.c; c.save(); c.translate(540, gy); c.scale(rx, ry); c.new_path(); c.arc(0, 0, 1, math.pi, 2 * math.pi); c.restore()
    cv.col(SKY, 0.13); c.fill_preserve(); cv.col(SKY, 0.9); c.set_line_width(7); c.stroke(); c.new_path()
    c.save(); c.translate(540, gy); c.scale(rx * 0.92, ry * 0.92); c.new_path(); c.arc(0, 0, 1, math.pi * 1.1, math.pi * 1.35); c.restore()
    cv.col("#FFFFFF", 0.45); c.set_line_width(8); c.stroke(); c.new_path()
    if q < 0.99: return
    for k in range(5):                                                   # rockets fall, stop on the dome in a yellow puff
        per = 1.6; ph = ((u - 0.9) / per + k * 0.37) % 1.0
        if u < 0.9 + k * 0.25: continue
        side = -1 if k % 2 else 1; ang0 = math.pi * (1.5 + side * (0.18 + 0.07 * k))
        hx, hy = 540 + 440 * math.cos(ang0), gy + 640 * math.sin(ang0)
        sx, sy = hx + side * 260, hy - 420
        if ph < 0.6:
            v = ph / 0.6; x = sx + (hx - sx) * v; y = sy + (hy - sy) * v
            cv.line([(x, y), (x + side * 60, y - 95)], "#FFFFFF", 6, 0.35)
            ell(cv, x, y, 12, 22, "#9AA6B8", 1.0, ang=-side * 0.55)
        else:
            v = (ph - 0.6) / 0.4
            cv.circ(hx, hy, 20 + 60 * v, YEL, 0.85 * (1 - v)); cv.circ(hx, hy, 10 + 30 * v, "#FFFFFF", 0.8 * (1 - v))


def pr_bark(cv, u, t, B, G):
    X = B.get("extra") or {}; words = X.get("words", ["WOOF!", "WOOF!", "WOOF!"])
    spots = [(170, 560, -1), (900, 660, 1), (150, 860, -1), (920, 940, 1)]
    for k, w in enumerate(words[:4]):
        q = WG.pop_s(u, 0.25 + k * 0.35)
        if q < 0.02: continue
        x, y, sd = spots[k]; wob = math.sin(u * 7 + k) * 0.05
        c = cv.c; c.save(); c.translate(x, y); c.rotate(wob); c.scale(q, q)
        poly(cv, [(sd * -20, 50), (sd * -80, 110), (sd * 30, 52)], "#FFFFFF")
        cv.rrect(-125, -62, 250, 124, 50, "#FFFFFF"); cv.text(w, 0, 0, 56, INK, 1, shadow=False, maxw=220, minsz=40)
        c.restore()
    bt = X.get("blame")
    if bt:
        q = WG.pop_s(u, 0.3 + len(words[:4]) * 0.35)
        if q > 0.02:
            for sd in (-1, 1):
                x0 = 540 + sd * 400; x1 = 540 + sd * 190; y = G["head_y"] + 230
                cv.line([(x0, y + 60), (x1 + sd * 40 * (1 - q), y)], RED, 14, q)
                poly(cv, [(x1, y), (x1 + sd * 46, y - 26), (x1 + sd * 40, y + 28)], RED, q)
            c = cv.c; c.save(); c.translate(540, G["head_y"] - 200); c.rotate(-0.06); c.scale(q, q)
            cv.rrect(-150, -56, 300, 112, 20, RED); cv.text(bt, 0, 0, 72, WHITE, 1, shadow=False); c.restore()


def pr_drip(cv, u, t, B, G):
    gy = G["gy"]; q = seg(u, 0.1, 0.7, eout)
    if q <= 0: return
    cv.rrect(0, gy - 4, W, 90, 0, "#C9A26A", 0.9 * q)
    cv.rrect(40, gy + 14, (W - 80) * q, 16, 8, "#2B2B2B", 1)
    for k in range(6):
        x = 100 + k * 176
        if x > 40 + (W - 80) * q or 380 < x < 700: continue
        g = seg(u, 0.9 + k * 0.15, 2.2 + k * 0.15, eout)
        ph = (u * 1.2 + k * 0.3) % 1.0
        ell(cv, x, gy + 34 + 30 * ph, 7, 10, SKY, 1 - ph)
        if g > 0:
            cv.line([(x, gy + 14), (x, gy + 14 - 120 * g)], "#2E9E5B", 9)
            for sd in (-1, 1): ell(cv, x + sd * 30 * g, gy - 60 * g - 20, 34 * g, 15 * g, "#3FBF7F", ang=sd * -0.5)


def pr_bowl(cv, u, t, B, G):
    X = B.get("extra") or {}; gy = G["gy"]; bx = 760
    q = seg(u, 0.1, 0.5, eout)
    if q <= 0: return
    ell(cv, bx, gy + 8, 120, 18, "#000000", 0.25)
    poly(cv, [(bx - 120, gy - 70), (bx + 120, gy - 70), (bx + 92, gy), (bx - 92, gy)], "#45A4DD")
    ell(cv, bx, gy - 70, 120, 22, "#2A86CC"); ell(cv, bx, gy - 72, 100, 15, "#BFE6FF", 0.9)
    for k in range(2):
        ph = (u * 0.7 + k * 0.5) % 1.0; ell(cv, bx, gy - 72, 30 + 70 * ph, 5 + 10 * ph, "#FFFFFF", 0.5 * (1 - ph), stroke=None)
    fa = G["A"](X.get("friend_at"), G["s0"] + 0.8) - G["s0"]
    f = seg(u, fa, fa + 1.0, eout)
    if f > 0:
        fx = 1180 - 260 * f
        draw_pet(cv, G["kind"], fx, gy, 0.42, t + 1.3, MOODS["happy"], dict(G["C"], body="#C7C1B6", light="#F4F1EA", dark="#5A5048"), tag=X.get("friend_flag", "JO"))
        if f > 0.95:
            for k in range(3):
                ph = ((u - fa - 1.0) * 0.5 + k / 3) % 1.0
                heart(cv, 650 + k * 40, gy - 330 - 200 * ph, 24, RED, 0.85 * (1 - ph))


def pr_spot(cv, u, t, B, G):
    q = seg(u, 0.0, 0.5)
    if q <= 0: return
    gy = G["gy"]
    poly(cv, [(470, 0), (610, 0), (900, gy + 10), (180, gy + 10)], "#FFF6D0", 0.13 * q)
    ell(cv, 540, gy + 4, 330, 40, "#FFF6D0", 0.22 * q)
    for k in range(18):                                                   # confetti
        ph = ((u - 0.3) * 0.35 + k / 18) % 1.0
        if u < 0.3: continue
        x = 60 + (k * 137) % 960 + 30 * math.sin(u * 2 + k); y = 380 + 760 * ph
        cv.rrect(x, y, 16, 26, 4, (YEL, SKY, RED, GREEN, WHITE)[k % 5], 0.9 * (1 - ph * 0.6))


def pr_flag(cv, u, t, B, G):
    q = seg(u, 0.1, 0.8, eout)
    if q <= 0: return
    gy = G["gy"]; px = 175; top = gy - 620 * q
    cv.rrect(px - 9, top, 18, gy - top, 9, "#C9CED8"); cv.circ(px, top, 16, YEL)
    fw, fh = 330, 230; x0 = px + 9; y0 = top + 20
    def pt(i, j):
        xx = x0 + fw * i; return (xx, y0 + fh * j + 16 * math.sin(u * 4 - i * 5) * i)
    n = 16; c = cv.c
    c.new_path(); c.move_to(*pt(0, 0))
    for k in range(1, n + 1): c.line_to(*pt(k / n, 0))
    for k in range(n, -1, -1): c.line_to(*pt(k / n, 1))
    c.close_path(); cv.col("#FFFFFF"); c.fill()
    for j0 in (0.12, 0.76):
        c.new_path(); c.move_to(*pt(0, j0))
        for k in range(1, n + 1): c.line_to(*pt(k / n, j0))
        for k in range(n, -1, -1): c.line_to(*pt(k / n, j0 + 0.12))
        c.close_path(); cv.col(IL_BLUE); c.fill()
    sx, sy = pt(0.5, 0.5)
    for ph in (-math.pi / 2, math.pi / 2):
        pts = [(sx + 44 * math.cos(ph + i * 2 * math.pi / 3), sy + 44 * math.sin(ph + i * 2 * math.pi / 3)) for i in range(3)]
        c.new_path(); c.move_to(*pts[0]); c.line_to(*pts[1]); c.line_to(*pts[2]); c.close_path(); cv.col(IL_BLUE); c.set_line_width(6); c.stroke()


def pr_medals(cv, u, t, B, G):
    n = (B.get("extra") or {}).get("n", 14); hy = G["head_y"]
    for k in range(n):
        q = WG.pop_s(u, 0.3 + k * 0.12)
        if q < 0.02: continue
        side = -1 if k % 2 == 0 else 1; i = k // 2; m = (n + 1) // 2
        ang = math.pi * (0.95 - 0.75 * i / max(1, m - 1))
        r = i / max(1, m - 1)                                             # two columns of medals either side of the pet
        x = 540 + side * (300 + 120 * math.sin(math.pi * r)); y = hy - 170 + r * 560
        bob = 6 * math.sin(u * 2 + k)
        cv.line([(x - 14 * q, y - 50 * q + bob), (x, y + bob)], RED, 10 * q); cv.line([(x + 14 * q, y - 50 * q + bob), (x, y + bob)], SKY, 10 * q)
        cv.circ(x, y + 22 * q + bob, 34 * q, "#E8B83A"); cv.circ(x, y + 22 * q + bob, 24 * q, YEL); cv.star(x, y + 22 * q + bob, 12 * q, "#E8B83A")


def pr_devices(cv, u, t, B, G):
    items = (B.get("extra") or {}).get("items", [])
    spots = [(190, 640), (890, 640), (190, 960), (890, 960)]
    for k, it in enumerate(items[:4]):
        kind, t0 = it[0], it[2]
        q = WG.pop_s(t, t0)
        if q < 0.02: continue
        x, y = spots[k]; bob = 8 * math.sin(t * 2 + k)
        c = cv.c; c.save(); c.translate(x, y + bob); c.scale(q, q)
        cv.circ(0, 0, 120, "#0B1730", 0.85, stroke=SKY, lw=4)
        if kind == "phone":
            cv.rrect(-46, -80, 92, 160, 18, "#2B3346"); cv.rrect(-38, -68, 76, 132, 10, "#BFE6FF")
            cv.line([(-30, 20), (0, -10), (30, 30)], "#7FB6E0", 8)
            c.new_path(); c.arc(0, -28, 20, math.pi, 2 * math.pi); c.line_to(0, 4); c.close_path(); cv.col(RED); c.fill(); cv.circ(0, -28, 8, "#FFFFFF")
        elif kind == "car":
            cv.rrect(-90, -10, 180, 50, 18, SKY); poly(cv, [(-56, -10), (-30, -50), (34, -50), (60, -10)], SKY)
            poly(cv, [(-44, -14), (-24, -42), (24, -42), (46, -14)], "#BFE6FF")
            for sx in (-1, 1): cv.circ(sx * 52, 42, 22, "#2B3346"); cv.circ(sx * 52, 42, 9, "#C9CED8")
            cv.circ(0, -34, 9, YEL)
            for k2 in range(3): cv.line([(10 + k2 * 4, -34), (90, -70 + k2 * 36)], YEL, 3, 0.6)
        else:
            cv.rrect(-60, -60, 120, 120, 12, "#2B3346"); cv.rrect(-36, -36, 72, 72, 8, "#8A96AA")
            for k2 in range(4):
                for sd in (-1, 1):
                    cv.line([(-45 + k2 * 30, sd * 60), (-45 + k2 * 30, sd * 80)], "#C9CED8", 6)
                    cv.line([(sd * 60, -45 + k2 * 30), (sd * 80, -45 + k2 * 30)], "#C9CED8", 6)
        c.restore()


def pr_zzz(cv, u, t, B, G):
    for k in range(3):
        ph = (u * 0.5 + k / 3) % 1.0
        cv.text("z", 640 + 60 * ph + k * 6, G["head_y"] - 120 - 160 * ph, 48 + 30 * ph, WHITE, 0.8 * (1 - ph), shadow=False)


PROPS = dict(lens=pr_lens, bulb=pr_bulb, home=pr_home, dome=pr_dome, bark=pr_bark, drip=pr_drip, bowl=pr_bowl, spot=pr_spot,
             flag=pr_flag, medals=pr_medals, devices=pr_devices, zzz=pr_zzz)


def _photo(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(p):                          # the CI checkout is sparse (templates/ only): pull the photo in
        import subprocess
        subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + path], check=False, capture_output=True)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"pet_story: cannot read photo {path}")
    return im


def _panel_img(im, w, h, cx, cy, z):
    ih, iw = im.shape[:2]; s = max(w / iw, h / ih) * max(1.0, z); cw, ch = w / s, h / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def run(S):
    D = S["data"]; work = workdir(S); kind = D.get("pet", "dog")
    C = dict(PAL[kind], **(D.get("colors") or {}))
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.6)); A = T.at
    GY = D.get("ground_y", 1090); PS = D.get("pet_scale", 1.05)
    Q = D.get("quiz") or {}; Q_UNTIL = A(Q.get("until"), T.E(0)); Q_BACK = A(Q.get("back_at"), T.END + 9)
    H_BACK = A(D.get("hook_back"), Q_BACK)
    BTS = []
    for b in D["beats"]:
        b = dict(b); b["t0"] = A(b["at"]); ex = dict(b.get("extra") or {})
        if ex.get("items"): ex["items"] = [[i[0], i[1], A(i[1])] for i in ex["items"]]
        if ex.get("flip_at") is not None: ex["_flip"] = A(ex["flip_at"])
        b["extra"] = ex; BTS.append(b)
    for k, b in enumerate(BTS): b["t1"] = BTS[k + 1]["t0"] if k + 1 < len(BTS) else Q_BACK
    CARDS = []
    for cd in D.get("cards", []):
        cd = dict(cd); cd["t0"] = A(cd["at"]); cd["t1"] = A(cd.get("until"), T.END - 0.6)
        if cd["kind"] == "photo": cd["im"] = _photo(cd["file"])
        CARDS.append(cd)
    LV = dict(D["lives"]) if D.get("lives") else None
    if LV:
        LV["t0"] = A(LV.get("at"), T.S(1)); LV["t1"] = A(LV.get("until"), T.END - 0.6); LV["used_t"] = [A(x) for x in LV.get("used", [])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    TY = D.get("title_y", 300)

    def mood_at(t):
        seq = [("neutral", -1e9)] + [(b.get("mood", "neutral"), b["t0"]) for b in BTS] + [("neutral", Q_BACK)]
        cur, prev, tc = MOODS["neutral"], MOODS["neutral"], -1e9
        for m, t0 in seq:
            if t >= t0: prev, cur, tc = cur, MOODS[m], t0
        k = seg(t, tc, tc + MOOD_T)
        keys = set(prev) | set(cur)
        return {kk: prev.get(kk, 0.0) + (cur.get(kk, 0.0) - prev.get(kk, 0.0)) * k for kk in keys}

    def pet_pose(t):
        """scale, y offset, rotation, squash"""
        sc, dy, rot, sq = PS, 0.0, 0.0, 1.0
        for b in BTS:
            mv = b.get("move")
            if not mv or t < b["t0"] - 0.05: continue
            if mv == "shrink":
                k = seg(t, b["t0"] + 0.1, b["t0"] + 0.6, lambda x: back(x, 1.4)) * (1 - seg(t, b["t1"] - 0.3, b["t1"] + 0.2))
                sc = PS * (1 - 0.38 * k)
            elif mv == "flip" and t < b["t1"]:
                f0 = b["extra"].get("_flip", b["t0"] + 0.25); d = 1.1; u = (t - f0) / d
                if 0 <= u <= 1:
                    dy = -380 * math.sin(math.pi * u); rot = 2 * math.pi * ease(u)
                    sq = 1 - 0.12 * math.sin(math.pi * min(1, u / 0.12)) if u < 0.12 else 1.0
                elif 1 < u < 1.35: sq = 1 - 0.16 * math.sin(math.pi * (u - 1) / 0.35)
        return sc, dy, rot, sq

    def info(cv, t):
        for cd in CARDS:
            a = win(t, cd["t0"], cd["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            rise = 40 * (1 - seg(t, cd["t0"], cd["t0"] + 0.4, eout))
            if cd["kind"] == "num":
                WG.panel(cv, 60, 1110 + rise, 960, 330, a)
                v = WG.roll(t, cd["t0"] + 0.15, cd.get("from", 0), cd["value"], cd.get("dur", 1.0))
                WG.counter(cv, v, 540, 1235 + rise, 150, a, cd.get("dec", 0), cd.get("prefix", ""), cd.get("suffix", ""))
                if cd.get("sub"): WG.title(cv, cd["sub"], 1365 + rise, a, 64, maxw=900)
            elif cd["kind"] == "text":
                WG.panel(cv, 60, 1110 + rise, 960, 330, a)
                WG.title(cv, cd["text"], 1275 + rise, a, cd.get("size", 76), maxw=900)
            else:
                kb = cd.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - cd["t0"]) / max(0.5, cd["t1"] - cd["t0"]))
                w, h = 960, 340
                img = _panel_img(cd["im"], w, h, kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
                g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]; img = (img.astype(np.float32) * (1 - 0.6 * np.clip((g - 0.6) / 0.4, 0, 1))).astype(np.uint8)
                rgba = np.dstack([img[..., ::-1], np.full((h, w), 255, np.uint8)])
                y0 = 1100 + rise; c = cv.c
                cv.rrect(60 + 6, y0 + 10, w, h, 36, "#000000", 0.3 * a)
                c.save(); cv.rrect(60, y0, w, h, 36, None); c.new_sub_path()
                r = 36; c.new_path(); c.arc(60 + w - r, y0 + r, r, -math.pi / 2, 0); c.arc(60 + w - r, y0 + h - r, r, 0, math.pi / 2)
                c.arc(60 + r, y0 + h - r, r, math.pi / 2, math.pi); c.arc(60 + r, y0 + r, r, math.pi, 1.5 * math.pi); c.close_path(); c.clip()
                cv.image(rgba, 540, y0 + h / 2, 1.0, a); c.restore()
                cv.rrect(60, y0, w, h, 36, None, a, stroke=SKY, lw=3, sa=0.6 * a)
                if cd.get("credit"): cv.text(cd["credit"], 540, y0 + h - 30, 36, "#E6ECF5", a, bold=False, maxw=900, minsz=36)

    def lives(cv, t):
        if not LV: return
        a = win(t, LV["t0"], LV["t1"], 0.3, 0.3)
        if a <= 0.01: return
        n = LV.get("n", 9); y = LV.get("y", 430); step = 92; x0 = 540 - step * (n - 1) / 2
        for k in range(n):
            ut = LV["used_t"][k] if k < len(LV["used_t"]) else None
            g = seg(t, ut, ut + 0.35) if ut is not None else 0.0
            pulse = 1 + 0.35 * math.sin(math.pi * g) if 0 < g < 1 else 1.0
            colr = RED if g < 0.5 else "#5C6B85"
            heart(cv, x0 + k * step, y + 8, 36 * pulse, colr, a)

    def frame(t):
        cv = Canvas(BG.copy())
        for k in range(5):                                               # soft drifting light circles (motion from frame 0)
            cx_ = (180 + k * 230 + 40 * math.sin(t * 0.3 + k)) % 1080; cy_ = 560 + 260 * math.sin(t * 0.21 + k * 1.9)
            cv.circ(cx_, cy_, 160 + 40 * k % 3, SKY, 0.045)
        quiz = t < Q_UNTIL + 0.4 or t >= Q_BACK
        qa = 1.0 if t < Q_UNTIL else (1 - seg(t, Q_UNTIL, Q_UNTIL + 0.4) if t < Q_BACK else seg(t, Q_BACK, Q_BACK + 0.4))
        sc, dy, rot, sq = pet_pose(t); P = mood_at(t)
        G = dict(gy=GY, head_y=GY - 440 * sc + dy, A=A, kind=kind, C=C, s0=0.0)
        cur = [b for b in BTS if b["t0"] - 0.05 <= t < b["t1"] + 0.35]
        for b in cur:                                                     # props behind the pet: dome, spot, flag, home, drip
            pa = 1 - seg(t, b["t1"], b["t1"] + 0.35)
            if b.get("prop") in ("spot", "flag", "home", "drip") and pa > 0.01:
                G["s0"] = b["t0"]; c = cv.c; c.push_group(); PROPS[b["prop"]](cv, t - b["t0"], t, b, G); c.pop_group_to_source(); c.paint_with_alpha(pa)
        if qa < 1.0: draw_pet(cv, kind, 540, GY + dy, sc, t, P, C, 1 - qa, rot=rot, squash=sq)
        if qa > 0.0:
            draw_pet(cv, kind, 540, GY, PS, t, MOODS["neutral"], C, qa, sil="#2B4F86")
            q = 1 + 0.04 * math.sin(t * 3)
            cv.text("?", 540, GY - 330 * PS, 300 * q, YEL, qa)
        for b in cur:
            pa = 1 - seg(t, b["t1"], b["t1"] + 0.35)
            if b.get("prop") and b["prop"] not in ("spot", "flag", "home", "drip", "none") and pa > 0.01:
                G["s0"] = b["t0"]; c = cv.c; c.push_group(); PROPS[b["prop"]](cv, t - b["t0"], t, b, G); c.pop_group_to_source(); c.paint_with_alpha(pa)
            if b.get("mood") == "sleep" and pa > 0.01: pr_zzz(cv, t - b["t0"], t, b, G)
        info(cv, t); lives(cv, t)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, TY, 84)
        draw_sources(cv, S, T, t, D.get("src_y", 1505)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, Q_UNTIL, D.get("hook_y", TY), 1.0)
        if t >= H_BACK: out = HK.paste(out, hook_img, 540, D.get("hook_y", TY), 1.0, seg(t, H_BACK, H_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1650)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(b["t0"] - 0.15, "whoosh", -30) for b in BTS] + [(Q_BACK - 0.15, "whoosh", -30)]
    ev += [(cd["t0"], "pop", -27) for cd in CARDS]
    ev += [(b["t0"] + 0.3, "ding", -28) for b in BTS if b.get("prop") == "spot"]
    ev += [(x, "thud", -26) for x in (LV["used_t"] if LV else [])]
    ev += [(it[2], "pop", -26) for b in BTS for it in b["extra"].get("items", [])]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=cd["file"], credit=cd.get("credit")) for cd in CARDS if cd["kind"] == "photo"])
