"""pun_sequence (@noamroz__ humour): one stick figure ("little Roth") grows up through life, one pun per stage.
Each stage is a quick scene that slides in from the right; a big label pops when the pun is spoken, with the changed
part in yellow (*...*). Hook card from frame 0 over the first scene; the last 0.5 s dissolves into frame 0 (loop).

data keys:
  stages  [{scene, at, tag, label, label_at, note, note_at}]
          scene     baby | teen | army | train | fridge | standup | schnitzel | party | guitar | wrath | retired | grandson
          at        anchor where the stage slides in (the first stage is on screen from frame 0)
          tag       small sky chip at the top ("Age 13")
          label     the pun, *part* in yellow ("Roth-*teen*")
          label_at  anchor where the label pops (the pun word)
          note      optional small line under the label ("a US retirement account"), note_at (default label_at + 0.4)
          wrath / grandson use label_at to switch the scene's second beat (angry face, baby pops in)
  hook_y  default 420;  tail  seconds after the last word (default 1.2);  subs_y  default 1612
cover.bg "scene": the cover background is a clean frame of the first stage (no hook, no subtitles), see common.cover.
"""
import os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

ORANGE = "#F2A93B"; NEW = "#FF5A5F"; INK = "#0E2250"; LIGHT = "#E8EEF8"; SOFT = "#C9D6EA"
GROUND = 1330

# ------------------------------------------------------------------ drawing helpers
def ellipse(cv, x, y, rx, ry, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.save(); c.translate(x, y); c.scale(rx, ry); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


PROP = {  # leg, torso, head radius (in units of sc)
    "baby": (26, 34, 31), "teen": (56, 58, 24), "adult": (64, 66, 24), "old": (60, 62, 24)}


def roth(cv, x, y, sc, col=SKY, a=1.0, age="adult", hands=((-28, 44), (28, 44)), eyes="dot", mouth="smile",
         sit=False, walk=0.0, lw=7, hair=False):
    """y = feet line (or seat line when sit). hands: offsets from the shoulder in units. Returns key points."""
    leg, tor, hr = PROP[age]; u = sc
    hip = (x, y - (leg * 0.45 if sit else leg) * u)
    neck = (x, hip[1] - tor * u); head = (x, neck[1] - hr * u - 3 * u); sh = (x, neck[1] + 10 * u)
    W_ = lw * u
    cv.line([hip, neck], col, W_, a)
    if sit:
        if age == "baby": cv.line([hip, (x - 34 * u, y)], col, W_, a); cv.line([hip, (x + 34 * u, y)], col, W_, a)
        else: cv.line([hip, (x + 34 * u, hip[1]), (x + 34 * u, y)], col, W_, a); cv.line([hip, (x - 30 * u, hip[1] + 4 * u), (x - 32 * u, y)], col, W_, a)
    else:
        sw = math.sin(walk) * 16 * u
        cv.line([hip, (x - 20 * u + sw, y)], col, W_, a); cv.line([hip, (x + 20 * u - sw, y)], col, W_, a)
    hs = []
    for dx, dy in hands:
        e = (sh[0] + dx * u, sh[1] + dy * u); cv.line([sh, e], col, W_, a); hs.append(e)
    cv.circ(*head, hr * u, INK, a, stroke=col, lw=W_)
    if hair:   # baby curl
        cv.c.new_path(); cv.c.arc(head[0] + 4 * u, head[1] - hr * u - 6 * u, 7 * u, math.pi * 0.2, math.pi * 1.6); cv.col(col, a); cv.c.set_line_width(4 * u); cv.c.stroke()
    ex, ey = 10 * u, head[1] - 3 * u
    if eyes == "dot":
        for s in (-1, 1): cv.circ(head[0] + s * ex, ey, 3.4 * u, col, a)
    elif eyes == "roll":
        for s in (-1, 1): cv.circ(head[0] + s * ex, ey, 6.5 * u, LIGHT, a); cv.circ(head[0] + s * ex + 1 * u, ey - 3.8 * u, 2.8 * u, INK, a)
    elif eyes == "happy":
        for s in (-1, 1):
            cv.c.new_path(); cv.c.arc(head[0] + s * ex, ey + 2 * u, 4.5 * u, math.pi * 1.1, math.pi * 1.9); cv.col(col, a); cv.c.set_line_width(3 * u); cv.c.stroke()
    elif eyes == "shades":
        cv.rrect(head[0] - 19 * u, ey - 5 * u, 16 * u, 10 * u, 3 * u, "#05070D", a); cv.rrect(head[0] + 3 * u, ey - 5 * u, 16 * u, 10 * u, 3 * u, "#05070D", a)
        cv.line([(head[0] - 4 * u, ey - 3 * u), (head[0] + 4 * u, ey - 3 * u)], "#05070D", 2.5 * u, a)
    elif eyes == "angry":
        for s in (-1, 1):
            cv.circ(head[0] + s * ex, ey + 1 * u, 3.4 * u, col, a)
            cv.line([(head[0] + s * (ex + 6 * u), ey - 8 * u), (head[0] + s * 3 * u, ey - 3 * u)], col, 3.2 * u, a)
    mx, my = head[0], head[1] + 9 * u
    if mouth == "smile":
        cv.c.new_path(); cv.c.arc(mx, my - 4 * u, 7 * u, math.pi * 0.15, math.pi * 0.85); cv.col(col, a); cv.c.set_line_width(3 * u); cv.c.stroke()
    elif mouth == "grin":
        cv.c.new_path(); cv.c.arc(mx, my - 3 * u, 8 * u, 0, math.pi); cv.c.close_path(); cv.col(col, a); cv.c.fill()
    elif mouth == "flat": cv.line([(mx - 6 * u, my + 1 * u), (mx + 6 * u, my + 1 * u)], col, 3 * u, a)
    elif mouth == "frown":
        cv.c.new_path(); cv.c.arc(mx, my + 7 * u, 7 * u, math.pi * 1.2, math.pi * 1.8); cv.col(col, a); cv.c.set_line_width(3 * u); cv.c.stroke()
    elif mouth == "o": cv.circ(mx, my + 1 * u, 3.6 * u, col, a)
    return dict(head=head, hr=hr * u, neck=neck, hip=hip, sh=sh, hands=hs, u=u)


# ------------------------------------------------------------------ scenes: draw(cv, P) with P = dict(t, tl, lab, a)
def floor(cv, a, col="#2B4A80"): cv.line([(60, GROUND + 6), (1020, GROUND + 6)], col, 8, a)


def sc_baby(cv, P, x=540, sc=3.4):
    t, a = P["t"], P["a"]
    cv.rrect(250, GROUND - 10, 580, 40, 20, "#2B4A80", a)                  # rug
    for k, (mx, ch) in enumerate(((330, "⭐"), (540, "✡️"), (750, "⭐"))):   # mobile
        sw = math.sin(t * 2.2 + k) * 10
        cv.line([(mx, 640), (mx + sw, 720)], SOFT, 3, a * 0.8); WG.emo(cv, ch, mx + sw, 755, 64, a)
    cv.line([(300, 640), (780, 640)], SOFT, 4, a * 0.8)
    ang = math.sin(t * 11) * 0.5
    R = roth(cv, x, GROUND - 4, sc, SKY, a, "baby", hands=((-26, 20), (24 + 10 * math.cos(ang), -18 + 10 * math.sin(ang))), eyes="happy", mouth="grin", sit=True, hair=True)
    hx, hy = R["hands"][1]; u = R["u"]; tip = (hx + 28 * u * math.sin(ang + 0.3), hy - 28 * u * math.cos(ang + 0.3))
    cv.line([(hx, hy), tip], SOFT, 4 * u, a); cv.circ(*tip, 12 * u, YEL, a, stroke="#E0A33A", lw=3 * u)
    for k in range(3): cv.circ(tip[0] - 5 * u + k * 5 * u, tip[1] - 2 * u + (k % 2) * 4 * u, 2 * u, NEW, a)
    return R


def sc_teen(cv, P, x=540, sc=3.0):
    t, a = P["t"], P["a"]; floor(cv, a)
    R = roth(cv, x, GROUND, sc, SKY, a, "teen", hands=((-30, 40), (18, 8)), eyes="roll", mouth="flat")
    hx, hy = R["head"]; r = R["hr"]; u = R["u"]
    cv.c.new_path(); cv.c.arc(hx, hy, r + 7 * u, math.pi * 1.05, math.pi * 1.95); cv.col(LIGHT, a); cv.c.set_line_width(5 * u); cv.c.stroke()
    for s in (-1, 1): cv.rrect(hx + s * (r + 4 * u) - 7 * u, hy - 12 * u, 14 * u, 26 * u, 6 * u, NEW, a)
    px, py = R["hands"][1]; cv.rrect(px - 10 * u, py - 30 * u, 20 * u, 34 * u, 4 * u, "#05070D", a, stroke=SOFT, lw=2 * u)
    cv.rrect(px - 7 * u, py - 26 * u, 14 * u, 24 * u, 2 * u, SKY, a * 0.8)
    for k, ch in enumerate(("\U0001F3B5", "\U0001F3B6")):
        ph = (t * 0.7 + k * 0.5) % 1.0; WG.emo(cv, ch, x + (170 if k else -190) + math.sin(t * 3 + k) * 20, 900 - ph * 160, 80, a * math.sin(math.pi * ph))
    WG.emo(cv, "\U0001F644", x + 230, 760, 96, a * seg(P["tl"], 1.0, 1.3))
    return R


def star_david(cv, x, y, r, col, a, lw):
    for ph in (-math.pi / 2, math.pi / 2):
        pts = [(x + r * math.cos(ph + i * 2 * math.pi / 3), y + r * math.sin(ph + i * 2 * math.pi / 3)) for i in range(3)]
        poly(cv, pts, None, a, stroke=col, lw=lw)


def sc_army(cv, P, x=520, sc=3.0):
    t, a = P["t"], P["a"]; floor(cv, a)
    WG.emo(cv, "\U0001F1EE\U0001F1F1", 860, 760, 110, a); cv.line([(818, 730), (818, GROUND)], SOFT, 6, a)
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-34, 22), (40, 6)), eyes="dot", mouth="flat")
    hx, hy = R["head"]; r = R["hr"]; u = R["u"]
    cv.c.new_path(); cv.c.arc(hx, hy - 2 * u, r + 4 * u, math.pi, 2 * math.pi); cv.c.close_path(); cv.col("#6B7B3A", a); cv.c.fill()
    cv.line([(hx - r - 10 * u, hy - 2 * u), (hx + r + 10 * u, hy - 2 * u)], "#55632C", 4 * u, a)
    sx, sy = R["hands"][0]; sx -= 14 * u; bob = math.sin(t * 2) * 3
    pts = [(sx - 42 * u, sy - 46 * u + bob), (sx + 42 * u, sy - 46 * u + bob), (sx + 40 * u, sy + 10 * u + bob), (sx, sy + 52 * u + bob), (sx - 40 * u, sy + 10 * u + bob)]
    poly(cv, pts, "#2B4A80", a, stroke=LIGHT, lw=5 * u)
    star_david(cv, sx, sy + 2 * u + bob, 26 * u, LIGHT, a, 4.5 * u)
    return R


def sc_train(cv, P, x=520, sc=3.0):
    t, a = P["t"], P["a"]
    for k in range(2):
        wx = 120 + k * 450
        cv.rrect(wx, 680, 390, 300, 30, "#0B2A55", a, stroke=SOFT, lw=5)
        cv.c.save(); cv.c.rectangle(wx + 6, 686, 378, 288); cv.c.clip()
        for j in range(8):
            px = wx + 400 - ((t * 700 + j * 140 + k * 60) % 560)
            cv.line([(px, 900 + (j % 3) * 22), (px + 70, 900 + (j % 3) * 22)], SKY, 6, a * 0.6)
            cv.rrect(px, 820 - (j % 2) * 40, 40, 160, 4, "#163261", a)
        cv.c.restore()
    cv.line([(880, 640), (880, GROUND)], SOFT, 10, a); floor(cv, a)
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-26, 44), (34, 2)), eyes="dot", mouth="smile")
    u = R["u"]; bx, by = R["hands"][0]
    cv.rrect(bx - 26 * u, by, 52 * u, 38 * u, 5 * u, "#C9A15B", a); cv.line([(bx - 8 * u, by), (bx - 8 * u, by - 7 * u), (bx + 8 * u, by - 7 * u), (bx + 8 * u, by)], "#C9A15B", 3 * u, a)
    cx, cy = R["hands"][1]; cv.rrect(cx - 9 * u, cy - 26 * u, 20 * u, 28 * u, 4 * u, LIGHT, a); cv.rrect(cx - 9 * u, cy - 16 * u, 20 * u, 7 * u, 0, SKY, a)
    for k in range(2):
        ph = (t * 0.8 + k * 0.5) % 1.0
        cv.line([(cx - 3 * u + k * 8 * u + math.sin(ph * 6 + q) * 3 * u, cy - 30 * u - ph * 30 * u - q * 4 * u) for q in range(4)], LIGHT, 2 * u, a * math.sin(math.pi * ph))
    return R


def sc_fridge(cv, P, x=400, sc=3.0):
    t, a = P["t"], P["a"]; floor(cv, a)
    fx, fy, fw, fh = 650, 760, 300, GROUND - 760
    cv.rrect(fx, fy, fw, fh, 24, LIGHT, a, stroke=SOFT, lw=4)
    cv.rrect(fx + 18, fy + 18, fw - 36, fh - 36, 14, "#DDF1FF", a)
    glow = 0.25 + 0.1 * math.sin(t * 5); cv.rrect(fx + 18, fy + 18, fw - 36, fh - 36, 14, "#FFF6C8", a * glow)
    for k in range(3): cv.line([(fx + 24, fy + 150 + k * 140), (fx + fw - 24, fy + 150 + k * 140)], SOFT, 5, a)
    WG.emo(cv, "\U0001F9C0", fx + 80, fy + 112, 64, a); WG.emo(cv, "\U0001F955", fx + 210, fy + 252, 64, a)
    poly(cv, [(fx + fw, fy), (fx + fw + 90, fy + 40), (fx + fw + 90, fy + fh - 40), (fx + fw, fy + fh)], "#CFD9E8", a, stroke=SOFT, lw=4)
    reach = eout(P["tl"] / 1.2)
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-28, 44), (40 + 60 * reach, -10)), eyes="shades", mouth="smile")
    u = R["u"]; bx, by = R["hands"][1]
    cv.rrect(bx - 6 * u, by - 34 * u, 14 * u, 40 * u, 4 * u, GREEN, a); cv.rrect(bx - 3 * u, by - 46 * u, 8 * u, 14 * u, 2 * u, GREEN, a)
    for k in range(3):
        ph = (t * 0.5 + k / 3) % 1.0; WG.emo(cv, "❄️", fx + 40 + k * 110, fy - 30 - ph * 120, 60, a * math.sin(math.pi * ph))
    return R


def sc_standup(cv, P, x=520, sc=3.0):
    t, a = P["t"], P["a"]
    for r_ in range(9):
        for c_ in range(7):
            bx = 40 + c_ * 150 - (75 if r_ % 2 else 0); cv.rrect(bx, 600 + r_ * 70, 140, 60, 6, "#3A2440", a * 0.55)
    poly(cv, [(x - 60, 560), (x + 60, 560), (x + 300, GROUND + 6), (x - 300, GROUND + 6)], "#FFF2B0", a * 0.16)
    floor(cv, a, "#5A3A2A")
    mx = x + 120; cv.line([(mx, GROUND), (mx, 1040)], SOFT, 8, a); cv.line([(mx - 50, GROUND), (mx + 50, GROUND)], SOFT, 8, a)
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-40, -6), (22, -26)), eyes="dot", mouth="grin")
    u = R["u"]; hx, hy = R["hands"][1]
    cv.rrect(hx - 5 * u, hy - 6 * u, 10 * u, 26 * u, 3 * u, "#05070D", a); cv.circ(hx, hy - 10 * u, 8 * u, "#555B66", a)
    for k in range(4):
        cv.c.new_path(); cv.c.arc(150 + k * 260, 1415, 70, math.pi, 2 * math.pi); cv.col(ORANGE, a * 0.85); cv.c.fill()
        if P["lab"] > 0: WG.emo(cv, "\U0001F602", 150 + k * 260, 1300 - 30 * abs(math.sin(t * 6 + k)), 70, a * seg(P["lab"], 0.1 * k, 0.1 * k + 0.2))
    return R


def schnitzel(cv, x, y, s, a, bite=0.0):
    ellipse(cv, x, y + 10 * s, 150 * s, 42 * s, LIGHT, a, stroke=SOFT, lw=4)
    n = 14; pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n; rr = 1 + 0.12 * math.sin(i * 2.7)
        pts.append((x + 110 * s * rr * math.cos(ang), y + 26 * s * rr * math.sin(ang)))
    poly(cv, pts, "#E0A33A", a, stroke="#B5762A", lw=4)
    for i in range(10): cv.circ(x - 80 * s + i * 17 * s, y - 6 * s + (i % 3) * 8 * s, 3 * s, "#B5762A", a)
    cv.c.new_path(); cv.c.arc(x + 90 * s, y - 12 * s, 22 * s, math.pi * 0.5, math.pi * 1.5); cv.c.close_path(); cv.col(YEL, a); cv.c.fill()
    if bite > 0:   # bites out of the right end
        for k in range(int(1 + bite * 3)): cv.circ(x + 105 * s - k * 34 * s, y - 14 * s + (k % 2) * 10 * s, 24 * s, LIGHT, a)


def table(cv, x, y, w, a):
    cv.rrect(x - w / 2, y, w, 24, 10, "#2B4A80", a); cv.line([(x - w / 2 + 40, y + 24), (x - w / 2 + 40, GROUND)], "#2B4A80", 10, a)
    cv.line([(x + w / 2 - 40, y + 24), (x + w / 2 - 40, GROUND)], "#2B4A80", 10, a)


def sc_schnitzel(cv, P, x=380, sc=3.0):
    t, a = P["t"], P["a"]; floor(cv, a)
    WG.emo(cv, "\U0001F1E6\U0001F1F9", 900, 700, 96, a)
    cv.c.new_path(); cv.c.arc(900, 860, 70, 0, 2 * math.pi); cv.col(LIGHT, a); cv.c.set_line_width(6); cv.c.stroke()
    cv.line([(900, 860), (900, 820)], LIGHT, 6, a); cv.line([(900, 860), (930, 875)], LIGHT, 6, a)
    table(cv, 640, 1130, 520, a)
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-30, 40), (52, 4 + 6 * math.sin(t * 4))), eyes="happy", mouth="grin", sit=False)
    u = R["u"]; fx, fy = R["hands"][1]
    cv.line([(fx, fy), (fx + 40, fy - 70)], LIGHT, 6, a)
    for k in range(3): cv.line([(fx + 34 + k * 7, fy - 70), (fx + 40 + k * 7, fy - 100)], LIGHT, 4, a)
    schnitzel(cv, 700, 1100, 1.15, a)
    return R


def sc_party(cv, P, x=540, sc=2.6):
    t, a = P["t"], P["a"]; floor(cv, a)
    for k in range(9): cv.circ(100 + k * 110, 640 + 20 * math.sin(k * 1.7), 9, (YEL, NEW, SKY)[k % 3], a * 0.8)
    cv.line([(60, 640), (1020, 640)], SOFT, 2, a * 0.5)
    for ox, fl in ((180, 1), (900, -1)):
        roth(cv, ox, GROUND, 2.3, ORANGE, a, "adult", hands=((-26, 40), (fl * 30, 6)), eyes="dot", mouth="smile")
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-46, -8 + 8 * math.sin(t * 5)), (40, 4)), eyes="happy", mouth="o" if int(t * 4) % 2 else "grin")
    u = R["u"]; gx, gy = R["hands"][1]
    poly(cv, [(gx - 12 * u, gy - 30 * u), (gx + 12 * u, gy - 30 * u), (gx + 8 * u, gy + 4 * u), (gx - 8 * u, gy + 4 * u)], None, a, stroke=LIGHT, lw=3 * u)
    poly(cv, [(gx - 10 * u, gy - 14 * u), (gx + 10 * u, gy - 14 * u), (gx + 8 * u, gy + 3 * u), (gx - 8 * u, gy + 3 * u)], NEW, a * 0.85)
    for k, (bx, ch) in enumerate(((300, "\U0001F44B"), (540, "\U0001F604"), (780, "\U0001F91D"))):
        ba = a * WG.pop_s(P["tl"], 0.3 + 0.35 * k) if P["tl"] > 0.3 + 0.35 * k else 0
        if ba > 0.01:
            cv.rrect(bx - 75, 735, 150, 110, 40, LIGHT, min(1, ba) * 0.95); WG.emo(cv, ch, bx, 790, 70, min(1, ba))
    return R


def guitar(cv, x, y, ang, s, a):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.rrect(10 * s, -7 * s, 190 * s, 14 * s, 4 * s, "#7A4A22", a); cv.rrect(196 * s, -12 * s, 40 * s, 24 * s, 6 * s, "#05070D", a)
    cv.circ(-20 * s, 0, 48 * s, NEW, a); cv.circ(-82 * s, 0, 60 * s, NEW, a); cv.circ(-50 * s, 0, 14 * s, "#05070D", a)
    for k in (-3, 0, 3): cv.line([(-100 * s, k * s), (230 * s, k * s)], LIGHT, 1.2 * s, a)
    cv.rrect(-110 * s, -20 * s, 22 * s, 40 * s, 4 * s, LIGHT, a)
    c.restore()


def sc_guitar(cv, P, x=520, sc=3.0):
    t, a = P["t"], P["a"]; floor(cv, a)
    cv.rrect(780, 1030, 220, 300, 18, "#05070D", a, stroke=SOFT, lw=4)
    for k in range(2): cv.circ(890, 1120 + k * 120, 46 + 4 * math.sin(t * 16), "#2B2F38", a, stroke=SOFT, lw=3)
    strum = math.sin(t * 14) * 10
    R = roth(cv, x, GROUND, sc, SKY, a, "adult", hands=((-38, 52 + strum * 0.4), (50, 0)), eyes="shades", mouth="grin", walk=0.6)
    u = R["u"]; hip = R["hip"]
    guitar(cv, hip[0] - 20 * u, hip[1] - 6 * u, -0.45, 1.55, a)
    for k, ch in enumerate(("⚡", "\U0001F3B6", "⚡")):
        ph = (t * 0.9 + k / 3) % 1.0; WG.emo(cv, ch, 200 + k * 300, 860 - ph * 150, 84, a * math.sin(math.pi * ph))
    return R


def sc_wrath(cv, P, x=300, sc=3.0):
    t, a, lab = P["t"], P["a"], P["lab"]; floor(cv, a)
    table(cv, 700, 1130, 440, a)
    roth(cv, 860, GROUND, 2.6, ORANGE, a, "adult", hands=((-46, 4 + 8 * math.sin(t * 9)), (26, 40)), eyes="happy", mouth="o" if int(t * 5) % 2 else "grin")
    schnitzel(cv, 650, 1100, 0.95, a, bite=min(1.0, max(0.0, P["tl"] / 2.2)))
    if lab <= 0:
        walk = min(1.0, P["tl"] / 1.3)
        roth(cv, x - 180 * (1 - eout(walk)), GROUND, sc, SKY, a, "adult", hands=((-28, 44), (28, 44)), eyes="dot", mouth="smile", walk=t * 8 if walk < 1 else 0)
        return
    sh = math.sin(t * 60) * 6 * seg(lab, 0, 0.1)
    R = roth(cv, x + sh, GROUND, sc * (1 + 0.06 * eout(lab / 0.3)), NEW, a, "adult", hands=((-44, -10), (44, -10)), eyes="angry", mouth="frown")
    hx, hy = R["head"]; u = R["u"]
    for k in range(5):
        ph = (t * 1.4 + k / 5) % 1.0; s_ = -1 if k % 2 else 1
        cv.circ(hx + s_ * (40 + ph * 70), hy - 60 - ph * 160, 18 + 26 * ph, LIGHT, a * 0.7 * (1 - ph))
    WG.emo(cv, "\U0001F4A2", hx + 110, hy - 70, 100, a * seg(lab, 0, 0.2))
    return R


def rocker(cv, x, y, a, ang):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.line([(-150, 0), (-170, -260)], "#C9A15B", 12, a); cv.line([(-150, -90), (120, -90)], "#C9A15B", 12, a)
    cv.line([(-120, -90), (-110, 10)], "#C9A15B", 10, a); cv.line([(100, -90), (110, 10)], "#C9A15B", 10, a)
    c.new_path(); c.arc(-10, -330, 350, math.pi * 0.36, math.pi * 0.64); cv.col("#C9A15B", a); c.set_line_width(12); c.stroke()
    c.restore()


def sc_retired(cv, P, x=500, baby=False):
    t, a = P["t"], P["a"]; floor(cv, a)
    cv.circ(880, 720, 70, YEL, a); WG.emo(cv, "\U0001F334", 170, 1100, 330, a)
    ang = math.sin(t * 2.2) * 0.07; c = cv.c
    c.save(); c.translate(x, GROUND - 22); c.rotate(ang); c.translate(-x, -(GROUND - 22))
    rocker(cv, x, GROUND - 22, a, 0)
    hands = ((-30, 34), (36, 30)) if not baby else ((30, 34), (52, 20))
    R = roth(cv, x, GROUND - 22 - 90 + 4, 3.0, SKY, a, "old", hands=hands, eyes="happy", mouth="smile", sit=True)
    hx, hy = R["head"]; u = R["u"]
    cv.line([(hx - 12 * u, hy + 5 * u), (hx + 12 * u, hy + 5 * u)], LIGHT, 4 * u, a)            # moustache
    cv.c.new_path(); cv.c.arc(hx, hy, R["hr"] + 2 * u, math.pi * 1.15, math.pi * 1.85); cv.col(LIGHT, a); cv.c.set_line_width(4 * u); cv.c.stroke()   # white hair
    c.restore()
    if not baby:
        cv.line([(x + 230, GROUND), (x + 250, 1050)], "#C9A15B", 9, a); cv.c.new_path(); cv.c.arc(x + 225, 1050, 25, math.pi, 2 * math.pi); cv.col("#C9A15B", a); cv.c.set_line_width(9); cv.c.stroke()
    else:
        ba = a * WG.pop_s(P["lab"], 0) if P["lab"] > 0 else 0
        if ba > 0.01:
            roth(cv, x + 170, GROUND - 175, 1.9 * min(1, ba), SKY, min(1, ba), "baby", hands=((-24, 18), (24, -16)), eyes="happy", mouth="grin", sit=True, hair=True)
            WG.emo(cv, "❤️", x + 170, 820 - 20 * math.sin(t * 4), 90, min(1, ba))
    return R


SCENES = dict(baby=sc_baby, teen=sc_teen, army=sc_army, train=sc_train, fridge=sc_fridge, standup=sc_standup, schnitzel=sc_schnitzel,
              party=sc_party, guitar=sc_guitar, wrath=sc_wrath, retired=sc_retired, grandson=lambda cv, P: sc_retired(cv, P, baby=True))


def label_parts(txt):
    out = []
    for tok in re.split(r"(\*[^*]+\*)", txt):
        if tok: out.append((tok[1:-1], YEL) if tok.startswith("*") else (tok, WHITE))
    return out


def draw_label(cv, txt, y, a, pop, sz=150):
    c = cv.c; s = 0.6 + 0.4 * pop
    c.save(); c.translate(540, y); c.scale(s, s); c.translate(-540, -y)
    cv.rich(label_parts(txt), 540, y, sz, a, 1000)
    c.restore()


# ------------------------------------------------------------------ render
def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.2)); A = T.at
    ST = []
    for k, s in enumerate(D["stages"]):
        t0 = 0.0 if k == 0 else A(s["at"])
        tl = A(s["label_at"]) if s.get("label_at") is not None else None
        ST.append(dict(s, t0=t0, tlab=tl, tnote=A(s["note_at"]) if s.get("note_at") else (tl + 0.4 if tl is not None else None)))
    for k, s in enumerate(ST): s["t1"] = ST[k + 1]["t0"] if k + 1 < len(ST) else T.END + 1
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 420); tH = T.E(0)
    CH = sub_chunks(T, S.get("subs_replace")); SL = 0.35

    def scene(cv, s, t, dx, a=1.0):
        c = cv.c; c.save(); c.translate(dx, 0)
        c.translate(540, 1440); c.scale(1.1, 1.1); c.translate(-540, GROUND * -1)   # scene drawn on a 1330 ground, shown at 1440
        lab = t - s["tlab"] if s["tlab"] is not None and t >= s["tlab"] else -1
        SCENES[s["scene"]](cv, dict(t=t, tl=t - s["t0"], lab=lab, a=a))
        c.restore()

    def layer(t, hook=True, subs=True):
        cv = Canvas(bars_moving_bg(t, 8))
        for k, s in enumerate(ST):
            if t < s["t0"] - 0.01 or t > s["t1"] + SL: continue
            din = 0 if k == 0 else 1080 * (1 - eout((t - s["t0"]) / SL))
            dout = -1080 * ease((t - s["t1"]) / SL) if t > s["t1"] else 0
            scene(cv, s, t, din + dout)
            ta = win(t, s["t0"] + 0.1, s["t1"], 0.25, 0.2) * (seg(t, tH + 0.3, tH + 0.6) if (k == 0 and hook) else 1)
            if s.get("tag") and ta > 0.01:
                w = cv.tw(s["tag"], 52) + 70; cv.rrect(540 - w / 2 + din + dout, 222, w, 84, 42, SKY, ta * 0.95)
                cv.text(s["tag"], 540 + din + dout, 264, 52, WHITE, ta, shadow=False)
            if s["tlab"] is not None and t >= s["tlab"]:
                la = win(t, s["tlab"], s["t1"], 0.12, 0.2); pop = back(seg(t, s["tlab"], s["tlab"] + 0.35, lambda u: u))
                draw_label(cv, s["label"], 440, la, pop)
                if s.get("note") and t >= s["tnote"]:
                    na = win(t, s["tnote"], s["t1"], 0.25, 0.2)
                    cv.text(s["note"], 540, 560, 50, SOFT, na, bold=False, maxw=980, shadow=True)
        out = cv.out().astype(np.float32)
        if hook: out = paste_hook(out, hook_img, t, tH, HY, 1.0)
        c2 = Canvas(out)
        if subs: draw_subs(c2, CH, t, D.get("subs_y", 1612))
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    if (S.get("cover") or {}).get("bg") == "scene":   # clean first-stage frame for the cover (no hook, no subtitles)
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(0.0, hook=False, subs=False)); S["_cover_bg"] = p
    ev = []
    for k, s in enumerate(ST):
        if k: ev.append((s["t0"], "whoosh", -26))
        if s["tlab"] is not None: ev.append((s["tlab"], "thud" if s["scene"] == "wrath" else "pop", -18))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, stages=[dict(scene=s["scene"], label=s.get("label"), t0=round(s["t0"], 2)) for s in ST])
