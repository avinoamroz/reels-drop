"""story_climb_full: the long-form version of story_climb (same figure, palette and primitives), one distinct scene per paragraph or idea.
Built for "The Warrior Within" full length (Anna Os's own words, read by a stock female voice, about 8 to 10 minutes).

Spec: like story_climb, plus
  data.scenes   [{scene, at, tag, k: {name: anchor}, d: {data overrides}}]; each scene runs from its anchor until the next (0.5 s
                cross dissolve) with a slow camera drift; k gives the scene its own beats (e.g. {"piano": "L6:piano"}).
  data.music    [[anchor, mood], ...] chapter moods for the procedural bed: dark, tense, child, quiet, warm, hope, rise
  data.start_credit {text, until, y}, data.titles, data.notes, data.credit, data.end_notes (like story_climb), data.crf (default 22)
Scenes: every scene of story_climb (tornado, wall, fall, child, pharmacy, peaks, five, voice, view, together) plus the ones in SCENES
below. Nothing graphic: the age 5 scene shows only a shadow and a closing door; the break is one quiet, dimmed beat.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np, cv2
from common import *
import story_climb as SB
from story_climb import person, poly, arc, mountains, rock, film_strip, bubble, pill_box, backdrop, climber_on_wall, wall_set
from story_climb import INK, LIGHT, SOFT, ROCK, ROCK2, GROUND, GIRL, DRESS

GREY_F = "#7F8FA9"; WARM = "#F2A93B"; NIGHT = "#070D1C"


def hexmix(a, b, u):
    ra, rb = rgb(a), rgb(b); u = max(0.0, min(1.0, u))
    return "#" + "".join(f"{int(round((x * (1 - u) + y * u) * 255)):02X}" for x, y in zip(ra, rb))


def K(s, n, d=1e9): return s["K"].get(n, d)


def floor(cv, y=GROUND, col="#2B4A80", a=1.0): cv.line([(0, y + 4), (W, y + 4)], col, 8, a)


def walk(cv, x, yg, sc, ph, a=1.0, col=LIGHT, dress=DRESS, hands=None, eyes="dot", mouth="soft", look=0.6, lean=0.04):
    """walking pose on ground yg: phase ph (rad)"""
    s_ = math.sin(ph); c_ = math.cos(ph); hip = (x, yg - 64 * sc - 2 * sc * abs(c_))
    feet = [(x + 22 * sc * s_, yg - 7 * sc * max(0, c_)), (x - 22 * sc * s_, yg - 7 * sc * max(0, -c_))]
    sh = (hip[0] + math.sin(lean) * 54 * sc, hip[1] - 54 * sc)
    hands = hands or [(sh[0] - 20 * sc * s_ - 6 * sc, sh[1] + 52 * sc), (sh[0] + 20 * sc * s_ + 6 * sc, sh[1] + 52 * sc)]
    return person(cv, hip, sc, lean=lean, hands=hands, feet=feet, a=a, col=col, dress=dress, eyes=eyes, mouth=mouth, look=look, knee=(1, 1))


def sitting(cv, x, yg, sc, a=1.0, eyes="dot", mouth="soft", hands=None, col=LIGHT, dress=DRESS, look=0.0):
    hip = (x, yg - 10 * sc)
    return person(cv, hip, sc, hands=hands or [(x - 14 * sc, yg - 34 * sc), (x + 30 * sc, yg - 34 * sc)],
                  feet=[(x + 40 * sc, yg), (x + 54 * sc, yg)], eyes=eyes, mouth=mouth, a=a, col=col, dress=dress, knee=(-1, -1), look=look)


def cloud(cv, x, y, s, col, a):
    for dx, dy, r in ((-60, 10, 52), (-10, -20, 64), (50, 4, 54), (0, 24, 56), (-90, 30, 36), (90, 30, 38)):
        cv.circ(x + dx * s, y + dy * s, r * s, col, a)


def eyes_pair(cv, x, y, s, a, look=(0, 0), blink=1.0):
    for k in (-1, 1):
        c = cv.c; c.save(); c.translate(x + k * 34 * s, y); c.scale(1, max(0.08, blink)); cv.circ(0, 0, 24 * s, "#E8EEF8", a); c.restore()
        if blink > 0.3: cv.circ(x + k * 34 * s + look[0] * 10 * s, y + look[1] * 8 * s, 9 * s, NIGHT, a)


def stamp_text(cv, txt, x, y, sz, col, a, ang=0.0):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang); w = cv.tw(txt, sz) + 60
    cv.rrect(-w / 2, -sz * 0.8, w, sz * 1.6, 14, None, a, stroke=col, lw=8, sa=a); cv.text(txt, 0, 0, sz, col, a, shadow=False); c.restore()


def heart_line(cv, y, calm, t, a=1.0, col=SKY):
    pts = []
    for i in range(0, 1081, 8):
        ph = (i + t * 220) % 270
        spike = (1 - calm) * (300 if 120 < ph < 130 else (-200 if 130 <= ph < 140 else 0)) * (0.5 + 0.5 * math.sin(i * 0.07))
        pts.append((i, y - spike * 0.35 - 30 * calm * math.sin((i + t * 120) * 0.018)))
    cv.line(pts, col, 6, a)


def battery(cv, x, y, lvl, a):
    cv.rrect(x - 110, y - 55, 220, 110, 18, None, a, stroke=LIGHT, lw=8, sa=a); cv.rrect(x + 110, y - 22, 20, 44, 6, LIGHT, a)
    n = int(math.ceil(lvl * 4 - 1e-6))
    for k in range(n): cv.rrect(x - 96 + k * 50, y - 40, 42, 80, 8, "#E5484D" if n <= 1 else GREEN, a)


def room_bg(cv, wallcol="#13284D"):
    poly(cv, [(0, 420), (W, 420), (W, 1300), (0, 1300)], wallcol, 1.0)
    poly(cv, [(0, 1300), (W, 1300), (W, H), (0, H)], "#0F2142", 1.0)
    for k in range(7): cv.line([(-200 + k * 230, H), (180 + k * 150, 1300)], "#1A3460", 4, 0.8)   # floor boards
    cv.line([(0, 1300), (W, 1300)], "#2B4A80", 6)


def piano(cv, x, y, a):
    cv.rrect(x, y, 300, 330, 10, "#2A1F3A", a, stroke="#5C4A7A", lw=4, sa=a)
    cv.rrect(x + 14, y + 120, 272, 60, 4, LIGHT, a)
    for q in range(17): cv.line([(x + 14 + q * 16, y + 120), (x + 14 + q * 16, y + 180)], "#8FA3C2", 2, a)
    for q in (1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16): cv.rrect(x + 14 + q * 16 - 5, y + 120, 10, 36, 2, NIGHT, a)
    cv.rrect(x + 20, y + 190, 260, 140, 6, "#33264A", a)


def stairs(cv, x, y, a):
    for q in range(6): poly(cv, [(x + q * 55, y + q * 70), (W + 20, y + q * 70), (W + 20, y + q * 70 + 70), (x + q * 55, y + q * 70 + 70)], "#1E3A66" if q % 2 else "#24426F", a)
    cv.line([(x - 10, y - 110), (x + 6 * 55, y + 6 * 70 - 110)], "#5C7FB6", 7, a)
    for q in range(6): cv.line([(x + q * 55, y + q * 70), (x + q * 55, y + q * 70 - 110)], "#5C7FB6", 4, a)


def picture(cv, x, y, w, h, kind, a):
    cv.rrect(x, y, w, h, 6, "#3B2F22", a); cv.rrect(x + 10, y + 10, w - 20, h - 20, 3, ["#3A5A8C", "#2F6B78", "#5C4A7A"][kind], a)
    if kind == 0: poly(cv, [(x + 14, y + h - 14), (x + w * 0.4, y + h * 0.35), (x + w * 0.7, y + h - 14)], "#1C3157", a); cv.circ(x + w * 0.72, y + h * 0.32, 12, YEL, a * 0.8)
    elif kind == 1:
        for q in range(3): cv.line([(x + 18, y + 30 + q * 22), (x + w - 18, y + 26 + q * 22)], LIGHT, 3, a * 0.6)
    else: cv.circ(x + w / 2, y + h / 2, min(w, h) * 0.22, "#F7C8C8", a * 0.8)


def lying_child(cv, x, y, sc, a=1.0, eyes="wide"):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(-math.pi / 2)
    person(cv, (0, 0), sc, child=True, eyes=eyes, mouth="flat", a=a, hands=[(-30 * sc, 30 * sc), (30 * sc, 34 * sc)], feet=[(-12 * sc, 56 * sc), (12 * sc, 58 * sc)])
    c.restore()


def tally(cv, n, x, y, a):
    for k in range(n):
        g, q = divmod(k, 5); gx = x + (g % 5) * 120; gy = y + (g // 5) * 110
        if q < 4: cv.line([(gx + q * 20, gy), (gx + q * 20, gy + 80)], YEL, 7, a)
        else: cv.line([(gx - 10, gy + 66), (gx + 76, gy + 14)], YEL, 7, a)


# ------------------------------------------------------------------ scenes (cv, t, tl, s, D, A)
def sc_room(cv, t, tl, s, D, A, full=False):
    cu = 1.0 if full else seg(t, K(s, "color"), K(s, "color") + 1.4)
    room_bg(cv, hexmix("#13284D", "#2E5E63", cu))
    for k, (x, y, w, h) in enumerate(((380, 520, 150, 120), (560, 500, 120, 160), (710, 540, 150, 110))):
        a = 1.0 if full else seg(t, K(s, "pictures") + 0.15 * k, K(s, "pictures") + 0.15 * k + 0.5)
        if a > 0.01: picture(cv, x, y, w, h, k, a)
    pa = 1.0 if full else seg(t, K(s, "piano"), K(s, "piano") + 0.6)
    if pa > 0.01: piano(cv, 40, 975, pa)
    sa = 1.0 if full else seg(t, K(s, "stairs"), K(s, "stairs") + 0.6)
    if sa > 0.01: stairs(cv, 800, 900, sa)
    lying_child(cv, 420, 1262, 2.4, eyes="wide")
    for k in range(14):   # dust in the light
        x = (k * 173 + tl * 14 * (1 + k % 3)) % 1080; y = 600 + (k * 97 + tl * 6) % 640
        cv.circ(x, y, 3 + k % 3, LIGHT, 0.18)


def sc_memgap(cv, t, tl, s, D, A):
    sc_room(cv, t, tl, s, D, A, full=True)
    for k, (x, y) in enumerate(((455, 580), (190, 1100), (920, 1010), (620, 580))):     # every detail lights up
        a = seg(tl, 0.3 + 0.35 * k, 0.6 + 0.35 * k)
        if a > 0.01: cv.circ(x, y, 70, None, 1, stroke=YEL, lw=5 * a); cv.c.new_path()
    g = seg(tl, 1.6, 2.4)          # except what happened: a torn blank patch where she lies
    if g > 0.01:
        poly(cv, [(260, 1120), (330, 1100), (400, 1130), (470, 1095), (560, 1118), (600, 1180), (585, 1260), (610, 1330), (540, 1360), (450, 1338),
                  (360, 1365), (280, 1330), (250, 1250)], "#C9D6EA", g * 0.92)
        cv.text("?", 430, 1230, 120, "#7F8FA9", g * 0.8, shadow=False)
    fz = seg(t, K(s, "frozen"), K(s, "frozen") + 1.2)
    if fz > 0.01:
        cv.rrect(0, 0, W, H, 0, "#9CC7F0", 0.16 * fz)
        for cx_, cy_ in ((0, 420), (W, 420), (0, 1500), (W, 1500)):
            for k in range(6):
                ang = k * math.pi / 6 + (0 if cx_ == 0 else math.pi / 2); L_ = 120 * fz
                cv.line([(cx_, cy_), (cx_ + (1 if cx_ == 0 else -1) * abs(L_ * math.cos(ang)), cy_ + L_ * math.sin(ang) * (1 if cy_ < 900 else -1))], LIGHT, 3, 0.5 * fz)
        cv.circ(900, 640, 70, "#0B1A3C", fz, stroke=LIGHT, lw=5); cv.line([(900, 640), (900, 590)], LIGHT, 6, fz); cv.line([(900, 640), (935, 660)], YEL, 6, fz)


def calm_wall(cv, t, tl, s, D, A, look=0.0, shake=0.0, eyes="dot", mouth="soft", ruler=True):
    mountains(cv, 0.8, shift=t * 3, y0=1500); wall_set(cv, t)
    R = climber_on_wall(cv, t, 560, 900, 3.0, shake=shake, eyes=eyes, mouth=mouth)
    cv.line([(560, 300), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4); cv.line([(560, 300), (448, 1440)], LIGHT, 3.5, 0.9)
    person(cv, (440, 1500 - 68 * 1.1), 1.1, hands=[(425, 1460), (450, 1440)], mouth="flat", dress="#5A6F95")
    if ruler:
        cv.line([(1000, 1500), (1000, 900)], YEL, 5, 0.9, dash=[18, 14]); cv.line([(975, 900), (1025, 900)], YEL, 5, 0.9)
        cv.line([(975, 1500), (1025, 1500)], YEL, 5, 0.9); WG.tag(cv, "20 m", 940, 830, 64, YEL)
    return R


def sc_eyes(cv, t, tl, s, D, A):
    calm_wall(cv, t, tl, s, D, A, shake=1.4, eyes="wide", mouth="flat", ruler=False)
    op = seg(tl, 0.2, 0.9) * (1 - 0.7 * seg(tl, 0.9, 1.05) + 0.7 * seg(tl, 1.05, 1.25)) * 0.35 + 0.65 * seg(tl, 1.2, 2.2)
    for sg in (-1, 1):
        pts = [(x, 960 + sg * (1 - (((x - 540) / 760) ** 2)) * op * 1150) for x in range(-40, 1121, 40)]
        poly(cv, ([(-40, 0 if sg < 0 else H)] + pts + [(1120, 0 if sg < 0 else H)]), NIGHT, 1.0)


def sc_rope(cv, t, tl, s, D, A):
    tz = K(s, "human", t - tl + 4); cam = 1900 * seg(t, t - tl + 0.4, tz - 0.2)
    mountains(cv, 0.6, shift=t * 3, y0=1500 - cam * 0.25)
    rr = np.random.default_rng(8)
    poly(cv, [(150, -50), (930, -50), (960, 3800 - cam), (120, 3800 - cam)], ROCK, 1.0, stroke=ROCK2, lw=5)
    for j in range(60):
        hx = 200 + rr.uniform(0, 680); hy = rr.uniform(0, 3400) - cam
        if -40 < hy < H + 40: cv.circ(hx, hy, rr.uniform(9, 15), [WARM, SKY, YEL][j % 3], 0.8)
    for j in range(24):
        fx = 200 + rr.uniform(0, 600); fy = rr.uniform(0, 3400) - cam; sz = rr.uniform(60, 130)
        if -150 < fy < H: poly(cv, [(fx, fy), (fx + sz, fy + sz * 0.4), (fx + sz * 0.4, fy + sz)], ROCK2, 0.5)
    poly(cv, [(0, 3500 - cam), (W, 3480 - cam), (W, H + 4000), (0, H + 4000)], "#0F2142", 1.0)
    R = climber_on_wall(cv, t, 560, 600 - cam, 3.0, shake=1.2, eyes="wide", mouth="flat")
    by = 3500 - cam
    cv.line([(560, 200 - cam), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 5); cv.line([(560, 200 - cam), (470, by - 110)], LIGHT, 4)
    cv.circ(560, 200 - cam, 14, None, 1, stroke=LIGHT, lw=4); cv.c.new_path()
    person(cv, (460, by - 64 * 1.7), 1.7, hands=[(440, by - 120), (480, by - 140)], mouth="flat", dress="#5A6F95", look=0.3)
    ha = win(t, K(s, "harness"), K(s, "rope"), 0.3, 0.3)
    if ha > 0.01: cv.circ(R["hip"][0], R["hip"][1] - 10, 70, None, 1, stroke=YEL, lw=6 * ha); cv.c.new_path()
    ra = win(t, K(s, "rope"), K(s, "human") + 0.5, 0.3, 0.4)
    if ra > 0.01: cv.line([(R["hip"][0], R["hip"][1] - 20), (560, 200 - cam), (470, by - 110)], YEL, 12, 0.35 * ra)
    pa = seg(t, K(s, "human"), K(s, "human") + 0.5)
    if pa > 0.01: cv.circ(460, by - 120, 120, None, 1, stroke=YEL, lw=6 * pa); cv.c.new_path()


def sc_breathe(cv, t, tl, s, D, A):
    c = cv.c; c.save(); c.translate(540, 1000); c.scale(1.9, 1.9); c.translate(-560, -820)
    wall_set(cv, t)
    stop = K(s, "cant"); sh = 1.2 + 1.6 * seg(t, stop, stop + 0.6)
    tw_ = math.sin(t * 9) * 0.15 * win(t, K(s, "move"), K(s, "move") + 1.2, 0.2, 0.4)
    R = climber_on_wall(cv, t, 560, 900, 3.0, reach=max(0, tw_), shake=sh, eyes="wide", mouth="o" if t < stop else "flat")
    cv.line([(560, 300), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4)
    c.restore()
    for k in range(4):   # short breaths
        ph = (tl * 1.6 + k / 4) % 1; a = (1 - ph) * 0.6 * (1 - seg(t, stop, stop + 0.4))
        if a > 0.01: cv.circ(700 + ph * 160, 330 + ph * 40, 14 + 30 * ph, LIGHT, a)
    ra = win(t, K(s, "reality"), stop + 0.6, 0.3, 0.4)
    if ra > 0.01:
        for k in range(3): cv.circ(540, 900, 520 + 60 * k + 30 * math.sin(t * 3 + k), None, 1, stroke=SKY, lw=4 * ra); cv.c.new_path()


def face(cv, x, y, r, a=1.0, eyes="wide", tears=0.0, t=0.0, mouth="flat"):
    cv.line([(x + r * 0.6, y - r * 0.7), (x + r * 1.35, y - r * 0.2), (x + r * 1.2, y + r * 0.9)], LIGHT, r * 0.3, a)
    cv.rrect(x - r * 0.28, y + r * 0.9, r * 0.56, r * 0.7, r * 0.1, LIGHT, a)
    poly(cv, [(x - r * 1.3, y + r * 2.4), (x - r * 0.4, y + r * 1.5), (x + r * 0.4, y + r * 1.5), (x + r * 1.3, y + r * 2.4)], DRESS, a, stroke=LIGHT, lw=r * 0.12)
    cv.circ(x, y, r, INK, a, stroke=LIGHT, lw=r * 0.28)
    arc(cv, x, y, r * 1.02, math.pi * 1.08, math.pi * 1.92, LIGHT, r * 0.24, a)
    for k in (-1, 1):
        if eyes == "wide": cv.circ(x + k * r * 0.36, y - r * 0.08, r * 0.2, LIGHT, a); cv.circ(x + k * r * 0.36, y - r * 0.08, r * 0.09, INK, a)
        else: cv.circ(x + k * r * 0.36, y - r * 0.08, r * 0.12, LIGHT, a)
    if mouth == "wobble":
        cv.line([(x - r * 0.26 + q * r * 0.13, y + r * 0.42 + (r * 0.05 if q % 2 else -r * 0.03)) for q in range(5)], LIGHT, r * 0.09, a)
    else: cv.line([(x - r * 0.22, y + r * 0.42), (x + r * 0.22, y + r * 0.42)], LIGHT, r * 0.09, a)
    if tears > 0.01:
        for k in (-1, 1):
            for q in range(3):
                ph = (t * 0.9 + q / 3 + (0.15 if k > 0 else 0)) % 1
                cv.circ(x + k * r * 0.4, y + r * 0.1 + ph * r * 1.4, r * 0.07 * (1 - ph * 0.3), SKY, tears * (1 - ph))


def sc_clock(cv, t, tl, s, D, A):
    m = WG.roll(t, t - tl + 0.3, 0, 40, 2.6); cx_, cy_, r = 540, 640, 250
    cv.circ(cx_, cy_, r, "#0B1A3C", 0.95, stroke=SKY, lw=8)
    for q in range(12):
        ang = q * math.pi / 6; cv.line([(cx_ + (r - 34) * math.cos(ang), cy_ + (r - 34) * math.sin(ang)), (cx_ + (r - 12) * math.cos(ang), cy_ + (r - 12) * math.sin(ang))], LIGHT, 6)
    c = cv.c; c.new_path(); c.move_to(cx_, cy_); c.arc(cx_, cy_, r - 50, -math.pi / 2, -math.pi / 2 + 2 * math.pi * m / 60); c.close_path(); cv.col(YEL, 0.18); c.fill()
    ang = -math.pi / 2 + 2 * math.pi * m / 60; cv.line([(cx_, cy_), (cx_ + (r - 60) * math.cos(ang), cy_ + (r - 60) * math.sin(ang))], YEL, 10)
    cv.line([(cx_, cy_), (cx_, cy_ - 120)], LIGHT, 10); cv.circ(cx_, cy_, 14, LIGHT)
    cv.text(f"{int(round(m))} min", cx_, cy_ + r + 90, 92, YEL)
    tr = seg(t, K(s, "crying"), K(s, "crying") + 0.6)
    face(cv, 540, 1240, 130, eyes="wide", tears=tr, t=t, mouth="wobble" if tr > 0.5 else "flat")


def sc_stuck(cv, t, tl, s, D, A):
    mountains(cv, 0.8, shift=t * 3, y0=1500); wall_set(cv, t)
    osc = 0.5 + 0.5 * math.sin(tl * 2.2)
    R = climber_on_wall(cv, t, 560, 900, 3.0, reach=0.55 * osc, shake=1.0, eyes="wide", mouth="flat")
    cv.line([(560, 300), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4)
    nh = (790, 560); ga = 0.5 + 0.5 * math.sin(t * 4)
    cv.circ(*nh, 22, YEL, 1.0); cv.circ(*nh, 44 + 8 * ga, None, 1, stroke=YEL, lw=4); cv.c.new_path()
    a1 = seg(t, K(s, "commit"), K(s, "commit") + 0.4)
    if a1 > 0.01: cv.line([(720, 700), (775, 600)], YEL, 6, a1, dash=[14, 10]); WG.tag(cv, "the next move", 820, 470, 48, YEL, a1)
    a2 = seg(t, K(s, "let"), K(s, "let") + 0.4)
    if a2 > 0.01: cv.line([(640, 1120), (640, 1420)], SOFT, 6, a2, dash=[14, 10]); poly(cv, [(615, 1410), (665, 1410), (640, 1450)], SOFT, a2); WG.tag(cv, "let go", 820, 1300, 48, SOFT, a2)


def sc_realize(cv, t, tl, s, D, A):
    floor(cv, 1300)
    ra = seg(t, K(s, "rock"), K(s, "rock") + 0.4); ma = seg(t, K(s, "memories"), K(s, "memories") + 0.4)
    poly(cv, [(80, 1300), (110, 860), (220, 760), (360, 790), (470, 1300)], ROCK, 1.0, stroke=YEL if ra > 0.5 else ROCK2, lw=6)
    for hx, hy in ((180, 950), (300, 880), (250, 1080), (380, 1010)): cv.circ(hx, hy, 12, WARM, 0.85)
    person(cv, (260, 1000), 1.3, hands=[(220, 940), (300, 935)], feet=[(240, 1080), (290, 1075)], eyes="wide", mouth="flat", knee=(1, -1))
    for k in range(3): film_strip(cv, 790, 820 + k * 150, 0.12 * math.sin(t + k), 1.25, 0.95, k)
    if ma > 0.5: cv.rrect(640, 720, 300, 470, 22, None, 1, stroke=YEL, lw=6)
    cv.text("Not on a rock", 275, 1390, 50, YEL if ra > 0.5 else SOFT, max(0.35, ra), maxw=420)
    cv.text("not inside those memories", 790, 1390, 46, YEL if ma > 0.5 else SOFT, max(0.35, ma), maxw=460)


def sc_again(cv, t, tl, s, D, A):
    SB.sc_wall(cv, t, tl, D, A, fall=True)
    for g in (0.8, 1.6):   # ghost falls
        ph = (tl + g) % 2.4; dy = 210 * (1 - math.exp(-max(0, ph - 0.75) * 7)) * (1 - seg(ph, 1.7, 2.35)) if ph > 0.75 else 0
        person(cv, (560 + 120 * (1 if g < 1 else -1), 900 + dy), 3.0, a=0.18, eyes="dot", mouth="flat")
    n = min(25, int(tl / 0.32)); cv.rrect(90, 250, 640, 260, 26, "#0B1730", 0.75); tally(cv, n, 130, 290, 1.0)


def sc_pattern(cv, t, tl, s, D, A):
    floor(cv)
    pts = []; per = 300; off = tl * 120
    for x in range(0, 1081, 6):
        ph = (x + off) % per; y = 760 - (220 if 140 < ph < 160 else (-90 if 160 <= ph < 175 else 0)) * math.sin(math.pi * (ph - 140) / 35 if 140 < ph < 175 else 0)
        pts.append((x, y))
    cv.line(pts, SKY, 7)
    an = seg(t, K(s, "anticipate"), K(s, "anticipate") + 0.5)
    for q in range(5):
        px = q * per - (off % per) + 150
        if 0 < px < W:
            cv.circ(px, 520, 16, YEL, 0.9); cv.line([(px, 540), (px, 720)], YEL, 3, 0.5, dash=[10, 10])
            if an > 0.01 and px > 560:
                cv.line([(px - 60, 700), (px - 10, 700), (px, 470), (px + 12, 790), (px + 24, 700), (px + 80, 700)], YEL, 5, 0.8 * an, dash=[12, 10])
    person(cv, (240, GROUND - 64 * 2.6), 2.6, eyes="dot", mouth="soft", look=1, hands=[(200, GROUND - 120), (340, GROUND - 260)])


def sc_talk(cv, t, tl, s, D, A):
    floor(cv)
    person(cv, (330, GROUND - 64 * 2.6), 2.6, eyes="dot", mouth="o" if (tl * 4) % 1 < 0.5 else "soft", look=1, hands=[(290, GROUND - 120), (440, GROUND - 230)])
    nod = 0.06 * math.sin(tl * 5) if tl > 1.0 else 0
    person(cv, (790, GROUND - 64 * 2.5), 2.5, eyes="dot", mouth="soft", look=-1, lean=-nod, col="#BFD3EE", dress="#3C70B0")
    g = seg(tl, 0.2, 0.9, eout); bubble(cv, 560, 560, 420 * g, 220 * g, g, tail_to=(400, 860))
    if g > 0.6:
        pts = [(380 + x, 560 - (60 if 150 < x < 170 else (-30 if 170 <= x < 185 else 0))) for x in range(0, 361, 6)]
        cv.line(pts, NAVY1, 6, g)
    sa = seg(t, K(s, "strangers"), K(s, "strangers") + 0.5)
    if sa > 0.01: WG.tag(cv, "complete strangers", 790, 1450, 46, SOFT, sa)


def sc_shame(cv, t, tl, s, D, A):
    backdrop(cv, t)
    u = seg(tl, 0.4, max(1.0, s["dur"] - 0.4))
    R = person(cv, (540, GROUND - 64 * 3.0), 3.0, lean=0.0, eyes="closed" if u < 0.5 else "dot", mouth="flat" if u < 0.6 else "soft")
    cloud(cv, 540, R["head"][1] - 150 - 60 * u, 1.6 * (1 - 0.7 * u), "#3C4A63", 0.9 * (1 - u))
    if u < 0.9:
        for k in range(5): cv.line([(470 + k * 35, R["head"][1] - 80 - 60 * u), (460 + k * 35, R["head"][1] - 30 - 60 * u)], "#5E6E88", 4, 0.6 * (1 - u))


def sc_proud(cv, t, tl, s, D, A):
    backdrop(cv, t)
    af = seg(t, K(s, "afraid"), K(s, "afraid") + 1.0)
    for i in range(6):
        ang = i * math.pi / 3 + t * 0.3; r = 330 + 220 * af; sh = math.sin(t * 30 + i) * 6 * af
        film_strip(cv, 540 + r * math.cos(ang) + sh, 900 + r * 0.8 * math.sin(ang), 0.3 * math.sin(ang), 0.75 - 0.3 * af, 0.8 - 0.35 * af, i)
    person(cv, (540, GROUND - 64 * 3.0), 3.0, eyes="dot", mouth="smile", hands=[(470, GROUND - 210), (610, GROUND - 210)])


def sc_quake(cv, t, tl, s, D, A):
    amp = 18 * math.exp(-tl * 0.8) + 4; c = cv.c; c.save(); c.translate(amp * math.sin(t * 47), amp * 0.6 * math.cos(t * 39))
    backdrop(cv, t)
    cr = seg(tl, 0.1, 1.4)
    path = [(540 + 70 * math.sin(k * 1.9) * (k > 0), GROUND + 4 + k * 70) for k in range(int(1 + 8 * cr))]
    left = [(540 - 60 * k * cr, GROUND + 4 + 10 * math.sin(k * 2.3)) for k in range(int(1 + 8 * cr))]
    if len(path) > 1: cv.line(path, NIGHT, 12)
    if len(left) > 1: cv.line(left, NIGHT, 8)
    rr = np.random.default_rng(5)
    for k in range(12):
        ph = (tl * 1.7 + k / 12) % 1; x = rr.uniform(80, 1000); cv.circ(x, GROUND - 120 * math.sin(math.pi * ph), 8 + k % 3 * 3, "#5C7FB6", 0.8)
    person(cv, (540, GROUND - 52 * 3.0), 3.0, eyes="wide", mouth="o", hands=[(380, GROUND - 260), (700, GROUND - 260)], feet=[(450, GROUND), (630, GROUND)])
    c.restore()


def sc_follows(cv, t, tl, s, D, A, day=False):
    sky = "#2C5A8E" if day else None
    if day:
        poly(cv, [(0, 0), (W, 0), (W, GROUND), (0, GROUND)], "#1E4A7C", 1.0); cv.circ(860, 360, 90, YEL, 0.55)
    off = (tl * 90) % 1400
    rr = np.random.default_rng(4)
    for k in range(10):
        bw, bh = rr.uniform(140, 220), rr.uniform(380, 820); x = k * 160 - off + (1400 if k * 160 - off < -250 else 0)
        cv.rrect(x, GROUND - bh, bw, bh, 6, "#1B3460" if not day else "#2B5687", 1.0)
        for q in range(int(bh // 90)):
            for p in range(2):
                lit = (k * 7 + q * 3 + p) % 4 != 0
                cv.rrect(x + 26 + p * (bw / 2 - 10), GROUND - bh + 40 + q * 90, 34, 44, 4, YEL if lit and not day else "#9CC7F0", 0.35 if not day else 0.3)
    floor(cv)
    R = walk(cv, 600, GROUND, 2.8, tl * 5.2, eyes="dot", mouth="flat" if not day else "smile", hands=None if not day else None)
    if not day:   # the silence: a dark bubble with a muted speaker, following a little behind
        bx = 330 + 16 * math.sin(tl * 1.3); by = R["head"][1] - 120 + 14 * math.sin(tl * 2.1); a = seg(tl, 0.3, 1.0)
        bubble(cv, bx, by, 220, 150, a * 0.85, col="#2A3346")
        poly(cv, [(bx - 54, by - 18), (bx - 30, by - 18), (bx, by - 44), (bx, by + 44), (bx - 30, by + 18), (bx - 54, by + 18)], SOFT, a)
        cv.line([(bx + 18, by - 30), (bx + 70, by + 30)], "#E5484D", 8, a); cv.line([(bx + 70, by - 30), (bx + 18, by + 30)], "#E5484D", 8, a)
    else:          # quietness carried into daily life: a soft halo and a coffee cup
        for k in range(3): cv.circ(R["head"][0], R["head"][1] + 80, 210 + 40 * k + 14 * math.sin(tl * 1.5 + k), None, 1, stroke=YEL, lw=3); cv.c.new_path()
        hx, hy = R["hands"][1]; cv.rrect(hx - 4, hy - 46, 44, 54, 8, LIGHT, 1.0); cv.line([(hx + 40, hy - 34), (hx + 54, hy - 26), (hx + 40, hy - 14)], LIGHT, 5)
        for q in range(2): cv.line([(hx + 10 + q * 14, hy - 60 - 10 * math.sin(tl * 3 + q)), (hx + 14 + q * 14, hy - 90)], SOFT, 3, 0.6)


def sc_daily(cv, t, tl, s, D, A): sc_follows(cv, t, tl, s, D, A, day=True)


def sc_judged(cv, t, tl, s, D, A):
    cv.rrect(0, 0, W, H, 0, NIGHT, 0.55)
    floor(cv, GROUND, "#1B2B4A")
    rr = np.random.default_rng(12)
    for k in range(14):
        x, y = rr.uniform(120, 960), rr.uniform(320, 1200)
        if abs(x - 540) < 200 and y > 800: continue
        a = seg(tl, 0.15 * k, 0.15 * k + 0.4); bl = 0.15 if (t * 0.7 + k * 0.37) % 1 < 0.05 else 1.0
        dx, dy = 540 - x, 1100 - y; d = math.hypot(dx, dy) + 1e-6
        if a > 0.01: eyes_pair(cv, x, y, 0.9 + 0.3 * (k % 3) / 2, a, (dx / d, dy / d), bl)
    sc = 2.6 - 0.5 * seg(tl, 0, 3)
    person(cv, (540, GROUND - 64 * sc), sc, eyes="wide", mouth="flat", hands=[(520, GROUND - 100 * sc), (560, GROUND - 100 * sc)])


def sc_risks(cv, t, tl, s, D, A):
    mountains(cv, 0.7, shift=t * 4, y0=1300)
    poly(cv, [(-20, 1150), (380, 1150), (420, H), (-20, H)], ROCK, 1.0, stroke=ROCK2, lw=5)
    poly(cv, [(700, 1150), (1100, 1150), (1100, H), (660, H)], ROCK, 1.0, stroke=ROCK2, lw=5)
    cv.line([(980, 1150), (980, 920)], SOFT, 5); poly(cv, [(980, 920), (1050, 945), (980, 970)], YEL, 1.0)
    WG.tag(cv, "enough", 930, 860, 48, YEL)
    u = seg(tl, 0.5, 2.0, lambda x: x)
    x = 180 + 680 * u; y = 1150 - 380 * math.sin(math.pi * u) * (0 < u < 1)
    if 0 < u < 1: person(cv, (x, y - 64 * 2.4), 2.4, lean=0.3, hands=[(x - 100, y - 260), (x + 120, y - 280)], feet=[(x - 90, y - 40), (x + 60, y - 10)], eyes="wide", mouth="o")
    else: walk(cv, x, 1150, 2.4, tl * 6 if u == 0 else 0, eyes="dot", mouth="flat")


def sc_push(cv, t, tl, s, D, A):
    sl = lambda x: 1500 - (x / 1080) * 420
    poly(cv, [(0, sl(0)), (W, sl(W)), (W, H), (0, H)], "#1B3460", 1.0, stroke="#3C70B0", lw=5)
    hurt = K(s, "hurt"); p = 0.35 * seg(tl, 0, hurt - (t - tl)) - 0.25 * seg(t, hurt, hurt + 0.8)
    bx = 640 + 260 * p; br = 150; ang = bx / br
    by = sl(bx) - br
    cv.circ(bx, by, br, ROCK, 1.0, stroke=ROCK2, lw=6)
    for k in range(3): cv.line([(bx + 60 * math.cos(ang + k * 2.1), by + 60 * math.sin(ang + k * 2.1)), (bx + 110 * math.cos(ang + k * 2.1 + 0.3), by + 110 * math.sin(ang + k * 2.1 + 0.3))], ROCK2, 6)
    x = bx - br - 90
    if t < hurt + 0.4:
        person(cv, (x, sl(x) - 56 * 2.6), 2.6, lean=0.65, hands=[(bx - br + 6, by - 30), (bx - br + 10, by + 20)], feet=[(x - 110, sl(x - 110)), (x - 40, sl(x - 40))], eyes="closed", mouth="flat", knee=(1, 1))
    else:
        x -= 60; sitting(cv, x, sl(x) - 4, 2.6, eyes="closed", mouth="flat", hands=[(x + 70, sl(x) - 60), (x + 100, sl(x) - 70)])


def sc_toomuch(cv, t, tl, s, D, A):
    floor(cv)
    for k, x in enumerate((90, 220, 340, 740, 860, 990)):
        person(cv, (x, GROUND - 64 * 1.5), 1.5, col="#8FA3C2", dress="#4A5F86", mouth="flat", a=0.75)
    pu = 1 + 0.04 * math.sin(tl * 3)
    for k in range(3): cv.circ(540, GROUND - 200, (170 + 40 * k) * pu, None, 1, stroke=WARM, lw=3); cv.c.new_path()
    person(cv, (540, GROUND - 64 * 2.9), 2.9, eyes="dot", mouth="flat", dress=WARM)
    for name, (x, y) in (("emotional", (300, 650)), ("intense", (790, 560)), ("much", (540, 430))):
        a = seg(t, K(s, name), K(s, name) + 0.4)
        if a > 0.01: WG.tag(cv, "too " + name, x, y, 56, YEL if name == "much" else LIGHT, a)


def sc_diagnosis(cv, t, tl, s, D, A):
    floor(cv)
    g = seg(tl, 0.2, 0.9); cv.rrect(180, 420, 720, 880, 28, "#E8EEF8", g, stroke=SOFT, lw=4)
    cv.rrect(180, 420, 720, 120, 28, SKY, g); cv.text("Diagnosis", 540, 480, 64, WHITE, g, shadow=False)
    for k in range(3): cv.rrect(250, 600 + k * 50, 580 - 120 * k, 20, 10, "#B7C4D9", g)
    ba = K(s, "borderline")
    for k, w in enumerate(("Borderline", "Personality", "Disorder")):
        a = seg(t, ba + 0.35 * k, ba + 0.35 * k + 0.3)
        if a > 0.01: cv.text(w, 540, 870 + k * 120, 92, INK if k else "#B03A3F", a, shadow=False)
    person(cv, (120, GROUND - 64 * 1.6), 1.6, eyes="wide", mouth="flat", look=1)


def sc_name(cv, t, tl, s, D, A):
    floor(cv)
    person(cv, (540, GROUND - 64 * 2.7), 2.7, eyes="wide", mouth="flat")
    pre = 1 - seg(t, K(s, "fear") - 0.4, K(s, "fear"))
    if pre > 0.01:
        for k in range(9):
            ang = k * 0.7 + tl * 1.8; r = 330 + 40 * math.sin(tl * 3 + k)
            x0, y0 = 540 + r * math.cos(ang), 840 + r * 0.7 * math.sin(ang)
            cv.line([(x0, y0), (x0 + 30, y0 + 40), (x0 - 10, y0 + 60), (x0 + 24, y0 + 110)], YEL if k % 2 else "#E5484D", 6, 0.6 * pre)
    for k, (w, (x, y)) in enumerate((("fear", (250, 560)), ("intensity", (790, 470)), ("chaos", (560, 330)))):
        a = seg(t, K(s, w), K(s, w) + 0.5)
        if a > 0.01:
            dx = 16 * math.sin(t * 1.3 + k); cloud(cv, x + dx, y, 1.25, "#3C4A63", 0.9 * a); cv.text(w, x + dx, y + 8, 60, YEL, a, shadow=False)


def sc_pharm_meds(cv, t, tl, s, D, A):
    SB.sc_pharmacy(cv, t, tl, dict(D, pills=None), A)
    a = seg(t, K(s, "three"), K(s, "three") + 0.5)
    for k in range(3):
        aa = seg(t, K(s, "three") + 0.2 * k, K(s, "three") + 0.2 * k + 0.4)
        if aa > 0.01:
            x = 700 + k * 100; cv.rrect(x, 780, 64, 82, 10, ["#F7C8C8", "#C8E2F7", "#F7EAB6"][k], aa, stroke=LIGHT, lw=3)
            cv.rrect(x - 4, 762, 72, 24, 6, LIGHT, aa)
    if a > 0.01: cv.text("3", 800, 560, 120, YEL, a)


def sc_wonderful(cv, t, tl, s, D, A):
    g = seg(t, K(s, "feel") - 0.2, K(s, "feel") + 1.2)
    cv.rrect(120, 330, 840, 880, 20, "#3B2F22", 1.0)
    x0, y0, w, h = 150, 360, 780, 820
    cv.rrect(x0, y0, w, h, 6, hexmix("#6FB6F0", "#55606F", g), 1.0)
    cv.circ(760, 520, 80, hexmix(YEL, "#9AA0A8", g), 1.0)
    poly(cv, [(x0, y0 + h), (x0, 900), (380, 820), (620, 900), (x0 + w, 840), (x0 + w, y0 + h)], hexmix("#3FBF7F", "#4E5560", g), 1.0)
    for k in range(9):
        fx = 200 + k * 85; fy = 1080 + 30 * math.sin(k)
        cv.line([(fx, fy + 60), (fx, fy)], hexmix("#2E8B57", "#4E5560", g), 5)
        cv.circ(fx, fy, 18, hexmix(["#F7C8C8", YEL, "#E3D2F5"][k % 3], "#7A808A", g), 1.0)
    for k in range(3):
        bx = (300 + k * 170 + tl * 60) % 760 + 160; by = 470 + 40 * k
        cv.line([(bx - 24, by - 10), (bx, by), (bx + 24, by - 10)], hexmix(NAVY1, "#3A3F48", g), 5)
    cv.rrect(x0 + w / 2 - 8, y0, 16, h, 0, "#3B2F22", 1.0); cv.rrect(x0, y0 + h / 2 - 8, w, 16, 0, "#3B2F22", 1.0)
    floor(cv)
    person(cv, (880, GROUND - 64 * 2.2), 2.2, eyes="dot", mouth="soft" if g < 0.5 else "flat", look=-1)


def sc_organizer(cv, t, tl, s, D, A): SB.sc_pharmacy(cv, t, tl, dict(D, pills=K(s, "pills", t - tl + 1.0)), A)


def slope_y(x): return 1500 - (x / 1080) * 560


def sc_hike(cv, t, tl, s, D, A):
    sun = 1.0; cv.circ(880, 330, 110, WARM, 0.75)
    for k in range(12):
        ang = k * math.pi / 6 + t * 0.2; cv.line([(880 + 140 * math.cos(ang), 330 + 140 * math.sin(ang)), (880 + 190 * math.cos(ang), 330 + 190 * math.sin(ang))], WARM, 6, 0.6)
    poly(cv, [(0, slope_y(0)), (W, slope_y(W)), (W, H), (0, H)], "#1B3460", 1.0, stroke="#3C70B0", lw=5)
    fa = seg(t, K(s, "faster"), K(s, "faster") + 0.5); spd = 4.2 + 3.6 * fa
    off = (tl * (70 + 60 * fa)) % 200
    for k in range(8):
        x = k * 200 - off; cv.circ(x, slope_y(x) + 30, 10, "#3C70B0", 0.8)
    x = 470; yg = slope_y(x); ph = tl * spd
    R = walk(cv, x, yg, 2.6, ph, lean=0.28, eyes="closed" if (tl % 3) < 0.15 else "dot", mouth="flat")
    hx, hy = R["sh"]; cv.rrect(hx - 120, hy - 10, 80, 140, 16, "#5A6F95", 1.0, stroke=LIGHT, lw=4)
    for k in range(3):
        ph2 = (tl * 1.2 + k / 3) % 1; cv.circ(R["head"][0] + 50 + 14 * k, R["head"][1] - 40 + ph2 * 80, 8, SKY, 0.7 * (1 - ph2))
    da = seg(t, K(s, "dehydrated"), K(s, "dehydrated") + 0.4)
    if da > 0.01:
        cv.rrect(150, 470, 70, 150, 14, None, da, stroke=LIGHT, lw=5, sa=da); cv.rrect(166, 440, 38, 34, 6, LIGHT, da); cv.rrect(160, 590, 50, 24, 6, SKY, da)
    an = seg(t, K(s, "anger"), K(s, "anger") + 0.6)
    if an > 0.01:
        for k in range(3): cv.circ(R["hip"][0], R["hip"][1] - 60, 200 + 34 * k + 12 * math.sin(t * 6 + k), None, 1, stroke="#E5484D", lw=4 * an); cv.c.new_path()


def sc_summit(cv, t, tl, s, D, A):
    mountains(cv, 0.8, shift=t * 4, y0=1300)
    poly(cv, [(80, H), (540, 980), (1000, H)], "#2B4A80", 1.0, stroke="#3C70B0", lw=5)
    tm = K(s, "made"); tsit = K(s, "sit")
    u = seg(tl, 0.0, max(0.6, tm - (t - tl) - 0.3))
    if t < tsit:
        x = 540 - 230 * (1 - u); y = 980 + 230 * (1 - u)
        up = seg(t, tm, tm + 0.4)
        if u < 1: walk(cv, x, y, 2.3, tl * 5, lean=0.25)
        else:
            person(cv, (540, 980 - 64 * 2.3), 2.3, hands=[(470 - 30 * up, 820 - 140 * up), (610 + 30 * up, 820 - 140 * up)], eyes="dot", mouth="smile" if up > 0.3 else "flat")
        fa = seg(t, tm - 0.6, tm)
        if fa > 0.01: cv.line([(640, 980), (640, 820)], SOFT, 5, fa); poly(cv, [(640, 820), (720, 845), (640, 870)], YEL, fa)
        if up > 0.01:
            for k in range(10):
                ang = k * math.pi / 5; r = 120 + 200 * up; cv.line([(540 + r * math.cos(ang), 760 + r * math.sin(ang)), (540 + (r + 40) * math.cos(ang), 760 + (r + 40) * math.sin(ang))], YEL, 5, 0.6 * (1 - seg(t, tm + 0.6, tm + 1.4)))
    else:
        sitting(cv, 520, 985, 2.3, eyes="closed", mouth="smile"); cv.line([(640, 980), (640, 820)], SOFT, 5); poly(cv, [(640, 820), (720, 845), (640, 870)], YEL, 1.0)
        cv.circ(540, 700, 260, YEL, 0.08 * seg(t, tsit, tsit + 1.0))


def sc_endless(cv, t, tl, s, D, A):
    pan = seg(tl, 0.0, 3.5)
    mountains(cv, 1.0, shift=pan * 1100 + tl * 20, y0=1150 - 100 * pan)
    poly(cv, [(-200, H), (260, 1380), (720, H)], "#2B4A80", 1.0, stroke="#3C70B0", lw=4)
    sitting(cv, 250, 1385, 1.4, eyes="dot", mouth="flat", look=1)
    for k in range(7):
        a = seg(tl, 0.6 + 0.7 * k, 1.0 + 0.7 * k)
        if a > 0.01:
            x = 440 + k * 95; y = 900 - k * 40; sc_ = 1 - k * 0.1
            cv.line([(x, y), (x, y - 70 * sc_)], SOFT, 3, a); poly(cv, [(x, y - 70 * sc_), (x + 36 * sc_, y - 58 * sc_), (x, y - 46 * sc_)], YEL, a)
    br = seg(t, K(s, "breaks"), K(s, "breaks") + 0.8)
    if br > 0.01: cv.rrect(0, 0, W, H, 0, NIGHT, 0.22 * br)


def sc_break(cv, t, tl, s, D, A):
    mountains(cv, 0.5, shift=t * 2, y0=1250)
    poly(cv, [(160, H), (540, 1150), (920, H)], "#20365E", 1.0)
    sitting(cv, 520, 1150, 2.0, eyes="closed", mouth="flat", hands=[(560, 1080), (590, 1090)])
    d = seg(tl, 0.3, 2.5); cv.rrect(0, 0, W, H, 0, NIGHT, 0.55 * d)
    cr = seg(tl, 1.2, 2.8)
    if cr > 0.01:
        pts = [(130, 380), (290, 470), (380, 430), (520, 560), (610, 520), (760, 650), (900, 620)]
        n = 1 + int(cr * (len(pts) - 1)); cv.line(pts[:n + 1], LIGHT, 2.5, 0.45)


def sc_exhausted(cv, t, tl, s, D, A):
    mountains(cv, 0.6, shift=t * 2, y0=1250)
    poly(cv, [(160, H), (540, 1150), (920, H)], "#20365E", 1.0)
    c = cv.c; c.save(); c.translate(540, 1112); c.rotate(-math.pi / 2)
    person(cv, (0, 0), 1.9, eyes="closed", mouth="flat", hands=[(-20, 70), (20, 74)], feet=[(-14, 120), (14, 122)]); c.restore()
    cv.line([(700, 1180), (700, 1040)], SOFT, 5); poly(cv, [(700, 1040), (740, 1090), (700, 1100)], YEL, 0.8)
    lv = 1 - 0.75 * seg(tl, 0.3, 2.5) - 0.25 * seg(t, K(s, "life"), K(s, "life") + 1.2)
    battery(cv, 540, 640, max(0.0, lv), 1.0)
    cv.rrect(0, 0, W, H, 0, NIGHT, 0.25)


def sc_boxmount(cv, t, tl, s, D, A):
    floor(cv, 1380)
    bw = 900; bh = 220; x = 90; y = 1160
    gm = seg(t, K(s, "mountain"), K(s, "mountain") + 1.4)
    cw = bw / 7
    for k in range(7):
        h_ = gm * (180 + 90 * ((k * 3) % 4)) * seg(gm, k / 10, k / 10 + 0.4)
        if h_ > 2: poly(cv, [(x + cw * k + 6, y + 30), (x + cw * k + cw / 2, y + 30 - h_), (x + cw * (k + 1) - 6, y + 30)], ["#1A3466", "#16305C", "#22406F"][k % 3], 1.0, stroke="#3C70B0", lw=3)
    pill_box(cv, x, y, bw, bh, 1.0, t)
    fu = seg(t, K(s, "further"), K(s, "further") + 2.0)
    fx, fy = 540 + 120 * fu, 680 - 260 * fu; sc_ = 1 - 0.6 * fu
    if gm > 0.3:
        poly(cv, [(fx - 120 * sc_, fy + 140 * sc_), (fx, fy), (fx + 120 * sc_, fy + 140 * sc_)], "#2B4A80", gm)
        cv.line([(fx, fy), (fx, fy - 70 * sc_)], SOFT, 4, gm); poly(cv, [(fx, fy - 70 * sc_), (fx + 40 * sc_, fy - 58 * sc_), (fx, fy - 46 * sc_)], YEL, gm)
    for k, wd in enumerate(("mountain", "battle", "summit")):
        a = seg(t, K(s, wd), K(s, wd) + 0.4)
        if a > 0.01: WG.tag(cv, "another " + wd, 540, 380 + 0 * k, 56, LIGHT, a * (1 - seg(t, K(s, ("battle", "summit", "zz")[k], 1e9), K(s, ("battle", "summit", "zz")[k], 1e9) + 0.2)))


def sc_research(cv, t, tl, s, D, A):
    cv.rrect(0, 0, W, H, 0, NIGHT, 0.5)
    cv.circ(860, 520, 70, "#FFE9A8", 0.25); cv.line([(860, 590), (900, 960)], SOFT, 6); cv.rrect(820, 950, 140, 20, 6, SOFT, 1.0)
    cv.rrect(80, 1180, 920, 30, 8, "#3B2F22", 1.0)
    cv.rrect(330, 980, 380, 200, 12, "#1B2B4A", 1.0, stroke=SOFT, lw=4); cv.rrect(350, 1000, 340, 160, 6, "#9CC7F0", 0.35 + 0.1 * math.sin(t * 5))
    poly(cv, [(290, 1180), (750, 1180), (790, 1205), (250, 1205)], SOFT, 1.0)
    for k in range(int(tl / 0.45) + 1):
        if k > 9: break
        x = 260 + (k % 4) * 40 + (k * 37) % 120; y = 460 + k * 38
        cv.rrect(x, y, 470, 260, 14, "#E8EEF8", 0.92); cv.rrect(x, y, 470, 44, 14, SKY, 0.95)
        for q in range(3): cv.rrect(x + 26, y + 80 + q * 46, 380 - 80 * q, 18, 9, "#B7C4D9", 0.9)
    sitting(cv, 180, 1180, 2.2, eyes="dot", mouth="flat", look=1, hands=[(300, 1100), (330, 1110)])


def sc_furious(cv, t, tl, s, D, A):
    floor(cv)
    cv.line([(460, 1240), (1060, 1240)], SKY, 8, 0.8, dash=[30, 18])
    for k in range(4): cv.circ(560 + k * 150, 1240, 16, SKY, 0.9)
    pu = 0.5 + 0.5 * math.sin(t * 7)
    for k in range(3): cv.circ(270, GROUND - 200, 200 + 30 * k + 10 * pu, None, 1, stroke="#E5484D", lw=4); cv.c.new_path()
    person(cv, (270, GROUND - 64 * 2.8), 2.8, eyes="wide", mouth="flat", hands=[(200, GROUND - 120), (340, GROUND - 120)], look=1)
    for k in (-1, 1): cv.circ(270 + k * 70, GROUND - 120, 18, LIGHT, 1.0)
    st = K(s, "decided"); a = seg(t, st, st + 0.25)
    if a > 0.01:
        z = 1 + 0.6 * (1 - a); c = cv.c; c.save(); c.translate(760, 1000); c.scale(z, z); stamp_text(cv, "DECIDED?", 0, 0, 84, "#E5484D", a, -0.12); c.restore()


def sc_explain(cv, t, tl, s, D, A):
    floor(cv)
    cv.rrect(110, 620, 380, 480, 22, "#E8EEF8", 1.0, stroke=SOFT, lw=4); cv.rrect(110, 620, 380, 70, 22, SKY, 1.0)
    pts = [(150 + x, 900 + 60 * math.sin(x * 0.05)) for x in range(0, 301, 6)]; cv.line(pts, NAVY1, 6)
    ex = seg(t, K(s, "explain"), K(s, "explain") + 0.5)
    if ex > 0.01: cv.circ(300, 900, 120, None, 1, stroke=YEL, lw=6 * ex); cv.c.new_path()
    dc = K(s, "decide"); u = seg(t, dc, dc + 2.4, lambda x: x)
    if u > 0.01:
        path = [(560, 1300), (640, 1180), (700, 1210), (780, 1020), (840, 1060), (900, 820), (960, 600)]
        n = max(2, int(1 + u * (len(path) - 1))); cv.line(path[:n], YEL, 10)
        if u > 0.95: cv.star(960, 560, 46, YEL, 1.0)
    person(cv, (560, GROUND - 64 * 2.5), 2.5, eyes="dot", mouth="soft" if u < 0.5 else "smile", look=1, hands=[(500, GROUND - 120), (640, GROUND - 260)])


def sc_unmute(cv, t, tl, s, D, A):
    backdrop(cv, t)
    R = person(cv, (440, GROUND - 64 * 3.0), 3.0, eyes="dot", mouth="flat" if t < K(s, "refuse") else "o", look=1,
               hands=[(380, GROUND - 120), (440 + 220 * seg(t, K(s, "refuse") - 0.3, K(s, "refuse")), GROUND - 330)])
    br = seg(t, K(s, "refuse"), K(s, "refuse") + 1.6)
    bx, by = 720, R["head"][1]
    if br < 0.25:
        cv.rrect(bx - 110, by - 120, 220, 240, 30, "#9CC7F0", 0.22, stroke="#9CC7F0", lw=5, sa=0.6)
        poly(cv, [(bx - 50, by - 20), (bx - 25, by - 20), (bx + 5, by - 50), (bx + 5, by + 50), (bx - 25, by + 20), (bx - 50, by + 20)], SOFT, 0.9)
        cv.line([(bx + 25, by - 30), (bx + 70, by + 30)], "#E5484D", 8); cv.line([(bx + 70, by - 30), (bx + 25, by + 30)], "#E5484D", 8)
        if br > 0.01:
            for k in range(5): cv.line([(bx, by), (bx + 110 * math.cos(k * 1.3), by + 120 * math.sin(k * 1.3))], LIGHT, 3, br * 4)
    else:
        rr = np.random.default_rng(2)
        for k in range(14):
            vx, vy = rr.uniform(-260, 260), rr.uniform(-300, 50); g_ = (br - 0.25) * 2.2
            x = bx + rr.uniform(-100, 100) + vx * g_; y = by + rr.uniform(-110, 110) + vy * g_ + 900 * g_ * g_
            poly(cv, [(x, y), (x + 30, y + 10), (x + 8, y + 36)], "#9CC7F0", max(0, 0.6 * (1 - g_)))
        for k in range(4):
            ph = (tl * 0.8 + k / 4) % 1; arc(cv, R["head"][0] + 40, R["head"][1], 80 + 360 * ph, -0.6, 0.6, YEL, 6, 0.8 * (1 - ph))


def sc_voice2(cv, t, tl, s, D, A):
    SB.sc_voice(cv, t, tl, D, A)
    a = seg(t, K(s, "alone"), K(s, "alone") + 0.8)
    person(cv, (980, GROUND - 64 * 1.3), 1.3, col=hexmix("#6F7F99", "#FFE9A8", a), dress=hexmix("#3A4A66", WARM, a), look=-1, mouth="smile" if a > 0.5 else "flat")
    if a > 0.01: cv.circ(980, GROUND - 110, 120, None, 1, stroke=YEL, lw=4 * a); cv.c.new_path()


def sc_stage(cv, t, tl, s, D, A):
    poly(cv, [(470, 0), (610, 0), (900, GROUND + 60), (180, GROUND + 60)], "#FFE9A8", 0.13)
    cv.rrect(80, GROUND, 920, 60, 12, "#2B4A80", 1.0)
    pr = seg(t, K(s, "proud"), K(s, "proud") + 0.8)
    person(cv, (540, GROUND - 64 * 3.0), 3.0, eyes="dot", mouth="smile", hands=[(470 - 60 * pr, GROUND - 210 - 120 * pr), (610 + 60 * pr, GROUND - 210 - 120 * pr)])
    sh = 1 - seg(t, K(s, "ashamed"), K(s, "ashamed") + 1.0)
    if sh > 0.01: cloud(cv, 540, 420 - 120 * (1 - sh), 0.6 * sh + 0.2, "#3C4A63", 0.8 * sh)
    if pr > 0.01:
        for k in range(10):
            x = 140 + (k * 97) % 800; y = 300 + (k * 131) % 700; tw_ = 0.5 + 0.5 * math.sin(t * 4 + k)
            cv.star(x, y, 10 + 8 * tw_, YEL, pr * (0.4 + 0.5 * tw_))


def sc_wall_again(cv, t, tl, s, D, A):
    lk = seg(t, K(s, "look"), K(s, "look") + 0.8)
    calm_wall(cv, t, tl, s, D, A, shake=0.0, eyes="dot", mouth="soft" if lk < 0.5 else "smile")
    if lk > 0.01:
        for k in range(3): arc(cv, 700, 600, 120 + 60 * k, -0.5, 0.5, YEL, 4, 0.5 * lk)


def sc_view2(cv, t, tl, s, D, A):
    SB.sc_view(cv, t, tl, D, A)
    mv = K(s, "movies"); fa = 1 - seg(t, mv + 0.3, mv + 1.6)
    if fa > 0.01:
        for i in range(4):
            sp = 0.6 * (1 - seg(t, mv, mv + 0.4)); ang = i * math.pi / 2 + min(t, mv) * sp
            film_strip(cv, 700 + 260 * math.cos(ang), 560 + 120 * math.sin(ang), 0.2 * math.sin(ang), 0.6, 0.55 * fa, i)


def sc_now(cv, t, tl, s, D, A):
    cv.line([(40, 1050), (1040, 1050)], SOFT, 6, 0.6)
    pa = seg(t, K(s, "past"), K(s, "past") + 0.4); fa_ = seg(t, K(s, "future"), K(s, "future") + 0.4)
    gone = seg(tl, s["dur"] - 1.6, s["dur"] - 0.4)
    for k in range(3): film_strip(cv, 140 + k * 70, 860 - k * 40, -0.2, 0.7, (0.75 - 0.25 * k) * (1 - pa * 0.7) * (1 - gone), k)
    for k in range(3): cloud(cv, 880 + k * 40, 820 - k * 60, 0.9, "#5E6E88", (0.55 - 0.15 * k) * (1 - fa_ * 0.7) * (1 - gone))
    if pa > 0.01: WG.tag(cv, "past", 200, 1150, 52, SOFT, pa * (1 - gone))
    if fa_ > 0.01: WG.tag(cv, "future", 880, 1150, 52, SOFT, fa_ * (1 - gone))
    for k in range(3): cv.circ(540, 1050, 40 + 40 * k + 10 * math.sin(t * 2 + k), None, 1, stroke=YEL, lw=4); cv.c.new_path()
    cv.circ(540, 1050, 28, YEL, 1.0)
    person(cv, (540, 1050 - 64 * 2.6 - 30), 2.6, eyes="dot", mouth="soft")


def sc_next(cv, t, tl, s, D, A):
    cam = -140 * seg(tl, 0.5, s["dur"])
    cv.circ(540, 150 - cam * 0.3, 220, YEL, 0.12)
    mountains(cv, 0.8, shift=t * 3, y0=1550 - cam * 0.4)
    c = cv.c; c.save(); c.translate(0, -cam); rock(cv, 170, 920, 260, 1.0, seed=5); c.restore()
    ph = (tl % 3.0) / 3.0; reach = seg(ph, 0.1, 0.6); up = seg(ph, 0.6, 0.95)
    R = climber_on_wall(cv, t, 560, 900 - up * 40, 3.0, reach=reach * (1 - up), shake=0, eyes="dot", mouth="soft")
    cv.line([(560, 200), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4)


def sc_lower(cv, t, tl, s, D, A):
    mountains(cv, 0.8, shift=t * 3, y0=1500); wall_set(cv, t)
    u = seg(tl, 0.3, max(1.0, s["dur"] - 1.0)); y = 760 + 560 * u
    if u < 0.97:
        R = person(cv, (600, y), 2.6, lean=-0.35, hands=[(560, y - 190), (585, y - 180)], feet=[(520, y + 140), (510, y + 110)], eyes="dot", mouth="smile" if u > 0.5 else "soft", knee=(1, 1))
        cv.line([(560, 300), (560, y - 190), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4)
    else: person(cv, (600, 1500 - 64 * 2.2), 2.2, eyes="dot", mouth="smile")
    cv.line([(560, 300), (448, 1440)], LIGHT, 3.5, 0.9)
    person(cv, (440, 1500 - 68 * 1.1), 1.1, hands=[(425, 1460), (450, 1440)], mouth="smile", dress="#5A6F95")
    cv.rrect(40, 230, 1000, 200, 26, "#0B1730", 0.7); heart_line(cv, 330, seg(tl, 0.5, 4.0), t)


def sc_help(cv, t, tl, s, D, A):
    floor(cv)
    th = K(s, "help"); ts = K(s, "strength")
    ff_ = seg(tl, 0.0, 0.6)
    if t < ts:
        sitting(cv, 470, GROUND, 2.6, eyes="closed" if t < th else "dot", mouth="flat", look=1, hands=[(470 + 60 + 140 * seg(t, th + 0.3, th + 0.9), GROUND - 150 - 60 * seg(t, th + 0.3, th + 0.9)), (440, GROUND - 60)])
    else:
        up = seg(t, ts, ts + 0.8); person(cv, (480, GROUND - 64 * 2.6 * (0.6 + 0.4 * up)), 2.6, eyes="dot", mouth="smile", look=1, hands=[(660, GROUND - 230), (420, GROUND - 120)])
    ha = seg(t, th, th + 0.6)
    if ha > 0.01: person(cv, (760, GROUND - 64 * 2.6), 2.6, a=ha, col="#BFD3EE", dress="#3C70B0", look=-1, mouth="smile", lean=-0.15 * (1 - seg(t, ts, ts + 0.8)), hands=[(670, GROUND - 220), (830, GROUND - 120)])


def sc_afraid(cv, t, tl, s, D, A):
    off = (tl * 110) % 240
    for k in range(6): x = k * 240 - off; cv.circ(x, GROUND + 40, 8, "#3C70B0", 0.7)
    floor(cv)
    R = walk(cv, 560, GROUND, 2.8, tl * 5, eyes="dot", mouth="soft")
    sx = 330 + 6 * math.sin(t * 25); walk(cv, sx, GROUND, 1.5, tl * 5 + 1, a=0.45, col="#8FA3C2", dress="#4A5F86", eyes="wide", mouth="flat")
    mv = seg(t, K(s, "moving"), K(s, "moving") + 1.2)
    cx, cy = R["sh"][0], R["sh"][1] + 40
    pu = 0.5 + 0.5 * math.sin(t * 5)
    cv.circ(cx, cy - 60 * mv, 14 + 10 * mv + 4 * pu, YEL, 0.5 + 0.5 * mv)
    if mv > 0.01:
        for k in range(3): cv.circ(cx, cy - 60 * mv, 40 + 30 * k + 10 * pu, None, 1, stroke=YEL, lw=3 * mv); cv.c.new_path()


def sc_girl_light(cv, t, tl, s, D, A):
    poly(cv, [(0, 520), (W, 520), (W, GROUND), (0, GROUND)], "#13284D", 1.0)
    poly(cv, [(0, GROUND), (W, GROUND), (W, H), (0, H)], "#0F2142", 1.0); cv.line([(0, GROUND), (W, GROUND)], "#2B4A80", 6)
    op = seg(tl, 0.3, 2.6); dx0, dw, dy0 = 720, 230, 760
    cv.rrect(dx0 - 14, dy0 - 14, dw + 28, GROUND - dy0 + 14, 6, "#2B4A80", 1.0); cv.rrect(dx0, dy0, dw, GROUND - dy0, 0, "#0A1326", 1.0)
    if op > 0.01:
        cv.rrect(dx0 + 4, dy0 + 4, max(4, dw * op - 8), GROUND - dy0 - 8, 0, "#FFE9A8", 0.85 * op)
        poly(cv, [(dx0, GROUND), (dx0 + dw * op, GROUND), (dx0 + dw * op - 640 * op, H), (dx0 - 300, H)], "#FFE9A8", 0.18 * op)
        cv.rrect(0, 0, W, H, 0, "#FFE9A8", 0.05 * op)
    poly(cv, [(dx0 + dw * op, dy0), (dx0 + dw, dy0 - 20 * op), (dx0 + dw, GROUND + 20 * op), (dx0 + dw * op, GROUND)], "#3B5C92", 1.0, stroke="#5C7FB6", lw=3)
    up = seg(tl, 1.4, 2.6)
    person(cv, (260, GROUND - 47 * 2.8 * (0.55 + 0.45 * up)), 2.8, child=True, eyes="dot" if up > 0.3 else "closed", mouth="soft", look=1,
           feet=[(250, GROUND), (290, GROUND)], hands=[(230, GROUND - 60), (300 + 120 * up, GROUND - 120 - 60 * up)])


def sc_two(cv, t, tl, s, D, A):
    floor(cv)
    sp = seg(t, K(s, "one"), K(s, "one") + 1.2)
    cv.circ(540, 900, 380 - 60 * sp, None, 1, stroke=SOFT, lw=4); cv.c.new_path()
    gx = 540 - 230 * sp; bx = 540 + 230 * sp
    sitting(cv, gx - 20, GROUND, 2.4, col="#8FA3C2", dress="#4A5F86", eyes="closed", mouth="flat", hands=[(gx + 10, GROUND - 90), (gx + 50, GROUND - 90)])
    person(cv, (bx, GROUND - 64 * 2.6), 2.6, a=0.6 + 0.4 * sp, dress=WARM, eyes="dot", mouth="smile", hands=[(bx - 80, GROUND - 330 - 60 * sp), (bx + 80, GROUND - 330 - 60 * sp)])
    if sp > 0.5:
        WG.tag(cv, "terrified of life", gx, 560, 46, SOFT, seg(sp, 0.5, 1.0)); WG.tag(cv, "wanted to live it", bx, 480, 46, YEL, seg(t, K(s, "other"), K(s, "other") + 0.4))


def sc_choice(cv, t, tl, s, D, A):
    poly(cv, [(470, H), (610, H), (580, 1250), (500, 1250)], "#2B4A80", 1.0)
    poly(cv, [(500, 1250), (580, 1250), (190, 700), (110, 720)], "#24426F", 1.0)
    poly(cv, [(500, 1250), (580, 1250), (950, 700), (870, 690)], "#2B4A80", 1.0)
    rk = 1.0; poly(cv, [(820, 700), (860, 470), (930, 420), (1010, 450), (1040, 700)], ROCK, 1.0, stroke=ROCK2, lw=4)
    for k in range(3): film_strip(cv, 130 + k * 60, 560 - k * 30, -0.3, 0.45, 0.7, k)
    ra = seg(t, K(s, "running"), K(s, "running") + 0.5)
    if ra > 0.01:
        u = (tl * 0.4) % 1; x = 520 - 300 * u; y = 1240 - 520 * u
        walk(cv, x, y, 1.3 * (1 - 0.5 * u), tl * 9, a=0.45 * ra, col="#8FA3C2", dress="#4A5F86", lean=-0.25, look=-1)
    for k, (nm, y) in enumerate((("diagnosis", 380), ("memories", 470))):
        a = seg(t, K(s, nm), K(s, nm) + 0.4)
        if a > 0.01: WG.tag(cv, ("not whether to have this diagnosis", "not whether the memories will return")[k], 540, y, 44, SOFT, a)
    turn = seg(t, K(s, "running"), K(s, "running") + 1.0)
    person(cv, (540, 1300 - 64 * 2.7), 2.7, eyes="dot", mouth="soft", look=-0.8 + 1.8 * turn)


def sc_warrior(cv, t, tl, s, D, A):
    sun = seg(tl, 0, 2.0)
    cv.circ(540, 820 - 120 * sun, 230, YEL, 0.32 + 0.25 * sun)
    for r_ in (320, 430): cv.circ(540, 820 - 120 * sun, r_, YEL, 0.06 * sun)
    mountains(cv, 1.0, shift=t * 5, warm=sun, y0=1350)
    poly(cv, [(250, H), (330, 1260), (460, 1180), (640, 1190), (760, 1270), (840, H)], ROCK, 1.0, stroke=ROCK2, lw=5)
    wind = math.sin(t * 3.0)
    R = person(cv, (540, 1185 - 64 * 3.2), 3.2, eyes="dot", mouth="soft", feet=[(470, 1185), (610, 1185)], hands=[(440, 1185 - 64 * 3.2 + 60), (650, 1185 - 64 * 3.2 - 230)])
    hx, hy = R["head"]; hr = R["hr"]; cv.line([(hx + hr * 0.9, hy - hr * 0.5), (hx + hr * 1.9 + 10 * wind, hy - hr * 0.2 + 8 * wind), (hx + hr * 2.7 + 16 * wind, hy - hr * 0.5 + 16 * wind)], LIGHT, 14)
    fa = win(t, K(s, "falling"), K(s, "afraid") + 0.3, 0.3, 0.4)
    if fa > 0.01:
        ph = (tl % 1.6) / 1.6; person(cv, (190, 600 + 500 * ph), 1.3, a=0.35 * fa, eyes="wide", mouth="o", hands=[(140, 520 + 500 * ph), (240, 520 + 500 * ph)])
    af = seg(t, K(s, "afraid"), K(s, "afraid") + 0.4)
    if af > 0.01: person(cv, (900 + 5 * math.sin(t * 28), 1400 - 64 * 1.4), 1.4, a=0.4 * af, col="#8FA3C2", dress="#4A5F86", eyes="wide", mouth="flat")


def sc_together2(cv, t, tl, s, D, A): SB.sc_together(cv, t, tl, D, A)


def sc_strips(cv, t, tl, s, D, A):
    """another one, another one: film strips fly at the camera one after another, she small in the middle"""
    backdrop(cv, t, 0.6)
    person(cv, (540, GROUND - 68 * 3.0), 3.0, eyes="wide", mouth="flat")
    for i in range(6):
        ph = (tl * 0.55 + i / 6) % 1; z = 0.4 + 3.6 * ph ** 2; a = min(1, ph * 5) * (1 - seg(ph, 0.8, 1.0))
        ang = i * 1.1 + ph * 0.8; r = 120 + 420 * ph
        film_strip(cv, 540 + r * math.cos(ang), 860 + r * 0.9 * math.sin(ang), 0.4 * math.sin(i + ph), z, a * 0.95, i)


def sc_steps(cv, t, tl, s, D, A):
    """every step feels impossible: close on her climbing big rock steps, faster than she should"""
    fa = seg(t, K(s, "faster"), K(s, "faster") + 0.5); sp = 0.45 + 0.5 * fa
    ph = (tl * sp) % 1; off = ph * 170
    for k in range(-1, 9):
        x0 = -200 + k * 170 - off * 1.0; y0 = 1500 - k * 120 + off * 120 / 170
        poly(cv, [(x0, y0), (x0 + 170, y0), (x0 + 170, y0 - 120), (x0 + 340, y0 - 120), (x0 + 340, H + 400), (x0, H + 400)], "#24426F" if k % 2 else "#1E3A66", 1.0, stroke="#3C70B0", lw=4)
    cv.circ(880, 330, 110, WARM, 0.6)
    lift = math.sin(math.pi * ph); x = 470; yg = 1500 - 2 * 120 + 0
    R = person(cv, (x, yg - 64 * 2.6 - 40 * lift), 2.6, lean=0.3, feet=[(x - 60 + 30 * ph, yg), (x + 60 + 40 * ph, yg - 120 + 120 * (1 - lift) * 0)], eyes="closed", mouth="flat",
               hands=[(x + 60, yg - 200), (x - 40, yg - 120)], knee=(1, 1))
    hx, hy = R["sh"]; cv.rrect(hx - 120, hy - 10, 80, 140, 16, "#5A6F95", 1.0, stroke=LIGHT, lw=4)
    for k in range(3):
        ph2 = (tl * 1.2 + k / 3) % 1; cv.circ(R["head"][0] + 50 + 14 * k, R["head"][1] - 40 + ph2 * 80, 8, SKY, 0.7 * (1 - ph2))


def wrap(f): return lambda cv, t, tl, s, D, A: f(cv, t, tl, dict(D, **s.get("d", {})), A)


SCENES = dict(tornado=wrap(SB.sc_tornado), wall=wrap(SB.sc_wall), fall=wrap(SB.sc_fall), child=wrap(SB.sc_child), five=wrap(SB.sc_five),
              view=sc_view2, together=wrap(SB.sc_together), room=sc_room, memgap=sc_memgap, eyes=sc_eyes, rope=sc_rope, breathe=sc_breathe,
              clock=sc_clock, stuck=sc_stuck, realize=sc_realize, again=sc_again, pattern=sc_pattern, talk=sc_talk, shame=sc_shame, proud=sc_proud,
              quake=sc_quake, follows=sc_follows, daily=sc_daily, judged=sc_judged, risks=sc_risks, push=sc_push, toomuch=sc_toomuch,
              diagnosis=sc_diagnosis, name=sc_name, pharm_meds=sc_pharm_meds, wonderful=sc_wonderful, organizer=sc_organizer, hike=sc_hike,
              summit=sc_summit, endless=sc_endless, brk=sc_break, exhausted=sc_exhausted, boxmount=sc_boxmount, research=sc_research,
              furious=sc_furious, explain=sc_explain, unmute=sc_unmute, voice=sc_voice2, stage=sc_stage, wall_again=sc_wall_again, now=sc_now,
              next=sc_next, lower=sc_lower, help=sc_help, afraid=sc_afraid, girl_light=sc_girl_light, strips=sc_strips, steps=sc_steps, two=sc_two, choice=sc_choice, warrior=sc_warrior)
NOZOOM = {"eyes", "rope", "next", "quake"}


# ------------------------------------------------------------------ music: chapter moods
def story_bed(dur, marks, bpm=62, seed=9):
    import music as MU
    tt, hz, saw, lp, env, room = MU.tt, MU.hz, MU.saw, MU.lp, MU.env, MU.room
    r = np.random.default_rng(seed); n = int(dur * SR); L = np.zeros(n); R = np.zeros(n)
    beat = 60 / bpm; bar = 4 * beat; nb = int(dur / bar) + 2
    PR = dict(dark=[(45, [57, 60, 64]), (41, [57, 60, 65]), (48, [55, 60, 64]), (43, [55, 59, 62])],
              tense=[(45, [57, 60, 64]), (40, [55, 59, 64]), (41, [57, 60, 65]), (40, [56, 59, 64])],
              child=[(45, [57, 60, 64]), (41, [57, 60, 65]), (48, [55, 60, 64]), (43, [55, 59, 62])],
              quiet=[(45, [57, 60, 64]), (41, [57, 60, 65])] * 2,
              warm=[(41, [57, 60, 65]), (48, [55, 60, 64]), (43, [55, 59, 62]), (45, [57, 60, 64])],
              hope=[(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])],
              rise=[(41, [60, 65, 69]), (48, [60, 64, 67]), (43, [62, 67, 71]), (45, [60, 64, 69])])
    def mood_at(x):
        m = marks[0][1]
        for a, mm in marks:
            if x + 0.01 >= a: m = mm
        return m
    def put(sig, st, g, pan=0.0):
        i = int(st * SR)
        if i >= n or i < 0: return
        j = min(n, i + len(sig)); L[i:j] += g * (1 - pan) * sig[:j - i]; R[i:j] += g * (1 + pan) * sig[:j - i]
    def piano(m, d=2.6):
        tc = tt(d); f = hz(m)
        s = sum(np.sin(2 * np.pi * k * f * tc * (1 + 0.0006 * k)) * np.exp(-tc * (1.4 + 1.1 * k)) / k ** 1.5 for k in range(1, 6))
        return lp(s, 1900) * env(len(tc), 0.006, 0.5)
    def bell(m, d=2.2):
        tc = tt(d); f = hz(m); return (np.sin(2 * np.pi * f * tc) + 0.3 * np.sin(2 * np.pi * 2.76 * f * tc) * np.exp(-tc * 6)) * np.exp(-tc * 2.6) * env(len(tc), 0.003, 0.3)
    for b in range(nb):
        st = b * bar; md = mood_at(st); root, ch = PR[md][b % 4]
        t = tt(bar + 1.6)
        pad = sum(sum(saw(2 * np.pi * hz(m) * (1 + dt) * t + r.uniform(0, 6)) for dt in (-0.004, 0.0, 0.004)) for m in ch)
        cut = dict(dark=850, tense=800, child=700, quiet=600, warm=1100, hope=1300, rise=1600)[md]
        pad = lp(pad, cut, 2) * np.minimum(1, t / 1.4) * np.minimum(1, (bar + 1.6 - t) / 1.4)
        put(pad, st, dict(dark=0.010, tense=0.011, child=0.007, quiet=0.008, warm=0.011, hope=0.013, rise=0.015)[md], -0.1)
        tb = tt(bar + 0.5)
        if md != "quiet" and md != "child": put(np.sin(2 * np.pi * hz(root - 12) * tb) * env(len(tb), 0.3, 0.8), st, 0.07 if md != "rise" else 0.085)
        if md == "tense":
            for q in range(8):
                tq = tt(beat * 0.45); put(np.sin(2 * np.pi * hz(root) * tq) * env(len(tq), 0.01, 0.2), st + q * beat / 2, 0.035)
        if md == "child":
            for q, off in enumerate((0, 1.5, 3)): put(bell(ch[q % 3] + 24), st + off * beat, 0.03, pan=0.3 * (q - 1))
            continue
        if md == "quiet": continue
        notes = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12, ch[2] + 12, ch[0] + 24]
        offs = {"dark": [0, 1.5, 2.5] if b % 2 else [0, 2], "tense": [0, 1, 2, 3], "warm": [0, 1, 2, 2.5, 3],
                "hope": [0, 1, 1.5, 2, 3, 3.5], "rise": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]}[md]
        for k, off in enumerate(offs):
            put(piano(notes[k % len(notes)]), st + off * beat, (0.05 if k else 0.065), pan=0.25 * (1 if k % 2 else -1))
        if md in ("hope", "rise"):
            tl_ = tt(bar + 0.8); m = ch[2] + 24 - (0 if md == "rise" else 12 * (b % 2))
            line = np.sin(2 * np.pi * hz(m) * tl_ + 0.004 * np.sin(2 * np.pi * 5 * tl_) * 60) * np.minimum(1, tl_ / 1.2) * np.minimum(1, (bar + 0.8 - tl_) / 0.8)
            put(lp(line, 3000), st, 0.022 if md == "hope" else 0.03, 0.2)
    x = np.stack([room(L, 2.4, 0.34, 1), room(R, 2.4, 0.34, 2)])
    fade = np.minimum(1, np.arange(n) / (1.5 * SR)) * np.minimum(1, (n - np.arange(n)) / (4.0 * SR))
    return x * fade / (np.abs(x).max() + 1e-9)


# ------------------------------------------------------------------ render
def wide_chunks(T, repl, sz, maxw):
    """subtitle chunks broken by width at the burned-in size (never shrunk), and at sentence punctuation"""
    import re
    m = Canvas(BG.copy()); CH = []
    def fix(x):
        x = re.sub(r"[\"“”]", "", x).rstrip(",;:")
        for a, b in (repl or {}).items(): x = x.replace(a, b)
        return x
    for l in T.L:
        cur = []
        for w in l["words"]:
            if cur and m.tw(fix(" ".join(x["w"] for x in cur + [w])), sz) > maxw:
                CH.append([fix(" ".join(x["w"] for x in cur)), cur[0]["t0"], cur[-1]["t1"]]); cur = []
            cur.append(w)
            if re.search(r"[.?!:]$", w["w"]) and len(cur) >= 2 or (re.search(r",$", w["w"]) and len(cur) >= 4):
                CH.append([fix(" ".join(x["w"] for x in cur)), cur[0]["t0"], cur[-1]["t1"]]); cur = []
        if cur: CH.append([fix(" ".join(x["w"] for x in cur)), cur[0]["t0"], cur[-1]["t1"]])
    for j in range(len(CH) - 1):
        CH[j][2] = min(CH[j + 1][1] - 0.02, CH[j][2] + 0.5) if CH[j + 1][1] - CH[j][2] < 0.5 else CH[j][2] + 0.25
    if CH: CH[-1][2] += 0.4
    return CH


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 6.0)); A = lambda a, d=0.0: T.at(a, d)
    SC = []
    for k, s in enumerate(D["scenes"]):
        s = dict(s, t0=0.0 if k == 0 else A(s["at"])); s["K"] = {n: A(a) for n, a in (s.get("k") or {}).items()}; SC.append(s)
    for k, s in enumerate(SC):
        s["t1"] = SC[k + 1]["t0"] if k + 1 < len(SC) else T.END + 1; s["dur"] = s["t1"] - s["t0"]
        if s["dur"] < 1.0: raise SystemExit(f"scene {k} {s['scene']} only {s['dur']:.2f} s")
    X = 0.5
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 400); tH = A(D.get("hook_until", "L1e"))
    CH = wide_chunks(T, S.get("subs_replace"), D.get("subs_size", 70), D.get("subs_w", 900))
    TI = [dict(x, t0=A(x["at"]), t1=A(x["until"]) if x.get("until") else None) for x in D.get("titles", [])]
    NO = [dict(x, t0=A(x["at"]), t1=A(x["until"]) if x.get("until") else T.END) for x in D.get("notes", [])]
    for k, x in enumerate(TI):
        if x["t1"] is None: x["t1"] = TI[k + 1]["t0"] if k + 1 < len(TI) else T.END
    CR = D.get("credit") or {}; SCR = D.get("start_credit") or {}

    def draw_scene(s, t):
        cv = Canvas(BG.copy()); tl = t - s["t0"]
        z = 0.0 if s["scene"] in NOZOOM else s.get("zoom", 0.045) * min(1.0, max(0.0, (tl + X) / (s["dur"] + X)))
        if z: cv.c.save(); cv.c.translate(540, 900); cv.c.scale(1 + z, 1 + z); cv.c.translate(-540, -900)
        SCENES[s["scene"]](cv, t, tl, s, D, A)
        if z: cv.c.restore()
        if s.get("tag"):
            ta = win(t, s["t0"] + 0.1, s["t1"], 0.3, 0.2) * (seg(t, tH + 0.3, tH + 0.7) if s["t0"] == 0 else 1)
            if ta > 0.01:
                w = cv.tw(s["tag"], 52) + 70; cv.rrect(540 - w / 2, 160, w, 84, 42, SKY, ta * 0.95); cv.text(s["tag"], 540, 202, 52, WHITE, ta, shadow=False)
        return cv.out().astype(np.float32)

    def layer(t, hook=True, subs=True, overlays=True):
        act = [s for s in SC if s["t0"] - X / 2 <= t < s["t1"] + X / 2]
        if not act: act = [SC[-1]]
        out = None
        for s in act:
            img = draw_scene(s, t)
            if out is None: out = img; continue
            u = ease((t - (s["t0"] - X / 2)) / X); out = out * (1 - u) + img * u
        if hook: out = paste_hook(out, hook_img, t, tH, HY, 1.0)
        cv = Canvas(out)
        if overlays:
            for x in TI:
                a = win(t, x["t0"] - 0.05, x["t1"], 0.3, 0.3)
                if a > 0.01:
                    y = x.get("y", 400); sz = x.get("size", 80)
                    WG.panel(cv, 50, y - 110 * (sz / 80) - 10, 980, 240 * (sz / 80) + 20, a * 0.9)
                    WG.title(cv, x["text"], y, a, sz)
            for x in NO:
                a = win(t, x["t0"], x["t1"], 0.3, 0.3)
                if a > 0.01:
                    y = x.get("y", 1470); ls = x["text"].split("|"); sz = x.get("size", 42); w = min(1000, max(cv.tw(l, sz, False) for l in ls) + 60); lh = sz * 1.3
                    cv.rrect(540 - w / 2, y - sz * 0.9 - lh * (len(ls) - 1) / 2, w, sz * 1.8 + lh * (len(ls) - 1), 24, "#0B1730", 0.8 * a)
                    for k, l in enumerate(ls): cv.text(l, 540, y + (k - (len(ls) - 1) / 2) * lh, sz, x.get("col", SOFT), a, bold=False, maxw=960, minsz=40)
            if SCR:
                a = win(t, -1, A(SCR.get("until", "L2")), 0.3, 0.5)
                if a > 0.01: cv.text(SCR["text"], 540, SCR.get("y", 1480), SCR.get("size", 44), SOFT, a, bold=False, maxw=980)
            if CR and t >= A(CR["at"]):
                a = win(t, A(CR["at"]), T.END + 1, 0.4, 0.3); cv.text(CR["text"], 540, CR.get("y", 1440), CR.get("size", 54), YEL, a, maxw=980)
        if subs: draw_subs(cv, CH, t, D.get("subs_y", 1640), D.get("subs_size", 70))
        return np.clip(cv.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    C = S.get("cover") or {}
    if C.get("bg") == "scene":
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(A(C.get("t", 3.0)), hook=False, subs=False, overlays=False)); S["_cover_bg"] = p
    marks = [(A(a), m) for a, m in D.get("music", [[0, "dark"]])]
    bp_ = f"{work}/bed.wav"
    import soundfile as sf
    sf.write(bp_, story_bed(T.END + 1.0, marks).T.astype(np.float32), SR)
    S.setdefault("music", {}); S["music"]["bed"] = "file:" + bp_
    ev = []
    F = D.get("five") or {}
    if F.get("at"): ev.append((A(F["at"]) + 1.4, "ding", -32))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True, crf=D.get("crf", 22))
    return dict(voice=J["voice"], mix=aud, scenes=[dict(scene=s["scene"], t0=round(s["t0"], 2)) for s in SC])
