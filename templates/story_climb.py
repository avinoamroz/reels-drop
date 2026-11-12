"""story_climb: a first-person story told with one minimal female stick figure (the climber), scene by scene.
Built for "The Warrior Within" (a speaker's own words, read by a stock female voice). No faces, no photos, nothing graphic.

Scenes (data.scenes [{scene, at, tag}], each from its anchor until the next; 0.45 s cross dissolves; the first is on screen
from frame 0, the last dissolves back into frame 0 so the reel loops):
  tornado   she stands at the foot of the rock, film strips spin around her like a tornado (stop at data.tornado_stop)
  wall      frozen high on a rock wall on a rope, belayer far below, dashed height ruler with data.height ("20 m")
  fall      reaches for the next hold, slips, drops on the rope, climbs back, again and again
  child     a small girl curled on a floor; a long shadow in a doorway; the door closes at data.door_close (no people shown hurting)
  pharmacy  a queue at a pharmacy counter; at data.pills a weekly pill organizer whose boxes multiply into a grid, her tiny figure inside one
  peaks     she reaches a summit (arms up), then the view pans out to endless peaks; at data.dim she sits and the scene dims softly
  five      a big count up to data.five.value with a card; at data.five.dots a 10 x 10 dot grid, 5 light up, she stands by one
  voice     her speech bubble grows, small figures turn to listen
  view      back high on the wall, calm, looking at mountains and valleys at sunrise
  together  the woman and the little girl side by side at the foot of the rock (same backdrop as tornado: circular)
Overlays: data.titles [{at, until, text, y, size}] (*word* yellow), data.notes [{at, until, text, y}] small lines (>= 40 px),
data.five {at, value, suffix, card, chip, dots}, data.credit {at, text}. Hook from frame 0 (data.hook_y), subtitles (data.subs_size).
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

INK = "#0E2250"; LIGHT = "#E8EEF8"; SOFT = "#C9D6EA"; ROCK = "#2A426E"; ROCK2 = "#33507F"; ROCK_D = "#1C3157"
DRESS = SKY; GIRL = YEL; GROUND = 1360


# ------------------------------------------------------------------ primitives
def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def arc(cv, x, y, r, a0, a1, col, lw, a=1.0):
    cv.c.new_path(); cv.c.arc(x, y, r, a0, a1); cv.col(col, a); cv.c.set_line_width(lw); cv.c.stroke(); cv.c.new_path()


def ik(a, b, l1, l2, sgn):
    """joint between a and b (two segments l1, l2), bending to side sgn"""
    dx, dy = b[0] - a[0], b[1] - a[1]; d = math.hypot(dx, dy) + 1e-6
    if d > l1 + l2 - 0.5: k = (l1 + l2 - 0.5) / d; dx, dy, d = dx * k, dy * k, l1 + l2 - 0.5
    ca = max(-1, min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))); al = math.acos(ca) * sgn; th = math.atan2(dy, dx) + al
    return (a[0] + l1 * math.cos(th), a[1] + l1 * math.sin(th))


def person(cv, hip, sc=3.0, lean=0.0, hands=None, feet=None, a=1.0, col=LIGHT, dress=DRESS, child=False, eyes="dot", mouth="soft",
           elbow=(1, -1), knee=(-1, 1), look=0.0):
    """female stick figure. hip (x, y) in px; lean (rad, + leans right); hands/feet: absolute px targets (None = rest pose).
    Returns key points. child=True: smaller torso, bigger head, two pigtails, yellow dress."""
    u = sc; T, HR, A1, A2, L1, L2 = ((38, 22, 20, 20, 23, 23) if child else (62, 22, 30, 30, 34, 34))
    up = (math.sin(lean), -math.cos(lean))
    neck = (hip[0] + up[0] * T * u, hip[1] + up[1] * T * u); head = (neck[0] + up[0] * (HR + 3) * u, neck[1] + up[1] * (HR + 3) * u)
    sh = (neck[0] - up[0] * 8 * u, neck[1] - up[1] * 8 * u); lw = (6.5 if not child else 6) * u
    feet = feet or [(hip[0] - 18 * u, hip[1] + (L1 + L2 - 4) * u), (hip[0] + 18 * u, hip[1] + (L1 + L2 - 4) * u)]
    hands = hands or [(sh[0] - 24 * u, sh[1] + 52 * u), (sh[0] + 24 * u, sh[1] + 52 * u)]
    for k, f in enumerate(feet):
        kn = ik(hip, f, L1 * u, L2 * u, knee[k]); cv.line([hip, kn, f], col, lw, a)
    # dress: trapezoid from the waist to below the hip, along the torso
    px = (-up[1], up[0]); w0, w1 = 9 * u, (26 if not child else 20) * u; top = 14 * u; bot = (24 if not child else 18) * u
    wst = (hip[0] + up[0] * top, hip[1] + up[1] * top); hem = (hip[0] - up[0] * bot, hip[1] - up[1] * bot)
    poly(cv, [(wst[0] - px[0] * w0, wst[1] - px[1] * w0), (wst[0] + px[0] * w0, wst[1] + px[1] * w0),
              (hem[0] + px[0] * w1, hem[1] + px[1] * w1), (hem[0] - px[0] * w1, hem[1] - px[1] * w1)], dress if not child else GIRL, a, stroke=col, lw=3 * u)
    cv.line([wst, neck], col, lw, a)
    for k, h in enumerate(hands):
        el = ik(sh, h, A1 * u, A2 * u, elbow[k]); cv.line([sh, el, h], col, lw, a)
    hr = HR * u
    if child:
        for s in (-1, 1): cv.circ(head[0] + s * (hr + 6 * u), head[1] - 4 * u, 8 * u, GIRL, a, stroke=col, lw=2.5 * u)
    else:   # ponytail
        tail = [(head[0] - look * 4 * u + 14 * u, head[1] - hr * 0.7), (head[0] + 30 * u, head[1] - hr * 0.2), (head[0] + 26 * u, head[1] + hr * 0.9)]
        cv.line(tail, col, 7 * u, a)
    cv.circ(*head, hr, INK, a, stroke=col, lw=lw)
    arc(cv, head[0], head[1], hr + 1 * u, math.pi * 1.08, math.pi * 1.92, col, 6 * u, a)      # hair line
    ex, ey = head[0] + look * 5 * u, head[1] - 2 * u
    if eyes == "dot":
        for s in (-1, 1): cv.circ(ex + s * 8 * u, ey, 3 * u, col, a)
    elif eyes == "closed":
        for s in (-1, 1): arc(cv, ex + s * 8 * u, ey - 1 * u, 4 * u, math.pi * 0.15, math.pi * 0.85, col, 2.5 * u, a)
    elif eyes == "wide":
        for s in (-1, 1): cv.circ(ex + s * 8 * u, ey, 5 * u, LIGHT, a); cv.circ(ex + s * 8 * u, ey, 2.4 * u, INK, a)
    my = head[1] + 8 * u
    if mouth == "soft": arc(cv, ex, my - 4 * u, 6 * u, math.pi * 0.25, math.pi * 0.75, col, 2.5 * u, a)
    elif mouth == "smile": arc(cv, ex, my - 5 * u, 7 * u, math.pi * 0.15, math.pi * 0.85, col, 3 * u, a)
    elif mouth == "flat": cv.line([(ex - 5 * u, my), (ex + 5 * u, my)], col, 2.5 * u, a)
    elif mouth == "o": cv.circ(ex, my, 3.4 * u, col, a)
    return dict(head=head, hr=hr, neck=neck, hip=hip, sh=sh, hands=hands, feet=feet, u=u)


def mountains(cv, a=1.0, shift=0.0, warm=0.0, y0=1180):
    """three layers of soft peaks (navy shades); shift pans the layers at different speeds"""
    for k, (yy, h, col, sp, per) in enumerate(((y0 - 120, 260, "#1A3466", 0.3, 360), (y0 - 30, 220, "#16305C", 0.6, 300), (y0 + 60, 180, "#122A52", 1.0, 260))):
        x0 = -((shift * sp) % per) - per; pts = [(x0, H)]
        x = x0
        while x < W + per:
            pts += [(x, yy + h * 0.15), (x + per * 0.5, yy - h * (0.75 + 0.25 * math.sin(x * 0.013 + k))), (x + per, yy + h * 0.15)]; x += per
        pts.append((x, H)); poly(cv, pts, col, a)
        if warm > 0: poly(cv, pts, "#F2A93B", a * warm * (0.10 - 0.03 * k))
        for j in range(int((W + 2 * per) / per) + 1):   # snow caps on the far layer
            if k == 0:
                px = x0 + j * per + per * 0.5; py = yy - h * (0.75 + 0.25 * math.sin((x0 + j * per) * 0.013 + k))
                poly(cv, [(px, py), (px - 34, py + 40), (px + 34, py + 40)], LIGHT, a * 0.35)


def rock(cv, x0, x1, top, a=1.0, seed=3):
    """rock wall from x0 to x1, from top down to the bottom of the frame, faceted, with holds"""
    rr = np.random.default_rng(seed); pts = [(x0 + 30, H), (x0, top + 300), (x0 + 40, top + 80), (x0 + 120, top)]
    xs = np.linspace(x0 + 120, x1 - 40, 7)
    for x in xs[1:]: pts.append((float(x), top + float(rr.uniform(-40, 30))))
    pts += [(x1, top + 120), (x1 + 20, H)]
    poly(cv, pts, ROCK, a, stroke=ROCK2, lw=5)
    for j in range(9):   # facets
        fx = x0 + 80 + rr.uniform(0, x1 - x0 - 160); fy = top + 160 + rr.uniform(0, H - top - 300); s = rr.uniform(60, 130)
        poly(cv, [(fx, fy), (fx + s, fy + s * 0.4), (fx + s * 0.4, fy + s)], ROCK2, a * 0.55)
    for j in range(16):  # holds
        hx = x0 + 70 + rr.uniform(0, x1 - x0 - 140); hy = top + 120 + rr.uniform(0, H - top - 260)
        cv.circ(hx, hy, rr.uniform(9, 15), ["#F2A93B", SKY, YEL][j % 3], a * 0.85)


def film_strip(cv, cx, cy, ang, sc, a, seed):
    c = cv.c; w, h = 190 * sc, 104 * sc
    c.save(); c.translate(cx, cy); c.rotate(ang)
    cv.rrect(-w / 2, -h / 2, w, h, 8 * sc, "#0A0F1C", a, stroke=SOFT, lw=2 * sc, sa=0.5 * a)
    for k in range(7):
        for s in (-1, 1): cv.rrect(-w / 2 + 10 * sc + k * 26 * sc, s * (h / 2 - 11 * sc) - 4 * sc, 13 * sc, 8 * sc, 2 * sc, LIGHT, a * 0.7)
    for k in range(3):   # three frames: faint memory shapes (a frame on a wall, piano keys, stairs)
        fx = -w / 2 + 12 * sc + k * 58 * sc; fy = -h / 2 + 22 * sc; fw, fh = 52 * sc, 60 * sc
        cv.rrect(fx, fy, fw, fh, 4 * sc, ["#3A5A8C", "#5C4A7A", "#2F6B78"][(k + seed) % 3], a * 0.85)
        kind = (k + seed) % 3
        if kind == 0: cv.rrect(fx + 14 * sc, fy + 14 * sc, 24 * sc, 18 * sc, 2 * sc, None, a, stroke=LIGHT, lw=2 * sc, sa=0.7 * a)
        elif kind == 1:
            for q in range(5): cv.rrect(fx + 4 * sc + q * 9 * sc, fy + 30 * sc, 8 * sc, 22 * sc, 1, LIGHT, a * 0.8)
        else: cv.line([(fx + 6 * sc, fy + 52 * sc), (fx + 6 * sc, fy + 40 * sc), (fx + 20 * sc, fy + 40 * sc), (fx + 20 * sc, fy + 28 * sc),
                       (fx + 34 * sc, fy + 28 * sc), (fx + 34 * sc, fy + 16 * sc), (fx + 48 * sc, fy + 16 * sc)], LIGHT, 2.5 * sc, a * 0.8)
    c.restore()


def bubble(cv, x, y, w, h, a, tail_to=None, col=LIGHT):
    if a <= 0.01 or w < 4: return
    if tail_to: poly(cv, [(x - w * 0.2, y + h * 0.35), (x - w * 0.02, y + h * 0.45), tail_to], col, a)
    cv.rrect(x - w / 2, y - h / 2, w, h, min(h / 2, 50), col, a)


# ------------------------------------------------------------------ scenes: f(cv, t, tl, D, A) where tl = time since scene start
def backdrop(cv, t, a=1.0):
    mountains(cv, a, shift=t * 6)
    rock(cv, 760, 1180, 520, a)
    cv.line([(0, GROUND + 4), (W, GROUND + 4)], "#2B4A80", 8, a)


def sc_tornado(cv, t, tl, D, A):
    backdrop(cv, t)
    stop = A(D.get("tornado_stop"), 1e9); sp = 1.4 + 2.2 * seg(t, 0.0, 3.0)
    frozen = t >= stop; th0 = (min(t, stop)) * sp; fa = 1 - seg(t, stop + 0.15, stop + 0.9)
    N = 9; items = []
    for i in range(N):
        y = 1240 - i * 78 + 10 * math.sin(th0 * 0.7 + i)
        r = 120 + (1300 - y) * 0.55; th = th0 + i * 2 * math.pi / N
        items.append((math.sin(th), 540 + r * math.cos(th), y, 0.55 + 0.25 * (1 + math.sin(th)) / 2 + (1300 - y) * 0.0006, math.cos(th) * 0.5, i))
    items.sort()
    behind = [it for it in items if it[0] < 0]; front = [it for it in items if it[0] >= 0]
    for d, x, y, s, ang, i in behind: film_strip(cv, x, y, ang, s, fa * 0.6, i)
    person(cv, (540, GROUND - 68 * 3.0), 3.0, eyes="wide" if not frozen else "closed", mouth="flat")
    for d, x, y, s, ang, i in front: film_strip(cv, x, y, ang, s, fa * 0.95, i)


def climber_on_wall(cv, t, x, y, sc, a=1.0, reach=0.0, eyes="wide", mouth="flat", shake=0.0):
    """she hangs on the wall face: hands up on holds, feet on holds below, harness rope up to the anchor"""
    jx = math.sin(t * 40) * shake
    hip = (x + jx, y)
    hl = (x - 52 * sc + jx, y - 128 * sc); hr_ = (x + 46 * sc + 26 * sc * reach + jx, y - 124 * sc - 46 * sc * reach)
    fl = (x - 44 * sc, y + 52 * sc); fr = (x + 40 * sc, y + 40 * sc)
    for p in (hl, (x + 46 * sc, y - 124 * sc), fl, fr): cv.circ(p[0], p[1] + 6 * sc, 9 * sc, "#F2A93B", a)
    return person(cv, hip, sc, lean=0.02, hands=[hl, hr_], feet=[fl, fr], a=a, eyes=eyes, mouth=mouth, elbow=(-1, 1), knee=(1, -1))


def wall_set(cv, t, cam=0.0):
    rock(cv, 170, 920, 260 - cam, 1.0, seed=5)
    cv.line([(560, 300 - cam), (560, 270 - cam)], SOFT, 6)
    cv.circ(560, 300 - cam, 12, None, 1, stroke=LIGHT, lw=4)


def sc_wall(cv, t, tl, D, A, fall=False):
    mountains(cv, 0.8, shift=t * 3, y0=1500)
    wall_set(cv, t)
    x, y = 560, 900; sc = 3.0
    dy = 0.0; reach = 0.0
    if fall:
        ph = tl % 2.4
        reach = seg(ph, 0.1, 0.6)
        if ph > 0.75: dy = 210 * (1 - math.exp(-(ph - 0.75) * 7)) * (1 - seg(ph, 1.7, 2.35)); reach = 0 if ph > 0.75 else reach
        swing = math.sin((ph - 0.75) * 8) * 26 * math.exp(-(ph - 0.75) * 2.5) if 0.75 < ph < 2.2 else 0
        x += swing
    R = climber_on_wall(cv, t, x, y + dy, sc, reach=reach, shake=0 if fall else 1.4, eyes="wide" if not fall else "dot")
    cv.line([(560, 300), (R["hip"][0], R["hip"][1] - 20)], LIGHT, 4)                    # rope up to the anchor
    cv.line([(560, 300), (448, 1440)], LIGHT, 3.5, 0.9)                                   # and down to the belayer
    person(cv, (440, 1500 - 68 * 1.1), 1.1, hands=[(425, 1460), (450, 1440)], mouth="flat", dress="#5A6F95")
    if not fall:   # dashed ruler and height
        cv.line([(1000, 1500), (1000, 900)], YEL, 5, 0.9, dash=[18, 14]); cv.line([(975, 900), (1025, 900)], YEL, 5, 0.9)
        cv.line([(975, 1500), (1025, 1500)], YEL, 5, 0.9); WG.tag(cv, D.get("height", "20 m"), 940, 830, 64, YEL)
        t40 = A(D.get("clock_at"), 1e9)
        if t >= t40 - 0.2:   # a small clock: minutes frozen
            ca = seg(t, t40 - 0.2, t40 + 0.2) * (1 - seg(t, A(D.get("clock_until"), 1e9) - 0.3, A(D.get("clock_until"), 1e9))); m = int(round(WG.roll(t, t40, 0, D.get("minutes", 40), 1.8)))
            cx_, cy_ = 150, 1180
            cv.circ(cx_, cy_, 72, "#0B1A3C", 0.9 * ca, stroke=SKY, lw=4)
            ang = -math.pi / 2 + 2 * math.pi * m / 60; cv.line([(cx_, cy_), (cx_ + 52 * math.cos(ang), cy_ + 52 * math.sin(ang))], YEL, 6, ca)
            cv.line([(cx_, cy_), (cx_, cy_ - 40)], LIGHT, 6, ca)
            WG.tag(cv, f"{m} min", cx_, cy_ + 120, 56, YEL, ca)


def sc_fall(cv, t, tl, D, A): sc_wall(cv, t, tl, D, A, fall=True)


def sc_child(cv, t, tl, D, A):
    poly(cv, [(0, 520), (W, 520), (W, GROUND), (0, GROUND)], "#13284D", 1.0)
    poly(cv, [(0, GROUND), (W, GROUND), (W, H), (0, H)], "#0F2142", 1.0)
    cv.line([(0, GROUND), (W, GROUND)], "#2B4A80", 6)
    tc = A(D.get("door_close"), 1e9); op = 1 - seg(t, tc, tc + 1.4)          # door opening 1 -> 0
    dx0, dw, dy0 = 720, 230, 760
    cv.rrect(dx0 - 14, dy0 - 14, dw + 28, GROUND - dy0 + 14, 6, "#2B4A80", 1.0)
    cv.rrect(dx0, dy0, dw, GROUND - dy0, 0, "#0A1326", 1.0)
    if op > 0.01:   # light from the doorway and a long shadow standing in it (no person shown)
        poly(cv, [(dx0, GROUND), (dx0 + dw * op, GROUND), (dx0 + dw * op - 520 * op, H), (dx0 - 260, H)], "#FFE9A8", 0.10 * op)
        cv.rrect(dx0 + 4, dy0 + 4, dw * op - 8 if dw * op > 12 else 4, GROUND - dy0 - 8, 0, "#FFE9A8", 0.5 * op)
        sh = op * seg(tl, 0.4, 1.4)
        if sh > 0.01:
            c = cv.c; c.save(); c.translate(dx0 + dw * 0.5, GROUND); c.scale(1.0, 1.0)
            poly(cv, [(-36, 0), (36, 0), (30, -320), (48, -400), (0, -470), (-48, -400), (-30, -320)], "#05070D", 0.75 * sh)
            cv.circ(0, -515, 40, "#05070D", 0.75 * sh); c.restore()
            poly(cv, [(dx0 - 10, GROUND), (dx0 + 60, GROUND), (dx0 - 420, GROUND + 260), (dx0 - 560, GROUND + 220)], "#05070D", 0.35 * sh)
    door = dw * op                                                                     # the door leaf swings shut
    poly(cv, [(dx0 + door, dy0), (dx0 + dw, dy0 - 20 * op), (dx0 + dw, GROUND + 20 * op), (dx0 + door, GROUND)], "#3B5C92", 1.0, stroke="#5C7FB6", lw=3)
    # the girl, curled on her side by the far wall, knees to chest, very still
    u = 3.4; hx, hy = 210, GROUND - 26 * u
    cv.line([(hx + 26 * u, hy + 6 * u), (hx + 62 * u, hy + 12 * u)], LIGHT, 6 * u)                                   # torso
    cv.line([(hx + 62 * u, hy + 12 * u), (hx + 44 * u, hy - 12 * u), (hx + 70 * u, hy - 6 * u)], LIGHT, 6 * u)         # legs up
    poly(cv, [(hx + 40 * u, hy), (hx + 66 * u, hy + 2 * u), (hx + 70 * u, hy + 20 * u), (hx + 36 * u, hy + 18 * u)], GIRL, 1.0, stroke=LIGHT, lw=3 * u)
    cv.line([(hx + 30 * u, hy + 4 * u), (hx + 48 * u, hy - 10 * u), (hx + 60 * u, hy - 12 * u)], LIGHT, 6 * u)         # arms around knees
    cv.circ(hx - 12 * u, hy - 22 * u, 8 * u, GIRL, 1.0, stroke=LIGHT, lw=2.5 * u)
    cv.circ(hx, hy, 22 * u, INK, 1.0, stroke=LIGHT, lw=6 * u)
    for s in (-1, 1): arc(cv, hx + 2 * u + s * 7 * u, hy - 2 * u, 4 * u, math.pi * 0.15, math.pi * 0.85, LIGHT, 2.5 * u)
    qa = seg(t, tc + 0.4, tc + 1.0)
    cv.line([(hx - 4 * u, hy + 9 * u), (hx + 8 * u, hy + 9 * u)], LIGHT, 2.5 * u)
    for k in range(3):   # small sound rings that fade to nothing: she learns to be quiet
        rr_ = 30 * u + k * 22 * u + (tl * 30 % 22)
        arc(cv, hx, hy, rr_, -math.pi * 0.35, math.pi * 0.05, SOFT, 4, 0.5 * (1 - qa) * (1 - k / 3))


def pill_box(cv, x, y, w, h, a, t, glow=-1):
    cv.rrect(x, y, w, h, h * 0.14, "#DDE6F4", a, stroke=SOFT, lw=3)
    cw = w / 7
    for k in range(7):
        cx = x + cw * k
        cv.rrect(cx + cw * 0.08, y + h * 0.1, cw * 0.84, h * 0.8, h * 0.1, ["#F7C8C8", "#C8E2F7", "#F7EAB6", "#CFF0D9", "#E3D2F5", "#F7D9B6", "#C8F0F0"][k], a)
        if w > 300:
            cv.text("SMTWTFS"[k], cx + cw / 2, y + h * 0.3, min(44, h * 0.3), INK, a, shadow=False)
            for q in range(3): cv.circ(cx + cw * (0.3 + 0.2 * q), y + h * 0.66, h * 0.07, [SKY, "#F2A93B", WHITE][q], a)
        if k == glow: cv.rrect(cx + cw * 0.08, y + h * 0.1, cw * 0.84, h * 0.8, h * 0.1, None, a, stroke=YEL, lw=4)


def sc_pharmacy(cv, t, tl, D, A):
    tp = A(D.get("pills"), 1e9); z = seg(t, tp, tp + 1.0)
    qa = 1 - z
    if qa > 0.01:
        cv.rrect(620, 860, 400, 60, 10, "#24406E", qa)                     # counter
        poly(cv, [(640, 920), (1000, 920), (1000, GROUND), (640, GROUND)], "#1B3460", qa)
        cv.rrect(700, 640, 260, 120, 18, "#0B1A3C", qa, stroke=GREEN, lw=4)
        cv.rrect(800, 664, 22, 72, 4, GREEN, qa); cv.rrect(775, 689, 72, 22, 4, GREEN, qa)
        cv.line([(0, GROUND + 4), (W, GROUND + 4)], "#2B4A80", 8, qa)
        for k, x in enumerate((560, 430, 300)):
            person(cv, (x, GROUND - 68 * 2.0), 2.0, a=qa * 0.8, col="#8FA3C2", dress="#4A5F86", look=1)
        person(cv, (160, GROUND - 68 * 2.0), 2.0, a=qa, look=1, eyes="dot", mouth="flat")
    if z > 0.01:
        # one weekly organizer, then it multiplies into weeks and years of little boxes
        mult = seg(t, tp + 1.6, tp + 4.0); bw = 900 - 520 * mult; bh = bw * 0.26
        rows = 1 + int(mult * 14); cols = 1 if mult < 0.05 else 2
        for r_ in range(rows):
            for c_ in range(cols):
                fade = z * (1 if r_ == 0 and c_ == 0 else seg(mult, r_ / 16, r_ / 16 + 0.1))
                x = 540 - (bw if cols == 2 else bw / 2) + c_ * (bw + 6) * (1 if cols == 2 else 0) - (3 if cols == 2 else 0)
                y = 640 + r_ * (bh + 14) - (0 if cols == 1 else 0)
                if y > 1500: continue
                pill_box(cv, x, y, bw, bh, fade * (1 - 0.35 * (r_ > 0)), t, glow=3 if (r_ == 0 and c_ == 0) else -1)
        if mult > 0.5:   # her tiny figure standing in one little box
            fa = seg(mult, 0.5, 0.8); cw = bw / 7; bx = 540 - bw + cw * 3.5; by = 640 + 3 * (bh + 14)
            cv.rrect(bx - cw * 0.42, by + bh * 0.1, cw * 0.84, bh * 0.8, bh * 0.1, "#0B1730", fa, stroke=YEL, lw=5)
            person(cv, (bx, by + bh * 0.86 - 66 * 0.42), 0.42, a=fa)


def sc_peaks(cv, t, tl, D, A):
    pan = seg(tl, 1.4, 4.0); td = A(D.get("dim"), 1e9); dm = seg(t, td, td + 1.0)
    mountains(cv, 1.0, shift=pan * 900, y0=1100 - 120 * pan)
    poly(cv, [(140 - 300 * pan, H), (540, 980), (940 + 300 * pan, H)], "#2B4A80", 1.0, stroke="#3C70B0", lw=4)
    walk = 1 - seg(tl, 0.0, 1.2)
    x = 540 - 160 * walk; y = 980 + 120 * walk
    if dm < 0.5:
        up = seg(tl, 1.2, 1.6)
        R = person(cv, (x, y - 68 * 2.2), 2.2, hands=[(x - 60 - 20 * up, y - 160 - 120 * up), (x + 60 + 20 * up, y - 160 - 120 * up)],
                   eyes="dot", mouth="smile" if up > 0.5 and pan < 0.4 else "flat")
        cv.line([(x + 80, y), (x + 80, y - 150)], SOFT, 5); poly(cv, [(x + 80, y - 150), (x + 150, y - 128), (x + 80, y - 106)], YEL, seg(tl, 1.0, 1.4))
    else:   # sits down, head low
        person(cv, (540, 980 - 30), 2.2, hands=[(500, 945), (580, 945)], feet=[(470, 985), (610, 985)], eyes="closed", mouth="flat", knee=(1, -1))
        cv.line([(620, 980), (620, 830)], SOFT, 5); poly(cv, [(620, 830), (690, 852), (620, 874)], YEL, 1.0)
    if dm > 0.01: cv.rrect(0, 0, W, H, 0, "#050B18", 0.38 * dm)


def sc_five(cv, t, tl, D, A):
    F = D.get("five") or {}; t0 = A(F.get("at"), t - tl); td = A(F.get("dots"), 1e9)
    ca = 1 - seg(t, td - 0.2, td + 0.4)
    if ca > 0.01:
        v = WG.roll(t, t0 + 0.2, 0, F.get("value", 5), 1.2)
        cv.text(f"{v:.0f}{F.get('suffix', '%')}", 540, 780, 360, YEL, ca * seg(t, t0, t0 + 0.3))
        if F.get("card") and t >= t0 + 1.0:
            a2 = ca * seg(t, t0 + 1.0, t0 + 1.4); WG.title(cv, F["card"], 1080, a2, 68)
            if F.get("chip"): WG.tag(cv, F["chip"], 540, 1230, 46, SOFT, a2)
    if t >= td - 0.2:
        da = seg(t, td - 0.2, td + 0.4); x0, y0, g = 540 - 4.5 * 78, 620, 78
        lit = [(2, 1), (6, 3), (1, 6), (8, 5), (4, 8)]
        for i in range(10):
            for j in range(10):
                k = lit.index((i, j)) if (i, j) in lit else -1
                on = k >= 0 and t >= td + 0.6 + 0.25 * k
                cv.circ(x0 + i * g, y0 + j * g, 16 if not on else 22, YEL if on else "#3C5A8A", da)
        her_at = (x0 + 4 * g, y0 + 8 * g); ha = seg(t, td + 2.0, td + 2.5)
        if ha > 0.01:
            cv.circ(*her_at, 34, None, 1, stroke=YEL, lw=5); cv.c.new_path()
            person(cv, (her_at[0], her_at[1] - 36 - 68 * 1.2), 1.2, a=ha, eyes="dot", mouth="smile")


def sc_voice(cv, t, tl, D, A):
    cv.line([(0, GROUND + 4), (W, GROUND + 4)], "#2B4A80", 8)
    g = seg(tl, 0.2, 1.6, eout); x = 300
    R = person(cv, (x, GROUND - 68 * 2.6), 2.6, hands=[(x - 70, GROUND - 330), (x + 110, GROUND - 400)], eyes="dot", mouth="o" if (tl * 4) % 1 < 0.5 else "smile", look=1)
    bw, bh = 520 * g, 230 * g; bx, by = 640, 640
    bubble(cv, bx, by, bw, bh, g, tail_to=(x + 60, R["head"][1] - 20))
    if g > 0.6:
        for k in range(3): cv.rrect(bx - 190 + 0, by - 60 + k * 50, (380 - 80 * k) * seg(tl, 1.0 + 0.2 * k, 1.5 + 0.2 * k), 22, 11, SKY if k != 1 else YEL, g)
    for k, (fx, fy) in enumerate(((640, 1180), (790, 1220), (940, 1180), (720, 1350), (880, 1360))):
        fa = seg(tl, 0.8 + 0.3 * k, 1.2 + 0.3 * k)
        if fa > 0.01: person(cv, (fx, fy - 68 * 1.1 + 160), 1.1, a=fa, col="#BFD3EE", dress="#3C70B0", look=-1, mouth="smile")


def sc_view(cv, t, tl, D, A):
    sun = seg(tl, 0.0, 3.0)
    cv.circ(820, 900 - 80 * sun, 120, YEL, 0.35 + 0.3 * sun)
    for r_ in (180, 260): cv.circ(820, 900 - 80 * sun, r_, YEL, 0.06 * sun)
    mountains(cv, 1.0, shift=t * 8, warm=sun, y0=1250)
    poly(cv, [(0, 1600), (300, 1520), (520, 1620), (760, 1540), (1080, 1640), (1080, H), (0, H)], "#0E2347", 1.0)   # valley floor
    cv.c.save(); cv.c.translate(0, 0)
    poly(cv, [(-40, H), (-40, 260), (180, 230), (330, 300), (360, 760), (350, 1200), (300, H)], ROCK, 1.0, stroke=ROCK2, lw=5)   # wall at the left edge
    cv.c.restore()
    br = 1 + 0.015 * math.sin(tl * 1.6)
    hip = (470, 930)
    R = person(cv, hip, 2.4 * br, lean=0.42, hands=[(400, 690), (470, 870)], feet=[(352, 1010), (350, 940)], eyes="dot", mouth="smile",
               look=1, elbow=(1, -1), knee=(-1, -1))
    cv.line([(250, 230), (400, 690), (hip[0] - 4, hip[1] - 24)], LIGHT, 4)


def sc_together(cv, t, tl, D, A):
    backdrop(cv, t)
    wx = 540 - 30
    R = person(cv, (wx - 60, GROUND - 68 * 3.0), 3.0, hands=None, eyes="dot", mouth="smile", look=0.6)
    R["hands"][1] = (wx - 60 + 72, GROUND - 68 * 3.0 + 60)
    g = person(cv, (wx + 150, GROUND - 47 * 3.0), 3.0, child=True, hands=[(wx + 150 - 60, GROUND - 47 * 3.0 - 10), (wx + 150 + 50, GROUND - 47 * 3.0 - 20)],
               feet=[(wx + 132, GROUND), (wx + 168, GROUND)], eyes="dot", mouth="o" if (tl * 3.5) % 1 < 0.5 else "smile", look=-0.4)
    cv.line([(wx - 60 + 30, GROUND - 68 * 3.0 - 38), (wx + 150 - 60, GROUND - 47 * 3.0 - 10)], LIGHT, 6.5 * 3.0 * 0.7)   # holding hands
    sp = A(D.get("speak_at"), 1e9); ga = seg(t, sp, sp + 0.8, eout)
    if ga > 0.01:
        bubble(cv, wx + 300, 760, 300 * ga, 150 * ga, ga, tail_to=(g["head"][0] + 30, g["head"][1] - 50), col=YEL)
        for k in range(3): cv.circ(wx + 300 - 70 + k * 70, 760, 14 * ga, INK, ga)


SCENES = dict(tornado=sc_tornado, wall=sc_wall, fall=sc_fall, child=sc_child, pharmacy=sc_pharmacy, peaks=sc_peaks, five=sc_five,
              voice=sc_voice, view=sc_view, together=sc_together)


# ------------------------------------------------------------------ render
def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.4)); A = lambda a, d=0.0: T.at(a, d)
    if S.get("music", {}).get("lift"): S["music"]["_lift"] = A(S["music"]["lift"])
    SC = [dict(s, t0=0.0 if k == 0 else A(s["at"])) for k, s in enumerate(D["scenes"])]
    for k, s in enumerate(SC): s["t1"] = SC[k + 1]["t0"] if k + 1 < len(SC) else T.END + 1
    X = 0.45
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 400); tH = T.E(0)
    CH = sub_chunks(T, S.get("subs_replace"))
    TI = [dict(x, t0=A(x["at"]), t1=A(x["until"]) if x.get("until") else None) for x in D.get("titles", [])]
    NO = [dict(x, t0=A(x["at"]), t1=A(x["until"]) if x.get("until") else T.END) for x in D.get("notes", [])]
    for k, x in enumerate(TI):
        if x["t1"] is None: x["t1"] = TI[k + 1]["t0"] if k + 1 < len(TI) else T.END
    CR = D.get("credit") or {}

    def draw_scene(s, t):
        cv = Canvas(BG.copy()); SCENES[s["scene"]](cv, t, t - s["t0"], D, A)
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
                    y = x.get("y", 400)
                    WG.panel(cv, 50, y - 110 * (x.get("size", 80) / 80) - 10, 980, 240 * (x.get("size", 80) / 80) + 20, a * 0.9)
                    WG.title(cv, x["text"], y, a, x.get("size", 80))
            for x in NO:
                a = win(t, x["t0"], x["t1"], 0.3, 0.3)
                if a > 0.01:
                    y = x.get("y", 1470); ls = x["text"].split("|"); w = min(1000, max(cv.tw(l, 42, False) for l in ls) + 60); lh = 54
                    cv.rrect(540 - w / 2, y - 38 - lh * (len(ls) - 1) / 2, w, 76 + lh * (len(ls) - 1), 24, "#0B1730", 0.8 * a)
                    for k, l in enumerate(ls): cv.text(l, 540, y + (k - (len(ls) - 1) / 2) * lh, 42, SOFT, a, bold=False, maxw=960, minsz=40)
            if CR and t >= A(CR["at"]):
                a = win(t, A(CR["at"]), T.END + 1, 0.4, 0.3); y = CR.get("y", 1460)
                cv.text(CR["text"], 540, y, CR.get("size", 54), YEL, a, maxw=980)
        if subs: draw_subs(cv, CH, t, D.get("subs_y", 1640), D.get("subs_size", 64))
        return np.clip(cv.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    C = S.get("cover") or {}
    if C.get("bg") == "scene":
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(A(C.get("t", 3.0)), hook=False, subs=False, overlays=False)); S["_cover_bg"] = p
    ev = [(s["t0"], "whoosh", -34) for s in SC[1:]]
    F = D.get("five") or {}
    if F.get("at"): ev.append((A(F["at"]) + 1.4, "ding", -30))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, scenes=[dict(scene=s["scene"], t0=round(s["t0"], 2)) for s in SC])
