"""boxing_ring (@noamroz__ story): an underdog boxing movie told in rounds. A lit ring in a dark arena stays on screen the
whole reel (spotlights sweep and the camera pushes in from frame 0). The small fighter (sky blue, white star on the trunks)
takes punches from labelled opponents (empires and governments only, never peoples), a big energy bar at the top drops and
recovers, round cards (ROUND n + name + year) open each round and shrink into a chip, a helper appears in the corner and
the bar recovers, real photos play in a picture-in-picture band above the ring (credit under the band), the darkest round
is still and quiet (no hits, no sound effects), he gets back up, trains, walks a school hallway where everybody steps
aside, an Iron Dome arc over the ring stops incoming rockets, warnings (leaflets) fall, crates of food go over the ropes,
counters count in the band, end card, and the last 0.5 s dissolves into frame 0 (circular).

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  hook_y        hook card centre at frame 0 (default 640, inside the band)
  energy        [[anchor, value 0..100], ...]   bar eases to each value over 0.35 s (heal keyframes ease over 0.9 s)
  bar_labels    [[anchor, text], ...]           text above the bar (e.g. "UNDERDOG", later "STILL STANDING")
  rounds        [{at, num, name, sub, card: seconds of the big card (default 1.3)}]  big card in the band, then a chip
  opponents     [{label, at, until}]            opponent slides in from the right, label tag over the head
  hits          [anchor, ...]                   impact times: opponent's jab lands, flash, camera shake, bar already dropping
  helpers       [{label, at, until, kind: "corner"|"plane"}]  corner man with a towel, or a plane crossing with crates
  photos        [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], label}]  picture-in-picture band (960x500)
  big           [{at, until, value, from, prefix, suffix, dec, sub, emo, col, roll, static}]  number card in the band
  chips         {at, until, items: [{at, emo, text}]}   training chips in the band
  counters      {at, until, rows: [{at, emo, value, dec, suffix, label}]}  counting rows in the band
  dark          [at, until]          darkest round: lights down, hero falls (lying), one cold spotlight, no SFX
  down          [at, up]             hero lying from `at`, back up at `up`
  ropes         [at, until]          hero pushed to the ropes (left)
  train         [at, until]          hero throws fast combos, grows to `grow` (default 1.22) by `until`
  hallway       [at, until]          school hallway scene instead of the ring (hero walks, everybody steps aside)
  dome          [at, until]          Iron Dome arc over the ring, 10 rockets, 9 stopped
  warn          {at, until, emo}     megaphone pulse at the hero + falling leaflets
  crates        {at, until}          crates passed over the ropes
  final_text    {at, text}           big line in the band for the last line (optional)
  tail          seconds after the last word (default 0.9)
  music flip:   music.flip anchor = where the doc_piano harmony lifts (he gets up)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2, cairo
import common as CMN
from common import *

HERO, HERO_TR, HERO_GL = SKY, "#0E2250", "#F5F5F7"
OPP, OPP_TR, OPP_GL = "#6E7896", "#7A2730", "#E5484D"
FLOOR, APRON = "#DCE6F2", "#163261"
BL, BR, FL, FR = (190, 1010), (890, 1010), (20, 1390), (1060, 1390)
HERO_X, OPP_X, FEET_Y = 375, 725, 1300
HSC, OSC = 1.3, 1.27
BAND = (60, 262, 960, 500)


def lerp(a, b, u): return a + (b - a) * u


def H01(i, k=0): return ((i * 2654435761 + k * 40503) % 1000003) / 1000003.0


# ------------------------------------------------------------------ figures
def boxer(cv, x, y, sc, col, trunks, glove, flip=1, punch=0.0, rear=0.0, recoil=0.0, lying=0.0, walk=None, star=False, a=1.0, guard=1.0):
    """silhouette boxer, feet at (x, y), facing right when flip=1"""
    if a <= 0.01: return
    c = cv.c; c.save()
    c.translate(x, y); c.rotate(-flip * (0.2 * recoil + (math.pi / 2) * lying)); c.translate(-flip * 26 * recoil * sc, 0)
    s = sc; f = flip
    def P(px, py): return (px * s * f, py * s)
    sw = math.sin(walk) if walk is not None else 0.0
    # legs
    for side in (-1, 1):
        kx = 26 * side + (14 * sw * side if walk is not None else 0); fx = 44 * side + (30 * sw * side if walk is not None else 0)
        cv.line([P(18 * side, -150), P(kx, -78), P(fx, -4)], col, 30 * s, a)
        cv.rrect(*P(fx - (10 if f * side > 0 else 30), -18), 40 * s, 20 * s, 8 * s, "#0B1730", a)
    # torso
    c.new_path(); pts = [P(-44, -278), P(44, -278), P(34, -176), P(-34, -176)]
    c.move_to(*pts[0]); [c.line_to(*p) for p in pts[1:]]; c.close_path(); cv.col(col, a); c.fill()
    cv.rrect(-50 * s, -192 * s, 100 * s, 66 * s, 14 * s, trunks, a)
    cv.rrect(-50 * s, -196 * s, 100 * s, 12 * s, 6 * s, "#FFFFFF", 0.85 * a)
    if star: cv.star(P(16, -158)[0], -158 * s, 14 * s, "#FFFFFF", a)
    # head
    cv.circ(*P(10, -318), 36 * s, col, a)
    # arms: rear (far) first, then lead
    def arm(shx, ext, hy):
        sh = P(shx, -262); gx = 70 + 165 * ext; gy = hy - 6 * ext
        el = P((shx + gx) / 2 - 6, (-262 + gy) / 2 + 34 * (1 - ext))
        g = P(gx, gy)
        cv.line([sh, el, g], col, 24 * s, a)
        cv.circ(g[0], g[1], 28 * s, glove, a); cv.circ(g[0] - 8 * s * f, g[1] - 8 * s, 9 * s, "#FFFFFF", 0.25 * a)
    if walk is not None:
        arm(-30, 0.0, -230 + 40 * (1 - guard)); arm(30, 0.0, -230 + 40 * (1 - guard))
    else:
        arm(-26, rear, -300 + 12 * (1 - guard)); arm(30, punch, -285 + 30 * (1 - guard))
    c.restore()


def kid(cv, x, y, sc, a=1.0, col="#8EA0BC"):
    """small neutral classmate silhouette"""
    cv.line([(x - 12 * sc, y), (x, y - 60 * sc), (x + 12 * sc, y)], col, 14 * sc, a)
    cv.rrect(x - 22 * sc, y - 130 * sc, 44 * sc, 74 * sc, 16 * sc, col, a)
    cv.circ(x, y - 152 * sc, 22 * sc, col, a)


# ------------------------------------------------------------------ scenes
def arena(cv, t, dark=0.0):
    c = cv.c
    for k, (x0, ph) in enumerate(((250, 0.0), (830, 1.7), (540, 3.1))):     # sweeping spotlights
        sway = 160 * math.sin(t * 0.7 + ph); tx = 540 + sway * (0.6 if k == 2 else 1.0)
        g = cairo.LinearGradient(x0, 0, tx, 1200)
        a0 = (0.20 if k < 2 else 0.14) * (1 - 0.85 * dark)
        g.add_color_stop_rgba(0, 1, 1, 0.92, a0); g.add_color_stop_rgba(1, 0.27, 0.64, 0.87, 0.0)
        c.new_path(); c.move_to(x0 - 30, -10); c.line_to(x0 + 30, -10); c.line_to(tx + 260, 1250); c.line_to(tx - 260, 1250); c.close_path()
        c.set_source(g); c.fill()
    for r in range(4):                                                        # crowd rows
        yy = 850 + r * 42; n = 16 + r * 2
        for i in range(n):
            x = (i + 0.5) * 1080 / n + (H01(i, r) - 0.5) * 30; bob = 3 * math.sin(t * 3 + i * 1.3 + r)
            cc = "#0A1630" if r % 2 else "#0D1C3C"
            cv.circ(x, yy - 18 + bob, 15 + 3 * H01(i, r + 7), cc, 1.0)
            cv.rrect(x - 22, yy - 4 + bob, 44, 40, 14, cc, 1.0)
            if H01(i, r + int(t * 6)) > 0.985 and dark < 0.5: cv.circ(x, yy - 30, 7, "#FFFFFF", 0.9)   # camera flashes


def ring(cv, t, dark=0.0, front=False, rope_push=0.0):
    """floor + posts + back/side ropes (front=False) or the front ropes and apron (front=True)"""
    c = cv.c
    if not front:
        c.new_path(); c.move_to(*BL); c.line_to(*BR); c.line_to(*FR); c.line_to(*FL); c.close_path()
        cv.col(FLOOR, 1.0 - 0.6 * dark); c.fill()
        c.save(); c.translate(540, 1230); c.scale(1.0, 0.32); cv.circ(0, 0, 190, SKY, 0.18 * (1 - dark), stroke=SKY, lw=6); c.restore()
        for (p, hcol, hh) in ((BL, SKY, 150), (BR, RED, 150)):
            cv.line([p, (p[0], p[1] - hh)], hcol, 12, 1.0)
        for hgt in (0.35, 0.65, 0.95):
            cv.line([(BL[0], BL[1] - 150 * hgt), (BR[0], BR[1] - 150 * hgt)], "#F5F5F7", 6, 0.85 - 0.4 * dark)
            cv.line([(BL[0], BL[1] - 150 * hgt), (FL[0], FL[1] - 210 * hgt)], "#F5F5F7", 7, 0.85 - 0.4 * dark)
            cv.line([(BR[0], BR[1] - 150 * hgt), (FR[0], FR[1] - 210 * hgt)], "#F5F5F7", 7, 0.85 - 0.4 * dark)
    else:
        cv.rrect(FL[0] - 6, FL[1], FR[0] - FL[0] + 12, 80, 4, APRON, 1.0)
        cv.text("THE LONGEST FIGHT", 540, FL[1] + 42, 40, SKY, 0.9 - 0.5 * dark, shadow=False)
        for (p, hcol) in ((FL, SKY), (FR, RED)):
            cv.line([p, (p[0], p[1] - 210)], hcol, 18, 1.0)
        for k, hgt in enumerate((0.35, 0.65, 0.95)):
            sag = rope_push * (18 + 8 * k)
            pts = [(FL[0], FL[1] - 210 * hgt), (540 - 260, FL[1] - 210 * hgt + sag), (FR[0], FR[1] - 210 * hgt)]
            cv.line(pts, "#FFFFFF", 9, 0.92 - 0.4 * dark)


def hallway(cv, t):
    c = cv.c
    def poly_(pts, col, a=1.0):
        c.new_path(); c.move_to(*pts[0]); [c.line_to(*p) for p in pts[1:]]; c.close_path(); cv.col(col, a); c.fill()
    poly_([(0, 0), (1080, 0), (1080, 1920), (0, 1920)], "#0D1D3F")
    poly_([(420, 780), (660, 780), (660, 1000), (420, 1000)], "#2A5C8E")                      # lit far wall
    poly_([(420, 1000), (660, 1000), (1080, 1520), (0, 1520)], "#1B355F")                      # floor
    poly_([(0, 1520), (1080, 1520), (1080, 1920), (0, 1920)], "#1B355F")
    for k in range(9):                                                                         # floor tile lines
        u = (k / 9 + (t * 0.12) % (1 / 9)); y = 1000 + 520 * u ** 2
        cv.line([(420 - 420 * u ** 2, y), (660 + 420 * u ** 2, y)], "#2B4C80", 3, 0.6)
    for side in (-1, 1):                                                                       # lockers
        for k in range(10):
            u0, u1 = k / 10, (k + 1) / 10
            def X(u): return 540 + side * (120 + 420 * u ** 1.6)
            x0, x1 = X(u0), X(u1)
            def Yt(u): return 780 - 480 * u ** 1.6
            def Yb(u): return 1000 + 520 * u ** 1.6
            col = SKY if k % 2 == 0 else "#3A86BE"
            poly_([(x0, Yt(u0) + 30), (x1, Yt(u1) + 30), (x1, Yb(u1) - 40), (x0, Yb(u0) - 40)], col, 0.85)
            cv.line([(x1, Yt(u1) + 30), (x1, Yb(u1) - 40)], "#0D1D3F", 4, 0.9)
            cv.line([(x0 + (x1 - x0) * 0.75, lerp(Yt(u0), Yt(u1), 0.75) + 120), (x0 + (x1 - x0) * 0.75, lerp(Yt(u0), Yt(u1), 0.75) + 170)], "#0D1D3F", 3, 0.8)
    for k in range(5):                                                                         # ceiling lights
        u = k / 5; y = 760 - 460 * u ** 1.6; w = 40 + 260 * u ** 1.6
        cv.rrect(540 - w / 2, y - 6 - 10 * u, w, 12 + 18 * u, 6, "#F5F5F7", 0.75)


def rocket(cv, x, y, ang, a=1.0):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.line([(-70, 0), (-10, 0)], "#FFB36B", 6, 0.45 * a)
    cv.rrect(-14, -6, 34, 12, 6, "#C9D6EA", a); cv.circ(22, 0, 6, RED, a)
    c.restore()


def burst(cv, x, y, u, a=1.0):
    r = 14 + 70 * eout(u); al = a * (1 - u)
    cv.circ(x, y, r, YEL, 0.35 * al); cv.circ(x, y, r * 0.55, "#FFFFFF", 0.7 * al)
    for k in range(8):
        ang = k * math.pi / 4 + 0.3
        cv.line([(x + math.cos(ang) * r * 0.6, y + math.sin(ang) * r * 0.6), (x + math.cos(ang) * r * 1.15, y + math.sin(ang) * r * 1.15)], YEL, 5, al)


def crate(cv, x, y, s, a=1.0):
    cv.rrect(x - 30 * s, y - 24 * s, 60 * s, 48 * s, 6 * s, "#C9A15B", a, stroke="#8A6A3A", lw=3)
    cv.line([(x - 30 * s, y), (x + 30 * s, y)], "#8A6A3A", 3, a)
    WG.emo(cv, "🍞", x, y - 2 * s, 30 * s, a)


def leaflet(cv, x, y, ang, a=1.0):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.rrect(-20, -13, 40, 26, 3, "#F5F5F7", a); cv.line([(-12, -4), (12, -4)], "#9FB0C8", 2, a); cv.line([(-12, 4), (6, 4)], "#9FB0C8", 2, a)
    c.restore()


def _load(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"boxing_ring: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z, w, h):
    ih, iw = im.shape[:2]; s = max(w / iw, h / ih) * max(1.0, z)
    cw, ch = w / s, h / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def _custom_bed(S):
    """doc_piano with the harmony lift at music.flip (he gets up)"""
    import music as MU
    orig = CMN.bed
    def bed(name, dur, lift=None):
        if name == "doc_piano" and lift is not None: return MU.doc_music(dur, flip_at=lift)
        return orig(name, dur, lift)
    CMN.bed = bed


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.9)); A = T.at
    if isinstance((S.get("cover") or {}).get("t"), str): S["cover"]["t"] = A(S["cover"]["t"])
    rng = lambda k, d=None: (A(D[k][0]), A(D[k][1])) if D.get(k) else d
    EN = sorted([(A(a), float(v)) for a, v in D.get("energy", [[0, 100]])])
    LAB = sorted([(A(a), s_) for a, s_ in D.get("bar_labels", [[0, "UNDERDOG"]])])
    RD = [dict(r, t0=A(r["at"])) for r in D.get("rounds", [])]
    for k, r in enumerate(RD): r["t1"] = RD[k + 1]["t0"] if k + 1 < len(RD) else A(r.get("until"), T.END)
    OP = [dict(o, t0=A(o["at"]), t1=A(o["until"])) for o in D.get("opponents", [])]
    HITS = sorted(A(h) for h in D.get("hits", []))
    HELP = [dict(h, t0=A(h["at"]), t1=A(h["until"])) for h in D.get("helpers", [])]
    PH = [dict(p, t0=A(p["at"]), t1=A(p["until"]), im=_load(p["file"])) for p in D.get("photos", [])]
    BIG = [dict(g, t0=A(g["at"]), t1=A(g["until"])) for g in D.get("big", [])]
    CHP = dict(D["chips"], t0=A(D["chips"]["at"]), t1=A(D["chips"]["until"]), items=[dict(i, t0=A(i["at"])) for i in D["chips"]["items"]]) if D.get("chips") else None
    CNT = dict(D["counters"], t0=A(D["counters"]["at"]), t1=A(D["counters"]["until"]), rows=[dict(r, t0=A(r["at"])) for r in D["counters"]["rows"]]) if D.get("counters") else None
    DARK = rng("dark"); DOWN = rng("down"); ROPES = rng("ropes"); TRAIN = rng("train"); HALL = rng("hallway"); DOME = rng("dome")
    WARN = dict(D["warn"], t0=A(D["warn"]["at"]), t1=A(D["warn"]["until"])) if D.get("warn") else None
    CRT = dict(D["crates"], t0=A(D["crates"]["at"]), t1=A(D["crates"]["until"])) if D.get("crates") else None
    FIN = dict(D["final_text"], t0=A(D["final_text"]["at"])) if D.get("final_text") else None
    END_AT = A((S.get("end") or {}).get("at")) if S.get("end") else T.END + 9
    GROW = float(D.get("grow", 1.22))
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"], size=D.get("hook_size", 132)); HY = D.get("hook_y", 512)
    bx, by, bw, bh = BAND
    mask = np.zeros((bh, bw), np.float32); cv2.rectangle(mask, (0, 0), (bw - 1, bh - 1), 1, -1)
    rr = 34; mk = np.zeros((bh, bw), np.uint8); cv2.rectangle(mk, (rr, 0), (bw - rr, bh), 255, -1); cv2.rectangle(mk, (0, rr), (bw, bh - rr), 255, -1)
    for cx_, cy_ in ((rr, rr), (bw - rr, rr), (rr, bh - rr), (bw - rr, bh - rr)): cv2.circle(mk, (cx_, cy_), rr, 255, -1, cv2.LINE_AA)
    MASK = (mk.astype(np.float32) / 255)[..., None]
    gy = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]; BANDGRAD = 1 - 0.7 * np.clip((gy - 0.6) / 0.4, 0, 1)
    ARENA0 = (BG * 0.55).astype(np.float32)

    def energy(t):
        v = EN[0][1]
        for k in range(1, len(EN)):
            t0, v1 = EN[k]; d = 0.9 if v1 > EN[k - 1][1] else 0.35
            if t >= t0: v = lerp(v, v1, ease((t - t0) / d)) if t < t0 + d else v1
        return max(0.0, min(100.0, v))

    def label(t):
        s_ = LAB[0][1]
        for t0, x in LAB:
            if t >= t0: s_ = x
        return s_

    def inwin(t, w, fi=0.3, fo=0.3): return win(t, w[0], w[1], fi, fo) if w else 0.0

    def hit_state(t):
        """(punch extension of the current opponent, hero recoil, shake)"""
        p = r_ = sh = 0.0
        for h in HITS:
            u = t - h
            if -0.22 <= u < 0: p = max(p, ease((u + 0.22) / 0.22))
            elif 0 <= u < 0.35: p = max(p, 1 - ease(u / 0.35)); r_ = max(r_, 1 - ease(u / 0.5)); sh = max(sh, 1 - u / 0.35)
            elif 0.35 <= u < 0.6: r_ = max(r_, 1 - ease(u / 0.5))
        return p, r_, sh

    def hero_scale(t):
        if not TRAIN: return 1.0
        return lerp(1.0, GROW, ease((t - TRAIN[0]) / max(0.5, TRAIN[1] - TRAIN[0])))

    def scene_ring(cv, t, dark):
        c = cv.c
        arena(cv, t, dark)
        p, rec, sh = hit_state(t)
        push = 1.0 + 0.035 * min(1, t / 6) + 0.02 * sh
        jx = 10 * sh * math.sin(t * 90); jy = 8 * sh * math.cos(t * 77)
        c.save(); c.translate(540 + jx, 1200 + jy); c.scale(push, push); c.translate(-540, -1200)
        ring(cv, t, dark)
        # opponent
        for o in OP:
            oa = win(t, o["t0"], o["t1"], 0.3, 0.3)
            if oa <= 0.01: continue
            slide = 1 - eout((t - o["t0"]) / 0.45); ox = OPP_X + 380 * slide + 280 * seg(t, o["t1"] - 0.3, o["t1"])
            boxer(cv, ox, FEET_Y, OSC, OPP, OPP_TR, OPP_GL, flip=-1, punch=p, a=oa, rear=0.15 * math.sin(t * 4) ** 2)
            WG.tag(cv, o["label"], ox + 30, FEET_Y - 470, 54, WHITE, oa, bg="#5E1E2A")
        # hero
        sc = HSC * hero_scale(t); hx = HERO_X; lying = 0.0; punch = 0.0; rear = 0.0; rp = 0.0
        if DOWN:
            lying = seg(t, DOWN[0], DOWN[0] + 0.7) * (1 - seg(t, DOWN[1], DOWN[1] + 0.9))
        if ROPES:
            rp = inwin(t, ROPES, 0.4, 0.6); hx -= 150 * rp
        if TRAIN and TRAIN[0] <= t < TRAIN[1]:
            ph = (t - TRAIN[0]) * 5.0; punch = max(0, math.sin(ph * math.pi)) ** 2; rear = max(0, -math.sin(ph * math.pi)) ** 2
        if OP and not any(o["t0"] <= t < o["t1"] for o in OP) and not (TRAIN and TRAIN[0] <= t < TRAIN[1]): hx = lerp(hx, 540 - 60, 0.0)
        bob = 0 if lying > 0.01 else 6 * math.sin(t * 5); hx += 230 * lying
        boxer(cv, hx, FEET_Y + bob, sc, HERO, HERO_TR, HERO_GL, flip=1, punch=punch, rear=rear, recoil=rec, lying=lying, star=True)
        # helpers in the corner (behind the ropes, back left)
        for h in HELP:
            ha = win(t, h["t0"], h["t1"], 0.35, 0.35)
            if ha <= 0.01: continue
            if h.get("kind", "corner") == "corner":
                hx = -60 + 150 * eout((t - h["t0"]) / 0.5); hy = FL[1] + 40
                boxer(cv, hx, hy, 0.95, "#C9D6EA", "#0E2250", "#C9D6EA", flip=1, a=ha, guard=0.0)
                cv.rrect(hx + 90, hy - 300, 80, 46, 10, "#FFFFFF", ha)              # towel
                for k in range(3):
                    u = ((t - h["t0"]) * 1.2 + k / 3) % 1.0
                    cv.text("+", HERO_X - 70 + 70 * k, FEET_Y - 480 - 110 * u, 80, GREEN, ha * (1 - u), shadow=False)
                WG.tag(cv, h["label"], hx + 60, hy - 390, 50, YEL, ha)
            else:
                u = (t - h["t0"]) / max(0.5, h["t1"] - h["t0"]); px = lerp(1160, -120, u); py = 930 - 60 * math.sin(u * math.pi)
                WG.emo(cv, "✈️", px, py, 110, ha); WG.tag(cv, h["label"], px, py - 85, 46, YEL, ha)
                for k in range(4):
                    cu = (t - h["t0"] - 0.25 - 0.3 * k) / 0.8
                    if 0 < cu < 1.6:
                        cx_ = lerp(1160, -120, (h["t0"] + 0.25 + 0.3 * k - h["t0"]) / max(0.5, h["t1"] - h["t0"]))
                        crate(cv, lerp(cx_, HERO_X - 150 + 70 * k, min(1, cu)), lerp(930, FEET_Y + 20, ease(min(1, cu))), 1.1, ha)
        if TRAIN and TRAIN[0] <= t < TRAIN[1]:                          # heavy bag
            cv.line([(hx + 290 * sc, 790), (hx + 290 * sc, 900)], "#9FB0C8", 4, 1.0)
            sw = 0.12 * math.sin((t - TRAIN[0]) * 5 * math.pi)
            c.save(); c.translate(hx + 290 * sc, 900); c.rotate(sw); cv.rrect(-56, 0, 112, 270, 36, "#7A2730", 1.0); c.restore()
        if sh > 0.05:
            ix = HERO_X + 75; iy = FEET_Y - 395
            for k in range(8):
                ang = k * math.pi / 4; r1 = 30 + 60 * (1 - sh)
                cv.line([(ix + math.cos(ang) * r1 * 0.5, iy + math.sin(ang) * r1 * 0.5), (ix + math.cos(ang) * r1, iy + math.sin(ang) * r1)], YEL, 6, sh)
        # Iron Dome arc + rockets
        if DOME:
            da = inwin(t, DOME, 0.4, 0.4)
            if da > 0.01:
                c.save(); c.translate(540, 1330); c.scale(1.0, 0.86)
                c.new_path(); c.arc(0, 0, 600, math.pi, 2 * math.pi); cv.col(SKY, 0.07 * da); c.fill()
                c.new_path(); c.arc(0, 0, 600, math.pi, 2 * math.pi); cv.col(SKY, 0.55 * da); c.set_line_width(6); c.stroke(); c.restore()
                n = 10; span = DOME[1] - DOME[0] - 0.6
                for k in range(n):
                    t0 = DOME[0] + 0.2 + span * k / n; u = (t - t0) / 0.9
                    if u < 0 or u > 1.5: continue
                    side = 1 if k % 2 == 0 else -1
                    sx, sy = 540 + side * 640, 600 + 80 * H01(k, 3); ex, ey = 540 + side * 200, 1150
                    if k == 6:                                                 # the one in ten: falls short, outside
                        if u < 1:
                            x = lerp(sx, sx - side * 260, u); y = lerp(sy, sy + 340, u ** 1.4)
                            rocket(cv, x, y, math.atan2(340, -side * 260), da * (1 - u))
                        continue
                    hit = 0.62
                    if u < hit:
                        x = lerp(sx, ex, u); y = lerp(sy, ey, u); rocket(cv, x, y, math.atan2(ey - sy, ex - sx), da)
                    else:
                        x = lerp(sx, ex, hit); y = lerp(sy, ey, hit); burst(cv, x, y, min(1, (u - hit) / 0.5), da)
        c.restore()
        ring(cv, t, dark, front=True, rope_push=rp)
        if WARN:
            wa = win(t, WARN["t0"], WARN["t1"], 0.3, 0.3)
            if wa > 0.01:
                for k in range(34):
                    fall = ((t - WARN["t0"]) * (180 + 90 * H01(k, 1)) + 900 * H01(k, 2)) % 1100
                    leaflet(cv, 40 + 1000 * H01(k, 4) + 30 * math.sin(t * 2 + k), 300 + fall, math.sin(t * 3 + k) * 0.8, wa * 0.9)
                hx2 = HERO_X; hy2 = FEET_Y - 300 * sc
                WG.emo(cv, WARN.get("emo", "📢"), hx2 + 120 * sc, hy2, 90, wa)
                for k in range(3):
                    u = ((t - WARN["t0"]) * 1.4 + k / 3) % 1.0
                    cv.circ(hx2 + 150 * sc, hy2, 30 + 120 * u, None, 1, stroke=YEL, lw=4); cv.c.new_path()
        if CRT:
            ca = win(t, CRT["t0"], CRT["t1"], 0.3, 0.3)
            if ca > 0.01:
                for k in range(6):
                    u = ((t - CRT["t0"]) * 0.55 - k * 0.16)
                    if 0 < u < 1:
                        x = lerp(HERO_X + 120, 1150, u); y = FEET_Y - 330 - 180 * math.sin(u * math.pi)
                        crate(cv, x, y, 1.4, ca)

    def scene_hall(cv, t):
        hallway(cv, t)
        u = ease((t - HALL[0]) / max(0.5, HALL[1] - HALL[0]))
        for k, (side, depth) in enumerate(((-1, 0.25), (1, 0.4), (-1, 0.6), (1, 0.75))):
            y = 1010 + 500 * depth ** 1.4; s_ = 0.5 + 1.0 * depth; step = seg(u, depth - 0.45, depth - 0.15)
            x = 540 + side * (60 + 220 * depth + (90 + 120 * depth) * step)
            kid(cv, x, y, s_, 0.95)
        sc = lerp(0.62, 1.12, u); y = lerp(1060, 1420, u)
        boxer(cv, 540, y, sc, HERO, HERO_TR, HERO_GL, walk=t * 7, star=True)

    def band_photo(out, t):
        for p in PH:
            a = win(t, p["t0"], p["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            img = _kb(p["im"], lerp(kb[0], kb[3], u), lerp(kb[1], kb[4], u), lerp(kb[2], kb[5], u), bw, bh) * BANDGRAD
            pop = 0.96 + 0.04 * eout((t - p["t0"]) / 0.35)
            if pop < 0.999:
                img = cv2.resize(img, None, fx=pop, fy=pop); ph_, pw_ = img.shape[:2]; pad = np.zeros((bh, bw, 3), np.float32)
                oy, ox = (bh - ph_) // 2, (bw - pw_) // 2; pad[oy:oy + ph_, ox:ox + pw_] = img; img = pad
            reg = out[by:by + bh, bx:bx + bw]
            out[by:by + bh, bx:bx + bw] = reg * (1 - MASK * a) + img * MASK * a
        return out

    def ui(cv, t, dark):
        # energy bar
        e = energy(t); col = GREEN if e > 60 else (YEL if e > 30 else RED)
        cv.text(label(t), 70, 62, 46, SKY if e > 30 else "#C9D6EA", 1.0, align="l", shadow=False, maxw=700)
        cv.star(1000, 62, 18, WHITE, 1.0)
        cv.rrect(60, 92, 960, 56, 28, "#0B1730", 0.95, stroke="#C9D6EA", lw=3, sa=0.6)
        fw = max(0.0, 948 * e / 100)
        if fw > 2: cv.rrect(66, 98, max(44, fw), 44, 22, col, 0.95)
        lowp = (1 - e / 30) if e < 30 else 0.0
        if lowp > 0 and not (DARK and DARK[0] <= t < DARK[1]): cv.rrect(60, 92, 960, 56, 28, None, 1, stroke=RED, lw=4, sa=lowp * (0.5 + 0.5 * math.sin(t * 9)))
        # round cards / chips
        for r in RD:
            if not (r["t0"] - 0.1 <= t < r["t1"]): continue
            busy = any(p["t0"] <= t < p["t1"] for p in PH) or any(g["t0"] <= t < g["t1"] for g in BIG)
            cd = float(r.get("card", 1.3)); big = (t < r["t0"] + cd) and not busy
            ba = win(t, r["t0"] - 0.1, r["t0"] + cd, 0.2, 0.3) if not busy else 0.0
            if ba > 0.01:
                k = 1 + 0.25 * (1 - eout((t - r["t0"]) / 0.25))
                WG.panel(cv, 110, 312, 860, 400, ba, stroke=YEL)
                if r.get("num"): cv.text(r["num"], 540, 412, int(110 * k), YEL, ba, maxw=800)
                cv.text(r["name"], 540, 542 if r.get("num") else 472, 84, WHITE, ba, maxw=800)
                if r.get("sub"): cv.text(r["sub"], 540, 642 if r.get("num") else 592, 54, "#C9D6EA", ba, bold=False, maxw=800)
            ca = (1 - ba) * win(t, r["t0"] + (0 if busy else cd - 0.2), r["t1"], 0.25, 0.2)
            if ca > 0.01:
                txt = "  ".join(x for x in (r.get("num", ""), r["name"], r.get("chip_sub", r.get("sub", ""))) if x)
                w = min(1000, cv.tw(txt, 50) + 70)
                cv.rrect(540 - w / 2, 168, w, 76, 38, "#0B1730", 0.85 * ca, stroke=YEL, lw=3, sa=ca)
                cv.text(txt, 540, 206, 50, WHITE, ca, maxw=940, shadow=False)
        # big number cards
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga <= 0.01: continue
            col = g.get("col", YEL); WG.panel(cv, 110, 312, 860, 400, ga, stroke=col)
            v = float(g["value"]) if g.get("static") else WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.1))
            txt = WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", ""))
            if g.get("emo"): WG.emo(cv, g["emo"], 540, 392, 80, ga)
            cv.text(txt, 540, 512 if g.get("emo") else 472, g.get("size", 150), col, ga, maxw=800)
            if g.get("sub"): cv.text(g["sub"], 540, 632, 56, WHITE, ga, maxw=820)
        if CHP:
            pa = win(t, CHP["t0"], CHP["t1"], 0.3, 0.3)
            for k, it in enumerate(CHP["items"]):
                ia = pa * seg(t, it["t0"] - 0.05, it["t0"] + 0.25)
                if ia <= 0.01: continue
                y = 362 + 140 * k; s_ = WG.pop_s(t, it["t0"])
                cv.rrect(160, y - 55, 760, 110, 55, "#0B1730", 0.9 * ia, stroke=SKY, lw=3, sa=ia)
                WG.emo(cv, it["emo"], 250, y, 70 * max(0.3, s_), ia); cv.text(it["text"], 320, y, 70, WHITE, ia, align="l", maxw=580)
        if CNT:
            na = win(t, CNT["t0"], CNT["t1"], 0.3, 0.3)
            if na > 0.01:
                WG.panel(cv, 60, 282, 960, 460, na, stroke=YEL)
                for k, r in enumerate(CNT["rows"]):
                    ra = na * seg(t, r["t0"] - 0.1, r["t0"] + 0.2)
                    if ra <= 0.01: continue
                    y = 372 + 140 * k; v = WG.roll(t, r["t0"], 0.0, float(r["value"]), 0.9)
                    WG.emo(cv, r["emo"], 140, y, 70, ra)
                    cv.text(WG.fmtn(v, r.get("dec", 0), "", r.get("suffix", "")), 210, y, 96, YEL, ra, align="l", maxw=420)
                    cv.text(r["label"], 1000, y, 64, WHITE, ra, align="r", maxw=380)
        for p in PH:                                                           # photo credit + label over the photo
            pa = win(t, p["t0"], p["t1"], 0.3, 0.3)
            if pa <= 0.01: continue
            cv.rrect(bx, by, bw, bh, 34, None, 1, stroke=YEL if p.get("label") else "#C9D6EA", lw=3, sa=0.7 * pa)
            if p.get("label"): WG.title(cv, p["label"], by + bh - 105, pa, 76)
            if p.get("credit"): cv.text("Photo: " + p["credit"], 540, by + bh - 30, 36, "#FFFFFF", 0.95 * pa, bold=False, maxw=920, minsz=36)
        if FIN:
            fa = seg(t, FIN["t0"], FIN["t0"] + 0.35) * (1 - seg(t, T.END - 0.6, T.END - 0.3))
            if fa > 0.01: WG.title(cv, FIN["text"], 512, fa, 100)
        draw_sources(cv, S, T, t, D.get("src_y", 1522)); draw_end(cv, S, T, t)

    def frame(t):
        dark = inwin(t, DARK, 0.6, 0.8) if DARK else 0.0
        hall = inwin(t, HALL, 0.3, 0.3) if HALL else 0.0
        cv = Canvas(ARENA0)
        if hall < 0.999: scene_ring(cv, t, dark)
        out = cv.out().astype(np.float32)
        if hall > 0.001:
            ch_ = Canvas(BG); scene_hall(ch_, t); out = out * (1 - hall) + ch_.out().astype(np.float32) * hall
        if dark > 0.001:                                                       # lights down, one cold spotlight on the canvas
            yy, xx = np.ogrid[0:H, 0:W]
            spot = np.exp(-(((xx - 330) / 360.0) ** 2 + ((yy - 1230) / 280.0) ** 2)).astype(np.float32)[..., None]
            out = out * (1 - dark * (0.78 - 0.6 * spot))
        if t >= END_AT - 0.2: out = out * (1 - 0.35 * seg(t, END_AT - 0.2, END_AT + 0.3))
        out = band_photo(out, t)
        cv2_ = Canvas(out); ui(cv2_, t, dark)
        out = paste_hook(cv2_.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c3 = Canvas(out); draw_subs(c3, CH, t, D.get("subs_y", 1640)); return np.clip(c3.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    quiet = DARK or (1e9, 1e9)
    ev = [(0.05, "ding", -20)] + [(r["t0"], "ding", -22) for r in RD if not (quiet[0] - 0.2 <= r["t0"] < quiet[1])]
    ev += [(h, "thud", -16) for h in HITS if not (quiet[0] <= h < quiet[1])] + [(h - 0.18, "whoosh", -30) for h in HITS if not (quiet[0] <= h < quiet[1])]
    ev += [(p["t0"], "whoosh", -30) for p in PH]
    ev += [(g["t0"], "roll", g["t0"] + 1.1) for g in BIG if not g.get("static")] + [(g["t0"] + 1.1, "ding", -28) for g in BIG if not g.get("static")]
    if CNT: ev += [(r["t0"], "roll", r["t0"] + 0.9) for r in CNT["rows"]]
    if CHP: ev += [(i["t0"], "pop", -24) for i in CHP["items"]]
    if TRAIN: ev += [(TRAIN[0] + k / 5.0 + 0.1, "thud", -30) for k in range(int((TRAIN[1] - TRAIN[0]) * 5))]
    if DOME:
        span = DOME[1] - DOME[0] - 0.6
        ev += [(DOME[0] + 0.2 + span * k / 10 + 0.56, "pop", -28) for k in range(10) if k != 6]
    M = S.setdefault("music", {})
    if M.get("flip"): M["_lift"] = A(M["flip"]); _custom_bed(S)
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
