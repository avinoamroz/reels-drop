"""pun_ladder (@noamroz__ wordplay): a stick-figure professor (wild white hair, chalkboard) climbs a ladder of puns on one
word ending. Each card: the big word with the changing part in yellow, its meaning, a drawn icon (a pile of stones or one big
tinted stone + an emoji), a chalkboard gloss, and a tiny source label. Counting cards stack stones in the centre; a "bury"
anchor drops the pile on the professor; a "scream" card makes him scream; the last card throws the stone and flies it back
to the frame-0 position so the loop is seamless. Made for einstein_stones; reusable for any "-stein / -berg / -man" ladder.

Text markup: *part* = yellow. "|" = line break on the chalkboard.
data keys:
  board0     chalkboard text at frame 0 (with the hook)
  cards      [{at, until?, word, mean?, stones?: n (counting) | tint: "#hex" (one big stone), emo?, board?, kind?}]
             kind: "title" (word only, big), "scream" (Frankenstein style), "throw" (the loop card: two birds, one stone)
  bury       anchor: the counting pile falls on the professor
  unbury     anchor: the pile rolls away, the professor pops back up
  cta        {at, text}  small pill near the end ("Tag your favorite *-stein*")
  tail       seconds after the last word (default 1.4)
"""
import os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
import common as CMOD
from common import *

CHALK = "#EDEFE8"; BOARD = "#1E3A2E"; WOOD = "#9A6A3A"; STONE = "#9AA3AE"
PX, PY, PS = 230, 1400, 2.2            # professor feet x, y, scale
SC0 = (540, 800, 120)                  # the hook stone: x, y, r (frame 0 and the loop target)


def mk(txt, base=WHITE):
    out = []
    for tok in re.split(r"(\*[^*]+\*)", txt):
        if tok: out.append((tok[1:-1], YEL) if tok.startswith("*") else (tok, base))
    return out


def rich_line(cv, txt, cx, cy, sz, a, maxw=980, base=WHITE, bold=True):
    ps = mk(txt, base)
    tot = lambda z: sum(cv.tw(p, z, bold) for p, _ in ps)
    while tot(sz) > maxw and sz > 64: sz -= 2
    x = cx - tot(sz) / 2
    for p, c in ps:
        cv.text(p, x, cy, sz, c, a, bold=bold, align="l", maxw=2000, shadow=True, minsz=36); x += cv.tw(p, sz, bold)


def shade(h, f):
    r, g, b = rgb(h); return "#%02X%02X%02X" % tuple(int(max(0, min(1, v * f)) * 255) for v in (r, g, b))


_SH = {}
def shape(seed, n=11):
    if seed not in _SH:
        rng = np.random.default_rng(seed + 7)
        _SH[seed] = [(2 * math.pi * k / n + rng.uniform(-0.12, 0.12), 0.84 + rng.uniform(0, 0.2)) for k in range(n)]
    return _SH[seed]


def stone(cv, x, y, r, col=STONE, a=1.0, seed=0, rot=0.0, facets=False, specks=False):
    """a rounded irregular stone (flatter than tall), outline, highlight"""
    if a <= 0.01: return
    pts = [(x + r * q * math.cos(th + rot) * 1.12, y + r * q * math.sin(th + rot) * 0.82) for th, q in shape(seed)]
    c = cv.c; n = len(pts); c.new_path()
    mids = [((pts[i][0] + pts[(i + 1) % n][0]) / 2, (pts[i][1] + pts[(i + 1) % n][1]) / 2) for i in range(n)]
    c.move_to(*mids[-1])
    for i in range(n): c.curve_to(pts[i][0], pts[i][1], pts[i][0], pts[i][1], *mids[i])
    c.close_path(); cv.col(col, a); c.fill_preserve(); cv.col(shade(col, 0.55), a); c.set_line_width(max(3, r * 0.05)); c.stroke(); c.new_path()
    c.save(); c.translate(x - r * 0.35, y - r * 0.32); c.scale(r * 0.32, r * 0.16); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    cv.col("#FFFFFF", 0.32 * a); c.fill(); c.new_path()
    if facets:
        for (ax, ay), (bx, by) in (((-0.5, -0.1), (0, -0.55)), ((0, -0.55), (0.55, -0.05)), ((-0.5, -0.1), (0, 0.5)), ((0.55, -0.05), (0, 0.5)), ((0, -0.55), (0, 0.5))):
            cv.line([(x + ax * r, y + ay * r), (x + bx * r, y + by * r)], "#FFFFFF", max(2, r * 0.025), 0.55 * a)
    if specks:
        rng = np.random.default_rng(seed + 3)
        for _ in range(14):
            u, v = rng.uniform(-0.7, 0.7), rng.uniform(-0.45, 0.45); cv.circ(x + u * r, y + v * r, r * 0.04, "#FFF6E0", 0.85 * a)


def hair(cv, hx, hy, r, a, wild=1.0, t=0.0):
    """Einstein style wild white hair + moustache around a head at (hx, hy) radius r"""
    for k in range(13):
        th = math.pi * (0.95 + 1.1 * k / 12)
        L = r * (0.55 + 0.35 * ((k * 37) % 5) / 4) * wild * (1 + 0.06 * math.sin(t * 9 + k))
        x0, y0 = hx + r * math.cos(th), hy + r * math.sin(th)
        x1, y1 = hx + (r + L) * math.cos(th + 0.12), hy + (r + L) * math.sin(th + 0.12)
        cv.line([(x0, y0), ((x0 + x1) / 2 + 4, (y0 + y1) / 2 - 3), (x1, y1)], "#FFFFFF", r * 0.16, a)
    for s_ in (-1, 1):
        cv.line([(hx + s_ * r * 0.95, hy - r * 0.1), (hx + s_ * r * 1.45, hy + r * 0.25)], "#FFFFFF", r * 0.16, a)
        cv.line([(hx + s_ * r * 0.05, hy + r * 0.42), (hx + s_ * r * 0.5, hy + r * 0.30)], "#FFFFFF", r * 0.14, a)


def professor(cv, x, y, a, arms="down", scream=0.0, t=0.0, bob=0.0):
    jit = (math.sin(t * 61) * 6, math.cos(t * 53) * 4) if scream > 0.5 else (0, 0)
    x += jit[0]; y += jit[1] - bob
    head = figure(cv, x, y, PS, WHITE, badge=False, arms=("up" if scream > 0.5 else arms), a=a, flip=1)
    hx, hy = head; r = 22 * PS
    hair(cv, hx, hy, r, a, 1.0 + 0.7 * scream, t)
    if scream > 0.5:
        c = cv.c; c.save(); c.translate(hx, hy + r * 0.5); c.scale(r * 0.26, r * 0.36); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
        cv.col("#0B1730", a); c.fill(); c.new_path()
        for s_ in (-1, 1): cv.circ(hx + s_ * 9 * PS, hy - 4 * PS, 5 * PS, "#FFFFFF", a)
    else:
        cv.c.new_path(); cv.c.arc(hx, hy + r * 0.38, r * 0.32, math.pi * 0.2, math.pi * 0.8); cv.col("#FFFFFF", a); cv.c.set_line_width(r * 0.1); cv.c.stroke(); cv.c.new_path()
    return head


def chalkboard(cv, txt, a, t_in, t):
    x, y, w, h = 440, 1010, 600, 330
    cv.rrect(x - 14, y - 14, w + 28, h + 28, 20, WOOD, a); cv.rrect(x, y, w, h, 12, BOARD, a)
    cv.line([(x + 120, y + h + 14), (x + 120, y + h + 70)], WOOD, 12, a); cv.line([(x + w - 120, y + h + 14), (x + w - 120, y + h + 70)], WOOD, 12, a)
    if not txt: return
    ta = a * (seg(t, t_in, t_in + 0.25) if t_in > 0 else 1.0)
    ls = txt.split("|"); lh = 92; y0 = y + h / 2 - lh * (len(ls) - 1) / 2
    for i, l in enumerate(ls): rich_line(cv, l, x + w / 2, y0 + i * lh, 70, ta, w - 50, CHALK, bold=False)


def emo_at(cv, ch, x, y, sz, a): WG.emo(cv, ch, x, y, sz, a)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.4)); A = T.at
    C = []
    for c in D["cards"]:
        C.append(dict(c, t0=A(c["at"])))
    for k, c in enumerate(C): c["t1"] = A(c["until"]) if c.get("until") else (C[k + 1]["t0"] if k + 1 < len(C) else T.END)
    t_bury = A(D["bury"]) if D.get("bury") else 1e9; t_unb = A(D["unbury"]) if D.get("unbury") else 1e9
    CTA = D.get("cta") or {}; t_cta = A(CTA["at"]) if CTA else 1e9
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"], size=128); HY = D.get("hook_y", 360)
    t_hook_end = T.E(0)
    count_cards = [c for c in C if c.get("stones")]
    throw = next((c for c in C if c.get("kind") == "throw"), None)
    END = T.END

    def cur(t):
        for c in C:
            if c["t0"] - 0.05 <= t < c["t1"]: return c
        return None

    def n_stones(t):
        n = 1
        for c in count_cards:
            if t >= c["t0"]: n = c["stones"]
        return n

    def row_pos(n, i):
        r = 120 if n <= 2 else (105 if n == 3 else 92)
        gap = 2 * r * 1.15; x0 = 540 - gap * (n - 1) / 2
        return x0 + i * gap, 800, r

    PILE = [(150, 1335, 82), (270, 1340, 86), (390, 1338, 80), (205, 1225, 80), (330, 1228, 82), (268, 1120, 78)]

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 10))
        c = cur(t); kind = (c or {}).get("kind")
        scream = 1.0 if kind == "scream" and t > c["t0"] + 0.15 else 0.0
        # --- professor + chalkboard (always, except while buried)
        board_txt, board_t = D.get("board0", ""), 0.0
        for cc in C:
            if cc.get("board") and t >= cc["t0"]: board_txt, board_t = cc["board"], cc["t0"]
        chalkboard(cv, board_txt, 1.0, board_t, t)
        buried = t_bury <= t < t_unb
        pop = back(seg(t, t_unb, t_unb + 0.45, lambda u: u)) if t >= t_unb else 1.0
        if not buried:
            arms = "point" if c and kind not in ("throw",) and c.get("tint") else "down"
            if throw and t >= throw["t0"]: arms = "up" if t < A(throw.get("hit", throw["at"])) else "point"
            yb = PY + (1 - pop) * 260 if t >= t_unb else PY
            professor(cv, PX, yb, 1.0, arms, scream, t, bob=6 * math.sin(t * 3.0))
        # --- counting stones (centre row), then the bury
        if t < t_unb + 0.6:
            n = n_stones(t)
            tc = max([cc["t0"] for cc in count_cards if cc["t0"] <= t] or [-9]); n_prev = n_stones(tc - 0.01) if tc > 0 else n
            uu = ease(seg(t, tc, tc + 0.3, lambda v: v))
            for i in range(n):
                x, y, r = row_pos(n, i)
                if i < n_prev:
                    x0_, _, r0_ = row_pos(n_prev, i) if n_prev > 1 else SC0; x, r = x0_ + (x - x0_) * uu, r0_ + (r - r0_) * uu
                born = 0.0 if i == 0 else next(cc["t0"] for cc in count_cards if cc["stones"] > i)
                drop = eout(seg(t, born, born + 0.35, lambda u: u)) if i > 0 else 1.0
                bounce = abs(math.sin(min(1, max(0, (t - born - 0.35) / 0.3)) * math.pi)) * 18 * (1 - min(1, max(0, (t - born - 0.35) / 0.3))) if i > 0 else 0
                yy = -200 + (y + 200) * drop - bounce
                if t >= t_bury:
                    px, py, pr = PILE[i]; u = ease(seg(t, t_bury, t_bury + 0.5, lambda v: v))
                    x, yy, r = x + (px - x) * u, yy + (py - yy) * u, r + (pr - r) * u
                if t >= t_unb:
                    u = ease(seg(t, t_unb, t_unb + 0.6, lambda v: v)); x += 900 * u; yy -= 80 * math.sin(u * math.pi)
                stone(cv, x, yy, r, STONE, 1.0, seed=i, rot=0.3 * (t - t_bury) if t >= t_bury else 0)
            if t >= t_bury:
                for j in (4, 5):
                    px, py, pr = PILE[j]; u = eout(seg(t, t_bury + 0.25 + 0.12 * (j - 4), t_bury + 0.6 + 0.12 * (j - 4), lambda v: v))
                    yy = -200 + (py + 200) * u; xx = px
                    if t >= t_unb:
                        w_ = ease(seg(t, t_unb, t_unb + 0.6, lambda v: v)); xx += 900 * w_
                    stone(cv, xx, yy, pr, STONE, 1.0, seed=j)
                if buried:
                    hair(cv, 268, 1020, 22 * PS * 0.9, 1.0, 1.4, t)
                    wv = math.sin(t * 12) * 20
                    cv.line([(410, 1250), (470, 1150 + wv)], WHITE, 7 * PS, 1.0)
        # --- the card: big word + meaning + icon
        if c:
            a = win(t, c["t0"] - 0.05, c["t1"], 0.12, 0.15)
            sp = 0.9 + 0.1 * back(seg(t, c["t0"] - 0.05, c["t0"] + 0.3, lambda u: u))
            if kind == "title":
                rich_line(cv, c["word"], 540, 420, int(110 * sp), a)
            else:
                rich_line(cv, c["word"], 540, 330, int(150 * sp), a)
                if c.get("mean"): rich_line(cv, c["mean"], 540, 480, 76, a * seg(t, c["t0"] + 0.25, c["t0"] + 0.5))
            if c.get("tint"):
                ia = a * seg(t, c["t0"], c["t0"] + 0.2); s_ = back(seg(t, c["t0"], c["t0"] + 0.4, lambda u: u))
                wob = 0.08 * math.sin((t - c["t0"]) * 5)
                fr = kind == "scream"
                stone(cv, 540 + (math.sin(t * 47) * 5 if fr else 0), 790, 150 * s_, c["tint"], ia, seed=11, rot=wob,
                      facets=c.get("facets", False), specks=c.get("specks", False))
                if fr and ia > 0.01:   # bolts, stitches, eyes
                    for s2 in (-1, 1): cv.rrect(540 + s2 * 170 - 22, 770, 44, 40, 8, "#B8C0CA", ia, stroke="#5B6470", lw=3)
                    cv.line([(470, 720), (610, 720)], "#2B3A20", 6, ia)
                    for k in range(5): cv.line([(480 + k * 30, 706), (480 + k * 30, 734)], "#2B3A20", 5, ia)
                    for s2 in (-1, 1): cv.circ(540 + s2 * 45, 780, 16, "#FFFFFF", ia); cv.circ(540 + s2 * 45, 782, 7, "#0B1730", ia)
                    cv.line([(495, 845), (585, 845)], "#2B3A20", 7, ia)
                if c.get("sparks") and ia > 0.01:
                    for k in range(8):
                        th = k * math.pi / 4 + t * 2; fl = 0.5 + 0.5 * math.sin(t * 23 + k * 1.7)
                        r0 = 175; r1 = 175 + 60 * fl
                        cv.line([(540 + r0 * math.cos(th), 790 + r0 * math.sin(th) * 0.8), (540 + r1 * math.cos(th), 790 + r1 * math.sin(th) * 0.8)], YEL, 8, ia * fl)
                if c.get("emo"): emo_at(cv, c["emo"], 540 + c.get("emo_dx", 175), 790 + c.get("emo_dy", -120), 120, ia)
            if kind == "throw":
                th0 = c["t0"]; hit = A(c.get("hit", c["at"]))
                for k, bx in enumerate((380, 700)):
                    by = 640 + 20 * math.sin(t * 6 + k * 2)
                    if t < hit + 0.05: emo_at(cv, "\U0001F426", bx, by, 120, a)
                    else:
                        u = seg(t, hit, hit + 0.8, lambda v: v)
                        emo_at(cv, "\U0001F426", bx + (k * 2 - 1) * 260 * u, by - 420 * u, 120, a * (1 - u))
                        emo_at(cv, "\U0001F4AB", bx, by - 70, 100, a * (1 - u))
                # the stone: from the professor's hand, through both birds, then back to the frame-0 spot
                hand = (PX + 70, PY - 290)
                if t < hit:
                    u = seg(t, th0 + 0.2, hit, lambda v: v); pts = beam_path(hand, (700, 640), bend=-120)
                    x, y = pts[int(u * (len(pts) - 1))]; stone(cv, x, y, 50, STONE, a if t > th0 + 0.2 else 0, seed=0, rot=t * 8)
                else:
                    u = ease(seg(t, hit + 0.15, END - 0.55, lambda v: v))
                    x = 700 + (SC0[0] - 700) * u; y = 640 + (SC0[1] - 640) * u - 80 * math.sin(u * math.pi); r = 50 + (SC0[2] - 50) * u
                    stone(cv, x, y, r, STONE, 1.0, seed=0, rot=(1 - u) * t * 8)
        # --- sources, cta, hook, subtitles
        draw_sources(cv, S, T, t)
        if t >= t_cta:
            ca = seg(t, t_cta, t_cta + 0.3) * (1 - seg(t, END - 0.6, END - 0.45))
            txt = CTA.get("text", ""); tw = min(960, cv.tw(txt.replace("*", ""), 60) + 90)
            cv.rrect(540 - tw / 2, 1426, tw, 110, 55, SKY, ca); rich_line(cv, txt, 540, 1481, 60, ca, 900)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, t_hook_end, HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    CMOD.cover = lambda S_, final, out_jpg: my_cover(S_, out_jpg)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    for c in C:
        ev.append((c["t0"], "pop", -22))
        if c.get("stones"): ev.append((c["t0"] + 0.33, "thud", -16))
        if c.get("kind") == "scream": ev += [(c["t0"] + 0.15, "buzz", -20), (c["t0"] + 0.4, "buzz", -22)]
        if c.get("kind") == "throw": ev += [(c["t0"] + 0.2, "whoosh", -24), (A(c.get("hit", c["at"])), "pop", -16), (A(c.get("hit", c["at"])) + 0.2, "whoosh", -24)]
    if t_bury < 1e8: ev += [(t_bury, "whoosh", -22), (t_bury + 0.5, "thud", -14), (t_bury + 0.7, "thud", -16)]
    if t_unb < 1e8: ev += [(t_unb, "whoosh", -24), (t_unb + 0.3, "pop", -18)]
    if t_cta < 1e8: ev.append((t_cta, "ding", -24))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)


def my_cover(S, out_jpg):
    """brand gradient, the hook huge, one big stone, the professor peeking in (no real face)"""
    Cv = S.get("cover") or {}
    cv = Canvas(BG.copy())
    stone(cv, 540, 1250, 230, STONE, 1.0, seed=0)
    professor(cv, 150, 1660, 1.0, "point")
    out = cv.out().astype(np.float32)
    img = HK.make(wrap_hook(Cv.get("text") or S.get("hook", "")), size=int(Cv.get("size", 150)), maxw=960)
    out = HK.paste(out, img, 540, 640, 1.0, 1.0)
    cv2.imwrite(out_jpg, np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
