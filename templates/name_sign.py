"""name_sign (story): famous Israeli street names that are surnames, and the hidden meaning of each surname.
Frame 0: a street signpost (one arrow per name) under the hook. Each beat: a big blue Israeli street sign (Hebrew on top,
English below) drops in across the top, and a full-width drawn scene acts out the surname's meaning. Beats change with an
eased slide (the old scene slides out left while the new one slides in from the right, both fully drawn, so no frame is
half faded or empty); the old sign lifts away as the new one drops. The loop line brings the signpost back: arrows sway
and light up one by one, a yellow "Your name?" arrow pops in, the hook returns, and the last 0.5 s dissolves into frame 0.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  theme    "noam" (default, @noamroz__) | "xp" (@expressenglishcom: brand navy #163261, sky signs)
  post     {until, back_at, names: [...], ask: "Your name?", ask_at}  signpost from frame 0 until `until` (default end of
           line 0); back at `back_at`; `ask` = extra yellow arrow that pops at `ask_at` (default back_at)
  hook_back  anchor where the hook comes back over the post (default post.back_at)
  beats    [{at, he, en, tag, scene, swap_at, extra}]   a beat lasts until the next beat's `at` (the last until post.back_at)
           scene     shield | green | falcon | cape | island
           green: swap_at = anchor where the name badge flips; extra {old, new, year}
           cape:  extra {chips: [[text, anchor], ...]} (the last chip is yellow)
           island: extra {town}
  he_lines [{at, until, text}]   small Hebrew line at the bottom (Express rule: Hebrew only as a small bottom line)
  sign_y default 310;  scene band y 470 to 1340;  title_y default 1405;  src_y default 1500;  subs_y default 1640;  he_y 1745
  tail     seconds after the last word (default 0.6)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

THEMES = {
    "noam": dict(sign="#1E5BB0", inner="#F5F5F7", pole="#8FA3C0", bg=None),
    "xp": dict(sign="#2B8FD0", inner="#F5F5F7", pole="#9FC3E3", bg=("#163261", "#1E4583")),
}
INK = "#0E2250"; SOFT = "#C9D6EA"; WOOD = "#8A5A3C"; LEAF = "#3FBF7F"; SAND = "#F2D58A"
BT, BB = 470, 1340                                       # scene band (edge to edge)
SLIDE = 0.6                                             # slide transition seconds


def mix(h1, h2, k):
    a, b = rgb(h1), rgb(h2); k = min(1.0, max(0.0, k))
    return "#%02X%02X%02X" % tuple(int(255 * (x + (y - x) * k)) for x, y in zip(a, b))


def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def ell(cv, x, y, rx, ry, col, a=1.0, ang=0.0):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang); c.scale(rx, ry); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    cv.col(col, a); c.fill(); c.new_path()


def street_sign(cv, TH, cx, cy, he, en, tag, a=1.0, ang=0.0, w=1010, h=290):
    if a <= 0.01: return
    c = cv.c; c.save(); c.translate(cx, cy); c.rotate(ang)
    cv.rrect(-w / 2 + 8, -h / 2 + 12, w, h, 30, "#000000", 0.32 * a)
    cv.rrect(-w / 2, -h / 2, w, h, 30, TH["sign"], a)
    cv.rrect(-w / 2 + 18, -h / 2 + 18, w - 36, h - 36, 20, None, a, stroke=TH["inner"], lw=7, sa=a)
    if he: hebrew(cv, he, 0, -62, 96, (245, 245, 247), a)
    cv.text(en, 0, 64, 96, WHITE, a, maxw=w - (330 if tag else 140), shadow=False)
    if tag: cv.text(tag, w / 2 - 92, 64, 44, SOFT, a, shadow=False, bold=False)
    c.restore()


def arrow_sign(cv, TH, cx, cy, txt, right, a=1.0, lit=0.0, w=560, h=118, ask=False):
    if a <= 0.01: return
    x0 = cx if right else cx - w; tip = 52
    pts = ([(x0, cy - h / 2), (x0 + w - tip, cy - h / 2), (x0 + w, cy), (x0 + w - tip, cy + h / 2), (x0, cy + h / 2)] if right else
           [(x0 + tip, cy - h / 2), (x0 + w, cy - h / 2), (x0 + w, cy + h / 2), (x0 + tip, cy + h / 2), (x0, cy)])
    poly(cv, [(x + 6, y + 9) for x, y in pts], "#000000", 0.3 * a)
    if lit > 0.01: poly(cv, pts, None, a, stroke=YEL, lw=6 + 16 * lit)                  # yellow glow when lit
    poly(cv, pts, YEL if ask else TH["sign"], a, stroke=YEL if lit > 0.3 else TH["inner"], lw=6)
    cv.text(txt, x0 + w / 2 + (-tip / 2 if right else tip / 2), cy, 64, INK if ask else mix(WHITE, YEL, lit), a, maxw=w - tip - 50, shadow=False)


# ------------------------------------------------------------------ scenes: (cv, cx, u, B, t), cx = scene centre x
def sc_shield(cv, cx, u, B, t):
    gy = BB - 34; hx = cx - 190; hw, hh = 440, 600; top = gy - hh
    cv.rrect(cx - 700, gy, 1400, 18, 4, "#5C6B85")
    poly(cv, [(hx - hw / 2 - 20, top + 4), (hx, top - 200), (hx + hw / 2 + 20, top + 4)], "#B5523B")
    cv.rrect(hx - hw / 2, top, hw, hh, 6, "#F1E3C8")
    for k in range(5): cv.rrect(hx - hw / 2, top + 14 + k * 140, hw, 14, 2, WOOD)
    for xx in (hx - hw / 2, hx - 7, hx + hw / 2 - 14): cv.rrect(xx, top, 14, hh, 2, WOOD)
    for row in range(3):
        for s_ in (-1, 1): cv.rrect(hx + s_ * 108 - 48, top + 52 + row * 140, 96, 84, 10, "#7FB6E0", 1, stroke=WOOD, lw=5, sa=1)
    cv.rrect(hx - 62, gy - 170, 124, 170, 16, WOOD)
    bx, by = hx + hw / 2, top + 150
    cv.line([(bx, by), (bx + 300, by)], "#2B2B2B", 12); cv.line([(bx, by + 70), (bx + 90, by)], "#2B2B2B", 8)
    p = seg(u, 0.15, 0.75, eout); sw = math.sin(max(0, u - 0.15) * 3.0) * 0.22 * math.exp(-max(0, u - 0.15) * 0.9)
    c = cv.c; c.save(); c.translate(bx + 230, by); c.rotate(sw * p); c.scale(0.4 + 0.6 * p, 0.4 + 0.6 * p)
    a_ = min(1.0, p * 3)
    cv.line([(-70, 0), (-70, 64)], "#2B2B2B", 6, a_); cv.line([(70, 0), (70, 64)], "#2B2B2B", 6, a_)
    sh = [(-140, 64), (140, 64), (140, 220), (0, 360), (-140, 220)]
    poly(cv, [(x + 7, y + 10) for x, y in sh], "#000000", 0.3 * a_)
    poly(cv, sh, RED, a_, stroke=YEL, lw=9)
    c.restore()
    g = seg(u, 1.0, 1.6)
    if g > 0: cv.circ(bx + 230, by + 210, 215 + 12 * math.sin(u * 3.5), None, 1, stroke=YEL, lw=6 * g)


def sc_green(cv, cx, u, B, t):
    X = B.get("extra", {}); sw = B.get("_swap", 1e9)
    f = seg(u, 0.1, 1.0)                                              # field rises once, then stays
    for k, (col, h0) in enumerate((("#2E9E5B", 520), ("#3FBF7F", 300))):
        top = BB - h0 * f
        if f > 0.01:
            pts = [(cx - 560, BB)] + [(cx - 560 + x, top + 22 * math.sin(x / 160 + k * 1.7)) for x in range(0, 1141, 40)] + [(cx + 580, BB)]
            poly(cv, pts, col, 1.0)
    p = seg(u, 0.25, 0.8, lambda x: back(x, 1.3))                     # badge scales in once
    ph = seg(t, sw - 0.25, sw + 0.25) if sw < 1e8 else 0.0            # one clean flip at the swap
    sx = abs(math.cos(math.pi * ph)); new = ph >= 0.5
    if p <= 0.01: return
    c = cv.c; c.save(); c.translate(cx, 830); c.scale(max(0.02, sx) * p, p)
    cv.rrect(-440, -200, 880, 400, 36, "#000000", 0.25)
    cv.rrect(-450, -210, 900, 400, 36, "#FFFFFF", 1, stroke=INK, lw=7, sa=1)
    cv.rrect(-450, -210, 900, 104, 36, RED if not new else SKY, 1); cv.rrect(-450, -150, 900, 44, 0, RED if not new else SKY, 1)
    cv.text("HELLO, MY NAME IS", 0, -158, 52, WHITE, 1, shadow=False)
    if not new: cv.text(X.get("old", "GRÜN"), 0, 50, 180, "#2E9E5B", 1, shadow=False, maxw=820)
    else: cv.text(X.get("new", "BEN GURION"), 0, 50, 128, INK, 1, shadow=False, maxw=820)
    c.restore()
    if X.get("year") and sw < 1e8 and t > sw + 0.3:
        q = WG.pop_s(t, sw + 0.3); cv.rrect(cx - 160 * q, 1120 - 60 * q, 320 * q, 120 * q, 60 * q, YEL, 1)
        cv.text(X["year"], cx, 1120, 84 * q, INK, 1, shadow=False, minsz=8)


def falcon(cv, x, y, s, ang, fold):
    """peregrine in a stoop: pointed swept wings, hooked beak, dark helmet and moustache stripe. Local +x = head."""
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang); c.scale(s, s)
    SL, DK, PALE = "#4A5568", "#262C38", "#EFE8D8"
    sp = 70 + 50 * (1 - fold)                                         # wing spread (folds as it dives)
    poly(cv, [(70, -10), (-60, -sp - 30), (-300, -sp - 10), (-90, -14)], DK)            # far wing
    poly(cv, [(-120, -6), (-250, -26), (-262, 0), (-250, 26), (-120, 6)], SL)            # tail
    for k in range(3): cv.line([(-150 - k * 32, -14 + k * 2), (-150 - k * 32, 14 - k * 2)], DK, 5)
    ell(cv, 10, 0, 160, 50, SL)                                        # body
    ell(cv, 20, 18, 120, 30, PALE)                                     # pale belly
    for k in range(6): cv.line([(-60 + k * 30, 14), (-48 + k * 30, 30)], "#8A8070", 4)  # bars
    ell(cv, 160, -4, 54, 48, DK)                                       # dark helmet
    ell(cv, 170, 18, 34, 22, PALE)                                     # white cheek
    cv.line([(168, 2), (158, 46)], "#111111", 15)                     # moustache stripe
    cv.circ(178, -12, 15, YEL); cv.circ(180, -12, 7, "#111111")       # eye ring + eye
    poly(cv, [(204, -22), (218, -24), (222, -6), (206, -4)], YEL)      # cere
    poly(cv, [(216, -24), (244, -16), (252, 4), (242, 18), (238, 4), (220, -2)], "#2E3440")  # hooked beak
    poly(cv, [(80, -6), (-40, sp * 1.05 + 40), (-330, sp * 0.9 + 40), (-100, 12)], SL)  # near wing (long, pointed)
    poly(cv, [(80, -6), (-40, sp * 1.05 + 40), (-120, sp * 0.95 + 36), (40, 4)], DK)     # dark leading edge
    c.restore()


def sc_falcon(cv, cx, u, B, t):
    for k in range(3):
        x = cx - 380 + k * 380 - u * 30; y = BT + 120 + (k % 2) * 160
        for dx, r in ((-70, 60), (0, 84), (75, 64)): cv.circ(x + dx, y, r, "#FFFFFF", 0.12)
    cv.circ(cx + 380, BT + 150, 90, YEL, 0.9)
    p = seg(u, 0.1, 1.9)                                              # one eased dive into the centre, then hover
    x = cx - 300 + 290 * p; y = BT + 170 + (440 + 12 * math.sin(u * 2.4) * p) * p
    s = 0.62 + 1.0 * p; ang = 0.95 * (1 - p) + 0.12 * p; fold = 1 - p
    if p < 0.95:
        for k in range(4):                                            # speed lines behind the dive
            ox = -(220 + 60 * k) * s; oy = (k - 1.5) * 34 * s
            ca, sa_ = math.cos(ang), math.sin(ang)
            p0 = (x + ox * ca - oy * sa_, y + ox * sa_ + oy * ca); p1 = (p0[0] - 140 * ca, p0[1] - 140 * sa_)
            cv.line([p0, p1], "#FFFFFF", 6, 0.35 * (1 - p))
    falcon(cv, x, y, s, ang, fold)


def chaplain(cv, fx, gy, u, a=1.0):
    sway = math.sin(u * 1.8) * 10
    top = gy - 640
    poly(cv, [(fx - 80, top + 150), (fx + 80, top + 150), (fx + 150, gy), (fx - 150, gy)], "#1E2230", a)   # cassock
    poly(cv, [(fx - 95, top + 140), (fx + 95, top + 140), (fx + 210 + sway, gy - 120), (fx - 210 + sway, gy - 120)], "#7B2D8E", a)  # cape
    cv.rrect(fx - 214 + sway, gy - 132, 428, 18, 6, YEL, a)
    poly(cv, [(fx - 95, top + 140), (fx + 95, top + 140), (fx + 60, top + 230), (fx - 60, top + 230)], "#5A1F6A", a)
    cv.rrect(fx - 70, gy - 22, 60, 22, 10, "#111111", a); cv.rrect(fx + 10, gy - 22, 60, 22, 10, "#111111", a)
    cv.rrect(fx - 36, top + 118, 72, 34, 8, "#FFFFFF", a)            # clerical collar
    cv.circ(fx, top + 60, 70, "#E9C29A", a)                           # head
    c = cv.c; c.save(); c.new_path(); c.arc(fx, top + 54, 72, math.pi * 1.05, math.pi * 1.95); c.close_path(); cv.col("#3A2A1E", a); c.fill(); c.restore()
    cv.circ(fx - 24, top + 64, 7, "#111111", a); cv.circ(fx + 24, top + 64, 7, "#111111", a)
    cv.line([(fx - 18, top + 92), (fx, top + 100), (fx + 18, top + 92)], "#8A4B3A", 5, a)
    cv.rrect(fx - 70, top + 300, 140, 104, 10, "#3B2A1A", a, stroke=YEL, lw=4, sa=a)   # book
    cv.circ(fx - 70, top + 352, 22, "#E9C29A", a); cv.circ(fx + 70, top + 352, 22, "#E9C29A", a)


def sc_cape(cv, cx, u, B, t):
    p = seg(u, 0.05, 0.6, lambda x: back(x, 1.2))
    if p > 0.01:
        c = cv.c; c.save(); c.translate(cx - 250, BB - 30); c.scale(p, p); c.translate(-(cx - 250), -(BB - 30))
        chaplain(cv, cx - 250, BB - 30, u); c.restore()
    chips = (B.get("extra") or {}).get("chips", [])
    for k, ch in enumerate(chips):
        t0 = ch[2]
        if t < t0: continue
        q = WG.pop_s(t, t0); hl = k == len(chips) - 1; y = 640 + k * 190; x = cx + 265
        if k: cv.line([(x, y - 140), (x, y - 62)], SOFT, 6); poly(cv, [(x - 16, y - 76), (x + 16, y - 76), (x, y - 56)], SOFT)
        c = cv.c; c.save(); c.translate(x, y); c.scale(q, q)
        cv.rrect(-245, -60, 490, 120, 60, YEL if hl else "#0B1730", 0.96, stroke=YEL, lw=4, sa=1)
        cv.text(ch[0], 0, 0, 56, INK if hl else WHITE, 1, shadow=False, maxw=440)
        c.restore()


def sc_island(cv, cx, u, B, t):
    sea = 1040
    cv.circ(cx - 360, BT + 150, 80, YEL, 0.9)
    r = seg(u, 0.15, 1.3); iy = sea + 10 + 170 * (1 - r)
    c = cv.c; c.save(); c.new_path(); c.rectangle(cx - 600, BT, 1200, sea + 12 - BT); c.clip()
    c.save(); c.translate(cx, iy); c.scale(470, 180); c.new_path(); c.arc(0, 0, 1, math.pi, 2 * math.pi); c.restore()
    cv.col(SAND); c.fill()
    for dx, hh in ((-250, 150), (-140, 200), (20, 170), (130, 130)):
        hb = iy - 70
        cv.rrect(cx + dx - 52, hb - hh + 80, 104, hh, 6, "#F5F5F7")
        poly(cv, [(cx + dx - 64, hb - hh + 84), (cx + dx, hb - hh + 20), (cx + dx + 64, hb - hh + 84)], "#B5523B")
        cv.rrect(cx + dx - 18, hb - hh + 120, 36, 40, 6, "#7FB6E0")
    px = cx + 290
    cv.line([(px, iy - 40), (px - 20, iy - 360)], WOOD, 26)
    for ang in (-2.7, -2.1, -1.3, -0.6, 0.15):
        tip = (px - 20 + 190 * math.cos(ang + 0.04 * math.sin(u * 2)), iy - 360 + 110 * math.sin(ang) + 50)
        cv.line([(px - 20, iy - 360), tip], LEAF, 30)
    c.restore()
    cv.rrect(cx - 600, sea, 1200, BB - sea, 0, "#0F4C85")
    for k in range(6):
        y = sea + 26 + k * 46; pts = [(x, y + 10 * math.sin(x / 80 + u * 2.4 + k)) for x in range(int(cx - 600), int(cx + 620), 40)]
        cv.line(pts, "#1A6FB3" if k % 2 else "#2A86CC", 8, 0.85)
    town = (B.get("extra") or {}).get("town")
    if town and u > 1.3:
        q = WG.pop_s(t, B["_s0"] + 1.3); w = 380
        cv.rrect(cx + 120 - w / 2 * q, BT + 110 - 55 * q, w * q, 110 * q, 55 * q, YEL, 1)
        cv.text(town, cx + 120, BT + 110, 72 * q, INK, 1, shadow=False, minsz=8)


SCENES = dict(shield=sc_shield, green=sc_green, falcon=sc_falcon, cape=sc_cape, island=sc_island)


def run(S):
    D = S["data"]; work = workdir(S)
    TH = THEMES[D.get("theme", "noam")]
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.6)); A = T.at
    SY = D.get("sign_y", 310)
    PO = D.get("post") or {}; P_UNTIL = A(PO.get("until"), T.E(0)); P_BACK = A(PO.get("back_at"), T.END + 9)
    H_BACK = A(D.get("hook_back"), P_BACK); ASK = PO.get("ask"); ASK_AT = A(PO.get("ask_at"), P_BACK)
    NAMES = PO.get("names") or [b["en"].title() for b in D["beats"]]
    BTS = []
    for b in D["beats"]:
        b = dict(b); b["t0"] = A(b["at"]); b["_s0"] = b["t0"] - SLIDE / 2
        b["_swap"] = A(b["swap_at"]) if b.get("swap_at") else 1e9
        ex = dict(b.get("extra") or {})
        if ex.get("chips"): ex["chips"] = [[c[0], c[1], A(c[1])] for c in ex["chips"]]
        b["extra"] = ex; BTS.append(b)
    for k, b in enumerate(BTS): b["t1"] = BTS[k + 1]["t0"] if k + 1 < len(BTS) else P_BACK
    HEL = [dict(h, t0=A(h["at"]), t1=A(h.get("until"), T.END - 0.4)) for h in D.get("he_lines", [])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    if TH["bg"]:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        c0 = np.array(rgb(TH["bg"][0])[::-1], np.float32) * 255; c1 = np.array(rgb(TH["bg"][1])[::-1], np.float32) * 255
        BASE = np.broadcast_to(c0 * (1 - yy) + c1 * yy, (H, W, 3)).astype(np.float32)
    else: BASE = BG
    SEQ = [("post", -1e9, BTS[0]["t0"])] + [(b, b["t0"], b["t1"]) for b in BTS] + [("post", P_BACK, 1e9)]
    FIRST_DROP = max(BTS[0]["t0"], P_UNTIL + 0.4)

    def slide_x(s0, s1, t):
        """x offset of an item shown from s0 to s1 (eased slide in from the right, out to the left); None = off screen"""
        if t < s0 - SLIDE / 2 or t > s1 + SLIDE / 2: return None
        if t < s0 + SLIDE / 2: return W * (1 - seg(t, s0 - SLIDE / 2, s0 + SLIDE / 2))
        if t > s1 - SLIDE / 2: return -W * seg(t, s1 - SLIDE / 2, s1 + SLIDE / 2)
        return 0.0

    def post(cv, cx, t):
        names = list(NAMES); extra = ASK and t >= ASK_AT - 0.05
        n = len(names) + (1 if ASK else 0); pitch = 136 if ASK else 160; y0 = 560
        cv.rrect(cx - 15, 500, 30, BB - 30 - 500, 10, TH["pole"]); cv.rrect(cx - 90, BB - 40, 180, 30, 12, "#5C6B85")
        end_ph = (t - P_BACK) * 2.4 if t >= P_BACK else None            # names light up one by one, round and round
        for k in range(n):
            last = ASK and k == n - 1
            if last and not extra: continue
            nm = ASK if last else names[k]
            lit = 0.0
            if end_ph is not None and not last:
                d = (end_ph % (len(names) + 1.5)) - k; lit = max(0.0, 1 - abs(d) / 0.9) ** 1.5
            q = WG.pop_s(t, ASK_AT) if last else 1.0
            c = cv.c; c.save(); c.translate(cx, y0 + k * pitch); c.rotate(math.sin(t * 2.3 + k * 1.3) * 0.035); c.scale(q * (1 + 0.07 * lit), q * (1 + 0.07 * lit))
            arrow_sign(cv, TH, 0, 0, nm, k % 2 == 0, 1.0, 0.0 if last else lit, ask=bool(last))
            c.restore()

    def frame(t):
        cv = Canvas(BASE.copy())
        c = cv.c; c.save(); c.new_path(); c.rectangle(0, BT - 40, W, BB - BT + 40); c.clip()
        for it, s0, s1 in SEQ:
            ox = slide_x(s0, s1, t)
            if ox is None: continue
            if it == "post": post(cv, 540 + ox, t)
            else: SCENES[it["scene"]](cv, 540 + ox, max(0.0, t - it["_s0"]), it, t)
        c.restore()
        for k, b in enumerate(BTS):                                    # signs: drop in, lift out as the next drops
            t_in = FIRST_DROP if k == 0 else b["t0"]
            if t < t_in or t > b["t1"] + 0.4: continue
            d = t - t_in; drop = back(min(1.0, d / 0.5), 1.2)
            lift = seg(t, b["t1"] - 0.1, b["t1"] + 0.3)
            ang = math.sin(d * 6.5) * 0.05 * math.exp(-d * 2.2)
            street_sign(cv, TH, 540, SY - 460 * (1 - drop) - 480 * lift, b.get("he"), b["en"], b.get("tag"), 1.0, ang)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 1405), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); draw_end(cv, S, T, t)
        for h in HEL:
            ha = win(t, h["t0"], h["t1"], 0.25, 0.25)
            if ha > 0.01:
                im = hebrew_img(h["text"], 50); w = min(1000, im.shape[1] + 50)
                cv.rrect(540 - w / 2, D.get("he_y", 1745) - 38, w, 76, 38, "#0B1730", 0.7 * ha)
                cv.image(im, 540, D.get("he_y", 1745), 1.0, ha)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, P_UNTIL, D.get("hook_y", 300), 1.0)
        if t >= H_BACK: out = HK.paste(out, hook_img, 540, D.get("hook_y", 300), 1.0, seg(t, H_BACK, H_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(b["t0"] - 0.2, "whoosh", -28) for b in BTS] + [(P_BACK - 0.2, "whoosh", -30)]
    ev += [((FIRST_DROP if k == 0 else b["t0"]) + 0.45, "thud", -26) for k, b in enumerate(BTS)]
    ev += [(b["_swap"], "pop", -24) for b in BTS if b["_swap"] < 1e8]
    ev += [(c[2], "pop", -26) for b in BTS for c in b["extra"].get("chips", [])]
    if ASK: ev += [(ASK_AT, "pop", -24)]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
