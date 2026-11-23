"""name_sign (@noamroz__ story): famous Israeli street names that are surnames, and the hidden meaning of each surname.
Frame 0: a street signpost with one arrow sign per name and the hook above it. Then, beat by beat, a blue Israeli street
sign drops in (Hebrew on top, English below), and a drawn scene under it acts out what the surname means. Screen cards
(lines' "screen") carry the gloss ("Rot + Schild = *red shield*"), source chips the reference, burned-in subtitles the
narration. The last line brings the signpost back, and the last 0.5 s dissolves into frame 0 (loop).

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  post     {until, back_at, names: ["Rothschild", ...]}   frame-0 signpost: shows until `until` (default end of line 0)
           and comes back at `back_at` (optional, the loop line)
  beats    [{at, until, he, en, tag, scene, scene_at, swap_at, extra}]
           at        the sign drops in; the beat lasts until the next beat's `at` (or `until`)
           he / en   Hebrew and English name on the sign; tag = small word on the sign ("BLVD", "ST")
           scene     shield | green | falcon | cape | island
           scene_at  the scene starts (default at + 0.5)
           swap_at   green: the name tag flips from the old name to `extra.new` ("BEN GURION"); cape: the chain of chips
           extra     green: {old: "GRUN", new: "BEN GURION", year: "1910"}; cape: {chips: [[text, at], ...]};
                     island: {town: "Ostrov"}
  hook_back  anchor where the hook comes back over the post (default post.back_at; use the loop line)
  sign_y   default 330;  scene_y default 930 (scene centre);  title_y default 1340;  src_y default 1500;  subs_y default 1640
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

SIGN = "#1E5BB0"; SIGN_IN = "#F5F5F7"; INK = "#0E2250"; SOFT = "#C9D6EA"
SEA0, SEA1 = "#1A6FB3", "#0F4C85"; SAND = "#F2D58A"; LEAF = "#3FBF7F"; WOOD = "#8A5A3C"


def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def street_sign(cv, cx, cy, he, en, tag, a=1.0, sc=1.0, ang=0.0, w=900, h=250):
    """blue Israeli street sign: white inner frame, Hebrew top line, English bottom line"""
    if a <= 0.01: return
    c = cv.c; c.save(); c.translate(cx, cy); c.rotate(ang); c.scale(sc, sc)
    cv.rrect(-w / 2 + 6, -h / 2 + 10, w, h, 26, "#000000", 0.3 * a)
    cv.rrect(-w / 2, -h / 2, w, h, 26, SIGN, a)
    cv.rrect(-w / 2 + 16, -h / 2 + 16, w - 32, h - 32, 18, None, a, stroke=SIGN_IN, lw=6, sa=a)
    if he: hebrew(cv, he, 0, -h * 0.2, int(h * 0.3), (245, 245, 247), a)
    cv.text(en, 0, h * 0.2, h * 0.3, WHITE, a, maxw=w - (300 if tag else 120), shadow=False)
    if tag: cv.text(tag, w / 2 - 74, h * 0.2, 36, SOFT, a, shadow=False, bold=False)
    c.restore()


def arrow_sign(cv, cx, cy, txt, right, a=1.0, w=470, h=104):
    if a <= 0.01: return
    x0 = cx if right else cx - w; tip = 46
    pts = ([(x0, cy - h / 2), (x0 + w - tip, cy - h / 2), (x0 + w, cy), (x0 + w - tip, cy + h / 2), (x0, cy + h / 2)] if right else
           [(x0 + tip, cy - h / 2), (x0 + w, cy - h / 2), (x0 + w, cy + h / 2), (x0 + tip, cy + h / 2), (x0, cy)])
    poly(cv, [(x + 5, y + 8) for x, y in pts], "#000000", 0.3 * a)
    poly(cv, pts, SIGN, a, stroke=SIGN_IN, lw=5)
    cv.text(txt, x0 + w / 2 + (-tip / 2 if right else tip / 2), cy, 54, WHITE, a, maxw=w - tip - 40, shadow=False)


# ------------------------------------------------------------------ scenes (cx, cy = scene centre; u = seconds into scene)
def sc_shield(cv, cx, cy, u, a, B):
    hx, gy = cx - 120, cy + 330                                     # narrow gabled Frankfurt house, timber frame
    hw, hh = 330, 470; top = gy - hh
    poly(cv, [(hx - hw / 2, top), (hx, top - 170), (hx + hw / 2, top)], "#B5523B", a)
    cv.rrect(hx - hw / 2, top, hw, hh, 6, "#F1E3C8", a)
    for k in range(4):
        y = top + 20 + k * 112; cv.rrect(hx - hw / 2, y, hw, 12, 2, WOOD, a)
    for xx in (hx - hw / 2, hx - 6, hx + hw / 2 - 12): cv.rrect(xx, top, 12, hh, 2, WOOD, a)
    for row in range(3):
        for col_ in (-1, 1):
            cv.rrect(hx + col_ * 85 - 38, top + 50 + row * 112, 76, 70, 8, "#7FB6E0", a, stroke=WOOD, lw=4, sa=a)
    cv.rrect(hx - 50, gy - 140, 100, 140, 14, WOOD, a)
    cv.rrect(cx - 520, gy, 1040, 14, 4, "#5C6B85", a)
    bx = hx + hw / 2; by = top + 120                                 # bracket and the swinging red shield
    cv.line([(bx, by), (bx + 210, by)], "#2B2B2B", 10, a)
    drop = eout(u / 0.6); sw = math.sin(u * 3.2) * 0.32 * math.exp(-u * 0.6)
    c = cv.c; c.save(); c.translate(bx + 170, by); c.rotate(sw); c.translate(0, -180 * (1 - drop))
    cv.line([(-50, 0), (-50, 50)], "#2B2B2B", 5, a); cv.line([(50, 0), (50, 50)], "#2B2B2B", 5, a)
    sh = [(-95, 50), (95, 50), (95, 150), (0, 250), (-95, 150)]
    poly(cv, [(x + 5, y + 7) for x, y in sh], "#000000", 0.3 * a)
    poly(cv, sh, RED, a, stroke=YEL, lw=7)
    c.restore()
    g = seg(u, 1.2, 1.8)                                             # glow ring once it settles
    if g > 0: cv.circ(bx + 170, by + 160, 150 + 20 * math.sin(u * 4), None, 1, stroke=YEL, lw=4 * g)


def sc_green(cv, cx, cy, u, a, B, t):
    X = B.get("extra", {}); sw = B.get("_swap", 1e9)
    fill = eout(seg(u, 0.4, 1.6, lambda x: x))                       # green rises from the bottom of the scene band
    top = cy + 380 - 760 * fill
    if fill > 0: cv.rrect(0, top, W, cy + 380 - top, 0, "#2E9E5B", 0.85 * a)
    for k in range(6):
        y = top + 8 * math.sin(u * 3 + k)
        if fill > 0.02: cv.circ(90 + k * 180, y, 40, "#2E9E5B", 0.85 * a)
    flip = seg(t, sw - 0.2, sw + 0.2, lambda x: x)                  # tag flips (scale x 1 -> 0 -> 1)
    sx = abs(math.cos(math.pi * flip)); new = flip >= 0.5
    c = cv.c; c.save(); c.translate(cx, cy - 40); c.scale(max(0.02, sx), 1)
    cv.rrect(-360, -150, 720, 300, 30, "#FFFFFF", a, stroke=INK, lw=6, sa=a)
    cv.rrect(-360, -150, 720, 80, 30, RED if not new else SIGN, a)
    cv.text("HELLO, MY NAME IS", 0, -110, 40, WHITE, a, shadow=False)
    if not new: cv.text(X.get("old", "GRUN"), 0, 30, 130, "#2E9E5B", a, shadow=False, maxw=640)
    else: cv.text(X.get("new", "BEN GURION"), 0, 30, 104, INK, a, shadow=False, maxw=640)
    c.restore()
    if X.get("year") and t > sw + 0.25:                               # yellow year chip
        p = WG.pop_s(t, sw + 0.25); cv.rrect(cx - 130 * p, cy + 170, 260 * p, 100 * p, 50 * p, YEL, a)
        cv.text(X["year"], cx, cy + 170 + 50 * p, 70 * p, INK, a, shadow=False, minsz=8)


def falcon(cv, x, y, s, flap, a, ang=0.0):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang); c.scale(s, s)
    wy = -60 * flap
    poly(cv, [(-10, -6), (-150, wy - 30), (-230, wy + 10), (-120, 10)], "#5A4632", a)     # far wing
    ellipse_ = lambda cx_, cy_, rx, ry, col: (c.save(), c.translate(cx_, cy_), c.scale(rx, ry), c.new_path(),
                                              c.arc(0, 0, 1, 0, 2 * math.pi), c.restore(), cv.col(col, a), c.fill())
    poly(cv, [(-120, 0), (-190, -18), (-190, 22)], "#4A3A2A", a)                          # tail
    ellipse_(0, 0, 120, 38, "#7A5C3E"); ellipse_(10, 10, 80, 22, "#E8D9B5")
    ellipse_(105, -18, 38, 34, "#4A3A2A")
    poly(cv, [(138, -24), (168, -10), (138, -4)], YEL, a)                                 # beak
    cv.circ(112, -26, 9, YEL, a); cv.circ(114, -26, 4, "#111111", a)
    poly(cv, [(10, -6), (-60, wy * 1.25 - 40), (-170, wy * 1.2 - 10), (-60, 14)], "#8B6B47", a)  # near wing
    c.restore()


def sc_falcon(cv, cx, cy, u, a, B):
    for k in range(3):                                               # clouds drift
        x = (180 + k * 380 - u * 40) % 1300 - 110; y = cy - 260 + k * 170
        for dx, r in ((-50, 44), (0, 60), (55, 46)): cv.circ(x + dx, y, r, "#FFFFFF", 0.14 * a)
    cv.circ(cx + 330, cy - 300, 80, YEL, 0.9 * a)
    p = min(1.0, u / 3.2)                                            # flies in on an arc, then hovers big in the middle
    x = -300 + (cx - 60 + 300) * eout(p); y = cy + 120 - 260 * math.sin(math.pi * p * 0.9) * (1 - p) + 0 * p
    s = 1.3 + 0.9 * eout(p)
    falcon(cv, x, y, s, math.sin(u * 9), a, ang=-0.12 * (1 - p))


def sc_cape(cv, cx, cy, u, a, B, t):
    sway = math.sin(u * 2.2) * 10                                    # a cape on a hanger
    hx, hy = cx, cy - 370
    cv.line([(hx - 120, hy + 40), (hx, hy), (hx + 120, hy + 40)], SOFT, 8, a); cv.circ(hx, hy - 22, 18, None, a, stroke=SOFT, lw=6)
    poly(cv, [(hx - 110, hy + 40), (hx + 110, hy + 40), (hx + 170 + sway, hy + 280), (hx - 170 + sway, hy + 280)], "#7B2D8E", a)
    poly(cv, [(hx - 70, hy + 40), (hx + 70, hy + 40), (hx + 40, hy + 120), (hx - 40, hy + 120)], "#5A1F6A", a)
    cv.rrect(hx - 170 + sway, hy + 268, 340, 16, 6, YEL, a)
    chips = (B.get("extra") or {}).get("chips", [])
    y0 = cy + 20
    for k, ch in enumerate(chips):                                   # cappa -> chaplain -> Kaplan
        t0 = ch[2]
        if t < t0: continue
        p = WG.pop_s(t, t0); txt = ch[0]; hl = k == len(chips) - 1
        w = min(980, cv.tw(txt, 58) + 70); y = y0 + k * 118
        if k: cv.line([(cx, y - 70), (cx, y - 50)], SOFT, 5, a)
        c = cv.c; c.save(); c.translate(cx, y); c.scale(p, p)
        cv.rrect(-w / 2, -48, w, 96, 48, YEL if hl else "#0B1730", 0.95 * a, stroke=YEL, lw=3, sa=a)
        cv.text(txt, 0, 0, 58, INK if hl else WHITE, a, shadow=False, maxw=w - 40)
        c.restore()


def sc_island(cv, cx, cy, u, a, B):
    sea = cy + 120
    cv.rrect(0, sea, W, cy + 380 - sea, 0, SEA1, a)
    rise = eout(seg(u, 0.2, 1.4, lambda x: x)); iy = sea + 4 + 116 * (1 - rise)
    c = cv.c; c.save(); c.new_path(); c.rectangle(0, 0, W, sea + 8); c.clip()
    c.save(); c.translate(cx, iy); c.scale(330, 110); c.new_path(); c.arc(0, 0, 1, math.pi, 2 * math.pi); c.restore()
    cv.col(SAND, a); c.fill()
    for k, (dx, hh) in enumerate(((-150, 90), (-80, 120), (40, 100))):   # little town on the island
        hb = iy - 60
        cv.rrect(cx + dx - 34, hb - hh + 60, 68, hh, 4, "#F5F5F7", a)
        poly(cv, [(cx + dx - 42, hb - hh + 62), (cx + dx, hb - hh + 20), (cx + dx + 42, hb - hh + 62)], "#B5523B", a)
    px, py = cx + 170, iy - 80                                       # palm
    cv.line([(px, iy - 10), (px - 10, py - 120)], WOOD, 16, a)
    for ang in (-2.6, -2.0, -1.2, -0.5, 0.2):
        tip = (px - 10 + 110 * math.cos(ang + 0.05 * math.sin(u * 2)), py - 120 + 70 * math.sin(ang) + 30)
        cv.line([(px - 10, py - 120), tip], LEAF, 18, a)
    c.restore()
    for k in range(5):                                               # waves
        y = sea + 30 + k * 48; pts = [(x, y + 9 * math.sin(x / 70 + u * 2.5 + k)) for x in range(0, W + 40, 40)]
        cv.line(pts, SEA0 if k % 2 else "#2A86CC", 6, 0.8 * a)
    town = (B.get("extra") or {}).get("town")
    if town and u > 1.4:
        p = WG.pop_s(u, 1.4); cv.rrect(cx - 230 * p, cy - 330, 460 * p, 96 * p, 48 * p, YEL, a)
        cv.text(town, cx, cy - 330 + 48 * p, 60 * p, INK, a, shadow=False, minsz=8)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.8)); A = T.at
    SY = D.get("sign_y", 330); CY = D.get("scene_y", 930)
    PO = D.get("post") or {}; P_UNTIL = A(PO.get("until"), T.E(0)); P_BACK = A(PO.get("back_at"), T.END + 9)
    H_BACK = A(D.get("hook_back"), P_BACK)
    NAMES = PO.get("names") or [b["en"] for b in D["beats"]]
    BT = []
    for k, b in enumerate(D["beats"]):
        b = dict(b); b["t0"] = A(b["at"]); b["s0"] = A(b.get("scene_at"), b["t0"] + 0.5)
        b["_swap"] = A(b.get("swap_at"), 1e9) if b.get("swap_at") else 1e9
        ex = dict(b.get("extra") or {})
        if ex.get("chips"): ex["chips"] = [[c[0], c[1], A(c[1])] for c in ex["chips"]]
        b["extra"] = ex; BT.append(b)
    for k, b in enumerate(BT):
        b["t1"] = A(b["until"]) if b.get("until") else (BT[k + 1]["t0"] if k + 1 < len(BT) else P_BACK - 0.1)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    SC = dict(shield=sc_shield, falcon=sc_falcon, island=sc_island)

    def post(cv, t, a):
        if a <= 0.01: return
        y0, y1 = 540, 1225
        cv.rrect(528, y0, 24, y1 - y0, 8, "#8FA3C0", a); cv.rrect(470, y1 - 10, 140, 24, 10, "#5C6B85", a)
        n = len(NAMES); pitch = min(150, (y1 - y0 - 120) / max(1, n))
        for k, nm in enumerate(NAMES):
            sa = a                                                    # all arrows on screen from frame 0
            sw = math.sin(t * 2 + k) * 0.015
            c = cv.c; c.save(); c.translate(540, y0 + 90 + k * pitch); c.rotate(sw)
            arrow_sign(cv, 0, 0, nm, k % 2 == 0, sa); c.restore()

    def frame(t):
        cv = Canvas(BG.copy())
        pa = max(1 - seg(t, P_UNTIL, P_UNTIL + 0.35), seg(t, P_BACK, P_BACK + 0.4))
        post(cv, t, pa)
        for b in BT:
            ba = win(t, b["t0"] - 0.05, b["t1"], 0.2, 0.3)
            if ba <= 0.01: continue
            u = t - b["s0"]
            if u > -0.2:
                sa = ba * seg(t, b["s0"] - 0.2, b["s0"] + 0.2)
                if b["scene"] == "green": sc_green(cv, 540, CY, max(0, u), sa, b, t)
                elif b["scene"] == "cape": sc_cape(cv, 540, CY, max(0, u), sa, b, t)
                else: SC[b["scene"]](cv, 540, CY, max(0, u), sa, b)
            d = t - b["t0"]; drop = back(min(1.0, d / 0.45), 1.3)        # sign drops in and swings to rest
            ang = math.sin(d * 7) * 0.06 * math.exp(-d * 2.2)
            street_sign(cv, 540, SY - 320 * (1 - drop), b.get("he"), b["en"], b.get("tag"), ba, 1.0, ang)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 1340), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, P_UNTIL, D.get("hook_y", 330), 1.0)
        if t >= H_BACK:                                               # the loop line brings the hook back over the post
            out = HK.paste(out, hook_img, 540, D.get("hook_y", 330), 1.0, seg(t, H_BACK, H_BACK + 0.4))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(b["t0"], "whoosh", -28) for b in BT] + [(b["t0"] + 0.42, "thud", -26) for b in BT]
    ev += [(b["_swap"], "pop", -24) for b in BT if b["_swap"] < 1e8]
    ev += [(c[2], "pop", -26) for b in BT for c in b["extra"].get("chips", [])]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
