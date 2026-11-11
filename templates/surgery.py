"""surgery (@noamroz__): the medical metaphor, clean and animated (no gore, no injuries, no real images).
A scan monitor shows a patient (top view, lying on the table). The patient is Gaza: inside the body the lights are its
people (healthy cells, drawn with sympathy), with homes, schools, a hospital and the UN HQ as small icons. The tumour is
HAMAS only: a dark red mass whose roots (tunnels) grow under the icons. A locator inset shows the whole land of Israel
(gold, West Bank and Gaza darker gold, Green Line green) with Gaza ringed. A heartbeat line runs along the monitor.
Beats (all anchors in data): icons light up as named; a "41 km vs 720 km" bar panel; two option pills; option 1 drops one
bomb (flash, the scan greys out, the heartbeat goes flat, then rewinds); option 2 sends small surgeon figures in along the
roots, which retract; some surgeons fade and leave a candle (soldiers killed); a few healthy cells grey out and the camera
zooms into one: a person; the law beat puts a crosshair on one root with a "military gain / civilian harm" scale; the
tumour is removed, the cells brighten, the heartbeat steadies, end card; the hook returns and the last 0.5 s dissolves
into frame 0 (circular).

data keys:
  roots0           root growth at frame 0 (default 0.45); grow_at / grow_until: roots grow to full
  icons            [{kind: home|school|hospital|un, x, y, at?, label?, label_at?, label_until?, rockets_at?}]
  ruler            {at, until, rows: [{label, value, txt, color?}]}       bar panel (Gaza length vs tunnel length)
  options          {at, until, one_at, two_at, one, two}                 the two option pills
  bomb_at, rewind_at                                                      option 1 (bomb lands 0.6 s after bomb_at)
  surgery_at, cut_until                                                   surgeons go in; roots retract until cut_until
  fallen_at        surgeons fade to outlines and a candle appears (count = data.fallen, default 2 of 5)
  lost_at          a few healthy cells grey out;  zoom_at / zoom_until: zoom into one lost cell (a person)
  law_at, law_until    crosshair + scale panel ({law_left, law_right} labels)
  heal_at          tumour removed, cells brighten;  hook_back (default END-1.4)
cover.bg "scene": a clean frame of the opening scan behind the cover text.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

INK = "#0E2250"; SOFT = "#C9D6EA"; SCAN = "#10284F"; CELL = "#BFE3FF"; TUM = "#8E2436"; TUM_L = "#E5484D"; ROOT_D = "#5E1622"
MON = (60, 430, 960, 990)                     # monitor rect x, y, w, h
CORE = (540, 870)
EMO = {"home": "🏠", "school": "🏫", "hospital": "🏥", "un": "🏢"}


def lerp(a, b, u): return a + (b - a) * u


def emo(cv, ch, x, y, sz, a):
    if a > 0.01 and sz > 1: WG.emo(cv, ch, x, y, sz, min(1.0, a))


# ------------------------------------------------------------------ body geometry (scene coords = screen coords at zoom 1)
HEAD = (540, 548, 62)
TORSO = (360, 628, 360, 440, 110)             # x, y, w, h, r
ARMS = [((392, 672), (300, 1060)), ((688, 672), (780, 1060))]
LEGS = [((478, 1040), (462, 1300)), ((602, 1040), (618, 1300))]
ARM_W, LEG_W = 74, 104


def body_path(c):
    """one cairo path (winding = union) for the patient's silhouette"""
    c.new_path()
    x, y, r = HEAD; c.new_sub_path(); c.arc(x, y, r, 0, 2 * math.pi)
    c.rectangle(516, 590, 48, 60)
    x, y, w, h, r = TORSO; c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0); c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi); c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi); c.close_path()
    for (p, q), wd in [(a, ARM_W) for a in ARMS] + [(l, LEG_W) for l in LEGS]:
        ang = math.atan2(q[1] - p[1], q[0] - p[0]); nx, ny = -math.sin(ang) * wd / 2, math.cos(ang) * wd / 2
        c.new_sub_path(); c.move_to(p[0] + nx, p[1] + ny); c.line_to(q[0] + nx, q[1] + ny)
        c.line_to(q[0] - nx, q[1] - ny); c.line_to(p[0] - nx, p[1] - ny); c.close_path()
        for e in (p, q): c.new_sub_path(); c.arc(e[0], e[1], wd / 2, 0, 2 * math.pi)
    c.set_fill_rule(0)      # cairo.FILL_RULE_WINDING


def body_mask():
    m = np.zeros((H, W), np.uint8)
    x, y, r = HEAD; cv2.circle(m, (x, y), r, 255, -1)
    x, y, w, h, r = TORSO; cv2.rectangle(m, (x + r, y), (x + w - r, y + h), 255, -1); cv2.rectangle(m, (x, y + r), (x + w, y + h - r), 255, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)): cv2.circle(m, (cx, cy), r, 255, -1)
    for (p, q), wd in [(a, ARM_W) for a in ARMS] + [(l, LEG_W) for l in LEGS]: cv2.line(m, p, q, 255, wd)
    return m


def blob(cv, x, y, r, t, col, a, stroke=None, lw=4, k=9, seed=0.0):
    c = cv.c; c.new_path()
    for i in range(k * 4 + 1):
        ang = i * 2 * math.pi / (k * 4)
        rr = r * (1 + 0.13 * math.sin(ang * 3 + seed + t * 1.3) + 0.08 * math.sin(ang * 5 - t * 0.9 + seed * 2))
        (c.move_to if i == 0 else c.line_to)(x + rr * math.cos(ang), y + rr * math.sin(ang))
    c.close_path(); cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def make_roots(icons, rng):
    """polylines from the core to each icon (passing under it) plus a few side roots"""
    ends = [(i["x"], i["y"] + 26) for i in icons] + [(395, 860), (690, 880), (540, 1120), (330, 780), (750, 780), (500, 640), (580, 1250)]
    roots = []
    for k, (ex, ey) in enumerate(ends):
        n = 14; pts = []
        for j in range(n + 1):
            u = j / n; bend = math.sin(u * math.pi) * rng.uniform(-38, 38)
            dx, dy = ex - CORE[0], ey - CORE[1]; L = math.hypot(dx, dy) + 1e-6
            px, py = CORE[0] + dx * u - dy / L * bend, CORE[1] + dy * u + dx / L * bend
            pts.append((px + rng.uniform(-5, 5) * (0 < j < n), py + rng.uniform(-5, 5) * (0 < j < n)))
        roots.append(dict(pts=pts, delay=0.0 if k < len(icons) else rng.uniform(0.05, 0.3), w=1.0 if k < len(icons) else 0.75))
    return roots


def partial(pts, f):
    if f <= 0: return []
    n = len(pts) - 1; m = f * n; i = int(m)
    out = pts[:i + 1]
    if i < n: p, q = pts[i], pts[i + 1]; u = m - i; out = out + [(lerp(p[0], q[0], u), lerp(p[1], q[1], u))]
    return out


def ecg_y(ph):
    """one heartbeat (ph 0..1) as a vertical offset (up = negative)"""
    if 0.10 < ph < 0.16: return -10 * math.sin((ph - 0.10) / 0.06 * math.pi)
    if 0.20 < ph < 0.23: return 12 * (ph - 0.20) / 0.03
    if 0.23 < ph < 0.27: return 12 - 70 * (ph - 0.23) / 0.04
    if 0.27 < ph < 0.31: return -58 + 78 * (ph - 0.27) / 0.04
    if 0.31 < ph < 0.34: return 20 - 20 * (ph - 0.31) / 0.03
    if 0.45 < ph < 0.58: return -16 * math.sin((ph - 0.45) / 0.13 * math.pi)
    return 0.0


def locator():
    """whole land of Israel (MapR rules) with Gaza ringed: RGBA crop, rendered once"""
    MR = MapR(); k = 74.0; cosf = math.cos(math.radians(31.5)); bh = 420
    img, M = MR.render(MR.view(35.05, 31.45, k, cosf, bh / 2), bh)
    gx, gy = geo2px(M, 34.39, 31.42)
    x0 = 540 - 118; crop = img[40:bh - 40, x0:x0 + 190].copy()
    cv2.circle(crop, (int(gx - x0), int(gy - 40)), 26, (63, 210, 255), 3, cv2.LINE_AA)
    rgba = np.dstack([crop[..., ::-1], np.full(crop.shape[:2], 255, np.uint8)])
    return rgba, (gx - x0, gy - 40)


# ------------------------------------------------------------------ render
def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.6)); A = T.at; END = T.END
    rng = np.random.default_rng(7)
    icons = [dict(i, t=A(i.get("at"), -1), lt0=A(i.get("label_at"), 1e9), lt1=A(i.get("label_until"), 1e9), rk=A(i.get("rockets_at"), 1e9)) for i in D["icons"]]
    roots = make_roots(icons, rng)
    # healthy cells: inside the body, away from the core and the icons
    mask = body_mask()
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cs = max(cs, key=cv2.contourArea)[:, 0, :]
    global OUTLINE; OUTLINE = [(float(x), float(y)) for x, y in cs] + [(float(cs[0][0]), float(cs[0][1]))]
    er = cv2.erode(mask, np.ones((23, 23), np.uint8)); cells = []
    while len(cells) < D.get("cells", 95):
        x, y = rng.uniform(270, 810), rng.uniform(490, 1340)
        if er[int(y), int(x)] == 0 or math.hypot(x - CORE[0], y - CORE[1]) < 95: continue
        if any(math.hypot(x - i["x"], y - i["y"]) < 44 for i in icons) or any(math.hypot(x - c["x"], y - c["y"]) < 30 for c in cells): continue
        cells.append(dict(x=x, y=y, r=rng.uniform(9, 13), ph=rng.uniform(0, 6.3), lost=False))
    near = sorted(range(len(cells)), key=lambda j: math.hypot(cells[j]["x"] - 600, cells[j]["y"] - 980))
    for j in near[:D.get("lost_n", 6)]: cells[j]["lost"] = True
    zc = cells[near[0]]
    tG0, tG1 = A(D.get("grow_at"), 0.0), A(D.get("grow_until"), 6.0); g0 = D.get("roots0", 0.45)
    O = D.get("options") or {}; tO0, tO1 = A(O.get("at"), 1e9), A(O.get("until"), 1e9); tO1a, tO2a = A(O.get("one_at"), 1e9), A(O.get("two_at"), 1e9)
    tBomb = A(D["bomb_at"]); tHit = tBomb + 0.6; tRew = A(D["rewind_at"])
    tSurg, tCut = A(D["surgery_at"]), A(D["cut_until"]); tFall = A(D["fallen_at"]); tLost = A(D["lost_at"])
    tZ0, tZ1 = A(D["zoom_at"]), A(D["zoom_until"]); tL0, tL1 = A(D["law_at"]), A(D["law_until"]); tHeal = A(D["heal_at"])
    R = D.get("ruler") or {}; tR0, tR1 = A(R.get("at"), 1e9), A(R.get("until"), 1e9)
    tHB = A(D.get("hook_back"), END - 1.4)
    surgeons = [dict(x=x, y=y, k=k) for k, (x, y) in enumerate([(470, 790), (612, 800), (452, 950), (630, 940), (540, 1110)])]
    fallen = set(D.get("fallen_idx", [1, 4]))
    LOC, LOCG = locator()
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 262); tHk = T.E(0)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    E = S.get("end") or {}; tEnd = A(E.get("at"), tHeal)

    def grey_u(t):          # option 1: the scan greys out on impact, rewinds at rewind_at
        if t < tHit: return 0.0
        return min(seg(t, tHit, tHit + 0.25), 1 - seg(t, tRew, tRew + 0.6))

    def root_f(t):          # how much of each root exists
        g = lerp(g0, 1.0, seg(t, tG0, tG1)); cut = seg(t, tSurg + 0.6, tCut, lambda u: u) * 0.55 + 0.45 * seg(t, tHeal, tHeal + 1.4)
        return g * (1 - cut)

    def cam(t):
        z = 1.0 + 5.0 * (seg(t, tZ0, tZ0 + 1.0) - seg(t, tZ1, tZ1 + 0.8))
        u = (z - 1) / 5.0
        return z, lerp(540, zc["x"], u), lerp(925, zc["y"], u)

    def scene(cv, t, clean=False):
        c = cv.c; mx, my, mw, mh = MON
        WG.panel(cv, mx, my, mw, mh, 1.0)
        c.save(); c.new_path(); c.rectangle(mx + 4, my + 4, mw - 8, mh - 8); c.clip()
        z, cx, cy = cam(t)
        c.translate(540, 925); c.scale(z, z); c.translate(-cx, -cy)
        # table + body
        cv.rrect(250, 470, 580, 900, 60, "#0D2246", 0.9)
        body_path(c); cv.col(SCAN, 1.0); c.fill(); c.new_path()
        cv.line(OUTLINE, SKY, 5, 0.85)
        c.save(); body_path(c); c.clip()
        sy = 470 + ((t * 260) % 1100)                                   # scan bar sweeping down
        for k in range(6): cv.rrect(240, sy - k * 9, 600, 9, 0, SKY, 0.05 * (6 - k))
        # roots (tunnels)
        f = root_f(t); heal = seg(t, tHeal, tHeal + 1.4)
        for r in roots:
            ff = max(0.0, min(1.0, (f - r["delay"]) / (1 - r["delay"]))) if f < 1 else 1.0
            p = partial(r["pts"], ff)
            if len(p) < 2: continue
            cv.line(p, ROOT_D, 15 * r["w"], 0.95); cv.line(p, TUM, 8 * r["w"], 0.95); cv.line(p, "#E06A78", 2.5, 0.6, dash=[10, 12])
        # healthy cells (people)
        lost_u = seg(t, tLost, tLost + 0.8)
        for k, ce in enumerate(cells):
            br = 0.75 + 0.25 * math.sin(t * 2.2 + ce["ph"]) + 0.25 * heal
            dx = math.sin(t * 0.7 + ce["ph"]) * 3; dy = math.cos(t * 0.6 + ce["ph"] * 1.3) * 3
            if ce["lost"] and lost_u > 0:
                col = CELL if lost_u < 0.5 else "#6E7A90"; aa = lerp(0.95, 0.55, lost_u)
                rr = ce["r"] * (1 + 1.6 * seg(z, 1.5, 5.0)) if ce is zc else ce["r"]
                cv.circ(ce["x"] + dx, ce["y"] + dy, rr, col, aa)
                if z > 1.6 and ce is zc:      # the zoom: a person inside the cell
                    pa = seg(z, 2.2, 4.8, lambda u: u)
                    cv.circ(ce["x"] + dx, ce["y"] + dy, rr * 0.94, "#22304A", pa, stroke=CELL, lw=1.2)
                    figure(cv, ce["x"] + dx, ce["y"] + dy + rr * 0.6, sc=rr * 0.0085, col="#E8EEF8", badge=False, a=pa, lw=7)
                continue
            cv.circ(ce["x"] + dx, ce["y"] + dy, ce["r"] * 1.9, CELL, 0.10 * br)
            cv.circ(ce["x"] + dx, ce["y"] + dy, ce["r"], CELL, min(1, 0.85 * br))
        c.restore()
        # icons
        for i in icons:
            pa = WG.pop_s(t, i["t"]) if i["t"] > 0 else 1.0
            hl = win(t, i["t"], i["t"] + 1.6, 0.15, 0.5) if i["t"] > 0 else 0.0
            if i.get("label") and win(t, i["lt0"], i["lt1"], 0.2, 0.3) > 0.01: hl = max(hl, win(t, i["lt0"], i["lt1"], 0.2, 0.3))
            if hl > 0.01: cv.circ(i["x"], i["y"], 44 + 6 * math.sin(t * 6), None, 1, stroke=YEL, lw=4 * hl); c.new_path()
            emo(cv, EMO[i["kind"]], i["x"], i["y"], 66 * (1 + 0.15 * hl) * min(1.15, pa), min(1, pa))
            if i["rk"] < 1e8:
                for q in range(2): emo(cv, "🚀", i["x"] - 16 + q * 30, i["y"] + 34, 30 * WG.pop_s(t, i["rk"] + q * 0.12), min(1, WG.pop_s(t, i["rk"] + q * 0.12)))
        # tumour (HAMAS)
        ts = 1 - seg(t, tHeal, tHeal + 1.2) * 1.0 - 0.25 * seg(t, tSurg + 0.6, tCut, lambda u: u)
        if ts > 0.02:
            pul = 1 + 0.05 * math.sin(t * 3.1)
            blob(cv, *CORE, 92 * ts * pul, t, TUM, 0.25)
            blob(cv, *CORE, 70 * ts * pul, t, TUM, 0.97, stroke=TUM_L, lw=4, seed=1.0)
            if ts > 0.45: cv.text("HAMAS", CORE[0], CORE[1], 44 * min(1, ts), WHITE, min(1, (ts - 0.45) * 3), maxw=150 * ts + 20, minsz=36)
        # surgeons (option 2)
        if t > tSurg:
            for s in surgeons:
                u = seg(t, tSurg + 0.15 * s["k"], tSurg + 1.1 + 0.15 * s["k"])
                x = lerp(540 + (s["x"] - 540) * 0.3, s["x"], u); y = lerp(420, s["y"], u)
                fa = 1.0
                if s["k"] in fallen: fa = 1 - 0.75 * seg(t, tFall + 0.2 * s["k"] / 4, tFall + 0.6 + 0.2 * s["k"] / 4)
                fa *= 1 - seg(t, tHeal + 0.6, tHeal + 1.4)
                if fa > 0.01:
                    cv.circ(x, y - 60, 46, SKY, 0.18 * fa)
                    figure(cv, x, y, sc=0.6, col=SKY if fa > 0.6 else SOFT, badge=True, arms="point", a=fa, lw=7, walk=t * 9 * (u < 1))
                    if fa > 0.6: cv.line([(x + 36, y - 92), (x + 58, y - 110)], WHITE, 4, fa)       # scalpel
                if s["k"] in fallen and t > tFall:
                    emo(cv, "🕯️", x + 4, y - 70, 52, seg(t, tFall + 0.3, tFall + 0.8) * (1 - seg(t, tHeal + 0.6, tHeal + 1.4)) * (1 - seg(z, 1.1, 1.6)))
        # law: crosshair on one root
        la = win(t, tL0, tL1, 0.3, 0.3)
        if la > 0.01:
            px, py = 650, 965; rr = 46 + 8 * math.sin(t * 5)
            cv.circ(px, py, rr, None, 1, stroke=YEL, lw=5 * la); c.new_path()
            for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)): cv.line([(px + dx_ * (rr - 16), py + dy_ * (rr - 16)), (px + dx_ * (rr + 18), py + dy_ * (rr + 18))], YEL, 5, la)
        c.restore()
        # header, locator, heartbeat (fixed on the monitor)
        ha = 1 - seg(z, 1.05, 1.6)
        cv.text("PATIENT: GAZA", mx + 40, my + 50, 40, SOFT, ha, align="l", shadow=False)
        cv.image(LOC, mx + mw - 120, my + 200, 1.0, ha)
        cv.rrect(mx + mw - 215, my + 30, 190, 340, 16, None, 1, stroke=SKY, lw=2, sa=0.5 * ha)
        cv.text("Gaza", mx + mw - 120, my + 395, 38, YEL, ha, shadow=False)
        gy_ = grey_u(t); hb = 1 - gy_
        ex0, ex1, ey = mx + 40, mx + mw - 40, my + mh - 60
        cv.rrect(ex0 - 10, ey - 52, ex1 - ex0 + 20, 92, 18, "#081530", 0.85)
        bpm = 1.25 + 0.15 * seg(t, tHeal, tHeal + 1.0)
        pts = []
        for xx in range(int(ex0), int(ex1), 4):
            ph = ((t * bpm) - (ex1 - xx) / 420.0) % 1.0
            pts.append((xx, ey + ecg_y(ph) * 0.62 * hb))
        cv.line(pts, GREEN if gy_ < 0.5 else "#E5484D", 4, 0.95)
        return z

    def option_pills(cv, t):
        a = win(t, tO0, tO1, 0.3, 0.3)
        if a <= 0.01: return
        y = 1250; WG.panel(cv, 70, y - 95, 940, 190, a)
        for k, (txt, ta, em) in enumerate(((O.get("one", "One bomb"), tO1a, "💣"), (O.get("two", "Surgery"), tO2a, "🩺"))):
            on = seg(t, ta, ta + 0.3) * (1 - seg(t, tO2a, tO2a + 0.3) if k == 0 else 1)
            x = 95 + k * 460
            cv.rrect(x, y - 70, 430, 140, 34, YEL if on > 0.5 else "#16305C", a, stroke=YEL, lw=3, sa=a * 0.7)
            WG.emo(cv, em, x + 66, y, 66, a)
            cv.text(txt, x + 255, y, 56, INK if on > 0.5 else WHITE, a, maxw=300, shadow=False)

    def law_panel(cv, t):
        a = win(t, tL0 + 0.3, tL1, 0.3, 0.3)
        if a <= 0.01: return
        y = 1250; WG.panel(cv, 70, y - 95, 940, 190, a)
        WG.emo(cv, "⚖️", 540, y, 96, a)
        cv.text(D.get("law_left", "Military gain"), 290, y, 54, WHITE, a, maxw=380)
        cv.text(D.get("law_right", "Civilian harm"), 790, y, 54, WHITE, a, maxw=380)

    def ruler(cv, t):
        a = win(t, tR0, tR1, 0.3, 0.3)
        if a <= 0.01: return
        rows = [dict(r, at=tR0 + 0.3 + 0.9 * k) for k, r in enumerate(R["rows"])]
        WG.panel(cv, 60, 1075, 960, 290, a)
        WG.bars(cv, rows, t, a, 1160, rowh=120, vmax=max(r["value"] for r in rows), lab_w=330, grow=1.4)

    def end_card(cv, t):
        a = win(t, tEnd - 0.1, tHB + 0.1, 0.35, 0.35)
        if a <= 0.01: return
        y = E.get("y", 1220); WG.panel(cv, 60, y - 215, 960, 430, a)
        if E.get("balance"): WG.title(cv, E["balance"], y - 80, a, 64)
        if E.get("cta"):
            ca = a * seg(t, tEnd + 0.6, tEnd + 1.0); two = cv.tw(E["cta"].replace("*", ""), 54) > 820
            tw = 900 if two else cv.tw(E["cta"].replace("*", ""), 54) + 90; ph = 150 if two else 110
            cv.rrect(540 - tw / 2, y + 40, tw, ph, 44, SKY, ca); WG.title(cv, E["cta"], y + 40 + ph / 2, ca, 54, maxw=780, col=INK)

    def layer(t, hook=True, subs=True, overlays=True):
        cv = Canvas(bars_moving_bg(t, 6)); scene(cv, t)
        out = cv.out().astype(np.float32)
        # bomb: falls, flash, grey
        gy_ = grey_u(t)
        if gy_ > 0.001:
            mx, my, mw, mh = MON; reg = out[my:my + mh, mx:mx + mw]
            g = reg.mean(axis=2, keepdims=True) * 0.5; out[my:my + mh, mx:mx + mw] = reg * (1 - gy_) + g * gy_
        c2 = Canvas(out)
        if tBomb <= t < tHit + 0.1:
            u = seg(t, tBomb, tHit, lambda z: z * z); WG.emo(c2, "💣", 540, lerp(380, CORE[1], u), 120, 1.0)
        fl = 1 - seg(t, tHit, tHit + 0.45, lambda z: z)
        if tHit <= t and fl > 0.01: c2.rrect(*MON, 34, WHITE, 0.85 * fl)
        if tRew <= t < tRew + 0.6: c2.text("REWIND", 540, 925, 72, WHITE, 0.6 * win(t, tRew, tRew + 0.6, 0.1, 0.25), shadow=True)
        if overlays:
            ruler(c2, t); option_pills(c2, t); law_panel(c2, t)
            draw_screen(c2, cards, t, y=262, sz=78)
            draw_sources(c2, S, T, t)
            end_card(c2, t)
        out = c2.out().astype(np.float32)
        if hook:
            out = paste_hook(out, hook_img, t, tHk, HY, 1.0)
            if t > tHB: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, tHB, tHB + 0.4))
        c3 = Canvas(out)
        if subs: draw_subs(c3, CH, t, D.get("subs_y", 1612))
        return np.clip(c3.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    if (S.get("cover") or {}).get("bg") == "scene":
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(1.2, hook=False, subs=False, overlays=False)); S["_cover_bg"] = p
    ev = [(tG0, "whoosh", -26), (tBomb, "whoosh", -24), (tHit, "thud", -16), (tHit + 0.1, "buzz", -24), (tRew, "whoosh", -22),
          (tSurg, "whoosh", -26), (tFall, "pop", -24), (tLost, "tick", -20), (tZ0, "whoosh", -26), (tL0, "pop", -22), (tHeal, "ding", -20)]
    for i in icons:
        if i["t"] > 0: ev.append((i["t"], "pop", -22))
        if i["rk"] < 1e8: ev.append((i["rk"], "pop", -20))
    for k in range(len(R.get("rows", []))): ev.append((tR0 + 0.3 + 0.9 * k, "pop", -22))
    for ta in (tO1a, tO2a):
        if ta < 1e8: ev.append((ta, "check", -22))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, phases=dict(bomb=round(tBomb, 2), surgery=round(tSurg, 2), zoom=round(tZ0, 2), heal=round(tHeal, 2)))
