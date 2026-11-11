"""build_vs_war (@noamroz__): "one hand builds, one hand shields".
Israeli side: three stick figures at laptops, one hand typing, the other holding a shield that deflects rocks and
rockets, a skyline of startups rising behind them, a GDP per person counter. At `split_at` the screen splits: the
Israeli scene moves to the left half and the right half shows generic militia figures (no ethnic or religious markers)
throwing rocks and firing rockets across the line; the right panel steps through the rivals (flag, name, armed group
chip, value, a small bar against Israel's). At `rank_at` a full width ranked bar chart. At `hope_at` the right side
puts the weapons down and gets laptops, schools and hospitals (the hopeful line) while the end card shows. At
`back_at` the full Israeli scene and the hook return, and the last 0.5 s dissolves into frame 0 (loop).

data keys:
  split_at, rank_at, hope_at, back_at, stop_fire_at   anchors
  israel   {label, flag, value, roll_at, pulse_at}      "?" until roll_at, then the counter rolls up; pulses at pulse_at
  unit     small label under the panels
  rivals   [{label, flag, value, group, at, year?}]     shown one after another in the right panel
  rank     {y, rows (WG.bars rows with "at" anchors), note}
  hope_icons  emoji list for the hopeful beat;  hook_y (default 300);  tail (default 1.2)
cover.bg "scene": a clean frame of the opening scene behind the cover text.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

INK = "#0E2250"; LIGHT = "#E8EEF8"; SOFT = "#C9D6EA"; FOE = "#9AA3B5"; FOE_D = "#5B6478"; ROCK = "#8C8577"; WOOD = "#2E5A9A"
GROUND = 1405


def lerp(a, b, u): return a + (b - a) * u


# ------------------------------------------------------------------ the Israeli side
def builder(cv, x, g, u, t, k, hit=0.0, a=1.0):
    """front view, sitting behind a desk: left hand typing behind the laptop lid, right arm up holding a shield.
    Returns the shield centre."""
    dy = g - 74 * u                       # desk top
    neck = (x, dy - 84 * u); head = (x, neck[1] - 30 * u); sh = (x, neck[1] + 12 * u)
    cv.line([(x, dy + 4 * u), neck], SKY, 7 * u, a)                                   # torso
    tap = math.sin(t * 19 + k * 1.7) * 3.2 * u
    cv.line([sh, (x - 26 * u, sh[1] + 34 * u), (x - 12 * u, dy - 6 * u + tap)], SKY, 7 * u, a)   # typing arm
    jolt = hit * 9 * u
    sx, sy = x + 58 * u + jolt, neck[1] - 26 * u - jolt * 0.4
    cv.line([sh, (x + 30 * u, sh[1] - 4 * u), (sx - 8 * u, sy + 6 * u)], SKY, 7 * u, a)  # shield arm
    cv.circ(*head, 23 * u, INK, a, stroke=SKY, lw=7 * u)
    for s in (-1, 1): cv.circ(head[0] + s * 8 * u, head[1] - 3 * u, 2.8 * u, SKY, a)
    cv.c.new_path(); cv.c.arc(head[0], head[1] + 4 * u, 8 * u, 0.2 * math.pi, 0.8 * math.pi); cv.col(SKY, a); cv.c.set_line_width(2.6 * u); cv.c.stroke()
    # desk + laptop (lid back towards us, it hides the typing hand)
    cv.rrect(x - 82 * u, dy, 164 * u, 11 * u, 4 * u, WOOD, a)
    cv.line([(x - 70 * u, dy + 10 * u), (x - 70 * u, g)], WOOD, 6 * u, a); cv.line([(x + 70 * u, dy + 10 * u), (x + 70 * u, g)], WOOD, 6 * u, a)
    glow = 0.5 + 0.5 * math.sin(t * 3 + k)
    cv.rrect(x - 44 * u, dy - 50 * u, 72 * u, 50 * u, 6 * u, SOFT, a)
    cv.circ(x - 8 * u, dy - 25 * u, 6 * u, SKY, a * (0.6 + 0.4 * glow))
    # shield: an upright oval, sky with a white rim and a small star
    c = cv.c; c.save(); c.translate(sx, sy); c.rotate(0.18); c.scale(30 * u, 44 * u); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    cv.col(SKY, a); c.fill_preserve(); cv.col(WHITE, a); c.set_line_width(4.5 * u); c.stroke(); c.new_path()
    cv.star(sx, sy, 13 * u, WHITE, a)
    if hit > 0.01: cv.circ(sx + 26 * u, sy, (18 + 30 * (1 - hit)) * u, None, 1, stroke=YEL, lw=4 * u); cv.c.new_path()
    return (sx + 22 * u, sy)


def skyline(cv, x0, x1, g, grow, t, a=1.0, hmax=560):
    """startup towers rising behind the desks (heights grow with `grow` 0..1)"""
    n = 7; w = (x1 - x0) / n
    hs = [0.55, 0.85, 0.65, 1.0, 0.72, 0.9, 0.6]
    for i in range(n):
        h = hmax * hs[i] * (0.25 + 0.75 * eout(min(1, grow * 1.15 - i * 0.02)))
        bx = x0 + i * w + w * 0.12; bw = w * 0.76; by = g - h
        cv.rrect(bx, by, bw, h, 6, "#1C3D72", 0.92 * a)
        cols = max(2, int(bw // 34)); rows = int(h // 44)
        for r in range(rows):
            for cc in range(cols):
                lit = (math.sin(i * 7.1 + r * 3.3 + cc * 5.7 + t * 0.6) > -0.1)
                cv.rrect(bx + 10 + cc * (bw - 20) / cols, by + 14 + r * 44, (bw - 20) / cols - 8, 22, 3, YEL if lit else "#29508A", (0.55 if lit else 0.5) * a)
        if i in (1, 3, 5): cv.line([(bx + bw / 2, by), (bx + bw / 2, by - 30)], SOFT, 3, a); cv.circ(bx + bw / 2, by - 32, 5, "#FF5A5F", a * (0.5 + 0.5 * math.sin(t * 4 + i)))


def floats(cv, xs, g, u, t, a=1.0):
    """ideas rising out of the laptops: chart, pill, rocket, bulb"""
    em = ["📈", "💊", "🚀", "💡"]
    for k, x in enumerate(xs):
        for j in range(2):
            ph = ((t * 0.55 + k * 0.37 + j * 0.5) % 1.0)
            y = g - 140 * u - ph * 260 * u; al = a * min(1, ph * 5) * (1 - ph)
            WG.emo(cv, em[(k * 2 + j) % 4], x - 8 * u + math.sin(ph * 6 + k) * 10 * u, y, 46 * u / 1.0, al)


# ------------------------------------------------------------------ the other side
def attacker(cv, x, g, u, t, kind, k, a=1.0, calm=0.0):
    """side view facing left. kind: 'throw' (arm swings, releases a rock) or 'launch' (tube on the shoulder).
    calm 0..1: weapons fade, a laptop appears (hope beat). Returns the hand / muzzle point."""
    hip = (x, g - 64 * u); neck = (x, hip[1] - 66 * u); head = (x, neck[1] - 28 * u); sh = (x, neck[1] + 12 * u)
    cv.line([hip, (x - 20 * u, g)], FOE, 7 * u, a); cv.line([hip, (x + 22 * u, g)], FOE, 7 * u, a); cv.line([hip, neck], FOE, 7 * u, a)
    cv.circ(*head, 22 * u, INK, a, stroke=FOE, lw=7 * u)
    if calm < 0.99: cv.rrect(head[0] - 22 * u, head[1] - 9 * u, 44 * u, 12 * u, 5 * u, FOE_D, a * (1 - calm))      # mask band
    cv.circ(head[0] - 9 * u, head[1] - 3 * u, 3 * u, FOE if calm < 0.5 else SKY, a)
    if calm > 0.5:
        cv.c.new_path(); cv.c.arc(head[0] - 4 * u, head[1] + 5 * u, 7 * u, 0.25 * math.pi, 0.8 * math.pi); cv.col(FOE, a); cv.c.set_line_width(2.5 * u); cv.c.stroke()
    if calm > 0.01:   # laptop on a crate in front of him, both hands on it
        lx = x - 62 * u; ly = g - 60 * u
        cv.rrect(lx - 30 * u, ly, 60 * u, 60 * u, 4 * u, "#3B4A66", a * calm)
        cv.rrect(lx - 30 * u, ly - 42 * u, 8 * u, 42 * u, 3 * u, SOFT, a * calm); cv.rrect(lx - 30 * u, ly - 4 * u, 58 * u, 6 * u, 3 * u, SOFT, a * calm)
        cv.line([sh, (x - 26 * u, sh[1] + 30 * u), (lx + 10 * u, ly - 6 * u)], FOE, 7 * u, a * calm)
    if kind == "throw":
        ph = (t * 0.8 + k * 0.31) % 1.0; sw = math.sin(ph * 2 * math.pi)
        ang = math.radians(200 + 70 * sw) if calm < 0.5 else math.radians(120)
        hand = (sh[0] + math.cos(ang) * 50 * u, sh[1] - math.sin(ang) * 50 * u)
        if calm < 0.5:
            cv.line([sh, hand], FOE, 7 * u, a * (1 - calm * 2))
            if sw < 0.3: poly_rock(cv, hand[0], hand[1] - 6 * u, 9 * u, a * (1 - calm * 2))
        cv.line([sh, (x + 18 * u, sh[1] + 40 * u)], FOE, 7 * u, a)
        return hand
    tube0, tube1 = (x + 34 * u, sh[1] - 2 * u), (x - 56 * u, sh[1] - 22 * u)
    if calm < 0.99:
        cv.line([tube0, tube1], "#4D5A42", 15 * u, a * (1 - calm))
        cv.line([sh, (x - 22 * u, sh[1] + 2 * u)], FOE, 7 * u, a * (1 - calm))
    return tube1


def poly_rock(cv, x, y, r, a, rot=0.0):
    pts = [(x + r * (0.8 + 0.25 * math.sin(i * 2.3)) * math.cos(rot + i * 2 * math.pi / 7), y + r * (0.8 + 0.25 * math.cos(i * 1.7)) * math.sin(rot + i * 2 * math.pi / 7)) for i in range(7)]
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path(); cv.col(ROCK, a); c.fill(); c.new_path()


def rocket(cv, x, y, ang, s, a):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.rrect(-22 * s, -6 * s, 40 * s, 12 * s, 6 * s, "#E5484D", a)
    c.new_path(); c.move_to(18 * s, -6 * s); c.line_to(28 * s, 0); c.line_to(18 * s, 6 * s); c.close_path(); cv.col("#E5484D", a); c.fill()
    c.new_path(); c.move_to(-22 * s, -4 * s); c.line_to(-36 * s - 6 * s * math.sin(x * 0.3), 0); c.line_to(-22 * s, 4 * s); c.close_path(); cv.col(YEL, a); c.fill()
    c.restore(); c.new_path()


def foe_town(cv, x0, x1, g, t, a=1.0, grow=0.0):
    """low, dim buildings (not ruins); during the hope beat they grow and light up"""
    n = 5; w = (x1 - x0) / n; hs = [0.32, 0.22, 0.38, 0.26, 0.3]
    for i in range(n):
        h = 520 * (hs[i] + grow * (0.55 - hs[i] * 0.4)); bx = x0 + i * w + w * 0.14; bw = w * 0.72; by = g - h
        cv.rrect(bx, by, bw, h, 5, "#25304A", 0.9 * a)
        for r in range(int(h // 46)):
            for cc in range(2):
                lit = grow > 0.2 and math.sin(i * 5.3 + r * 2.9 + cc * 4.1) > 0.6 - grow * 1.3
                cv.rrect(bx + 10 + cc * (bw - 20) / 2, by + 14 + r * 46, (bw - 20) / 2 - 8, 22, 3, YEL if lit else "#323F5C", (0.6 if lit else 0.6) * a)


# ------------------------------------------------------------------ render
def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.2)); A = T.at
    tS, tR, tH, tB = A(D["split_at"]), A(D["rank_at"]), A(D["hope_at"]), A(D["back_at"])
    tStop = A(D.get("stop_fire_at", D["hope_at"])); END = T.END
    IS = D["israel"]; tPulse = A(IS.get("pulse_at"), 1.0)
    RV = [dict(r, t=A(r["at"])) for r in D["rivals"]]
    RK = D["rank"]; rows = [dict(r, at=A(r["at"])) for r in RK["rows"]]
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300); tHk = T.E(0)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    E = S.get("end") or {}; tEnd = A(E.get("at"), tH)
    # shots: one every 1.15 s from 0.35 s; flight 0.75 s; the impact shakes the target's shield
    shots = [dict(t0=0.35 + 1.15 * k, k=k, fig=(k * 2) % 3, kind="rock" if k % 2 == 0 else "rocket") for k in range(400)]
    shots = [s for s in shots if s["t0"] < tStop - 0.8 and not (tR - 0.9 < s["t0"] < tH)]
    FL = 0.75

    def split_u(t):   # 0 = full Israeli scene, 1 = split
        if t < tS: return 0.0
        if t < tR: return eout((t - tS) / 0.55)
        if t < tH: return 1.0           # (scene hidden during the ranking)
        if t < tB: return 1.0
        return 1 - ease((t - tB) / 0.5)

    def israel_layout(t):
        u = split_u(t); x1 = lerp(1080, 540, u); sc = lerp(1.8, 1.0, u)
        xs = [lerp(185, 92, u), lerp(530, 272, u), lerp(875, 450, u)]
        return u, x1, sc, xs

    def scene(cv, t, a=1.0):
        u, x1, sc, xs = israel_layout(t)
        grow = seg(t, 0.0, 2.2) * 0.75 + 0.25 * seg(t, tPulse - 0.2, tPulse + 0.8)
        skyline(cv, 0, x1, GROUND - 8, grow, t, a, hmax=lerp(640, 470, u))
        cv.line([(0, GROUND), (x1, GROUND)], "#2B4A80", 8, a)
        hits = [0.0, 0.0, 0.0]
        for s in shots:
            dt = t - (s["t0"] + FL)
            if 0 <= dt < 0.35: hits[s["fig"]] = max(hits[s["fig"]], 1 - dt / 0.35)
        shields = [builder(cv, xs[k], GROUND, sc, t, k, hits[k], a) for k in range(3)]
        floats(cv, xs, GROUND, sc, t, a * (0.9 if u < 0.5 else 0.7))
        # the other side
        hu = seg(t, tH, tH + 1.2) if t < tB else 1.0
        fx = []
        if u > 0.01:
            ox = 540 + (1 - u) * 560
            c = cv.c; c.save(); c.rectangle(540, 0, 540, H); c.clip()
            foe_town(cv, ox, ox + 540, GROUND - 8, t, a, hu)
            cv.line([(ox, GROUND), (ox + 540, GROUND)], "#3A4560", 8, a)
            kinds = ["throw", "launch", "throw"]
            for k, xx in enumerate((ox + 125, ox + 300, ox + 465)):
                fx.append(attacker(cv, xx, GROUND, 1.15, t, kinds[k], k, a, hu))
            if hu > 0.05:   # hope icons over the town
                for k, ch in enumerate(D.get("hope_icons", [])):
                    ia = a * WG.pop_s(t, tH + 0.3 + 0.35 * k)
                    if ia <= 0.01: continue
                    WG.emo(cv, ch, ox + 90 + k * 120, GROUND - 470 + 14 * math.sin(t * 2 + k), 84 * min(1.2, ia), min(1, ia))
            c.restore()
            cv.line([(540, 700), (540, GROUND + 20)], YEL, 5, a * u)                          # the line between the sides
        # projectiles (drawn last, across the line)
        for s in shots:
            p = (t - s["t0"]) / FL
            if p < 0 or p > 1.9: continue
            tx, ty = shields[s["fig"]]
            if fx: sx0, sy0 = fx[0] if s["kind"] == "rock" else fx[1]
            else: sx0, sy0 = 1160, GROUND - 420
            if p <= 1:
                x = lerp(sx0, tx, p); y = lerp(sy0, ty, p) - math.sin(p * math.pi) * (150 if s["kind"] == "rock" else 50)
                if s["kind"] == "rock": poly_rock(cv, x, y, 16 * sc / 1.2 + 6, a, rot=p * 9)
                else:
                    dx = (tx - sx0); dy_ = (ty - sy0) - math.cos(p * math.pi) * (50 * math.pi)
                    rocket(cv, x, y, math.atan2(dy_, dx), 0.9 * sc / 1.2 + 0.3, a)
            else:   # bounced off: falls down and away, fades
                q = p - 1; x = tx + 120 * q; y = ty - 90 * q + 420 * q * q
                al = a * max(0, 1 - q / 0.9)
                if s["kind"] == "rock": poly_rock(cv, x, y, 12 * sc / 1.2 + 4, al, rot=q * 14)
                else: rocket(cv, x, y, 1.4 + q * 3, 0.9 * sc / 1.2 + 0.3, al * 0.8)

    def panels(cv, t, a=1.0):
        u = split_u(t)
        tRoll = A(IS.get("roll_at"), 0.15); v = WG.roll(t, tRoll, 0, IS["value"], 1.6)
        pul = 1 + 0.12 * math.sin(math.pi * seg(t, tPulse, tPulse + 0.45, lambda z: z)) if t > tPulse else 1
        x0 = lerp(140, 30, u); w = lerp(800, 500, u)
        WG.panel(cv, x0, 510, w, 200, a)
        WG.flag(cv, IS["flag"], x0 + 60, 562, 56, a); cv.text(IS["label"].upper(), x0 + 104, 562, 48, SKY, a, align="l", maxw=w - 140)
        c = cv.c; c.save(); c.translate(x0 + w / 2, 640); c.scale(pul, pul); c.translate(-(x0 + w / 2), -640)
        cv.text(WG.fmtn(v, 0, "$") if t >= tRoll else "?", x0 + w / 2, 640, lerp(104, 84, u), YEL, a, maxw=w - 40); c.restore()
        if u > 0.01:   # bars under the panels: Israel full, rival tiny
            cv.rrect(x0 + 20, 732, (w - 40) * eout(u), 22, 11, SKY, a)
        if u > 0.01:
            rx = 1080 - 30 - 500 + (1 - u) * 560
            WG.panel(cv, rx, 510, 500, 200, a, stroke="#9AA3B5")
            cur = None
            for r in RV:
                if t >= r["t"] - 0.05: cur = r
            if cur is None:
                cv.text("?", rx + 250, 610, 110, YEL, a)
            else:
                ca = a * seg(t, cur["t"] - 0.05, cur["t"] + 0.25)
                if cur.get("flag"): WG.flag(cv, cur["flag"], rx + 60, 562, 56, ca)
                nm = cur["label"].upper() + (f" ({cur['year']})" if cur.get("year") else "")
                cv.text(nm, rx + (104 if cur.get("flag") else 40), 562, 48, SOFT, ca, align="l", maxw=420)
                cv.text(WG.fmtn(WG.roll(t, cur["t"], 0, cur["value"], 0.8), 0, "$"), rx + 250, 640, 84, YEL, ca, maxw=460)
                bw = max(8, 460 * cur["value"] / IS["value"] * eout((t - cur["t"]) / 0.6))
                cv.rrect(rx + 20, 732, bw, 22, 11, "#E5484D", ca)
                WG.tag(cv, cur["group"], 810, 830, 44, WHITE, ca * (1 - seg(t, tH - 0.3, tH)), bg="#5B2230")
            if RV and t < RV[0]["t"]: cv.text(D.get("unit", ""), 810, 830, 40, SOFT, a * u, bold=False)

    def end_card(cv, t):   # common.draw_end, but it gives way to the closing scene at back_at
        a = win(t, tEnd - 0.1, tB + 0.1, 0.35, 0.35)
        if a <= 0.01: return
        y = E.get("y", 1010); WG.panel(cv, 60, y - 230, 960, 460, a)
        if E.get("balance"): WG.title(cv, E["balance"], y - 90, a, 66)
        if E.get("cta"):
            ca = a * seg(t, tEnd + 0.6, tEnd + 1.0); tw = min(940, cv.tw(E["cta"].replace("*", ""), 60) + 90)
            cv.rrect(540 - tw / 2, y + 60, tw, 120, 60, SKY, ca); WG.title(cv, E["cta"], y + 120, ca, 60)

    def layer(t, hook=True, subs=True, overlays=True):
        cv = Canvas(bars_moving_bg(t, 8))
        sa = max(1 - seg(t, tR - 0.25, tR + 0.1), seg(t, tH - 0.3, tH + 0.1))
        if sa > 0.01: scene(cv, t, sa)
        if overlays:
            pa = sa * (1 - seg(t, tEnd - 0.3, tEnd + 0.1) + seg(t, tB - 0.2, tB + 0.3)) if t > tEnd - 0.3 else sa
            if pa > 0.01: panels(cv, t, min(1, pa))
            ra = win(t, tR - 0.1, tH, 0.3, 0.3)
            if ra > 0.01:
                WG.bars(cv, rows, t, ra, RK.get("y", 600), rowh=128, vmax=IS["value"], lab_w=370)
                if RK.get("note"): cv.text(RK["note"], 540, RK.get("y", 600) + 128 * len(rows) + 10, 40, SOFT, ra, bold=False)
            draw_screen(cv, cards, t, y=270, sz=80)
            draw_sources(cv, S, T, t)
            end_card(cv, t)
        out = cv.out().astype(np.float32)
        if hook:
            out = paste_hook(out, hook_img, t, tHk, HY, 1.0)
            if t > tB: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, tB, tB + 0.4))
        c2 = Canvas(out)
        if subs: draw_subs(c2, CH, t, D.get("subs_y", 1612))
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    if (S.get("cover") or {}).get("bg") == "scene":
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(2.6, hook=False, subs=False, overlays=False)); S["_cover_bg"] = p
    ev = [(tS, "whoosh", -24), (tR, "whoosh", -24), (tH, "ding", -22), (tB, "whoosh", -26), (tPulse, "pop", -18)]
    for s in shots: ev.append((s["t0"] + FL, "thud", -27))
    for r in RV: ev.append((r["t"], "pop", -20))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, shots=len(shots), phases=dict(split=round(tS, 2), rank=round(tR, 2), hope=round(tH, 2), back=round(tB, 2)))
