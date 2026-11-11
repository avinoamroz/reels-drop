"""two_paths (@noamroz__): a split screen of two sides, a symbolic cutaway of the ground on each half.
Left (Israel): a soldier with a shield in front of a house, a family; a protective dome; a shelter under the house that the
family walks down into. Right (HAMAS): houses with civilians above ground; a tunnel drawn under them with fighters inside
(generic grey stick figures, no ethnic or religious markers). The camera moves between "split", "left" (left half grows to
the full width) and "right". Quote cards (real, sourced quotes) sit in the top area so the scene stays visible under them;
focus rings point at who a quote is about. The hook sits over the split scene from frame 0, and the last 0.5 s dissolves
back into frame 0 (circular). No blood, no bodies, no weapons fired: symbolic visuals only.

data keys:
  views        [{at, view: split | left | right}]   camera keyframes (0.8 s eased moves)
  dome_at      the protective dome over the house appears
  shelter_at   the shelter under the house appears and the family walks down into it
  tunnel_at    the tunnel draws from left to right (2 s); fighters_at: the fighters walk in
  big          [{at, until, value, from, suffix, sub, side, plain}]   a big yellow number over one side (plain = no count up)
  quotes       [{at, until, text, who}]   *word* in yellow, max 12 words, up to 3 lines at 68 px
  focus        [{at, until, target: soldier | family | fighters | civilians}]   pulsing yellow ring
  tags         {at, left, right}   small labels on the shelter and the tunnel (end beat)
  intro_tags   {until, left, right}   side labels from frame 0 (default "IDF" / "HAMAS", until L2)
cover.bg "scene": the cover background is a clean split frame (both sides complete, no text).
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

GY = 1060                     # ground line on screen
DEEP = 300                    # depth of the cutaway (scene units)
INK = "#0E2250"; SOFT = "#C9D6EA"; FOE = "#9AA3B5"; HOUSE_L = "#1C3D72"; HOUSE_R = "#2A3550"
VIEW_S = {"split": 0.5, "left": 1.0, "right": 0.0}


def lerp(a, b, u): return a + (b - a) * u


def wrap_rich(cv, txt, y, a, sz=68, maxw=900, maxlines=3, col=WHITE):
    """*word* in yellow, wraps to up to maxlines (shrinks to fit), centred on y"""
    if a <= 0.01: return
    words = txt.split()
    while True:
        lines = [[]]
        for w in words:
            test = " ".join(x.replace("*", "") for x in lines[-1] + [w])
            if lines[-1] and cv.tw(test, sz) > maxw: lines.append([w])
            else: lines[-1].append(w)
        if len(lines) <= maxlines or sz <= 56: break
        sz -= 4
    lh = sz * 1.2; y0 = y - lh * (len(lines) - 1) / 2; hl = False
    for li, ln in enumerate(lines):
        parts = []
        for w in ln:
            on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
            parts.append((w.replace("*", "") + " ", YEL if (on or hl) else col)); hl = (hl or on) and not off
        parts[-1] = (parts[-1][0].rstrip(), parts[-1][1])
        cv.rich(parts, 540, y0 + li * lh, sz, a, maxw + 60)


def ellipse(cv, x, y, rx, ry, col, a, lw=6, fill=None, fa=0.0):
    c = cv.c; c.save(); c.translate(x, y); c.scale(rx, ry); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    if fill: cv.col(fill, fa); c.fill_preserve()
    cv.col(col, a); c.set_line_width(lw); c.stroke(); c.new_path()


def house(cv, x, w, h, fill, stroke, a, lit=0.9, t=0.0):
    x0 = x - w / 2
    cv.rrect(x0, -h, w, h, 6, fill, a, stroke=stroke, lw=4, sa=0.8 * a)
    c = cv.c; c.new_path(); c.move_to(x0 - 14, -h); c.line_to(x, -h - 0.45 * w); c.line_to(x0 + w + 14, -h); c.close_path()
    cv.col(fill, a); c.fill_preserve(); cv.col(stroke, 0.8 * a); c.set_line_width(4); c.stroke(); c.new_path()
    for k in (-1, 1):
        cv.rrect(x + k * w * 0.25 - 22, -h * 0.72, 44, 40, 4, YEL, a * lit * (0.85 + 0.15 * math.sin(t * 2 + k)))


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.9)); A = T.at
    VK = sorted([(A(v["at"]), VIEW_S[v["view"]]) for v in D.get("views", [{"at": 0, "view": "split"}])])
    t_dome = A(D.get("dome_at"), 1e9); t_shel = A(D.get("shelter_at"), 1e9)
    t_tun = A(D.get("tunnel_at"), 1e9); t_fig = A(D.get("fighters_at"), 1e9)
    BIG = [dict(b, _a=A(b["at"]), _u=A(b.get("until"), T.END - 0.6)) for b in D.get("big", [])]
    QU = [(A(q["at"]), A(q.get("until"), T.END - 0.6), q["text"], q.get("who", "")) for q in D.get("quotes", [])]
    FO = [(A(f["at"]), A(f.get("until"), T.END - 0.6), f["target"]) for f in D.get("focus", [])]
    TG = D.get("tags"); t_tag = A(TG["at"]) if TG else 1e9
    IT = D.get("intro_tags", {}); t_it = A(IT.get("until"), T.S(2) - 0.3 if len(T.L) > 2 else 3.0)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 430)
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 1)) if S.get("end") else 1e9

    def share(t):
        s = VK[0][1]
        for (a0, v0) in VK[1:]:
            if t >= a0: s = lerp(s, v0, ease((t - a0) / 0.8))
        return s

    # ------------------------------------------------------------------ the two scenes (local units: x from the side centre, y up from the ground = negative)
    def fam_pos(t):
        u = ease((t - t_shel - 0.3) / 1.3)
        base = [(80, 0.72), (140, 0.5), (200, 0.72)]
        out = []
        for k, (x, sc) in enumerate(base):
            uk = ease((t - t_shel - 0.3 - 0.18 * k) / 1.2)
            out.append((lerp(x, 100 + 50 * k, uk), lerp(0, DEEP - 82, uk), sc))
        return out, u

    def left_scene(cv, t, a):
        cv.rrect(-700, 0, 1400, DEEP + 40, 0, "#0A1A3C", 0.92 * a)
        for k in range(6): cv.line([(-700, 40 + k * 48), (700, 40 + k * 48)], "#12284F", 2, 0.5 * a)
        cv.line([(-700, 0), (700, 0)], "#3C70B0", 5, a)
        da = seg(t, t_dome, t_dome + 0.8)
        if da > 0.01:
            p = 0.5 + 0.5 * math.sin(t * 2.2); c = cv.c; c.new_path(); c.arc(40, 2, 340 + 8 * p, math.pi, 2 * math.pi)
            cv.col(SKY, 0.10 * da * a); c.fill_preserve(); cv.col(SKY, (0.45 + 0.25 * p) * da * a); c.set_line_width(5); c.stroke(); c.new_path()
        house(cv, 140, 240, 200, HOUSE_L, SKY, a, t=t)
        cv.rrect(118, -86, 44, 86, 4, INK, a)
        sa = seg(t, t_shel - 0.5, t_shel + 0.2)
        if sa > 0.01:
            cv.rrect(25, 70, 250, DEEP - 60, 18, "#2B4F86", sa * a, stroke=SOFT, lw=12, sa=sa * a)
            cv.rrect(45, 90, 210, DEEP - 100, 12, YEL, 0.12 * sa * a)
            for k in range(5): cv.line([(275 + 4, 4 + k * 14), (300, 4 + k * 14)], SOFT, 5, sa * a)      # stairs
            cv.line([(290, 0), (290, 70)], SOFT, 5, sa * a)
        fam, _ = fam_pos(t)
        for k, (x, y, sc) in enumerate(fam): figure(cv, x, y, sc, WHITE, badge=False, a=a, lw=7)
        bob = 3 * math.sin(t * 2.4)
        figure(cv, -200, bob, 1.3, SKY, badge=True, arms="down", a=a, lw=7)
        sx, sy = -122, -128 + bob
        cv.line([(-200, -112 + bob), (-160, -104 + bob), (sx - 18, sy + 8)], SKY, 9, a)
        c = cv.c; c.save(); c.translate(sx, sy); c.rotate(0.12); c.scale(40, 60); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
        cv.col(SKY, a); c.fill_preserve(); cv.col(WHITE, a); c.set_line_width(5); c.stroke(); c.new_path()
        cv.star(sx, sy, 17, WHITE, a)

    def right_scene(cv, t, a):
        cv.rrect(-700, 0, 1400, DEEP + 40, 0, "#0A0F1C", 0.94 * a)
        for k in range(6): cv.line([(-700, 40 + k * 48), (700, 40 + k * 48)], "#141B2C", 2, 0.5 * a)
        cv.line([(-700, 0), (700, 0)], "#5B6478", 5, a)
        house(cv, -200, 200, 170, HOUSE_R, FOE, a, lit=0.6, t=t + 1)
        house(cv, 180, 200, 210, HOUSE_R, FOE, a, lit=0.6, t=t + 2)
        for x, sc in ((-265, 0.7), (-205, 0.48), (-140, 0.7), (130, 0.7), (230, 0.7)):
            figure(cv, x, 0, sc, WHITE, badge=False, a=a, lw=7)
        pr = eout((t - t_tun) / 2.0) if t >= t_tun else 0.0
        if pr > 0.005:
            x0, x1 = -720, lerp(-720, 720, pr)
            cv.rrect(x0, 95, x1 - x0, 150, 40, "#03050B", a, stroke="#5B6478", lw=6, sa=a)
            if pr > 0.5:
                sh = seg(t, t_tun + 1.0, t_tun + 1.4); cv.rrect(-20, -2, 46, 100, 6, "#03050B", sh * a, stroke="#5B6478", lw=5, sa=sh * a)
            for k in range(-8, 9):
                lx = k * 85
                if lx < x1 - 30: cv.circ(lx, 112, 6, YEL, a * (0.55 + 0.35 * math.sin(t * 5 + k * 1.3)))
        fa = seg(t, t_fig, t_fig + 0.6)
        if fa > 0.01:
            off = lerp(-90, 0, eout((t - t_fig) / 1.4))
            for k, x in enumerate((-150, -60, 30, 120)):
                figure(cv, x + off, 236, 0.6, FOE, badge=False, a=fa * a, lw=7, walk=t * 6 + k if t < t_fig + 1.4 else 0.0)

    def place(s, side):
        if side == "left": return W * s / 2, max(0.6, 0.5 + 0.9 * s), (0, W * s)
        return W * s + W * (1 - s) / 2, max(0.6, 0.5 + 0.9 * (1 - s)), (W * s, W)

    def draw_side(cv, t, s, side, fn, tint, ta):
        ox, sc, (xa, xb) = place(s, side)
        if xb - xa < 2: return ox, sc
        c = cv.c; c.save(); c.rectangle(xa, 0, xb - xa, H); c.clip()
        cv.rrect(xa, 0, xb - xa, H, 0, tint, ta)
        c.translate(ox, GY); c.scale(sc, sc); fn(cv, t, 1.0); c.restore()
        return ox, sc

    def ring(cv, t, s, target, a):
        side = "left" if target in ("soldier", "family") else "right"
        ox, sc, (xa, xb) = place(s, side)
        if xb - xa < 40: return
        P = 0.5 + 0.5 * math.sin(t * 5)
        if target == "soldier": pts = [(-190, -110, 130, 190)]
        elif target == "family": pts = [(150, DEEP / 2 + 20, 160, 110)] if t > t_shel + 1.5 else [(140, -50, 120, 90)]
        elif target == "fighters": pts = [(-15, 170, 220, 100)]
        else: pts = [(-205, -50, 105, 80), (180, -50, 95, 80)]
        for x, y, rx, ry in pts:
            ellipse(cv, ox + sc * x, GY + sc * y, sc * rx * (1 + 0.04 * P), sc * ry * (1 + 0.04 * P), YEL, a * (0.7 + 0.3 * P), lw=6)

    def layer(t, text=True):
        cv = Canvas(bars_moving_bg(t, 10)); s = share(t)
        lo, lsc = draw_side(cv, t, s, "left", left_scene, "#1E4A86", 0.22)
        ro, rsc = draw_side(cv, t, s, "right", right_scene, "#000000", 0.30)
        if 0.01 < s < 0.99: cv.line([(W * s, 0), (W * s, H)], WHITE, 4, 0.75)
        for a0, a1, tg in FO:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: ring(cv, t, s, tg, a)
        if not text: return cv.out()
        # side labels from frame 0
        ia = 1 - seg(t, t_it - 0.3, t_it)
        if ia > 0.01:
            if s > 0.2: WG.tag(cv, IT.get("left", "IDF"), lo, 700, 56, SKY, ia * min(1, s * 2))
            if s < 0.8: WG.tag(cv, IT.get("right", "HAMAS"), ro, 700, 56, WHITE, ia * min(1, (1 - s) * 2))
        if TG and t >= t_tag:
            a = seg(t, t_tag, t_tag + 0.4) * (1 - seg(t, T.END - 0.55, T.END - 0.4))
            WG.tag(cv, TG["left"], lo + lsc * 150, GY + lsc * (DEEP + 10), 44, SKY, a)
            WG.tag(cv, TG["right"], ro, GY + rsc * (DEEP + 10), 44, WHITE, a)
        if t > T.E(0) + 0.3:
            fo = 1 - seg(t, END_AT - 0.3, END_AT)
            draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 250), 78) if fo > 0.5 else None
        for b in BIG:
            a = win(t, b["_a"] - 0.1, b["_u"], 0.25, 0.3)
            if a <= 0.01: continue
            cx = lo if b.get("side") == "left" else ro
            v = b["value"] if b.get("plain") else WG.roll(t, b["_a"], b.get("from", 0), b["value"], 1.2)
            txt = str(int(v)) if b.get("plain") else WG.fmtn(v, 0, "", b.get("suffix", ""))
            sz = 190 * (WG.pop_s(t, b["_a"]) if b.get("plain") else 1.0)
            cv.text(txt, cx, 455, max(40, sz), YEL, a, maxw=980)
            if b.get("sub"): cv.text(b["sub"], cx, 572, 46, SOFT, a, bold=True, maxw=980)
        for a0, a1, txt, who in QU:
            a = win(t, a0, a1, 0.25, 0.25)
            if a <= 0.01: continue
            WG.panel(cv, 40, 120, 1000, 500, a, stroke=YEL)
            cv.text("“", 110, 205, 170, SKY, a, maxw=200)
            wrap_rich(cv, "“" + txt + "”", 345, a, 70, 900, 3)
            cv.text(who, 540, 560, 44, SOFT, a, bold=True, maxw=940)
        draw_sources(cv, S, T, t, D.get("src_y", 1540)); end_card(cv, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.95)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1650)); return c2.out()

    def end_card(cv, t):
        """balance line (up to 3 lines) + CTA pill in the top area, so the split scene stays visible under it"""
        E = S.get("end") or {}
        if not E: return
        a = win(t, END_AT - 0.1, T.END - 0.5, 0.35, 0.3)
        if a <= 0.01: return
        WG.panel(cv, 40, 110, 1000, 540, a, stroke=YEL)
        if E.get("balance"): wrap_rich(cv, E["balance"], 290, a, 68, 900, 3)
        if E.get("cta"):
            ca = a * seg(t, END_AT + 0.6, END_AT + 1.0); txt = E["cta"]
            tw = min(960, cv.tw(txt.replace("*", ""), 54) + 80)
            cv.rrect(540 - tw / 2, 500, tw, 100, 50, SKY, ca); wrap_rich(cv, txt, 550, ca, 54, tw - 60, 1)

    def frame(t): return np.clip(layer(t), 0, 255).astype(np.uint8)

    if (S.get("cover") or {}).get("bg") == "scene":
        p = f"{work}/cover_bg.png"; tc = A(D.get("cover_t"), max(T.S(min(13, len(T.L) - 1)) + 1.0, 0.0))
        cv2.imwrite(p, np.clip(layer(tc, text=False), 0, 255).astype(np.uint8)); S["_cover_bg"] = p
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(a0 + 0.05, "whoosh", -24) for a0, _ in VK[1:]]
    ev += [(t_dome, "ding", -30), (t_shel, "thud", -26), (t_tun, "whoosh", -28), (t_fig, "tick", -30)]
    for b in BIG:
        ev.append((b["_a"], "pop", -24))
        if not b.get("plain"): ev.append((b["_a"], "roll", b["_a"] + 1.1))
    ev += [(a0, "thud", -24) for a0, _, _, _ in QU]
    ev = [e for e in ev if e[0] < T.END]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
