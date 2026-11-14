"""game_show: a funny "Would you rather?" game show. Three doors on a stage (price tags swinging from frame 0, spotlights
sweeping), the camera zooms into a door and it swings open to show what is behind it (icon, name, big yellow value, small
line); a live scoreboard of pluses (green ticks) and minuses (red crosses) per door; a stylised contestant silhouette
(no face, no real person) at a podium with a red buzzer that flashes "BUZZ!" on every reveal, speech bubbles for the
contestant, rubber stamps (e.g. "HARD PASS"), real photos in a band across the stage (credit inside the band) so the game
show stays on screen, source chips, burned-in subtitles, hook from frame 0, an end card, and a reprise: the doors close,
the camera goes wide and the hook comes back, so the loop into frame 0 is seamless.

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  doors    [{num, tag, icon, name, value, sub, color, open, close}]   exactly 3. tag = price tag text on the closed door,
           icon = emoji behind the door, value = big yellow text, sub = small line. open/close = anchors (door swings).
  cams     [{at, view: "wide" | 0 | 1 | 2, dur}]                       camera moves (zoom 1.5 into a door)
  points   [{door, kind: "Y" | "N", at}]                               scoreboard: +1 tick or +1 cross for that door
  heads    [{icon, name}]                                               scoreboard column headers (default: doors' icon/name)
  buzz     [{at, kind: "buzz" | "wrong" | "ding"}]                      buzzer flash + sound ("wrong" = long game show buzzer)
  bubbles  [{at, until, text}]                                          contestant speech bubble (*word* = gold)
  poses    [{at, pose: idle | excited | recoil | think}]                contestant pose from `at`
  stamp    [{at, until, text, color: red|green|yellow, y, angle, size}] rubber stamp that slams on
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1]}] photo band (y 470 to 950) over the stage
  endcard  {at, until, balance, cta}                                    end card (balance line + CTA pill)
  reprise  anchor: from here the hook comes back over the closed doors (the circular ending)
  hook_y   default 250;  title_y default 285;  subs_y default 1612;  src_y default 1488
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as _CM

STAMP_COL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)
DX = (185.0, 540.0, 895.0); DW, DH, DY0 = 310.0, 560.0, 440.0       # door centres, size, top (stage coords)
FOCUS = (540.0, 745.0); ZOOM = 1.5


def _gs_sfx():
    base = _sfx_lib0(); tt = lambda d: np.arange(int(d * SR)) / SR; rng = np.random.default_rng(11)
    def wrong():                                   # game show "wrong answer" buzzer: two detuned saws, 0.8 s
        x = tt(0.8); s = lambda f: 2 * ((f * x) % 1.0) - 1
        y = (s(146) + s(155) + 0.5 * s(73)) * np.minimum(1, x / 0.01) * np.clip((0.8 - x) / 0.08, 0, 1)
        return y / np.abs(y).max()
    def gbuzz():                                   # short reveal buzzer
        x = tt(0.32); s = lambda f: 2 * ((f * x) % 1.0) - 1
        y = (s(220) + s(227)) * np.minimum(1, x / 0.005) * np.clip((0.32 - x) / 0.05, 0, 1); return y / np.abs(y).max()
    def gding():                                   # bright double ding (a plus point)
        x = tt(0.7); n1 = (np.sin(2 * np.pi * 1568 * x) + 0.4 * np.sin(2 * np.pi * 3136 * x)) * np.exp(-x * 6)
        x2 = np.clip(x - 0.11, 0, None); n2 = (np.sin(2 * np.pi * 2093 * x2) + 0.4 * np.sin(2 * np.pi * 4186 * x2)) * np.exp(-x2 * 6) * (x > 0.11)
        y = 0.8 * n1 + n2; return y / np.abs(y).max()
    def door():                                    # door swing: soft whoosh + low knock
        x = tt(0.5); ph = 2 * np.pi * np.cumsum(220 + 900 * (x / 0.5) ** 2) / SR
        y = (rng.standard_normal(len(x)) * 0.3 + np.sin(ph) * 0.15) * np.sin(np.pi * x / 0.5) ** 2
        y[:int(0.12 * SR)] += np.sin(2 * np.pi * 90 * x[:int(0.12 * SR)]) * np.exp(-x[:int(0.12 * SR)] * 40) * 0.8
        return y / np.abs(y).max()
    def drum():                                    # short snare roll (suspense), 1.0 s
        x = tt(1.0); y = np.zeros_like(x)
        for t0 in np.arange(0, 0.95, 0.05):
            i = int(t0 * SR); n = int(0.06 * SR); seg_ = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 60)
            y[i:i + n] += seg_[:len(y) - i] * (0.35 + 0.65 * t0)
        return y / np.abs(y).max()
    base.update(wrong=wrong, gbuzz=gbuzz, gding=gding, door=door, drum=drum)
    return base


_sfx_lib0 = _CM._sfx_lib


def _load(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"game_show: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z, w, h):
    ih, iw = im.shape[:2]; s = max(w / iw, h / ih) * max(1.0, z); cw, ch = w / s, h / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _wrapped(cv, txt, cx, cy, a, sz, maxw, col=WHITE):
    toks, hl = [], False
    for w in txt.split():
        on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
        toks.append((w.replace("*", ""), hl or on)); hl = (hl or on) and not off
    lines = [[]]
    for tk in toks:
        if lines[-1] and cv.tw(" ".join(x[0] for x in lines[-1] + [tk]), sz) > maxw: lines.append([tk])
        else: lines[-1].append(tk)
    lh = sz * 1.18; y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        parts = [(w + (" " if k < len(ln) - 1 else ""), YEL if h else col) for k, (w, h) in enumerate(ln)]
        tot = sum(cv.tw(p, sz) for p, _ in parts); x = cx - tot / 2
        for p, c in parts: cv.text(p, x, y0 + i * lh, sz, c, a, align="l", maxw=2000, shadow=(col == WHITE)); x += cv.tw(p, sz)
    return len(lines)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    DO = [dict(d, t_open=A(d.get("open"), 1e9), t_close=A(d.get("close"), 1e9)) for d in D["doors"]]
    assert len(DO) == 3, "game_show needs exactly 3 doors"
    CAM = sorted([dict(c, t0=A(c["at"])) for c in D.get("cams", [])], key=lambda c: c["t0"])
    PTS = [dict(p, t0=A(p["at"])) for p in D.get("points", [])]
    HEADS = D.get("heads") or [dict(icon=d.get("icon", ""), name=d.get("name", "")) for d in DO]
    BZ = [dict(b, t0=A(b["at"])) for b in D.get("buzz", [])]
    BUB = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END)) for b in D.get("bubbles", [])]
    POS = sorted([dict(p, t0=A(p["at"])) for p in D.get("poses", [])], key=lambda p: p["t0"])
    STM = [dict(s_, t0=A(s_["at"]), t1=A(s_.get("until"), T.END - 0.6)) for s_ in D.get("stamp", [])]
    PH = [dict(p, t0=A(p["at"]), im=_load(p["file"])) for p in D.get("photos", [])]
    for k, p in enumerate(PH): p["t1"] = A(p["until"]) if p.get("until") else (PH[k + 1]["t0"] if k + 1 < len(PH) else T.END)
    EC = dict(D["endcard"]) if D.get("endcard") else None
    if EC: EC["t0"] = A(EC["at"]); EC["t1"] = A(EC.get("until"), T.END - 0.5)
    REP = A(D.get("reprise"), 1e9)
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 250)
    TY = D.get("title_y", 285)
    BAND = (470, 950)

    # ------------------------------------------------------------ background: studio (static part cached)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    studio = BG.copy()
    glow = np.exp(-(((xx - 540) / 620) ** 2 + ((yy - 760) / 520) ** 2))[..., None]
    studio = studio + glow * np.array([120, 70, 25], np.float32) * 0.55
    floor = np.clip((yy - 1010) / 60, 0, 1)[..., None]
    studio = studio * (1 - 0.35 * floor) + np.array([60, 30, 14], np.float32) * 0.35 * floor
    def beams(t):
        out = np.zeros((H, W), np.float32)
        for k, (x0, ph) in enumerate(((120, 0.0), (960, 1.7), (540, 3.1))):
            ang = 0.32 * math.sin(t * 0.9 + ph)                   # sweeping spotlight from the top
            d = (xx - (x0 + (yy - 0) * math.tan(ang))) / (60 + yy * 0.22)
            out += np.exp(-d ** 2) * np.clip(yy / 700, 0, 1) * np.clip((1400 - yy) / 400, 0, 1)
        return out
    BEAM_T = {}

    def cam_at(t):
        """-> (scale, cx, cy): stage point (cx, cy) lands on the screen point FOCUS (or (540, 745) wide)"""
        def view(v):
            if v == "wide" or v is None: return (1.0, 540.0, 745.0)
            return (ZOOM, DX[int(v)], 730.0)
        cur = view("wide")
        for c in CAM:
            if t < c["t0"]: break
            nxt = view(c["view"]); u = ease((t - c["t0"]) / c.get("dur", 0.7))
            cur = tuple(a + (b - a) * u for a, b in zip(cur, nxt))
        return cur

    def pose_at(t):
        p = "idle"
        for q in POS:
            if t >= q["t0"]: p = q["pose"]
        return p

    # ------------------------------------------------------------ stage drawing (stage coords, under the camera transform)
    def draw_door(cv, k, t):
        d = DO[k]; cx = DX[k]; x0 = cx - DW / 2; y0 = DY0; col = d.get("color", SKY)
        lit = 1.0
        # frame + bulbs
        cv.rrect(x0 - 16, y0 - 16, DW + 32, DH + 30, 22, "#0A1530", 1.0, stroke=col, lw=6, sa=0.9)
        for j in range(9):
            bx = x0 - 6 + j * (DW + 12) / 8; on = (int(t * 6) + j + k) % 3 == 0
            cv.circ(bx, y0 - 16, 7, YEL if on else "#8A7A3A", 1.0)
        # interior (visible as the door opens)
        u = seg(t, d["t_open"], d["t_open"] + 0.55) * (1 - seg(t, d["t_close"], d["t_close"] + 0.45))
        if u > 0.01:
            cv.rrect(x0, y0, DW, DH, 10, "#07102A", 1.0)
            cv.circ(cx, y0 + 170, 120, YEL, 0.10 * u); cv.circ(cx, y0 + 170, 80, YEL, 0.10 * u)
            ia = seg(t, d["t_open"] + 0.2, d["t_open"] + 0.55) * (1 - seg(t, d["t_close"] - 0.05, d["t_close"] + 0.2))
            if d.get("icon"): WG.emo(cv, d["icon"], cx, y0 + 165, 120 * (0.85 + 0.15 * WG.pop_s(t, d["t_open"] + 0.2)), ia)
            if d.get("name"): cv.text(d["name"], cx, y0 + 300, 46, WHITE, ia, maxw=DW - 20, minsz=36)
            if d.get("value"): cv.text(d["value"], cx, y0 + 385, 66, YEL, ia, maxw=DW - 16, minsz=36)
            if d.get("sub"): _wrapped(cv, d["sub"], cx, y0 + 470, ia, 38, DW - 24, "#C9D6EA")
        # the door panel (hinged on the left edge, swings away)
        ang = u * 1.95; w = DW * math.cos(ang); sk = 34 * math.sin(ang)
        if abs(w) > 2 or u > 0.5:
            c = cv.c; c.new_path()
            c.move_to(x0, y0); c.line_to(x0 + w, y0 - sk); c.line_to(x0 + w, y0 + DH + sk); c.line_to(x0, y0 + DH); c.close_path()
            r_, g_, b_ = rgb(col); sh = 0.55 + 0.45 * abs(math.cos(ang))
            c.set_source_rgba(r_ * 0.55 * sh, g_ * 0.55 * sh, b_ * 0.55 * sh + 0.05, 1.0); c.fill_preserve()
            c.set_source_rgba(1, 1, 1, 0.35); c.set_line_width(3); c.stroke()
            if math.cos(ang) > 0.35:                                 # front face details only while it faces us
                fa = min(1.0, (math.cos(ang) - 0.35) / 0.4)
                c.save(); c.translate(x0, y0); c.scale(w / DW, 1.0)
                cv.rrect(28, 34, DW - 56, DH * 0.42, 14, None, stroke="#FFFFFF", lw=3, sa=0.25 * fa)
                cv.rrect(28, 60 + DH * 0.42, DW - 56, DH * 0.42, 14, None, stroke="#FFFFFF", lw=3, sa=0.25 * fa)
                cv.text(str(d.get("num", k + 1)), DW / 2, DH * 0.33, 170, YEL, fa, maxw=DW - 40)
                cv.circ(DW - 40, DH * 0.55, 13, YEL, fa)               # knob
                c.restore()
                # price tag on a string from the knob, swinging
                kx, ky = x0 + w * (DW - 40) / DW, y0 + DH * 0.55
                sw = math.sin(t * 2.6 + k * 1.3) * (0.22 if t < 2.0 else 0.10) + (0.25 * math.exp(-t * 2.5) * math.sin(t * 14 + k))
                L_ = 110; tx, ty = kx + math.sin(sw) * L_ - 90 * (1 - fa) * 0, ky + math.cos(sw) * L_
                cv.line([(kx, ky), (tx, ty - 34)], "#E9E1C8", 3, fa)
                c.save(); c.translate(tx, ty); c.rotate(sw * 0.8)
                tw_ = min(260, cv.tw(d.get("tag", ""), 40) + 40)
                cv.rrect(-tw_ / 2, -34, tw_, 76, 12, "#F4E8C1", fa, stroke="#B8A672", lw=3, sa=fa)
                cv.circ(0, -22, 6, "#0A1530", fa)
                cv.text(d.get("tag", ""), 0, 10, 40, "#163261", fa, maxw=tw_ - 20, shadow=False, minsz=36)
                c.restore()

    def draw_stage(cv, t):
        sc, cx, cy = cam_at(t)
        c = cv.c; c.save(); c.translate(FOCUS[0], FOCUS[1]); c.scale(sc, sc); c.translate(-cx, -cy)
        # stage floor edge
        cv.rrect(-200, DY0 + DH + 24, W + 400, 14, 6, SKY, 0.35)
        for k in range(3):
            u = back(min(1.0, max(0.0, (t - 0.15 * k) / 0.6)), 2.2) if t < 1.6 else 1.0
            s_ = 0.84 + 0.16 * u; c.save(); c.translate(DX[k], DY0 + DH / 2); c.scale(s_, s_); c.translate(-DX[k], -DY0 - DH / 2)
            draw_door(cv, k, t); c.restore()
        c.restore()

    # ------------------------------------------------------------ HUD: contestant, podium, buzzer, scoreboard
    def draw_contestant(cv, t):
        x, base = 175.0, 1300.0; p = pose_at(t)
        bob = math.sin(t * 3.2) * 4
        lean = {"excited": -10, "recoil": 22, "think": 4}.get(p, 0); jump = abs(math.sin(t * 9)) * 12 if p == "excited" else 0
        hx, hy = x + lean * 0.6, base - 200 + bob - jump
        c = cv.c
        for rr, aa in ((118, 0.10), (96, 0.14)): cv.circ(hx, hy + 60, rr, SKY, aa)
        # torso (rounded trapezoid)
        c.new_path(); c.move_to(x - 78, base + 10); c.curve_to(x - 80, base - 90, hx - 60, hy + 50, hx, hy + 44)
        c.curve_to(hx + 60, hy + 50, x + 80, base - 90, x + 78, base + 10); c.close_path()
        c.set_source_rgba(*rgb("#2E6FA8"), 1.0); c.fill_preserve(); c.set_source_rgba(*rgb(SKY), 1.0); c.set_line_width(4); c.stroke()
        # arms
        sh_l, sh_r = (hx - 54, hy + 78), (hx + 54, hy + 78)
        if p == "excited": arms = [(sh_l, (hx - 92, hy - 40)), (sh_r, (hx + 96, hy - 44))]
        elif p == "recoil": arms = [(sh_l, (hx - 70, hy + 10)), (sh_r, (hx + 40, hy + 0))]
        elif p == "think": arms = [(sh_l, (hx - 60, hy + 150)), (sh_r, (hx + 30, hy + 20))]
        else:
            press = any(0 <= t - b["t0"] < 0.35 for b in BZ)
            arms = [(sh_l, (hx - 66, hy + 160)), (sh_r, (x + 120, base - 4 if press else base - 30))]
        for a_, b_ in arms: cv.line([a_, b_], "#2E6FA8", 30, 1.0); cv.line([a_, b_], SKY, 4, 0.7)
        for a_, b_ in arms: cv.circ(b_[0], b_[1], 17, "#2E6FA8", 1.0, stroke=SKY, lw=4)
        # head: plain silhouette, no face
        cv.circ(hx, hy, 52, "#2E6FA8", 1.0, stroke=SKY, lw=5)
        if p == "recoil":                                       # sweat drop
            cv.circ(hx + 62, hy - 30, 9, "#BFE6FF", 0.9)
        # podium
        cv.rrect(x - 130, base - 10, 260, 150, 18, "#0A1530", 0.96, stroke=YEL, lw=4, sa=0.8)
        WG.flag(cv, "IL", x - 70, base + 64, 50, 1.0)
        cv.text("ISRAELI", x + 32, base + 64, 40, WHITE, 1.0, maxw=150, minsz=36)
        # buzzer
        fl = max([math.exp(-(t - b["t0"]) * 4) if t >= b["t0"] else 0 for b in BZ] + [0])
        bx, by = x + 120, base - 18
        if fl > 0.02:
            for rr, aa in ((80, 0.18), (56, 0.25)): cv.circ(bx, by - 8, rr * (0.7 + 0.3 * fl), RED, aa * fl)
        cv.rrect(bx - 40, by - 6, 80, 22, 8, "#3A3A48", 1.0)
        c.new_path(); c.arc(bx, by - 4, 32, math.pi, 2 * math.pi); c.close_path()
        c.set_source_rgba(*rgb("#FF6B6E" if fl > 0.3 else RED), 1.0); c.fill()

    def draw_board(cv, t):
        x0, y0, w, h = 340, 1078, 710, 352
        cv.rrect(x0, y0, w, h, 28, "#0A1530", 0.92, stroke=SKY, lw=3, sa=0.6)
        cv.text("SCOREBOARD", x0 + w / 2, y0 + 30, 36, "#C9D6EA", 1.0, maxw=400, shadow=False, minsz=36)
        cw = w / 3
        for k in range(3):
            cx = x0 + cw * (k + 0.5)
            act = max([win(t, DO[k]["t_open"], DO[k]["t_close"], 0.3, 0.3)] + [0])
            if act > 0.01: cv.rrect(x0 + cw * k + 8, y0 + 54, cw - 16, h - 64, 20, YEL, 0.10 * act, stroke=YEL, lw=3, sa=0.6 * act)
            hd = HEADS[k]
            if hd.get("icon"): WG.emo(cv, hd["icon"], cx, y0 + 104, 50, 1.0)
            cv.text(hd.get("name", ""), cx, y0 + 160, 40, WHITE, 1.0, maxw=cw - 20, minsz=36)
            for row, kind in enumerate(("Y", "N")):
                ps = [p for p in PTS if p["door"] == k and p["kind"] == kind and t >= p["t0"]]
                n = len(ps); yrow = y0 + 230 + row * 82
                last = max([p["t0"] for p in ps] + [-9])
                WG.mark(cv, kind, cx - 46, yrow, 30, 1.0 if n else 0.35, WG.pop_s(t, last) if n and t - last < 0.4 else 1.0)
                bump = 1 + 0.35 * max(0, 1 - (t - last) / 0.35) if n else 1
                cv.text(("+" if kind == "Y" else "-") + str(n), cx + 34, yrow, int(64 * bump), GREEN if kind == "Y" else RED,
                        1.0 if n else 0.45, maxw=120)

    def draw_burst(cv, t):
        c = cv.c; bx, by = 295.0, 1282.0
        for b in BZ:                                            # "BUZZ!" burst
            if 0 <= t - b["t0"] < 0.75:
                u = (t - b["t0"]) / 0.75; s_ = back(min(1, u * 3), 1.8); a_ = 1 - seg(t, b["t0"] + 0.5, b["t0"] + 0.75)
                txt = {"ding": "DING!", "wrong": "BZZZT!"}.get(b.get("kind"), "BUZZ!")
                colr = GREEN if b.get("kind") == "ding" else RED
                c.save(); c.translate(bx + 40, by - 110); c.rotate(-0.15); c.scale(s_, s_)
                cv.rrect(-120, -44, 240, 88, 30, colr, 0.95 * a_)
                cv.text(txt, 0, 0, 58, WHITE, a_, maxw=220, shadow=False); c.restore()

    def draw_bubble(cv, t):
        for b in BUB:
            a = win(t, b["t0"] - 0.05, b["t1"], 0.18, 0.2)
            if a <= 0.01: continue
            s_ = back(seg(t, b["t0"] - 0.05, b["t0"] + 0.3, lambda u: u), 1.6)
            c = cv.c; c.save(); c.translate(540, 980); c.scale(s_, s_); c.translate(-540, -980)
            cv.rrect(60, 880, 960, 200, 44, "#FFFFFF", 0.97 * a, stroke="#163261", lw=4, sa=a)
            c.new_path(); c.move_to(170, 1074); c.line_to(210, 1074); c.line_to(185, 1112); c.close_path()
            c.set_source_rgba(1, 1, 1, 0.97 * a); c.fill()
            _wrapped(cv, b["text"], 540, 980, a, 64, 880, "#163261")
            c.restore()

    def draw_band(cv, t):
        for p in PH:
            a = win(t, p["t0"] - 0.1, p["t1"], 0.25, 0.25)
            if a <= 0.01: continue
            slide = (1 - eout((t - p["t0"] + 0.1) / 0.4)) * 1100 if t < p["t0"] + 0.3 else 0
            y0, y1 = BAND; h = y1 - y0
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            img = _kb(p["im"], *(kb[i] + (kb[i + 3] - kb[i]) * u for i in range(3)), W - 40, h)
            rgba = np.dstack([img[..., ::-1], np.full(img.shape[:2], 255, np.uint8)])
            cv.rrect(10 - slide, y0 - 10, W - 20, h + 20, 26, "#0A1530", a, stroke=YEL, lw=5, sa=a)
            cv.image(rgba, W / 2 - slide, y0 + h / 2, 1.0, a)
            if p.get("credit"):
                cv.rrect(20 - slide, y1 - 58, W - 40, 58, 0, "#000000", 0.55 * a)
                cv.text("Photo: " + p["credit"], 540 - slide, y1 - 29, 36, "#E6ECF5", a, bold=False, maxw=1000, minsz=36, shadow=False)

    def draw_stamps(cv, t):
        for s_ in STM:
            sa = win(t, s_["t0"] - 0.05, s_["t1"], 0.08, 0.3)
            if sa <= 0.01: continue
            k = 1.0 + 0.9 * (1 - eout((t - s_["t0"]) / 0.18)) if t < s_["t0"] + 0.18 else 1.0
            col = STAMP_COL.get(s_.get("color", "red"), s_.get("color", RED)); sz = s_.get("size", 130) * k
            y = s_.get("y", 760); ang = math.radians(s_.get("angle", -9)); txt = s_["text"]
            c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.translate(-540, -y)
            tw = min(cv.tw(txt, sz), 940) + 80
            cv.rrect(540 - tw / 2, y - sz * 0.72, tw, sz * 1.44, 18, "#0B1730", 0.55 * sa, stroke=col, lw=10 * k, sa=sa)
            cv.text(txt, 540, y, sz, col, sa, maxw=900, shadow=False, minsz=36)
            c.restore()

    def draw_titles(cv, t):
        for txt, t0, t1, yy_, ss in cards:
            if t0 < T.E(0) - 0.05: continue
            a = win(t, t0 - 0.1, t1, 0.25, 0.25)
            if a <= 0.01: continue
            y = yy_ or TY; sz = ss or 74
            cv.rrect(30, y - 120, 1020, 240, 36, "#0A1530", 0.78 * a)
            _wrapped(cv, txt.replace("|", " "), 540, y, a, sz, 960)

    def draw_endcard(cv, t):
        if not EC: return
        a = win(t, EC["t0"] - 0.1, EC["t1"], 0.35, 0.3)
        if a <= 0.01: return
        y = EC.get("y", 760)
        WG.panel(cv, 50, y - 250, 980, 500, a, stroke=YEL)
        if EC.get("balance"): _wrapped(cv, EC["balance"], 540, y - 70, a, 70, 900)
        if EC.get("cta"):
            ca = a * seg(t, EC["t0"] + 0.5, EC["t0"] + 0.9)
            tw = min(940, cv.tw(EC["cta"].replace("*", ""), 56) + 90)
            cv.rrect(540 - tw / 2, y + 90, tw, 112, 56, SKY, ca)
            WG.title(cv, EC["cta"], y + 146, ca, 56)

    def frame(t):
        tb = round(t * 15) / 15
        if tb not in BEAM_T:
            if len(BEAM_T) > 64: BEAM_T.clear()
            BEAM_T[tb] = beams(tb)
        base = studio + BEAM_T[tb][..., None] * np.array([200, 225, 255], np.float32) * 0.07
        cv = Canvas(base)
        draw_stage(cv, t)
        dimmer = max([win(t, p["t0"] - 0.1, p["t1"], 0.25, 0.25) for p in PH] + ([win(t, EC["t0"] - 0.1, EC["t1"], 0.35, 0.3)] if EC else []) + [0])
        if dimmer > 0.01: cv.rrect(0, 380, W, 700, 0, "#050B1A", 0.45 * dimmer)
        draw_band(cv, t)
        draw_contestant(cv, t); draw_board(cv, t); draw_burst(cv, t)
        draw_endcard(cv, t)
        draw_bubble(cv, t); draw_stamps(cv, t)
        draw_titles(cv, t)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0), HY, 1.0)
        if t >= REP - 0.3:                                       # reprise: the hook comes back for the loop
            out = HK.paste(out, hook_img, 540, HY, 0.94 + 0.06 * eout((t - REP + 0.3) / 0.3), seg(t, REP - 0.3, REP + 0.1))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    for d in DO:
        if d["t_open"] < 1e8: ev += [(d["t_open"] - 1.0, "drum", -30), (d["t_open"], "door", -24)]
        if d["t_close"] < 1e8: ev += [(d["t_close"], "door", -30)]
    ev += [(b["t0"], {"wrong": "wrong", "ding": "gding"}.get(b.get("kind"), "gbuzz"), -17 if b.get("kind") == "wrong" else -23) for b in BZ]
    ev += [(p["t0"], "gding" if p["kind"] == "Y" else "buzz", -29) for p in PTS]
    ev += [(s_["t0"], "thud", -18) for s_ in STM] + [(p["t0"], "whoosh", -30) for p in PH] + [(b["t0"], "pop", -28) for b in BUB]
    _CM._sfx_lib = _gs_sfx
    try: aud = build_mix(S, T, work, ev)
    finally: _CM._sfx_lib = _sfx_lib0
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
