"""holiday_wheel: a circular calendar. 12 months on a ring turn under a fixed yellow pointer to each beat's date (the
wheel spins from frame 0 and lands on the first date), a LIVE SCOREBOARD at the top (up to 3 counters that tick on
anchors and roll back to 0 in the outro, so the loop is seamless), a centre plate (a real photo clipped to a circle, or
a big emoji on a white plate), a holiday card under the wheel (name, one line, food chip), a rubber-stamp verdict,
a final score card, an end card, hook over the spinning wheel at frame 0, ratchet ticks as months pass, circular ending.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  months    12 names clockwise from the top (default Tishrei ... Elul)
  cy, R     wheel centre y (default 735) and outer radius (default 395)
  counters  [{key, label}]  up to 3 scoreboard tiles, left to right
  ticks     [{key, at}]     +1 on a counter at `at`
  spin      {turns: 2}      frame-0 spin, lands on the first beat's date at beats[0].at
  beats     [{at, until, name, story, food, food_emoji, month, day, photo, credit, kb: [z0, z1], emoji, plate: "white"|"dark",
              stamp: {text ("|" = new line), color: red|green|yellow|sky|grey, at}, grave: true, dot: "meal"|"fast"|"grave"}]
            month (0 based) + day move the wheel; a beat without month keeps the wheel; photo/emoji change the plate.
            grave: the scene greys out, the scoreboard dims, no stamp (for remembrance days).
  final     {at, until, rows: [{key, label}]}  big score card (counts up)
  end       {at, until, balance, cta}
  outro     {at, month, day, dur}   the wheel turns to this date (e.g. 12, 1 = Tishrei 1 a year later), counters roll to 0
  hook_y    default = cy;  card_y default 1300;  stamp_y default 1085;  src_y 1505;  subs_y 1612;  credit_y 1712
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

MONTHS = ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar", "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"]
SCOL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY, grey="#C9D6EA")
DOTC = dict(meal=YEL, fast=SKY, grave="#C9D6EA", kill=RED)


def _square(path, size=560):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"holiday_wheel: cannot read photo {path}")
    h, w = im.shape[:2]; s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
    im = cv2.resize(im[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    rgba = np.dstack([im[..., ::-1], np.full((size, size), 255, np.uint8)])
    return np.ascontiguousarray(rgba)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    MN = D.get("months", MONTHS); CX, CY, R = 540, D.get("cy", 745), D.get("R", 410)
    RP = R * 0.64                                           # plate radius
    ang = lambda m, d: (m + (d - 0.5) / 30.0) * 30.0        # degrees clockwise from the top
    B = []
    for b in D["beats"]:
        b = dict(b); b["t0"] = A(b["at"]); B.append(b)
    for k, b in enumerate(B):
        b["t1"] = A(b["until"]) if b.get("until") else (B[k + 1]["t0"] if k + 1 < len(B) else A((D.get("final") or {}).get("at"), T.END))
        if b.get("photo"): b["img"] = _square(b["photo"])
        if b.get("stamp"): b["st0"] = A(b["stamp"].get("at"), b["t0"] + 1.0)
    # wheel keyframes (angle of the date under the pointer)
    KF = []; first = next(b for b in B if b.get("month") is not None)
    a_first = ang(first["month"], first.get("day", 15))
    turns = (D.get("spin") or {}).get("turns", 2)
    KF.append((0.0, a_first - 360 * turns)); KF.append((first["t0"], a_first))
    last = a_first
    for b in B:
        if b.get("month") is None or b is first: continue
        a = ang(b["month"], b.get("day", 15))
        while a < last - 0.01: a += 360
        KF.append((b["t0"], a)); last = a
    OUT = D.get("outro")
    if OUT:
        o0 = A(OUT["at"]); a = ang(OUT.get("month", 0) % 12, OUT.get("day", 1))
        while a <= last: a += 360
        KF.append((o0, a)); KF[-1] = (o0, a)
        o_dur = float(OUT.get("dur", 1.6))
    else: o0, o_dur = None, 1.0
    SPIN0 = KF[1][0]

    def wheel_angle(t):
        if t <= SPIN0: return KF[0][1] + (KF[1][1] - KF[0][1]) * eout(t / max(0.1, SPIN0)) ** 1.0
        cur = KF[1][1]
        for k in range(2, len(KF)):
            t0, a1 = KF[k]; a0 = KF[k - 1][1]
            d = o_dur if (o0 is not None and k == len(KF) - 1) else min(0.75, max(0.3, t0 - KF[k - 1][0]))
            if t >= t0: cur = a1 + 0 * a0 if t >= t0 + d else a0 + (a1 - a0) * ease((t - t0) / d)
        return cur

    # counters
    CNT = D.get("counters", [])[:3]; TK = [(A(x["at"]), x["key"]) for x in D.get("ticks", [])]
    def cval(key, t):
        v = sum(1 for t0, k in TK if k == key and t >= t0)
        if o0 is not None and t > o0: v = v * (1 - seg(t, o0, o0 + o_dur))
        return v
    def cpop(key, t):
        ds = [t - t0 for t0, k in TK if k == key and 0 <= t - t0 < 0.45]
        return (1 - min(ds) / 0.45) if ds else 0.0

    FN = D.get("final"); F0 = A(FN["at"]) if FN else None; F1 = A(FN.get("until"), T.END) if FN else None
    EN = D.get("end"); E0 = A(EN["at"]) if EN else None; E1 = A(EN.get("until"), T.END - 0.5) if EN else None
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", CY)
    CARD_Y = D.get("card_y", 1325); ST_Y = D.get("stamp_y", 975)

    def grave_at(t):
        return max([win(t, b["t0"], b["t1"], 0.5, 0.5) for b in B if b.get("grave")] + [0])

    def draw_scoreboard(cv, t, a):
        n = len(CNT)
        if not n or a <= 0.01: return
        gap = 14; w = (1000 - gap * (n - 1)) / n; y0, h = 44, 200
        for k, c in enumerate(CNT):
            x = 40 + k * (w + gap); p = cpop(c["key"], t)
            col = c.get("color", YEL)
            cv.rrect(x, y0, w, h, 28, "#0B1A3C", 0.9 * a, stroke=YEL if p > 0 else SKY, lw=2.5 + 4 * p, sa=(0.55 + 0.45 * p) * a)
            cv.text(c["label"], x + w / 2, y0 + 50, 40, "#C9D6EA", a, maxw=w - 24, minsz=36)
            v = cval(c["key"], t); txt = str(int(round(v)))
            cv.text(txt, x + w / 2, y0 + 133, int(112 * (1 + 0.28 * p)), col, a, maxw=w - 20)

    def plate_state(t):
        """(beat for the plate now, previous beat, crossfade 0..1)"""
        pl = [b for b in B if (b.get("photo") or b.get("emoji"))]
        cur, prev = None, None
        for b in pl:
            if t >= b["t0"] - 0.15: prev, cur = cur, b
        if cur is None: return None, None, 1.0
        return cur, prev, seg(t, cur["t0"] - 0.15, cur["t0"] + 0.2)

    def draw_plate(cv, b, t, a):
        if b is None or a <= 0.01: return
        c = cv.c
        if b.get("img") is not None:
            kb = b.get("kb", [1.0, 1.1]); u = (t - b["t0"]) / max(0.5, b["t1"] - b["t0"])
            z = kb[0] + (kb[1] - kb[0]) * min(1.2, max(0, u))
            c.save(); c.new_path(); c.arc(CX, CY, RP - 6, 0, 2 * math.pi); c.clip()
            cv.image(b["img"], CX, CY, (2 * RP / b["img"].shape[0]) * z, a); c.restore(); c.new_path()
        else:
            dark = b.get("plate") == "dark"
            cv.circ(CX, CY, RP - 6, "#1A2433" if dark else "#F5F5F7", a)
            cv.circ(CX, CY, RP - 40, None, 1, stroke="#3A4558" if dark else "#DCE3EE", lw=3); c.new_path()
            WG.emo(cv, b["emoji"], CX, CY, 230, a)

    def draw_wheel(cv, t):
        c = cv.c; rot = wheel_angle(t); spd = 0
        cv.circ(CX, CY, R + 8, "#0B1730", 0.95)
        cur_m = int(((rot % 360) // 30)) % 12
        for m in range(12):
            a0 = math.radians(m * 30 - rot - 90); a1 = math.radians((m + 1) * 30 - rot - 90)
            c.new_path(); c.arc(CX, CY, R, a0, a1); c.arc_negative(CX, CY, RP + 8, a1, a0); c.close_path()
            hl = 1.0 if m == cur_m else 0.0
            cv.col(SKY if hl else ("#173A6E" if m % 2 == 0 else "#122F5C"), 0.95 if hl else 1.0); c.fill()
            c.new_path(); c.move_to(CX + (RP + 8) * math.cos(a0), CY + (RP + 8) * math.sin(a0)); c.line_to(CX + R * math.cos(a0), CY + R * math.sin(a0))
            cv.col("#0B1730", 1.0); c.set_line_width(4); c.stroke()
        # month names, tangent, flipped on the lower half so they stay readable
        rl = (R + RP + 8) / 2 + 22
        for m, name in enumerate(MN):
            phi = (m + 0.5) * 30 - rot; ph = math.radians(phi)
            x, y = CX + rl * math.sin(ph), CY - rl * math.cos(ph)
            nphi = ((phi + 180) % 360) - 180; flip = abs(nphi) > 95
            c.save(); c.translate(x, y); c.rotate(math.radians(nphi + (180 if flip else 0)))
            cv.text(name, 0, (12 if flip else -6) * 0, 44, "#0B1730" if m == cur_m else WHITE, 1.0, maxw=210, minsz=36, shadow=False)
            c.restore()
        # holiday dots (appear once the pointer has visited them)
        for b in B:
            if b.get("month") is None or t < b["t0"]: continue
            phi = math.radians(ang(b["month"], b.get("day", 15)) - rot); rr = RP + 30
            x, y = CX + rr * math.sin(phi), CY - rr * math.cos(phi)
            cv.circ(x, y, 12 * min(1, WG.pop_s(t, b["t0"]) + 0.01), DOTC.get(b.get("dot", "meal"), YEL), 1.0, stroke="#0B1730", lw=3)
        # plate rim
        cv.circ(CX, CY, RP + 4, "#E8EDF5", 1.0)
        cur, prev, u = plate_state(t)
        if cur is None:
            cv.circ(CX, CY, RP - 6, "#F5F5F7", 1.0); WG.emo(cv, "🍽️", CX, CY, 230, 1.0)
        else:
            if prev is not None and u < 1: draw_plate(cv, prev, t, 1.0)
            elif prev is None and u < 1: cv.circ(CX, CY, RP - 6, "#F5F5F7", 1.0); WG.emo(cv, "🍽️", CX, CY, 230, 1 - u)
            draw_plate(cv, cur, t, u if prev is not None or u < 1 else 1.0)
        # pointer
        py = CY - R - 4
        c.new_path(); c.move_to(CX - 34, py - 40); c.line_to(CX + 34, py - 40); c.line_to(CX, py + 18); c.close_path()
        cv.col(YEL, 1.0); c.fill_preserve(); cv.col("#0B1730", 1.0); c.set_line_width(4); c.stroke(); c.new_path()

    def draw_stamp(cv, st, t0, t1, t):
        sa = win(t, t0 - 0.05, t1, 0.08, 0.3)
        if sa <= 0.01: return
        k = 1.0 + 0.9 * (1 - eout((t - t0) / 0.18)) if t < t0 + 0.18 else 1.0
        col = SCOL.get(st.get("color", "red"), st.get("color", RED)); lines = st["text"].split("|")
        sz = st.get("size", 84) * k; ang_ = math.radians(st.get("angle", -7)); c = cv.c
        c.save(); c.translate(540, ST_Y); c.rotate(ang_); c.translate(-540, -ST_Y)
        tw = min(max(cv.tw(l, sz) for l in lines), 900) + 70; th = sz * (1.25 * len(lines) + 0.35)
        cv.rrect(540 - tw / 2, ST_Y - th / 2, tw, th, 18, "#0B1730", 0.82 * sa, stroke=col, lw=9 * k, sa=sa)
        for i, l in enumerate(lines):
            cv.text(l, 540, ST_Y - sz * 1.25 * (len(lines) - 1) / 2 + i * sz * 1.25, sz, col, sa, maxw=880, shadow=False, minsz=36)
        c.restore()

    def draw_card(cv, b, t, g):
        a = win(t, b["t0"] - 0.1, b["t1"] + 0.05, 0.3, 0.25)
        if F0 is not None: a *= 1 - seg(t, F0 - 0.2, F0 + 0.1)
        if a <= 0.01: return
        y0 = CARD_Y - 155; WG.panel(cv, 40, y0, 1000, 310, a, stroke="#C9D6EA" if b.get("grave") else SKY)
        cv.text(b["name"], 540, y0 + 58, 78, WHITE if b.get("grave") else YEL, a, maxw=930, minsz=40)
        if b.get("story"): WG.title(cv, b["story"], y0 + 158, a, 64, maxw=950)
        if b.get("food"):
            fy = y0 + 262; e = b.get("food_emoji"); tw = cv.tw(b["food"], 50) + (70 if e else 0); x0 = 540 - tw / 2
            if e: WG.emo(cv, e, x0 + 26, fy, 52, a)
            cv.text(b["food"], x0 + (70 if e else 0), fy, 50, SKY if not b.get("grave") else "#C9D6EA", a, align="l", maxw=900, minsz=36)

    def draw_final(cv, t):
        if F0 is None: return
        a = win(t, F0 - 0.1, F1, 0.3, 0.3)
        if a <= 0.01: return
        rows = FN["rows"]; y0 = CY - 330; WG.panel(cv, 60, y0, 960, 660, a, stroke=YEL)
        WG.title(cv, FN.get("title", "Final *score*"), y0 + 85, a, 76)
        for k, r in enumerate(rows):
            ra = a * seg(t, F0 + 0.25 + 0.45 * k, F0 + 0.5 + 0.45 * k); y = y0 + 220 + k * 150
            if ra <= 0.01: continue
            cv.text(r["label"], 110, y, 66, WHITE, ra, align="l", maxw=640)
            v = sum(1 for t0, kk in TK if kk == r["key"] and t0 <= F0 + 0.01)
            vv = WG.roll(t, F0 + 0.25 + 0.45 * k, 0, v, 0.6)
            cv.text(str(int(round(vv))), 930, y, 130, YEL, ra, align="r", maxw=200)

    def draw_endcard(cv, t):
        if E0 is None: return
        a = win(t, E0 - 0.1, E1, 0.35, 0.35)
        if a <= 0.01: return
        y = CY; WG.panel(cv, 60, y - 260, 960, 520, a, stroke=YEL)
        for i, l in enumerate(EN["balance"].split("|")): WG.title(cv, l, y - 140 + i * 100, a, 78)
        if EN.get("cta"):
            ca = a * seg(t, E0 + 0.6, E0 + 1.0); tw = min(cv.tw(EN["cta"].replace("*", ""), 54) + 90, 940)
            cv.rrect(540 - tw / 2, y + 95, tw, 116, 58, SKY, ca); WG.title(cv, EN["cta"], y + 153, ca, 54)

    def frame(t):
        g = grave_at(t)
        cv = Canvas(bars_moving_bg(t, 10))
        draw_wheel(cv, t)
        ha = 1 - seg(t, T.E(0) + 0.15, T.E(0) + 0.55)            # hook legible over the spinning wheel
        if ha > 0.01: cv.circ(CX, CY, R + 10, "#0B1730", 0.55 * ha)
        ov = max([win(t, F0 - 0.1, F1, 0.3, 0.3)] if F0 is not None else [0] + [0])
        if E0 is not None: ov = max(ov, win(t, E0 - 0.1, E1, 0.35, 0.35))
        if ov > 0.01: cv.circ(CX, CY, R + 10, "#0B1730", 0.7 * ov)
        draw_scoreboard(cv, t, 1.0)
        out = cv.out().astype(np.float32)
        if g > 0.01:
            gr = out.mean(axis=2, keepdims=True); out = out * (1 - 0.85 * g) + gr * 0.85 * g; out *= (1 - 0.35 * g)
        cv = Canvas(out)
        for b in B: draw_card(cv, b, t, g)
        for b in B:
            if b.get("stamp") and not b.get("grave"): draw_stamp(cv, b["stamp"], b["st0"], b["t1"] + 0.05, t)
        for b in B:                                            # photo credit
            ca = win(t, b["t0"], b["t1"], 0.25, 0.25)
            if ca > 0.01 and b.get("credit"): cv.text("Photo: " + b["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ca, bold=False, maxw=1000, minsz=36)
        draw_final(cv, t); draw_endcard(cv, t)
        draw_sources(cv, S, T, t, D.get("src_y", 1505))
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.95)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    # SFX: a ratchet tick each time a month line passes the pointer, ding per score, thud per stamp, pop per card
    ev = []; prev = wheel_angle(0.0); tt = 1 / 60
    while tt < T.END:
        a = wheel_angle(tt)
        if math.floor(a / 30) != math.floor(prev / 30): ev.append((tt, "tick", -20))
        prev = a; tt += 1 / 60
    ev += [(t0, "ding", -24) for t0, _ in TK]
    ev += [(b["st0"], "thud", -18) for b in B if b.get("stamp") and not b.get("grave")]
    ev += [(b["t0"], "pop", -30) for b in B if not b.get("grave")]
    if F0 is not None: ev.append((F0, "whoosh", -26))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=b["photo"], credit=b.get("credit")) for b in B if b.get("photo")],
                score={c["key"]: sum(1 for _, k in TK if k == c["key"]) for c in CNT})
