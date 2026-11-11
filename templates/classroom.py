"""classroom (@noamroz__ story + data): a cartoon classroom where every kid is a country (small flag on the desk, name
under it). A board at the front carries the hook at frame 0, then the class question. One kid (the hero, "hl") gets
picked on by labelled troublemakers in the back row (paper balls, spitballs, paper planes, a paper supplier) while he
types with one hand and blocks with a book in the other, and still passes notes (emoji) to classmates. Then the board
zooms to full screen: a stat card, a ranking (sorted best first, the hero's slot waits until its anchor, "no data" pills,
never drop a row), a two-number verdict, and the board zooms back for the end card; the last 0.5 s dissolves into
frame 0 (loop). Classmates are neutral and never mocked; troublemakers are groups or regimes only, never peoples.

data keys:
  kids      [{label, flag, row: "mid"|"front", x, act, hl}]   act = emoji bubble during the "normal day" (e.g. "📖")
  title     [{at, l1, l2}]          chalk text on the small board after the hook (l2 *word* in yellow)
  day_at    anchor where the classmates' activity bubbles start;  hero_at  spotlight on the hero
  bullies   [{label, x, at, kind: paper|spit|plane|supply}]  label tag above the head, pops in at `at`
  hands     {homework: anchor, block: anchor}   small tags at the hero's hands
  notes     [{at, emo, to}]          a note flies from the hero to kid label `to`, emoji bubble stays a moment
  reveal    {at, back}               board zooms to full screen at `at`, back to the classroom at `back`
  stat      {at, until, title, value, dec, suffix, count_at, who, flag, pill, pill_at}
  ranking   {at, until, title, sub, rows: [{label, flag, rank|null, at, hl, badge, badge_at}]}  (rows sorted by rank, null last)
  verdict   {at, until, items: [{big, sub}]}
  homework_at  anchor where the troublemakers drop the paper and hold books (end message)
  tail      seconds after the last word (default 1.0)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *
from pun_sequence import roth, poly, LIGHT, SOFT, INK

BOARD = "#10283F"; WOOD = "#C9A15B"; WOOD_D = "#8A6A3A"; KID = "#C9D6EA"; BULLY = "#F2A93B"; CHALK = "#F5F5F7"
ROWS = {"mid": dict(y=985, sc=1.55), "front": dict(y=1372, sc=1.7)}
BACK_Y, BACK_SC = 765, 1.3
R0 = (90, 120, 900, 340); R1 = (40, 170, 1000, 1010)   # board x, y, w, h (classroom / zoomed)


def lerp(a, b, u): return a + (b - a) * u


def desk(cv, x, y, w, a):
    cv.rrect(x - w / 2 + 12, y - 48, w - 24, 92, 10, WOOD_D, a)
    cv.rrect(x - w / 2, y - 62, w, 20, 8, WOOD, a)


def book(cv, x, y, ang, s, a, col="#2B6CB0"):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    cv.rrect(-46 * s, -34 * s, 92 * s, 68 * s, 6 * s, col, a, stroke=LIGHT, lw=3 * s)
    cv.line([(0, -32 * s), (0, 32 * s)], LIGHT, 3 * s, a * 0.8)
    for k in range(3): cv.line([(-36 * s, (-16 + k * 14) * s), (-10 * s, (-16 + k * 14) * s)], LIGHT, 2 * s, a * 0.6)
    c.restore()


def laptop(cv, x, y, s, a, t):
    poly(cv, [(x - 52 * s, y), (x + 52 * s, y), (x + 62 * s, y + 10 * s), (x - 62 * s, y + 10 * s)], "#9FB0C8", a)
    cv.rrect(x - 46 * s, y - 64 * s, 92 * s, 62 * s, 6 * s, "#2B2F38", a)
    cv.rrect(x - 40 * s, y - 58 * s, 80 * s, 50 * s, 3 * s, SKY, a * (0.75 + 0.15 * math.sin(t * 6)))
    for k in range(3): cv.line([(x - 32 * s, y - (48 - k * 13) * s), (x + (10 - 12 * (k % 2)) * s, y - (48 - k * 13) * s)], WHITE, 3 * s, a * 0.8)


def paper_ball(cv, x, y, r, a):
    cv.circ(x, y, r, LIGHT, a, stroke="#9FB0C8", lw=2)
    cv.line([(x - r * 0.5, y - r * 0.2), (x, y + r * 0.1), (x + r * 0.4, y - r * 0.4)], "#9FB0C8", 2, a)


def plane(cv, x, y, ang, s, a):
    c = cv.c; c.save(); c.translate(x, y); c.rotate(ang)
    poly(cv, [(30 * s, 0), (-24 * s, -16 * s), (-12 * s, 0), (-24 * s, 16 * s)], LIGHT, a, stroke="#9FB0C8", lw=2)
    c.restore()


def bubble(cv, emo, x, y, a, s=1.0):
    if a <= 0.01: return
    cv.rrect(x - 50 * s, y - 44 * s, 100 * s, 88 * s, 30 * s, LIGHT, a * 0.95)
    poly(cv, [(x - 10 * s, y + 40 * s), (x + 8 * s, y + 40 * s), (x - 4 * s, y + 60 * s)], LIGHT, a * 0.95)
    WG.emo(cv, emo, x, y, 60 * s, a)


def arc_pt(p0, p1, u, h):
    return (lerp(p0[0], p1[0], u), lerp(p0[1], p1[1], u) - h * 4 * u * (1 - u))


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.0)); A = T.at
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); tH = T.E(0)
    kids = []
    for k in D["kids"]:
        r = ROWS[k.get("row", "front")]; kids.append(dict(k, y=r["y"], sc=r["sc"], w=200 if k.get("row") == "front" else 230))
    hero = next(k for k in kids if k.get("hl")); KX = {k["label"]: k for k in kids}
    t_day = A(D.get("day_at"), 2.0); t_hero = A(D.get("hero_at"), 6.0)
    titles = [(A(x["at"]), x["l1"], x["l2"]) for x in D.get("title", [])]
    bullies = [dict(b, t0=A(b["at"])) for b in D.get("bullies", [])]
    H_ = D.get("hands", {}); t_hw = A(H_.get("homework")) if H_.get("homework") else None; t_bl = A(H_.get("block")) if H_.get("block") else None
    notes = [dict(n, t0=A(n["at"])) for n in D.get("notes", [])]
    RV = D.get("reveal", {}); t_rv = A(RV.get("at"), T.END); t_bk = A(RV.get("back"), T.END)
    t_home = A(D.get("homework_at"), t_bk)
    ST = D.get("stat"); RK = D.get("ranking"); VD = D.get("verdict")
    if ST: ST = dict(ST, t0=A(ST["at"]), t1=A(ST.get("until"), t_bk), tc=A(ST.get("count_at", ST["at"])), tp=A(ST.get("pill_at", ST["at"])))
    if RK:
        rows = sorted(RK["rows"], key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
        RK = dict(RK, rows=[dict(r, t0=A(r["at"]), tb=A(r["badge_at"]) if r.get("badge_at") else None) for r in rows], t0=A(RK["at"]), t1=A(RK.get("until"), t_bk))
    if VD: VD = dict(VD, t0=A(VD["at"]), t1=A(VD.get("until"), t_bk))
    t_stop = t_rv - 0.2                       # throwing stops when the board zooms

    # hero geometry (shared by drawing and projectile targets)
    hx, hy, hsc = hero["x"], hero["y"], hero["sc"]
    def hero_pts(t):
        u = hsc; leg, tor, hr = 56, 58, 24
        hip = (hx, hy - leg * 0.45 * u); neck = (hx, hip[1] - tor * u); sh = (hx, neck[1] + 10 * u)
        blk = seg(t, t_hero - 0.2, t_hero + 0.4) if t < t_bk else 1 - seg(t, t_home, t_home + 0.5)
        lh = (lerp(-26, -46, blk), lerp(40, -62, blk)); rh = (30 + 3 * math.sin(t * 22), 34 + 3 * abs(math.sin(t * 17)))
        return dict(sh=sh, lh=lh, rh=rh, blk=blk, book=(sh[0] + lh[0] * u - 14 * u, sh[1] + lh[1] * u - 30 * u))

    def bully_hand(b, t):
        x, y, u = b["x"], BACK_Y, BACK_SC
        neck_y = y - 56 * u - 58 * u; sh = (x, neck_y + 10 * u); dirn = 1 if x < hx else -1
        return (sh[0] + dirn * 34 * u, sh[1] - 30 * u)

    def throws(b):
        if b["kind"] == "supply": return []
        per = {"paper": 1.25, "spit": 0.8, "plane": 1.6}[b["kind"]]; dur = {"paper": 0.62, "spit": 0.36, "plane": 0.9}[b["kind"]]
        out, t = [], b["t0"] + 0.35
        while t + dur < t_stop: out.append((t, dur)); t += per
        return out

    TH = {b["label"]: throws(b) for b in bullies}

    def draw_classroom(cv, t, dim=1.0):
        P = hero_pts(t)
        # back row troublemakers
        for b in bullies:
            ba = seg(t, b["t0"] - 0.05, b["t0"] + 0.25) * (1 - seg(t, T.END - 1.6, T.END - 0.7)) * dim
            if ba <= 0.01: continue
            pop = WG.pop_s(t, b["t0"]); sc = BACK_SC * min(1.0, pop)
            dirn = 1 if b["x"] < hx else -1; home = t >= t_home
            if home: hands = ((-30, 38), (30, 38))
            else:
                ph = 0.0
                for (t0, d) in TH[b["label"]]:
                    if t0 - 0.3 <= t <= t0: ph = seg(t, t0 - 0.3, t0)
                hands = ((-dirn * 28, 40), (dirn * lerp(20, 34, ph), lerp(10, -30, ph)))
            R = roth(cv, b["x"], BACK_Y, sc, BULLY, ba, "teen", hands=hands, eyes="dot", mouth="flat" if home else "grin")
            u = R["u"]
            if home: book(cv, b["x"], R["hands"][0][1] - 4 * u, 0, u * 0.9, ba, "#3C70B0")
            elif b["kind"] == "spit":
                hx_, hy_ = R["head"]; cv.line([(hx_ + dirn * 8 * u, hy_ + 8 * u), (hx_ + dirn * 40 * u, hy_ + 18 * u)], LIGHT, 4 * u, ba)
            elif b["kind"] == "supply":
                sx, sy = R["hands"][1]
                for q in range(4): cv.rrect(sx - 24 * u, sy - 6 * u - q * 5 * u, 48 * u, 6 * u, 1, LIGHT, ba, stroke="#9FB0C8", lw=1)
            WG.tag(cv, b["label"], b["x"], R["head"][1] - R["hr"] - 40, 38, WHITE, ba, bg="#7A2E1E")
        # supply: sheets fly from the supplier to the others
        for b in bullies:
            if b["kind"] != "supply" or t < b["t0"] + 0.5 or t > t_stop or t >= t_home: continue
            others = [o for o in bullies if o is not b and t >= o["t0"]]
            if not others: continue
            ph = (t - b["t0"] - 0.5) / 0.9; k = int(ph); u = ph - k; o = others[k % len(others)]
            p0 = bully_hand(b, t); p1 = (o["x"], BACK_Y - 120 * BACK_SC)
            x, y = arc_pt(p0, p1, u, 90); cv.rrect(x - 18, y - 12, 36, 24, 3, LIGHT, dim * math.sin(math.pi * u), stroke="#9FB0C8", lw=1)
        # spotlight
        sa = seg(t, t_hero - 0.2, t_hero + 0.5) * (1 - seg(t, t_rv, t_rv + 0.4)) * dim
        if sa > 0.01:
            for rr, aa in ((260, 0.05), (200, 0.07), (140, 0.08)): cv.circ(hx, hy - 120, rr, YEL, aa * sa)
        # kids
        for k in kids:
            u = k["sc"]; ka = dim * (0.82 if (not k.get("hl") and sa > 0.5) else 1.0)
            if k.get("hl"):
                R = roth(cv, hx, hy, u, SKY, ka, "teen", hands=(P["lh"], P["rh"]), eyes="dot", mouth="flat" if P["blk"] > 0.5 and t < t_home else "smile", sit=True)
                desk(cv, hx, hy, k["w"], ka); laptop(cv, hx + 40 * u, hy - 62, 0.9 * u, ka, t)
                if P["blk"] > 0.05: book(cv, *P["book"], -0.5 * P["blk"], u * 1.05, ka)
            else:
                bob = math.sin(t * 2.4 + k["x"] * 0.01) * 3
                R = roth(cv, k["x"], k["y"], u, KID, ka, "teen", hands=((-26, 40 + bob), (26, 40 - bob)), eyes="dot", mouth="smile", sit=True)
                desk(cv, k["x"], k["y"], k["w"], ka)
            fa = ka * seg(t, A(k.get("flag_at"), 0.0) - 0.05, A(k.get("flag_at"), 0.0) + 0.25)
            if fa > 0.01:
                WG.flag(cv, k["flag"], k["x"], k["y"] - 6, 52 * min(1.0, WG.pop_s(t, A(k.get("flag_at"), 0.0)) or 1.0), fa)
                cv.text(k["label"], k["x"], k["y"] + 70, 36, YEL if k.get("hl") else KID, fa, maxw=k["w"] + 20, minsz=36)
            if k.get("act") and not k.get("hl"):
                t0 = t_day + 0.25 * kids.index(k); ba = win(t, t0, t_hero, 0.25, 0.3) * dim
                bubble(cv, k["act"], k["x"] + 46, R["head"][1] - 90, ba, 0.9)
        # hand tags
        if t_hw is not None:
            a = win(t, t_hw, t_rv, 0.2, 0.3) * dim; WG.tag(cv, "Homework", hx + 190, hy - 250, 44, YEL, a)
        if t_bl is not None:
            a = win(t, t_bl, t_rv, 0.2, 0.3) * dim; WG.tag(cv, "Block", hx - 210, hy - 300, 44, YEL, a)
        # projectiles
        for b in bullies:
            for (t0, d) in TH[b["label"]]:
                if not (t0 <= t <= t0 + d + 0.5): continue
                p0 = bully_hand(b, t0); p1 = P["book"]
                if t <= t0 + d:
                    u = (t - t0) / d; x, y = arc_pt(p0, p1, u, {"paper": 120, "spit": 30, "plane": 60}[b["kind"]])
                    if b["kind"] == "paper": paper_ball(cv, x, y, 16, dim)
                    elif b["kind"] == "spit": cv.circ(x, y, 8, LIGHT, dim)
                    else: plane(cv, x, y, math.atan2(p1[1] - p0[1], p1[0] - p0[0]), 1.3, dim)
                else:
                    u = (t - t0 - d) / 0.5; dirn = -1 if b["x"] < hx else 1
                    x = p1[0] + dirn * 140 * u; y = p1[1] - 60 * u + 380 * u * u; a = dim * (1 - u)
                    if b["kind"] == "paper": paper_ball(cv, x, y, 14, a)
                    elif b["kind"] == "spit": cv.circ(x, y, 7, LIGHT, a)
                    else: plane(cv, x, y, 2.4 * u, 1.2, a)
                    if u < 0.3: WG.emo(cv, "💥", p1[0], p1[1], 70, dim * (1 - u / 0.3))
        # notes from the hero
        for n in notes:
            to = KX[n["to"]]; tn = n["t0"]
            if t < tn: continue
            p0 = (hx, hy - 120); p1 = (to["x"] + 46, to["y"] - 300 * to["sc"] / 1.7)
            u = (t - tn) / 0.7
            if u < 1:
                x, y = arc_pt(p0, p1, u, 160); cv.rrect(x - 26, y - 20, 52, 40, 4, YEL, dim); cv.line([(x - 16, y - 6), (x + 16, y - 6)], "#B5762A", 3, dim)
            else: bubble(cv, n["emo"], p1[0], p1[1], win(t, tn + 0.7, t_rv, 0.15, 0.3) * dim, 0.9)

    def board(cv, t):
        z = seg(t, t_rv, t_rv + 0.6) * (1 - seg(t, t_bk, t_bk + 0.6))
        x, y, w, h = [lerp(a, b, z) for a, b in zip(R0, R1)]
        cv.rrect(x - 14, y - 14, w + 28, h + 28, 22, WOOD, 1.0); cv.rrect(x, y, w, h, 12, BOARD, 1.0)
        cv.rrect(x + 40, y + h + 8, w - 80, 12, 6, WOOD_D, 1.0)
        cv.rrect(x + w - 150, y + h - 6, 46, 14, 4, CHALK, 0.9 * (1 - z))
        if z < 0.05 and t > tH + 0.4:                       # small board text
            for k, (t0, l1, l2) in enumerate(titles):
                t1 = titles[k + 1][0] if k + 1 < len(titles) else t_rv
                a = win(t, max(t0, tH + 0.4), t1, 0.3, 0.2)
                if a > 0.01:
                    cv.text(l1, 540, y + 110, 60, KID, a, maxw=820); WG.title(cv, l2, y + 220, a, 76, maxw=820)
        if z < 0.95: return z
        if ST and ST["t0"] - 0.2 <= t <= ST["t1"] + 0.3:
            a = win(t, ST["t0"] - 0.2, ST["t1"], 0.3, 0.3)
            WG.title(cv, ST["title"], 300, a, 66, maxw=900)
            v = WG.roll(t, ST["tc"], 0, ST["value"], 1.0) if t >= ST["tc"] else 0
            cv.text(WG.fmtn(v, ST.get("dec", 0), "", ST.get("suffix", "")), 540, 580, 230, YEL, a, maxw=900)
            if ST.get("flag"): WG.flag(cv, ST["flag"], 540 - cv.tw(ST["who"], 64) / 2 - 50, 780, 64, a)
            cv.text(ST.get("who", ""), 540 + 10, 780, 64, WHITE, a, maxw=800)
            if t >= ST["tp"]:
                pa = a * min(1, WG.pop_s(t, ST["tp"])); pw = cv.tw(ST["pill"], 76) + 90
                cv.rrect(540 - pw / 2, 900, pw, 130, 65, SKY, pa); cv.text(ST["pill"], 540, 965, 76, WHITE, pa, maxw=900)
        if RK and RK["t0"] - 0.2 <= t <= RK["t1"] + 0.3:
            a = win(t, RK["t0"] - 0.2, RK["t1"], 0.3, 0.3) * (1 - 0.85 * (win(t, VD["t0"], VD["t1"], 0.3, 0.3) if VD else 0))
            WG.title(cv, RK["title"], 255, a, 66, maxw=900); cv.text(RK.get("sub", ""), 540, 325, 44, KID, a, bold=False, maxw=900)
            rowh = 98
            for i, r in enumerate(RK["rows"]):
                if t < r["t0"] - 0.05: continue
                ra = a * seg(t, r["t0"] - 0.05, r["t0"] + 0.2); yy = 420 + i * rowh; p = min(1.0, WG.pop_s(t, r["t0"]))
                if r.get("hl"): cv.rrect(60, yy - rowh / 2 + 6, 960, rowh - 12, 20, YEL, 0.16 * ra)
                if r.get("rank") is None:
                    cv.rrect(80, yy - 26, 150, 52, 26, "#5B6B85", ra); cv.text("no data", 155, yy, 36, WHITE, ra, maxw=140, minsz=36)
                else: cv.text(f"#{r['rank']}", 155, yy, 66 * (0.7 + 0.3 * p), YEL, ra, maxw=150)
                WG.flag(cv, r["flag"], 290, yy, 58, ra); cv.text(r["label"], 340, yy, 64, YEL if r.get("hl") else WHITE, ra, align="l", maxw=430)
                if r.get("badge") and r["tb"] is not None and t >= r["tb"]:
                    ba = ra * min(1, WG.pop_s(t, r["tb"])); bw = cv.tw(r["badge"], 44) + 50
                    cv.rrect(990 - bw, yy - 34, bw, 68, 34, SKY, ba); cv.text(r["badge"], 990 - bw / 2, yy, 44, WHITE, ba, maxw=bw)
        if VD and VD["t0"] - 0.2 <= t <= VD["t1"] + 0.3:
            a = win(t, VD["t0"], VD["t1"], 0.3, 0.3); n = len(VD["items"])
            WG.panel(cv, 70, 470, 940, 440, a, stroke=YEL)
            for i, it in enumerate(VD["items"]):
                cx = 540 + (i - (n - 1) / 2) * 440; p = min(1.0, WG.pop_s(t, VD["t0"] + 0.35 * i))
                cv.text(it["big"], cx, 640, 210 * (0.6 + 0.4 * p), YEL, a * min(1, p * 1.2), maxw=400)
                cv.text(it["sub"], cx, 800, 56, WHITE, a * min(1, p * 1.2), maxw=400)
            if n == 2: cv.text("or", 540, 650, 64, KID, a, maxw=120)
        return z

    def layer(t, hook=True, subs=True, end=True):
        cv = Canvas(bars_moving_bg(t, 8))
        cv.rrect(0, 1430, 1080, 490, 0, "#0A1630", 0.5)               # floor
        z = seg(t, t_rv, t_rv + 0.6) * (1 - seg(t, t_bk, t_bk + 0.6))
        draw_classroom(cv, t, 1.0 - 0.92 * z)
        board(cv, t)
        draw_sources(cv, S, T, t, D.get("src_y", 1262))
        if end: draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        if hook: out = paste_hook(out, hook_img, t, tH, R0[1] + R0[3] / 2, D.get("hook_s", 0.95))
        c2 = Canvas(out)
        if subs: draw_subs(c2, CH, t, D.get("subs_y", 1612))
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    frame = lambda t: layer(t)
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    if (S.get("cover") or {}).get("bg") == "scene":
        ct = A((S.get("cover") or {}).get("t"), t_rv - 1.0)
        p = f"{work}/cover_bg.png"; cv2.imwrite(p, layer(ct, hook=False, subs=False, end=False)); S["_cover_bg"] = p
    ev = [(b["t0"], "pop", -22) for b in bullies]
    for b in bullies:
        for (t0, d) in TH[b["label"]]: ev.append((t0 + d, "tick", -24))
    ev += [(n["t0"], "whoosh", -30) for n in notes] + [(n["t0"] + 0.7, "pop", -26) for n in notes]
    ev += [(t_rv, "whoosh", -24), (t_bk, "whoosh", -26)]
    if ST: ev.append((ST["tp"], "ding", -20))
    if RK: ev += [(r["t0"], "ding" if r.get("hl") else "pop", -20 if r.get("hl") else -26) for r in RK["rows"]]
    if VD: ev.append((VD["t0"], "thud", -20))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, reveal=round(t_rv, 2), back=round(t_bk, 2))
