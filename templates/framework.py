"""framework (@noamroz__ business lens): a strategy framework diagram that stays on screen the whole reel, filled in beat by
beat. Two layouts:
  "diamond"  nodes at free positions joined by edges (Porter's Diamond: 4 corners + 2 outside forces with dashed links)
  "matrix"   a 2x2 grid (BCG growth-share: Stars, Question marks, Cash cows, Dogs) with items that fly into a quadrant
Frame 0: the hook over the top band and an empty diagram frame; the diagram draws itself in by about 1.5 s (edges grow,
nodes pop). A node or quadrant can glow while the voice talks about it (the rest dims), yellow number chips stick to
nodes so the finished diagram is a summary, and a fact card (big yellow number, counting, plus one line) sits under
the diagram. Real photos play inside the top band (Ken Burns, credit line inside the band) so the framework never leaves
the screen. End card (balance line, optional second line, CTA pill) over the dimmed diagram; circular ending (the last
0.5 s dissolves into frame 0, the hook).

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  layout    "diamond" | "matrix"
  draw      {at: 0.15, dur: 1.25}            diagram draw-in window (default 0.15 s, 1.25 s)
  nodes     diamond: [{id, label: "Factor|conditions", icon, x, y, w, outer}]  label lines split by |, outer = dashed links
  edges     diamond: [[id_a, id_b], ...]  (edges touching an outer node are dashed)
  quads     matrix: 4 x {name, icon, sub, col}  order: top left, top right, bottom left, bottom right
  axes      matrix: {left: "High share", right: "Low share", y: "Market growth"}
  items     matrix: [{label, icon, value, quad, fx, fy, at}]  fx, fy = 0..1 inside the quadrant; value = yellow line
  focus     [{target, at, until}]   target = node id (diamond) or quad index 0..3 (matrix); glow + dim the rest
  chips     [{node, text, at, dx, dy}]  yellow pill under a node (diamond) that stays
  facts     [{at, until, big, text, count: {from, to, dec, prefix, suffix}}]  card under the diagram (one line each)
  photos    [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]   inside the top band
  band      {y0: 170, y1: 560}   top band;  title_y (screen titles, default band centre);  hook_y (default band centre)
  grid      matrix: {x0: 40, x1: 1040, y0: 650, y1: 1200}
  fact_y    default 1335;  src_y default 1500;  subs_y default 1640
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *

LAB = "#C9D6EA"; CARD = "#0B1A3C"; DIMC = "#2A3B5C"
QCOL = {"sky": SKY, "yellow": YEL, "green": "#3DDC84", "grey": "#8FA3C0", "coral": "#F28C6B", "violet": "#9B8CFF"}


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    LAY = D.get("layout", "diamond")
    DR = D.get("draw") or {}; D0 = A(DR.get("at"), 0.15); DD = float(DR.get("dur", 1.25))
    B = D.get("band") or {}; BY0, BY1 = int(B.get("y0", 170)), int(B.get("y1", 560)); BX0, BX1 = 24, 1056
    BW, BH = BX1 - BX0, BY1 - BY0; BCY = (BY0 + BY1) / 2
    HY = D.get("hook_y", BCY); TY = D.get("title_y", BCY); FY = D.get("fact_y", 1335)
    FOC = [dict(f, t0=A(f["at"]), t1=A(f.get("until"), T.END - 0.6)) for f in D.get("focus", [])]
    CHP = [dict(c, t0=A(c["at"])) for c in D.get("chips", [])]
    FAC = [dict(f, t0=A(f["at"]), t1=A(f.get("until"), T.END - 0.6)) for f in D.get("facts", [])]
    from photo_story import _load
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6), im=_load(p["file"], p.get("gray", False))) for p in D.get("photos", [])]
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    E = S.get("end") or {}; END0 = A(E.get("at"), T.END - 3) if E else T.END + 9

    # ------------------------------------------------------------- band (photos)
    msk = np.zeros((BH, BW), np.uint8); r = 30
    cv2.rectangle(msk, (r, 0), (BW - r, BH), 255, -1); cv2.rectangle(msk, (0, r), (BW, BH - r), 255, -1)
    for cx_, cy_ in ((r, r), (BW - r, r), (r, BH - r), (BW - r, BH - r)): cv2.circle(msk, (cx_, cy_), r, 255, -1)
    BM = (msk.astype(np.float32) / 255)[..., None]
    byy = np.arange(BH, dtype=np.float32)[:, None]
    BVIG = (1 - np.clip((byy - BH * 0.55) / (BH * 0.45), 0, 1) ** 1.3 * 0.6)[..., None]

    def band_kb(im, cx, cy, z):
        h_, w_ = im.shape[:2]; s_ = max(BW / w_, BH / h_) * max(1.0, z)
        cw, ch = BW / s_, BH / s_
        x0_ = min(max(cx * w_ - cw / 2, 0), w_ - cw); y0_ = min(max(cy * h_ - ch / 2, 0), h_ - ch)
        M_ = np.array([[s_, 0, -x0_ * s_], [0, s_, -y0_ * s_]], np.float32)
        return cv2.warpAffine(im, M_, (BW, BH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    def ph_a(p, t): return win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)

    def band(base, t):
        img, tot = None, 0.0
        for p in PH:
            a_ = ph_a(p, t)
            if a_ <= 0.001: continue
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            ph = band_kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
            ph = ph * (1 - 0.6 * p.get("dim", 0.2)) * BVIG
            img = ph * a_ if img is None else img * (1 - a_) + ph * a_
            tot = 1 - (1 - tot) * (1 - a_)
        if img is None: return base, 0.0
        reg = base[BY0:BY1, BX0:BX1]; m = BM * tot
        base[BY0:BY1, BX0:BX1] = reg * (1 - m) + (img / max(tot, 1e-3)) * m
        return base, tot

    # ------------------------------------------------------------- diagram
    def focus_level(target, t):
        """(glow 0..1 for this target, dim 0..1 for everything not focused)"""
        g, any_ = 0.0, 0.0
        for f in FOC:
            w_ = win(t, f["t0"], f["t1"], 0.25, 0.3)
            if w_ <= 0: continue
            any_ = max(any_, w_)
            if f["target"] == target: g = max(g, w_)
        return g, any_

    def node_card(cv, n, t, a, sc=1.0):
        x, y = n["x"], n["y"]; w = n.get("w", 300) * sc; h = 128 * sc
        g, anyf = focus_level(n["id"], t); da = a * (1 - 0.55 * anyf * (1 - g))
        if da <= 0.01: return
        if g > 0.01:
            cv.rrect(x - w / 2 - 12, y - h / 2 - 12, w + 24, h + 24, 36, SKY, 0.25 * g * a)
        cv.rrect(x - w / 2, y - h / 2, w, h, 30, CARD, 0.92 * da, stroke=SKY if g > 0.01 or not n.get("outer") else LAB, lw=3 + 3 * g, sa=da)
        lines = n["label"].split("|"); ix = x - w / 2 + 52 * sc
        WG.emo(cv, n.get("icon", "⭐"), ix, y, 58 * sc, da)
        tx = x - w / 2 + 96 * sc; mw = w - 110 * sc
        if len(lines) == 1: cv.text(lines[0], tx, y, 46 * sc, WHITE, da, align="l", maxw=mw, minsz=36)
        else:
            cv.text(lines[0], tx, y - 25 * sc, 44 * sc, WHITE, da, align="l", maxw=mw, minsz=36)
            cv.text(lines[1], tx, y + 25 * sc, 40 * sc, LAB, da, align="l", maxw=mw, minsz=36)

    def chip(cv, txt, x, y, a, p):
        if a <= 0.01: return
        sz = 44; w = cv.tw(txt, sz) + 40; s = min(1.15, p)
        x = min(max(x, w / 2 + 14), W - 14 - w / 2)
        cv.rrect(x - w * s / 2, y - 30 * s, w * s, 60 * s, 30 * s, YEL, a * min(1, p))
        cv.text(txt, x, y, sz * s, "#0B1A3C", a * min(1, p), maxw=w * s, shadow=False, minsz=36)

    NODES = {n["id"]: n for n in D.get("nodes", [])}

    def draw_diamond(cv, t, a):
        NL = list(NODES.values()); EDG = D.get("edges", [])
        for k, (ia, ib) in enumerate(EDG):
            u = seg(t, D0 + DD * 0.5 * k / max(1, len(EDG)), D0 + DD * 0.5 * k / max(1, len(EDG)) + DD * 0.5, eout)
            if u <= 0.001: continue
            na, nb = NODES[ia], NODES[ib]; outer = na.get("outer") or nb.get("outer")
            ga = max(focus_level(ia, t)[0], focus_level(ib, t)[0]); anyf = focus_level(None, t)[1]
            ea = a * (1 - 0.5 * anyf * (1 - ga))
            p0 = (na["x"], na["y"]); p1 = (na["x"] + (nb["x"] - na["x"]) * u, na["y"] + (nb["y"] - na["y"]) * u)
            cv.line([p0, p1], SKY if ga < 0.01 else YEL, 6 if not outer else 4, ea * (0.85 if not outer else 0.7), dash=[14, 12] if outer else None)
        for k, n in enumerate(NL):
            t0 = D0 + DD * 0.35 + DD * 0.65 * k / max(1, len(NL))
            if t < t0: continue
            p = WG.pop_s(t, t0); node_card(cv, n, t, a * min(1, p * 1.4), 0.85 + 0.15 * min(1.1, p))
        for c in CHP:
            if t < c["t0"]: continue
            n = NODES[c["node"]]; p = WG.pop_s(t, c["t0"])
            chip(cv, c["text"], n["x"] + c.get("dx", 0), n["y"] + c.get("dy", 98), a, p)

    G = D.get("grid") or {}; GX0, GX1, GY0, GY1 = G.get("x0", 40), G.get("x1", 1040), G.get("y0", 650), G.get("y1", 1200)
    QW, QH = (GX1 - GX0) / 2, (GY1 - GY0) / 2
    QUADS = D.get("quads", [])
    ITEMS = [dict(it, t0=A(it["at"])) for it in D.get("items", [])]

    def quad_rect(q): return GX0 + (q % 2) * QW, GY0 + (q // 2) * QH

    def draw_matrix(cv, t, a):
        AX = D.get("axes") or {}
        ha = a * seg(t, D0, D0 + 0.4)
        if AX.get("left"): cv.text(AX["left"], GX0 + QW / 2, GY0 - 34, 40, LAB, ha, maxw=QW - 20, minsz=36)
        if AX.get("right"): cv.text(AX["right"], GX0 + QW * 1.5, GY0 - 34, 40, LAB, ha, maxw=QW - 20, minsz=36)
        anyf = focus_level(None, t)[1]
        for q, Q in enumerate(QUADS):
            t0 = D0 + DD * 0.15 * q
            if t < t0: continue
            p = min(1.0, WG.pop_s(t, t0)); x, y = quad_rect(q); col = QCOL.get(Q.get("col", "sky"), Q.get("col", SKY))
            g = focus_level(q, t)[0]; qa = a * p * (1 - 0.55 * anyf * (1 - g))
            pad = 10; s = 0.9 + 0.1 * p
            cx, cy = x + QW / 2, y + QH / 2; w_, h_ = (QW - 2 * pad) * s, (QH - 2 * pad) * s
            if g > 0.01: cv.rrect(cx - w_ / 2 - 8, cy - h_ / 2 - 8, w_ + 16, h_ + 16, 34, col, 0.22 * g * a)
            cv.rrect(cx - w_ / 2, cy - h_ / 2, w_, h_, 28, CARD, 0.9 * qa, stroke=col, lw=3 + 3 * g, sa=qa)
            cv.rrect(cx - w_ / 2, cy - h_ / 2, w_, 12, 6, col, 0.9 * qa)
            # quadrant name: icon + name (top), sub line
            nm = Q["name"]; sz = 52; tw = cv.tw(nm, sz) + 70
            WG.emo(cv, Q.get("icon", "⭐"), cx - tw / 2 + 26, y + 62, 56, qa)
            cv.text(nm, cx - tw / 2 + 64, y + 62, sz, WHITE, qa, align="l", maxw=QW - 110, minsz=36)
            if Q.get("sub"): cv.text(Q["sub"], cx, y + 116, 36, col if col != YEL else YEL, qa * 0.95, bold=False, maxw=QW - 40, minsz=36)
        for it in ITEMS:
            if t < it["t0"]: continue
            q = it["quad"]; x, y = quad_rect(q); u = eout((t - it["t0"]) / 0.55); p = WG.pop_s(t, it["t0"])
            g = focus_level(q, t)[0]; ia = a * min(1, u * 1.5) * (1 - 0.55 * anyf * (1 - g))
            tx, ty = x + QW * it.get("fx", 0.5), y + QH * it.get("fy", 0.62)
            sx, sy = 540, 1350
            px, py = sx + (tx - sx) * u, sy + (ty - sy) * u
            sz = 46; lab = it["label"]; val = it.get("value")
            w_ = max(cv.tw(lab, sz) + 100, (cv.tw(val, 40) + 40) if val else 0); w_ = min(w_, QW - 24); h_ = 104 if val else 72
            s = min(1.1, 0.7 + 0.3 * p)
            cv.rrect(px - w_ * s / 2, py - h_ * s / 2, w_ * s, h_ * s, 22 * s, "#13264D", 0.95 * ia, stroke=WHITE, lw=2, sa=0.35 * ia)
            yy = py - (18 if val else 0) * s
            WG.emo(cv, it.get("icon", "•"), px - w_ * s / 2 + 34 * s, yy, 46 * s, ia)
            cv.text(lab, px - w_ * s / 2 + 66 * s, yy, sz * s, WHITE, ia, align="l", maxw=w_ * s - 80 * s, minsz=36)
            if val: cv.text(val, px, py + 30 * s, 40 * s, YEL, ia, maxw=w_ * s - 20, minsz=36)

    def draw_fact(cv, t, dim):
        for f in FAC:
            a = win(t, f["t0"], f["t1"], 0.25, 0.25) * (1 - dim)
            if a <= 0.01: continue
            big = f.get("big"); c = f.get("count")
            if c:
                v = WG.roll(t, f["t0"] + 0.1, c.get("from", 0), c["to"], c.get("dur", 0.9))
                big = WG.fmtn(v, c.get("dec", 0), c.get("prefix", ""), c.get("suffix", ""))
            has_b = bool(big); h = 230 if has_b else 140; y0 = FY - h / 2
            WG.panel(cv, 40, y0, 1000, h, a)
            if has_b:
                cv.text(big, 540, y0 + 76, 104, YEL, a, maxw=940, minsz=60)
                WG.title(cv, f["text"], y0 + 172, a, 64, maxw=940)
            else: WG.title(cv, f["text"], FY, a, 64, maxw=940)

    def end_card(cv, t):
        if not E: return
        a = win(t, END0 - 0.1, T.END - 0.5, 0.35, 0.3)
        if a <= 0.01: return
        y = E.get("y", 930); b2 = E.get("balance2"); t2 = A(E.get("balance2_at"), END0) if b2 else None
        WG.panel(cv, 40, y - 300, 1000, 640, a)
        WG.title(cv, E["balance"], y - 170, a, 66)
        if b2: WG.title(cv, b2, y + 10, a * seg(t, t2 - 0.1, t2 + 0.3), 58, col=LAB)
        if E.get("cta"):
            ct = A(E.get("cta_at"), END0 + 0.6); ca = a * seg(t, ct, ct + 0.4)
            txt = E["cta"].replace("*", ""); sz = 60; tw = min(cv.tw(txt, sz) + 90, 980)
            cv.rrect(540 - tw / 2, y + 140, tw, 120, 60, SKY, ca)
            cv.text(txt, 540, y + 200, sz, WHITE, ca, maxw=940, minsz=40)

    def frame(t):
        base = bars_moving_bg(t, 10)
        base, pa = band(base, t)
        cv = Canvas(base)
        ba = 1 - pa
        cv.rrect(BX0, BY0, BW, BH, 30, None, 1, stroke=DIMC, lw=3, sa=0.9)
        for k_, p in enumerate(PH):
            a_ = ph_a(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_a(q, t)
            if a_ > 0.01 and p.get("credit"):
                txt = "Photo: " + p["credit"]; tw = min(BW - 30, cv.tw(txt, 36, bold=False) + 30)
                cv.rrect(540 - tw / 2, BY1 - 62, tw, 50, 14, "#0B1322", 0.72 * a_)
                cv.text(txt, 540, BY1 - 37, 36, "#E6EEF8", a_, bold=False, maxw=BW - 40, minsz=36, shadow=False)
        dime = win(t, END0 - 0.1, T.END - 0.5, 0.35, 0.3) if E else 0
        da = 1 - 0.8 * dime
        if LAY == "matrix": draw_matrix(cv, t, da)
        else: draw_diamond(cv, t, da)
        draw_fact(cv, t, dime)
        # screen titles in the band (over a photo they sit on a dark pill)
        for txt, t0, t1, yy, ss in cards:
            if t0 <= T.E(0) - 0.05: continue
            a = win(t, t0 - 0.1, t1, 0.3, 0.3) * (1 - dime)
            if a <= 0.01: continue
            if pa > 0.05: cv.rrect(BX0 + 20, (yy or TY) - 100, BW - 40, 200, 26, "#0B1322", 0.55 * a * pa)
            WG.title(cv, txt.replace("|", " "), yy or TY, a, ss or 72, maxw=960)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); end_card(cv, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(D0, "whoosh", -28)]
    if LAY == "matrix": ev += [(D0 + DD * 0.15 * q, "pop", -30) for q in range(len(QUADS))] + [(it["t0"], "pop", -26) for it in ITEMS]
    else: ev += [(D0 + DD * 0.35 + DD * 0.65 * k / max(1, len(NODES)), "pop", -30) for k in range(len(NODES))]
    ev += [(c["t0"], "ding", -30) for c in CHP] + [(f["t0"], "tick", -28) for f in FAC] + [(p["t0"], "whoosh", -34) for p in PH if p["t0"] > 0.5]
    ev += [(f["t0"], "whoosh", -34) for f in FOC]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, layout=LAY, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
