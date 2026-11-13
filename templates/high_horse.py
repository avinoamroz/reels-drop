"""high_horse: "easy to judge from up here". A stick-figure critic on a very tall horse (frame 0, hook over a map of the place
being judged) points down at accusation chips; he climbs down and walks through real cases at ground level (full-bleed real
photos, big yellow number cards, a list card); the close is a vertical "utopia scale" (Utopia at the top, the empty horse
standing on it, then the rows below), an end card, and the loop back to frame 0 (rider on the horse again).

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "END-2"):
  map      {view: [lon, lat, width_deg, centre_y], until, labels: [{text, lon, lat, size}], credit}
  horse    {x, y (feet line), sc, climb: [a, b] (rider climbs down), until (horse leaves), back: {at, x, y, sc} (empty horse returns)}
  chips    [{at, until, big, sub, y, x}]   accusation chips (yellow number line + sub line), right column by default
  bubble   [{at, until, text}]             speech bubble from the rider
  walker   {at, until, y}                  small figure walking left to right at ground level during the cases
  photos   [{file, at, until, credit, kb, dim, gray}]  as photo_story
  big      [{at, until, value, from, prefix, suffix, dec, sub, sub2, head, flag, y, size, txt, roll}]  counting number card
           (txt: fixed text instead; head: country line with its flag on the left; sub2: small grey line)
  listcard [{at, until, title, rows: [{v, label, flag, sub, at}], y, vx, vsize, lsize}]   rows pop in at their anchors (yellow value + white label)
  versus   [{at, until, y0, pitch, left: {title, flag, col, mark: "N", items: [{text, src, at}]}, right: {..., mark: "Y", note, note_at}}]
           split screen, two columns of items with a red cross or green tick each
  ladder   {at, until, rows: [{label, sub ("|" = 2 lines), flag, col, at, hl}], top (y of first row, default 470), pitch (default 150)}
  hook_y default 300; title_y default 300; src_y 1488; subs_y 1612; credit_y 1712
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
from photo_story import _load, _kb

DARK = np.array([30, 18, 11], np.float32)
COLS = dict(green="#22C55E", sky=SKY, yellow=YEL, orange="#F2A93B", red="#EF4444", grey="#9FB0C8")


def horse(cv, x, y, s, a, col=WHITE):
    """tall stick horse facing right, feet on y. returns the saddle point"""
    if a <= 0.01: return (x - 10 * s, y - 440 * s)
    by = y - 420 * s                                      # body line
    for dx, sw in ((-105, -14), (-80, 10), (78, -10), (102, 14)):    # four long legs
        cv.line([(x + dx * s, by + 10 * s), (x + (dx + sw) * s, y - 70 * s), (x + (dx + sw * 0.6) * s, y)], col, 13 * s, a)
    cv.line([(x - 118 * s, by), (x + 104 * s, by)], col, 46 * s, a)                       # body
    cv.line([(x + 92 * s, by - 6 * s), (x + 150 * s, by - 120 * s)], col, 30 * s, a)       # neck
    cv.line([(x + 146 * s, by - 124 * s), (x + 222 * s, by - 78 * s)], col, 32 * s, a)     # head
    cv.line([(x + 140 * s, by - 136 * s), (x + 150 * s, by - 172 * s)], col, 10 * s, a)    # ear
    cv.circ(x + 172 * s, by - 112 * s, 5 * s, "#0E2250", a)                                 # eye
    cv.line([(x + 110 * s, by - 112 * s), (x + 86 * s, by - 60 * s)], "#C9D6EA", 8 * s, a)   # mane
    cv.line([(x - 120 * s, by - 6 * s), (x - 165 * s, by + 40 * s), (x - 170 * s, by + 110 * s)], col, 10 * s, a)   # tail
    cv.rrect(x - 40 * s, by - 32 * s, 64 * s, 16 * s, 6 * s, SKY, a)                        # saddle
    return (x - 8 * s, by - 26 * s)


def bubble(cv, txt, x, y, a, maxw=760, sz=56):
    """speech bubble with a tail pointing to (x, y) from above-right"""
    if a <= 0.01: return
    words = txt.split(); lines = [[]]
    for w in words:
        if lines[-1] and cv.tw(" ".join(lines[-1] + [w]).replace("*", ""), sz) > maxw - 70: lines.append([w])
        else: lines[-1].append(w)
    lh = sz * 1.2; hh = lh * len(lines) + 60; ww = max(cv.tw(" ".join(l).replace("*", ""), sz) for l in lines) + 80
    bx = min(max(x - 60, 40), 1040 - ww); byy = y - 60 - hh
    c = cv.c; c.new_path(); c.move_to(x + 30, byy + hh - 2); c.line_to(x, y - 10); c.line_to(x + 90, byy + hh - 2); c.close_path(); cv.col("#FFFFFF", 0.96 * a); c.fill()
    cv.rrect(bx, byy, ww, hh, 34, "#FFFFFF", 0.96 * a)
    for i, l in enumerate(lines):
        parts = []; hl = False
        for w in l:
            on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
            parts.append((w.replace("*", "") + " ", "#B8860B" if (on or hl) else "#0B1730")); hl = (hl or on) and not off
        parts[-1] = (parts[-1][0].rstrip(), parts[-1][1])
        cv.rich(parts, bx + ww / 2, byy + 30 + lh * (i + 0.5), sz, a, maxw)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    # ---- map (rendered once)
    M_ = D["map"]; MR = MapR(); lon, lat, wd, my = M_["view"]
    cosf = max(0.6, math.cos(math.radians(min(62, abs(lat)))))
    mimg, MM = MR.render(MR.view(lon, lat, W / (wd * cosf), cosf, my), H)
    mimg = mimg.astype(np.float32) * 0.8 + DARK * 0.2
    map_until = A(M_.get("until"), 3.0)
    def map_alpha(t): return 1.0 - seg(t, map_until - 0.35, map_until)
    # ---- horse
    HS = D["horse"]; hx, hy, hs = HS.get("x", 300), HS.get("y", 1250), HS.get("sc", 1.0)
    c0, c1 = A(HS["climb"][0]), A(HS["climb"][1]); h_until = A(HS.get("until"), c1 + 1.0)
    BK = HS.get("back"); b0 = A(BK["at"]) if BK else 1e9
    CHP = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6)) for c in D.get("chips", [])]
    BUB = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("bubble", [])]
    WK = D.get("walker"); wk0, wk1 = (A(WK["at"]), A(WK.get("until"), T.END)) if WK else (1e9, 1e9)
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))) for p in D.get("photos", [])]
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    LC = [dict(l, t0=A(l["at"]), t1=A(l.get("until"), T.END - 0.6), rows=[dict(r, t0=A(r["at"])) for r in l["rows"]]) for l in D.get("listcard", [])]
    VS = []
    for V in D.get("versus", []):
        V = dict(V, t0=A(V["at"]), t1=A(V.get("until"), T.END - 0.6))
        for side in ("left", "right"):
            V[side] = dict(V[side], items=[dict(it, t0=A(it.get("at"), V["t0"] + 0.3 * k)) for k, it in enumerate(V[side]["items"])])
        VS.append(V)
    LD = D.get("ladder")
    if LD:
        LD = dict(LD, t0=A(LD["at"]), t1=A(LD.get("until"), T.END - 0.6)); LD["rows"] = [dict(r, t0=A(r.get("at"), LD["t0"] + 0.2 * k)) for k, r in enumerate(LD["rows"])]
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.62, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    def ph_alpha(p, t): return min(seg(t, p["t0"] - 0.25, p["t0"]), 1 - seg(t, p["t1"] - 0.25, p["t1"]))
    def ph_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.3); return img * (1 - dim) * VIG + DARK * dim
    cards = screen_cards(S, T); SUBS = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    E = S.get("end") or {}; e0 = A(E.get("at"), T.END - 3) if E else 1e9; e1 = A(D.get("end_until"), T.END - 0.5)

    def draw_rider_and_horse(cv, t):
        # horse 1: rider on it, climbs down, horse slides out left
        ha = 1 - seg(t, h_until - 0.4, h_until)
        if ha > 0.01:
            dx = -260 * seg(t, h_until - 0.6, h_until)
            sad = horse(cv, hx + dx, hy, hs, ha)
            u = seg(t, c0, c1, lambda v: v)
            ground = (hx - 250 * hs, hy)
            if u < 0.5:      # still seated, then sliding down
                k = ease(u * 2); px = sad[0] + (ground[0] - sad[0]) * k * 0.3; py = sad[1] + (ground[1] - 200 * hs - sad[1]) * k
                pt = "point" if t < c0 else "up"
                figure(cv, px, py, 1.15 * hs, WHITE, badge=False, arms=pt, sit=True, flip=1, a=ha, lw=8)
            else:            # on the ground, walking off to the right
                k = ease((u - 0.5) * 2); px = sad[0] + (ground[0] - sad[0]) * (0.3 + 0.7 * k); py = ground[1] - 200 * hs * (1 - k)
                wx = px + 420 * seg(t, c1, h_until)
                figure(cv, wx, py, 1.15 * hs, WHITE, badge=False, arms="down", walk=t * 9 if t > c1 else 0, a=ha, lw=8)
        # horse 2: back, empty
        if BK and t >= b0 - 0.3:
            a2 = seg(t, b0 - 0.3, b0) * (1 - seg(t, T.END - 0.45, T.END))
            horse(cv, BK.get("x", 860), BK.get("y", 520), BK.get("sc", 0.33), a2)

    def frame(t):
        base = bars_moving_bg(t) * 0.55 + DARK * 0.45
        ma = map_alpha(t)
        if ma > 0.001: base = base * (1 - ma) + mimg * ma
        pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BIG + LC + VS] + [win(t, e0 - 0.1, e1, 0.3, 0.3) if E else 0, 0])
        if over > 0: base = base * (1 - 0.4 * over)
        cv = Canvas(base); mla = ma * (1 - pa)
        if mla > 0.01:
            for L in M_.get("labels", []):
                x, y = geo2px(MM, L["lon"], L["lat"]); cv.text(L["text"], x, y, L.get("size", 50), L.get("col", WHITE), mla, maxw=500)
        # accusation chips
        for c in CHP:
            ca = win(t, c["t0"] - 0.05, c["t1"], 0.25, 0.3)
            if ca <= 0.01: continue
            k = back(seg(t, c["t0"], c["t0"] + 0.35, lambda v: v), 1.3); x0 = c.get("x", 560); y = c.get("y", 700)
            cc = cv.c; cc.save(); cc.translate(x0 + 230, y); cc.scale(k, k); cc.translate(-(x0 + 230), -y)
            cv.rrect(x0, y - 95, 460, 190, 28, "#0B1730", 0.9 * ca, stroke="#EF4444", lw=3, sa=0.9 * ca)
            cv.text(c["big"], x0 + 230, y - 30, 80, YEL, ca, maxw=420)
            cv.text(c["sub"], x0 + 230, y + 48, 40, WHITE, ca, maxw=430, bold=False, minsz=36)
            cc.restore()
        draw_rider_and_horse(cv, t)
        for b in BUB:
            ba = win(t, b["t0"] - 0.05, b["t1"], 0.2, 0.25)
            if ba > 0.01: bubble(cv, b["text"], hx + 10 * hs, hy - 650 * hs, ba, sz=b.get("size", 56))
        # small walker at ground level during the cases
        if wk0 - 0.3 <= t <= wk1 + 0.3:
            wa = win(t, wk0 - 0.3, wk1 + 0.3, 0.3, 0.3); u = (t - wk0) / max(1, wk1 - wk0)
            figure(cv, 90 + 900 * min(1, max(0, u)), WK.get("y", 1395), 0.8, WHITE, badge=False, arms="down", walk=t * 9, a=wa, lw=7)
        # big number cards
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); tall = bool(g.get("head") or g.get("sub2"))
                top, bot = (y - 250, y + 230) if tall else (y - 175, y + 175)
                cv.rrect(80, top, 920, bot - top, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=4, sa=ga)
                if g.get("head"):
                    hw = min(700, cv.tw(g["head"], 54)); fx = 540 - hw / 2 - 52 if g.get("flag") else None
                    if fx: WG.flag(cv, g["flag"], fx, y - 175, 70, ga)
                    cv.text(g["head"], 540 + (24 if fx else 0), y - 175, 54, WHITE, ga, maxw=700)
                ny = y - (10 if tall else 35)
                if g.get("txt"): cv.text(g["txt"], 540, ny, g.get("size", 150), YEL, ga, maxw=860)
                else:
                    v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.0))
                    k = 1.0 + 0.08 * (1 - eout((t - g["t0"] - g.get("roll", 1.0)) / 0.35)) if t > g["t0"] + g.get("roll", 1.0) else 1.0
                    cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, ny, g.get("size", 170) * k, YEL, ga, maxw=880)
                if g.get("sub"): cv.text(g["sub"], 540, ny + 125, 54, WHITE, ga, maxw=860)
                if g.get("sub2"): cv.text(g["sub2"], 540, ny + 185, 40, "#C9D6EA", ga, maxw=860, bold=False, minsz=36)
        # list card
        for l in LC:
            la = win(t, l["t0"] - 0.1, l["t1"], 0.3, 0.3)
            if la <= 0.01: continue
            rows = l["rows"]; y = l.get("y", 960); hh = 150 + 130 * len(rows); top = y - hh / 2
            WG.panel(cv, 60, top, 960, hh, la, stroke=YEL)
            if l.get("title"): WG.title(cv, l["title"], top + 80, la, 56, col="#C9D6EA")
            for k, r in enumerate(rows):
                ra = la * seg(t, r["t0"] - 0.05, r["t0"] + 0.25); yy = top + 190 + 130 * k
                if ra <= 0.01: continue
                vx = l.get("vx", 470); cv.text(r["v"], vx, yy, l.get("vsize", 92), YEL, ra, align="r", maxw=vx - 90)
                lx = vx + 30
                if r.get("flag"): WG.flag(cv, r["flag"], lx + 30, yy, 56, ra); lx += 70
                cv.text(r["label"], lx, yy, l.get("lsize", 60), WHITE, ra, align="l", maxw=1010 - lx)
                if r.get("sub"): cv.text(r["sub"], lx, yy + 50, 38, "#C9D6EA", ra, align="l", bold=False, maxw=1010 - lx, minsz=36)
        # split screen: two columns of marked items
        for V in VS:
            va = win(t, V["t0"] - 0.1, V["t1"], 0.3, 0.3)
            if va <= 0.01: continue
            top = V.get("y0", 400); P_ = V.get("pitch", 150)
            for ci, (side, x0) in enumerate(((V["left"], 40), (V["right"], 550))):
                col = COLS.get(side.get("col", "red"), side.get("col")); n = len(side["items"])
                cv.rrect(x0, top, 490, 110 + P_ * n + 20, 30, "#0B1730", 0.88 * va, stroke=col, lw=3, sa=0.8 * va)
                if side.get("flag"): WG.flag(cv, side["flag"], x0 + 60, top + 58, 56, va)
                cv.text(side["title"], x0 + (100 if side.get("flag") else 245), top + 58, 54, col, va, align="l" if side.get("flag") else "c", maxw=380)
                for k, it in enumerate(side["items"]):
                    ia = va * seg(t, it["t0"] - 0.05, it["t0"] + 0.25)
                    if ia <= 0.01: continue
                    y = top + 110 + P_ * k + P_ / 2
                    WG.mark(cv, "Y" if side.get("mark", "N") == "Y" else "N", x0 + 48, y, 26, ia, WG.pop_s(t, it["t0"]) if t < it["t0"] + 0.35 else 1)
                    words = it["text"].split(); lines = [[]]
                    for w in words:
                        if lines[-1] and cv.tw(" ".join(lines[-1] + [w]), V.get("fsz", 44)) > 390: lines.append([w])
                        else: lines[-1].append(w)
                    yy = y - (len(lines) - 1) * 26 - (14 if it.get("src") else 0)
                    for j, ln in enumerate(lines): cv.text(" ".join(ln), x0 + 88, yy + 50 * j, V.get("fsz", 44), WHITE, ia, align="l", maxw=395)
                    if it.get("src"): cv.text(it["src"], x0 + 88, yy + 50 * (len(lines) - 1) + 44, 36, "#C9D6EA", ia, align="l", bold=False, maxw=395, minsz=36)
                if side.get("note"):
                    na = va * seg(t, A(side.get("note_at"), V["t0"] + 1.0), A(side.get("note_at"), V["t0"] + 1.0) + 0.3)
                    cv.text(side["note"], 540, top + 110 + P_ * n + 70, 44, WHITE, na, bold=False, maxw=1000, minsz=36)
        # utopia scale
        if LD:
            la = win(t, LD["t0"] - 0.1, LD["t1"], 0.35, 0.35) * (1 - pa)
            if la > 0.01:
                top = LD.get("top", 470); P_ = LD.get("pitch", 150); n = len(LD["rows"])
                ax = 104; cv.line([(ax, top), (ax, top + P_ * (n - 1))], "#3A4E6E", 10, la)
                for k, r in enumerate(LD["rows"]):
                    ra = la * seg(t, r["t0"] - 0.05, r["t0"] + 0.3); y = top + P_ * k
                    if ra <= 0.01: continue
                    col = COLS.get(r.get("col", "grey"), r.get("col"))
                    if r.get("hl"): cv.rrect(50, y - P_ / 2 + 10, 980, P_ - 20, 26, SKY, 0.16 * ra, stroke=SKY, lw=3, sa=0.8 * ra)
                    cv.circ(ax, y, 22 * (0.6 + 0.4 * WG.pop_s(t, r["t0"]) if t < r["t0"] + 0.35 else 1), col, ra, stroke="#FFFFFF", lw=4)
                    subs = [x for x in (r.get("sub") or "").split("|") if x]; ns = len(subs)
                    ty = y - (0 if not ns else 24 + 22 * (ns - 1))
                    lx = 150
                    if r.get("flag"): WG.flag(cv, r["flag"], 180, ty, 52, ra); lx = 222
                    cv.text(r["label"], lx, ty, r.get("size", 58), WHITE if not r.get("hl") else YEL, ra, align="l", maxw=r.get("maxw", 1020 - lx))
                    for j, sl in enumerate(subs): cv.text(sl, lx, ty + 58 + 44 * j, 40, "#C9D6EA", ra, align="l", bold=False, maxw=1020 - lx, minsz=36)
        # photo credits
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1000, minsz=36)
        if mla > 0.01 and M_.get("credit"):
            cv.text(M_["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.8 * mla, bold=False, maxw=1000, minsz=36)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if E:
            a = win(t, e0 - 0.1, e1, 0.35, 0.3)
            if a > 0.01:
                y = E.get("y", 1010); WG.panel(cv, 60, y - 230, 960, 460, a)
                if E.get("balance"): WG.title(cv, E["balance"], y - 90, a, 66)
                if E.get("cta"):
                    cs = E.get("cta_size", 60); ca = a * seg(t, e0 + 0.6, e0 + 1.0); tw = min(960, cv.tw(E["cta"].replace("*", ""), cs) + 90)
                    cv.rrect(540 - tw / 2, y + 60, tw, 120, 60, SKY, ca); WG.title(cv, E["cta"], y + 120, ca, cs, maxw=900)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, SUBS, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -30) for p in PH] + [(c["t0"], "thud", -22) for c in CHP] + [(b["t0"], "pop", -26) for b in BUB]
    ev += [(g["t0"], "roll", g["t0"] + 1.0) for g in BIG if not g.get("txt")] + [(g["t0"] + (0 if g.get("txt") else 1.0), "ding", -26) for g in BIG]
    for l in LC: ev += [(r["t0"], "pop", -26) for r in l["rows"]]
    for V in VS: ev += [(it["t0"], "check" if V[sd].get("mark") == "Y" else "thud", -26) for sd in ("left", "right") for it in V[sd]["items"]]
    if LD: ev += [(r["t0"], "tick", -20) for r in LD["rows"]]
    ev += [(c0, "whoosh", -30)]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
