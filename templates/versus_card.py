"""versus_card (@noamroz__ data story): one live scorecard per country, "<country> today" vs "at Israel's level", on the
same 4 to 6 measures. Each card slides in, its rows fill in (on their own anchors, or quickly one after the other),
a row can glow while the voice talks about it, the "Upgrades" score counts up live (a row counts when Israel's value is
better; "same" rows show a grey = and do not count), and a yellow sticker (e.g. "16x income") pops on the card.
Summary bar cards (sorted best first, "no data" pills) can follow the cards. Circular: the end card clears before the
last line and the last 0.5 s dissolves into frame 0 (the hook).

The contrast is laws and regimes, never peoples: country values are white, Israel's values yellow, no red.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  measures   [{label, icon}]  max 6 (label max 2 words, wraps to 2 lines), icon = emoji
  israel     {label: "Israel", flag: "IL", head: "At Israel's level", values: [txt per measure]}
  cards      [{name, flag, head, at, until, values: [txt | null], cmp: ["up" | "same" | null], row_at: [anchor | null],
              glow: [{row, at, until}], sticker: {txt, sub, at}}]
              values null = "no data" pill; row_at missing = fast fill (0.35 s after the card, 0.16 s apart)
  score      {label: "Upgrades"}   live score in the card header ("5/6")
  bars       [{at, until, title, rows: [{label, flag, value, txt, hl}], vmax, lab_w}]  sorted as given (best first)
  hook_y     default 960 (frame 0, over the empty background)
  panel_y    default 470 (top of the card)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

GREEN_UP = "#3DDC84"; GREY = "#5B6B85"; LAB = "#C9D6EA"


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    MS = D["measures"]; nm = len(MS)
    if nm > 6: raise SystemExit("versus_card: max 6 measures")
    IL = D["israel"]
    if len(IL["values"]) != nm: raise SystemExit("versus_card: israel.values must match measures")
    C = [dict(c) for c in D["cards"]]
    for i, c in enumerate(C):
        if len(c["values"]) != nm: raise SystemExit(f"versus_card card {c['name']}: {len(c['values'])} values for {nm} measures")
        c["t0"] = A(c["at"])
    for i, c in enumerate(C):
        c["t1"] = A(c["until"]) if c.get("until") else (C[i + 1]["t0"] if i + 1 < len(C) else T.END - 0.6)
        ra = c.get("row_at") or [None] * nm
        c["ra"] = [A(x) if x is not None else c["t0"] + 0.35 + 0.16 * k for k, x in enumerate(ra)]
        c["cmp"] = c.get("cmp") or ["up"] * nm
        c["glow"] = [(g["row"], A(g["at"]), A(g.get("until"), c["t1"])) for g in c.get("glow", [])]
        st = c.get("sticker")
        c["st"] = (st, A(st["at"])) if st else None
    BARS = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("bars", [])]
    for x in BARS:
        for k, r in enumerate(x["rows"]): r["at"] = A(r["_at"]) if r.get("_at") is not None else x["t0"] + 0.3 + 0.12 * k
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 960)
    E = S.get("end") or {}; END0 = A(E.get("at"), T.END - 3) if E else None
    PY = D.get("panel_y", 470); HEAD = 150; ROWH = min(112, 680 // max(1, nm)); PH = HEAD + ROWH * nm + 40
    X_IC, X_LAB, X_C, X_AR, X_IL = 92, 140, 560, 712, 870
    SCL = (D.get("score") or {}).get("label", "Upgrades")

    def draw_card(cv, c, t, a, dx):
        WG.panel(cv, 30 + dx, PY, 1020, PH, a)
        hy = PY + 72
        # header: score | country | Israel
        ups = sum(1 for k in range(nm) if c["cmp"][k] == "up" and t >= c["ra"][k])
        cv.text(SCL, X_LAB + 40 + dx, hy - 28, 40, LAB, a, maxw=300, minsz=36)
        cv.text(f"{ups}/{nm}", X_LAB + 40 + dx, hy + 28, 64, YEL, a, maxw=300)
        WG.flag(cv, c["flag"], X_C + dx, hy - 30, 60, a)
        cv.text(c.get("head", c["name"]), X_C + dx, hy + 30, 44, WHITE, a, maxw=250, minsz=36)
        WG.flag(cv, IL.get("flag", "IL"), X_IL + dx, hy - 30, 60, a)
        cv.text(IL.get("head", "Israel"), X_IL + dx, hy + 30, 44, YEL, a, maxw=280, minsz=36)
        y0 = PY + HEAD
        for k, m in enumerate(MS):
            y = y0 + k * ROWH + ROWH / 2
            if k: cv.rrect(60 + dx, y - ROWH / 2 - 1, 960, 2, 1, "#FFFFFF", 0.10 * a)
            g = max([win(t, g0, g1, 0.25, 0.25) for r_, g0, g1 in c["glow"] if r_ == k] + [0])
            if g > 0.01: cv.rrect(44 + dx, y - ROWH / 2 + 6, 992, ROWH - 12, 22, SKY, 0.22 * g * a, stroke=SKY, lw=3, sa=0.9 * g * a)
            WG.emo(cv, m["icon"], X_IC + dx, y, 58, a)
            wl = m["label"].split()
            if len(wl) == 1: cv.text(wl[0], X_LAB + dx, y, 50, WHITE, a, align="l", maxw=300, minsz=36)
            else:
                cv.text(wl[0], X_LAB + dx, y - 27, 46, WHITE, a, align="l", maxw=300, minsz=36)
                cv.text(" ".join(wl[1:]), X_LAB + dx, y + 27, 46, WHITE, a, align="l", maxw=300, minsz=36)
            t_ = c["ra"][k]
            if t < t_: continue
            p = min(1.0, WG.pop_s(t, t_)); p2 = min(1.0, WG.pop_s(t, t_ + 0.18))
            v = c["values"][k]
            if v is None:
                cv.rrect(X_C - 80 + dx, y - 28, 160, 56, 28, GREY, a * p)
                cv.text("no data", X_C + dx, y, 38, WHITE, a * p, maxw=150, minsz=36)
            else: cv.text(v, X_C + dx, y, 64 * (0.85 + 0.15 * p), WHITE, a * p, maxw=270, minsz=40)
            if t >= t_ + 0.18:
                kind = c["cmp"][k]
                if kind == "up":
                    s = 26 * p2; cx_ = X_AR + dx
                    cv.c.new_path(); cv.c.move_to(cx_ - s, y + s * 0.7); cv.c.line_to(cx_ + s, y + s * 0.7); cv.c.line_to(cx_, y - s * 0.9); cv.c.close_path()
                    cv.col(GREEN_UP, a * p2); cv.c.fill()
                elif kind == "same": cv.text("=", X_AR + dx, y, 64, LAB, a * p2, maxw=80)
                cv.text(IL["values"][k], X_IL + dx, y, 64 * (0.85 + 0.15 * p2), YEL, a * p2, maxw=290, minsz=40)
        if c["st"]:
            st, ts = c["st"]
            if t >= ts:
                p = WG.pop_s(t, ts); sub = st.get("sub"); bh = 130 if sub else 110
                w = max(cv.tw(st["txt"], 88) + 70, (cv.tw(sub, 38) + 50) if sub else 0, 240); sx, sy = 1030 - w / 2 + dx, PY + PH + 50
                cv.rrect(sx - w / 2, sy - bh / 2, w, bh, 30, YEL, a * min(1, p))
                cv.text(st["txt"], sx, sy - (20 if sub else 0), 84 * min(1.15, p), "#0B1A3C", a * min(1, p), maxw=w - 20, shadow=False)
                if sub: cv.text(sub, sx, sy + 40, 36, "#0B1A3C", a * min(1, p), maxw=w - 20, shadow=False, minsz=36)

    def end_card(cv, t):
        if not E: return
        a = win(t, END0 - 0.1, T.END - 0.5, 0.35, 0.3)
        if a <= 0.01: return
        y = E.get("y", 1000); b2 = E.get("balance2"); t2 = A(E.get("balance2_at"), END0) if b2 else None
        WG.panel(cv, 40, y - 300, 1000, 680, a)
        WG.title(cv, E["balance"], y - 170, a, 66)
        if b2: WG.title(cv, b2, y + 10, a * seg(t, t2 - 0.1, t2 + 0.3), 58, col=LAB)
        if E.get("cta"):
            ca = a * seg(t, A(E.get("cta_at"), END0 + 0.6), A(E.get("cta_at"), END0 + 0.6) + 0.4)
            txt = E["cta"].replace("*", ""); sz = 60
            if cv.tw(txt, sz) <= 880: lines_ = [txt]
            else:
                w_ = txt.split(); best = None
                for k in range(1, len(w_)):
                    l1, l2 = " ".join(w_[:k]), " ".join(w_[k:]); m = max(cv.tw(l1, sz), cv.tw(l2, sz))
                    if best is None or m < best[0]: best = (m, [l1, l2])
                lines_ = best[1]
            tw = min(max(cv.tw(l, sz) for l in lines_) + 90, 980); ph = 60 + 72 * len(lines_)
            cv.rrect(540 - tw / 2, y + 130, tw, ph, 50, SKY, ca)
            for q, l in enumerate(lines_): cv.text(l, 540, y + 130 + 30 + 36 + 72 * q, sz, WHITE, ca, maxw=940, minsz=40)

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 10))
        dimb = max([win(t, x["t0"], x["t1"], 0.3, 0.3) for x in BARS] + [0])
        dime = win(t, END0 - 0.1, T.END - 0.5, 0.35, 0.3) if E else 0
        for c in C:
            if not (c["t0"] - 0.05 <= t <= c["t1"] + 0.45): continue
            u_in = seg(t, c["t0"] - 0.05, c["t0"] + 0.3); u_out = seg(t, c["t1"] - 0.05, c["t1"] + 0.3)
            dx = 1080 * (1 - u_in) - 1080 * u_out
            a = (1 - 0.85 * max(dimb, dime)) * min(1.0, 0.3 + 0.7 * u_in) * (1 - 0.7 * u_out)
            if a > 0.01: draw_card(cv, c, t, a, dx)
        for x in BARS:
            a = win(t, x["t0"], x["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            n = len(x["rows"]); rowh = min(118, 820 // max(1, n)); y0 = x.get("y", 600)
            WG.panel(cv, 30, y0 - 150, 1020, rowh * n + 190, a)
            WG.title(cv, x["title"], y0 - 80, a, 64)
            rows = []
            for r in x["rows"]:
                rr = dict(r)
                if r.get("value") is None: rr["value"] = 0; rr["txt"] = "no data"; rr["tcol"] = LAB
                if r.get("hl"): rr["color"] = YEL
                rows.append(rr)
            WG.bars(cv, rows, t, a, y0 + 30, rowh=rowh, vmax=x.get("vmax"), lab_w=x.get("lab_w", 360))
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); end_card(cv, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(c["t0"] - 0.05, "whoosh", -26) for c in C]
    for c in C:
        for k in range(nm): ev.append((c["ra"][k], "tick", -30))
        if c["st"]: ev.append((c["st"][1], "thud", -22))
    ev += [(x["t0"], "whoosh", -26) for x in BARS]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, cards=[c["name"] for c in C])
