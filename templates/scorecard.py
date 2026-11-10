"""scorecard: a live "0 out of 4" scorecard. Countries around Israel first (Egypt, Jordan, Lebanon, Syria, then Iraq,
Saudi Arabia ...), sorted best first by score (the renderer re-sorts, stable), Israel highlighted in its true place,
"no data" shown as a grey pill (never drop a country). Each column (a test, with an icon that shows the hard word)
fills in at its anchor, row by row, and every row's score counts up live.

data keys:
  columns   [{label, icon, at}]   max 4 (label max 2 words, icon an emoji that shows the meaning)
  rows      [{label, flag, cells: ["Y" | "N" | "P" | null], hl}]   Y = yes (green), N = no (red), P = partly, null = no data
  sort      true (default): best first by score (Y 1, P 0.5, N or no data 0)
  at        anchor when the empty table appears (default 0.4 s, under the hook)
  hook_y    default 300
  verdict   [{at, until, text}]  a card over the table (max 12 words)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

SC = {"Y": 1.0, "P": 0.5, "N": 0.0, None: 0.0}


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    cols = D["columns"][:4]; rows = list(D["rows"]); nc = len(cols)
    if len(D["columns"]) > 4: raise SystemExit("scorecard: max 4 columns")
    for r in rows:
        if len(r["cells"]) != nc: raise SystemExit(f"scorecard row {r['label']}: {len(r['cells'])} cells for {nc} columns")
        r["score"] = sum(SC[c] for c in r["cells"])
    if D.get("sort", True): rows.sort(key=lambda r: -r["score"])
    t_tab = A(D.get("at"), 0.4); cat = [A(c["at"]) for c in cols]
    n = len(rows); rowh = min(118, 900 // max(1, n)); y0 = 640 + (8 - min(8, n)) * rowh * 0.25
    cx = [430 + k * 130 for k in range(nc)]; sx = cx[-1] + 95
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    VD = [(A(v["at"]), A(v.get("until"), T.END - 0.6), v["text"]) for v in D.get("verdict", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    cell_at = lambda k, ri: cat[k] + 0.12 * ri

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 10)); ta = seg(t, t_tab, t_tab + 0.5)
        vd = max([win(t, a0, a1, 0.3, 0.3) for a0, a1, _ in VD] + ([seg(t, END_AT - 0.1, END_AT + 0.3)] if END_AT else []) + [0]); ta *= (1 - 0.8 * vd)
        if ta > 0.01:
            WG.panel(cv, 30, y0 - 210, 1020, rowh * n + 250, ta)
            for k, c in enumerate(cols):
                ha = ta * (0.45 + 0.55 * seg(t, cat[k] - 0.3, cat[k]))
                if c.get("icon"): WG.emo(cv, c["icon"], cx[k], y0 - 150, 64, ha)
                lw_ = c["label"].split()
                for q, wd in enumerate(lw_[:2]): cv.text(wd, cx[k], y0 - 92 + q * 40 - (20 if len(lw_) == 1 else 0), 36, "#C9D6EA", ha, bold=True, maxw=128, minsz=36)
            cv.text("Score", sx + 50, y0 - 72, 36, "#C9D6EA", ta, maxw=130)
            for ri, r in enumerate(rows):
                y = y0 + ri * rowh + rowh / 2 - 10
                if r.get("hl"): cv.rrect(40, y - rowh / 2 + 6, 1000, rowh - 12, 20, "#FFD23F", 0.14 * ta)
                if r.get("flag"): WG.flag(cv, r["flag"], 85, y, 56, ta)
                cv.text(r["label"], 125, y, 54, YEL if r.get("hl") else WHITE, ta, align="l", maxw=cx[0] - 185, minsz=38)
                sc = 0.0
                for k, cell in enumerate(r["cells"]):
                    t_ = cell_at(k, ri)
                    if t < t_: continue
                    p = WG.pop_s(t, t_)
                    if cell is None:
                        cv.rrect(cx[k] - 58, y - 24, 116, 48, 24, "#5B6B85", ta * min(1, p))
                        cv.text("no data", cx[k], y, 36, WHITE, ta * min(1, p), bold=True, maxw=110, minsz=36)
                    else: WG.mark(cv, cell, cx[k], y, 34, ta, p); sc += SC[cell] * min(1, (t - t_) / 0.3)
                done = all(t >= cell_at(k, ri) for k in range(nc))
                sc = math.floor(sc * 2 + 1e-6) / 2
                txt = (f"{sc:.1f}" if sc % 1 else f"{int(sc)}") + f"/{nc}"
                cv.text(txt, sx + 50, y, 54, YEL if (r.get("hl") or done) else WHITE, ta * (0.5 + 0.5 * seg(t, cat[0], cat[0] + 0.3)), maxw=125, minsz=36)
        for a0, a1, txt in VD:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: WG.panel(cv, 70, 830, 940, 300, a, stroke=YEL); WG.title(cv, txt, 980, a, 72)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 290), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t_tab, "whoosh", -26)]
    for k in range(nc):
        for ri, r in enumerate(rows):
            c = r["cells"][k]; ev.append((cell_at(k, ri), {"Y": "check", "N": "buzz", "P": "tick"}.get(c, "tick"), -30 if c == "N" else -26))
    ev += [(a0, "thud", -22) for a0, _, _ in VD]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, order=[r["label"] for r in rows])
