"""scorecard: a live "0 out of 4" scorecard. Countries around Israel first (Egypt, Jordan, Lebanon, Syria, then Iraq,
Saudi Arabia ...), sorted best first by score (the renderer re-sorts, stable), Israel highlighted in its true place,
"no data" shown as a grey pill (never drop a country). Each column (a test, with an icon that shows the hard word)
fills in at its anchor, row by row, and every row's score counts up live.

data keys:
  columns   [{label, icon, at}]   max 4 (label max 2 words, icon an emoji that shows the meaning); up to 6 short one-word
            labels when "score" is false (anthems_compare: Land, Hope, Free, God, King, War)
  rows      [{label, flag, cells: ["Y" | "N" | "P" | null], hl}]   Y = yes (green), N = no (red), P = partly, null = no data
  sort      true (default): best first by score (Y 1, P 0.5, N or no data 0)
  at        anchor when the empty table appears (default 0.4 s, under the hook)
  hook_y    default 300
  verdict   [{at, until, text}]  a card over the table (max 12 words)
  score     true (default): live score column; false: no score (themes, not a test: nothing is "better")
  fill      "columns" (default): each column fills at its own anchor, row by row; "rows": each row fills at its own
            row "at" (one beat per country), cells left to right, and the row being talked about glows sky
  focus_until  rows mode: anchor where the row glow stops (default: the first verdict or the end card)
  photos    [{file, credit, at, until, kb}]  full-bleed real photo between beats (repo path, pulled into the sparse CI
            checkout if missing), slow Ken Burns, table fades out under it, "Photo: <credit>" line
  rows[].flag "SY" is drawn by hand as the current Syrian flag (green, white, black, three red stars, since 2025):
            the emoji font on the runner still has the old flag
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

SC = {"Y": 1.0, "P": 0.5, "N": 0.0, None: 0.0}


def _photo(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(p):                          # the CI checkout is sparse (templates/ only): pull the photo in
        import subprocess
        subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + path], check=False, capture_output=True)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"scorecard: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z):
    h, w = im.shape[:2]; s = max(W / w, H / h) * max(1.0, z); cw, ch = W / s, H / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


_yy = np.arange(H, dtype=np.float32)[:, None, None]
PH_SHADE = (0.80 - 0.45 * np.clip(1 - _yy / 620, 0, 1) - 0.40 * np.clip((_yy - 1300) / 620, 0, 1)).astype(np.float32)
del _yy


def _star5(cv, x, y, r, col, a):
    pts = [(x + (r if k % 2 == 0 else r * 0.4) * math.cos(-math.pi / 2 + k * math.pi / 5),
            y + (r if k % 2 == 0 else r * 0.4) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    c = cv.c; c.move_to(*pts[0])
    for q in pts[1:]: c.line_to(*q)
    c.close_path(); cv.col(col, a); c.fill()


def draw_flag(cv, code, cx, cy, sz, a):
    if code != "SY": return WG.flag(cv, code, cx, cy, sz, a)
    w = sz * 1.08; h = w * 2 / 3; x0 = cx - w / 2; y0 = cy - h / 2
    cv.rrect(x0, y0, w, h / 3 + 1, 2, "#007A3D", a); cv.rrect(x0, y0 + h / 3, w, h / 3 + 1, 0, "#FFFFFF", a)
    cv.rrect(x0, y0 + 2 * h / 3, w, h / 3, 2, "#000000", a)
    for k in (-1, 0, 1): _star5(cv, cx + k * w * 0.26, cy, h * 0.13, "#CE1126", a)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    SCORE = D.get("score", True); ROWS = D.get("fill", "columns") == "rows"
    MAXC = 4 if SCORE else 6
    cols = D["columns"][:MAXC]; rows = list(D["rows"]); nc = len(cols)
    if len(D["columns"]) > MAXC: raise SystemExit(f"scorecard: max {MAXC} columns")
    for r in rows:
        if len(r["cells"]) != nc: raise SystemExit(f"scorecard row {r['label']}: {len(r['cells'])} cells for {nc} columns")
        r["score"] = sum(SC[c] for c in r["cells"])
    if D.get("sort", True): rows.sort(key=lambda r: -r["score"])
    t_tab = A(D.get("at"), 0.4); cat = [A(c["at"]) for c in cols] if not ROWS else [t_tab] * nc
    rat = [A(r["at"]) for r in rows] if ROWS else None
    n = len(rows); rowh = min(118, 900 // max(1, n)); y0 = 640 + (8 - min(8, n)) * rowh * 0.25
    if nc <= 4: cx = [430 + k * 130 for k in range(nc)]
    else: cx = [400 + k * (600 / (nc - 1)) for k in range(nc)]
    sx = cx[-1] + 95; colw = 128 if nc <= 4 else 112
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    VD = [(A(v["at"]), A(v.get("until"), T.END - 0.6), v["text"]) for v in D.get("verdict", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    cell_at = (lambda k, ri: rat[ri] + 0.16 * k) if ROWS else (lambda k, ri: cat[k] + 0.12 * ri)
    PH = []
    for p_ in D.get("photos", []):
        PH.append(dict(im=_photo(p_["file"]), t0=A(p_["at"]), t1=A(p_["until"]), credit=p_.get("credit"),
                       kb=p_.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.47, 1.10]))
    FOC_END = A(D.get("focus_until"), min([a0 for a0, _, _ in VD] + ([END_AT] if END_AT else []) + [T.END]))

    def frame(t):
        bg = bars_moving_bg(t, 10); pa = 0.0; pcred = None
        for ph in PH:
            a_ = win(t, ph["t0"], ph["t1"], 0.25, 0.3)
            if a_ > 0.01:
                kb = ph["kb"]; u = min(1.0, max(0.0, (t - ph["t0"]) / max(0.5, ph["t1"] - ph["t0"] + 0.3)))
                img = _kb(ph["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u) * PH_SHADE
                bg = bg.astype(np.float32) * (1 - a_) + img * a_; pa = max(pa, a_); pcred = (ph["credit"], a_) if ph["credit"] else pcred
        cv = Canvas(bg); ta = seg(t, t_tab, t_tab + 0.5)
        vd = max([win(t, a0, a1, 0.3, 0.3) for a0, a1, _ in VD] + ([seg(t, END_AT - 0.1, END_AT + 0.3)] if END_AT else []) + [0]); ta *= (1 - 0.8 * vd); ta *= (1 - pa)
        if ta > 0.01:
            WG.panel(cv, 30, y0 - 210, 1020, rowh * n + 250, ta)
            for k, c in enumerate(cols):
                ha = ta * (1.0 if ROWS else (0.45 + 0.55 * seg(t, cat[k] - 0.3, cat[k])))
                if c.get("icon"): WG.emo(cv, c["icon"], cx[k], y0 - 150, 64, ha)
                lw_ = c["label"].split()
                for q, wd in enumerate(lw_[:2]): cv.text(wd, cx[k], y0 - 92 + q * 40 - (20 if len(lw_) == 1 else 0), 36, "#C9D6EA", ha, bold=True, maxw=colw, minsz=36)
            if SCORE: cv.text("Score", sx + 50, y0 - 72, 36, "#C9D6EA", ta, maxw=130)
            for ri, r in enumerate(rows):
                y = y0 + ri * rowh + rowh / 2 - 10
                if r.get("hl"): cv.rrect(40, y - rowh / 2 + 6, 1000, rowh - 12, 20, "#FFD23F", 0.14 * ta)
                if ROWS:
                    nxt = rat[ri + 1] if ri + 1 < n else FOC_END
                    fa = win(t, rat[ri] - 0.15, min(nxt, FOC_END), 0.2, 0.25)
                    if fa > 0.01: cv.rrect(40, y - rowh / 2 + 6, 1000, rowh - 12, 20, "#45A4DD", 0.20 * ta * fa, stroke="#45A4DD", lw=4, sa=0.9 * ta * fa)
                if r.get("flag"): draw_flag(cv, r["flag"], 85, y, 56, ta)
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
                if SCORE: cv.text(txt, sx + 50, y, 54, YEL if (r.get("hl") or done) else WHITE, ta * (0.5 + 0.5 * seg(t, cat[0], cat[0] + 0.3)), maxw=125, minsz=36)
        for a0, a1, txt in VD:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: WG.panel(cv, 70, 830, 940, 300, a, stroke=YEL); WG.title(cv, txt, 980, a, 72)
        if pcred: cv.text("Photo: " + pcred[0], 540, D.get("credit_y", 1430), 36, "#D5DCE8", 0.85 * pcred[1], bold=False, maxw=1000, minsz=36)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 290), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1500)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t_tab, "whoosh", -26)] + [(ph["t0"], "whoosh", -30) for ph in PH]
    for k in range(nc):
        for ri, r in enumerate(rows):
            c = r["cells"][k]; ev.append((cell_at(k, ri), {"Y": "check", "N": "buzz", "P": "tick"}.get(c, "tick"), -30 if c == "N" else -26))
    ev += [(a0, "thud", -22) for a0, _, _ in VD]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, order=[r["label"] for r in rows],
                photos=[dict(file=p_["file"], credit=p_.get("credit")) for p_ in D.get("photos", [])])
