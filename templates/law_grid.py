"""law_grid: a live comparison grid, phone friendly. Countries are COLUMNS (max 5, flag LEFT of a short code in the
header), situations are ROWS (icon + one short question on its own line, the marks on the line under it, so labels can
be long and marks big). Rows fill in at their anchors (cells left to right, or per cell anchors), the row being talked
about glows, a highlighted column (e.g. Israel) gets a soft yellow strip, and every column's score counts up live in a
score row under the grid. Real photos slide in as a band across the top (edge to edge, credit inside the band) while
the grid stays on screen. Hook from frame 0 over the moving grid; circular ending.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  columns  [{code: "IL", label: "IL", hl}]   max 5; code = flag (ISO 2), label = short text right of the flag
  rows     [{label, icon, at, cells: ["Y"|"N"|"P"|null], cells_at: [anchor|null, ...], note}]
           Y green tick, N red cross, P amber (partly), null = "no data" pill. cells_at: optional per cell anchors
           (null = sequential after `at`). note: optional small line (>= 36 px) under the label, shown from `at`.
  focus    [{at, until, row}]   glow ring on a row (default: each row glows from its `at` for 2.2 s)
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]   top band photos (repo paths)
  band     {y: 40, h: 400}     photo band geometry
  verdict  [{at, until, text}]  a card over the grid (max 12 words)
  grid_y   top of the grid panel (default 470); hook_y (default 250); title_y (default 250); src_y (default 1508);
  subs_y   (default 1640); score_label (default "Score")
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

SC = {"Y": 1.0, "P": 0.5, "N": 0.0, None: 0.0}


def _band(im, cx, cy, z, bw, bh):
    """cover crop of im into a bw x bh band at zoom z centred at (cx, cy) (0..1 image coords) -> float32 BGR"""
    h, w = im.shape[:2]; s = max(bw / w, bh / h) * max(1.0, z)
    cw, ch = bw / s, bh / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    cols = D["columns"]; rows = D["rows"]; nc = len(cols); nr = len(rows)
    if nc > 5: raise SystemExit("law_grid: max 5 columns")
    for r in rows:
        if len(r["cells"]) != nc: raise SystemExit(f"law_grid row {r['label']}: {len(r['cells'])} cells for {nc} columns")
    # ---- timing per cell
    r_at = [A(r["at"]) for r in rows]
    c_at = []
    for ri, r in enumerate(rows):
        ca = r.get("cells_at") or [None] * nc; out = []; k_seq = 0
        for k in range(nc):
            if ca[k] is not None: out.append(A(ca[k]))
            else: out.append(r_at[ri] + 0.14 * k_seq); k_seq += 1
        c_at.append(out)
    FOC = [(A(f["at"]), A(f.get("until"), A(f["at"]) + 2.2), f["row"]) for f in D.get("focus", [])] or \
          [(r_at[i], r_at[i] + 2.2, i) for i in range(nr)]
    # ---- geometry
    GY = D.get("grid_y", 470); HEAD = 96; ROWH = D.get("row_h", 132)
    cx = [140 + k * 200 for k in range(nc)] if nc == 5 else [540 + (k - (nc - 1) / 2) * 220 for k in range(nc)]
    y_rows = GY + HEAD + 20; y_score = y_rows + nr * ROWH + 52
    PH_ = D.get("band", {}); BY, BHh = PH_.get("y", 40), PH_.get("h", 400)
    PH = []
    if D.get("photos"):
        from photo_story import _load
        for p in D["photos"]: PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"])))
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 250)
    VD = [(A(v["at"]), A(v.get("until"), T.END - 0.6), v["text"]) for v in D.get("verdict", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    t_tab = A(D.get("at"), 0.0)

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 14))
        vd = max([win(t, a0, a1, 0.3, 0.3) for a0, a1, _ in VD] + ([seg(t, END_AT - 0.1, END_AT + 0.3)] if END_AT else []) + [0])
        ta = 1 - 0.82 * vd
        bot = y_score + 70
        WG.panel(cv, 24, GY, 1032, bot - GY, ta)
        # highlighted column strip
        for k, c in enumerate(cols):
            if c.get("hl"): cv.rrect(cx[k] - 92, GY + 12, 184, bot - GY - 24, 26, "#FFD23F", 0.13 * ta)
        # header: flag LEFT of the code, on screen from frame 0; each flag bounces in turn 0.3 to 1.4 s (motion by 1.5 s)
        for k, c in enumerate(cols):
            u = (t - (t_tab + 0.3 + 0.2 * k)) / 0.35
            p = 1 + 0.22 * math.sin(math.pi * u) if 0 < u < 1 else 1.0
            yh = GY + HEAD / 2 + 8; lab = c.get("label", c["code"])
            tw = cv.tw(lab, 54); x0 = cx[k] - (58 + 10 + tw) / 2
            WG.flag(cv, c["code"], x0 + 29, yh, 58 * min(1.15, p), ta)
            cv.text(lab, x0 + 68, yh, 54, YEL if c.get("hl") else WHITE, ta * min(1, p), align="l", maxw=110, minsz=40)
        cv.line([(50, GY + HEAD + 10), (1030, GY + HEAD + 10)], "#45A4DD", 2, 0.5 * ta)
        # rows
        score = [0.0] * nc
        for ri, r in enumerate(rows):
            y = y_rows + ri * ROWH
            ra = ta * seg(t, r_at[ri] - 0.25, r_at[ri] + 0.15)
            fo = max([win(t, a0, a1, 0.25, 0.35) for a0, a1, i in FOC if i == ri] + [0])
            if fo > 0.01: cv.rrect(34, y - 8, 1012, ROWH - 6, 22, "#45A4DD", 0.18 * fo * ta, stroke=YEL, lw=3, sa=0.9 * fo * ta)
            # label line (dim placeholder before its anchor)
            if ra < 0.01:
                sh = 0.10 + 0.08 * max(0, math.sin(2.2 * t - 0.5 * ri))
                cv.rrect(110, y + 16, 520, 18, 9, "#C9D6EA", sh * ta)
            else:
                if r.get("icon"): WG.emo(cv, r["icon"], 78, y + 28, 50, ra)
                cv.text(r["label"], 118, y + 28, 50, WHITE, ra, align="l", maxw=900, minsz=40)
            ym = y + 88
            for k, cell in enumerate(r["cells"]):
                t_ = c_at[ri][k]
                if t < t_:
                    cv.circ(cx[k], ym, 8, "#C9D6EA", 0.25 * ta); continue
                p = WG.pop_s(t, t_)
                if cell is None:
                    cv.rrect(cx[k] - 66, ym - 24, 132, 48, 24, "#5B6B85", ta * min(1, p))
                    cv.text("no data", cx[k], ym, 36, WHITE, ta * min(1, p), maxw=126, minsz=36)
                else:
                    WG.mark(cv, cell, cx[k], ym, 30, ta, p); score[k] += SC[cell] * min(1, (t - t_) / 0.3)
            if ri < nr - 1: cv.line([(60, y + ROWH - 10), (1020, y + ROWH - 10)], "#C9D6EA", 1, 0.18 * ta)
        # score row
        sa = ta * seg(t, min(min(c) for c in c_at) - 0.2, min(min(c) for c in c_at) + 0.2)
        if sa > 0.01:
            cv.line([(50, y_score - 48), (1030, y_score - 48)], "#45A4DD", 2, 0.5 * ta)
            for k in range(nc):
                sc = math.floor(score[k] * 2 + 1e-6) / 2
                txt = (f"{sc:.1f}" if sc % 1 else f"{int(sc)}") + f"/{nr}"
                cv.text(txt, cx[k], y_score + 8, 60, YEL, sa, maxw=180, minsz=40)
        # photo band (top, edge to edge) with credit inside the band
        for j, p in enumerate(PH):
            a_ = win(t, p["t0"], p["t1"], 0.3, 0.3)
            if a_ <= 0.01: continue
            u = min(1, max(0, (t - p["t0"]) / max(0.1, p["t1"] - p["t0"])))
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            band = _band(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u, W, BHh)
            band *= (1 - p.get("dim", 0.1))
            g = np.linspace(0, 1, BHh, dtype=np.float32); g = np.clip((g - 0.62) / 0.38, 0, 1)[:, None, None]
            band = band * (1 - 0.6 * g) + np.array([97, 50, 22], np.float32)[None, None] * 0.6 * g
            slide = (1 - eout(seg(t, p["t0"], p["t0"] + 0.4, lambda v: v))) * 60
            y0 = int(BY - slide); y1 = min(H, y0 + BHh)
            if y1 > max(0, y0):
                src = band[max(0, -y0):max(0, -y0) + (y1 - max(0, y0))]
                cv.surf.flush()
                region = cv.buf[max(0, y0):y1, :, :3].astype(np.float32)
                cv.buf[max(0, y0):y1, :, :3] = np.clip(region * (1 - a_) + src * a_, 0, 255).astype(cv.buf.dtype)
                cv.surf.mark_dirty()
            if p.get("credit"): cv.text("Photo: " + p["credit"], 540, BY + BHh - 30, 36, "#E6EEF8", 0.9 * a_, bold=False, maxw=1000, minsz=36)
            if p.get("tag"): WG.tag(cv, p["tag"], 540, BY + 60, 46, WHITE, a_)
        for a0, a1, txt in VD:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: WG.panel(cv, 70, 830, 940, 300, a, stroke=YEL); WG.title(cv, txt, 980, a, 72)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 250), 70)
        draw_sources(cv, S, T, t, D.get("src_y", 1508)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.86)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1640)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t_tab + 0.3 + 0.2 * k, "pop", -32) for k in range(nc)]
    for ri, r in enumerate(rows):
        for k, c in enumerate(r["cells"]):
            ev.append((c_at[ri][k], {"Y": "check", "N": "buzz", "P": "tick"}.get(c, "tick"), -31 if c == "N" else -27))
    ev += [(p["t0"], "whoosh", -26) for p in PH] + [(a0, "thud", -22) for a0, _, _ in VD]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
