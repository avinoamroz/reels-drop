"""photo_story: a photo-led story reel. Full-bleed real photos (slow Ken Burns, quick cuts or a crossfade), each beat a photo
plus a real number: big yellow number cards, a rubber stamp, a sourced quote card, an arithmetic card, screen titles,
source chips, a credit line per photo, end card, burned-in subtitles, hook from frame 0, circular ending (loops to frame 0).

Photos live in the repo under templates/photos/<id>/ (the spec-render checkout only has /templates/). Freely licensed or
official-release only; put every photo's author + licence in `credit` (shown on screen, 36 px) and in the caption.

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim, gray}]
           file: repo path. Shows from `at` until the next photo's `at` (or `until`). kb: Ken Burns from (cx0, cy0) at zoom
           z0 to (cx1, cy1) at z1 over the photo's window (centre as 0..1 of the image, zoom 1 = cover crop). dim: 0..1
           extra darkening (default 0.25). gray: true = desaturate (for the "blockade" irony beats).
  cut      crossfade seconds between photos (default 0.25; 0 = hard cut)
  big      [{at, until, value, from, prefix, suffix, dec, sub, y}]          counting number card (yellow)
  stamp    [{at, until, text, color: red|green|yellow, y, angle, size}]     rubber stamp that slams on
  quote    [{at, until, text, who, y}]                                       sourced quote card (*word* = gold)
  math     [{at, until, title, rows: [text, ...], result, y}]                arithmetic card (rows 52 px, result yellow)
  hook_y   default 900;  title_y default 300;  subs_y default 1612;  src_y default 1488;  credit_y default 1712
  stamp    optional per stamp: mark "Y" (green tick) | "N" (red cross) drawn inside the stamp after the text
  board    optional two-column scoreboard in the lower third (cards that chips drop into):
           {at, until, y (top, default 1050), pitch (row pitch, default 72), rows (default 4),
            cols: [{label: "NOT ETHNIC|CLEANSING", color: "sky"|"red"|hex}, {...}],
            chips: [{col, row, label, emoji, at}]}   a chip drops from above into its slot at `at`, then its tick pops
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

STAMP_COL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)


def _load(path, gray=False):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"photo_story: cannot read photo {path}")
    if gray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); im = cv2.merge([g, g, g])
    return im


def _kb(im, cx, cy, z):
    """cover crop of im to 1080x1920 at zoom z centred at (cx, cy) in 0..1 image coords -> float32 BGR"""
    h, w = im.shape[:2]; s = max(W / w, H / h) * max(1.0, z)
    cw, ch = W / s, H / s                                  # crop size in source px
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def _wrapped(cv, txt, cy, a, sz, maxw):
    """any number of centred lines, *span* in yellow even across line breaks"""
    toks, hl = [], False
    for w in txt.split():
        on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
        toks.append((w.replace("*", ""), hl or on)); hl = (hl or on) and not off
    lines = [[]]
    for tk in toks:
        if lines[-1] and cv.tw(" ".join(x[0] for x in lines[-1] + [tk]), sz) > maxw: lines.append([tk])
        else: lines[-1].append(tk)
    lh = sz * 1.2; y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        parts = [(w + (" " if k < len(ln) - 1 else ""), YEL if h else WHITE) for k, (w, h) in enumerate(ln)]
        cv.rich(parts, 540, y0 + i * lh, sz, a, maxw + 60)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    P = []
    for k, p in enumerate(D["photos"]):
        P.append(dict(p, t0=A(p["at"]), im=_load(p["file"], p.get("gray", False))))
    for k, p in enumerate(P):
        p["t1"] = A(p["until"]) if p.get("until") else (P[k + 1]["t0"] if k + 1 < len(P) else T.END)
    XF = float(D.get("cut", 0.25))
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    STM = [dict(s_, t0=A(s_["at"]), t1=A(s_.get("until"), T.END - 0.6)) for s_ in D.get("stamp", [])]
    QT = [dict(q, t0=A(q["at"]), t1=A(q.get("until"), T.END - 0.6)) for q in D.get("quote", [])]
    MT = [dict(m, t0=A(m["at"]), t1=A(m.get("until"), T.END - 0.6)) for m in D.get("math", [])]
    BD = dict(D["board"]) if D.get("board") else None
    if BD:
        BD["t0"] = A(BD.get("at"), 0.0); BD["t1"] = A(BD.get("until"), T.END + 1)
        BD["chips"] = [dict(c, t0=A(c["at"])) for c in BD.get("chips", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 900)
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.62, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    NAVY = np.array([97, 50, 22], np.float32)

    def photo_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"] + XF))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.25)
        return img * (1 - dim) * VIG + NAVY * dim * 0.35

    def base_at(t):
        lin = lambda x: min(1.0, max(0.0, x))
        out = BG.copy()
        for k, p in enumerate(P):                              # later photos fade in over earlier ones
            nxt = P[k + 1]["t0"] if k + 1 < len(P) else None
            gap = nxt is None or nxt > p["t1"] + 0.01           # nothing follows straight away: fade out to the brand bg
            if t < p["t0"] - XF or t >= p["t1"] + (XF if gap else 0): continue
            a = 1.0 if (p["t0"] <= 0.01 or not XF) else seg(t, p["t0"] - XF, p["t0"], lin)
            if gap and t >= p["t1"]: a *= 1.0 - (seg(t, p["t1"], p["t1"] + XF, lin) if XF else 1.0)
            if a <= 0.001: continue
            lay = photo_layer(p, t); out = lay if a >= 0.999 else out * (1 - a) + lay * a
        return out

    BCOL = dict(sky=SKY, red=RED, green=GREEN, yellow=YEL)
    BFILL = {SKY: "#123C66", RED: "#5E1E2A", GREEN: "#14492F", YEL: "#4D4214"}

    def draw_board(cv, t):
        a = (1.0 if BD["t0"] <= 0.01 else seg(t, BD["t0"], BD["t0"] + 0.3)) * (1 - seg(t, BD["t1"] - 0.3, BD["t1"]))
        if a <= 0.01: return
        y0 = BD.get("y", 1050); P_ = BD.get("pitch", 72); nr = BD.get("rows", 4); cw = 476; xs = (52, 552)
        hh = 108 + P_ * nr + 8
        cv.rrect(40, y0, 1000, hh, 30, "#0B1730", 0.9 * a, stroke=SKY, lw=2, sa=0.35 * a)
        cv.rrect(539, y0 + 18, 2, hh - 36, 1, "#C9D6EA", 0.25 * a)
        for k, c in enumerate(BD["cols"][:2]):
            col = BCOL.get(c.get("color", "sky"), c.get("color", SKY)); ls = c["label"].split("|")
            for j, l in enumerate(ls): cv.text(l, xs[k] + cw / 2, y0 + 54 - 21 * (len(ls) - 1) + 42 * j, 40, col, a, maxw=cw - 30, shadow=False)
            for r in range(nr): cv.rrect(xs[k] + 8, y0 + 108 + r * P_, cw - 16, P_ - 12, 16, "#FFFFFF", 0.06 * a)
        for ch in BD["chips"]:
            if t < ch["t0"]: continue
            k = ch.get("col", 0); col = BCOL.get(BD["cols"][k].get("color", "sky"), SKY); fill = BFILL.get(col, "#123C66")
            x = xs[k] + 8; y = y0 + 108 + ch["row"] * P_; w = cw - 16; h = P_ - 12; cx_, cy_ = x + w / 2, y + h / 2
            u = (t - ch["t0"]) / 0.42; p = back(min(1.0, u), 1.2); s_ = 1 + 0.35 * (1 - eout(min(1.0, u)))
            ca = a * min(1.0, u * 4); c = cv.c; c.save()
            c.translate(cx_, cy_ - 260 * (1 - p)); c.scale(s_, s_); c.translate(-cx_, -cy_)
            cv.rrect(x, y, w, h, 16, fill, 0.97 * ca, stroke=col, lw=3, sa=ca)
            if ch.get("emoji"): WG.emo(cv, ch["emoji"], x + 38, cy_, 42, ca)
            cv.text(ch["label"], x + 72, cy_, 44, WHITE, ca, align="l", maxw=w - 140, shadow=False)
            c.restore()
            land = ch["t0"] + 0.42
            if t >= land:                                      # yellow "sorted" tick
                r_ = 21 * WG.pop_s(t, land); mx = x + w - 34
                if r_ > 0.5:
                    cv.circ(mx, cy_, r_, YEL, a)
                    cv.line([(mx - r_ * 0.45, cy_ + r_ * 0.02), (mx - r_ * 0.12, cy_ + r_ * 0.36), (mx + r_ * 0.5, cy_ - r_ * 0.34)], "#0B1730", r_ * 0.24, a)

    def frame(t):
        base = base_at(t)
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BIG + QT + MT] +
                   ([seg(t, END_AT - 0.1, END_AT + 0.3)] if END_AT else []) + [0])
        if over > 0: base = base * (1 - 0.45 * over)
        cv = Canvas(base)
        if BD: draw_board(cv, t)
        for p in P:                                            # credit line of the photo on screen
            ca = win(t, p["t0"], p["t1"], 0.25, 0.25)
            if ca > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ca, bold=False, maxw=1000, minsz=36)
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(110, y - 175, 860, 350, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.1))
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 35, g.get("size", 170), YEL, ga, maxw=800)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 56, WHITE, ga, maxw=800)
        for m in MT:
            ma = win(t, m["t0"] - 0.1, m["t1"], 0.35, 0.35)
            if ma > 0.01:
                rows = m.get("rows", []); y = m.get("y", 930); hh = 150 + 78 * len(rows) + (150 if m.get("result") else 0)
                top = y - hh / 2; WG.panel(cv, 60, top, 960, hh, ma, stroke=YEL)
                yy = top + 80
                if m.get("title"): WG.title(cv, m["title"], yy, ma, 60, col="#C9D6EA"); yy += 90
                for k, r in enumerate(rows):
                    ra = ma * seg(t, m["t0"] + 0.25 * k, m["t0"] + 0.25 * k + 0.3)
                    WG.title(cv, r, yy, ra, 56); yy += 78
                if m.get("result"):
                    ra = ma * seg(t, m["t0"] + 0.25 * len(rows) + 0.2, m["t0"] + 0.25 * len(rows) + 0.6)
                    cv.rrect(120, yy - 10, 840, 2, 1, "#C9D6EA", 0.5 * ra)
                    WG.title(cv, m["result"], yy + 70, ra, 84, col=YEL)
        for q in QT:
            qa = win(t, q["t0"] - 0.1, q["t1"], 0.35, 0.35)
            if qa > 0.01:
                y = q.get("y", 920); WG.panel(cv, 60, y - 260, 960, 520, qa, stroke=YEL)
                cv.text("“", 140, y - 190, 150, YEL, qa, align="l", shadow=False)
                _wrapped(cv, q["text"], y - 30, qa, q.get("size", 66), 860)
                if q.get("who"): cv.text(q["who"], 540, y + 175, 44, "#C9D6EA", qa, bold=False, maxw=900, minsz=36)
        for s_ in STM:
            sa = win(t, s_["t0"] - 0.05, s_["t1"], 0.08, 0.3)
            if sa > 0.01:
                k = 1.0 + 0.9 * (1 - eout((t - s_["t0"]) / 0.18)) if t < s_["t0"] + 0.18 else 1.0
                col = STAMP_COL.get(s_.get("color", "red"), s_.get("color", RED)); sz = s_.get("size", 128) * k
                y = s_.get("y", 1180); ang = math.radians(s_.get("angle", -9)); txt = s_["text"]
                c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.translate(-540, -y)
                mk = s_.get("mark"); mr = sz * 0.38 if mk else 0
                tw = min(cv.tw(txt, sz), 940 - 2.6 * mr) + 80 + 2.6 * mr
                cv.rrect(540 - tw / 2, y - sz * 0.72, tw, sz * 1.44, 18, "#0B1730", 0.35 * sa, stroke=col, lw=10 * k, sa=sa)
                cv.text(txt, 540 - 1.3 * mr, y, sz, col, sa, maxw=900 - 2.6 * mr, shadow=False, minsz=36)
                if mk: WG.mark(cv, mk, 540 + tw / 2 - 40 - mr, y, mr, sa)
                c.restore()
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 300), 76)
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -30) for p in P[1:]]
    ev += [(s_["t0"], "thud", -18) for s_ in STM]
    ev += [(g["t0"], "roll", g["t0"] + 1.1) for g in BIG] + [(g["t0"] + 1.1, "ding", -26) for g in BIG]
    ev += [(q["t0"], "pop", -26) for q in QT] + [(m["t0"], "pop", -26) for m in MT]
    if BD: ev += [(c["t0"] + 0.42, "check", -28) for c in BD["chips"]]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in P])
