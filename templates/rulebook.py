"""rulebook: a satirical "rulebook" reel. A physical book on the desk; frame 0 riffles pages fast and stops (boom) on a
rule; later beats flip (with a page-flip sound) forwards or backwards to other rules, non-sequential; after each rule the
reel cuts to a full-bleed real photo with an evidence card (sourced quote, stamp); at the end the book slams shut and the
reel loops back into the frame-0 riffle.

The book is our own clean design (navy board, cream pages, Inter Display); no logos of anyone.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  book     {title: "THE BBC RULEBOOK", tag: "SATIRE", footer: "Unofficial. Not a BBC product.",
            cover_title: "THE BBC|RULEBOOK", cover_sub: "Unofficial. Satire."}
  pages    {"5": "Rule text with *red* words", ...}   rules that are read; other numbers show as placeholder pages
  scenes   in order; each starts at `at` (a book scene's flip ENDS at `at`, it starts `flip` seconds earlier):
           {kind: "book", at, page: 5, flip: 0.5, from: 1}   from = page to riffle from (frame 0); default = last page shown
           {kind: "photo", at, file, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}
           {kind: "close", at, dur: 0.35}                   the cover slams shut over the page (thud)
  cards    [{type: "quote", at, until, text, who, y, size}  sourced quote / evidence card (*word* = gold)
            {type: "stamp", at, until, text, color: red|green|yellow, y, angle, size}]
  cut      crossfade seconds between scenes (default 0.12)
  hook_y   default 250;  hook_s default 0.85;  subs_y default 1612;  credit_y default 1712
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as CMN
from photo_story import _wrapped

# ------------------------------------------------------------------ geometry and colours
PX0, PY0, PW, PH = 112, 448, 872, 990          # the right-hand page (spine on its left edge)
CREAM, CREAM_D, INK, RED_INK, MUTE = "#F4EEE1", "#E4DCCB", "#14284D", "#C2313A", "#7C8597"
STAMP_COL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)


def _sfx_patch():
    """adds a paper page-flip and a book slam to the shared SFX library (used by build_mix)"""
    orig = CMN._sfx_lib
    def lib():
        d = orig(); rng = np.random.default_rng(11); tt = lambda d_: np.arange(int(d_ * SR)) / SR
        def flip():
            x = tt(0.16); n = rng.standard_normal(len(x))
            n = np.convolve(n, np.ones(6) / 6, "same") - np.convolve(n, np.ones(40) / 40, "same")   # papery band
            env = np.minimum(1, x / 0.012) * np.exp(-x * 26)
            return n * env / (np.abs(n * env).max() + 1e-9)
        def slam():
            x = tt(0.5); low = np.sin(2 * np.pi * (85 - 30 * x) * x) * np.exp(-x * 14)
            n = rng.standard_normal(len(x)) * np.exp(-x * 45)
            y = low + 0.6 * np.convolve(n, np.ones(5) / 5, "same"); return y / np.abs(y).max()
        d.update(flip=flip, slam=slam); return d
    CMN._sfx_lib = lib


def _rich_wrap(cv, txt, x, cy, sz, col, hl, maxw, align="l"):
    """left aligned wrapped text, *span* in hl colour; returns nothing"""
    toks, on_ = [], False
    for w in txt.split():
        on = w.startswith("*"); off = w.rstrip(".,!?:;\"").endswith("*")
        toks.append((w.replace("*", ""), on_ or on)); on_ = (on_ or on) and not off
    lines = [[]]
    for tk in toks:
        if lines[-1] and cv.tw(" ".join(a for a, _ in lines[-1] + [tk]), sz) > maxw: lines.append([tk])
        else: lines[-1].append(tk)
    lh = sz * 1.18; y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        xx = x
        for k, (w, h) in enumerate(ln):
            s_ = w + (" " if k < len(ln) - 1 else "")
            cv.text(s_, xx, y0 + i * lh, sz, hl if h else col, 1.0, align="l", maxw=2000, shadow=False); xx += cv.tw(s_, sz)


def _page_img(B, num, text):
    """one cream page (PH x PW BGR float32), header, RULE n, rule text or placeholder lines, footer"""
    base = np.zeros((H, W, 3), np.float32); cv = Canvas(base)
    cv.rrect(PX0, PY0, PW, PH, 6, CREAM, 1.0)
    x0 = PX0 + 70; xr = PX0 + PW - 60
    cv.text(B.get("title", "THE RULEBOOK"), x0, PY0 + 62, 36, MUTE, 1.0, align="l", shadow=False, maxw=600)
    if B.get("tag"): cv.text(B["tag"], xr, PY0 + 62, 36, RED_INK, 1.0, align="r", shadow=False, maxw=260)
    cv.rrect(x0, PY0 + 98, xr - x0, 3, 1, MUTE, 0.45)
    if num is not None:
        cv.text(f"RULE {num}", x0, PY0 + 230, 132, INK if text else "#B9B2A4", 1.0, align="l", shadow=False, maxw=760)
        cv.rrect(x0, PY0 + 318, 150, 12, 6, RED_INK if text else "#CFC7B6", 1.0)
    if text:
        _rich_wrap(cv, text, x0, PY0 + 600, 80, INK, RED_INK, xr - x0)
    else:                                                   # an unread page: grey bars where the text would be
        rng = np.random.default_rng(int(num or 0) + 7)
        for k in range(6):
            w_ = (xr - x0) * (0.55 + 0.4 * rng.random()) if k < 5 else (xr - x0) * 0.35
            cv.rrect(x0, PY0 + 420 + k * 82, w_, 30, 15, "#D6CEBD", 1.0)
    if B.get("footer"): cv.text(B["footer"], x0, PY0 + PH - 58, 36, MUTE, 1.0, align="l", bold=False, shadow=False, maxw=620)
    if num is not None: cv.text(str(num), xr, PY0 + PH - 58, 36, MUTE, 1.0, align="r", bold=False, shadow=False)
    im = cv.out().astype(np.float32)[PY0:PY0 + PH, PX0:PX0 + PW]
    g = np.linspace(0.86, 1.0, 46, dtype=np.float32)        # soft gutter shadow at the spine
    im[:, :46] *= g[None, :, None]
    return im


def _cover_img(B):
    base = np.zeros((H, W, 3), np.float32); cv = Canvas(base)
    cv.rrect(PX0, PY0, PW, PH, 10, "#132B55", 1.0)
    cv.rrect(PX0 + 34, PY0 + 34, PW - 68, PH - 68, 6, None, 1.0, stroke="#C9A54A", lw=4, sa=0.9)
    cv.rrect(PX0 + 50, PY0 + 50, PW - 100, PH - 100, 4, None, 1.0, stroke="#C9A54A", lw=1.5, sa=0.6)
    ls = B.get("cover_title", "THE|RULEBOOK").split("|"); ty = PY0 + 210
    for k, l in enumerate(ls):
        cv.text(l, PX0 + PW / 2, ty + k * 140, 124, "#F1D27A", 1.0, shadow=False, maxw=PW - 140)
    yb = ty + 140 * (len(ls) - 1) + 110
    cv.rrect(PX0 + PW / 2 - 90, yb, 180, 6, 3, "#C9A54A", 0.9)
    if B.get("cover_sub"): cv.text(B["cover_sub"], PX0 + PW / 2, yb + 70, 48, "#D9DEE8", 1.0, bold=False, shadow=False, maxw=PW - 160)
    im = cv.out().astype(np.float32)[PY0:PY0 + PH, PX0:PX0 + PW]
    g = np.linspace(0.7, 1.0, 40, dtype=np.float32); im[:, :40] *= g[None, :, None]
    return im


def _desk():
    """brand gradient desk with a soft spotlight and the book boards / page stack under the page"""
    b = BG.copy(); yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-(((xx - 560) / 700) ** 2 + ((yy - 950) / 900) ** 2))[..., None]
    b = b * (0.75 + 0.45 * spot)
    cv = Canvas(b)
    cv.rrect(PX0 - 30, PY0 - 26, PW + 62, PH + 60, 14, "#08132A", 0.55)            # drop shadow
    cv.rrect(PX0 - 22, PY0 - 20, PW + 46, PH + 44, 12, "#132B55", 1.0)             # back board
    for k in range(5, 0, -1):                                                       # page stack edges
        cv.rrect(PX0 + 2, PY0 + 2, PW + k * 3, PH + k * 3, 4, CREAM_D if k % 2 else "#CFC6B2", 1.0)
    return cv.out().astype(np.float32)


def _warp(img, quad):
    """warp a page image (h, w, 3) onto quad [(x,y) tl, tr, br, bl] in frame coords -> (frame img, mask)"""
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    return out, m[..., None]


def _turn_quad(theta):
    """quad of a page rotated theta (0 = flat on the right, pi = flat on the left) about the spine x = PX0"""
    c = math.cos(theta); lift = 0.10 * math.sin(theta)
    xe = PX0 + PW * c; dy = PH * lift / 2
    if c >= 0: return [(PX0, PY0), (xe, PY0 - dy), (xe, PY0 + PH + dy), (PX0, PY0 + PH)]
    return [(xe, PY0 - dy), (PX0, PY0), (PX0, PY0 + PH), (xe, PY0 + PH + dy)]


def run(S):
    _sfx_patch()
    D = S["data"]; B = D.get("book", {}); work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    PAGES = {int(k): v for k, v in D.get("pages", {}).items()}
    cache = {}
    def page(n, ph=False):
        key = (n, ph)
        if key not in cache: cache[key] = _page_img(B, n, None if ph else PAGES.get(n))
        return cache[key]
    COVER = _cover_img(B); BACK = np.full((PH, PW, 3), 0, np.float32); BACK[:] = np.array([200, 214, 226], np.float32)  # back of a page (BGR cream)
    DESK = _desk()

    # ---- scenes: resolve times; a book scene's flip ends at `at`
    SC, last_page, read = [], D.get("first_from", 1), set()
    for s_ in D["scenes"]:
        s_ = dict(s_); k = s_["kind"]
        if k == "book":
            s_["t1f"] = A(s_["at"]); s_["dur"] = float(s_.get("flip", 0.5)); s_["t0"] = max(0.0, s_["t1f"] - s_["dur"])
            if s_["t1f"] <= 0.01 and s_["dur"]: s_["t1f"] = s_["dur"]                 # frame-0 riffle
            frm = int(s_.get("from", last_page)); to = int(s_["page"])
            seq = list(range(frm, to + 1)) if to >= frm else list(range(frm, to - 1, -1))
            s_["seq"] = seq; s_["fwd"] = to >= frm; last_page = to
            s_["ph"] = [n_ for n_ in seq[1:-1] if n_ not in read]; read.update([frm, to])   # unread pages flick by blank
            if "u0" in s_: s_["u0"] = float(s_["u0"])
        elif k == "photo":
            s_["t0"] = A(s_["at"]); s_["im"] = cv2.imread(s_["file"] if os.path.isabs(s_["file"]) else os.path.join(REPO, s_["file"]))
            if s_["im"] is None: raise SystemExit(f"rulebook: cannot read photo {s_['file']}")
        elif k == "close":
            s_["t0"] = A(s_["at"]); s_["dur"] = float(s_.get("dur", 0.35)); s_["page"] = last_page
        SC.append(s_)
    for i, s_ in enumerate(SC): s_["t_end"] = SC[i + 1]["t0"] if i + 1 < len(SC) else T.END + 1
    XF = float(D.get("cut", 0.12))
    CARDS = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6)) for c in D.get("cards", [])]
    END_AT = A((S.get("end") or {}).get("at"), T.END - 3) if S.get("end") else None
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 250); HS = D.get("hook_s", 0.85)
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.55, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    NAVY = np.array([97, 50, 22], np.float32)

    def comp(out, img, mask): return out * (1 - mask) + img * mask

    def flat(out, img):
        out[PY0:PY0 + PH, PX0:PX0 + PW] = img; return out

    def book_at(s_, t):
        out = DESK.copy(); seq = s_["seq"]; n = len(seq) - 1
        pg = lambda n_: page(n_, n_ in s_["ph"])
        if n <= 0 or t >= s_["t1f"]: return flat(out, pg(seq[-1]))
        u = (t - s_["t0"]) / max(1e-3, s_["t1f"] - s_["t0"]); u0 = s_.get("u0", 0.0)
        u = u0 + (1 - u0) * min(1.0, max(0.0, u))
        p = eout(u) ** 0.9 * n                                  # fast first, then the stop
        k = min(n - 1, int(p)); f = p - k
        th = math.pi * ease(f) if n == 1 else math.pi * f
        if s_["fwd"]:                                           # the page on top turns away to the left
            out = flat(out, pg(seq[k + 1]))
            if th < math.pi / 2:
                im, m = _warp(pg(seq[k]) * (0.78 + 0.22 * math.cos(th)), _turn_quad(th))
                sh = np.clip((np.arange(W, dtype=np.float32) - (PX0 + PW * math.cos(th))) / 140, 0, 1)
                out = out * (1 - 0.35 * (1 - sh)[None, :, None] * (1 - math.cos(th)) * (np.arange(W) > PX0)[None, :, None])
                out = comp(out, im, m)
            else:
                im, m = _warp(BACK * (0.7 + 0.3 * abs(math.cos(th))), _turn_quad(th)); out = comp(out, im, m)
        else:                                                   # a page comes back from the left and lands
            th = math.pi - th
            out = flat(out, pg(seq[k]))
            if th >= math.pi / 2:
                im, m = _warp(BACK * (0.7 + 0.3 * abs(math.cos(th))), _turn_quad(th)); out = comp(out, im, m)
            else:
                im, m = _warp(pg(seq[k + 1]) * (0.78 + 0.22 * math.cos(th)), _turn_quad(th)); out = comp(out, im, m)
        return out

    def close_at(s_, t):
        out = flat(DESK.copy(), page(s_["page"]))
        u = (t - s_["t0"]) / s_["dur"]
        if u >= 1:
            return flat(out, COVER)
        th = math.pi * (1 - ease(max(0.0, u)))
        if th >= math.pi / 2: return out
        im, m = _warp(COVER, _turn_quad(th)); return comp(out, im, m)

    def photo_at(s_, t):
        kb = s_.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - s_["t0"]) / max(0.5, s_["t_end"] - s_["t0"]))
        im = s_["im"]; h, w = im.shape[:2]; z = kb[2] + (kb[5] - kb[2]) * u
        cx, cy = kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u
        sc = max(W / w, H / h) * max(1.0, z); cw, ch = W / sc, H / sc
        x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
        M = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]], np.float32)
        img = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        dim = s_.get("dim", 0.3)
        return img * (1 - dim) * VIG + NAVY * dim * 0.35

    def scene_img(s_, t):
        return book_at(s_, t) if s_["kind"] == "book" else close_at(s_, t) if s_["kind"] == "close" else photo_at(s_, t)

    def base_at(t):
        cur = 0
        for i, s_ in enumerate(SC):
            if s_["t0"] <= t: cur = i
        s_ = SC[cur]; img = scene_img(s_, t)
        if cur > 0 and XF and t < s_["t0"] + XF and (s_["kind"] == "photo" or SC[cur - 1]["kind"] == "photo"):
            a = (t - s_["t0"]) / XF; img = scene_img(SC[cur - 1], t) * (1 - a) + img * a
        return img, s_

    def frame(t):
        base, s_ = base_at(t)
        over = max([win(t, c["t0"] - 0.1, c["t1"], 0.25, 0.25) for c in CARDS if c["type"] == "quote"] + [0])
        if END_AT is not None: over = max(over, seg(t, END_AT - 0.1, END_AT + 0.3) * 0.6)
        if over > 0: base = base * (1 - 0.4 * over)
        cv = Canvas(base)
        if s_["kind"] == "photo" and s_.get("credit"):
            ca = win(t, s_["t0"], s_["t_end"], 0.15, 0.15)
            cv.text("Photo: " + s_["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * ca, bold=False, maxw=1000, minsz=36)
        for c in CARDS:
            if c["type"] == "quote":
                qa = win(t, c["t0"] - 0.08, c["t1"], 0.2, 0.2)
                if qa <= 0.01: continue
                y = c.get("y", 900); hh = c.get("h", 560)
                pop_ = 0.96 + 0.04 * eout((t - c["t0"] + 0.08) / 0.2)
                cc = cv.c; cc.save(); cc.translate(540, y); cc.scale(pop_, pop_); cc.translate(-540, -y)
                WG.panel(cv, 60, y - hh / 2, 960, hh, qa, stroke=YEL)
                if c.get("mark", True): cv.text("“", 130, y - hh / 2 + 80, 150, YEL, qa, align="l", shadow=False)
                _wrapped(cv, c["text"], y - 30, qa, c.get("size", 70), 840)
                if c.get("who"):
                    for j, wl in enumerate(c["who"].split("|")):
                        cv.text(wl, 540, y + hh / 2 - 92 + 50 * j - 25 * (len(c["who"].split("|")) - 1), 40, "#C9D6EA", qa, bold=False, maxw=900, minsz=36)
                cc.restore()
            elif c["type"] == "stamp":
                sa = win(t, c["t0"] - 0.05, c["t1"], 0.08, 0.2)
                if sa <= 0.01: continue
                k = 1.0 + 0.9 * (1 - eout((t - c["t0"]) / 0.16)) if t < c["t0"] + 0.16 else 1.0
                col = STAMP_COL.get(c.get("color", "red"), RED); sz = c.get("size", 110) * k
                y = c.get("y", 1180); ang = math.radians(c.get("angle", -8)); txt = c["text"]
                cc = cv.c; cc.save(); cc.translate(540, y); cc.rotate(ang); cc.translate(-540, -y)
                tw = min(cv.tw(txt, sz), 900) + 80
                cv.rrect(540 - tw / 2, y - sz * 0.72, tw, sz * 1.44, 18, "#0B1730", 0.55 * sa, stroke=col, lw=10 * k, sa=sa)
                cv.text(txt, 540, y, sz, col, sa, maxw=880, shadow=False, minsz=36)
                if c.get("who"): cv.text(c["who"], 540, y + sz * 0.72 + 50, 40, "#E8EDF5", sa, bold=False, maxw=900, minsz=36)
                cc.restore()
        if D.get("draw_sources", False): draw_sources(cv, S, T, t, D.get("src_y", 1488))
        draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0) if D.get("hook_until") is None else A(D["hook_until"]) - 0.15, HY, HS)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    for s_ in SC:
        if s_["kind"] == "book":
            n = len(s_["seq"]) - 1; u0 = s_.get("u0", 0.0)
            for j in range(n):                                   # one paper flip per page, on its turn start
                # invert p = eout(u)**0.9 * n for p = j (page j starts turning)
                pj = j / n; u = 1 - (1 - min(0.999, pj ** (1 / 0.9))) ** (1 / 3)
                if u < u0: continue
                tj = s_["t0"] + (u - u0) / (1 - u0) * (s_["t1f"] - s_["t0"]) if u0 < 1 else s_["t0"]
                ev.append((max(0.0, tj), "flip", -14 if j == n - 1 else -17))
            ev.append((s_["t1f"] - 0.02, "thud", -24 if s_["t0"] > 0.01 else -18))
        elif s_["kind"] == "photo": ev.append((max(0.0, s_["t0"] - 0.15), "whoosh", -28))
        elif s_["kind"] == "close": ev.append((s_["t0"] + s_["dur"] - 0.03, "slam", -12))
    ev += [(c["t0"], "thud", -16) for c in CARDS if c["type"] == "stamp"]
    ev += [(c["t0"], "pop", -26) for c in CARDS if c["type"] == "quote"]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=s_["file"], credit=s_.get("credit")) for s_ in SC if s_["kind"] == "photo"],
                flips=[dict(page=s_["page"], seq=s_.get("seq"), start=round(s_["t0"], 2), land=round(s_.get("t1f", s_["t0"]), 2))
                       for s_ in SC if s_["kind"] in ("book", "close")])
