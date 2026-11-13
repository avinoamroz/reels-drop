"""irony_cine: cinematic version of irony_reasons ("8 reasons I 'hate' X", every reason a good deed). Same safety design:
  * "SARCASM" eye-roll badge on EVERY frame (frame 0 included);
  * frame 0: a moving grid of the reasons' visuals behind the hook ("hate" in quotes) and a big eye-roll emoji;
  * reason 1 is on screen by `data.first_at` (default 1.2 s); subtitles keep "hate" in quotes (spec subs_replace);
  * warm reveal end card, then the dissolve back into frame 0.
Presentation:
  * shots: every reason has 2 or 3 full-bleed shots (real video clips or stills with a strong Ken Burns push), so the
    picture changes at least every ~1.5 s; a whip pan (motion-blurred slide) between reasons, a zoom-punch cut inside one;
  * kinetic text: the deed slams in (scale + tilt), who/when slides up, the punch word hits with a scale bounce,
    a screen shake and a comedy sting (boing / rimshot);
  * animated counter badge "Reason 3 of 8" (flips in), and a GRUMPY METER that fills by reason and maxes out on 8.

data keys (times are anchors: 12.5, "L3", "L3:word", "L3:word+0.2", "END-2"):
  badge, badge_emoji, hook_emoji, hook_y, first_at, of, meter (bool, default true)
  reasons [{n, at, title, who, src, punch, punch_at, emoji, sting: boing|rim|horn,
            shots: [{at, clip | photo, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], gray, dim, ss (clip start offset s)}]}]
            the first shot starts at the reason's start (reason 1: first_at); each shot lasts until the next one
  end     {at, lines: [{text, at, size, y}], pill: {text, at, y}, emoji, emoji_y}
Clips: repo mp4s (any size, cover-cropped to 1080x1920), read frame by frame, looping ping-pong if the shot outlasts the clip.
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as _CM

PCOL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)


# ------------------------------------------------------------------ comedy stings (added to the shared SFX set, this module only)
def _sfx_plus():
    K = _ORIG_SFX()
    tt = lambda d: np.arange(int(d * SR)) / SR
    def boing():
        x = tt(0.55); f = 180 + 260 * np.exp(-x * 6) * (1 + 0.35 * np.sin(2 * np.pi * 14 * x))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 5.5)
    def rim():
        x = tt(0.5); rng = np.random.default_rng(7)
        snare = rng.standard_normal(len(x)) * np.exp(-x * 22) * 0.6 + np.sin(2 * np.pi * 190 * x) * np.exp(-x * 30)
        y = snare.copy(); d = int(0.13 * SR); y[d:] += snare[:-d] * 0.8
        cym = rng.standard_normal(len(x)) * np.exp(-x * 6) * 0.25
        y[2 * d:] += cym[:-2 * d] if 2 * d < len(x) else 0
        return y / (np.abs(y).max() + 1e-9)
    def horn():
        x = tt(0.45); f = 240 - 60 * x / 0.45
        y = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5 * np.exp(-x * 2.5) * np.clip(x / 0.02, 0, 1)
        return y
    def swish():
        x = tt(0.28); rng = np.random.default_rng(11); n = rng.standard_normal(len(x))
        return n * np.sin(np.pi * x / 0.28) ** 2 * 0.6
    K.update(boing=boing, rim=rim, horn=horn, swish=swish)
    return K


_ORIG_SFX = _CM._sfx_lib


class Clip:
    """whole clip decoded up front by an ffmpeg subprocess (no in-process decoder threads, so the forked render workers
    cannot deadlock), 540x960 at 24 fps; frame at local time t, ping-pong looping when the shot outlasts the clip"""
    FPS_ = 24.0
    def __init__(s, path):
        p = path if os.path.isabs(path) else os.path.join(REPO, path)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vf", "scale=540:960:force_original_aspect_ratio=increase,crop=540:960,fps=24",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
        s.fr = np.frombuffer(raw, np.uint8).reshape(-1, 960, 540, 3)
        if not len(s.fr): raise SystemExit(f"irony_cine: no frames in clip {path}")
        s.n = len(s.fr); s.fps = s.FPS_
    def frame(s, t):
        k = int(t * s.fps); m = max(1, s.n - 1); k = k % (2 * m); k = k if k <= m else 2 * m - k
        return s.fr[k]


def _load(path, gray=False):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"irony_cine: cannot read photo {path}")
    if gray: g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); im = cv2.merge([g, g, g])
    return im


def _kb(im, cx, cy, z, w_=W, h_=H):
    h, w = im.shape[:2]; s = max(w_ / w, h_ / h) * max(1.0, z)
    cw, ch = w_ / s, h_ / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w_, h_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    E = S.get("end") or {}
    END_AT = A(E.get("at"), T.S(len(T.L) - 2))
    FIRST = float(D.get("first_at", 1.2))
    R = []
    for k, r in enumerate(D["reasons"]):
        r = dict(r); r["t0"] = FIRST if k == 0 else A(r["at"]); R.append(r)
    for k, r in enumerate(R):
        r["t1"] = R[k + 1]["t0"] if k + 1 < len(R) else END_AT
        r["pt"] = A(r["punch_at"]) if r.get("punch_at") else r["t1"] - 1.2
    OF = int(D.get("of", len(R)))
    cache = {}
    SH = []                                              # all shots, flattened
    for k, r in enumerate(R):
        for j, s_ in enumerate(r["shots"]):
            s_ = dict(s_); s_["t0"] = r["t0"] if j == 0 else A(s_["at"]); s_["first"] = j == 0; s_["r"] = k
            src = s_.get("clip") or s_.get("photo")
            if src not in cache: cache[src] = Clip(src) if s_.get("clip") else _load(src, s_.get("gray", False))
            s_["src"] = cache[src]; SH.append(s_)
    for k, s_ in enumerate(SH): s_["t1"] = SH[k + 1]["t0"] if k + 1 < len(SH) else END_AT + 0.35
    WHIP = 0.24; ZP = 0.22

    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.5 * 0.55, np.clip((y_ - 860) / 1060, 0, 1) ** 1.0 * 0.86))[..., None]
    NAVY = np.array([97, 50, 22], np.float32)

    def shot_img(s_, t):
        kb = s_.get("kb") or ([0.5, 0.5, 1.0, 0.5, 0.5, 1.1] if s_.get("clip") else [0.5, 0.45, 1.0, 0.5, 0.42, 1.28])
        u = ease((t - s_["t0"]) / max(0.4, s_["t1"] - s_["t0"] + WHIP))
        if s_.get("clip"):
            fr = s_["src"].frame(max(0.0, t - s_["t0"]) + float(s_.get("ss", 0)))
            if fr is None: fr = np.zeros((H, W, 3), np.uint8)
            if s_.get("gray"): g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); fr = cv2.merge([g, g, g])
        else: fr = s_["src"]
        img = _kb(fr, kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = s_.get("dim", 0.1)
        return img * (1 - dim) * VIG + NAVY * dim * 0.35

    # frame-0 grid of the first shot of each reason (moving tiles)
    G = []
    for r in R[:8]:
        s_ = next(x for x in SH if x["r"] == R.index(r))
        G.append(s_)
    def grid_at(t):
        out = np.zeros((H, W, 3), np.float32)
        for i, s_ in enumerate(G):
            cx_, cy_ = (i % 2) * 540, (i // 2) * 480
            src = s_["src"].frame(t + float(s_.get("ss", 0))) if s_.get("clip") else s_["src"]
            kb = s_.get("kb") or [0.5, 0.5, 1.0]
            z = 1.0 + 0.06 * t + 0.02 * (i % 3)
            out[cy_:cy_ + 480, cx_:cx_ + 540] = _kb(src, kb[0], kb[1], z, 540, 480)
        out = out * 0.45 + NAVY * 0.2
        out[:, 537:543] = (48, 26, 11)
        for yy in (477, 957, 1437): out[yy:yy + 6, :] = (48, 26, 11)
        return out

    def hblur(img, k):
        k = int(k)
        return cv2.blur(img, (k, 1)) if k > 2 else img

    def base_at(t):
        if t < FIRST: return grid_at(t)
        cur = None
        for k, s_ in enumerate(SH):
            if s_["t0"] <= t < s_["t1"]: cur = k; break
        if t >= END_AT:
            a = seg(t, END_AT, END_AT + 0.3)
            last = shot_img(SH[-1], min(t, END_AT + 0.3)) if t < END_AT + 0.3 else None
            return BG.copy() if last is None else last * (1 - a) + BG * a
        if cur is None: return BG.copy()
        s_ = SH[cur]; img = shot_img(s_, t); dt = t - s_["t0"]
        prev = SH[cur - 1] if cur > 0 else None
        if s_["first"] and dt < WHIP:                      # whip pan in from the right
            u = eout(dt / WHIP); off = int(W * (1 - u)); k = 140 * (1 - u) + 1
            old = grid_at(t) if prev is None else shot_img(prev, t)
            fr = np.empty_like(img)
            oo = int(-W * 0.55 * u)
            fr[:] = np.roll(hblur(old, k), oo, axis=1)
            nw = hblur(img, k)
            if off < W: fr[:, off:] = nw[:, :W - off]
            return fr
        if not s_["first"] and dt < ZP:                    # zoom punch + flash
            u = eout(dt / ZP); z = 1.16 - 0.16 * u
            M = cv2.getRotationMatrix2D((540, 960), 0, z); img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            img = img + 255 * 0.28 * (1 - u)
        return img

    BADGE = D.get("badge", "SARCASM"); BEMO = D.get("badge_emoji", "🙄")
    def badge(cv):
        tw = cv.tw(BADGE, 44) + 130
        cv.rrect(540 - tw / 2, 92, tw, 84, 42, "#0B1730", 0.88, stroke=SKY, lw=3, sa=0.9)
        WG.emo(cv, BEMO, 540 - tw / 2 + 52, 134, 54, 1.0)
        cv.text(BADGE, 540 + 32, 134, 44, SKY, 1.0, maxw=tw - 100, shadow=False)

    def counter(cv, r, t):
        a = win(t, r["t0"], r["t1"] + 0.05, 0.12, 0.15)
        if a <= 0.01: return
        u = (t - r["t0"]) / 0.4; sc = back(min(1.0, max(0.0, u)), 2.2) if u < 1 else 1.0
        ang = math.radians(-14 * (1 - eout(min(1.0, max(0.0, u)))) - 3)
        y = 268; c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.scale(max(0.02, sc), max(0.02, sc)); c.translate(-540, -y)
        tw = cv.tw(f"Reason {r['n']} of {OF}", 84) + 110
        cv.rrect(540 - tw / 2, y - 68, tw, 136, 68, YEL, 0.97 * a)
        cv.rich([("Reason ", "#0B1730"), (str(r["n"]), RED), (f" of {OF}", "#0B1730")], 540, y, 86, a, 1000)
        c.restore()

    def meter(cv, t):
        if not D.get("meter", True) or t < FIRST: return
        a = seg(t, FIRST, FIRST + 0.3) * (1 - seg(t, END_AT - 0.1, END_AT + 0.2))
        if a <= 0.01: return
        # level from the timeline (frames render out of order): steps at each reason start and punch, eased
        lv = lv0 = 0.0; last_t = None
        for r in R:
            for tt_, v in ((r["t0"], (r["n"] - 0.6) / OF), (r["pt"], r["n"] / OF)):
                if t >= tt_: lv0 = lv; lv = v; last_t = tt_
        if last_t is not None: lv = lv0 + (lv - lv0) * eout((t - last_t) / 0.35)
        x0, y0, w, h = 150, 370, 780, 50
        shake = 0
        for r in R:
            if r["pt"] <= t < r["pt"] + 0.4: shake = 6 * math.sin((t - r["pt"]) * 70) * (1 - (t - r["pt"]) / 0.4)
        cv.rrect(x0 - 8 + shake, y0 - 8, w + 16, h + 16, (h + 16) / 2, "#0B1730", 0.85 * a, stroke="#C9D6EA", lw=2, sa=0.6 * a)
        col = YEL if lv < 0.55 else ("#F2A93B" if lv < 0.85 else RED)
        if lv > 0.01: cv.rrect(x0 + shake, y0, w * min(1.0, lv), h, h / 2, col, a)
        cv.text("GRUMPY METER", x0 + w / 2 + shake, y0 + h / 2 + 1, 36, "#0B1730" if lv > 0.62 else WHITE, a, maxw=w - 20, shadow=False, minsz=36)
        WG.emo(cv, "😤" if lv >= 0.99 else ("😒" if lv > 0.4 else "😐"), x0 + w + 52 + shake, y0 + h / 2, 64, a)

    def card(cv, r, t):
        a = win(t, r["t0"] + 0.05, r["t1"] + 0.05, 0.12, 0.18)
        if a <= 0.01: return
        u = (t - (r["t0"] + 0.05)) / 0.32; sc = 1.0 + 0.7 * (1 - back(min(1.0, max(0.0, u)), 1.6)) if u < 1 else 1.0
        ty = D.get("title_y", 1070); c = cv.c; tsz = r.get("size", 112)
        tw_ = cv.tw(r["title"].replace("*", ""), tsz)
        if tw_ > 960: tsz = int(tsz * 960 / tw_)
        c.save(); c.translate(540, ty); c.rotate(math.radians(-4 * (1 - eout(min(1.0, max(0.0, u)))))); c.scale(sc, sc); c.translate(-540, -ty)
        WG.title(cv, r["title"], ty, a * min(1.0, max(0.0, u * 3)), tsz)
        c.restore()
        if r.get("who"):
            wu = eout((t - r["t0"] - 0.3) / 0.3)
            if wu > 0.01: cv.text(r["who"], 540, D.get("who_y", 1185) + 40 * (1 - wu), 52, "#E6EEF8", a * wu, maxw=990, minsz=40)
        if r.get("src"): WG.src(cv, r["src"], a, D.get("src_y", 1488))

    def credit(cv, t):
        for s_ in SH:
            if s_["t0"] <= t < s_["t1"] and s_.get("credit") and t >= FIRST and t < END_AT:
                cv.text(("Video: " if s_.get("clip") else "Photo: ") + s_["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85, bold=False, maxw=1000, minsz=36)

    def punch(cv, r, t):
        pt = r["pt"]
        if not r.get("punch") or t < pt - 0.02 or t > r["t1"] + 0.05: return
        pa = win(t, pt - 0.02, r["t1"] + 0.05, 0.05, 0.15)
        u = (t - pt) / 0.38
        k = 1.0 + 1.2 * (1 - back(min(1.0, max(0.0, u)), 2.6)) if u < 1 else 1.0 + 0.03 * math.sin((t - pt) * 9)
        col = PCOL.get(r.get("punch_color", "yellow"), YEL); sz = r.get("punch_size", 140) * k; y = D.get("punch_y", 1335)
        ang = math.radians(r.get("angle", -7))
        c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.translate(-540, -y)
        tw = min(cv.tw(r["punch"], sz), 800) + 200
        cv.rrect(540 - tw / 2, y - sz * 0.68, tw, sz * 1.36, 24, col, 0.96 * pa)
        cv.text(r["punch"], 540 - 50, y, sz, "#0B1730", pa, maxw=800, shadow=False, minsz=40)
        WG.emo(cv, r.get("emoji", "🙄"), 540 + tw / 2 - 74, y, sz * 0.78, pa)
        c.restore()

    def end_card(cv, t):
        if t < END_AT - 0.1: return
        for ln in E.get("lines", []):
            t0 = A(ln.get("at"), END_AT); la = seg(t, t0 - 0.05, t0 + 0.3)
            if la > 0.01:
                u = (t - t0) / 0.35; sc = 1.0 + 0.5 * (1 - back(min(1.0, max(0.0, u)), 1.6)) if u < 1 else 1.0
                y = ln.get("y", 900); c = cv.c; c.save(); c.translate(540, y); c.scale(sc, sc); c.translate(-540, -y)
                WG.title(cv, ln["text"], y, la, ln.get("size", 96)); c.restore()
        if E.get("emoji"):
            ea = seg(t, END_AT, END_AT + 0.3); esz = 190 * WG.pop_s(t, END_AT) if t < END_AT + 0.35 else 190
            if ea > 0.01 and esz > 4: WG.emo(cv, E["emoji"], 540, E.get("emoji_y", 560), esz, ea)
        P_ = E.get("pill")
        if P_:
            t0 = A(P_.get("at"), END_AT + 1.0); pa = seg(t, t0, t0 + 0.3)
            if pa > 0.01:
                tw = min(cv.tw(P_["text"].replace("*", ""), 64) + 110, 980); y = P_.get("y", 1300)
                cv.rrect(540 - tw / 2, y - 64, tw, 128, 64, SKY, pa)
                WG.title(cv, P_["text"], y, pa, 64)

    def shake_off(t):
        for r in R:
            if r.get("punch") and r["pt"] <= t < r["pt"] + 0.35:
                d = (t - r["pt"]) / 0.35; amp = 22 * (1 - d) ** 2
                return int(amp * math.sin((t - r["pt"]) * 95)), int(amp * 0.7 * math.cos((t - r["pt"]) * 77))
        return 0, 0

    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"], size=130); HY = D.get("hook_y", 820)

    def frame(t):
        base = base_at(t); cv = Canvas(base)
        for r in R:
            if r["t0"] - 0.05 <= t < r["t1"] + 0.3: card(cv, r, t); punch(cv, r, t)
        meter(cv, t)
        for r in R:
            if r["t0"] - 0.05 <= t < r["t1"] + 0.3: counter(cv, r, t)
        end_card(cv, t); credit(cv, t)
        ha = 1 - seg(t, FIRST - 0.05, FIRST + 0.15)
        if ha > 0.01: WG.emo(cv, D.get("hook_emoji", "🙄"), 540, HY + 330, 230 * (0.94 + 0.06 * eout(t / 0.2)) * (1 + 0.04 * math.sin(t * 7)), ha)
        out = cv.out().astype(np.float32)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 0.94 + 0.06 * eout(t / 0.2), ha)
        dx, dy = shake_off(t)
        if dx or dy: out = cv2.warpAffine(out, np.float32([[1.03, 0, dx - 16], [0, 1.03, dy - 29]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        c2 = Canvas(out); badge(c2); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(r["t0"], "swish", -24) for r in R]
    ev += [(s_["t0"], "pop", -30) for s_ in SH if not s_["first"]]
    ev += [(r["pt"], r.get("sting", "boing"), -17) for r in R if r.get("punch")]
    ev += [(r["pt"], "thud", -22) for r in R if r.get("punch")]
    ev += [(END_AT, "ding", -24)]
    _CM._sfx_lib = _sfx_plus
    try: aud = build_mix(S, T, work, ev)
    finally: _CM._sfx_lib = _ORIG_SFX
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, shots=[dict(file=s_.get("clip") or s_.get("photo"), credit=s_.get("credit")) for s_ in SH])
