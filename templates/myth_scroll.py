"""myth_scroll: deadpan "myth busting" comedy reel. A parchment scroll unrolls with a historic accusation, a red MYTH stamp slams
on it, a running counter at the top ("Myths busted 1 of 4") ticks up, a deadpan joke card follows with a rimshot sting, then the
mood turns: a fact card (real photo band edge to edge, big yellow year, what really happened, source chip) under a soft piano bed,
then back to the light comic bed for the next myth. Closes on a serious two-line end card and loops back into frame 0.

Music: the comic bed (`music.bed`, default "skip") plays under myths and jokes; under fact and end scenes it crossfades into
`music.serious` (default "soft_piano"). Both at -16 LUFS, ducked under speech. SFX: paper unroll, stamp slam, counter ding, rimshot.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  title    {head: "MEDIEVAL MYTHS ABOUT JEWS", items: ["Blood in matzah", ...]}   the frame-0 scroll (a numbered list)
  counter  {label: "Myths busted", of: 4, y: 128}
  scenes   in order, each from `at` until the next scene:
           {kind: "title", at: 0}
           {kind: "myth", at, n, text ("|" = line break, *word* = red ink), stamp_at, stamp: "MYTH"}
           {kind: "joke", at, text (*word* = yellow), emoji, sting (anchor of the rimshot), photo: {file, credit, kb}}
           {kind: "fact", at, kicker, big, text (*word* = yellow), photo: {file, credit, kb}, quote: true}
           {kind: "end", at, l1, l2, sub}
  cut      crossfade seconds between scenes (default 0.22);  tighten (default 0.16): max pause inside a line
  hook_y   default 300;  subs_y default 1612;  src_y default 1488
"""
import os, sys, math, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
from common import *
import common as CMN

PARCH, PARCH_D, PARCH_E, INK, RED_INK, MUTE = "#F3E6C8", "#E3CFA2", "#C9AE78", "#3A2614", "#B3262E", "#8A6A3E"
SX0, SX1 = 80, 1000                       # scroll paper x range
BAND_H = 760                              # fact/joke photo band height (edge to edge, from y 0)


# ------------------------------------------------------------------ sound
def _sfx_patch():
    orig = CMN._sfx_lib
    def lib():
        d = orig(); rng = np.random.default_rng(17); tt = lambda d_: np.arange(int(d_ * SR)) / SR
        def tom(f0, d_=0.22):
            x = tt(d_); f = f0 * (1 + 0.6 * np.exp(-x * 30)); y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 13)
            nz = rng.standard_normal(len(x)) * np.exp(-x * 60) * 0.25; return y + nz
        def cym(d_=1.1):
            x = tt(d_); nz = rng.standard_normal(len(x))
            from scipy.signal import butter, sosfilt
            nz = sosfilt(butter(2, 5500, "hp", fs=SR, output="sos"), nz)
            return nz * (np.minimum(1, x / 0.003) * np.exp(-x * 3.2)) * 1.3
        def rimshot():                                  # ba-dum-tss
            y = np.zeros(int(1.6 * SR))
            for t0, sig, g in ((0.0, tom(210), 0.8), (0.15, tom(150), 0.9), (0.33, cym(), 0.55)):
                i = int(t0 * SR); y[i:i + len(sig)] += g * sig[:len(y) - i]
            return y / np.abs(y).max()
        def unroll():
            x = tt(0.5); n = rng.standard_normal(len(x))
            n = np.convolve(n, np.ones(5) / 5, "same") - np.convolve(n, np.ones(60) / 60, "same")
            env = np.sin(np.pi * np.clip(x / 0.5, 0, 1)) ** 1.2 * (0.6 + 0.4 * np.sin(2 * np.pi * 13 * x) ** 2)
            return n * env / (np.abs(n * env).max() + 1e-9)
        def slam():
            x = tt(0.5); low = np.sin(2 * np.pi * (85 - 30 * x) * x) * np.exp(-x * 14)
            n = rng.standard_normal(len(x)) * np.exp(-x * 45)
            y = low + 0.6 * np.convolve(n, np.ones(5) / 5, "same"); return y / np.abs(y).max()
        d.update(rimshot=rimshot, unroll=unroll, slam=slam); return d
    CMN._sfx_lib = lib


def _mix(S, T, work, events, serious):
    """voice + comic bed crossfading into a serious bed inside `serious` windows + SFX -> work/mix.m4a (loudnorm)"""
    import soundfile as sf
    M = S.get("music") or {}; n = int(T.END * SR)
    vo, _ = sf.read(f"{work}/vo.wav"); vo = np.pad(vo, (0, max(0, n - len(vo))))[:n]
    def norm_bed(name):
        b = bed(name, T.END + 0.6)[:, :n]; b = np.pad(b, ((0, 0), (0, max(0, n - b.shape[1]))))
        return b * 10 ** ((M.get("bed_lufs", -16) - lufs_of(b)) / 20)
    bc = norm_bed(M.get("bed", "skip")); bs = norm_bed(M.get("serious", "soft_piano")) * 10 ** (M.get("serious_db", 1.5) / 20)
    s = np.zeros(n)
    for a, b in serious: s[max(0, int(a * SR)):int(b * SR)] = 1
    k = int(0.45 * SR); s = np.convolve(s, np.ones(k) / k, "same")
    x = bc * (1 - s)[None] + bs * s[None]
    m = np.zeros(n)
    for l in T.L: m[max(0, int((l["t0"] - 0.08) * SR)):int((l["t1"] + 0.12) * SR)] = 1
    k = int(0.12 * SR); m = np.convolve(m, np.ones(k) / k, "same"); x *= (10 ** (-M.get("duck_db", 9) * m / 20))[None]
    sfx = np.zeros(n); K = CMN._sfx_lib()
    for t0, kind, db in events:
        y = K[kind](); i = int(t0 * SR); j = min(n, i + len(y))
        if 0 <= i < n: sfx[i:j] += y[:j - i] * 10 ** (db / 20)
    return loudnorm(np.stack([vo, vo]) + x + np.stack([sfx, sfx]), f"{work}/mix.m4a", M.get("lufs", -16))


def _tighten(J, work, maxgap=0.16):
    """Noam's pacing rule: pauses inside a line cut to maxgap seconds; audio and timings shift together (as rulebook)"""
    import soundfile as sf
    y, sr = sf.read(f"{work}/vo.wav"); cuts = []
    hop = int(0.01 * sr); n = len(y) // hop
    rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)); quiet = rms < 0.004
    for L in J["lines"]:
        i0, i1 = int(L["t0"] / 0.01), min(n, int(L["t1"] / 0.01)); k = i0
        while k < i1:
            if quiet[k]:
                j = k
                while j < i1 and quiet[j]: j += 1
                run = (j - k) * 0.01
                if run > maxgap + 0.07 and j < i1: cuts.append((k * 0.01 + 0.09, run - maxgap - 0.02))
                k = j
            else: k += 1
    if not cuts: return J, 0.0
    def sh(t): return t - sum(r for c, r in cuts if c < t)
    keep, last = [], 0
    for c, r in cuts:
        i0, i1 = int(c * sr), int((c + r) * sr); keep.append(y[last:i0]); last = i1
    keep.append(y[last:]); f = int(0.004 * sr); out = []
    for k_ in keep:
        k_ = k_.copy()
        if len(k_) > 2 * f: k_[:f] *= np.linspace(0, 1, f); k_[-f:] *= np.linspace(1, 0, f)
        out.append(k_)
    sf.write(f"{work}/vo.wav", np.concatenate(out), sr)
    for L in J["lines"]:
        L["t0"] = round(sh(L["t0"]), 3); L["t1"] = round(sh(L["t1"]), 3)
        for w in L["words"]: w["t0"] = round(sh(w["t0"]), 3); w["t1"] = round(sh(w["t1"]), 3)
    tot = sum(r for _, r in cuts); J["dur"] = round(J["dur"] - tot, 3); J["tightened_s"] = round(tot, 2)
    return J, tot


# ------------------------------------------------------------------ drawing helpers
def _toks(txt):
    out, on_ = [], False
    for w in txt.split():
        on = w.startswith("*"); off = w.rstrip(".,!?:;\"'").endswith("*")
        out.append((w.replace("*", ""), on_ or on)); on_ = (on_ or on) and not off
    return out


def _wrap(cv, txt, cx, cy, sz, col, hl, maxw, a=1.0, lh=1.2, shadow=False):
    """centred multi-line text; "|" forces a break; *span* in hl colour. returns the block height"""
    if a <= 0.01: return 0
    lines = []
    for part in txt.split("|"):
        cur = []
        for tk in _toks(part):
            if cur and cv.tw(" ".join(w for w, _ in cur + [tk]), sz) > maxw: lines.append(cur); cur = [tk]
            else: cur.append(tk)
        if cur: lines.append(cur)
    while sz > 56 and any(cv.tw(" ".join(w for w, _ in ln), sz) > maxw for ln in lines): sz -= 2
    hh = sz * lh; y0 = cy - hh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        tot = cv.tw(" ".join(w for w, _ in ln), sz); x = cx - tot / 2
        for k, (w, h) in enumerate(ln):
            s_ = w + (" " if k < len(ln) - 1 else "")
            cv.text(s_, x, y0 + i * hh, sz, hl if h else col, a, align="l", maxw=2000, shadow=shadow); x += cv.tw(s_, sz)
    return hh * len(lines)


def _scroll(cv, y0, y1, open_, a=1.0):
    """parchment scroll between y0 and y1, unrolled by open_ (0..1) from the centre. returns visible (top, bottom)"""
    cy = (y0 + y1) / 2; hh = (y1 - y0) / 2 * max(0.04, open_); top, bot = cy - hh, cy + hh
    cv.rrect(SX0 + 14, top + 16, SX1 - SX0, bot - top, 10, "#000000", 0.35 * a)        # shadow
    cv.rrect(SX0, top, SX1 - SX0, bot - top, 6, PARCH, a)
    cv.rrect(SX0, top, SX1 - SX0, bot - top, 6, None, a, stroke=PARCH_E, lw=3, sa=0.8 * a)
    for k in range(3):                                                                    # age spots
        cv.circ(SX0 + 140 + k * 330, top + (bot - top) * (0.25 + 0.28 * k), 70 + 20 * k, PARCH_D, 0.22 * a)
    for yy in (top, bot):                                                                 # the two rolls
        cv.rrect(SX0 - 26, yy - 30, SX1 - SX0 + 52, 60, 30, "#D8BE86", a)
        cv.rrect(SX0 - 26, yy - 30, SX1 - SX0 + 52, 22, 11, "#EAD6A6", 0.7 * a)
        cv.rrect(SX0 - 26, yy - 30, SX1 - SX0 + 52, 60, 30, None, a, stroke="#9C7E47", lw=3, sa=a)
        for xx in (SX0 - 40, SX1 + 40):
            cv.circ(xx, yy, 24, "#6B4A22", a); cv.circ(xx, yy, 10, "#9C7E47", a)
    return top, bot


def _stamp(cv, txt, cx, cy, t, t0, a=1.0, size=170, ang=-11, col=RED):
    if t < t0 - 0.02 or a <= 0.01: return
    u = (t - t0) / 0.16; k = 1.0 + 1.1 * (1 - eout(u)) if u < 1 else 1.0
    sa = a * min(1.0, (t - t0 + 0.02) / 0.06)
    cc = cv.c; cc.save(); cc.translate(cx, cy); cc.rotate(math.radians(ang)); cc.scale(k, k); cc.translate(-cx, -cy)
    tw = cv.tw(txt, size) + 90
    cv.rrect(cx - tw / 2, cy - size * 0.7, tw, size * 1.4, 22, None, sa, stroke=col, lw=12, sa=0.92 * sa)
    cv.rrect(cx - tw / 2 + 14, cy - size * 0.7 + 14, tw - 28, size * 1.4 - 28, 14, None, sa, stroke=col, lw=4, sa=0.7 * sa)
    cv.text(txt, cx, cy, size, col, 0.92 * sa, shadow=False)
    cc.restore()


def _photo_band(im, kb, u, bh=BAND_H):
    h, w = im.shape[:2]; z = kb[2] + (kb[5] - kb[2]) * u; cx = kb[0] + (kb[3] - kb[0]) * u; cy = kb[1] + (kb[4] - kb[1]) * u
    s = max(W / w, bh / h) * max(1.0, z); cw, ch = W / s, bh / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


# ------------------------------------------------------------------ render
def run(S):
    _sfx_patch()
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work)
    if D.get("tighten", 0.16): J, _cut = _tighten(J, work, float(D.get("tighten", 0.16))); log(f"pauses tightened by {_cut:.2f} s")
    T = Timing(J, D.get("tail", 0.9)); A = T.at
    TI = D.get("title", {}); CO = D.get("counter", {}); OF = int(CO.get("of", 4))
    SC = []
    for s_ in D["scenes"]:
        s_ = dict(s_); s_["t0"] = A(s_["at"])
        if s_.get("stamp_at") is not None: s_["ts"] = A(s_["stamp_at"])
        if s_.get("sting") is not None: s_["tst"] = A(s_["sting"])
        ph = s_.get("photo")
        if ph:
            p = ph["file"] if os.path.isabs(ph["file"]) else os.path.join(REPO, ph["file"])
            ph["im"] = cv2.imread(p, cv2.IMREAD_COLOR)
            if ph["im"] is None: raise SystemExit(f"myth_scroll: cannot read photo {ph['file']}")
        SC.append(s_)
    for i, s_ in enumerate(SC): s_["t1"] = SC[i + 1]["t0"] if i + 1 < len(SC) else T.END + 1
    STAMPS = sorted(s_["ts"] for s_ in SC if s_["kind"] == "myth")
    XF = float(D.get("cut", 0.22)); HY = D.get("hook_y", 300)
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"])
    HOOK_END = A(D.get("hook_until"), T.E(0))
    yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
    SPOT = (0.72 + 0.5 * np.exp(-(((xx_ - 540) / 760) ** 2 + ((yy_ - 900) / 820) ** 2)))[..., None]
    DESK = BG * SPOT; DARK = BG * (0.55 + 0.35 * np.exp(-(((xx_ - 540) / 700) ** 2 + ((yy_ - 1000) / 700) ** 2)))[..., None]
    del yy_, xx_
    fade = np.clip(np.arange(BAND_H, dtype=np.float32) / BAND_H, 0, 1)
    BAND_FADE = np.clip((fade - 0.72) / 0.28, 0, 1)[:, None, None]                         # photo fades into navy at its bottom

    def busted(t): return sum(1 for x in STAMPS if t >= x)

    def scene_img(s_, t):
        k = s_["kind"]; u_in = (t - s_["t0"])
        if k in ("fact", "end"): base = DARK.copy()
        else: base = DESK.copy()
        ph = s_.get("photo")
        if ph:
            kb = ph.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.48, 1.12]
            u = min(1.0, max(0.0, (t - s_["t0"]) / max(0.5, s_["t1"] - s_["t0"])))
            band = _photo_band(ph["im"], kb, u) * (1 - ph.get("dim", 0.12))
            if ph.get("fit"):                                    # whole image, contained, over a blurred cover crop
                band = cv2.GaussianBlur(band, (0, 0), 28) * 0.45
                ih, iw = ph["im"].shape[:2]; sc_ = min(1040 / iw, (BAND_H - 80) / ih) * (1 + 0.06 * u)
                fw, fh = int(iw * sc_), int(ih * sc_); fx, fy = (W - fw) // 2, max(0, (BAND_H - 40 - fh) // 2)
                band[fy:fy + fh, fx:fx + fw] = cv2.resize(ph["im"], (fw, fh), interpolation=cv2.INTER_CUBIC).astype(np.float32)[:BAND_H - fy]
            if ph.get("gray"):
                g = band.mean(2, keepdims=True); band = band * 0.25 + g * 0.75
            base[:BAND_H] = band * (1 - BAND_FADE) + base[:BAND_H] * BAND_FADE
        cv = Canvas(base)
        if k == "title":
            op = 0.62 + 0.38 * eout(t / 1.4)
            top, bot = _scroll(cv, 560, 1400, op)
            ca = seg(t, 0.0, 0.25) if t > 0 else 1.0
            cv.c.save(); cv.c.rectangle(0, top + 34, W, max(0, bot - top - 68)); cv.c.clip()
            cv.text(TI.get("head", "MEDIEVAL MYTHS"), 540, 680, 50, MUTE, 1.0, shadow=False, maxw=820)
            cv.rrect(390, 722, 300, 4, 2, MUTE, 0.6)
            for i, it in enumerate(TI.get("items", [])):
                yy = 830 + i * 135
                cv.text(f"{i + 1}.", 190, yy, 72, RED_INK, 1.0, align="l", shadow=False)
                cv.text(it, 290, yy, 72, INK, 1.0, align="l", shadow=False, maxw=640)
            cv.c.restore()
        elif k == "myth":
            op = eout(u_in / 0.45) if u_in < 0.45 else 1.0
            shake = 0.0
            if s_.get("ts") is not None and 0 <= t - s_["ts"] < 0.25: shake = 9 * math.sin((t - s_["ts"]) * 90) * (1 - (t - s_["ts"]) / 0.25)
            cv.c.save(); cv.c.translate(shake, shake * 0.4)
            top, bot = _scroll(cv, 330, 1400, op)
            cv.c.save(); cv.c.rectangle(0, top + 34, W, max(0, bot - top - 68)); cv.c.clip()
            cv.text(f"MYTH No. {s_.get('n', 1)}", 540, 470, 54, MUTE, 1.0, shadow=False)
            cv.rrect(420, 512, 240, 4, 2, MUTE, 0.6)
            cv.text("Accused, in medieval Europe:", 540, 590, 44, MUTE, 1.0, bold=False, shadow=False, maxw=820)
            _wrap(cv, s_["text"], 540, 830, 86, INK, RED_INK, 800)
            cv.c.restore()
            if s_.get("ts") is not None: _stamp(cv, s_.get("stamp", "MYTH"), 560, 1160, t, s_["ts"], size=176)
            cv.c.restore()
        elif k == "joke":
            ph = s_.get("photo"); ey = 470 if not ph else None
            pa = seg(t, s_["t0"] - 0.05, s_["t0"] + 0.3, eout)
            y0 = 790 if ph else 600; hh = 660 if ph else 760
            pop_ = 0.92 + 0.08 * back(seg(t, s_["t0"], s_["t0"] + 0.35, lambda x: x))
            cc = cv.c; cc.save(); cc.translate(540, y0 + hh / 2); cc.scale(pop_, pop_); cc.translate(-540, -(y0 + hh / 2))
            cv.rrect(60, y0, 960, hh, 40, "#F5F5F7", 0.96 * pa)
            cv.rrect(60, y0, 960, hh, 40, None, pa, stroke=SKY, lw=6, sa=pa)
            if s_.get("emoji"):
                WG.emo(cv, s_["emoji"], 540, y0 + 115, 150, pa)
                ty = y0 + 210 + (hh - 210) / 2
            else: ty = y0 + hh / 2
            _wrap(cv, s_["text"], 540, ty, 74, "#0B1730", "#1C6FB0", 860, pa)
            cc.restore()
            if s_.get("tst") is not None and t >= s_["tst"]:                               # the sting: a little badge
                ba = win(t, s_["tst"], s_["t1"], 0.12, 0.15)
                bs = 0.7 + 0.3 * back(seg(t, s_["tst"], s_["tst"] + 0.3, lambda x: x))
                cc.save(); cc.translate(880, y0 - 10); cc.rotate(math.radians(10)); cc.scale(bs, bs)
                cv.rrect(-120, -50, 240, 100, 50, YEL, ba); cv.text("BA DUM TSS", 0, 0, 40, "#0B1730", ba, shadow=False, maxw=210)
                cc.restore()
        elif k == "fact":
            fa = seg(t, s_["t0"] - 0.05, s_["t0"] + 0.35, eout)
            BY = BAND_H if ph else 420
            cv.text(s_.get("kicker", "WHAT REALLY HAPPENED"), 540, BY + 70, 48, SKY, fa, maxw=960)
            yb = BY + 70
            if s_.get("big"):
                ba = seg(t, s_["t0"] + 0.1, s_["t0"] + 0.5, eout)
                cv.text(s_["big"], 540, BY + 230, 200 if ph else 260, YEL, ba, maxw=980); yb = BY + (330 if ph else 380)
            if s_.get("quote"):
                cv.text("“", 110, yb + 70, 150, YEL, fa, align="l", shadow=False)
            _wrap(cv, s_["text"], 540, (yb + 1440) / 2 + 20, 66 if ph else 72, WHITE, YEL, 940, fa * seg(t, s_["t0"] + 0.2, s_["t0"] + 0.6), lh=1.22, shadow=True)
            if ph and ph.get("credit"):
                cv.text("Photo: " + ph["credit"], 1040, BAND_H - 40, 36, "#E8EDF5", 0.9 * fa, bold=False, align="r", maxw=1000, minsz=36)
        elif k == "end":
            ea = seg(t, s_["t0"] - 0.05, s_["t0"] + 0.4, eout)
            cv.text(s_.get("l1", ""), 540, 720, 100, WHITE, ea, maxw=980)
            ea2 = seg(t, A(s_.get("l2_at"), s_["t0"] + 1.0), A(s_.get("l2_at"), s_["t0"] + 1.0) + 0.4, eout)
            _wrap(cv, s_.get("l2", ""), 540, 960, 104, YEL, YEL, 980, ea2, shadow=True)
            if s_.get("sub"): cv.text(s_["sub"], 540, 1220, 48, "#C9D6EA", ea2, bold=False, maxw=960)
        if k == "joke" and ph and ph.get("credit"):
            cv.text("Photo: " + ph["credit"], 1040, BAND_H - 100, 36, "#E8EDF5", 0.9, bold=False, align="r", maxw=1000, minsz=36)
        return cv.out().astype(np.float32)

    def base_at(t):
        cur = 0
        for i, s_ in enumerate(SC):
            if s_["t0"] <= t: cur = i
        s_ = SC[cur]; img = scene_img(s_, t)
        if cur > 0 and t < s_["t0"] + XF:
            a = ease((t - s_["t0"]) / XF); img = scene_img(SC[cur - 1], t) * (1 - a) + img * a
        return img, s_

    def frame(t):
        base, s_ = base_at(t)
        cv = Canvas(base)
        # running counter (after the hook leaves, until the end card)
        end_t = next((x["t0"] for x in SC if x["kind"] == "end"), T.END)
        ca = win(t, HOOK_END + 0.2, end_t + 0.3, 0.35, 0.3)
        if ca > 0.01:
            y = CO.get("y", 128); n = busted(t)
            cv.rrect(60, y - 62, 960, 124, 62, "#0B1730", 0.86 * ca, stroke=SKY, lw=3, sa=0.7 * ca)
            cv.text(CO.get("label", "Myths busted"), 110, y, 50, WHITE, ca, align="l", shadow=False, maxw=420)
            last = max([x for x in STAMPS if x <= t] or [-9])
            pz = 1 + 0.35 * (1 - eout((t - last) / 0.35)) if 0 <= t - last < 0.35 else 1.0
            cc = cv.c; cc.save(); cc.translate(660, y); cc.scale(pz, pz)
            cv.text(f"{n} of {OF}", 0, 0, 64, YEL, ca, shadow=False); cc.restore()
            for i in range(OF):
                px = 800 + i * 58
                if i < n:
                    cv.circ(px, y, 22, RED, ca)
                    cv.line([(px - 9, y - 9), (px + 9, y + 9)], WHITE, 5, ca); cv.line([(px - 9, y + 9), (px + 9, y - 9)], WHITE, 5, ca)
                else: cv.circ(px, y, 20, None, ca, stroke="#9FB0C8", lw=3)
        if s_["kind"] in ("fact",) or s_.get("src_any"):
            pass
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, HOOK_END - 0.15, HY, D.get("hook_s", 1.0))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev, serious = [(0.05, "unroll", -22)], []
    for s_ in SC:
        if s_["kind"] == "myth":
            ev += [(s_["t0"], "unroll", -22), (s_["ts"], "slam", -12), (s_["ts"] + 0.05, "ding", -26)]
        elif s_["kind"] == "joke":
            ev.append((s_["t0"], "pop", -24))
            if s_.get("tst") is not None: ev.append((s_["tst"], "rimshot", -14))
        elif s_["kind"] == "fact":
            ev.append((max(0.0, s_["t0"] - 0.2), "whoosh", -30)); serious.append((s_["t0"] - 0.1, s_["t1"] - 0.05))
        elif s_["kind"] == "end":
            serious.append((s_["t0"] - 0.1, T.END + 1))
    aud = _mix(S, T, work, ev, serious)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, tightened_s=J.get("tightened_s", 0),
                photos=[dict(file=s_["photo"]["file"], credit=s_["photo"].get("credit")) for s_ in SC if s_.get("photo")],
                scenes=[dict(kind=s_["kind"], start=round(s_["t0"], 2)) for s_ in SC])
