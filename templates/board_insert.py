"""board_insert: re-cut one stretch of a published scoreboard reel (the ethnic_cleansing "Not / Yes" sorting board) with a
new voiced segment, keeping everything before and after it from the published file.

  head    source [0, cut_a] with optional silent holds dropped (`drop`), video and audio untouched
  segment new Cartesia lines over a clean plate of the source: a photo card (real photos, Ken Burns), a title, two
          stamps per beat (word + tick or cross), chips that fly into the NOT column, burned-in subtitles in the
          source's own style (black pill, Heebo/Roboto bold)
  tail    source [cut_b, end] with the NOT column repainted (the new chips, and the source's later chip moved to the
          next free row), so the board stays true to the new opening. The source's own loop dissolve into frame 0 is kept.

Assets live under templates/photos/<id>/ (the spec-render checkout only has /templates/): src.mp4 (the published reel),
plate.png, notcol.png, chip_blank.png, chip_gaza.png, check.png, hdr.png (made from the source by a one-off prep).

data keys (segment times are anchors: 1.2, "L3", "L3e", "L3:word", "L3:word+0.2"; source times are plain seconds):
  source, plate, notcol, chip_blank, check     repo paths
  cut_a, cut_b                                  source seconds: head ends at cut_a, tail starts at cut_b
  drop        [[a, b], ...]                     source holds (no voice, no motion) cut out of the head
  xf          video cross-fade at both joins (default 0.2)
  photos      [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], fx: "scrub"|"suds"|""}]
  titles      [{at, until, text}]               *word* = yellow
  stamps      [{at, until, text, ok, x}]        ok true = green tick, false = red cross
  chips       [{label, emoji, at, row}]         flies from the card into NOT row `row` (0 based) at `at`
  tail_chips  [{row, sprite, fly: [t0, t1], check, from: [x, y, s], hdr: {file, t0, t1, y0}}]  source-time chip moves
  keep_src    [[a, b], ...]                     source windows where other chips fly over the NOT column (kept on top)
  card        [x, y, w, h]                      photo card box (default [120, 250, 840, 560])
"""
import os, sys, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
from PIL import Image, ImageDraw, ImageFont

HEEBO7 = os.path.join(TPL, "fonts", "Heebo-700.ttf")
NC_X, NC_Y = 60, 1160                         # NOT column patch origin (notcol.png is 473 x 356)
CHIP_X, ROW0, PITCH = 74, 1171, 84             # chip box left, row 0 top, row pitch
CHK_DX, CHK_DY = 408, 38                       # check centre inside the chip


def _p(p): return p if os.path.isabs(p) else os.path.join(REPO, p)


def _imread(p, flag=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(_p(p), flag)
    if im is None: raise SystemExit(f"board_insert: cannot read {p}")
    return im


_fonts = {}
def _font(sz):
    if sz not in _fonts: _fonts[sz] = ImageFont.truetype(HEEBO7, int(sz))
    return _fonts[sz]


def text_bgra(txt, sz, col=(255, 255, 255), hl=(255, 210, 63)):
    """Heebo 700 (Roboto Latin, the source's font) -> BGRA; *word* in yellow"""
    assert not DASH_RE.search(txt), txt
    f = _font(sz); parts = []; on = False
    for w in txt.split(" "):
        a = w.startswith("*"); b = w.rstrip(".,!?:;").endswith("*")
        parts.append((w.replace("*", ""), hl if (on or a) else col)); on = (on or a) and not b
    full = " ".join(p for p, _ in parts); wd = int(f.getlength(full)) + 16; ht = int(sz * 1.45)
    im = Image.new("RGBA", (wd, ht), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 8
    for i, (p, c) in enumerate(parts):
        s = p + (" " if i < len(parts) - 1 else "")
        d.text((x, ht / 2), s, font=f, fill=tuple(c) + (255,), anchor="lm"); x += f.getlength(s)
    a = np.asarray(im).copy(); return a[..., [2, 1, 0, 3]]


def emoji_bgra(ch, px):
    a = emoji(ch); a = cv2.resize(a, (int(a.shape[1] * px / 109), int(a.shape[0] * px / 109)), interpolation=cv2.INTER_AREA)
    return a[..., [2, 1, 0, 3]].copy()


def paste(dst, bgra, cx, cy, s=1.0, a=1.0):
    """alpha-paste a BGRA sprite centred at (cx, cy), scaled by s, into float BGR dst (in place)"""
    if a <= 0.004 or s <= 0.01: return dst
    if abs(s - 1) > 1e-3:
        bgra = cv2.resize(bgra, (max(1, int(bgra.shape[1] * s)), max(1, int(bgra.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
    h, w = bgra.shape[:2]; x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if xa >= xb or ya >= yb: return dst
    sp = bgra[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32); al = sp[..., 3:4] / 255 * a
    dst[ya:yb, xa:xb] = dst[ya:yb, xa:xb] * (1 - al) + sp[..., :3] * al
    return dst


def chip_sprite(blank, label, emo):
    c = blank.astype(np.float32).copy(); bgr = c[..., :3]
    paste(bgr, emoji_bgra(emo, 50), 48, 39)
    t = text_bgra(label, 48); paste(bgr, t, 89 - 8 + t.shape[1] / 2, 39)
    c[..., :3] = bgr; return np.clip(c, 0, 255).astype(np.uint8)


def row_c(r): return CHIP_X + 227, ROW0 + PITCH * r + 39


def round_mask(w, h, r):
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1); cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32)[..., None] / 255


def pill_bgra(text, ok):
    """stamp: rounded pill, word + green tick or red cross"""
    t = text_bgra(text, 54); w = t.shape[1] + 96; h = 92
    cv = Canvas(np.zeros((H, W, 3), np.float32)); col = "#1C7A4C" if ok else "#9E2A31"; stroke = GREEN if ok else RED
    cv.rrect(4, 4, w, h, 46, col, 0.96, stroke=stroke, lw=5)
    WG.mark(cv, "Y" if ok else "N", w - 36, 4 + h / 2, 28, 1.0)
    rgb = cv.out()[:h + 8, :w + 8].astype(np.float32)
    al = np.zeros((h + 8, w + 8), np.float32); m = round_mask(w + 2, h + 2, 47)[..., 0]; al[3:h + 5, 3:w + 5] = m
    out = np.dstack([rgb, al * 255]); paste(out[..., :3], t, 4 + 18 + t.shape[1] / 2 - 8, 4 + h / 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def read_src(src, t0, dur=None):
    """yield BGR uint8 frames of src from t0 (accurate seek)"""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", src] + (["-t", f"{dur:.3f}"] if dur else []) + \
          ["-vf", f"scale={W}:{H},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE); n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.wait()


def grab_src(src, t): return next(read_src(src, t, 0.1))


def src_audio(src, t0=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", src] + (["-t", f"{dur:.3f}"] if dur else []) + ["-vn", "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).T.astype(np.float64)


def edge_fade(x, d=0.008, head=True, tail=True):
    n = min(int(d * SR), x.shape[1] // 2); r = np.linspace(0, 1, n)
    x = x.copy()
    if head and n: x[:, :n] *= r
    if tail and n: x[:, -n:] *= r[::-1]
    return x


def run(S):
    D = S["data"]; work = workdir(S); src = _p(D["source"]); Ls = dur_of(src)
    cut_a, cut_b = float(D["cut_a"]), float(D["cut_b"]); XF = float(D.get("xf", 0.2)); nX = int(round(XF * FPS))
    drops = [tuple(map(float, d)) for d in D.get("drop", [])]

    # ---------------------------------------------------------------- segment
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.35)); A = T.at; SEG = T.END
    plate = _imread(D["plate"], cv2.IMREAD_COLOR).astype(np.float32)
    notcol = _imread(D["notcol"], cv2.IMREAD_COLOR).astype(np.float32)
    blank = _imread(D["chip_blank"]); CHECK = _imread(D["check"])
    cx_, cy_, cw_, ch_ = D.get("card", [120, 250, 840, 560]); CM_ = round_mask(cw_, ch_, 36)
    P = []
    for p in D["photos"]:
        im = _imread(p["file"], cv2.IMREAD_COLOR); P.append(dict(p, im=im, t0=A(p["at"])))
    for k, p in enumerate(P): p["t1"] = A(p["until"]) if p.get("until") else (P[k + 1]["t0"] if k + 1 < len(P) else SEG)
    TT = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), SEG), img=text_bgra(x["text"], 76)) for x in D.get("titles", [])]
    ST = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), SEG), img=pill_bgra(x["text"], x.get("ok", True))) for x in D.get("stamps", [])]
    CH = [dict(x, t0=A(x["at"]), img=chip_sprite(blank, x["label"], x["emoji"])) for x in D.get("chips", [])]
    SUB = sub_chunks(T, S.get("subs_replace"))
    SUBIMG = {c[0]: text_bgra(c[0], 50) for c in SUB}
    CRED = {k: text_bgra("Photo: " + p["credit"], 36, (201, 214, 234)) if p.get("credit") else None for k, p in enumerate(P)}
    fade_out = SEG - 0.42
    rng = np.random.default_rng(7); BUB = rng.random((26, 4))
    _mk = np.zeros((H, W, 1), np.float32); _mk[cy_:cy_ + ch_, cx_:cx_ + cw_] = CM_
    MK1 = (_mk, cv2.GaussianBlur(_mk[..., 0], (0, 0), 18)[..., None])

    def kb(im, k, u):
        z = k[2] + (k[5] - k[2]) * u; cx = k[0] + (k[3] - k[0]) * u; cy = k[1] + (k[4] - k[1]) * u
        h, w = im.shape[:2]; s = max(cw_ / w, ch_ / h) * z; vw, vh = cw_ / s, ch_ / s
        x0 = min(max(cx * w - vw / 2, 0), w - vw); y0 = min(max(cy * h - vh / 2, 0), h - vh)
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        return cv2.warpAffine(im, M, (cw_, ch_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    def board(out, t, upto=None):
        """NOT column with the chips that have landed by t (in place)"""
        for c in CH:
            land = c["t0"] + 0.45
            if t >= land:
                cxr, cyr = row_c(c["row"]); paste(out, c["img"], cxr, cyr)
                paste(out, CHECK, CHIP_X + CHK_DX, cyr, back(seg(t, land + 0.05, land + 0.35, lambda u: u)) if t < land + 0.35 else 1.0)
        return out

    def frame(t):
        out = plate.copy(); out[NC_Y:NC_Y + notcol.shape[0], NC_X:NC_X + notcol.shape[1]] = notcol
        ca = seg(t, 0.0, 0.3) * (1 - seg(t, fade_out, fade_out + 0.32))       # card alpha
        pop = 0.92 + 0.08 * eout(t / 0.35)
        if ca > 0.01:
            card = np.zeros((ch_, cw_, 3), np.float32); have = False
            for k, p in enumerate(P):
                if t < p["t0"] - 0.22 or t > p["t1"] + 0.22: continue
                u = (t - p["t0"]) / max(0.5, p["t1"] - p["t0"] + 0.3)
                lay = kb(p["im"], p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.1], min(1, max(0, u)))
                al = 1.0 if (k == 0 or not have) else seg(t, p["t0"] - 0.2, p["t0"] + 0.05, lambda v: v)
                card = lay if al >= 0.999 or not have else card * (1 - al) + lay * al; have = True
            # cartoon scrub: sponge emoji + suds
            for k, p in enumerate(P):
                fx = p.get("fx")
                if not fx or not (p["t0"] - 0.2 <= t <= p["t1"] + 0.2): continue
                fa = win(t, p["t0"] - 0.2, p["t1"] + 0.2, 0.2, 0.2)
                ccv = Canvas(card if card.shape == (H, W, 3) else np.pad(card, ((0, H - ch_), (0, W - cw_), (0, 0))))
                for i, (bx, by, br, bs) in enumerate(BUB):
                    ph = (t * (0.35 + 0.4 * bs) + by) % 1.0
                    x = cw_ * (0.15 + 0.7 * bx) + 14 * math.sin(6 * t + i); y = ch_ * (0.95 - 0.85 * ph)
                    ccv.circ(x, y, 8 + 22 * br, None, 1, stroke="#FFFFFF", lw=3); ccv.c.new_path()
                    ccv.circ(x - 4, y - 4, 3 + 4 * br, "#FFFFFF", 0.55 * fa * (1 - ph))
                card = ccv.out()[:ch_, :cw_].astype(np.float32) * fa + card * (1 - fa)
                if fx == "scrub":
                    sx = cw_ * 0.5 + 150 * math.sin(2 * math.pi * 2.4 * t); sy = ch_ * 0.42 + 40 * math.cos(2 * math.pi * 2.4 * t)
                    paste(card, emoji_bgra("\U0001F9FD", 150), sx, sy, 1.0, fa)
            # card into the frame: rounded mask, sky border, pop
            big = np.zeros((H, W, 3), np.float32); big[cy_:cy_ + ch_, cx_:cx_ + cw_] = card
            if abs(pop - 1) > 1e-3:
                mk = np.zeros((H, W, 1), np.float32); mk[cy_:cy_ + ch_, cx_:cx_ + cw_] = CM_
                M = cv2.getRotationMatrix2D((cx_ + cw_ / 2, cy_ + ch_ / 2), 0, pop)
                big = cv2.warpAffine(big, M, (W, H)); mk = cv2.warpAffine(mk, M, (W, H))[..., None]
                shadow = cv2.GaussianBlur(mk[..., 0], (0, 0), 18)[..., None]
            else:
                mk, shadow = MK1
            out = out * (1 - 0.45 * shadow * ca) + 0
            out = out * (1 - mk * ca) + big * mk * ca
            cv = Canvas(out); w2, h2 = cw_ * pop, ch_ * pop
            cv.rrect(540 - w2 / 2, cy_ + ch_ / 2 - h2 / 2, w2, h2, 36, None, 1, stroke=SKY, lw=4, sa=0.9 * ca)
            out = cv.out().astype(np.float32)
            for k, p in enumerate(P):
                a = win(t, p["t0"], p["t1"], 0.2, 0.2) * ca
                if CRED[k] is not None and a > 0.01: paste(out, CRED[k], 540, 1012, 1.0, 0.9 * a)
        for x in TT:
            a = win(t, x["t0"] - 0.05, x["t1"], 0.25, 0.2) * (1 - seg(t, fade_out, fade_out + 0.32))
            paste(out, x["img"], 540, 165, 0.94 + 0.06 * eout((t - x["t0"]) / 0.25), a)
        for x in ST:
            a = win(t, x["t0"], x["t1"], 0.12, 0.2) * (1 - seg(t, fade_out, fade_out + 0.32))
            if a > 0.01:
                s = 1.0 + 0.45 * (1 - eout((t - x["t0"]) / 0.18))           # slams down
                paste(out, x["img"], x.get("x", 540), x.get("y", 885), s, a)
        board(out, t)
        for c in CH:                                                       # flying chips
            u = (t - c["t0"]) / 0.45
            if 0 <= u < 1:
                cxr, cyr = row_c(c["row"]); e = ease(u)
                x = 540 + (cxr - 540) * e; y = 760 + (cyr - 760) * e - 120 * math.sin(math.pi * e)
                paste(out, c["img"], x, y, 1.12 - 0.12 * e, min(1, u * 6))
        for txt, a0, b0 in SUB:                                            # subtitles, source style
            if a0 - 0.05 <= t <= b0:
                al = min(1, (t - (a0 - 0.05)) / 0.08) * min(1, (b0 - t) / 0.08 + 1e-4); im = SUBIMG[txt]
                w = im.shape[1] + 44; cv = Canvas(out)
                cv.rrect(540 - w / 2, 1650 - 46, w, 92, 20, "#070C1C", 0.92 * al); out = cv.out().astype(np.float32)
                paste(out, im, 540, 1650, 1.0, al); break
        return np.clip(out, 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)

    # segment audio: voice + bed + SFX (whoosh at both joins, stamp thuds, chip pops)
    ev = [(0.0, "whoosh", -20), (max(0, SEG - 0.45), "whoosh", -22)]
    for x in ST: ev.append((x["t0"], "thud", -18))
    for c in CH: ev += [(c["t0"], "whoosh", -26), (c["t0"] + 0.45, "pop", -20), (c["t0"] + 0.5, "check", -22)]
    build_mix(S, T, work, ev, dur=SEG)
    seg_v = f"{work}/seg.mp4"; encode(frame, SEG, None, seg_v, loop=False)
    seg_a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f"{work}/mix.m4a", "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"],
                                         capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).T.astype(np.float64)
    n_seg = int(round(SEG * FPS)); seg_a = np.pad(seg_a, ((0, 0), (0, max(0, int(n_seg / FPS * SR) - seg_a.shape[1]))))[:, :int(n_seg / FPS * SR)]

    # ---------------------------------------------------------------- head (source, holds dropped)
    keep, t = [], 0.0
    for a, b in sorted(drops):
        keep.append((t, a)); t = b
    keep.append((t, cut_a))
    keep = [(round(a * FPS) / FPS, round(b * FPS) / FPS) for a, b in keep]
    fc = "".join(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}];" for i, (a, b) in enumerate(keep))
    fc += "".join(f"[v{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0,fps={FPS},format=yuv420p[v]"
    head_v = f"{work}/head.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "16",
                    "-preset", "medium", head_v], check=True)
    sa = src_audio(src)
    head_a = np.concatenate([edge_fade(sa[:, int(a * SR):int(b * SR)], head=k > 0, tail=k < len(keep) - 1) for k, (a, b) in enumerate(keep)], 1)
    head_d = sum(b - a for a, b in keep); n_head = int(round(head_d * FPS))

    # ---------------------------------------------------------------- tail (source, NOT column repainted)
    tmp = np.zeros((H, W, 3), np.float32); tmp[NC_Y:NC_Y + notcol.shape[0], NC_X:NC_X + notcol.shape[1]] = notcol
    board(tmp, 1e9); base_nc = tmp[NC_Y:NC_Y + notcol.shape[0], NC_X:NC_X + notcol.shape[1]].copy()
    hN, wN = base_nc.shape[:2]; reg = (slice(NC_Y, NC_Y + hN), slice(NC_X, NC_X + wN))
    TC = []
    for c in D.get("tail_chips", []):
        TC.append(dict(c, img=_imread(c["sprite"]), hdr=dict(c["hdr"], img=_imread(c["hdr"]["file"], cv2.IMREAD_COLOR).astype(np.float32)) if c.get("hdr") else None))
    keep_src = [tuple(map(float, k)) for k in D.get("keep_src", [])]
    refs = sorted(((float(r["from"]), grab_src(src, float(r["t"]))[reg].astype(np.float32)) for r in D.get("refs", [])), key=lambda x: x[0])
    f0 = grab_src(src, 0.0).astype(np.float32); fin = grab_src(src, Ls - 0.6).astype(np.float32)
    ex = (slice(1170, 1500), slice(560, 1020)); dv = (f0[ex] - fin[ex]); dd = float((dv * dv).sum()) + 1e-6
    tail_v = f"{work}/tail.mp4"; tail_d = Ls - cut_b
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", tail_v], stdin=subprocess.PIPE)
    n_tail = 0
    for i, fr in enumerate(read_src(src, cut_b)):
        ts = cut_b + i / FPS; o = fr.astype(np.float32)
        mine = base_nc.copy()
        full = np.zeros((H, W, 3), np.float32); full[reg] = mine
        flying = []
        for c in TC:
            f0t, f1t = c["fly"]
            if ts >= f1t:
                cxr, cyr = row_c(c["row"]); paste(full, c["img"], cxr, cyr)
                paste(full, CHECK, CHIP_X + CHK_DX, cyr, back(seg(ts, c["check"], c["check"] + 0.3, lambda u: u)) if ts < c["check"] + 0.3 else 1.0)
            elif ts >= f0t: flying.append(c)
            h_ = c.get("hdr")
            if h_ and h_["t0"] <= ts < h_["t1"]:
                y0 = int(h_["y0"]); o[y0:y0 + h_["img"].shape[0]] = h_["img"]
        mine = full[reg]
        if any(a <= ts <= b for a, b in keep_src) and refs:
            ref = refs[0][1]
            for rfrom, rimg in refs:
                if ts >= rfrom: ref = rimg
            dm = (np.abs(o[reg] - ref).sum(2) > 110).astype(np.uint8)
            dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))          # flying chips only, not codec specks
            dm = cv2.GaussianBlur(cv2.dilate(dm, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.5)[..., None]
            mine = mine * (1 - dm) + o[reg] * dm
        if ts > Ls - 0.75:                                                 # source loop dissolve into its frame 0
            u = float(((o[ex] - fin[ex]) * dv).sum()) / dd; u = min(1.0, max(0.0, u))
            mine = mine * (1 - u) + f0[reg] * u
        o[reg] = mine
        for c in flying:
            f0t, f1t = c["fly"]; u = ease((ts - f0t) / (f1t - f0t)); x0, y0, s0 = c["from"]; cxr, cyr = row_c(c["row"])
            paste(o, c["img"], x0 + (cxr - x0) * u, y0 + (cyr - y0) * u, s0 + (1 - s0) * u, min(1, (ts - f0t) / 0.06))
        enc.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes()); n_tail += 1
    enc.stdin.close(); assert enc.wait() == 0
    tail_a = src_audio(src, cut_b)

    # ---------------------------------------------------------------- join
    off1 = (n_head - nX) / FPS; off2 = off1 + (n_seg - nX) / FPS; total = off2 + n_tail / FPS
    joined = f"{work}/joined.mp4"
    fc = (f"[0:v]fps={FPS},settb=AVTB,format=yuv420p,setsar=1[a];[1:v]fps={FPS},settb=AVTB,format=yuv420p,setsar=1[b];"
          f"[2:v]fps={FPS},settb=AVTB,format=yuv420p,setsar=1[c];[a][b]xfade=transition=fade:duration={XF}:offset={off1:.4f}[ab];"
          f"[ab][c]xfade=transition=fade:duration={XF}:offset={off2:.4f}[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", head_v, "-i", seg_v, "-i", tail_v, "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", joined], check=True)
    # audio: switch at quiet points inside each video cross-fade (no voice cut), 0.3 s equal-power
    aj = float(D.get("audio_join", 0.3))
    a1 = int((off1 + D.get("a1", XF / 2)) * SR); a2 = int((off2 + D.get("a2", XF / 2)) * SR)
    seg_full = np.concatenate([np.zeros((2, int(off1 * SR))), seg_a], 1)                     # segment on the output clock
    m1 = xfade_at(head_a, seg_full, a1, aj)
    tail_full = np.concatenate([np.zeros((2, int(off2 * SR))), tail_a], 1)
    m2 = xfade_at(m1, tail_full, a2, aj)[:, :int(total * SR)]
    aud = loudnorm(m2, f"{work}/final.m4a", (S.get("music") or {}).get("lufs", -16))
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-i", f"{work}/final.m4a", "-map", "0:v", "-map", "1:a", "-c", "copy",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    return dict(voice=J.get("voice"), source=D["source"], head_s=round(head_d, 3), segment_s=round(SEG, 3), tail_s=round(n_tail / FPS, 3),
                cut_a=cut_a, cut_b=cut_b, dropped=drops, mix=aud, segment_lines=[(l["text"], l["t0"], l["t1"]) for l in T.L])


def xfade_at(a, b, at, d):
    """a and b on the same clock (sample 0 = output 0): a until `at`, b after, equal-power cross-fade of d s centred on at"""
    n = int(d * SR); i0 = max(0, at - n // 2); L = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, L - a.shape[1]))); b = np.pad(b, ((0, 0), (0, L - b.shape[1])))
    k = np.clip((np.arange(L) - i0) / max(1, n), 0, 1); fo = np.cos(k * np.pi / 2); fi = np.sin(k * np.pi / 2)
    return a * fo[None] + b * fi[None]
