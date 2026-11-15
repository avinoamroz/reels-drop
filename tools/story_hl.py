#!/usr/bin/env python3
"""tools/story_hl.py <spec.json> [frame numbers...] : story highlight frames, one short MP4 per frame (1080x1920, 30 fps).

Frame types (mix them, about a third each): still (photo, slow Ken Burns), broll (video clips, cover crop),
text (navy card, optional big yellow number, rows list or split bar), tm (testimonial: landscape clip in a panel over a
blurred copy, the speaker's real audio). Music bed: soft_piano from templates/core/music.py (one bed per set, sliced per
frame). Narration (spec frame "vo": line index) comes from voice/<voice_id>_vo.wav + _words.json (tools/tts_chunks.py),
house chain from templates/core/common.py. Rules: no narration on frames with a still photo of Noam's face.
Outputs: <out_dir>/<NN>_<name>.mp4 and <preview> (540p, all frames in order). Nothing here posts anything."""
import json, os, subprocess, sys, math
import numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "templates", "core"))
import music as MU

W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY, SKY, YEL, WHITE = (22, 50, 97), (69, 164, 221), (255, 205, 60), (255, 255, 255)
IF = "/usr/share/fonts/opentype/inter/"; HF = os.path.join(REPO, "templates", "fonts")
FONTS = dict(en_h=IF + "InterDisplay-Bold.otf", en_x=IF + "InterDisplay-ExtraBold.otf", en_b=IF + "Inter-Medium.otf",
             en_s=IF + "Inter-SemiBold.otf", he_h=HF + "/Heebo-800.ttf", he_x=HF + "/Heebo-800.ttf", he_b=HF + "/Heebo-700.ttf", he_s=HF + "/Heebo-700.ttf")
RTL = dict(direction="rtl", language="he")


def font(k, sz): return ImageFont.truetype(FONTS[k], sz)


def is_he(s): return any("֐" <= c <= "׿" for c in s)


def run(*a): subprocess.run(a, check=True)


# ------------------------------------------------------------------ text overlay (RGBA, static) + animated parts
class Overlay:
    def __init__(s, F, S):
        s.F, s.S = F, S
        s.main_he = S.get("lang") == "he"
        s.items = []          # (img RGBA, x, y, t_in)

    def text_block(s, txt, sz, y, kind="h", col=WHITE, maxw=W - 128, he=None, t_in=0.3, align=None, lh=1.12, shadow=True):
        he = is_he(txt) if he is None else he
        f = font(("he_" if he else "en_") + kind, sz)
        kw = RTL if he else {}
        tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        words, lines, cur = txt.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if tmp.textlength(t.replace("*", ""), font=f, **kw) <= maxw or not cur: cur = t
            else: lines.append(cur); cur = w_
        if cur: lines.append(cur)
        lhp = int(sz * lh); hh = lhp * len(lines) + int(sz * 0.35)
        im = Image.new("RGBA", (W, hh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        align = align or ("right" if he else "left")
        yy = 0
        for ln in lines:
            plain = ln.replace("*", ""); tw = d.textlength(plain, font=f, **kw)
            x0 = (W - tw) / 2 if align == "center" else ((W - 64 - tw) if align == "right" else 64)
            # *span* in yellow: draw segment by segment (LTR only; Hebrew lines draw whole in col)
            if "*" in ln and not he:
                x = x0; hl = False
                for seg in ln.split("*"):
                    if seg:
                        if shadow: d.text((x + 2, yy + 3), seg, font=f, fill=(0, 0, 0, 140))
                        d.text((x, yy), seg, font=f, fill=YEL if hl else col)
                        x += d.textlength(seg, font=f)
                    hl = not hl
            else:
                c = YEL if "*" in ln else col
                if shadow: d.text((x0 + 2, yy + 3), plain, font=f, fill=(0, 0, 0, 140), **kw)
                d.text((x0, yy), plain, font=f, fill=c, **kw)
            yy += lhp
        s.items.append((im, 0, y, t_in))
        return y + hh

    def pill(s, txt, y, t_in=0.15, fill=SKY, col=WHITE, sz=38, align="left"):
        he = is_he(txt); f = font("he_b" if he else "en_s", sz); kw = RTL if he else {}
        d0 = ImageDraw.Draw(Image.new("RGBA", (10, 10))); tw = d0.textlength(txt, font=f, **kw)
        pw, ph = int(tw + 56), int(sz * 1.75)
        im = Image.new("RGBA", (W, ph + 4), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        x = 64 if align == "left" else (W - 64 - pw if align == "right" else (W - pw) // 2)
        d.rounded_rectangle((x, 0, x + pw, ph), ph // 2, fill=fill + (235,))
        d.text((x + 28, (ph - sz * 1.2) / 2), txt, font=f, fill=col, **kw)
        s.items.append((im, 0, y, t_in)); return y + ph

    def render(s):
        """-> list of (y0, rgb float BGR, alpha float, cropped to the layer's rows, t_in)"""
        out = []
        for im, x, y, t_in in s.items:
            a = np.asarray(im).astype(np.float32) / 255
            h = min(a.shape[0], H - y)
            if h <= 0 or y < 0: continue
            a = a[:h]; rows = np.where(a[..., 3].max(1) > 0)[0]
            if not len(rows): continue
            r0, r1 = rows[0], rows[-1] + 1
            out.append((y + r0, a[r0:r1, :, 2::-1].copy(), a[r0:r1, :, 3:4].copy(), t_in))
        return out


def comp(frame, layers, t):
    """frame float32 BGR 0..1; each layer fades + rises in over 0.45 s from t_in"""
    for y0, rgb, al, t_in in layers:
        p = min(1.0, max(0.0, (t - t_in) / 0.45))
        if p <= 0: continue
        e = 1 - (1 - p) ** 3; y = y0 + int(26 * (1 - e)); h = min(rgb.shape[0], H - y)
        if h <= 0: continue
        a = al[:h] * e
        frame[y:y + h] = frame[y:y + h] * (1 - a) + rgb[:h] * a
    return frame


# ------------------------------------------------------------------ backgrounds
def grad_dark(top=0.55, bottom=0.92, start=0.38):
    y = np.linspace(0, 1, H)[:, None]
    g = np.maximum(np.clip(1 - y / 0.22, 0, 1) ** 1.4 * top, np.clip((y - start) / (1 - start), 0, 1) ** 0.9 * bottom)
    return g[..., None].astype(np.float32)


def text_bg(t, dur, seed=0):
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    base = np.zeros((H // 4, W // 4, 3), np.float32)
    top = np.array(NAVY[::-1], np.float32) / 255 * 1.05; bot = np.array((10, 22, 46)[::-1], np.float32) / 255
    k = (yy / H)[..., None]; base[:] = top * (1 - k) + bot * k
    cx = W * (0.25 + 0.5 * (0.5 + 0.5 * math.sin(t * 0.35 + seed))); cy = H * (0.3 + 0.12 * math.cos(t * 0.27 + seed))
    glow = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 360.0 ** 2)))[..., None]
    base += glow * (np.array(SKY[::-1], np.float32) / 255) * 0.28
    return cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)


def load_still(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def kb(im, cx, cy, z):
    h, w = im.shape[:2]; s = max(W / w, H / h) * z
    cw, ch = W / s, H / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255


def prep_broll(F, work, dur):
    """concat clips [{src, ss, t, x (0..1 crop centre), speed}] -> work/bg.mp4 at 1080x1920 30 fps, exactly dur seconds"""
    parts = []
    for i, c in enumerate(F["clips"]):
        p = f"{work}/c{i}.mp4"; sp = float(c.get("speed", 1.0)); x = c.get("x", 0.5)
        vf = (f"setpts=PTS/{sp},scale='if(gt(a,{W}/{H}),-2,{W})':'if(gt(a,{W}/{H}),{H},-2)',"
              f"crop={W}:{H}:(iw-{W})*{x}:(ih-{H})*{c.get('y', 0.5)},fps={FPS},format=yuv420p")
        args = ["ffmpeg", "-v", "error", "-y"]
        if c.get("ss") is not None: args += ["-ss", str(c["ss"])]
        args += ["-i", os.path.join(REPO, c["src"]) if not os.path.isabs(c["src"]) else c["src"]]
        if c.get("t"): args += ["-t", str(float(c["t"]))]
        run(*args, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", p)
        parts.append(p)
    lst = f"{work}/l.txt"; open(lst, "w").write("".join(f"file '{p}'\n" for p in parts * 4))
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "16",
        "-preset", "veryfast", f"{work}/bg.mp4")
    return f"{work}/bg.mp4"


class VidReader:
    def __init__(s, path): s.cap = cv2.VideoCapture(path); s.last = None
    def next(s):
        ok, fr = s.cap.read()
        if ok: s.last = fr.astype(np.float32) / 255
        return s.last


# ------------------------------------------------------------------ audio
def voice_lines(vid):
    """voice/<vid>_vo.wav + words -> list of mono 48 kHz arrays, one per chunk line (house chain, tempo 1.05)"""
    S = json.load(open(os.path.join(REPO, f"voice/chunks/{vid}.json")))
    raw = os.path.join(REPO, f"voice/{vid}_vo.wav"); Wd = json.load(open(os.path.join(REPO, f"voice/{vid}_vo_words.json")))
    y, k = house_chain(raw, 1.05)
    out, i = [], 0
    for ln in S["lines"]:
        n = len(ln.split()); ws = Wd[i:i + n]; i += n
        a = max(0, ws[0]["start"] * k - 0.06); b = ws[-1]["end"] * k + 0.12
        seg = y[int(a * SR):int(b * SR)].astype(np.float32).copy(); f = int(0.012 * SR)
        seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        out.append(seg / (np.abs(seg).max() + 1e-9) * 0.89)
    return out


def house_chain(raw, tempo):
    """same chain as templates/core/common.py house_chain (Noam's reel voice): pitch 0.9715, bass +2 dB at 135 Hz, atempo"""
    af = f"rubberband=pitch=0.9715:formant=preserved,bass=g=2:f=135,atempo={tempo}"
    y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", raw, "-af", af, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                                     check=True, capture_output=True).stdout, np.float32).copy()
    dry = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw], capture_output=True, text=True).stdout)
    return y, (len(y) / SR) / dry


def lufs(x):
    """x (2, n) float -> integrated loudness (ffmpeg ebur128), as templates/core/common.py lufs_of"""
    o = subprocess.run(["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af", "ebur128", "-f", "null", "-"],
                       input=x.T.astype(np.float32).tobytes(), capture_output=True).stderr.decode()
    return float(o[o.rfind("I:"):].split()[1])


def mix_audio(F, bed, t0, dur, work, VO):
    n = int(dur * SR)
    m = bed[:, int(t0 * SR):int(t0 * SR) + n]
    if m.shape[1] < n: m = np.pad(m, ((0, 0), (0, n - m.shape[1])))
    fade = np.minimum(1, np.arange(n) / (0.25 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.35 * SR))
    m = m * fade
    fg = np.zeros(n, np.float32); has_fg = False
    if F.get("vo") is not None:
        v = VO[F["vo"]]; st = int(float(F.get("vo_at", 0.45)) * SR); e = min(n, st + len(v)); fg[st:e] += v[:e - st]; has_fg = True
    if (F.get("type") == "tm" and not F.get("mute")) or F.get("clip_audio"):
        a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", str(F["ss"]), "-t", str(F["t"]), "-i", F["src_abs"], "-af",
                                          "highpass=f=90,loudnorm=I=-18:TP=-2", "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                                         check=True, capture_output=True).stdout, np.float32)
        st = int((0.35 if F.get("type") == "tm" else 0.0) * SR); e = min(n, st + len(a)); fg[st:e] += a[:e - st]; has_fg = True
    if has_fg:
        fg = fg / (np.abs(fg).max() + 1e-9) * 0.9
        mus_db = -17.0                                   # bed about 17 dB under speech (music 3 to 6 % feel)
        mm = m / (np.sqrt((m ** 2).mean()) + 1e-9) * (np.sqrt((fg[fg != 0] ** 2).mean()) * 10 ** (mus_db / 20))
        x = np.stack([fg, fg]) + mm; target = -16.0
    else:
        x = m; target = -20.0                            # quiet bed on music-only frames
    L = lufs(x)
    if np.isfinite(L): x = x * 10 ** ((target - L) / 20)
    x = np.clip(x, -0.98, 0.98)
    sf.write(f"{work}/a.wav", x.T.astype(np.float32), SR)
    return f"{work}/a.wav"


# ------------------------------------------------------------------ frame
def build_frame(F, S, bed, t0, VO, out):
    work = f"/tmp/story_hl_{S['id']}_{F['n']}"; os.makedirs(work, exist_ok=True)
    dur = float(F.get("dur", 6.0)); typ = F["type"]; heM = S.get("lang") == "he"
    O = Overlay(F, S)
    # --- text layout
    tag = F.get("tag", S.get("tag"))
    if tag: O.pill(tag, 200, align="right" if is_he(tag) else "left")
    if typ == "tm":
        y = 330
        if F.get("who"): y = O.text_block(F["who"], 60, y, kind="h", t_in=0.2)
        yq = 1180
        if F.get("head"): yq = O.text_block(F["head"], 72, yq, kind="h", t_in=0.9)
        if F.get("quote"): yq = O.text_block("“" + F["quote"] + "”", 46, yq + 10, kind="b", col=(225, 233, 245), t_in=1.2)
        if F.get("second"): yq = O.text_block(F["second"], 48, yq + 10, kind="b", col=(190, 214, 240), t_in=1.4, shadow=False)
    else:
        top = F.get("y")
        blocks = []
        if F.get("big"): blocks.append(("big", F["big"]))
        if F.get("head"): blocks.append(("head", F["head"]))
        if F.get("rows"): blocks.append(("rows", F["rows"]))
        if F.get("bar"): blocks.append(("bar", F["bar"]))
        if F.get("sub"): blocks.append(("sub", F["sub"]))
        if F.get("second"): blocks.append(("second", F["second"]))
        # estimate height to place the stack: lower third on photo/video, centred on text cards
        est = 0
        for k, v in blocks:
            est += {"big": 260, "head": 3 * 100, "rows": len(v) * 128, "bar": 230, "sub": 2 * 82, "second": 2 * 62}[k]
        y = top if top is not None else (int((H - est) / 2) - 40 if typ == "text" else int(1560 - est))
        y = max(300, y); t_in = 0.25; al = F.get("align"); k0 = len(O.items); y_first = y
        for k, v in blocks:
            if k == "big":
                y = O.text_block("*" + v + "*" if not is_he(v) else v, 230, y, kind="x", t_in=t_in, align=al, lh=1.0, col=YEL) + 10
            elif k == "head":
                y = O.text_block(v, F.get("head_size", 84), y, kind="h", t_in=t_in, align=al) + 14
            elif k == "rows":
                for j, r in enumerate(v):
                    he = is_he(r); f = font("he_h" if he else "en_h", 64); kw = RTL if he else {}
                    im = Image.new("RGBA", (W, 112), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
                    d.rounded_rectangle((64, 4, W - 64, 104), 30, fill=(255, 255, 255, 34), outline=SKY + (255,), width=3)
                    num = str(j + 1); fn = font("en_x", 60)
                    if he:
                        d.text((W - 64 - 30 - d.textlength(num, font=fn), 16), num, font=fn, fill=YEL)
                        tw = d.textlength(r, font=f, **kw); d.text((W - 64 - 100 - tw, 12), r, font=f, fill=WHITE, **kw)
                    else:
                        d.text((94, 16), num, font=fn, fill=YEL); d.text((164, 12), r, font=f, fill=WHITE)
                    O.items.append((im, 0, y, t_in + 0.35 * j)); y += 128
                t_in += 0.35 * len(v)
            elif k == "bar":
                O.items.append(("BAR", v, y, t_in)); y += 230
            elif k == "sub":
                y = O.text_block(v, 72 if F.get("sub_big") else 64, y, kind="b" if not is_he(v) else "b", col=(232, 238, 248), t_in=t_in + 0.35, align=al) + 8
            elif k == "second":
                y = O.text_block(v, 48, y + 6, kind="b", col=(190, 214, 240), t_in=t_in + 0.6, align=al, shadow=False)
            t_in += 0.25
        # bottom-anchor the stack on photo / video frames (or centre it on text cards) from its real height
        if top is None and typ != "tm":
            hgt = y - y_first; ny = int(1610 - hgt) if typ != "text" else int((H - hgt) / 2) - 30
            ny = max(300, ny); dy = ny - y_first
            O.items = O.items[:k0] + [(a, b, c + dy, d) for a, b, c, d in O.items[k0:]]
            y_first += dy
        O.text_top = y_first; O.text_end = y
    if F.get("label"): O.text_block(F["label"], 36, 1700, kind="b", col=(200, 214, 234), t_in=0.0, align="center", shadow=True)
    bars = [it for it in O.items if it[0] == "BAR"]; O.items = [it for it in O.items if it[0] != "BAR"]
    layers = O.render()
    # --- background
    if typ == "broll": vr = VidReader(prep_broll(F, work, dur))
    if typ == "tm":
        F["src_abs"] = F["src"] if os.path.isabs(F["src"]) else os.path.join(REPO, F["src"])
        run("ffmpeg", "-v", "error", "-y", "-ss", str(F["ss"]), "-t", str(F["t"]), "-i", F["src_abs"], "-vf",
            (f"setpts=PTS/{F['speed']}," if F.get("speed") else "") + f"scale={W}:-2,fps={FPS},format=yuv420p", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", f"{work}/tm.mp4")
        vr = VidReader(f"{work}/tm.mp4"); lead = 0.35
    if typ == "still":
        im = load_still(F["src"] if os.path.isabs(F["src"]) else os.path.join(REPO, F["src"]))
        fx, fy = F.get("focus", [0.5, 0.42]); z0, z1 = F.get("kb", [1.0, 1.08])
        if F.get("framed"):        # low-res or square photo: sharp, full width, over a blurred darker copy of itself
            h0, w0 = im.shape[:2]; fg = cv2.resize(im, (W, int(h0 * W / w0)), interpolation=cv2.INTER_CUBIC)
            bgf = cv2.GaussianBlur(cv2.resize(im, (int(w0 * H / h0), H)), (0, 0), 40)
            xo = (bgf.shape[1] - W) // 2; bgf = (bgf[:, xo:xo + W].astype(np.float32) * 0.5).astype(np.uint8)
            yo = int(F.get("frame_y", 300)); bgf[yo:yo + fg.shape[0]] = fg[:H - yo]; im = bgf; fx, fy = 0.5, 0.5
    g = None
    if typ in ("still", "broll"):
        tt_ = getattr(O, "text_top", 1100); yy = np.arange(H, dtype=np.float32)[:, None]
        if F.get("y") is not None:       # text at the top (CTA frames keep the lower half free for the link sticker)
            te = getattr(O, "text_end", 900); sc = np.clip(((te + 220) - yy) / 260, 0, 1) ** 1.2 * F.get("scrim", 0.85)
        else:
            sc = np.clip((yy - (tt_ - 260)) / 260, 0, 1) ** 1.2 * F.get("scrim", 0.82)    # scrim behind the text stack
        tp = np.clip(1 - yy / 330, 0, 1) ** 1.4 * 0.55                                    # top fade for the tag
        g = np.maximum(sc, tp)[..., None].astype(np.float32)
    # --- encode
    if F.get("clip_audio"):
        c0 = F["clips"][0]; F["src_abs"] = c0["src"] if os.path.isabs(c0["src"]) else os.path.join(REPO, c0["src"])
        F["ss"] = c0.get("ss", 0); F["t"] = c0.get("t", dur)
    a = mix_audio(F, bed, t0, dur, work, VO)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-i", a, "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                          "-ar", str(SR), "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    N = int(round(dur * FPS)); last_clip = None
    for i in range(N):
        t = i / FPS
        if typ == "text": fr = text_bg(t, dur, seed=hash(F["n"]) % 7)
        elif typ == "still":
            u = t / dur; e = u * u * (3 - 2 * u); z = z0 + (z1 - z0) * e
            fr = kb(im, fx, fy, z); fr = fr * (1 - g) + np.array([46, 22, 9], np.float32) / 255 * g
        elif typ == "broll":
            fr = vr.next().copy(); fr = fr * (1 - g) + np.array([46, 22, 9], np.float32) / 255 * g
        else:  # tm
            if t >= lead:
                c = vr.next()
                if c is not None: last_clip = c
            if last_clip is None:
                cap = cv2.VideoCapture(f"{work}/tm.mp4"); ok, c0 = cap.read(); last_clip = c0.astype(np.float32) / 255; cap.release()
            ph = last_clip.shape[0]
            bg = cv2.resize(last_clip, (W * ph // last_clip.shape[0] if False else int(last_clip.shape[1] * H / ph), H))
            x0 = (bg.shape[1] - W) // 2; bg = bg[:, x0:x0 + W]
            sm = cv2.resize(bg, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            bg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3.5), (W, H), interpolation=cv2.INTER_LINEAR) * 0.42
            fr = bg.copy(); py = 520; fr[py:py + ph, :] = last_clip[:min(ph, H - py)]
            cv2.rectangle(fr, (0, py - 3), (W - 1, py + ph + 2), (SKY[2] / 255, SKY[1] / 255, SKY[0] / 255), 3)
        fr = comp(fr, layers, t)
        for _, v, y, t_in in bars:
            fr = draw_bar(fr, v, y, t, t_in)
        p.stdin.write((np.clip(fr, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    if p.returncode: raise SystemExit(f"ffmpeg failed for {out}")
    import shutil; shutil.rmtree(work, ignore_errors=True)  # disk is tight: drop the per-frame work dir
    return out


def draw_bar(fr, v, y, t, t_in):
    """split bar animating from v['from'] to v['to'] (share of the left part, %), labels v['left'], v['right']"""
    p = min(1.0, max(0.0, (t - t_in) / 0.4))
    if p <= 0: return fr
    q = min(1.0, max(0.0, (t - t_in - 0.8) / 1.6)); q = q * q * (3 - 2 * q)
    share = v["from"] + (v["to"] - v["from"]) * q
    x0, x1 = 64, W - 64; bw = x1 - x0; h = 120
    im = Image.new("RGBA", (W, 230), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    xs = x0 + int(bw * share / 100)
    # Hebrew reads right to left: English (left label) part sits on the right when v.rtl
    if v.get("rtl"):
        d.rounded_rectangle((x0, 0, x1, h), 26, fill=(255, 255, 255, 60))
        d.rounded_rectangle((x1 - int(bw * share / 100), 0, x1, h), 26, fill=SKY + (255,))
    else:
        d.rounded_rectangle((x0, 0, x1, h), 26, fill=(255, 255, 255, 60)); d.rounded_rectangle((x0, 0, xs, h), 26, fill=SKY + (255,))
    fl = font("en_x", 64); a_ = f"{round(share)}%"; b_ = f"{100 - round(share)}%"
    if v.get("rtl"):
        d.text((x1 - 30 - d.textlength(a_, font=fl), 22), a_, font=fl, fill=WHITE); d.text((x0 + 30, 22), b_, font=fl, fill=WHITE)
    else:
        d.text((x0 + 30, 22), a_, font=fl, fill=WHITE); d.text((x1 - 30 - d.textlength(b_, font=fl), 22), b_, font=fl, fill=WHITE)
    fs = font("he_b", 46) if is_he(v["left"]) else font("en_s", 44)
    la, lb = v["left"], v["right"]; kw = RTL if is_he(la) else {}
    if v.get("rtl"):
        d.text((x1 - d.textlength(la, font=fs, **kw), h + 22), la, font=fs, fill=SKY, **kw); d.text((x0, h + 22), lb, font=fs, fill=(220, 228, 240), **kw)
    else:
        d.text((x0, h + 22), la, font=fs, fill=SKY, **kw); d.text((x1 - d.textlength(lb, font=fs, **kw), h + 22), lb, font=fs, fill=(220, 228, 240), **kw)
    a = np.asarray(im).astype(np.float32) / 255
    al = a[..., 3:4] * p; rgb = a[..., :3][..., ::-1]
    fr[y:y + 230] = fr[y:y + 230] * (1 - al) + rgb * al
    return fr


def main():
    S = json.load(open(sys.argv[1])); only = set(sys.argv[2:])
    out_dir = os.path.join(REPO, S["out_dir"]); os.makedirs(out_dir, exist_ok=True)
    total = sum(float(F.get("dur", 6)) for F in S["frames"]) + 4
    bed = MU.soft_piano(total, seed=S.get("seed", 7), lift_at=S.get("lift"))
    if bed.ndim == 1: bed = np.stack([bed, bed])
    if MU.SR != SR:
        import scipy.signal as ss
        bed = ss.resample_poly(bed, SR, MU.SR, axis=1)
    VO = voice_lines(S["voice_id"]) if any(F.get("vo") is not None for F in S["frames"]) else []
    t0, outs = 0.5, []
    for F in S["frames"]:
        o = os.path.join(out_dir, f"{F['n']}_{F['name']}.mp4"); outs.append(o)
        if not only or F["n"] in only:
            print("frame", F["n"], F["type"], flush=True); build_frame(F, S, bed, t0, VO, o)
        t0 += float(F.get("dur", 6))
    if S.get("preview") and not only:
        os.makedirs(os.path.dirname(os.path.join(REPO, S["preview"])), exist_ok=True)
        lst = f"/tmp/story_hl_{S['id']}_list.txt"; open(lst, "w").write("".join(f"file '{o}'\n" for o in outs))
        run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "scale=540:960", "-c:v", "libx264", "-crf", "24",
            "-preset", "veryfast", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", os.path.join(REPO, S["preview"]))
        print("preview", S["preview"])


if __name__ == "__main__":
    main()
