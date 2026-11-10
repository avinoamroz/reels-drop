"""hook.py: the reel's hook line on screen from frame 0 of an opener (click study, 3 Nov: what is SEEN at frame 0
decides the skip; hook text visible by about 0.3 s). Lines split with "|", *word* in gold.
The hook sits at the top while the opener plays, then (when a reel follows) moves and grows into the position of the
reel's own frame-0 text, so the cross-dissolve into frame 0 lands text on text."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FONT = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
GOLD_RGB = (255, 205, 80)


def make(text, size=124, maxw=1000):
    fnt = ImageFont.truetype(FONT, size); lines = [l.strip() for l in text.split("|")]
    lh = int(size * 1.1); pad = 40
    widths = [sum(fnt.getlength(w.strip("*") + " ") for w in ln.split()) - fnt.getlength(" ") for ln in lines]
    s = min(1.0, maxw / max(widths))
    if s < 1: return make(text, int(size * s), maxw)
    im = Image.new("RGBA", (maxw + 2 * pad, lh * len(lines) + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        x = (maxw + 2 * pad - widths[i]) / 2; y = pad + i * lh + lh / 2
        for w in ln.split():
            gold = w.startswith("*") and w.rstrip(".,!?").endswith("*"); wt = w.replace("*", "")
            d.text((x, y), wt, font=fnt, fill=GOLD_RGB + (255,) if gold else (255, 255, 255, 255), anchor="lm")
            x += fnt.getlength(wt + " ")
    return np.asarray(im).astype(np.float32) / 255


def paste(fr, rgba, cx, cy, scale, alpha):
    if alpha <= 0.01: return fr
    im = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = im.shape[:2]; x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya: return fr
    sub = im[ya - y0:yb - y0, xa - x0:xb - x0]; a = sub[..., 3:4] * alpha
    sh = cv2.GaussianBlur(sub[..., 3], (0, 0), 12)[..., None] * alpha * 0.6
    reg = fr[ya:yb, xa:xb] * (1 - sh); col = sub[..., 2::-1] * 255
    fr[ya:yb, xa:xb] = reg * (1 - a) + col * a
    return fr


def _e(u): u = min(1.0, max(0.0, u)); return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def overlay(fr, img, t, T, z0, to_reel, y_top=250, s_top=0.68, y_reel=930, s_reel=1.0, alpha=1.0):
    """frame 0: already on screen (scale pops 0.94 -> 1 over 0.2 s); with a reel it travels to the frame-0 text during
    the zoom; the caller's dissolve into frame 0 covers the last 0.33 s"""
    pop = 0.94 + 0.06 * (1 - (1 - min(1, t / 0.2)) ** 3)
    if to_reel:
        e = _e((t - z0) / max(1e-3, (T - 0.2) - z0)) if t > z0 else 0.0
        y = y_top + (y_reel - y_top) * e; s = (s_top + (s_reel - s_top) * e) * pop
    else: y, s = y_top, s_top * pop
    return paste(fr, img, 540, y, s, alpha)
