"""trend_text: a SILENT text clip for a trending Instagram sound (the sound is added in the Instagram app, never baked in).
Typewriter text (white, thin black outline, centred) over a slow push-in background. No voice, no music: the mp4 carries a
silent audio track only. Loops: the last 0.4 s dissolves into frame 0.

data keys:
  dur      seconds (default 9.0)
  text     the trend text; "|" forces a line break; *word* yellow
  type     [t0, t1] typing window in seconds (default [0.0, 4.2]); start_chars (default 13) are on screen at frame 0
  size     text size (default 84)
  y        text centre y (default 820)
  bg       "map" (whole land of Israel lighting up, slow push-in), "map_counter" (map + yellow counter rolling to `count`),
           "express" (Express navy #163261 with sky #45A4DD rings)
  count    {to, label}  for bg map_counter
  he       small Hebrew line at the bottom (express)
  mark     small top-left mark text (default "Israel Facts" on map backgrounds, "English Express" on express)
"""
import os, sys, math, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

NAVY = "#163261"


def outline_text(cv, parts, cx, cy, sz, a, lw=5):
    """parts [(txt, col)] on one line, centred at cx; thin black outline under white/yellow fill"""
    c = cv.c; cv.face(True); c.set_font_size(sz)
    tot = sum(c.text_extents(p).x_advance for p, _ in parts); x = cx - tot / 2; y = cy + sz * 0.35
    for p, col in parts:
        c.move_to(x, y); c.text_path(p); cv.col("#000000", a); c.set_line_width(lw); c.set_line_join(cairo.LINE_JOIN_ROUND)
        c.stroke_preserve(); cv.col(col, a); c.fill(); c.new_path()
        x += c.text_extents(p).x_advance


def layout(cv, text, sz, maxw=940):
    """-> list of lines, each a list of (word, yellow?)"""
    cv.face(True); out = []
    for seg_ in text.split("|"):
        line = []
        for w in seg_.split():
            hl = w.startswith("*") or w.rstrip(".,!?:;").endswith("*"); ww = w.replace("*", "")
            test = " ".join(x for x, _ in line + [(ww, hl)])
            if line and cv.tw(test, sz) > maxw: out.append(line); line = []
            line.append((ww, hl))
        if line: out.append(line)
    return out


def run(S):
    D = S["data"]; work = workdir(S); dur = float(D.get("dur", 9.0))
    text = D["text"]; sz = int(D.get("size", 84)); ty = float(D.get("y", 820)); t0, t1 = D.get("type", [0.0, 4.2])
    kind = D.get("bg", "map"); mark = D.get("mark", "English Express" if kind == "express" else "Israel Facts")
    nchar = len(text.replace("*", "").replace("|", " "))
    MR = MapR() if kind != "express" else None; cache = {}
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.6 * 0.45, np.clip((y_ - 1180) / 740, 0, 1) ** 1.2 * 0.62))[..., None]

    def mapbg(t):
        u = t / dur; wd = 4.8 - 0.9 * ease(u); lat = 31.4; cosf = math.cos(math.radians(lat))
        key = round(wd, 3)
        if key not in cache:
            if len(cache) > 6: cache.clear()
            img, _ = MR.render(MR.view(35.0, lat, W / (wd * cosf), cosf, 1180.0), H)
            cache[key] = img.astype(np.float32)
        img = cache[key] * VIG
        light = 0.45 + 0.55 * ease(min(1.0, t / 3.0))                      # the land lights up
        return img * light

    def expressbg(t):
        cv = Canvas(BG.copy() * 0.0 + np.array([0x61, 0x32, 0x16], np.float32))   # navy #163261 (BGR)
        for k in range(5):
            r = 160 + k * 150 + 40 * math.sin(t * 0.8 + k)
            cv.c.arc(540, 1240, r, 0, 2 * math.pi); cv.col(SKY, 0.10 + 0.04 * (4 - k)); cv.c.set_line_width(6); cv.c.stroke()
        return cv.out().astype(np.float32)

    def frame(t):
        base = expressbg(t) if kind == "express" else mapbg(t)
        tz = 1.0 + 0.06 * (t / dur)                                          # slow push-in on the whole background
        M = cv2.getRotationMatrix2D((W / 2, H * 0.6), 0, tz); base = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        cv = Canvas(base)
        if kind == "map_counter":
            C = D.get("count") or {}; u = eout(seg(t, 0.6, 5.0, lambda x: x)); v = C.get("to", 0) * u
            WG.panel(cv, 140, 1250, 800, 300, 1.0)
            cv.text(C.get("label", ""), 540, 1315, 48, "#C9D6EA", 1.0, maxw=720)
            cv.text(f"{v:,.0f}", 540, 1430, 150, YEL, 1.0, maxw=720)
        # mark
        w_ = cv.tw(mark, 38) + 56
        cv.rrect(70, 196, w_, 64, 32, "#0B1730", 0.8, stroke=SKY, lw=2, sa=0.7); cv.circ(100, 228, 9, YEL, 1.0)
        cv.text(mark, 120, 228, 38, WHITE, 1.0, maxw=w_, align="l", shadow=False)
        # typewriter
        st = int(D.get("start_chars", 13))                                  # first words already on screen at frame 0
        shown = int(round(st + (nchar - st) * seg(t, t0, t1, lambda x: x))) if t < t1 else nchar
        lines = layout(cv, text, sz); lh = sz * 1.22; y0 = ty - lh * (len(lines) - 1) / 2; k = 0
        for li, ln in enumerate(lines):
            parts = []
            for wi, (w, hl) in enumerate(ln):
                piece = w + (" " if wi < len(ln) - 1 else ""); take = max(0, min(len(piece), shown - k)); k += len(piece) + (1 if wi == len(ln) - 1 else 0)
                if take: parts.append((piece[:take], YEL if hl else WHITE))
            # keep the line centred on its FULL width so typing does not jitter
            full = " ".join(w for w, _ in ln); cv.face(True); cv.c.set_font_size(sz)
            fw = cv.c.text_extents(full).x_advance
            if parts:
                tw_ = sum(cv.c.text_extents(p).x_advance for p, _ in parts)
                outline_text(cv, parts, 540 - fw / 2 + tw_ / 2, y0 + li * lh, sz, 1.0)
        if D.get("he"): hebrew(cv, D["he"], 540, D.get("he_y", 1500), D.get("he_size", 56), (255, 255, 255), seg(t, t1 - 0.5, t1))
        return np.clip(cv.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    sil = f"{work}/silence.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{dur:.3f}",
                    "-c:a", "aac", "-b:a", "96k", sil], check=True)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, dur, sil, final, loop=True)
    return dict(voice="none (silent: trending sound is added in the Instagram app)", body_s=dur, opener=None,
                mix=dict(lufs=None, true_peak=None))
