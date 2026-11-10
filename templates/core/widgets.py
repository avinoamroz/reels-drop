"""templates/core/widgets.py: title, panel, counter, bars, checks, pins, tags (from /home/claude/fix_1109/rmk/eng.py)."""
import math
from base import *

# ------------------------------------------------------------------ widgets (cfg draw() calls these)
def title(cv, txt, y, a=1.0, sz=80, col=WHITE, maxw=980):
    """*word* in yellow; wraps to 2 lines"""
    if a <= 0.01: return
    words = txt.split(); lines = [[]]
    for w in words:
        test = " ".join(x.replace("*", "") for x in lines[-1] + [w])
        if lines[-1] and cv.tw(test, sz) > maxw: lines.append([w])
        else: lines[-1].append(w)
    if len(lines) > 2: sz = int(sz * 0.86); return title(cv, txt, y, a, sz, col, maxw) if sz > 60 else None
    lh = sz * 1.18; y0 = y - lh * (len(lines) - 1) / 2
    for li, ln in enumerate(lines):
        parts = []; hl = False
        for w in ln:
            on = w.startswith("*"); off = w.rstrip(".,!?:;").endswith("*")
            ww = w.replace("*", "")
            parts.append((ww + " ", YEL if (on or hl) else col)); hl = (hl or on) and not off
        parts[-1] = (parts[-1][0].rstrip(), parts[-1][1])
        cv.rich(parts, 540, y0 + li * lh, sz, a, maxw)
def panel(cv, x, y, w, h, a=1.0, stroke=SKY):
    cv.rrect(x, y, w, h, 34, "#0B1A3C", 0.84 * a, stroke=stroke, lw=2.5, sa=0.55 * a)
def fmtn(v, dec=0, prefix="", suffix=""):
    s = f"{v:,.{dec}f}"; return f"{prefix}{s}{suffix}"
def counter(cv, v, cx, cy, sz=180, a=1.0, dec=0, prefix="", suffix="", col=YEL):
    cv.text(fmtn(v, dec, prefix, suffix), cx, cy, sz, col, a, maxw=1000)
def roll(t, t0, v0, v1, d=1.0):
    """count from v0 to v1 starting at t0 (log-eased for big jumps)"""
    u = eout((t - t0) / d)
    if u >= 0.999: return v1
    if v1 > 10 * max(v0, 1): return math.exp(math.log(max(v0, 1)) + (math.log(v1) - math.log(max(v0, 1))) * u) if u > 0 else v0
    return v0 + (v1 - v0) * u
def src(cv, txt, a=1.0, y=1488):
    if a <= 0.01: return
    if cv.tw(txt, 38, False) <= 960: cv.text(txt, 540, y, 38, "#C9D6EA", a, bold=False, maxw=960, shadow=True, minsz=36); return
    w = txt.split(); best = None
    for k in range(1, len(w)):
        l1, l2 = " ".join(w[:k]), " ".join(w[k:]); m = max(cv.tw(l1, 38, False), cv.tw(l2, 38, False))
        if best is None or m < best[0]: best = (m, l1, l2)
    cv.text(best[1], 540, y - 24, 38, "#C9D6EA", a, bold=False, maxw=960, shadow=True, minsz=36)
    cv.text(best[2], 540, y + 24, 38, "#C9D6EA", a, bold=False, maxw=960, shadow=True, minsz=36)
def flag(cv, code, cx, cy, sz=64, a=1.0):
    ch = "".join(chr(0x1F1E6 + ord(c) - 65) for c in code.upper())
    cv.image(emoji(ch), cx, cy, sz / 109, a)
def emo(cv, ch, cx, cy, sz=64, a=1.0): cv.image(emoji(ch), cx, cy, sz / 109, a)
def mark(cv, kind, cx, cy, r=34, a=1.0, sc=1.0):
    """Y green tick, N red cross, P amber half"""
    if a <= 0.01: return
    r *= sc; colr = {"Y": GREEN, "N": RED, "P": "#F2A93B"}[kind]
    cv.circ(cx, cy, r, colr, a)
    if kind == "Y": cv.line([(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.12, cy + r * 0.36), (cx + r * 0.5, cy - r * 0.34)], "#FFFFFF", r * 0.22, a)
    elif kind == "N":
        cv.line([(cx - r * 0.38, cy - r * 0.38), (cx + r * 0.38, cy + r * 0.38)], "#FFFFFF", r * 0.22, a)
        cv.line([(cx - r * 0.38, cy + r * 0.38), (cx + r * 0.38, cy - r * 0.38)], "#FFFFFF", r * 0.22, a)
    else: cv.line([(cx - r * 0.42, cy), (cx + r * 0.42, cy)], "#FFFFFF", r * 0.24, a)
def pop_s(t, t0): return back(seg(t, t0, t0 + 0.35, lambda u: u)) if t >= t0 else 0.0
def bars(cv, rows, t, a, y0, rowh=118, vmax=None, x0=70, w=940, dec=0, suffix="", lab_w=300, grow=0.8):
    """rows: dict(label, flag|emo, value, at, color, txt); bar grows from 0 at `at`"""
    vmax = vmax or max(r["value"] for r in rows)
    for k, r in enumerate(rows):
        ra = a * seg(t, r["at"] - 0.1, r["at"] + 0.2)
        if ra <= 0.01: continue
        y = y0 + k * rowh; v = r["value"] * eout((t - r["at"]) / grow)
        ix = x0 + 40
        if r.get("flag"): flag(cv, r["flag"], ix, y, 60, ra)
        elif r.get("emo"): emo(cv, r["emo"], ix, y, 60, ra)
        cv.text(r["label"], ix + 46, y, 58, WHITE, ra, align="l", maxw=lab_w - 80)
        bx = x0 + lab_w; bw = (w - lab_w - 170) * max(0.004, v / vmax)
        cv.rrect(bx, y - 28, max(10, bw), 56, 14, r.get("color", SKY), ra)
        cv.text(r.get("txt") or fmtn(v, r.get("dec", dec), "", r.get("suffix", suffix)), bx + max(10, bw) + 16, y, 60,
                r.get("tcol", YEL), ra, align="l", maxw=300)
def checks(cv, cols, rows, t, a, y0, rowh=128, x_lab=80, col_x=(560, 720, 880), head_sz=40):
    """cols: short headers (with emoji); rows: dict(label, flag, cells=[(kind, at)], at)"""
    for j, c in enumerate(cols):
        if isinstance(c, tuple): emo(cv, c[0], col_x[j], y0 - 120, 56, a); cv.text(c[1], col_x[j], y0 - 62, head_sz, "#C9D6EA", a, bold=True, maxw=170, minsz=36)
        else: cv.text(c, col_x[j], y0 - 70, head_sz, "#C9D6EA", a, maxw=170, minsz=36)
    for k, r in enumerate(rows):
        ra = a * seg(t, r["at"] - 0.1, r["at"] + 0.2)
        if ra <= 0.01: continue
        y = y0 + k * rowh
        if r.get("hl"): cv.rrect(50, y - rowh / 2 + 8, 980, rowh - 16, 22, "#FFD23F", 0.13 * ra)
        if r.get("flag"): flag(cv, r["flag"], x_lab + 30, y, 58, ra)
        cv.text(r["label"], x_lab + 74, y, 60, WHITE, ra, align="l", maxw=col_x[0] - x_lab - 140)
        for j, (kind, at) in enumerate(r["cells"]):
            if kind and t >= at: mark(cv, kind, col_x[j], y, 36, ra, pop_s(t, at))
def pin(cv, x, y, col=YEL, a=1.0, t=0.0, r=11):
    if a <= 0.01: return
    pr = (t * 0.9) % 1.0
    cv.circ(x, y, r + 26 * pr, None, 1, stroke=col, lw=3); cv.c.new_path()
    cv.circ(x, y, r, col, a, stroke="#FFFFFF", lw=3)
def tag(cv, txt, x, y, sz=46, col=WHITE, a=1.0, bg="#0B1730"):
    if a <= 0.01: return
    w = cv.tw(txt, sz) + 34; x = min(max(x, w / 2 + 12), W - 12 - w / 2); cv.rrect(x - w / 2, y - sz * 0.75, w, sz * 1.5, 16, bg, 0.85 * a)
    cv.text(txt, x, y, sz, col, a, shadow=False, maxw=1000)

