"""column_recut: A/B re-cut of a published bar race with VERTICAL COLUMNS instead of horizontal bars. Everything else in
the published reel stays as it was (hook, frame 0, timing, voice, music, length, counters, sources, subtitles, loop,
ending): the source mp4 is decoded frame by frame, only the bar panel rectangle is wiped (filled with the frame's own
background gradient, sampled from a clean strip) and redrawn as columns rising from a baseline, leader on the left,
smooth position swaps on overtakes, value on top, flag/label under each column. Audio is copied untouched.

data keys (times in seconds of the source):
  source     repo path of the published reel mp4 (scaled to 1080x1920 if smaller)
  rect       [x, y, w, h]  bar panel to wipe
  bg_x       [x0, x1]  clean background strip used to rebuild the gradient inside rect (default [2, 14])
  area       {x0, x1, base, top, lab_h}  column area: columns rise from `base` up to at most `top` (leader never above)
  windows    [[t0, t1, fade_in, fade_out], ...]  when the columns show (match the original panel's fades)
  years      [[t, year], ...]  piecewise linear timeline (from the source's year label)
  series     [{id, label, flag (ISO2) | dot, col, pts: [[year, v], ...], last, first, hl, show: [t0, t1, fade]}]
             value = linear interpolation; before `first` (or the first point): "no data"; after `last`: carried and striped
  fmt        "dec1" (1 decimal, * when striped) | "pop" (174k, 1.96M)
  rings      [{t0, t1, ids}]  white outline around those columns
  hide_zero  true: a column with value 0 leaves (pop race)
  gaps       "hold": between non consecutive reported years hold the last value, striped (UNODC style)
  sort       "asc": lowest value is the leader (left), e.g. murder rates; default highest first
  label      "flag_name" (flag + short name under the column) | "rotated" (name rotated 90 degrees, colour dot)
  vmax_fix   fixed scale maximum (else the smoothed leader)
  overlays   [{at, until, text, y, size, pill, cue}]  extra on-screen pills (map_counter style), e.g. the follow line overlay
  loop_blend seconds at the end over which the columns blend into their frame 0 state (the source's own loop dissolve)
"""
import os, sys, math, subprocess, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import map_counter as MC


def fmt_val(v, fmt, striped):
    if fmt == "pop":
        if v >= 1e6: return f"{v / 1e6:.2f}M"
        if v >= 1e4: return f"{v / 1e3:.0f}k"
        if v >= 1e3: return f"{v / 1e3:.1f}k"
        return f"{v:.0f}"
    return f"{v:.1f}" + ("*" if striped else "")


def run(S):
    D = S["data"]
    if D.get("file"): D = dict(json.load(open(os.path.join(REPO, D["file"]))), **{k: v for k, v in D.items() if k != "file"})
    work = workdir(S); src = os.path.join(REPO, D["source"])
    dur = dur_of(src); n_fr = int(round(dur * FPS))
    X, Y, RW, RH = D["rect"]; bx0, bx1 = D.get("bg_x", [2, 14]); AR = D["area"]
    YR = np.array(D["years"], float)
    def year_at(t): return float(np.interp(t, YR[:, 0], YR[:, 1]))
    SER = D["series"]
    for s_ in SER:
        P = np.array(s_["pts"], float); s_["_y"], s_["_v"] = P[:, 0], P[:, 1]
        s_["_first"] = s_.get("first", P[0, 0]); s_["_last"] = s_.get("last", P[-1, 0])
    WIN = D["windows"]; fmt = D.get("fmt", "dec1"); RINGS = D.get("rings", []); lbl = D.get("label", "flag_name")

    def wa(t, t0, t1, fi, fo):
        if t < t0 or t > t1: return 0.0
        ain = 1.0 if fi <= 0 else min(1.0, (t - t0) / fi); aout = 1.0 if fo <= 0 else min(1.0, (t1 - t) / fo)
        return max(0.0, min(ain, aout))

    def alpha_at(t): return max([wa(t, *w) for w in WIN] or [0.0])

    def show_a(s_, t):
        sh = s_.get("show")
        if not sh: return 1.0
        f = sh[2] if len(sh) > 2 else 0.3
        return wa(t, sh[0], sh[1], f, f)

    def state(t):
        """[(series, value or None, striped, show alpha)] at time t"""
        y = year_at(t); out = []
        for s_ in SER:
            sa = show_a(s_, t)
            if sa <= 0.01: continue
            if y < s_["_first"] - 1e-6:
                if not D.get("hide_zero"): out.append((s_, None, False, sa))
                continue
            if D.get("gaps") == "hold":          # interpolate only between consecutive reported years, else hold the last one (striped)
                yy = s_["_y"]; k = int(np.searchsorted(yy, y, side="right")) - 1
                if k + 1 < len(yy) and yy[k + 1] - yy[k] <= 1.0: v = float(np.interp(y, yy, s_["_v"]))
                else: v = float(s_["_v"][k])
                st = int(math.floor(y + 1e-6)) not in set(int(a) for a in yy)
            else:
                v = float(np.interp(min(y, s_["_last"]), s_["_y"], s_["_v"])); st = y > s_["_last"] + 1e-6
            if D.get("hide_zero") and v < 0.5: continue
            out.append((s_, v, st, sa))
        return out

    pos = {}; vmax_s = [None]

    def draw(cv, t, a, smooth=True):
        ST = state(t)
        if D.get("sort") == "asc": order = sorted(ST, key=lambda e: (e[1] is None, e[1] if e[1] is not None else 0))
        else: order = sorted(ST, key=lambda e: -(e[1] if e[1] is not None else -1e9))
        if D.get("top"):
            keep = set(e[0]["id"] for e in order[:D["top"]]); order = order[:D["top"]]; ST = [e for e in ST if e[0]["id"] in keep]
        n = max(1, len(order)); pitch = (AR["x1"] - AR["x0"]) / max(n, AR.get("min_slots", n))
        for i, e in enumerate(order):
            k = e[0]["id"]; p = pos.get(k)
            pos[k] = float(i) if (p is None or not smooth) else p + (i - p) * 0.2
        vals = [e[1] for e in ST if e[1] is not None]; mx = max(vals) if vals else 1.0
        if D.get("vmax_fix"): mx = D["vmax_fix"]
        vmax_s[0] = mx if (vmax_s[0] is None or not smooth) else vmax_s[0] + (mx - vmax_s[0]) * 0.15
        hmax = AR["base"] - AR["top"]; cw = min(110, pitch * 0.64)
        for s_, v, st, sa in sorted(ST, key=lambda e: -pos[e[0]["id"]]):
            aa = a * sa; cx = AR["x0"] + pitch * (pos[s_["id"]] + 0.5); col = s_.get("col", SKY)
            hl = s_.get("hl"); vtxt_y = None
            if v is None:
                cv.text("no", cx, AR["base"] - 70, 36, "#9AA6BD", aa, maxw=pitch); cv.text("data", cx, AR["base"] - 32, 36, "#9AA6BD", aa, maxw=pitch)
            else:
                h = max(6.0, hmax * v / max(1e-9, vmax_s[0])); h = min(h, hmax)
                x0 = cx - cw / 2; y0 = AR["base"] - h
                cv.rrect(x0, y0, cw, h, min(12, cw / 4), col, 0.95 * aa)
                if st:      # striped = carried from the last reported year
                    c = cv.c; c.save(); c.rectangle(x0, y0, cw, h); c.clip()
                    for k_ in range(-int(cw), int(h) + int(cw), 22):
                        c.move_to(x0, y0 + k_); c.line_to(x0 + cw, y0 + k_ - cw)
                    cv.col("#0B1730", 0.55 * aa); c.set_line_width(8); c.stroke(); c.restore()
                if hl: cv.rrect(x0 - 4, y0 - 4, cw + 8, h + 8, min(14, cw / 4 + 3), None, aa, stroke=YEL, lw=4, sa=aa)
                vtxt_y = y0 - 34
                cv.text(fmt_val(v, fmt, st), cx, vtxt_y, AR.get("val_size", 40), YEL if hl else WHITE, aa, maxw=pitch - 8, minsz=36)
            for R in RINGS:
                if s_["id"] in R["ids"]:
                    ra = aa * win(t, R["t0"], R["t1"], 0.2, 0.2)
                    if ra > 0.01:
                        top = (AR["base"] - (min(hmax, hmax * v / vmax_s[0]) if v else 90)) - 70
                        cv.rrect(cx - pitch / 2 + 3, top, pitch - 6, AR["base"] + AR.get("lab_h", 120) - top, 16, None, ra, stroke="#FFFFFF", lw=4, sa=ra)
            # label under the column
            ly = AR["base"] + 14
            if lbl == "rotated":
                cv.circ(cx, ly + 12, 10, col, aa)
                c = cv.c; c.save(); c.translate(cx, ly + 30); c.rotate(-math.pi / 2)
                cv.text(s_["label"], 0, 0, AR.get("lab_size", 38), YEL if hl else WHITE, aa, maxw=AR.get("lab_h", 200) - 40, align="r", minsz=36)
                c.restore()
            else:
                if s_.get("flag"): WG.flag(cv, s_["flag"], cx, ly + 30, min(60, pitch * 0.62), aa)
                cv.text(s_["label"], cx, ly + 84, AR.get("lab_size", 38), YEL if hl else WHITE, aa, maxw=pitch * 1.25, minsz=36)

    # frame 0 column state (for the loop blend at the end)
    f0_cache = {}

    def composite(img, t):
        a = alpha_at(t)
        base = img.astype(np.float32)
        OV = [o for o in D.get("overlays", []) if o["at"] - 0.1 <= t <= o["until"]]
        if a <= 0.01 and not OV: return img
        if a <= 0.01:
            cv = Canvas(img.astype(np.float32))
            for o in OV: MC.draw_overlay(cv, o, win(t, o["at"], o["until"], 0.3, 0.3), 1.0)
            return cv.out()
        strip = np.median(base[Y:Y + RH, bx0:bx1], axis=1)                      # (RH, 3) background gradient
        base[Y:Y + RH, X:X + RW] = base[Y:Y + RH, X:X + RW] * (1 - a) + strip[:, None, :] * a
        cv = Canvas(base); lb = float(D.get("loop_blend", 0.0))
        if lb and t > dur - lb:
            u = ease((t - (dur - lb)) / lb)
            if "img" not in f0_cache:
                cv0 = Canvas(base.copy()); pos_bak = dict(pos); vb = vmax_s[0]
                pos.clear(); vmax_s[0] = None; draw(cv0, 0.0, 1.0, smooth=False); f0_cache["state"] = (dict(pos), vmax_s[0])
                pos.clear(); pos.update(pos_bak); vmax_s[0] = vb
            draw(cv, t, a * (1 - u))
            p_now, v_now = dict(pos), vmax_s[0]
            pos.clear(); pos.update(f0_cache["state"][0]); vmax_s[0] = f0_cache["state"][1]
            draw(cv, 0.0, a * u, smooth=False)
            pos.clear(); pos.update(p_now); vmax_s[0] = v_now
        else:
            draw(cv, t, a)
        for o in OV: MC.draw_overlay(cv, o, win(t, o["at"], o["until"], 0.3, 0.3), 1.0)
        return cv.out()

    if os.environ.get("TPL_SNAPS"):
        ts = [float(x) for x in os.environ["TPL_SNAPS"].split(",")]; tiles = []
        for t in ts:
            img = grab(src, t); img = cv2.resize(img, (W, H)) if img.shape[1] != W else img
            pos.clear(); vmax_s[0] = None
            for k in range(30): composite(img, max(0, t - (30 - k) / FPS))     # settle the smoothing
            o = composite(img, t); tiles.append(np.hstack([cv2.resize(img, (360, 640)), cv2.resize(o, (360, 640))]))
        p = f"{work}/snaps.png"; cv2.imwrite(p, np.vstack(tiles)); print("snaps", p); return {"snaps": p}

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    sil = f"{work}/v.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", sil], stdin=subprocess.PIPE)
    i = 0; fb = W * H * 3
    while True:
        b = dec.stdout.read(fb)
        if len(b) < fb: break
        img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        enc.stdin.write(np.ascontiguousarray(composite(img, i / FPS)).tobytes()); i += 1
        if i % 300 == 0: log(f"frame {i}/{n_fr}")
    enc.stdin.close(); assert enc.wait() == 0; dec.wait()
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", sil, "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", final], check=True)
    return dict(voice="original audio track kept (voice and music unchanged)", body_s=round(i / FPS, 2), opener=None, mix=measure(final))
