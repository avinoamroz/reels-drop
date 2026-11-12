#!/usr/bin/env python3
"""Face version of a map_counter reel: Noam's own recording (cut out with Robust Video Matting) over the reel's visuals.

  python3 tools/face/render_face.py tools/face/<id>.json

The spec is a normal map_counter spec plus a "face" block:
  face.sources  {key: {wav, fgr, alpha, words}}   48 kHz mono wav/flac of the take, RVM foreground + alpha videos (tools/face/matte.py,
                already at the placement scale), word timestamps [{w, s, e}] (tools/tx.py format); repo-relative paths
  face.clips    [{src, a, b, line}]                cut list in seconds of the source; clips play back to back, in order
                {..., hold: true, rev: true}      silent visual hold (no words, voice muted), optionally played backwards
  face.tail_from_cut  true                        the end hold is whatever the cut runs past the last word (data.tail)
  face.place    {x0, y0, side_feather}            where the matte's top left corner lands on the 1080x1920 canvas
  face.say_fix  {line: "text"}                    subtitle text per line (same word count as the words in its clips;
                                                  a "+" token merges that heard word into the previous one, e.g. "60% +")
  face.clean    {k, white}                        optional matte clean up (largest blob after a k px opening, white wall leaks)
  face.counter  {x, y, w, h}                      compact counter card (the template's full width counter is moved off canvas)
Works with any template that has run(S) + build_voice/encode/draw_subs globals (map_counter, photo_story).
Pass 1 renders the template visuals and the mix (his voice + music bed + SFX, -16 LUFS) with no subtitles,
pass 2 composites him, then subtitles and the counter on top, loops the last 0.5 s into frame 0, and runs finish().
Nothing is posted.
"""
import os, sys, json, subprocess, re, math
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "templates", "core")); sys.path.insert(0, os.path.join(REPO, "templates"))
import numpy as np, cv2, soundfile as sf
import common as CM
from common import Canvas, Timing, sub_chunks, draw_subs, seg, win, YEL, WHITE
import widgets as WG

SR = 48000; FPS = 30; W, H = 1080, 1920
rp = lambda p: p if os.path.isabs(p) else os.path.join(REPO, p)


def set_tail(S, J):
    if S["face"].get("tail_from_cut"):
        S["data"]["tail"] = round(J["dur"] - J["lines"][-1]["t1"], 3)


def build_cut(S, work):
    F = S["face"]; srcs = F["sources"]; lead = F.get("lead", 0.0)
    wav = {k: sf.read(rp(v["wav"]))[0].astype(np.float32) for k, v in srcs.items()}
    words = {k: json.load(open(rp(v["words"]))) for k, v in srcs.items()}
    nl = len([L for L in S["lines"] if L.get("say", "").strip()])
    lines = [dict(i=i, text="", t0=None, t1=None, words=[]) for i in range(nl)]
    audio, vlist, t = [], [], 0.0
    xf = int(0.012 * SR)
    clips = []
    if isinstance(F["clips"], str): F["clips"] = json.load(open(os.path.join(REPO, F["clips"])))
    for c in F["clips"]:
        a = round(c["a"] * FPS) / FPS; b = round(c["b"] * FPS) / FPS
        seg_ = wav[c["src"]][int(a * SR):int(b * SR)].copy()
        if c.get("hold"): seg_[:] = 0                                # visual hold: his silent footage, the music carries it
        seg_[:xf] *= np.linspace(0, 1, xf); seg_[-xf:] *= np.linspace(1, 0, xf)
        audio.append(seg_); vlist.append(dict(src=c["src"], f0=int(round(a * FPS)), n=int(round((b - a) * FPS)), t=t, rev=bool(c.get("rev"))))
        clips.append(dict(c, a=a, b=b, t=t)); t += (b - a)
    # words: each clip group (one take range) carries word indices [i0, i1] of its take; whisper's times are rough, so the
    # group's word span is stretched onto the group's speech span (first clip start + 0.08 to last clip end - 0.13), then
    # mapped through the cut (a word inside a removed pause snaps to the next kept clip)
    for g in sorted({c["g"] for c in clips if not c.get("hold")}):
        cg = [c for c in clips if c.get("g") == g]; i0, i1 = cg[0]["w"]; ws_ = words[cg[0]["src"]][i0:i1 + 1]
        s0, s1 = cg[0]["a"] + 0.08, cg[-1]["b"] - 0.13; w0, w1 = ws_[0]["s"], ws_[-1]["e"]
        def m(x):
            x = s0 + (x - w0) * (s1 - s0) / max(0.1, w1 - w0)
            for c in cg:
                if x <= c["b"]: return c["t"] + max(0.0, x - c["a"])
            return cg[-1]["t"] + cg[-1]["b"] - cg[-1]["a"]
        L = lines[cg[0]["line"]]
        for w in ws_:
            ta, tb = m(w["s"]), m(max(w["e"], w["s"] + 0.05))
            L["words"].append(dict(w=w["w"], t0=round(ta, 3), t1=round(max(tb, ta + 0.05), 3)))
    for L in lines:
        fix = F.get("say_fix", {}).get(str(L["i"]))
        if fix:
            fw = fix.split()
            if len(fw) != len(L["words"]):
                raise SystemExit(f"say_fix line {L['i']}: {len(fw)} words vs {len(L['words'])} heard: {' '.join(w['w'] for w in L['words'])}")
            for w, x in zip(L["words"], fw): w["w"] = x
            ws_ = []
            for w in L["words"]:
                if w["w"] == "+" and ws_: ws_[-1]["t1"] = w["t1"]
                else: ws_.append(w)
            L["words"] = ws_
        L["text"] = " ".join(w["w"] for w in L["words"])
        L["t0"], L["t1"] = L["words"][0]["t0"], L["words"][-1]["t1"]
    vo = np.concatenate(audio)
    # light clean up: rumble cut + gentle FFT denoise, then peak normalise (loudness is set by the mix)
    raw = f"{work}/vo_raw.wav"; sf.write(raw, vo, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", F.get("af", "highpass=f=75,afftdn=nr=8:nf=-42:tn=1"),
                    "-ar", str(SR), "-ac", "1", f"{work}/vo.wav"], check=True)
    y, _ = sf.read(f"{work}/vo.wav"); y = y / (np.abs(y).max() + 1e-9) * 0.89; sf.write(f"{work}/vo.wav", y.astype(np.float32), SR)
    J = dict(dur=round(t, 3), voice="Noam's own recording (" + ", ".join(sorted(srcs)) + ")", lines=lines)
    set_tail(S, J)
    json.dump(J, open(f"{work}/timing.json", "w"), indent=0)
    json.dump(vlist, open(f"{work}/vlist.json", "w"))
    for L in lines: CM.log(f"L{L['i']} {L['t0']:6.2f}-{L['t1']:6.2f}  {L['text']}")
    return J


def pass1(S, work):
    import importlib
    MC = importlib.import_module(S["template"])
    J = build_cut(S, work)
    MC.build_voice = lambda S_, w_: J
    enc = MC.encode
    MC.encode = lambda frame, dur, audio, out, loop=True, crf=18: enc(frame, dur, audio, f"{work}/body.mp4", loop=False, crf=crf)
    MC.draw_subs = lambda *a, **k: None
    if os.environ.get("TPL_SNAPS"): MC.run(S); raise SystemExit
    MC.run(S)
    return J


def reader(cmd, shape):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE); n = int(np.prod(shape))
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(shape)


def face_frames(S, work, n_total):
    """yields (fgr float32 BGR, alpha float32) at matte size along the cut, holding the last frame to the end"""
    F = S["face"]; vlist = json.load(open(f"{work}/vlist.json")); last = None; k = 0; XF = int(F.get("cut_xfade", 3))
    for v in vlist:
        s_ = F["sources"][v["src"]]
        cap = cv2.VideoCapture(rp(s_["fgr"])); mw, mh = int(cap.get(3)), int(cap.get(4)); cap.release()
        tr = f"trim=start_frame={v['f0']}:end_frame={v['f0'] + v['n']},setpts=PTS-STARTPTS" + (",reverse" if v.get("rev") else "")
        fg = reader(["ffmpeg", "-v", "error", "-i", rp(s_["fgr"]), "-vf", tr, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], (mh, mw, 3))
        al = reader(["ffmpeg", "-v", "error", "-i", rp(s_["alpha"]), "-vf", tr, "-f", "rawvideo", "-pix_fmt", "gray", "-"], (mh, mw))
        prev = last
        for j, (f_, a_) in enumerate(zip(fg, al)):
            cur = (f_, a_)
            if prev is not None and j < XF:                          # soften the jump cut: short dissolve from the last kept frame
                u = (j + 1) / (XF + 1)
                cur = ((prev[0] * (1 - u) + f_ * u).astype(np.uint8), (prev[1] * (1 - u) + a_ * u).astype(np.uint8))
            last = (f_, a_); k += 1; yield cur
    while k < n_total:
        k += 1; yield last


def clean_alpha(a, fg, C):
    """matte leaks (face.clean {k, white}): fade near-white background (bright, unsaturated wall) along the edge only, then a
    morphological opening (k px ellipse) keeps the largest blob, so thin protrusions and floating bits drop out"""
    if C.get("white", True):
        hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).astype(np.float32)
        white = np.clip((hsv[..., 2] - 170) / 40, 0, 1) * np.clip((70 - hsv[..., 1]) / 30, 0, 1)
        core = cv2.erode((a > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 4)
        a = a * (1 - white * (1 - core))
    k = int(C.get("k", 15))
    op = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(op)
    if n > 1: op = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    op = cv2.dilate(op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return np.minimum(a, cv2.GaussianBlur(op.astype(np.float32), (0, 0), 2.0))


def prep_alpha(a, feather, fg=None, C=None):
    a = a.astype(np.float32) / 255
    if C and fg is not None: a = clean_alpha(a, fg, C)
    a = np.clip((a - 0.04) / 0.92, 0, 1)                          # kill faint background haze, keep hair soft
    a = cv2.GaussianBlur(a, (0, 0), 0.9)                            # soft matte edge
    h, w = a.shape
    if feather:                                                     # shirt meets the matte's sides: fade it out
        x = np.arange(w, dtype=np.float32)
        r = np.clip(np.minimum(x, w - 1 - x) / feather, 0, 1) ** 1.3
        yy = np.clip((np.arange(h, dtype=np.float32) - h * 0.55) / (h * 0.2), 0, 1)[:, None]
        a = a * (1 - yy * (1 - r[None, :]))
    return a


def bust_mask(a, cx, F):
    """head and upper shoulders only: torso fades out below the chin, shoulders fade out sideways (no hard edges)"""
    B = F.get("bust", {}); h, w = a.shape
    yy = np.arange(h, dtype=np.float32)[:, None]; xx = np.arange(w, dtype=np.float32)[None, :]
    v = np.clip((B.get("fade_end", 830) - yy) / (B.get("fade_end", 830) - B.get("fade_start", 640)), 0, 1); v = v * v * (3 - 2 * v)
    hw = B.get("half_width", 255); soft = B.get("soft", 80)
    hz = np.clip((hw - np.abs(xx - cx)) / soft, 0, 1); hz = hz * hz * (3 - 2 * hz)
    hz = np.maximum(hz, (yy < B.get("chin", 600)).astype(np.float32))          # never cut the head itself
    return a * v * hz


def place_at(t, keys):
    """keys [(t, X, Y, s)] -> eased position; each move takes 0.45 s starting at its t"""
    X, Y, s_ = keys[0][1:]
    for k in keys[1:]:
        if t < k[0]: break
        u = CM.ease(min(1.0, (t - k[0]) / 0.45)); X, Y, s_ = X + (k[1] - X) * u, Y + (k[2] - Y) * u, s_ + (k[3] - s_) * u
    return X, Y, s_


def paste_bust(img, fg, a, cx, X, Y, sc, F):
    """in place: his bust (alpha a already cleaned, at matte size) with its anchor point at canvas (X, Y), scale sc, drop shadow"""
    P = F.get("place", {}); ax, ay = P.get("anchor", [325, 400]); R0, R1 = P.get("rows", [140, 860])
    a = bust_mask(a, cx, F)[R0:R1]; fgc = fg[R0:R1].astype(np.float32)
    nw, nh = int(round(a.shape[1] * sc)), int(round(a.shape[0] * sc))
    ip = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
    a = cv2.resize(a, (nw, nh), interpolation=ip); fgc = cv2.resize(fgc, (nw, nh), interpolation=ip)
    x0 = int(round(X - ax * sc)); y0 = int(round(Y - (ay - R0) * sc))
    ya, yb = max(0, y0), min(H, y0 + nh); xa, xb = max(0, x0), min(W, x0 + nw)
    if yb <= ya or xb <= xa: return
    A_ = a[ya - y0:yb - y0, xa - x0:xb - x0][..., None]; FG = fgc[ya - y0:yb - y0, xa - x0:xb - x0]
    pad = 40; big = np.zeros((nh + 2 * pad, nw + 2 * pad), np.float32); big[pad + 14:pad + 14 + nh, pad:pad + nw] = a
    sh = cv2.GaussianBlur(big, (0, 0), 16 * max(0.6, sc))
    sya, syb = max(0, y0 - pad), min(H, y0 + nh + pad); sxa, sxb = max(0, x0 - pad), min(W, x0 + nw + pad)
    S_ = sh[sya - (y0 - pad):syb - (y0 - pad), sxa - (x0 - pad):sxb - (x0 - pad)][..., None]
    img[sya:syb, sxa:sxb] *= (1 - F.get("shadow", 0.55) * np.clip(S_ * 1.3, 0, 1))
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - A_) + FG * A_


def pass2(S, work, J):
    F = S["face"]
    body = f"{work}/body.mp4"; T = Timing(J, S["data"].get("tail", 0.75)); dur = T.END; n_total = int(round(dur * FPS))
    MODES = F["modes"]; keys = [(T.at(m["at"]) - (0.35 if m["at"] not in (0, "0") else 0), *MODES[m["mode"]]) for m in F["moves"]]
    CH = sub_chunks(T, S.get("subs_replace")); SY = F.get("subs_y", 1090)
    C = F.get("counter"); CT = S["data"].get("counter")
    if C and CT:
        ct0 = T.at(CT.get("at")); ct1 = T.at(CT.get("until"), T.END - 0.6); cst = [(T.at(s_["at"]), float(s_["value"])) for s_ in CT["steps"]]
    sil = f"{work}/comp_silent.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", sil], stdin=subprocess.PIPE)
    bg_it = reader(["ffmpeg", "-v", "error", "-i", body, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], (H, W, 3))
    f0 = None; cxs = None
    for n, (bg, (fg, al)) in enumerate(zip(bg_it, face_frames(S, work, n_total))):
        if n >= n_total: break
        t = n / FPS
        img = bg.astype(np.float32)
        a = prep_alpha(al, 0, fg, F.get("clean"))
        band = a[250:520]; m_ = band.sum(0); cx = float((m_ * np.arange(len(m_))).sum() / max(1e-3, m_.sum()))
        cxs = cx if cxs is None else cxs * 0.85 + cx * 0.15                  # follow his head slowly for the shoulder fade
        paste_bust(img, fg, a, cxs, *place_at(t, keys), F)
        cv = Canvas(img)
        if C and CT:
            ca = win(t, ct0, ct1, 0.35, 0.35)
            if ca > 0.01:
                v, prev = 0.0, 0.0
                for t0, val in cst:
                    if t >= t0: v = WG.roll(t, t0, prev, val, 0.7)
                    prev = val
                WG.panel(cv, C["x"], C["y"], C["w"], C["h"], ca)
                cx = C["x"] + C["w"] / 2
                for k_, ln in enumerate(C.get("label", ["Layers of", "control"])):
                    cv.text(ln, cx, C["y"] + 52 + k_ * 46, 40, "#C9D6EA", ca, maxw=C["w"] - 30)
                cv.text(WG.fmtn(v), cx, C["y"] + C["h"] - 78, 130, YEL, ca, maxw=C["w"] - 30)
        draw_subs(cv, CH, t, SY)
        out = np.clip(cv.out(), 0, 255).astype(np.uint8)
        if n == 0: f0 = out.astype(np.float32)
        if t > dur - 0.5:                                           # circular: dissolve into frame 0
            u = CM.ease((t - (dur - 0.5)) / 0.5); out = (out.astype(np.float32) * (1 - u) + f0 * u).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        if n % 150 == 0: CM.log(f"comp {t:5.1f}/{dur:.1f}")
    enc.stdin.close(); enc.wait()
    final = os.path.join(REPO, "out", f"{S['id']}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sil, "-i", f"{work}/mix.m4a", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    return final


def main():
    S = CM.load_spec(sys.argv[1]); work = CM.workdir(S)
    for w in CM.lint(S): CM.log("lint warning:", w)
    if "--pass2" in sys.argv: J = json.load(open(f"{work}/timing.json")); set_tail(S, J)     # reuse the rendered body + mix
    else: J = pass1(S, work)
    if "--pass1" in sys.argv: return
    final = pass2(S, work, J)
    rep = CM.finish(S, final, dict(voice=J["voice"], body_s=Timing(J, S["data"].get("tail", 0.75)).END,
                                   transcript_lines=[L["text"] for L in J["lines"]]))
    cover_face(S, work)
    print(json.dumps({k: rep[k] for k in ("id", "duration", "audio", "outputs")}, indent=1))


def cover_face(S, work):
    """the standard cover (blurred frame + hook + big number) with his floating bust sharp on top"""
    F = S["face"]; p = os.path.join(REPO, "covers", f"{S['id']}_cover.jpg"); img = cv2.imread(p).astype(np.float32)
    k = int(F.get("cover_frame", 0))
    for j, (fg, al) in enumerate(face_frames(S, work, k + 1)):
        if j == k: break
    a = prep_alpha(al, 0, fg, F.get("clean")); m_ = a[250:520].sum(0); cx = float((m_ * np.arange(len(m_))).sum() / max(1e-3, m_.sum()))
    paste_bust(img, fg, a, cx, *F["modes"][F.get("cover_mode", F["moves"][0]["mode"])], F)
    cv2.imwrite(p, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
