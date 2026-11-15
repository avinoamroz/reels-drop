"""talk_cut: Noam on camera the whole way (15 Nov 2026). One long talking-head take is cut down to the kept segments
(retakes and long pauses out), his real voice with burned-in subtitles of exactly what he says (the take's word
timestamps, with spec fixes), the hook on screen at frame 0, picture-in-picture artifact photos over his chest with a
small label (what, year, source, photo credit) while he keeps talking, his recorded follow line inserted over a still
of him, and the last 0.5 s dissolving into frame 0 (circular). A quiet bed sits about 20 dB under his voice; the mix
is levelled to -16 LUFS. The report lists what whisper hears in every kept segment and in every gap that was cut.

data keys:
  clip        repo path of the take; flip false (default) or true
  words       repo path of its word timestamps json ({"segs": [{"w": [[word, t0, t1], ...]}]})
  keep        [[a, b], ...] source seconds, in order
  fix         [{"near": t, "from": "that", "to": "Dan"}]   subtitle word fixes (the word nearest t)
  case        {"history": "HISTORY"}                        subtitle casing
  photos      [{"file": "photos/x.jpg", "from": src_t, "until": src_t, "title": "...", "label": "...", "credit": "..."}]
  follow      {"wav": "ref/follow_line_noam.wav", "text": "...", "before": src_t}
  hook_until  source seconds the hook stays on screen (default: end of the first sentence)
"""
import os, sys, re, json, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
import common as CM


def run(S):
    import face_hook as FH, map_counter as MC, soundfile as sf
    D = S["data"]; work = workdir(S); M = S.get("music") or {}; LUFS = float(M.get("lufs", -16))
    F = D.get("follow") or {}
    need = [D["clip"], D["words"], F.get("wav", "ref/follow_line_noam.wav")] + [p["file"] for p in D.get("photos", [])]
    miss = [f for f in need if not os.path.exists(os.path.join(REPO, f))]
    if miss: subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", *["/" + f for f in miss]], check=False)
    clip = os.path.join(REPO, D["clip"])
    keep = [(float(a), float(b)) for a, b in D["keep"]]
    # words: merge "3" + ",000"; fixes; casing
    WD = []
    for s in json.load(open(os.path.join(REPO, D["words"])))["segs"]:
        for w, a, b in s["w"]:
            if WD and re.match(r"^,\d", w): WD[-1] = [WD[-1][0] + w, WD[-1][1], b]
            else: WD.append([w, a, b])
    for fx in D.get("fix", []):
        k = min(range(len(WD)), key=lambda i: abs(WD[i][1] - fx["near"]) + (0 if WD[i][0].strip(".,?!").lower() == fx["from"].lower() else 99))
        WD[k][0] = re.sub(re.escape(fx["from"]), fx["to"], WD[k][0], flags=re.I)
    for a_, b_ in (D.get("case") or {}).items():
        for w in WD: w[0] = re.sub(rf"\b{re.escape(a_)}\b", b_, w[0], flags=re.I)
    # timeline: clip segments + the follow still
    fwav = os.path.join(REPO, F.get("wav", "ref/follow_line_noam.wav"))
    fw, fsr = sf.read(fwav); fw = fw if fw.ndim == 1 else fw.mean(1)
    if fsr != SR: fw = np.interp(np.arange(int(len(fw) * SR / fsr)) / SR, np.arange(len(fw)) / fsr, fw)
    fdur = math.ceil((len(fw) / SR + 0.25) * FPS) / FPS
    fb = float(F["before"]) if F else None
    parts = []   # (kind, a, b) in source time, or ("follow", t_src, dur)
    for a, b in keep:
        if fb is not None and a <= fb < b:
            parts += [("clip", a, fb), ("follow", fb, fdur), ("clip", fb, b)]
        elif fb is not None and abs(a - fb) < 1e-6:
            parts += [("follow", fb, fdur), ("clip", a, b)]
        else: parts.append(("clip", a, b))
    parts = [p for p in parts if p[0] == "follow" or p[2] - p[1] > 0.02]
    # output time map
    T, segs = 0.0, []
    for p in parts:
        n = int(round(((p[2] - p[1]) if p[0] == "clip" else p[2]) * FPS))
        segs.append((p, T, n)); T += n / FPS
    tot = T
    def out_t(ts):         # source time -> output time (inside a kept clip part)
        for p, t0, n in segs:
            if p[0] == "clip" and p[1] - 1e-6 <= ts <= p[2] + 1e-6: return t0 + (ts - p[1])
        return None
    # subtitles: words whose middle is inside a kept part, grouped by sentence (halves of 6+ words)
    OW = []
    for w, a, b in WD:        # a word is kept where most of it (or at least 0.15 s of it) is inside a kept clip part
        best = None
        for p, t0, n in segs:
            if p[0] != "clip": continue
            ov = min(b, p[2]) - max(a, p[1])
            if b - a < 0.02 and p[1] <= a <= p[2]: ov = 0.1        # zero-length ASR word inside the part
            if ov > 0 and (ov >= 0.1 or ov >= 0.4 * (b - a)) and (best is None or ov > best[0]): best = (ov, p, t0)
        if best is None: continue
        ov, p, t0 = best; oa = t0 + max(a, p[1]) - p[1]; ob = t0 + min(b, p[2]) - p[1]
        if b - a > 1.0: oa = max(oa, ob - 0.6)       # long ASR spans hold a pause: show the word near its end
        if b - a < 0.02: oa, ob = oa - 0.12, ob + 0.12
        if (not OW or re.search(r"[.?!]$", OW[-1][0])) and w[:1].islower(): w = w[:1].upper() + w[1:]
        OW.append([w, oa, ob])
    CH = FH.chunks(OW, 0.0)
    if F:
        t_f = next(t0 for p, t0, n in segs if p[0] == "follow")
        ft = F.get("text", "If you love Israeli facts, follow Noam Roz for more.")
        fwords, fsc = FH.align_said(ft, FH.asr_words(fwav, prompt=ft))
        FCH = FH.chunks([[w, a + t_f + 0.1, b + t_f + 0.1] for w, a, b in fwords], 0.0); FCH[-1][2] = t_f + fdur - 0.05
        CH = sorted(CH + FCH, key=lambda c: c[1])
        for j in range(len(CH) - 1): CH[j][2] = min(CH[j][2], CH[j + 1][1] - 0.02)
    # photos in output time
    PH = []
    for ph in D.get("photos", []):
        a = out_t(ph["from"]); b = out_t(ph["until"])
        if a is None or b is None: raise SystemExit(f"photo window outside the kept parts: {ph['file']}")
        img = cv2.imread(os.path.join(REPO, ph["file"]))
        PH.append(dict(ph, a=a, b=b, img=img))
    hook_img = HK.make(S["_hook"], int(D.get("hook_size", 118)), maxw=960)
    hook_end = out_t(float(D.get("hook_until", keep[0][1]))) or 2.5
    BX = D.get("photo_box", [90, 980, 900, 560])    # x, y, w, h of the picture-in-picture panel (over his chest)

    def overlay(fr, t):
        cv = Canvas(fr.astype(np.float32))
        for ph in PH:
            a = win(t, ph["a"], ph["b"], 0.25, 0.25)
            if a <= 0.01: continue
            x, y, w, h = BX; ih = h - 180
            cv.rrect(x - 8, y - 8, w + 16, h + 16, 34, "#0B1730", 0.92 * a, stroke=SKY, lw=3, sa=0.8 * a)
            out = cv.out().astype(np.float32)
            im = ph["img"]; z = 1.0 + 0.04 * min(1, (t - ph["a"]) / max(0.5, ph["b"] - ph["a"]))
            if "bgc" not in ph:                     # blurred cover fill behind the whole (contained) photo
                s0 = max(w / im.shape[1], ih / im.shape[0]); r0 = cv2.resize(im, (int(im.shape[1] * s0) + 1, int(im.shape[0] * s0) + 1), interpolation=cv2.INTER_AREA)
                oy, ox = (r0.shape[0] - ih) // 2, (r0.shape[1] - w) // 2
                ph["bgc"] = cv2.GaussianBlur(r0[oy:oy + ih, ox:ox + w], (0, 0), 18).astype(np.float32) * 0.45
            s1 = min(w / im.shape[1], ih / im.shape[0]) * z
            r = cv2.resize(im, (max(1, int(im.shape[1] * s1)), max(1, int(im.shape[0] * s1))), interpolation=cv2.INTER_AREA).astype(np.float32)
            crop = ph["bgc"].copy(); rh, rw = min(ih, r.shape[0]), min(w, r.shape[1])
            cy0, cx0 = (ih - rh) // 2, (w - rw) // 2; ry0, rx0 = (r.shape[0] - rh) // 2, (r.shape[1] - rw) // 2
            crop[cy0:cy0 + rh, cx0:cx0 + rw] = r[ry0:ry0 + rh, rx0:rx0 + rw]
            reg = out[y:y + ih, x:x + w]; out[y:y + ih, x:x + w] = reg * (1 - a) + crop * a
            cv = Canvas(out)
            WG.title(cv, ph["title"], y + ih + 52, a, 50, maxw=w - 40)
            for li, ln in enumerate(ph["label"].split("|")):
                cv.text(ln, x + w / 2, y + ih + 106 + 42 * li, 36, "#C9D6EA", a, bold=False, maxw=w - 40, minsz=36)
        out = cv.out().astype(np.float32)
        if t < hook_end + 0.6:
            out = HK.paste(out, hook_img, 540, int(D.get("hook_y", 1240)), 0.94 + 0.06 * eout(t / 0.2), 1 - seg(t, hook_end + 0.1, hook_end + 0.5))
        cv = Canvas(out)
        MC.draw_brand(cv, "Israel Facts", 1 - seg(t, 2.5, 3.0) if t < tot - 1 else seg(t, tot - 1.0, tot - 0.6))
        if F and t_f - 0.05 <= t <= t_f + fdur:
            al = min(1.0, (t - t_f) / 0.2) * min(1.0, (t_f + fdur - t) / 0.15 + 1e-4)
            MC.draw_overlay(cv, {"text": F.get("screen", "Like Israel facts?|Follow *@noamroz__*"), "y": int(F.get("y", 1250)), "size": 76, "cue": True}, max(0.0, al), 0.92 + 0.08 * eout(min(1.0, (t - t_f) / 0.25)))
        draw_subs(cv, CH, t, D.get("subs_y", 1612))
        return cv.out()

    if os.environ.get("TPL_SNAPS"):
        ts = [float(x) for x in os.environ["TPL_SNAPS"].split(",")]
        tiles = []
        for t in ts:
            p, t0, n = next(s_ for s_ in segs if s_[1] <= t < s_[1] + s_[2] / FPS)
            st = p[1] + (t - t0) if p[0] == "clip" else p[1] - 0.04
            fr = cv2.resize(grab(clip, st), (W, H)); fr = cv2.flip(fr, 1) if D.get("flip") else fr; tiles.append(cv2.resize(overlay(fr, t), (360, 640)))
        pth = f"{work}/snaps.png"; cv2.imwrite(pth, np.hstack(tiles)); print("snaps", pth); return {"snaps": pth}
    # video: one sequential decode of the take, frames picked by index, overlays drawn, piped to the encoder
    f0 = None
    vf = ("hflip," if D.get("flip") else "") + f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}"
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    sil = f"{work}/v.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", sil], stdin=subprocess.PIPE)
    cur_i, cur = -1, None; frames_out = []; ntot = sum(n for _, _, n in segs); k = 0
    def read_to(idx):
        nonlocal cur_i, cur
        while cur_i < idx:
            b = dec.stdout.read(W * H * 3)
            if len(b) < W * H * 3: break
            cur = np.frombuffer(b, np.uint8).reshape(H, W, 3); cur_i += 1
        return cur
    for p, t0, n in segs:
        for j in range(n):
            t = t0 + j / FPS
            if p[0] == "clip": fr = read_to(int(round((p[1] + j / FPS) * FPS)))
            else: fr = read_to(max(0, int(round(p[1] * FPS)) - 1))
            img = overlay(fr, t)
            if k == 0: f0 = img.astype(np.float32)
            if t > tot - 0.5:
                u = ease((t - (tot - 0.5)) / 0.5); img = (img.astype(np.float32) * (1 - u) + f0 * u).astype(np.uint8)
            enc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes()); k += 1
            if k % 300 == 0: log(f"frame {k}/{ntot}")
    enc.stdin.close(); assert enc.wait() == 0; dec.kill()
    # audio: his kept speech + his follow line + a quiet bed, levelled
    src = FH.audio_of(clip); parts_a = []
    for p, t0, n in segs:
        L = int(round(n / FPS * SR))
        if p[0] == "clip":
            i = int(round(p[1] * SR)); x = src[:, i:i + L]; x = np.pad(x, ((0, 0), (0, L - x.shape[1]))); parts_a.append(FH.fade(x, 12))
        else: parts_a.append(None)
    speech = np.concatenate([x for x in parts_a if x is not None], axis=1); sp_l = lufs_of(speech)
    fw2 = np.stack([fw, fw]); fw2 = fw2 * 10 ** ((sp_l - lufs_of(fw2)) / 20)
    out = []
    for (p, t0, n), x in zip(segs, parts_a):
        if x is None:
            L = int(round(n / FPS * SR)); v = np.zeros((2, L)); i0 = int(0.1 * SR); m = min(fw2.shape[1], L - i0); v[:, i0:i0 + m] = fw2[:, :m]; x = FH.fade(v, 12)
        out.append(x)
    x = np.concatenate(out, axis=1); n = int(round(tot * SR)); x = np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))))[:, :n]
    bd = M.get("bed", "doc_piano")
    if bd != "none":
        b = bed(bd, tot + 0.5)[:, :n]; b = np.pad(b, ((0, 0), (0, n - b.shape[1])))
        b = b * 10 ** ((sp_l + float(M.get("bed_rel", -20)) - lufs_of(b)) / 20)
        fade_ = np.minimum(1, (n - np.arange(n)) / (0.4 * SR)); x = x + b * fade_[None]
    aud = loudnorm(x, f"{work}/mix.m4a", LUFS)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sil, "-i", f"{work}/mix.m4a", "-map", "0:v", "-map", "1:a", "-c", "copy",
                    "-t", f"{tot:.3f}", "-movflags", "+faststart", final], check=True)
    # report: what whisper hears in each kept part and in each cut gap (no prompt)
    heard_k, heard_gap = [], []
    prev = None
    for a, b in keep:
        if prev is not None and a - prev > 0.35:
            hw = FH.asr_words(clip, prev, a - prev); heard_gap.append([round(prev, 2), round(a, 2), " ".join(w[0] for w in hw)])
        hw = FH.asr_words(clip, a, b - a); heard_k.append([round(a, 2), round(b, 2), " ".join(w[0] for w in hw)]); prev = b
    lv = {"speech": round(sp_l, 1), "follow_before_mix": round(lufs_of(fw2), 1)}
    return dict(voice="Noam's own voice (one take, cut), his recorded follow line, quiet bed", clip=D["clip"], keep=keep, parts=[list(p) for p, _, _ in segs],
                mix=aud, levels=lv, subs=CH, heard_kept=heard_k, heard_cut=heard_gap, said=" ".join(w[0] for w in OW),
                follow_at=round(t_f, 3) if F else None, photos=[{k_: ph[k_] for k_ in ("file", "title", "a", "b")} for ph in PH])
