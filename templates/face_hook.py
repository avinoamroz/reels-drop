"""face_hook: Noam's own face clip opens a published reel (15 Nov 2026). Frame 0 is Noam on camera, his real voice with
the lips in sync (the clip's own audio), big hook text over his chest, the Israel Facts mark, and burned-in subtitles of
exactly what he says (spec data.said; the words are timed to the clip by faster-whisper, the text shown is data.said).
Then a hard cut into the source reel at `cut` (just after the source's own hook line, so the hook is not said twice);
the rest of the source is kept as it is, except the follow line: the reel clone's follow line take
(ref/follow_line_tts_chain.wav, house chain applied; Noam 22 Nov; "own_voice": true keeps ref/follow_line_noam.wav) replaces the TTS follow line, or is inserted right before the final lead-in line (freeze frame with the on-screen
"Like Israel facts? Follow @noamroz__" pill and the sky chevron cue), as Noam asked on 15 Nov. The last 1 s of the
body dissolves into the new frame 0 (circular). Everything is levelled to the spec loudness (default -16 LUFS):
the face clip is levelled on its own first so his real voice and the Cartesia narration match.

data keys:
  clip        repo path of the face clip, e.g. "hooks/face/h_violent_country.mp4"
  flip        true (default): hflip the clip (Noam's phone clips are mirrored)
  said        exact words Noam says in the clip (subtitles and the transcript check use this text)
  clip_in, clip_out   optional trim in clip seconds (default: 0.08 s before his first word, 0.14 s after his last)
  source      repo path of the source reel mp4
  cut         seconds in the source where the body starts, or
  cut_after   a phrase of the source's narration; the body starts right after its first occurrence (+ cut_pad, default 0.06)
  follow      {"wav": "ref/follow_line_tts_chain.wav", "text": "If you love Israeli facts, follow Noam Roz for more.",
               "mode": "insert" | "replace" | "none",
               "before": phrase (insert: the final lead-in line; its LAST occurrence) or "at": source seconds,
               "replace": phrase of the TTS follow line to replace ("replace_end": its last words), "bed": music bed}
  hook_y      default 1240 (over his chest); hook_size default 118; brand true adds the Israel Facts mark
  zoom        push-in over the face clip, default 0.04
"""
import os, sys, re, json, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
from common import *
import common as CM

_M = [None]


def model():
    if _M[0] is None:
        from faster_whisper import WhisperModel
        _M[0] = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    return _M[0]


def pcm16k(path, t0=0.0, dur=None):
    a = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-i", path] + (["-t", f"{dur:.3f}"] if dur else [])
    raw = subprocess.run(a + ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def asr_words(path, t0=0.0, dur=None, prompt=None):
    segs, _ = model().transcribe(pcm16k(path, t0, dur), language="en", beam_size=5, word_timestamps=True,
                                 condition_on_previous_text=False, initial_prompt=prompt)
    return [(w.word.strip(), w.start + t0, w.end + t0) for s in segs for w in (s.words or [])]


def norm(s):
    import check
    return check.norm(s)


def tokens(words):
    """[(word, t0, t1)] -> [(token, t0, t1, word index)]"""
    out = []
    for k, (w, a, b) in enumerate(words):
        for t in norm(w): out.append((t, a, b, k))
    return out


def find_phrase(TK, phrase, last=False):
    """(start, end) seconds of a phrase in the token list; exact match first, else the best fuzzy window (ratio >= 0.6)"""
    import difflib
    P = norm(phrase); T = [t[0] for t in TK]; n = len(P); hits = [i for i in range(len(T) - n + 1) if T[i:i + n] == P]
    if hits:
        i = hits[-1] if last else hits[0]; return TK[i][1], TK[i + n - 1][2], 1.0
    best = (0, None)
    rng = range(len(T) - n, -1, -1) if last else range(len(T) - n + 1)
    for i in rng:
        r = difflib.SequenceMatcher(None, P, T[i:i + n], autojunk=False).ratio()
        if r > best[0] + 1e-9: best = (r, i)
    if best[1] is None or best[0] < 0.6: raise SystemExit(f"phrase not found in the source narration: {phrase!r}")
    i = best[1]; return TK[i][1], TK[i + n - 1][2], round(best[0], 3)


def align_said(said, heard):
    """time each word of `said` with the clip's ASR words -> [[word, t0, t1]], plus a match score"""
    import difflib
    SW = said.split(); E = []
    for k, w in enumerate(SW):
        for t in norm(w): E.append((t, k))
    HT = tokens(heard)
    sm = difflib.SequenceMatcher(None, [e[0] for e in E], [h[0] for h in HT], autojunk=False)
    t0 = [None] * len(SW); t1 = [None] * len(SW); hit = 0
    for b in sm.get_matching_blocks():
        for j in range(b.size):
            k = E[b.a + j][1]; h = HT[b.b + j]; hit += 1
            t0[k] = h[1] if t0[k] is None else min(t0[k], h[1]); t1[k] = h[2] if t1[k] is None else max(t1[k], h[2])
    known = [k for k in range(len(SW)) if t0[k] is not None]
    if not known:      # nothing matched: spread the words over the heard span
        a = heard[0][1] if heard else 0.2; b = heard[-1][2] if heard else 2.5
        return [[w, a + (b - a) * k / len(SW), a + (b - a) * (k + 1) / len(SW)] for k, w in enumerate(SW)], 0.0
    k = 0
    while k < len(SW):           # unmatched runs: spread evenly between the matched neighbours
        if t0[k] is not None: k += 1; continue
        j = k
        while j < len(SW) and t0[j] is None: j += 1
        a = t1[k - 1] if k > 0 else max(0.0, t0[j] - 0.35 * (j - k))
        b = t0[j] if j < len(SW) else t1[k - 1] + 0.35 * (j - k)
        for q in range(k, j):
            t0[q] = a + (b - a) * (q - k) / (j - k); t1[q] = a + (b - a) * (q - k + 1) / (j - k)
        k = j
    return [[SW[k], round(t0[k], 3), round(t1[k], 3)] for k in range(len(SW))], round(hit / max(1, len(E)), 3)


def chunks(words, shift):
    """subtitle chunks [text, t0, t1] in reel time: max 6 words, break after punctuation"""
    CH, sent = [], []
    sents, cur = [], []
    for w in words:                      # sentences (break after . ? ! and after a comma once 3+ words), then halves of 6+
        cur.append(w)
        if re.search(r"[.?!]$", w[0]) or (w[0].endswith(",") and len(cur) >= 3): sents.append(cur); cur = []
    if cur: sents.append(cur)
    for s_ in sents:
        parts = [s_] if len(s_) <= 5 else [s_[:(len(s_) + 1) // 2], s_[(len(s_) + 1) // 2:]]
        for p_ in parts: CH.append([" ".join(x[0] for x in p_), p_[0][1] - shift, p_[-1][2] - shift])
    for c in CH: c[0] = c[0].rstrip(",;:")
    for j in range(len(CH) - 1): CH[j][2] = CH[j + 1][1] - 0.02
    return CH


def read_frames(src, t0, dur, vf):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", src, "-t", f"{dur:.3f}", "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3); return np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)


def audio_of(path, t0=0.0, dur=None):
    a = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-i", path] + (["-t", f"{dur:.3f}"] if dur else [])
    raw = subprocess.run(a + ["-vn", "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.astype(np.float64)


def fade(x, ms=12):
    n = int(SR * ms / 1000); x = x.copy()
    if x.shape[1] > 2 * n: x[:, :n] *= np.linspace(0, 1, n)[None]; x[:, -n:] *= np.linspace(1, 0, n)[None]
    return x


def gain_to(x, target):
    return x * 10 ** ((target - lufs_of(x)) / 20)


def run(S):
    D = S["data"]; work = workdir(S); M = S.get("music") or {}; M = {"bed": M} if isinstance(M, str) else M
    F0 = D.get("follow")                 # Noam 22 Nov: follow line in the reel clone (house-chained TTS take), not his recording
    if isinstance(F0, dict) and not F0.get("own_voice") and F0.get("wav", OWN_FOLLOW[0]) in OWN_FOLLOW:
        F0["wav"] = "ref/follow_line_tts_chain.wav"; F0["text"] = FOLLOW_TEXT
    LUFS = float(M.get("lufs", -16))
    clip = os.path.join(REPO, D["clip"]); src = os.path.join(REPO, D["source"])
    need = [D["clip"], D["source"], (D.get("follow") or {}).get("wav", "ref/follow_line_tts_chain.wav")]
    miss = [f for f in need if not os.path.exists(os.path.join(REPO, f))]
    if miss:   # the CI checkout is sparse: pull the clip, the source reel and the follow line in
        subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", *["/" + f for f in miss]], check=False)
    for p in (clip, src):
        if not os.path.exists(p): raise SystemExit(f"face_hook: file not found: {os.path.relpath(p, REPO)}")
    said = D["said"].strip(); L = dur_of(src); CL = dur_of(clip)
    # 1. time Noam's words in the clip
    heard = asr_words(clip, 0.0, None, prompt=said)
    words, wscore = align_said(said, heard)
    if wscore < 0.8:                     # the prompt can make whisper drop words: try without it and keep the better timing
        h2 = asr_words(clip, 0.0, None); w2, s2 = align_said(said, h2)
        if s2 > wscore: heard, words, wscore = h2, w2, s2
    cin = float(D.get("clip_in", max(0.0, words[0][1] - 0.08)))
    cout = float(D.get("clip_out", min(CL, words[-1][2] + 0.14)))
    fd = round(round((cout - cin) * FPS) / FPS, 4)
    CH = chunks(words, cin)
    log(f"clip {D['clip']}: heard '{' '.join(w[0] for w in heard)}' (match {wscore}), trim {cin:.2f} to {cout:.2f} s")
    # 2. resolve the cut and the follow line in the source
    F = D.get("follow") or {}; fmode = F.get("mode", "insert")
    speech = D.get("source_speech", True)          # false for music-only sources
    need_asr = speech or bool(D.get("cut_after")) or fmode == "replace" or (fmode == "insert" and F.get("at") is None)
    STK = tokens(asr_words(src)) if need_asr else []
    if D.get("cut_after"):
        _, e, r = find_phrase(STK, D["cut_after"]); cut = e + float(D.get("cut_pad", 0.06)); cut_how = f"after '{D['cut_after']}' ({r})"
    else:
        cut = float(D.get("cut", 0.0)); cut_how = "fixed"
    if speech and D.get("snap", True) and STK:      # land the cut on the next spoken word so the old hook's subtitle is gone
        nxt = [t[1] for t in STK if t[1] >= cut - 0.02]
        if nxt and nxt[0] - cut <= 1.2 and nxt[0] - 0.03 > cut:
            cut_how += f", snapped from {cut:.2f} to the next word"; cut = nxt[0] - 0.03
    fwav = os.path.join(REPO, F.get("wav", "ref/follow_line_tts_chain.wav"))
    fa = fb = None
    if fmode == "replace":
        fa, _, r1 = find_phrase(STK, F["replace"], last=True)
        _, fb, r2 = find_phrase(STK, F.get("replace_end", "for more"), last=True)
        fa -= 0.05; fb += 0.08; fhow = f"replaced TTS follow line {fa:.2f} to {fb:.2f} s ({r1}, {r2})"
    elif fmode == "insert":
        if F.get("at") is not None: fa = float(F["at"]); fhow = f"inserted at {fa:.2f} s (fixed)"
        else:
            a_, _, r1 = find_phrase(STK, F["before"], last=True); fa = a_ - float(F.get("lead", 0.12)); fhow = f"inserted before '{F['before']}' at {fa:.2f} s ({r1})"
        fb = fa
    else: fhow = "none"
    if fa is not None and fa <= cut + 1: raise SystemExit("follow line falls before the cut")
    # 3. face part: frames
    vf = ("hflip," if D.get("flip", True) else "") + f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}"
    FR = read_frames(clip, cin, fd + 0.1, vf)
    hook_img = HK.make(S["_hook"], int(D.get("hook_size", 118)), maxw=960); HY = int(D.get("hook_y", 1240)); BR = D.get("brand", True)
    zoom = float(D.get("zoom", 0.04)); yy = np.arange(H, dtype=np.float32)[:, None, None] / H
    shade = (1 - 0.62 * np.clip((yy - 0.52) / 0.3, 0, 1)).astype(np.float32)
    import map_counter as MC

    def frame(t):
        k = min(len(FR) - 1, int(round(t * FPS))); z = 1.0 + zoom * min(1.0, t / max(fd, 0.1))
        Mz = cv2.getRotationMatrix2D((W / 2, H * 0.4), 0, z)
        bg = cv2.warpAffine(FR[k], Mz, (W, H), borderMode=cv2.BORDER_REFLECT).astype(np.float32) * shade
        cv = Canvas(bg)
        if BR: MC.draw_brand(cv, BR if isinstance(BR, str) else "Israel Facts", 1.0)
        out = HK.paste(cv.out().astype(np.float32), hook_img, 540, HY, 0.94 + 0.06 * eout(t / 0.2), 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); out = c2.out().astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    face_mp4 = f"{work}/face.mp4"; encode(frame, fd, None, face_mp4, loop=False)
    # 4. face audio: his real voice, cleaned and levelled on its own
    fx = audio_of(clip, cin, fd)
    fx = np.pad(fx, ((0, 0), (0, max(0, int(fd * SR) - fx.shape[1]))))[:, :int(fd * SR)]
    import soundfile as sf
    sf.write(f"{work}/face_raw.wav", fx.T.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{work}/face_raw.wav", "-af",
                    "highpass=f=80,lowpass=f=14000,afftdn=nr=8:nf=-45,acompressor=threshold=-22dB:ratio=2.5:attack=8:release=120:makeup=2",
                    "-ar", str(SR), "-ac", "2", f"{work}/face_proc.wav"], check=True)
    fx, _ = sf.read(f"{work}/face_proc.wav"); fx = fx.T[:, :int(fd * SR)]
    fx = np.pad(fx, ((0, 0), (0, max(0, int(fd * SR) - fx.shape[1]))))
    fx = fade(gain_to(fx, LUFS), 8)
    face_lufs = round(lufs_of(fx), 1)
    # 5. body: source from the cut, follow line in Noam's own voice
    segs_v = []   # (kind, a, b, extra)
    if fmode in ("insert", "replace"):
        fw, fsr = sf.read(fwav); fw = fw if fw.ndim == 1 else fw.mean(1)
        if fsr != SR: fw = np.interp(np.arange(int(len(fw) * SR / fsr)) / SR, np.arange(len(fw)) / fsr, fw)
        fw = np.stack([fw, fw]); fw = gain_to(fw, LUFS)
        fdur = round(round((fw.shape[1] / SR + 0.32) * FPS) / FPS, 4)
        segs_v = [("src", cut, fa), ("follow", fa, fb, fdur), ("src", fb, L)]
    else:
        segs_v = [("src", cut, L)]
    body_a = []
    for sg in segs_v:
        if sg[0] == "src":
            n = int(round((sg[2] - sg[1]) * FPS)) / FPS; x = audio_of(src, sg[1], n)
            x = np.pad(x, ((0, 0), (0, max(0, int(n * SR) - x.shape[1]))))[:, :int(n * SR)]; body_a.append(fade(x, 15))
        else:
            n = int(sg[3] * SR); bn = F.get("bed", M.get("bed", "doc_piano"))
            if bn == "none": b_ = np.zeros((2, n))
            else:
                b_ = bed(bn, sg[3] + 0.3)[:, :n]; b_ = np.pad(b_, ((0, 0), (0, max(0, n - b_.shape[1])))); b_ = gain_to(b_, LUFS - 14)
            v = np.zeros((2, n)); i0 = int(0.12 * SR); m = min(fw.shape[1], n - i0); v[:, i0:i0 + m] = fw[:, :m]
            body_a.append(fade(v + b_, 25))
    # follow part video
    fol_mp4 = None; fol_info = None
    for sg in segs_v:
        if sg[0] != "follow": continue
        fol_mp4 = f"{work}/follow.mp4"; fdur = sg[3]
        ft = F.get("text", "If you love Israeli facts, follow Noam Roz for more.")
        fwords, fscore = align_said(ft, asr_words(fwav, prompt=ft))
        FCH = chunks([[w, a_ + 0.12, b_ + 0.12] for w, a_, b_ in fwords], 0.0); FCH[-1][2] = fdur - 0.05
        tf = float(F["freeze_at"]) if F.get("freeze_at") is not None else (sg[1] - 0.04 if fmode == "insert" else (sg[1] + sg[2]) / 2)
        still = cv2.resize(grab(src, max(0.0, tf)), (W, H)).astype(np.float32)
        pill = F.get("pill", fmode == "insert"); dim = float(F.get("dim", 0.45 if pill else 0.0))
        ov = {"text": F.get("screen", "Like Israel facts?|Follow *@noamroz__*"), "y": int(F.get("y", 1350)), "size": 76, "cue": True}

        def ffr(t, still=still, ov=ov, fdur=fdur, FCH=FCH, pill=pill, dim=dim):
            a = min(1.0, t / 0.25) * min(1.0, (fdur - t) / 0.2 + 1e-4)
            cv = Canvas(still * (1 - dim * a))
            cv.rrect(0, 1530, 1080, 170, 0, "#0B1730", 0.96)          # hides the frozen source subtitle
            if pill: MC.draw_overlay(cv, ov, max(0.0, a), 0.92 + 0.08 * eout(min(1.0, t / 0.25)))
            draw_subs(cv, FCH, t, D.get("subs_y", 1612))
            return np.clip(cv.out(), 0, 255).astype(np.uint8)
        encode(ffr, fdur, None, fol_mp4, loop=False)
        fol_info = dict(freeze_at=round(tf, 3), dur=fdur, subs=FCH, match=fscore)
    BA = np.concatenate(body_a, axis=1); body_lufs = round(lufs_of(BA), 1)
    # 6. video join: face | src parts | follow, hard cuts, last 1 s of the body dissolves into the new frame 0
    f0 = f"{work}/f0.png"; cv2.imwrite(f0, frame(0.0))
    ins, flt, labs = ["-i", face_mp4], [f"[0:v]fps={FPS},setsar=1,format=yuv420p[v0]"], ["[v0]"]; k = 1
    for sg in segs_v:
        if sg[0] == "src":
            n = int(round((sg[2] - sg[1]) * FPS)) / FPS
            ins += ["-ss", f"{sg[1]:.3f}", "-t", f"{n:.3f}", "-i", src]
            flt.append(f"[{k}:v]scale={W}:{H},fps={FPS},setsar=1,format=yuv420p,trim=duration={n:.3f},setpts=PTS-STARTPTS[v{k}]")
        else:
            ins += ["-i", fol_mp4]; flt.append(f"[{k}:v]scale={W}:{H},fps={FPS},setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{k}]")
        labs.append(f"[v{k}]"); k += 1
    tot = math.floor((fd + BA.shape[1] / SR) * FPS) / FPS     # whole frames: audio never outlasts the video
    ins += ["-loop", "1", "-framerate", str(FPS), "-t", f"{tot:.3f}", "-i", f0]
    flt.append(f"{''.join(labs)}concat=n={len(labs)}:v=1:a=0[cat]")
    flt.append(f"[{k}:v]format=rgba,fade=in:st={tot - 1.0:.3f}:d=0.9:alpha=1[o];[cat][o]overlay=shortest=1:format=auto,tpad=stop_mode=clone:stop_duration=1,format=yuv420p[v]")
    vid = f"{work}/joined.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(flt), "-map", "[v]", "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-t", f"{tot + 0.1:.3f}", vid], check=True)
    # 7. audio join + loudness; his voice is matched to the body on the final file (one correction pass if they differ by > 0.7 LU)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    fx = gain_to(fx, body_lufs) if np.abs(fx).max() > 0 else fx
    n = int(tot * SR)

    def mixdown(fx_):
        x = np.concatenate([fx_, BA], axis=1); x = np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))))[:, :n]
        a_ = loudnorm(x, f"{work}/mix.m4a", LUFS)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", f"{work}/mix.m4a", "-map", "0:v", "-map", "1:a", "-c", "copy",
                        "-t", f"{tot:.3f}", "-movflags", "+faststart", final], check=True)
        sl = {nm: round(lufs_of(audio_of(final, a, b - a)), 1) for nm, (a, b) in (("face", (0, fd)), ("body", (fd, tot)))}
        return a_, sl
    aud, seg_l = mixdown(fx); passes = 1
    if abs(seg_l["face"] - seg_l["body"]) > 0.7:
        fx = fx * 10 ** ((seg_l["body"] - seg_l["face"]) / 20); aud, seg_l = mixdown(fx); passes = 2
    return dict(voice="Noam's own face clip (real voice) + source reel audio after the cut" + (" + Noam's own follow line" if fmode != "none" else ""),
                clip=D["clip"], said=said, clip_heard=" ".join(w[0] for w in heard), said_match=wscore, said_words=words,
                clip_trim=[round(cin, 3), round(cout, 3)], face_s=fd, source=D["source"], cut=round(cut, 3), cut_how=cut_how,
                follow=fhow, follow_part=fol_info, follow_text=F.get("text", "If you love Israeli facts, follow Noam Roz for more.") if fmode != "none" else None,
                levels_before_final=dict(face=face_lufs, body=body_lufs), levels_final=seg_l, level_passes=passes, mix=aud,
                body_kept_s=round(L - cut, 2), subs=CH)
