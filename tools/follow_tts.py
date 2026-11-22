#!/usr/bin/env python3
"""tools/follow_tts.py ref/follow_tts/job.json : the @noamroz__ follow line voiced by the reel clone, in context (Noam, 22 Nov 2026).

Noam 22 Nov: his own pasted recording of the follow line stood out at the end of the reels. So the line is now voiced by the
same Cartesia voice, model and pronunciation pass as the narration (templates/core/common.py cartesia() + core/pron.py), IN
CONTEXT: each take is one Cartesia call of "<preceding line> <follow line> <lead-in line>", and the follow line is cut out
by the word timestamps, so its intonation is a mid-ending sentence like in a reel, not a standalone read.

Job: {"follow": "If you love Israeli facts, follow Noam Roz for more.", "takes": [{"before": "...", "after": "..."}, ...],
      "ref": "voice/<id>_vo.wav"}   (ref: a reel narration to compare pitch, pitch variety and rate against)
For each take: ref/follow_tts/take<k>.wav (raw Cartesia 44.1 kHz cut, no house chain: the render applies it, like every
narration line) + take<k>_words.json (one entry per follow line word, seconds inside the cut), and in ref/follow_tts/report.json:
whisper small.en transcript of the house-chained cut, name check (Noam + Roz/Raz), pitch median / variety / rate vs the ref.
No key is printed. Nothing here posts anything."""
import os, sys, json, re, subprocess
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "templates", "core"))
import pron as PR

OUT = os.path.join(REPO, "ref", "follow_tts")


def cartesia(text, pre):
    # same call as templates/core/common.py cartesia() (copied: importing common pulls in cv2 and the render stack)
    import requests, base64, wave
    key = os.environ.get("CARTESIA_API_KEY", "").strip()
    if not key: raise SystemExit("CARTESIA_API_KEY not set")
    V = json.load(open(os.path.join(REPO, "templates", "voices.json")))
    vid = V.get("en", {}).get("id") if isinstance(V.get("en"), dict) else V.get("en")
    vid = vid or "cd06f613-0c92-4347-9b94-b34cb4a0239f"
    body = {"model_id": "sonic-3", "transcript": text, "voice": {"mode": "id", "id": vid},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100},
            "add_timestamps": True, "language": "en"}
    r = requests.post("https://api.cartesia.ai/tts/sse", json=body, timeout=300, stream=True,
                      headers={"X-API-Key": key, "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"})
    if r.status_code != 200: raise RuntimeError(f"Cartesia HTTP {r.status_code}: {r.text[:200]}")
    pcm, words = bytearray(), []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"): continue
        try: ev = json.loads(line[5:].strip())
        except Exception: continue
        if ev.get("type") == "error": raise RuntimeError(f"Cartesia error: {str(ev)[:200]}")
        if ev.get("type") == "chunk" and ev.get("data"): pcm += base64.b64decode(ev["data"])
        wt = ev.get("word_timestamps")
        if wt:
            for w, s, e in zip(wt["words"], wt["start"], wt["end"]): words.append({"word": w, "start": s, "end": e})
    y = np.frombuffer(bytes(pcm), np.int16).astype(np.float32) / 32768.0
    return y, words, vid


def chain(y, sr=44100, tempo=1.1):
    """the reels' house chain (common.house_chain): pitch 0.9715 formant preserved, bass +2 dB at 135 Hz, atempo"""
    af = f"rubberband=pitch=0.9715:formant=preserved,bass=g=2:f=135,atempo={tempo}"
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-af", af,
                        "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], input=y.astype(np.float32).tobytes(), capture_output=True)
    if p.returncode:
        af = f"asetrate={sr}*0.9715,aresample={sr},atempo={tempo / 0.9715:.5f},bass=g=2:f=135"
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-af", af,
                            "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], input=y.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def stats(y, words, sr=44100):
    import librosa
    z = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, v, _ = librosa.pyin(z, fmin=70, fmax=400, sr=16000, frame_length=1024)
    f = f0[v & ~np.isnan(f0)]
    st = 12 * np.log2(f / np.median(f)) if len(f) else np.array([0.0])
    span = words[-1]["end"] - words[0]["start"]
    gaps = sum(max(0.0, words[k + 1]["start"] - words[k]["end"] - 0.2) for k in range(len(words) - 1))
    return dict(pitch_hz=round(float(np.median(f)), 1) if len(f) else 0.0,
                pitch_var_st=round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
                rate_wps=round(len(words) / max(0.3, span - gaps), 2))


def main(job_path):
    J = json.load(open(job_path)); os.makedirs(OUT, exist_ok=True)
    follow = J["follow"].strip(); fw = follow.split()
    from faster_whisper import WhisperModel
    W = WhisperModel("small.en", device="cpu", compute_type="int8")
    ref = None
    if J.get("ref") and os.path.exists(os.path.join(REPO, J["ref"])):
        ry, rsr = sf.read(os.path.join(REPO, J["ref"]), dtype="float32")
        rw = json.load(open(os.path.join(REPO, J["ref"])[:-4] + "_words.json"))
        if ry.ndim > 1: ry = ry.mean(1)
        ref = stats(ry, rw, rsr)
    tag = J.get("tag", "")   # e.g. "r3_": files r3_take<k>.* and report_r3.json, so earlier takes are not overwritten
    rep = dict(follow=follow, ref=J.get("ref"), ref_stats=ref, takes=[], tag=tag)
    for k, T in enumerate(J["takes"], 1):
        # a take may carry "follow_tts": the raw Cartesia text for the follow part (phonemes, <break time=".."/>,
        # <speed ratio=".."/>); the follow span is then found by Cartesia word counts of the context around it
        bp = PR.prepare(T["before"], "en") if T.get("before") else []
        ap = PR.prepare(T["after"], "en") if T.get("after") else []
        fp = PR.prepare(follow, "en"); ftts = T.get("follow_tts") or PR.tts_text(fp)
        tts_txt = " ".join(x for x in (PR.tts_text(bp), ftts, PR.tts_text(ap)) if x)
        text = " ".join(x for x in (T.get("before", ""), follow, T.get("after", "")) if x).strip()
        pairs = bp + fp + ap
        y, cw, vid = cartesia(tts_txt, None)
        nbc = len(PR.tts_text(bp).split()) if bp else 0; nac = len(PR.tts_text(ap).split()) if ap else 0
        exact = len(cw) == len(PR.tts_text(pairs).split())
        span = cw[nbc:len(cw) - nac] if nac else cw[nbc:]
        if len(span) == len(fw): seg = [dict(start=s["start"], end=s["end"]) for s in span]
        else:                 # Cartesia split the follow part differently: spread the 10 words over its span by length
            t0, t1 = span[0]["start"], span[-1]["end"]; w8 = np.array([len(re.sub(r"[^\w]", "", w)) + 2 for w in fw], float)
            cum = np.concatenate([[0], np.cumsum(w8)]) / w8.sum()
            seg = [dict(start=t0 + (t1 - t0) * cum[j], end=t0 + (t1 - t0) * cum[j + 1] - 0.02) for j in range(len(fw))]
        g = [dict(word=w["word"], start=w["start"], end=w["end"]) for w in cw]
        prev_end = cw[nbc - 1]["end"] if nbc else 0.0
        next_start = cw[len(cw) - nac]["start"] if nac else len(y) / 44100
        a = max(prev_end + (seg[0]["start"] - prev_end) * 0.4, seg[0]["start"] - 0.05, 0.0)
        b = min(seg[-1]["end"] + 0.10, seg[-1]["end"] + (next_start - seg[-1]["end"]) * 0.6, len(y) / 44100)
        cut = y[int(a * 44100):int(b * 44100)].copy(); fr = int(0.01 * 44100)
        cut[:fr] *= np.linspace(0, 1, fr); cut[-fr:] *= np.linspace(1, 0, fr)
        words = [dict(word=w, start=round(s["start"] - a, 3), end=round(s["end"] - a, 3)) for w, s in zip(fw, seg)]
        sf.write(f"{OUT}/{tag}take{k}.wav", cut, 44100, subtype="PCM_16")
        json.dump(words, open(f"{OUT}/{tag}take{k}_words.json", "w"), indent=0)
        sf.write(f"{OUT}/{tag}take{k}_context.wav", y, 44100, subtype="PCM_16")
        json.dump(g, open(f"{OUT}/{tag}take{k}_context_words.json", "w"), indent=0)
        h = chain(cut)
        segs, _ = W.transcribe(h, language="en", beam_size=5)       # 16 kHz array (no file decode)
        tx = " ".join(s.text.strip() for s in segs).strip()
        hc = chain(y)
        segs, _ = W.transcribe(hc, language="en", beam_size=5)
        txc = " ".join(s.text.strip() for s in segs).strip()
        name_ok = bool(re.search(r"\bnoam\b", tx, re.I)) and bool(re.search(r"\b(roz|raz|ra's|rahz|rose?s?)\b", tx, re.I))
        name_exact = bool(re.search(r"\bnoam\s+r[oa]z\b", tx, re.I))
        st = stats(cut, words)
        d = None
        if ref:
            d = round(abs(st["pitch_hz"] / ref["pitch_hz"] - 1) * 100 + abs(st["rate_wps"] / ref["rate_wps"] - 1) * 100
                      + abs(st["pitch_var_st"] - ref["pitch_var_st"]) * 5, 2)
        r = dict(take=k, name=T.get("name", ""), file=f"ref/follow_tts/{tag}take{k}.wav", context=text, tts_text=tts_txt, exact_words=exact,
                 secs=round(len(cut) / 44100, 2), transcript=tx, transcript_context=txc, name_ok=name_ok, name_exact=name_exact,
                 stats=st, dist_to_ref=d, voice=vid)
        rep["takes"].append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    ok = [t for t in rep["takes"] if t["name_ok"]] or rep["takes"]
    best = sorted(ok, key=lambda t: (not t["name_exact"], t["dist_to_ref"] if t["dist_to_ref"] is not None else 0))[0]
    rep["best"] = best["take"]
    json.dump(rep, open(f"{OUT}/report{'_' + tag.strip('_') if tag else ''}.json", "w"), indent=1, ensure_ascii=False)
    print("best take:", best["take"], best["transcript"])


if __name__ == "__main__":
    main(sys.argv[1])
