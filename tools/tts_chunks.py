#!/usr/bin/env python3
"""tools/tts_chunks.py voice/chunks/<id>.json : long narration in several Cartesia sonic-3 calls (one per chapter), made consistent.

Request: {"id", "voice_id", "lines": [...], "chunks": [[i0, i1], ...]}  (line ranges, i1 exclusive)
For every chunk: one SSE call (same body and headers as templates/core/common.py cartesia()), word count checked against the text.
Consistency: pitch median (pyin) and articulation rate per chunk; a chunk more than 1.8% off the median pitch or 7% off the
median rate is generated again (up to 3 extra tries, the closest take wins). A residual pitch error over 1.5% is corrected with
rubberband (formant preserved), a residual rate error over 6% with atempo (word times scaled). Loudness: every chunk scaled to the
same active-speech RMS. Output (picked up by spec-render as a cached voice, same hash as common.get_voice):
  voice/<id>_vo.wav, voice/<id>_vo_words.json, voice/<id>_vo.hash, voice/<id>_chunks_report.json
No key is printed."""
import os, sys, json, hashlib, base64, subprocess, time
import numpy as np, requests, soundfile as sf

URL = "https://api.cartesia.ai/tts/sse"; VER = "2025-04-16"; SR = 44100


def tts(text, vid):
    key = os.environ.get("CARTESIA_API_KEY", "").strip()
    if not key: raise SystemExit("CARTESIA_API_KEY not set")
    body = {"model_id": "sonic-3", "transcript": text, "voice": {"mode": "id", "id": vid},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SR}, "add_timestamps": True, "language": "en"}
    r = requests.post(URL, json=body, timeout=300, stream=True, headers={"X-API-Key": key, "Cartesia-Version": VER, "Content-Type": "application/json"})
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
    return y, words


def pitch_median(y):
    import librosa
    z = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0, v, _ = librosa.pyin(z, fmin=110, fmax=420, sr=16000, frame_length=1024)
    f = f0[v & ~np.isnan(f0)]
    return float(np.median(f)) if len(f) else 0.0


def rate(words):
    """articulation rate: words per second of speech, pauses over 0.2 s left out"""
    if len(words) < 2: return 0.0
    span = words[-1]["end"] - words[0]["start"]
    gaps = sum(max(0.0, words[k + 1]["start"] - words[k]["end"] - 0.2) for k in range(len(words) - 1))
    return len(words) / max(0.5, span - gaps)


def active_rms(y):
    n = int(0.05 * SR); fr = y[:len(y) // n * n].reshape(-1, n); r = np.sqrt((fr ** 2).mean(1) + 1e-12)
    act = r[r > r.max() * 10 ** (-35 / 20)]
    return float(np.sqrt((act ** 2).mean()))


def ff(y, af):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", "-af", af, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                       input=y.astype(np.float32).tobytes(), capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode()[:300])
    return np.frombuffer(p.stdout, np.float32).copy()


def main(req_path):
    R = json.load(open(req_path)); sid = R["id"]; vid = R["voice_id"]; lines = [l.strip() for l in R["lines"]]
    h = hashlib.sha1("\n".join(lines + [f"voice:{vid}"]).encode()).hexdigest()[:12]
    pre = os.path.join("voice", f"{sid}_vo")
    if os.path.exists(pre + ".hash") and open(pre + ".hash").read().strip() == h and not R.get("force"):
        print(sid, "voice up to date"); return
    takes = []
    for ci, (i0, i1) in enumerate(R["chunks"]):
        text = " ".join(lines[i0:i1]); nw = len(text.split()); cand = []
        for tr in range(3):
            try:
                y, w = tts(text, vid)
                if len(w) != nw: print(f"chunk {ci}: {len(w)} words vs {nw} in text, retry"); continue
                cand.append(dict(y=y, w=w, p=pitch_median(y), r=rate(w))); break
            except Exception as e: print(f"chunk {ci} try {tr}: {e}"); time.sleep(3)
        if not cand: raise SystemExit(f"chunk {ci} failed")
        takes.append(cand); print(f"chunk {ci}: {nw} words, pitch {cand[0]['p']:.1f} Hz, rate {cand[0]['r']:.2f} w/s", flush=True)
    for rnd in range(3):
        P = np.median([min(c, key=lambda x: x.get("score", 0))["p"] for c in takes]) if rnd else np.median([c[0]["p"] for c in takes])
        Q = np.median([c[0]["r"] for c in takes])
        bad = []
        for ci, c in enumerate(takes):
            for x in c: x["score"] = abs(x["p"] / P - 1) / 0.018 + abs(x["r"] / Q - 1) / 0.07
            best = min(c, key=lambda x: x["score"])
            if abs(best["p"] / P - 1) > 0.018 or abs(best["r"] / Q - 1) > 0.09: bad.append(ci)
        print(f"round {rnd}: median pitch {P:.1f} Hz, rate {Q:.2f} w/s, off: {bad}", flush=True)
        if not bad: break
        for ci in bad:
            i0, i1 = R["chunks"][ci]; text = " ".join(lines[i0:i1])
            try:
                y, w = tts(text, vid)
                if len(w) == len(text.split()): takes[ci].append(dict(y=y, w=w, p=pitch_median(y), r=rate(w)))
            except Exception as e: print(f"retake {ci}: {e}")
    for c in takes:
        for x in c: x.setdefault("score", 99.0)
    P = float(np.median([min(c, key=lambda x: x["score"])["p"] for c in takes])); Q = float(np.median([min(c, key=lambda x: x["score"])["r"] for c in takes]))
    rep = dict(id=sid, voice_id=vid, hash=h, median_pitch_hz=round(P, 2), median_rate_wps=round(Q, 3), chunks=[])
    outs, target = [], None
    for ci, c in enumerate(takes):
        x = min(c, key=lambda q: abs(q["p"] / P - 1) / 0.018 + abs(q["r"] / Q - 1) / 0.07); y = x["y"]; w = [dict(d) for d in x["w"]]
        fix = []
        dp = x["p"] / P - 1
        if abs(dp) > 0.015:
            y = ff(y, f"rubberband=pitch={1 / (1 + dp):.5f}:formant=preserved"); fix.append(f"pitch x{1 / (1 + dp):.4f}")
        dr = x["r"] / Q - 1
        if abs(dr) > 0.05:      # atempo > 1 speeds a slow chunk up; word times shrink by the same factor
            at = Q / x["r"]; y = ff(y, f"atempo={at:.5f}"); fix.append(f"atempo {at:.4f}")
            for d in w: d["start"] /= at; d["end"] /= at
        os.makedirs(os.path.join("voice", "raw_" + sid), exist_ok=True)
        sf.write(os.path.join("voice", "raw_" + sid, f"c{ci:02d}.flac"), x["y"], SR); json.dump(x["w"], open(os.path.join("voice", "raw_" + sid, f"c{ci:02d}.json"), "w"))
        p2 = pitch_median(y) if fix else x["p"]
        rr = active_rms(y); target = target or rr; y = y * (target / rr)
        outs.append((y, w))
        rep["chunks"].append(dict(chunk=ci, lines=R["chunks"][ci], takes=len(c), pitch_hz=round(p2, 2), pitch_dev_pct=round((p2 / P - 1) * 100, 2),
                                  rate_wps=round(x["r"], 3), rate_dev_pct=round((x["r"] / Q - 1) * 100, 2), rate_after_wps=round(rate(w), 3), fixes=fix, dur_s=round(len(y) / SR, 2)))
    pk = max(np.abs(y).max() for y, _ in outs); allw, buf, t = [], [], 0.0; sil = np.zeros(int(0.5 * SR), np.float32)
    for y, w in outs:
        y = y / pk * 0.89
        for d in w: allw.append({"word": d["word"], "start": round(d["start"] + t, 4), "end": round(d["end"] + t, 4)})
        buf += [y, sil]; t += (len(y) + len(sil)) / SR
    total = sum(len(l.split()) for l in lines)
    if len(allw) != total: raise SystemExit(f"word count {len(allw)} vs script {total}")
    sf.write(pre + ".wav", np.concatenate(buf), SR, subtype="PCM_16")
    json.dump(allw, open(pre + "_words.json", "w"), indent=0)
    open(pre + ".hash", "w").write(h)
    devs = [abs(c["pitch_dev_pct"]) for c in rep["chunks"]]; rep["max_pitch_dev_pct"] = max(devs); rep["total_s"] = round(t, 1)
    json.dump(rep, open(os.path.join("voice", f"{sid}_chunks_report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("median_pitch_hz", "median_rate_wps", "max_pitch_dev_pct", "total_s")}))




def reassemble(req_path, target_rate):
    """rebuild voice/<id>_vo.* from the stored raw takes (voice/raw_<id>/cNN.flac + .json), no new TTS: one rubberband pass per chunk
    sets pitch to the median and tempo to target_rate words per second (formant preserved), then loudness match and concatenation."""
    R = json.load(open(req_path)); sid = R["id"]; vid = R["voice_id"]; lines = [l.strip() for l in R["lines"]]
    h = hashlib.sha1("\n".join(lines + [f"voice:{vid}"]).encode()).hexdigest()[:12]; raw = os.path.join("voice", "raw_" + sid)
    takes = []
    for ci in range(len(R["chunks"])):
        y, _ = sf.read(os.path.join(raw, f"c{ci:02d}.flac"), dtype="float32"); w = json.load(open(os.path.join(raw, f"c{ci:02d}.json")))
        takes.append(dict(y=y, w=w, p=pitch_median(y), r=rate(w)))
    P = float(np.median([x["p"] for x in takes])); rep = dict(id=sid, voice_id=vid, hash=h, mode="reassembled from raw takes", median_pitch_hz=round(P, 2),
                                                               target_rate_wps=target_rate, chunks=[])
    outs, target = [], None
    for ci, x in enumerate(takes):
        tempo = target_rate / x["r"]; pitch = P / x["p"]
        for it in range(4):       # pitch measured after the tempo change, corrected until within 1.2% of the median
            y = ff(x["y"], f"rubberband=pitch={pitch:.5f}:formant=preserved,rubberband=tempo={tempo:.5f}")   # one instance with both ignores pitch
            p2 = pitch_median(y)
            if abs(p2 / P - 1) <= 0.012: break
            pitch *= P / p2
        w = [dict(d, start=d["start"] / tempo, end=d["end"] / tempo) for d in x["w"]]
        rr = active_rms(y); target = target or rr; y = y * (target / rr); outs.append((y, w))
        rep["chunks"].append(dict(chunk=ci, lines=R["chunks"][ci], raw_pitch_hz=round(x["p"], 2), raw_rate_wps=round(x["r"], 3), tempo=round(tempo, 4),
                                  pitch_shift=round(pitch, 4), pitch_hz=round(p2, 2), rate_wps=round(rate(w), 3), dur_s=round(len(y) / SR, 2)))
    Pm = float(np.median([c["pitch_hz"] for c in rep["chunks"]]))
    for c in rep["chunks"]: c["pitch_dev_pct"] = round((c["pitch_hz"] / Pm - 1) * 100, 2)
    pk = max(np.abs(y).max() for y, _ in outs); allw, buf, t = [], [], 0.0; sil = np.zeros(int(0.5 * SR), np.float32)
    for y, w in outs:
        y = (y / pk * 0.89).astype(np.float32)
        for d in w: allw.append({"word": d["word"], "start": round(d["start"] + t, 4), "end": round(d["end"] + t, 4)})
        buf += [y, sil]; t += (len(y) + len(sil)) / SR
    if len(allw) != sum(len(l.split()) for l in lines): raise SystemExit("word count mismatch")
    pre = os.path.join("voice", f"{sid}_vo")
    sf.write(pre + ".wav", np.concatenate(buf), SR, subtype="PCM_16"); json.dump(allw, open(pre + "_words.json", "w"), indent=0); open(pre + ".hash", "w").write(h)
    rep["median_after_hz"] = round(Pm, 2); rep["max_pitch_dev_pct"] = max(abs(c["pitch_dev_pct"]) for c in rep["chunks"]); rep["total_s"] = round(t, 1)
    json.dump(rep, open(os.path.join("voice", f"{sid}_chunks_report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("median_after_hz", "max_pitch_dev_pct", "total_s")}))


if __name__ == "__main__":
    if sys.argv[1] == "--reassemble": reassemble(sys.argv[3], float(sys.argv[2]))
    else:
        for p in sys.argv[1:]: main(p)
