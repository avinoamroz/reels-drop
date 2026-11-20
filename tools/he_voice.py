#!/usr/bin/env python3
"""tools/he_voice.py: Hebrew voice jobs for the he-voice workflow (20 Nov 2026). Reads hevoice/jobs.json, a list of
{"do": ...} jobs, writes results to hevoice/*.json (+ wav cuts). Never prints the API key. Nothing is posted.
  {"do": "voices"}                                    Cartesia voices that speak Hebrew -> hevoice/voices_he.json
  {"do": "nakdan", "text": "..."}                     Dicta Nakdan test (raw answer + vowelised text) -> hevoice/nakdan_test.json
  {"do": "fetch", "name": "x", "drive": "<file id>"}  public Drive file -> audio, Hebrew transcript with segment times,
                                                      noise floor -> hevoice/x_tx.json (audio kept for later jobs)
  {"do": "cut", "name": "x", "from": [[a, b], ...], "out": "ref/noam_he_sample.wav"}  joined clean sample, 44.1 kHz mono,
                                                      loudness -16 LUFS, + its transcript -> hevoice/<out name>.json
  {"do": "audition", "text": "...", "voices": ["id", ...], "nikud": true}   TTS each voice (pron pass), Hebrew ASR score
                                                      -> hevoice/audition.json (+ hevoice/aud_<n>.wav)
  {"do": "clone", "clip": "ref/noam_he_sample.wav", "name": "Noam Roz - Hebrew", "language": "he"}   Cartesia clone,
                                                      once (skipped if hevoice/clone.json has an id) -> hevoice/clone.json"""
import os, sys, json, subprocess, re, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(REPO, "templates", "core"))
OUT = os.path.join(REPO, "hevoice"); KEY = os.environ.get("CARTESIA_API_KEY", "").strip()
API = "https://api.cartesia.ai"; HDR = {"X-API-Key": KEY, "Cartesia-Version": "2025-04-16"}
HE_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
_M = {}


def model():
    if "he" not in _M:
        from faster_whisper import WhisperModel
        _M["he"] = WhisperModel(HE_MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    return _M["he"]


def wav16(src):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype("float32") / 32768.0


def noise_floor(x):
    import numpy as np
    fr = 800; e = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2) + 1e-12) for i in range(0, len(x) - fr, fr)])
    db = 20 * np.log10(e + 1e-9); return round(float(np.percentile(db, 8)), 1), round(float(np.percentile(db, 60)), 1)


def he_norm(s):
    s = re.sub(r"[֑-ׇ]", "", s).replace("־", " ").replace("-", " ")
    return re.sub(r"[^א-ת0-9 ]+", " ", s).split()


def save(name, obj): json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, ensure_ascii=False); print("wrote", name)


def j_voices(J):
    import requests
    allv, after = [], None
    for _ in range(30):
        p = {"limit": 100}
        if after: p["starting_after"] = after
        r = requests.get(f"{API}/voices", headers=HDR, params=p, timeout=60)
        if r.status_code != 200: save("voices_he.json", {"error": r.status_code, "body": r.text[:300]}); return
        js = r.json(); data = js.get("data", js) if isinstance(js, dict) else js
        allv += data
        if not (isinstance(js, dict) and js.get("has_more")) or not data: break
        after = data[-1]["id"]
    he = [dict(id=v["id"], name=v.get("name"), description=(v.get("description") or "")[:160], gender=v.get("gender"),
               is_owner=v.get("is_owner"), language=v.get("language")) for v in allv if v.get("language") == "he"]
    mine = [dict(id=v["id"], name=v.get("name"), language=v.get("language")) for v in allv if v.get("is_owner")]
    save("voices_he.json", dict(total=len(allv), hebrew=he, mine=mine))


def j_nakdan(J):
    import pron as PR, requests
    raw = None
    for u in PR.NAKDAN_URLS:
        try:
            r = requests.post(u, json={"task": "nakdan", "data": J["text"], "genre": "modern", "addmorph": False, "keepqq": False,
                                       "nodageshdefmem": False, "patachma": False, "keepmetagim": False, "useTokenization": True}, timeout=40)
            raw = dict(url=u, status=r.status_code, body=r.text[:1500]); break
        except Exception as e: raw = dict(url=u, error=str(e)[:200])
    nt = []
    pairs = PR.prepare(J["text"], "he", notes=nt)
    save("nakdan_test.json", dict(raw=raw, vowelised=PR.tts_text(pairs), notes=nt))


def j_fetch(J):
    import requests
    os.makedirs("/tmp/hv", exist_ok=True); src = f"/tmp/hv/{J['name']}.bin"
    u = f"https://drive.usercontent.google.com/download?id={J['drive']}&export=download&confirm=t"
    with requests.get(u, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            for ch in r.iter_content(1 << 20): f.write(ch)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "44100", f"/tmp/hv/{J['name']}.wav"], check=True)
    x = wav16(f"/tmp/hv/{J['name']}.wav")
    segs, info = model().transcribe(x, language="he", beam_size=5, word_timestamps=False, vad_filter=True)
    S = [dict(a=round(s.start, 2), b=round(s.end, 2), t=s.text.strip()) for s in segs]
    nf = noise_floor(x)
    save(f"{J['name']}_tx.json", dict(drive=J["drive"], secs=round(len(x) / 16000, 1), noise_floor_db=nf[0], speech_db=nf[1], segments=S))


def j_cut(J):
    import numpy as np, soundfile as sf
    src = f"/tmp/hv/{J['name']}.wav"
    if not os.path.exists(src): j_fetch(dict(name=J["name"], drive=J["drive"]))
    y, sr = sf.read(src, dtype="float32")
    parts = []
    for a, b in J["from"]:
        p = y[int(a * sr):int(b * sr)].copy(); f = int(0.015 * sr); p[:f] *= np.linspace(0, 1, f); p[-f:] *= np.linspace(1, 0, f)
        parts += [p, np.zeros(int(0.25 * sr), "float32")]
    z = np.concatenate(parts[:-1]); out = os.path.join(REPO, J["out"]); os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = "/tmp/hv/cut.wav"; sf.write(tmp, z, sr)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "1", out], check=True)
    x = wav16(out); segs, _ = model().transcribe(x, language="he", beam_size=5)
    nf = noise_floor(x)
    save(os.path.basename(out).rsplit(".", 1)[0] + ".json", dict(file=J["out"], secs=round(len(x) / 16000, 1), noise_floor_db=nf[0], speech_db=nf[1],
                                                                 text=" ".join(s.text.strip() for s in segs)))


class CM:
    @staticmethod
    def cartesia(text, pre, lang, vid):
        """same body as templates/core/common.py cartesia(), bytes endpoint -> pre.wav"""
        import requests, wave
        r = requests.post(f"{API}/tts/bytes", headers=dict(HDR, **{"Content-Type": "application/json"}), timeout=300,
                          json={"model_id": "sonic-3", "transcript": text, "voice": {"mode": "id", "id": vid}, "language": lang,
                                "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100}})
        if r.status_code != 200: raise RuntimeError(f"Cartesia HTTP {r.status_code}: {r.text[:300]}")
        with wave.open(pre + ".wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(r.content)
        return len(r.content) / 88200, len(text.split())


def j_join(J):
    import numpy as np, soundfile as sf
    parts = []
    for f in J["parts"]:
        y, sr = sf.read(os.path.join(REPO, f), dtype="float32"); parts += [y, np.zeros(int(0.3 * sr), "float32")]
    out = os.path.join(REPO, J["out"]); sf.write(out, np.concatenate(parts[:-1]), sr)
    save(os.path.basename(out).rsplit(".", 1)[0] + ".json", dict(file=J["out"], secs=round(sum(len(p) for p in parts[:-1]) / sr, 1), parts=J["parts"]))


def j_audition(J):
    import difflib, pron as PR
    res = []
    for k, vid in enumerate(J["voices"]):
        if vid.endswith(".json"):                      # a clone made earlier in this run: read its id
            try: vid = json.load(open(os.path.join(OUT, vid)))["id"]
            except Exception as e: res.append(dict(voice=vid, error="no clone id: " + str(e)[:100])); continue
        for nk in J.get("modes") or ([True, False] if J.get("both") else [J.get("nikud", True)]):
            pairs = PR.prepare(J["text"], "he", {"nikud": nk})
            pre = os.path.join(OUT, f"{J.get('prefix', 'aud')}_{k}_{nk if isinstance(nk, str) else ('full' if nk else 'plain')}")
            try: sec, n = CM.cartesia(PR.tts_text(pairs), pre, "he", vid)
            except Exception as e: res.append(dict(voice=vid, nikud=nk, error=str(e)[:300])); continue
            segs, _ = model().transcribe(wav16(pre + ".wav"), language="he", beam_size=5)
            got = " ".join(s.text.strip() for s in segs); E, G = he_norm(J["text"]), he_norm(got)
            sm = difflib.SequenceMatcher(None, E, G, autojunk=False); h = sum(b.size for b in sm.get_matching_blocks())
            wrong = [dict(said=" ".join(E[a:b]), heard=" ".join(G[c:d])) for op, a, b, c, d in sm.get_opcodes() if op in ("replace", "delete")]
            res.append(dict(voice=vid, name=J.get("names", {}).get(J["voices"][k], ""), nikud=nk, wav=os.path.relpath(pre + ".wav", REPO), secs=round(sec, 2), score=round(h / max(1, len(E)), 3),
                            heard=got, misread=wrong, sent=PR.tts_text(pairs)))
            try: os.remove(pre + "_words.json")
            except OSError: pass
    save(J.get("out", "audition.json"), dict(text=J["text"], results=res))


def j_ipa(J):
    """IPA of chosen Hebrew words in a repo audio/video (Hebrew ASR cannot hear vowels: חָלָב and חֵלֶב are spelt
    the same). Hebrew whisper word timestamps, each matching word cut (+0.06 s) and read by the wav2vec2 phoneme
    recogniser (facebook/wav2vec2-xlsr-53-espeak-cv-ft, as in phon.yml) -> hevoice/<name>_ipa.json"""
    import torch, soundfile as sf
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + J["src"]], check=False)
    x = wav16(os.path.join(REPO, J["src"])); rx = re.compile(J["words"])
    P = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft"); M = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").eval()
    segs, _ = model().transcribe(x, language="he", beam_size=5, word_timestamps=True, condition_on_previous_text=False)
    words = [w for s in segs for w in s.words]; out = []
    for k, w in enumerate(words):
        bare = re.sub(r"[^\u05D0-\u05EA]", "", w.word)
        if not rx.fullmatch(bare): continue
        a, b = max(0, int((w.start - 0.06) * 16000)), int((w.end + 0.06) * 16000)
        with torch.no_grad(): lg = M(P(x[a:b], sampling_rate=16000, return_tensors="pt").input_values).logits
        ctx = "".join(v.word for v in words[max(0, k - 4):k + 4]).strip()
        out.append(dict(word=bare, start=round(w.start, 2), ipa=P.batch_decode(torch.argmax(lg, -1))[0], context=ctx))
    save(f"{J['name']}_ipa.json", dict(src=J["src"], hits=out))


def j_clone(J):
    import requests
    out = J.get("out", "clone.json"); p = os.path.join(OUT, out)
    if os.path.exists(p) and json.load(open(p)).get("id"): print("clone exists, skipped"); return
    with open(os.path.join(REPO, J["clip"]), "rb") as f:
        r = requests.post(f"{API}/voices/clone", headers=HDR, timeout=180,
                          files={"clip": (os.path.basename(J["clip"]), f, "audio/wav")},
                          data={"name": J.get("name", "Noam Roz - Hebrew"), "description": J.get("description", "Noam Roz, Hebrew, cloned from his own Hebrew reels (20 Nov 2026)"),
                                "language": J.get("language", "he"), "mode": J.get("mode", "similarity"), "enhance": str(J.get("enhance", False)).lower()})
    if r.status_code not in (200, 201):
        save(out, dict(error=r.status_code, body=r.text[:500])); return
    v = r.json(); save(out, dict(id=v.get("id"), name=v.get("name"), language=v.get("language"), created=time.strftime("%Y-%m-%d %H:%M")))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for J in json.load(open(os.path.join(OUT, "jobs.json"))):
        print("==", J.get("do"), J.get("name", ""))
        try: globals()["j_" + J["do"]](J)
        except Exception as e:
            import traceback; traceback.print_exc(); save(f"err_{J['do']}_{J.get('name', '')}.json", dict(error=str(e)[:500]))
