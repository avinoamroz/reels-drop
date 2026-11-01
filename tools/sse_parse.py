#!/usr/bin/env python3
"""Cartesia /tts/sse response (TTS1 endpoint=sse) -> <out_prefix>.wav + <out_prefix>_words.json.
Accepts the raw SSE text (GitHub download) or a Drive-tool JSON wrapper with base64 "content".
usage: sse_parse.py <src> <out_prefix>"""
import base64, json, sys, wave

src, out = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read().decode("utf-8", "replace")
if raw.lstrip().startswith("{"):
    try:
        raw = base64.b64decode(json.loads(raw)["content"]).decode("utf-8", "replace")
    except Exception:
        pass
pcm, words = bytearray(), []
for line in raw.splitlines():
    if not line.startswith("data:"):
        continue
    try:
        ev = json.loads(line[5:].strip())
    except Exception:
        continue
    if ev.get("type") == "chunk" and ev.get("data"):
        pcm += base64.b64decode(ev["data"])
    wt = ev.get("word_timestamps")
    if wt:
        for w, s, e in zip(wt["words"], wt["start"], wt["end"]):
            words.append({"word": w, "start": s, "end": e})
if not pcm:
    sys.exit("no audio chunks found (file not public, or not an SSE response)")
with wave.open(out + ".wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(bytes(pcm))
json.dump(words, open(out + "_words.json", "w"), indent=0)
print(len(pcm) / 88200, "s audio,", len(words), "words")
