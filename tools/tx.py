import json, os, sys, traceback
try:
    from faster_whisper import WhisperModel
    m = WhisperModel("small.en", device="cpu", compute_type="int8")
    for name in open("heygen/transcribe.txt").read().split():
        out = "heygen/" + name.rsplit(".", 1)[0] + ".words.json"
        if os.path.exists(out):
            continue
        segs, _ = m.transcribe("heygen/" + name, word_timestamps=True, language="en")
        words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for s in segs for w in s.words]
        json.dump(words, open(out, "w"), indent=0)
        print(name, " ".join(x["w"] for x in words))
except Exception:
    open("heygen/tx.log", "w").write(traceback.format_exc())
    raise
# rerun
