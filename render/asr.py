#!/usr/bin/env python3
"""Cloud ASR QA, run by render.yml after a successful render. Never fatal for the render.

usage: python render/asr.py <id> <rendered.mp4> <job.json>

Writes, next to render/done/<id>.json:
  render/done/<id>.asr.json   {"id","lang","model","duration","segments":[{"s","e","text","words":[{"w","s","e","p"}]}]}
  render/done/<id>.asr.txt    one line per segment: "[start end] text"
If job.json has "asr_lang" (e.g. "he") and "asr_clips" (absolute /home/claude/... paths from the bundle):
  render/done/<id>.asr_<lang>.json  {"lang","model","clips":{path: {"duration","segments":[...]}}}
  render/done/<id>.asr_<lang>.txt   "## path" then one line per segment
Optional job.json keys: "asr": false (skip), "asr_video_lang" (default "en"),
"asr_model" (video model, default small.en for English, large-v3-turbo otherwise),
"asr_clip_model" (default large-v3-turbo, or small.en when asr_lang is "en").
"""
import json, os, sys, time, traceback

ID, MP4, JOB = sys.argv[1:4]
J = json.load(open(JOB))
DONE = "render/done"
os.makedirs(DONE, exist_ok=True)


def default_model(lang):
    return "small.en" if lang == "en" else "large-v3-turbo"


_models = {}


def model(name):
    if name not in _models:
        from faster_whisper import WhisperModel
        _models[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    return _models[name]


def tx(name, path, lang):
    segs, info = model(name).transcribe(path, language=lang, word_timestamps=True, beam_size=5,
                                        condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append({"s": round(s.start, 2), "e": round(s.end, 2), "text": s.text.strip(),
                    "words": [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2),
                               "p": round(w.probability, 2)} for w in (s.words or [])]})
    return out, round(info.duration, 2)


def lines(segs):
    return [f"[{s['s']:.2f} {s['e']:.2f}] {s['text']}" for s in segs]


def main():
    if J.get("asr") is False:
        print("asr disabled in job.json"); return
    t0 = time.time()
    vlang = J.get("asr_video_lang", "en")
    vmodel = J.get("asr_model") or default_model(vlang)
    segs, dur = tx(vmodel, MP4, vlang)
    json.dump({"id": ID, "lang": vlang, "model": vmodel, "duration": dur, "segments": segs},
              open(f"{DONE}/{ID}.asr.json", "w"), ensure_ascii=False, indent=0)
    open(f"{DONE}/{ID}.asr.txt", "w").write("\n".join(lines(segs)) + "\n")
    print(f"video: {len(segs)} segments, {dur} s audio, {time.time()-t0:.0f} s, model {vmodel}")
    print("\n".join(lines(segs)))

    lang, clips = J.get("asr_lang"), J.get("asr_clips") or []
    if lang and clips:
        cmodel = J.get("asr_clip_model") or default_model(lang)
        res, txt = {}, []
        for p in clips:
            src = p if os.path.exists(p) else None
            if not src:
                res[p] = {"error": "not found in bundle"}; txt += [f"## {p}", "(not found in bundle)"]; continue
            try:
                s, d = tx(cmodel, src, lang)
                res[p] = {"duration": d, "segments": s}; txt += [f"## {p}"] + lines(s)
            except Exception as e:
                res[p] = {"error": str(e)[:300]}; txt += [f"## {p}", f"(error {e})"]
        json.dump({"id": ID, "lang": lang, "model": cmodel, "clips": res},
                  open(f"{DONE}/{ID}.asr_{lang}.json", "w"), ensure_ascii=False, indent=0)
        open(f"{DONE}/{ID}.asr_{lang}.txt", "w").write("\n".join(txt) + "\n")
        print(f"clips ({lang}, {cmodel}):"); print("\n".join(txt))
    print(f"asr total {time.time()-t0:.0f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        open(f"{DONE}/{ID}.asr.err", "w").write(traceback.format_exc()[-3000:])
        sys.exit(1)
