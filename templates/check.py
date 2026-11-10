#!/usr/bin/env python3
"""templates/check.py specs/<id>.json : transcript check of out/<id>.mp4 (faster-whisper small.en on the CPU).
Expected text = the spec's spoken lines; for a rehook, the source reel's own transcript after the cut (the body must be
intact). Writes "transcript" into out/<id>.report.json: score (0 to 1, matched words / expected words), per line scores
(MISS under 0.8), and the heard text. Never blocks the render: on an error it writes the error into the report."""
import os, sys, re, json, difflib, subprocess, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def norm(s):
    from num2words import num2words
    s = s.lower().replace("u.n.", "un").replace("%", " percent")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s.replace("-", " "))
    return s.split()


def pcm(path, start=0.0):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def heard(model, path, start=0.0):
    segs, _ = model.transcribe(pcm(path, start), language="en", beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs)


def main(S, rp):
    R = json.load(open(rp))
    from faster_whisper import WhisperModel
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    got = heard(m, os.path.join(REPO, "out", f"{S['id']}.mp4"))
    if S.get("template") == "rehook":
        D = S["data"]; exp_lines = [heard(m, os.path.join(REPO, D["source"]), float(D.get("cut", 3.0)) + 0.3)]
    else:
        exp_lines = [L["say"] for L in S.get("lines", []) if L.get("say")]
    G = norm(got); per = []; tot = hit = 0
    for line in exp_lines:
        E = norm(line)
        sm = difflib.SequenceMatcher(None, E, G, autojunk=False); h = sum(b.size for b in sm.get_matching_blocks())
        sc = h / max(1, len(E))
        per.append(dict(line=line[:160], score=round(sc, 3), flag="MISS" if sc < 0.8 else "ok"))
        tot += len(E); hit += h
    R["transcript"] = dict(score=round(hit / max(1, tot), 3), model="faster-whisper small.en", lines=per, heard=got,
                           expected="spec lines" if S.get("template") != "rehook" else "source reel transcript after the cut")
    json.dump(R, open(rp, "w"), indent=1, ensure_ascii=False)
    print("transcript score", R["transcript"]["score"])
    for p in per: print(p["flag"], p["score"], p["line"][:90])


if __name__ == "__main__":
    S = json.load(open(sys.argv[1], encoding="utf-8")); S.setdefault("id", os.path.splitext(os.path.basename(sys.argv[1]))[0])
    rp = os.path.join(REPO, "out", f"{S['id']}.report.json")
    try: main(S, rp)
    except Exception as e:
        traceback.print_exc()
        try:
            R = json.load(open(rp)); R["transcript"] = dict(score=None, error=str(e)[:300]); json.dump(R, open(rp, "w"), indent=1, ensure_ascii=False)
        except Exception: pass
