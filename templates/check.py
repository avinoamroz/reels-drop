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


HE_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"      # Hebrew ASR (ivrit.ai fine-tune of whisper large-v3-turbo, CTranslate2)


def he_norm(s):
    s = re.sub(r"[֑-ׇ]", "", s).replace("־", " ").replace("-", " ")
    s = re.sub(r"\d+(\.\d+)?", lambda m: " #" + m.group() + " ", s.replace(",", ""))
    return re.sub(r"[^א-תa-z0-9#' ]+", " ", s.lower()).split()


def is_he(S): return S.get("voice", {}).get("lang") == "he" or any(L.get("lang") == "he" for L in S.get("lines", []))


def main_he(S, rp):
    """Hebrew (or mixed) reel: each spoken line cut from the mp4 by its timing and heard in its own language
    (Hebrew: ivrit-ai whisper large-v3-turbo; English: small.en). Report: score per line + every misread word."""
    import numpy as np
    from faster_whisper import WhisperModel
    import tempfile
    R = json.load(open(rp))
    T = json.load(open(os.path.join(tempfile.gettempdir(), "tpl_" + S["id"], "timing.json")))   # left by the render step (common.workdir)
    x = pcm(os.path.join(REPO, "out", f"{S['id']}.mp4"))
    SAY = [L for L in S["lines"] if L.get("say", "").strip()]
    nth = os.cpu_count() or 4
    mh = WhisperModel(HE_MODEL, device="cpu", compute_type="int8", cpu_threads=nth)
    me = None; per = []; tot = hit = 0; mis = []
    for L, tl in zip(SAY, T["lines"]):
        lg = L.get("lang") or S.get("voice", {}).get("lang", "en")
        a, b = max(0, int((tl["t0"] - 0.15) * 16000)), int((tl["t1"] + 0.3) * 16000); cut = x[a:b]
        if lg == "he":
            segs, _ = mh.transcribe(cut, language="he", beam_size=5, condition_on_previous_text=False, vad_filter=False)
            got = " ".join(s.text.strip() for s in segs); E, G = he_norm(L["say"]), he_norm(got)
        else:
            if me is None: me = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=nth)
            segs, _ = me.transcribe(cut, language="en", beam_size=5, condition_on_previous_text=False)
            got = " ".join(s.text.strip() for s in segs); E, G = norm(L["say"]), norm(got)
        sm = difflib.SequenceMatcher(None, E, G, autojunk=False); h = sum(bk.size for bk in sm.get_matching_blocks())
        wrong = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("replace", "delete"): wrong.append(dict(said=" ".join(E[i1:i2]), heard=" ".join(G[j1:j2])))
        sc = h / max(1, len(E)); tot += len(E); hit += h
        per.append(dict(line=L["say"][:160], lang=lg, nikud=L.get("nikud", True) if lg == "he" else None, heard=got,
                        score=round(sc, 3), flag="MISS" if sc < 0.8 else "ok", misread=wrong))
        mis += [dict(w, line=len(per) - 1) for w in wrong]
    R["transcript"] = dict(score=round(hit / max(1, tot), 3), model=f"{HE_MODEL} (he) + faster-whisper small.en (en), per line",
                           lines=per, misread=mis, expected="spec lines, each heard in its own language")
    json.dump(R, open(rp, "w"), indent=1, ensure_ascii=False)
    print("transcript score", R["transcript"]["score"])
    for p in per: print(p["flag"], p["score"], p["lang"], p["line"][:70], "| heard:", p["heard"][:70], "| misread:", p["misread"])


def main(S, rp):
    if is_he(S): return main_he(S, rp)
    R = json.load(open(rp))
    from faster_whisper import WhisperModel
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    got = heard(m, os.path.join(REPO, "out", f"{S['id']}.mp4"))
    if S.get("template") == "rehook":
        D = S["data"]; exp_lines = [heard(m, os.path.join(REPO, D["source"]), float(D.get("cut", 3.0)) + 0.3)]
    elif S.get("template") == "talk_cut":       # Noam's own words as subtitled (kept parts) + his follow line
        exp_lines = [R.get("said", "")] + ([S["data"]["follow"].get("text", "If you love Israeli facts, follow Noam Roz for more.")] if S["data"].get("follow") else [])
    elif S.get("template") == "face_hook":     # Noam's own words + the source after the cut (+ his follow line)
        D = S["data"]; cut = float(R.get("cut", D.get("cut", 0.0)))
        exp_lines = [D["said"]] + ([heard(m, os.path.join(REPO, D["source"]), cut + 0.3)] if D.get("source_speech", True) else [])
        if R.get("follow_text"): exp_lines.append(R["follow_text"])
    elif S.get("data", {}).get("take"):       # Noam's own take under an animation: the exact words of the kept parts
        exp_lines = [k[2] for k in S["data"]["take"]["keep"]]
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
                           expected={"rehook": "source reel transcript after the cut", "face_hook": "Noam's words, source transcript after the cut, follow line"}.get(S.get("template"), "spec lines"))
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
