#!/usr/bin/env python3
"""tools/follow_mix.py <spec id> : mix test of follow line takes inside one real reel (Noam 22 Nov 2026, name clarity).

For each take in ref/follow_tts/mix_job.json ({"takes": ["ref/follow_line_tts.wav", "ref/follow_tts/take3.wav", ...]})
the reel's own spec is loaded, its follow line pointed at that take, and the narration (cached voice, no new TTS) plus
the music bed are mixed exactly like the render does (build_voice + build_mix, without SFX). The whole mix is transcribed
with faster-whisper small.en (same settings as templates/check.py) and the words heard between "follow" and "for more"
are kept. Output: ref/follow_tts/mix/<id>.json. Nothing renders video, nothing posts."""
import os, sys, json, re, shutil, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "templates", "core"))
import numpy as np
import common as C


def heard_name(tx):
    m = re.search(r"follow\b(.{0,40}?)\bfor more", tx, re.I | re.S)
    return (m.group(1).strip(" ,.") if m else None)


def main(sid):
    J = json.load(open(os.path.join(REPO, "ref", "follow_tts", "mix_job.json")))
    from faster_whisper import WhisperModel
    W = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    out = dict(id=sid, takes=[])
    for tk in J["takes"]:
        S = C.load_spec(os.path.join(REPO, "specs", f"{sid}.json"))
        fl = [L for L in S["lines"] if L.get("wav") == C.FOLLOW_TTS]
        if not fl: out["error"] = "no follow line on ref/follow_line_tts.wav"; break
        for L in fl: L["wav"] = tk
        work = tempfile.mkdtemp(prefix="fm_")
        try:
            tj = C.build_voice(S, work); T = C.Timing(tj)
            C.build_mix(S, T, work)
            pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f"{work}/mix.m4a", "-ar", "16000", "-ac", "1",
                                                "-f", "f32le", "-"], check=True, capture_output=True).stdout, np.float32)
            segs, _ = W.transcribe(pcm, language="en", beam_size=5, condition_on_previous_text=False)
            tx = " ".join(s.text.strip() for s in segs).strip()
            nm = heard_name(tx)
            r = dict(take=tk, heard=nm, exact=bool(nm) and bool(re.fullmatch(r"noam\s+roz", nm, re.I)),
                     transcript=tx, lufs=C.measure(f"{work}/mix.m4a")["lufs"])
        except Exception as e:
            r = dict(take=tk, error=str(e)[:300])
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["takes"].append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    os.makedirs(os.path.join(REPO, "ref", "follow_tts", "mix"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "ref", "follow_tts", "mix", f"{sid}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1])
