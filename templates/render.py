#!/usr/bin/env python3
"""templates/render.py: one JSON spec -> one reel.

  python3 templates/render.py specs/<id>.json            full render: out/<id>.mp4, covers/<id>_cover.jpg,
                                                          previews/<id>_540.mp4, out/frames/<id>_{0,1,3,last}.jpg, out/<id>.report.json
  python3 templates/render.py specs/<id>.json --lint     spec checks only (no voice, no render)
  python3 templates/render.py specs/<id>.json --voice-only   Cartesia TTS (secret) or check voice/<id>_*.wav, nothing else
  TPL_SNAPS=0,3,10 python3 templates/render.py spec.json  a few frames to /tmp/tpl_<id>/snaps.png (local look, no video)

Templates: map_counter, data_race, scorecard, slang_scene, rehook, pun_sequence, pun_ladder, classroom (see templates/README.md).
"""
import os, sys, time, json, importlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import common as CM

TEMPLATES = ("map_counter", "data_race", "scorecard", "slang_scene", "rehook", "pun_sequence", "pun_ladder", "classroom", "build_vs_war", "two_paths", "surgery", "crowd_equals", "versus_card", "board_insert", "story_climb", "story_climb_full")


def main():
    S = CM.load_spec(sys.argv[1])
    if S.get("template") not in TEMPLATES + ("photo_story",): raise SystemExit(f"template must be one of {TEMPLATES + ('photo_story',)}")
    warn = CM.lint(S)
    for w in warn: CM.log("lint warning:", w)
    if "--lint" in sys.argv: print("lint ok"); return
    if "--voice-only" in sys.argv:          # workflow step 1: make sure a voice exists (Cartesia or voice/<id>_*.wav)
        if S["template"] == "rehook" and not S.get("lines"): print("rehook: no new voice needed"); return
        raw, wj, how = CM.get_voice(S); print("voice ok:", how, os.path.relpath(raw, CM.REPO)); return
    t0 = time.time(); mod = importlib.import_module(S["template"])
    extra = mod.run(S) or {}
    if os.environ.get("TPL_SNAPS"): return
    final = os.path.join(CM.REPO, "out", f"{S['id']}.mp4")
    extra.update(lint_warnings=warn, render_s=round(time.time() - t0, 1))
    rep = CM.finish(S, final, extra)
    print(json.dumps({k: rep[k] for k in ("id", "duration", "audio", "outputs", "render_s")}, indent=1))


if __name__ == "__main__":
    main()
