#!/usr/bin/env python3
"""Rough line ranges (seconds in the take) -> tight clips: snap to speech on/offsets, shorten pauses inside a line.

  python3 tools/face/autocut.py <take.wav> <ranges.json> <out_clips.json>
ranges: [{line, src, a, b}]  (a, b loose, from the word timestamps; several ranges per line splice takes)
"""
import sys, json, numpy as np, soundfile as sf
wav, rng, out = sys.argv[1:4]
y, sr = sf.read(wav); h = sr // 100
db = 20 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same")[::h]) + 1e-9)
TH, MINSIL, PRE, POST, KEEP = -40.0, 0.30, 0.08, 0.13, 0.10
clips = []
for g, r in enumerate(json.load(open(rng))):
    i0, i1 = int(r["a"] * 100), int(r["b"] * 100)
    sp = db[i0:i1] > TH
    # close tiny gaps (<60 ms) and drop blips (<50 ms)
    idx = np.flatnonzero(sp)
    if not len(idx): continue
    runs, s, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - p > 6: runs.append((s, p)); s = k
        p = k
    runs.append((s, p)); runs = [(a, b) for a, b in runs if b - a >= 5]
    segs, cur = [], [runs[0][0], runs[0][1]]
    for a, b in runs[1:]:
        if (a - cur[1]) / 100 < MINSIL: cur[1] = b
        else: segs.append(cur); cur = [a, b]
    segs.append(cur)
    n = len(segs)
    for k, (a, b) in enumerate(segs):
        ta = (i0 + a) / 100 - (PRE if k == 0 else KEEP); tb = (i0 + b) / 100 + (POST if k == n - 1 else KEEP)
        clips.append(dict(src=r["src"], a=float(round(max(r["a"], ta), 3)), b=float(round(min(r["b"], tb), 3)), line=r["line"], g=g, w=r["w"]))
json.dump(clips, open(out, "w"), indent=0)
tot = sum(c["b"] - c["a"] for c in clips)
for c in clips: print(c)
print("total", round(tot, 2))
