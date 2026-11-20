"""he_cards: Hebrew text cards over the brand background, one card per spoken line (20 Nov 2026, Hebrew voice tests and
simple Hebrew reels). Every line shows its "screen" text (or its "say" text) big, right to left (Heebo + bidi), wrapped
to 3 lines; an optional small "label" above it (e.g. "עם ניקוד" / "בלי ניקוד"); burned-in subtitles off by default
(data.subs true to show them). Lines may set "lang" ("he" / "en") and "nikud" (false = plain text to the voice).
data keys: hook_y (default 330), card_y (default 960), size (default 92), subs (default false),
  show_tts (true: nikud lines show the vowelised text sent to the voice), hook_hold (seconds the hook stays)."""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *


def he_lines(txt, sz, maxw=940):
    words = txt.split(); lines = [[]]
    for w in words:
        test = " ".join(lines[-1] + [w])
        if lines[-1] and hebrew_img(test, sz).shape[1] > maxw: lines.append([w])
        else: lines[-1].append(w)
    return [" ".join(l) for l in lines]


def card_imgs(txt, sz):
    while True:
        ls = he_lines(txt, sz)
        if len(ls) <= 3 or sz <= 60: return [hebrew_img(l, sz) for l in ls], sz
        sz -= 6


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.2))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 330); CY = D.get("card_y", 960); SZ = D.get("size", 92)
    SAY = [L for L in S["lines"] if L.get("say", "").strip()]
    shown = {}
    if D.get("show_tts"):          # show the vowelised text the voice got (nikud lines), so the intended reading is visible
        plan, _ = pron_plan(S)
        for L, lg, v, pr in plan:
            if lg == "he" and not L.get("pron_off"): shown[id(L)] = " ".join(c for _, c in pr)
    cards = []
    for i, L in enumerate(SAY):
        im, sz = card_imgs(L.get("screen") or shown.get(id(L)) or L["say"], SZ)
        lab = hebrew_img(L["label"], 48, (69, 164, 221)) if L.get("label") else None
        t0 = T.S(i) - 0.15; t1 = T.S(i + 1) - 0.15 if i + 1 < len(SAY) else T.END - 0.3
        cards.append((t0, t1, im, sz, lab))
    CH = sub_chunks(T, S.get("subs_replace")) if D.get("subs") else []

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 8))
        for t0, t1, im, sz, lab in cards:
            a = win(t, t0, t1, 0.18, 0.18)
            if a <= 0.01: continue
            h = len(im) * sz * 1.25
            WG.panel(cv, 50, CY - h / 2 - 90, 980, h + 180, a)
            if lab is not None: cv.image(lab, 540, CY - h / 2 - 40, 1.0, a)
            for k, li in enumerate(im): cv.image(li, 540, CY - h / 2 + sz * 1.25 * (k + 0.5) + 20, 1.0, a)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0) if D.get("hook_hold") is None else D["hook_hold"], HY, 0.8)
        c2 = Canvas(out)
        if CH: draw_subs(c2, CH, t, D.get("subs_y", 1612))
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, [(c[0] + 0.15, "tick", -24) for c in cards])
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=D.get("loop", False))
    return dict(voice=J["voice"], mix=aud)
