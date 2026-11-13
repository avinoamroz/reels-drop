"""irony_reasons: the "backhanded compliment" list reel ("8 reasons I 'hate' X" where every reason is a good deed).
Built for satire that must read as satire in every frame (Instagram auto-moderation + clipping risk):
  * a "SARCASM" eye-roll badge sits at the top of EVERY frame, frame 0 included;
  * frame 0: a grid of the reasons' real photos (all good deeds) behind the hook, a big eye-roll emoji under it;
  * the first reason card is on screen by `data.first_at` (default 1.2 s), before the hook line has finished;
  * each reason: full-bleed real photo (Ken Burns), counter pill "Reason 3 of 8", the deed in big text, who/when,
    a punch stamp ("UGH.") with an eye-roll, small source label, photo credit;
  * subtitles: spec `subs_replace` (e.g. {"hate": "“hate”"}) keeps the word in scare quotes on screen;
  * end: warm reveal lines + a pill (e.g. "Part 2 on my profile"), then the usual dissolve back into frame 0.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  badge     text of the top badge (default "SARCASM"); badge_emoji (default "🙄")
  hook_emoji emoji under the hook (default "🙄");  hook_y (default 820)
  first_at  when reason 1 replaces the hook grid (default 1.2)
  of        total reasons shown on the counter (default len(reasons))
  reasons   [{n, at, photo, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], gray, dim, title, who, punch, punch_color, emoji, src}]
            shows from `at` until the next reason's `at` (the last one until end.at)
  end       {at, lines: [{text, at, size, y}], pill: {text, at}, emoji, photo (optional, dimmed)}
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

PCOL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)


def _load(path, gray=False):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"irony_reasons: cannot read photo {path}")
    if gray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); im = cv2.merge([g, g, g])
    return im


def _kb(im, cx, cy, z, w_=W, h_=H):
    h, w = im.shape[:2]; s = max(w_ / w, h_ / h) * max(1.0, z)
    cw, ch = w_ / s, h_ / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w_, h_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    E = S.get("end") or {}
    END_AT = A(E.get("at"), T.S(len(T.L) - 2))
    R = []
    for k, r in enumerate(D["reasons"]):
        t0 = float(D.get("first_at", 1.2)) if k == 0 else A(r["at"])
        R.append(dict(r, t0=t0, im=_load(r["photo"], r.get("gray", False))))
    for k, r in enumerate(R): r["t1"] = R[k + 1]["t0"] if k + 1 < len(R) else END_AT
    OF = int(D.get("of", len(R)))
    FIRST = R[0]["t0"]
    END_IM = _load(E["photo"], True) if E.get("photo") else None

    # frame-0 grid: the reasons' photos, 2 columns x 4 rows (all good deeds behind the hook)
    grid = np.zeros((H, W, 3), np.float32)
    cells = [r for r in R][:8]
    for i, r in enumerate(cells):
        cx_, cy_ = (i % 2) * 540, (i // 2) * 480
        kb = r.get("kb") or [0.5, 0.5, 1.0]
        grid[cy_:cy_ + 480, cx_:cx_ + 540] = _kb(r["im"], kb[0], kb[1], 1.0, 540, 480)
    grid = grid * 0.42 + np.array([97, 50, 22], np.float32) * 0.2
    grid[:, 538:542] = (48, 26, 11); grid[478:482, :] = (48, 26, 11); grid[958:962, :] = (48, 26, 11); grid[1438:1442, :] = (48, 26, 11)

    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.5 * 0.6, np.clip((y_ - 900) / 1020, 0, 1) ** 1.05 * 0.86))[..., None]
    NAVY = np.array([97, 50, 22], np.float32)
    XF = 0.22
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"], size=130)
    HY = D.get("hook_y", 820); BADGE = D.get("badge", "SARCASM"); BEMO = D.get("badge_emoji", "🙄")

    def photo_layer(r, t):
        kb = r.get("kb") or [0.5, 0.42, 1.0, 0.5, 0.42, 1.12]
        u = ease((t - r["t0"]) / max(0.5, r["t1"] - r["t0"] + XF))
        img = _kb(r["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = r.get("dim", 0.12)
        return img * (1 - dim) * VIG + NAVY * dim * 0.35

    def base_at(t):
        if t < FIRST + XF: out = grid.copy()
        else: out = BG.copy()
        for k, r in enumerate(R):
            if t < r["t0"] or t >= r["t1"] + XF: continue
            a = seg(t, r["t0"], r["t0"] + XF, lambda x: min(1.0, max(0.0, x)))
            if k == len(R) - 1 and t >= r["t1"]: a *= 1 - seg(t, r["t1"], r["t1"] + XF, lambda x: min(1.0, max(0.0, x)))
            lay = photo_layer(r, t); out = lay if a >= 0.999 else out * (1 - a) + lay * a
        if END_IM is not None and t >= END_AT:
            a = seg(t, END_AT, END_AT + 0.3)
            lay = _kb(END_IM, 0.5, 0.45, 1.05 + 0.05 * seg(t, END_AT, T.END)) * 0.3 + BG * 0.55
            out = out * (1 - a) + lay * a
        elif t >= END_AT:
            out = out * (1 - seg(t, END_AT, END_AT + 0.3)) + BG * seg(t, END_AT, END_AT + 0.3)
        return out

    def badge(cv, t):
        tw = cv.tw(BADGE, 44) + 130
        cv.rrect(540 - tw / 2, 92, tw, 84, 42, "#0B1730", 0.88, stroke=SKY, lw=3, sa=0.9)
        WG.emo(cv, BEMO, 540 - tw / 2 + 52, 134, 54, 1.0)
        cv.text(BADGE, 540 + 32, 134, 44, SKY, 1.0, maxw=tw - 100, shadow=False)

    def counter(cv, r, t):
        a = win(t, r["t0"], r["t1"] + 0.05, 0.2, 0.2)
        if a <= 0.01: return
        u = (t - r["t0"]) / 0.35; s_ = back(min(1.0, max(0.0, u)), 1.4) if u < 1 else 1.0
        c = cv.c; c.save(); c.translate(540, 290); c.scale(max(0.01, s_), max(0.01, s_)); c.translate(-540, -290)
        parts = [("Reason ", WHITE), (str(r["n"]), YEL), (f" of {OF}", WHITE)]
        tw = cv.tw(f"Reason {r['n']} of {OF}", 84) + 90
        cv.rrect(540 - tw / 2, 290 - 66, tw, 132, 66, "#0B1730", 0.9 * a, stroke=YEL, lw=4, sa=a)
        cv.rich(parts, 540, 290, 84, a, 1000)
        c.restore()

    def card(cv, r, t):
        a = win(t, r["t0"] + 0.08, r["t1"] + 0.05, 0.25, 0.2)
        if a <= 0.01: return
        WG.title(cv, r["title"], D.get("title_y", 1105), a, r.get("size", 96))
        if r.get("who"): cv.text(r["who"], 540, D.get("who_y", 1222), 52, "#C9D6EA", a, maxw=980, minsz=40)
        if r.get("src"): WG.src(cv, r["src"], a, D.get("src_y", 1488))
        if r.get("credit"):
            cv.text("Photo: " + r["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a, bold=False, maxw=1000, minsz=36)
        if r.get("punch"):
            pt = A(r["punch_at"]) if r.get("punch_at") else r["t1"] - 1.25
            pa = win(t, pt - 0.05, r["t1"] + 0.05, 0.08, 0.2)
            if pa > 0.01:
                k = 1.0 + 0.9 * (1 - eout((t - pt) / 0.18)) if t < pt + 0.18 else 1.0
                col = PCOL.get(r.get("punch_color", "yellow"), YEL); sz = r.get("punch_size", 104) * k; y = D.get("punch_y", 1355)
                ang = math.radians(r.get("angle", -6))
                c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.translate(-540, -y)
                tw = min(cv.tw(r["punch"], sz), 760) + 190
                cv.rrect(540 - tw / 2, y - sz * 0.7, tw, sz * 1.4, 20, "#0B1730", 0.6 * pa, stroke=col, lw=9 * k, sa=pa)
                cv.text(r["punch"], 540 - 45, y, sz, col, pa, maxw=760, shadow=False, minsz=40)
                WG.emo(cv, r.get("emoji", "🙄"), 540 + tw / 2 - 72, y, sz * 0.82, pa)
                c.restore()

    def end_card(cv, t):
        if t < END_AT - 0.1: return
        for ln in E.get("lines", []):
            t0 = A(ln.get("at"), END_AT); la = seg(t, t0 - 0.05, t0 + 0.3)
            if la > 0.01:
                rise = 30 * (1 - eout((t - t0) / 0.4))
                WG.title(cv, ln["text"], ln.get("y", 900) + rise, la, ln.get("size", 96))
        if E.get("emoji"):
            ea = seg(t, END_AT, END_AT + 0.3); esz = 190 * WG.pop_s(t, END_AT) if t < END_AT + 0.35 else 190
            if ea > 0.01 and esz > 4: WG.emo(cv, E["emoji"], 540, E.get("emoji_y", 600), esz, ea)
        P_ = E.get("pill")
        if P_:
            t0 = A(P_.get("at"), END_AT + 1.0); pa = seg(t, t0, t0 + 0.3)
            if pa > 0.01:
                tw = min(cv.tw(P_["text"].replace("*", ""), 64) + 110, 980); y = P_.get("y", 1330)
                cv.rrect(540 - tw / 2, y - 64, tw, 128, 64, SKY, pa)
                WG.title(cv, P_["text"], y, pa, 64)

    def frame(t):
        base = base_at(t); cv = Canvas(base)
        for r in R:
            if r["t0"] - 0.05 <= t < r["t1"] + 0.3: counter(cv, r, t); card(cv, r, t)
        end_card(cv, t)
        ha = 1 - seg(t, FIRST - 0.05, FIRST + 0.2)
        if ha > 0.01:                                    # big eye roll under the hook (frame 0)
            WG.emo(cv, D.get("hook_emoji", "🙄"), 540, HY + 330, 230 * (0.94 + 0.06 * eout(t / 0.2)), ha)
        badge(cv, t)
        out = cv.out().astype(np.float32)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 0.94 + 0.06 * eout(t / 0.2), ha)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(r["t0"], "whoosh", -28) for r in R] + [(r["t0"] + 0.05, "pop", -26) for r in R]
    ev += [((A(r["punch_at"]) if r.get("punch_at") else r["t1"] - 1.25), "thud", -20) for r in R if r.get("punch")]
    ev += [(END_AT, "ding", -24)]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=r["photo"], credit=r.get("credit")) for r in R])
