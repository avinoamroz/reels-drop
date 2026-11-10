"""map_counter: motion opener (globe spin / map tear / fold ...) lands on a map, region zooms, a big live counter, pins
with labels, bar cards and a big-number card, source chips, end card (balance line + CTA), loop back into the opener.
Map rules built in: whole land of Israel gold, West Bank and Gaza a darker gold, Green Line green (core/base.py MapR).

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  cam0        [lon, lat, width_deg] where the opener lands (frame 0 of the body)
  cams        [{at, view: [lon, lat, width_deg, screen_y], dur}]       camera moves
  highlight   [{at, until, code}]                                        countries tinted red (ISO3, e.g. "RUS")
  pins        [{at, until, lon, lat, label, color: red|yellow|grey}]
  counter     {label, of, at, until, y, steps: [{at, value}], prefix, suffix, dec}
  bars        [{at, until, title, y, rows: [{label, flag, value, txt, at, hl}], vmax}]
  big         [{at, until, value, from, prefix, suffix, sub, y, dec}]
  dim         [{at, until}]                                              darken the map behind cards
  hook_y      y of the hook text in the body's frame 0 (default 930)
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
import common as CM

COL = dict(red=RED, yellow=YEL, grey=GREY, sky=SKY, green=GREEN)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75))
    A = T.at
    def cam(lon, lat, wd, sy=960.0):
        cosf = max(0.6, math.cos(math.radians(min(62, abs(lat))))); return np.array([lon, lat, math.log(W / (wd * cosf)), cosf, sy], float)
    K0 = cam(*D["cam0"]); keys = [(0.0, 0.01, K0)] + [(A(c["at"]), float(c.get("dur", 1.3)), cam(*c["view"])) for c in D.get("cams", [])]
    def cam_at(t):
        cur = keys[0][2]
        for i, (t0, d, k) in enumerate(keys):
            if t < t0: break
            prev = keys[i - 1][2] if i else k; u = ease((t - t0) / d); cur = prev + (k - prev) * u
            if u < 1: return cur
        return cur
    HL = [(A(h["at"]), A(h.get("until"), T.END), h["code"]) for h in D.get("highlight", [])]
    PINS = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END - 0.6)) for p in D.get("pins", [])]
    CT = D.get("counter")
    if CT:
        CT = dict(CT, t0=A(CT.get("at"), A(CT["steps"][0]["at"]) - 0.4), t1=A(CT.get("until"), T.END - 0.6),
                  st=[(A(s["at"]), float(s["value"])) for s in CT["steps"]])
    BARS = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END - 0.6)) for b in D.get("bars", [])]
    for b in BARS:
        for k, r in enumerate(b["rows"]): r["_at"] = A(r["at"]) if r.get("at") is not None else b["t0"] + 0.35 + 0.18 * k
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]
    DIM = [(A(d_["at"]), A(d_.get("until"), T.END)) for d_ in D.get("dim", [])] + [(b["t0"], b["t1"]) for b in BARS] + [(g["t0"], g["t1"]) for g in BIG]
    if S.get("end"): DIM.append((A(S["end"].get("at"), T.S(len(T.L) - 2)), T.END))
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 930)
    MR = MapR(); cache = {}

    def mapimg(cm, hi):
        key = (tuple(np.round(cm, 4)), tuple(sorted((k, round(v, 2)) for k, v in hi.items())))
        if key not in cache:
            if len(cache) > 12: cache.clear()
            lon, lat, lk, cosf, sy = cm
            cache[key] = MR.render(MR.view(lon, lat, math.exp(lk), cosf, sy), H, hi=[k for k, v in hi.items() if v > 0.02], hi_a=hi)
        return cache[key]
    y_ = np.arange(H, dtype=np.float32)[:, None]
    VIG = (1 - np.maximum(np.clip(1 - y_ / 520, 0, 1) ** 1.6 * 0.45, np.clip((y_ - 1180) / 740, 0, 1) ** 1.2 * 0.62))[..., None]

    def frame(t):
        hi = {}
        for t0, t1, code in HL:
            v = 0.85 * seg(t, t0, t0 + 0.4) * (1 - seg(t, t1 - 0.3, t1 + 0.2))
            if v > 0.02: hi[code] = max(hi.get(code, 0), v)
        img, M = mapimg(cam_at(t), hi); base = img.astype(np.float32) * VIG
        dm = max([win(t, a - 0.3, b, 0.4, 0.4) for a, b in DIM] or [0])
        if dm > 0: base *= (1 - 0.5 * dm)
        cv = Canvas(base); geo = lambda lon, lat: geo2px(M, lon, lat)
        for p in PINS:
            pa = win(t, p["t0"] - 0.1, p["t1"], 0.3, 0.3)
            if pa > 0.01:
                x, y = geo(p["lon"], p["lat"]); c = COL.get(p.get("color", "red"), p.get("color", RED))
                WG.pin(cv, x, y, c, pa, t)
                if p.get("label"): WG.tag(cv, p["label"], x, y - 50, p.get("size", 40), YEL if c == YEL else WHITE, pa)
        if CT:
            ca = win(t, CT["t0"], CT["t1"], 0.35, 0.35)
            if ca > 0.01:
                v = 0.0; prev = 0.0
                for t0, val in CT["st"]:
                    if t >= t0: v = WG.roll(t, t0, prev, val, 0.7)
                    prev = val
                y = CT.get("y", 1240); WG.panel(cv, 90, y - 150, 900, 300, ca)
                cv.text(CT.get("label", ""), 540, y - 85, 48, "#C9D6EA", ca, maxw=820)
                txt = WG.fmtn(v, CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", "")) + (f" / {CT['of']}" if CT.get("of") else "")
                cv.text(txt, 540, y + 40, 150, YEL, ca, maxw=820)
        for b in BARS:
            ba = win(t, b["t0"] - 0.1, b["t1"], 0.4, 0.4)
            if ba > 0.01:
                n = len(b["rows"]); rowh = min(110, 640 // max(1, n)); y0 = b.get("y", 560)
                WG.panel(cv, 50, y0 - 90, 980, rowh * n + 120, ba)
                if b.get("title"): WG.title(cv, b["title"], y0 - 190, ba, 76)
                rows = [dict(label=r["label"], flag=r.get("flag"), emo=r.get("emo"), value=float(r["value"]), at=r["_at"],
                             color=YEL if r.get("hl") else r.get("color", RED), txt=r.get("txt"), tcol=YEL if r.get("hl") else WHITE) for r in b["rows"]]
                WG.bars(cv, rows, t, ba, y0, rowh=rowh, vmax=b.get("vmax"), lab_w=b.get("lab_w", 400))
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(120, y - 170, 840, 340, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), 1.2)
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 30, 170, YEL, ga, maxw=780)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 52, WHITE, ga, maxw=780)
        draw_screen(cv, cards, t, D.get("title_y", 270))
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        out = cv.out().astype(np.float32)
        ha = 1 - seg(t, T.E(0) + 0.15, T.E(0) + 0.55)
        if ha > 0.01: out = HK.paste(out, hook_img, 540, HY, 1.0, ha)
        cv2_ = Canvas(out); draw_subs(cv2_, CH, t, D.get("subs_y", 1612))
        return np.clip(cv2_.out(), 0, 255).astype(np.uint8)

    ev = [(p["t0"], "pop", -22) for p in PINS] + [(b["t0"] - 0.2, "whoosh", -24) for b in BARS] + [(g["t0"], "roll", g["t0"] + 1.2) for g in BIG]
    ev += [(g["t0"] + 1.2, "ding", -26) for g in BIG]
    if CT: ev += [(t0, "roll", t0 + 0.7) for t0, _ in CT["st"]]
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, ev)
    body = f"{work}/body.mp4"
    encode(frame, T.END, f"{work}/mix.m4a", body, loop=not S.get("opener"))
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    if S.get("opener"):
        op = wrap_opener(body, S, final, D["cam0"], HY, work)
    else:
        shutil.copy(body, final); op = None
    return dict(voice=J["voice"], body_s=T.END, opener=op, mix=aud)

