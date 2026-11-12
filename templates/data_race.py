"""data_race: a bar race over time from a table (inline JSON or a CSV in the repo). Hook on screen from frame 0 at the top,
the bars are already moving by 0.5 s. Ranks glide between keys, values count, a big yellow key label (the year).

data keys:
  csv        repo path to a CSV: first column = label, header row = keys (years), cells = numbers (empty = no data)
  table      inline instead of csv: {"keys": ["1950", ...], "series": [{"label", "flag", "emo", "values": [...], "hl": true}]}
  meta       {"Israel": {"flag": "IL", "hl": true}, ...}       flags / highlight for CSV rows
  at, until  anchors: the race runs from key 0 to the last key between these times (default 0.5 s to END-3)
  holds      {"1967": 1.5}   extra seconds to hold on a key (slow-down at the reveal)
  top        bars on screen (default 8, max 10)
  dec, prefix, suffix         value format
  unit       small label under the chart, e.g. "per 100,000 people"
  callouts   [{at, until, text}]  a short line on a card (max 12 words), e.g. the reveal
  hook_y     default 300
  optional (opt-in, nothing changes without them):
  nodata     true = rows with no value at the current key are drawn at the bottom as a grey "no data" pill
  photos     [{file, at, until, credit, kb, dim}] full-bleed real photos (as in photo_story); the race fades out while
             a photo is up, credit line at credit_y (default 1712). A photo with at 0 is on screen from frame 0.
  src_y      source chip y (default 1488)
  big        [{at, until, value, from, prefix, suffix, dec, sub, y, size}] counting number card (yellow), as in photo_story
"""
import os, sys, csv, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *


def load_table(D):
    if D.get("csv"):
        rows = list(csv.reader(open(os.path.join(REPO, D["csv"]), encoding="utf-8")))
        keys = rows[0][1:]; meta = D.get("meta", {}); series = []
        for r in rows[1:]:
            if not r or not r[0].strip(): continue
            vals = [float(x) if x.strip() else None for x in r[1:]]
            series.append(dict(meta.get(r[0], {}), label=r[0], values=vals))
        return keys, series
    return D["table"]["keys"], D["table"]["series"]


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    keys, series = load_table(D); nk = len(keys)
    for s in series:   # fill gaps by interpolation, keep None before the first value (bar hidden = no data yet)
        v = s["values"]; idx = [i for i, x in enumerate(v) if x is not None]
        s["v"] = np.array([np.interp(i, idx, [v[j] for j in idx]) if idx and i >= idx[0] else np.nan for i in range(nk)], float)
    t0, t1 = A(D.get("at"), 0.5), A(D.get("until"), T.END - 3.0)
    holds = D.get("holds", {}); w = np.array([1.0 + float(holds.get(k, 0)) for k in keys])   # time weight per key step
    cum = np.concatenate([[0], np.cumsum(w[1:])]); cum = cum / cum[-1] if cum[-1] > 0 else cum
    def kpos(t):     # fractional key index at time t
        u = min(1, max(0, (t - t0) / max(0.1, t1 - t0)))
        i = int(np.searchsorted(cum, u, "right") - 1); i = min(i, nk - 2)
        return i + (u - cum[i]) / max(1e-9, cum[i + 1] - cum[i]) if nk > 1 else 0.0
    V = np.array([s["v"] for s in series])                        # (n, nk)
    def ranks(i):
        v = np.nan_to_num(V[:, i], nan=-1e18); order = np.argsort(-v, kind="stable"); r = np.empty(len(v)); r[order] = np.arange(len(v)); return r
    R = np.array([ranks(i) for i in range(nk)]).T
    top = min(10, D.get("top", 8)); cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    CO = [(A(c["at"]), A(c.get("until"), T.END - 0.6), c["text"]) for c in D.get("callouts", [])]
    y0, rowh = 560, min(100, 820 // top); x0, lab_w = 60, 330; bw_max = 1000 - x0 - lab_w - 180
    PH = []
    if D.get("photos"):
        from photo_story import _load, _kb
        for p in D["photos"]:
            PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))))
        y_ = np.arange(H, dtype=np.float32)[:, None]
        VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.62, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
        NAVYC = np.array([97, 50, 22], np.float32)
    BIG = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), T.END - 0.6)) for g in D.get("big", [])]

    for p in PH:     # a photo followed straight away by another stays full until the next one has faded in over it
        p["cont"] = any(abs(q["t0"] - p["t1"]) < 0.05 for q in PH if q is not p)

    def ph_alpha(p, t):
        a_in = 1.0 if p["t0"] <= 0.01 else seg(t, p["t0"] - 0.3, p["t0"])
        a_out = (1.0 if t < p["t1"] + 0.05 else 0.0) if p["cont"] else 1.0 - seg(t, p["t1"] - 0.3, p["t1"])
        return min(a_in, a_out)

    def ph_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.25)
        return img * (1 - dim) * VIG + NAVYC * dim * 0.35

    def frame(t):
        base = bars_moving_bg(t); pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        over = max([win(t, b["t0"] - 0.2, b["t1"], 0.3, 0.3) for b in BIG] + [0])
        if over > 0: base = base * (1 - 0.45 * over)
        cv = Canvas(base); kp = kpos(t); i = int(math.floor(kp)); f = kp - i; j = min(nk - 1, i + 1); fe = ease(f)
        vals = V[:, i] + (V[:, j] - V[:, i]) * f
        def rk_at(tt):
            kq = kpos(tt); a_ = int(math.floor(kq)); b_ = min(nk - 1, a_ + 1); vv = V[:, a_] + (V[:, b_] - V[:, a_]) * (kq - a_)
            v = np.nan_to_num(vv, nan=-1e18); o = np.argsort(-v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
        rk = np.mean([rk_at(t - d_) for d_ in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)], axis=0)
        vmax = np.nanmax(np.where(rk < top, vals, np.nan)) if np.any(rk < top) else 1.0
        ca = 1.0 - pa
        for n_, s in enumerate(series):
            if ca <= 0.01: break
            if np.isnan(vals[n_]) and D.get("nodata") and rk[n_] <= top - 0.05:
                y = y0 + rk[n_] * rowh; a = ca * min(1, top - rk[n_]) * 0.85; ix = x0 + 34
                if s.get("flag"): WG.flag(cv, s["flag"], ix, y, 56, a)
                elif s.get("emo"): WG.emo(cv, s["emo"], ix, y, 56, a)
                cv.text(s["label"], ix + 42, y, 52, "#C9D6EA", a, align="l", maxw=lab_w - 70, minsz=38)
                cv.rrect(x0 + lab_w, y - rowh * 0.30, 230, rowh * 0.60, 14, "#3A4A66", a)
                cv.text("no data", x0 + lab_w + 115, y, 40, "#C9D6EA", a, maxw=210, minsz=36, bold=False)
                continue
            if np.isnan(vals[n_]) or rk[n_] > top - 0.05: continue
            y = y0 + rk[n_] * rowh; a = ca * min(1, top - rk[n_])
            col = YEL if s.get("hl") else s.get("color", SKY)
            ix = x0 + 34
            if s.get("flag"): WG.flag(cv, s["flag"], ix, y, 56, a)
            elif s.get("emo"): WG.emo(cv, s["emo"], ix, y, 56, a)
            cv.text(s["label"], ix + 42, y, 52, YEL if s.get("hl") else WHITE, a, align="l", maxw=lab_w - 70, minsz=38)
            bw = bw_max * max(0.004, vals[n_] / vmax)
            cv.rrect(x0 + lab_w, y - rowh * 0.34, max(10, bw), rowh * 0.68, 14, col, a)
            cv.text(WG.fmtn(vals[n_], D.get("dec", 0), D.get("prefix", ""), D.get("suffix", "")), x0 + lab_w + max(10, bw) + 14, y, 50,
                    YEL if s.get("hl") else WHITE, a, align="l", maxw=200, minsz=38)
        kl = keys[min(nk - 1, int(round(kp)))]
        if ca > 0.01: cv.text(str(kl), 1010 - cv.tw(str(kl), 150) / 2, y0 + top * rowh + 40, 150, YEL, 0.9 * ca, maxw=500)
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1000, minsz=36)
        for g in BIG:
            ga = win(t, g["t0"] - 0.1, g["t1"], 0.35, 0.35)
            if ga > 0.01:
                y = g.get("y", 960); cv.rrect(110, y - 175, 860, 350, 40, "#0B1730", 0.9 * ga, stroke=YEL, lw=3, sa=ga)
                v = WG.roll(t, g["t0"], float(g.get("from", 0)), float(g["value"]), g.get("roll", 1.1))
                cv.text(WG.fmtn(v, g.get("dec", 0), g.get("prefix", ""), g.get("suffix", "")), 540, y - 35, g.get("size", 170), YEL, ga, maxw=800)
                if g.get("sub"): cv.text(g["sub"], 540, y + 105, 56, WHITE, ga, maxw=800)
        if D.get("unit") and ca > 0.01: cv.text(D["unit"], 540, y0 - 70, 44, "#C9D6EA", ca, bold=False, maxw=960)
        for a0, a1, txt in CO:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: WG.panel(cv, 80, 1230, 920, 200, a, stroke=YEL); WG.title(cv, txt, 1330, a, 64)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 290), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t0, "whoosh", -26)] + [(a0, "pop", -22) for a0, _, _ in CO] + [(p["t0"], "whoosh", -30) for p in PH if p["t0"] > 0.01] + [(g["t0"], "pop", -22) for g in BIG]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
