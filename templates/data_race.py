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

    def frame(t):
        cv = Canvas(bars_moving_bg(t)); kp = kpos(t); i = int(math.floor(kp)); f = kp - i; j = min(nk - 1, i + 1); fe = ease(f)
        vals = V[:, i] + (V[:, j] - V[:, i]) * f
        def rk_at(tt):
            kq = kpos(tt); a_ = int(math.floor(kq)); b_ = min(nk - 1, a_ + 1); vv = V[:, a_] + (V[:, b_] - V[:, a_]) * (kq - a_)
            v = np.nan_to_num(vv, nan=-1e18); o = np.argsort(-v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
        rk = np.mean([rk_at(t - d_) for d_ in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)], axis=0)
        vmax = np.nanmax(np.where(rk < top, vals, np.nan)) if np.any(rk < top) else 1.0
        ca = 1.0
        for n_, s in enumerate(series):
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
        cv.text(str(kl), 1010 - cv.tw(str(kl), 150) / 2, y0 + top * rowh + 40, 150, YEL, 0.9 * ca, maxw=500)
        if D.get("unit"): cv.text(D["unit"], 540, y0 - 70, 44, "#C9D6EA", ca, bold=False, maxw=960)
        for a0, a1, txt in CO:
            a = win(t, a0, a1, 0.3, 0.3)
            if a > 0.01: WG.panel(cv, 80, 1230, 920, 200, a, stroke=YEL); WG.title(cv, txt, 1330, a, 64)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 290), 72)
        draw_sources(cv, S, T, t); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t0, "whoosh", -26)] + [(a0, "pop", -22) for a0, _, _ in CO]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
