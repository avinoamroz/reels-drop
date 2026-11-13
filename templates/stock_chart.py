"""stock_chart: numbers over time drawn like stock charts. Frame 0 is a map of the places compared (highlighted, hook over
it). Then one full-screen chart per place, each on its own scale on a shared year axis: the line draws left to right,
green where the number rises, red where it falls, a ticker header ("IRAQ <150k  [down] 90%") counts along with the line
head, tick sounds while it draws. Real photos between beats, a recap grid of every line (sparklines), an end card, and
the map again at the end so the loop back to frame 0 is seamless.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "END-2"):
  map      {view: [lon, lat, width_deg, centre_y], hi: ["TUR", ...] (tinted), labels: [{text, lon, lat, size}],
            pins: [{lon, lat, label}], until: anchor (first window, from 0), again: anchor (shown again, with the hook card, from here to END)}
  per chart optional: years [y0, y1] x range (default: its points plus a small pad), xticks [..] (default: about 4 round years)
  charts   [{name, sub, points: [[year, value, "label"?, est?], ...], at, until, draw: [anchor, anchor],
             prefix_last: "<" (upper bound, shown on the ticker), hl: true}]
           points draw in order between draw[0] and draw[1]; a point's 3rd item overrides its value label,
           4th item true = "est." under it
  photos   [{file, at, until, credit, kb, dim}] full-bleed photos (as data_race), credit at credit_y (default 1712)
  recap    {at, until, y0 (default 470), rowh (default 190)}  every chart as a sparkline row, hl row boxed
  end_until  anchor: the end card (spec "end") shows from end.at until this (default END-0.5)
  hook_y   default 300;  title_y default 300;  src_y default 1488;  subs_y default 1612
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

UP, DOWN, FLAT = "#22C55E", "#EF4444", "#9FB0C8"
def scol(a, b): return UP if b > a else (DOWN if b < a else FLAT)
DARK = np.array([30, 18, 11], np.float32)       # BGR near-black navy (approved mock look)


def fmt(v, pre=""):
    a = abs(v)
    if a >= 1e6: s = f"{v / 1e6:.1f}M".replace(".0M", "M")
    elif a >= 1e4: s = f"{v / 1e3:.0f}k"
    elif a >= 1e3: s = f"{v / 1e3:.1f}k".replace(".0k", "k")
    else: s = f"{v:.0f}"
    return pre + s


def nice_max(v):
    """4 gridline steps of a round size (1, 2, 2.5, 5 x 10^n) covering v with some headroom"""
    m = 10 ** math.floor(math.log10(v * 1.12 / 4))
    return 4 * next(k * m for k in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if 4 * k * m >= v * 1.12)


def xrange_of(c, pts):
    if c.get("years"): return c["years"]
    a, b = pts[0][0], pts[-1][0]; pad = max(2.0, 0.07 * (b - a)); return [a - pad, b + pad]


def xticks_of(c, y0, y1):
    if c.get("xticks"): return c["xticks"]
    st = next(k for k in (2, 5, 10, 20, 25, 30, 50) if (y1 - y0) / k <= 4.5)
    return [y for y in range(int(math.ceil(y0 / st) * st), int(y1) + 1, st)]


def tri(cv, x, y, s, up, col, a):
    c = cv.c; c.new_path()
    if up: c.move_to(x - s, y + s * 0.8); c.line_to(x + s, y + s * 0.8); c.line_to(x, y - s * 0.9)
    else: c.move_to(x - s, y - s * 0.8); c.line_to(x + s, y - s * 0.8); c.line_to(x, y + s * 0.9)
    c.close_path(); cv.col(col, a); c.fill()


# country name -> ISO2 for WG.flag; a chart can set "flag": "XX" (or false to hide). Non-countries (towns, regions) get none.
FLAGS = {"ISRAEL": "IL", "IRAQ": "IQ", "SYRIA": "SY", "TURKEY": "TR", "TURKIYE": "TR", "LEBANON": "LB", "EGYPT": "EG",
         "JORDAN": "JO", "IRAN": "IR", "SAUDI ARABIA": "SA", "SAUDI": "SA", "QATAR": "QA", "UAE": "AE", "KUWAIT": "KW",
         "BAHRAIN": "BH", "OMAN": "OM", "YEMEN": "YE", "LIBYA": "LY", "TUNISIA": "TN", "ALGERIA": "DZ", "MOROCCO": "MA",
         "CYPRUS": "CY", "GREECE": "GR", "SUDAN": "SD", "USA": "US", "UK": "GB"}


def flag_of(ch):
    f = ch.get("flag")
    if f is False: return None
    return f or FLAGS.get(ch["name"].upper())


def ticker(cv, ch, val, pct, x, y, a, sz=92, sub_sz=46, show_sub=True):
    """[flag] NAME  value  [arrow] pct%   (left aligned at x, centred on y)"""
    fl = flag_of(ch)
    if "_tsz" not in ch:                              # one text size per chart: the widest row it will show fits the panel (x 1010)
        p0 = ch["pts"][0][1]; vals = [q[1] for q in ch["pts"]]
        def row_w(z):
            w = (z * 1.26 if fl else 0) + min(560, cv.tw(ch["name"], z)) + 34
            ws = [cv.tw(fmt(v), z) + (40 + z * 0.7 + cv.tw(f"{abs(v / p0 - 1) * 100:.0f}%", z) if k else 0) for k, v in enumerate(vals)]
            ws.append(cv.tw(fmt(vals[-1], ch.get("prefix_last", "")), z) + 40 + z * 0.7 + cv.tw(f"{abs(vals[-1] / p0 - 1) * 100:.0f}%", z))
            return w + max(ws)
        z = sz
        for _ in range(3):
            w = row_w(z)
            if x + w > 1010: z = z * (1010 - x) / w
        ch["_tsz"] = min(sz, z)
    sub_y = y + sz * 0.9; sz = ch["_tsz"]               # the sub line keeps its place
    if fl:                                            # flag sized to the name, left of it; the row shifts right
        fw = sz * 1.04; WG.flag(cv, fl, x + fw / 2, y, sz * 0.9, a); x += fw + sz * 0.22
    if pct is None:                                   # before the line moves: name + first value, neutral
        cv.text(ch["name"], x, y, sz, WHITE, a, align="l", maxw=560)
        cv.text(fmt(val), x + min(560, cv.tw(ch["name"], sz)) + 34, y, sz, "#C9D6EA", a, align="l", maxw=420)
        if show_sub and ch.get("sub"): cv.text(ch["sub"], x, sub_y, sub_sz, "#C9D6EA", a, align="l", bold=False, maxw=940)
        return
    up = pct >= 0; col = UP if up else DOWN
    cv.text(ch["name"], x, y, sz, WHITE, a, align="l", maxw=560)
    x2 = x + min(560, cv.tw(ch["name"], sz)) + 34
    vs = fmt(val, ch.get("prefix_last", "") if ch.get("_done") else "")
    cv.text(vs, x2, y, sz, col, a, align="l", maxw=420); x3 = x2 + cv.tw(vs, sz) + 40
    tri(cv, x3 + sz * 0.3, y, sz * 0.3, up, col, a)
    cv.text(f"{abs(pct):.0f}%", x3 + sz * 0.7, y, sz, col, a, align="l", maxw=360)
    if show_sub and ch.get("sub"): cv.text(ch["sub"], x, sub_y, sub_sz, "#C9D6EA", a, align="l", bold=False, maxw=940)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    CX0, CX1, CY0, CY1 = 140, 1000, 640, 1330          # chart plot area
    CH = []
    for c in D["charts"]:
        pts = [(float(p[0]), float(p[1]), p[2] if len(p) > 2 else None, bool(p[3]) if len(p) > 3 else False) for p in c["points"]]
        vmax = nice_max(max(p[1] for p in pts))
        d0, d1 = A(c["draw"][0]), A(c["draw"][1])
        xr = xrange_of(c, pts)
        CH.append(dict(c, pts=pts, vmax=vmax, xr=xr, xt=xticks_of(c, *xr), t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6), d0=d0, d1=d1,
                       pct=(pts[-1][1] / pts[0][1] - 1) * 100))
    # the line head moves at constant speed along x between d0 and d1 (time per year constant)
    def head(ch, t):
        p = ch["pts"]; u = min(1, max(0, (t - ch["d0"]) / max(0.1, ch["d1"] - ch["d0"])))
        yr = p[0][0] + (p[-1][0] - p[0][0]) * u
        for i in range(len(p) - 1):
            if yr <= p[i + 1][0] + 1e-9:
                f = (yr - p[i][0]) / max(1e-9, p[i + 1][0] - p[i][0]); return yr, i, f, p[i][1] + (p[i + 1][1] - p[i][1]) * f
        return p[-1][0], len(p) - 2, 1.0, p[-1][1]

    # map (rendered once)
    M_ = D["map"]; MR = MapR(); lon, lat, wd, my = M_["view"]
    cosf = max(0.6, math.cos(math.radians(min(62, abs(lat)))))
    mimg, MM = MR.render(MR.view(lon, lat, W / (wd * cosf), cosf, my), H, hi=tuple(M_.get("hi", ())), hi_a={k: 0.75 for k in M_.get("hi", ())})
    mimg = mimg.astype(np.float32); geo = lambda lo, la: geo2px(MM, lo, la)
    map_until = A(M_.get("until"), 1.6); map_again = A(M_.get("again"), T.END + 1)
    def map_layer(t):
        u = t if t < map_again else (t - map_again)
        z = 1.0 + 0.03 * min(1, u / 4.0); Mz = np.float32([[z, 0, 540 * (1 - z)], [0, z, 960 * (1 - z)]])
        return cv2.warpAffine(mimg, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), z
    def map_alpha(t):
        if t >= map_again - 0.3: return seg(t, map_again - 0.3, map_again)
        return 1.0 - seg(t, map_until - 0.3, map_until)

    PH = []
    if D.get("photos"):
        from photo_story import _load, _kb
        for p in D["photos"]: PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))))
        y_ = np.arange(H, dtype=np.float32)[:, None]
        VIG = (1 - np.maximum(np.clip(1 - y_ / 560, 0, 1) ** 1.5 * 0.62, np.clip((y_ - 1150) / 770, 0, 1) ** 1.1 * 0.7))[..., None]
    def ph_alpha(p, t): return min(seg(t, p["t0"] - 0.25, p["t0"]), 1 - seg(t, p["t1"] - 0.25, p["t1"]))
    def ph_layer(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        dim = p.get("dim", 0.3); return img * (1 - dim) * VIG + DARK * dim
    RC = D.get("recap"); rc0, rc1 = (A(RC["at"]), A(RC.get("until"), T.END - 0.6)) if RC else (1e9, 1e9)
    cards = screen_cards(S, T); SUBS = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 300)
    E = S.get("end") or {}; e0 = A(E.get("at"), T.END - 3) if E else 1e9; e1 = A(D.get("end_until"), T.END - 0.5)

    def bg(t):
        b = bars_moving_bg(t) * 0.55 + DARK * 0.45; return b

    def draw_chart(cv, ch, t, a):
        # grid + y labels
        vm = ch["vmax"]; xa_, xb_ = ch["xr"]; xp = lambda yr: CX0 + (yr - xa_) / (xb_ - xa_) * (CX1 - CX0)
        for k in range(5):
            v = vm * k / 4; y = CY1 - (CY1 - CY0) * k / 4
            cv.line([(CX0, y), (CX1, y)], "#2A3A55", 2, a)
            cv.text(fmt(v), CX0 - 14, y, 38, "#8FA3C0", a, align="r", bold=False, maxw=130, shadow=False)
        for yr in ch["xt"]:
            x = xp(yr); cv.line([(x, CY0), (x, CY1)], "#22314A", 2, a * 0.8)
            cv.text(str(yr), x, CY1 + 44, 40, "#8FA3C0", a, bold=False, shadow=False)
        yv = lambda v: CY1 - (CY1 - CY0) * v / vm
        yr, i, f, hv = head(ch, t); p = ch["pts"]
        if t < ch["d0"] - 0.05: return hv
        path = [(xp(q[0]), yv(q[1])) for q in p[:i + 1]] + [(xp(yr), yv(hv))]
        # area under the line, per segment colour
        c = cv.c
        for k in range(len(path) - 1):
            (xa, ya), (xb, yb) = path[k], path[k + 1]
            col = scol(p[k][1], p[min(k + 1, len(p) - 1)][1])
            c.new_path(); c.move_to(xa, CY1); c.line_to(xa, ya); c.line_to(xb, yb); c.line_to(xb, CY1); c.close_path(); cv.col(col, 0.16 * a); c.fill()
        for k in range(len(path) - 1):
            col = scol(p[k][1], p[min(k + 1, len(p) - 1)][1])
            cv.line([path[k], path[k + 1]], col, 12, a)
        # points reached: dot + value label
        for k, q in enumerate(p):
            if q[0] > yr + 1e-6: break
            x, y = xp(q[0]), yv(q[1]); col = scol(q[1], p[1][1]) if k == 0 else scol(p[k - 1][1], q[1])
            pa = a * seg(t, ch["d0"] + (ch["d1"] - ch["d0"]) * (q[0] - p[0][0]) / max(1e-9, p[-1][0] - p[0][0]) - 0.02, 0.25 + ch["d0"] + (ch["d1"] - ch["d0"]) * (q[0] - p[0][0]) / max(1e-9, p[-1][0] - p[0][0]))
            cv.circ(x, y, 13, col, pa, stroke="#FFFFFF", lw=4)
            lab = q[2] if q[2] is not None else fmt(q[1]); ly = y - 54 if y - 54 > CY0 - 30 else y + 54
            lx = min(max(x, CX0 + 90), CX1 - 70)
            if lab: cv.text(lab, lx, ly, 46, WHITE, pa, maxw=260)
            if q[3]: cv.text("est.", lx, ly + (44 if ly < y else 40), 38, "#C9D6EA", pa, bold=False, maxw=200)
        hx, hy = path[-1]; cv.circ(hx, hy, 9, "#FFFFFF", a)
        return hv

    def frame(t):
        base = bg(t)
        ma = map_alpha(t)
        if ma > 0.001: ml, _ = map_layer(t); base = base * (1 - ma) + ml * ma
        pa = 0.0
        for p in PH:
            a_ = ph_alpha(p, t)
            if a_ > 0.001: base = base * (1 - a_) + ph_layer(p, t) * a_; pa = 1 - (1 - pa) * (1 - a_)
        cv = Canvas(base)
        # map labels and pins
        if ma > 0.01:
            z = map_layer(t)[1]
            zp = lambda lo, la: (540 + (geo(lo, la)[0] - 540) * z, 960 + (geo(lo, la)[1] - 960) * z)
            for L in M_.get("labels", []):
                x, y = zp(L["lon"], L["lat"]); cv.text(L["text"], x, y, L.get("size", 58), L.get("col", WHITE), ma, maxw=500)
            for P in M_.get("pins", []):
                x, y = zp(P["lon"], P["lat"]); WG.pin(cv, x, y, YEL, ma, t)
                if P.get("label"): cv.text(P["label"], x + P.get("dx", 30), y + P.get("dy", 0), P.get("size", 50), YEL, ma, align="l", maxw=500)
        # charts
        for ch in CH:
            a = win(t, ch["t0"] - 0.05, ch["t1"], 0.25, 0.25) * (1 - pa)
            if a <= 0.01: continue
            cv.rrect(40, 250, 1000, 1180, 40, "#0A1222", 0.55 * a)
            draw_chart(cv, ch, t, a)
            # the ticker steps to the last REAL data point the line has reached (no interpolated values on screen)
            yr = head(ch, t)[0]; reached = [q for q in ch["pts"] if q[0] <= yr + 1e-6] or ch["pts"][:1]
            ch["_done"] = len(reached) == len(ch["pts"])
            pct = (reached[-1][1] / ch["pts"][0][1] - 1) * 100 if len(reached) > 1 else None
            ticker(cv, ch, reached[-1][1], pct, 90, 340, a)
        # recap: every line as a sparkline row
        ra = win(t, rc0, rc1, 0.3, 0.3) * (1 - pa) if RC else 0
        if ra > 0.01:
            y0, rh = RC.get("y0", 520), RC.get("rowh", 190)
            for k, ch in enumerate(CH):
                rka = ra * seg(t, rc0 + 0.12 * k, rc0 + 0.12 * k + 0.3); y = y0 + k * rh
                if ch.get("hl"): cv.rrect(40, y - rh / 2 + 8, 1000, rh - 16, 26, UP, 0.16 * rka, stroke=UP, lw=3, sa=0.8 * rka)
                else: cv.rrect(40, y - rh / 2 + 8, 1000, rh - 16, 26, "#0A1222", 0.6 * rka)
                ch["_done"] = True
                up = ch["pct"] >= 0; col = UP if up else DOWN
                fl = flag_of(ch); nx = 80
                if fl: WG.flag(cv, fl, 80 + 29, y - 30, 50, rka); nx = 80 + 58 + 12
                cv.text(ch.get("short", ch["name"]), nx, y - 30, 56, WHITE, rka, align="l", maxw=540 - nx)
                vs = fmt(ch["pts"][-1][1], ch.get("prefix_last", "")); cv.text(vs, 80, y + 36, 54, col, rka, align="l", maxw=240)
                xx = 80 + cv.tw(vs, 54) + 30; tri(cv, xx + 16, y + 36, 16, up, col, rka)
                cv.text(f"{abs(ch['pct']):.0f}%", xx + 40, y + 36, 54, col, rka, align="l", maxw=220)
                p = ch["pts"]; sx0, sx1, sy0, sy1 = 560, 1000, y - rh / 2 + 34, y + rh / 2 - 34
                spx = lambda yr, xr=ch["xr"]: sx0 + (yr - xr[0]) / (xr[1] - xr[0]) * (sx1 - sx0); spy = lambda v: sy1 - (sy1 - sy0) * v / ch["vmax"]
                cv.line([(sx0, sy1), (sx1, sy1)], "#2A3A55", 2, rka)
                for j in range(len(p) - 1):
                    cv.line([(spx(p[j][0]), spy(p[j][1])), (spx(p[j + 1][0]), spy(p[j + 1][1]))], scol(p[j][1], p[j + 1][1]), 8, rka)
        # photo credits
        for k_, p in enumerate(PH):
            a_ = ph_alpha(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_alpha(q, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.85 * a_, bold=False, maxw=1000, minsz=36)
        if ma > 0.01 and M_.get("credit"):
            cv.text(M_["credit"], 540, D.get("credit_y", 1712), 36, "#C9D6EA", 0.8 * ma, bold=False, maxw=1000, minsz=36)
        draw_screen(cv, [c for c in cards if c[1] > T.E(0)], t, D.get("title_y", 300), 72)
        draw_sources(cv, S, T, t, D.get("src_y", 1488))
        if E:
            a = win(t, e0 - 0.1, e1, 0.35, 0.3)
            if a > 0.01:
                y = E.get("y", 1010); WG.panel(cv, 60, y - 230, 960, 460, a)
                if E.get("balance"): WG.title(cv, E["balance"], y - 90, a, 66)
                if E.get("cta"):
                    ca = a * seg(t, e0 + 0.6, e0 + 1.0); tw = min(940, cv.tw(E["cta"].replace("*", ""), 60) + 90)
                    cv.rrect(540 - tw / 2, y + 60, tw, 120, 60, SKY, ca); WG.title(cv, E["cta"], y + 120, ca, 60)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, map_until - 0.55, HY, 0.9)
        if t >= map_again - 0.3: out = HK.paste(out, hook_img, 540, HY, 0.9, map_alpha(t))   # same hook card: seamless loop
        c2 = Canvas(out); draw_subs(c2, SUBS, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -30) for p in PH]
    for ch in CH:
        ev.append((ch["t0"], "whoosh", -28)); ev.append((ch["d0"], "roll", ch["d1"]))
        p = ch["pts"]
        for q in p: ev.append((ch["d0"] + (ch["d1"] - ch["d0"]) * (q[0] - p[0][0]) / max(1e-9, p[-1][0] - p[0][0]), "tick", -20))
        ev.append((ch["d1"], "ding" if ch["pct"] >= 0 else "thud", -24))
    if RC: ev.append((rc0, "pop", -24))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
