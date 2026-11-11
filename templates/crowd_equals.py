"""crowd_equals (@noamroz__ data story): "1 Israeli = N <neighbours>". One sky-blue stick figure on the left, a regional
map on top with the beat's country tinted sky (never red: the contrast is economies, not peoples), and for each beat a
neutral white crowd of N figures pops in on the right (fractional N: the last figure is cut, so 8.8 shows 8.8 figures)
while the equation counts up. Bar cards (ranking, resources) take the middle of the screen between beats. Circular:
the end card clears before the last line, the screen goes back to the single figure and dissolves into frame 0.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  cam         [lon, lat, width_deg]  map view (default [40, 25, 40]); map_band [y0, y1] (default [320, 965]), map_y = screen y of the view centre (default 630)
  israel      {label: "Israeli", flag: "IL", tag: "$54,213", lon, lat}   the single figure + its value tag on the map
  beats       [{code (ISO3), name, demonym, flag, ratio, txt, dec, tag, lon, lat, at, until}]
               ratio = figures in the crowd (exact, fractional ok), txt = the number shown at the end of the count
  bars        [{at, until, title, y, rows: [{label, flag, value, txt, hl}], vmax, lab_w}]   cards over a dimmed scene
  icons       [{at, until, emo | flag, x, y, size, fig}]   pop-in emoji or flags on the scene (fig: a white figure under it)
  hook_y      default 630 (over the map at frame 0)
  end_until   anchor when the end card clears (default END-0.5); the single figure stays until the loop
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

CROWD = (300, 1090, 1040, 1435)        # x0, y0, x1, y1 (feet of the last row on y1)
FIG_H = 172.0                          # figure() height at sc = 1 (feet to head top)


def grid(n):
    x0, y0, x1, y1 = CROWD; w, h = x1 - x0, y1 - y0; k = max(1, math.ceil(n))
    best = None
    for cols in range(1, k + 1):
        rows = math.ceil(k / cols); cell = min(w / cols, h / rows / 1.35)
        if best is None or cell > best[0]: best = (cell, cols, rows)
    cell, cols, rows = best; sc = min(1.05, cell * 1.12 / FIG_H)
    gw, gh = cols * cell, rows * cell * 1.35; ox = x0 + (w - gw) / 2 + cell / 2; oy = y1 - (h - gh) / 2
    pts = []
    for i in range(k):
        r, c = divmod(i, cols); pts.append((ox + c * cell, oy - (rows - 1 - r) * cell * 1.35))
    return pts, sc, cell


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    B = [dict(b) for b in D["beats"]]
    for i, b in enumerate(B):
        b["t0"] = A(b["at"]); b["t1"] = A(b["until"]) if b.get("until") else None
    for i, b in enumerate(B):
        if b["t1"] is None: b["t1"] = A(B[i + 1]["at"]) if i + 1 < len(B) else T.END - 0.6
        b["pts"], b["sc"], b["cell"] = grid(b["ratio"]); n = len(b["pts"])
        b["pop"] = min(1.6, max(0.5, (b["t1"] - b["t0"]) * 0.45)); b["n"] = n
    BARS = [dict(x, t0=A(x["at"]), t1=A(x.get("until"), T.END - 0.6)) for x in D.get("bars", [])]
    for x in BARS:
        for k, r in enumerate(x["rows"]): r["_at"] = A(r["at"]) if r.get("at") is not None else x["t0"] + 0.3 + 0.12 * k
    E = S.get("end") or {}; END0 = A(E.get("at"), T.END - 3) if E else None; END1 = A(D.get("end_until"), T.END - 0.5)
    DIM = [(x["t0"], x["t1"]) for x in BARS] + ([(END0, END1)] if E else [])
    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 630)
    IL = D.get("israel", {})
    ICONS = [dict(ic, t0=A(ic["at"]), t1=A(ic.get("until"), T.END - 0.6)) for ic in D.get("icons", [])]
    HOOK_OFF = T.E(0) + 0.5

    # static map band
    lon, lat, wd = D.get("cam", [40, 25, 40]); my = D.get("map_y", 630)
    cosf = max(0.6, math.cos(math.radians(min(62, abs(lat))))); MR = MapR()
    mimg, M = MR.render(MR.view(lon, lat, W / (wd * cosf), cosf, my), H)
    yy = np.arange(H, dtype=np.float32)[:, None]
    mb0, mb1 = D.get("map_band", [320, 965])
    mask = (np.clip((yy - mb0) / 70, 0, 1) * np.clip((mb1 - yy) / 70, 0, 1))[..., None]
    geo = lambda lo, la: geo2px(M, lo, la)

    def tint(cv, code, a):
        c = cv.c; c.new_path()
        for ring, _ in MR.country(code):
            pts = [geo(*p) for p in ring]; c.move_to(*pts[0])
            for p in pts[1:]: c.line_to(*p)
            c.close_path()
        cv.col(SKY, 0.62 * a); c.fill_preserve(); cv.col("#FFFFFF", 0.85 * a); c.set_line_width(2.5); c.stroke(); c.new_path()

    def scene_a(t):
        return 1 - 0.85 * max([win(t, a0 - 0.2, a1, 0.35, 0.35) for a0, a1 in DIM] or [0])

    def frame(t):
        bg = bars_moving_bg(t, 10)
        dm = 1 - scene_a(t)
        base = bg * (1 - mask) + mimg.astype(np.float32) * mask * (1 - 0.55 * dm)
        cv = Canvas(base); sa = scene_a(t)
        # beat country on the map + value tags
        for b in B:
            ba = win(t, b["t0"] - 0.1, b["t1"], 0.3, 0.25)
            if ba > 0.01:
                if t < HOOK_OFF: ba *= seg(t, HOOK_OFF - 0.3, HOOK_OFF)
                tint(cv, b["code"], ba * sa)
                if b.get("tag"):
                    x, y = geo(b["lon"], b["lat"]); WG.tag(cv, b["tag"], x, y, 46, WHITE, ba * sa)
        if IL.get("tag"):
            ia = sa * max([win(t, b["t0"] - 0.1, b["t1"], 0.3, 0.25) for b in B] + [0]) * seg(t, HOOK_OFF - 0.3, HOOK_OFF)
            if ia > 0.01:
                x, y = geo(IL.get("lon", 34.9), IL.get("lat", 31.6)); WG.tag(cv, IL["tag"], x, y, 46, YEL, ia)
        # the single Israeli (always there, the loop lands on it)
        fx, fy, fs = 150, 1430, 1.45
        bob = 4 * math.sin(t * 2.4)
        figure(cv, fx, fy + bob * 0, fs, SKY, badge=True, arms="down", a=0.25 + 0.75 * sa, lw=7)
        if IL.get("flag"): WG.flag(cv, IL["flag"], fx, fy - FIG_H * fs - 50, 64, 0.25 + 0.75 * sa)
        # crowd + equation for the live beat
        for b in B:
            ba = win(t, b["t0"] - 0.05, b["t1"], 0.15, 0.25) * sa
            if ba <= 0.01: continue
            n = b["n"]; fr = b["ratio"] - math.floor(b["ratio"]) if b["ratio"] % 1 else 0.0
            for i, (x, y) in enumerate(b["pts"]):
                ti = b["t0"] + b["pop"] * (i / max(1, n - 1)) ** 0.8
                if t < ti: break
                p = WG.pop_s(t, ti); s_ = b["sc"] * max(0.05, p)
                last_cut = (i == n - 1 and fr > 0)
                if last_cut:
                    cv.c.save(); cv.c.rectangle(x - b["cell"], 0, b["cell"] * (2 * fr), H); cv.c.clip()
                figure(cv, x, y, s_, WHITE, badge=False, a=0.92 * ba, lw=max(8.0, 4.0 / b["sc"]))
                if last_cut: cv.c.restore()
            u = seg(t, b["t0"], b["t0"] + b["pop"], eout); v = 1 + (b["ratio"] - 1) * u
            num = b["txt"] if u >= 0.999 else (f"{v:.{b.get('dec', 0)}f}")
            cv.rich([("1 ", YEL), (IL.get("label", "Israeli") + " = ", WHITE), (num + " ", YEL), (b["demonym"], WHITE)], 540, 1018, 84, ba, maxw=1000)
        # bar cards
        for x in BARS:
            xa = win(t, x["t0"] - 0.1, x["t1"], 0.4, 0.35)
            if xa > 0.01:
                n = len(x["rows"]); rowh = x.get("rowh", min(96, 760 // max(1, n))); y0 = x.get("y", 560)
                WG.panel(cv, 40, y0 - 80, 1000, rowh * (n - 1) + 160, xa)
                if x.get("title"): WG.title(cv, x["title"], y0 - 190, xa, 72)
                rows = [dict(label=r["label"], flag=r.get("flag"), value=float(r["value"]), at=r["_at"], txt=r.get("txt"),
                             color=YEL if r.get("hl") else r.get("color", SKY), tcol=YEL if r.get("hl") else WHITE) for r in x["rows"]]
                WG.bars(cv, rows, t, xa, y0, rowh=rowh, vmax=x.get("vmax"), x0=60, w=960, lab_w=x.get("lab_w", 380))
                if x.get("note"): cv.text(x["note"], 540, y0 + rowh * (n - 1) + 110, 38, "#C9D6EA", xa * seg(t, x["t0"] + 0.8, x["t0"] + 1.2), bold=False, maxw=960)
        for ic in ICONS:
            ia = win(t, ic["t0"] - 0.05, ic["t1"], 0.2, 0.3) * sa
            if ia > 0.01:
                p = WG.pop_s(t, ic["t0"]); sz = ic.get("size", 110) * max(0.05, p)
                if ic.get("flag"): WG.flag(cv, ic["flag"], ic["x"], ic["y"], sz, ia)
                else: WG.emo(cv, ic["emo"], ic["x"], ic["y"], sz, ia)
                if ic.get("fig"): figure(cv, ic["x"], ic["y"] + 230, 0.85, WHITE, badge=False, a=0.92 * ia, lw=8)
        draw_screen(cv, cards, t, D.get("title_y", 270), 76)
        draw_sources(cv, S, T, t)
        if E:
            ea = win(t, END0 - 0.1, END1, 0.35, 0.3)
            if ea > 0.01:
                y = E.get("y", 900); WG.panel(cv, 60, y - 230, 960, 460, ea)
                if E.get("balance"): WG.title(cv, E["balance"], y - 90, ea, 66)
                if E.get("cta"):
                    ca = ea * seg(t, END0 + 0.6, END0 + 1.0); cv.rrect(80, y + 40, 920, 160, 60, SKY, ca)
                    WG.title(cv, E["cta"], y + 120, ca, 54, maxw=860)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = []
    for b in B:
        ev.append((b["t0"], "whoosh", -28))
        for i in range(min(b["n"], 10)): ev.append((b["t0"] + b["pop"] * (i / max(1, b["n"] - 1)) ** 0.8, "pop", -30))
        ev.append((b["t0"] + b["pop"], "ding", -30))
    ev += [(x["t0"] - 0.2, "whoosh", -26) for x in BARS] + ([(END0, "thud", -26)] if E else [])
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, beats=[(b["code"], b["ratio"]) for b in B])
