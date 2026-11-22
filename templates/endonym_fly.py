"""endonym_fly: "what countries call themselves" (endonyms). Dark minimal full-bleed world map (cine_timeline look), the
camera flies country to country; at each stop the country lights up in sky #45A4DD, its English name appears and gets
struck through, and the native name lands huge in yellow (pre-rendered, shaped PNG) with a small transliteration and an
optional meaning line in sky. Big white karaoke subtitle one word at a time (lower third), small "ISRAEL FACTS" mark,
small source label, hook from frame 0 over a wide view that is already moving (first flight lands by about 1.4 s),
screen lines (debate question, follow line, "Part 2 on my profile") as plain shadowed text in the card area, balance
line, circular ending (the camera flies back to the opening view and the last 0.5 s dissolves into frame 0).
Whole land of Israel always drawn: 1949 lines a muted slate, West Bank and Gaza a slightly different slate, Green Line
green; when Israel is a stop both light up in sky shades.

data keys (times are anchors: 12.5, "L3", "L3:word", "END-2"):
  start   [lon, lat, k]   camera at frame 0 (k = px per degree); the loop flies back here
  fly     flight seconds between stops (default 1.0)
  stops   [{at, code, cam: [lon, lat, k], en, img, tr, mean, strike_at, strike: true}]
          code: ISO3 (ISR = the whole land of Israel); img: PNG under templates/data/endonyms/ (no .png);
          en struck through at strike_at (default at + 1.0); native PNG pops at strike_at; tr (transliteration, 64 px
          white) and mean (64 px sky) below it. A stop without en/img is a camera move only (e.g. the pull back).
  card_y  [en_y, native_y, tr_y] (default [250, 430, 600]); subs_y (default 1590), subs_size (default 116)
  src_y   source label y (default 1800); the map centre sits at y 1110, below the card
"""
import os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

SEA, LANDC, BORDER = "#0A0C10", "#1B1E23", "#7B8494"
ISR_C, TERR_C, GREENLINE = "#33404F", "#3B4A5C", "#3FBF7F"
SKY2 = "#7CC3EC"
CY = 1110.0                                                        # map centre y: the country sits below the name card
NAMES_DIR = os.path.join(TPL, "data", "endonyms")
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.5 * np.clip(((_xx - W / 2) / (W * 0.75)) ** 2 + ((_yy - H * 0.5) / (H * 0.65)) ** 2, 0, 1))[..., None].astype(np.float32)
del _yy, _xx
_y = np.arange(H, dtype=np.float32)[:, None, None]
SHADE = (1 - 0.45 * np.clip(1 - _y / 820, 0, 1) ** 1.2 - 0.45 * np.clip((_y - 1480) / 440, 0, 1) ** 1.2).astype(np.float32)
del _y


class WorldMap:
    def __init__(s): s.m = MapR()
    def rings(s, code, k): return s.m.country(code) if (k >= 50 and code in s.m.local) else s.m.world[code]
    def render(s, cam, hi):
        """cam (lon, lat, k); hi {code: alpha} lit in sky -> float32 BGR"""
        lon, lat, k = cam; v = mt.View((lon, lat), (540.0, CY), k, 0.0, math.cos(math.radians(max(-70, min(70, lat)))))
        M = v.M(); inv = np.linalg.inv(M); c = mt.apply(inv, np.array([[-40, -40], [W + 40, -40], [-40, H + 40], [W + 40, H + 40]], float))
        vb = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
        cv = np.empty((H, W, 3), np.uint8); cv[:] = mt.hexc(SEA).astype(np.uint8)
        land, bord = mt.hexc(LANDC), mt.hexc(BORDER); lw = max(1.0, min(k / 90, 2.2))
        for code, rings in s.m.world.items():
            if code in ("ISR", "PSE"): continue
            mt.fill(cv, M, rings, land, 1.0, vb, outline=bord, lw=lw, oalpha=0.5)
        for code, a in hi.items():
            if code == "ISR" or a <= 0.01: continue
            mt.fill(cv, M, s.rings(code, k), mt.hexc(SKY), 0.95 * a, vb, outline=mt.hexc("#BFE3F7"), lw=lw + 0.6, oalpha=0.9 * a)
        ia = hi.get("ISR", 0.0)
        mix = lambda c0, c1: mt.hexc(c0) * (1 - ia) + mt.hexc(c1) * ia
        mt.fill(cv, M, s.m.isr, mix(ISR_C, SKY), 1.0, vb, outline=bord, lw=lw, oalpha=0.7)
        mt.fill(cv, M, s.m.terr, mix(TERR_C, SKY2), 1.0, vb, outline=mt.hexc(GREENLINE), lw=max(2, min(k / 60, 4)), oalpha=0.95)
        return cv.astype(np.float32)


def _png(key):
    p = os.path.join(NAMES_DIR, key + ".png")
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None: raise SystemExit(f"endonym_fly: missing {p}")
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)


def _lerp_cam(c0, c1, u):
    """fly: lon/lat eased, zoom geometric with a dip in the middle (pull out, then dive in)"""
    e = ease(u); far = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))
    k = math.exp(math.log(c0[2]) * (1 - e) + math.log(c1[2]) * e)
    dip = min(0.6, 0.012 * far + 0.15) * math.sin(math.pi * u)
    return [c0[0] + (c1[0] - c0[0]) * e, c0[1] + (c1[1] - c0[1]) * e, k * (1 - dip)]


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    FLY = float(D.get("fly", 1.0)); START = D.get("start", [20, 40, 12])
    ST = []
    for st in D["stops"]:
        st = dict(st); st["t0"] = A(st["at"])
        st["ts"] = A(st["strike_at"]) if st.get("strike_at") else st["t0"] + FLY + 0.35
        if st.get("img"): st["im"] = _png(st["img"])
        ST.append(st)
    for k, st in enumerate(ST): st["t1"] = ST[k + 1]["t0"] if k + 1 < len(ST) else T.END + 1
    LOOP0 = T.END - 1.6                                            # fly back to the opening view for the loop

    def cam_at(t):
        if t >= LOOP0:
            last = ST[-1]; c0 = cam_after(last, LOOP0); return _lerp_cam(c0, START, min(1.0, (t - LOOP0) / 1.5))
        k = max([i for i, st in enumerate(ST) if t >= st["t0"]] or [-1])
        if k < 0: return list(START)
        st = ST[k]; prev = cam_after(ST[k - 1], st["t0"]) if k else list(START)
        u = (t - st["t0"]) / FLY
        if u < 1: return _lerp_cam(prev, st["cam"], u)
        return cam_after(st, t)

    def cam_after(st, t):                                           # slow push after landing
        u = min(1.0, max(0.0, (t - st["t0"] - FLY) / max(0.5, st["t1"] - st["t0"] - FLY)))
        c = st["cam"]; return [c[0], c[1], c[2] * (1 + 0.05 * u)]

    def hi_at(t):
        hi = {}
        for k, st in enumerate(ST):
            if not st.get("code"): continue
            same = k + 1 < len(ST) and ST[k + 1].get("code") == st["code"]
            a0 = st["t0"] + FLY * 0.6 if not (k and ST[k - 1].get("code") == st["code"]) else st["t0"] - 1.0
            a = win(t, a0, st["t1"] + (FLY + 1.0 if same else 0.25), 0.35, 0.35)
            if t >= LOOP0: a *= 1 - seg(t, LOOP0, LOOP0 + 0.5)
            hi[st["code"]] = max(hi.get(st["code"], 0), a)
        return hi

    WM = WorldMap()
    # karaoke words
    repl = S.get("subs_replace") or {}
    WD = []
    for l in T.L:
        ws = l["words"]; i = 0
        while i < len(ws):
            hit = None
            for a_, b_ in repl.items():
                n = len(a_.split())
                if " ".join(w["w"] for w in ws[i:i + n]).rstrip(".,!?;:") == a_ or " ".join(w["w"] for w in ws[i:i + n]) == a_:
                    hit = (b_, n); break
            if hit: WD.append([hit[0], ws[i]["t0"], ws[i + hit[1] - 1]["t1"]]); i += hit[1]
            else: WD.append([ws[i]["w"], ws[i]["t0"], ws[i]["t1"]]); i += 1
    for w in WD: w[0] = re.sub(r"[\"“”֑-ׇ]", "", w[0]).strip().rstrip(",;:.?!")
    for j in range(len(WD)):
        nxt = WD[j + 1][1] if j + 1 < len(WD) else WD[j][2] + 0.5
        WD[j][2] = min(nxt - 0.01, max(WD[j][2], WD[j][1] + 0.18) + 0.25) if nxt > WD[j][1] else WD[j][2]
    cards = [c for c in screen_cards(S, T) if c[1] > T.E(0) - 0.05]
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 560)
    E = S.get("end") or {}; END_AT = A(E.get("at"), T.S(len(T.L) - 2)) if E else None
    EY, NY, TY = D.get("card_y", [250, 430, 600]); SUBY, SUBS = D.get("subs_y", 1590), D.get("subs_size", 116)
    SRCY = D.get("src_y", 1800)

    def frame(t):
        base = WM.render(cam_at(t), hi_at(t)) * VIG ** 0.7
        cv = Canvas(base * SHADE)
        cv.text("ISRAEL FACTS", 64, 120, 36, "#C9CED6", 0.8, bold=True, align="l", shadow=True, maxw=500)
        for st in ST:                                              # the name card
            if not st.get("en") and st.get("im") is None: continue
            ca = win(t, st["t0"] + FLY * 0.55, st["t1"] + 0.05, 0.3, 0.25)
            if t >= LOOP0: ca *= 1 - seg(t, LOOP0 - 0.2, LOOP0 + 0.2)
            if ca <= 0.01: continue
            if st.get("en"):
                strike = st.get("strike", True)
                ea = ca * (1 - 0.5 * seg(t, st["ts"] + 0.25, st["ts"] + 0.6)) if strike else ca
                sz = st.get("en_size", 96); cv.text(st["en"], 540, EY, sz, WHITE, ea, bold=True, maxw=980)
                if strike:
                    u = seg(t, st["ts"] - 0.05, st["ts"] + 0.3)
                    if u > 0:
                        w = min(980, cv.tw(st["en"], sz)) + 30
                        cv.line([(540 - w / 2, EY + 2), (540 - w / 2 + w * u, EY + 2)], "#E5484D", 10, ca)
            if st.get("im") is not None:
                u = (t - st["ts"]) / 0.4
                if u > 0:
                    sc = 0.7 + 0.3 * back(min(1.0, u)); a = ca * min(1.0, u * 2.5)
                    cv.image(st["im"], 540, NY, sc, a)
                    if st.get("tr"): cv.text(st["tr"], 540, TY, 64, WHITE, a * seg(t, st["ts"] + 0.2, st["ts"] + 0.5), bold=True, maxw=1000, minsz=44)
                    if st.get("mean"): cv.text(st["mean"], 540, TY + 86, 64, SKY, a * seg(t, st["ts"] + 0.4, st["ts"] + 0.7), bold=True, maxw=1000, minsz=44)
        for s_ in S["sources"]:
            t0 = A(s_.get("at"), T.S(min(1, len(T.L) - 1))); t1 = A(s_.get("until"), T.END - 0.6); sa = win(t, t0, t1, 0.25, 0.25)
            if sa > 0.01:
                lab = f"{s_['label']}, {s_['year']}" if s_.get("year") and str(s_["year"]) not in s_["label"] else s_["label"]
                cv.text("Source: " + lab, 540, SRCY, 36, "#C9CED6", 0.85 * sa, bold=False, maxw=1000, minsz=36)
        for txt, t0, t1, yy, ss in cards:
            a = win(t, t0 - 0.1, t1, 0.25, 0.25)
            if a > 0.01: WG.title(cv, txt.replace("|", " "), yy or 500, a, ss or 84)
        if E and END_AT is not None:
            ea = win(t, END_AT - 0.1, A(E.get("until"), T.END - 0.5), 0.35, 0.3)
            if ea > 0.01 and E.get("balance"): WG.title(cv, E["balance"], E.get("y", 760), ea, 68)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); hook_on = t < T.E(0) + 0.35
        for w, a_, b_ in WD:
            if hook_on and a_ < T.E(0) + 0.05: continue
            if a_ - 0.03 <= t < b_ and w:
                pop = 1.0 + 0.08 * (1 - eout((t - a_ + 0.03) / 0.09)); s2 = SUBS * pop
                c2.face(True)
                while c2.tw(w, s2) > 1000 and s2 > 60: s2 -= 4
                for dx, dy, al in ((0, 8, 0.28), (0, 4, 0.35), (3, 3, 0.45)):
                    c2.col("#000000", al); c2.c.set_font_size(s2); e = c2.c.text_extents(w)
                    c2.c.move_to(540 - e.x_advance / 2 + dx, SUBY + s2 * 0.35 + dy); c2.c.show_text(w)
                c2.text(w, 540, SUBY, s2, WHITE, 1.0, bold=True, maxw=1000, shadow=False, minsz=60)
                break
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(st["t0"], "whoosh", -32) for st in ST] + [(st["ts"], "tick", -24) for st in ST if st.get("im") is not None]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
