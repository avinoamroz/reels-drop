"""cine_timeline: "history in 60 seconds" documentary look (style of the viral remerged.ai history reels, our rules).

Full bleed 1080x1920, no cards, no frames. Two beat types alternate every 2 to 4 s:
  map    dark minimal map (near-black sea, charcoal land, thin light borders, small uppercase grey country labels);
         territory fills change in soft blues and muted browns as the story moves through the years. House rule kept:
         the whole land of Israel always drawn (1949 lines soft blue, West Bank and Gaza a slightly different blue,
         Green Line green). Slow camera push on every map beat.
  photo  full-bleed REAL archival photo (GPO Israel, Wikimedia Commons PD/CC, US government) with a slow Ken Burns
         push and a muted film grade (desaturated, lifted blacks, warm highlights, grain, vignette); tiny credit line.
Overlays: big bold white subtitle ONE WORD at a time (karaoke, word-synced to the narration, soft shadow, lower middle),
year label top left (bold white, rolls to the new year), small "ISRAEL FACTS" mark above it, thin timeline scrubber along
the bottom (start year left, end year right, a dot gliding to the current year), small source label under the year,
hook from frame 0 (house hook card), screen lines (debate question, follow line) as plain shadowed text, balance line
(no panel), music bed, soft whoosh on cuts and a tick on year changes, circular ending (dissolves into frame 0).

data keys (all times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  span     [1923, 2024]  timeline ends; span_labels ["1923", "TODAY"] optional text for the two ends
  years    [{at, year, label}]  year label + scrubber target from `at` (label optional, e.g. "TODAY"; year still places the dot)
  beats    [{at, kind: "map" | "photo", ...}]  each beat runs until the next beat's `at` (the first starts at 0)
           map:   cam0 [lon, lat, k] -> cam1 [lon, lat, k] (k = px per degree, default push 6 %), states apply
           photo: file (repo path, e.g. photos/gl_x.jpg; fetched into the sparse CI checkout if missing), credit,
                  kb [cx0, cy0, z0, cx1, cy1, z1] (defaults to a slow centre push), gray true for black and white,
                  dim 0..1 (default 0.15)
  states   [{at, fills: {shape: colour}, fade}]  map fills from `at` (crossfade `fade` s, default 0.6). shape: golan, sinai,
           an ISO3 code (SYR, EGY, JOR, LBN ...), or "isr49" / "terr" to tint Israel itself. colour: blue, blue2, brown,
           blue3, sand, sand2, grey, land, red, green, sky or #hex. Unlisted shapes keep the previous state.
  labels   [{text, lon, lat, size, at, until}]  uppercase grey labels (default: EGYPT, JORDAN, SYRIA, LEBANON, ISRAEL)
  marks    [{at, until, lon, lat, text, side}]  small white dot + label on the map (e.g. Quneitra, Sea of Galilee)
  subs_y   word subtitle centre (default 1180), subs_size (default 132)
  year_xy  [64, 250] top-left year label position; year_size 112
  scrub_y  timeline y (default 1650); label_min_k: map labels hidden below this zoom (default 250)
  cut      crossfade seconds between beats (default 0.22)
  end      (spec top level) {at, until, balance, cta, y}: balance as shadowed text (no panel); cta optional pill
"""
import os, sys, math, re, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

PAL = dict(blue="#5E8FBF", blue2="#3F6A93", blue3="#86AED3", sand2="#6E604C", brown="#8A6A4C", sand="#A88F6B", grey="#3A3F47", land="#1B1E23",
           red="#9C4A47", green="#3FBF7F", sky=SKY)
SEA, LANDC, BORDER, LABEL = "#0A0C10", "#1B1E23", "#7B8494", "#8E97A6"
ISR_C, TERR_C, GREENLINE = PAL["blue"], "#4A77A3", "#3FBF7F"
DEF_LABELS = [dict(text="EGYPT", lon=31.6, lat=29.6), dict(text="JORDAN", lon=36.9, lat=31.0), dict(text="SYRIA", lon=37.4, lat=33.9),
              dict(text="LEBANON", lon=35.9, lat=34.25), dict(text="ISRAEL", lon=34.62, lat=30.8)]


def _hex_bgr(h):
    if h in PAL: h = PAL[h]
    return mt.hexc(h)


def _photo(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(p):                          # the CI checkout is sparse (templates/ only): pull the photo in
        subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + path], check=False, capture_output=True)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"cine_timeline: cannot read photo {path}")
    return im


def _kb(im, cx, cy, z):
    h, w = im.shape[:2]; s = max(W / w, H / h) * max(1.0, z); cw, ch = W / s, H / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * np.clip(((_xx - W / 2) / (W * 0.72)) ** 2 + ((_yy - H * 0.47) / (H * 0.62)) ** 2, 0, 1))[..., None].astype(np.float32)
del _yy, _xx
_y = np.arange(H, dtype=np.float32)[:, None, None]
SHADE = (1 - 0.42 * np.clip(1 - _y / 430, 0, 1) ** 1.3 - 0.40 * np.clip((_y - 1460) / 460, 0, 1) ** 1.3).astype(np.float32)   # legible year/source/scrubber
del _y
GRAIN = [cv2.GaussianBlur(np.random.default_rng(s).normal(0, 9.0, (H // 2, W // 2)).astype(np.float32), (0, 0), 0.6) for s in range(6)]


def film(img, t, gray=False, dim=0.15):
    """muted documentary grade: desaturate, lift blacks, soft contrast, warm highs / cool lows, grain, vignette"""
    g = img.mean(2, keepdims=True)
    x = g.repeat(3, 2) if gray else g + (img - g) * 0.58
    x = 18 + x * 0.86                                          # lifted blacks, softer whites
    x = 255 * (0.5 + (x / 255 - 0.5) * 1.06)
    lum = np.clip(g / 255, 0, 1)
    if not gray: x = x + (lum - 0.5) * np.array([-10, 2, 12], np.float32)   # BGR: warm highlights, teal shadows
    gr = cv2.resize(GRAIN[int(t * 12) % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    x = (x + gr) * VIG * (1 - dim)
    return x


class DarkMap:
    def __init__(s):
        s.m = MapR()
    def rings(s, shape):
        m = s.m
        if shape == "golan": return m.golan
        if shape == "sinai": return m.sinai
        if shape == "isr49": return m.isr
        if shape == "terr": return m.terr
        return m.country(shape)
    def render(s, cam, fills, mix=None):
        """cam (lon, lat, k); fills {shape: bgr}; mix (fills_prev, u) crossfade -> float32 BGR (H, W, 3)"""
        lon, lat, k = cam; v = mt.View((lon, lat), (540.0, 960.0), k, 0.0, math.cos(math.radians(lat)))
        M = v.M(); inv = np.linalg.inv(M); c = mt.apply(inv, np.array([[-40, -40], [W + 40, -40], [-40, H + 40], [W + 40, H + 40]], float))
        vb = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
        cv = np.empty((H, W, 3), np.uint8); cv[:] = mt.hexc(SEA).astype(np.uint8)
        land, bord = mt.hexc(LANDC), mt.hexc(BORDER); lw = max(1.0, min(k / 90, 2.2))
        for code, rings in s.m.world.items():
            if code in ("ISR", "PSE"): continue
            mt.fill(cv, M, s.m.country(code), land, 1.0, vb, outline=bord, lw=lw, oalpha=0.55)
        mt.fill(cv, M, s.m.sinai, land, 1.0, vb, outline=bord, lw=lw, oalpha=0.55)
        mt.fill(cv, M, s.m.golan, land, 1.0, vb, outline=bord, lw=lw, oalpha=0.55)
        def paint(F, alpha):
            for shape, col in F.items():
                if shape in ("isr49", "terr"): continue
                mt.fill(cv, M, s.rings(shape), col, alpha * 0.92, vb, outline=bord, lw=lw, oalpha=0.6 * alpha)
        if mix:
            Fp, u = mix; paint(Fp, 1.0)
            if u > 0: paint(fills, u)
        else: paint(fills, 1.0)
        def col(shape, dflt):
            c1 = fills.get(shape, mt.hexc(dflt))
            if not mix: return c1
            c0 = mix[0].get(shape, mt.hexc(dflt)); return c0 * (1 - mix[1]) + c1 * mix[1]
        mt.fill(cv, M, s.m.isr, col("isr49", ISR_C), 1.0, vb, outline=bord, lw=lw, oalpha=0.7)
        mt.fill(cv, M, s.m.terr, col("terr", TERR_C), 1.0, vb, outline=mt.hexc(GREENLINE), lw=max(2, min(k / 60, 4)), oalpha=0.95)
        return cv.astype(np.float32), M


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    XF = float(D.get("cut", 0.22))
    # ---------------------------------------------------------------- beats
    B = []
    for b in D["beats"]:
        b = dict(b); b["t0"] = 0.0 if not B else A(b["at"])
        if b.get("kind", "map") == "photo": b["im"] = _photo(b["file"])
        B.append(b)
    for k, b in enumerate(B): b["t1"] = B[k + 1]["t0"] if k + 1 < len(B) else T.END + 0.01
    # ---------------------------------------------------------------- map states
    ST = []
    for st in D.get("states", []):
        ST.append(dict(t0=A(st.get("at"), 0.0), fade=float(st.get("fade", 0.6)), fills={k: _hex_bgr(v) for k, v in st["fills"].items()}))
    ST.sort(key=lambda x: x["t0"])
    cum, acc = [], {}
    for st in ST: acc = dict(acc, **st["fills"]); cum.append(dict(st, all=dict(acc)))
    def fills_at(t):
        cur, prev = {}, {}
        for i, st in enumerate(cum):
            if t >= st["t0"]: prev = cum[i - 1]["all"] if i else {}; cur = st;
        if not cur: return {}, None
        u = 1.0 if cur["t0"] <= 0.01 else seg(t, cur["t0"], cur["t0"] + cur["fade"])
        return cur["all"], ((prev, u) if u < 1 else None)
    # ---------------------------------------------------------------- years / scrubber
    span = D.get("span", [1900, 2025]); sl = D.get("span_labels") or [str(span[0]), str(span[1])]
    YR = [dict(y, t0=A(y.get("at"), 0.0)) for y in D.get("years", [])]; YR.sort(key=lambda y: y["t0"])
    def year_at(t):
        if not YR: return None, None, 1.0, span[0]
        i = max([k for k, y in enumerate(YR) if t >= y["t0"]] or [0]); y = YR[i]
        y0 = YR[i - 1]["year"] if i else y["year"]; u = eout((t - y["t0"]) / 0.55) if i else 1.0
        return y, y0, u, y0 + (y["year"] - y0) * u
    MK = [dict(m, t0=A(m.get("at"), 0.0), t1=A(m.get("until"), T.END)) for m in D.get("marks", [])]
    LB = [dict(l, t0=A(l.get("at"), -1.0), t1=A(l.get("until"), T.END + 1)) for l in D.get("labels", DEF_LABELS)]
    DM = DarkMap()
    # ---------------------------------------------------------------- words (karaoke)
    repl = S.get("subs_replace") or {}
    WD = []
    for l in T.L:
        ws = l["words"]; i = 0
        while i < len(ws):
            hit = None
            for a_, b_ in repl.items():                      # multi-word replacements (phoneme markup -> plain word)
                n = len(a_.split())
                if " ".join(w["w"] for w in ws[i:i + n]) .rstrip(".,!?;:") == a_ or " ".join(w["w"] for w in ws[i:i + n]) == a_:
                    hit = (b_, n); break
            if hit: WD.append([hit[0], ws[i]["t0"], ws[i + hit[1] - 1]["t1"]]); i += hit[1]
            else: WD.append([ws[i]["w"], ws[i]["t0"], ws[i]["t1"]]); i += 1
    for w in WD: w[0] = re.sub(r"[\"“”֑-ׇ]", "", w[0]).strip().rstrip(",;:.")
    for j in range(len(WD)):
        nxt = WD[j + 1][1] if j + 1 < len(WD) else WD[j][2] + 0.5
        WD[j][2] = min(nxt - 0.01, max(WD[j][2], WD[j][1] + 0.18) + 0.25) if nxt > WD[j][1] else WD[j][2]
    # ---------------------------------------------------------------- cards (screen lines), hook, end
    cards = [c for c in screen_cards(S, T) if c[1] > T.E(0) - 0.05]
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 860)
    E = S.get("end") or {}; END_AT = A(E.get("at"), T.S(len(T.L) - 2)) if E else None
    SUBY, SUBS = D.get("subs_y", 1180), D.get("subs_size", 132)
    YX, YY = D.get("year_xy", [64, 250]); YS = D.get("year_size", 112); SCY = D.get("scrub_y", 1650)

    def beat_img(b, t):
        if b.get("kind", "map") == "photo":
            kb = b.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.48, 1.10]
            u = (t - b["t0"]) / max(0.5, b["t1"] - b["t0"] + XF)
            u = min(1.0, max(0.0, u))
            img = _kb(b["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
            return film(img, t, b.get("gray", False), b.get("dim", 0.15)), None
        c0 = b.get("cam0") or [35.6, 32.2, 300]; c1 = b.get("cam1") or [c0[0], c0[1], c0[2] * 1.06]
        u = ease(min(1.0, max(0.0, (t - b["t0"]) / max(0.5, b["t1"] - b["t0"]))))
        cam = [c0[i] + (c1[i] - c0[i]) * u for i in range(3)]
        F, mix = fills_at(t); img, M = DM.render(cam, F, mix)
        return img * VIG ** 0.6, (M, b, cam[2])

    def frame(t):
        k = max([i for i, b in enumerate(B) if t >= b["t0"]] or [0]); b = B[k]
        base, mp = beat_img(b, t)
        if k > 0 and XF > 0 and t < b["t0"] + XF:                 # crossfade from the previous beat
            pb, _ = beat_img(B[k - 1], t); u = (t - b["t0"]) / XF; base = pb * (1 - u) + base * u
            if u < 0.5: mp = None
        over = max([seg(t, END_AT - 0.1, END_AT + 0.4)] if END_AT else [0])
        if over > 0: base = base * (1 - 0.35 * over)
        cv = Canvas(base * SHADE)
        if mp:                                                    # map labels + marks
            M, _, kk = mp
            for l in (LB if kk >= D.get("label_min_k", 250) else []):     # wide views: no labels (they crowd)
                la = win(t, l["t0"], l["t1"], 0.3, 0.3) if l["t0"] > 0 else (1 - seg(t, l["t1"] - 0.3, l["t1"]))
                x, y = geo2px(M, l["lon"], l["lat"])
                if la > 0.01 and -100 < x < W + 100 and 0 < y < H:
                    txt = " ".join(l["text"].upper())               # letter spaced
                    cv.text(txt, x, y, l.get("size", 38), LABEL, 0.9 * la, bold=False, shadow=False, maxw=600, minsz=36)
            for m in MK:
                ma = win(t, m["t0"], m["t1"], 0.25, 0.25)
                if ma > 0.01:
                    x, y = geo2px(M, m["lon"], m["lat"]); cv.circ(x, y, 9, WHITE, ma)
                    side = m.get("side", "r"); cv.text(m["text"], x + (22 if side == "r" else -22), y, 40, WHITE, ma, bold=True,
                                                       align="l" if side == "r" else "r", maxw=520, minsz=36)
        if b.get("kind") == "photo" and b.get("credit"):          # tiny credit
            ca = win(t, b["t0"], b["t1"], 0.2, 0.2)
            cv.text("Photo: " + b["credit"], 540, SCY - 70, 36, "#B8BEC8", 0.8 * ca, bold=False, maxw=1000, minsz=36)
        # year label + mark
        cv.text("ISRAEL FACTS", YX, YY - 92, 36, "#C9CED6", 0.75, bold=True, align="l", shadow=True, maxw=500)
        y, y0, u, yv = year_at(t)
        if y:
            lab = y.get("label") or str(int(round(yv)))
            if u < 1 and not y.get("label"): lab = str(int(round(yv)))
            cv.text(lab, YX, YY, YS, WHITE, 1.0, bold=True, align="l", shadow=True, maxw=900)
        # sources (small, under the year)
        for s_ in S["sources"]:
            t0 = A(s_.get("at"), T.S(min(1, len(T.L) - 1))); t1 = A(s_.get("until"), T.END - 0.6); sa = win(t, t0, t1, 0.25, 0.25)
            if sa > 0.01:
                lab = f"{s_['label']}, {s_['year']}" if s_.get("year") and str(s_["year"]) not in s_["label"] else s_["label"]
                cv.text("Source: " + lab, YX, YY + 92, 36, "#C9CED6", 0.85 * sa, bold=False, align="l", maxw=950, minsz=36)
        # timeline scrubber
        x0, x1 = 90, 990; p = 0.0 if not y else min(1, max(0, (yv - span[0]) / max(1e-6, span[1] - span[0])))
        cv.line([(x0, SCY), (x1, SCY)], WHITE, 3, 0.35)
        cv.line([(x0, SCY), (x0 + (x1 - x0) * p + 0.01, SCY)], WHITE, 4, 0.9)
        cv.circ(x0 + (x1 - x0) * p, SCY, 22, WHITE, 0.18); cv.circ(x0 + (x1 - x0) * p, SCY, 12, WHITE, 1.0)
        cv.text(sl[0], x0, SCY + 52, 36, "#C9CED6", 0.85, bold=True, align="l", maxw=300)
        cv.text(sl[1], x1, SCY + 52, 36, "#C9CED6", 0.85, bold=True, align="r", maxw=300)
        # screen lines (debate question, follow line): plain shadowed text
        for txt, t0, t1, yy, ss in cards:
            a = win(t, t0 - 0.1, t1, 0.25, 0.25)
            if a > 0.01: WG.title(cv, txt.replace("|", " "), yy or 560, a, ss or 76)
        if E and END_AT is not None:
            ea = win(t, END_AT - 0.1, A(E.get("until"), T.END - 0.5), 0.35, 0.3)
            if ea > 0.01:
                if E.get("balance"): WG.title(cv, E["balance"], E.get("y", 820), ea, 66)
                if E.get("cta"):
                    ca = ea * seg(t, END_AT + 0.6, END_AT + 1.0); tw = min(940, cv.tw(E["cta"].replace("*", ""), 58) + 90)
                    cv.rrect(540 - tw / 2, E.get("y", 820) + 130, tw, 112, 56, SKY, ca); WG.title(cv, E["cta"], E.get("y", 820) + 186, ca, 58)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        # karaoke: one big word at a time, soft shadow
        c2 = Canvas(out)
        hook_on = t < T.E(0) + 0.35                               # the hook card already shows the first spoken line
        for w, a_, b_ in WD:
            if hook_on and a_ < T.E(0) + 0.05: continue
            if a_ - 0.03 <= t < b_ and w:
                pop = 1.0 + 0.08 * (1 - eout((t - a_ + 0.03) / 0.09)); sz = SUBS * pop
                cv_ = c2; cv_.face(True); s2 = sz
                while cv_.tw(w, s2) > 1000 and s2 > 60: s2 -= 4
                for dx, dy, al in ((0, 8, 0.28), (0, 4, 0.35), (3, 3, 0.45)):
                    cv_.col("#000000", al); cv_.c.set_font_size(s2); e = cv_.c.text_extents(w)
                    cv_.c.move_to(540 - e.x_advance / 2 + dx, SUBY + s2 * 0.35 + dy); cv_.c.show_text(w)
                cv_.text(w, 540, SUBY, s2, WHITE, 1.0, bold=True, maxw=1000, shadow=False, minsz=60)
                break
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(b["t0"], "whoosh", -34) for b in B[1:]]
    ev += [(y["t0"], "tick", -22) for y in YR[1:]]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=b["file"], credit=b.get("credit")) for b in B if b.get("kind") == "photo"])
