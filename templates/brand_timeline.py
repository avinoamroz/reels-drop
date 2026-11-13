"""brand_timeline: a "logo evolution" history reel. A photo band on top (real photos, Ken Burns, credit line), a big yellow
year counter with a timeline track and playhead under it, and brand cards that "rebrand" on cue (old name in a paper
masthead, a red strike, the new name slides in), sourced info cards, a drawn flag family (flag LEFT of each name), stamps,
end card, burned-in subtitles, hook from frame 0, circular ending (dissolves back into frame 0).

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  band_h   photo band height (default 700)
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim, gray}]   band photos (repo paths)
           or {flag: "ARAB"|"PSE"|"JOR"|"SDN"|"KWT"|"ARE", at, until} a big drawn flag in the band instead of a photo
  years    {from, to, y (counter, default 790), track_y (default 900), ticks: [1916, ...], minor: [1911, ...],
            steps: [{at, value}], hide: [[at, until], ...]}       counter rolls between steps; playhead on the track
  cards    [{at, until, old, new, change_at, old_tag, new_tag, font: serif|sans, y, sub}]   paper brand card
  info     [{at, until, title, rows: [..], y}]                    navy panel (title sky, rows white 64 px)
  flags    {at, until, y, pitch, rows: [{code, label, at, hl}]}  drawn flags, flag left of the name
  stamp    [{at, until, text, color, y, angle, size}]
  hook_y default 1150; subs_y 1612; src_y 1488
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *
from PIL import Image, ImageDraw, ImageFont

SERIF = os.path.join(HERE, "fonts", "DejaVuSerif-Bold.ttf")
FB, FG, FW, FR = "#111111", "#007A3D", "#FFFFFF", "#CE1126"
STAMP_COL = dict(red=RED, green=GREEN, yellow=YEL, sky=SKY)
_txt = {}


def serif_img(txt, sz, col):
    k = (txt, sz, col)
    if k not in _txt:
        f = ImageFont.truetype(SERIF, int(sz)); w = int(f.getlength(txt)) + 16; h = int(sz * 1.45)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(im).text((8, h / 2), txt, font=f, fill=col, anchor="lm")
        _txt[k] = np.asarray(im)
    return _txt[k]


def serif(cv, txt, cx, cy, sz, col=(20, 20, 20, 255), a=1.0, maxw=860):
    while sz > 40 and ImageFont.truetype(SERIF, int(sz)).getlength(txt) > maxw: sz -= 2
    cv.image(serif_img(txt, sz, col), cx, cy, 1.0, a)


def _load(path, gray=False):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit(f"brand_timeline: cannot read photo {path}")
    if gray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); im = cv2.merge([g, g, g])
    return im


def _kb(im, cx, cy, z, bw, bh):
    h, w = im.shape[:2]; s = max(bw / w, bh / h) * max(1.0, z)
    cw, ch = bw / s, bh / s
    x0 = min(max(cx * w - cw / 2, 0), w - cw); y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def _poly(cv, pts, col, a):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path(); cv.col(col, a); c.fill()


def _star7(cv, x, y, r, col, a):
    pts = []
    for i in range(14):
        rr = r if i % 2 == 0 else r * 0.45; ang = -math.pi / 2 + i * math.pi / 7
        pts.append((x + rr * math.cos(ang), y + rr * math.sin(ang)))
    _poly(cv, pts, col, a)


def draw_flag(cv, code, x, y, w, h, a=1.0):
    """drawn flags (simplified official geometry)"""
    if a <= 0.01: return
    def stripes(cols):
        n = len(cols)
        for i, c_ in enumerate(cols): cv.rrect(x, y + h * i / n, w, h / n + 0.6, 0, c_, a)
    if code == "ARAB": stripes([FB, FG, FW]); _poly(cv, [(x, y), (x + w * 0.42, y + h / 2), (x, y + h)], FR, a)
    elif code == "PSE": stripes([FB, FW, FG]); _poly(cv, [(x, y), (x + w / 3, y + h / 2), (x, y + h)], FR, a)
    elif code == "JOR":
        stripes([FB, FW, FG]); _poly(cv, [(x, y), (x + w / 2, y + h / 2), (x, y + h)], FR, a)
        _star7(cv, x + w * 0.17, y + h / 2, h * 0.085, FW, a)
    elif code == "SDN": stripes([FR, FW, FB]); _poly(cv, [(x, y), (x + w / 3, y + h / 2), (x, y + h)], FG, a)
    elif code == "KWT": stripes([FG, FW, FR]); _poly(cv, [(x, y), (x + w / 4, y + h / 3), (x + w / 4, y + 2 * h / 3), (x, y + h)], FB, a)
    elif code == "ARE": stripes([FG, FW, FB]); cv.rrect(x, y, w / 4, h, 0, FR, a)
    cv.rrect(x, y, w, h, 4, None, a, stroke="#FFFFFF", lw=2, sa=0.35 * a)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    BH = int(D.get("band_h", 700))
    P = []
    for p in D["photos"]:
        q = dict(p, t0=A(p["at"]))
        if p.get("file"): q["im"] = _load(p["file"], p.get("gray", False))
        P.append(q)
    for k, p in enumerate(P):
        p["t1"] = A(p["until"]) if p.get("until") else (P[k + 1]["t0"] if k + 1 < len(P) else T.END)
    XF = float(D.get("cut", 0.25))
    Y = dict(D.get("years") or {})
    YST = [(A(s_["at"]), float(s_["value"])) for s_ in Y.get("steps", [])]
    YHIDE = [(A(a), A(b)) for a, b in Y.get("hide", [])]
    CARDS = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6), tc=A(c["change_at"]) if c.get("change_at") else None) for c in D.get("cards", [])]
    INFO = [dict(i, t0=A(i["at"]), t1=A(i.get("until"), T.END - 0.6)) for i in D.get("info", [])]
    STM = [dict(s_, t0=A(s_["at"]), t1=A(s_.get("until"), T.END - 0.6)) for s_ in D.get("stamp", [])]
    FL = dict(D["flags"]) if D.get("flags") else None
    if FL:
        FL["t0"] = A(FL["at"]); FL["t1"] = A(FL.get("until"), T.END - 0.6)
        FL["rows"] = [dict(r, t0=A(r.get("at"), FL["t0"])) for r in FL["rows"]]
    END_AT = A((S.get("end") or {}).get("at"), T.S(len(T.L) - 2)) if S.get("end") else None
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 1150)
    yb = np.arange(BH, dtype=np.float32)[:, None]
    BVIG = (1 - np.clip((yb - (BH - 260)) / 260, 0, 1) ** 1.3 * 0.92 - np.clip(1 - yb / 260, 0, 1) ** 1.5 * 0.35)[..., None]
    NAVY = np.array([48, 23, 11], np.float32)

    def band_photo(p, t):
        if p.get("flag"):
            cv = Canvas(np.zeros((H, W, 3), np.float32) + NAVY)
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"])); z = 1.0 + 0.06 * u
            fw, fh = 760 * z, 380 * z
            draw_flag(cv, p["flag"], 540 - fw / 2, BH / 2 - fh / 2 - 30, fw, fh, 1.0)
            return cv.out()[:BH].astype(np.float32) * BVIG + NAVY * (1 - BVIG)
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
        u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"] + XF))
        img = _kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u, W, BH)
        dim = p.get("dim", 0.15)
        return img * (1 - dim) * BVIG + NAVY * (1 - BVIG)

    def base_at(t):
        out = BG.copy(); band = out[:BH].copy() * 0 + NAVY
        lin = lambda x: min(1.0, max(0.0, x))
        for k, p in enumerate(P):
            nxt = P[k + 1]["t0"] if k + 1 < len(P) else None
            gap = nxt is None or nxt > p["t1"] + 0.01
            if t < p["t0"] - XF or t >= p["t1"] + (XF if gap else 0): continue
            a = 1.0 if (p["t0"] <= 0.01 or not XF) else seg(t, p["t0"] - XF, p["t0"], lin)
            if gap and t >= p["t1"]: a *= 1.0 - (seg(t, p["t1"], p["t1"] + XF, lin) if XF else 1.0)
            if a <= 0.001: continue
            lay = band_photo(p, t); band = lay if a >= 0.999 else band * (1 - a) + lay * a
        out[:BH] = band
        return out

    def year_at(t):
        if not YST: return None
        v = YST[0][1]
        for k, (ts, val) in enumerate(YST):
            if t >= ts:
                prev = YST[k - 1][1] if k else val
                v = prev + (val - prev) * eout((t - ts) / 0.9)
        return v

    def draw_years(cv, t):
        if not YST: return
        a = 1.0
        for h0, h1 in YHIDE: a = min(a, 1 - win(t, h0, h1, 0.3, 0.3))
        if a <= 0.01: return
        v = year_at(t); y0, y1 = float(Y.get("from", 1905)), float(Y.get("to", 1975))
        vp = v if not Y.get("intro") else y0 + (v - y0) * eout(t / float(Y["intro"])) if t < float(Y["intro"]) else v
        cy = Y.get("y", 790); ty = Y.get("track_y", 900); X0, X1 = 90, 990
        X = lambda yr: X0 + (X1 - X0) * (yr - y0) / (y1 - y0)
        cv.text(f"{int(round(v))}", 540, cy, 150, YEL, a)
        cv.rrect(X0, ty - 4, X1 - X0, 8, 4, "#C9D6EA", 0.35 * a)
        cv.rrect(X0, ty - 4, max(0, X(vp) - X0), 8, 4, YEL, 0.9 * a)
        for yr in Y.get("minor", []):
            cv.circ(X(yr), ty, 7, YEL if vp >= yr - 0.5 else "#6F86A8", a)
        for yr in Y.get("ticks", []):
            on = vp >= yr - 0.5
            cv.circ(X(yr), ty, 11, YEL if on else "#6F86A8", a)
            cv.text(str(yr), X(yr), ty + 48, 40, WHITE if on else "#9FB0C8", a, shadow=False)
        cv.circ(X(vp), ty, 19, YEL, a); cv.circ(X(vp), ty, 9, NAVY0, a)

    def draw_card(cv, c, t):
        a = win(t, c["t0"] - 0.1, c["t1"], 0.3, 0.3)
        if a <= 0.01: return
        y = c.get("y", 1190); sp = back(min(1.0, (t - c["t0"] + 0.1) / 0.4), 1.1)
        top = y - 190 + 60 * (1 - sp); hh = 380
        cv.rrect(70, top, 940, hh, 26, "#F3EBD9", 0.97 * a, stroke=YEL, lw=3, sa=0.8 * a)
        cv.rrect(110, top + 30, 860, 3, 1, "#1A1A1A", 0.6 * a); cv.rrect(110, top + hh - 33, 860, 3, 1, "#1A1A1A", 0.6 * a)
        tc = c.get("tc"); ch = 0.0 if tc is None else seg(t, tc, tc + 0.45)
        sans = c.get("font") == "sans"
        oy = top + hh / 2 - 70 * ch
        old_a = a * (1 - 0.55 * ch)
        if sans: cv.text(c["old"], 540, oy, 76, "#1A1A1A", old_a, shadow=False, maxw=840, minsz=56)
        else: serif(cv, c["old"], 540, oy, 76, (24, 24, 24, 255), old_a)
        if c.get("old_tag"):
            tw = cv.tw(c["old_tag"], 44) + 40
            cv.rrect(110, top - 34, tw, 64, 32, NAVY1, a); cv.text(c["old_tag"], 110 + tw / 2, top - 2, 44, YEL, a, shadow=False)
        if tc is not None and t >= tc:
            sw = seg(t, tc, tc + 0.35)
            if sw > 0: cv.line([(120, oy), (120 + 840 * sw, oy)], RED, 9, a)
            na = a * seg(t, tc + 0.25, tc + 0.7); ny = top + hh / 2 + 80 + 40 * (1 - na / max(a, 1e-3))
            if na > 0.01:
                if sans: cv.text(c["new"], 540, ny, 80, NAVY1, na, shadow=False, maxw=840, minsz=56)
                else: serif(cv, c["new"], 540, ny, 80, (22, 50, 97, 255), na)
                if c.get("new_tag"):
                    tw = cv.tw(c["new_tag"], 44) + 40
                    cv.rrect(970 - tw, top + hh - 30, tw, 64, 32, YEL, na); cv.text(c["new_tag"], 970 - tw / 2, top + hh + 2, 44, NAVY0, na, shadow=False)
        elif c.get("sub"):
            cv.text(c["sub"], 540, top + hh / 2 + 95, 52, "#4A4A4A", a, bold=False, shadow=False, maxw=840, minsz=40)

    def draw_info(cv, i, t):
        a = win(t, i["t0"] - 0.1, i["t1"], 0.3, 0.3)
        if a <= 0.01: return
        rows = i.get("rows", []); y = i.get("y", 1190); hh = 110 + 84 * len(rows) + (70 if i.get("title") else 0)
        top = y - hh / 2; WG.panel(cv, 60, top, 960, hh, a, stroke=YEL); yy = top + 75
        if i.get("title"): WG.title(cv, i["title"], yy, a, 70, col=SKY); yy += 95
        for k, r in enumerate(rows):
            ra = a * seg(t, i["t0"] + 0.2 * k, i["t0"] + 0.2 * k + 0.3)
            WG.title(cv, r, yy, ra, 64); yy += 84

    def draw_flags(cv, t):
        a = win(t, FL["t0"] - 0.1, FL["t1"], 0.3, 0.3)
        if a <= 0.01: return
        y0 = FL.get("y", 760); pt = FL.get("pitch", 118); n = len(FL["rows"])
        WG.panel(cv, 50, y0 - 30, 980, pt * n + 50, a)
        for k, r in enumerate(FL["rows"]):
            ra = a * seg(t, r["t0"], r["t0"] + 0.3)
            if ra <= 0.01: continue
            yy = y0 + k * pt; s_ = 0.85 + 0.15 * back(seg(t, r["t0"], r["t0"] + 0.35, lambda u: u))
            if r.get("hl"): cv.rrect(64, yy - 4, 952, pt - 12, 22, YEL, 0.16 * ra, stroke=YEL, lw=3, sa=0.8 * ra)
            fw, fh = 150 * s_, 75 * s_
            draw_flag(cv, r["code"], 100 + (150 - fw) / 2, yy + (pt - 16) / 2 - fh / 2, fw, fh, ra)
            cv.text(r["label"], 290, yy + (pt - 16) / 2, 64, YEL if r.get("hl") else WHITE, ra, align="l", maxw=700, shadow=False)

    def frame(t):
        base = base_at(t)
        endov = seg(t, END_AT - 0.1, END_AT + 0.3) if END_AT else 0
        if endov > 0: base = base * (1 - 0.45 * endov)
        cv = Canvas(base)
        for p in P:
            ca = win(t, p["t0"], p["t1"], 0.25, 0.25)
            if ca > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, BH - 40, 36, "#C9D6EA", 0.9 * ca, bold=False, maxw=1000, minsz=36)
        draw_years(cv, t)
        for c in CARDS: draw_card(cv, c, t)
        for i in INFO: draw_info(cv, i, t)
        if FL: draw_flags(cv, t)
        for s_ in STM:
            sa = win(t, s_["t0"] - 0.05, s_["t1"], 0.08, 0.3)
            if sa > 0.01:
                k = 1.0 + 0.9 * (1 - eout((t - s_["t0"]) / 0.18)) if t < s_["t0"] + 0.18 else 1.0
                col = STAMP_COL.get(s_.get("color", "red"), s_.get("color", RED)); sz = s_.get("size", 120) * k
                y = s_.get("y", 1180); ang = math.radians(s_.get("angle", -9)); txt = s_["text"]
                c = cv.c; c.save(); c.translate(540, y); c.rotate(ang); c.translate(-540, -y)
                tw = min(cv.tw(txt, sz), 900) + 80
                cv.rrect(540 - tw / 2, y - sz * 0.72, tw, sz * 1.44, 18, "#0B1730", 0.55 * sa, stroke=col, lw=10 * k, sa=sa)
                cv.text(txt, 540, y, sz, col, sa, maxw=900, shadow=False, minsz=36)
                c.restore()
        draw_sources(cv, S, T, t, D.get("src_y", 1488)); draw_end(cv, S, T, t)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 1.0)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -30) for p in P[1:]]
    ev += [(s_["t0"], "thud", -18) for s_ in STM]
    ev += [(c["tc"], "pop", -24) for c in CARDS if c.get("tc") is not None]
    ev += [(c["t0"], "whoosh", -32) for c in CARDS]
    ev += [(i["t0"], "pop", -26) for i in INFO]
    if FL: ev += [(r["t0"], "pop", -28) for r in FL["rows"]]
    ev += [(ts, "tick", -30) for ts, _ in YST[1:]]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p.get("file"), credit=p.get("credit")) for p in P])
