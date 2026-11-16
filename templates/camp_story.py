"""camp_story (@noamroz__ deadpan humour): a flat-cartoon summer camp (sky, sun, clouds, hills, pines, a lake, two tents,
a wooden SUMMER CAMP sign). One grumpy small kid ("hero": side parting, tiny mustache) sits alone on a log at frame 0
while three kids pass a ball; a grandpa can wave him in; speech bubbles carry the dialogue; a soft camera zooms on whoever
speaks. Gags: shrug, hoard (he holds every toy), possessed (swirl eyes, purple glow, kids lean back), sweat. A book card
(plain navy cover, no symbols) for the title reveal. The final beat returns to the exact frame-0 state, and the last 0.5 s
dissolves into frame 0, so the reel loops seamlessly (a hanging line can be finished by the hook).
Built for mein_kampf_camp (15 Nov). No flags, uniforms, fences or any war imagery by design.

data keys (all times are anchors):
  stand      {at, until}         hero stands near the kids (otherwise sits on the log, frame-0 pose)
  grandpa    {at, until}         grandpa walks in and waves, "my grandpa" tag
  shrug | hoard | possessed | sweat   {at, until}
  bubbles    [{at, until, who: hero|kids|grandpa, text}]   *word* in red, | = line break
  titles     [{at, until, text}]  navy pill title at title_y (*word* yellow)
  book       {at, until, title, sub, sub_at, src}
  tags       [{at, until, who, text}]
  cams       [{at, who: wide|hero|kids, z, dur}]   the camera eases to the speaker
  hook_y (520), title_y (520), subs_y (1690), tail (0.25)
  shift_y    move the whole scene up (negative px, grass fills the bottom), e.g. to make room for a floating head
  src_y      y of the book's source label (default 1430); book_dy moves the book card up or down; subs_size (default 56)
  take       Noam's real voice instead of TTS: {clip, words (longtx json), keep: [[a, b, "exact words"], ...]}.
             The kept parts are joined (his own audio, his own frame timing); lines L0, L1 ... are the kept parts.
  head       floating head cut out of the take (core/head_cut.py, u2net person model, never mirrored), lips in sync:
             {crop: [x, y, w, h], cut_w: 480, mode: fixed | presenter, w (px wide), cx, bottom,
              spots: {name: [cx, bottom]}, cues: [{at, spot}], pops: [anchor], pop: 1.18, bob: 8}
"""
import os, sys, math, re, shutil, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

INK = "#0E2250"; SKIN = "#F6CFA8"; HAIR = "#2B2118"; NAVY = "#163261"; REDW = "#E5484D"
FEET = 1600; HORIZON = 1120
HERO_SIT = (220, 1600); HERO_STAND = (360, 1600)
KIDS = [(620, "#E5484D", "#7A4A2A"), (790, "#FFD23F", "#E8B04A"), (960, "#3FBF7F", "#3A2A1E")]
GRANDPA_X = 520


def poly(cv, pts, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.new_path(); c.move_to(*pts[0])
    for p in pts[1:]: c.line_to(*p)
    c.close_path()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def ellipse(cv, x, y, rx, ry, col, a=1.0, stroke=None, lw=3):
    c = cv.c; c.save(); c.translate(x, y); c.scale(rx, ry); c.new_path(); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    if col: cv.col(col, a); c.fill_preserve() if stroke else c.fill()
    if stroke: cv.col(stroke, a); c.set_line_width(lw); c.stroke()
    c.new_path()


def arcl(cv, x, y, r, a0, a1, col, lw, a=1.0):
    c = cv.c; c.new_path(); c.arc(x, y, r, a0, a1); cv.col(col, a); c.set_line_width(lw); c.set_line_cap(1); c.stroke(); c.new_path()


# ------------------------------------------------------------------ backdrop
def backdrop(t):
    y = np.arange(H, dtype=np.float32)[:, None] / HORIZON
    top, bot = np.array([242, 198, 124], np.float32), np.array([250, 235, 205], np.float32)      # BGR sky
    img = np.empty((H, W, 3), np.float32)
    g = np.clip(y, 0, 1)[..., None]; img[:] = top * (1 - g) + bot * g
    cv = Canvas(img)
    cv.circ(870, 300, 78, "#FFD23F", 1.0); cv.circ(870, 300, 110, "#FFD23F", 0.18)
    for k, (x0, yy, s) in enumerate(((120, 230, 1.0), (520, 380, 0.8), (980, 520, 0.7))):
        x = (x0 + t * (8 + 4 * k)) % 1400 - 160
        for dx, dy, r in ((0, 0, 46), (46, -18, 56), (100, 0, 44), (50, 14, 50)): ellipse(cv, x + dx * s, yy + dy * s, r * s * 1.2, r * s * 0.8, "#FFFFFF", 0.95)
    poly(cv, [(0, 1030), (180, 960), (380, 1010), (560, 940), (760, 1000), (930, 950), (1080, 990), (1080, HORIZON + 10), (0, HORIZON + 10)], "#7BC47F")
    cv.rrect(0, HORIZON, 1080, H - HORIZON, 0, "#6CC070")
    cv.rrect(0, 1640, 1080, H - 1640, 0, "#5DB062")
    for x, s in ((60, 1.0), (140, 0.8), (1000, 1.05), (1060, 0.85)):
        cv.rrect(x - 8 * s, HORIZON - 30 * s, 16 * s, 40 * s, 3, "#8A5A2B")
        for k in range(3): poly(cv, [(x, HORIZON - (210 - 50 * k) * s), (x - (52 + 12 * k) * s, HORIZON - (80 - 40 * k) * s - 20 * s), (x + (52 + 12 * k) * s, HORIZON - (80 - 40 * k) * s - 20 * s)], "#2F8F4E")
    ellipse(cv, 250, 1250, 300, 80, "#3E9BD8"); ellipse(cv, 250, 1240, 270, 58, "#5DB3E8")
    for k in range(3): cv.line([(140 + 90 * k + 10 * math.sin(t * 2 + k), 1240 + 14 * k), (190 + 90 * k + 10 * math.sin(t * 2 + k), 1240 + 14 * k)], "#E8F6FF", 4, 0.8)
    for x, w, h, col in ((730, 210, 160, "#F2A93B"), (930, 170, 130, "#E5484D")):
        poly(cv, [(x - w / 2, HORIZON + 70), (x, HORIZON + 70 - h), (x + w / 2, HORIZON + 70)], col, stroke=INK, lw=4)
        poly(cv, [(x - w * 0.14, HORIZON + 70), (x, HORIZON + 70 - h * 0.55), (x + w * 0.14, HORIZON + 70)], INK, 0.85)
    for x in (430, 650): cv.rrect(x - 7, 1000, 14, 140, 4, "#8A5A2B")
    cv.rrect(400, 990, 280, 84, 14, "#C9A15B", stroke="#8A5A2B", lw=5)
    cv.text("SUMMER CAMP", 540, 1030, 44, "#5A3A1A", 1.0, maxw=250, shadow=False, minsz=36)
    cv.rrect(100, FEET - 34, 260, 44, 18, "#8A5A2B", stroke="#5A3A1A", lw=4)            # the log
    return cv


# ------------------------------------------------------------------ characters
def kid(cv, x, y, h, shirt, hair=HAIR, sit=False, arms="down", eyes="dot", mouth="smile", hero=False, a=1.0, lean=0.0, glow=0.0, t=0.0):
    r = h * 0.15; lw = max(4, h * 0.028)
    hip = (x, y - (h * 0.22 if sit else h * 0.40)); neck = (x + lean, hip[1] - h * 0.28); head = (x + lean * 1.2, neck[1] - r * 1.05)
    if glow > 0.01:
        for k in range(3): cv.circ(x, y - h * 0.5, h * (0.55 + 0.08 * k + 0.03 * math.sin(t * 6 + k)), "#9B5DE5", 0.12 * glow)
    if sit: cv.line([hip, (x + h * 0.2, hip[1]), (x + h * 0.2, y)], INK, lw, a); cv.line([hip, (x + h * 0.12, hip[1] + 6), (x + h * 0.12, y)], INK, lw, a)
    else: cv.line([hip, (x - h * 0.09, y)], INK, lw, a); cv.line([hip, (x + h * 0.09, y)], INK, lw, a)
    sh = (neck[0], neck[1] + h * 0.05)
    A = {"down": [(-0.13, 0.2), (0.13, 0.2)], "wave": [(-0.13, 0.2), (0.16, -0.2)], "shrug": [(-0.22, -0.04), (0.22, -0.04)],
         "hold": [(-0.12, 0.12), (0.12, 0.12)], "up": [(-0.16, -0.18), (0.16, -0.18)], "toss": [(-0.15, -0.08), (0.15, -0.08)]}[arms]
    hands = []
    for dx, dy in A:
        e = (sh[0] + dx * h, sh[1] + dy * h); cv.line([sh, e], INK, lw, a); hands.append(e)
        if arms == "shrug": arcl(cv, e[0], e[1] - 6, 9, math.pi * 1.1, math.pi * 1.9, INK, lw * 0.8, a)
    cv.rrect(neck[0] - h * 0.11, neck[1], h * 0.22, hip[1] - neck[1] + 6, h * 0.05, shirt, a, stroke=INK, lw=lw * 0.8)
    cv.circ(*head, r, SKIN, a, stroke=INK, lw=lw * 0.8)
    hx, hy = head
    if hero:      # side parting: dark cap with a fringe swept across the forehead, tiny mustache
        c = cv.c; c.new_path(); c.arc(hx, hy, r, math.pi * 1.02, math.pi * 1.98); c.close_path(); cv.col(HAIR, a); c.fill(); c.new_path()
        poly(cv, [(hx + r * 0.55, hy - r * 0.86), (hx - r * 1.0, hy - r * 0.02), (hx - r * 0.98, hy - r * 0.5), (hx - r * 0.2, hy - r * 0.98)], HAIR, a)
        cv.line([(hx + r * 0.42, hy - r * 0.9), (hx + r * 0.18, hy - r * 0.55)], SKIN, lw * 0.5, a)
        cv.rrect(hx - r * 0.17, hy + r * 0.3, r * 0.34, r * 0.17, 2, HAIR, a)
    else:
        c = cv.c; c.new_path(); c.arc(hx, hy, r, math.pi * 1.05, math.pi * 1.95); c.close_path(); cv.col(hair, a); c.fill(); c.new_path()
    ey = hy - r * 0.08; ex = r * 0.38
    if eyes == "swirl":
        for s in (-1, 1):
            c = cv.c; c.new_path()
            for k in range(40):
                th = k * 0.45 + t * 9; rr = r * 0.02 + k * r * 0.0055; px, py = hx + s * ex + rr * math.cos(th), ey + rr * math.sin(th)
                c.line_to(px, py) if k else c.move_to(px, py)
            cv.col("#9B5DE5", a); c.set_line_width(lw * 0.45); c.stroke(); c.new_path()
    elif eyes == "wide":
        for s in (-1, 1): cv.circ(hx + s * ex, ey, r * 0.17, "#FFFFFF", a, stroke=INK, lw=2); cv.circ(hx + s * ex, ey, r * 0.07, INK, a)
    else:
        for s in (-1, 1): cv.circ(hx + s * ex, ey, r * 0.09, INK, a)
    if hero and eyes != "swirl":   # grumpy brows
        for s in (-1, 1): cv.line([(hx + s * ex * 1.5, ey - r * 0.32), (hx + s * ex * 0.4, ey - r * 0.2)], INK, lw * 0.6, a)
    my = hy + r * (0.62 if hero else 0.45)
    if mouth == "smile": arcl(cv, hx, my - r * 0.2, r * 0.3, math.pi * 0.15, math.pi * 0.85, INK, lw * 0.6, a)
    elif mouth == "frown": arcl(cv, hx, my + r * 0.12, r * 0.24, math.pi * 1.2, math.pi * 1.8, INK, lw * 0.6, a)
    elif mouth == "o": cv.circ(hx, my, r * 0.11, INK, a)
    return dict(head=(hx, hy - r), hands=hands, r=r)


def grandpa(cv, x, y, a, t):
    h = 430; kid(cv, x, y, h, "#F5F5F7", hair="#D9D9D9", arms="wave", a=a, t=t)
    r = h * 0.15; hx, hy = x, y - h * 0.40 - h * 0.28 - r * 1.05
    c = cv.c; c.new_path(); c.arc(hx, hy, r * 1.02, math.pi, 2 * math.pi); c.close_path(); cv.col("#45A4DD", a); c.fill(); c.new_path()
    cv.rrect(hx - r * 0.2, hy - r * 0.12, r * 1.5, r * 0.22, 6, "#45A4DD", a)
    for s in (-1, 1): cv.circ(hx + s * r * 0.38, hy - r * 0.08, r * 0.2, None, a, stroke=INK, lw=3)
    cv.circ(x - h * 0.04, y - h * 0.62, 7, "#C9D6EA", a)     # whistle


# ------------------------------------------------------------------ overlays
def parts_of(txt):
    out = []
    for tok in re.split(r"(\*[^*]+\*)", txt):
        if tok: out.append((tok.strip("*"), REDW if tok.startswith("*") else INK))
    return out


def bubble(cv, text, tx, ty, a, sc):
    """white speech bubble above (tx, ty) = the speaker's head top (screen px), text lines split by |"""
    if a <= 0.01: return
    sz = 64; lines = text.split("|")
    ws = [sum(cv.tw(p, sz) for p, _ in parts_of(l)) for l in lines]; w = min(1000, max(ws) + 90); h = 60 + 80 * len(lines)
    bx = min(max(tx - w / 2, 30), 1050 - w); by = ty - 70 - h
    c = cv.c; c.save(); c.translate(tx, ty - 40); c.scale(sc, sc); c.translate(-tx, -(ty - 40))
    poly(cv, [(min(max(tx - 26, bx + 30), bx + w - 80), by + h - 4), (tx, ty - 22), (min(max(tx + 26, bx + 80), bx + w - 30), by + h - 4)], "#FFFFFF", a, stroke=INK, lw=4)
    cv.rrect(bx, by, w, h, 40, "#FFFFFF", a, stroke=INK, lw=4)
    cv.rrect(bx + 3, by + h - 14, w - 6, 12, 4, "#FFFFFF", a)
    for i, l in enumerate(lines):
        ps = parts_of(l); x = bx + w / 2 - ws[i] / 2; yy = by + 70 + 80 * i
        for p, col in ps: cv.text(p, x, yy, sz, col, a, align="l", shadow=False, maxw=2000); x += cv.tw(p, sz)
    c.restore()


def pill_title(cv, txt, y, a):
    if a <= 0.01: return
    w = min(1000, cv.tw(txt.replace("*", ""), 76) + 110)
    cv.rrect(540 - w / 2, y - 72, w, 144, 44, NAVY, 0.92 * a)
    WG.title(cv, txt, y, a, 76)


def take_voice(S, TK, work):
    """his kept parts joined: work/vo.wav (48 kHz mono), timing json, and the source frame for every output frame"""
    import soundfile as sf
    need = [TK["clip"], TK["words"]]; miss = [f for f in need if not os.path.exists(os.path.join(REPO, f))]
    if miss: subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", *["/" + f for f in miss]], check=False)
    clip = os.path.join(REPO, TK["clip"])
    WD = [w for sg in json.load(open(os.path.join(REPO, TK["words"])))["segs"] for w in sg["w"]]
    au = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                      check=True, capture_output=True).stdout, np.float32)
    fr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", clip],
                        capture_output=True, text=True).stdout.strip().split("/")
    sfps = float(fr[0]) / float(fr[1] if len(fr) > 1 else 1)
    lines, parts, ot, f_ = [], [], 0.0, int(0.008 * SR)
    for i, k in enumerate(TK["keep"]):
        a, b, txt = float(k[0]), float(k[1]), k[2]
        ws = [w for w in WD if a - 0.05 <= w[1] < b]; tok = txt.split()
        if len(ws) == len(tok): tw = [(tk, w[1], w[2]) for tk, w in zip(tok, ws)]
        else:     # spread the exact words over the heard span by length
            s0, s1 = (ws[0][1], ws[-1][2]) if ws else (a + 0.05, b - 0.05); L8 = np.array([len(x) + 2 for x in tok], float); c = np.concatenate([[0], np.cumsum(L8)]) / L8.sum()
            tw = [(tk, s0 + (s1 - s0) * c[j], s0 + (s1 - s0) * c[j + 1] - 0.02) for j, tk in enumerate(tok)]
        seg_ = au[int(a * SR):int(b * SR)].copy(); seg_[:f_] *= np.linspace(0, 1, f_); seg_[-f_:] *= np.linspace(1, 0, f_); parts.append(seg_)
        lines.append(dict(i=i, text=txt, t0=round(ot + tw[0][1] - a, 3), t1=round(ot + tw[-1][2] - a, 3),
                          words=[dict(w=x, t0=round(ot + p - a, 3), t1=round(ot + q - a, 3)) for x, p, q in tw], src=[a, b], out=round(ot, 3)))
        ot += len(seg_) / SR
    sf.write(f"{work}/vo.wav", np.concatenate(parts), SR)
    src_idx = []
    for i in range(int(round(ot * FPS))):
        t = i / FPS; L = [l for l in lines if l["out"] <= t + 1e-6][-1]
        src_idx.append(int(round((L["src"][0] + t - L["out"]) * sfps)))
    J = dict(dur=round(ot, 3), voice=f"Noam's own voice from {TK['clip']}", lines=lines)
    json.dump(J, open(f"{work}/timing.json", "w"), indent=0)
    return J, src_idx, clip


def run(S):
    D = S["data"]; work = workdir(S); TK = D.get("take"); HD = D.get("head")
    if TK:
        J, SRC_IDX, CLIP = take_voice(S, TK, work); T = Timing(J, J["dur"] - J["lines"][-1]["t1"])
    else:
        J = build_voice(S, work); T = Timing(J, D.get("tail", 0.25))
    A = T.at
    HEADS = None
    if HD:
        import importlib
        try: importlib.import_module("onnxruntime")
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
        import head_cut as HC
        idx = SRC_IDX
        if os.environ.get("TPL_SNAPS"):
            want = [int(round(float(x) * FPS)) for x in os.environ["TPL_SNAPS"].split(",")]; idx = [SRC_IDX[min(i, len(SRC_IDX) - 1)] for i in want]
        HEADS = HC.cut(CLIP, idx, tuple(HD["crop"]), int(HD.get("cut_w", 480)), f"{work}/heads.npy", log=log)
        HSNAP = {w_: j for j, w_ in enumerate(want)} if os.environ.get("TPL_SNAPS") else None
        SPOTS = {k: tuple(v) for k, v in (HD.get("spots") or {}).items()}
        HCUE = [(A(c["at"]), SPOTS[c["spot"]]) for c in HD.get("cues", [])]
        POPS = [A(p) for p in HD.get("pops", [])]
    def W_(k): v = D.get(k); return (A(v["at"]), A(v.get("until"), T.END)) if v else None
    def on(k, t, f=0.3):
        w = W_(k); return 0.0 if not w else win(t, w[0], w[1], f, f)
    BUB = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END)) for b in D.get("bubbles", [])]
    TIT = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END)) for b in D.get("titles", [])]
    TAG = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), T.END)) for b in D.get("tags", [])]
    BK = D.get("book"); BK = dict(BK, t0=A(BK["at"]), t1=A(BK.get("until"), T.END), ts=A(BK.get("sub_at"), A(BK["at"]) + 1)) if BK else None
    FOC = {"wide": (540, 1180, 1.12), "hero": (HERO_STAND[0] + 60, 1260, 1.3), "kids": (790, 1260, 1.2), "log": (HERO_SIT[0] + 80, 1280, 1.25)}
    CAMS = [(0.0, 0.01, FOC["wide"])] + [(A(c["at"]), float(c.get("dur", 0.6)), (FOC[c["who"]][0], FOC[c["who"]][1], float(c.get("z", FOC[c["who"]][2])))) for c in D.get("cams", [])]
    def cam(t):
        cur = np.array(CAMS[0][2], float)
        for i, (t0, d, k) in enumerate(CAMS):
            if t < t0: break
            prev = np.array(CAMS[i - 1][2] if i else k, float); u = ease((t - t0) / d); cur = prev + (np.array(k, float) - prev) * u
        return cur
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 520); TY = D.get("title_y", 520); tH = T.E(0)
    CH = sub_chunks(T, S.get("subs_replace"))

    def frame(t):
        cv = backdrop(t)
        st = on("stand", t, 0.25); hd = on("hoard", t); ps = on("possessed", t); sw = on("sweat", t); sr = on("shrug", t, 0.2)
        gp = on("grandpa", t, 0.4)
        # kids passing the ball (hidden while the hero hoards every toy)
        lean = 40 * ps
        P = {}
        for i, (x, shirt, hair) in enumerate(KIDS):
            arms = "toss" if hd < 0.5 and (i in (0, 2)) else ("down" if ps < 0.5 else "up")
            if any(b["who"] == "kids" and b["t0"] - 0.1 <= t <= b["t1"] for b in BUB) and i == 1: arms = "wave"
            P[i] = kid(cv, x + lean, FEET, 420, shirt, hair, arms=arms, eyes="wide" if ps > 0.5 else "dot", mouth="o" if ps > 0.5 else "smile", lean=8 * ps, t=t)
        if hd < 0.5:
            ph = (t % 1.6) / 1.6; u = ph * 2 if ph < 0.5 else 2 - ph * 2
            bx = KIDS[0][0] + 40 + (KIDS[2][0] - KIDS[0][0] - 80) * u + lean; by = FEET - 330 - 200 * math.sin(math.pi * ((ph * 2) % 1))
            WG.emo(cv, "⚽", bx, by, 80, 1.0 - 2 * hd)
        if gp > 0.01:
            gx = GRANDPA_X + 200 * (1 - gp); grandpa(cv, gx, FEET, gp, t)
        hx = HERO_SIT[0] + (HERO_STAND[0] - HERO_SIT[0]) * st
        arms = "shrug" if sr > 0.5 else ("hold" if hd > 0.5 else "down")
        P["hero"] = kid(cv, hx, FEET, 360, "#8E7CC3", sit=st < 0.5, arms=arms, eyes="swirl" if ps > 0.5 else ("wide" if sw > 0.5 else "dot"),
                        mouth="frown", hero=True, glow=ps, t=t)
        if hd > 0.01:
            for k, (e, dx, dy) in enumerate((("⚽", -40, -40), ("\U0001FAA3", 30, -60), ("\U0001F3D3", -10, -110), ("\U0001F388", 60, -170), ("\U0001F6F6", -70, -150), ("\U0001F94F", 20, -10))):
                WG.emo(cv, e, hx + dx * 1.5, FEET - 170 + dy * 1.5, 100 * max(0.05, back(seg(t, W_("hoard")[0] + 0.08 * k, W_("hoard")[0] + 0.08 * k + 0.35, lambda q: q))), hd)
        if sw > 0.01:
            r = P["hero"]["r"]; hx_, hy_ = P["hero"]["head"]
            ellipse(cv, hx_ + r * 1.05, hy_ + r * 0.6 + 6 * math.sin(t * 5), 9, 15, "#5DB3E8", sw, stroke=INK, lw=2)
        for tg in TAG:
            ta = win(t, tg["t0"], tg["t1"], 0.25, 0.25)
            if ta > 0.01:
                if tg["who"] == "grandpa": x_, y_ = GRANDPA_X + 200 * (1 - gp), FEET + 50
                else: x_, y_ = hx, FEET + 55
                WG.tag(cv, tg["text"], x_, y_, 42, YEL, ta)
        img = cv.out().astype(np.float32)
        cx, cy, z = cam(t)
        x0 = min(0.0, max(W - z * W, 540 - z * cx)); y0 = min(0.0, max(H - z * H, 960 - z * cy)) + float(D.get("shift_y", 0))
        M = np.array([[z, 0, x0], [0, z, y0]], np.float32)
        if z > 1.001 or abs(y0) > 0.5: img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sp = lambda p: (p[0] * z + x0, p[1] * z + y0)
        cv = Canvas(img)
        for b in BUB:
            ba = win(t, b["t0"] - 0.05, b["t1"], 0.15, 0.2)
            if ba > 0.01:
                if b["who"] == "kids": hp = P[1]["head"]
                elif b["who"] == "grandpa": hp = (GRANDPA_X, FEET - 430)
                else: hp = P["hero"]["head"]
                tx, ty = sp(hp); bubble(cv, b["text"], tx, max(ty, 700), ba, 0.8 + 0.2 * back(seg(t, b["t0"] - 0.05, b["t0"] + 0.25, lambda q: q)))
        for ti in TIT: pill_title(cv, ti["text"], TY, win(t, ti["t0"], ti["t1"], 0.25, 0.25))
        if BK:
            ba = win(t, BK["t0"] - 0.1, BK["t1"], 0.35, 0.3)
            if ba > 0.01:
                s_ = 0.9 + 0.1 * eout(seg(t, BK["t0"], BK["t0"] + 0.4)); c = cv.c; c.save(); c.translate(540, 900 + float(D.get("book_dy", 0))); c.scale(s_, s_); c.translate(-540, -900)
                cv.rrect(250, 520, 600, 780, 26, "#F5F5F7", ba, stroke=INK, lw=4)
                cv.rrect(230, 500, 600, 780, 26, NAVY, ba, stroke=INK, lw=4)
                cv.rrect(250, 500, 24, 780, 6, "#0B1730", ba)
                cv.text(BK["title"], 540, 860, 104, YEL, ba, maxw=540)
                cv.line([(330, 960), (750, 960)], YEL, 4, 0.6 * ba)
                sa = ba * seg(t, BK["ts"] - 0.05, BK["ts"] + 0.3)
                if sa > 0.01: cv.text(BK["sub"], 540, 1070, 76, WHITE, sa, maxw=540)
                c.restore()
                if BK.get("src"):
                    sy_ = D.get("src_y", 1430); sw_ = min(1020, cv.tw(BK["src"], 38, False) + 60); cv.rrect(540 - sw_ / 2, sy_ - 38, sw_, 76, 30, NAVY, 0.9 * ba)
                    cv.text(BK["src"], 540, sy_, 38, "#F5F5F7", ba, bold=False, maxw=980, shadow=False, minsz=36)
        out = cv.out().astype(np.float32)
        if HEADS is not None:
            fi = int(round(t * FPS)); fi = HSNAP.get(fi, 0) if HSNAP is not None else min(fi, len(HEADS) - 1)
            rgba = np.asarray(HEADS[fi]); sc = float(HD.get("w", 432)) / rgba.shape[1]
            if HD.get("mode", "fixed") == "fixed": hx_, hb_ = float(HD.get("cx", 250)), float(HD.get("bottom", 1950))
            else:
                (hx_, hb_) = HCUE[0][1] if HCUE else (540.0, 1900.0)
                for k_ in range(1, len(HCUE)):
                    if t >= HCUE[k_][0]:
                        u_ = ease((t - HCUE[k_][0]) / 0.6); p0, p1 = HCUE[k_ - 1][1], HCUE[k_][1]
                        if u_ >= 1: hx_, hb_ = p1
                        else:   # glide from wherever it was
                            q = HCUE[k_ - 1][1]; hx_, hb_ = q[0] + (p1[0] - q[0]) * u_, q[1] + (p1[1] - q[1]) * u_
                b_ = float(HD.get("bob", 8)); hb_ += b_ * math.sin(2 * math.pi * t / 2.6); hx_ += b_ * 0.6 * math.sin(2 * math.pi * t / 3.7)
                pk = max([seg(t, p, p + 0.25, eout) * (1 - seg(t, p + 0.85, p + 1.2)) for p in POPS] or [0])
                sc *= 1 + (float(HD.get("pop", 1.18)) - 1) * pk
            out = HC.paste(out, rgba, hx_, hb_, sc)
        ha = 1 - seg(t, tH + 0.15, tH + 0.55)
        if ha > 0.01:
            c2 = Canvas(out); c2.rrect(40, HY - 140, 1000, 280, 50, NAVY, 0.9 * ha); out = c2.out().astype(np.float32)
            out = paste_hook(out, hook_img, t, tH, HY, 1.0)
        c3 = Canvas(out); draw_subs(c3, CH, t, D.get("subs_y", 1690), D.get("subs_size", 56))
        return np.clip(c3.out(), 0, 255).astype(np.uint8)

    ev = [(b["t0"], "pop", -24) for b in BUB] + [(ti["t0"], "whoosh", -30) for ti in TIT]
    if BK: ev += [(BK["t0"] - 0.2, "whoosh", -24), (BK["ts"], "ding", -24)]
    if D.get("possessed"): ev.append((A(D["possessed"]["at"]), "buzz", -30))
    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    aud = build_mix(S, T, work, ev, vo_path=f"{work}/vo.wav" if TK else None, dur=T.END if TK else None)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, keep=[l.get("src") for l in J["lines"]] if TK else None)
