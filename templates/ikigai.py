"""ikigai: the four-circle ikigai chart (LOVE, GOOD AT, WORLD NEEDS, PAID FOR; centre = IKIGAI) as a reel. The circles are on
screen and breathing from frame 0 with the hook over the top band. Per country: flag + name header (flag LEFT of the name),
an optional real photo in the top band (credit inside the band), and items that fly in from the side of their circle and
snap into its lobe (pop + ring), then the centre word lands in yellow. An "intruder" chip (e.g. "Colonizer") can be dropped
into the centre and tested circle by circle (red cross + note per circle), then fall out of the diagram under a stamp.
A recap cycles every country's centre word, the end card sits in the top band, and the reel loops back to frame 0.

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  geom       {cx: 540, cy: 1050, dx: 200, dy: 200, r: 290}   circle centres: love (cx, cy-dy), good (cx-dx, cy),
             needs (cx+dx, cy), paid (cx, cy+dy)
  labels     {love: {text, at}, good: {...}, needs: {...}, paid: {...}}   circle titles ("|" = line break), pop at `at`
             (missing at = shown from frame 0)
  countries  [{name, flag (ISO2), at, until, items: {love|good|needs|paid: {text, at}}, center: {text, at}}]
             text: "*x*" = yellow, "|" = line break. An item without `at` flies in 0.25 s after the previous one.
  intruder   {text, at, tests: [{key, at, note}], fall: anchor, stamp: {text, at, until}}
  photos     [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1]}]   shown in the top band (y 225 to 545)
  recap      {at, until}   the header and the centre cycle through every country (items hidden)
  hook_y     default 385;  header_y default 165;  title_y (screen cards, in the band) default 385
  src_y      default 1712; subs_y default 1612; end card: spec "end" drawn in the band
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

COLS = dict(love="#FF6B8B", good="#45A4DD", needs="#3FBF7F", paid="#FFD23F")
KEYS = ("love", "good", "needs", "paid")
BAND = (225, 545)


def _toks(txt):
    """'a *b c* d|e' -> lines of (word, yellow)"""
    out, hl = [[]], False
    for raw in txt.replace("|", " | ").split():
        if raw == "|": out.append([]); continue
        on = raw.startswith("*"); off = raw.rstrip(".,!?:;").endswith("*")
        out[-1].append((raw.replace("*", ""), hl or on)); hl = (hl or on) and not off
    return out


def rich_block(cv, txt, cx, cy, sz, maxw, a=1.0, col=WHITE, lh=1.12, min_sz=36):
    """centred multi-line text, *span* yellow, wraps at maxw, shrinks (not under min_sz) if a single word is too wide"""
    if a <= 0.01 or not txt: return
    hard = _toks(txt)
    while True:
        lines = []
        for hl_ in hard:
            cur = []
            for tk in hl_:
                if cur and cv.tw(" ".join(x[0] for x in cur + [tk]), sz) > maxw: lines.append(cur); cur = [tk]
                else: cur.append(tk)
            lines.append(cur)
        widest = max((cv.tw(" ".join(x[0] for x in ln), sz) for ln in lines if ln), default=0)
        if widest <= maxw or sz <= min_sz: break
        sz -= 2
    y0 = cy - sz * lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        if not ln: continue
        parts = [(w + (" " if k < len(ln) - 1 else ""), YEL if h else col) for k, (w, h) in enumerate(ln)]
        cv.rich(parts, cx, y0 + i * sz * lh, sz, a, maxw + 40)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 0.75)); A = T.at
    G = dict(cx=540, cy=1050, dx=200, dy=200, r=290); G.update(D.get("geom") or {})
    cx, cy, dx, dy, R = G["cx"], G["cy"], G["dx"], G["dy"], G["r"]
    CEN = dict(love=(cx, cy - dy), good=(cx - dx, cy), needs=(cx + dx, cy), paid=(cx, cy + dy))
    # lobe anchor points: title, item, note, cross mark; max text width; fly-in start point
    side = R - dx                                             # how far the side lobes reach past the centre circles
    LOBE = dict(
        love=dict(lab=(cx, cy - dy - R + 52), item=(cx, cy - dy - R + 158), note=(cx, cy - dy - R + 165), w=440, sz=64, start=(cx, -160)),
        paid=dict(lab=(cx, cy + dy + R - 52), item=(cx, cy + dy + R - 158), note=(cx, cy + dy + R - 165), w=440, sz=64, start=(cx, H + 160)),
        good=dict(lab=(cx - dx - R * 0.55, cy - 135), item=(cx - dx - R * 0.53, cy + 25), note=(cx - dx - R * 0.53, cy + 75), w=215, sz=54, start=(-320, cy)),
        needs=dict(lab=(cx + dx + R * 0.55, cy - 135), item=(cx + dx + R * 0.53, cy + 25), note=(cx + dx + R * 0.53, cy + 75), w=215, sz=54, start=(W + 320, cy)))
    LAB = {k: dict(text=(D.get("labels") or {}).get(k, {}).get("text", k.upper()),
                   t=A((D.get("labels") or {}).get(k, {}).get("at"), -1.0)) for k in KEYS}

    C = []
    for c in D["countries"]:
        t0 = A(c["at"]); t1 = A(c.get("until"), T.END + 1); its = {}; prev = t0 - 0.25
        for k in KEYS:
            it = (c.get("items") or {}).get(k)
            if not it: continue
            ta = A(it.get("at"), prev + 0.25); prev = ta; its[k] = dict(text=it["text"], t=ta)
        ce = c.get("center"); C.append(dict(c, t0=t0, t1=t1, its=its, ct=A(ce.get("at"), t0 + 1.2) if ce else None, ctext=(ce or {}).get("text")))
    IN = D.get("intruder")
    if IN:
        IN = dict(IN, t0=A(IN["at"]), tf=A(IN["fall"]), tests=[dict(x, t=A(x["at"])) for x in IN.get("tests", [])])
        st = IN.get("stamp") or {}; IN["s0"] = A(st.get("at"), IN["tf"]); IN["s1"] = A(st.get("until"), IN["s0"] + 1.6)
    RC = D.get("recap"); rc0, rc1 = (A(RC["at"]), A(RC.get("until"))) if RC else (1e9, 1e9)
    RCL = [c for c in C if c.get("recap", True) and c.get("ctext")]

    PH = []
    if D.get("photos"):
        from photo_story import _load, _kb
        for p in D["photos"]:
            im = _load(p["file"]); PH.append(dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=im))
    bh = BAND[1] - BAND[0]
    yb = np.arange(bh, dtype=np.float32)[:, None]
    BMASK = (np.clip(yb / 46, 0, 1) * np.clip((bh - yb) / 46, 0, 1))[..., None]            # soft top and bottom edge
    BSHADE = (1 - np.clip((yb - (bh - 120)) / 120, 0, 1) * 0.55)[..., None]                 # darker at the credit line

    def band_img(p, t):
        kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]; u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
        zc, xc, yc = kb[2] + (kb[5] - kb[2]) * u, kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u
        h, w = p["im"].shape[:2]; s = max(W / w, bh / h) * max(1.0, zc); cw, ch = W / s, bh / s
        x0 = min(max(xc * w - cw / 2, 0), w - cw); y0 = min(max(yc * h - ch / 2, 0), h - ch)
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        return cv2.warpAffine(p["im"], M, (W, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) * BSHADE

    def ph_a(p, t): return min(seg(t, p["t0"] - 0.25, p["t0"]), 1 - seg(t, p["t1"] - 0.25, p["t1"]))

    cards = screen_cards(S, T); SUBS = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 385); hook_end = T.E(0)
    E = S.get("end") or {}; e0 = A(E.get("at"), T.END - 3) if E else 1e9; e1 = A(D.get("end_until"), T.END - 0.5) if E else 1e9

    def active(t):
        """countries visible at t with their alpha"""
        out = []
        for c in C:
            a = min(seg(t, c["t0"] - 0.05, c["t0"] + 0.15), 1 - seg(t, c["t1"] - 0.25, c["t1"]))
            if a > 0.005: out.append((c, a))
        return out

    def fly(k, t, ta):
        """item position/scale/alpha: flies in from the side of its circle (0.45 s, back ease), then sits"""
        u = (t - ta) / 0.45
        if u < 0: return None
        sx, sy = LOBE[k]["start"]; ex, ey = LOBE[k]["item"]; e = back(min(1, u), 1.25)
        return sx + (ex - sx) * e, sy + (ey - sy) * e, min(1, u * 2.2)

    def frame(t):
        base = bars_moving_bg(t)
        # top band photo
        for p in PH:
            a_ = ph_a(p, t)
            if a_ > 0.003:
                sl = base[BAND[0]:BAND[1]]; m = BMASK * a_; base[BAND[0]:BAND[1]] = sl * (1 - m) + band_img(p, t) * m
        cv = Canvas(base); c = cv.c
        AC = active(t)
        # ---- circles (breathing from frame 0), fill alpha builds overlaps
        br = 1 + 0.012 * math.sin(t * 2.1)
        hit = {k: 0.0 for k in KEYS}; red = {k: 0.0 for k in KEYS}
        for cc, a in AC:
            for k, it in cc["its"].items():
                hit[k] = max(hit[k], a * (1 - seg(t, it["t"] + 0.45, it["t"] + 1.1)) * (t >= it["t"] + 0.4))
        if IN:
            for x in IN["tests"]:
                red[x["key"]] = max(red[x["key"]], seg(t, x["t"], x["t"] + 0.15) * (1 - seg(t, IN["s1"] - 0.3, IN["s1"])))
        for k in KEYS:
            x, y = CEN[k]; ph_ = {"love": 0, "good": 1.6, "needs": 3.1, "paid": 4.7}[k]
            rr = R * br + 4 * math.sin(t * 1.3 + ph_)
            col = COLS[k]
            cv.circ(x, y, rr, col, 0.13 + 0.10 * hit[k])
            if red[k] > 0.01: cv.circ(x, y, rr, RED, 0.16 * red[k])
            c.new_path(); c.arc(x, y, rr, 0, 2 * math.pi); cv.col(RED if red[k] > 0.5 else col, 0.85); c.set_line_width(5 + 5 * hit[k]); c.stroke()
            # a slow orbiting highlight on each ring: motion from frame 0
            ang = t * 0.7 + ph_; c.new_path(); c.arc(x, y, rr, ang, ang + 0.55); cv.col("#FFFFFF", 0.55); c.set_line_width(7); c.stroke(); c.new_path()
        # ---- circle titles
        for k in KEYS:
            la = seg(t, LAB[k]["t"], LAB[k]["t"] + 0.25) if LAB[k]["t"] >= 0 else 1.0
            if la <= 0.01: continue
            sc = WG.pop_s(t, LAB[k]["t"]) if LAB[k]["t"] >= 0 and t < LAB[k]["t"] + 0.4 else 1.0
            lx, ly = LOBE[k]["lab"]; parts = LAB[k]["text"].split("|"); lz = 42 * sc
            for i, ptxt in enumerate(parts):
                cv.text(ptxt, lx, ly + (i - (len(parts) - 1) / 2) * 46, lz, COLS[k], la, maxw=LOBE[k]["w"], minsz=36)
        # ---- centre
        ctr = None
        for cc, a in AC:
            if cc["ctext"] and t >= cc["ct"]: ctr = (cc, a)
        if RC and rc0 <= t < rc1:
            k_ = min(len(RCL) - 1, int((t - rc0) / max(0.2, (rc1 - rc0) / len(RCL)))); ctr = (RCL[k_], 1.0)
        if ctr:
            cc, a = ctr; ct = cc["ct"] if not (RC and rc0 <= t < rc1) else rc0 + (rc1 - rc0) / len(RCL) * RCL.index(cc)
            sc = WG.pop_s(t, ct) if t < ct + 0.4 else 1.0
            cv.circ(cx, cy, 112 * sc, "#0B1730", 0.92 * a, stroke=YEL, lw=5)
            if t < ct + 0.6:
                u = (t - ct) / 0.6; c.new_path(); c.arc(cx, cy, 112 + 120 * u, 0, 2 * math.pi); cv.col(YEL, 0.6 * (1 - u) * a); c.set_line_width(6); c.stroke(); c.new_path()
            rich_block(cv, cc["ctext"], cx, cy, 46 * sc, 200, a, col=YEL)
        else:
            cv.text("IKIGAI", cx, cy, 40, "#C9D6EA", 0.9, maxw=200)
        # ---- items
        if not (RC and rc0 <= t < rc1):
            for cc, a in AC:
                for k, it in cc["its"].items():
                    f = fly(k, t, it["t"])
                    if not f: continue
                    x, y, fa = f; L_ = LOBE[k]
                    if t < it["t"] + 0.75 and t >= it["t"] + 0.42:          # snap ring
                        u = (t - it["t"] - 0.42) / 0.33; c.new_path(); c.arc(x, y, 60 + 120 * u, 0, 2 * math.pi)
                        cv.col(COLS[k], 0.7 * (1 - u) * a); c.set_line_width(5); c.stroke(); c.new_path()
                    rich_block(cv, it["text"], x, y, L_["sz"], L_["w"], a * fa)
        # ---- intruder: chip in the centre, tests, fall, stamp
        if IN and IN["t0"] - 0.05 <= t < IN["s1"]:
            for x in IN["tests"]:
                if t < x["t"]: continue
                ta = 1 - seg(t, IN["s1"] - 0.3, IN["s1"]); L_ = LOBE[x["key"]]
                lx, ly = L_["lab"]
                if x["key"] in ("love", "paid"):
                    mx = lx + cv.tw(LAB[x["key"]]["text"].replace("|", " "), 42) / 2 + 44; my = ly
                else: mx, my = lx, ly + 80
                WG.mark(cv, "N", mx, my, 30, ta, WG.pop_s(t, x["t"]))
                if x.get("note"):
                    nx, ny = L_["note"]
                    if x["key"] in ("good", "needs"): ny += 40
                    rich_block(cv, x["note"], nx, ny, 50 if x["key"] in ("love", "paid") else 44, L_["w"], ta * seg(t, x["t"], x["t"] + 0.2))
            # the chip
            u = (t - IN["t0"]) / 0.5
            if t < IN["tf"]:
                e = back(min(1, max(0, u)), 1.3); px, py = cx, HY + (cy - HY) * e; rot = 0.0; ca = min(1, max(0, u * 2))
                shake = 6 * math.sin(t * 40) * sum(seg(t, x["t"], x["t"] + 0.05) * (1 - seg(t, x["t"] + 0.25, x["t"] + 0.35)) for x in IN["tests"])
                px += shake
            else:
                g = t - IN["tf"]; px, py = cx + 140 * g, cy + 1200 * g * g + 80 * g; rot = 1.4 * g; ca = 1.0
            if ca > 0.01 and py < H + 200:
                tw = cv.tw(IN["text"], 66) + 80
                c.save(); c.translate(px, py); c.rotate(rot)
                cv.rrect(-tw / 2, -58, tw, 116, 58, RED, 0.95 * ca, stroke="#FFFFFF", lw=4, sa=ca)
                cv.text(IN["text"], 0, 0, 66, "#FFFFFF", ca, maxw=900)
                c.restore()
            st = IN.get("stamp")
            if st and t >= IN["s0"]:
                sa_ = 1 - seg(t, IN["s1"] - 0.3, IN["s1"]); sc = 1 + 0.7 * (1 - eout((t - IN["s0"]) / 0.18))
                c.save(); c.translate(cx, cy); c.rotate(math.radians(-9)); c.scale(sc, sc)
                tw = cv.tw(st["text"], 92) + 90
                cv.rrect(-tw / 2, -90, tw, 180, 26, "#0B1730", 0.55 * sa_, stroke=RED, lw=10, sa=sa_)
                cv.text(st["text"], 0, 0, 92, RED, sa_, maxw=960, shadow=False)
                c.restore()
        # ---- header: flag LEFT of the name
        hy = D.get("header_y", 165)
        hdr = [(cc, a) for cc, a in AC]
        if RC and rc0 <= t < rc1: hdr = [(ctr[0], 1.0)] if ctr else []
        for cc, a in hdr:
            sz = 84; nw = min(820, cv.tw(cc["name"], sz)); fw = sz * 1.1 if cc.get("flag") else 0; gap = 24 if fw else 0
            x0 = 540 - (fw + gap + nw) / 2
            if cc.get("flag"): WG.flag(cv, cc["flag"], x0 + fw / 2, hy, sz, a)
            cv.text(cc["name"], x0 + fw + gap, hy, sz, WHITE, a, align="l", maxw=820)
        # ---- band: screen cards (only where no photo), photo credits, end card
        pa = max([ph_a(p, t) for p in PH] + [0.0])
        draw_screen(cv, [k for k in cards if k[1] > hook_end], t, D.get("title_y", 385), 72)
        for k_, p in enumerate(PH):
            a_ = ph_a(p, t)
            if a_ > 0.01 and p.get("credit"):
                cv.text("Photo: " + p["credit"], 540, BAND[1] - 32, 36, "#E3EAF5", 0.9 * a_, bold=False, maxw=1010, minsz=36)
        if E:
            a = win(t, e0 - 0.1, e1, 0.35, 0.3)
            if a > 0.01:
                WG.panel(cv, 40, BAND[0], 1000, BAND[1] - BAND[0], a)
                if E.get("balance"): WG.title(cv, E["balance"], BAND[0] + 105, a, 62)
                if E.get("cta"):
                    ca = a * seg(t, e0 + 0.6, e0 + 1.0); tw = min(940, cv.tw(E["cta"].replace("*", ""), 54) + 80)
                    cv.rrect(540 - tw / 2, BAND[1] - 120, tw, 96, 48, SKY, ca); WG.title(cv, E["cta"], BAND[1] - 72, ca, 54)
        draw_sources(cv, S, T, t, D.get("src_y", 1712))
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, hook_end, HY, 0.9)
        c2 = Canvas(out); draw_subs(c2, SUBS, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p["t0"], "whoosh", -32) for p in PH]
    for k in KEYS:
        if LAB[k]["t"] >= 0: ev.append((LAB[k]["t"], "pop", -26))
    for cc in C:
        for k, it in cc["its"].items(): ev += [(it["t"], "whoosh", -34), (it["t"] + 0.42, "pop", -24)]
        if cc["ct"]: ev.append((cc["ct"], "ding", -26))
    if IN:
        ev.append((IN["t0"], "whoosh", -28))
        for x in IN["tests"]: ev.append((x["t"], "buzz", -28))
        ev.append((IN["tf"], "whoosh", -30)); ev.append((IN["s0"], "thud", -20))
    if RC:
        for k in range(len(RCL)): ev.append((rc0 + (rc1 - rc0) / len(RCL) * k, "tick", -22))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
