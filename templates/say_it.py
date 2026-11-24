"""say_it: "Words for what you already feel" (24 Nov 2026, series say_it_1 to say_it_5). A typographic reel that hands
the viewer ONE quotable sentence for something they feel but struggle to say, built for saves and shares.

Beats (each a full screen phase, eased slide + fade between phases):
  frame 0  hook (max 6 words, a number) over a pulsing "..." speech bubble and a small sky "you" figure: the thing you
           can't quite say. Motion from frame 0 (dots bounce, bubble breathes), "Israel Facts" mark top left, Part pill.
  scene    1 to 2 s situation, drawn simply, no faces: "dinner" (table, seated stick figures), "chat" (phone group chat),
           "comments" (comment section), "office" (two figures at a desk). The other person's line pops in a bubble; the
           sky "you" figure answers with typing dots.
  hero     THE SENTENCE: huge type, each word lands on its voice timestamp (spoken slowly: split it over 1 or 2 lines
           with a gap), *key words* yellow, a "SAY THIS" pill and a big quote mark.
  facts    1 to 3 sourced facts: real photo band edge to edge on top (Ken Burns, credit inside the band), a big yellow
           number that counts up (or a short big word), one line of text; the source chip comes from spec "sources".
  save     the sentence again, shorter: bookmark, "SAVE THIS", short line, "send it to the friend who..." line.
  debate   neutral question ("Which sentence do you need most?") with the series' sentences as chips (this part lit).
  follow   the standard follow line ({"follow": true} spec line), sky chevron cue, "Part N+1 on my profile".
  loop     the last line brings back the frame-0 composition, then the last 0.5 s dissolves into frame 0.

data keys (anchors as everywhere: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  part      {"n": 1, "of": 5, "next": "Part 2 on my profile"}
  hook_y    hook centre y (default 520)
  scene     {kind: dinner|chat|comments|office, at, says: "their line", says_at, label: "Dinner with friends",
             msgs: [{who: "them"|"me", text, at}] (chat/comments), reply_at}
  hero      {lines: [3, 4], text: "I'm *pro-Israel* and *pro-peace.*|Those are not *opposites.*"}  text words = the
            spoken words of those lines, in order ("|" = forced break); at defaults to the first line's start
  facts     [{at, photos: [{file, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], at}], big: "1979" | number,
              count: {from, to, dec, prefix, suffix}, text: "Peace with *Egypt.*", label: "small line over big"}]
  save      {at, text: "Pro-Israel. Pro-peace.|*Not opposites.*", sub: "Send it to the friend who gets asked this."}
  debate    {at, text: "Which sentence do you need most?", chips: ["Pro-peace", ...], hl: 0}
  loop      {at}  (default: start of the last line)
  duck_db   music duck under speech (default 14: music about 4% under the voice, Noam's One Peak rule)
"""
import os, sys, re, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np, cv2
from common import *

INK = "#0B1730"


def _fake_timing(S):
    """local look without a voice (TPL_FAKE=1): 2.7 words per second, spec gaps"""
    t, out = 0.3, []
    for i, L in enumerate([L for L in S["lines"] if L.get("say", "").strip()]):
        ws = L["say"].split(); d = len(ws) / 2.7
        out.append(dict(i=i, text=L["say"], t0=t, t1=t + d, words=[dict(w=w, t0=t + d * k / len(ws), t1=t + d * (k + 1) / len(ws)) for k, w in enumerate(ws)]))
        t += d + float(L.get("gap", 0.15))
    return dict(dur=t, voice="fake", lines=out)


def _cover(im, w, h, cx, cy, z):
    ih, iw = im.shape[:2]; s = max(w / iw, h / ih) * max(1.0, z); cw, ch = w / s, h / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def _tokens(txt):
    """'I'm *pro-Israel* and|Those' -> [(word, yellow, break_before)]"""
    out, hl, brk = [], False, False
    for chunk in txt.replace("|", " | ").split():
        if chunk == "|": brk = True; continue
        on = chunk.startswith("*"); off = chunk.rstrip(".,!?:;").endswith("*")
        out.append((chunk.replace("*", ""), on or hl, brk)); hl = (hl or on) and not off; brk = False
    return out


def _layout(cv, toks, sz, maxw=960):
    """greedy wrap honouring forced breaks -> list of lines of token indexes"""
    lines, cur, w = [], [], 0.0
    sp = cv.tw(" ", sz)
    for k, (wd, _, brk) in enumerate(toks):
        ww = cv.tw(wd, sz)
        if cur and (brk or w + sp + ww > maxw): lines.append(cur); cur, w = [], 0.0
        w = ww if not cur else w + sp + ww; cur.append(k)
    if cur: lines.append(cur)
    return lines


def _rich_block(cv, txt, cy, sz, a, maxw=960, maxlines=4, col=WHITE):
    """centred multi line text with *yellow* words, auto shrink to maxlines"""
    toks = _tokens(txt)
    while True:
        lines = _layout(cv, toks, sz, maxw)
        if len(lines) <= maxlines or sz <= 64: break
        sz -= 4
    lh = sz * 1.16; y0 = cy - lh * (len(lines) - 1) / 2; sp = cv.tw(" ", sz)
    for li, ln in enumerate(lines):
        tot = sum(cv.tw(toks[k][0], sz) for k in ln) + sp * (len(ln) - 1); x = 540 - tot / 2
        for k in ln:
            wd, yl, _ = toks[k]
            cv.text(wd, x, y0 + li * lh, sz, YEL if yl else col, a, align="l", maxw=2000)
            x += cv.tw(wd, sz) + sp
    return sz, len(lines)


def _bubble(cv, x, y, w, h, a, fill=WHITE, tail="left", stroke=None):
    """speech bubble, top left (x, y); tail at the bottom left or right"""
    if a <= 0.01: return
    cv.rrect(x, y, w, h, 36, fill, a, stroke=stroke, lw=3, sa=a)
    tx = x + 70 if tail == "left" else x + w - 70; d = -1 if tail == "left" else 1
    c = cv.c; c.new_path(); c.move_to(tx - 26, y + h - 2); c.line_to(tx + 26, y + h - 2); c.line_to(tx + d * 40, y + h + 44); c.close_path()
    cv.col(fill, a); c.fill(); c.new_path()


def _dots(cv, cx, cy, t, a, col=INK, r=13):
    for k in range(3):
        b = max(0.0, math.sin((t * 2 * math.pi * 1.3) - k * 0.9))
        cv.circ(cx + (k - 1) * 40, cy - 12 * b, r, col, a * (0.5 + 0.5 * b))


def _wrap(cv, txt, sz, maxw):
    out, cur = [], ""
    for w in txt.split():
        tst = (cur + " " + w).strip()
        if cur and cv.tw(tst.replace("*", ""), sz) > maxw: out.append(cur); cur = w
        else: cur = tst
    if cur: out.append(cur)
    return out


def _bubble_text(cv, txt, x, y, w, sz, a, col=INK, fill=WHITE, tail="left"):
    """bubble sized to its wrapped text (max 3 lines), returns its height"""
    lines = _wrap(cv, txt, sz, w - 80)
    while len(lines) > 3 and sz > 44: sz -= 4; lines = _wrap(cv, txt, sz, w - 80)
    h = len(lines) * sz * 1.2 + 64
    _bubble(cv, x, y, w, h, a, fill, tail)
    for k, ln in enumerate(lines):
        cv.text(ln.replace("*", ""), x + w / 2, y + 32 + sz * 1.2 * (k + 0.5), sz, col, a, shadow=False, maxw=w - 60, minsz=40)
    return h


def run(S):
    D = S["data"]; work = workdir(S)
    J = _fake_timing(S) if os.environ.get("TPL_FAKE") else build_voice(S, work)
    T = Timing(J, D.get("tail", 0.8)); A = T.at; END = T.END
    SAY = [L for L in S["lines"] if L.get("say", "").strip()]
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 520)
    P = D.get("part") or {}
    fidx = next((i for i, L in enumerate(SAY) if L.get("follow")), None)

    # ---------------------------------------------------------------- phases
    ph = []
    def add(kind, at, obj=None): ph.append([A(at), kind, obj])
    add("intro", 0.0)
    if D.get("scene"): add("scene", D["scene"].get("at", "L1"), D["scene"])
    H_ = D["hero"]; hl = H_.get("lines", [2]); add("hero", H_.get("at", f"L{hl[0]}-0.15"), H_)
    for f in D.get("facts", []): add("fact", f["at"], f)
    if D.get("save"): add("save", D["save"]["at"], D["save"])
    if D.get("debate"): add("debate", D["debate"]["at"], D["debate"])
    if fidx is not None: add("follow", T.S(fidx) - 0.15)
    add("loop", (D.get("loop") or {}).get("at", f"L{len(SAY) - 1}-0.1"))
    ph.sort(key=lambda p: p[0])
    for k, p in enumerate(ph): p.append(ph[k + 1][0] if k + 1 < len(ph) else END + 1)

    # hero words: one token per spoken word of the hero lines
    toks = _tokens(H_["text"])
    hw = [w for i in hl for w in T.L[i]["words"]]
    if len(hw) != len(toks):
        raise SystemExit(f"hero text has {len(toks)} words, hero lines {hl} speak {len(hw)}: make them the same words")
    hero_t = [w["t0"] - 0.06 for w in hw]

    # photos
    PH = {}
    for f in D.get("facts", []):
        for p in f.get("photos", []):
            if p["file"] not in PH:
                im = cv2.imread(os.path.join(REPO, p["file"]))
                if im is None: raise SystemExit(f"photo missing: {p['file']}")
                PH[p["file"]] = im
    BAND_H = 760
    yy = np.linspace(0, 1, BAND_H, dtype=np.float32)[:, None, None]
    BAND_FADE = np.clip((1 - yy) / 0.22, 0, 1)            # bottom 22 % of the band fades into the background

    CH = sub_chunks(T, S.get("subs_replace"))
    LOOP_T = next(p[0] for p in ph if p[1] == "loop")

    def slide(t, t0, t1):
        """phase alpha and y offset: eased slide up on entry, fade on exit"""
        a = win(t, t0, t1, 0.3, 0.25); dy = 46 * (1 - eout((t - t0) / 0.45)) if t < t0 + 0.45 else 0.0
        return a, dy

    # ---------------------------------------------------------------- drawers
    def draw_intro(cv, t, a, dy, t0):
        """frame-0 composition (also the loop target): typing bubble + small sky figure"""
        br = 1 + 0.025 * math.sin(t * 3.1)
        bw, bh = 520 * br, 210 * br; bx, by = 540 - bw / 2, 860 - bh / 2 + dy
        _bubble(cv, bx, by, bw, bh, a * 0.96, WHITE, "left")
        _dots(cv, 540, by + bh / 2, t, a, INK, 20)
        figure(cv, 400, 1330 + dy, 1.9, SKY, badge=True, arms="shrug" if (t % 2.4) < 1.2 else "down", a=a, lw=8)

    def draw_scene(cv, t, a, dy, t0, sc):
        k = sc.get("kind", "dinner")
        if sc.get("label"):
            cv.text(sc["label"], 540, 300 + dy, 64, "#C9D6EA", a, maxw=960)
        says_t = A(sc.get("says_at"), t0 + 0.25); rep_t = A(sc.get("reply_at"), says_t + 0.9)
        if k == "dinner":
            ty = 1180 + dy
            for x, col in ((270, WHITE), (540, WHITE), (810, SKY)):
                figure(cv, x, ty + 10, 1.7, col, badge=(col == SKY), sit=True, a=a, lw=8, flip=1 if x < 540 else -1)
            cv.rrect(100, ty - 30, 880, 70, 22, "#8A5A35", a); cv.rrect(80, ty - 44, 920, 28, 14, "#F5F5F7", a)
            for x in (270, 540, 810): cv.circ(x, ty - 52, 34, "#FFFFFF", a, stroke="#C9D6EA", lw=3)
            for x in (380, 680): cv.rrect(x - 8, ty - 120, 16, 76, 6, "#C9D6EA", 0.8 * a); cv.circ(x, ty - 128, 22, "#E5484D", 0.55 * a)
            ba = a * seg(t, says_t - 0.05, says_t + 0.25)
            if ba > 0.01:
                s = 0.9 + 0.1 * WG.pop_s(t, says_t)
                _bubble_text(cv, sc["says"], 120, 520 + dy - 30 * (1 - s), 700, 62, ba, INK, WHITE, "left")
            ra = a * seg(t, rep_t, rep_t + 0.25)
            if ra > 0.01:
                _bubble(cv, 700, 830 + dy, 300, 140, ra, SKY, "right"); _dots(cv, 850, 900 + dy, t, ra, WHITE, 14)
        elif k == "office":
            gy = 1320 + dy
            cv.rrect(140, gy - 150, 800, 34, 12, "#C9D6EA", a)                          # desk
            cv.rrect(440, gy - 290, 200, 140, 14, "#2A4A80", a, stroke="#C9D6EA", lw=4)   # laptop
            cv.rrect(410, gy - 156, 260, 12, 6, "#C9D6EA", a)
            figure(cv, 260, gy, 1.9, WHITE, badge=False, arms="point", a=a, lw=8, flip=1)
            figure(cv, 820, gy, 1.9, SKY, badge=True, arms="shrug", a=a, lw=8)
            ba = a * seg(t, says_t - 0.05, says_t + 0.25)
            if ba > 0.01: _bubble_text(cv, sc["says"], 90, 560 + dy, 720, 62, ba, INK, WHITE, "left")
            ra = a * seg(t, rep_t, rep_t + 0.25)
            if ra > 0.01: _bubble(cv, 680, 880 + dy, 300, 140, ra, SKY, "right"); _dots(cv, 830, 950 + dy, t, ra, WHITE, 14)
        else:                                                    # chat / comments: a phone
            px, py, pw, ph_ = 150, 430 + dy, 780, 1080
            cv.rrect(px, py, pw, ph_, 60, "#0E1F45", a, stroke="#C9D6EA", lw=5, sa=0.8 * a)
            cv.rrect(px + 300, py + 22, 180, 22, 11, "#1D3566", a)
            head = "Group chat" if k == "chat" else "Comments"
            cv.text(head, px + pw / 2, py + 100, 52, WHITE, a, shadow=False)
            cv.line([(px + 30, py + 150), (px + pw - 30, py + 150)], "#2A4A80", 3, a)
            y = py + 190
            for m in sc.get("msgs", []):
                mt_ = A(m.get("at"), says_t); ma = a * seg(t, mt_ - 0.05, mt_ + 0.25)
                if ma <= 0.01: continue
                me = m.get("who") == "me"
                if k == "comments":
                    cv.circ(px + 70, y + 40, 30, SKY if me else "#9FB0C8", ma)
                    cv.text(m.get("name", "you" if me else "user"), px + 120, y + 20, 38, "#9FB0C8", ma, align="l", bold=False, shadow=False)
                    lines = _wrap(cv, m["text"], 54, pw - 170)
                    for j, ln in enumerate(lines): cv.text(ln, px + 120, y + 74 + j * 64, 54, WHITE, ma, align="l", shadow=False, maxw=pw - 170)
                    cv.text("♥", px + pw - 50, y + 40, 44, "#E5484D", ma * 0.8, shadow=False)
                    y += 110 + 64 * len(lines)
                else:
                    lines = _wrap(cv, m["text"], 54, 520)
                    bw = min(600, max(cv.tw(l, 54) for l in lines) + 70); bh = len(lines) * 66 + 44
                    bx = px + pw - 40 - bw if me else px + 40
                    cv.rrect(bx, y, bw, bh, 30, SKY if me else "#E9EEF6", ma)
                    for j, ln in enumerate(lines):
                        cv.text(ln, bx + 35, y + 22 + 66 * (j + 0.5), 54, WHITE if me else INK, ma, align="l", shadow=False, maxw=bw - 50)
                    y += bh + 30
            ra = a * seg(t, rep_t, rep_t + 0.25)
            if ra > 0.01:
                if k == "comments":
                    cv.rrect(px + 40, py + ph_ - 150, pw - 80, 100, 50, "#1D3566", ra, stroke=SKY, lw=3, sa=ra)
                    _dots(cv, px + 200, py + ph_ - 100, t, ra, WHITE, 12)
                else:
                    cv.rrect(px + pw - 260, y, 220, 100, 30, SKY, ra); _dots(cv, px + pw - 150, y + 52, t, ra, WHITE, 12)

    def draw_hero(cv, t, a, dy, t0):
        pa = a * seg(t, t0, t0 + 0.3)
        tw_ = cv.tw("SAY THIS", 46) + 70
        cv.rrect(540 - tw_ / 2, 430 + dy, tw_, 84, 42, SKY, pa); cv.text("SAY THIS", 540, 472 + dy, 46, WHITE, pa, shadow=False)
        cv.text("“", 140, 640 + dy, 260, YEL, pa * 0.9, align="c", shadow=False)
        sz = 112
        while True:
            lines = _layout(cv, toks, sz, 940)
            if len(lines) <= 5 or sz <= 72: break
            sz -= 4
        lh = sz * 1.17; y0 = 1000 - lh * (len(lines) - 1) / 2 + dy; sp = cv.tw(" ", sz)
        done = t > hero_t[-1] + 0.6
        pulse = 1.0
        for li, ln in enumerate(lines):
            tot = sum(cv.tw(toks[k][0], sz) for k in ln) + sp * (len(ln) - 1); x = 540 - tot / 2
            for k in ln:
                wd, yl, _ = toks[k]; u = eout((t - hero_t[k]) / 0.28) if t >= hero_t[k] else 0.0
                if u > 0.01:
                    cv.text(wd, x, y0 + li * lh + 26 * (1 - u), sz, YEL if yl else WHITE, a * u, align="l", maxw=3000)
                    if yl and done:          # underline sweep under the yellow words once the sentence is complete
                        g = seg(t, hero_t[-1] + 0.6, hero_t[-1] + 1.1)
                        cv.rrect(x, y0 + li * lh + sz * 0.55, cv.tw(wd, sz) * g, 10, 5, YEL, a * 0.85)
                x += cv.tw(wd, sz) + sp

    def fact_photo(t, a, t0, f):
        """-> (band image, alpha, credit) of the photo showing now, or None"""
        ps = f.get("photos", [])
        if not ps: return None
        cur = None
        for p in ps:
            if t >= A(p.get("at"), t0) - 0.01: cur = p
        cur = cur or ps[0]; pt0 = max(t0, A(cur.get("at"), t0))
        kb = cur.get("kb", [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]); u = min(1.0, max(0.0, (t - pt0) / 6.0))
        img = _cover(PH[cur["file"]], W, BAND_H, kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
        return img, a * seg(t, pt0 - 0.05, pt0 + 0.3), cur.get("credit")

    def draw_fact(cv, t, a, dy, t0, f):
        ps = f.get("photos", [])
        y_big = 1040 if ps else 820
        if f.get("label"): cv.text(f["label"], 540, y_big - 130 + dy, 52, "#C9D6EA", a, bold=False, maxw=960)
        c = f.get("count")
        if c:
            v = WG.roll(t, t0 + 0.15, c.get("from", 0), c["to"], c.get("dur", 1.0))
            WG.counter(cv, v, 540, y_big + dy, f.get("size", 190), a, c.get("dec", 0), c.get("prefix", ""), c.get("suffix", ""))
        elif f.get("big"):
            cv.text(str(f["big"]), 540, y_big + dy, f.get("size", 170), YEL, a * WG.pop_s(t, t0 + 0.1) if t < t0 + 0.45 else a, maxw=980)
        if f.get("text"):
            ta = a * seg(t, t0 + 0.35, t0 + 0.65)
            _rich_block(cv, f["text"], y_big + 210 + dy, 70, ta, 960, 2)

    def draw_save(cv, t, a, dy, t0, sv):
        WG.emo(cv, "\U0001F516", 540, 420 + dy, 130, a * WG.pop_s(t, t0) if t < t0 + 0.4 else a)
        cv.text("SAVE THIS", 540, 560 + dy, 52, SKY, a)
        WG.panel(cv, 50, 650 + dy, 980, 520, a, stroke=YEL)
        _rich_block(cv, sv["text"], 910 + dy, 104, a, 900, 3)
        if sv.get("sub"):
            sa = a * seg(t, t0 + 0.5, t0 + 0.8)
            WG.emo(cv, "\U0001F4E4", 540, 1270 + dy, 80, sa)
            _rich_block(cv, sv["sub"], 1390 + dy, 58, sa, 940, 2, "#C9D6EA")

    def draw_debate(cv, t, a, dy, t0, db):
        WG.emo(cv, "\U0001F4AC", 540, 430 + dy, 120, a)
        _rich_block(cv, db["text"], 640 + dy, 88, a, 940, 2)
        chips = db.get("chips", []); y = 840 + dy
        for k, ch in enumerate(chips):
            ca = a * seg(t, t0 + 0.25 + 0.12 * k, t0 + 0.5 + 0.12 * k); on = k == db.get("hl")
            w = min(940, cv.tw(f"{k + 1}.  {ch}", 56) + 80)
            cv.rrect(540 - w / 2, y - 50, w, 100, 50, YEL if on else "#13274F", ca, stroke=None if on else SKY, lw=3, sa=ca * 0.7)
            cv.text(f"{k + 1}.  {ch}", 540, y, 56, INK if on else WHITE, ca, shadow=False, maxw=900)
            y += 128

    def draw_follow(cv, t, a, dy, t0):
        WG.panel(cv, 60, 720 + dy, 960, 360, a)
        WG.title(cv, FOLLOW_SCREEN, 860 + dy, a, 72)
        k = (t * 1.6) % 1.0
        for j in range(2):
            yy_ = 990 + dy + j * 30 + 14 * k
            cv.line([(500, yy_), (540, yy_ + 30), (580, yy_)], SKY, 10, a * (0.5 + 0.5 * j))
        if P.get("next"): cv.text(P["next"], 540, 1200 + dy, 58, "#C9D6EA", a * seg(t, t0 + 0.3, t0 + 0.6), maxw=960)

    # ---------------------------------------------------------------- frame
    def frame(t):
        base = bars_moving_bg(t, 8)
        act = []
        for t0, kind, obj, t1 in ph:
            if t < t0 - 0.05 or t > t1 + 0.3: continue
            if kind != "intro" and t < t0: continue
            a, dy = slide(t, t0, t1 + 0.08 if kind != "loop" else END + 2)
            if kind == "intro": dy = 0.0; a = 1 - seg(t, t1 - 0.17, t1 + 0.08)     # frame 0 fully drawn (no fade in)
            if a > 0.01: act.append((t0, kind, obj, a, dy))
        credits = []
        for t0, kind, obj, a, dy in act:               # photo band first (edge to edge, fades into the background)
            if kind != "fact": continue
            r = fact_photo(t, a, t0, obj)
            if not r: continue
            img, pa, cr = r; m = BAND_FADE * pa
            base[150:150 + BAND_H] = base[150:150 + BAND_H] * (1 - m) + img * m
            g = np.linspace(0, 0.6, 220, dtype=np.float32)[:, None, None] * pa          # dark foot behind the credit
            base[150 + BAND_H - 300:150 + BAND_H - 80] *= (1 - g)
            if cr: credits.append((cr, pa))
        cv = Canvas(base)
        for cr, pa in credits: cv.text(cr, 40, 150 + BAND_H - 130, 36, "#E3EAF5", pa, align="l", bold=False, maxw=1000, minsz=36)
        for t0, kind, obj, a, dy in act:
            if kind in ("intro", "loop"): draw_intro(cv, t, a, dy, t0)
            elif kind == "scene": draw_scene(cv, t, a, dy, t0, obj)
            elif kind == "hero": draw_hero(cv, t, a, dy, t0)
            elif kind == "fact": draw_fact(cv, t, a, dy, t0, obj)
            elif kind == "save": draw_save(cv, t, a, dy, t0, obj)
            elif kind == "debate": draw_debate(cv, t, a, dy, t0, obj)
            elif kind == "follow": draw_follow(cv, t, a, dy, t0)
        out = cv.out().astype(np.float32)
        # hook: frame 0 until the end of line 0, and back in the loop phase (so the dissolve lands text on text)
        out = paste_hook(out, hook_img, t, T.E(0), HY, 0.92)
        if t >= LOOP_T: out = HK.paste(out, hook_img, 540, HY, 0.92, seg(t, LOOP_T, LOOP_T + 0.4))
        c2 = Canvas(out)
        c2.circ(62, 92, 8, YEL); c2.text("Israel Facts", 80, 92, 38, "#C9D6EA", 1.0, align="l", shadow=True, bold=True)
        if P.get("n"):
            lab = f"Part {P['n']}"; w = c2.tw(lab, 38) + 44
            c2.rrect(1040 - w, 62, w, 60, 30, "#13274F", 0.9, stroke=SKY, lw=2, sa=0.8); c2.text(lab, 1040 - w / 2, 92, 38, WHITE, 1.0, shadow=False)
        draw_sources(c2, S, T, t)
        draw_subs(c2, CH, t, D.get("subs_y", 1640))
        return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(p[0], "whoosh", -30) for p in ph if p[1] not in ("intro",)]
    ev += [(x, "tick", -32) for x in hero_t]
    if D.get("scene"): ev.append((A(D["scene"].get("says_at"), A(D["scene"].get("at", "L1")) + 0.25), "pop", -26))
    if D.get("save"): ev.append((A(D["save"]["at"]), "ding", -30))
    S.setdefault("music", {}).setdefault("duck_db", D.get("duck_db", 14))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, phases=[[round(p[0], 2), p[1]] for p in ph])
