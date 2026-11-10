"""slang_scene (Express): a stick-figure scene first (someone says the line), the Hebrew word as a small bottom line,
2 or 3 English options, a mini chat that uses one of them, and "Save this for your next meeting".
Colour code: blue = active (the speaker and the words to say), orange = passive (the listener), red = new words.
Brand: navy #163261, sky #45A4DD, yellow.

Text markup in bubble / options / chat: *word* = red (new), _word_ = orange (passive). Everything else white.
data keys:
  he         the Hebrew word or phrase (small bottom line, Heebo, no niqqud)
  setting    office | coffee | zoom (simple props)
  speaker    {name, says, at}                 the scene: name tag + speech bubble
  options    [{tag, text, at}]                2 or 3 ("Casual", "Or", "At work")
  chat       [{who: them | me, text, at}]     mini chat, 2 to 4 bubbles
  save       {at, text}                       default "Save this for your next meeting"
  hook_y     default 330
"""
import os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *

ORANGE = "#F2A93B"; NEW = "#FF5A5F"; BLUE = SKY


def parts(txt, base=WHITE):
    out = []
    for tok in re.split(r"(\*[^*]+\*|_[^_]+_)", txt):
        if not tok: continue
        if tok.startswith("*"): out.append((tok[1:-1], NEW))
        elif tok.startswith("_"): out.append((tok[1:-1], ORANGE))
        else: out.append((tok, base))
    return out


def rich_wrap(cv, txt, cx, cy, sz, a, maxw, base=WHITE, align="c"):
    """two-line wrap of marked-up text, centred (or left at cx)"""
    ps = parts(txt, base); words = []
    for p, c in ps:
        if p[:1].isspace() and words: words[-1] = (words[-1][0].rstrip() + " ", words[-1][1])
        for w in re.findall(r"\S+\s*", p): words.append((w, c))
    lines = [[]]
    for w in words:
        test = "".join(x for x, _ in lines[-1] + [w]).strip()
        if lines[-1] and cv.tw(test, sz) > maxw: lines.append([w])
        else: lines[-1].append(w)
    while len(lines) > 2 and sz > 48: return rich_wrap(cv, txt, cx, cy, sz - 6, a, maxw, base, align)
    lh = sz * 1.2; y0 = cy - lh * (len(lines) - 1) / 2
    for li, ln in enumerate(lines):
        if ln: ln[-1] = (ln[-1][0].rstrip(), ln[-1][1])
        wsum = sum(cv.tw(x, sz) for x, _ in ln); x = cx - wsum / 2 if align == "c" else cx
        for w, c in ln: cv.text(w, x, y0 + li * lh, sz, c, a, align="l", maxw=2000, shadow=False); x += cv.tw(w, sz)
    return len(lines) * lh


def props(cv, kind, a):
    if kind == "office":
        cv.rrect(170, 1090, 740, 26, 10, "#2B4A80", a); cv.line([(220, 1116), (220, 1250)], "#2B4A80", 10, a); cv.line([(860, 1116), (860, 1250)], "#2B4A80", 10, a)
        cv.rrect(470, 1020, 140, 72, 8, "#0E2250", a, stroke=SKY, lw=3)
    elif kind == "coffee":
        cv.rrect(400, 1095, 280, 22, 10, "#2B4A80", a); cv.line([(540, 1117), (540, 1250)], "#2B4A80", 10, a); cv.line([(480, 1250), (600, 1250)], "#2B4A80", 10, a)
        for x in (465, 615): cv.rrect(x - 20, 1049, 40, 46, 9, "#E8EEF8", a); cv.line([(x + 20, 1062), (x + 32, 1062), (x + 32, 1080), (x + 20, 1080)], "#E8EEF8", 5, a)
    elif kind == "zoom":
        for k in range(2):
            cv.rrect(110 + k * 450, 760, 410, 300, 22, "#0E2250", a, stroke=SKY, lw=3)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 2.0)); A = T.at
    SP = D["speaker"]; t_sp = A(SP.get("at"), 0.5) if SP.get("at") is not None else 0.5
    OP = [dict(o, t0=A(o["at"])) for o in D.get("options", [])][:3]
    CHT = [dict(c, t0=A(c["at"])) for c in D.get("chat", [])][:4]
    SV = D.get("save") or {}; t_sv = A(SV.get("at"), T.S(len(T.L) - 1)); sv_txt = SV.get("text", "Save this for your next meeting")
    t_op = OP[0]["t0"] - 0.2 if OP else T.END; t_ch = CHT[0]["t0"] - 0.3 if CHT else t_sv
    CH = sub_chunks(T, S.get("subs_replace")); hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", 330)
    he_img = hebrew_img(D["he"], 60, (201, 214, 234)) if D.get("he") else None

    def frame(t):
        cv = Canvas(bars_moving_bg(t, 8))
        # 1. scene
        sa = 1 - seg(t, t_op - 0.2, t_op + 0.2)
        if sa > 0.01:
            props(cv, D.get("setting", "office"), sa)
            walk = eout(t / 0.9)
            xs = 300 - 160 * (1 - walk); xl = 780 + 160 * (1 - walk)
            talk = t_sp <= t < t_op
            figure(cv, xs, 1250, 2.2, BLUE, badge=False, arms="point" if talk else "down", a=sa, walk=t * 7 if walk < 1 else 0)
            figure(cv, xl, 1250, 2.2, ORANGE, badge=False, arms="down", a=sa, flip=-1, walk=t * 7 if walk < 1 else 0)
            cv.text(SP.get("name", "Boss"), xs, 1305, 46, BLUE, sa, maxw=300)
            ba = sa * WG.pop_s(t, t_sp) if t >= t_sp else 0
            if ba > 0.01:
                cv.rrect(90, 590, 900, 230, 46, BLUE, min(1, ba) * 0.95)
                cv.c.move_to(260, 815); cv.c.line_to(330, 815); cv.c.line_to(280, 890); cv.c.close_path(); cv.col(BLUE, min(1, ba) * 0.95); cv.c.fill()
                rich_wrap(cv, SP["says"], 540, 705, 70, min(1, ba), 820)
        # 2. options
        oa = win(t, t_op, t_ch, 0.3, 0.3)
        if oa > 0.01:
            for k, o in enumerate(OP):
                a = oa * seg(t, o["t0"] - 0.1, o["t0"] + 0.25)
                if a <= 0.01: continue
                y = 640 + k * 260; sc = 0.92 + 0.08 * eout((t - o["t0"]) / 0.3)
                cv.rrect(70, y - 105 * sc, 940, 210 * sc, 34, "#0B1A3C", 0.9 * a, stroke=BLUE, lw=3, sa=a)
                cv.text(o.get("tag", ""), 120, y - 58, 40, "#C9D6EA", a, align="l", maxw=400, bold=False)
                rich_wrap(cv, o["text"], 540, y + 25, 74, a, 860, base=BLUE)
        # 3. chat
        ca = win(t, t_ch, t_sv, 0.3, 0.3)
        if ca > 0.01:
            cv.rrect(110, 520, 860, 940, 50, "#0B1A3C", 0.92 * ca, stroke="#2B4A80", lw=3, sa=ca)
            for k, c in enumerate(CHT):
                a = ca * seg(t, c["t0"] - 0.1, c["t0"] + 0.2)
                if a <= 0.01: continue
                me = c["who"] == "me"; y = 640 + k * 210; w = min(700, cv.tw(re.sub(r"[*_]", "", c["text"]), 58) + 80)
                x = 930 - w if me else 150
                cv.rrect(x, y - 70, w, 140, 36, "#123C73" if me else "#2B4A80", a, stroke=BLUE if me else None, lw=3, sa=a)
                rich_wrap(cv, c["text"], x + w / 2, y, 58, a, w - 60, base=WHITE if me else "#E8EEF8")
        # 4. save
        va = seg(t, t_sv - 0.1, t_sv + 0.3) * (1 - seg(t, T.END - 0.6, T.END - 0.5))
        if va > 0.01:
            WG.panel(cv, 90, 760, 900, 380, va, stroke=YEL)
            WG.emo(cv, "\U0001F516", 540, 860, 110, va)
            WG.title(cv, sv_txt, 1040, va, 70)
        if he_img is not None:
            cv.image(he_img, 540, 1490, 1.0, 0.9)
        out = paste_hook(cv.out().astype(np.float32), hook_img, t, T.E(0), HY, 0.85)
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(t_sp, "pop", -22)] + [(o["t0"], "pop", -22) for o in OP] + [(c["t0"], "tick", -20) for c in CHT] + [(t_sv, "ding", -24)]
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud)
