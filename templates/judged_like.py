"""templates/judged_like.py: "<Country>, judged like Israel." deadpan satire report (the France reel format, 9 Nov, as a template).

Look: navy gradient with a drifting sky grid, "Israel Facts" mark top left, "OFFICIAL REPORT: <COUNTRY>" pill (fictional, no logos,
no faces), frame 0 = hook over the country map card with a yellow scanner sweep and an UNDER REVIEW stamp. Then scenes: a small
FINDING n line, a title (*word* gold), one or more visuals, rubber stamps that thud in, and a note box labelled TRUE PART,
SATIRE, REALITY CHECK or THE TEST (source + year inside it). Burned-in subtitles, music bed, last 0.5 s dissolves into frame 0.

spec "data":
  country   "Britain"            name in the report pill
  hook_map  {cam: [lon, lat, width_deg], hi: ["GBR"], color: "sky", stamp: "UNDER REVIEW"}   frame 0 map
  hook_y    310                  hook text centre
  duck_db   music duck (default 14)
  scenes    [{at, kind?: "hook", header?: "FINDING 1", title?: "...", vis: [..], stamps: [..], note: {..}}]
            a scene shows from its "at" until the next scene's "at" (or "until"); the first scene starts at 0
    vis item (each may carry at / until inside the scene; default = the scene's span):
      {"type": "map", cam, hi: [codes], color, israel: true, pins: [{lon, lat, label, at, dx, dy}],
                       arc: {from: [lon, lat], to: [lon, lat], at, dur, bend, counter: {to, suffix, label}}}
      {"type": "photo", file: "photos/x.jpg", credit: "...", kb: [cx0, cy0, z0, cx1, cy1, z1]}
      {"type": "emoji", ch: "...", size: 300, y: 820}
      {"type": "big", value, from, prefix, suffix, dec, sub, y}
      {"type": "chain", rows: [{l, r, at}], y0}
      {"type": "sign", text, sub}
      {"type": "flag", code: "GB", size: 360, y}
      {"type": "israel", stats: [{big, sub, at}]}                 Israel map card left, stat cards right
      {"type": "versus", items: [{flag, name, sub}], y}
      {"type": "quote", text, by, y}                              definition card (e.g. Rome Statute wording)
      {"type": "checks", rows: [{label, mark: "N" | "Y" | "P", at}], y0, title}
      {"type": "scales", y}
      {"type": "text", text, y, size}
    stamps: [{text, at, x, y, rot, col: "red" | "yellow" | "green", size}]
    note:   {kind: "TRUE PART" | "SATIRE" | "REALITY CHECK" | "THE TEST", text, at}
Local look without a voice: TPL_FAKE=1 TPL_SNAPS=0,5,20 python3 templates/render.py specs/<id>.json
"""
import os, sys, re, math, json, subprocess
import numpy as np, cv2
from common import *
from base import *
import widgets as WG
import map_tear as mt

COL = dict(red=RED, yellow=YEL, green=GREEN, sky=SKY, white=WHITE, grey=GREY)
KIND_COL = {"TRUE PART": GREEN, "SATIRE": "#F2A93B", "REALITY CHECK": SKY, "THE TEST": SKY, "THE DEFINITION": SKY}
CARD = (60, 470, 960, 760)          # main visual card x, y, w, h


def _fake_timing(S):
    """local look without a voice (TPL_FAKE=1): 2.7 words per second, spec gaps"""
    t, out = 0.3, []
    for i, L in enumerate([L for L in S["lines"] if L.get("say", "").strip()]):
        ws = L["say"].split(); d = len(ws) / 2.7
        out.append(dict(i=i, text=L["say"], t0=t, t1=t + d, words=[dict(w=w, t0=t + d * k / len(ws), t1=t + d * (k + 1) / len(ws)) for k, w in enumerate(ws)]))
        t += d + float(L.get("gap", 0.15))
    return dict(dur=t, voice="fake", lines=out)


def _mask(w, h, r):
    m = np.ones((h, w), np.float32); yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        q = ((xx < r) | (xx >= w - r)) & ((yy < r) | (yy >= h - r)) & (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        m[q] = np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0, 1)[q]
    return m[..., None]


def _wrap(cv, txt, sz, maxw, bold=False):
    out, cur = [], ""
    for w in txt.split():
        tst = (cur + " " + w).strip()
        if cur and cv.tw(tst, sz, bold) > maxw: out.append(cur); cur = w
        else: cur = tst
    if cur: out.append(cur)
    return out


def _cover(im, w, h, cx, cy, z):
    ih, iw = im.shape[:2]; s = max(w / iw, h / ih) * max(1.0, z); cw, ch = w / s, h / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def run(S):
    D = S["data"]; work = workdir(S)
    J = _fake_timing(S) if os.environ.get("TPL_FAKE") else build_voice(S, work)
    T = Timing(J, D.get("tail", 0.7)); A = T.at; END = T.END
    MR = MapR(); MCACHE = {}
    CX, CY, CW, CH_ = CARD; CMASK = _mask(CW, CH_, 34); IMASK = _mask(360, CH_, 30)

    def mapcard(cam, hi=(), color="sky", israel=True):
        key = (tuple(cam), tuple(hi), color, israel)
        if key not in MCACHE:
            lon, lat, wd = cam; cosf = max(0.6, math.cos(math.radians(min(62, abs(lat)))))
            v = MR.view(lon, lat, CW / (wd * cosf), cosf, CH_ / 2.0)
            img, M = MR.render(v, CH_, hi=list(hi), hi_a={c: 0.95 for c in hi}, israel=israel, hi_c={c: COL.get(color, color) for c in hi})
            x0 = (W - CW) // 2
            MCACHE[key] = (img[:, x0:x0 + CW].astype(np.float32), M, x0)
        return MCACHE[key]

    # ---------------- scenes
    SC = []
    for k, s in enumerate(D["scenes"]):
        s = dict(s); s["t0"] = A(s["at"]) if k else 0.0; SC.append(s)
    for k, s in enumerate(SC):
        s["t1"] = A(s["until"]) if s.get("until") else (SC[k + 1]["t0"] if k + 1 < len(SC) else END + 1)
        s["vis"] = [dict(v, t0=A(v["at"]) if v.get("at") is not None else s["t0"], t1=A(v["until"]) if v.get("until") else s["t1"]) for v in s.get("vis", [])]
        s["stamps"] = [dict(st, t0=A(st["at"])) for st in s.get("stamps", [])]
        if s.get("note"): s["note"] = dict(s["note"], t0=A(s["note"].get("at")) if s["note"].get("at") is not None else s["t0"] + 0.3)
    HM = D["hook_map"]
    hook_img = HK.make(S["_hook"], size=int(D.get("hook_size", 118)), maxw=980)
    HY = D.get("hook_y", 300)
    PHOTOS = {}
    for s in SC:
        for v in s["vis"]:
            if v["type"] == "photo" and v["file"] not in PHOTOS:
                p = os.path.join(REPO, v["file"])
                if not os.path.exists(p):
                    subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + v["file"]], check=False, capture_output=True)
                im = cv2.imread(p)
                if im is None and os.environ.get("TPL_FAKE"): im = np.full((1080, 1440, 3), 90, np.uint8)
                if im is None: raise SystemExit(f"photo missing: {v['file']}")
                PHOTOS[v["file"]] = im
            if v["type"] == "photo" and str(v.get("credit", "")).startswith("CREDIT:"):
                js = os.path.join(REPO, v["file"][:-4] + ".json")
                if not os.path.exists(js):
                    subprocess.run(["git", "-C", REPO, "sparse-checkout", "add", "/" + v["file"][:-4] + ".json"], check=False, capture_output=True)
                try:
                    pj = json.load(open(js)); au = re.split(r",| from | \(", pj.get("author") or "")[0].strip()[:28]
                    v["credit"] = "Photo: " + ", ".join(x for x in (au, pj.get("license"), "Wikimedia Commons") if x)
                except Exception: v["credit"] = "Photo: Wikimedia Commons"
                v["credit"] = DASH_RE.sub(",", v["credit"])[:90]
    CHS = sub_chunks(T, S.get("subs_replace"))
    SAYL = [L for L in S["lines"] if L.get("say", "").strip()]
    fl_end = max([T.E(i) for i, L in enumerate(SAYL) if "noamroz" in (L.get("screen") or "")] or [END])
    follow_cards = [(c[0], c[1], min(c[2], fl_end + 0.3), c[3], c[4]) for c in screen_cards(S, T) if "noamroz" in c[0]]
    events = [(0.05, "thud", -20)]
    for s in SC:
        if s["t0"] > 0.2: events.append((s["t0"], "whoosh", -30))
        for st in s["stamps"]: events.append((st["t0"], "thud", -18))
        for v in s["vis"]:
            if v["type"] == "big": events.append((v["t0"] + 0.05, "roll", v["t0"] + 1.0))
            if v["type"] == "map" and v.get("arc"): events.append((A(v["arc"]["at"]), "roll", A(v["arc"]["at"]) + float(v["arc"].get("dur", 1.4))))
            if v["type"] == "checks":
                for r in v["rows"]: events.append((A(r["at"]), "buzz" if r["mark"] == "N" else "check", -26))
            if v["type"] == "chain":
                for r in v["rows"]: events.append((A(r["at"]), "pop", -28))
            if v["type"] == "israel":
                for st in v.get("stats", []): events.append((A(st["at"]), "pop", -26))

    def s_alpha(s, t):
        return win(t, s["t0"], s["t1"], 0.22, 0.18) if s["t0"] > 0 else (1 - seg(t, s["t1"] - 0.18, s["t1"]))

    def paste_card(out, img, a, x=CX, w=CW, m=None):
        if a <= 0.01: return out
        m = (CMASK if m is None else m) * a
        out[CY:CY + CH_, x:x + w] = out[CY:CY + CH_, x:x + w] * (1 - m) + img * m
        return out

    def zoomed(img, z):
        if abs(z - 1) < 1e-3: return img
        h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, z)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def stamp(cv, st, t, a):
        u = seg(t, st["t0"], st["t0"] + 0.22, lambda x: x)
        if u <= 0 or a <= 0.01: return
        sc = 1.9 - 0.9 * eout(u); al = a * min(1, u * 2.2)
        col = COL.get(st.get("col", "red"), st.get("col", "red")); sz = st.get("size", 70)
        x, y = st.get("x", 540), st.get("y", 900); rot = math.radians(st.get("rot", -8))
        tw = cv.tw(st["text"], sz) + 60
        if tw > 960: sz = int(sz * 900 / (tw - 60)); tw = cv.tw(st["text"], sz) + 60
        th = sz * 1.45
        c = cv.c; c.save(); c.translate(x, y); c.rotate(rot); c.scale(sc, sc)
        cv.rrect(-tw / 2, -th / 2, tw, th, 14, "#0B1730", 0.55 * al, stroke=col, lw=7, sa=al)
        cv.rrect(-tw / 2 + 10, -th / 2 + 10, tw - 20, th - 20, 8, None, al, stroke=col, lw=2.5, sa=0.8 * al)
        cv.text(st["text"], 0, 0, sz, col, al, maxw=980, shadow=False)
        c.restore()

    def note(cv, n, t, a):
        a = a * seg(t, n["t0"], n["t0"] + 0.3)
        if a <= 0.01: return
        y = D.get("note_y", 1340); col = KIND_COL.get(n["kind"], SKY)
        lines = _wrap(cv, n["text"], 38, 880)[:3]; h = 30 + 48 * len(lines)
        cv.rrect(70, y - 16, 940, h + 16, 22, "#0B1A3C", 0.92 * a, stroke=col, lw=2.5, sa=0.8 * a)
        kw = cv.tw(n["kind"], 36) + 40
        cv.rrect(540 - kw / 2, y - 42, kw, 50, 25, col, a)
        cv.text(n["kind"], 540, y - 17, 36, "#0B1730", a, shadow=False, maxw=900)
        for i, ln in enumerate(lines): cv.text(ln, 540, y + 36 + 48 * i, 38, WHITE, a, bold=False, maxw=900, shadow=False)

    def frame(t):
        out = bars_moving_bg(t)
        cur = [s for s in SC if s["t0"] <= t < s["t1"]]
        for s in cur:                                     # raster layers (maps, photos)
            sa = s_alpha(s, t)
            if sa <= 0.01: continue
            if s.get("kind") == "hook":
                img, M, x0 = mapcard(HM["cam"], HM.get("hi", ()), HM.get("color", "sky"), HM.get("israel", True))
                out = paste_card(out, zoomed(img, 1.0 + 0.04 * max(0, t - s["t0"]) / 3.0), sa)
                ph = ((t - s["t0"]) / 1.5) % 1.0; yy = CH_ * ph; r = np.arange(CH_)
                prof = np.exp(-((r - yy) / 22.0) ** 2) + np.where(r < yy, np.exp(-(yy - r) / 140.0), 0) * 0.22
                out[CY:CY + CH_, CX:CX + CW] += prof[:, None, None] * CMASK * sa * np.array([63, 210, 255], np.float32) * 0.5
                continue
            for v in s["vis"]:
                va = sa * win(t, v["t0"] - 0.15, v["t1"] + 0.15, 0.3, 0.3)
                if va <= 0.01: continue
                if v["type"] == "map":
                    img, M, x0 = mapcard(v["cam"], tuple(v.get("hi", ())), v.get("color", "sky"), v.get("israel", True))
                    out = paste_card(out, zoomed(img, 1.0 + 0.03 * max(0, t - v["t0"]) / 3.0), va)
                elif v["type"] == "photo":
                    im = PHOTOS[v["file"]]; kb = v.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
                    u = ease((t - v["t0"]) / max(0.5, v["t1"] - v["t0"]))
                    out = paste_card(out, _cover(im, CW, CH_, kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u) * 0.92, va)
                elif v["type"] == "israel":
                    img, M, x0 = mapcard((35.0, 31.4, 4.6), (), "sky", True)
                    out = paste_card(out, img[:, 300:660], va, CX, 360, IMASK)
        out = np.clip(out, 0, 255)
        cv = Canvas(out)
        cv.circ(62, 92, 8, YEL); cv.text("Israel Facts", 80, 92, 38, "#C9D6EA", 1.0, align="l", shadow=False)
        pill = f"OFFICIAL REPORT: {D['country'].upper()}"
        pw = cv.tw(pill, 40) + 60; cv.rrect(540 - pw / 2, 136, pw, 62, 31, SKY, 0.95)
        cv.text(pill, 540, 167, 40, "#0B1730", 1.0, shadow=False)
        for s in cur:                                     # vector layers
            sa = s_alpha(s, t)
            if sa <= 0.01: continue
            if s.get("kind") == "hook":
                cv.rrect(CX - 4, CY - 4, CW + 8, CH_ + 8, 38, None, sa, stroke=SKY, lw=3, sa=0.65 * sa)
                stamp(cv, dict(text=HM.get("stamp", "UNDER REVIEW"), t0=s["t0"] - 0.3, x=540, y=CY + CH_ - 90, rot=-6, col="red", size=64), t, sa)
                o2 = HK.paste(cv.out().astype(np.float32), hook_img, 540, HY, 0.94 + 0.06 * eout(max(0, t - s["t0"]) / 0.2), sa)
                cv = Canvas(o2); continue
            if s.get("header"): cv.text(s["header"], 540, 252, 50, YEL, sa, shadow=False)
            if s.get("title"): WG.title(cv, s["title"], 352, sa, s.get("title_size", 72))
            for v in s["vis"]:
                va = sa * win(t, v["t0"] - 0.15, v["t1"] + 0.15, 0.3, 0.3)
                if va > 0.01: _vector(cv, v, t, va, mapcard, A)
            for st in s["stamps"]: stamp(cv, st, t, sa)
            if s.get("note"): note(cv, s["note"], t, sa)
        for txt, t0, t1, yy, ss in follow_cards:
            a = win(t, t0 - 0.1, t1, 0.25, 0.25)
            if a > 0.01:
                cv.rrect(60, 1150, 960, 170, 40, "#0B1A3C", 0.92 * a, stroke=SKY, lw=3, sa=0.8 * a)
                WG.title(cv, txt, 1235, a, 64)
        draw_subs(cv, CHS, t)
        return cv.out()

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    M_ = dict(S.get("music") or {}); M_.setdefault("duck_db", D.get("duck_db", 14))
    mix = build_mix(dict(S, music=M_), T, work, events)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, END, f"{work}/mix.m4a", final, loop=True)
    return dict(mix=mix, scenes=len(SC), min_text_px=MINSZ[0])


def _vector(cv, v, t, a, mapcard, A):
    CX, CY, CW, CH_ = CARD
    ty = v["type"]
    def pop(): return max(0.01, WG.pop_s(t, v["t0"])) if t < v["t0"] + 0.4 else 1.0
    if ty in ("map", "photo"):
        cv.rrect(CX - 4, CY - 4, CW + 8, CH_ + 8, 38, None, a, stroke=SKY, lw=3, sa=0.65 * a)
    if ty == "map":
        img, M, x0 = mapcard(v["cam"], tuple(v.get("hi", ())), v.get("color", "sky"), v.get("israel", True))
        def px(lon, lat):
            p = mt.apply(M, np.array([[lon, lat]], float))[0]; return CX + p[0] - x0, CY + p[1]
        for p in v.get("pins", []):
            pt = A(p["at"]) if p.get("at") is not None else v["t0"]; pa = a * seg(t, pt, pt + 0.3)
            if pa <= 0.01: continue
            x, y = px(p["lon"], p["lat"]); WG.pin(cv, x, y, YEL, pa, t)
            if p.get("label"): WG.tag(cv, p["label"], x + p.get("dx", 0), y + p.get("dy", -64), 46, WHITE, pa)
        ar = v.get("arc")
        if ar:
            t0 = A(ar["at"]); d = float(ar.get("dur", 1.4)); u = eout((t - t0) / d)
            if t >= t0:
                p0, p1 = px(*ar["from"]), px(*ar["to"])
                pulse_beam(cv, p0, p1, t, bend=ar.get("bend", 120), col=YEL, prog=max(0.03, u), a=a, waves=False)
                cv.circ(*p0, 10, YEL, a)
                if u > 0.98: cv.circ(*p1, 12, YEL, a)
                c = ar.get("counter")
                if c:
                    val = WG.roll(t, t0, 0, float(c["to"]), d)
                    bx, bw = (CX + CW - 520, 490) if c.get("box", "right") == "right" else (CX + 40, CW - 80); mx = bx + bw / 2
                    cv.rrect(bx, CY + CH_ - 200, bw, 170, 30, "#0B1730", 0.88 * a)
                    cv.text(WG.fmtn(val, 0, c.get("prefix", ""), c.get("suffix", "")), mx, CY + CH_ - 140, 86, YEL, a, maxw=bw - 30)
                    if c.get("label"): cv.text(c["label"], mx, CY + CH_ - 72, 40, WHITE, a, maxw=bw - 30)
    elif ty == "photo" and v.get("credit"):
        if cv.tw(v["credit"], 36, False) > 900: v["credit"] = v["credit"].replace(", Wikimedia Commons", ", Commons")
        if cv.tw(v["credit"], 36, False) > 900: v["credit"] = v["credit"].replace("Photo: ", "")
        while cv.tw(v["credit"], 36, False) > 900 and "," in v["credit"]:
            head, rest = v["credit"].split(",", 1); v["credit"] = head[:max(8, len(head) - 4)].rstrip() + "," + rest
        cw = min(940, cv.tw(v["credit"], 36, False) + 30)
        cv.rrect(540 - cw / 2, CY + CH_ - 60, cw, 48, 14, "#0B1730", 0.75 * a)
        cv.text(v["credit"], 540, CY + CH_ - 36, 36, "#C9D6EA", a, bold=False, maxw=920, shadow=False)
    elif ty == "emoji":
        WG.emo(cv, v["ch"], 540, v.get("y", 820), v.get("size", 300) * pop(), a)
    elif ty == "flag":
        WG.flag(cv, v["code"], 540, v.get("y", 800), v.get("size", 360) * pop(), a)
    elif ty == "big":
        val = WG.roll(t, v["t0"], float(v.get("from", 0)), float(v["value"]), 1.0); y = v.get("y", 760)
        WG.counter(cv, val, 540, y, v.get("sz", 230), a, v.get("dec", 0), v.get("prefix", ""), v.get("suffix", ""))
        if v.get("sub"): WG.title(cv, v["sub"], y + 170, a, 62)
    elif ty == "chain":
        y0 = v.get("y0", 560)
        for k, r in enumerate(v["rows"]):
            ra = a * seg(t, A(r["at"]), A(r["at"]) + 0.25)
            if ra <= 0.01: continue
            y = y0 + k * 132
            cv.rrect(110, y - 52, 860, 104, 26, "#0B1A3C", 0.92 * ra, stroke=SKY, lw=2.5, sa=0.8 * ra)
            if r.get("r"):
                cv.text(r["l"], 150, y, 62, WHITE, ra, align="l", maxw=470, shadow=False)
                cv.text(r["r"], 930, y, 48, YEL, ra, align="r", maxw=380, shadow=False)
            else:
                cv.text(r["l"], 540, y, 66, YEL, ra, shadow=False)
    elif ty == "sign":
        p = pop(); c = cv.c
        c.save(); c.translate(540, 760); c.rotate(math.radians(-4 + 2 * math.sin(t * 2.2))); c.scale(p, p)
        w_ = min(940, max(560, cv.tw(v["text"], 96) + 100))
        cv.rrect(-14, 70, 28, 360, 6, "#8B6A43", a)
        cv.rrect(-w_ / 2, -150, w_, 240, 20, "#F5F5F7", a, stroke="#C9D6EA", lw=4, sa=a)
        cv.text(v["text"], 0, -62, 96, "#0B1730", a, shadow=False, maxw=w_ - 40)
        if v.get("sub"): cv.text(v["sub"], 0, 34, 46, RED, a, shadow=False, maxw=w_ - 40)
        c.restore()
    elif ty == "israel":
        cv.rrect(CX - 4, CY - 4, 368, CH_ + 8, 34, None, a, stroke=SKY, lw=3, sa=0.65 * a)
        cv.text("Israel", CX + 180, CY + 52, 50, WHITE, a)
        for k, st in enumerate(v.get("stats", [])):
            sa = a * seg(t, A(st["at"]), A(st["at"]) + 0.3)
            if sa <= 0.01: continue
            y = CY + k * 256
            cv.rrect(450, y, 570, 240, 28, "#0B1A3C", 0.92 * sa, stroke=SKY, lw=2.5, sa=0.8 * sa)
            cv.text(st["big"], 735, y + 78, 92, YEL, sa, maxw=530)
            for i, ln in enumerate(_wrap(cv, st["sub"], 44, 520, True)[:2]): cv.text(ln, 735, y + 150 + 48 * i, 44, WHITE, sa, maxw=540)
    elif ty == "versus":
        its = v["items"]; n = len(its); y = v.get("y", 640); cw = 420
        for k, it in enumerate(its):
            x = 540 + (k - (n - 1) / 2) * 470
            cv.rrect(x - cw / 2, y - 150, cw, 300, 30, "#0B1A3C", 0.92 * a, stroke=SKY, lw=2.5, sa=0.8 * a)
            WG.flag(cv, it["flag"], x, y - 52, 130, a)
            cv.text(it["name"], x, y + 46, 58, WHITE, a, maxw=cw - 30)
            if it.get("sub"): cv.text(it["sub"], x, y + 108, 42, COL.get(it.get("sub_col", "green"), GREEN), a, maxw=cw - 30)
    elif ty == "quote":
        y = v.get("y", 820)
        cv.rrect(80, y - 270, 920, 540, 34, "#0B1A3C", 0.92 * a, stroke=SKY, lw=3, sa=0.8 * a)
        ls = _wrap(cv, v["text"], 66, 820, True)
        for i, ln in enumerate(ls): cv.text(ln, 540, y - 40 - 40 * (len(ls) - 1) + 80 * i, 66, WHITE, a, maxw=860)
        if v.get("by"): cv.text(v["by"], 540, y + 205, 44, YEL, a, maxw=860)
    elif ty == "checks":
        y0 = v.get("y0", 600)
        if v.get("title"): cv.text(v["title"], 540, y0 - 92, 54, "#C9D6EA", a, maxw=900)
        for k, r in enumerate(v["rows"]):
            at = A(r["at"]); ra = a * seg(t, at - 0.1, at + 0.2)
            if ra <= 0.01: continue
            y = y0 + k * 128
            cv.rrect(80, y - 56, 920, 112, 24, "#0B1A3C", 0.92 * ra, stroke=SKY, lw=2, sa=0.6 * ra)
            cv.text(r["label"], 120, y, 58, WHITE, ra, align="l", maxw=740, shadow=False)
            WG.mark(cv, r["mark"], 930, y, 38, ra, WG.pop_s(t, at) if t < at + 0.4 else 1.0)
    elif ty == "scales":
        y = v.get("y", 1060); sw = 6 * math.sin(t * 1.6)
        cv.line([(540, y - 150), (540, y + 110)], YEL, 12, a); cv.line([(430, y + 110), (650, y + 110)], YEL, 14, a)
        lx, ly = 320, y - 150 + sw; rx, ry = 760, y - 150 - sw
        cv.line([(lx, ly), (rx, ry)], YEL, 10, a)
        for x_, y_ in ((lx, ly), (rx, ry)):
            cv.line([(x_, y_), (x_ - 70, y_ + 110)], YEL, 4, a); cv.line([(x_, y_), (x_ + 70, y_ + 110)], YEL, 4, a)
            cv.c.new_path(); cv.c.arc(x_, y_ + 110, 80, 0, math.pi); cv.col(YEL, a); cv.c.fill()
    elif ty == "text":
        WG.title(cv, v["text"], v.get("y", 900), a, v.get("size", 80))
