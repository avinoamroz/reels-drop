"""bracket: a sports-broadcast playoff bracket of history. Two rivals per match, the winner moves UP the bracket (the
camera climbs match by match), the winner's connector lights up and its pill pops, the winner's line carries into the
next match; a gold "lifeline" runs up the left edge through every pill of one team (the survivor) into the FINAL box.
Top: scorebug (LIVE chip, tournament title, round, the year as the game clock with a flip). Band (top third): a regional
map with the rivals' pins and an attack arc, real photos (Ken Burns, credit line), quote / text / scoreboard cards, so
the bracket never leaves the screen. Frame 0: bright map, hook over the band, bracket already moving (light chase);
the map dims outside the band by about 1.4 s. Ending: camera rewinds to match 1, map brightens, hook returns, and the
last 0.5 s dissolves into frame 0 (circular).

data keys (times are anchors: 12.5, "L3", "L3e", "L3:word", "L3:word+0.2", "END-2"):
  map      {view: [lon, lat, k]}   map centred on the band centre (default [32.5, 33.5, 28.8])
  places   {CODE: [lon, lat, "Label"]}            pins on the map
  teams    {KEY: {name, abbr, col, place, hero}}  hero = the lifeline team (gold)
  matches  [{a, b, at, win: "a"|"b", win_at, note, chip_lose, arrow: true}]  a = top pill, b = bottom pill (the team
           carried from the previous match should be b, so the carry line enters from below)
  final    {at, title, name, sub}                  the FINAL box above the last match (lifeline ends in it)
  cams     [{at, k, dur}]   camera on match k (index; len(matches) = the final box); default follows match.at
  clock    [{at, text, round}]   game clock text (yellow) and round label
  stamps   [{at, k, text}]   red rubber stamp on match k's winner pill ("FELL 609 BCE")
  life     {at, dur}         the lifeline lights up bottom to top
  banners  [{at, until, text, col: yellow|sky|gold|red}]   big sports banner across the bracket
  photos   [{file, at, until, credit, kb: [cx0, cy0, z0, cx1, cy1, z1], dim}]    shown in the band
  cards    [{kind: quote|text|table, at, until, text, who, y, title, rows: [{name, val, at, ok}]}]   in the band
  tags     [{at, until, text, lon, lat, dx, dy, col}]   map labels in the band
  map_until  anchor after which match pins and arcs leave the map
  hook_back  anchor where the hook returns at the end (default the last line);  rewind anchor (default last line)
  cta      {at, text}   CTA pill at the bottom of the band
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core")); sys.path.insert(0, HERE)
import numpy as np
import cairo
from common import *

GOLD = "#F2C14E"
PANEL = "#0B1730"
BAND = (24, 1056, 296, 796)          # x0, x1, y0, y1
VP = (24, 1056, 822, 1446)           # bracket viewport
VC = 1186                            # viewport centre y (current match centre)
PITCH = 380
PX0, PX1, PH_ = 92, 524, 118         # team pills
JX = 574                             # connector join x
WX0, WX1, WH = 612, 1046, 140        # winner pill
DY = 84                              # pill centre offset from the match mid
LX = 50                              # lifeline x
COLS = dict(yellow=YEL, sky=SKY, gold=GOLD, red=RED, green=GREEN)


def run(S):
    D = S["data"]; work = workdir(S)
    J = build_voice(S, work); T = Timing(J, D.get("tail", 1.0)); A = T.at
    TEAMS = D["teams"]; PL = D.get("places", {})
    MT = []
    for k, m in enumerate(D["matches"]):
        MT.append(dict(m, k=k, t0=A(m["at"]), tw=A(m.get("win_at"), A(m["at"]) + 2.0), mid=-k * PITCH))
    NM = len(MT)
    FIN = dict(D.get("final", {})); FIN["t0"] = A(FIN.get("at"), T.END + 9); FIN["mid"] = -NM * PITCH - 20
    # carry: the winner of match k is the b team of match k+1 (same key or same "line")
    for k, m in enumerate(MT):
        m["wkey"] = m["a"] if m.get("win", "a") == "a" else m["b"]
        m["lkey"] = m["b"] if m.get("win", "a") == "a" else m["a"]
        m["carry"] = k + 1 < NM and (MT[k + 1]["b"] == m["wkey"] or MT[k + 1]["b"] in m.get("carry_to", []))

    CAMS = [dict(c, t0=A(c["at"])) for c in D.get("cams", [])] or [dict(k=m["k"], t0=m["t0"] - 0.6) for m in MT]
    # make cam continuous: integrate sequentially
    def cam_y(t):
        y = MT[0]["mid"]
        for c in CAMS:
            if t < c["t0"]: break
            tgt = FIN["mid"] if c["k"] >= NM else MT[c["k"]]["mid"]
            u = ease((t - c["t0"]) / float(c.get("dur", 0.8))); y = y + (tgt - y) * u
        return y

    CLK = [dict(c, t0=A(c["at"])) for c in D.get("clock", [])]
    STP = [dict(s, t0=A(s["at"])) for s in D.get("stamps", [])]
    LIFE = D.get("life", {}); LIFE_T = A(LIFE.get("at"), T.END + 9); LIFE_D = float(LIFE.get("dur", 1.4))
    BAN = [dict(b, t0=A(b["at"]), t1=A(b.get("until"), A(b["at"]) + 1.6)) for b in D.get("banners", [])]
    CARDS = [dict(c, t0=A(c["at"]), t1=A(c.get("until"), T.END - 0.6)) for c in D.get("cards", [])]
    TAGS = [dict(g, t0=A(g["at"]), t1=A(g.get("until"), A(g["at"]) + 2.5)) for g in D.get("tags", [])]
    CTA = D.get("cta"); CTA_T = A(CTA["at"]) if CTA else None
    MAP_UNTIL = A(D.get("map_until"), T.END + 9)
    REW = A(D.get("rewind"), T.S(len(T.L) - 1) + 0.3); HOOKB = A(D.get("hook_back"), REW + 0.4)

    # ------------------------------------------------------------- map (rendered once, full screen, centred on the band)
    bx0, bx1, by0, by1 = BAND; bw, bh = bx1 - bx0, by1 - by0; bcy = (by0 + by1) / 2
    lon, lat, kk = D.get("map", {}).get("view", [32.5, 33.5, 28.8])
    MR = MapR(); cosf = math.cos(math.radians(lat))
    mimg, MM = MR.render(MR.view(lon, lat, kk, cosf, bcy), H); MIMG = mimg.astype(np.float32)
    PXY = {c: geo2px(MM, v[0], v[1]) for c, v in PL.items()}
    yy = np.arange(H, dtype=np.float32)[:, None]
    NAVYC = np.array([97, 50, 22], np.float32)
    def mz(t):
        if t < REW: return 1 + 0.05 * eout(t / 1.6)
        return 1.05 - 0.05 * ease((t - REW) / 1.2)
    def zp(x, y, z): return 540 + (x - 540) * z, bcy + (y - bcy) * z
    def map_img(z):
        Mz = cv2.getRotationMatrix2D((540, bcy), 0, z)
        return cv2.warpAffine(MIMG, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    def lay(t):         # 0 = frame-0 bright map, 1 = broadcast layout (map dimmed outside the band)
        return seg(t, 0.8, 1.4) * (1 - seg(t, REW + 0.2, REW + 1.2))
    bmask = np.zeros((H, W), np.float32); r = 30
    cv2.rectangle(bmask, (bx0 + r, by0), (bx1 - r, by1), 1, -1); cv2.rectangle(bmask, (bx0, by0 + r), (bx1, by1 - r), 1, -1)
    for cx_, cy_ in ((bx0 + r, by0 + r), (bx1 - r, by0 + r), (bx0 + r, by1 - r), (bx1 - r, by1 - r)): cv2.circle(bmask, (cx_, cy_), r, 1, -1)
    BMASK = bmask[..., None]
    DIMBG = (BG * 0.8 + NAVYC * 0.2)

    # ------------------------------------------------------------- photos (band)
    from photo_story import _load
    PH = [dict(p, t0=A(p["at"]), t1=A(p.get("until"), T.END), im=_load(p["file"], p.get("gray", False))) for p in D.get("photos", [])]
    def band_kb(im, cx, cy, z):
        h_, w_ = im.shape[:2]; s_ = max(bw / w_, bh / h_) * max(1.0, z)
        cw, ch = bw / s_, bh / s_
        x0_ = min(max(cx * w_ - cw / 2, 0), w_ - cw); y0_ = min(max(cy * h_ - ch / 2, 0), h_ - ch)
        M_ = np.array([[s_, 0, -x0_ * s_], [0, s_, -y0_ * s_]], np.float32)
        return cv2.warpAffine(im, M_, (bw, bh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    byy = np.arange(bh, dtype=np.float32)[:, None]
    BVIG = (1 - np.clip((byy - bh * 0.6) / (bh * 0.4), 0, 1) ** 1.3 * 0.5)[..., None]
    def ph_a(p, t): return win(t, p["t0"] - 0.15, p["t1"] + 0.15, 0.3, 0.3)

    cards = screen_cards(S, T); CH = sub_chunks(T, S.get("subs_replace"))
    hook_img = HK.make(S["_hook"]); HY = D.get("hook_y", bcy)

    def team(key): return TEAMS[key]
    def tcol(key): return GOLD if team(key).get("hero") else team(key).get("col", SKY)

    # ------------------------------------------------------------- drawing helpers (world coords under a cairo transform)
    def pill(cv, key, cy, a, lost=False, chip=None, glow=0.0):
        tm = team(key); col = tcol(key)
        if glow > 0.01: cv.rrect(PX0 - 8, cy - PH_ / 2 - 8, PX1 - PX0 + 16, PH_ + 16, 30, col, 0.25 * glow * a)
        cv.rrect(PX0, cy - PH_ / 2, PX1 - PX0, PH_, 24, "#13254A", 0.95 * a, stroke=col, lw=4, sa=(0.45 if lost else 1.0) * a)
        aa = a * (0.42 if lost else 1.0)
        cv.circ(PX0 + 60, cy, 42, col, aa)
        cv.text(tm.get("abbr", key)[:3], PX0 + 60, cy, 36, "#0B1730", aa, shadow=False, maxw=74, minsz=36)
        cv.text(tm["name"], PX0 + 118, cy, 64, WHITE, aa, align="l", maxw=PX1 - PX0 - 132, shadow=False)
        if chip:
            ca = a * chip[1]
            if ca > 0.01:
                ccol = GOLD if tm.get("hero") else RED; w = cv.tw(chip[0], 36) + 30
                cv.rrect(PX1 - w + 14, cy - PH_ / 2 - 26, w, 50, 14, ccol, ca)
                cv.text(chip[0], PX1 - w / 2 + 14, cy - PH_ / 2 - 1, 36, "#0B1730", ca, shadow=False, minsz=36)

    def path_pts(pts, u):
        """first u (0..1) of a polyline"""
        L = [math.dist(p, q) for p, q in zip(pts, pts[1:])]; tot = sum(L); want = tot * max(0, min(1, u)); out = [pts[0]]
        for (p, q), l in zip(zip(pts, pts[1:]), L):
            if want >= l: out.append(q); want -= l
            else:
                f = want / l if l else 0; out.append((p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)); break
        return out

    def chase(cv, pts, t, col, a, period=1.3):
        L = [math.dist(p, q) for p, q in zip(pts, pts[1:])]; tot = sum(L)
        for ph in (0.0, 0.5):
            u = ((t / period) + ph) % 1.0; d = u * tot
            for (p, q), l in zip(zip(pts, pts[1:]), L):
                if d <= l:
                    f = d / l if l else 0; x, y = p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f
                    cv.circ(x, y, 16, col, 0.25 * a); cv.circ(x, y, 7, WHITE, 0.95 * a); break
                d -= l

    def stamp(cv, x, y, txt, a, s=1.0):
        if a <= 0.01: return
        c = cv.c; c.save(); c.translate(x, y); c.rotate(-0.13); c.scale(s, s)
        w = cv.tw(txt, 50) + 44
        cv.rrect(-w / 2, -40, w, 80, 12, "#2A0A12", 0.55 * a, stroke=RED, lw=6, sa=a)
        cv.text(txt, 0, 0, 50, "#FF6B70", a, shadow=False, maxw=600)
        c.restore()

    def draw_match(cv, m, t, a_vp):
        mid = m["mid"]; ya, yb = mid - DY, mid + DY
        tw = m["tw"]; k = m["k"]
        on = seg(t, m["t0"] - 0.25, m["t0"] + 0.2)            # pills arrive
        skel = 0.55 * a_vp
        # skeleton connectors
        cv.line([(PX1, ya), (JX, ya), (JX, yb), (PX1, yb)], "#3A5A8C", 4, skel)
        cv.line([(JX, mid), (WX0, mid)], "#3A5A8C", 4, skel)
        if on < 0.999:
            for yy_ in (ya, yb): cv.rrect(PX0, yy_ - PH_ / 2, PX1 - PX0, PH_, 24, None, 1, stroke="#3A5A8C", lw=3, sa=skel * (1 - on))
            cv.rrect(WX0, mid - WH / 2, WX1 - WX0, WH, 26, None, 1, stroke="#3A5A8C", lw=3, sa=skel)
            cv.text("?", (WX0 + WX1) / 2, mid, 64, "#3A5A8C", skel * (1 - on), shadow=False)
        if on <= 0.01: return
        res = seg(t, tw + 0.3, tw + 0.55)                   # winner known
        wy = ya if m.get("win", "a") == "a" else yb
        lost_a = res > 0.5 and m.get("win", "a") != "a"; lost_b = res > 0.5 and m.get("win", "a") == "a"
        def chip_for(key, lost):
            if not lost: return None
            txt = m.get("chip_lose") or ("OUT" if not team(key).get("hero") else "STILL IN")
            return (txt, seg(t, tw + 0.5, tw + 0.8))
        pop = lambda t0_: back(seg(t, t0_ - 0.25, t0_ + 0.15, lambda u: u))
        for key, yy_, lost, t_in in ((m["a"], ya, lost_a, m["t0"] - 0.1), (m["b"], yb, lost_b, m["t0"] + 0.15)):
            sc = pop(t_in)
            if sc <= 0.01: continue
            c = cv.c; c.save(); c.translate((PX0 + PX1) / 2, yy_); c.scale(sc, sc); c.translate(-(PX0 + PX1) / 2, -yy_)
            glow = (1 - res) * (0.5 + 0.5 * math.sin(t * 6)) if t > m["t0"] and t < tw + 0.3 else 0.0
            pill(cv, key, yy_, a_vp, lost, chip_for(key, lost), glow)
            c.restore()
        # live chase on the connectors before the result
        if m["t0"] <= t < tw:
            chase(cv, [(PX1, ya), (JX, ya), (JX, mid), (WX0, mid)], t, SKY, a_vp * on)
            chase(cv, [(PX1, yb), (JX, yb), (JX, mid), (WX0, mid)], t + 0.4, SKY, a_vp * on)
        # winner path lights up
        wk = m["wkey"]; wc = tcol(wk)
        if t >= tw:
            pts = [(PX1, wy), (JX, wy), (JX, mid), (WX0, mid)]; u = (t - tw) / 0.4
            pp = path_pts(pts, u)
            if len(pp) > 1: cv.line(pp, wc, 18, 0.28 * a_vp); cv.line(pp, wc, 7, a_vp); cv.line(pp, WHITE, 2, 0.8 * a_vp)
        # winner pill
        sc = back(seg(t, tw + 0.3, tw + 0.65, lambda u: u))
        if sc > 0.01:
            c = cv.c; c.save(); c.translate((WX0 + WX1) / 2, mid); c.scale(sc, sc); c.translate(-(WX0 + WX1) / 2, -mid)
            fl = max(0.0, 1 - (t - tw - 0.3) / 0.5)
            cv.rrect(WX0 - 10, mid - WH / 2 - 10, WX1 - WX0 + 20, WH + 20, 34, wc, (0.22 + 0.5 * fl) * a_vp)
            cv.rrect(WX0, mid - WH / 2, WX1 - WX0, WH, 26, wc, 0.96 * a_vp, stroke=WHITE, lw=3, sa=0.7 * a_vp)
            nm = team(wk)["name"]; note = m.get("note")
            dark = "#0B1730"
            if note:
                cv.text(nm, (WX0 + WX1) / 2, mid - 26, 62, dark, a_vp, shadow=False, maxw=WX1 - WX0 - 40)
                cv.text(note, (WX0 + WX1) / 2, mid + 36, 42, dark, a_vp, bold=False, shadow=False, maxw=WX1 - WX0 - 30, minsz=36)
            else: cv.text(nm, (WX0 + WX1) / 2, mid, 60, dark, a_vp, shadow=False, maxw=WX1 - WX0 - 40)
            c.restore()
            tag_ = m.get("win_tag", "WIN")
            ta = a_vp * seg(t, tw + 0.55, tw + 0.8)
            if ta > 0.01:
                w = cv.tw(tag_, 36) + 28; cv.rrect(WX1 - w - 6, mid - WH / 2 - 30, w, 50, 14, YEL, ta)
                cv.text(tag_, WX1 - w / 2 - 6, mid - WH / 2 - 5, 36, "#0B1730", ta, shadow=False)
        # carry line up into the next match (enters the next match's bottom pill connector)
        if m["carry"] and t >= tw + 0.7:
            nm_ = MT[k + 1]; yb2 = nm_["mid"] + DY; xc = (WX0 + WX1) / 2
            pts = [(xc, mid - WH / 2), (xc, yb2 + 30), (xc - 30, yb2), (JX, yb2)]
            pp = path_pts(pts, (t - tw - 0.7) / 0.7)
            if len(pp) > 1: cv.line(pp, wc, 6, 0.75 * a_vp); cv.line(pp, wc, 16, 0.18 * a_vp)
        for s in STP:
            if s["k"] == k and t >= s["t0"]:
                u = min(1.0, max(0.0, (t - s["t0"]) / 0.22)); pos = s.get("pos", "w")
                sx, sy = ((WX0 + WX1) / 2 + 10, mid + 4) if pos == "w" else ((PX0 + PX1) / 2 + 30, ya if pos == "a" else yb)
                stamp(cv, sx, sy, s["text"], u, 1.6 - 0.6 * u)

    def hero_ys():
        ys = []
        for m in MT:
            for key, yy_ in ((m["a"], m["mid"] - DY), (m["b"], m["mid"] + DY)):
                if team(key).get("hero"): ys.append((yy_, m["t0"]))
        return ys
    HYS = hero_ys()
    Y_BOT = MT[0]["mid"] + DY + 160 if MT else 0

    def draw_life(cv, t, a_vp):
        top = FIN["mid"]
        cv.line([(LX, Y_BOT), (LX, top)], GOLD, 5, 0.35 * a_vp, dash=[14, 12])
        for yy_, t0 in HYS:
            sa = a_vp * seg(t, t0, t0 + 0.4)
            if sa > 0.01: cv.line([(LX, yy_), (PX0, yy_)], GOLD, 5, 0.8 * sa); cv.circ(LX, yy_, 9, GOLD, sa)
        u = ease((t - LIFE_T) / LIFE_D)
        if t >= LIFE_T:
            ytop = Y_BOT + (top - Y_BOT) * u
            cv.line([(LX, Y_BOT), (LX, ytop)], GOLD, 26, 0.25 * a_vp); cv.line([(LX, Y_BOT), (LX, ytop)], GOLD, 9, a_vp)
            cv.line([(LX, Y_BOT), (LX, ytop)], WHITE, 3, 0.8 * a_vp); cv.circ(LX, ytop, 18, WHITE, a_vp * (1 - 0.5 * u))

    def draw_final(cv, t, a_vp):
        mid = FIN["mid"]; x0, x1, h = PX0, WX1, 190
        on = seg(t, FIN["t0"], FIN["t0"] + 0.5)
        cv.rrect(x0, mid - h / 2, x1 - x0, h, 34, None, 1, stroke="#3A5A8C", lw=4, sa=0.55 * a_vp * (1 - on))
        cv.text(FIN.get("title", "FINAL"), (x0 + x1) / 2, mid, 64, "#3A5A8C", 0.55 * a_vp * (1 - on), shadow=False)
        cv.line([(LX, mid), (x0, mid)], GOLD, 5, 0.35 * a_vp, dash=[14, 12])
        if on <= 0.01: return
        sc = back(seg(t, FIN["t0"], FIN["t0"] + 0.45, lambda u: u))
        c = cv.c; c.save(); c.translate(540, mid); c.scale(sc, sc); c.translate(-540, -mid)
        gl = 0.5 + 0.5 * math.sin(t * 4)
        cv.rrect(x0 - 14, mid - h / 2 - 14, x1 - x0 + 28, h + 28, 44, GOLD, (0.18 + 0.15 * gl) * a_vp)
        cv.rrect(x0, mid - h / 2, x1 - x0, h, 34, "#13254A", 0.97 * a_vp, stroke=GOLD, lw=6, sa=a_vp)
        cv.text(FIN.get("name", ""), 540, mid - 32, 80, GOLD, a_vp, maxw=x1 - x0 - 60)
        cv.text(FIN.get("sub", ""), 540, mid + 50, 50, WHITE, a_vp * seg(t, FIN["t0"] + 0.4, FIN["t0"] + 0.7), maxw=x1 - x0 - 60)
        c.restore()
        w = cv.tw(FIN.get("title", "FINAL"), 44) + 50
        cv.rrect(540 - w / 2, mid - h / 2 - 34, w, 64, 18, GOLD, a_vp * on)
        cv.text(FIN.get("title", "FINAL"), 540, mid - h / 2 - 2, 44, "#0B1730", a_vp * on, shadow=False)
        cv.line([(LX, mid), (x0, mid)], GOLD, 9, a_vp * on)

    def draw_bracket(cv, t):
        c = cv.c; vx0, vx1, vy0, vy1 = VP
        cv.rrect(vx0, vy0, vx1 - vx0, vy1 - vy0, 30, PANEL, 0.86 * lay_b(t), stroke="#2A3B5C", lw=3, sa=lay_b(t))
        c.save(); c.rectangle(vx0, vy0, vx1 - vx0, vy1 - vy0); c.clip()
        c.push_group()
        cy = cam_y(t)
        c.translate(0, VC - cy)
        draw_life(cv, t, 1.0)
        for m in MT:
            if abs(m["mid"] - cy) < PITCH * 1.6: draw_match(cv, m, t, 1.0)
        if abs(FIN["mid"] - cy) < PITCH * 1.6: draw_final(cv, t, 1.0)
        pat = c.pop_group()
        c.identity_matrix(); c.set_source(pat)
        g = cairo.LinearGradient(0, vy0, 0, vy1)
        for off, al in ((0, 0), (0.1, 1), (0.9, 1), (1, 0)): g.add_color_stop_rgba(off, 0, 0, 0, al)
        c.mask(g); c.restore()

    def lay_b(t): return 0.75 + 0.25 * lay(t)

    def draw_scorebug(cv, t):
        a = 1.0
        cv.rrect(24, 118, 1032, 150, 30, PANEL, 0.9, stroke="#2A3B5C", lw=3)
        # LIVE chip
        pl = 0.6 + 0.4 * math.sin(t * 5)
        cv.rrect(48, 140, 134, 50, 14, RED, 0.95)
        cv.circ(74, 165, 9, WHITE, pl); cv.text("LIVE", 128, 165, 36, WHITE, 1, shadow=False)
        cv.text(D.get("title", "THE TOURNAMENT"), 198, 165, 38, SKY, 1, align="l", maxw=466, shadow=False, minsz=36)
        cur = None; prev = None
        for c_ in CLK:
            if t >= c_["t0"]: prev, cur = cur, c_
        rnd = (cur or {}).get("round", "")
        if rnd: cv.text(rnd, 60, 230, 56, WHITE, 1, align="l", maxw=560)
        # clock box
        cv.rrect(680, 132, 360, 122, 22, "#060E1F", 0.95, stroke=YEL, lw=3, sa=0.6)
        c = cv.c; c.save(); c.rectangle(682, 134, 356, 118); c.clip()
        if cur:
            u = ease((t - cur["t0"]) / 0.35)
            if not prev: u = 1.0
            if prev and u < 1:
                cv.text(prev["text"], 860, 193 - 80 * u, 84, YEL, 1 - u, maxw=330, shadow=False)
            cv.text(cur["text"], 860, 193 + 80 * (1 - u), 84, YEL, u, maxw=330, shadow=False)
        c.restore()

    def draw_band_map_overlays(cv, t, a):
        """pins + attack arc for the current match (map coords follow the zoom)"""
        if a <= 0.01: return
        z = mz(t)
        cur = None
        for m in MT:
            if t >= m["t0"] - 0.3: cur = m
        shown = set()
        if cur and t < min(REW, MAP_UNTIL):
            ka, kb = team(cur["a"]).get("place"), team(cur["b"]).get("place")
            for key, pc in ((cur["a"], ka), (cur["b"], kb)):
                if pc in PXY and pc not in shown:
                    x, y = zp(*PXY[pc], z); col = tcol(key); shown.add(pc)
                    WG.pin(cv, x, y, col, a, t, 13)
                    pv = PL[pc]
                    la_ = a * seg(t, T.E(0) + 0.1, T.E(0) + 0.4)
                    if len(pv) > 2 and la_ > 0.01: WG.tag(cv, pv[2], x + (pv[3] if len(pv) > 3 else 0), y - 52 + (pv[4] if len(pv) > 4 else 0), 40, WHITE, la_)
            if cur.get("arrow", True) and t >= cur["t0"]:
                src_, dst = (ka, kb) if cur.get("win", "a") == "a" else (kb, ka)
                if src_ in PXY and dst in PXY and src_ != dst:
                    (x0, y0), (x1, y1) = zp(*PXY[src_], z), zp(*PXY[dst], z)
                    mx, my = (x0 + x1) / 2, min(y0, y1) - 90 - 0.15 * abs(x1 - x0)
                    pts = [((1 - u) ** 2 * x0 + 2 * u * (1 - u) * mx + u * u * x1, (1 - u) ** 2 * y0 + 2 * u * (1 - u) * my + u * u * y1) for u in np.linspace(0, 1, 40)]
                    u = seg(t, cur["tw"] - 0.2, cur["tw"] + 0.6) if t < cur["tw"] + 0.6 else 1.0
                    n = max(2, int(len(pts) * u))
                    wc = tcol(cur["wkey"])
                    cv.line(pts[:n], wc, 16, 0.25 * a); cv.line(pts[:n], wc, 7, a)
                    if n >= 3:
                        (px, py), (qx, qy) = pts[n - 2], pts[n - 1]; ang = math.atan2(qy - py, qx - px)
                        hd = [(qx + 26 * math.cos(ang), qy + 26 * math.sin(ang)), (qx + 18 * math.cos(ang + 2.4), qy + 18 * math.sin(ang + 2.4)),
                              (qx + 18 * math.cos(ang - 2.4), qy + 18 * math.sin(ang - 2.4))]
                        cv.c.new_path(); cv.c.move_to(*hd[0]); cv.c.line_to(*hd[1]); cv.c.line_to(*hd[2]); cv.c.close_path(); cv.col(wc, a); cv.c.fill()
        for g in TAGS:
            ga = a * win(t, g["t0"], g["t1"], 0.25, 0.25)
            if ga > 0.01:
                x, y = zp(*geo2px(MM, g["lon"], g["lat"]), z)
                WG.tag(cv, g["text"], x + g.get("dx", 0), y + g.get("dy", 0), g.get("size", 44), COLS.get(g.get("col", "yellow"), YEL), ga)

    def draw_cards(cv, t):
        for cd in CARDS:
            a = win(t, cd["t0"] - 0.1, cd["t1"], 0.3, 0.3)
            if a <= 0.01: continue
            kind = cd.get("kind", "text")
            if kind == "table":
                cv.rrect(bx0 + 10, by0 + 10, bw - 20, bh - 20, 26, "#081226", 0.93 * a, stroke=SKY, lw=2.5, sa=0.6 * a)
                cv.text(cd.get("title", ""), 540, by0 + 58, 44, SKY, a, maxw=bw - 80)
                rows = cd["rows"]; rh = (bh - 110) / max(1, len(rows))
                for i, rw in enumerate(rows):
                    ra = a * seg(t, A(rw["at"]) - 0.05, A(rw["at"]) + 0.25)
                    if ra <= 0.01: continue
                    y = by0 + 112 + rh * (i + 0.5) - 6
                    ok = rw.get("ok")
                    if ok: cv.rrect(bx0 + 24, y - rh / 2 + 4, bw - 48, rh - 8, 16, GOLD, 0.16 * ra)
                    WG.mark(cv, "Y" if ok else "N", bx0 + 70, y, 26, ra, WG.pop_s(t, A(rw["at"])))
                    cv.text(rw["name"], bx0 + 112, y, 50, GOLD if ok else WHITE, ra, align="l", maxw=560)
                    cv.text(rw["val"], bx1 - 36, y, 50, GREEN if ok else "#FF8A8D", ra, align="r", maxw=330)
                continue
            y = cd.get("y", bcy)
            txt = cd["text"]; sz = cd.get("size", 62)
            # measure lines via WG.title wrapping (2 lines max)
            hh = cd.get("h", 250 if not cd.get("who") else 300)
            cv.rrect(bx0 + 20, y - hh / 2, bw - 40, hh, 26, "#081226", 0.88 * a, stroke=GOLD if kind == "quote" else SKY, lw=3, sa=0.8 * a)
            parts = txt.split("|"); lh = sz * 1.2
            y_ = y - (30 if (kind == "quote" and cd.get("who")) else 0) - lh * (len(parts) - 1) / 2
            if kind == "quote": cv.text("\u201C", bx0 + 76, y - hh / 2 + 56, 110, GOLD, a, shadow=False)
            for i_, ptxt in enumerate(parts): WG.title(cv, ptxt.strip(), y_ + i_ * lh, a, sz, WHITE, maxw=bw - 120)
            if kind == "quote" and cd.get("who"): cv.text(cd["who"], 540, y + hh / 2 - 42, 40, SKY, a, bold=False, maxw=bw - 100)

    def draw_banners(cv, t):
        for b in BAN:
            a = win(t, b["t0"], b["t1"], 0.15, 0.3)
            if a <= 0.01: continue
            col = COLS.get(b.get("col", "yellow"), YEL); sc = back(seg(t, b["t0"], b["t0"] + 0.3, lambda u: u))
            c = cv.c; c.save(); c.translate(540, VC); c.rotate(-0.06); c.scale(sc, sc)
            w = min(1000, cv.tw(b["text"], 110) + 120)
            cv.rrect(-w / 2 - 10, -86, w + 20, 172, 22, "#000000", 0.35 * a)
            cv.rrect(-w / 2, -78, w, 156, 18, col, a)
            cv.text(b["text"], 0, 0, 110, "#0B1730", a, shadow=False, maxw=w - 60)
            c.restore()

    def frame(t):
        L = lay(t); z = mz(t)
        mp = map_img(z)
        # outside the band the map dims as the layout comes in
        outside = mp * (1 - 0.82 * L) + DIMBG * 0.82 * L
        band = mp * 0.92
        # photos in the band
        reg = band[by0:by1, bx0:bx1]
        pa_tot = 0.0
        for p in PH:
            a_ = ph_a(p, t)
            if a_ <= 0.001: continue
            kb = p.get("kb") or [0.5, 0.5, 1.0, 0.5, 0.5, 1.12]
            u = ease((t - p["t0"]) / max(0.5, p["t1"] - p["t0"]))
            ph = band_kb(p["im"], kb[0] + (kb[3] - kb[0]) * u, kb[1] + (kb[4] - kb[1]) * u, kb[2] + (kb[5] - kb[2]) * u)
            reg = reg * (1 - a_) + ph * (1 - 0.6 * p.get("dim", 0.2)) * a_; pa_tot = max(pa_tot, a_)
        band[by0:by1, bx0:bx1] = reg * (BVIG * (0.5 + 0.5 * L) + (1 - (0.5 + 0.5 * L)))
        bm = BMASK * L + (1 - L)
        base = band * bm + outside * (1 - bm)
        cv = Canvas(base)
        cv.rrect(bx0, by0, bw, bh, 30, None, 1, stroke="#2A3B5C", lw=3, sa=L)
        tab_a = max([win(t, c_["t0"] - 0.1, c_["t1"], 0.3, 0.3) for c_ in CARDS if c_.get("kind") == "table"] + [0.0])
        draw_band_map_overlays(cv, t, (1 - pa_tot) * (1 - tab_a))
        for k_, p in enumerate(PH):
            a_ = ph_a(p, t)
            for q in PH[k_ + 1:]: a_ *= 1 - ph_a(q, t)
            if a_ > 0.01 and p.get("credit"):
                txt = "Photo: " + p["credit"]; tw_ = min(bw - 30, cv.tw(txt, 36, bold=False) + 30)
                cv.rrect(540 - tw_ / 2, by1 - 60, tw_, 48, 14, "#0B1322", 0.72 * a_)
                cv.text(txt, 540, by1 - 36, 36, "#E6EEF8", a_, bold=False, maxw=bw - 40, minsz=36, shadow=False)
        draw_cards(cv, t)
        if CTA and t >= CTA_T:
            ca = seg(t, CTA_T, CTA_T + 0.35) * (1 - seg(t, HOOKB - 0.1, HOOKB + 0.3))
            if ca > 0.01:
                w = min(980, cv.tw(CTA["text"].replace("*", ""), 60) + 100)
                cv.rrect(540 - w / 2, by1 - 150, w, 116, 58, SKY, ca); WG.title(cv, CTA["text"], by1 - 92, ca, 60)
        draw_scorebug(cv, t)
        draw_bracket(cv, t)
        draw_banners(cv, t)
        draw_screen(cv, [c_ for c_ in cards if c_[1] > T.E(0) - 0.05], t, D.get("title_y", 860), 64)
        if t < REW: draw_sources(cv, S, T, t, D.get("src_y", 1490))
        out = cv.out().astype(np.float32)
        out = paste_hook(out, hook_img, t, T.E(0), HY, 1.0)
        if t >= HOOKB: out = HK.paste(out, hook_img, 540, HY, 1.0, seg(t, HOOKB, HOOKB + 0.35))
        c2 = Canvas(out); draw_subs(c2, CH, t, D.get("subs_y", 1612)); return np.clip(c2.out(), 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    ev = [(c["t0"], "whoosh", -32) for c in CAMS if c["t0"] > 0.5]
    for m in MT: ev += [(m["t0"] - 0.1, "pop", -30), (m["tw"] + 0.3, "ding", -27)]
    ev += [(s["t0"], "thud", -22) for s in STP]
    ev += [(b["t0"], "whoosh", -28) for b in BAN]
    ev += [(c["t0"], "tick", -30) for c in CLK if c["t0"] > 0.2]
    if LIFE_T < T.END: ev.append((LIFE_T, "roll", LIFE_T + LIFE_D))
    if FIN["t0"] < T.END: ev.append((FIN["t0"], "ding", -24))
    aud = build_mix(S, T, work, ev)
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    encode(frame, T.END, f"{work}/mix.m4a", final, loop=True)
    return dict(voice=J["voice"], mix=aud, photos=[dict(file=p["file"], credit=p.get("credit")) for p in PH])
