#!/usr/bin/env python3
"""map_tear_v2.py (Noam 3 Nov 07:45: "the tear is pretty good, you can improve it even more"): v2 of map_tear.py.
Changes: fibrous torn edge (soft translucent fibres + white core + more hairs), richer paper grain, shadow lifts as the
halves separate, crisper rip sound, a 0.15 s hang at the moment of the tear, glued back with a snap and a light flash
that runs along the seam. map_tear.py itself is unchanged (the control).

map_tear.py: reusable 2.0 to 2.5 s animated opener for @noamroz__ map reels (Noam, 3 Nov: "The map is tearing itself,
and then it's glued back together"). No face.

  0.000 to 0.375 T  a paper world map (full frame width, slightly rotated, paper grain, soft shadow) rips by itself almost
                    fully across along a jagged line, the halves pull apart with a small rotation, white fibres on the edge
  0.375 to 0.625 T  VHS fast rewind (scanlines, RGB split, tracking band, "◀◀"): the halves fly back, the seam seals with a
                    soft gold glue glow
  0.625 to 1.000 T  fast ease-in zoom into the target view; when a target reel is given, its frame 0 is warped onto the same
                    geography and cross-dissolved in, so the cut is a match cut (the reel's frame-0 hook is fully visible
                    on the first reel frame)
T = 2.4 s by default (--dur 2.0 to 2.5). Every frame is piped to ffmpeg (nothing written to disk). SFX are synthesised:
tear, rewind whirr, glue shimmer, zoom whoosh. The reel's own audio starts exactly at the cut (concat, no ducking).

Target view, one of:
  --cam lon,lat,width_deg,cy   the same numbers as rlib.camw() in the reel engines (exact match for those reels)
  --box lon0,lat0,lon1,lat1    a lon/lat box, fitted inside the 1080x1920 frame and centred

Examples:
  python3 map_tear.py --box 33.6,29.3,36.4,33.6 --style empires --hold 1.5 --out demo.mp4          # opener only
  python3 map_tear.py --reel v5c.mp4 --cam 15,14,330,1130 --style empires --out v5c_tear.mp4      # opener + reel
"""
import sys, os, math, json, argparse, subprocess, shutil, time, wave
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hook as hk_

GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "geo.json")     # Natural Earth shapes (world, empires, isr49, wb, gaza), read only
W, H, FPS, SR = 1080, 1920, 30, 48000
SS = 4; SC = 1 << SS                                        # cv2 sub-pixel shift


def hexc(h): h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (4, 2, 0)], np.float32)   # BGR
NIGHT, NAVY, SKY, GOLD, RED = hexc("0B1426"), hexc("163261"), hexc("45A4DD"), hexc("FFCD50"), hexc("E5484D")
PAPER, OCEAN, LAND = hexc("EFE6D0"), hexc("C4D8E2"), hexc("EADDC0")
FIBRE, GREEN, TERR = hexc("FBF8F1"), hexc("2ECC71"), hexc("E3B347")


def ease_io(u): u = min(1.0, max(0.0, u)); return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2
def ease_out(u): u = min(1.0, max(0.0, u)); return 1 - (1 - u) ** 3
def ease_in(u): u = min(1.0, max(0.0, u)); return u ** 2.6
def seg(t, a, b, f=ease_io): return f((t - a) / (b - a)) if b > a else float(t >= a)


def disk_guard(path=os.path.dirname(os.path.abspath(__file__)), need_mb=500, wait_s=600):
    t0 = time.time()
    while shutil.disk_usage(path).free / 1e6 < need_mb:
        if time.time() - t0 > wait_s:
            sys.exit(f"map_tear: paused, disk free under {need_mb} MB for {wait_s} s, giving up")
        print(f"map_tear: disk free under {need_mb} MB, pausing", file=sys.stderr); time.sleep(15)


# ------------------------------------------------------------------ geometry
def load_geo(style):
    g = json.load(open(GEO))
    def rings(geom):
        cs = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out = []
        for poly in cs:
            for i, r in enumerate(poly):
                a = np.asarray(r, np.float64)
                if len(a) >= 3: out.append((a, i > 0))
        return out
    def pack(geoms):
        rs = [r for gm in geoms for r, hole in rings(gm) if not hole]
        return [(r, (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())) for r in rs]
    L = {"land": pack([v for k, v in g["world"].items() if k != "ATA"])}
    L["empires"] = pack([p["geom"] for v in g["empires"].values() for p in v]) if style == "empires" else []
    L["isr"] = pack([g["isr49"]]); L["terr"] = pack([g["wb"], g["gaza"]])
    return L


def M3(a, b, c, d, e, f): return np.array([[a, b, c], [d, e, f], [0, 0, 1.0]])


class View:
    """geo (lon, lat) -> screen: s = sT + R(rot) . diag(k*cosf, -k) . (g - T)"""
    def __init__(s, T, sT, k, rot, cosf): s.T, s.sT, s.k, s.rot, s.cosf = T, sT, k, rot, cosf
    def M(s):
        c, n = math.cos(s.rot), math.sin(s.rot); kx, ky = s.k * s.cosf, -s.k
        A = np.array([[c * kx, -n * ky], [n * kx, c * ky]]); b = np.array(s.sT) - A @ np.array(s.T)
        return M3(A[0, 0], A[0, 1], b[0], A[1, 0], A[1, 1], b[1])


def apply(M, pts): return pts @ M[:2, :2].T + M[:2, 2]


def view_box(M):   # geo bbox visible on screen (for culling)
    inv = np.linalg.inv(M); c = apply(inv, np.array([[-60, -60], [W + 60, -60], [-60, H + 60], [W + 60, H + 60]], float))
    return c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()


def fill(canvas, M, rings, color, alpha=1.0, vb=None, outline=None, lw=1.0, ocol=None, oalpha=1.0):
    polys = []
    for r, (x0, y0, x1, y1) in rings:
        if vb and (x1 < vb[0] or x0 > vb[2] or y1 < vb[1] or y0 > vb[3]): continue
        p = apply(M, r)
        if np.ptp(p[:, 0]) < 0.7 and np.ptp(p[:, 1]) < 0.7: continue
        polys.append(np.round(p * SC).astype(np.int32))
    if not polys: return
    if alpha >= 1: cv2.fillPoly(canvas, polys, color.tolist(), cv2.LINE_AA, SS)
    else:
        lay = canvas.copy(); cv2.fillPoly(lay, polys, color.tolist(), cv2.LINE_AA, SS); cv2.addWeighted(lay, alpha, canvas, 1 - alpha, 0, canvas)
    if outline is not None:
        lay = canvas.copy() if oalpha < 1 else canvas
        cv2.polylines(lay, polys, True, outline.tolist(), max(1, int(round(lw))), cv2.LINE_AA, SS)
        if oalpha < 1: cv2.addWeighted(lay, oalpha, canvas, 1 - oalpha, 0, canvas)


# ------------------------------------------------------------------ the paper sheet
class Sheet:
    MX, MY = 34.0, 34.0
    LAT0, LAT1 = 84.0, -60.0

    def __init__(s, cosf, width=1240):
        s.cosf = cosf; s.w = width; s.kx = (width - 2 * s.MX) / 360.0; s.k = s.kx / cosf
        s.h = 2 * s.MY + (s.LAT0 - s.LAT1) * s.k
        # sheet px -> geo
        s.G = M3(1 / s.kx, 0, -180 - s.MX / s.kx, 0, -1 / s.k, s.LAT0 + s.MY / s.k)
        rng = np.random.default_rng(7)
        gw, gh = int(s.w * 2), int(s.h * 2)
        n = rng.normal(0, 1, (gh, gw)).astype(np.float32)
        fine = cv2.GaussianBlur(n, (0, 0), 0.8); fib = cv2.GaussianBlur(rng.normal(0, 1, (gh, gw)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=1.2)
        blot = cv2.GaussianBlur(rng.normal(0, 1, (gh // 8, gw // 8)).astype(np.float32), (0, 0), 6); blot = cv2.resize(blot, (gw, gh))
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32); vx, vy = (xx / gw - 0.5) * 2, (yy / gh - 0.5) * 2
        light = 1 - 0.07 * (vx ** 2 + vy ** 2) - 0.03 * vx + 0.02 * vy
        g = light * (1 + 0.04 * fine / fine.std() + 0.035 * fib / fib.std() + 0.032 * blot / (blot.std() + 1e-6))
        s.grain = g.astype(np.float32); s.gscale = 2.0
        # jagged tear line in sheet px (top to bottom), set by tear_x()
        s.tear = None

    def geo2px(s, lon, lat): return apply(np.linalg.inv(s.G), np.array([[lon, lat]]))[0]

    def make_tear(s, lon):
        x0 = s.geo2px(lon, 0)[0]; rng = np.random.default_rng(11)
        ys = np.linspace(-2, s.h + 2, 140)
        wander = 26 * np.sin(ys / 61 + 0.7) + 14 * np.sin(ys / 23 + 2.1) + 7 * np.sin(ys / 9.5)
        jag = rng.uniform(-5.5, 5.5, ys.size); jag[::2] *= 0.4
        s.tear = np.stack([x0 + wander + jag, ys], 1)
        s.fib_w = rng.uniform(1.6, 4.6, ys.size)                    # white torn edge width per segment
        hy = rng.uniform(0, s.h, 620); s.hairs = (hy, rng.uniform(2.0, 12.0, 620), rng.uniform(-1.1, 1.1, 620), rng.integers(0, 2, 620))

    def tear_x(s, y): return np.interp(y, s.tear[:, 1], s.tear[:, 0])


def render_map(M_geo, style_layers, vb):
    """draw the printed map (ocean, graticule, land, empires, Israel) through M_geo onto a BGR canvas"""
    c = np.zeros((H, W, 3), np.uint8); c[:] = OCEAN.astype(np.uint8)
    lay = c.copy()
    for lon in range(-180, 181, 30):
        p = apply(M_geo, np.array([[lon, Sheet.LAT0], [lon, Sheet.LAT1]], float)); cv2.line(lay, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), NAVY.tolist(), 1, cv2.LINE_AA, SS)
    for lat in range(-60, 85, 30):
        p = apply(M_geo, np.array([[-180, lat], [180, lat]], float)); cv2.line(lay, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), NAVY.tolist(), 1, cv2.LINE_AA, SS)
    cv2.addWeighted(lay, 0.16, c, 0.84, 0, c)
    fill(c, M_geo, style_layers["land"], LAND, 1.0, vb, outline=NAVY, lw=1, oalpha=0.38)
    if style_layers["empires"]: fill(c, M_geo, style_layers["empires"], RED, 0.86, vb)
    fill(c, M_geo, style_layers["isr"], GOLD, 1.0, vb, outline=NAVY, lw=1, oalpha=0.6)
    big = math.hypot(M_geo[0, 0], M_geo[1, 0]) > 30          # Green Line only once Israel is big enough to read
    fill(c, M_geo, style_layers["terr"], TERR, 1.0, vb, outline=GREEN if big else None, lw=2, oalpha=0.95)
    return c


def sheet_layer(sh, M_sheet, layers, mask_poly=None, fibre=None):
    """the paper sheet through sheet-px -> screen affine: returns (bgr float, alpha float)"""
    M_geo = M_sheet @ np.linalg.inv(sh.G); vb = view_box(M_geo)
    img = render_map(M_geo, layers, vb).astype(np.float32)
    # printed-area mask, paper margin, frame lines
    pr = np.array([[sh.MX, sh.MY], [sh.w - sh.MX, sh.MY], [sh.w - sh.MX, sh.h - sh.MY], [sh.MX, sh.h - sh.MY]])
    full = np.array([[0, 0], [sh.w, 0], [sh.w, sh.h], [0, sh.h]], float)
    pm = np.zeros((H, W), np.uint8); cv2.fillPoly(pm, [np.round(apply(M_sheet, pr) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
    pm = pm.astype(np.float32)[..., None] / 255
    img = img * pm + PAPER * (1 - pm)
    for d, a in ((0, 0.55), (7, 0.3)):
        q = pr + np.array([[-d, -d], [d, -d], [d, d], [-d, d]]); cv2.polylines(img, [np.round(apply(M_sheet, q) * SC).astype(np.int32)], True, (NAVY * a + PAPER * (1 - a)).tolist(), 1, cv2.LINE_AA, SS)
    # paper grain, warped with the sheet
    Mg = M_sheet @ M3(1 / sh.gscale, 0, 0, 0, 1 / sh.gscale, 0)
    gr = cv2.warpAffine(sh.grain, Mg[:2], (W, H), flags=cv2.INTER_LINEAR, borderValue=1.0)
    zf = math.hypot(M_sheet[0, 0], M_sheet[1, 0]); fade = min(1.0, max(0.0, (3.0 - zf) / 2.0))   # big grain fades as we zoom
    img *= (1 + (gr - 1) * fade)[..., None]
    if fade < 1: img *= (1 + (1 - fade) * SCREEN_GRAIN)[..., None]
    # alpha: sheet rectangle, optionally clipped to one torn half
    al = np.zeros((H, W), np.uint8); cv2.fillPoly(al, [np.round(apply(M_sheet, full) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
    if mask_poly is not None:
        hm = np.zeros((H, W), np.uint8); cv2.fillPoly(hm, [np.round(apply(M_sheet, mask_poly) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
        al = np.minimum(al, hm)
    al = al.astype(np.float32) / 255
    if fibre is not None:   # white torn edge + loose hairs, only where torn (y < front)
        side, yf, amt = fibre
        if amt > 0.01 and yf > 0:
            fl = np.zeros((H, W), np.uint8); t = sh.tear; idx = np.where(t[:, 1] < yf)[0]
            for i in idx[:-1]:
                p = apply(M_sheet, t[i:i + 2]); cv2.line(fl, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), 255, max(1, int(round(sh.fib_w[i] * 1.6))), cv2.LINE_AA, SS)
            soft = cv2.GaussianBlur(fl.astype(np.float32) / 255, (0, 0), 3.2) * 1.6
            fl = fl.astype(np.float32) / 255 * al                # keep the strip inside this half
            hy, hl, ha, hs = sh.hairs; hl_ = np.zeros((H, W), np.uint8)
            for y, l, a, sd in zip(hy, hl, ha, hs):
                if y > yf or (sd == 0) != (side == "L"): continue
                x = sh.tear_x(y); d = 1 if side == "L" else -1
                p = apply(M_sheet, np.array([[x, y], [x + d * l * math.cos(a), y + l * math.sin(a)]])); cv2.line(hl_, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), 220, 1, cv2.LINE_AA, SS)
            hl_ = hl_.astype(np.float32) / 255 * amt
            soft = np.clip(soft, 0, 1) * amt * 0.5
            img = img * (1 - soft[..., None]) + FIBRE * soft[..., None]; al = np.maximum(al, soft * 0.85)
            fl = np.maximum(fl * amt, hl_)
            img = img * (1 - fl[..., None]) + FIBRE * fl[..., None]; al = np.maximum(al, hl_)
    return img, al


SCREEN_GRAIN = cv2.GaussianBlur(np.random.default_rng(9).normal(0, 0.022, (H, W)).astype(np.float32), (0, 0), 0.7)


def background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); r = np.sqrt(((xx - 540) / 900) ** 2 + ((yy - 900) / 1100) ** 2)
    m = np.clip(1 - r, 0, 1)[..., None] ** 1.5
    return NIGHT * (1 - 0.55 * m) + NAVY * 0.55 * m


def composite(bg, layers, lift=0.0):
    out = bg.copy()
    for img, al in layers:
        sh = cv2.GaussianBlur(al, (0, 0), 13 + 14 * lift); sh = np.roll(np.roll(sh, int(16 + 34 * lift), 0), int(9 + 20 * lift), 1) * (0.6 - 0.12 * lift)
        out = out * (1 - sh[..., None]); out = out * (1 - al[..., None]) + img * al[..., None]
    return out


# ------------------------------------------------------------------ VHS + glow
def vhs(frame, t, amt, rng):
    if amt <= 0.01: return frame
    f = frame.copy(); sh = int(round(7 * amt))
    f[..., 2] = np.roll(frame[..., 2], sh, 1); f[..., 0] = np.roll(frame[..., 0], -sh, 1)
    g = f.mean(2, keepdims=True); f = f * (1 - 0.25 * amt) + g * 0.25 * amt
    f = f * (1 - 0.05 * amt) + 14 * amt
    f[::3] *= 1 - 0.22 * amt; f[1::3] *= 1 - 0.08 * amt
    for k in range(2):   # tracking bands rolling down
        yc = int(((t * 2.6 + k * 0.53) % 1.0) * (H + 200)) - 100; y0, y1 = max(0, yc - 26 - 20 * k), min(H, yc + 26)
        if y1 > y0:
            f[y0:y1] = np.roll(f[y0:y1], int(rng.integers(-30, 30) * amt), 1)
            f[y0:y1] += rng.normal(0, 38 * amt, (y1 - y0, W, 1)).astype(np.float32)
    jit = int(round(rng.integers(-3, 4) * amt)); f = np.roll(f, jit, 0)
    f += rng.normal(0, 7 * amt, (H // 4, W // 4, 1)).repeat(4, 0).repeat(4, 1).astype(np.float32)
    # the "◀◀" mark, top left, blinking
    if (t * 6) % 1.0 < 0.72:
        a = amt; col = np.array([255, 255, 255], np.float32)
        lay = np.zeros((H, W), np.uint8)
        for dx in (0, 58):
            x0, y0 = 96 + dx, 200; tri = np.array([[x0 + 58, y0 - 34], [x0 + 58, y0 + 34], [x0, y0]], np.int32)
            cv2.fillPoly(lay, [tri * SC], 255, cv2.LINE_AA, SS)
        sh_ = cv2.GaussianBlur(lay.astype(np.float32) / 255, (0, 0), 6)
        m = lay.astype(np.float32)[..., None] / 255 * a
        f = f * (1 - sh_[..., None] * 0.5 * a); f = f * (1 - m) + col * m
    return f


def glow(frame, M_sheet, sh, a):
    if a <= 0.01: return frame
    lay = np.zeros((H, W), np.uint8); p = np.round(apply(M_sheet, sh.tear) * SC).astype(np.int32)
    cv2.polylines(lay, [p], False, 255, 4, cv2.LINE_AA, SS)
    l = lay.astype(np.float32) / 255; g = cv2.GaussianBlur(l, (0, 0), 14) * 2.2 + cv2.GaussianBlur(l, (0, 0), 4) * 0.9 + l * 0.5
    g = np.clip(g * a, 0, 1.6)[..., None]
    return frame + g * (GOLD * 0.9 + 255 * 0.1) * 0.62


def flash(frame, M_sheet, sh, t, seal):
    """v2: a bright point of light runs down the seam as it seals, then the whole seam flashes once"""
    u = (t - (seal - 0.03)) / 0.16
    if not (0 <= u <= 1.6): return frame
    p = apply(M_sheet, sh.tear); n = len(p); lay = np.zeros((H, W), np.uint8)
    if u <= 1:
        c = u * (n - 1); i0, i1 = max(0, int(c - n * 0.09)), min(n - 1, int(c + 2))
        cv2.polylines(lay, [np.round(p[i0:i1 + 1] * SC).astype(np.int32)], False, 255, 5, cv2.LINE_AA, SS)
        a_ = 1.0
    else:
        cv2.polylines(lay, [np.round(p * SC).astype(np.int32)], False, 255, 3, cv2.LINE_AA, SS); a_ = max(0.0, 1 - (u - 1) / 0.6) * 0.8
    l = lay.astype(np.float32) / 255; g = cv2.GaussianBlur(l, (0, 0), 3) * 1.2 + cv2.GaussianBlur(l, (0, 0), 16) * 2.4 + l
    return frame + np.clip(g * a_, 0, 1.8)[..., None] * (WHITE_F * 0.55 + GOLD * 0.45) * 0.8


WHITE_F = np.array([255, 255, 255], np.float32)


def zoom_blur(frame, center, amt):
    if amt <= 0.004: return frame
    acc = frame.copy(); n = 4
    for i in range(1, n):
        s = 1 + amt * i / (n - 1); cx, cy = center
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        acc += cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


# ------------------------------------------------------------------ sound
def synth(T, tl, sr=SR):
    n = int(round(T * sr)); out = np.zeros((n, 2)); rng = np.random.default_rng(3); t = np.arange(n) / sr
    def bp(x, lo, hi):
        X = np.fft.rfft(x); f = np.fft.rfftfreq(x.size, 1 / sr)
        X *= np.clip((f - lo * 0.7) / (lo * 0.3 + 1), 0, 1) * np.clip((hi * 1.3 - f) / (hi * 0.3), 0, 1); return np.fft.irfft(X, x.size)
    def put(x, at, g, pan=0.0):
        i = int(at * sr); x = x[: max(0, n - i)]
        out[i:i + x.size, 0] += x * g * (1 - max(0, pan)); out[i:i + x.size, 1] += x * g * (1 + min(0, pan))
    # v2 rip: crisp granular crackle (short grains, more clicks, less low end) for the rip, near silence in the hang
    # (one tiny creak), a second fast burst as the halves pull apart, plus a dry snap
    def rip(t0, t1, dens_s, g, seed):
        r = np.random.default_rng(seed); td = t1 - t0; m = int(td * sr); tt = np.arange(m) / sr
        env = np.clip(tt / 0.012, 0, 1) * np.clip((td - tt) / 0.05, 0, 1) * (0.8 + 0.2 * np.sin(tt * 47))
        grains = np.zeros(m); k = 0
        while k < m:
            L = int(r.uniform(0.0008, 0.0045) * sr)
            if k + L <= m: grains[k:k + L] += r.uniform(0.4, 1.0) * np.hanning(L)
            k += int(r.exponential(dens_s) * sr) + 1
        nz = r.normal(0, 1, m)
        x = bp(nz * (0.2 + grains), 1500, 12000) * env + 0.22 * bp(nz * grains, 400, 1800) * env
        cl = np.zeros(m); ci = r.integers(0, m, int(320 * td / 0.3)); cl[ci] = r.uniform(-1, 1, ci.size); x += 1.8 * bp(cl, 3000, 14000) * env
        x /= np.abs(x).max(); put(x, t0, g); return x
    tear = rip(tl["tear_start"], tl["rip_end"], 0.0028, 0.66, 1)
    rip(tl["hang_end"] - 0.01, tl["tear_end"], 0.0022, 0.5, 2)
    m = int(0.06 * sr); cr = bp(rng.normal(0, 1, m), 2500, 7000) * np.hanning(m); put(cr / np.abs(cr).max(), tl["rip_end"] + 0.06, 0.12)
    snap = bp(rng.normal(0, 1, int(0.04 * sr)), 2500, 12000) * np.exp(-np.arange(int(0.04 * sr)) / (0.004 * sr)); put(snap / np.abs(snap).max(), tl["hang_end"], 0.5)
    td = tl["rip_end"] - tl["tear_start"]
    # rewind: button clunk, tape whirr rising, the tear played backwards at double speed (chipmunk chatter)
    r0, r1 = tl["rew_start"], tl["rew_end"]; rd = r1 - r0; m = int(rd * sr); tt = np.arange(m) / sr
    clunk = np.sin(2 * np.pi * 95 * np.arange(int(0.07 * sr)) / sr) * np.exp(-np.arange(int(0.07 * sr)) / (0.012 * sr)); put(clunk, r0 - 0.01, 0.55)
    put(bp(rng.normal(0, 1, int(0.02 * sr)), 2000, 8000) * np.hanning(int(0.02 * sr)), r0 - 0.01, 0.35)
    f = 220 + 1300 * (tt / rd) ** 1.3 + 25 * np.sin(2 * np.pi * 13 * tt); ph = 2 * np.pi * np.cumsum(f) / sr
    whirr = (np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.25 * np.sign(np.sin(3 * ph))) * 0.3 + 0.6 * bp(rng.normal(0, 1, m), 1500, 6000) * (0.5 + 0.5 * np.sin(2 * np.pi * 37 * tt))
    wenv = np.clip(tt / 0.05, 0, 1) * np.clip((rd - tt) / 0.08, 0, 1); whirr *= wenv / np.abs(whirr).max()
    put(whirr, r0, 0.36)
    back = tear[::-1][::2]; back = back[:m] * wenv[: back[:m].size]; put(back, r0, 0.32)
    # glue: soft shimmer at the seal
    g0 = tl["seal"]; m = int(0.6 * sr); tt = np.arange(m) / sr
    shim = sum(np.sin(2 * np.pi * fr * tt + i) * a for i, (fr, a) in enumerate(((880, 0.5), (1318.5, 0.35), (1760, 0.22), (2637, 0.12))))
    shim *= np.clip(tt / 0.02, 0, 1) * np.exp(-tt / 0.18); put(shim / np.abs(shim).max(), g0 - 0.02, 0.26)
    m = int(0.09 * sr); tt2 = np.arange(m) / sr
    knock = bp(rng.normal(0, 1, m), 1800, 9000) * np.exp(-tt2 / 0.005) + 0.9 * np.sin(2 * np.pi * (140 + 300 * np.exp(-tt2 / 0.01)) * tt2) * np.exp(-tt2 / 0.03)
    put(knock / np.abs(knock).max(), g0 - 0.005, 0.7)
    # zoom whoosh: noise, cutoff and level rising (ease-in), cut right at the end
    z0, z1 = tl["zoom_start"], T; m = int((z1 - z0) * sr); tt = np.arange(m) / sr; u = tt / (z1 - z0)
    wn = rng.normal(0, 1, m); wh = np.zeros(m); lp1 = lp2 = 0.0     # two-pole lowpass, cutoff rising sample by sample
    cut = 250 + 5600 * u ** 2.2; al = 1 - np.exp(-2 * np.pi * cut / sr)
    for i in range(m):
        lp1 += al[i] * (wn[i] - lp1); lp2 += al[i] * (lp1 - lp2); wh[i] = lp2
    wh = wh - np.convolve(wh, np.ones(64) / 64, "same") * 0.6        # thin the low rumble a little
    wenv = (u ** 2.2) * np.clip((1 - u) / 0.04, 0, 1); wh = wh * wenv / (np.abs(wh * wenv).max() + 1e-9)
    put(wh, z0, 0.55, 0); put(np.roll(wh, 300), z0, 0.12, 0.6)
    out /= max(1e-9, np.abs(out).max()); out *= 10 ** (-3 / 20)
    return out


def write_wav(path, a, sr=SR):
    a = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(a.tobytes())


# ------------------------------------------------------------------ main
def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,sample_rate,channels", "-show_entries", "format=duration", "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def first_frame(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3).astype(np.float32)


def target_view(A):
    if A.cam:
        lon, lat, wd, cy = map(float, A.cam.split(","))
        cosf = max(0.6, math.cos(math.radians(min(62, abs(lat))))); k = W / (wd * cosf)
        return View((lon, lat), (540.0, cy), k, 0.0, cosf)
    lon0, lat0, lon1, lat1 = map(float, A.box.split(","))
    lat = (lat0 + lat1) / 2; cosf = max(0.6, math.cos(math.radians(min(62, abs(lat)))))
    k = min(W * 0.92 / ((lon1 - lon0) * cosf), H * 0.86 / (lat1 - lat0))
    return View(((lon0 + lon1) / 2, lat), (540.0, H / 2), k, 0.0, cosf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reel", help="target reel mp4 (opener + reel); omit for the opener alone")
    ap.add_argument("--cam", help="target view as lon,lat,width_deg,cy (rlib.camw numbers)")
    ap.add_argument("--box", help="target view as lon0,lat0,lon1,lat1")
    ap.add_argument("--style", choices=["empires", "plain"], default="empires")
    ap.add_argument("--dur", type=float, default=2.4, help="opener length, 2.0 to 2.5 s")
    ap.add_argument("--tear-lon", type=float, default=None, help="longitude the tear runs along (default: 0, or 45 deg away from the target)")
    ap.add_argument("--rot", type=float, default=-3.0, help="sheet rotation in degrees at the start")
    ap.add_argument("--hold", type=float, default=0.0, help="opener-only: hold the final frame this long")
    ap.add_argument("--hook"); ap.add_argument("--hook-y", type=float, default=930); ap.add_argument("--hook-s", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snaps", help="debug: comma list of times, writes PNGs next to --out instead of a video")
    A = ap.parse_args()
    if not (A.cam or A.box): ap.error("give --cam or --box")
    T = min(2.5, max(2.0, A.dur)); disk_guard()
    F = target_view(A); sh = Sheet(F.cosf)
    tlon = A.tear_lon if A.tear_lon is not None else (0.0 if abs(F.T[0]) > 25 or (A.cam and float(A.cam.split(",")[2]) > 120) else F.T[0] - 45)
    sh.make_tear(tlon); layers = load_geo(A.style); bg = background()
    tl = dict(tear_start=0.04 * T / 2.4, rip_end=0.255 * T, hang_end=0.255 * T + 0.15, tear_end=0.40 * T, rew_start=0.40 * T, rew_end=0.625 * T, zoom_start=0.625 * T)
    tl["seal"] = tl["rew_end"] - 0.09
    yf_max = sh.h * 0.93
    # start view: sheet centred, rotated; T's screen position on the sheet
    rot0 = math.radians(A.rot); ctr = np.array([540.0, 930.0])
    R = lambda a: np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    def sheet_M(rot, center):
        Rm = R(rot); b = center - Rm @ np.array([sh.w / 2, sh.h / 2]); return M3(Rm[0, 0], Rm[0, 1], b[0], Rm[1, 0], Rm[1, 1], b[1])
    M0 = sheet_M(rot0, ctr)
    sT0 = apply(M0, sh.geo2px(*F.T)[None])[0]
    reel_f0 = first_frame(A.reel) if A.reel else None
    MF = F.M()

    def tear_state(t):
        if t < tl["rip_end"]:      # fast rip almost all the way, the halves barely open
            u = seg(t, tl["tear_start"], tl["rip_end"], ease_out); return 0.9 * u, 0.2 * u, 0.0
        if t < tl["hang_end"]:     # the 0.15 s hang: it holds, trembling
            w = (t - tl["rip_end"]) / (tl["hang_end"] - tl["rip_end"]); return 0.9 + 0.02 * w, 0.2 + 0.012 * math.sin(w * 31), 0.0
        if t < tl["rew_start"]:    # then it gives: the halves pull apart and lift
            u = seg(t, tl["hang_end"], tl["tear_end"], ease_out); return 0.92 + 0.08 * u, 0.2 + 0.8 * u, 0.0
        if t < tl["rew_end"]:
            u = seg(t, tl["rew_start"], tl["seal"], lambda x: x ** 1.4)
            return 1 - u, 1 - u, 1.0
        return 0.0, 0.0, 0.0

    def half_M(Ms, side, p, s):
        yf = yf_max * p; hx, hy = sh.tear_x(yf), yf
        a = math.radians(6.5) * s * (-1 if side == "L" else 1); d = 12 * s * (-1 if side == "L" else 1)
        Rm = R(a); off = np.array([hx, hy]) - Rm @ np.array([hx, hy]) + np.array([d, 4 * s])
        return Ms @ M3(Rm[0, 0], Rm[0, 1], off[0], Rm[1, 0], Rm[1, 1], off[1])

    def half_poly(side, extra):
        t = sh.tear.copy(); t[:, 0] += extra
        if side == "L": return np.vstack([[[-50, -50]], t, [[-50, sh.h + 50]]])
        return np.vstack([[[sh.w + 50, -50]], t, [[sh.w + 50, sh.h + 50]]])

    rng = np.random.default_rng(5)
    def frame(t):
        z0 = tl["zoom_start"]
        if t < z0:
            p, s, vamt = tear_state(t)
            if p < 0.004 and s < 0.004:
                img, al = sheet_layer(sh, M0, layers); out = composite(bg, [(img, al)])
            else:
                yf = yf_max * p; amt = 1.0
                if tl["rew_start"] <= t: amt = 1 - seg(t, tl["seal"] - 0.06, tl["seal"])
                Lm, Rm_ = half_M(M0, "L", p, s), half_M(M0, "R", p, s)
                R_ = sheet_layer(sh, Rm_, layers, half_poly("R", -1.5), ("R", yf, amt))
                L_ = sheet_layer(sh, Lm, layers, half_poly("L", 1.5), ("L", yf, amt))
                out = composite(bg, [R_, L_], lift=s)
            gl = seg(t, tl["seal"] - 0.1, tl["seal"]) * (1 - seg(t, tl["seal"] + 0.02, z0 + 0.22))
            out = glow(out, M0, sh, gl)
            out = flash(out, M0, sh, t, tl["seal"])
            if t >= tl["rew_start"]:
                v = 1 - seg(t, tl["seal"] + 0.02, tl["rew_end"]); out = vhs(out, t, v, rng)
            return np.clip(out, 0, 255)
        # zoom: ease-in from the sheet view to the target view; the target point moves on a straight screen path
        u = (t - z0) / (T - z0); e = ease_in(u)
        k = math.exp(math.log(sh.k) + (math.log(F.k) - math.log(sh.k)) * e)
        sT = sT0 + (np.array(F.sT) - sT0) * e; rot = rot0 * (1 - e)
        V = View(F.T, tuple(sT), k, rot, F.cosf); Mg = V.M(); Ms = Mg @ sh.G
        img, al = sheet_layer(sh, Ms, layers); out = composite(bg, [(img, al)])
        gl = 1 - seg(t, tl["seal"] + 0.02, z0 + 0.22); out = glow(out, Ms, sh, gl)
        out = flash(out, Ms, sh, t, tl["seal"])
        du = 1.0 / (FPS * (T - z0)); e2 = ease_in(min(1, u + du)); kr = math.exp((math.log(F.k) - math.log(sh.k)) * (e2 - e))
        out = zoom_blur(out, sT, min(0.055, (kr - 1) * 0.7) * (1 - seg(u, 0.86, 1.0)))
        if reel_f0 is not None:   # warp reel frame 0 onto the same geography and dissolve in
            a = seg(t, T - 0.36, T - 1.0 / FPS)
            if a > 0:
                Mr = Mg @ np.linalg.inv(MF); rf = cv2.warpAffine(reel_f0, Mr[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out = out * (1 - a) + rf * a
        return np.clip(out, 0, 255)

    raw = frame; himg = hk_.make(A.hook) if A.hook else None
    def frame(t):
        f_ = raw(t)
        if himg is None: return f_
        a = seg(t, T - 0.36, T - 1.0 / FPS) if A.reel else 0.0
        return np.clip(hk_.overlay(f_, himg, t, T, tl["zoom_start"], bool(A.reel), y_reel=A.hook_y, s_reel=A.hook_s, alpha=1 - a), 0, 255)
    if A.snaps:
        for s_ in A.snaps.split(","):
            cv2.imwrite(f"{os.path.splitext(A.out)[0]}_{float(s_):.2f}.png", frame(float(s_)).astype(np.uint8))
        return
    nfr = int(round(T * FPS)); tmpd = os.path.dirname(os.path.abspath(A.out)); os.makedirs(tmpd, exist_ok=True)
    wav = os.path.join("/tmp", f"map_tear_{os.getpid()}.wav"); sfx = synth(T, tl)
    hold = 0.0 if A.reel else max(0.0, A.hold)
    if hold: sfx = np.vstack([sfx, np.zeros((int(round(hold * SR)), 2))])
    write_wav(wav, sfx)
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart"]
    src = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav]
    if A.reel:
        info = probe(A.reel); vs = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        has_a = any(s["codec_type"] == "audio" for s in info["streams"]); fr = vs["r_frame_rate"]
        ra = "[2:a]aformat=sample_rates=48000:channel_layouts=stereo[ra]" if has_a else f"anullsrc=r=48000:cl=stereo,atrim=0:{info['format']['duration']}[ra]"
        fc = (f"[0:v]fps={fr},format=yuv420p,setsar=1[ov];[2:v]scale={W}:{H},fps={fr},format=yuv420p,setsar=1[rv];"
              f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo[oa];{ra};[ov][oa][rv][ra]concat=n=2:v=1:a=1[v][a]")
        cmd = ["ffmpeg", "-y", "-v", "error", *src, "-i", A.reel, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", *enc, A.out]
    else:
        vf = f"tpad=stop_mode=clone:stop_duration={hold}" if hold else "null"
        cmd = ["ffmpeg", "-y", "-v", "error", *src, "-vf", vf, "-map", "0:v", "-map", "1:a", *enc, "-shortest", A.out] if not hold else \
              ["ffmpeg", "-y", "-v", "error", *src, "-vf", vf, "-map", "0:v", "-map", "1:a", *enc, A.out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(nfr):
        pr.stdin.write(frame(i / FPS).astype(np.uint8).tobytes())
    pr.stdin.close(); rc = pr.wait(); os.remove(wav)
    if rc: sys.exit(f"ffmpeg failed ({rc})")
    print(json.dumps(dict(out=A.out, opener_s=T, frames=nfr, render_s=round(time.time() - t0, 1), target=dict(lon=F.T[0], lat=F.T[1], k=round(F.k, 2)))))


if __name__ == "__main__":
    main()
