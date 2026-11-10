#!/usr/bin/env python3
"""map_open.py: 8 motion openers (2.0 to 2.5 s, no faces) for @noamroz__ map reels, for A/B tests (Noam, 3 Nov 07:34).
Same interface as map_tear.py, plus --style. Reuses map_tear.py's geodata, sheet, VHS, zoom and match-cut code.

  globe_spin_fast     3D globe spins fast, stops on Israel: a gold shockwave ripples out across the region, then unwraps into frame 0
  globe_spin_reverse  spins one way, overshoots past the Middle East, snaps back: the region lights up, ISRAEL rises out of the globe
  globe_spin_whip     whip-pan spin with heavy motion blur and a whoosh, a gold pin slams onto Israel with a thud and a shockwave
  globe_spin_night    night side with city lights spins, stops on Israel, Israel's lights switch on last and brightest, dawn sweeps in
  globe_unpeel        fast spin, a pin slams onto Israel, then the map unpeels from the globe into the flat map of frame 0
  map_crumple         a paper map scrunched into a ball, pops open and flattens (paper sounds), then zooms in
  map_tear            the existing opener (map_tear.py, unchanged), the control
  map_tear_v2         map_tear_v2.py: fibrous edge, shadow lift, crisper rip, 0.15 s hang, snap + light flash along the seam
  map_fold            an origami map unfolds panel by panel, the last fold (the north) reveals Israel
  map_puzzle          puzzle pieces fly in and snap together, the last piece is Israel's

Globes: orthographic projection of the Natural Earth shapes, per-pixel texture lookup with soft light, ocean specular,
rim light and atmosphere glow. The globe-to-target move is a continuous unwrap from orthographic to the flat view the
reel uses (angles scaled by a, a: 1 = globe, 0 = flat), so the cut into frame 0 is a match cut.
--hook puts the reel's hook on screen from frame 0 (hook.py) and moves it into the reel's frame-0 text for the cut.
Frames are piped to ffmpeg (nothing on disk). SFX synthesised per style. The reel's own audio starts at the cut.

  python3 map_open.py --style globe_spin_fast --box 33.6,29.3,36.4,33.6 --hold 0.5 --out demo.mp4
  python3 map_open.py --style map_fold --reel reel.mp4 --cam 25,32,310,950 --out reel_fold.mp4
"""
import sys, os, math, json, argparse, subprocess, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import map_tear as mt
import hook as hk_
from map_tear import W, H, FPS, SR, hexc, ease_io, ease_out, ease_in, seg, apply, M3, View

STYLES = ["globe_spin_fast", "globe_spin_reverse", "globe_spin_whip", "globe_spin_night", "globe_unpeel",
          "map_crumple", "map_tear", "map_tear_v2", "map_fold", "map_puzzle"]
NIGHT, NAVY, SKY, GOLD, RED = mt.NIGHT, mt.NAVY, mt.SKY, mt.GOLD, mt.RED
GREENL = hexc("3FBF7F")
G_OCEAN, G_LAND, G_BORDER, G_TERR = hexc("0E2250"), hexc("4A84C4"), hexc("A9D4F5"), hexc("D9A93F")
LIGHT_COL = hexc("FFD9A0"); WARM = hexc("FFB15A"); WHITE = np.array([255, 255, 255], np.float32)
ISR = (35.0, 31.45)
ISR_CITIES = [(34.78, 32.08, 1.0), (35.21, 31.77, 0.85), (34.99, 32.79, 0.7), (34.79, 31.25, 0.5), (34.95, 29.56, 0.35), (35.30, 32.70, 0.35)]
CITIES = os.path.join(HERE, "data", "cities.json")
SS, SC = mt.SS, mt.SC
D2R = math.pi / 180


def cross2(a, b): return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
def smooth(e0, e1, x): u = np.clip((x - e0) / (e1 - e0), 0, 1); return u * u * (3 - 2 * u)
def ease_out_back(u, s=1.4): u = min(1.0, max(0.0, u)); return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2
def lerp(a, b, e): return a + (b - a) * e


# ------------------------------------------------------------------ geodata (map_tear's file, plus Antarctica for the globe)
def load_layers(empires=False):
    L = mt.load_geo("empires" if empires else "plain")
    g = json.load(open(mt.GEO)); ata = g["world"].get("ATA")
    out = []
    if ata:
        cs = ata["coordinates"] if ata["type"] == "MultiPolygon" else [ata["coordinates"]]
        for poly in cs:
            r = np.asarray(poly[0], np.float64)
            if len(r) >= 3: out.append((r, (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())))
    L["ata"] = out
    return L


def render_eq(L, lon0, lat1, dens, wpx, hpx):
    """equirectangular texture: BGRA uint8, A = ocean mask (for specular)"""
    c = np.empty((hpx, wpx, 3), np.uint8); c[:] = G_OCEAN.astype(np.uint8)
    M = M3(dens, 0, -lon0 * dens, 0, -dens, lat1 * dens); vb = (lon0 - 1, lat1 - hpx / dens - 1, lon0 + wpx / dens + 1, lat1 + 1)
    lay = c.copy(); step = 15 if dens < 60 else 5
    for lon in range(-180, 181, step):
        x = int(round((lon - lon0) * dens * SC)); cv2.line(lay, (x, -10), (x, (hpx + 10) * SC), SKY.tolist(), 1, cv2.LINE_AA, SS)
    for lat in range(-90, 91, step):
        y = int(round((lat1 - lat) * dens * SC)); cv2.line(lay, (-10, y), ((wpx + 10) * SC, y), SKY.tolist(), 1, cv2.LINE_AA, SS)
    cv2.addWeighted(lay, 0.13, c, 0.87, 0, c)
    lw = max(1.0, min(dens / 14, 3.0))
    mt.fill(c, M, L["land"] + L["ata"], G_LAND, 1.0, vb, outline=G_BORDER, lw=lw, oalpha=0.32)
    mt.fill(c, M, L["isr"], GOLD, 1.0, vb)
    mt.fill(c, M, L["terr"], G_TERR, 1.0, vb, outline=GREENL if dens > 22 else None, lw=max(2, min(dens / 70, 5)), oalpha=0.95)
    lm = np.zeros((hpx, wpx), np.uint8); mt.fill(lm, M, L["land"] + L["ata"], np.array([255.0]), 1.0, vb)
    return np.dstack([c, 255 - lm])


# ------------------------------------------------------------------ globe renderer
class Globe:
    BASE_W = 4096

    def __init__(s, L, night=False):
        s.L = L; s.night = night
        bw = s.BASE_W; s.base_d = bw / 360.0
        t = render_eq(L, -180, 90, s.base_d, bw, bw // 2)
        s.levels = [(t, s.base_d)]
        for k in (2, 4):
            s.levels.append((cv2.resize(t, (bw // k, bw // (2 * k)), interpolation=cv2.INTER_AREA), s.base_d / k))
        s.levels = s.levels[::-1]                        # coarse first
        s.lights = None
        if night: s.lights = s.make_lights(t[..., 3])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); s.xx, s.yy = xx, yy
        Lv = np.array([-0.48, 0.52, 0.71]); s.Lv = Lv / np.linalg.norm(Lv)
        Hv = s.Lv + np.array([0, 0, 1.0]); s.Hv = Hv / np.linalg.norm(Hv)
        s.dyn = None

    def make_lights(s, ocean):
        bw, bh = s.BASE_W, s.BASE_W // 2; d = s.base_d
        lt = np.zeros((bh, bw), np.float32); rng = np.random.default_rng(21)
        for lon, lat, pop in json.load(open(CITIES)):
            if 34.0 < lon < 36.0 and 29.3 < lat < 33.5: continue   # Israel's lights switch on last (event)
            I = float(np.clip((math.log10(max(pop, 1)) - 4.3) / 2.8, 0, 1))
            if I <= 0: continue
            x, y = (lon + 180) * d, (90 - lat) * d
            n = int(4 + 34 * I * I); sp = (0.25 + 1.3 * I) * d
            px = rng.normal(x, sp, n); py = rng.normal(y, sp * 0.8, n)
            for a, b in zip(px, py):
                cv2.circle(lt, (int(a * 4), int(b * 4)), 4, float(rng.uniform(0.25, 0.8)), -1, cv2.LINE_AA, 2)
            cv2.circle(lt, (int(x * 4), int(y * 4)), int(4 * (1 + 2.2 * I)), 1.0, -1, cv2.LINE_AA, 2)
        land = cv2.dilate((ocean < 128).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        lt *= land
        lt = cv2.GaussianBlur(lt, (0, 0), 0.9) + 0.55 * cv2.GaussianBlur(lt, (0, 0), 5)
        lt = np.clip(lt, 0, 1.2)
        out = [lt]
        for k in (2, 4): out.append(cv2.resize(lt, (bw // k, bh // k), interpolation=cv2.INTER_AREA))
        return out[::-1]

    # geometry: screen px -> (lat, dlon) under the scaled-angle orthographic projection
    def geometry(s, P, roi):
        x0, y0, x1, y1 = roi
        R, cx, cy, a, xf, roll = P["R"], P["cx"], P["cy"], max(1e-4, P["a"]), P["xf"], P["roll"] * D2R
        X = (s.xx[y0:y1, x0:x1] - cx) / (R * xf); Y = -(s.yy[y0:y1, x0:x1] - cy) / R
        cr, sr = math.cos(roll), math.sin(roll); Xr = X * cr + Y * sr; Yr = -X * sr + Y * cr
        x = Xr * a; y = Yr * a; rho2 = x * x + y * y
        z = np.sqrt(np.clip(1 - rho2, 0, 1)); pc = P["phi"] * D2R * a
        latp = np.arcsin(np.clip(y * math.cos(pc) + z * math.sin(pc), -1, 1))
        dlp = np.arctan2(x, z * math.cos(pc) - y * math.sin(pc))
        lat = latp / (D2R * a); dlon = dlp / (D2R * a)
        rho = np.sqrt(rho2); ppd = R * D2R
        al = np.clip((1 - rho) * R / a + 0.5, 0, 1)
        al *= np.clip((180 - np.abs(dlon)) * ppd * xf * np.cos(np.clip(latp / a, -1.5707, 1.5707)) + 0.5, 0, 1)
        al *= np.clip((90 - np.abs(lat)) * ppd + 0.5, 0, 1)
        return dict(x=x, y=y, z=z, rho=rho, lat=lat, dlon=dlon, al=al.astype(np.float32), a=a, R=R)

    def pick(s, nd, Pm, G):
        for tex, d in s.levels:
            if d >= nd * 0.95: return tex, d, -180.0, 90.0, True, (s.lights[[l[1] for l in s.levels].index(d)] if s.lights else None)
        if nd <= s.base_d * 1.25: return s.levels[-1][0], s.base_d, -180.0, 90.0, True, (s.lights[-1] if s.lights else None)
        # dynamic regional texture for deep zooms
        v = G["al"] > 0.5; sub = (slice(None, None, 16), slice(None, None, 16))
        la = G["lat"][sub][v[sub]]; dl = G["dlon"][sub][v[sub]]
        if la.size == 0: return s.levels[-1][0], s.base_d, -180.0, 90.0, True, None
        lam = Pm[0]; lo0, lo1 = lam + dl.min() - 0.6, lam + dl.max() + 0.6; la0, la1 = la.min() - 0.6, la.max() + 0.6
        if lo1 - lo0 > 300: return s.levels[-1][0], s.base_d, -180.0, 90.0, True, None
        dens = min(nd * 1.3, 3000.0 / max(lo1 - lo0, la1 - la0))
        wpx, hpx = int((lo1 - lo0) * dens) + 2, int((la1 - la0) * dens) + 2
        key = (round(lo0, 3), round(la1, 3), round(dens, 2))
        if s.dyn is None or s.dyn[0] != key: s.dyn = (key, render_eq(s.L, lo0, la1, dens, wpx, hpx))
        return s.dyn[1], dens, lo0, la1, False, None

    def render(s, P, bg):
        """P: lams (list, motion-blur samples of the central longitude), phi, roll, R, cx, cy, a, xf, sun (dlon, lat) or None"""
        a = max(1e-4, P["a"]); span = P["R"] / a * 1.8
        if a > 0.999 and span < 1400:
            roi = (max(0, int(P["cx"] - span)), max(0, int(P["cy"] - span)), min(W, int(P["cx"] + span)), min(H, int(P["cy"] + span)))
        else: roi = (0, 0, W, H)
        x0, y0, x1, y1 = roi; G = s.geometry(P, roi)
        nd = P["R"] * D2R
        tex, dens, lon0, lat1, wrap, lights = s.pick(nd, P["lams"], G)
        my = ((lat1 - G["lat"]) * dens).astype(np.float32)
        acc = None; lacc = None
        for lam in P["lams"]:
            lon = lam + G["dlon"]
            mx = (((lon - lon0) % 360.0) * dens if wrap else (lon - lon0) * dens).astype(np.float32)
            smp = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE).astype(np.float32)
            acc = smp if acc is None else acc + smp
            if lights is not None:
                l_ = cv2.remap(lights, mx * lights.shape[1] / tex.shape[1], my * lights.shape[0] / tex.shape[0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                lacc = l_ if lacc is None else lacc + l_
        acc /= len(P["lams"]); col = acc[..., :3]; ocean = acc[..., 3] / 255
        if lacc is not None: lacc /= len(P["lams"])
        x, y, z = G["x"], G["y"], G["z"]
        sh = float(np.clip((a - 0.04) / 0.96, 0, 1)) ** 0.9
        Lv, Hv = s.Lv, s.Hv
        if P.get("sun") is not None and s.night:
            sdl, slat = P["sun"]; latr = G["lat"] * D2R
            cosd = np.sin(latr) * math.sin(slat * D2R) + np.cos(latr) * math.cos(slat * D2R) * np.cos((G["dlon"] - sdl) * D2R)
            day = smooth(-0.10, 0.14, cosd)
            dshade = 0.42 + 0.75 * np.clip(cosd + 0.1, 0, 1) ** 0.7
            ndl = x * Lv[0] + y * Lv[1] + z * Lv[2]; soft = 0.82 + 0.25 * np.clip(ndl, 0, 1)
            nightc = col * 0.075 + np.array([30, 14, 6], np.float32)
            out = col * (dshade * soft * day)[..., None] + nightc * (1 - day)[..., None]
            if lacc is not None: out += (lacc * (1 - day) ** 1.4)[..., None] * LIGHT_COL * 1.15
            term = np.exp(-((cosd + 0.01) / 0.075) ** 2) * 0.30
            out += term[..., None] * WARM
            spec = np.clip(x * Hv[0] + y * Hv[1] + z * Hv[2], 0, 1) ** 60 * 0.28 * sh * ocean * day
            out += spec[..., None] * 255
            rimw = 0.35 + 0.65 * day
        else:
            ndl = x * Lv[0] + y * Lv[1] + z * Lv[2]
            diff = 0.30 + 0.86 * np.clip(ndl * 0.8 + 0.3, 0, 1) ** 1.15
            shade = 1 + (diff * (0.86 + 0.14 * z ** 0.5) - 1) * sh
            out = col * shade[..., None]
            spec = np.clip(x * Hv[0] + y * Hv[1] + z * Hv[2], 0, 1) ** 55 * 0.34 * sh * ocean
            out += spec[..., None] * 255
            rimw = 1.0
        if P.get("wave") or P.get("region"):
            lon_t = P["lams"][len(P["lams"]) // 2] + G["dlon"]; latr = G["lat"] * D2R
            cang = np.sin(latr) * math.sin(ISR[1] * D2R) + np.cos(latr) * math.cos(ISR[1] * D2R) * np.cos((lon_t - ISR[0]) * D2R)
            ang = np.degrees(np.arccos(np.clip(cang, -1, 1)))
            for r_, w_, amp in (P.get("wave") or []):
                band = np.exp(-((ang - r_) / w_) ** 2) * amp; inner = np.clip(1 - ang / max(r_, 1e-3), 0, 1) * amp * 0.22
                out = out * (1 + band * 0.25)[..., None] + (band * 0.75 + inner)[..., None] * GOLD
            if P.get("region"):
                rg = smooth(18, 2, ang) * P["region"]
                out = out * (1 - 0.25 * rg)[..., None] + (rg * 0.45)[..., None] * GOLD
        fres = np.clip(1 - z, 0, 1) ** 3 * sh * 0.8 * rimw
        out += fres[..., None] * SKY
        if P.get("glow"):   # target pulse: brighten the gold of Israel
            gx, gy, ga = P["glow"]
            d2 = ((s.xx[y0:y1, x0:x1] - gx) ** 2 + (s.yy[y0:y1, x0:x1] - gy) ** 2)
            out += (np.exp(-d2 / (2 * 26 ** 2)) * 0.55 * ga)[..., None] * GOLD
        al = G["al"]
        reg = bg[y0:y1, x0:x1].copy()
        if sh > 0.01:   # atmosphere outside the limb
            dout = np.clip((G["rho"] - 1) * G["R"] / a, 0, None)
            glow = (np.exp(-dout / 13) * 0.55 + np.exp(-dout / 75) * 0.2) * (G["rho"] > 0.98) * sh
            reg = reg + (glow * (1 - al))[..., None] * SKY * (0.6 + 0.4 * (rimw if np.isscalar(rimw) else 1))
            # contact shadow under the globe (soft, very subtle)
        reg = reg * (1 - al[..., None]) + out * al[..., None]
        fr = bg.copy(); fr[y0:y1, x0:x1] = reg
        return fr

    @staticmethod
    def project(P, lon, lat):
        a = max(1e-4, P["a"]); lam = P["lams"][len(P["lams"]) // 2]
        dl = (lon - lam) * D2R * a; ph = lat * D2R * a; pc = P["phi"] * D2R * a
        x = math.cos(ph) * math.sin(dl); y = math.cos(pc) * math.sin(ph) - math.sin(pc) * math.cos(ph) * math.cos(dl)
        vis = math.sin(pc) * math.sin(ph) + math.cos(pc) * math.cos(ph) * math.cos(dl) > 0
        Xr, Yr = x / a, y / a; r = P["roll"] * D2R; cr, sr = math.cos(r), math.sin(r)
        X = Xr * cr - Yr * sr; Y = Xr * sr + Yr * cr
        return P["cx"] + X * P["R"] * P["xf"], P["cy"] - Y * P["R"], vis


# ------------------------------------------------------------------ overlays
def ring(fr, x, y, r, a, col=GOLD, th=4):
    if a <= 0.01 or r <= 1: return fr
    lay = np.zeros((H, W), np.uint8); cv2.circle(lay, (int(x * SC), int(y * SC)), int(r * SC), 255, th, cv2.LINE_AA, SS)
    m = cv2.GaussianBlur(lay.astype(np.float32) / 255, (0, 0), 1.2) * a + cv2.GaussianBlur(lay.astype(np.float32) / 255, (0, 0), 9) * a * 0.7
    return fr * (1 - np.clip(m, 0, 1)[..., None] * 0.2) + m[..., None] * col


def draw_pin(fr, x, y, s, sq, a, shadow_a):
    """gold map pin with its tip at (x, y); s scale, sq squash (1 = none)"""
    if a <= 0.01: return fr
    sx, sy = s * (1 + (1 - sq) * 0.9), s * sq
    r = 34; th = np.linspace(math.radians(-42), math.radians(222), 60)
    head = np.stack([r * np.cos(th), -86 + r * np.sin(-th)], 1)
    pts = np.vstack([[[0, 0]], head[::-1] * [1, 1]])
    pts = np.vstack([[[0, 0]], np.stack([r * np.cos(th), -86 - r * np.sin(th)], 1)])
    P = pts * [sx, sy] + [x, y]
    # shadow on the surface
    if shadow_a > 0.01:
        sl = np.zeros((H, W), np.uint8); cv2.ellipse(sl, (int(x * SC), int((y + 3) * SC)), (int(26 * s * SC), int(8 * s * SC)), 0, 0, 360, 255, -1, cv2.LINE_AA, SS)
        m = cv2.GaussianBlur(sl.astype(np.float32) / 255, (0, 0), 5) * shadow_a * 0.55
        fr = fr * (1 - m[..., None])
    lay = np.zeros((H, W), np.uint8); cv2.fillPoly(lay, [np.round(P * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
    m = lay.astype(np.float32)[..., None] / 255 * a
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    body = GOLD * 1.0
    fr = fr * (1 - m) + body * m
    # soft highlight and the white dot
    hl = np.zeros((H, W), np.uint8); cv2.circle(hl, (int((x - 12 * sx) * SC), int((y - 98 * sy) * SC)), int(10 * s * SC), 255, -1, cv2.LINE_AA, SS)
    hm = cv2.GaussianBlur(hl.astype(np.float32) / 255, (0, 0), 6)[..., None] * 0.45 * a
    fr = fr * (1 - hm) + WHITE * hm
    dl = np.zeros((H, W), np.uint8); cv2.circle(dl, (int(x * SC), int((y - 86 * sy) * SC)), int(12.5 * s * SC), 255, -1, cv2.LINE_AA, SS)
    dm = dl.astype(np.float32)[..., None] / 255 * a
    return fr * (1 - dm) + NIGHT * dm


# ------------------------------------------------------------------ sound
class Mix:
    def __init__(s, T, seed=3):
        s.T = T; s.n = int(round(T * SR)); s.out = np.zeros((s.n, 2)); s.rng = np.random.default_rng(seed)

    def put(s, x, at, g, pan=0.0):
        i = int(max(0, at) * SR); x = np.asarray(x)[: max(0, s.n - i)]
        s.out[i:i + x.size, 0] += x * g * (1 - max(0, pan)); s.out[i:i + x.size, 1] += x * g * (1 + min(0, pan))

    @staticmethod
    def bp(x, lo, hi):
        X = np.fft.rfft(x); f = np.fft.rfftfreq(x.size, 1 / SR)
        X *= np.clip((f - lo * 0.7) / (lo * 0.3 + 1), 0, 1) * np.clip((hi * 1.3 - f) / (hi * 0.3), 0, 1); return np.fft.irfft(X, x.size)

    @staticmethod
    def nz(x): return x / (np.abs(x).max() + 1e-9)

    def sweep(s, prof, lo=180, hi=6500, p=1.3):
        """noise through a 2-pole lowpass whose cutoff and level follow prof (0..1 per sample)"""
        m = prof.size; wn = s.rng.normal(0, 1, m); out = np.zeros(m); l1 = l2 = 0.0
        al = 1 - np.exp(-2 * np.pi * (lo + (hi - lo) * np.clip(prof, 0, 1) ** p) / SR)
        for i in range(m):
            l1 += al[i] * (wn[i] - l1); l2 += al[i] * (l1 - l2); out[i] = l2
        out = out - np.convolve(out, np.ones(48) / 48, "same") * 0.5
        out *= np.clip(prof, 0, 1) ** 0.9
        return s.nz(out)

    def whoosh_prof(s, at, prof_fn, dur, g, pan_sweep=0.0):
        tt = np.arange(int(dur * SR)) / SR; pr = np.array([prof_fn(t) for t in tt[::48]]); pr = np.interp(tt, tt[::48], pr)
        w = s.sweep(pr); s.put(w, at, g, 0)
        if pan_sweep: s.put(np.roll(w, 240), at, g * 0.25, pan_sweep)

    def zoom_whoosh(s, z0, z1, g=0.55):
        m = int((z1 - z0) * SR); u = np.arange(m) / max(1, m)
        pr = u ** 2.2 * np.clip((1 - u) / 0.04, 0, 1); w = s.sweep(pr, 250, 5800, 1.0); s.put(w, z0, g); s.put(np.roll(w, 300), z0, g * 0.2, 0.6)

    def shimmer(s, at, g=0.26, base=880):
        m = int(0.7 * SR); tt = np.arange(m) / SR
        x = sum(np.sin(2 * np.pi * base * r * tt + i) * a for i, (r, a) in enumerate(((1, 0.5), (1.4983, 0.35), (2, 0.22), (2.9966, 0.12))))
        x *= np.clip(tt / 0.015, 0, 1) * np.exp(-tt / 0.2); s.put(s.nz(x), at, g)

    def thud(s, at, g=0.8):
        m = int(0.35 * SR); tt = np.arange(m) / SR
        f = 48 + 60 * np.exp(-tt / 0.03); ph = 2 * np.pi * np.cumsum(f) / SR
        body = np.sin(ph) * np.exp(-tt / 0.09)
        click = s.bp(s.rng.normal(0, 1, m), 900, 4000) * np.exp(-tt / 0.006) * 0.5
        s.put(s.nz(body + click), at, g)

    def click(s, at, g=0.5, lo=1800, hi=7000, body=650, dec=0.008):
        m = int(0.08 * SR); tt = np.arange(m) / SR
        x = s.bp(s.rng.normal(0, 1, m), lo, hi) * np.exp(-tt / dec) + 0.6 * np.sin(2 * np.pi * body * tt) * np.exp(-tt / 0.018)
        s.put(s.nz(x), at, g, float(s.rng.uniform(-0.3, 0.3)))

    def crackle(s, at, dur, dens, g=0.5, lo=1200, hi=9000):
        m = int(dur * SR); tt = np.arange(m) / SR; x = np.zeros(m); k = 0
        while k < m:
            u = k / m; L = int(s.rng.uniform(0.002, 0.012) * SR)
            if k + L < m: x[k:k + L] += s.rng.uniform(0.3, 1.0) * np.hanning(L) * s.rng.normal(0, 1, L)
            k += int(s.rng.exponential(1.0 / max(1, dens(u))) * SR) + 1
        y = s.bp(x, lo, hi) + 0.45 * s.bp(x, 250, 1400)
        env = np.clip(tt / 0.02, 0, 1) * np.clip((dur - tt) / 0.05, 0, 1); s.put(s.nz(y * env), at, g)

    def fwip(s, at, dur=0.16, g=0.4, lo=500, hi=4500):
        m = int(dur * SR); u = np.arange(m) / m; pr = np.sin(np.pi * u) ** 1.5
        s.put(s.sweep(pr, lo, hi, 1.0), at, g, float(s.rng.uniform(-0.25, 0.25)))

    def fwump(s, at, g=0.6):
        m = int(0.28 * SR); tt = np.arange(m) / SR
        x = s.bp(s.rng.normal(0, 1, m), 120, 1100) * np.clip(tt / 0.012, 0, 1) * np.exp(-tt / 0.07)
        s.put(s.nz(x), at, g)

    def pad(s, at, dur, g=0.2, notes=(110.0, 164.81, 220.0, 277.18, 329.63)):
        m = int(dur * SR); tt = np.arange(m) / SR
        x = sum(np.sin(2 * np.pi * f * tt + i) * (0.9 ** i) + 0.3 * np.sin(2 * np.pi * f * 2.003 * tt) * (0.8 ** i) for i, f in enumerate(notes))
        env = np.clip(tt / (dur * 0.7), 0, 1) ** 2 * np.clip((dur - tt) / 0.08, 0, 1); s.put(s.nz(x) * env, at, g)

    def done(s):
        o = s.out / max(1e-9, np.abs(s.out).max()) * 10 ** (-3 / 20); return o


# ------------------------------------------------------------------ globe styles
R0, GCY = 505.0, 900.0


def lam_speed(fn, t, dt=1e-3): return (fn(t + dt) - fn(t - dt)) / (2 * dt)


def globe_style(style, T, F, glb, reel_f0, mix, title="ISRAEL"):
    """each globe style ends its spin with an EVENT on Israel (Noam, 3 Nov 07:45), then unwraps / zooms into frame 0"""
    lamI, phiI = ISR
    z0 = 0.625 * T; zend = T - 2.0 / FPS
    if style == "globe_unpeel": z0 = 0.50 * T
    R_end = F.k / D2R; zoom_in = R_end > 2 * R0
    roll0 = -16.0; sc = T / 2.4
    shake = []            # (time, px) screen shake on impacts
    pin = None            # (t_drop0, t_land)
    title_t = None
    region = lambda t: 0.0
    wave = lambda t: None
    isr_lights = lambda t: 0.0
    sun = None

    if style == "globe_spin_fast":
        # EVENT (a): a gold shockwave ripples out from Israel across the region, along the curve of the globe
        tl = 1.12 * sc
        def lam(t):
            u = min(1, max(0, t / tl)); return lamI + 560 * (1 - u) ** 3.2 if t < tl else lamI
        def wave(t):
            out = []
            for d, amp in ((0.0, 1.0), (0.13, 0.65), (0.26, 0.4)):
                u = (t - tl - d) / 0.62
                if 0 < u < 1: out.append((2 + 58 * ease_out(u), 2.2 + 5 * u, amp * (1 - u) ** 1.3))
            return out or None
        region = lambda t: seg(t, tl - 0.04, tl + 0.08) * (1 - seg(t, tl + 0.5, z0 + 0.3)) * 0.8
        glow = lambda t: seg(t, tl - 0.08, tl + 0.02) * (1 - seg(t, tl + 0.35, z0 + 0.3)) * 1.3
        shake = [(tl, 7)]
        mix.whoosh_prof(0, lambda t: min(1, abs(lam_speed(lam, t)) / 900), tl + 0.05, 0.7, 0.4)
        mix.thud(tl - 0.005, 0.85); mix.fwump(tl, 0.45); mix.shimmer(tl + 0.02, 0.3, 880); mix.shimmer(tl + 0.15, 0.16, 1320)
        phi = lambda t: lerp(12, phiI, ease_out(min(1, t / tl))); shutter = 0.6
        def extra(t, P, fr): return fr
    elif style == "globe_spin_reverse":
        # EVENT (b): the region lights up gold and the reel's title rises out of the globe
        t1, t2 = 0.80 * sc, 1.12 * sc; over = 42.0
        def lam(t):
            if t < t1: u = t / t1; return lamI - 320 + (320 + over) * (1 - (1 - u) ** 2.6)
            if t < t2: u = (t - t1) / (t2 - t1); return lamI + over * (1 - ease_out_back(u, 1.9))
            return lamI
        region = lambda t: seg(t, t2 - 0.1, t2 + 0.1) * (1 - seg(t, z0 + 0.1, z0 + 0.5))
        glow = lambda t: seg(t, t2 - 0.1, t2 + 0.05) * (1 - seg(t, z0, z0 + 0.4)) * 1.2
        title_t = (t2 - 0.02, t2 + 0.42)
        mix.whoosh_prof(0, lambda t: min(1, abs(lam_speed(lam, t)) / 700), t2, 0.6, 0.4)
        mix.fwip(t1 + 0.02, 0.18, 0.45, 600, 6000); mix.click(t2 - 0.03, 0.55)
        mix.pad(t2 - 0.25, 0.5, 0.22, (220.0, 277.18, 329.63, 440.0)); mix.shimmer(t2, 0.3, 990); mix.fwip(t2 + 0.02, 0.4, 0.3, 300, 3000)
        phi = lambda t: lerp(18, phiI, ease_io(min(1, t / t2))); shutter = 0.6
        def extra(t, P, fr): return fr
    elif style in ("globe_spin_whip", "globe_unpeel"):
        # whip EVENT: pin slams in with a thud and a shockwave. unpeel EVENT (c): pin slams in, then the map unpeels flat
        if style == "globe_spin_whip":
            ts = 0.78 * sc; pin = (0.92 * sc, 1.10 * sc); brake = 0.16; spin = 1900
        else:
            ts = 0.62 * sc; pin = (0.66 * sc, 0.80 * sc); brake = 0.18; spin = 1500
        lam_fast = lambda t: lamI + 720 - spin * t * (0.6 + 0.4 * min(1, t / 0.2))
        def lam(t):
            if t < ts - brake: return lam_fast(t)
            if t < ts: u = (t - (ts - brake)) / brake; a0 = lam_fast(ts - brake); return a0 + (lamI - 9 - a0) * (1 - (1 - u) ** 2.2)
            u = (t - ts) / 0.3; return lamI - 9 * math.exp(-u * 5) * math.cos(u * 9) if u < 1 else lamI
        def wave(t):
            u = (t - pin[1]) / 0.5
            return [(1 + 26 * ease_out(u), 1.6 + 2.5 * u, 0.9 * (1 - u) ** 1.4)] if 0 < u < 1 else None
        glow = lambda t: seg(t, pin[1] - 0.03, pin[1] + 0.04) * (1 - seg(t, pin[1] + 0.3, z0 + 0.3))
        shake = [(pin[1], 10)]
        mix.whoosh_prof(0, lambda t: min(1, abs(lam_speed(lam, t)) / 1200) ** 0.8, ts + 0.1, 0.85, 0.5)
        mix.crackle(ts - 0.04, 0.05, lambda u: 900, 0.5, 2500, 11000); mix.click(ts - 0.01, 0.4, 2500, 9000, 300, 0.004)
        m = int((pin[1] - pin[0]) * SR); tt = np.arange(m) / SR; f = 2400 - 1500 * (tt / tt[-1]) ** 1.5
        wh = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.clip(tt / 0.04, 0, 1) * (0.25 + 0.75 * tt / tt[-1]); mix.put(wh / np.abs(wh).max(), pin[0], 0.08)
        mix.thud(pin[1], 1.0); mix.shimmer(pin[1] + 0.02, 0.18, 660)
        if style == "globe_unpeel":   # paper peel: crackle + long swish through the unwrap
            mix.crackle(z0 + 0.05, 0.55, lambda u: 220 * math.sin(math.pi * u) + 30, 0.32, 1500, 9000); mix.fwip(z0, 0.75, 0.4, 300, 3500)
        phi = lambda t: phiI if t > ts else lerp(20, phiI, min(1, t / ts)); shutter = 1.0 if style == "globe_spin_whip" else 0.8
        def extra(t, P, fr): return fr
    elif style == "globe_spin_night":
        # EVENT (d): the globe stops on the night side, Israel's lights switch on last and brightest, then dawn sweeps in
        tl = 0.98 * sc; ton = 1.06 * sc; tsun0, tsun1 = 1.22 * sc, 1.75 * sc
        def lam(t):
            u = min(1, max(0, t / tl)); return lamI + 300 * (1 - u) ** 2.6 if t < tl else lamI
        def sun(t):
            e = ease_io((t - tsun0) / (tsun1 - tsun0)); return (lerp(158.0, -38.0, e), 22.0)
        def isr_lights(t):
            if t < ton: return 0.0
            w = t - ton; flick = (0.35 if w < 0.05 else (0.1 if w < 0.08 else 1.0))
            return flick * min(1.0, 0.4 + w / 0.12) * (1 + 0.6 * math.exp(-w / 0.12)) * (1 - seg(t, tsun1 - 0.15, z0 + 0.6))
        glow = lambda t: 0.0
        mix.whoosh_prof(0, lambda t: min(1, abs(lam_speed(lam, t)) / 700), tl, 0.4, 0.3)
        mix.click(ton, 0.55, 1500, 6000, 120, 0.01); mix.click(ton + 0.08, 0.4, 1500, 6000, 120, 0.01)
        mix.shimmer(ton + 0.09, 0.32, 880); mix.pad(ton - 0.1, 0.9, 0.36, (110.0, 164.81, 220.0, 329.63, 440.0)); mix.shimmer(tsun0 + 0.2, 0.18, 1320)
        phi = lambda t: lerp(14, phiI, ease_out(min(1, t / tl))); shutter = 0.6
        def extra(t, P, fr):
            a = isr_lights(t)
            if a <= 0.01: return fr
            lay = np.zeros((H, W), np.float32)
            for lon, lat, wgt in ISR_CITIES:
                x, y, vis = Globe.project(P, lon, lat)
                if not vis or not (-200 < x < W + 200 and -200 < y < H + 200): continue
                cv2.circle(lay, (int(x * 4), int(y * 4)), int(4 * (2.2 + 2.5 * wgt)), wgt, -1, cv2.LINE_AA, 2)
            core = cv2.GaussianBlur(lay, (0, 0), 1.5); bloom = cv2.GaussianBlur(lay, (0, 0), 14) * 9 + cv2.GaussianBlur(lay, (0, 0), 50) * 30 + cv2.GaussianBlur(lay, (0, 0), 120) * 60
            return fr + (np.clip(core * 2.2, 0, 1.6) * a)[..., None] * WHITE * 0.9 + (np.clip(bloom, 0, 1.4) * a)[..., None] * LIGHT_COL * 0.7
    mix.zoom_whoosh(z0, T, 0.5 if style != "globe_unpeel" else 0.35)

    def params(t):
        tc = min(t, z0)
        P = dict(phi=phi(tc), roll=roll0 * (1 - 0.35 * seg(t, 0, z0)), R=R0 * (1 + 0.03 * seg(t, 0, z0)), cx=540.0, cy=GCY, a=1.0, xf=1.0)
        lc = lam(tc); P["lams"] = [lc]
        if t < z0:
            v = abs(lam_speed(lam, t)) / FPS * shutter * R0 * D2R      # px of travel at the centre within the shutter
            n = int(np.clip(v / (2.5 if shutter > 0.9 else 4.0), 1, 30 if shutter > 0.9 else 16))
            if n > 1: P["lams"] = [lam(t + (i / (n - 1) - 0.5) * shutter / FPS) for i in range(n)]
        if sun: P["sun"] = sun(t)
        if t >= z0:   # unwrap / zoom into the target view
            u = min(1, (t - z0) / (zend - z0))
            if style == "globe_unpeel":   # unpeel first (globe to flat), then travel to the target scale
                ea = ease_io(min(1, u / 0.75)); e = ease_in(u) if zoom_in else ease_io(u)
            else:
                e = ease_in(u) if zoom_in else ease_io(u); ea = e
            P["a"] = max(1e-4, 1 - ea); P["xf"] = lerp(1, F.cosf, ea)
            P["R"] = math.exp(lerp(math.log(P["R"]), math.log(R_end), e))
            P["cx"], P["cy"] = lerp(P["cx"], F.sT[0], e), lerp(P["cy"], F.sT[1], e)
            P["lams"] = [lerp(lamI, F.T[0], e)]; P["phi"] = lerp(phiI, F.T[1], e); P["roll"] = lerp(P["roll"], 0, ea)
        g = glow(t)
        if g > 0.01: P["glow"] = (P["cx"], P["cy"], g)
        r_ = region(t)
        if r_ > 0.01: P["region"] = r_
        wv = wave(t)
        if wv: P["wave"] = wv
        return P

    bg = mt.background()
    timg = text_rgba(title) if title_t else None

    def frame(t):
        P = params(t); fr = glb.render(P, bg); fr = extra(t, P, fr)
        if pin and t >= pin[0] - 0.02:
            px, py, vis = Globe.project(P, lamI, phiI)
            u = (t - pin[0]) / (pin[1] - pin[0]); fade = 1 - seg(t, z0 + (0.35 if style == "globe_unpeel" else 0.15), z0 + (0.7 if style == "globe_unpeel" else 0.45))
            if u < 1: drop = (1 - u * u) * 620; sq = 1.0; sa = u
            else:
                w = t - pin[1]; drop = -abs(math.sin(w * 22) * math.exp(-w * 16)) * 26; sq = 1 - 0.25 * math.exp(-w * 18) * abs(math.cos(w * 20)); sa = 1
            s_ = 1.0 * (1 + 0.6 * seg(t, z0, T) if zoom_in else 1)
            fr = draw_pin(fr, px, py - drop, s_, sq, min(1, (t - pin[0] + 0.05) / 0.06) * fade, sa * fade)
            if t > pin[1]:
                for d in (0.0, 0.1):
                    uu = (t - pin[1] - d) / 0.5
                    if 0 < uu < 1: fr = ring(fr, px, py, 10 + 120 * ease_out(uu), (1 - uu) ** 1.5 * 0.9 * fade)
        if timg is not None and t >= title_t[0]:
            u = (t - title_t[0]) / (title_t[1] - title_t[0]); e = ease_out_back(min(1, u), 1.2)
            fade = min(1, u * 2.5) * (1 - seg(t, z0 + 0.05, z0 + 0.4))
            yc = lerp(P["cy"] + 10, GCY - 330, e) if reel_f0 is None else lerp(P["cy"] + 10, 940, e)
            fr = paste_rgba(fr, timg, 540, yc, lerp(0.35, 1.0, ease_out(min(1, u))), fade)
        for ts_, amp in shake:
            w = t - ts_
            if 0 <= w < 0.3:
                d = amp * math.exp(-w * 14) * math.sin(w * 70)
                fr = np.roll(fr, int(round(d)), 0); fr = np.roll(fr, int(round(d * 0.5)), 1)
        if t >= z0 and zoom_in:
            P2 = params(t + 1.0 / FPS); kr = P2["R"] / P["R"]
            fr = mt.zoom_blur(fr, (P["cx"], P["cy"]), min(0.05, (kr - 1) * 0.6) * (1 - seg(t, zend - 0.1, zend)))
        if reel_f0 is not None:
            a = seg(t, T - 0.33, T - 1.0 / FPS)
            if a > 0: fr = fr * (1 - a) + reel_f0 * a
        return np.clip(fr, 0, 255)
    return frame


FONT_EN = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"


def text_rgba(title, size=None, maxw=960):
    from PIL import Image, ImageDraw, ImageFont
    size = size or (170 if len(title) <= 10 else 104)
    fnt = ImageFont.truetype(FONT_EN, size); words = title.split(); lines = []; cur = ""
    for wd in words:
        t_ = (cur + " " + wd).strip()
        if fnt.getlength(t_) > maxw and cur: lines.append(cur); cur = wd
        else: cur = t_
    lines.append(cur); lh = int(size * 1.08)
    im = Image.new("RGBA", (maxw + 80, lh * len(lines) + 60), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text(((maxw + 80) / 2, 30 + i * lh + lh / 2), ln, font=fnt, fill=(255, 255, 255, 255), anchor="mm")
    a = np.asarray(im).astype(np.float32) / 255
    return a


def paste_rgba(fr, rgba, cx, cy, scale, alpha):
    if alpha <= 0.01: return fr
    im = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = im.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya: return fr
    sub = im[ya - y0:yb - y0, xa - x0:xb - x0]; a = sub[..., 3:4] * alpha
    sh = cv2.GaussianBlur(sub[..., 3], (0, 0), 14)[..., None] * alpha * 0.55
    reg = fr[ya:yb, xa:xb]; reg = reg * (1 - sh); col = sub[..., 2::-1] * 255   # RGBA -> BGR
    fr[ya:yb, xa:xb] = reg * (1 - a) + col * a
    return fr


# ------------------------------------------------------------------ paper styles (map_tear's sheet, rendered as a texture)
def sheet_tex(sh, layers):
    """the whole sheet (printed map, margin, frame lines, grain) at 2x sheet px; plus the paper back"""
    w2, h2 = int(sh.w * 2), int(sh.h * 2)
    Mg = M3(2, 0, 0, 0, 2, 0) @ np.linalg.inv(sh.G)
    c = np.zeros((h2, w2, 3), np.uint8); c[:] = mt.OCEAN.astype(np.uint8); lay = c.copy()
    for lon in range(-180, 181, 30):
        p = apply(Mg, np.array([[lon, sh.LAT0], [lon, sh.LAT1]], float)); cv2.line(lay, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), NAVY.tolist(), 2, cv2.LINE_AA, SS)
    for lat in range(-60, 85, 30):
        p = apply(Mg, np.array([[-180, lat], [180, lat]], float)); cv2.line(lay, tuple(np.round(p[0] * SC).astype(int)), tuple(np.round(p[1] * SC).astype(int)), NAVY.tolist(), 2, cv2.LINE_AA, SS)
    cv2.addWeighted(lay, 0.16, c, 0.84, 0, c)
    mt.fill(c, Mg, layers["land"], mt.LAND, 1.0, None, outline=NAVY, lw=2, oalpha=0.38)
    if layers["empires"]: mt.fill(c, Mg, layers["empires"], RED, 0.86, None)
    mt.fill(c, Mg, layers["isr"], GOLD, 1.0, None, outline=NAVY, lw=1, oalpha=0.6)
    mt.fill(c, Mg, layers["terr"], mt.TERR, 1.0, None)
    img = c.astype(np.float32)
    pm = np.zeros((h2, w2), np.uint8); pr2 = np.array([[sh.MX, sh.MY], [sh.w - sh.MX, sh.MY], [sh.w - sh.MX, sh.h - sh.MY], [sh.MX, sh.h - sh.MY]]) * 2
    cv2.fillPoly(pm, [np.round(pr2 * SC).astype(np.int32)], 255, cv2.LINE_AA, SS); pm = pm.astype(np.float32)[..., None] / 255
    img = img * pm + mt.PAPER * (1 - pm)
    for d, a in ((0, 0.55), (7, 0.3)):
        q = pr2 + np.array([[-d, -d], [d, -d], [d, d], [-d, d]]) * 2
        cv2.polylines(img, [np.round(q * SC).astype(np.int32)], True, (NAVY * a + mt.PAPER * (1 - a)).tolist(), 2, cv2.LINE_AA, SS)
    g = cv2.resize(sh.grain, (w2, h2))
    img *= g[..., None]
    back = (mt.PAPER * 0.94)[None, None, :] * np.ones((h2, w2, 1), np.float32)
    back *= (1 + (g - 1) * 1.3)[..., None]
    return np.clip(img, 0, 255).astype(np.float32), np.clip(back, 0, 255).astype(np.float32)


def composite_layer(bg, img, al, off=(9, 16), blur=13, sa=0.6):
    sh = cv2.GaussianBlur(al, (0, 0), blur); sh = np.roll(np.roll(sh, int(off[1]), 0), int(off[0]), 1) * sa
    out = bg * (1 - sh[..., None]); return out * (1 - al[..., None]) + img * al[..., None]


def rot3(axis, ang):
    c, s = math.cos(ang), math.sin(ang); x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def hinge(p0, axis, ang):
    """4x4: rotate about the line through p0 along axis"""
    Rm = rot3(axis, ang); M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = np.array(p0) - Rm @ np.array(p0); return M


def paper_style(style, T, F, reel_f0, mix, empires=True, rot_deg=-3.0):
    sh = mt.Sheet(F.cosf, width=980); layers = mt.load_geo("empires" if empires else "plain")
    tex, back = sheet_tex(sh, layers); bg = mt.background()
    z0 = 0.625 * T; rot0 = math.radians(rot_deg); ctr = np.array([540.0, 930.0])
    Rr = lambda a: np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    def sheet_M(rot, center):
        Rm = Rr(rot); b = center - Rm @ np.array([sh.w / 2, sh.h / 2]); return M3(Rm[0, 0], Rm[0, 1], b[0], Rm[1, 0], Rm[1, 1], b[1])
    M0 = sheet_M(rot0, ctr); sT0 = apply(M0, sh.geo2px(*F.T)[None])[0]; MF = F.M()
    w, h = sh.w, sh.h; f = 2400.0; Rm0 = Rr(rot0)

    def to_screen(P3, S=1.0, pitch=0.0, c3=(0, 0)):
        """sheet-local 3D (u, v, z up) -> screen px"""
        q = (P3 - np.array([c3[0], c3[1], 0.0])) * S
        if pitch:
            cp, sp = math.cos(pitch), math.sin(pitch); y, z = q[:, 1].copy(), q[:, 2].copy(); q[:, 1] = y * cp - z * sp; q[:, 2] = y * sp + z * cp
        k = f / (f - q[:, 2]); xy = (q[:, :2] * k[:, None]) @ Rm0.T + ctr
        return xy, q

    Lv = np.array([-0.35, -0.5, 0.8]); Lv /= np.linalg.norm(Lv); Lz = Lv[2]

    # ---------------- fold: 4 columns x 2 rows, accordion columns, the top row (Israel) swings up last
    if style == "map_fold":
        cw, hh = w / 4, h / 2; E = z0
        ts = dict(a2=(-0.06, 0.36), a3=(0.30, 0.64), a0=(0.58, 0.92), ph=(0.90, 1.38))
        ts = {k: (a * T / 2.4, b * T / 2.4) for k, (a, b) in ts.items()}
        FOLD = math.radians(176)
        def angles(t):
            g = lambda k: 1 - ease_io((t - ts[k][0]) / (ts[k][1] - ts[k][0]))
            return dict(a2=-FOLD * g("a2"), a3=FOLD * g("a3"), a0=-FOLD * g("a0"), ph=FOLD * g("ph"))
        for k, (a, b) in ts.items():
            mix.fwip(a + 0.04, (b - a) * 0.8, 0.42 if k != "ph" else 0.55, 450, 4200)
            mix.click(b - 0.01, 0.32 if k != "ph" else 0.45, 1200, 5000, 220, 0.012)
        mix.fwump(ts["ph"][1] - 0.02, 0.55); mix.shimmer(ts["ph"][1], 0.22, 880)
        panels = [(r, c) for r in range(2) for c in range(4)]
        def panel_M(r, c, A):
            Mb = {1: np.eye(4)}
            Mb[0] = hinge((cw, 0, 0), (0, 1, 0), A["a0"])
            Mb[2] = hinge((2 * cw, 0, 0), (0, 1, 0), A["a2"]); Mb[3] = Mb[2] @ hinge((3 * cw, 0, 0), (0, 1, 0), A["a3"])
            M = Mb[c]
            if r == 0: M = M @ hinge((0, hh, 0), (1, 0, 0), A["ph"])
            return M
        def effect(t):
            A = angles(t); quads = []
            for r, c in panels:
                M = panel_M(r, c, A)
                loc = np.array([[c * cw, r * hh, 0, 1], [(c + 1) * cw, r * hh, 0, 1], [(c + 1) * cw, (r + 1) * hh, 0, 1], [c * cw, (r + 1) * hh, 0, 1]], float)
                quads.append(((r, c), (M @ loc.T).T[:, :3]))
            allp = np.vstack([q for _, q in quads]); bx0, by0 = allp[:, 0].min(), allp[:, 1].min(); bx1, by1 = allp[:, 0].max(), allp[:, 1].max()
            p = min(1, t / ts["ph"][1]); S = 1 + 0.45 * (1 - ease_io(p)); pitch = math.radians(16) * (1 - ease_io(p))
            c3 = ((bx0 + bx1) / 2, (by0 + by1) / 2)
            img = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
            items = []
            for (r, c), q3 in quads:
                xy, qq = to_screen(q3, S, pitch, c3)
                n = np.cross(qq[1] - qq[0], qq[3] - qq[0]); n /= (np.linalg.norm(n) + 1e-9)
                items.append((qq[:, 2].mean(), r, c, xy, n))
            items.sort(key=lambda z: z[0])
            for _, r, c, xy, n in items:
                area = cross2(xy[1] - xy[0], xy[3] - xy[0])
                front = area > 0; src = tex if front else back
                nn = n if n[2] >= 0 else -n; b = (0.80 + 0.20 * float(nn @ Lv)) / (0.80 + 0.20 * Lz) * (1.0 if front else 0.93)
                sq = np.array([[c * cw, r * hh], [(c + 1) * cw, r * hh], [(c + 1) * cw, (r + 1) * hh], [c * cw, (r + 1) * hh]], np.float32) * 2
                x0, y0 = np.floor(xy.min(0)).astype(int) - 2; x1, y1 = np.ceil(xy.max(0)).astype(int) + 2
                x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
                if x1 - x0 < 2 or y1 - y0 < 2: continue
                Hm = cv2.getPerspectiveTransform(sq, (xy - [x0, y0]).astype(np.float32))
                wimg = cv2.warpPerspective(src, Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [np.round((xy - [x0, y0]) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
                m = m.astype(np.float32) / 255
                img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m[..., None]) + wimg * b * m[..., None]
                al[y0:y1, x0:x1] = np.maximum(al[y0:y1, x0:x1], m)
            lift = 1 - ease_io(p)
            return composite_layer(bg, img, al, (9 + 26 * lift, 16 + 40 * lift), 13 + 14 * lift, 0.6)

    # ---------------- crumple: a mesh scrunched into a ball, then popped open and flattened
    elif style == "map_crumple":
        nu, nv = 44, 22; uu, vv = np.meshgrid(np.linspace(0, w, nu), np.linspace(0, h, nv))
        rng = np.random.default_rng(17)
        def noise(sig, amp): n = cv2.GaussianBlur(rng.normal(0, 1, (nv, nu)).astype(np.float32), (0, 0), sig); return n / (n.std() + 1e-9) * amp
        n1, n2, n3 = noise(2.2, 1.0), noise(0.7, 1.0), noise(0.8, 1.0)
        th = (uu / w - 0.5) * 2 * math.pi * 0.9; ps = (vv / h - 0.5) * math.pi * 0.82; rb = 150.0
        rr = rb * (1 + 0.24 * n1 + 0.10 * n2)
        ball = np.stack([rr * np.cos(ps) * np.sin(th), rr * np.sin(ps), rr * np.cos(ps) * np.cos(th)], -1)
        flat = np.stack([uu - w / 2, vv - h / 2, np.zeros_like(uu)], -1)
        dist = np.sqrt(((uu - w / 2) / (w / 2)) ** 2 + ((vv - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        wr = np.stack([n2 * 0.5, n3 * 0.5, n3], -1)
        tri = []
        for j in range(nv - 1):
            for i in range(nu - 1):
                a_, b_, c_, d_ = j * nu + i, j * nu + i + 1, (j + 1) * nu + i + 1, (j + 1) * nu + i
                tri += [(a_, b_, c_), (a_, c_, d_)] if (i + j) % 2 == 0 else [(a_, b_, d_), (b_, c_, d_)]
        tri = np.array(tri); texuv = np.stack([uu, vv], -1).reshape(-1, 2) * 2
        t_sq0, t_sq1, t_hold, t_pop0, t_pop1, t_flat = -0.10, 0.40, 0.60, 0.62, 0.98, 1.40
        t_sq0, t_sq1, t_hold, t_pop0, t_pop1, t_flat = [x * T / 2.4 for x in (t_sq0, t_sq1, t_hold, t_pop0, t_pop1, t_flat)]
        def state(t):
            if t < t_sq1: s = ease_io((t - t_sq0) / (t_sq1 - t_sq0)); res = 0.0
            elif t < t_pop0: s = 1.0; res = 0.0
            elif t < t_pop1: u = (t - t_pop0) / (t_pop1 - t_pop0); s = 1 - ease_out_back(u, 1.1); res = 1.0
            else: s = 0.0; res = 1 - ease_io((t - t_pop1) / (t_flat - t_pop1))
            return s, res
        def verts(t):
            s, res = state(t)
            se = np.clip((s - 0.3 * (1 - dist)) / 0.7, 0, 1.0) if t < t_sq1 else np.full_like(dist, s)
            P = flat + (ball - flat) * se[..., None]
            P += wr * (math.sin(math.pi * np.clip(s, 0, 1)) * 52 + res * 9 * (1 if t >= t_pop1 else 0))
            if t_sq1 <= t < t_pop0 + 0.05:   # squeeze in the hand
                q = 1 - 0.08 * math.sin(math.pi * min(1, (t - t_sq1) / (t_pop0 + 0.05 - t_sq1)))
                P *= np.array([q * 1.02, q, q])
            ang = 0.55 * math.sin(math.pi * min(1.0, s)) * (1 if t < t_pop0 else -0.6)
            ca, sa = math.cos(ang), math.sin(ang); x, z = P[..., 0].copy(), P[..., 2].copy(); P[..., 0] = x * ca + z * sa; P[..., 2] = -x * sa + z * ca
            P[..., 1] += -60 * math.sin(math.pi * np.clip(s, 0, 1))
            return P.reshape(-1, 3), s
        mix.crackle(t_sq0, t_sq1 - t_sq0 + 0.04, lambda u: 260 + 900 * math.sin(math.pi * u) ** 0.7, 0.7)
        mix.crackle(t_sq1, t_pop0 - t_sq1, lambda u: 160, 0.35)
        mix.fwump(t_pop0 + 0.01, 0.65); mix.crackle(t_pop0, 0.22, lambda u: 700 * (1 - u) + 60, 0.55)
        mix.fwip(t_pop1 - 0.12, 0.42, 0.3, 300, 2600)
        def effect(t):
            P, s = verts(t)
            q = P.copy(); k = f / (f - q[:, 2]); xy = (q[:, :2] * k[:, None]) @ Rm0.T + ctr
            T3 = q[tri]; xyT = xy[tri]
            nrm = np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]); nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
            area = cross2(xyT[:, 1] - xyT[:, 0], xyT[:, 2] - xyT[:, 0])
            front = area > 0
            nn = np.where(nrm[:, 2:3] >= 0, nrm, -nrm)
            b = (0.58 + 0.42 * np.clip(nn @ Lv, 0, 1)) / (0.58 + 0.42 * Lz) * np.where(front, 1.0, 0.92)
            order = np.argsort(T3[:, :, 2].mean(1))
            idb = np.full((H, W), -1, np.float32)
            for i in order:
                if abs(area[i]) < 0.05: continue
                cv2.fillConvexPoly(idb, np.round(xyT[i] * 4).astype(np.int32), float(i), cv2.LINE_8, 2)
            m = idb >= 0; ids = idb[m].astype(np.int64)
            # affine screen -> texture per triangle
            Sm = np.concatenate([xyT.transpose(0, 2, 1), np.ones((len(tri), 1, 3))], 1)
            Sm[np.abs(np.linalg.det(Sm)) < 1e-6] += np.eye(3) * 1e-3
            Tm = texuv[tri].transpose(0, 2, 1); A = Tm @ np.linalg.inv(Sm)
            ys, xs = np.nonzero(m); Ai = A[ids]
            mx = Ai[:, 0, 0] * xs + Ai[:, 0, 1] * ys + Ai[:, 0, 2]; my = Ai[:, 1, 0] * xs + Ai[:, 1, 1] * ys + Ai[:, 1, 2]
            my = my + np.where(front[ids], 0, tex.shape[0])
            atlas = np.vstack([tex, back])
            MX = np.full((H, W), -10, np.float32); MY = np.full((H, W), -10, np.float32); MX[m] = mx; MY[m] = my
            img = cv2.remap(atlas, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            shd = np.ones((H, W), np.float32); shd[m] = b[ids]
            img *= shd[..., None]
            al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.6)
            lift = min(1, s * 1.2)
            return composite_layer(bg, img, al, (9 + 30 * lift, 16 + 60 * lift), 13 + 16 * lift, 0.6 - 0.15 * lift)

    # ---------------- puzzle: 5 x 3 jigsaw pieces fly in and snap; the last piece is Israel's
    elif style == "map_puzzle":
        nc, nr = 5, 3; pw, ph_ = w / nc, h / nr; rng = np.random.default_rng(4)
        vs = rng.choice([-1, 1], (nr, nc - 1)); hs = rng.choice([-1, 1], (nr - 1, nc))
        def edge(A, B, sign):
            A, B = np.array(A, float), np.array(B, float); d = B - A; L = np.linalg.norm(d); t_ = d / L; nrm = np.array([-t_[1], t_[0]]) * sign
            if sign == 0: return np.array([A, B])
            n1, n2 = A + d * 0.43, A + d * 0.57; C = A + d * 0.5 + nrm * L * 0.15; r = float(np.linalg.norm(n1 - C))
            far = math.atan2(nrm[1], nrm[0])
            wrap_ = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
            d1 = wrap_(math.atan2(*(n1 - C)[::-1]) - far); d2 = wrap_(math.atan2(*(n2 - C)[::-1]) - far)
            arc = far + np.linspace(d1, d2, 30)
            pts = [A, A + d * 0.40, n1] + [C + r * np.array([math.cos(a), math.sin(a)]) for a in arc] + [n2, A + d * 0.60, B]
            return np.array(pts)
        polys = {}
        for r in range(nr):
            for c in range(nc):
                x0_, x1_, y0_, y1_ = c * pw, (c + 1) * pw, r * ph_, (r + 1) * ph_
                top = edge((x0_, y0_), (x1_, y0_), 0 if r == 0 else -hs[r - 1, c])
                right = edge((x1_, y0_), (x1_, y1_), 0 if c == nc - 1 else -vs[r, c])
                bot = edge((x1_, y1_), (x0_, y1_), 0 if r == nr - 1 else hs[r, c])
                left = edge((x0_, y1_), (x0_, y0_), 0 if c == 0 else vs[r, c - 1])
                polys[(r, c)] = np.vstack([top[:-1], right[:-1], bot[:-1], left[:-1]])
        ix, iy = sh.geo2px(*ISR); tgt = (int(iy // ph_), int(ix // pw))
        others = [k for k in polys if k != tgt]; rng.shuffle(others)
        sc = T / 2.4; fl = 0.40 * sc
        starts = {k: (-0.16 + 0.78 * i / (len(others) - 1)) * sc for i, k in enumerate(others)}; starts[tgt] = 0.92 * sc
        land = {k: v + (fl if k != tgt else 0.46 * sc) for k, v in starts.items()}
        start_pose = {}
        for k in polys:
            ang = rng.uniform(0, 2 * math.pi) if k != tgt else math.radians(-60)
            dist_ = rng.uniform(900, 1300); start_pose[k] = (math.cos(ang) * dist_, math.sin(ang) * dist_ * 1.2, rng.uniform(-75, 75), 1.3)
        start_pose[tgt] = (240.0, -1250.0, 38.0, 1.5)
        t_glue = land[tgt] + 0.05
        for k, tl_ in land.items():
            if k == tgt: mix.click(tl_ - 0.005, 0.85, 1400, 6000, 330, 0.014); mix.thud(tl_, 0.35); mix.shimmer(tl_ + 0.02, 0.3, 880)
            else: mix.click(tl_ - 0.005, 0.45 + 0.1 * rng.random(), 1800, 8000, float(rng.uniform(500, 900)), 0.008)
            mix.fwip(starts[k] + 0.05, (land[k] - starts[k]) * 0.8, 0.07 if k != tgt else 0.25, 600, 5000)
        def pose(k, t):
            u = (t - starts[k]) / (land[k] - starts[k])
            if u <= 0: return None
            dx, dy, rt, s0 = start_pose[k]
            e = ease_out_back(u, 1.0) if u < 1 else 1.0; e2 = ease_out(u)
            return dx * (1 - e), dy * (1 - e), rt * (1 - e2), lerp(s0, 1.0, e2), max(0.0, 1 - u)
        def effect(t):
            img = bg.copy()
            ghost = 0.22 * (1 - seg(t, t_glue - 0.3, t_glue))
            if ghost > 0.01:
                lay = np.zeros((H, W), np.uint8); q = apply(M0, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
                cv2.polylines(lay, [np.round(q * SC).astype(np.int32)], True, 255, 2, cv2.LINE_AA, SS)
                m = lay.astype(np.float32)[..., None] / 255 * ghost; img = img * (1 - m) + SKY * m
            items = []
            for k, poly in polys.items():
                p = pose(k, t)
                if p is None: continue
                items.append((p[4], k, p))
            items.sort(key=lambda z: z[0])
            seam = 1 - seg(t, t_glue, z0 - 0.02)
            placed = [it for it in items if it[0] <= 0]; flying = [it for it in items if it[0] > 0]
            if placed:   # placed pieces share one shadow
                lay = np.zeros((H, W), np.uint8)
                for _, k, p in placed: cv2.fillPoly(lay, [np.round(apply(M0, polys[k]) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
                sh_ = cv2.GaussianBlur(lay.astype(np.float32) / 255, (0, 0), 13); sh_ = np.roll(np.roll(sh_, 16, 0), 9, 1) * 0.6
                img = img * (1 - sh_[..., None])
            for hgt, k, (dx, dy, rt, s_, _) in placed + flying:
                r, c = k; pc_ = np.array([(c + 0.5) * pw, (r + 0.5) * ph_])
                a_ = math.radians(rt); ca, sa = math.cos(a_), math.sin(a_)
                Tp = M3(1, 0, pc_[0], 0, 1, pc_[1]) @ M3(ca * s_, -sa * s_, 0, sa * s_, ca * s_, 0) @ M3(1, 0, -pc_[0], 0, 1, -pc_[1])
                Mp = M0 @ Tp; Ms = Mp.copy(); Ms[:2, 2] += Rm0 @ np.array([dx, dy]) * 0 + np.array([dx, dy])
                q = apply(Ms, polys[k]); x0, y0 = np.floor(q.min(0)).astype(int) - 3; x1, y1 = np.ceil(q.max(0)).astype(int) + 3
                if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H: continue
                if hgt > 0:   # own shadow, further away the higher it flies
                    lay = np.zeros((H, W), np.uint8); cv2.fillPoly(lay, [np.round((q + [9 + 50 * hgt, 16 + 80 * hgt]) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
                    sh_ = cv2.GaussianBlur(lay.astype(np.float32) / 255, (0, 0), 13 + 18 * hgt) * (0.55 - 0.2 * hgt); img = img * (1 - sh_[..., None])
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                if x1 - x0 < 2 or y1 - y0 < 2: continue   # piece fully clipped (OpenCV 5 rejects empty arrays)
                Mt = Ms @ M3(0.5, 0, 0, 0, 0.5, 0); Mt[:2, 2] -= [x0, y0]
                wimg = cv2.warpAffine(tex, Mt[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [np.round((q - [x0, y0]) * SC).astype(np.int32)], 255, cv2.LINE_AA, SS)
                m = m.astype(np.float32) / 255; bri = 1 + 0.07 * hgt
                reg = img[y0:y1, x0:x1]; reg = reg * (1 - m[..., None]) + wimg * bri * m[..., None]
                if seam > 0.01:
                    ol = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.polylines(ol, [np.round((q - [x0, y0]) * SC).astype(np.int32)], True, 255, 2, cv2.LINE_AA, SS)
                    om = ol.astype(np.float32)[..., None] / 255 * 0.32 * seam; reg = reg * (1 - om) + NAVY * om
                img[y0:y1, x0:x1] = reg
            if t >= land[tgt] - 0.02:   # gold flash along the last piece
                g = seg(t, land[tgt] - 0.02, land[tgt] + 0.04) * (1 - seg(t, land[tgt] + 0.08, z0 + 0.1))
                if g > 0.01:
                    lay = np.zeros((H, W), np.uint8); cv2.polylines(lay, [np.round(apply(M0, polys[tgt]) * SC).astype(np.int32)], True, 255, 4, cv2.LINE_AA, SS)
                    l = lay.astype(np.float32) / 255; gl = cv2.GaussianBlur(l, (0, 0), 12) * 2.0 + cv2.GaussianBlur(l, (0, 0), 3) * 0.8
                    img = img + np.clip(gl * g, 0, 1.5)[..., None] * GOLD * 0.6
            return img
    mix.zoom_whoosh(z0, T, 0.5)

    def frame(t):
        if t < z0: return np.clip(effect(t), 0, 255)
        u = (t - z0) / (T - z0); e = ease_in(u)
        k = math.exp(math.log(sh.k) + (math.log(F.k) - math.log(sh.k)) * e)
        sT = sT0 + (np.array(F.sT) - sT0) * e; rot = rot0 * (1 - e)
        V = View(F.T, tuple(sT), k, rot, F.cosf); Mg = V.M(); Ms = Mg @ sh.G
        img, al = mt.sheet_layer(sh, Ms, layers); out = mt.composite(bg, [(img, al)])
        du = 1.0 / (FPS * (T - z0)); e2 = ease_in(min(1, u + du)); kr = math.exp((math.log(F.k) - math.log(sh.k)) * (e2 - e))
        out = mt.zoom_blur(out, sT, min(0.055, (kr - 1) * 0.7) * (1 - seg(u, 0.86, 1.0)))
        if reel_f0 is not None:
            a = seg(t, T - 0.36, T - 1.0 / FPS)
            if a > 0:
                Mr = Mg @ np.linalg.inv(MF); rf = cv2.warpAffine(reel_f0, Mr[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out = out * (1 - a) + rf * a
        return np.clip(out, 0, 255)
    return frame


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=STYLES, required=True)
    ap.add_argument("--reel"); ap.add_argument("--cam"); ap.add_argument("--box")
    ap.add_argument("--dur", type=float, default=2.4); ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--plain", action="store_true", help="paper styles: no red empires layer")
    ap.add_argument("--title", default="ISRAEL", help="globe_spin_reverse: the line that rises out of the globe (the reel's hook)")
    ap.add_argument("--hook", help="the reel's hook, on screen from frame 0; lines split with |, *word* in gold")
    ap.add_argument("--hook-y", type=float, default=930, help="centre y of the hook text in the reel's frame 0")
    ap.add_argument("--hook-s", type=float, default=1.0, help="scale of the hook text in the reel's frame 0")
    ap.add_argument("--out", required=True); ap.add_argument("--snaps")
    A = ap.parse_args()
    if not (A.cam or A.box): ap.error("give --cam or --box")
    if A.style in ("map_tear", "map_tear_v2"):   # the control: map_tear.py unchanged; v2: map_tear_v2.py
        cmd = [sys.executable, os.path.join(HERE, A.style + ".py"), "--style", "plain" if A.plain else "empires", "--dur", str(A.dur), "--out", A.out]
        for k in ("reel", "cam", "box", "snaps") + (("hook", "hook_y", "hook_s") if A.style == "map_tear_v2" else ()):
            if getattr(A, k) is not None: cmd += [f"--{k.replace('_', '-')}", str(getattr(A, k))]
        if A.hold: cmd += ["--hold", str(A.hold)]
        sys.exit(subprocess.call(cmd))
    T = min(2.5, max(2.0, A.dur)); mt.disk_guard(need_mb=800, wait_s=1800)
    F = mt.target_view(A); reel_f0 = mt.first_frame(A.reel) if A.reel else None
    mix = Mix(T)
    t0 = time.time()
    if A.style.startswith("globe"):
        glb = Globe(load_layers(), night=A.style == "globe_spin_night")
        frame = globe_style(A.style, T, F, glb, reel_f0, mix, A.title)
    else:
        frame = paper_style(A.style, T, F, reel_f0, mix, empires=not A.plain)
    raw = frame; himg = hk_.make(A.hook) if A.hook else None; zz = (0.50 if A.style == "globe_unpeel" else 0.625) * T
    def frame(t):
        f_ = raw(t)
        if himg is None: return f_
        a = seg(t, T - 0.33, T - 1.0 / FPS) if A.reel else 0.0
        return np.clip(hk_.overlay(f_, himg, t, T, zz, bool(A.reel), y_reel=A.hook_y, s_reel=A.hook_s, alpha=1 - a), 0, 255)
    if A.snaps:
        for s_ in A.snaps.split(","):
            cv2.imwrite(f"{os.path.splitext(A.out)[0]}_{float(s_):.2f}.png", frame(float(s_)).astype(np.uint8))
        print(f"snaps in {time.time() - t0:.1f}s"); return
    nfr = int(round(T * FPS)); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    wav = os.path.join("/tmp", f"map_open_{os.getpid()}.wav"); sfx = mix.done()
    hold = 0.0 if A.reel else max(0.0, A.hold)
    if hold: sfx = np.vstack([sfx, np.zeros((int(round(hold * SR)), 2))])
    mt.write_wav(wav, sfx)
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart"]
    src = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav]
    if A.reel:
        info = mt.probe(A.reel); vs = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        has_a = any(s["codec_type"] == "audio" for s in info["streams"]); fr = vs["r_frame_rate"]
        ra = "[2:a]aformat=sample_rates=48000:channel_layouts=stereo[ra]" if has_a else f"anullsrc=r=48000:cl=stereo,atrim=0:{info['format']['duration']}[ra]"
        fc = (f"[0:v]fps={fr},format=yuv420p,setsar=1[ov];[2:v]scale={W}:{H},fps={fr},format=yuv420p,setsar=1[rv];"
              f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo[oa];{ra};[ov][oa][rv][ra]concat=n=2:v=1:a=1[v][a]")
        cmd = ["ffmpeg", "-y", "-v", "error", *src, "-i", A.reel, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", *enc, A.out]
    else:
        vf = f"tpad=stop_mode=clone:stop_duration={hold}" if hold else "null"
        cmd = ["ffmpeg", "-y", "-v", "error", *src, "-vf", vf, "-map", "0:v", "-map", "1:a", *enc] + ([] if hold else ["-shortest"]) + [A.out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(nfr):
        if i % 24 == 0: mt.disk_guard(need_mb=800, wait_s=1800)
        pr.stdin.write(frame(i / FPS).astype(np.uint8).tobytes())
    pr.stdin.close(); rc = pr.wait(); os.remove(wav)
    if rc: sys.exit(f"ffmpeg failed ({rc})")
    print(json.dumps(dict(out=A.out, style=A.style, opener_s=T, frames=nfr, render_s=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
