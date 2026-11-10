"""templates music beds (copied from /home/claude/safety_v4/audio_v4.py). music() = light documentary 100 BPM, doc_music() = serious piano 86 BPM.
Original note: safety_v4 audio (copied from /home/claude/island_israel/journey_audio.py, unchanged there). Adds doc_music() (serious,
modern documentary bed), stamp() and soft_hit(); mix() takes the music bed as an argument.
Original notes: Israel's Journey audio: original procedural music bed + synthesised SFX (all made here), plus one CC0 Kenney pluck
(copied from /home/claude/silly_maps/sfx_src) for speech-bubble pops. Everything float64 at 48 kHz.
Music: light, adventurous documentary bed in D (Dsus2, Bm7, Gmaj7, A6sus), 100 BPM: string pad, staccato string
ostinato, warm bass, soft low drums from bar 3, celesta motif. Ducked under the voice. SFX placed ~12 dB under the voice."""
import numpy as np, subprocess, json
from scipy.signal import butter, sosfilt, fftconvolve, lfilter

SR = 48000
rng = np.random.default_rng(11)
def bp(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], "bp", fs=SR, output="sos"), x)
def lp(x, f, o=2): return sosfilt(butter(o, f, "lp", fs=SR, output="sos"), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, "hp", fs=SR, output="sos"), x)
def norm(x, peak=1.0): return x / (np.abs(x).max() + 1e-12) * peak
def tt(d): return np.arange(int(d * SR)) / SR
def env(n, a, r):
    t = np.arange(n) / SR; d = n / SR; return np.minimum(1, t / max(a, 1e-4)) * np.minimum(1, (d - t) / max(r, 1e-4))
def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)
def room(x, t60=0.6, mix=0.2, seed=3):
    r = np.random.default_rng(seed); n = int(t60 * SR); ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir = lp(ir, 4500); ir /= np.sqrt((ir ** 2).sum()); y = fftconvolve(x, ir)[:len(x)]; return x * (1 - mix) + y * mix
def saw(ph): return 2 * ((ph / (2 * np.pi)) % 1.0) - 1

# ------------------------------------------------------------------ SFX
def sweep(nz, fc, lo=150, hi=6000, nb=10, q=0.35):
    """time-varying band-pass: crossfade a log-spaced filter bank by distance to fc(t) (no block clicks)"""
    cs = np.geomspace(lo, hi, nb); y = np.zeros(len(nz)); wsum = np.zeros(len(nz)) + 1e-9
    for c in cs:
        w = np.exp(-(np.log(fc / c) / q) ** 2); y += w * bp(nz, c / 1.35, min(c * 1.35, SR / 2 - 100)); wsum += w
    return y / np.sqrt(wsum)
def water_rush(d=2.6):
    t = tt(d); nz = rng.standard_normal(len(t)); b = np.cumsum(rng.standard_normal(len(t))); b = hp(b, 40); b /= np.abs(b).max()
    sw = np.sin(np.pi * np.minimum(1, t / d)) ** 1.5
    x = 0.8 * lp(b, 700) + 0.6 * bp(nz, 500, 2500) * sw + 0.3 * hp(nz, 3000) * sw ** 2
    bub = np.zeros(len(t))
    for i in rng.integers(0, len(t) - 3000, int(d * 40)):
        m = int(0.03 * SR); u = np.arange(m) / SR; f = rng.uniform(300, 900)
        bub[i:i + m] += np.sin(2 * np.pi * f * u * (1 + 8 * u)) * np.exp(-u * 90) * rng.uniform(0.2, 0.6)
    return norm(room((x * sw + 0.5 * bub * sw) * env(len(t), 0.25, 0.6), 0.8, 0.25))
def waves(d, soft=1.0):
    t = tt(d); nz = rng.standard_normal(len(t)); b = np.cumsum(nz); b = hp(b, 30); b /= np.abs(b).max()
    sw = 0.25 + 0.75 * (0.5 - 0.5 * np.cos(2 * np.pi * (t / 4.6 + 0.15))) ** 2
    x = lp(b, 450) * sw + 0.45 * soft * bp(nz, 1200, 3500) * sw ** 2
    return norm(x * env(len(t), 1.2, 1.5))
def tank_stop(d=2.2):
    t = tt(d); u = t / d; f = 34 * (1 - 0.45 * u ** 1.5) * (1 + 0.04 * np.sin(2 * np.pi * 7 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR; x = sum(np.sin(k * ph + k) / k ** 0.8 for k in range(1, 14))
    x = lp(np.tanh(1.6 * x), 500) * (1 - u) ** 0.8
    tr = np.zeros(len(t)); k = 0.0; rate = 9.0
    while k < d * 0.75:
        i = int(k * SR); m = int(0.025 * SR)
        if i + m < len(t): tr[i:i + m] += bp(rng.standard_normal(m), 800, 3000) * np.exp(-np.arange(m) / SR * 160) * (1 - k / d)
        k += 1 / (rate * (1 - 0.7 * k / d))
    clunk = np.zeros(len(t)); i = int((d - 0.55) * SR); m = int(0.35 * SR); w = np.arange(m) / SR
    clunk[i:i + m] = (np.sin(2 * np.pi * 70 * w) + 0.6 * bp(rng.standard_normal(m), 200, 1500)) * np.exp(-w * 18)
    return norm(room(x + 0.35 * tr + 0.8 * clunk, 0.5, 0.18))
def jet_whoosh(d=2.0):
    t = tt(d); u = t / d; nz = rng.standard_normal(len(t))
    fc = 500 + 2600 * np.exp(-((u - 0.45) / 0.18) ** 2)
    y = sweep(nz, fc, 300, 5000)
    amp = np.exp(-((u - 0.45) / 0.22) ** 2); rumble = lp(nz, 220) * amp
    x = norm(y * amp) + 0.6 * norm(rumble); pan = np.clip((u - 0.1) * 1.4 - 0.7, -0.8, 0.8)
    return np.stack([x * (1 - pan) / 2, x * (1 + pan) / 2]) * 2
def ship_horn(d=1.8):
    t = tt(d); vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)
    x = sum(0.7 ** j * saw(2 * np.pi * f * vib * t + j) for j, f in enumerate((98, 147, 196)))
    x = lp(x, 900, 3) * env(len(t), 0.09, 0.4)
    return norm(room(x, 1.4, 0.35))
def plane_takeoff(d=3.0):
    t = tt(d); u = t / d; nz = rng.standard_normal(len(t)); amp = np.sin(np.pi * np.minimum(1, u * 1.15)) ** 1.2
    roar = lp(nz, 260) * 1.5 + sweep(nz, 300 + 2200 * u, 200, 4000)
    whine = np.sin(2 * np.pi * np.cumsum(1700 + 1500 * u) / SR) * 0.08
    x = norm(roar) * amp + whine * amp; pan = -0.6 * u
    return np.stack([x * (1 - pan) / 2, x * (1 + pan) / 2]) * 2
def digging(d=2.6):
    t = tt(d); x = np.zeros(len(t)); hum = 0.15 * lp(saw(2 * np.pi * 52 * t), 300) * env(len(t), 0.2, 0.4)
    k = 0.15
    while k < d - 0.3:
        i = int(k * SR); m = int(0.22 * SR); w = np.arange(m) / SR
        g = np.zeros(m)
        for j in rng.integers(0, m - 400, 60): g[j:j + 200] += rng.uniform(-1, 1) * np.exp(-np.arange(200) / 30)
        crunch = bp(rng.standard_normal(m), 250, 2200) * np.exp(-w * 14) + 0.8 * bp(g, 1500, 6000)
        x[i:i + m] += crunch; k += rng.uniform(0.55, 0.75)
    return norm(room(x + hum, 0.4, 0.15))
def engine_start(d=2.4):
    t = tt(d); x = np.zeros(len(t)); n0 = int(0.75 * SR)
    crank = bp(rng.standard_normal(n0), 300, 1500) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 9 * t[:n0]))) * 0.6
    crank += 0.4 * np.sin(2 * np.pi * 60 * t[:n0]) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * t[:n0]))
    x[:n0] += crank
    w = t[n0:] - t[n0]; f = 28 + 30 * (1 - np.exp(-w * 5)) + 10 * np.exp(-((w - 0.25) / 0.12) ** 2)
    ph = 2 * np.pi * np.cumsum(f) / SR; r = sum(np.sin(k * ph) / k for k in range(1, 12)); r = lp(np.tanh(2 * r), 700)
    x[n0:] += r * np.minimum(1, w / 0.05) * env(len(w), 0.01, 0.5)
    return norm(room(x, 0.4, 0.15))
def prop_hum(d):
    t = tt(d); blade = 0.5 + 0.5 * np.sin(2 * np.pi * 14 * t)
    x = 0.7 * np.sin(2 * np.pi * 62 * t) + 0.3 * np.sin(2 * np.pi * 124 * t) + 0.5 * lp(rng.standard_normal(len(t)), 400) * blade
    return norm(x * (0.6 + 0.4 * blade) * env(len(t), 0.6, 0.8))
def scrape(d=1.6):
    t = tt(d); nz = rng.standard_normal(len(t)); y = sweep(nz, 600 * np.exp(0.35 * np.sin(2 * np.pi * 5.3 * t) * np.sin(2 * np.pi * 1.7 * t)), 300, 1400, 8, 0.25)
    stick = 0.5 + 0.5 * (np.sin(2 * np.pi * 23 * t + 3 * np.sin(2 * np.pi * 3 * t)) > 0.2)
    grind = lp(nz, 180)
    return norm(room((norm(y) * stick + 0.6 * norm(grind)) * env(len(t), 0.05, 0.4), 0.5, 0.2))
def boing(d=1.0):
    t = tt(d); f = 180 * (1 + 0.9 * np.exp(-t * 5)) * (1 + 0.18 * np.exp(-t * 3.5) * np.sin(2 * np.pi * 11 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR; x = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3.01 * ph)
    thud = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
    return norm(room(x * np.exp(-t * 3.2) * env(len(t), 0.004, 0.1) + 0.7 * thud, 0.3, 0.12))
def crowd_bicker(d, voices=7):
    t = tt(d); out = np.zeros(len(t))
    VOW = [(730, 1090), (530, 1840), (270, 2290), (570, 840), (440, 1020), (300, 870)]
    for v in range(voices):
        r = np.random.default_rng(100 + v); f0 = r.uniform(105, 230)
        cont = f0 * (1 + 0.18 * np.sin(2 * np.pi * r.uniform(0.3, 0.9) * t + r.uniform(0, 6)) + 0.06 * np.sin(2 * np.pi * 6 * t))
        src = saw(2 * np.pi * np.cumsum(cont) / SR)
        syl = np.zeros(len(t)); k = r.uniform(0, 0.4)
        while k < d:
            m = int(r.uniform(0.09, 0.22) * SR); i = int(k * SR)
            if i + m < len(t): syl[i:i + m] += np.hanning(m)
            k += r.uniform(0.12, 0.3) + (r.uniform(0.2, 0.6) if r.random() < 0.12 else 0)
        k_ = np.arange(len(t)) // int(0.16 * SR); idx = r.integers(0, len(VOW), k_.max() + 2)
        F1 = lp(np.array([VOW[i][0] for i in idx])[k_].astype(float), 30, 1); F2 = lp(np.array([VOW[i][1] for i in idx])[k_].astype(float), 30, 1)
        y = sweep(src, np.maximum(F1, 200), 220, 1000, 7, 0.3) + 0.6 * sweep(src, np.maximum(F2, 700), 750, 2600, 7, 0.3)
        pan = r.uniform(-0.6, 0.6); out += norm(y * syl) * (0.6 + 0.4 * r.random())
    out = lp(out, 3800) * np.minimum(1, t / 1.2) * env(len(t), 0.4, 0.8) * (0.55 + 0.45 * np.minimum(1, t / d * 1.6))
    return norm(room(out, 0.6, 0.25))
def bubble_pop():
    x = tt(0.14); return norm(np.sin(2 * np.pi * (500 + 5400 * x) * x) * np.exp(-x * 30))

# ------------------------------------------------------------------ music
def hz(m): return 440 * 2 ** ((m - 69) / 12)
def music(dur, bpm=100, seed=5, end_ring=2.5):
    r = np.random.default_rng(seed); n = int(dur * SR); L = np.zeros(n); R = np.zeros(n)
    beat = 60 / bpm; bar = 4 * beat; nb = int(dur / bar) + 2
    prog = [(38, [62, 64, 69, 74]), (35, [62, 66, 69, 71]), (31, [62, 66, 67, 71]), (33, [61, 64, 66, 69])]   # Dsus2 Bm7 Gmaj7 A6sus
    def put(sig, st, g, pan=0.0):
        i = int(st * SR)
        if i >= n: return
        j = min(n, i + len(sig)); L[i:j] += g * (1 - pan) * sig[:j - i]; R[i:j] += g * (1 + pan) * sig[:j - i]
    motif = [74, 76, 78, 81, 78, 76, 74, 73]
    for b in range(nb):
        root, ch = prog[b % 4]; st = b * bar; inten = min(1.0, 0.45 + 0.2 * b)
        t = tt(bar + 0.6)
        pad = sum(sum(saw(2 * np.pi * hz(m) * (1 + dt) * t + r.uniform(0, 6)) for dt in (-0.004, 0.004)) for m in ch)
        pad = lp(pad, 1400, 2) * np.minimum(1, t / 0.6) * np.minimum(1, (bar + 0.6 - t) / 0.6); put(pad, st, 0.020)
        tb = tt(bar); bass = (np.sin(2 * np.pi * hz(root) * tb) + 0.35 * np.sin(4 * np.pi * hz(root) * tb)) * env(len(tb), 0.05, 0.3)
        put(bass, st, 0.20 * inten)
        for q in range(8):    # staccato string ostinato
            m = [root + 24, ch[1], ch[2], ch[1]][q % 4] if q % 4 else root + 24; d = beat * 0.42; ts = tt(d)
            s = lp(saw(2 * np.pi * hz(m) * ts) + 0.5 * saw(2 * np.pi * hz(m) * 1.003 * ts), 2200) * np.exp(-ts * 12) * env(len(ts), 0.006, 0.05)
            put(s, st + q * beat / 2, (0.05 if q % 2 else 0.065) * inten, pan=0.3 * (1 if q % 2 else -1))
        if b >= 2:            # soft low drums: boom on 1, light on the and of 2, boom on 3
            for off, g in ((0, 0.34), (1.5, 0.16), (2, 0.26)):
                td = tt(0.5); f = 90 * np.exp(-td * 10) + 46
                put(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-td * 7) + 0.15 * lp(r.standard_normal(len(td)), 400) * np.exp(-td * 25), st + off * beat, g * inten)
            for q in range(16):
                nzs = hp(r.standard_normal(int(0.04 * SR)), 6000) * np.exp(-tt(0.04) * 80); put(nzs, st + q * beat / 4, 0.018 if q % 2 else 0.03, 0.4)
        if b % 2 == 1:        # celesta motif
            for q, m in enumerate(motif[:4] if b % 4 == 1 else motif[4:]):
                tc = tt(1.2); f = hz(m); s = (np.sin(2 * np.pi * f * tc) + 0.25 * np.sin(2 * np.pi * 2 * f * tc) * np.exp(-tc * 8)) * np.exp(-tc * 3.5) * env(len(tc), 0.003, 0.2)
                put(s, st + q * beat, 0.05, pan=-0.2 + 0.13 * q)
    x = np.stack([L, R]); x = np.stack([room(x[0], 1.2, 0.22, 1), room(x[1], 1.2, 0.22, 2)])
    fade = np.minimum(1, np.arange(n) / (0.5 * SR)) * np.minimum(1, (n - np.arange(n)) / (end_ring * SR))
    return x * fade / (np.abs(x).max() + 1e-9)

# ------------------------------------------------------------------ mix
def act_rms(x, sr=SR):
    n = int(0.05 * sr); e = np.array([np.sqrt((x[i:i + n] ** 2).mean()) for i in range(0, len(x) - n, n)]); th = e.max() * 10 ** (-30 / 20)
    return np.sqrt((e[e > th] ** 2).mean())
def mix(vo, dur, events, out_wav, music_db=-18.0, sfx_db=-12.0, mutes=(), bed=None):
    """vo mono at SR; events: list of (t_start, stereo-or-mono array, extra_db). returns report dict"""
    n = int(dur * SR); v = np.zeros(n); v[:min(n, len(vo))] = vo[:n]
    vr = act_rms(v); m = music(dur) if bed is None else bed
    e2 = lfilter([1], [1, -0.9995], v ** 2) * 0.0005; envv = np.sqrt(np.maximum(e2, 0)); envv = np.convolve(envv, np.ones(4800) / 4800, "same")
    sp = np.clip(envv / (np.percentile(envv[envv > 0], 75) + 1e-9), 0, 1)
    duck = 10 ** ((-7.0 * sp) / 20)          # up to 7 dB of extra ducking under speech
    mr = act_rms(m.mean(0)); m = m * (vr * 10 ** (music_db / 20) / mr) * duck
    tq = np.arange(n) / SR
    for a_, b_ in mutes:                      # music drops out (e.g. "Total quiet."), 0.2 s fades
        g = np.clip(np.maximum((a_ - tq) / 0.2, (tq - b_) / 0.35), 0, 1); m = m * g
    S = np.zeros((2, n)); rep = []
    bed_duck = 10 ** ((-10.0 * sp) / 20)     # continuous beds (waves, hum, crowd) dip 10 dB under speech
    for ev in events:
        t0, x, xdb = ev[:3]; bed = len(ev) > 3 and ev[3]
        x = np.atleast_2d(x); x = np.vstack([x, x]) if x.shape[0] == 1 else x
        g = vr * 10 ** ((sfx_db + xdb) / 20) / (act_rms(x.mean(0)) + 1e-12); i = int(t0 * SR); j = min(n, i + x.shape[1])
        if j > i:
            seg = x[:, :j - i] * g
            if bed: seg = seg * bed_duck[i:j]
            S[:, i:j] += seg; rep.append(round(t0, 2))
    y = np.stack([v, v]) + m + S
    raw = y.T.astype(np.float32).tobytes()
    meas = subprocess.run(["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af",
                           "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], input=raw, capture_output=True).stderr.decode()
    J = json.loads(meas[meas.rfind("{"):meas.rfind("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={J['input_i']}:measured_TP={J['input_tp']}:measured_LRA={J['input_lra']}:"
          f"measured_thresh={J['input_thresh']}:offset={J['target_offset']}:linear=true,aresample=48000")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", out_wav], input=raw, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()[-400:]
    return {"input_I": J["input_i"], "sfx_at": rep}

# ------------------------------------------------------------------ safety_v4 additions (all synthesised here)
def stamp(seed=0):
    """rubber-stamp thunk: short low thud with a pitch drop, a papery slap and a tiny room"""
    r = np.random.default_rng(seed); d = 0.32; t = tt(d); f = 62 + 70 * np.exp(-t * 38)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 22)
    slap = bp(r.standard_normal(len(t)), 900, 3800) * np.exp(-t * 95)
    tick = hp(r.standard_normal(len(t)), 4000) * np.exp(-t * 300)
    return norm(room(body + 0.45 * slap + 0.15 * tick, 0.25, 0.12, seed + 1))
def soft_hit(d=2.4):
    """low cinematic boom with a dark noise tail for the end cards"""
    t = tt(d); f = 38 + 30 * np.exp(-t * 6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) + 0.25 * lp(rng.standard_normal(len(t)), 300) * np.exp(-t * 3)
    return norm(room(x, 1.4, 0.3, 9))
def doc_music(dur, bpm=86, seed=21, flip_at=None, end_ring=3.0):
    """serious but modern documentary bed in D minor: low string pad (Dm, Bb, F/A, C), soft eighth-note pulse on the root,
    sparse felt-piano motif, sub hits on bar starts from bar 3; after flip_at the harmony lifts (Bb, F, C, Dm) and opens up."""
    r = np.random.default_rng(seed); n = int(dur * SR); L = np.zeros(n); R = np.zeros(n)
    beat = 60 / bpm; bar = 4 * beat; nb = int(dur / bar) + 2
    A = [(38, [62, 65, 69]), (34, [62, 65, 70]), (33, [60, 65, 69]), (36, [60, 64, 67])]
    B = [(34, [62, 65, 70, 74]), (29, [60, 65, 69, 72]), (36, [62, 67, 72, 76]), (38, [62, 65, 69, 74])]
    def put(sig, st, g, pan=0.0):
        i = int(st * SR)
        if i >= n or i < 0: return
        j = min(n, i + len(sig)); L[i:j] += g * (1 - pan) * sig[:j - i]; R[i:j] += g * (1 + pan) * sig[:j - i]
    def piano(m, d=1.6):
        tc = tt(d); f = hz(m)
        s = sum(np.sin(2 * np.pi * k * f * tc * (1 + 0.0007 * k)) * np.exp(-tc * (2.2 + 1.3 * k)) / k ** 1.3 for k in range(1, 7))
        return lp(s, 2600) * env(len(tc), 0.004, 0.3)
    motif = [(0, 74), (1.5, 72), (2, 69), (3, 70)]
    for b in range(nb):
        st = b * bar; after = flip_at is not None and st >= flip_at - 0.01
        root, ch = (B if after else A)[b % 4]; inten = min(1.0, 0.5 + 0.12 * b)
        t = tt(bar + 0.8)
        pad = sum(sum(saw(2 * np.pi * hz(m) * (1 + dt) * t + r.uniform(0, 6)) for dt in (-0.003, 0.003)) for m in ch)
        pad = lp(pad, 1100 if not after else 1700, 2) * np.minimum(1, t / 0.8) * np.minimum(1, (bar + 0.8 - t) / 0.8); put(pad, st, 0.016)
        for q in range(8):   # pulse: muted low root eighths, accent on 1
            tq = tt(beat * 0.45); s = (np.sin(2 * np.pi * hz(root + 12) * tq) + 0.4 * np.sin(2 * np.pi * hz(root + 24) * tq)) * np.exp(-tq * 9) * env(len(tq), 0.004, 0.04)
            put(s, st + q * beat / 2, (0.11 if q == 0 else 0.06) * inten)
        tb = tt(bar); put((np.sin(2 * np.pi * hz(root) * tb)) * env(len(tb), 0.08, 0.4), st, 0.13 * inten)
        if b >= 2:
            td = tt(0.9); f = 70 * np.exp(-td * 7) + 34
            put(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-td * 4.5), st, 0.26 * inten)
            for q in range(8):
                nzs = hp(r.standard_normal(int(0.03 * SR)), 7000) * np.exp(-tt(0.03) * 120); put(nzs, st + q * beat / 2 + beat / 4, 0.012 if not after else 0.02, 0.35)
        if b % 2 == 1:
            for off, m in motif: put(piano(m + (3 if after and m == 70 else 0)), st + off * beat, 0.05, pan=-0.15 + 0.1 * off)
    x = np.stack([L, R]); x = np.stack([room(x[0], 1.6, 0.25, 1), room(x[1], 1.6, 0.25, 2)])
    fade = np.minimum(1, np.arange(n) / (0.6 * SR)) * np.minimum(1, (n - np.arange(n)) / (end_ring * SR))
    return x * fade / (np.abs(x).max() + 1e-9)



# ------------------------------------------------------------------ bounce / skip / warm beds (copied from /home/claude/reelkit/kit.py music())
from scipy.signal import sosfiltfilt
def _env(n, a, d):
    t = np.arange(n) / SR; return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t * d)
def kit_music(dur, style, seed=1):
    rng = np.random.default_rng(seed); n = int(dur * SR); out = np.zeros((2, n))
    cfg = {"bounce": dict(bpm=118, T=0, prog=[(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])], tim="marimba"),
           "skip": dict(bpm=104, T=2, prog=[(41, [57, 60, 65]), (45, [57, 60, 64]), (38, [57, 62, 65]), (43, [55, 59, 62])], tim="bell"),
           "warm": dict(bpm=96, T=-3, prog=[(43, [59, 62, 67]), (40, [59, 64, 67]), (36, [60, 64, 67]), (38, [57, 62, 66])], tim="pluck")}[style]
    beat = 60 / cfg["bpm"]; bar = 4 * beat; T = cfg["T"]
    def hz(m): return 440 * 2 ** ((m - 69 + T) / 12)
    def put(sig, st, g, pan=0.0):
        i = int(st * SR); j = min(n, i + len(sig))
        if 0 <= i < n and j > i: out[0, i:j] += g * (1 - pan) * sig[:j - i]; out[1, i:j] += g * (1 + pan) * sig[:j - i]
    def tone(f, d, kind):
        m = int(d * SR); t = np.arange(m) / SR
        if kind == "marimba": return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t * 40)) * _env(m, 0.002, 9)
        if kind == "bell": return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 12)) * _env(m, 0.002, 5)
        sq = sum(np.sin(2 * np.pi * f * h * t) / h ** 1.3 for h in range(1, 7)); return sq * _env(m, 0.003, 7) * 0.6
    nb = int(dur / bar) + 2
    for b in range(nb):
        root, ch = cfg["prog"][b % 4]; st = b * bar; inten = 0.55 if b < 1 else 1.0
        tt = np.arange(int(bar * SR)) / SR  # pad
        pad = sum(sum(np.sin(2 * np.pi * hz(m) * hh * tt + rng.uniform(0, 6)) / hh ** 2 for hh in (1, 2, 3)) for m in ch)
        pad *= np.minimum(1, tt / 0.25) * np.minimum(1, (bar - tt) / 0.2); put(pad, st, 0.035)
        for q, off in enumerate((0, 1.5, 2, 3.5) if style != "warm" else (0, 2, 2.5)):  # bass
            m = root - 12; d = beat * 0.9; t2 = np.arange(int(d * SR)) / SR
            put(np.sin(2 * np.pi * hz(m) * t2) * _env(len(t2), 0.005, 5) + 0.25 * np.sin(4 * np.pi * hz(m) * t2) * _env(len(t2), 0.005, 9), st + off * beat, 0.22 * inten)
        pat = [0, 1, 2, 1, 2, 1, 0, 2] if b % 2 == 0 else [2, 1, 0, 1, 2, 0, 1, 2]
        for q in range(8):  # arpeggio / pluck line
            if style == "warm" and q % 2 == 1 and rng.random() < 0.5: continue
            m = ch[pat[q]] + 12 + (12 if (style == "skip" and q in (3, 7)) else 0)
            put(tone(hz(m), 0.5, cfg["tim"]), st + q * beat / 2, 0.075 * inten, pan=0.25 * (1 if q % 2 else -1))
        if b >= 1:  # drums
            for q in range(4):
                t3 = np.arange(int(0.25 * SR)) / SR; f = 110 * np.exp(-t3 * 18) + 45
                put(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t3 * 14), st + q * beat, 0.30 if style != "warm" or q % 2 == 0 else 0.0)
            for q in (1, 3):
                nz = rng.standard_normal(int(0.18 * SR)); nz = sosfiltfilt(butter(2, [900, 5000], "bp", fs=SR, output="sos"), nz)
                put(nz * _env(len(nz), 0.001, 22), st + q * beat, 0.16)
            for q in range(8 if style != "skip" else 16):
                sub = beat / 2 if style != "skip" else beat / 4
                nz = rng.standard_normal(int(0.05 * SR)); nz = sosfiltfilt(butter(2, 7000, "hp", fs=SR, output="sos"), nz)
                put(nz * _env(len(nz), 0.001, 70), st + q * sub, (0.05 if q % 2 else 0.08) * (0.7 if style == "skip" else 1), pan=0.3)
    fade = np.minimum(1, np.arange(n) / (0.4 * SR)) * np.minimum(1, (n - np.arange(n)) / (1.8 * SR))
    out *= fade; return out / (np.abs(out).max() + 1e-9)
