"""Overlay year counter, place labels, notes, music on the SimCity look test v2 frames -> mp4 1080x1920 30 fps."""
import json, os, sys, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
FR, OUT, REPO = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(REPO, 'templates', 'core'))
proj = json.load(open(os.path.join(FR, 'proj.json')))
N = len(proj); W, H = 1080, 1920
EB = '/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf'; BD = '/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf'
fY = ImageFont.truetype(EB, 150); fL = ImageFont.truetype(EB, 54); fS = ImageFont.truetype(BD, 40); fN = ImageFont.truetype(BD, 36)
fC = ImageFont.truetype(EB, 66); fM = ImageFont.truetype(BD, 38)
YEL, WHITE, NAVY = (255, 210, 63), (255, 255, 255), (22, 50, 97)

def pill(d, x, y, text, font, fill=(12, 29, 64, 210), col=WHITE, pad=16, anchor='mm'):
    w = d.textlength(text, font=font); h = font.size * 1.25
    d.rounded_rectangle([x - w / 2 - pad, y - h / 2 - 6, x + w / 2 + pad, y + h / 2 + 2], radius=18, fill=fill)
    d.text((x, y), text, font=font, fill=col, anchor=anchor)

def safe(l): return 0.02 < l[2] < 0.98 and 0.12 < l[3] < 0.88
SCHEDULE = {}
_cur = 3
_names = []
for f in range(N):
    for l in proj[str(f)]['labels']:
        if l[0] not in _names: _names.append((l[0], l[4])) if l[0] not in [n for n, _ in _names] else None
for name, a0 in sorted(_names, key=lambda x: x[1]):
    st = None
    for f in range(max(int(a0) - 2, _cur), N - 4):
        ls = [l for l in proj[str(f)]['labels'] if l[0] == name]
        if ls and safe(ls[0]):
            ok = sum(1 for g in range(f, min(N, f + 10)) if any(l[0] == name and safe(l) for l in proj[str(g)]['labels']))
            if ok >= 7: st = f; break
        if f > a0 + 40: break
    if st is not None:
        SCHEDULE[name] = (st, st + 10); _cur = st + 10
print('labels:', SCHEDULE)

def frame(i):
    p = proj[str(i)]; yr = int(round(p['year'])) if i == N - 1 else int(p['year'])
    fp = os.path.join(FR, 'f_%04d.png' % i); fp = fp if os.path.exists(fp) else fp[:-4] + '.jpg'
    im = Image.open(fp).convert('RGB').resize((W, H), Image.LANCZOS)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    # top band for legibility
    g = np.zeros((H, W, 4), np.uint8); a = np.clip(1 - np.arange(H) / 520, 0, 1) ** 1.6 * 170; g[..., 2] = 40; g[..., 1] = 18; g[..., 3] = a[:, None].astype(np.uint8)
    lay = Image.alpha_composite(lay, Image.fromarray(g)); d = ImageDraw.Draw(lay)
    d.rounded_rectangle([40, 52, 300, 108], radius=28, fill=(12, 29, 64, 200)); d.ellipse([60, 72, 76, 88], fill=YEL)
    d.text((88, 80), 'Israel Facts', font=fN, fill=WHITE, anchor='lm')
    d.text((60, 140), str(yr), font=fY, fill=YEL, stroke_width=4, stroke_fill=(12, 29, 64))
    if yr >= 1948 and yr < 2020:
        d.text((64, 310), '806,000 people', font=fL, fill=WHITE, stroke_width=3, stroke_fill=(12, 29, 64))
        d.text((64, 376), 'Israel CBS, May 1948', font=fM, fill=(220, 228, 240), stroke_width=2, stroke_fill=(12, 29, 64))
    if yr >= 2024:
        d.text((64, 310), '10.03M people', font=fL, fill=WHITE, stroke_width=3, stroke_fill=(12, 29, 64))
        d.text((64, 376), 'Israel CBS, end of 2024', font=fM, fill=(220, 228, 240), stroke_width=2, stroke_fill=(12, 29, 64))
    # opening caption
    if i < 52:
        al = int(255 * min(1, (52 - i) / 8))
        w1 = d.textlength('1900: before Tel Aviv.', font=fC)
        d.rounded_rectangle([540 - w1 / 2 - 34, 1290, 540 + w1 / 2 + 34, 1480], radius=26, fill=(12, 29, 64, int(al * 0.85)))
        d.text((540, 1345), '1900: before Tel Aviv.', font=fC, fill=YEL + (al,), anchor='mm')
        d.text((540, 1428), 'Watch what happens.', font=fC, fill=WHITE + (al,), anchor='mm')
    # place labels (only the newest few, fade in and out)
    pick = []
    for name, (s0, s1) in SCHEDULE.items():
        if s0 <= i < s1:
            l = [l for l in p['labels'] if l[0] == name]
            if l: pick.append(l[0][:4] + [s0, s1])
    for text, sub, x, y, a0, a1 in pick:
        al = min(1, (i - a0 + 2) / 3, (a1 - i + 1) / 3)
        if al <= 0: continue
        X, Y = x * W, y * H - 70
        A = int(255 * al)
        w = max(d.textlength(text, font=fL), d.textlength(sub, font=fS))
        X = min(max(X, w / 2 + 50), W - w / 2 - 50); Y = min(max(Y, 520), 1700)
        d.rounded_rectangle([X - w / 2 - 22, Y - 52, X + w / 2 + 22, Y + 50], radius=20, fill=(12, 29, 64, int(A * 0.86)))
        d.text((X, Y - 20), text, font=fL, fill=WHITE + (A,), anchor='mm')
        d.text((X, Y + 26), sub, font=fS, fill=YEL + (A,), anchor='mm')
        d.line([(x * W, y * H - 18), (X, Y + 50)], fill=(255, 255, 255, int(A * 0.6)), width=3)
    d.text((540, 1840), 'LOOK TEST: city sizes illustrative, not data', font=fN, fill=(235, 240, 250, 210), anchor='mm', stroke_width=2, stroke_fill=(12, 29, 64))
    return Image.alpha_composite(im.convert('RGBA'), lay).convert('RGB')

if __name__ == '__main__':
    if len(sys.argv) > 4:
        for i in map(int, sys.argv[4].split(',')): frame(i).resize((360, 640)).save(f'post_{i}.jpg')
        sys.exit()
    dur = N / 15.0
    import music as MU
    try: bed = MU.doc_music(dur + 0.5, bpm=86, seed=21, flip_at=7.6, end_ring=2.0)
    except TypeError: bed = MU.doc_music(dur + 0.5)
    bed = np.asarray(bed, np.float32)
    if bed.ndim > 1: bed = bed.mean(axis=1) if bed.shape[1] <= 2 else bed.mean(axis=0)
    import soundfile as sf
    sf.write('bed.wav', bed / (np.abs(bed).max() + 1e-9) * 0.8, MU.SR)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '15', '-i', '-',
                            '-i', 'bed.wav', '-filter_complex', "[0:v]minterpolate=fps=30:mi_mode=blend[v];[1:a]afade=t=in:d=0.6,afade=t=out:st=%.2f:d=1.5,loudnorm=I=-16:TP=-1.5:LRA=11[a]" % (dur - 1.5),
                            '-map', '[v]', '-map', '[a]', '-t', '%.2f' % dur, '-c:v', 'libx264', '-crf', '19', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(frame(i).tobytes())
    enc.stdin.close(); enc.wait(); print('ok', OUT)
