#!/usr/bin/env python3
"""Testimonial -> 9:16 Instagram story.
usage: story.py spec.json   (spec: id, clip, tx, segs [[s,e],...], hook, name, role)
"""
import json, sys, subprocess, os, re
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
NAVY, SKY, YEL, WHITE = (22, 50, 97), (69, 164, 221), (255, 198, 41), (255, 255, 255)
F = '/usr/share/fonts/truetype/inter/' if os.path.isdir('/usr/share/fonts/truetype/inter/') else ''
def font(weight, size):
    for p in [f'{F}Inter-{weight}.ttf'] + subprocess.run(['fc-match', '-f', '%{file}', f'Inter:style={weight}'], capture_output=True, text=True).stdout.split('\n'):
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
HEB = subprocess.run(['fc-match', '-f', '%{file}', 'DejaVu Sans:style=Bold'], capture_output=True, text=True).stdout

def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def sh(*a): subprocess.run(a, check=True)

spec = json.load(open(sys.argv[1]))
out = os.path.dirname(os.path.abspath(sys.argv[1]))
sid = spec['id']
tx = json.load(open(spec['tx']))
FIX = {'curse': 'course', 'Karine': 'Karien', 'ExpressEx': 'ExpressX', 'Cleel,': 'Klil,'}
words = [[FIX.get(w[0], w[0]), w[1], w[2]] for s in tx['segs'] for w in s.get('w', [])]
def snap(s, e):
    ins = [w for w in words if w[1] >= s - 0.3 and w[2] <= e + 0.3]
    if not ins: return s, e
    return max(0, ins[0][1] - 0.12), ins[-1][2] + 0.25
spec['segs'] = [list(snap(a, b)) for a, b in spec['segs']]

# 1. cut + join segments (video area 1080 wide, 16:9 kept)
parts, t0, timeline = [], 0.0, []
for i, (s, e) in enumerate(spec['segs']):
    p = f'{out}/{sid}_p{i}.mp4'
    sh('ffmpeg', '-v', 'error', '-y', '-ss', str(s), '-to', str(e), '-i', spec['clip'],
       '-vf', 'scale=1080:-2,fps=30,format=yuv420p', '-af', 'aresample=48000', '-ac', '2',
       '-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '160k', p)
    parts.append(p)
    for w, ws, we in words:
        if ws >= s - 0.05 and we <= e + 0.15:
            timeline.append([w, ws - s + t0, we - s + t0])
    t0 += e - s
main_dur = t0
with open(f'{out}/{sid}_parts.txt', 'w') as fh:
    fh.writelines(f"file '{p}'\n" for p in parts)
sh('ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', f'{out}/{sid}_parts.txt', '-c', 'copy', f'{out}/{sid}_main.mp4')
vh = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=height', '-of', 'csv=p=0', f'{out}/{sid}_main.mp4'], capture_output=True, text=True).stdout.strip())
VY = 700

# 2. background card
bg = Image.new('RGB', (W, H), NAVY); d = ImageDraw.Draw(bg)
fh_, LH = font('ExtraBold', 82), 96
lines = wrap(d, '“' + spec['hook'] + '”', fh_, W - 120)
if len(lines) > 3:
    fh_, LH = font('ExtraBold', 68), 80
    lines = wrap(d, '“' + spec['hook'] + '”', fh_, W - 100)
y = VY - 70 - len(lines) * LH
d.text((W // 2, min(250, y - 60)), 'ENGLISH EXPRESS  ·  REAL STORIES', font=font('SemiBold', 36), fill=SKY, anchor='mm')
for ln in lines:
    d.text((W // 2, y), ln, font=fh_, fill=WHITE, anchor='mt'); y += LH
d.text((W // 2, VY + vh + 60), spec['name'], font=font('Bold', 60), fill=WHITE, anchor='mt')
d.text((W // 2, VY + vh + 136), spec['role'], font=font('Medium', 40), fill=SKY, anchor='mt')
bg.save(f'{out}/{sid}_bg.png')

# 3. subtitles (ASS), chunks of up to 4 words, break on punctuation
chunks, cur = [], []
for w in timeline:
    cur.append(w)
    if len(cur) >= 4 or re.search(r'[.,?!]$', w[0]):
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
def ts(t):
    t = max(0, t); return f'{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}'
SUBY = VY + vh + 330
ass = ['[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', '',
       '[V4+ Styles]', 'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
       'Style: S,Inter SemiBold,72,&H00FFFFFF,&H00FFFFFF,&H00613216,&H80000000,1,0,0,0,100,100,0,0,1,0,0,8,60,60,0,1', '',
       '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
for i, c in enumerate(chunks):
    st = c[0][1]; en = chunks[i + 1][0][1] if i + 1 < len(chunks) else min(c[-1][2] + 0.4, main_dur)
    en = min(en, c[-1][2] + 0.6)
    txt = ' '.join(x[0] for x in c).replace('{', '').replace('}', '')
    ass.append(f'Dialogue: 0,{ts(st)},{ts(en)},S,,0,0,0,,{{\\pos({W//2},{SUBY})\\an8}}{txt}')
open(f'{out}/{sid}.ass', 'w').write('\n'.join(ass) + '\n')

# 4. end card
ec = Image.new('RGB', (W, H), NAVY); d = ImageDraw.Draw(ec)
y = 640
for ln in wrap(d, spec.get('end', 'Your story could be next.'), font('ExtraBold', 96), W - 140):
    d.text((W // 2, y), ln, font=font('ExtraBold', 96), fill=WHITE, anchor='mt'); y += 112
y += 70
f2 = font('Bold', 64); tw = d.textlength('Drop us a note', font=f2)
d.rounded_rectangle((W//2 - tw/2 - 56, y, W//2 + tw/2 + 56, y + 128), radius=64, fill=YEL)
d.text((W // 2, y + 64), 'Drop us a note', font=f2, fill=NAVY, anchor='mm')
d.text((W // 2, y + 230), 'שלחו לנו הודעה ונחזור אליכם', font=ImageFont.truetype(HEB, 46), fill=SKY, anchor='mm')
d.text((W // 2, H - 260), '@expressenglishcom', font=font('SemiBold', 40), fill=SKY, anchor='mm')
ec.save(f'{out}/{sid}_end.png')

# 5. compose
END = 3.0
fc = (f'[0:v][1:v]overlay=0:{VY}:shortest=1,ass={out}/{sid}.ass,fade=t=in:st=0:d=0.25[m];'
      f'[2:v]scale={W}:{H},fps=30,format=yuv420p,fade=t=in:st=0:d=0.3[e];'
      f'[1:a]loudnorm=I=-16:TP=-1.5,afade=t=out:st={main_dur-0.4:.2f}:d=0.4[ma];'
      f'anullsrc=r=48000:cl=stereo,atrim=0:{END}[ea];'
      f'[m][ma][e][ea]concat=n=2:v=1:a=1[v][a]')
sh('ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '30', '-i', f'{out}/{sid}_bg.png',
   '-i', f'{out}/{sid}_main.mp4', '-loop', '1', '-framerate', '30', '-t', str(END), '-i', f'{out}/{sid}_end.png',
   '-filter_complex', fc, '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', f'{out}/{sid}.mp4')
print(sid, round(main_dur + END, 1), 's')
