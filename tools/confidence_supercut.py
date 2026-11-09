#!/usr/bin/env python3
"""'Confidence' supercut from testimonial transcripts -> 9:16 reel."""
import json, re, os, subprocess, glob
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__)); os.chdir(D)
W, H, VY = 1080, 1920, 700
NAVY, SKY, YEL, WHITE = (22, 50, 97), (69, 164, 221), (255, 198, 41), (255, 255, 255)
def font(w, s): return ImageFont.truetype(subprocess.run(['fc-match', '-f', '%{file}', f'Inter:style={w}'], capture_output=True, text=True).stdout, s)
HEB = subprocess.run(['fc-match', '-f', '%{file}', 'DejaVu Sans:style=Bold'], capture_output=True, text=True).stdout
def sh(*a): subprocess.run(a, check=True)
NAMES = {'ohad': 'Ohad Dreifuss', 'klil': 'Klil Shafrir', 'yaniv': 'Yaniv Refael', 'tirosh': 'Noam Tirosh', 'yuvalmor': 'Yuval Mor',
         'gan': 'Gan Klein', 'dima': 'Dima Shaposhnikov', 'dima_day': 'Dima Shaposhnikov', 'eliya': 'Eliya Shtraser', 'karin': 'Karin Marton',
         'amir': 'Amir Shneider', 'amnon': 'Amnon Ben Zion', 'bensela': 'Ben Sela', 'marina': 'Marina Matatov', 'maya': 'Maya Halili', 'nate': 'Nate Leibzon'}
CLIP = lambda n: f"clips/vt_{'dimaday' if n == 'dima_day' else n}.mp4"
# lead phrases (person, approx word time) shown with context first
LEAD = [('ohad', 2.7), ('klil', 3.0), ('yaniv', 0.8), ('tirosh', 1.8), ('yuvalmor', 2.8)]

hits = []
for n in NAMES:
    if not os.path.exists(CLIP(n)): print('skip (no clip)', n); continue
    tx = json.load(open(f'tx/t_{n}.json'))
    ws = [w for s in tx['segs'] for w in s.get('w', [])]
    for i, w in enumerate(ws):
        if re.search(r'confiden', w[0], re.I):
            hits.append({'n': n, 'i': i, 'ws': ws, 's': w[1], 'e': w[2]})
lead = []
for n, t in LEAD:
    h = min((h for h in hits if h['n'] == n), key=lambda h: abs(h['s'] - t))
    lead.append(h)
rest = [h for h in hits if h not in lead]
# interleave people in the rapid part so the same face doesn't repeat back to back
rest.sort(key=lambda h: (sum(1 for x in rest if x['n'] == h['n'] and x['s'] < h['s']), h['n']))
order = [('lead', h) for h in lead] + [('fast', h) for h in rest]

parts, tl, t0 = [], [], 0.0
for k, (mode, h) in enumerate(order):
    ws, i = h['ws'], h['i']
    if mode == 'lead':
        j0 = i
        while j0 > 0 and h['s'] - ws[j0 - 1][1] < 1.8 and not re.search(r'[.?!]$', ws[j0 - 1][0]): j0 -= 1
        j1 = i
        while not re.search(r'[.,?!]$', ws[j1][0]) and j1 + 1 < len(ws) and ws[j1 + 1][2] - h['e'] < 1.0: j1 += 1
        s, e = ws[j0][1] - 0.12, ws[j1][2] + 0.22
        words = [x for x in ws if x[1] >= s - 0.02 and x[2] <= e + 0.05]
        cap = ' '.join(x[0] for x in words)
    else:
        s, e = h['s'] - 0.14, h['e'] + 0.22
        if e - s < 0.75: e = s + 0.75
        cap = re.sub(r'[^A-Za-z-]', '', h['ws'][i][0]).strip('-').lower()
    s = max(0, s); d = e - s
    p = f'sc/p{k:02d}.mp4'
    sh('ffmpeg', '-v', 'error', '-y', '-ss', f'{s:.3f}', '-i', CLIP(h['n']), '-t', f'{d:.3f}',
       '-vf', 'scale=1080:608:force_original_aspect_ratio=decrease,pad=1080:608:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p',
       '-af', f'aresample=48000,loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.03,afade=t=out:st={d-0.04:.3f}:d=0.04', '-ac', '2',
       '-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '160k', p)
    parts.append(p); tl.append((t0, t0 + d, k + 1, NAMES[h['n']], cap, mode)); t0 += d
open('sc/parts.txt', 'w').writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
sh('ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', 'sc/parts.txt', '-c', 'copy', 'sc/main.mp4')
main = t0
people = len({x[3] for x in tl}); count = len(tl)

def ts(t): return f'0:{int(t//60):02d}:{t%60:05.2f}'
ass = ['[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', '', '[V4+ Styles]',
       'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
       'Style: N,Inter ExtraBold,230,&H0029C6FF,&H0029C6FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1',
       'Style: L,Inter SemiBold,44,&H00DDA445,&H00DDA445,&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,0,0,8,0,0,0,1',
       'Style: P,Inter Bold,58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1',
       'Style: C,Inter ExtraBold,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,70,70,0,1', '',
       '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
for a, b, k, name, cap, mode in tl:
    ass.append(f'Dialogue: 0,{ts(a)},{ts(b)},N,,0,0,0,,{{\\pos({W//2},150)}}{k}')
    ass.append(f'Dialogue: 0,{ts(a)},{ts(b)},L,,0,0,0,,{{\\pos({W//2},420)}}TIMES OUR GRADS SAID IT')
    ass.append(f'Dialogue: 0,{ts(a)},{ts(b)},P,,0,0,0,,{{\\pos({W//2},{VY+608+60})}}{name}')
    if mode == 'lead':
        cap = re.sub(r'(?i)(confiden\w*)', r'{\\c&H0029C6FF&}\1{\\c&H00FFFFFF&}', cap)
    else:
        cap = '{\\c&H0029C6FF&}' + cap
    ass.append(f'Dialogue: 0,{ts(a)},{ts(b)},C,,0,0,0,,{{\\pos({W//2},{VY+608+200})}}{cap}')
open('sc/sc.ass', 'w').write('\n'.join(ass) + '\n')

def card(path, big, small=None, cta=False):
    im = Image.new('RGB', (W, H), NAVY); d = ImageDraw.Draw(im); y = 620
    for ln in big:
        d.text((W // 2, y), ln, font=font('ExtraBold', 104), fill=YEL if 'confidence' in ln.lower() else WHITE, anchor='mt'); y += 124
    if small: d.text((W // 2, y + 30), small, font=font('SemiBold', 46), fill=SKY, anchor='mt'); y += 110
    if cta:
        f2 = font('Bold', 64); tw = d.textlength('Drop us a note', font=f2); y += 60
        d.rounded_rectangle((W//2 - tw/2 - 56, y, W//2 + tw/2 + 56, y + 128), radius=64, fill=YEL)
        d.text((W // 2, y + 64), 'Drop us a note', font=f2, fill=NAVY, anchor='mm')
        d.text((W // 2, y + 230), 'שלחו לנו הודעה ונחזור אליכם', font=ImageFont.truetype(HEB, 46), fill=SKY, anchor='mm')
    im.save(path)
card('sc/intro.png', ['One word.', f'{people} grads.'], 'Listen to what they all said')
card('sc/end.png', [f'{count} times.', 'One word:', 'confidence.'], 'Want yours?', cta=True)
bg = Image.new('RGB', (W, H), NAVY); bg.save('sc/bg.png')
I, E = 2.2, 4.0
fc = (f'[0:v][1:v]overlay=0:{VY}:shortest=1,ass=sc/sc.ass[m];'
      f'[2:v]fps=30,format=yuv420p,fade=t=in:d=0.3[i];[3:v]fps=30,format=yuv420p,fade=t=in:d=0.3[e];'
      f'anullsrc=r=48000:cl=stereo,atrim=0:{I}[ia];anullsrc=r=48000:cl=stereo,atrim=0:{E}[ea];'
      f'[i][ia][m][1:a][e][ea]concat=n=3:v=1:a=1[v][a]')
sh('ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '30', '-i', 'sc/bg.png', '-i', 'sc/main.mp4',
   '-loop', '1', '-framerate', '30', '-t', str(I), '-i', 'sc/intro.png', '-loop', '1', '-framerate', '30', '-t', str(E), '-i', 'sc/end.png',
   '-filter_complex', fc, '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p',
   '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', 'confidence_supercut.mp4')
print('clips', count, 'people', people, 'dur', round(I + main + E, 1))
