import json, os, sys, subprocess
name, fid = sys.argv[1], sys.argv[2]
os.makedirs('long', exist_ok=True)
src = f'/tmp/{name}.src'
subprocess.run(['curl', '-sSL', '--fail', '-o', src, f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', '16000', f'/tmp/{name}.wav'], check=True)
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src], capture_output=True, text=True).stdout.strip() or 0)
info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', src], capture_output=True, text=True).stdout.strip()
for i, f in enumerate((0.1, 0.4, 0.7)):
    if info: subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(dur * f), '-i', src, '-frames:v', '1', '-vf', 'scale=480:-2', f'long/{name}_f{i}.jpg'])
from faster_whisper import WhisperModel
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')
segs, inf = m.transcribe(f'/tmp/{name}.wav', word_timestamps=True, vad_filter=True)
out = []
for s in segs:
    out.append({'s': round(s.start, 2), 'e': round(s.end, 2), 't': s.text.strip(), 'w': [[w.word.strip(), round(w.start, 2), round(w.end, 2)] for w in (s.words or [])]})
json.dump({'name': name, 'dur': dur, 'video': info, 'lang': inf.language, 'segs': out}, open(f'long/{name}.json', 'w'), ensure_ascii=False)
open(f'long/{name}.txt', 'w').write('\n'.join(f"[{x['s']:.1f}] {x['t']}" for x in out))
