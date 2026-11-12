#!/usr/bin/env python3
"""Robust Video Matting (MobileNetV3, ONNX, CPU) of a face recording -> foreground + alpha videos for render_face.py.

  python3 tools/face/matte.py <take.mov> <out_alpha.mkv> <out_fgr.mp4> [ranges.json] [--crf 19]

Model: tools/face/models/rvm_mnv3.onnx (RVM, PeterL1n/RobustVideoMatting, GPL-3.0). Output is 648x1152 (0.6 of a
1080x1920 take), 30 fps, frame for frame with the source. ranges.json [[a, b], ...] in seconds limits the work to the
spoken parts (frames outside get black and reset the recurrent state); without it every frame is matted.
"""
import os, sys, json, subprocess
import numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
crf = sys.argv[sys.argv.index("--crf") + 1] if "--crf" in sys.argv else "19"
if "--crf" in sys.argv: args.remove(crf)
src, outa, outf = args[:3]
R = json.load(open(args[3])) if len(args) > 3 else None
need = lambda t: R is None or any(a - 1.2 <= t <= b + 0.3 for a, b in R)
so = ort.SessionOptions(); so.intra_op_num_threads = 0
s = ort.InferenceSession(os.path.join(HERE, "models", "rvm_mnv3.onnx"), so, providers=["CPUExecutionProvider"])
zero = lambda: [np.zeros([1, 1, 1, 1], np.float32)] * 4
rec = zero(); dr = np.array([0.4], np.float32)
W, H = 648, 1152
pa = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                       "-c:v", "ffv1", outa], stdin=subprocess.PIPE)
pf = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                       "-c:v", "libx264", "-crf", crf, "-preset", "slow", "-pix_fmt", "yuv444p", outf], stdin=subprocess.PIPE)
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={W}:{H}:flags=area", "-r", "30", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
n = 0
while True:
    b = dec.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    t = n / 30
    if not need(t):
        rec = zero(); pa.stdin.write(bytes(W * H)); pf.stdin.write(bytes(W * H * 3)); n += 1; continue
    x = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32).transpose(2, 0, 1)[None] / 255
    fgr, pha, *rec = s.run(None, {"src": x, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3], "downsample_ratio": dr})
    pa.stdin.write((pha[0, 0] * 255).clip(0, 255).astype(np.uint8).tobytes())
    pf.stdin.write((fgr[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8).tobytes())
    n += 1
    if n % 300 == 0: print(n, flush=True)
pa.stdin.close(); pf.stdin.close(); pa.wait(); pf.wait(); print("done", n, flush=True)
