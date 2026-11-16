"""head_cut.py: Noam's head and shoulders cut out of an on-camera take, frame by frame, for floating-head layouts.
u2net_human_seg (the rembg person model) run directly with onnxruntime, refined with a guided filter on the frame
(clean edges), a 1 px erode and a slight blur (soft edge), a light temporal smoothing inside each continuous run of
frames (no flicker), and soft fades at the bottom and side crop edges so the bust floats. No mirroring.

cut(clip, src_idx, crop, out_w, out_path) -> np.memmap (N, oh, out_w, 4) uint8 BGRA, row i = output frame i
  src_idx  list of source frame numbers, one per output frame (the edit: keep parts mapped onto the output timeline)
  crop     (x, y, w, h) in source pixels
"""
import os, subprocess, numpy as np, cv2

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"


def _model(path=None):
    path = path or os.path.join(os.path.expanduser("~"), ".cache", "u2net_human_seg.onnx")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", path, MODEL_URL], check=True)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _pred(sess, bgr):
    x = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)
    x = x / max(1e-6, x.max())
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    y = (y - y.min()) / max(1e-6, y.max() - y.min())
    return cv2.resize(y, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def _guided(I, p, r=8, eps=2e-3):
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p); cov = box(I * p) - mI * mp; var = box(I * I) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return box(a) * I + box(b)


def cut(clip, src_idx, crop, out_w, out_path, model=None, log=print):
    x, y, w, h = crop; oh = int(round(out_w * h / w / 2) * 2)
    N = len(src_idx); M = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(N, oh, out_w, 4))
    want = {}
    for i, s in enumerate(src_idx): want.setdefault(int(s), []).append(i)
    last = max(want); sess = _model(model)
    yy = np.arange(oh, dtype=np.float32)[:, None]; xx = np.arange(out_w, dtype=np.float32)[None, :]
    fade = np.clip((oh - 1 - yy) / (0.24 * oh), 0, 1) ** 1.3 * np.clip(xx / (0.06 * out_w), 0, 1) * np.clip((out_w - 1 - xx) / (0.06 * out_w), 0, 1)
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip, "-vf", f"crop={w}:{h}:{x}:{y},scale={out_w}:{oh}:flags=area",
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    n = 0; prev = None; prev_k = -9; done = 0
    while n <= last:
        buf = pr.stdout.read(out_w * oh * 3)
        if len(buf) < out_w * oh * 3: break
        if n in want:
            f = np.frombuffer(buf, np.uint8).reshape(oh, out_w, 3)
            m = _pred(sess, f)
            if prev is not None and n == prev_k + 1: m = 0.55 * m + 0.45 * prev       # same run: smooth
            prev, prev_k = m, n
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            a = np.clip(_guided(g, m.astype(np.float32)), 0, 1)
            a = np.clip((a - 0.3) / 0.4, 0, 1); a = a * a * (3 - 2 * a)
            a = cv2.erode(a, np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(a, (0, 0), 1.1) * fade
            rgba = np.dstack([f, (a * 255).astype(np.uint8)])
            for i in want[n]: M[i] = rgba
            done += len(want[n])
            if done % 300 < len(want[n]): log(f"head cut {done}/{N}")
        n += 1
    pr.stdout.close(); pr.kill()
    for i, s in enumerate(src_idx):      # any frame past the clip's end repeats the last good one
        if s > n - 1 and i: M[i] = M[i - 1]
    M.flush(); return M


def paste(out, rgba, cx, bottom, scale, alpha=1.0, shadow=(10, 16, 0.35, 14)):
    """out float32 BGR (H, W, 3); rgba uint8 BGRA bust; placed with its bottom centre at (cx, bottom), soft drop shadow"""
    if alpha <= 0.01: return out
    im = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR).astype(np.float32)
    h, w = im.shape[:2]; H, W = out.shape[:2]
    def blit(src_a, col, ox, oy):
        x0, y0 = int(round(cx - w / 2 + ox)), int(round(bottom - h + oy))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb <= xa or yb <= ya: return
        a = src_a[ya - y0:yb - y0, xa - x0:xb - x0, None] * alpha
        reg = out[ya:yb, xa:xb]
        out[ya:yb, xa:xb] = reg * (1 - a) + (col[ya - y0:yb - y0, xa - x0:xb - x0] if col.ndim == 3 else col) * a
    al = im[..., 3] / 255
    if shadow:
        sa = cv2.GaussianBlur(al, (0, 0), shadow[3]) * shadow[2]
        blit(sa, np.zeros(3, np.float32), shadow[0], shadow[1])
    blit(al, im[..., :3], 0, 0)
    return out
