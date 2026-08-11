"""
Latency/throughput benchmark for the inference pipeline.

Mirrors the exact stages run_recognition() in app.py performs:
  1. MTCNN detect  (find face boxes)
  2. MTCNN align   (crop+align to 160x160 tensor)
  3. FaceNet embed (InceptionResnetV1 -> 512-d vector)
  4. SVM classify  (predict_proba)

Usage:
    python benchmark.py path/to/image1.jpg [path/to/image2.jpg ...]

Runs single-threaded on CPU, matching the Dockerfile's deployed environment.
Images are downscaled per config.MAX_INPUT_DIM, identically to app.py, so
results reflect actual deployed behavior rather than raw high-res photos.
"""
import sys
import time
import statistics

import numpy as np
import torch
import joblib
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from config import (
    DEVICE, MODEL_PATH, LABELS_PATH,
    FACE_MARGIN, MIN_FACE_SIZE, DETECT_THRESH, MAX_INPUT_DIM,
)

N_WARMUP = 10
N_RUNS = 100


def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(name, times_ms):
    print(f"\n{name}  (n={len(times_ms)})")
    print(f"  mean : {statistics.mean(times_ms):7.2f} ms")
    print(f"  p50  : {percentile(times_ms, 50):7.2f} ms")
    print(f"  p95  : {percentile(times_ms, 95):7.2f} ms")
    print(f"  p99  : {percentile(times_ms, 99):7.2f} ms")
    print(f"  min  : {min(times_ms):7.2f} ms")
    print(f"  max  : {max(times_ms):7.2f} ms")


def load_capped(path):
    """Mirrors the exact downscale app.py's run_recognition() applies before detection."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    return im


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    torch.set_num_threads(1)  # single-thread number = honest per-request cost, not a multi-core burst

    print("Loading models...")
    mtcnn = MTCNN(
        image_size=160, margin=FACE_MARGIN, keep_all=True,
        min_face_size=MIN_FACE_SIZE, thresholds=[0.6, 0.7, 0.7], device=DEVICE,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    for p in resnet.parameters():
        p.requires_grad = False
    model = joblib.load(MODEL_PATH)
    idx2label = list(np.load(LABELS_PATH, allow_pickle=True))
    print(f"Loaded. {len(idx2label)} identities.")

    imgs = [load_capped(p) for p in sys.argv[1:]]
    print("Benchmark image sizes:", [im.size for im in imgs])

    detect_ms, align_ms, embed_ms, classify_ms, e2e_ms = [], [], [], [], []

    total = N_WARMUP + N_RUNS
    for i in range(total):
        img = imgs[i % len(imgs)]
        is_warmup = i < N_WARMUP

        t0 = time.perf_counter()
        boxes, probs = mtcnn.detect(img)
        t1 = time.perf_counter()

        if boxes is None:
            continue
        valid = [(b, p) for b, p in zip(boxes, probs) if p >= DETECT_THRESH]
        if not valid:
            continue
        box, prob = valid[0]

        x1, y1, x2, y2 = [int(v) for v in box]
        fx1, fy1 = max(0, x1 - FACE_MARGIN), max(0, y1 - FACE_MARGIN)
        fx2, fy2 = min(img.width, x2 + FACE_MARGIN), min(img.height, y2 + FACE_MARGIN)
        face_crop = img.crop((fx1, fy1, fx2, fy2))

        with torch.no_grad():
            t2 = time.perf_counter()
            face_tensor = mtcnn(face_crop)
            t3 = time.perf_counter()

            if face_tensor is None:
                continue
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor[0:1].to(DEVICE)

            embedding = resnet(face_tensor).cpu().numpy()
            t4 = time.perf_counter()

        model.predict_proba(embedding)
        t5 = time.perf_counter()

        if not is_warmup:
            detect_ms.append((t1 - t0) * 1000)
            align_ms.append((t3 - t2) * 1000)
            embed_ms.append((t4 - t3) * 1000)
            classify_ms.append((t5 - t4) * 1000)
            e2e_ms.append((t5 - t0) * 1000)

    summarize("MTCNN detect", detect_ms)
    summarize("MTCNN align/crop", align_ms)
    summarize("FaceNet embed", embed_ms)
    summarize("SVM classify", classify_ms)
    summarize("End-to-end (single face)", e2e_ms)

    throughput = 1000 / statistics.mean(e2e_ms)
    print(f"\nThroughput (single-threaded, sequential): {throughput:.2f} faces/sec")


if __name__ == "__main__":
    main()
