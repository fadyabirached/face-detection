# 👤 Face Recognition System

[![CI](https://github.com/fadyabirached/face-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/face-recognition/actions/workflows/ci.yml)
[![Build and Push to ACR](https://github.com/fadyabirached/face-recognition/actions/workflows/deploy.yml/badge.svg)](https://github.com/fadyabirached/face-recognition/actions/workflows/deploy.yml)

A production-grade face recognition system built with **FaceNet**, **SVM**, and **Gradio**.  
Trained on the [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — achieving **98.18% test accuracy** across 62 identities.

---

## Architecture

```
Input Image
    │
    ▼
 MTCNN ──────────────── Face detection & alignment → 160×160
    │
    ▼
 InceptionResnetV1 ──── FaceNet embedding → 512-dim vector
    │
    ▼
 SVM (RBF, C=10) ─────── Classification → identity + confidence
    │
    ▼
 Gradio UI ──────────── Annotated image + confidence charts
```

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **98.18%** |
| Macro F1-Score | **98.85%** |
| Weighted F1-Score | **98.14%** |
| Val → Test Gap | **1.41%** (no overfitting) |
| Identities | 62 |
| Embedding Dim | 512 |

---

<img width="1301" height="722" alt="Screenshot 2026-05-10 130746" src="https://github.com/user-attachments/assets/367b8855-8235-43bd-b378-e25abb595817" />
<img width="1300" height="678" alt="Screenshot 2026-05-10 130818" src="https://github.com/user-attachments/assets/299824f4-8bc1-4b30-98e7-13d6d3395eb3" />
<img width="1242" height="601" alt="Screenshot 2026-05-10 130844" src="https://github.com/user-attachments/assets/c4ed0de1-85d8-49a5-bf20-382fc34a5813" />

---

## Project Structure

```
face-recognition/
├── app.py                       # Gradio application (UI + inference)
├── config.py                    # All configuration & hyperparameters
├── face-detection.ipynb         # Data prep, training & evaluation notebook
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Copy to .env to override config.py defaults
├── .gitignore                   # Git ignore rules
├── tests/                       # Model-free unit tests (pytest)
├── .github/workflows/ci.yml     # Lint + test on every push/PR to main
├── .github/workflows/deploy.yml # Build & push Docker image to Azure Container Registry on push to main
├── Dockerfile                   # CPU-only image for serving the Gradio app
├── .dockerignore                # Keeps dataset/venv/notebooks out of the image
├── README.md                    # This file
├── face_recognition_model.pkl   # Trained SVM classifier — committed, see below
└── idx2label.npy                # Class index → identity name mapping — committed, see below
```

> **`face_recognition_model.pkl` and `idx2label.npy` are committed to the repo on purpose.**
> They're trained artifacts, not source code, but the app and the Docker image need them
> present to run at all — committing them means a fresh clone or `docker build` works
> out of the box instead of requiring a Colab training run before anything is usable.
> If you want to train on a different dataset or identities, see
> [Reproducing the model](#reproducing-the-model) below.

---

## Quickstart

The trained model files are already in the repo, so a fresh clone runs as-is —
no training step required first.

### 1. Clone & enter the project
```bash
git clone https://github.com/fadyabirached/face-recognition.git
cd face-recognition
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)
```bash
cp .env.example .env            # edit if you want non-default paths/port
```

### 5. Run
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

Want to train on a different dataset or set of identities instead of the ones
already baked in? See [Reproducing the model](#reproducing-the-model).

---

## Reproducing the model

The committed `face_recognition_model.pkl` / `idx2label.npy` were trained on the 62 LFW
identities described below. To retrain on a different dataset or set of identities,
regenerate them yourself by running `face-detection.ipynb` end to end. The notebook was
written for **Google Colab** (it uses `google.colab.files` for upload/download), so the
easiest path is to run it there.

### What you need
- A free [Google Colab](https://colab.research.google.com/) account (a T4 GPU runtime
  makes embedding extraction much faster than CPU, but isn't required).
- The **[LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)** from
  Kaggle, downloaded as a zip (`FaceDetectionDataset.zip` in the notebook, ~180 MB).

### Steps
1. Open `face-detection.ipynb` in Colab.
2. Run the first cell to install extra dependencies (`facenet-pytorch`, `opencv-python`,
   `scikit-learn`, `matplotlib`, `joblib`, `ImageHash`) — everything else Colab ships with.
3. Run the upload cell and select the LFW zip you downloaded from Kaggle. The next cell
   extracts it to `lfw_dataset/`.
4. Run the dataset-exploration and pHash de-duplication cells. These filter to persons
   with ≥ 20 images (62 identities) and drop corrupted/duplicate images.
5. Run the **embedding extraction** cell — MTCNN aligns every face to 160×160, FaceNet
   (`InceptionResnetV1`, VGGFace2 weights) encodes each into a 512-dim vector, with 5×
   augmentation (flip, brightness ±, contrast +) applied to the train split only. This is
   the slowest step: roughly 10–20 minutes on a Colab T4 GPU, longer on CPU. Saves
   `embeddings.npz`.
6. Run the **classifier training** cell — trains an SVM (RBF, C=10) and an MLP on the
   embeddings, picks whichever scores higher on the held-out test split, and saves
   `face_recognition_model.pkl`. Takes a few minutes.
7. Run the **full evaluation** cell for the confusion matrix / classification report
   (optional, doesn't produce artifacts you need for the app).
8. Run the final cell (**"Download all files needed for local VS Code app"**) — it derives
   `idx2label.npy` from the saved embeddings and downloads both `face_recognition_model.pkl`
   and `idx2label.npy` to your machine via the browser.
9. Replace the committed `face_recognition_model.pkl` and `idx2label.npy` in the project
   root with the two downloaded files.

### Expected artifact sizes
Both files are small — the SVM classifier is typically a few MB (it stores support
vectors over 512-dim embeddings, not raw images) and the label map is a few KB. This is
**not** the same as the FaceNet backbone: `InceptionResnetV1(pretrained="vggface2")` is
~107 MB and is downloaded automatically by `facenet-pytorch` on first run, cached locally
(e.g. `~/.cache/torch/checkpoints`) — it's never stored in this repo either.

### Training on your own faces
The pipeline isn't LFW-specific. Point `DATASET_PATH` in the notebook at any folder laid
out as `<dataset>/<person_name>/<image files>` with ≥ `MIN_IMAGES_PER_PERSON` images per
person, and re-run cells 3 onward.

---

## Testing & CI

`tests/` covers everything that doesn't need the trained model files or the dataset:
`config.py` defaults/env overrides, and `app.py`'s pure logic (chart builders, face-count/
confidence-threshold control flow) with MTCNN/FaceNet/SVM replaced by lightweight mocks.

```bash
pip install pytest ruff
pytest -v                              # run the test suite
ruff check app.py config.py benchmark.py tests/     # lint
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push/PR to `main`. CI never
downloads the dataset or the trained model — it only exercises the model-free code paths.

---

## Performance

`benchmark.py` times each stage of the inference pipeline — MTCNN detect, MTCNN align/crop,
FaceNet embedding, SVM classification — the same code path `app.py` runs, on a handful of
warm-up-excluded runs so model-load time doesn't pollute the numbers.

```bash
python benchmark.py path/to/image1.jpg path/to/image2.jpg
```

Single-threaded CPU (matching the Dockerfile's deployed environment), on 100 runs after a
10-run warm-up. Detection cost scales with input pixel count (MTCNN's image-pyramid search),
so the first version of this benchmark ran against uncapped high-res images:

| Stage | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| MTCNN detect | 242.8 ms | 264.8 ms | 323.9 ms | 330.2 ms |
| MTCNN align/crop | 59.5 ms | 59.2 ms | 73.5 ms | 77.4 ms |
| FaceNet embed | 49.2 ms | 48.7 ms | 54.1 ms | 56.8 ms |
| SVM classify | 3.4 ms | 3.3 ms | 3.7 ms | 4.3 ms |
| **End-to-end (single face)** | **355.0 ms** | **374.3 ms** | **429.1 ms** | **437.1 ms** |

Since detection was ~68% of total latency and MTCNN's cost is driven by input resolution,
`app.py` now downscales uploads above `config.MAX_INPUT_DIM` (720px long edge, aspect
preserved) before detection — high enough that faces at typical webcam/upload framing stay
well above `MIN_FACE_SIZE`, low enough to cut the dominant cost:

| Stage | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| MTCNN detect | 94.2 ms | 102.6 ms | 124.0 ms | 127.1 ms |
| MTCNN align/crop | 33.9 ms | 33.8 ms | 38.9 ms | 45.1 ms |
| FaceNet embed | 50.1 ms | 49.6 ms | 54.3 ms | 61.1 ms |
| SVM classify | 3.6 ms | 3.5 ms | 4.0 ms | 4.6 ms |
| **End-to-end (single face)** | **181.9 ms** | **181.7 ms** | **211.9 ms** | **217.0 ms** |

**Result: 355ms → 182ms mean (-49%), throughput 2.8 → 5.5 faces/sec, single-threaded CPU.**

FaceNet embedding is now the largest single stage (~50ms) and is effectively a fixed cost —
it always processes a constant 160×160 crop, so it doesn't shrink with `MAX_INPUT_DIM`.
Cutting further would mean quantizing/compiling the FaceNet backbone or moving to GPU, not
just downscaling input; on CPU, ~180ms end-to-end is close to the floor for this pipeline as
architected. These numbers are single-threaded; exact milliseconds will vary by host, but the
relative breakdown across stages holds.

---

## CI/CD & Deployment

Two independent GitHub Actions workflows run on this repo:

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **CI** | `.github/workflows/ci.yml` | push/PR to `main`, manual | Installs deps, runs `ruff check`, runs `pytest -v` |
| **Build and Push to ACR** | `.github/workflows/deploy.yml` | push to `main`, manual | Builds the Docker image and pushes it to Azure Container Registry |

### Docker image

The `Dockerfile` builds a CPU-only image:
- Base: `python:3.11-slim` + the system libs OpenCV needs (`libgl1`, `libglib2.0-0`).
- Installs CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) in its own
  layer, so the image doesn't balloon past 6GB pulling in CUDA builds.
- Bakes the FaceNet VGGFace2 weights (~110MB) into the image at build time, so containers
  start instantly instead of re-downloading them on every cold start.
- Serves the Gradio app on `0.0.0.0:7860` (`EXPOSE 7860`, `CMD ["python", "app.py"]`).

Build and run it locally:
```bash
docker build -t face-recognition .
docker run -p 7860:7860 face-recognition
```

> `face_recognition_model.pkl` and `idx2label.npy` are committed to the repo, so `COPY . .`
> bakes them into the image automatically — no extra mount step needed at runtime.

### Cloud deployment (Azure Container Registry)

On every push to `main`, `deploy.yml`:
1. Logs into the registry `fadyfacerecog.azurecr.io` using the `ACR_USERNAME` / `ACR_PASSWORD`
   repo secrets.
2. Builds the image from the `Dockerfile` and pushes two tags:
   - `fadyfacerecog.azurecr.io/face-recognition:latest`
   - `fadyfacerecog.azurecr.io/face-recognition:<git-sha>` (immutable, one per commit)

This workflow only builds and publishes the image — it doesn't itself deploy to a running
service (e.g. an Azure Container App/App Service pulling `:latest` is a separate, manually
configured step outside this repo).

---

## Configuration

All parameters are in `config.py` and overridable via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `face_recognition_model.pkl` | Path to trained SVM |
| `LABELS_PATH` | `idx2label.npy` | Path to label map |
| `SERVER_NAME` | `0.0.0.0` | Gradio server bind address |
| `SERVER_PORT` | `7860` | Gradio server port |
| `SHARE` | `false` | Set `true` for public Gradio link |
| `DETECT_THRESH` | `0.85` | MTCNN minimum confidence |
| `MAX_INPUT_DIM` | `720` | Uploads above this (long edge, px) are downscaled before detection — see [Performance](#performance) |
| `DEFAULT_TOP_K` | `6` | Candidates shown per face |
| `MIN_MARGIN_RATIO` | `1.5` | Top-1 must be this many times more confident than the runner-up to be accepted (see below) |
| `MIN_ABS_CONFIDENCE` | `3.0` | Top-1 below this confidence (%) is rejected outright, regardless of margin |

---

## Handling unrecognized faces

The SVM is a **closed-set** classifier — it always outputs *some* class, even for a
face that isn't any of the 62 trained identities. To tell "not in the dataset" apart
from "in the dataset but genuinely hard to classify," the app rejects on the **margin**
between the top-1 and top-2 candidate confidences, not on raw top-1 confidence.

Raw confidence alone doesn't work here: it's diluted by how many classes are competing
and by how many training images a given identity had. A correct match for someone with
few training photos can legitimately score a low absolute confidence — e.g. an identity
with only a handful of training images might come back at ~20-25% — while still being
clearly the best candidate. Rejecting on raw confidence alone would wrongly label that
person `Unknown`. A genuinely unrecognized face, by contrast, tends to produce several
similarly-weak candidates with no clear winner, which shows up as a small margin between
1st and 2nd place regardless of the absolute numbers.

So a face is labeled with a name only if:
1. the top candidate's confidence is at least `MIN_ABS_CONFIDENCE` (guards against a
   near-uniform, all-noise distribution where a tiny margin could still exist by chance), **and**
2. the top candidate is at least `MIN_MARGIN_RATIO`× more confident than the runner-up.

Otherwise the face is labeled `Unknown` — shown in a neutral color in the annotated
image and charts instead of one of the per-face identity colors, so a rejected face
isn't visually mistaken for a confident identification. The full ranked candidate list
is still shown either way, so you can see what it *almost* matched to.

This is a threshold heuristic, not calibrated open-set verification (no rejection ROC
curve, no per-class thresholds) — a reasonable stopgap for a demo app, not a substitute
for a proper open-set evaluation.

---

## Pipeline Details

### Face Detection — MTCNN
Multi-task Cascaded Convolutional Network. Detects, aligns, and crops faces to 160×160px. Supports multiple faces per image.

### Feature Extraction — FaceNet
`InceptionResnetV1` pre-trained on VGGFace2 (3.3M images). Used as a **fixed feature extractor** — no fine-tuning needed. Outputs an L2-normalized 512-dimensional embedding per face.

### Classification — SVM (RBF)
Support Vector Machine with RBF kernel (C=10, gamma='scale'). Trained on augmented FaceNet embeddings (5× augmentation: flip, brightness ±, contrast +). Outputs per-class probabilities for Top-K display.

### Data Augmentation (train only)
| Transform | Value |
|---|---|
| Horizontal flip | Mirror |
| Brightness boost | +20% |
| Brightness drop | −15% |
| Contrast boost | +15% |

---

## Dataset

**LFW (Labeled Faces in the Wild)**  
- 5,749 total persons  
- Filtered to persons with ≥ 20 images → **62 eligible identities**  
- Duplicate removal via **perceptual hashing (pHash)**  
- Train / Validation / Test split: **70 / 10 / 20**

---

## Tech Stack

| Component | Library |
|---|---|
| Face detection | `facenet-pytorch` (MTCNN) |
| Face embedding | `facenet-pytorch` (InceptionResnetV1) |
| Classification | `scikit-learn` (SVC) |
| Deep learning | `PyTorch` |
| UI | `Gradio` |
| Visualization | `Matplotlib` |
| Image processing | `Pillow`, `OpenCV` |
| Testing / CI | `pytest`, `ruff`, GitHub Actions |

---

## Author
Built as an academic project - Face Recognition System using LFW Dataset.
