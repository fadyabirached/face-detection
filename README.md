# 👤 Face Recognition System

[![CI](https://github.com/fadyabirached/face-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/face-detection/actions/workflows/ci.yml)

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
face-detection/
├── app.py                       # Gradio application (UI + inference)
├── config.py                    # All configuration & hyperparameters
├── face-detection.ipynb         # Data prep, training & evaluation notebook
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Copy to .env to override config.py defaults
├── .gitignore                   # Git ignore rules
├── tests/                       # Model-free unit tests (pytest)
├── .github/workflows/ci.yml     # Lint + test on every push
├── README.md                    # This file
├── face_recognition_model.pkl   # Trained SVM classifier — NOT in this repo, see below
└── idx2label.npy                # Class index → identity name mapping — NOT in this repo, see below
```

> **`face_recognition_model.pkl` and `idx2label.npy` are gitignored on purpose.**
> They are trained artifacts derived from the LFW dataset, not source code — committing
> a model binary without the data/training run that produced it would be misleading.
> Generate them yourself with the steps below (~15–30 min on a free Colab GPU), or
> use your own dataset to train different identities.

---

## Quickstart

There are two separate paths depending on what you want to do:

- **"I just want to see the app run"** → you still need model files, since there's no
  universal pretrained face-*identity* classifier to ship (identities are whoever was in
  your training set). Follow **[Reproducing the model](#reproducing-the-model)** first,
  then come back here.
- **"I already have `face_recognition_model.pkl` and `idx2label.npy`"** → follow the steps
  below directly.

### 1. Clone & enter the project
```bash
git clone https://github.com/fadyabirached/face-detection.git
cd face-detection
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

### 4. Add model files
Place these two files (produced by the notebook — see [Reproducing the model](#reproducing-the-model))
in the project root:
- `face_recognition_model.pkl`
- `idx2label.npy`

### 5. Configure environment (optional)
```bash
cp .env.example .env            # edit if you want non-default paths/port
```

### 6. Run
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## Reproducing the model

The trained classifier is tied to whichever identities you train it on, so it isn't
included in the repo — you generate it yourself by running `face-detection.ipynb` end
to end. The notebook was written for **Google Colab** (it uses `google.colab.files` for
upload/download), so the easiest path is to run it there.

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
9. Move both downloaded files into the project root (see [Quickstart](#quickstart) step 4).

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
ruff check app.py config.py tests/     # lint
```

GitHub Actions (`.github/workflows/ci.yml`) runs both on every push/PR to `main`. CI never
downloads the dataset or the trained model — it only exercises the model-free code paths.

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
| `DEFAULT_TOP_K` | `6` | Candidates shown per face |

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
