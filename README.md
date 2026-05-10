# ⬡ Face Recognition System

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
face_recognition/
├── app.py                       # Gradio application (UI + inference)
├── config.py                    # All configuration & hyperparameters
├── requirements.txt             # Pinned dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── face_recognition_model.pkl   # Trained SVM classifier
└── idx2label.npy                # Class index → identity name mapping
```

---

## Quickstart

### 1. Clone & enter the project
```bash
git clone <your-repo-url>
cd face_recognition
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
Place these two files in the project root:
- `face_recognition_model.pkl`
- `idx2label.npy`

### 5. Configure environment
```bash
cp .env.example .env            # edit if needed
```

### 6. Run
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## Configuration

All parameters are in `config.py` and overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `face_recognition_model.pkl` | Path to trained SVM |
| `LABELS_PATH` | `idx2label.npy` | Path to label map |
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

---

## Author
Built as an academic project - Face Recognition System using LFW Dataset.
