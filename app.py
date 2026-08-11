#  FACE RECOGNITION SYSTEM
#  Requirements: pip install -r requirements.txt
#  Run: python app.py

import gradio as gr
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1

# Import all settings from config
from config import (
    DEVICE, MODEL_PATH, LABELS_PATH,
    DEFAULT_TOP_K, MAX_TOP_K,
    FACE_MARGIN, MIN_FACE_SIZE, DETECT_THRESH,
    SERVER_NAME, SERVER_PORT, SHARE,
    UNKNOWN_LABEL, UNKNOWN_COLOR, MIN_MARGIN_RATIO, MIN_ABS_CONFIDENCE,
    DARK_BG, PANEL_BG, NEON_GREEN, NEON_BLUE,
    NEON_PINK, TEXT_COLOR, FACE_COLORS
)


# STARTUP

print(f"\n{'='*50}")
print("  FACE RECOGNITION SYSTEM — Loading...")
print(f"{'='*50}")
print(f"  Device  : {DEVICE}")

mtcnn = MTCNN(
    image_size=160,
    margin=FACE_MARGIN,
    keep_all=True,
    min_face_size=MIN_FACE_SIZE,
    thresholds=[0.6, 0.7, 0.7],
    device=DEVICE
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
for p in resnet.parameters():
    p.requires_grad = False

model     = joblib.load(MODEL_PATH)
idx2label = list(np.load(LABELS_PATH, allow_pickle=True))

print(f"  Classes : {len(idx2label)} identities")
print(f"  Model   : {MODEL_PATH}")
print(f"{'='*50}\n")


# PREDICTION ENGINE

def decide_identity(
    top_names,
    top_confs,
    min_margin_ratio=MIN_MARGIN_RATIO,
    min_abs_confidence=MIN_ABS_CONFIDENCE,
):
    """Decide whether the top candidate is a confident match, or the face
    should be treated as unrecognized.

    Uses the *margin* between the top-1 and top-2 candidate confidences
    rather than raw top-1 confidence. Raw confidence is diluted by how
    many classes are competing and by how many training images a given
    identity had, so a correct match for someone with few training
    photos can legitimately score a low absolute confidence while still
    being clearly the best candidate — rejecting on raw confidence alone
    would wrongly call that person "Unknown". A genuinely unrecognized
    face, by contrast, tends to produce several similarly-weak
    candidates with no clear winner, which shows up as a small margin.

    `min_abs_confidence` is a secondary guard against the degenerate
    case where the whole distribution is near-uniform noise but a tiny
    margin still happens to exist between the top two entries.

    Returns (name, confidence, is_known).
    """
    if not top_names:
        return UNKNOWN_LABEL, 0.0, False

    top1_conf = top_confs[0]
    top2_conf = top_confs[1] if len(top_confs) > 1 else 0.0

    if top1_conf < min_abs_confidence:
        return UNKNOWN_LABEL, top1_conf, False

    margin_ratio = top1_conf / top2_conf if top2_conf > 0 else float("inf")
    if margin_ratio < min_margin_ratio:
        return UNKNOWN_LABEL, top1_conf, False

    return top_names[0], top1_conf, True


def run_recognition(pil_img, top_k=DEFAULT_TOP_K):
    """
    Full pipeline: PIL image -> detect faces -> embed -> classify.
    Returns annotated image + per-face result list.
    """
    if pil_img is None:
        return None, None, "No image provided."

    img_rgb = pil_img.convert("RGB")

    # Detect all faces
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        return pil_img, None, "No faces detected. Try a clearer image."

    # Filter low confidence detections
    valid = [(b, p) for b, p in zip(boxes, probs) if p >= DETECT_THRESH]
    if not valid:
        return pil_img, None, "Faces found but confidence too low."

    boxes, probs = zip(*valid)

    # Draw bounding boxes on image
    annotated = img_rgb.copy()
    draw      = ImageDraw.Draw(annotated)
    face_results = []

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Crop face with margin
        fx1 = max(0, x1 - FACE_MARGIN)
        fy1 = max(0, y1 - FACE_MARGIN)
        fx2 = min(img_rgb.width,  x2 + FACE_MARGIN)
        fy2 = min(img_rgb.height, y2 + FACE_MARGIN)
        face_crop = img_rgb.crop((fx1, fy1, fx2, fy2))

        # Extract embedding
        with torch.no_grad():
            face_tensor = mtcnn(face_crop)
            if face_tensor is None:
                continue
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor[0:1].to(DEVICE)
            embedding   = resnet(face_tensor).cpu().numpy()

        # Classify
        proba     = model.predict_proba(embedding)[0]
        top_idx   = np.argsort(proba)[::-1][:top_k]
        top_names = [idx2label[j].replace("_", " ") for j in top_idx]
        top_confs = [float(proba[j]) * 100 for j in top_idx]

        best_name, best_conf, is_known = decide_identity(top_names, top_confs)

        # Colour reflects the identity decision, not just detection: a
        # confident, known match gets its own colour from the palette;
        # a rejected/unrecognized face is drawn in a neutral colour so
        # it isn't mistaken for a confident identification.
        color = FACE_COLORS[i % len(FACE_COLORS)] if is_known else UNKNOWN_COLOR

        # Glow effect
        for thickness, alpha in [(8, 40), (5, 80), (2, 180), (1, 255)]:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # Corner brackets (cyberpunk HUD style)
        bracket = 20
        for (cx, cy, sx, sy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            draw.line([cx, cy, cx + sx*bracket, cy], fill=color, width=3)
            draw.line([cx, cy, cx, cy + sy*bracket], fill=color, width=3)

        # Face number label
        draw.rectangle([x1, y1-24, x1+28, y1], fill=color)
        draw.text((x1+4, y1-22), f"#{i+1}", fill="#000000")

        face_results.append({
            "face_num"    : i + 1,
            "color"       : color,
            "detect_conf" : float(prob) * 100,
            "crop"        : face_crop,
            "top_names"   : top_names,
            "top_confs"   : top_confs,
            "best_name"   : best_name,
            "best_conf"   : best_conf,
            "is_known"    : is_known,
        })

    return annotated, face_results, f"  {len(face_results)} face(s) detected"


# CHART GENERATORS

def make_confidence_chart(face_results):
    """Horizontal bar chart — top-K confidence per face."""
    if not face_results:
        return None

    n_faces = len(face_results)
    fig, axes = plt.subplots(1, n_faces, figsize=(7 * n_faces, 5), facecolor=DARK_BG)
    if n_faces == 1:
        axes = [axes]

    for ax, result in zip(axes, face_results):
        names = result["top_names"]
        confs = result["top_confs"]
        color = result["color"]

        bar_colors = [color if i == 0 else "#1A2744" for i in range(len(names))]

        bars = ax.barh(
            range(len(names))[::-1],
            confs,
            color=bar_colors,
            edgecolor=DARK_BG,
            linewidth=1.5,
            height=0.65
        )

        # Value labels on bars
        for bar, conf in zip(bars, confs):
            ax.text(
                min(conf + 1.5, 97), bar.get_y() + bar.get_height()/2,
                f"{conf:.1f}%",
                va="center", ha="left",
                color=TEXT_COLOR, fontsize=10,
                fontweight="bold" if conf == confs[0] else "normal",
                fontfamily="monospace"
            )

        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names, color=TEXT_COLOR, fontsize=10, fontfamily="monospace")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Confidence %", color=TEXT_COLOR, fontsize=10, labelpad=8)
        ax.set_title(
            f"FACE #{result['face_num']}  —  {result['best_name'].upper()}",
            color=color, fontsize=12, fontweight="bold",
            pad=14, fontfamily="monospace"
        )
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.spines[:].set_color("#1A3050")
        ax.axvline(50, color=NEON_PINK, linestyle="--", linewidth=0.8, alpha=0.5)

        # Detection confidence badge
        ax.text(
            0.98, 0.02,
            f"Detection: {result['detect_conf']:.1f}%",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color="#888888", fontsize=8,
            fontfamily="monospace"
        )

    fig.suptitle(
        "NEURAL FACE RECOGNITION  —  CONFIDENCE ANALYSIS",
        color=NEON_GREEN, fontsize=13, fontweight="bold",
        fontfamily="monospace", y=1.02
    )
    plt.tight_layout()
    return fig


def make_radar_chart(face_results):
    """Radar / polar chart comparing top identities across faces."""
    if not face_results:
        return None

    fig = plt.figure(figsize=(6 * len(face_results), 5), facecolor=DARK_BG)

    for fi, result in enumerate(face_results):
        ax = fig.add_subplot(
            1, len(face_results), fi + 1,
            polar=True, facecolor=PANEL_BG
        )
        names = result["top_names"][:5]
        confs = result["top_confs"][:5]
        N     = len(names)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals   = confs + confs[:1]

        ax.fill(angles, vals, alpha=0.25, color=result["color"])
        ax.plot(angles, vals, color=result["color"], linewidth=2, linestyle="solid")
        ax.scatter(angles[:-1], confs, color=result["color"], s=60, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [n.split()[-1] for n in names],
            color=TEXT_COLOR, fontsize=9, fontfamily="monospace"
        )
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], color="#555555", fontsize=7)
        ax.grid(color="#1A3050", linewidth=0.8)
        ax.spines["polar"].set_color("#1A3050")
        ax.set_title(
            f"FACE #{result['face_num']}",
            color=result["color"], fontsize=11,
            fontweight="bold", pad=16, fontfamily="monospace"
        )

    fig.suptitle(
        "IDENTITY CONFIDENCE RADAR",
        color=NEON_BLUE, fontsize=13,
        fontweight="bold", fontfamily="monospace"
    )
    plt.tight_layout()
    return fig


def make_face_strip(face_results):
    """Row of detected face crops with labels."""
    if not face_results:
        return None

    n = len(face_results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), facecolor=DARK_BG)
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, face_results):
        crop = result["crop"].resize((180, 180))
        ax.imshow(crop)
        ax.set_title(
            f"#{result['face_num']}  {result['best_name']}\n"
            f"{result['best_conf']:.1f}% confidence",
            color=result["color"], fontsize=9,
            fontweight="bold", fontfamily="monospace"
        )
        # Neon border
        for spine in ax.spines.values():
            spine.set_edgecolor(result["color"])
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PANEL_BG)

    fig.suptitle(
        "DETECTED FACES",
        color=NEON_GREEN, fontsize=12,
        fontweight="bold", fontfamily="monospace"
    )
    plt.tight_layout()
    return fig


# GRADIO HANDLER

def process_image(image, top_k_slider):
    """Main Gradio handler — runs full pipeline, returns all outputs."""
    if image is None:
        return (None, None, None, None, "### Please upload an image first.")

    pil_img = Image.fromarray(image).convert("RGB")
    annotated, face_results, status = run_recognition(pil_img, int(top_k_slider))

    if not face_results:
        return (np.array(annotated), None, None, None, f"### {status}")

    conf_fig  = make_confidence_chart(face_results)
    radar_fig = make_radar_chart(face_results)
    strip_fig = make_face_strip(face_results)

    # Build markdown summary
    summary_lines = ["## Recognition Results\n"]
    summary_lines.append(f"**{len(face_results)} face(s) detected**\n")
    summary_lines.append("---")

    for r in face_results:
        if r["is_known"]:
            bar = "█" * int(r["best_conf"] / 10) + "░" * (10 - int(r["best_conf"] / 10))
            summary_lines.append(
                f"### Face #{r['face_num']}\n"
                f"**Best match:** `{r['best_name']}`  \n"
                f"**Confidence:** `{bar}` {r['best_conf']:.1f}%  \n"
                f"**Detection score:** {r['detect_conf']:.1f}%\n"
            )
        else:
            summary_lines.append(
                f"### Face #{r['face_num']}\n"
                f"**Result:** `{UNKNOWN_LABEL}` — no confident match in the training set  \n"
                f"**Detection score:** {r['detect_conf']:.1f}%\n"
            )
        summary_lines.append("**Top candidates:**")
        for name, conf in zip(r["top_names"][:4], r["top_confs"][:4]):
            marker = "▶" if conf == r["top_confs"][0] else "  "
            summary_lines.append(f"- {marker} `{name}` — {conf:.2f}%")
        summary_lines.append("---")

    return (
        np.array(annotated),
        conf_fig,
        radar_fig,
        strip_fig,
        "\n".join(summary_lines)
    )


def handle_click(image, top_k):
    """Button click handler — wraps process_image and updates status bar."""
    ann, conf, radar, strip, summary = process_image(image, top_k)
    status_html = (
        '<div class="status-bar">  Processing complete</div>'
        if conf is not None else
        '<div class="status-bar" style="border-color:#FF4D6D;color:#FF4D6D;">'
        '  No faces detected</div>'
    )
    return ann, conf, radar, strip, summary, status_html


# GRADIO UI

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

* { font-family: 'Rajdhani', sans-serif !important; }
code, .monospace { font-family: 'Share Tech Mono', monospace !important; }

body, .gradio-container {
    background: #050D1A !important;
    color: #E8F4FD !important;
}
.app-header {
    background: linear-gradient(135deg, #050D1A 0%, #0A1F3C 50%, #050D1A 100%);
    border-bottom: 1px solid #00FFB222;
    padding: 32px 40px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-linear-gradient(90deg, transparent, transparent 60px, #00FFB208 60px, #00FFB208 61px),
        repeating-linear-gradient(0deg,  transparent, transparent 60px, #00FFB208 60px, #00FFB208 61px);
    pointer-events: none;
}
.app-title {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    color: #00FFB2 !important;
    text-shadow: 0 0 30px #00FFB255, 0 0 60px #00FFB222 !important;
    margin: 0 !important;
}
.app-subtitle {
    color: #4A8FA8 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    margin-top: 6px !important;
}
.panel {
    background: #0D1526 !important;
    border: 1px solid #1A3050 !important;
    border-radius: 8px !important;
}
.upload-box {
    border: 2px dashed #00FFB255 !important;
    border-radius: 8px !important;
    background: #080F1D !important;
    transition: border-color 0.3s !important;
}
.upload-box:hover { border-color: #00FFB2 !important; }
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, #00FFB2, #00B4D8) !important;
    color: #050D1A !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px #00FFB233 !important;
}
button.primary-btn:hover {
    box-shadow: 0 0 35px #00FFB266 !important;
    transform: translateY(-1px) !important;
}
input[type=range] { accent-color: #00FFB2 !important; }
.tab-nav button {
    color: #4A8FA8 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
}
.tab-nav button.selected {
    color: #00FFB2 !important;
    border-bottom-color: #00FFB2 !important;
}
.results-md {
    background: #080F1D !important;
    border: 1px solid #1A3050 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.results-md h2 { color: #00FFB2 !important; }
.results-md h3 { color: #00B4D8 !important; }
.results-md code { color: #FFD60A !important; background: #0A1F3C !important; }
.status-bar {
    background: #0A1F3C !important;
    border-left: 3px solid #00FFB2 !important;
    padding: 8px 16px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #00FFB2 !important;
}
img { border-radius: 6px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050D1A; }
::-webkit-scrollbar-thumb { background: #1A3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00FFB255; }
"""

# Build the interface
# NOTE: Gradio 6.0 moved `css` from the Blocks() constructor to launch().
# It is passed in demo.launch() at the bottom of this file.
with gr.Blocks(title="Face Recognition System") as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
            <div class="app-title">⬡ FACE RECOGNITION SYSTEM</div>
            <div class="app-subtitle">
                FaceNet · VGGFace2 · SVM · 98.18% Accuracy · LFW Dataset
            </div>
        </div>
    """)

    # Stats bar — values pulled from config at runtime
    gr.HTML(f"""
        <div style="display:flex; gap:24px; padding:16px 24px;
                    background:#080F1D; border-bottom:1px solid #1A3050;
                    font-family:'Share Tech Mono',monospace; font-size:0.78rem;">
            <span style="color:#00FFB2">◈ MODEL: SVM (RBF)</span>
            <span style="color:#4A8FA8">|</span>
            <span style="color:#00B4D8">◈ BACKBONE: InceptionResnetV1</span>
            <span style="color:#4A8FA8">|</span>
            <span style="color:#FFD60A">◈ EMBEDDINGS: 512-DIM</span>
            <span style="color:#4A8FA8">|</span>
            <span style="color:#FF4D6D">◈ CLASSES: {len(idx2label)}</span>
            <span style="color:#4A8FA8">|</span>
            <span style="color:#A9FF6B">◈ TEST ACC: 98.18%</span>
            <span style="color:#4A8FA8">|</span>
            <span style="color:#888888">◈ DEVICE: {DEVICE.upper()}</span>
        </div>
    """)

    with gr.Row():

        # Left column: input
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("""
                <div style="padding:16px 20px 8px; font-family:'Share Tech Mono',monospace;
                            color:#00FFB2; font-size:0.85rem; letter-spacing:0.15em;">
                    ▸ INPUT IMAGE
                </div>
            """)

            img_input = gr.Image(
                label="Upload or drag an image",
                type="numpy",
                elem_classes=["upload-box"],
                height=320,
            )

            top_k_slider = gr.Slider(
                minimum=3, maximum=MAX_TOP_K, value=DEFAULT_TOP_K, step=1,
                label="Top-K candidates to show",
                info="How many identity candidates to display per face"
            )

            run_btn = gr.Button(
                "⬡  IDENTIFY FACES",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            status_md = gr.HTML(
                '<div class="status-bar">Ready — upload an image to begin</div>'
            )

            gr.HTML("""
                <div style="padding:12px 20px; font-size:0.78rem;
                            color:#4A8FA8; font-family:'Share Tech Mono',monospace;
                            border-top:1px solid #1A3050; margin-top:8px;">
                    TIP: Works best with clear frontal face photos.<br>
                    Multiple faces in one image are supported.
                </div>
            """)

        # Right column: outputs
        with gr.Column(scale=2):
            with gr.Tabs():

                with gr.TabItem("🖼  Annotated Image"):
                    img_output = gr.Image(
                        label="Detected Faces with Bounding Boxes",
                        type="numpy",
                        height=420,
                    )

                with gr.TabItem("📊  Confidence Bars"):
                    conf_plot = gr.Plot(label="Top-K Confidence per Face")

                with gr.TabItem("🎯  Radar Chart"):
                    radar_plot = gr.Plot(label="Identity Confidence Radar")

                with gr.TabItem("👤  Face Crops"):
                    strip_plot = gr.Plot(label="Detected Face Crops")

                with gr.TabItem("📋  Summary"):
                    results_md = gr.Markdown(
                        value="*Results will appear here after processing...*",
                        elem_classes=["results-md"]
                    )

    # How it works section
    gr.HTML("""
        <div style="padding:20px 24px 8px; font-family:'Share Tech Mono',monospace;
                    color:#00B4D8; font-size:0.82rem; letter-spacing:0.12em;
                    border-top:1px solid #1A3050; margin-top:16px;">
            ▸ HOW IT WORKS
        </div>
    """)

    gr.HTML(f"""
        <div style="display:flex; gap:0; padding:0 24px 24px;">
            <div style="flex:1; padding:16px; background:#080F1D;
                        border:1px solid #1A3050; border-right:none;">
                <div style="color:#00FFB2; font-size:1.5rem; margin-bottom:8px;">①</div>
                <div style="color:#00FFB2; font-weight:600; font-size:0.9rem;">DETECT</div>
                <div style="color:#4A8FA8; font-size:0.8rem; margin-top:4px;">
                    MTCNN locates and aligns every face in the image to 160×160px
                </div>
            </div>
            <div style="flex:1; padding:16px; background:#080F1D;
                        border:1px solid #1A3050; border-right:none;">
                <div style="color:#FFD60A; font-size:1.5rem; margin-bottom:8px;">②</div>
                <div style="color:#FFD60A; font-weight:600; font-size:0.9rem;">EMBED</div>
                <div style="color:#4A8FA8; font-size:0.8rem; margin-top:4px;">
                    FaceNet (InceptionResnetV1) encodes each face into a 512-dim vector
                </div>
            </div>
            <div style="flex:1; padding:16px; background:#080F1D;
                        border:1px solid #1A3050; border-right:none;">
                <div style="color:#FF4D6D; font-size:1.5rem; margin-bottom:8px;">③</div>
                <div style="color:#FF4D6D; font-weight:600; font-size:0.9rem;">CLASSIFY</div>
                <div style="color:#4A8FA8; font-size:0.8rem; margin-top:4px;">
                    SVM (RBF, C=10) maps the embedding to one of {len(idx2label)} identities
                </div>
            </div>
            <div style="flex:1; padding:16px; background:#080F1D; border:1px solid #1A3050;">
                <div style="color:#A9FF6B; font-size:1.5rem; margin-bottom:8px;">④</div>
                <div style="color:#A9FF6B; font-weight:600; font-size:0.9rem;">VISUALIZE</div>
                <div style="color:#4A8FA8; font-size:0.8rem; margin-top:4px;">
                    Top-K predictions displayed with confidence scores and charts
                </div>
            </div>
        </div>
    """)

    # Wire up buttons
    run_btn.click(
        fn=handle_click,
        inputs=[img_input, top_k_slider],
        outputs=[img_output, conf_plot, radar_plot, strip_plot, results_md, status_md]
    )

    img_input.change(
        fn=lambda _: '<div class="status-bar" style="color:#FFD60A;">'
                     'Image loaded — click Identify</div>',
        inputs=[img_input],
        outputs=[status_md]
    )


# Launch
if __name__ == "__main__":
    demo.launch(
        css=CUSTOM_CSS,
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE,
        show_error=True,
        favicon_path=None,
    )