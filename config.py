#  config.py - Central Configuration
import os
from dotenv import load_dotenv

load_dotenv()   # loads variables from .env file

# Device
# Deployment (Dockerfile, ACR) is CPU-only by design — see README ->
# "Docker image". Override via env only for local benchmarking against
# a GPU (e.g. DEVICE=cuda python benchmark.py ..., in Colab); the
# deployed app never sets this and stays on cpu.
DEVICE = os.getenv("DEVICE", "cpu")

# Paths
MODEL_PATH  = os.getenv("MODEL_PATH",  "face_recognition_model.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "idx2label.npy")

# Model settings
IMG_SIZE       = 160      # FaceNet requirement — do not change
FACE_MARGIN    = 20       # pixels of context around detected face
MIN_FACE_SIZE  = 30       # minimum detectable face size in pixels
DETECT_THRESH  = 0.85     # minimum MTCNN detection confidence
EMBEDDING_DIM  = 512      # FaceNet output dimension

# MTCNN's detection cost scales with input pixel count (image-pyramid
# search), so uploads larger than this are downscaled (long edge, aspect
# preserved) before detection. 720px keeps faces at typical webcam/upload
# distance comfortably above MIN_FACE_SIZE while cutting detection latency
# substantially versus raw high-res photos. See README -> "Performance".
MAX_INPUT_DIM  = int(os.getenv("MAX_INPUT_DIM", 720))

# App settings
DEFAULT_TOP_K  = 6        # default number of candidate identities shown
MAX_TOP_K      = 10       # slider maximum
SERVER_NAME    = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT    = int(os.getenv("SERVER_PORT", 7860))
SHARE          = os.getenv("SHARE", "false").lower() == "true"

# Unknown-face rejection
#
# The SVM is a closed-set classifier: it always picks *some* class, even
# for a face that isn't any of the trained identities. Rejecting on raw
# top-1 confidence alone doesn't work well here, because that confidence
# is diluted by how many classes are competing and by how many training
# images a given identity had — a correct match for someone with few
# training photos can legitimately score a low absolute confidence while
# still being clearly the best candidate. What actually separates "a real
# but under-represented identity" from "not in the dataset at all" is the
# *margin* between the top and runner-up candidate: a genuine match tends
# to be decisively ahead of everyone else, while a true unknown produces
# several similarly-weak candidates with no clear winner.
UNKNOWN_LABEL       = "Unknown"
UNKNOWN_COLOR       = "#7A8699"
MIN_MARGIN_RATIO    = float(os.getenv("MIN_MARGIN_RATIO", 1.5))
# in percentage points (top_confs are already scaled to 0-100)
MIN_ABS_CONFIDENCE  = float(os.getenv("MIN_ABS_CONFIDENCE", 3.0))

# Visual theme
DARK_BG    = "#0A0F1E"
PANEL_BG   = "#0D1526"
NEON_GREEN = "#00FFB2"
NEON_BLUE  = "#00B4D8"
NEON_PINK  = "#FF4D6D"
NEON_GOLD  = "#FFD60A"
TEXT_COLOR = "#E8F4FD"

FACE_COLORS = [
    "#00FFB2", "#FF4D6D", "#FFD60A",
    "#00B4D8", "#FF9A3C", "#A9FF6B"
]
