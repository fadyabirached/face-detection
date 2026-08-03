#  config.py - Central Configuration
import os
from dotenv import load_dotenv

load_dotenv()   # loads variables from .env file

# Device
DEVICE = "cpu"

# Paths
MODEL_PATH  = os.getenv("MODEL_PATH",  "face_recognition_model.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "idx2label.npy")

# Model settings
IMG_SIZE       = 160      # FaceNet requirement — do not change
FACE_MARGIN    = 20       # pixels of context around detected face
MIN_FACE_SIZE  = 30       # minimum detectable face size in pixels
DETECT_THRESH  = 0.85     # minimum MTCNN detection confidence
EMBEDDING_DIM  = 512      # FaceNet output dimension

# App settings
DEFAULT_TOP_K  = 6        # default number of candidate identities shown
MAX_TOP_K      = 10       # slider maximum
SERVER_NAME    = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT    = int(os.getenv("SERVER_PORT", 7860))
SHARE          = os.getenv("SHARE", "false").lower() == "true"

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
