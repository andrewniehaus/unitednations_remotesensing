"""
Central Configuration Registry for Air-Gapped Geospatial ML Pipeline.
Manages absolute directory paths, local offline model weights,
CUDA runtime settings, and default tiling parameters across all sub-modules.
"""

from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# 1. ROOT & DIRECTORY PATHS
# -----------------------------------------------------------------------------
# Absolute path to the top-level repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Base Model Weights Directories (Local Air-Gapped Stores)
MODELS_DIR = PROJECT_ROOT / "models"
FLORENCE2_DIR = MODELS_DIR / "florence2"
GROUNDING_DINO_DIR = MODELS_DIR / "groundingdino"
CLAY_DIR = MODELS_DIR / "clayfoundation"

# Ensure runtime directories exist upon module load
for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FLORENCE2_DIR, GROUNDING_DINO_DIR, CLAY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. HARDWARE & RUNTIME SETTINGS
# -----------------------------------------------------------------------------
# Dynamic PyTorch execution target
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Floating-point precision selection for inference optimization on local GPU
# Defaulting to bfloat16 for RTX 4090 / A6000 architecture support; fall back to float32 on CPU
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# -----------------------------------------------------------------------------
# 3. GLOBAL TILING DEFAULTS
# -----------------------------------------------------------------------------
DEFAULT_TILE_SIZE = 512  # Pixel height/width for image patches
DEFAULT_STRIDE = 384     # Pixel step size (512 size with 384 stride gives 128px overlap)
DEFAULT_CRS = "EPSG:4326"  # Fallback spatial reference system

# Array Normalization Parameters
DEFAULT_RGB_BANDS = (0, 1, 2)  # Indices for Red, Green, Blue in your MS data (0-indexed)
DEFAULT_CLIP_PERCENTILE = (2.0, 98.0)  # Robust min/max scaling to ignore atmospheric outliers
