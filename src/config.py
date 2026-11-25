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

from pathlib import Path
from typing import Optional

def get_active_raw_image(raw_dir: Path) -> Optional[Path]:
    """
    Scans RAW_DIR for GeoTIFF files (.tif, .tiff), ignoring sidecar metadata.
    Returns the most recently modified image if multiple exist, or None if empty.
    """
    valid_extensions = {".tif", ".tiff"}
    tif_files = [
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not tif_files:
        return None
        
    # Sort by modification time descending so the newest drop is always selected
    tif_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return tif_files[0]

# Dynamically resolved active input image
ACTIVE_INPUT_IMAGE: Optional[Path] = get_active_raw_image(RAW_DIR)

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

# [EDIT 2026-09-28 | Claude Code for charliefp03-dg] Added settings referenced by chipper.py,
# batch_runner.py and vector_writer.py that were previously undefined (import-time AttributeError).
DEFAULT_PADDING_MODE = "constant"  # np.pad mode for partial edge tiles ('constant', 'reflect', 'edge')
DEFAULT_FILL_VALUE = 0             # Pad/NoData value for written tiles (constant padding mode)
MIN_CHIP_VALID_FRACTION = 0.05     # Chipper skips windows with less valid (non-NoData) area than this; SpatialTileFilter makes the stricter keep/drop decision
MANIFEST_FILENAME = "tile_manifest.json"  # Chipper manifest written to INTERIM_DIR
TILE_NAMING_SCHEMA = "{source_name}_r{row}_c{col}.tif"  # Tile filename template used by chipper
DEFAULT_OVERWRITE = False  # If False, existing inference JSONs are skipped (resumable runs); re-chipping always replaces tiles

# -----------------------------------------------------------------------------
# Target Prompt Profiles (Open-Vocabulary Detection)
# -----------------------------------------------------------------------------
TARGET_PROMPTS = {
    "disaster_response": [
        "building",
    ],
}

# Default prompt set for generic execution
DEFAULT_TARGETS = TARGET_PROMPTS["disaster_response"]
