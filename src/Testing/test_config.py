# src/testing/test_config.py

import sys
from pathlib import Path

# Ensure src is in the path for absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def run_diagnostics():
    print("--- Hardware Diagnostics ---")
    print(f"Device Resolution: {config.DEVICE}")
    print(f"Torch Precision: {config.TORCH_DTYPE}")
    
    print("\n--- Path Resolution Diagnostics ---")
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Data/Interim Path: {config.INTERIM_DIR}")
    print(f"Models/Florence-2 Path: {config.FLORENCE2_DIR}")
    
    print("\n--- Tiling Diagnostics ---")
    print(f"Default Tile Size: {config.DEFAULT_TILE_SIZE}")
    print(f"Default Stride: {config.DEFAULT_STRIDE}")

if __name__ == "__main__":
    run_diagnostics()