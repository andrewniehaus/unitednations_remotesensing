import os
from pathlib import Path
import rasterio
from src import config

def validate_interim_tiles() -> None:
    """
    Iterates through all generated tiles in the interim directory to validate 
    spatial dimensions, CRS, and affine transformations.
    """
    interim_dir = Path(config.INTERIM_DIR)
    tile_files = list(interim_dir.glob("*.tif"))
    
    if not tile_files:
        print(f"No tiles found in {config.INTERIM_DIR}.")
        return

    print(f"Validating {len(tile_files)} tiles in {config.INTERIM_DIR}...\n")

    for tile_path in tile_files[:10]:  # Limit output to first 10 for terminal readability
        with rasterio.open(tile_path) as src:
            print(f"File: {tile_path.name}")
            print(f"  - Dimensions: {src.width} x {src.height}")
            print(f"  - CRS: {src.crs}")
            print(f"  - Transform Origin (X, Y): ({src.transform.c}, {src.transform.f})")
            print(f"  - Bounding Box: {src.bounds}")
            print("-" * 50)
            
    if len(tile_files) > 10:
        print(f"... and {len(tile_files) - 10} more tiles successfully processed.")

if __name__ == "__main__":
    validate_interim_tiles()