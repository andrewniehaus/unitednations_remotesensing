# src/testing/create_synthetic_tiff.py

import sys
import os
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def create_mock_geotiff(filename: str = "synthetic_pipeline_test.tif"):
    """Generates a multi-band GeoTIFF designed to stress-test georeferencing and ingestion pipelines."""
    output_path = config.RAW_DIR / filename
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 2048, 2048
    count = 3
    
    data = np.random.randint(10, 240, size=(count, height, width), dtype=np.uint8)
    
    # Insert a nodata block to test blank tile skipping algorithms
    data[:, 1536:, 1536:] = 0

    # Strict geographic coordinates (EPSG:4326) mapping to Libya for georeferencing tests
    west, south, east, north = 13.18, 32.88, 13.19, 32.89
    transform = from_bounds(west, south, east, north, width, height)
    crs = CRS.from_epsg(4326)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "blockxsize": config.DEFAULT_TILE_SIZE,
        "blockysize": config.DEFAULT_TILE_SIZE,
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

if __name__ == "__main__":
    create_mock_geotiff()