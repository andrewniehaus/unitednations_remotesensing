import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src import config


def create_mock_geotiff(
    filename: str = "test_image.tif",
    width: int = 2048,
    height: int = 2048,
    count: int = 3,
    block_size: int = 512,
):
    """Generates a synthetic multi-band GeoTIFF with internal block structures for testing."""
    output_path = config.RAW_DIR / filename
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Generate synthetic image data (values 1-255) with some zero-filled (nodata) blocks to test filtering
    data = np.random.randint(10, 240, size=(count, height, width), dtype=np.uint8)
    
    # Introduce a fully 'nodata' block in the bottom-right corner to test blank tile skipping
    data[:, 1536:, 1536:] = 0

    # Define a realistic spatial extent
    west, south, east, north = 300000.0, 4200000.0, 302048.0, 4202048.0
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
        "blockxsize": block_size,
        "blockysize": block_size,
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    print(f"Mock GeoTIFF successfully created at: {output_path}")


if __name__ == "__main__":
    create_mock_geotiff()