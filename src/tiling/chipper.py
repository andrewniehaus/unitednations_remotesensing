import concurrent.futures

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from src import config

from src.tiling.normalizer import ArrayNormalizer


class RasterChipper:
    """Handles memory-efficient, georeferenced raster tiling for offline inference."""

    def __init__(
        self,
        tile_size: int = config.DEFAULT_TILE_SIZE,
        padding_mode: str = config.DEFAULT_PADDING_MODE,
        fill_value: Union[int, float] = config.DEFAULT_FILL_VALUE,
        output_dir: Optional[Path] = None,
        manifest_filename: str = config.MANIFEST_FILENAME,
    ):
        """Initializes the chipper with spatial parameters and output locations."""
        self.tile_size = tile_size
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        self.output_dir = Path(output_dir) if output_dir else config.INTERIM_DIR
        self.manifest_path = self.output_dir / manifest_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = ArrayNormalizer()  # Add this line

        # Ensure output directory exists for tile writes
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_tile_windows(
        self, src: rasterio.io.DatasetReader
    ) -> Generator[Tuple[int, int, Window], None, None]:
        """Yields native block windows directly from the GeoTIFF for optimized I/O.

        Yields:
            Tuple of (row_index, col_index, rasterio.windows.Window)
        """
        for (row_idx, col_idx), window in src.block_windows(1):
            yield row_idx, col_idx, window

    def calculate_tile_transform(
        self, src_transform: Affine, window: Window
    ) -> Affine:
        """Recalculates the 6-parameter affine transform for an individual tile window."""
        return rasterio.windows.transform(window, src_transform)

    def pad_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Pads partial edge tiles to ensure uniform spatial dimensions (C, H, W).

        Args:
            tile_data: Input numpy array with shape (bands, height, width).

        Returns:
            Padded numpy array with shape (bands, self.tile_size, self.tile_size).
        """
        _, h, w = tile_data.shape

        if h == self.tile_size and w == self.tile_size:
            return tile_data

        pad_h = self.tile_size - h
        pad_w = self.tile_size - w
        pad_width = ((0, 0), (0, pad_h), (0, pad_w))

        if self.padding_mode == "constant":
            return np.pad(
                tile_data,
                pad_width=pad_width,
                mode="constant",
                constant_values=self.fill_value,
            )
        else:
            return np.pad(tile_data, pad_width=pad_width, mode=self.padding_mode)

    def write_tile(
        self,
        tile_data: np.ndarray,
        transform: Affine,
        crs: rasterio.crs.CRS,
        source_stem: str,
        row_idx: int,
        col_idx: int,
    ) -> Path:
        """Writes a single georeferenced tile to the interim directory.

        Args:
            tile_data: Padded numpy array with shape (bands, tile_size, tile_size).
            transform: Recalculated 6-parameter affine transform for the tile bounds.
            crs: Coordinate Reference System from the original source raster.
            source_stem: Base filename of the source raster (without extension).
            row_idx: Row index position in the tiling grid.
            col_idx: Column index position in the tiling grid.

        Returns:
            Path object pointing to the written GeoTIFF tile.
        """
        tile_filename = config.TILE_NAMING_SCHEMA.format(
            source_name=source_stem, row=row_idx, col=col_idx
        )
        tile_path = self.output_dir / tile_filename

        if tile_path.exists() and not config.DEFAULT_OVERWRITE:
            return tile_path

        bands, height, width = tile_data.shape

        profile = {
            "driver": "GTiff",
            "dtype": tile_data.dtype,
            "nodata": self.fill_value,
            "width": width,
            "height": height,
            "count": bands,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        with rasterio.open(tile_path, "w", **profile) as dst:
            dst.write(tile_data)

        return tile_path

    def generate_manifest(
        self,
        tile_records: List[Dict[str, Any]],
        source_path: Path,
        crs: rasterio.crs.CRS,
    ) -> Path:
        """Generates and writes a spatial tracking manifest for all chipped tiles.

        Args:
            tile_records: List of dictionaries containing tile metadata.
            source_path: Path to the original un-chipped source raster.
            crs: Coordinate Reference System of the source raster.

        Returns:
            Path object pointing to the written JSON manifest file.
        """
        manifest_data = {
            "source_raster": str(source_path.resolve()),
            "crs": str(crs),
            "tile_size": self.tile_size,
            "total_tiles": len(tile_records),
            "tiles": tile_records,
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)

        return self.manifest_path

    def _process_and_write_tile(
        self,
        tile_data: np.ndarray,
        window: Window,
        src_transform: Affine,
        src_crs: rasterio.crs.CRS,
        source_stem: str,
        row_idx: int,
        col_idx: int,
    ) -> Dict[str, Any]:
        """Worker function to normalize, pad, georeference, and write a tile concurrently."""
        _, orig_h, orig_w = tile_data.shape
        pad_h = self.tile_size - orig_h
        pad_w = self.tile_size - orig_w

        # 1. Normalize raw array to standard 8-bit, 3-channel RGB for VLM ingestion
        tile_data = self.normalizer.normalize(tile_data)

        # 2. Pad the normalized array to uniform dimensions
        padded_data = self.pad_tile(tile_data)
        
        # 3. Calculate spatial extents
        transform = self.calculate_tile_transform(src_transform, window)
        world_bounds = rasterio.windows.bounds(window, src_transform)

        # 4. Pass the standardized, padded array to your existing write_tile method
        tile_path = self.write_tile(
            tile_data=padded_data,
            transform=transform,
            crs=src_crs,
            source_stem=source_stem,
            row_idx=row_idx,
            col_idx=col_idx,
        )

        return {
            "filename": tile_path.name,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "window": {
                "col_off": window.col_off,
                "row_off": window.row_off,
                "width": window.width,
                "height": window.height,
            },
            "padding": {"bottom": pad_h, "right": pad_w},
            "world_bounds": world_bounds,
            "transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
        }

    def chip_raster(self, input_path: Path) -> Path:4
    """Executes the complete tiling workflow for a single input raster using concurrent I/O.

        Args:
            input_path: Path to the source GeoTIFF.

        Returns:
            Path to the generated tile manifest JSON.
        """
    tile_records = []
    source_stem = input_path.stem
    futures = []

        with rasterio.open(input_path) as src:
            nodata_val = src.nodata if src.nodata is not None else self.fill_value

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for row_idx, col_idx, window in self.get_tile_windows(src):
                    tile_data = src.read(window=window)

                    if np.all(tile_data == nodata_val):
                        continue

                    future = executor.submit(
                        self._process_and_write_tile,
                        tile_data,
                        window,
                        src.transform,
                        src.crs,
                        source_stem,
                        row_idx,
                        col_idx,
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    tile_records.append(future.result())

            return self.generate_manifest(tile_records, input_path, src.crs)