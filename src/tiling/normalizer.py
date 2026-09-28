"""
src/tiling/normalizer.py

Tile Array Normalization Module.

Converts raw multi-band raster tiles (8/11/12/16-bit optical, or single-band SAR)
into standardized 8-bit, 3-channel arrays in (Bands, Height, Width) order for VLM
ingestion by RasterChipper.
"""

# [EDIT 2026-09-28 | Claude Code for charliefp03-dg] New module. The file previously existed as an
# empty "Normalizer.py"; it was renamed to lowercase to match the import in chipper.py and
# ArrayNormalizer was implemented here, mirroring the stretch logic in band_selector.py.

from typing import Optional, Tuple, Union

import numpy as np

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class ArrayNormalizer:
    """
    Maps raw (Bands, Height, Width) tiles to uint8 3-channel (3, Height, Width) arrays.
    """

    def __init__(
        self,
        rgb_band_indices: Tuple[int, int, int] = config.DEFAULT_RGB_BANDS,
        percentile_clip: Tuple[float, float] = config.DEFAULT_CLIP_PERCENTILE,
    ):
        """
        Initialize normalization rules.

        Args:
            rgb_band_indices (Tuple[int, int, int]): 0-based indices for Red, Green, Blue.
            percentile_clip (Tuple[float, float]): Lower/upper percentiles for contrast stretch.
        """
        self.rgb_indices = rgb_band_indices
        self.percentile_clip = percentile_clip

    def _stretch_band(
        self, band: np.ndarray, nodata: Optional[Union[int, float]] = None
    ) -> np.ndarray:
        """
        Percentile-clips a single band and rescales it to [0, 255] uint8.
        NoData pixels are excluded from the percentile calculation and written as 0.
        """
        band = band.astype(np.float32)
        valid = np.isfinite(band)
        if nodata is not None:
            valid &= band != nodata

        if not np.any(valid):
            return np.zeros(band.shape, dtype=np.uint8)

        low_p, high_p = np.percentile(band[valid], self.percentile_clip)
        if high_p == low_p:
            return np.zeros(band.shape, dtype=np.uint8)

        scaled = (np.clip(band, low_p, high_p) - low_p) / (high_p - low_p) * 255.0
        scaled[~valid] = 0
        return scaled.astype(np.uint8)

    def normalize(
        self, tile_data: np.ndarray, nodata: Optional[Union[int, float]] = None
    ) -> np.ndarray:
        """
        Converts a raw tile into an 8-bit, 3-channel array.

        Note: stretch statistics are computed per tile, so brightness may vary slightly
        between neighbouring tiles.

        Args:
            tile_data (np.ndarray): Input array of shape (Bands, Height, Width).
            nodata (Optional[Union[int, float]]): Source NoData value to ignore.

        Returns:
            np.ndarray: uint8 array of shape (3, Height, Width).
        """
        num_bands = tile_data.shape[0]

        # Single-band inputs (e.g. SAR or panchromatic) are replicated across 3 channels
        if num_bands == 1:
            band = self._stretch_band(tile_data[0], nodata)
            return np.stack([band] * 3, axis=0)

        # Clamp requested band indices to the available band count
        indices = [min(idx, num_bands - 1) for idx in self.rgb_indices]
        return np.stack(
            [self._stretch_band(tile_data[idx], nodata) for idx in indices], axis=0
        )
