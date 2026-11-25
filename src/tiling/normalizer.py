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

from typing import List, Optional, Tuple, Union

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

    def _valid_mask(self, band: np.ndarray, nodata: Optional[Union[int, float]]) -> np.ndarray:
        """True where a pixel holds data."""
        valid = np.isfinite(band)
        if nodata is not None and not np.isnan(nodata):
            valid &= band != nodata
        return valid

    def band_limits(
        self, data: np.ndarray, nodata: Optional[Union[int, float]] = None
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Percentile stretch limits for each band of a whole scene (or a downsampled copy).

        Passing these to normalize() stretches every tile identically, so the same surface
        looks the same in every tile. None for a band with no valid pixels.
        """
        limits = []
        for band in data.astype(np.float32):
            valid = self._valid_mask(band, nodata)
            if not np.any(valid):
                limits.append(None)
                continue
            low_p, high_p = np.percentile(band[valid], self.percentile_clip)
            limits.append((float(low_p), float(high_p)))
        return limits

    def _stretch_band(
        self,
        band: np.ndarray,
        nodata: Optional[Union[int, float]] = None,
        limits: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Percentile-clips a single band and rescales it to uint8.

        Uses the given (low, high) limits, or the band's own percentiles if none are given.
        NoData pixels are excluded and written as 0; real pixels are written as 1-255, so a
        dark real pixel is never mistaken for NoData downstream.
        """
        band = band.astype(np.float32)
        valid = self._valid_mask(band, nodata)

        if not np.any(valid):
            return np.zeros(band.shape, dtype=np.uint8)

        low_p, high_p = limits if limits is not None else np.percentile(band[valid], self.percentile_clip)
        if high_p == low_p:
            out = np.zeros(band.shape, dtype=np.uint8)
            out[valid] = 1
            return out

        scaled = (np.clip(band, low_p, high_p) - low_p) / (high_p - low_p) * 254.0 + 1.0
        scaled[~valid] = 0
        return scaled.astype(np.uint8)

    def normalize(
        self,
        tile_data: np.ndarray,
        nodata: Optional[Union[int, float]] = None,
        band_limits: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> np.ndarray:
        """
        Converts a raw tile into an 8-bit, 3-channel array.

        Args:
            tile_data (np.ndarray): Input array of shape (Bands, Height, Width).
            nodata (Optional[Union[int, float]]): Source NoData value to ignore.
            band_limits: Scene-wide (low, high) per source band from band_limits(). Without
                them, each tile is stretched on its own, so brightness can differ between tiles.

        Returns:
            np.ndarray: uint8 array of shape (3, Height, Width).
        """
        num_bands = tile_data.shape[0]

        def limits_for(idx: int) -> Optional[Tuple[float, float]]:
            return band_limits[idx] if band_limits is not None else None

        # Single-band inputs (e.g. SAR or panchromatic) are replicated across 3 channels
        if num_bands == 1:
            band = self._stretch_band(tile_data[0], nodata, limits_for(0))
            return np.stack([band] * 3, axis=0)

        # Clamp requested band indices to the available band count
        indices = [min(idx, num_bands - 1) for idx in self.rgb_indices]
        return np.stack(
            [self._stretch_band(tile_data[idx], nodata, limits_for(idx)) for idx in indices], axis=0
        )
