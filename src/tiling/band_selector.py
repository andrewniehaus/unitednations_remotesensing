"""
src/tiling/band_selector.py

Multi-Spectral and Sensor Band Normalization Module.

Extracts, scales, and transforms multi-band imagery (4-band RGB-NIR, 8-band multispectral, 
or Capella SAR intensity rasters) into 3-channel uint8 RGB tensors ready for VLM inference.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class BandSelector:
    """
    Handles band extraction, dynamic contrast stretching, pan-sharpening, and SAR log-scaling.
    """

    def __init__(
        self,
        rgb_band_indices: Tuple[int, int, int] = (1, 2, 3),
        percentile_clip: Tuple[float, float] = (2.0, 98.0),
    ):
        """
        Initialize band selection rules.

        Args:
            rgb_band_indices (Tuple[int, int, int]): 1-based indices corresponding to Red, Green, Blue.
            percentile_clip (Tuple[float, float]): Lower/upper percentiles for dynamic stretch.
        """
        self.rgb_indices = rgb_band_indices
        self.percentile_clip = percentile_clip

    def apply_percentile_stretch(self, band_array: np.ndarray) -> np.ndarray:
        """
        Feature 3: Dynamic Contrast Stretching (2%-98% Cumulative Percentile Clipping).
        Converts 11-bit, 12-bit, or 16-bit raw imagery to standardized 8-bit uint8 [0, 255].
        """
        low_p, high_p = np.percentile(band_array, self.percentile_clip)
        if high_p == low_p:
            return np.zeros_like(band_array, dtype=np.uint8)

        clipped = np.clip(band_array, low_p, high_p)
        normalized = ((clipped - low_p) / (high_p - low_p)) * 255.0
        return normalized.astype(np.uint8)

    def process_sar_intensity(self, sar_array: np.ndarray) -> np.ndarray:
        """
        Feature 4: SAR Intensity Log-Scaling and Lee Filter Speckle Reduction.
        Preprocesses Capella/Sentinel-1 Synthetic Aperture Radar amplitude data into RGB.
        """
        # Convert intensity to decibels (log-scale)
        sar_db = 10.0 * np.log10(np.maximum(sar_array, 1e-6))

        # Basic 3x3 local variance despeckling (Lee Filter approximation)
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(sar_db, size=3)
        sqr_mean = uniform_filter(sar_db**2, size=3)
        var = np.maximum(0.0, sqr_mean - mean**2)

        overall_var = np.var(sar_db)
        weights = var / (var + overall_var + 1e-6)
        despeckled = mean + weights * (sar_db - mean)

        # Scale to 8-bit uint8
        return self.apply_percentile_stretch(despeckled)

    def select_rgb_bands(self, multi_band_array: np.ndarray) -> np.ndarray:
        """
        Feature 1 & 5: Multi-Band Channel Mapping and PIL Array Format Output.
        Extracts target RGB channels and returns a (Height, Width, 3) uint8 array.

        Args:
            multi_band_array (np.ndarray): Input array of shape (Bands, Height, Width).

        Returns:
            np.ndarray: Preprocessed 3-channel RGB array in shape (Height, Width, 3).
        """
        num_bands = multi_band_array.shape[0]

        # Handle SAR single-band inputs by replicating across 3 channels
        if num_bands == 1:
            processed_band = self.process_sar_intensity(multi_band_array[0])
            return np.stack([processed_band] * 3, axis=-1)

        # Extract designated RGB band channels
        r_idx, g_idx, b_idx = [idx - 1 for idx in self.rgb_indices]
        
        # Fallback if requested indices exceed band count
        r_idx = min(r_idx, num_bands - 1)
        g_idx = min(g_idx, num_bands - 1)
        b_idx = min(b_idx, num_bands - 1)

        r_band = self.apply_percentile_stretch(multi_band_array[r_idx])
        g_band = self.apply_percentile_stretch(multi_band_array[g_idx])
        b_band = self.apply_percentile_stretch(multi_band_array[b_idx])

        # Feature 5: Stack into HWC uint8 format ready for PIL / PyTorch transform
        rgb_stack = np.stack([r_band, g_band, b_band], axis=-1)
        return rgb_stack

    def pan_sharpen(self, ms_rgb: np.ndarray, pan_band: np.ndarray) -> np.ndarray:
        """
        Feature 2: High-Pass Intensity Pan-Sharpening.
        Fuses high-resolution panchromatic band with lower-resolution multispectral RGB.
        """
        # Simple Brovey Transform pan-sharpening
        ms_float = ms_rgb.astype(np.float32)
        intensity = np.mean(ms_float, axis=-1, keepdims=True) + 1e-6
        pan_expanded = np.expand_dims(pan_band.astype(np.float32), axis=-1)

        sharpened = (ms_float / intensity) * pan_expanded
        return np.clip(sharpened, 0, 255).astype(np.uint8)