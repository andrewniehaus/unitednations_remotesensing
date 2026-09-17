"""
src/tiling/adaptive_overlap.py

Dynamic Stride and Overlap Calculation Module.

Calculates tile stride dynamically based on target object prompt dimensions 
(e.g., higher overlap for long linear trench lines, standard overlap for discrete tents) 
to prevent truncation across tile seams.
"""

from typing import List, Tuple
import numpy as np

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class AdaptiveOverlapCalculator:
    """
    Computes dynamic stride and overlap parameters based on target geometry scales.
    """

    def __init__(
        self,
        default_tile_size: int = config.DEFAULT_TILE_SIZE,
        default_stride: int = config.DEFAULT_STRIDE,
    ):
        """
        Initialize calculator with fallback config defaults.
        """
        self.tile_size = default_tile_size
        self.default_stride = default_stride

    def compute_prompt_adaptive_stride(
        self, prompt_type: str = "discrete"
    ) -> int:
        """
        Feature 1: Prompt-Driven Stride Calculation.
        Adjusts stride based on target visual geometry type:
        - 'linear' (trenches, fences): High overlap (50% tile size).
        - 'cluster' (tent encampments): Medium overlap (35% tile size).
        - 'discrete' (vehicles, buildings): Standard overlap (20% tile size).
        """
        if prompt_type == "linear":
            overlap = int(self.tile_size * 0.50)
        elif prompt_type == "cluster":
            overlap = int(self.tile_size * 0.35)
        else:
            overlap = int(self.tile_size * 0.20)

        stride = self.tile_size - overlap
        return max(32, stride)

    def compute_gsd_aware_stride(
        self, gsd_meters: float, target_metric_overlap: float = 25.0
    ) -> int:
        """
        Feature 2: Ground Sample Distance (GSD) Aware Stride Adjustment.
        Ensures a fixed metric overlap in meters regardless of pixel resolution.
        """
        pixel_overlap = int(target_metric_overlap / gsd_meters)
        pixel_overlap = min(pixel_overlap, int(self.tile_size * 0.60))

        stride = self.tile_size - pixel_overlap
        return max(32, stride)

    def estimate_edge_truncation_risk(
        self, box_width_px: int, box_height_px: int, current_stride: int
    ) -> float:
        """
        Feature 3: Edge Truncation Risk Estimation.
        Calculates the probability $P(\text{truncation}) = \frac{\max(W, H)}{\text{Overlap}}$ 
        that an object of given pixel size will be truncated across seams.
        """
        overlap = self.tile_size - current_stride
        max_dim = max(box_width_px, box_height_px)

        if overlap <= 0:
            return 1.0

        risk_score = float(max_dim) / float(overlap)
        return min(1.0, round(risk_score, 4))

    def generate_window_grid_coordinates(
        self, image_width: int, image_height: int, stride: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Features 4 & 5: Variable Row/Column Grid Step Generation & Border Adjustment.
        Generates list of (row_off, col_off, tile_height, tile_width) pixel windows.
        """
        window_coords = []

        y_offsets = list(range(0, image_height, stride))
        x_offsets = list(range(0, image_width, stride))

        # Adjust last offsets to ensure full coverage up to borders
        if y_offsets[-1] + self.tile_size < image_height:
            y_offsets.append(image_height - self.tile_size)
        if x_offsets[-1] + self.tile_size < image_width:
            x_offsets.append(image_width - self.tile_size)

        for y in y_offsets:
            for x in x_offsets:
                # Clamp window dimensions to image dimensions
                h = min(self.tile_size, image_height - y)
                w = min(self.tile_size, image_width - x)
                window_coords.append((y, x, h, w))

        return window_coords