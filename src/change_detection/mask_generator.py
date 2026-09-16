"""
src/change_detection/mask_generator.py

Binary Change Mask and Polygon Generator.

Converts continuous change intensity maps into binary masks, cleans morphological noise,
categorizes severity tiers, and extracts polygon coordinates for spatial reporting.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, binary_closing
from shapely.geometry import Polygon, Shape
from rasterio.features import shapes

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class ChangeMaskGenerator:
    """
    Generates cleaned binary masks and vector geometries from change intensity maps.
    """

    def __init__(
        self,
        default_threshold: float = 0.45,
        min_pixel_area: int = 16,
    ):
        """
        Initialize mask generator parameters.

        Args:
            default_threshold (float): Fixed threshold for change detection (0 to 1).
            min_pixel_area (int): Minimum pixel area required to retain a change polygon.
        """
        self.default_threshold = default_threshold
        self.min_pixel_area = min_pixel_area

    def compute_otsu_threshold(self, change_map: np.ndarray) -> float:
        """
        Feature 1: Otsu Adaptive Statistical Thresholding.
        Calculates optimal bimodal change threshold automatically from score distributions.
        """
        pixel_counts, bin_edges = np.histogram(change_map.flatten(), bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        weight1 = np.cumsum(pixel_counts)
        weight2 = np.cumsum(pixel_counts[::-1])[::-1]

        mean1 = np.cumsum(pixel_counts * bin_centers) / np.maximum(weight1, 1)
        mean2 = (np.cumsum((pixel_counts * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]

        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = np.argmax(variance)
        return float(bin_centers[idx])

    def clean_morphological_noise(
        self, binary_mask: np.ndarray, structure_size: int = 3
    ) -> np.ndarray:
        """
        Feature 2: Morphological Cleaning.
        Applies binary opening and closing to suppress isolated pixel artifacts.
        """
        struct = np.ones((structure_size, structure_size), dtype=bool)
        cleaned = binary_opening(binary_mask, structure=struct)
        cleaned = binary_closing(cleaned, structure=struct)
        return cleaned

    def classify_change_severity(self, change_map: np.ndarray) -> np.ndarray:
        """
        Feature 3: Multi-Class Change Intensity Binning.
        Categorizes continuous change into discrete severity tiers:
        0: No Change, 1: Minor, 2: Moderate, 3: Severe Destruction.
        """
        severity_grid = np.zeros_like(change_map, dtype=np.int32)
        severity_grid[change_map >= 0.30] = 1
        severity_grid[change_map >= 0.50] = 2
        severity_grid[change_map >= 0.70] = 3
        return severity_grid

    def extract_change_polygons(
        self,
        change_map: np.ndarray,
        use_otsu: bool = True,
        x_offset: int = 0,
        y_offset: int = 0
    ) -> pd.DataFrame:
        """
        Feature 4: Raster-to-Vector Polygon Extraction.
        Converts change grid into pixel-space Shapely Polygons and DataFrames.

        Args:
            change_map (np.ndarray): 2D array of change indices.
            use_otsu (bool): If True, uses dynamic Otsu thresholding instead of default.
            x_offset (int): Tile X position offset for mosaic reconstruction.
            y_offset (int): Tile Y position offset for mosaic reconstruction.

        Returns:
            pd.DataFrame: DataFrame of change polygons with pixel bounding boxes.
        """
        threshold = self.compute_otsu_threshold(change_map) if use_otsu else self.default_threshold
        raw_mask = (change_map >= threshold).astype(np.uint8)
        clean_mask = self.clean_morphological_noise(raw_mask)
        severity = self.classify_change_severity(change_map)

        records = []
        # Extract shapes using rasterio features
        for geom_dict, val in shapes(clean_mask, mask=clean_mask > 0):
            if val == 0:
                continue

            coords = geom_dict["coordinates"][0]
            # Apply tile offsets
            offset_coords = [(x + x_offset, y + y_offset) for x, y in coords]
            poly = Polygon(offset_coords)

            if poly.area < self.min_pixel_area:
                continue

            bounds = poly.bounds  # (xmin, ymin, xmax, ymax)
            
            # Estimate mean severity level inside polygon region
            minx, miny, maxx, maxy = int(bounds[0] - x_offset), int(bounds[1] - y_offset), int(bounds[2] - x_offset), int(bounds[3] - y_offset)
            poly_severity = int(np.mean(severity[max(0, miny):maxy+1, max(0, minx):maxx+1])) if miny < maxy and minx < maxx else 1

            records.append({
                "xmin": bounds[0],
                "ymin": bounds[1],
                "xmax": bounds[2],
                "ymax": bounds[3],
                "label": f"change_severity_{poly_severity}",
                "confidence": float(np.max(change_map[max(0, miny):maxy+1, max(0, minx):maxx+1])) if miny < maxy and minx < maxx else threshold,
                "geometry": poly,
            })

        return pd.DataFrame(records)