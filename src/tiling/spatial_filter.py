"""
src/tiling/spatial_filter.py

Spatial Tile Optimization and Pre-Inference Filtering.

Inspects generated tiles in the interim directory prior to Vision-Language Model 
inference. Drops tiles with excessive NoData, high cloud cover, or mathematically 
low informational entropy (featureless terrain) to conserve GPU VRAM and 
execution time during offline processing.
"""

import os
import json
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from pathlib import Path

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class SpatialTileFilter:
    """
    Evaluates and filters raster tiles based on pixel validity, 
    cloud heuristics, and information entropy.
    """

    def __init__(
        self,
        min_valid_pixels: float = 0.50,
        min_entropy: float = 3.5,
        max_cloud_cover: float = 0.60,
        cloud_brightness_thresh: int = 220,
    ):
        """
        Initialize the spatial filter with threshold parameters.

        Args:
            min_valid_pixels (float): Minimum percentage of valid (non-NoData) pixels required.
            min_entropy (float): Minimum Shannon entropy required to keep the tile.
            max_cloud_cover (float): Maximum allowed percentage of cloud-like pixels.
            cloud_brightness_thresh (int): 8-bit digital number threshold for cloud detection.
        """
        self.min_valid_pixels = min_valid_pixels
        self.min_entropy = min_entropy
        self.max_cloud_cover = max_cloud_cover
        self.cloud_brightness_thresh = cloud_brightness_thresh

    def calculate_entropy(self, image_array: np.ndarray) -> float:
        """
        Feature 1: Calculates Shannon entropy of a tile to determine informational complexity.
        Formula: $H = -\\sum p_i \\log_2 p_i$
        
        Args:
            image_array (np.ndarray): 2D or 3D numpy array of the image tile.
            
        Returns:
            float: Calculated entropy value.
        """
        # Flatten array and calculate histogram for pixel value probabilities
        histogram, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 255))
        probabilities = histogram / histogram.sum()
        
        # Remove zero probabilities to avoid log2(0) error
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def evaluate_tile(self, tile_path: Path) -> Tuple[bool, str]:
        """
        Feature 4 & 2: Evaluates a single tile for NoData, Alpha, and Cloud heuristics.
        
        Args:
            tile_path (Path): Path to the specific interim GeoTIFF tile.
            
        Returns:
            Tuple[bool, str]: (is_valid_to_keep, rejection_reason)
        """
        try:
            with rasterio.open(tile_path) as src:
                # Read all bands
                img = src.read()
                nodata = src.nodata
                
                # Check for alpha band (often the 4th band)
                has_alpha = src.count == 4
                
                # Determine valid pixel mask
                if has_alpha:
                    valid_mask = img[3] > 0
                elif nodata is not None:
                    valid_mask = img[0] != nodata
                else:
                    # Fallback assuming pure black is border NoData
                    valid_mask = np.any(img > 0, axis=0)

                valid_ratio = np.sum(valid_mask) / valid_mask.size
                if valid_ratio < self.min_valid_pixels:
                    return False, "low_valid_pixels"

                # Extract RGB/Grayscale for entropy and cloud checks
                analysis_img = img[:3] if src.count >= 3 else img[0:1]
                
                # Cloud heuristic: High brightness & low variance across RGB
                if analysis_img.shape[0] == 3:
                    gray = np.mean(analysis_img, axis=0)
                    cloud_mask = gray > self.cloud_brightness_thresh
                    cloud_ratio = np.sum(cloud_mask) / cloud_mask.size
                    if cloud_ratio > self.max_cloud_cover:
                        return False, "high_cloud_cover"

                # Entropy check for featureless terrain (e.g., solid desert/water)
                entropy_val = self.calculate_entropy(analysis_img)
                if entropy_val < self.min_entropy:
                    return False, f"low_entropy_{entropy_val:.2f}"

                return True, "valid"
                
        except Exception as e:
            # Drop unreadable or corrupted files generated during chipping
            return False, f"read_error_{str(e)}"

    def filter_paths(self, tile_paths: List[Path]) -> Tuple[List[Path], Dict[str, str]]:
        """
        Evaluates tiles in parallel and records why any were rejected.

        Writes the rejections to config.PROCESSED_DIR / "tile_rejection_manifest.json".

        Args:
            tile_paths (List[Path]): Tiles to evaluate.

        Returns:
            Tuple[List[Path], Dict[str, str]]: Kept tiles in input order, and rejection reasons by filename.
        """
        # Threads keep raster reads from blocking one another
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            outcomes = list(executor.map(self.evaluate_tile, tile_paths))

        kept = [path for path, (is_valid, _) in zip(tile_paths, outcomes) if is_valid]
        rejected = {path.name: reason for path, (is_valid, reason) in zip(tile_paths, outcomes) if not is_valid}

        manifest_path = Path(config.PROCESSED_DIR) / "tile_rejection_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=4)

        return kept, rejected

    def filter_directory(self) -> List[Path]:
        """
        Feature 3 & 5: Evaluates every tile in config.INTERIM_DIR and logs rejections.

        Returns:
            List[Path]: A list of file paths that passed all filtering checks.
        """
        tile_paths = list(Path(config.INTERIM_DIR).glob("*.tif"))
        if not tile_paths:
            return []
        return self.filter_paths(tile_paths)[0]

if __name__ == "__main__":
    # Example direct execution structure
    tile_filter = SpatialTileFilter()
    processed_tiles = tile_filter.filter_directory()
    print(f"Retained {len(processed_tiles)} tiles for VLM inference.")