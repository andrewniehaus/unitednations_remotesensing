"""
src/tiling/gsd_resampler.py

Ground Sample Distance (GSD) Resampling Engine.

Standardizes spatial resolution across heterogeneous satellite imagery providers 
(e.g., 0.3m Airbus/Vantor, 0.5m WorldView, 10m Sentinel-2) to ensure objects 
(tents, vehicles, trenches) maintain consistent pixel footprints for Vision-Language Models.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class GSDResampler:
    """
    Resamples GeoTIFF rasters to a uniform Ground Sample Distance (GSD) in meters/pixel.
    """

    def __init__(self, target_gsd_meters: float = 0.50):
        """
        Initialize GSD Resampler.

        Args:
            target_gsd_meters (float): Target spatial resolution in meters per pixel.
        """
        self.target_gsd = target_gsd_meters

    def calculate_current_gsd(self, transform: Affine, crs: rasterio.crs.CRS) -> Tuple[float, float]:
        """
        Feature 1: Dynamic GSD Calculation from Affine Transform.
        Extracts native pixel dimensions in metric units.

        Args:
            transform (Affine): Rasterio dataset affine transform matrix.
            crs (rasterio.crs.CRS): Native Coordinate Reference System.

        Returns:
            Tuple[float, float]: (GSD_x, GSD_y) in meters.
        """
        # Pixel width and height in native CRS units
        x_size = abs(transform.a)
        y_size = abs(transform.e)

        # Convert angular degrees to meters if CRS is geographic (e.g., EPSG:4326)
        if crs.is_geographic:
            # 1 degree lat ~ 111,320 meters
            x_meters = x_size * 111320.0
            y_meters = y_size * 111320.0
        else:
            x_meters = x_size
            y_meters = y_size

        return x_meters, y_meters

    def resample_raster(
        self,
        input_raster_path: Union[str, Path],
        output_raster_path: Union[str, Path],
        resampling_method: Resampling = Resampling.bilinear,
        tolerance: float = 0.05,
    ) -> Path:
        """
        Features 2, 3, 4 & 5: Bilinear/Cubic Resampling, Affine Matrix Recalculation, 
        Tolerance Check, and Memory-Mapped Streaming Write.

        Args:
            input_raster_path (Union[str, Path]): Path to source GeoTIFF in config.RAW_DIR.
            output_raster_path (Union[str, Path]): Path to save resampled raster.
            resampling_method (Resampling): Rasterio resampling algorithm.
            tolerance (float): Allowed deviation from target GSD before skipping resampling.

        Returns:
            Path: Path to the output resampled GeoTIFF raster.
        """
        input_path = Path(input_raster_path)
        output_path = Path(output_raster_path)

        with rasterio.open(input_path) as src:
            current_gsd_x, current_gsd_y = self.calculate_current_gsd(src.transform, src.crs)
            avg_current_gsd = (current_gsd_x + current_gsd_y) / 2.0

            # Feature 5: Automated Tolerance Check (skip if already within target resolution)
            if abs(avg_current_gsd - self.target_gsd) <= tolerance:
                # Copy/symlink or return original if resolution matches target
                return input_path

            # Calculate scaling factors
            scale_x = current_gsd_x / self.target_gsd
            scale_y = current_gsd_y / self.target_gsd

            # Feature 3: Recalculate target pixel dimensions and affine matrix
            new_width = int(round(src.width * scale_x))
            new_height = int(round(src.height * scale_y))

            new_transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )

            # Feature 2 & 4: Resample bands and stream to disk
            profile = src.profile.copy()
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
            })

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(
                        band_idx,
                        out_shape=(new_height, new_width),
                        resampling=resampling_method,
                    )
                    dst.write(data, band_idx)

        return output_path