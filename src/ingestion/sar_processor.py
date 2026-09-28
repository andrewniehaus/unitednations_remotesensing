"""
SAR Processor Module
====================
Handles dedicated ingestion, radiometric calibration, and speckle reduction for 
Synthetic Aperture Radar (SAR) data (e.g., Capella Space, ICEYE) within the air-gapped environment.

Extra Features Included:
- Radiometric calibration to Sigma Naught (dB) for standardized temporal comparison.
- Configurable speckle filtering (Median/Gaussian) to reduce radar noise prior to embedding extraction.
- Dual-polarization ratio generation (e.g., VV/VH) to highlight structural metallic anomalies (military compounds).
"""

import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from scipy.ndimage import median_filter, gaussian_filter

# Strict hardware and path configurations
from src.config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SARProcessor:
    def __init__(
        self,
        input_dir: Path = RAW_DIR,
        output_dir: Path = INTERIM_DIR,
    ):
        """
        Initializes the SAR Processor using configuration paths.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calibrate_to_db(self, array: np.ndarray, calibration_factor: float = 1.0) -> np.ndarray:
        """
        Converts Digital Numbers (DN) to radar backscatter in decibels (dB).
        Prevents log(0) errors using a small epsilon.
        """
        epsilon = 1e-10
        # Assuming amplitude data; if power, remove the square
        power = np.square(array.astype(np.float32)) * calibration_factor
        db_array = 10 * np.log10(power + epsilon)
        
        # Clip extreme outliers common in urban/metallic specular reflection
        return np.clip(db_array, -30.0, 5.0)

    def apply_speckle_filter(self, array: np.ndarray, filter_type: str = "median", size: int = 3) -> np.ndarray:
        """
        Applies local spatial filtering to reduce SAR speckle noise.
        """
        if filter_type.lower() == "median":
            return median_filter(array, size=size)
        elif filter_type.lower() == "gaussian":
            return gaussian_filter(array, sigma=size/2)
        else:
            logger.warning(f"Unknown filter type '{filter_type}'. Returning unfiltered array.")
            return array

    def generate_pol_ratio(self, band_1: np.ndarray, band_2: np.ndarray) -> np.ndarray:
        """
        Generates a cross-polarization ratio (e.g., VV / VH) in linear scale, 
        useful for identifying volume scattering (tents/camps) vs double-bounce (urban).
        """
        epsilon = 1e-10
        return band_1 / (band_2 + epsilon)

    def process_scene(
        self, 
        filepath: Path, 
        calibration_factor: float = 1.0, 
        filter_type: str = "median", 
        dual_pol: bool = False
    ) -> Optional[Path]:
        """
        End-to-end ingestion, calibration, filtering, and export of a SAR raster.
        """
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        output_filename = self.output_dir / f"{filepath.stem}_calibrated_db.tif"

        try:
            with rasterio.open(filepath) as src:
                meta = src.meta.copy()
                meta.update(dtype=rasterio.float32)

                # Process Band 1 (e.g., HH or VV)
                band_1_raw = src.read(1)
                band_1_db = self.calibrate_to_db(band_1_raw, calibration_factor)
                band_1_clean = self.apply_speckle_filter(band_1_db, filter_type=filter_type)

                output_bands = 1
                band_2_clean = None
                ratio_band = None

                # Process Band 2 and Ratio if dual-pol is flagged and exists
                if dual_pol and src.count >= 2:
                    band_2_raw = src.read(2)
                    band_2_db = self.calibrate_to_db(band_2_raw, calibration_factor)
                    band_2_clean = self.apply_speckle_filter(band_2_db, filter_type=filter_type)
                    ratio_band = self.generate_pol_ratio(band_1_clean, band_2_clean)
                    output_bands = 3
                    meta.update(count=output_bands)

                with rasterio.open(output_filename, 'w', **meta) as dst:
                    dst.write(band_1_clean.astype(rasterio.float32), 1)
                    if dual_pol and band_2_clean is not None and ratio_band is not None:
                        dst.write(band_2_clean.astype(rasterio.float32), 2)
                        dst.write(ratio_band.astype(rasterio.float32), 3)

            logger.info(f"Successfully processed SAR scene: {output_filename}")
            return output_filename

        except Exception as e:
            logger.error(f"Failed to process SAR scene {filepath.name}: {str(e)}")
            return None


if __name__ == "__main__":
    # Test execution snippet for air-gapped validation
    processor = SARProcessor()
    
    # Example: Look for a raw Capella SAR TIFF
    sample_sar = next(RAW_DIR.rglob("*capella*.tif"), None)
    if sample_sar:
        processed_path = processor.process_scene(
            filepath=sample_sar, 
            calibration_factor=1.0, 
            dual_pol=True
        )