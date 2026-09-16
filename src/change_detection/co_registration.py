"""
src/change_detection/co_registration.py

Sub-Pixel Spatial Co-Registration Module.

Aligns post-event (Time 2) rasters to pre-event (Time 1) rasters to eliminate 
false-positive change detection noise caused by orbital variations or sensor 
viewing angles prior to latent feature extraction.
"""

from pathlib import Path
from typing import Union, Tuple
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config

class SpatialCoRegistrator:
    """
    Handles image alignment and warping using ECC maximization.
    """

    def __init__(self, number_of_iterations: int = 50, termination_eps: float = 1e-4):
        """
        Initialize the co-registrator parameters.
        """
        self.number_of_iterations = number_of_iterations
        self.termination_eps = termination_eps
        
        # Define termination criteria for the ECC algorithm
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
            self.number_of_iterations, 
            self.termination_eps
        )

    def _get_reference_band(self, img_array: np.ndarray) -> np.ndarray:
        """
        Extracts a grayscale or primary reference band for alignment calculations.
        Assumes input is (Bands, Height, Width).
        """
        if img_array.shape[0] >= 3:
            # Simple average for RGB to Grayscale
            gray = np.mean(img_array[:3], axis=0).astype(np.float32)
            return gray
        return img_array[0].astype(np.float32)

    def compute_warp_matrix(self, t1_array: np.ndarray, t2_array: np.ndarray) -> np.ndarray:
        """
        Computes the affine warp matrix to align T2 to T1.
        """
        ref_t1 = self._get_reference_band(t1_array)
        tar_t2 = self._get_reference_band(t2_array)

        # Initialize a 2x3 identity matrix for the affine transformation
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Run the ECC algorithm (using Euclidean transformation: translation, rotation)
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_t1, tar_t2, warp_matrix, 
                cv2.MOTION_EUCLIDEAN, self.criteria
            )
        except cv2.error:
            # Fallback to identity if convergence fails (e.g., zero overlap)
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        return warp_matrix

    def align_and_save(
        self, 
        t1_path: Union[str, Path], 
        t2_path: Union[str, Path], 
        output_path: Union[str, Path]
    ) -> Path:
        """
        Aligns T2 raster to T1 raster and writes the aligned T2 to disk.
        """
        with rasterio.open(t1_path) as src_t1, rasterio.open(t2_path) as src_t2:
            t1_img = src_t1.read()
            t2_img = src_t2.read()
            t2_profile = src_t2.profile

            # Compute alignment
            warp_mat = self.compute_warp_matrix(t1_img, t2_img)
            
            # Apply warp to all bands in T2
            aligned_t2 = np.zeros_like(t1_img)
            height, width = t1_img.shape[1:]
            
            for band_idx in range(t2_img.shape[0]):
                aligned_t2[band_idx] = cv2.warpAffine(
                    t2_img[band_idx].astype(np.float32), 
                    warp_mat, 
                    (width, height), 
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )

            # Update profile to match T1 spatial dimensions (if they differ slightly)
            t2_profile.update(
                width=width,
                height=height,
                transform=src_t1.transform,
                dtype=aligned_t2.dtype
            )

            # Write aligned file
            output_path = Path(output_path)
            with rasterio.open(output_path, 'w', **t2_profile) as dst:
                dst.write(aligned_t2)

        return output_path