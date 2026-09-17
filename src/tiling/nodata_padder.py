"""
src/tiling/nodata_padder.py

Boundary Tile Padding and NoData Mask Generation Engine.

Fills incomplete edge chips along GeoTIFF borders using reflection, replication, or 
constant padding to guarantee uniform tile dimensions (config.DEFAULT_TILE_SIZE) 
without distorting object aspect ratios during VLM inference.
"""

from typing import Tuple
import numpy as np

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class NoDataPadder:
    """
    Handles array padding and binary mask generation for edge tiles.
    """

    def __init__(
        self, target_tile_size: int = config.DEFAULT_TILE_SIZE
    ):
        """
        Initialize padder parameters.
        """
        self.target_size = target_tile_size

    def pad_tile_array(
        self,
        tile_array: np.ndarray,
        mode: str = "reflect",
        constant_value: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Features 1, 2, 3 & 4: Border Padding, Padding Modes, Binary Mask Generation, 
        and Multi-Channel Array Support.

        Args:
            tile_array (np.ndarray): Input tile array of shape (Height, Width, Channels) or (Height, Width).
            mode (str): Padding strategy ('reflect', 'edge', 'constant').
            constant_value (int): Fill value if mode is 'constant'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Padded Tile Array, Binary Validity Mask).
        """
        if tile_array.ndim == 2:
            h, w = tile_array.shape
            channels = 1
            tile_array = np.expand_dims(tile_array, axis=-1)
        else:
            h, w, channels = tile_array.shape

        # Calculate required right and bottom padding
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)

        if pad_h == 0 and pad_w == 0:
            valid_mask = np.ones((h, w), dtype=bool)
            return tile_array.squeeze() if channels == 1 else tile_array, valid_mask

        # Feature 3: Binary NoData Mask (True = valid pixel, False = padded region)
        valid_mask = np.zeros((self.target_size, self.target_size), dtype=bool)
        valid_mask[:h, :w] = True

        # Feature 2 & 4: Multi-Channel Array Padding
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))

        if mode == "reflect":
            padded_array = np.pad(tile_array, pad_width, mode="reflect")
        elif mode == "edge":
            padded_array = np.pad(tile_array, pad_width, mode="edge")
        else:
            padded_array = np.pad(
                tile_array, pad_width, mode="constant", constant_values=constant_value
            )

        if channels == 1:
            padded_array = padded_array.squeeze()

        return padded_array, valid_mask

    def crop_valid_region(
        self, padded_prediction_mask: np.ndarray, original_h: int, original_w: int
    ) -> np.ndarray:
        """
        Feature 5: Aspect Ratio Preservation Recovery.
        Crops model predictions back to original non-padded spatial dimensions.
        """
        return padded_prediction_mask[:original_h, :original_w]