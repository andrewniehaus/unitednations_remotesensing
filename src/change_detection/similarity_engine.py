"""
src/change_detection/similarity_engine.py

Latent Space Cosine Similarity Engine.

Computes mathematical divergence and semantic shift matrices between Time 1 and Time 2 
embedding vectors, bypassing noisy pixel-level subtraction methods.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class LatentSimilarityEngine:
    """
    Computes metric distances and semantic shift matrices over latent embeddings.
    """

    def __init__(self, metric: str = "cosine"):
        """
        Initialize the similarity engine.

        Args:
            metric (str): Distance metric ('cosine', 'euclidean', 'manhattan').
        """
        self.metric = metric
        self.device = config.DEVICE

    def compute_similarity(
        self, emb_t1: torch.Tensor, emb_t2: torch.Tensor
    ) -> torch.Tensor:
        """
        Feature 1: Multi-Scale / Granular Distance Metrics.
        Calculates similarity score between Time 1 and Time 2 embedding vectors.

        Args:
            emb_t1 (torch.Tensor): Time 1 embedding tensor.
            emb_t2 (torch.Tensor): Time 2 embedding tensor.

        Returns:
            torch.Tensor: Similarity scores in range [0, 1].
        """
        emb_t1 = emb_t1.to(self.device)
        emb_t2 = emb_t2.to(self.device)

        if self.metric == "cosine":
            # Cosine similarity matrix
            sim = F.cosine_similarity(emb_t1, emb_t2, dim=-1)
            # Scale cosine score from [-1, 1] to [0, 1]
            sim = (sim + 1.0) / 2.0
        elif self.metric == "euclidean":
            dist = torch.norm(emb_t1 - emb_t2, p=2, dim=-1)
            sim = 1.0 / (1.0 + dist)
        elif self.metric == "manhattan":
            dist = torch.norm(emb_t1 - emb_t2, p=1, dim=-1)
            sim = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return sim

    def compute_spatial_change_map(
        self, spatial_t1: torch.Tensor, spatial_t2: torch.Tensor
    ) -> np.ndarray:
        """
        Feature 4: Temporal Shift Delta Matrix.
        Computes 2D grid of change intensity indices (0 = identical, 1 = total shift).

        Args:
            spatial_t1 (torch.Tensor): T1 spatial embedding map (B, C, H, W).
            spatial_t2 (torch.Tensor): T2 spatial embedding map (B, C, H, W).

        Returns:
            np.ndarray: 2D numpy array representing spatial change index values [0, 1].
        """
        spatial_t1 = spatial_t1.to(self.device)
        spatial_t2 = spatial_t2.to(self.device)

        # Compute cosine similarity along channel dimension
        sim_map = F.cosine_similarity(spatial_t1, spatial_t2, dim=1)
        
        # Convert similarity to change delta (Delta = 1 - Similarity)
        change_map = 1.0 - ((sim_map + 1.0) / 2.0)
        
        # Move to CPU numpy array
        change_grid = change_map.squeeze(0).cpu().numpy()
        return np.clip(change_grid, 0.0, 1.0)

    def apply_local_variance_normalization(
        self, change_map: np.ndarray, window_size: int = 3
    ) -> np.ndarray:
        """
        Feature 2: Spatial Anomaly Suppression via Local Variance Normalization.
        Suppresses false positives caused by uniform lighting or seasonal shifts.

        Args:
            change_map (np.ndarray): 2D change intensity matrix.
            window_size (int): Pixel neighborhood window size.

        Returns:
            np.ndarray: Normalized change map with background illumination noise suppressed.
        """
        from scipy.ndimage import uniform_filter

        mean = uniform_filter(change_map, size=window_size)
        sqr_mean = uniform_filter(change_map**2, size=window_size)
        variance = np.maximum(0.0, sqr_mean - mean**2)

        # Suppress regions with uniform low variance (broad environmental shifts)
        normalized_map = change_map * (variance / (variance + np.mean(variance) + 1e-6))
        return np.clip(normalized_map, 0.0, 1.0)