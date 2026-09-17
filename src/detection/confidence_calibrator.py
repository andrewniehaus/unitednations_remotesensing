"""
src/detection/confidence_calibrator.py

Confidence Calibration and Score Normalization Engine.

Applies Platt Scaling, temperature adjustments, token-length normalizations, and 
spatial density penalties to map raw Vision-Language Model generation logits into 
well-calibrated posterior probability estimates ($P(Y=1|S)$).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class ConfidenceCalibrator:
    """
    Calibrates unscaled confidence scores and sequence generation probabilities.
    """

    def __init__(self, calibration_profile_path: Optional[Path] = None):
        """
        Initialize calibrator and load offline parameter profiles if available.

        Args:
            calibration_profile_path (Optional[Path]): Path to JSON calibration file.
        """
        self.profile_path = calibration_profile_path or (
            config.MODELS_DIR / "calibration" / "vlm_calibration_profile.json"
        )
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self.temperature: float = 1.0
        self.load_profile()

    def load_profile(self) -> None:
        """
        Feature 5: Loads persistent calibration weights from config.MODELS_DIR.
        """
        if self.profile_path.exists():
            with open(self.profile_path, "r") as f:
                params = json.load(f)
                self.platt_a = params.get("platt_a", 1.0)
                self.platt_b = params.get("platt_b", 0.0)
                self.temperature = params.get("temperature", 1.0)

    def save_profile(self) -> None:
        """
        Feature 5: Saves calibrated profile parameters to local air-gapped directory.
        """
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w") as f:
            json.dump(
                {
                    "platt_a": self.platt_a,
                    "platt_b": self.platt_b,
                    "temperature": self.temperature,
                },
                f,
                indent=4,
            )

    def fit_platt_scaling(self, raw_scores: np.ndarray, ground_truth_labels: np.ndarray) -> None:
        """
        Feature 1: Platt Scaling Logistic Calibration.
        Fits sigmoid mapping parameters ($a, b$) over validation detection logs:
        $P(Y=1|S) = \frac{1}{1 + \exp(a \cdot S + b)}$

        Args:
            raw_scores (np.ndarray): Uncalibrated score array.
            ground_truth_labels (np.ndarray): Binary targets (1 = True Positive, 0 = False Positive).
        """
        # Loss function: Binary Cross Entropy with logistic transform
        def loss_func(params):
            a, b = params
            logits = a * raw_scores + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
            # Compute log loss
            eps = 1e-7
            loss = -np.mean(
                ground_truth_labels * np.log(probs + eps)
                + (1 - ground_truth_labels) * np.log(1 - probs + eps)
            )
            return loss

        res = minimize(loss_func, [1.0, 0.0], method="L-BFGS-B")
        self.platt_a, self.platt_b = res.x
        self.save_profile()

    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Applies learned Platt Scaling transformation to raw confidence scores.
        """
        logits = self.platt_a * scores + self.platt_b
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return np.round(calibrated, 4)

    def normalize_length_penalty(self, log_likelihood: float, token_length: int, alpha: float = 0.6) -> float:
        """
        Feature 4: Length Normalization for Florence-2 Sequence Probabilities.
        Prevents long, highly detailed open-vocabulary sequence prompts from receiving 
        artificially low probability scores. Formula: $S_{\text{norm}} = \frac{LL}{L^\alpha}$
        """
        if token_length <= 0:
            return float(np.exp(log_likelihood))
        
        penalty = (token_length ** alpha)
        normalized_ll = log_likelihood / penalty
        return float(np.exp(normalized_ll))

    def apply_spatial_density_discount(
        self,
        df: pd.DataFrame,
        bbox_cols: Tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
        score_col: str = "confidence",
        density_radius: float = 100.0,
        max_discount: float = 0.30,
    ) -> pd.DataFrame:
        """
        Feature 3: Spatial Density Score Discounting.
        Penalizes confidence scores in ultra-dense spatial prediction clusters where 
        spurious phantom bounding box detections occur frequently.
        """
        if df.empty or len(df) < 5:
            return df

        df = df.copy()
        centroids = np.column_stack([
            (df[bbox_cols[0]] + df[bbox_cols[2]]) / 2.0,
            (df[bbox_cols[1]] + df[bbox_cols[3]]) / 2.0,
        ])

        discounts = []
        for i in range(len(centroids)):
            # Calculate Euclidean distances to all other detection centroids
            dists = np.linalg.norm(centroids - centroids[i], axis=1)
            neighbor_count = np.sum(dists <= density_radius) - 1  # Exclude self

            # Compute progressive discount factor
            discount = min(max_discount, neighbor_count * 0.03)
            discounts.append(1.0 - discount)

        df[score_col] = np.clip(df[score_col].values * np.array(discounts), 0.0, 1.0)
        return df