"""
src/detection/hard_negative_miner.py

Uncertainty Mining and Edge-Case Archival Engine.

Flags ambiguous VLM predictions, spatially isolated anomalies, and edge cases. 
Extracts their image crops and formats them into an offline review dashboard and 
fine-tuning dataset (YOLO format) for continuous model improvement.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class HardNegativeMiner:
    """
    Mines difficult detections for manual review and offline dataset augmentation.
    """

    def __init__(self, lower_thresh: float = 0.25, upper_thresh: float = 0.40):
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.audit_dir = config.PROCESSED_DIR / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def extract_uncertain_predictions(self, df: pd.DataFrame, score_col: str = "confidence") -> pd.DataFrame:
        """
        Feature 1: Uncertainty Band Filtering.
        Extracts detections sitting in the ambiguous decision boundary.
        """
        if df.empty:
            return df
        mask = (df[score_col] >= self.lower_thresh) & (df[score_col] <= self.upper_thresh)
        return df[mask].copy()

    def flag_spatial_outliers(self, df: pd.DataFrame, bbox_cols: Tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax")) -> pd.DataFrame:
        """
        Feature 4: Spatial Outlier Flagging.
        Identifies confident predictions that are completely isolated using unsupervised LOF.
        """
        if len(df) < 10:
            return df
            
        centroids = np.column_stack([
            (df[bbox_cols[0]] + df[bbox_cols[2]]) / 2.0,
            (df[bbox_cols[1]] + df[bbox_cols[3]]) / 2.0,
        ])
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        outlier_labels = lof.fit_predict(centroids)
        
        # -1 indicates an outlier
        df["is_spatial_anomaly"] = outlier_labels == -1
        return df

    def mine_and_archive(self, image_path: Path, df_ambiguous: pd.DataFrame, image_id: str) -> None:
        """
        Feature 2 & 3: Automated Cropping & YOLO Export.
        Saves image chips of edge cases and their labels to config.PROCESSED_DIR / 'audit'.
        """
        if df_ambiguous.empty or not image_path.exists():
            return

        img = cv2.imread(str(image_path))
        if img is None:
            return

        yolo_dir = self.audit_dir / "yolo_dataset"
        yolo_dir.mkdir(exist_ok=True)

        for idx, row in df_ambiguous.iterrows():
            xmin, ymin = int(row["xmin"]), int(row["ymin"])
            xmax, ymax = int(row["xmax"]), int(row["ymax"])
            
            # Crop with a 20-pixel context padding
            crop = img[max(0, ymin-20):min(img.shape[0], ymax+20), max(0, xmin-20):min(img.shape[1], xmax+20)]
            crop_filename = f"{image_id}_ambiguous_{idx}.jpg"
            cv2.imwrite(str(self.audit_dir / crop_filename), crop)

            # Feature 3: Pseudo-label in YOLO format (class x_center y_center width height)
            with open(yolo_dir / f"{image_id}_ambiguous_{idx}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.9 0.9\n") # Placeholder normalized YOLO coord for the centered crop

    def generate_html_dashboard(self, df_ambiguous: pd.DataFrame) -> None:
        """
        Feature 5: Offline HTML Dashboard Generator.
        Creates a local gallery of uncertain crops for rapid stakeholder visual review.
        """
        html_content = "<html><head><title>Hard Negative Audit</title></head><body><h1>Ambiguous Detections Review</h1><div style='display:flex; flex-wrap:wrap;'>"
        
        for idx, row in df_ambiguous.iterrows():
            crop_path = f"{row.get('image_id', 'unknown')}_ambiguous_{idx}.jpg"
            conf = row['confidence']
            label = row['label']
            html_content += f"<div style='margin:10px; border:1px solid #ccc; padding:5px;'><img src='{crop_path}' width='200'><p>{label} (Conf: {conf:.2f})</p></div>"
            
        html_content += "</div></body></html>"
        
        with open(self.audit_dir / "audit_dashboard.html", "w") as f:
            f.write(html_content)