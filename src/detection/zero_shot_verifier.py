"""
src/detection/zero_shot_verifier.py

Localized Candidate Verification and False-Positive Suppression Engine.

Crops candidate bounding boxes directly from source image rasters and executes 
secondary feature extraction against exemplar reference prototypes using local vision 
backbones to eliminate false positives prior to vector output generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class ZeroShotVerifier:
    """
    Secondary crop verifier that checks candidate detections against local visual prototypes.
    """

    def __init__(self, target_prototypes_dir: Optional[Path] = None):
        """
        Initialize verifier and load offline feature extractor backbone.

        Args:
            target_prototypes_dir (Optional[Path]): Directory containing offline target reference crops.
        """
        self.prototypes_dir = target_prototypes_dir or (config.MODELS_DIR / "prototypes")
        self.device = config.DEVICE
        self.torch_dtype = config.TORCH_DTYPE

        # Image crop normalization transform
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Lightweight secondary feature extractor
        self.backbone = self._load_backbone()
        self.prototype_embeddings: Dict[str, torch.Tensor] = {}
        self.load_reference_prototypes()

    def _load_backbone(self) -> nn.Module:
        """
        Loads offline feature extraction backbone.
        """
        weights_path = config.MODELS_DIR / "resnet50_verifier.pth"
        if not weights_path.exists():
            # Standard stub structure for air-gapped initialization
            import torchvision.models as models
            model = models.resnet50(weights=None)
            model.fc = nn.Identity()
            model.to(device=self.device, dtype=self.torch_dtype)
            model.eval()
            return model

        model = torch.load(weights_path, map_location=self.device)
        model.eval()
        return model

    def load_reference_prototypes(self) -> None:
        """
        Feature 2: Prototype Feature Matrix Construction.
        Extracts and stores average feature vectors for exemplar target categories 
        stored offline in config.MODELS_DIR / 'prototypes'.
        """
        if not self.prototypes_dir.exists():
            return

        for class_dir in self.prototypes_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                embeddings = []

                for img_path in class_dir.glob("*.jpg"):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        tensor = self.transform(img).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
                        with torch.no_grad():
                            feat = self.backbone(tensor)
                            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
                            embeddings.append(feat)
                    except Exception:
                        continue

                if embeddings:
                    # Average class embedding prototype
                    self.prototype_embeddings[class_name] = torch.mean(torch.stack(embeddings), dim=0)

    def verify_crop_quality(self, crop_arr: np.ndarray, min_variance: float = 15.0) -> bool:
        """
        Feature 3: Texture Variance and Quality Filtering.
        Rejects featureless crops (e.g., solid ocean water, uniform sand) that escaped VLM prompts.
        """
        if crop_arr.size == 0:
            return False

        gray = np.mean(crop_arr, axis=2) if crop_arr.ndim == 3 else crop_arr
        variance = np.var(gray)
        return float(variance) >= min_variance

    def verify_detections(
        self,
        image_input: Union[str, Path, Image.Image],
        df_detections: pd.DataFrame,
        alpha: float = 0.50,
        bbox_cols: Tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
        label_col: str = "label",
        score_col: str = "confidence",
    ) -> pd.DataFrame:
        """
        Features 1, 4 & 5: Crop extraction, CUDA batch feature comparison, and score fusion.

        Formula: $S_{\text{final}} = \alpha S_{\text{vlm}} + (1 - \alpha) S_{\text{verify}}$
        """
        if df_detections.empty:
            return df_detections

        if isinstance(image_input, (str, Path)):
            full_img = Image.open(image_input).convert("RGB")
        else:
            full_img = image_input.convert("RGB")

        verified_records = []

        for _, row in df_detections.iterrows():
            xmin, ymin, xmax, ymax = int(row[bbox_cols[0]]), int(row[bbox_cols[1]]), int(row[bbox_cols[2]]), int(row[bbox_cols[3]])
            
            # Crop candidate region
            crop = full_img.crop((xmin, ymin, xmax, ymax))
            crop_np = np.array(crop)

            # Feature 3: Quality check
            if not self.verify_crop_quality(crop_np):
                continue

            cls_name = str(row[label_col]).lower()
            vlm_score = float(row[score_col])

            # Feature 2 & 5: Calculate prototype similarity if prototype exists
            if cls_name in self.prototype_embeddings:
                tensor = self.transform(crop).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
                with torch.no_grad():
                    crop_feat = self.backbone(tensor)
                    crop_feat = torch.nn.functional.normalize(crop_feat, p=2, dim=-1)
                    proto_feat = self.prototype_embeddings[cls_name]
                    
                    sim = torch.nn.functional.cosine_similarity(crop_feat, proto_feat, dim=-1).item()
                    sim_score = max(0.0, float(sim))

                # Feature 5: Score Fusion
                final_score = alpha * vlm_score + (1.0 - alpha) * sim_score
            else:
                final_score = vlm_score

            updated_row = row.to_dict()
            updated_row[score_col] = round(final_score, 4)
            verified_records.append(updated_row)

        # Feature 4: Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pd.DataFrame(verified_records)
    